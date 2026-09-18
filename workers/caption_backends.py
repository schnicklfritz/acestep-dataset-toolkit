"""Pluggable caption backends for the AI captioner.

Providers:
  * ``ace_step`` — Qwen2.5-Omni captioner on a Kaggle GPU (see workers/caption.py).
  * ``gemini``   — Google Gemini (audio-native) via ``google-genai`` or the
                   legacy ``google-generativeai`` SDK.
  * ``deepseek`` — DeepSeek LLM (text-only synthesis).
  * ``custom``   — any OpenAI-compatible endpoint (vLLM, Ollama, llama.cpp,
                   runpod, a rented or local GPU box). Text-only by default;
                   optional ``input_audio`` support for models that accept it.
"""
import base64
import os

from pathlib import Path

from modules.caption_quality import trim_to_caption

AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}


def _mime_for(path):
    return AUDIO_MIME.get(Path(path).suffix.lower(), "audio/mpeg")


def _audio_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _audio_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


class GeminiBackend:
    """Caption audio with Google Gemini (native audio understanding)."""

    def __init__(self, config):
        self.config = config
        self.api_key = (
            config.get("gemini_api_key", "").strip() or os.getenv("GEMINI_API_KEY", "")
        )
        self.model = (
            config.get("gemini_model", "gemini-2.5-flash").strip()
            or "gemini-2.5-flash"
        )

    def caption(self, audio_path, filename, prompt, metadata=None, system_prompt=""):
        if not self.api_key:
            raise ValueError(
                "Gemini API key missing — set it in ⚙ Settings (or GEMINI_API_KEY)."
            )
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        # Preferred SDK: google-genai (current).
        try:
            from google import genai as genai_sdk
            from google.genai import types as genai_types

            client = genai_sdk.Client(api_key=self.api_key)
            audio_part = genai_types.Part.from_bytes(
                data=_audio_bytes(audio_path), mime_type=_mime_for(audio_path)
            )
            gen_config = None
            if system_prompt:
                # The ACE-Step 1.5XL annotation schema belongs in the SYSTEM role.
                gen_config = genai_types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            response = client.models.generate_content(
                model=self.model, contents=[audio_part, prompt], config=gen_config
            )
            # Trim: a provider may ignore max_tokens entirely and return a tome.
            return trim_to_caption((response.text or "").strip())
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001 — real API errors surface clearly
            raise RuntimeError(f"Gemini request failed: {e}")

        # Legacy SDK fallback.
        try:
            import google.generativeai as legacy

            legacy.configure(api_key=self.api_key)
            uploaded = legacy.upload_file(audio_path)
            model = legacy.GenerativeModel(
                self.model, system_instruction=system_prompt or None
            )
            response = model.generate_content([uploaded, prompt])
            return trim_to_caption((response.text or "").strip())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Gemini SDK not installed or failed. Install with: "
                f"pip install google-genai  ({e})"
            )


class CustomOpenAICompatBackend:
    """Caption via any OpenAI-compatible endpoint (vLLM / Ollama / local GPU).

    GROUNDING CAVEAT: audio is only attached when ``custom_caption_audio`` is
    true. With it false (the default) this backend sends TEXT ONLY, so a model
    reached this way is describing a track it never heard — it will happily invent
    a genre and a BPM. The schema prompt now forbids BPM in the caption and
    modules/caption_quality.py flags it, but the real fix is to send the audio.
    """

    def __init__(self, config):
        self.config = config
        self.base_url = config.get("custom_caption_url", "").strip()
        self.model = (
            config.get("custom_caption_model", "").strip() or "caption-model"
        )
        self.send_audio = bool(config.get("custom_caption_audio", False))
        # Most local/self-hosted servers ignore the key; reuse the app's
        # custom auth token when the user runs a gated endpoint.
        self.api_key = config.get("custom_key", "").strip() or "sk-no-key"

    def caption(self, audio_path, filename, prompt, metadata=None, system_prompt=""):
        if not self.base_url:
            raise ValueError(
                "Custom endpoint base URL missing — set it in ⚙ Settings."
            )
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        content = [{"type": "text", "text": prompt}]
        if self.send_audio and os.path.exists(audio_path):
            fmt = Path(audio_path).suffix.lstrip(".").lower() or "wav"
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": _audio_b64(audio_path), "format": fmt},
                }
            )
        messages = []
        if system_prompt:
            # The ACE-Step 1.5XL annotation schema belongs in the SYSTEM role.
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=int(self.config.get("caption_max_tokens", 512)),
            # Without a penalty a caption can loop on a repeated phrase until the
            # token cap (observed: a lyric refrain repeated ~200x).
            frequency_penalty=float(
                self.config.get("caption_frequency_penalty", 0.3) or 0
            ),
            presence_penalty=float(
                self.config.get("caption_presence_penalty", 0.0) or 0
            ),
        )
        return trim_to_caption(
            (response.choices[0].message.content or "").strip()
        )