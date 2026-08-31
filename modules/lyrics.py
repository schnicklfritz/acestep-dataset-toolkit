"""Word-aligned lyrics transcription via WhisperX (optional).

WhisperX needs ``torch`` + ``transformers`` + ``faster-whisper`` (install:
``pip install whisperx``). This is a guarded module — if it isn't installed,
:func:`transcribe` raises a clear, actionable message.
"""
import os


def transcribe_available():
    try:
        import whisperx  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def transcribe(audio_path, model_size="small", language=None, device=None,
               initial_prompt=None):
    """Transcribe lyrics with WhisperX.

    ``initial_prompt`` biases the model toward context (artist/genre/known
    words) and ``language`` forces the language instead of auto-detection.

    Returns ``{"lyrics", "segments"}`` where ``segments`` is the word-aligned
    WhisperX segment list (start/end/text per segment).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    if not transcribe_available():
        raise RuntimeError(
            "WhisperX is not installed. Install it with:  pip install whisperx  "
            "(requires torch + transformers)."
        )
    import torch
    import whisperx

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device.startswith("cuda") else "float32"
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language=language, initial_prompt=initial_prompt)

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language"), device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception:  # noqa: BLE001 — alignment is best-effort
        pass

    segments = result.get("segments", [])
    lyrics = "\n".join((seg.get("text") or "").strip() for seg in segments).strip()
    return {"lyrics": lyrics, "segments": segments}


_TRANSCRIBE_PROMPT = (
    "Transcribe the lyrics of this song verbatim. Output only the lyrics text, "
    "one line per lyric, keeping the original words and pronunciation."
)


def transcribe_lyrics_engine(audio_path, engine="whisperx", config=None,
                             model_size="small", language=None, initial_prompt=None):
    """Transcribe lyrics via the chosen engine.

    Engines: ``whisperx`` (default), ``gemini`` (audio-native via the Gemini
    caption backend), ``acestep_transcriber`` (experimental — needs its HF
    model; falls back to a clear message when unavailable).
    """
    engine = (engine or "whisperx").strip().lower()
    if engine == "gemini":
        from workers.caption_backends import GeminiBackend

        gb = GeminiBackend(config or {})
        prompt = _TRANSCRIBE_PROMPT
        if initial_prompt:
            prompt = f"Context: {initial_prompt}\n\n" + prompt
        text = gb.caption(audio_path, os.path.basename(audio_path), prompt)
        return {"lyrics": (text or "").strip(), "segments": []}
    if engine in ("acestep_transcriber", "acestep-transcriber"):
        raise RuntimeError(
            "ace-step-transcriber is not wired yet — install WhisperX and use "
            "engine 'whisperx' (or 'gemini')."
        )
    return transcribe(audio_path, model_size=model_size, language=language,
                      initial_prompt=initial_prompt)