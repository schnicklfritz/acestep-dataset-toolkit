import os
from openai import OpenAI

class DeepSeekMusicOrchestrator:
    def __init__(self, api_key=None, base_url=None, config=None, role="aggregator"):
        """Provider-aware orchestrator.

        Pass ``config`` to use the configured LLM provider (DeepSeek / Gemini /
        Groq / OpenRouter / local), honouring the per-role override for the
        given ``role`` (aggregator / captioner / assistant). Otherwise falls
        back to a direct DeepSeek client with ``api_key``.
        """
        if config is not None:
            from modules.llm_client import get_client

            self.provider, self.info, self.client = get_client(config, role=role)
            self.api_key = (config.get(self.info["key"]) or "").strip()
            self.model = self.info.get("model") or "deepseek-chat"
        else:
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API Token missing.")
            self.client = OpenAI(
                api_key=self.api_key, base_url=base_url or "https://api.deepseek.com/v1"
            )
            self.provider = "deepseek"
            self.model = "deepseek-chat"

    def generate_master_dataset_prompt(self, target_genre, global_bpm, segments, spatial_tokens=None, lyrics=None):
        system_prompt = (
            "You are an elite music prompt engineer for ACE-Step. Synthesize a cohesive master prompt from structural segments, "
            "spatial instrument placement, and lyrical content. Output ONLY the final prompt, no introductory text. "
            "Structure: [Genre/Vibe], [Production Texture], [Instrumentation with spatial placement], [Dynamics/Energy], [Structural flow]."
        )
        user_context = f"TARGET GENRE: {target_genre}\nGLOBAL BPM: {global_bpm}\n\n"
        if spatial_tokens:
            user_context += "SPATIAL PLACEMENT:\n"
            for instr, pos in spatial_tokens.items():
                user_context += f"  {instr}: {pos}\n"
        user_context += "\nSTRUCTURAL SEGMENTS:\n"
        for seg in segments:
            user_context += f"  [{seg['name']}] {seg['start_sec']}s - {seg['end_sec']}s\n"
            user_context += f"  Caption: {seg.get('caption', '')}\n"
            if lyrics and seg['name'] in lyrics:
                user_context += f"  Lyrics: {lyrics[seg['name']]}\n"
            if 'spatial_tokens' in seg and seg['spatial_tokens']:
                user_context += f"  Spatial: {seg['spatial_tokens']}\n"
            user_context += "\n"
        user_context += "Compile final master caption now:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                temperature=0.4,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"{self.provider} aggregation error: {e}")
            return ""

    def recommend_instrument_models(self, instruments_text, available_models):
        """Ask DeepSeek which instrument-specific MVSEP models to run.

        ``available_models`` is the live MVSEP algorithm catalog (names), so
        DeepSeek can only recommend models that actually exist. Returns a
        comma-separated string of model names.
        """
        system_prompt = (
            "You are a music production expert and stem-separation specialist. "
            "Given the instruments detected in a song, recommend which instrument-"
            "specific stem-separation models should be run, chosen ONLY from the "
            "provided catalog of available MVSEP models. Return ONLY the model "
            "names as a comma-separated list — no explanations, no numbering."
        )
        user = (
            f"Detected instruments: {instruments_text}\n\n"
            f"Available MVSEP models:\n{', '.join(available_models)}\n\n"
            "Recommended instrument-specific models:"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"{self.provider} instrument recommendation error: {e}")
            return ""

# ============================================================================
# NEW: Advanced Structural Pipeline Worker
# ============================================================================
