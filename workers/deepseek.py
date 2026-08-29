import os
from openai import OpenAI

class DeepSeekMusicOrchestrator:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com/v1"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Token missing.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

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
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                temperature=0.4,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek error: {e}")
            return ""

# ============================================================================
# NEW: Advanced Structural Pipeline Worker
# ============================================================================
