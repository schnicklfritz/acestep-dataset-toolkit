import os
import numpy as np
import librosa
import soundfile as sf
from openai import OpenAI
from PySide6.QtCore import QThread, Signal

class DeepSeekMusicOrchestrator:
    def __init__(self, api_key=None, base_url=None, config=None, role="aggregator"):
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

class AdvancedDatasetOrchestratorWorker(QThread):
    progress = Signal(int, str)
    track_processing_complete = Signal(str, dict, str)
    error_occurred = Signal(str)

    def __init__(self, track_id, file_path, target_genre, api_key, use_spatial_module):
        super().__init__()
        self.track_id = track_id
        self.file_path = file_path
        self.target_genre = target_genre
        self.api_key = api_key
        self.use_spatial = use_spatial_module
        self._is_cancelled = False

    def run(self):
        try:
            self.progress.emit(10, "Loading audio...")
            y, sr = librosa.load(self.file_path, sr=None, mono=False)
            y_mono = librosa.to_mono(y) if y.ndim > 1 else y
            duration = librosa.get_duration(y=y_mono, sr=sr)

            mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
            bounds = librosa.segment.agglomerative(mfcc, k=9)
            bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]
            bound_times = sorted(set(bound_times))

            filtered = [bound_times[0]]
            for t in bound_times[1:]:
                if t - filtered[-1] >= 12.0:
                    filtered.append(t)
            if filtered[-1] < duration:
                filtered[-1] = duration
            bound_times = filtered

            audio_dir = os.path.dirname(self.file_path)
            slice_dir = os.path.join(audio_dir, "structural_slices")
            os.makedirs(slice_dir, exist_ok=True)
            file_base = os.path.splitext(os.path.basename(self.file_path))[0]

            segments = []
            for i in range(len(bound_times)-1):
                start, end = bound_times[i], bound_times[i+1]
                name = f"Section_{i+1:02d}"
                start_s = int(start * sr)
                end_s = int(end * sr)
                chunk = y[:, start_s:end_s] if y.ndim > 1 else y[start_s:end_s]
                out_path = os.path.join(slice_dir, f"{file_base}_{name}.wav")
                sf.write(out_path, chunk.T if y.ndim > 1 else chunk, sr)
                segments.append({
                    "name": name,
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "slice_path": out_path,
                    "caption": "",
                    "spatial_tokens": {}
                })
                self.progress.emit(50 + int(i*5), f"Sliced: {name}")

            if self.use_spatial and y.ndim > 1:
                for seg in segments:
                    slice_y, _ = librosa.load(seg["slice_path"], sr=None, mono=False)
                    if slice_y.ndim > 1:
                        left_en = np.sum(librosa.feature.rms(y=slice_y[0]))
                        right_en = np.sum(librosa.feature.rms(y=slice_y[1]))
                        ratio = left_en / (right_en + 1e-9)
                        if ratio > 2.0:
                            seg["spatial_tokens"]["stereo_balance"] = "heavy left"
                        elif ratio < 0.5:
                            seg["spatial_tokens"]["stereo_balance"] = "heavy right"
                        else:
                            seg["spatial_tokens"]["stereo_balance"] = "balanced"

            self.progress.emit(85, "Calling DeepSeek for aggregation...")
            orchestrator = DeepSeekMusicOrchestrator(api_key=self.api_key)
            onset = librosa.onset.onset_strength(y=y_mono, sr=sr)
            bpm = int(librosa.feature.tempo(onset_envelope=onset, sr=sr)[0])
            final_caption = orchestrator.generate_master_dataset_prompt(
                target_genre=self.target_genre,
                global_bpm=bpm,
                segments=segments
            )

            self.progress.emit(100, "Done.")
            self.track_processing_complete.emit(self.track_id, segments, final_caption)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self._is_cancelled = True

