import os, json, time, tempfile, subprocess, shutil
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from workers.deepseek import DeepSeekMusicOrchestrator

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

# ============================================================================
# NEW: Spatial Pipeline Worker (MVSEP, Slicing, L/R, Kaggle, DeepSeek)
# ============================================================================
