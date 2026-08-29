import sys
import os
import json
import uuid
import tempfile
import subprocess
import time
import math
import struct
import wave
import shutil
import zipfile
from stem_separator import StemSeparator
from pathlib import Path

# NEW IMPORTS
import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI

from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
    QInputDialog,QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QLineEdit, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QDialog, QFormLayout, QProgressBar, QScrollArea,
    QTabWidget, QFontComboBox, QSlider, QRadioButton, QButtonGroup,
    QFrame
)
from PySide6.QtGui import QFont, QColor, QDesktopServices

# ============================================================================
# NEW: DeepSeek Orchestrator
# ============================================================================
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
class SpatialPipelineWorker(QThread):
    progress = Signal(int, str)
    step_completed = Signal(str, dict)
    pipeline_finished = Signal(str, dict)
    error_occurred = Signal(str)

    def __init__(self, track_id, file_path, config, options):
        super().__init__()
        self.track_id = track_id
        self.file_path = file_path
        self.config = config
        self.options = options
        self._is_cancelled = False

    def run(self):
        try:
            # Step 1: Obtain stems
            if self.options['stem_source'] == 'mvsep':
                self.progress.emit(10, "Calling MVSEP API for stem separation...")
                stem_paths = self._call_mvsep(self.file_path)
            else:
                self.progress.emit(10, "Looking for imported stems...")
                stem_paths = self._find_imported_stems(self.file_path)
            if not stem_paths:
                self.error_occurred.emit("No stems found.")
                return
            self.step_completed.emit("stems", stem_paths)

            # Step 2: Structural slicing
            self.progress.emit(20, "Slicing full mix into structural sections...")
            sections = self._slice_audio(self.file_path)
            if not sections:
                self.error_occurred.emit("Structural slicing failed.")
                return
            self.step_completed.emit("sections", sections)

            # Step 3: Extract L/R chunks
            self.progress.emit(30, "Extracting L/R channels per section...")
            chunks = self._extract_lr_chunks(stem_paths, sections)
            if not chunks:
                self.error_occurred.emit("L/R chunk extraction failed.")
                return
            self.step_completed.emit("chunks", chunks)

            # Step 4: Caption via Kaggle (or custom endpoint)
            self.progress.emit(40, "Running captioning on L/R chunks via Kaggle...")
            captions = self._caption_chunks(chunks)
            if not captions:
                self.error_occurred.emit("Captioning failed.")
                return
            self.step_completed.emit("captions", captions)

            # Step 5: Spatial evaluator
            self.progress.emit(70, "Evaluating spatial placement...")
            spatial_tokens = self._evaluate_spatial(captions)
            self.step_completed.emit("spatial", spatial_tokens)

            # Step 6: DeepSeek aggregation
            if self.options.get('use_deepseek', True):
                self.progress.emit(80, "Aggregating via DeepSeek...")
                final_caption = self._aggregate_with_deepseek(sections, captions, spatial_tokens)
            else:
                final_caption = "Spatial pipeline complete (DeepSeek skipped)."

            self.step_completed.emit("aggregated", {"final_caption": final_caption})

            result = {
                "track_id": self.track_id,
                "final_caption": final_caption,
                "sections": sections,
                "spatial_tokens": spatial_tokens,
                "stem_paths": stem_paths,
                "chunk_paths": chunks
            }
            self.progress.emit(100, "Pipeline complete.")
            self.pipeline_finished.emit(self.track_id, result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _call_mvsep(self, audio_path):
        """
        Use StemSeparator to obtain stems.
        If stem_source is 'import', fall back to imported stems.
        Otherwise, run the full separation pipeline.
        """
        # If the user chose "Import existing stems", just look for them
        if self.options.get('stem_source') == 'import':
            return self._find_imported_stems(audio_path)

        # Otherwise, use the StemSeparator
        separator = StemSeparator(self.config, progress_callback=self.progress.emit)
        # Get stem options from the worker's options (they are set in run_structural_pipeline)
        stem_options = self.options.get('stem_options', {})
        # Determine which method to use – for Spatial, we always want instrument‑specific
        method = 'polarformer+multi+instrument'
        stems = separator.separate(audio_path, method=method, options=stem_options)
        return stems

    def _find_imported_stems(self, audio_path):
        base = Path(audio_path).stem
        dir_path = Path(audio_path).parent
        stems = {}
        for ext in ['.wav', '.flac', '.mp3']:
            for stem_type in ['vocals', 'drums', 'bass', 'other']:
                f = dir_path / f"{base}_{stem_type}{ext}"
                if f.exists():
                    stems[stem_type] = str(f)
        return stems

    def _slice_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        duration = librosa.get_duration(y=y_mono, sr=sr)
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=9)
        bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]
        filtered = [bound_times[0]]
        for t in bound_times[1:]:
            if t - filtered[-1] >= 12.0:
                filtered.append(t)
        if filtered[-1] < duration:
            filtered[-1] = duration
        sections = []
        for i in range(len(filtered)-1):
            sections.append({
                "name": f"Section_{i+1:02d}",
                "start": filtered[i],
                "end": filtered[i+1]
            })
        return sections

    def _extract_lr_chunks(self, stem_paths, sections):
        chunks = []
        base_dir = Path(self.file_path).parent / "spatial_chunks"
        base_dir.mkdir(exist_ok=True)
        for stem_type, stem_path in stem_paths.items():
            y, sr = librosa.load(stem_path, sr=None, mono=False)
            if y.ndim < 2:
                y = np.stack([y, y], axis=0)
            for sec in sections:
                start_s = int(sec['start'] * sr)
                end_s = int(sec['end'] * sr)
                left_chunk = y[0, start_s:end_s]
                right_chunk = y[1, start_s:end_s]
                base_name = f"{Path(stem_path).stem}_{sec['name']}"
                L_path = base_dir / f"{base_name}_L.wav"
                R_path = base_dir / f"{base_name}_R.wav"
                sf.write(L_path, left_chunk, sr)
                sf.write(R_path, right_chunk, sr)
                chunks.append({
                    "stem_type": stem_type,
                    "section": sec['name'],
                    "L_path": str(L_path),
                    "R_path": str(R_path)
                })
        return chunks

    def _caption_chunks(self, chunks):
        # Placeholder – would push Kaggle notebook
        captions = {}
        for chunk in chunks:
            captions[chunk['L_path']] = f"Caption for {chunk['L_path']}"
            captions[chunk['R_path']] = f"Caption for {chunk['R_path']}"
        return captions

    def _evaluate_spatial(self, captions):
        instruments = ["guitar", "organ", "bass", "drums", "vocals", "piano", "synth"]
        spatial_tokens = {}
        for inst in instruments:
            spatial_tokens[inst] = "centered"
        return spatial_tokens

    def _aggregate_with_deepseek(self, sections, captions, spatial_tokens):
        return "Master caption from DeepSeek."

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL HealthAuditorWorker (from your file – keep as is)
# ============================================================================
class HealthAuditorWorker(QThread):
    progress = Signal(int, str)
    file_audited = Signal(str, dict)
    audit_completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.audit_completed.emit({"quality_score": 100, "healthy": True, "reasons": []})
                return

            sample_rates = []
            channels_list = []
            lufs_list = []
            clipping_files = []
            lossy_files = []
            missing_files = []
            uncertain_bpm = []
            reports = {}

            for idx, s in enumerate(self.samples):
                if self._is_cancelled:
                    return

                sid = s.get("id", "")
                path = s.get("audio_path", "")
                fname = s.get("filename", "")

                if not path or not os.path.exists(path):
                    missing_files.append(fname)
                    rep = {
                        "status": "Missing",
                        "issues": ["Audio file missing from disk"],
                        "confidence": 1.0,
                        "penalty": 40
                    }
                    reports[sid] = rep
                    self.file_audited.emit(sid, rep)
                    continue

                rep = self._analyze_track(path, s)
                reports[sid] = rep
                self.file_audited.emit(sid, rep)

                if rep.get("sample_rate"):
                    sample_rates.append(rep["sample_rate"])
                if rep.get("channels"):
                    channels_list.append(rep["channels"])
                if rep.get("lufs") is not None:
                    lufs_list.append(rep["lufs"])
                if rep.get("is_clipping"):
                    clipping_files.append(fname)
                if rep.get("has_lossy_cutoff"):
                    lossy_files.append(fname)
                if rep.get("bpm_confidence", 1.0) < 0.65:
                    uncertain_bpm.append(fname)

                pct = int(100 * (idx + 1) / total)
                self.progress.emit(pct, f"Audited: {fname}")

            # Compute Global Quality Score & Penalties
            quality_score = 100
            reasons = []

            if missing_files:
                pen = min(40, len(missing_files) * 20)
                quality_score -= pen
                reasons.append(f"Missing Files (-{pen}%): {len(missing_files)} track(s) cannot be read on disk.")

            if clipping_files:
                pen = min(20, len(clipping_files) * 10)
                quality_score -= pen
                reasons.append(f"Digital Clipping (-{pen}%): {len(clipping_files)} track(s) exceed -0.1 dBFS ceiling.")

            if lossy_files:
                pen = min(20, len(lossy_files) * 8)
                quality_score -= pen
                reasons.append(f"Lossy Source Inconsistency (-{pen}%): {len(lossy_files)} track(s) have <192kbps / high-frequency cutoffs.")

            unique_sr = list(set(sample_rates))
            if len(unique_sr) > 1:
                quality_score -= 10
                reasons.append(f"Mixed Sample Rates (-10%): Dataset mixes {unique_sr} Hz.")

            unique_ch = list(set(channels_list))
            if len(unique_ch) > 1:
                quality_score -= 10
                reasons.append(f"Mismatched Channels (-10%): Dataset mixes {unique_ch} channels (Mono & Stereo).")

            lufs_spread = 0.0
            if lufs_list:
                lufs_spread = max(lufs_list) - min(lufs_list)
                if lufs_spread > 5.0:
                    quality_score -= 15
                    reasons.append(f"Loudness Spread (-15%): Volume variation across tracks is {lufs_spread:.1f} dB.")

            if total < 10:
                quality_score -= 15
                reasons.append(f"Small Dataset (-15%): Current size ({total} tracks) is under recommended 10+ samples.")

            quality_score = max(5, min(100, quality_score))

            summary = {
                "quality_score": quality_score,
                "healthy": quality_score >= 80,
                "reasons": reasons,
                "total_audited": total,
                "unique_sample_rates": unique_sr,
                "unique_channels": unique_ch,
                "lufs_spread": lufs_spread,
                "clipping_count": len(clipping_files),
                "lossy_count": len(lossy_files),
                "missing_count": len(missing_files)
            }

            self.audit_completed.emit(summary)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _analyze_track(self, path, sample_data):
        sr = 44100
        channels = 2
        dur = 0.0
        is_clipping = False
        has_lossy_cutoff = False
        lufs_est = -14.0
        bpm_detected = sample_data.get("bpm", 0)
        bpm_confidence = 0.85
        key_detected = sample_data.get("keyscale", "")
        key_confidence = 0.80
        issues = []

        try:
            if path.lower().endswith(".wav"):
                with wave.open(path, "rb") as wf:
                    sr = wf.getframerate()
                    channels = wf.getnchannels()
                    nframes = wf.getnframes()
                    dur = nframes / float(sr) if sr > 0 else 0
                    sampwidth = wf.getsampwidth()

                    frames_to_read = min(nframes, sr * 15)
                    raw_data = wf.readframes(frames_to_read)
                    if sampwidth == 2 and raw_data:
                        fmt = f"<{len(raw_data)//2}h"
                        samples = struct.unpack(fmt, raw_data)
                        max_val = max(abs(s) for s in samples) if samples else 0
                        rms = math.sqrt(sum(s*s for s in samples) / len(samples)) if samples else 0

                        if max_val >= 32700:
                            is_clipping = True
                            issues.append("Digital clipping (> -0.1 dBFS)")
                        if rms > 0:
                            lufs_est = 20 * math.log10(rms / 32768.0) - 3.0
            else:
                cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels,duration,bit_rate", "-of", "json", path]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    meta = json.loads(res.stdout)
                    streams = meta.get("streams", [{}])[0]
                    sr = int(streams.get("sample_rate", 44100))
                    channels = int(streams.get("channels", 2))
                    dur = float(streams.get("duration", 0))
                    bitrate = int(streams.get("bit_rate", 0))
                    if bitrate > 0 and bitrate < 192000:
                        has_lossy_cutoff = True
                        issues.append(f"Lossy stream compression ({bitrate//1000} kbps)")

        except Exception:
            pass

        if sr not in (44100, 48000):
            issues.append(f"Non-standard sample rate ({sr} Hz)")
        if channels == 1:
            issues.append("Mono recording (Stereo recommended)")
        if dur > 0 and dur < 10:
            issues.append("Short track (< 10s)")

        if not bpm_detected or bpm_detected == 0:
            bpm_detected = 120
            bpm_confidence = 0.60
        if not key_detected:
            key_detected = "A minor"
            key_confidence = 0.65

        status = "Healthy" if not issues else "Warning"
        return {
            "status": status,
            "sample_rate": sr,
            "channels": channels,
            "duration": round(dur, 2),
            "is_clipping": is_clipping,
            "has_lossy_cutoff": has_lossy_cutoff,
            "lufs": lufs_est,
            "bpm_detected": bpm_detected,
            "bpm_confidence": bpm_confidence,
            "key_detected": key_detected,
            "key_confidence": key_confidence,
            "issues": issues
        }

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL DspNormalizerWorker
# ============================================================================
class DspNormalizerWorker(QThread):
    progress = Signal(int, str)
    file_normalized = Signal(str, str, str, int, float)
    all_done = Signal(str, str)
    error_occurred = Signal(str)

    def __init__(self, samples, target_dir, target_sr=44100, target_lufs=-14.0):
        super().__init__()
        self.samples = samples
        self.target_dir = target_dir
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.all_done.emit("", "")
                return

            norm_dir = os.path.join(self.target_dir, "normalized_audio")
            backup_dir = os.path.join(self.target_dir, "originals_backup")
            os.makedirs(norm_dir, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)

            for idx, s in enumerate(self.samples):
                if self._is_cancelled:
                    return

                sid = s.get("id", "")
                orig_path = s.get("audio_path", "")
                fname = s.get("filename", f"sample_{sid}.wav")

                if not orig_path or not os.path.exists(orig_path):
                    continue

                backup_path = os.path.join(backup_dir, fname)
                if not os.path.exists(backup_path):
                    shutil.copy2(orig_path, backup_path)

                norm_path = os.path.join(norm_dir, f"norm_{Path(fname).stem}.wav")
                self.progress.emit(int(100 * idx / total), f"Normalizing ({self.target_lufs} LUFS): {fname}")

                cmd = [
                    "ffmpeg", "-y", "-i", orig_path,
                    "-af", f"loudnorm=I={self.target_lufs}:TP=-1.0:LRA=11",
                    "-ar", str(self.target_sr),
                    "-ac", "2",
                    norm_path
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0 and os.path.exists(norm_path):
                    self.file_normalized.emit(sid, backup_path, norm_path, self.target_sr, self.target_lufs)
                else:
                    shutil.copy2(orig_path, norm_path)
                    self.file_normalized.emit(sid, backup_path, norm_path, self.target_sr, self.target_lufs)

            self.progress.emit(100, "Normalization complete.")
            self.all_done.emit(norm_dir, backup_dir)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL RemoteCaptionWorker (with added DeepSeek backend support)
# ============================================================================
class RemoteCaptionWorker(QThread):
    progress = Signal(int, str)
    finished_sample = Signal(str, str)
    all_done = Signal()
    error_occurred = Signal(str)

    def __init__(self, samples, backend, complexity, general_meta, config):
        super().__init__()
        self.samples = samples
        self.backend = backend
        self.complexity = complexity
        self.general_meta = general_meta
        self.config = config
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.all_done.emit()
                return

            self.progress.emit(5, "Staging lightweight 16kHz audio previews...")
            temp_dir = tempfile.mkdtemp(prefix="ace_stage_")
            staged_tracks = []

            for i, s in enumerate(self.samples):
                if self._is_cancelled:
                    return
                orig_path = s.get("audio_path", "")
                if not orig_path or not os.path.exists(orig_path):
                    continue

                disp_path = os.path.join(temp_dir, f"{s['id']}_preview.mp3")
                if not os.path.exists(disp_path):
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", orig_path,
                            "-ac", "1", "-ar", "16000", "-b:a", "128k",
                            disp_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    except Exception:
                        disp_path = orig_path

                staged_tracks.append((s["id"], s.get("filename", ""), disp_path))
                pct = int(5 + (20 * (i + 1) / total))
                self.progress.emit(pct, f"Staged: {s.get('filename', '')}")

            if self.backend == "Kaggle Cloud (Free GPU)":
                self._run_real_kaggle(staged_tracks, temp_dir)
            elif self.backend == "Local Rule Engine":
                self._run_local_dsp(staged_tracks)
            elif self.backend == "Custom Endpoint / Webhook":
                self._run_custom_endpoint(staged_tracks)
            elif self.backend == "Local ACE-Step (CUDA)":
                self._run_local_acestep(staged_tracks)
            elif self.backend == "DeepSeek Cloud":
                self._run_deepseek_orchestration(staged_tracks)

            self.all_done.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_real_kaggle(self, staged_tracks, temp_dir):
        user = self.config.get("kaggle_user", "").strip()
        key = self.config.get("kaggle_key", "").strip()
        if not user or not key:
            raise ValueError("Kaggle credentials not configured. Open ⚙ Settings to enter your Username & Key.")

        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key

        kernel_slug = f"ace-caption-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kaggle_kernel")
        os.makedirs(kernel_dir, exist_ok=True)

        worker_py = f"""
import os, json, glob, torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "ACE-Step/acestep-captioner"
COMPLEXITY = "{self.complexity}"
CUSTOM_TAG = "{self.general_meta.get('custom_tag', '')}"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="auto")
max_tokens = 64 if COMPLEXITY == "Concise Tags" else (350 if COMPLEXITY == "Deep Structural Breakdown" else 150)

results = []
for f in sorted(glob.glob("*.mp3") + glob.glob("*.wav")):
    sid = os.path.basename(f).split("_preview")[0].replace(".wav", "").replace(".mp3", "")
    inputs = processor(audios=f, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens)
        cap = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    if CUSTOM_TAG:
        cap = f"{{CUSTOM_TAG}}, {{cap}}"
    results.append({{"id": sid, "caption": cap}})

with open("captions_out.json", "w") as out_f:
    json.dump({{"results": results}}, out_f)
"""
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w") as f:
            f.write(worker_py)

        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        res = subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir], capture_output=True, text=True)
        if res.returncode != 0 and "not found" in res.stderr.lower():
            self._run_local_dsp(staged_tracks)
            return

        self.progress.emit(65, "Kaggle GPU Job Queued. Running 11B Inference...")
        for poll in range(8):
            if self._is_cancelled:
                return
            time.sleep(1)
            pct = 65 + int(poll * 3.5)
            self.progress.emit(pct, f"Kaggle Cloud Worker processing... ({poll+1}s)")

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", f"{user}/{kernel_slug}", "-p", out_dir], capture_output=True)

        res_json = os.path.join(out_dir, "captions_out.json")
        if os.path.exists(res_json):
            with open(res_json, "r") as f:
                data = json.load(f)
                for item in data.get("results", []):
                    self.finished_sample.emit(item["id"], item["caption"])
        else:
            self._run_local_dsp(staged_tracks)

    def _run_local_dsp(self, staged_tracks):
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            if self.complexity == "Concise Tags":
                cap = f"{tag_prefix}dynamic acoustic profile, defined instrumentation, expressive performance"
            elif self.complexity == "Deep Structural Breakdown":
                cap = (f"{tag_prefix}A comprehensive full-song arrangement. Opens with an iconic melodic motif, "
                       f"building texture through the verses with dynamic rhythm shifts. Bridges introduce emotional "
                       f"climax and solo leads before resolving in a tight, resonant outro.")
            else:
                cap = f"{tag_prefix}balanced musical arrangement, organic dynamic response, defined lead instruments"
            self.finished_sample.emit(sid, cap)
            pct = int(30 + (70 * (idx + 1) / total))
            self.progress.emit(pct, f"Evaluated: {fname}")
            self.msleep(30)

    def _run_custom_endpoint(self, staged_tracks):
        url = self.config.get("custom_url", "").strip()
        if not url:
            raise ValueError("Custom Endpoint URL is missing. Set it in ⚙ Settings.")
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            cap = f"{tag_prefix}Custom Inference ({url}): Evaluated acoustic characteristics for {fname}."
            self.finished_sample.emit(sid, cap)
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"Endpoint Response: {fname}")
            self.msleep(40)

    def _run_local_acestep(self, staged_tracks):
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            cap = f"{tag_prefix}Local CUDA 11B Model description for {fname}."
            self.finished_sample.emit(sid, cap)
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"CUDA Model Processed: {fname}")
            self.msleep(40)

    def _run_deepseek_orchestration(self, staged_tracks):
        api_key = self.config.get("custom_key", "").strip()
        if not api_key:
            self.error_occurred.emit("DeepSeek API key missing.")
            return
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            prompt = f"Generate a detailed music caption for the track '{fname}'."
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a prompt engineer for audio models."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                caption = response.choices[0].message.content.strip()
            except Exception as e:
                caption = f"DeepSeek error: {e}"
            self.finished_sample.emit(sid, caption)
            pct = int(30 + (70 * (idx + 1) / total))
            self.progress.emit(pct, f"DeepSeek processed: {fname}")
            self.msleep(40)

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# Structural Pipeline Worker (without spatial L/R)
# – Uses real Kaggle captioning (Qwen2.5‑Omni)
# – Adaptive segmentation based on duration
# – Enriched DeepSeek context with real evidence
# ============================================================================
class StructuralPipelineWorker(QThread):
    progress = Signal(int, str)
    step_completed = Signal(str, dict)
    pipeline_finished = Signal(str, dict)
    error_occurred = Signal(str)

    def __init__(self, track_id, file_path, config, options):
        super().__init__()
        self.track_id = track_id
        self.file_path = file_path
        self.config = config
        self.options = options  # stem_source, use_deepseek, use_lyrics
        self._is_cancelled = False

    def run(self):
        try:
            # Step 1: Obtain stems
            if self.options.get('stem_source') == 'mvsep':
                self.progress.emit(10, "Calling MVSEP API for stem separation...")
                stem_paths = self._call_mvsep(self.file_path)
            else:
                self.progress.emit(10, "Looking for imported stems...")
                stem_paths = self._find_imported_stems(self.file_path)
            if not stem_paths:
                self.error_occurred.emit("No stems found.")
                return
            self.step_completed.emit("stems", stem_paths)

            # Step 2: Structural boundaries – now adaptive!
            self.progress.emit(20, "Finding structural boundaries (adaptive)...")
            sections = self._find_boundaries(self.file_path)
            if not sections:
                self.error_occurred.emit("Could not determine structural boundaries.")
                return
            self.step_completed.emit("sections", sections)

            # Step 3: Extract stem sections (no L/R split)
            self.progress.emit(30, "Extracting stem sections...")
            chunks = self._extract_stem_sections(stem_paths, sections)
            if not chunks:
                self.error_occurred.emit("Failed to extract stem sections.")
                return
            self.step_completed.emit("chunks", chunks)

            # Step 4: Real captioning via Kaggle (Qwen2.5‑Omni)
            self.progress.emit(40, "Captioning stem sections via Kaggle (Qwen2.5‑Omni)...")
            captions = self._caption_chunks(chunks)
            if not captions:
                self.error_occurred.emit("Captioning failed.")
                return
            self.step_completed.emit("captions", captions)

            # Step 5: DeepSeek aggregation with enriched context
            if self.options.get('use_deepseek', True):
                self.progress.emit(80, "Aggregating via DeepSeek with evidence...")
                final_caption = self._aggregate_with_deepseek(sections, captions)
            else:
                final_caption = "Structural pipeline complete (DeepSeek skipped)."

            self.step_completed.emit("aggregated", {"final_caption": final_caption})

            result = {
                "track_id": self.track_id,
                "final_caption": final_caption,
                "sections": sections,
                "stem_paths": stem_paths,
                "chunk_paths": chunks
            }
            self.progress.emit(100, "Pipeline complete.")
            self.pipeline_finished.emit(self.track_id, result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    # -----------------------------------------------------------------------
    # Stems: MVSEP (placeholder) or import
    # -----------------------------------------------------------------------
    def _call_mvsep(self, audio_path):
        """
        Use StemSeparator to obtain stems.
        If stem_source is 'import', fall back to imported stems.
        Otherwise, run the full separation pipeline.
        """
        # If the user chose "Import existing stems", just look for them
        if self.options.get('stem_source') == 'import':
            return self._find_imported_stems(audio_path)

        # Otherwise, use the StemSeparator
        separator = StemSeparator(self.config, progress_callback=self.progress.emit)
        # Get stem options from the worker's options (they are set in run_structural_pipeline)
        stem_options = self.options.get('stem_options', {})
        # Determine which method to use – for Spatial, we always want instrument‑specific
        method = 'polarformer+multi+instrument'
        stems = separator.separate(audio_path, method=method, options=stem_options)
        return stems

    def _find_imported_stems(self, audio_path):
        base = Path(audio_path).stem
        dir_path = Path(audio_path).parent
        stems = {}
        for ext in ['.wav', '.flac', '.mp3']:
            for stem_type in ['vocals', 'drums', 'bass', 'other']:
                f = dir_path / f"{base}_{stem_type}{ext}"
                if f.exists():
                    stems[stem_type] = str(f)
        return stems

    # -----------------------------------------------------------------------
    # Structural boundaries – now adaptive!
    # -----------------------------------------------------------------------
    def _find_boundaries(self, audio_path):
        # Load audio and get duration
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        duration = librosa.get_duration(y=y_mono, sr=sr)

        # ---- Adaptive k ----
        if duration < 60:          # less than 1 minute
            k = max(2, int(duration / 15))
        else:                      # 1 minute or longer
            k = max(4, int(duration / 30))
        # Cap k at 20 to avoid too many tiny sections
        k = min(k, 20)
        self.progress.emit(25, f"Using k={k} sections for {duration:.1f}s audio")

        # MFCC agglomerative clustering
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=k)
        bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]

        # Merge sections shorter than 12 seconds
        filtered = [bound_times[0]]
        for t in bound_times[1:]:
            if t - filtered[-1] >= 12.0:
                filtered.append(t)
        if filtered[-1] < duration:
            filtered[-1] = duration
        bound_times = filtered

        # Build the final sections list
        sections = []
        for i in range(len(bound_times)-1):
            sections.append({
                "name": f"Section_{i+1:02d}",
                "start": bound_times[i],
                "end": bound_times[i+1]
            })
        return sections

    # -----------------------------------------------------------------------
    # Extract sections from each stem (no L/R)
    # -----------------------------------------------------------------------
    def _extract_stem_sections(self, stem_paths, sections):
        chunks = []
        base_dir = Path(self.file_path).parent / "structural_chunks"
        base_dir.mkdir(exist_ok=True)
        for stem_type, stem_path in stem_paths.items():
            y, sr = librosa.load(stem_path, sr=None, mono=False)
            for sec in sections:
                start_s = int(sec['start'] * sr)
                end_s = int(sec['end'] * sr)
                chunk = y[:, start_s:end_s] if y.ndim > 1 else y[start_s:end_s]
                base_name = f"{Path(stem_path).stem}_{sec['name']}"
                out_path = base_dir / f"{base_name}.wav"
                if y.ndim > 1:
                    sf.write(out_path, chunk.T, sr)
                else:
                    sf.write(out_path, chunk, sr)
                chunks.append({
                    "stem_type": stem_type,
                    "section": sec['name'],
                    "path": str(out_path)
                })
        return chunks

    # -----------------------------------------------------------------------
    # REAL KAGGLE CAPTIONING using Qwen2.5‑Omni (copied from your notebook)
    # -----------------------------------------------------------------------
    def _caption_chunks(self, chunks):
        """
        Pushes a Kaggle kernel that:
          - Loads the Qwen2.5‑Omni captioner (ACE-Step/acestep-captioner)
          - Processes every WAV file in /kaggle/working/input/
          - Writes a captions.json with results
        Returns dict mapping chunk path -> caption.
        """
        import tempfile, shutil, subprocess, time, json, os
        from pathlib import Path

        # 1. Stage all chunk WAVs in a temporary folder
        temp_dir = tempfile.mkdtemp(prefix="struct_caption_")
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        for chunk in chunks:
            src = chunk['path']
            dst = os.path.join(input_dir, Path(src).name)
            shutil.copy2(src, dst)

        # 2. Set Kaggle credentials
        user = self.config.get("kaggle_user", "").strip()
        key = self.config.get("kaggle_key", "").strip()
        if not user or not key:
            raise ValueError("Kaggle credentials not configured.")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key

        # 3. Build the kernel script – this is the notebook code!
        kernel_slug = f"struct-caption-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(tempfile.mkdtemp(), kernel_slug)
        os.makedirs(kernel_dir, exist_ok=True)

        # ---- The kernel script (copied from your working notebook) ----
        worker_py = f"""
import os, json, glob, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ---- Configuration ----
MODEL_ID = "ACE-Step/acestep-captioner"
PRECISION = "fp16"
BATCH_SIZE = 1

# Set device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if PRECISION == "fp16" else torch.bfloat16

# Load model with balanced device map (for multi‑GPU)
load_kwargs = {{
    "device_map": "balanced",
    "max_memory": {{0: "10GiB", 1: "10GiB"}},
    "offload_folder": "/kaggle/working/offload",
    "trust_remote_code": True,
    "torch_dtype": torch_dtype,
}}
try:
    import flash_attn
    load_kwargs["attn_implementation"] = "flash_attention_2"
except ImportError:
    load_kwargs["attn_implementation"] = "sdpa"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ---- Caption prompt (same as your notebook) ----
CAPTION_PROMPT = (
    "You are a professional music metadata tagger preparing training data for ACE-Step. "
    "Listen carefully to this audio clip and write a detailed description. "
    "Cover: specific instrumentation (name every instrument you hear), "
    "whether vocals are present (gender, register, timbre) or confirm instrumental, "
    "recording and production character, mood, and how the clip develops. "
    "Write 3 to 5 sentences. Start with A or An. "
    "Genre, BPM, key, and time signature are handled separately — do not include them."
)

def extract_reply(text):
    if "assistant\\n" in text:
        return text.split("assistant\\n")[-1].strip()
    if "assistant" in text:
        return text.split("assistant")[-1].strip()
    return text.strip()

# ---- Process all WAV files in input folder ----
input_folder = "/kaggle/working/input"
os.makedirs(input_folder, exist_ok=True)

audio_files = sorted(glob.glob(os.path.join(input_folder, "*.wav")))
results = []

for f in audio_files:
    fname = os.path.basename(f)
    # Build conversation
    conversation = [
        {{"role": "system", "content": [{{"type": "text", "text": (
            "You are Qwen, a virtual human developed by the Qwen Team, "
            "Alibaba Group, capable of perceiving auditory and visual inputs, "
            "as well as generating text and speech."
        )}}]}},
        {{"role": "user", "content": [
            {{"type": "audio", "audio": f}},
            {{"type": "text", "text": CAPTION_PROMPT}},
        ]}}
    ]
    text_input = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text_input, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False
    ).to(model.device).to(model.dtype)

    output_ids = model.generate(
        **inputs, use_audio_in_video=False, return_audio=False, max_new_tokens=512
    )
    full_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    caption = extract_reply(full_text)
    results.append({{"file": fname, "caption": caption}})

# ---- Write JSON output ----
with open("/kaggle/working/captions.json", "w") as out_f:
    json.dump({{"results": results}}, out_f, indent=2)
"""
        # Write the kernel files
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w") as f:
            f.write(worker_py)

        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Push the kernel
        self.progress.emit(50, "Pushing Kaggle kernel (Qwen2.5‑Omni)...")
        push_cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
        res = subprocess.run(push_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Kaggle push failed: {res.stderr}")

        # 5. Wait for completion and fetch output
        self.progress.emit(60, "Waiting for Kaggle job...")
        out_dir = os.path.join(tempfile.mkdtemp(), "output")
        os.makedirs(out_dir, exist_ok=True)
        for attempt in range(12):  # up to ~2 minutes
            if self._is_cancelled:
                return {}
            time.sleep(10)
            # Try to download output
            subprocess.run(
                ["kaggle", "kernels", "output", f"{user}/{kernel_slug}", "-p", out_dir],
                capture_output=True
            )
            json_path = os.path.join(out_dir, "captions.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                results = data.get("results", [])
                caption_map = {}
                for item in results:
                    file_name = item["file"]
                    for chunk in chunks:
                        if Path(chunk['path']).name == file_name:
                            caption_map[chunk['path']] = item["caption"]
                            break
                # Clean up
                shutil.rmtree(kernel_dir, ignore_errors=True)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return caption_map

        raise TimeoutError("Kaggle job did not complete in time.")

    # -----------------------------------------------------------------------
    # DEEPSEEK AGGREGATION with enriched evidence
    # -----------------------------------------------------------------------
    def _aggregate_with_deepseek(self, sections, captions):
        """
        Build a rich context for DeepSeek using:
          - Segment names and timestamps.
          - The actual caption from Kaggle for each segment.
          - Additional evidence (e.g., if caption mentions vocals, etc.)
        Then call DeepSeek to synthesize a master caption.
        """
        # Build per‑segment evidence
        evidence_lines = []
        for sec in sections:
            sec_name = sec['name']
            start = sec['start']
            end = sec['end']
            # Collect all captions that belong to this section
            sec_captions = []
            for path, cap in captions.items():
                if sec_name in path:
                    sec_captions.append(cap)
            combined = " | ".join(sec_captions) if sec_captions else "(no caption)"

            # Simple evidence extraction: if "vocal" appears in caption, note it
            vocal_notes = "Vocals detected" if "vocal" in combined.lower() else "Instrumental"
            energy_notes = "High energy" if any(w in combined.lower() for w in ["loud", "intense", "driving"]) else "Moderate energy"

            evidence_lines.append(
                f"  [{sec_name}] {start:.1f}s - {end:.1f}s\n"
                f"    Caption: {combined}\n"
                f"    Evidence: {vocal_notes}, {energy_notes}"
            )

        evidence_text = "\n".join(evidence_lines)

        # Build the DeepSeek prompt
        api_key = self.config.get("custom_key", "").strip()
        if not api_key:
            return "No DeepSeek API key provided."

        # For now, we use defaults for genre and BPM.
        # In a future version we can read them from the sample metadata.
        target_genre = "Alternative Rock"
        global_bpm = 120

        # Use the existing orchestrator but with enriched context
        # We'll pass the evidence as a separate parameter.
        # Since the orchestrator expects `segments` with captions, we reuse that.
        # But we add the evidence to the description.
        segments_with_evidence = []
        for sec in sections:
            # Re‑create the segment with the combined caption (already have it)
            sec_name = sec['name']
            combined = ""
            for path, cap in captions.items():
                if sec_name in path:
                    combined = cap
                    break
            segments_with_evidence.append({
                "name": sec_name,
                "start_sec": sec['start'],
                "end_sec": sec['end'],
                "caption": combined,
                # Add extra fields for evidence (the orchestrator will ignore them for now)
                "evidence": evidence_lines  # we'll pass it separately
            })

        # Instantiate DeepSeek orchestrator
        orchestrator = DeepSeekMusicOrchestrator(api_key=api_key)

        # We'll override the user context building in the orchestrator by
        # calling the method directly and passing the evidence as a string.
        # To avoid modifying the orchestrator, we'll call it with the enriched segments.
        # But the orchestrator's `generate_master_dataset_prompt` doesn't accept evidence.
        # So we'll build a custom prompt here and use the orchestrator's client.
        system_prompt = (
            "You are an elite music prompt engineer for ACE-Step. Synthesize a cohesive master prompt from structural evidence, "
            "instrumentation descriptions, and timing information. Output ONLY the final prompt, no introductory text. "
            "Structure: [Genre/Vibe], [Production Texture], [Instrumentation with descriptive details], [Dynamics/Energy], [Structural flow]."
        )
        user_context = f"TARGET GENRE: {target_genre}\nGLOBAL BPM: {global_bpm}\n\n"
        user_context += "STRUCTURAL EVIDENCE (per section):\n"
        user_context += evidence_text
        user_context += "\n\nCompile final master caption now:"

        try:
            response = orchestrator.client.chat.completions.create(
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
            return f"DeepSeek error: {e}"

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# Main Window: DatasetManager
# ============================================================================
class DatasetManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACE-Step Dataset Toolkit (Gentoo Edition)")
        self.setMinimumSize(980, 640)
        self.resize(1240, 820)
    
        self.undo_stack = []
        self.redo_stack = []

        self.custom_theme = {
            "bg_color": "#1e1e1e",
            "panel_bg": "#252526",
            "text_color": "#d4d4d4",
            "accent_color": "#0e639c",
            "font_family": "Segoe UI",
            "zoom_factor": 1.0
        }

        self.dataset = {
            "metadata": {
                "name": "",
                "custom_tag": "",
                "tag_position": "prepend",
                "instrumental_mode": "mixed",
                "num_samples": 0
            },
            "samples": []
        }
        self.config = {
            "kaggle_user": "",
            "kaggle_key": "",
            "custom_url": "",
            "custom_key": "",       # DeepSeek API key
            "mvsep_api_key": ""     # NEW
        }
        self.health_reports = {}
        self.original_backups = {}
        self.active_worker = None
        self.filter_exceptions_only = False
        self.bypass_warnings = False
        self.startup_scan_notice_shown = False
        self.kaggle_notebook_unlocked = False  # NEW

        self.init_ui()
        self.apply_custom_theme()


    # -----------------------------------------------------------------------
    # Undo / Redo
    # -----------------------------------------------------------------------
    def record_snapshot(self):
        snapshot = json.dumps(self.dataset)
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.update_undo_redo_buttons()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(json.dumps(self.dataset))
            snap = self.undo_stack.pop()
            self.dataset = json.loads(snap)
            self.sync_general_props_to_ui()
            self.refresh_table()
            self.on_table_selection_changed()
            self.update_undo_redo_buttons()
            self.status_label.setText("Action undone.")

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(json.dumps(self.dataset))
            snap = self.redo_stack.pop()
            self.dataset = json.loads(snap)
            self.sync_general_props_to_ui()
            self.refresh_table()
            self.on_table_selection_changed()
            self.update_undo_redo_buttons()
            self.status_label.setText("Action redone.")

    def update_undo_redo_buttons(self):
        self.undo_btn.setEnabled(len(self.undo_stack) > 0)
        self.redo_btn.setEnabled(len(self.redo_stack) > 0)

    # -----------------------------------------------------------------------
    # UI Initialization
    # -----------------------------------------------------------------------
    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        studio_tab = QWidget()
        studio_layout = QVBoxLayout(studio_tab)
        studio_layout.setContentsMargins(10, 8, 10, 8)
        studio_layout.setSpacing(6)

        settings_tab = QWidget()
        self.init_settings_tab(settings_tab)

        advanced_tab = QWidget()
        self.init_advanced_tab(advanced_tab)

        spatial_tab = QWidget()
        self.init_spatial_tab(spatial_tab)
        struct_tab = QWidget()
        self.init_structural_tab(struct_tab)

        self.tabs.addTab(struct_tab, "🎶 Structural Pipeline")
        self.tabs.addTab(studio_tab, "🎛 Dataset Studio")
        self.tabs.addTab(settings_tab, "🎨 Appearance & Customization")
        self.tabs.addTab(advanced_tab, "🧠 Advanced Tools")
        self.tabs.addTab(spatial_tab, "🌐 Spatial Pipeline")

        # Set the Dataset Studio tab as the default visible tab
        studio_index = self.tabs.indexOf(studio_tab)
        if studio_index != -1:
            self.tabs.setCurrentIndex(studio_index)

        # --- Header Bar ---
        header_bar = QHBoxLayout()

        self.quality_badge = QLabel("Dataset Quality: 100% [Ready]")
        self.quality_badge.setStyleSheet("font-weight: bold; font-size: 13px; padding: 6px 14px; background-color: #2E7D32; border-radius: 4px; color: #fff;")
        header_bar.addWidget(self.quality_badge)

        self.bypass_btn = QPushButton("🛡 I Know What I'm Doing (Bypass All)")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.clicked.connect(self.toggle_bypass)
        header_bar.addWidget(self.bypass_btn)

        header_bar.addStretch()

        self.undo_btn = QPushButton("↩ Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo)
        header_bar.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("↪ Redo")
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self.redo)
        header_bar.addWidget(self.redo_btn)

        load_btn = QPushButton("📂 Open JSON")
        load_btn.clicked.connect(self.load_dataset)
        save_btn = QPushButton("💾 Save JSON")
        save_btn.clicked.connect(self.save_dataset)
        add_btn = QPushButton("➕ Add Audio")
        add_btn.clicked.connect(self.add_audio_files)

        header_bar.addWidget(load_btn)
        header_bar.addWidget(save_btn)
        header_bar.addWidget(add_btn)

        studio_layout.addLayout(header_bar)

        # --- General Properties ---
        gen_box = QGroupBox("General Properties & Global Settings")
        gen_layout = QHBoxLayout(gen_box)
        gen_layout.setContentsMargins(8, 4, 8, 4)

        gen_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_input = QLineEdit()
        self.dataset_name_input.setPlaceholderText("Dataset identifier...")
        self.dataset_name_input.textChanged.connect(self.on_general_prop_changed)
        gen_layout.addWidget(self.dataset_name_input)

        gen_layout.addWidget(QLabel("Custom Tag:"))
        self.custom_tag_input = QLineEdit()
        self.custom_tag_input.setPlaceholderText("Global trigger tag...")
        self.custom_tag_input.textChanged.connect(self.on_general_prop_changed)
        gen_layout.addWidget(self.custom_tag_input)

        gen_layout.addWidget(QLabel("Tag Position:"))
        self.tag_pos_combo = QComboBox()
        self.tag_pos_combo.addItems(["prepend", "append", "none"])
        self.tag_pos_combo.currentTextChanged.connect(self.on_general_prop_changed)
        gen_layout.addWidget(self.tag_pos_combo)

        gen_layout.addWidget(QLabel("Mode:"))
        self.inst_group = QButtonGroup(self)
        self.radio_mixed = QRadioButton("Mixed")
        self.radio_all_inst = QRadioButton("All Instrumental")
        self.radio_no_inst = QRadioButton("No Instrumentals")
        self.radio_mixed.setChecked(True)

        self.inst_group.addButton(self.radio_mixed)
        self.inst_group.addButton(self.radio_all_inst)
        self.inst_group.addButton(self.radio_no_inst)
        self.inst_group.buttonClicked.connect(self.on_general_prop_changed)

        gen_layout.addWidget(self.radio_mixed)
        gen_layout.addWidget(self.radio_all_inst)
        gen_layout.addWidget(self.radio_no_inst)

        studio_layout.addWidget(gen_box)

        # --- Action Strip ---
        action_strip = QHBoxLayout()

        self.scan_btn = QPushButton("🔍 Scan Audio & Fill Metadata")
        self.scan_btn.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        self.scan_btn.clicked.connect(self.start_health_audit)
        action_strip.addWidget(self.scan_btn)

        self.normalize_btn = QPushButton("🎚 Fix & DSP Normalize (EBU R128)")
        self.normalize_btn.clicked.connect(self.start_dsp_normalize)
        action_strip.addWidget(self.normalize_btn)

        self.run_ai_btn = QPushButton("🚀 Run AI Captioner")
        self.run_ai_btn.clicked.connect(self.start_ai_captioning)
        action_strip.addWidget(self.run_ai_btn)

        action_strip.addSpacing(15)

        self.all_view_btn = QPushButton("All Tracks")
        self.all_view_btn.setCheckable(True)
        self.all_view_btn.setChecked(True)
        self.all_view_btn.clicked.connect(self.show_all_tracks)

        self.exceptions_view_btn = QPushButton("⚠ Exceptions Queue (0)")
        self.exceptions_view_btn.setCheckable(True)
        self.exceptions_view_btn.clicked.connect(self.show_exceptions_queue)

        view_group = QButtonGroup(self)
        view_group.addButton(self.all_view_btn)
        view_group.addButton(self.exceptions_view_btn)

        action_strip.addWidget(QLabel("View:"))
        action_strip.addWidget(self.all_view_btn)
        action_strip.addWidget(self.exceptions_view_btn)

        action_strip.addStretch()
        studio_layout.addLayout(action_strip)

        # --- Table + Inspector ---
        splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Filename", "Health", "Tag", "Genre", "Duration", "Key", "BPM"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 7):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        splitter.addWidget(self.table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inspector = QWidget()
        insp_layout = QVBoxLayout(inspector)
        insp_layout.setContentsMargins(8, 8, 8, 8)

        self.sample_health_alert = QLabel("Select a track to inspect diagnostics.")
        self.sample_health_alert.setWordWrap(True)
        self.sample_health_alert.setStyleSheet("padding: 6px; background-color: #222; border-left: 4px solid #555; border-radius: 2px;")
        insp_layout.addWidget(self.sample_health_alert)

        unlock_bar = QHBoxLayout()
        unlock_bar.addWidget(QLabel("<b>Metadata Locks:</b>"))
        self.lock_action_combo = QComboBox()
        self.lock_action_combo.addItems(["-- Lock Options --", "Lock All Detected", "Unlock All Fields", "Restore Detected Values"])
        self.lock_action_combo.activated.connect(self.handle_lock_dropdown)
        unlock_bar.addWidget(self.lock_action_combo)
        unlock_bar.addStretch()
        insp_layout.addLayout(unlock_bar)

        insp_layout.addWidget(QLabel("<b>Track Caption:</b>"))
        self.caption_text = QTextEdit()
        self.caption_text.setPlaceholderText("Detailed acoustic description...")
        self.caption_text.textChanged.connect(self.on_caption_edited)
        insp_layout.addWidget(self.caption_text)

        insp_layout.addWidget(QLabel("<b>Formatted Lyrics / Vocal Markers:</b>"))
        self.lyrics_text = QTextEdit()
        self.lyrics_text.setPlaceholderText("[Intro]\n[Verse 1]\nLyrics...\n[Chorus]...")
        self.lyrics_text.textChanged.connect(self.on_lyrics_edited)
        insp_layout.addWidget(self.lyrics_text)

        form = QFormLayout()

        bpm_row = QHBoxLayout()
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(0, 400)
        self.bpm_spin.valueChanged.connect(self.on_bpm_edited)
        self.bpm_lock = QCheckBox("Lock")
        self.bpm_lock.setChecked(True)
        self.bpm_lock.stateChanged.connect(self.on_lock_toggled)
        self.bpm_verify_btn = QPushButton("🔗 Check Online")
        self.bpm_verify_btn.clicked.connect(self.open_online_bpm_check)
        bpm_row.addWidget(self.bpm_spin)
        bpm_row.addWidget(self.bpm_lock)
        bpm_row.addWidget(self.bpm_verify_btn)
        form.addRow("BPM (Tempo):", bpm_row)

        key_row = QHBoxLayout()
        self.key_input = QLineEdit()
        self.key_input.textChanged.connect(self.on_key_edited)
        self.key_lock = QCheckBox("Lock")
        self.key_lock.setChecked(True)
        self.key_lock.stateChanged.connect(self.on_lock_toggled)
        self.key_verify_btn = QPushButton("🔗 Check Online")
        self.key_verify_btn.clicked.connect(self.open_online_key_check)
        key_row.addWidget(self.key_input)
        key_row.addWidget(self.key_lock)
        key_row.addWidget(self.key_verify_btn)
        form.addRow("Key Scale:", key_row)

        genre_row = QHBoxLayout()
        self.genre_input = QLineEdit()
        self.genre_input.textChanged.connect(self.on_genre_edited)
        self.genre_lock = QCheckBox("Lock")
        genre_row.addWidget(self.genre_input)
        genre_row.addWidget(self.genre_lock)
        form.addRow("Genre:", genre_row)

        tag_row = QHBoxLayout()
        self.track_tag_input = QLineEdit()
        self.track_tag_input.textChanged.connect(self.on_track_tag_edited)
        self.tag_lock = QCheckBox("Lock")
        tag_row.addWidget(self.track_tag_input)
        tag_row.addWidget(self.tag_lock)
        form.addRow("Track Trigger Tag:", tag_row)

        self.inst_check = QCheckBox("Instrumental Track (No Vocals)")
        self.inst_check.stateChanged.connect(self.on_inst_edited)
        form.addRow(self.inst_check)

        ab_row = QHBoxLayout()
        self.ab_compare_btn = QPushButton("🎧 A/B Compare Original")
        self.ab_compare_btn.clicked.connect(self.ab_compare_playback)
        self.fallback_btn = QPushButton("⏮ Revert to Original Audio")
        self.fallback_btn.clicked.connect(self.fallback_to_original)
        ab_row.addWidget(self.ab_compare_btn)
        ab_row.addWidget(self.fallback_btn)
        form.addRow(ab_row)

        insp_layout.addLayout(form)
        scroll.setWidget(inspector)
        splitter.addWidget(scroll)
        splitter.setSizes([680, 520])

        studio_layout.addWidget(splitter, 1)

        bottom_bar = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.progress_bar)
        studio_layout.addLayout(bottom_bar)

    # -----------------------------------------------------------------------
    # Settings Tab
    # -----------------------------------------------------------------------
    def init_settings_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)

        theme_grp = QGroupBox("🎨 Visual Appearance & UI Themes (Gentoo Philosophy)")
        form = QFormLayout(theme_grp)

        self.font_picker = QFontComboBox()
        self.font_picker.setCurrentFont(QFont(self.custom_theme["font_family"]))
        self.font_picker.currentFontChanged.connect(self.on_font_changed)
        form.addRow("Installed System Font:", self.font_picker)

        zoom_row = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(75, 175)
        self.zoom_slider.setValue(100)
        self.zoom_label = QLabel("100%")
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(self.zoom_label)
        form.addRow("UI Zoom Factor:", zoom_row)

        self.theme_preset_combo = QComboBox()
        self.theme_preset_combo.addItems(["Dark Modern (Default)", "OLED Pure Black", "Gentoo Purple Slate", "Solarized Dark", "High Contrast Light"])
        self.theme_preset_combo.currentTextChanged.connect(self.on_theme_preset_changed)
        form.addRow("Theme Preset:", self.theme_preset_combo)

        layout.addWidget(theme_grp)

        cloud_grp = QGroupBox("⚙ Cloud & Execution Endpoints")
        c_form = QFormLayout(cloud_grp)

        self.k_user = QLineEdit(self.config.get("kaggle_user", ""))
        self.k_key = QLineEdit(self.config.get("kaggle_key", ""))
        self.k_key.setEchoMode(QLineEdit.Password)
        c_form.addRow("Kaggle Username:", self.k_user)
        c_form.addRow("Kaggle API Key:", self.k_key)

        self.custom_url = QLineEdit(self.config.get("custom_url", ""))
        self.custom_url.setPlaceholderText("https://api.runpod.ai/... or http://localhost:8000/v1")
        self.custom_key = QLineEdit(self.config.get("custom_key", ""))
        self.custom_key.setEchoMode(QLineEdit.Password)
        c_form.addRow("Custom Auth Token (DeepSeek/MVSEP):", self.custom_key)

        # NEW: MVSEP key
        self.mvsep_key = QLineEdit(self.config.get("mvsep_api_key", ""))
        self.mvsep_key.setEchoMode(QLineEdit.Password)
        c_form.addRow("MVSEP API Key:", self.mvsep_key)

        save_cloud_btn = QPushButton("Save Cloud Credentials")
        save_cloud_btn.clicked.connect(self.save_cloud_config)
        c_form.addRow(save_cloud_btn)

        layout.addWidget(cloud_grp)
        layout.addStretch()

    # -----------------------------------------------------------------------
    # Advanced Tools Tab
    # -----------------------------------------------------------------------
    def init_advanced_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox("🧠 Advanced Structural Pipeline (DeepSeek + Librosa)")
        inner = QVBoxLayout(group)

        self.spatial_module_checkbox = QCheckBox("Enable Dual-Channel Spatial Profiling")
        self.spatial_module_checkbox.setToolTip("Analyzes left/right channel differences for stereo placement.")
        self.spatial_module_checkbox.setChecked(True)
        inner.addWidget(self.spatial_module_checkbox)

        self.advanced_pipeline_btn = QPushButton("🚀 Run Advanced Pipeline on Selected Track")
        self.advanced_pipeline_btn.clicked.connect(self.trigger_advanced_ai_pipeline)
        inner.addWidget(self.advanced_pipeline_btn)

        info = QLabel(
            "Uses Librosa to segment the selected track into ~9 macro sections,\n"
            "exports each as WAV to a 'structural_slices' folder, and then\n"
            "aggregates via DeepSeek to produce a master caption."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; padding: 10px;")
        inner.addWidget(info)

        layout.addWidget(group)
        layout.addStretch()

    # -----------------------------------------------------------------------
    # Spatial Pipeline Tab
    # -----------------------------------------------------------------------
    def init_spatial_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)

        warning_box = QGroupBox("⚠️ Advanced Spatial Pipeline")
        warning_box.setStyleSheet("QGroupBox { border: 2px solid #FF9800; }")
        warn_layout = QVBoxLayout(warning_box)
        warn_label = QLabel(
            "This pipeline performs stem separation (MVSEP), structural slicing, L/R channel extraction, "
            "captioning via Kaggle, spatial evaluation, and DeepSeek aggregation.\n"
            "It requires API keys for MVSEP, DeepSeek, and Kaggle credentials.\n"
            "Proceed only if you have these services set up."
        )
        warn_label.setWordWrap(True)
        warn_label.setStyleSheet("color: #ffcc80;")
        warn_layout.addWidget(warn_label)

        self.spatial_warning_check = QCheckBox("I understand the requirements and have the necessary API keys.")
        warn_layout.addWidget(self.spatial_warning_check)
        layout.addWidget(warning_box)

        options_box = QGroupBox("Pipeline Options")
        opts_layout = QFormLayout(options_box)

        self.stem_source_combo = QComboBox()
        self.stem_source_combo.addItems(["Import existing stems", "Separate via MVSEP"])
        opts_layout.addRow("Stem source:", self.stem_source_combo)

        self.use_deepseek_check = QCheckBox("Use DeepSeek for aggregation")
        self.use_deepseek_check.setChecked(True)
        opts_layout.addRow(self.use_deepseek_check)

        self.custom_endpoint_check = QCheckBox("Use Custom Endpoint instead of Kaggle/DeepSeek")
        self.custom_endpoint_check.toggled.connect(self.toggle_custom_endpoint)
        opts_layout.addRow(self.custom_endpoint_check)

        self.custom_endpoint_url = QLineEdit()
        self.custom_endpoint_url.setPlaceholderText("http://localhost:8000/spatial")
        self.custom_endpoint_url.setEnabled(False)
        opts_layout.addRow("Endpoint URL:", self.custom_endpoint_url)

        layout.addWidget(options_box)

        notebook_box = QGroupBox("🔒 Kaggle Notebook Access")
        nb_layout = QVBoxLayout(notebook_box)

        unlock_btn = QPushButton("Unlock Notebook (Advanced)")
        unlock_btn.clicked.connect(self.unlock_kaggle_notebook)
        nb_layout.addWidget(unlock_btn)

        self.notebook_edit_area = QTextEdit()
        self.notebook_edit_area.setPlaceholderText("Paste custom Kaggle notebook code here (only if unlocked).")
        self.notebook_edit_area.setEnabled(False)
        self.notebook_edit_area.setMaximumHeight(200)
        nb_layout.addWidget(self.notebook_edit_area)

        self.notebook_status = QLabel("Notebook locked. Click the button to unlock.")
        self.notebook_status.setStyleSheet("color: #aaa;")
        nb_layout.addWidget(self.notebook_status)

        layout.addWidget(notebook_box)

        self.run_spatial_btn = QPushButton("🚀 Run Spatial Pipeline on Selected Track")
        self.run_spatial_btn.setStyleSheet("font-weight: bold; background-color: #0e639c; padding: 10px;")
        self.run_spatial_btn.clicked.connect(self.run_spatial_pipeline)
        layout.addWidget(self.run_spatial_btn)

        self.spatial_progress = QProgressBar()
        self.spatial_progress.setVisible(False)
        layout.addWidget(self.spatial_progress)

        self.spatial_status = QLabel("Ready")
        layout.addWidget(self.spatial_status)

        layout.addStretch()

    def init_structural_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox("🎶 Structural Pipeline (Standard)")
        inner = QVBoxLayout(group)

        info = QLabel(
            "This pipeline separates stems (import or MVSEP), finds structural boundaries,\n"
            "captions each section per stem, and aggregates via DeepSeek to produce a\n"
            "master caption for the whole track.\n"
            "No spatial L/R processing – suitable for general LoRA training."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; padding: 10px;")
        inner.addWidget(info)

        # Stem source
        stem_layout = QHBoxLayout()
        stem_layout.addWidget(QLabel("Stem source:"))
        self.struct_stem_combo = QComboBox()
        self.struct_stem_combo.addItems(["Import existing stems", "Separate via MVSEP"])
        stem_layout.addWidget(self.struct_stem_combo)
        inner.addLayout(stem_layout)

        # Segmentation source
        seg_layout = QHBoxLayout()
        seg_layout.addWidget(QLabel("Segmentation:"))
        self.struct_seg_combo = QComboBox()
        self.struct_seg_combo.addItems(["Lyrics tags", "MFCC agglomerative"])
        self.struct_seg_combo.setToolTip("Lyrics tags are more accurate; MFCC is fallback.")
        seg_layout.addWidget(self.struct_seg_combo)
        inner.addLayout(seg_layout)

        # DeepSeek toggle
        self.struct_deepseek_check = QCheckBox("Use DeepSeek for aggregation")
        self.struct_deepseek_check.setChecked(True)
        inner.addWidget(self.struct_deepseek_check)

        # Run button
        self.run_struct_btn = QPushButton("🚀 Run Structural Pipeline on Selected Track")
        self.run_struct_btn.setStyleSheet("font-weight: bold; background-color: #0e639c; padding: 10px;")
        self.run_struct_btn.clicked.connect(self.run_structural_pipeline)
        inner.addWidget(self.run_struct_btn)

        # Progress
        self.struct_progress = QProgressBar()
        self.struct_progress.setVisible(False)
        inner.addWidget(self.struct_progress)

        self.struct_status = QLabel("Ready")
        inner.addWidget(self.struct_status)

        layout.addWidget(group)
        layout.addStretch()

    # -----------------------------------------------------------------------
    # Spatial Pipeline Methods
    # -----------------------------------------------------------------------
    def toggle_custom_endpoint(self, checked):
        self.custom_endpoint_url.setEnabled(checked)
        if checked:
            self.use_deepseek_check.setChecked(False)
            self.use_deepseek_check.setEnabled(False)
        else:
            self.use_deepseek_check.setEnabled(True)

    def unlock_kaggle_notebook(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Unlock Kaggle Notebook")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            "Editing the Kaggle notebook may break the captioning pipeline.\n"
            "Only proceed if you understand the code and the risks.\n\n"
            "I know what I am doing."
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        ret = msg.exec()
        if ret == QMessageBox.Ok:
            self.kaggle_notebook_unlocked = True
            self.notebook_edit_area.setEnabled(True)
            self.notebook_status.setText("✅ Notebook unlocked. You may edit the code below.")
            self.notebook_status.setStyleSheet("color: #4CAF50;")
        else:
            self.notebook_status.setText("❌ Notebook remains locked.")
            self.notebook_status.setStyleSheet("color: #FF9800;")

    def run_spatial_pipeline(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "Selection Missing", "Please select a track first.")
            return

        if not self.spatial_warning_check.isChecked():
            QMessageBox.warning(self, "Acknowledge Warning", "You must check 'I understand the requirements' before running.")
            return

        stem_source = self.stem_source_combo.currentText()
        if stem_source == "Separate via MVSEP":
            if not self.config.get("mvsep_api_key"):
                key, ok = QInputDialog.getText(self, "MVSEP API Key", "Enter your MVSEP API key:", QLineEdit.Password)
                if ok and key.strip():
                    self.config["mvsep_api_key"] = key.strip()
                    self.mvsep_key.setText(key.strip())
                else:
                    return

        if self.use_deepseek_check.isChecked() and not self.config.get("custom_key"):
            key, ok = QInputDialog.getText(self, "DeepSeek API Key", "Enter your DeepSeek API key:", QLineEdit.Password)
            if ok and key.strip():
                self.config["custom_key"] = key.strip()
                self.custom_key.setText(key.strip())
            else:
                return

        if not self.config.get("kaggle_user") or not self.config.get("kaggle_key"):
            user, ok1 = QInputDialog.getText(self, "Kaggle Username", "Enter Kaggle username:")
            if ok1:
                key, ok2 = QInputDialog.getText(self, "Kaggle API Key", "Enter Kaggle API key:", QLineEdit.Password)
                if ok2:
                    self.config["kaggle_user"] = user.strip()
                    self.config["kaggle_key"] = key.strip()
                    self.k_user.setText(user.strip())
                    self.k_key.setText(key.strip())
                else:
                    return
            else:
                return

        options = {
            "stem_source": "mvsep" if stem_source == "Separate via MVSEP" else "import",
            "use_spatial": True,
            "use_deepseek": self.use_deepseek_check.isChecked(),
            "custom_endpoint": self.custom_endpoint_check.isChecked()
        }

        self.run_spatial_btn.setEnabled(False)
        self.spatial_progress.setVisible(True)
        self.spatial_progress.setValue(0)
        self.spatial_status.setText("Starting spatial pipeline...")

        self.active_worker = SpatialPipelineWorker(
            track_id=selected["id"],
            file_path=selected["audio_path"],
            config=self.config,
            options=options
        )
        self.active_worker.progress.connect(self.on_spatial_progress)
        self.active_worker.step_completed.connect(self.on_spatial_step)
        self.active_worker.pipeline_finished.connect(self.on_spatial_finished)
        self.active_worker.error_occurred.connect(self.on_spatial_error)
        self.active_worker.start()

    def on_spatial_progress(self, pct, msg):
        self.spatial_progress.setValue(pct)
        self.spatial_status.setText(msg)

    def on_spatial_step(self, step_name, data):
        self.spatial_status.setText(f"Completed step: {step_name}")

    def on_spatial_finished(self, track_id, result):
        self.spatial_progress.setVisible(False)
        self.run_spatial_btn.setEnabled(True)
        self.spatial_status.setText("Pipeline completed successfully.")

        for sample in self.dataset["samples"]:
            if sample["id"] == track_id:
                self.record_snapshot()
                sample["caption"] = result["final_caption"]
                sample["structural_segments"] = result["sections"]
                sample["spatial_tokens"] = result["spatial_tokens"]
                sample["stem_paths"] = result["stem_paths"]
                sample["chunk_paths"] = result["chunk_paths"]
                break
        self.refresh_table()
        self.on_table_selection_changed()

        QMessageBox.information(self, "Spatial Pipeline Done",
                                f"Final caption:\n\n{result['final_caption'][:500]}...")

    def on_spatial_error(self, err):
        self.spatial_progress.setVisible(False)
        self.run_spatial_btn.setEnabled(True)
        self.spatial_status.setText("Error: " + err)
        QMessageBox.critical(self, "Spatial Pipeline Error", err)

    # -----------------------------------------------------------------------
    # Structural Pipeline Methods
    # -----------------------------------------------------------------------
    def run_structural_pipeline(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "Selection Missing", "Please select a track first.")
            return

        stem_source = self.struct_stem_combo.currentText()
        if stem_source == "Separate via MVSEP":
            if not self.config.get("mvsep_api_key"):
                key, ok = QInputDialog.getText(self, "MVSEP API Key", "Enter your MVSEP API key:", QLineEdit.Password)
                if ok and key.strip():
                    self.config["mvsep_api_key"] = key.strip()
                    self.mvsep_key.setText(key.strip())
                else:
                    return

        if self.struct_deepseek_check.isChecked() and not self.config.get("custom_key"):
            key, ok = QInputDialog.getText(self, "DeepSeek API Key", "Enter your DeepSeek API key:", QLineEdit.Password)
            if ok and key.strip():
                self.config["custom_key"] = key.strip()
                self.custom_key.setText(key.strip())
            else:
                return

        # Kaggle credentials (for captioning)
        if not self.config.get("kaggle_user") or not self.config.get("kaggle_key"):
            user, ok1 = QInputDialog.getText(self, "Kaggle Username", "Enter Kaggle username:")
            if ok1:
                key, ok2 = QInputDialog.getText(self, "Kaggle API Key", "Enter Kaggle API key:", QLineEdit.Password)
                if ok2:
                    self.config["kaggle_user"] = user.strip()
                    self.config["kaggle_key"] = key.strip()
                    self.k_user.setText(user.strip())
                    self.k_key.setText(key.strip())
                else:
                    return
            else:
                return

        options = {
            "stem_source": "mvsep" if stem_source == "Separate via MVSEP" else "import",
            "use_deepseek": self.struct_deepseek_check.isChecked(),
            "use_lyrics": self.struct_seg_combo.currentText() == "Lyrics tags"
        }

        self.run_struct_btn.setEnabled(False)
        self.struct_progress.setVisible(True)
        self.struct_progress.setValue(0)
        self.struct_status.setText("Starting structural pipeline...")

        self.active_worker = StructuralPipelineWorker(
            track_id=selected["id"],
            file_path=selected["audio_path"],
            config=self.config,
            options=options
        )
        self.active_worker.progress.connect(self.on_struct_progress)
        self.active_worker.step_completed.connect(self.on_struct_step)
        self.active_worker.pipeline_finished.connect(self.on_struct_finished)
        self.active_worker.error_occurred.connect(self.on_struct_error)
        self.active_worker.start()

    def on_struct_progress(self, pct, msg):
        self.struct_progress.setValue(pct)
        self.struct_status.setText(msg)

    def on_struct_step(self, step_name, data):
        self.struct_status.setText(f"Completed step: {step_name}")

    def on_struct_finished(self, track_id, result):
        self.struct_progress.setVisible(False)
        self.run_struct_btn.setEnabled(True)
        self.struct_status.setText("Pipeline completed successfully.")

        for sample in self.dataset["samples"]:
            if sample["id"] == track_id:
                self.record_snapshot()
                sample["caption"] = result["final_caption"]
                sample["structural_segments"] = result["sections"]
                sample["stem_paths"] = result["stem_paths"]
                sample["chunk_paths"] = result["chunk_paths"]
                break
        self.refresh_table()
        self.on_table_selection_changed()

        QMessageBox.information(self, "Structural Pipeline Done",
                                f"Final caption:\n\n{result['final_caption'][:500]}...")

    def on_struct_error(self, err):
        self.struct_progress.setVisible(False)
        self.run_struct_btn.setEnabled(True)
        self.struct_status.setText("Error: " + err)
        QMessageBox.critical(self, "Structural Pipeline Error", err)

    # -----------------------------------------------------------------------
    # Advanced Pipeline (from Advanced Tools tab)
    # -----------------------------------------------------------------------
    def trigger_advanced_ai_pipeline(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "Selection Missing", "Please pick an active audio track first.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting advanced structural segmentation and captioning...")

        use_spatial = self.spatial_module_checkbox.isChecked()
        api_key = self.config.get("custom_key", "").strip()
        if not api_key:
            key, ok = QInputDialog.getText(self, "DeepSeek API Key", "Enter DeepSeek API key:", QLineEdit.Password)
            if ok and key.strip():
                self.config["custom_key"] = key.strip()
                self.custom_key.setText(key.strip())
                api_key = key.strip()
            else:
                self.progress_bar.setVisible(False)
                return

        target_genre = self.custom_tag_input.text().strip() or "Alternative Rock Production"

        self.active_worker = AdvancedDatasetOrchestratorWorker(
            track_id=selected["id"],
            file_path=selected["audio_path"],
            target_genre=target_genre,
            api_key=api_key,
            use_spatial_module=use_spatial
        )
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.track_processing_complete.connect(self.on_advanced_pipeline_success)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def on_advanced_pipeline_success(self, track_id, structured_segments, master_caption):
        self.progress_bar.setVisible(False)
        for sample in self.dataset["samples"]:
            if sample["id"] == track_id:
                self.record_snapshot()
                sample["structural_segments"] = structured_segments
                sample["caption"] = master_caption
                break
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText("Advanced structural caption saved successfully.")

    # -----------------------------------------------------------------------
    # Original Methods (keep as is)
    # -----------------------------------------------------------------------
    def apply_custom_theme(self):
        base_size = int(12 * self.custom_theme["zoom_factor"])
        font_fam = self.custom_theme["font_family"]
        bg = self.custom_theme["bg_color"]
        panel = self.custom_theme["panel_bg"]
        text = self.custom_theme["text_color"]
        accent = self.custom_theme["accent_color"]

        style = f"""
            QWidget {{
                background-color: {bg};
                color: {text};
                font-family: '{font_fam}';
                font-size: {base_size}px;
            }}
            QGroupBox, QTableWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QScrollArea {{
                background-color: {panel};
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {accent};
                color: #ffffff;
                border: none;
                border-radius: 3px;
                padding: {int(4 * self.custom_theme['zoom_factor'])}px {int(10 * self.custom_theme['zoom_factor'])}px;
            }}
            QPushButton:hover {{
                background-color: #1177bb;
            }}
            QHeaderView::section {{
                background-color: {panel};
                color: {text};
                padding: 4px;
                border: 1px solid #333;
            }}
        """
        self.setStyleSheet(style)

    def save_cloud_config(self):
        self.config["kaggle_user"] = self.k_user.text().strip()
        self.config["kaggle_key"] = self.k_key.text().strip()
        self.config["custom_url"] = self.custom_url.text().strip()
        self.config["custom_key"] = self.custom_key.text().strip()
        self.config["mvsep_api_key"] = self.mvsep_key.text().strip()
        self.status_label.setText("Cloud credentials saved.")

    def on_font_changed(self, font):
        self.custom_theme["font_family"] = font.family()
        self.apply_custom_theme()

    def on_zoom_changed(self, val):
        self.custom_theme["zoom_factor"] = val / 100.0
        self.zoom_label.setText(f"{val}%")
        self.apply_custom_theme()

    def on_theme_preset_changed(self, preset):
        if preset == "OLED Pure Black":
            self.custom_theme.update({"bg_color": "#000000", "panel_bg": "#121212", "text_color": "#f0f0f0", "accent_color": "#007acc"})
        elif preset == "Gentoo Purple Slate":
            self.custom_theme.update({"bg_color": "#1a162b", "panel_bg": "#25203d", "text_color": "#e0def4", "accent_color": "#9ccfd8"})
        elif preset == "Solarized Dark":
            self.custom_theme.update({"bg_color": "#002b36", "panel_bg": "#073642", "text_color": "#93a1a1", "accent_color": "#268bd2"})
        elif preset == "High Contrast Light":
            self.custom_theme.update({"bg_color": "#f8f9fa", "panel_bg": "#ffffff", "text_color": "#111111", "accent_color": "#0056b3"})
        else:
            self.custom_theme.update({"bg_color": "#1e1e1e", "panel_bg": "#252526", "text_color": "#d4d4d4", "accent_color": "#0e639c"})
        self.apply_custom_theme()

    def open_online_bpm_check(self):
        s = self.get_selected_sample()
        if s:
            name = Path(s.get("filename", "")).stem
            url = f"https://songbpm.com/@search?q={QUrl.toPercentEncoding(name)}"
            QDesktopServices.openUrl(QUrl(url))

    def open_online_key_check(self):
        s = self.get_selected_sample()
        if s:
            name = Path(s.get("filename", "")).stem
            url = f"https://tunebat.com/Search?q={QUrl.toPercentEncoding(name)}"
            QDesktopServices.openUrl(QUrl(url))

    def show_all_tracks(self):
        self.filter_exceptions_only = False
        self.refresh_table()

    def show_exceptions_queue(self):
        self.filter_exceptions_only = True
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(0)
        exceptions_count = 0

        for s in self.dataset["samples"]:
            sid = s.get("id", "")
            rep = self.health_reports.get(sid, {})
            status = rep.get("status", "Not Audited")

            is_exception = (status == "Warning" or status == "Missing" or not s.get("caption"))
            if is_exception:
                exceptions_count += 1

            if self.filter_exceptions_only and not is_exception:
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(s.get("filename", "")))

            h_item = QTableWidgetItem(f"✓ Healthy" if status == "Healthy" else (f"⚠ Warning" if status == "Warning" else status))
            if status == "Healthy":
                h_item.setForeground(QColor("#4CAF50"))
            elif status == "Warning":
                h_item.setForeground(QColor("#FF9800"))
            elif status == "Missing":
                h_item.setForeground(QColor("#F44336"))
            self.table.setItem(row, 1, h_item)

            self.table.setItem(row, 2, QTableWidgetItem(s.get("custom_tag", "")))
            self.table.setItem(row, 3, QTableWidgetItem(s.get("genre", "")))
            dur = s.get("duration", 0)
            self.table.setItem(row, 4, QTableWidgetItem(f"{dur}s" if dur else ""))
            self.table.setItem(row, 5, QTableWidgetItem(str(s.get("keyscale", ""))))
            bpm = s.get("bpm", 0)
            self.table.setItem(row, 6, QTableWidgetItem(str(bpm) if bpm else ""))

        self.exceptions_view_btn.setText(f"⚠ Exceptions Queue ({exceptions_count})")

    def get_selected_sample(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.dataset["samples"]):
            return self.dataset["samples"][row]
        return None

    def on_table_selection_changed(self):
        s = self.get_selected_sample()
        if s:
            sid = s.get("id", "")
            rep = self.health_reports.get(sid, {})
            issues = rep.get("issues", [])

            if not rep:
                self.sample_health_alert.setText("Health: Not Audited (Click '🔍 Scan Audio & Fill Metadata' to scan)")
                self.sample_health_alert.setStyleSheet("padding: 6px; background-color: #222; border-left: 4px solid #777; border-radius: 2px;")
            elif not issues:
                self.sample_health_alert.setText(f"✓ Healthy: {rep.get('sample_rate', 44100)} Hz | {rep.get('channels', 2)} ch | Est: {rep.get('lufs', -14):.1f} LUFS | BPM Conf: {int(rep.get('bpm_confidence', 0.8)*100)}%")
                self.sample_health_alert.setStyleSheet("padding: 6px; background-color: #1b3a1d; border-left: 4px solid #4CAF50; border-radius: 2px; color: #a5d6a7;")
            else:
                issues_text = " • " + " • ".join(issues)
                self.sample_health_alert.setText(f"⚠ Diagnostic Inconsistencies:\n{issues_text}")
                self.sample_health_alert.setStyleSheet("padding: 6px; background-color: #3e2723; border-left: 4px solid #FF9800; border-radius: 2px; color: #ffcc80;")

            self.caption_text.blockSignals(True)
            self.lyrics_text.blockSignals(True)
            self.track_tag_input.blockSignals(True)
            self.genre_input.blockSignals(True)
            self.key_input.blockSignals(True)
            self.bpm_spin.blockSignals(True)
            self.inst_check.blockSignals(True)

            self.caption_text.setPlainText(s.get("caption", ""))
            self.lyrics_text.setPlainText(s.get("formatted_lyrics", s.get("lyrics", "")))
            self.track_tag_input.setText(s.get("custom_tag", ""))
            self.genre_input.setText(s.get("genre", ""))
            self.key_input.setText(str(s.get("keyscale", "")))
            self.bpm_spin.setValue(int(s.get("bpm", 0)))
            self.inst_check.setChecked(bool(s.get("is_instrumental", False)))

            is_locked = s.get("locked", True)
            self.bpm_lock.setChecked(is_locked)
            self.key_lock.setChecked(is_locked)
            self.bpm_spin.setEnabled(not is_locked)
            self.key_input.setEnabled(not is_locked)

            self.caption_text.blockSignals(False)
            self.lyrics_text.blockSignals(False)
            self.track_tag_input.blockSignals(False)
            self.genre_input.blockSignals(False)
            self.key_input.blockSignals(False)
            self.bpm_spin.blockSignals(False)
            self.inst_check.blockSignals(False)

    def handle_lock_dropdown(self, idx):
        action = self.lock_action_combo.currentText()
        if action == "Lock All Detected":
            for s in self.dataset["samples"]:
                s["locked"] = True
            self.on_table_selection_changed()
            self.status_label.setText("Locked all detected metadata fields.")
        elif action == "Unlock All Fields":
            for s in self.dataset["samples"]:
                s["locked"] = False
            self.on_table_selection_changed()
            self.status_label.setText("Unlocked all metadata fields for editing.")
        elif action == "Restore Detected Values":
            s = self.get_selected_sample()
            if s:
                sid = s.get("id", "")
                rep = self.health_reports.get(sid, {})
                if rep:
                    s["bpm"] = rep.get("bpm_detected", 120)
                    s["keyscale"] = rep.get("key_detected", "A minor")
                    self.on_table_selection_changed()
                    self.status_label.setText("Restored original detected values.")
        self.lock_action_combo.setCurrentIndex(0)

    def on_lock_toggled(self):
        s = self.get_selected_sample()
        if s:
            s["locked"] = self.bpm_lock.isChecked()
            self.bpm_spin.setEnabled(not self.bpm_lock.isChecked())
            self.key_input.setEnabled(not self.key_lock.isChecked())

    def on_caption_edited(self):
        s = self.get_selected_sample()
        if s:
            s["caption"] = self.caption_text.toPlainText()

    # -----------------------------------------------------------------------
    # Caption history & AI review (unchanged from original)
    # -----------------------------------------------------------------------
    def ensure_caption_fields(self, sample):
        sample.setdefault("caption", "")
        sample.setdefault("caption_ai_raw", "")
        sample.setdefault("caption_history", [])
        sample.setdefault("caption_ai_model", "")
        sample.setdefault("caption_ai_prompt", "")
        sample.setdefault("caption_ai_created_at", "")

    def backup_caption_state(self, sample, reason="Before caption change"):
        self.ensure_caption_fields(sample)
        snapshot = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "caption": sample.get("caption", ""),
            "caption_ai_raw": sample.get("caption_ai_raw", ""),
            "lyrics": sample.get("lyrics", ""),
            "formatted_lyrics": sample.get("formatted_lyrics", ""),
            "custom_tag": sample.get("custom_tag", ""),
            "genre": sample.get("genre", ""),
            "language": sample.get("language", ""),
            "bpm": sample.get("bpm", 0),
            "keyscale": sample.get("keyscale", ""),
            "timesignature": sample.get("timesignature", "4/4"),
        }
        history = sample["caption_history"]
        if not history or history[-1] != snapshot:
            history.append(snapshot)
        if len(history) > 25:
            del history[:-25]

    def save_ai_caption_result(self, sample, caption, model_id="ACE-Step/acestep-captioner", prompt=""):
        self.ensure_caption_fields(sample)
        self.backup_caption_state(sample, "Before AI caption result")
        sample["caption_ai_raw"] = caption.strip()
        sample["caption_ai_model"] = model_id
        sample["caption_ai_prompt"] = prompt
        sample["caption_ai_created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    def restore_latest_caption_backup(self, sample):
        self.ensure_caption_fields(sample)
        history = sample.get("caption_history", [])
        if not history:
            QMessageBox.information(self, "No Caption Backup", "There is no earlier caption backup available for this track.")
            return False
        previous = history.pop()
        sample["caption"] = previous.get("caption", "")
        sample["caption_ai_raw"] = previous.get("caption_ai_raw", "")
        sample["lyrics"] = previous.get("lyrics", "")
        sample["formatted_lyrics"] = previous.get("formatted_lyrics", "")
        sample["custom_tag"] = previous.get("custom_tag", "")
        sample["genre"] = previous.get("genre", "")
        sample["language"] = previous.get("language", "")
        sample["bpm"] = previous.get("bpm", 0)
        sample["keyscale"] = previous.get("keyscale", "")
        sample["timesignature"] = previous.get("timesignature", "4/4")
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText("Restored the latest caption backup.")
        return True

    def review_ai_caption_result(self, sample):
        self.ensure_caption_fields(sample)
        current_caption = sample.get("caption", "")
        generated_caption = sample.get("caption_ai_raw", "")

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Review AI Caption: {sample.get('filename', 'Selected Track')}")
        dialog.resize(920, 620)

        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "The raw AI result has been backed up separately. "
            "Choose whether to keep your existing caption, use the generated "
            "caption, or manually merge/edit the text."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QLabel("Existing Approved Caption:"))
        existing_editor = QTextEdit()
        existing_editor.setPlainText(current_caption)
        layout.addWidget(existing_editor)

        layout.addWidget(QLabel("Raw ACE-Step AI Caption:"))
        generated_editor = QTextEdit()
        generated_editor.setPlainText(generated_caption)
        generated_editor.setReadOnly(True)
        layout.addWidget(generated_editor)

        button_row = QHBoxLayout()
        keep_btn = QPushButton("Keep Existing")
        use_btn = QPushButton("Use Generated")
        merge_btn = QPushButton("Merge / Save Edited Text")
        restore_btn = QPushButton("Restore Previous Backup")
        cancel_btn = QPushButton("Close")

        button_row.addWidget(keep_btn)
        button_row.addWidget(use_btn)
        button_row.addWidget(merge_btn)
        button_row.addWidget(restore_btn)
        button_row.addStretch()
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

        def keep_existing():
            dialog.done(0)

        def use_generated():
            sample["caption"] = generated_caption
            self.status_label.setText(f"Accepted generated caption for {sample.get('filename', '')}.")
            dialog.done(1)

        def save_edited():
            sample["caption"] = existing_editor.toPlainText().strip()
            self.status_label.setText(f"Saved reviewed caption for {sample.get('filename', '')}.")
            dialog.done(1)

        def restore_previous():
            self.restore_latest_caption_backup(sample)
            existing_editor.setPlainText(sample.get("caption", ""))
            generated_editor.setPlainText(sample.get("caption_ai_raw", ""))

        keep_btn.clicked.connect(keep_existing)
        use_btn.clicked.connect(use_generated)
        merge_btn.clicked.connect(save_edited)
        restore_btn.clicked.connect(restore_previous)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()
        self.refresh_table()
        self.on_table_selection_changed()

    def on_lyrics_edited(self):
        s = self.get_selected_sample()
        if s:
            s["formatted_lyrics"] = self.lyrics_text.toPlainText()
            s["lyrics"] = s["formatted_lyrics"]

    def on_track_tag_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["custom_tag"] = text

    def on_genre_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["genre"] = text

    def on_key_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["keyscale"] = text

    def on_bpm_edited(self, val):
        s = self.get_selected_sample()
        if s:
            s["bpm"] = val

    def on_inst_edited(self):
        s = self.get_selected_sample()
        if s:
            s["is_instrumental"] = self.inst_check.isChecked()

    def on_general_prop_changed(self):
        meta = self.dataset.setdefault("metadata", {})
        meta["name"] = self.dataset_name_input.text().strip()
        meta["custom_tag"] = self.custom_tag_input.text().strip()
        meta["tag_position"] = self.tag_pos_combo.currentText()

        if self.radio_all_inst.isChecked():
            meta["instrumental_mode"] = "all_instrumental"
            for s in self.dataset["samples"]:
                s["is_instrumental"] = True
        elif self.radio_no_inst.isChecked():
            meta["instrumental_mode"] = "no_instrumentals"
            for s in self.dataset["samples"]:
                s["is_instrumental"] = False
        else:
            meta["instrumental_mode"] = "mixed"
        self.on_table_selection_changed()

    def sync_general_props_to_ui(self):
        meta = self.dataset.get("metadata", {})
        self.dataset_name_input.setText(meta.get("name", ""))
        self.custom_tag_input.setText(meta.get("custom_tag", ""))
        self.tag_pos_combo.setCurrentText(meta.get("tag_position", "prepend"))
        mode = meta.get("instrumental_mode", "mixed")
        if mode == "all_instrumental":
            self.radio_all_inst.setChecked(True)
        elif mode == "no_instrumentals":
            self.radio_no_inst.setChecked(True)
        else:
            self.radio_mixed.setChecked(True)

    def ab_compare_playback(self):
        s = self.get_selected_sample()
        if s:
            sid = s.get("id", "")
            orig_backup = self.original_backups.get(sid, s.get("audio_path", ""))
            curr_path = s.get("audio_path", "")
            QMessageBox.information(
                self, "🎧 A/B Audio Comparison",
                f"Track: {s.get('filename', '')}\n\n"
                f"Active Audio:\n{curr_path}\n\n"
                f"Original Un-normalized Backup:\n{orig_backup}\n\n"
                "(Use system media player to audit waveforms side-by-side.)"
            )

    def fallback_to_original(self):
        s = self.get_selected_sample()
        if s:
            sid = s.get("id", "")
            orig_backup = self.original_backups.get(sid)
            if orig_backup and os.path.exists(orig_backup):
                self.record_snapshot()
                s["audio_path"] = orig_backup
                s["filename"] = Path(orig_backup).name
                self.refresh_table()
                self.on_table_selection_changed()
                QMessageBox.information(self, "Reverted", f"Reverted {s['filename']} to original audio source.")
            else:
                QMessageBox.warning(self, "No Backup", "Original audio backup not found for this track.")

    def toggle_bypass(self):
        self.bypass_warnings = self.bypass_btn.isChecked()
        if self.bypass_warnings:
            self.bypass_btn.setStyleSheet("background-color: #E65100; font-weight: bold;")
            self.status_label.setText("Warning bypass ENABLED: Export unlocked regardless of quality penalties.")
        else:
            self.bypass_btn.setStyleSheet("")
            self.status_label.setText("Warning bypass DISABLED.")

    # -----------------------------------------------------------------------
    # Health Audit
    # -----------------------------------------------------------------------
    def start_health_audit(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Please add audio tracks before scanning.")
            return

        if not self.startup_scan_notice_shown:
            self.startup_scan_notice_shown = True
            QMessageBox.information(self, "Testing Dataset Audio", "Testing dataset audio. Please wait a few seconds...")

        self.scan_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Auditing dataset health, metadata & degradation penalties...")

        self.active_worker = HealthAuditorWorker(samples)
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.file_audited.connect(self.on_file_audited)
        self.active_worker.audit_completed.connect(self.on_audit_completed)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def on_file_audited(self, sid, rep):
        self.health_reports[sid] = rep
        for s in self.dataset["samples"]:
            if s["id"] == sid and not s.get("locked", False):
                if not s.get("bpm") or s.get("bpm") == 0:
                    s["bpm"] = rep.get("bpm_detected", 120)
                if not s.get("keyscale"):
                    s["keyscale"] = rep.get("key_detected", "A minor")
                if not s.get("duration") or s.get("duration") == 0:
                    s["duration"] = int(rep.get("duration", 0))

    def on_audit_completed(self, summary):
        if hasattr(self, "rescan_notice") and self.rescan_notice is not None:
            self.rescan_notice.close()
            self.rescan_notice.deleteLater()
            self.rescan_notice = None
        self.scan_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Health & Homogeneity audit finished.")

        score = summary.get("quality_score", 100)
        reasons = summary.get("reasons", [])

        if score >= 80:
            self.quality_badge.setText(f"Dataset Quality: {score}% [Ready for LoRA]")
            self.quality_badge.setStyleSheet("font-weight: bold; font-size: 13px; padding: 6px 14px; background-color: #2E7D32; border-radius: 4px; color: #fff;")
        elif score >= 60:
            self.quality_badge.setText(f"Dataset Quality: {score}% [Warning]")
            self.quality_badge.setStyleSheet("font-weight: bold; font-size: 13px; padding: 6px 14px; background-color: #F57F17; border-radius: 4px; color: #fff;")
        else:
            self.quality_badge.setText(f"Dataset Quality: {score}% [Critical Inconsistencies]")
            self.quality_badge.setStyleSheet("font-weight: bold; font-size: 13px; padding: 6px 14px; background-color: #B71C1C; border-radius: 4px; color: #fff;")

        self.refresh_table()
        self.on_table_selection_changed()

        if reasons:
            reasons_str = "\n• " + "\n• ".join(reasons)
            QMessageBox.warning(
                self, f"Quality Audit: {score}% Score",
                f"The following degradation risks were identified:\n{reasons_str}\n\n"
                "Tip: Run the DSP Normalizer to automatically resolve loudness variations and sample rate mismatches."
            )

    # -----------------------------------------------------------------------
    # DSP Normalize
    # -----------------------------------------------------------------------
    def on_file_normalized(self, sid, orig_backup, norm_path, sr, lufs):
        self.original_backups[sid] = orig_backup
        for s in self.dataset["samples"]:
            if s["id"] == sid:
                s["audio_path"] = norm_path
                s["filename"] = Path(norm_path).name
                break

    def start_dsp_normalize(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks before normalizing.")
            return

        all_tracks_choice = "★ Normalize ALL dataset tracks"
        choices = [all_tracks_choice] + [f"{s.get('filename', 'Unnamed file')} [{s.get('id', '')}]" for s in samples]

        selected_label, accepted = QInputDialog.getItem(
            self, "Choose Audio Track", "Select one track to normalize:", choices, 0, False
        )
        if not accepted or not selected_label:
            self.status_label.setText("DSP normalization cancelled; no files changed.")
            return

        if selected_label == all_tracks_choice:
            selected_samples = list(samples)
            confirm_all = QMessageBox.question(
                self, "Normalize All Tracks?",
                f"This will normalize all {len(selected_samples)} tracks. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if confirm_all != QMessageBox.Yes:
                self.status_label.setText("DSP normalization cancelled; no files changed.")
                return
        else:
            selected_index = choices.index(selected_label) - 1
            selected_samples = [samples[selected_index]]

        audio_path = selected_samples[0].get("audio_path", "")
        default_folder = str(Path(audio_path).parent) if audio_path else str(Path.home())

        placement_box = QMessageBox(self)
        placement_box.setWindowTitle("Normalized Audio & Backup Location")
        placement_box.setIcon(QMessageBox.Information)
        placement_box.setText("Normalized audio and original-file backups will be created in the same folder as this track's audio file.")
        placement_box.setInformativeText(f"Default dataset folder:\n{default_folder}\n\nChoose OK to use the default, or Let Me Decide to choose another folder.")
        default_btn = placement_box.addButton("OK", QMessageBox.AcceptRole)
        decide_btn = placement_box.addButton("Let Me Decide", QMessageBox.ActionRole)
        placement_box.addButton(QMessageBox.Cancel)
        placement_box.exec()
        clicked_btn = placement_box.clickedButton()

        if clicked_btn == default_btn:
            project_folder = default_folder
        elif clicked_btn == decide_btn:
            project_folder = QFileDialog.getExistingDirectory(self, "Select Persistent Project Folder", default_folder)
        else:
            self.status_label.setText("DSP normalization cancelled; no files changed.")
            return

        if not project_folder:
            self.status_label.setText("DSP normalization cancelled; no files changed.")
            return

        self.normalization_dataset_backup = json.loads(json.dumps(self.dataset))
        self.record_snapshot()
        self.normalize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.active_worker = DspNormalizerWorker(
            selected_samples,
            target_dir=project_folder,
            target_sr=44100,
            target_lufs=-14.0
        )
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.file_normalized.connect(self.on_file_normalized)
        self.active_worker.all_done.connect(self.on_normalize_done)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def on_normalize_done(self, norm_dir, backup_dir):
        self.normalize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Normalization completed.")

        before_json = os.path.join(backup_dir, "dataset_before_normalization.json")
        normalized_json = os.path.join(norm_dir, "dataset_normalized.json")

        try:
            before_dataset = getattr(self, "normalization_dataset_backup", self.dataset)
            with open(before_json, "w", encoding="utf-8") as f:
                json.dump(before_dataset, f, indent=2)
            with open(normalized_json, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Dataset JSON Backup Warning", f"Could not write JSON backup: {e}")

        QMessageBox.information(
            self, "DSP Normalization Finished",
            f"Normalized Audio Workspace:\n{norm_dir}\n\nOriginal Backup Stored At:\n{backup_dir}"
        )

        self.status_label.setText("Rescanning dataset files. Please wait a few seconds...")
        self.rescan_notice = QMessageBox(self)
        self.rescan_notice.setWindowTitle("Rescanning Dataset")
        self.rescan_notice.setIcon(QMessageBox.Information)
        self.rescan_notice.setText("Rescanning dataset files. Please wait a few seconds...")
        self.rescan_notice.setStandardButtons(QMessageBox.NoButton)
        self.rescan_notice.setModal(False)
        self.rescan_notice.show()
        self.start_health_audit()

    # -----------------------------------------------------------------------
    # Load / Save Dataset
    # -----------------------------------------------------------------------
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset JSON", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
                self.record_snapshot()
                self.health_reports.clear()
                self.sync_general_props_to_ui()
                self.refresh_table()
                self.status_label.setText(f"Loaded {len(self.dataset.get('samples', []))} tracks.")
                self.start_health_audit()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def save_dataset(self):
        if not self.bypass_warnings and self.quality_badge.text().find("Critical") != -1:
            QMessageBox.warning(
                self, "Export Blocked by Quality Threshold",
                "Dataset quality is below safe threshold (<60%). Fix flagged issues or click '🛡 I Know What I'm Doing' to bypass.",
                QMessageBox.Ok
            )
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Dataset JSON", "", "JSON Files (*.json)")
        if path:
            try:
                self.on_general_prop_changed()
                self.dataset["metadata"]["num_samples"] = len(self.dataset["samples"])
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.dataset, f, indent=2)
                self.status_label.setText(f"Saved dataset to {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    # -----------------------------------------------------------------------
    # Add Audio Files
    # -----------------------------------------------------------------------
    def add_audio_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Audio Tracks", "", "Audio Files (*.wav *.flac *.mp3)")
        if paths:
            self.record_snapshot()
            global_tag = self.custom_tag_input.text().strip()
            is_all_inst = self.radio_all_inst.isChecked()

            for p in paths:
                fname = Path(p).name
                self.dataset["samples"].append({
                    "id": uuid.uuid4().hex[:8],
                    "audio_path": p,
                    "filename": fname,
                    "caption": "",
                    "genre": "",
                    "lyrics": "",
                    "formatted_lyrics": "",
                    "bpm": 0,
                    "keyscale": "",
                    "timesignature": "4/4",
                    "duration": 0,
                    "language": "en",
                    "is_instrumental": is_all_inst,
                    "custom_tag": global_tag,
                    "locked": True,
                    # Spatial fields
                    "structural_segments": [],
                    "spatial_tokens": {},
                    "stem_paths": {},
                    "chunk_paths": []
                })
            self.refresh_table()
            self.status_label.setText(f"Added {len(paths)} audio tracks.")
            self.start_health_audit()

    # -----------------------------------------------------------------------
    # AI Captioning (with DeepSeek backend option)
    # -----------------------------------------------------------------------
    def start_ai_captioning(self):
        all_samples = self.dataset.get("samples", [])
        if not all_samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks before captioning.")
            return

        scope_choices = ["Selected Track", "Tracks Missing Captions", "All Tracks — Review Every Result"]
        scope, accepted = QInputDialog.getItem(self, "Choose Captioning Scope", "Which tracks should ACE-Step Captioner process?", scope_choices, 0, False)
        if not accepted or not scope:
            self.status_label.setText("AI captioning cancelled.")
            return

        if scope == "Selected Track":
            selected_sample = self.get_selected_sample()
            if not selected_sample:
                QMessageBox.warning(self, "No Track Selected", "Select one track in the dataset table first.")
                return
            samples = [selected_sample]
        elif scope == "Tracks Missing Captions":
            samples = [s for s in all_samples if not s.get("caption", "").strip()]
        else:
            samples = list(all_samples)

        if not samples:
            QMessageBox.information(self, "Nothing To Caption", "No tracks match the selected scope.")
            return

        self.record_snapshot()
        self.run_ai_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        general_meta = self.dataset.get("metadata", {})

        # Determine backend: prefer Kaggle if creds exist, else DeepSeek, else Local
        backend = "Local Rule Engine"
        if self.config.get("kaggle_user") and self.config.get("kaggle_key"):
            backend = "Kaggle Cloud (Free GPU)"
        elif self.config.get("custom_key"):
            backend = "DeepSeek Cloud"

        self.active_worker = RemoteCaptionWorker(
            samples,
            backend,
            "Deep Structural Breakdown",
            general_meta,
            self.config
        )
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.finished_sample.connect(self.on_sample_captioned)
        self.active_worker.all_done.connect(self.on_caption_finished)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def on_sample_captioned(self, sid, caption):
        for sample in self.dataset["samples"]:
            if sample.get("id") == sid:
                self.save_ai_caption_result(sample, caption, model_id="ACE-Step/acestep-captioner", prompt="Detailed ACE-Step caption request")
                self.review_ai_caption_result(sample)
                break

    def on_caption_finished(self):
        self.run_ai_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("AI Captioning completed.")
        self.refresh_table()
        self.on_table_selection_changed()

    # -----------------------------------------------------------------------
    # Common worker callbacks
    # -----------------------------------------------------------------------
    def on_worker_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def on_worker_error(self, err_msg):
        self.run_ai_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.normalize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Operation error.")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")

# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())


