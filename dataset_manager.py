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

from config import DEFAULT_CONFIG
from modules.config_store import load_config, save_config
from modules.model_manager import load_catalog, leaderboards, find_model, is_downloaded, remove_model
from workers.model_manager import ModelDownloadWorker

# Modern worker implementations (split into workers/ modules).
from workers.caption import RemoteCaptionWorker, resolve_backend
from workers.spatial import SpatialPipelineWorker
from workers.structural import (
    StructuralPipelineWorker,
    StructuralPipelineBatchWorker,
)

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
    QFrame, QListWidget
)
from PySide6.QtGui import QFont, QColor, QDesktopServices

# ============================================================================
# NEW: DeepSeek Orchestrator
# ============================================================================
class DeepSeekMusicOrchestrator:
    def __init__(self, api_key=None, base_url=None, config=None):
        """Provider-aware orchestrator (mirrors workers/deepseek.py)."""
        if config is not None:
            from modules.llm_client import get_client

            self.provider, self.info, self.client = get_client(config)
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
# Main Window: DatasetManager
# ============================================================================
class DatasetManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACE-Step Dataset Toolkit")
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
        # Config: DEFAULT_CONFIG (best-working defaults) + settings.json + the
        # encrypted secrets store. Each secret has a per-key "remember on this
        # device" policy — when unchecked it is kept for the session only.
        self.config = load_config(DEFAULT_CONFIG)
        self.health_reports = {}
        self.original_backups = {}
        self.active_worker = None
        self.filter_exceptions_only = False
        self.bypass_warnings = False
        self.startup_scan_notice_shown = False
        self.kaggle_notebook_unlocked = False  # NEW

        self.init_ui()
        self.apply_custom_theme()
        self.refresh_band_profiles()

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
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)   
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
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
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
        self.remember_kaggle = QCheckBox("Remember on this device (encrypted)")
        self.remember_kaggle.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
        self.remember_kaggle.setChecked(bool(self.config.get("remember_kaggle_key", True)))
        c_form.addRow("", self.remember_kaggle)

        self.custom_url = QLineEdit(self.config.get("custom_url", ""))
        self.custom_url.setPlaceholderText("https://api.runpod.ai/... or http://localhost:8000/v1")
        self.custom_key = QLineEdit(self.config.get("custom_key", ""))
        self.custom_key.setEchoMode(QLineEdit.Password)
        c_form.addRow("Custom Auth Token (DeepSeek/MVSEP):", self.custom_key)
        self.remember_custom = QCheckBox("Remember on this device (encrypted)")
        self.remember_custom.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
        self.remember_custom.setChecked(bool(self.config.get("remember_custom_key", True)))
        c_form.addRow("", self.remember_custom)

        # NEW: MVSEP key
        self.mvsep_key = QLineEdit(self.config.get("mvsep_api_key", ""))
        self.mvsep_key.setEchoMode(QLineEdit.Password)
        c_form.addRow("MVSEP API Key:", self.mvsep_key)
        self.remember_mvsep = QCheckBox("Remember on this device (encrypted)")
        self.remember_mvsep.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
        self.remember_mvsep.setChecked(bool(self.config.get("remember_mvsep_api_key", True)))
        c_form.addRow("", self.remember_mvsep)

        sec_note = QLabel(
            "Secrets are stored encrypted (OS keyring, or an encrypted file as a fallback) — "
            "never in settings.json. Uncheck 'Remember' to keep a key for the current session only."
        )
        sec_note.setWordWrap(True)
        sec_note.setStyleSheet("color: #aaa; font-size: 9px; padding: 2px;")
        c_form.addRow(sec_note)

        # ---- Caption backend (pluggable providers) ----
        self.caption_backend_combo = QComboBox()
        self.caption_backend_combo.addItems([
            "ace_step — ACE-Step captioner (Kaggle GPU, default)",
            "gemini — Google Gemini (audio-native)",
            "deepseek — DeepSeek LLM",
            "custom — OpenAI-compatible endpoint (local/rented GPU)",
        ])
        cur_backend = (self.config.get("caption_backend") or "ace_step").strip().lower()
        for i in range(self.caption_backend_combo.count()):
            if self.caption_backend_combo.itemText(i).startswith(cur_backend):
                self.caption_backend_combo.setCurrentIndex(i)
                break
        c_form.addRow("Caption Backend:", self.caption_backend_combo)

        self.gemini_key = QLineEdit(self.config.get("gemini_api_key", ""))
        self.gemini_key.setEchoMode(QLineEdit.Password)
        c_form.addRow("Gemini API Key:", self.gemini_key)
        self.remember_gemini = QCheckBox("Remember on this device (encrypted)")
        self.remember_gemini.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
        self.remember_gemini.setChecked(bool(self.config.get("remember_gemini_key", True)))
        c_form.addRow("", self.remember_gemini)

        self.gemini_model_combo = QComboBox()
        self.gemini_model_combo.setEditable(True)
        self.gemini_model_combo.addItems(["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"])
        cur_model = (self.config.get("gemini_model") or "gemini-2.5-flash").strip()
        idx = self.gemini_model_combo.findText(cur_model)
        if idx >= 0:
            self.gemini_model_combo.setCurrentIndex(idx)
        else:
            self.gemini_model_combo.setEditText(cur_model)
        c_form.addRow("Gemini Model:", self.gemini_model_combo)

        self.custom_url_edit = QLineEdit(self.config.get("custom_caption_url", ""))
        self.custom_url_edit.setPlaceholderText("e.g. http://localhost:8000/v1 (vLLM/Ollama/runpod)")
        c_form.addRow("Custom Endpoint Base URL:", self.custom_url_edit)

        self.custom_model_edit = QLineEdit(self.config.get("custom_caption_model", ""))
        self.custom_model_edit.setPlaceholderText("model name served by the endpoint")
        c_form.addRow("Custom Endpoint Model:", self.custom_model_edit)

        self.custom_audio_check = QCheckBox("Send audio to the endpoint (OpenAI input_audio)")
        self.custom_audio_check.setChecked(bool(self.config.get("custom_caption_audio", False)))
        c_form.addRow("", self.custom_audio_check)

        save_cloud_btn = QPushButton("Save Cloud Credentials")
        save_cloud_btn.clicked.connect(self.save_cloud_config)
        c_form.addRow(save_cloud_btn)

        layout.addWidget(cloud_grp)

        llm_grp = QGroupBox("🧠 LLM Provider")
        llm_form = QFormLayout(llm_grp)

        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems([
            "deepseek (paid, cheap — default)",
            "gemini (free tier)",
            "groq (free tier)",
            "openrouter (free models)",
            "local (custom endpoint)",
        ])
        cur_prov = str(self.config.get("llm_provider", "deepseek") or "deepseek").lower()
        for i in range(self.llm_provider_combo.count()):
            if self.llm_provider_combo.itemText(i).startswith(cur_prov):
                self.llm_provider_combo.setCurrentIndex(i)
                break
        self.llm_provider_combo.currentIndexChanged.connect(self._on_llm_provider_changed)
        llm_form.addRow("Provider:", self.llm_provider_combo)

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        llm_form.addRow("Model:", self.llm_model_combo)

        self.llm_base_url_edit = QLineEdit()
        self.llm_base_url_edit.setPlaceholderText("auto-filled from the provider (editable)")
        llm_form.addRow("Base URL:", self.llm_base_url_edit)

        self.openrouter_key = QLineEdit(self.config.get("openrouter_key", ""))
        self.openrouter_key.setEchoMode(QLineEdit.Password)
        llm_form.addRow("OpenRouter Key:", self.openrouter_key)
        self.remember_openrouter = QCheckBox("Remember on this device (encrypted)")
        self.remember_openrouter.setChecked(bool(self.config.get("remember_openrouter_key", True)))
        llm_form.addRow("", self.remember_openrouter)

        self.groq_key = QLineEdit(self.config.get("groq_key", ""))
        self.groq_key.setEchoMode(QLineEdit.Password)
        llm_form.addRow("Groq Key:", self.groq_key)
        self.remember_groq = QCheckBox("Remember on this device (encrypted)")
        self.remember_groq.setChecked(bool(self.config.get("remember_groq_key", True)))
        llm_form.addRow("", self.remember_groq)

        self.llm_note = QLabel("")
        self.llm_note.setWordWrap(True)
        self.llm_note.setStyleSheet("color: #aaa; font-size: 9px;")
        llm_form.addRow(self.llm_note)
        self._on_llm_provider_changed()

        layout.addWidget(llm_grp)

        pipe_grp = QGroupBox("🎛 Pipeline & Model Defaults")
        p_form = QFormLayout(pipe_grp)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(self.config.get("caption_prompt", ""))
        self.prompt_edit.setMaximumHeight(120)
        self.prompt_edit.setPlaceholderText("Prompt used by the caption backends (ACE-Step / Gemini / custom).")
        p_form.addRow("Caption Prompt:", self.prompt_edit)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(32, 2048)
        self.max_tokens_spin.setValue(int(self.config.get("caption_max_tokens", 512)))
        p_form.addRow("Caption Max Tokens:", self.max_tokens_spin)

        self.max_dur_spin = QSpinBox()
        self.max_dur_spin.setRange(0, 600)
        self.max_dur_spin.setValue(int(self.config.get("caption_max_audio_duration", 120)))
        self.max_dur_spin.setToolTip("Max audio length fed to the captioner in seconds (0 = whole file).")
        p_form.addRow("Max Audio Duration (s):", self.max_dur_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 8)
        self.batch_size_spin.setValue(int(self.config.get("caption_batch_size", 1)))
        self.batch_size_spin.setToolTip("Chunks processed per captioner forward pass on the Kaggle GPU. 1 is safest; 2-4 is faster when VRAM allows (32 GB total across both T4s).")
        p_form.addRow("Caption Batch Size:", self.batch_size_spin)

        self.tag_ratio_spin = QSpinBox()
        self.tag_ratio_spin.setRange(0, 100)
        self.tag_ratio_spin.setSuffix("%")
        self.tag_ratio_spin.setValue(int(self.config.get("tag_caption_ratio", 0)))
        self.tag_ratio_spin.setToolTip("Hybrid captions: 0% = prose only (default), 100% = tag block only, in between = both.")
        p_form.addRow("Hybrid Tag Ratio:", self.tag_ratio_spin)

        self.clap_tagger_combo = QComboBox()
        self.clap_tagger_combo.addItems(["auto (use CLAP if installed)", "on", "off"])
        cur_clap = str(self.config.get("use_clap_tagger", "auto") or "auto").lower()
        for i in range(self.clap_tagger_combo.count()):
            if self.clap_tagger_combo.itemText(i).lower().startswith(cur_clap):
                self.clap_tagger_combo.setCurrentIndex(i)
                break
        self.clap_tagger_combo.setToolTip("CLAP zero-shot tagging names specific instruments (needs torch + transformers). 'auto' uses it when available.")
        p_form.addRow("Instrument Tagger:", self.clap_tagger_combo)

        self.auto_recommend_check = QCheckBox("Auto-recommend instrument models from detected tags")
        self.auto_recommend_check.setChecked(bool(self.config.get("auto_recommend_models", True)))
        p_form.addRow(self.auto_recommend_check)

        self.lead_vocal_combo = QComboBox()
        self.lead_vocal_combo.addItems(["off", "mvsep (backing-vocal model)", "heuristic (experimental)"])
        cur_lv = str(self.config.get("lead_vocal_splitter", "off") or "off").lower()
        if cur_lv == "mvsep":
            self.lead_vocal_combo.setCurrentIndex(1)
        elif cur_lv == "heuristic":
            self.lead_vocal_combo.setCurrentIndex(2)
        p_form.addRow("Lead/Backing Vocal Split:", self.lead_vocal_combo)

        self.min_sec_spin = QDoubleSpinBox()
        self.min_sec_spin.setRange(1.0, 120.0)
        self.min_sec_spin.setSingleStep(0.5)
        self.min_sec_spin.setDecimals(1)
        self.min_sec_spin.setValue(float(self.config.get("segment_min_sec", 12.0)))
        p_form.addRow("Min Section Length (s):", self.min_sec_spin)

        self.max_k_spin = QSpinBox()
        self.max_k_spin.setRange(2, 40)
        self.max_k_spin.setValue(int(self.config.get("segment_max_k", 20)))
        p_form.addRow("Max Sections:", self.max_k_spin)

        self.structure_backend_combo = QComboBox()
        self.structure_backend_combo.addItems([
            "librosa (default)",
            "songformer (functional labels, Kaggle)",
        ])
        cur_struct = str(self.config.get("structure_backend", "librosa") or "librosa").lower()
        self.structure_backend_combo.setCurrentIndex(1 if cur_struct.startswith("song") else 0)
        self.structure_backend_combo.setToolTip(
            "songformer splits sections into real labels (intro/verse/chorus/bridge/solo/outro) "
            "via a Kaggle GPU kernel; falls back to librosa automatically."
        )
        p_form.addRow("Structure Backend:", self.structure_backend_combo)

        self.stem_model_combo = QComboBox()
        self.stem_model_combo.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"])
        cur_stem = self.config.get("kaggle_stem_model", "htdemucs_ft")
        for i in range(self.stem_model_combo.count()):
            if self.stem_model_combo.itemText(i) == cur_stem:
                self.stem_model_combo.setCurrentIndex(i)
                break
        p_form.addRow("Kaggle Stem Model:", self.stem_model_combo)

        stem_out_row = QHBoxLayout()
        self.stem_out_edit = QLineEdit(self.config.get("stem_output_dir", ""))
        self.stem_out_edit.setPlaceholderText("Default: ~/mvsep_stems")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_stem_dir)
        stem_out_row.addWidget(self.stem_out_edit)
        stem_out_row.addWidget(browse_btn)
        p_form.addRow("Stem Output Folder:", stem_out_row)

        self.lufs_spin = QDoubleSpinBox()
        self.lufs_spin.setRange(-30.0, 0.0)
        self.lufs_spin.setDecimals(1)
        self.lufs_spin.setValue(float(self.config.get("dsp_target_lufs", -14.0)))
        p_form.addRow("Normalize Target (LUFS):", self.lufs_spin)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setSingleStep(1000)
        self.sr_spin.setValue(int(self.config.get("dsp_target_sr", 44100)))
        p_form.addRow("Normalize Target SR (Hz):", self.sr_spin)

        save_pipe_btn = QPushButton("Save Pipeline Defaults")
        save_pipe_btn.clicked.connect(self.save_pipeline_defaults)
        p_form.addRow(save_pipe_btn)

        layout.addWidget(pipe_grp)

        mm_grp = QGroupBox("🧰 Model Manager")
        mm_form = QFormLayout(mm_grp)

        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems(["hf (Hugging Face)", "github (my repo)"])
        cur_src = str(self.config.get("model_download_source", "hf") or "hf").lower()
        self.model_source_combo.setCurrentIndex(1 if cur_src.startswith("git") else 0)
        mm_form.addRow("Download Source:", self.model_source_combo)

        self.model_pick_combo = QComboBox()
        self.model_pick_combo.setMinimumWidth(340)
        self._populate_model_picker()
        mm_form.addRow("Model:", self.model_pick_combo)

        self.model_status = QLabel("Select a model to see its status.")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet("color: #aaa; font-size: 9px;")
        mm_form.addRow(self.model_status)

        dl_row = QHBoxLayout()
        download_btn = QPushButton("⬇ Download")
        download_btn.clicked.connect(self.download_selected_model)
        remove_btn = QPushButton("🗑 Remove")
        remove_btn.clicked.connect(self.remove_selected_model)
        refresh_btn = QPushButton("↻ Status")
        refresh_btn.clicked.connect(self.refresh_model_status)
        dl_row.addWidget(download_btn)
        dl_row.addWidget(remove_btn)
        dl_row.addWidget(refresh_btn)
        mm_form.addRow(dl_row)

        self.leaderboard_combo = QComboBox()
        for item in leaderboards():
            self.leaderboard_combo.addItem(item["name"], item["url"])
        open_lb = QPushButton("Open")
        open_lb.clicked.connect(self.open_selected_leaderboard)
        lb_row = QHBoxLayout()
        lb_row.addWidget(self.leaderboard_combo, 1)
        lb_row.addWidget(open_lb)
        mm_form.addRow("Leaderboards:", lb_row)

        self.hf_token_edit = QLineEdit(self.config.get("hf_token", ""))
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        mm_form.addRow("Hugging Face Token:", self.hf_token_edit)
        self.remember_hf = QCheckBox("Remember on this device (encrypted)")
        self.remember_hf.setChecked(bool(self.config.get("remember_hf_token", True)))
        mm_form.addRow("", self.remember_hf)

        self.model_dir_edit = QLineEdit(self.config.get("model_dir", "models"))
        self.model_dir_edit.setToolTip("Folder where downloaded models are stored (gitignored).")
        mm_form.addRow("Models Folder:", self.model_dir_edit)

        layout.addWidget(mm_grp)
        layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

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
        self.stem_source_combo.addItems(["Import existing stems", "Separate via MVSEP", "Separate via Kaggle (Demucs)"])
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

        # ---- Scope selection ----
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))
        self.struct_scope_combo = QComboBox()
        self.struct_scope_combo.addItems([
            "All Tracks",
            "Tracks Missing Captions",
            "Selected Tracks (from list)"
        ])
        self.struct_scope_combo.currentTextChanged.connect(self.on_struct_scope_changed)
        scope_layout.addWidget(self.struct_scope_combo)
        inner.addLayout(scope_layout)

         # ---- Track list (selectable, visible only when needed) ----
        self.track_list_widget = QListWidget()
        self.track_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.track_list_widget.setMaximumHeight(100)
        self.track_list_widget.setVisible(False)
        inner.addWidget(self.track_list_widget)

        # ---- Track number input (visible when scope is "Selected Tracks (from list)") ----
        number_input_layout = QHBoxLayout()
        number_input_layout.addWidget(QLabel("Track numbers (or select from list above):"))
        self.track_numbers_input = QLineEdit()
        self.track_numbers_input.setPlaceholderText("e.g., 1,3,5 or 1-3,5")
        self.track_numbers_input.setVisible(False)
        number_input_layout.addWidget(self.track_numbers_input)
        inner.addLayout(number_input_layout)  

        # ---- Populate list if dataset has tracks ----
        self.refresh_track_list()

        # Stem source
        stem_layout = QHBoxLayout()
        stem_layout.addWidget(QLabel("Stem source:"))
        self.struct_stem_combo = QComboBox()
        self.struct_stem_combo.addItems(["Import existing stems", "Separate via MVSEP", "Separate via Kaggle (Demucs)"])
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

        # Humanization preset
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Humanization Preset:"))
        self.humanize_preset_combo = QComboBox()
        self.humanize_preset_combo.addItems([
            "None",
            "Hank Williams",
            "Kurt Cobain",
            "Jimi Hendrix",
            "Janis Joplin",
            "Bob Dylan",
            "Pink Floyd (Gilmour)",
            "Ozzy Osbourne"
        ])
        preset_layout.addWidget(self.humanize_preset_combo)
        inner.addLayout(preset_layout)

        self.humanize_check = QCheckBox("Apply humanization")
        self.humanize_check.setChecked(True)
        inner.addWidget(self.humanize_check)

        # ---- Instrument extraction group ----
        sep_group = QGroupBox("Instrument‑Specific Stem Extraction")
        sep_layout2 = QVBoxLayout(sep_group)

        self.instrument_extraction_check = QCheckBox("Enable instrument‑specific extraction (recommended)")
        self.instrument_extraction_check.setChecked(True)
        sep_layout2.addWidget(self.instrument_extraction_check)

        disclaimer = QLabel(
            "⚠️ Disclaimer: The song‑specific recommendation may not be perfect.\n"
            "If instruments are not removed by the recommended options,\n"
            "you must experiment with other models that may or may not be on the list."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #ffcc80; font-size: 10px; padding: 4px;")
        sep_layout2.addWidget(disclaimer)

        detect_layout = QHBoxLayout()
        detect_layout.addWidget(QLabel("Detected instruments:"))
        self.detect_instruments_btn = QPushButton("Detect via Captioner")
        self.detect_instruments_btn.clicked.connect(self.detect_instruments_for_separation)
        detect_layout.addWidget(self.detect_instruments_btn)
        detect_layout.addStretch()
        sep_layout2.addLayout(detect_layout)

        self.detected_instruments_list = QTextEdit()
        self.detected_instruments_list.setPlaceholderText("Run 'Detect' to see instruments...")
        self.detected_instruments_list.setMaximumHeight(80)
        self.detected_instruments_list.setReadOnly(True)
        sep_layout2.addWidget(self.detected_instruments_list)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Additional models to run:"))
        self.extra_models_input = QLineEdit()
        self.extra_models_input.setPlaceholderText("e.g., MVSep Organ, MVSep Harpsichord")
        model_layout.addWidget(self.extra_models_input)
        sep_layout2.addLayout(model_layout)

        note = QLabel("You can also manually type additional MVSEP model names above.")
        note.setStyleSheet("color: #aaa; font-size: 9px;")
        sep_layout2.addWidget(note)

        inner.addWidget(sep_group)

        # Band profile
        band_layout = QHBoxLayout()
        band_layout.addWidget(QLabel("Band:"))
        self.band_combo = QComboBox()
        self.band_combo.addItem("None")
        self.band_combo.currentTextChanged.connect(self.on_band_changed)
        band_layout.addWidget(self.band_combo)

        band_layout.addWidget(QLabel("Era:"))
        self.era_combo = QComboBox()
        self.era_combo.addItem("None")
        band_layout.addWidget(self.era_combo)

        band_layout.addWidget(QLabel("Extra Notes:"))
        self.band_notes = QLineEdit()
        self.band_notes.setPlaceholderText("e.g., specific amp, recording notes")
        band_layout.addWidget(self.band_notes)

        inner.addLayout(band_layout)

        # DeepSeek toggle
        self.struct_deepseek_check = QCheckBox("Use DeepSeek for aggregation")
        self.struct_deepseek_check.setChecked(True)
        inner.addWidget(self.struct_deepseek_check)

        # Run button
        self.run_struct_btn = QPushButton("🚀 Run Structural Pipeline")
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
            "stem_source": ("kaggle_demucs" if stem_source == "Separate via Kaggle (Demucs)" else ("mvsep" if stem_source == "Separate via MVSEP" else "import")),
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

    def run_structural_pipeline(self):
        # ---- Determine which tracks to process based on scope ----
        scope = self.struct_scope_combo.currentText()
        if scope == "All Tracks":
            tracks = self.dataset.get("samples", [])
            if not tracks:
                QMessageBox.warning(self, "No Tracks", "The dataset is empty.")
                return
        elif scope == "Tracks Missing Captions":
            tracks = [s for s in self.dataset.get("samples", []) if not s.get("caption", "").strip()]
            if not tracks:
                QMessageBox.information(self, "Nothing to Process", "All tracks already have captions.")
                return
        else:  # "Selected Tracks (from list)"
            # First, try to use the text input if non‑empty
            numbers_text = self.track_numbers_input.text().strip()
            if numbers_text:
                indices = self._parse_track_numbers(numbers_text)
                if indices is None:
                    QMessageBox.warning(self, "Invalid Input", "Invalid track number format. Use numbers separated by spaces, commas, or ranges (e.g., 1-3,5).")
                    return
                all_samples = self.dataset.get("samples", [])
                valid_indices = [i for i in indices if 1 <= i <= len(all_samples)]
                if not valid_indices:
                    QMessageBox.warning(self, "No Valid Tracks", "None of the numbers correspond to existing tracks.")
                    return
                tracks = [all_samples[i-1] for i in valid_indices]
            else:
                # Fallback to list selection
                selected_items = self.track_list_widget.selectedItems()
                if not selected_items:
                    QMessageBox.warning(self, "No Selection", "Please select at least one track from the list or enter numbers.")
                    return
                indices = []
                for item in selected_items:
                    try:
                        num = int(item.text().split(' - ')[0])
                        indices.append(num)
                    except:
                        continue
                if not indices:
                    QMessageBox.warning(self, "Invalid Selection", "Could not parse track numbers.")
                    return
                all_samples = self.dataset.get("samples", [])
                tracks = [all_samples[i-1] for i in indices if 1 <= i <= len(all_samples)]

        # ---- Read UI options ----
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

        # ---- Band profile ----
        band = self.band_combo.currentText()
        era = self.era_combo.currentText()
        extra_notes = self.band_notes.text()
        band_context = ""
        instrument_context = ""
        production_context = ""
        vocal_context = ""
        humanize_preset = "None"

        if band != "None" and era != "None":
            profiles = self.load_band_profiles()
            band_data = profiles.get(band, {})
            era_data = next((e for e in band_data.get("eras", []) if e["name"] == era), {})
            band_context = f"{band} – {era}"
            instrument_context = era_data.get("instruments", "")
            production_context = era_data.get("production", "")
            vocal_context = era_data.get("vocal_character", "")
            humanize_preset = era_data.get("humanization_preset", "None")

        # ---- Humanization and stem options ----
        humanize = self.humanize_check.isChecked()
        stem_options = {}
        if self.instrument_extraction_check.isChecked():
            sample = tracks[0] if tracks else None
            caption_text = sample.get("caption", "") if sample else ""
            if caption_text:
                stem_options['use_caption_recommendation'] = True
                stem_options['caption_text'] = caption_text
            extra_models = self.extra_models_input.text().strip()
            if extra_models:
                models = [m.strip() for m in extra_models.split(',') if m.strip()]
                stem_options['instrument_models'] = models

        options = {
            "stem_source": ("kaggle_demucs" if stem_source == "Separate via Kaggle (Demucs)" else ("mvsep" if stem_source == "Separate via MVSEP" else "import")),
            "use_deepseek": self.struct_deepseek_check.isChecked(),
            "use_lyrics": self.struct_seg_combo.currentText() == "Lyrics tags",
            "humanize": humanize,
            "humanize_preset": humanize_preset,
            "band_context": band_context,
            "instrument_context": instrument_context,
            "production_context": production_context,
            "vocal_context": vocal_context,
            "extra_notes": extra_notes,
            "stem_options": stem_options
        }

        # ---- Start batch worker ----
        self.run_struct_btn.setEnabled(False)
        self.struct_progress.setVisible(True)
        self.struct_progress.setValue(0)
        self.struct_status.setText(f"Starting structural pipeline on {len(tracks)} track(s)...")

        self.batch_worker = StructuralPipelineBatchWorker(
            tracks=tracks,
            config=self.config,
            options=options
        )
        self.batch_worker.progress.connect(self.on_struct_progress)
        self.batch_worker.track_done.connect(self.on_struct_track_done)
        self.batch_worker.all_done.connect(self.on_struct_batch_done)
        self.batch_worker.error_occurred.connect(self.on_struct_error)
        self.batch_worker.start()

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
                tags = result.get("tags") or {}
                if tags:
                    sample["tags"] = tags
                    if tags.get("bpm"):
                        sample["bpm"] = tags["bpm"]
                    if tags.get("key"):
                        sample["keyscale"] = tags["key"]
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
        scope = self.struct_scope_combo.currentText()
        if scope == "All Tracks":
            tracks = self.dataset.get("samples", [])
            if not tracks:
                QMessageBox.warning(self, "No Tracks", "The dataset is empty.")
                return
        elif scope == "Tracks Missing Captions":
            tracks = [s for s in self.dataset.get("samples", []) if not s.get("caption", "").strip()]
            if not tracks:
                QMessageBox.information(self, "Nothing to Process", "All tracks already have captions.")
                return

            # Validate indices
            all_samples = self.dataset.get("samples", [])
            valid_indices = [i for i in indices if 1 <= i <= len(all_samples)]
            if not valid_indices:
                QMessageBox.warning(self, "No Valid Tracks", "None of the numbers correspond to existing tracks.")
                return
            tracks = [all_samples[i-1] for i in valid_indices]

        # ---- Read UI options ----
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

        # ---- Band profile ----
        band = self.band_combo.currentText()
        era = self.era_combo.currentText()
        extra_notes = self.band_notes.text()
        band_context = ""
        instrument_context = ""
        production_context = ""
        vocal_context = ""
        humanize_preset = "None"

        if band != "None" and era != "None":
            profiles = self.load_band_profiles()
            band_data = profiles.get(band, {})
            era_data = next((e for e in band_data.get("eras", []) if e["name"] == era), {})
            band_context = f"{band} – {era}"
            instrument_context = era_data.get("instruments", "")
            production_context = era_data.get("production", "")
            vocal_context = era_data.get("vocal_character", "")
            humanize_preset = era_data.get("humanization_preset", "None")

        # ---- Humanization and stem options ----
        humanize = self.humanize_check.isChecked()
        stem_options = {}
        if self.instrument_extraction_check.isChecked():
            # Use the first track in the list for caption recommendation
            sample = tracks[0] if tracks else None
            caption_text = sample.get("caption", "") if sample else ""
            if caption_text:
                stem_options['use_caption_recommendation'] = True
                stem_options['caption_text'] = caption_text
            extra_models = self.extra_models_input.text().strip()
            if extra_models:
                models = [m.strip() for m in extra_models.split(',') if m.strip()]
                stem_options['instrument_models'] = models

        options = {
            "stem_source": ("kaggle_demucs" if stem_source == "Separate via Kaggle (Demucs)" else ("mvsep" if stem_source == "Separate via MVSEP" else "import")),
            "use_deepseek": self.struct_deepseek_check.isChecked(),
            "use_lyrics": self.struct_seg_combo.currentText() == "Lyrics tags",
            "humanize": humanize,
            "humanize_preset": humanize_preset,
            "band_context": band_context,
            "instrument_context": instrument_context,
            "production_context": production_context,
            "vocal_context": vocal_context,
            "extra_notes": extra_notes,
            "stem_options": stem_options
        }

        # ---- Start batch worker ----
        self.run_struct_btn.setEnabled(False)
        self.struct_progress.setVisible(True)
        self.struct_progress.setValue(0)
        self.struct_status.setText(f"Starting structural pipeline on {len(tracks)} track(s)...")

        self.batch_worker = StructuralPipelineBatchWorker(
            tracks=tracks,
            config=self.config,
            options=options
        )
        self.batch_worker.progress.connect(self.on_struct_progress)
        self.batch_worker.track_done.connect(self.on_struct_track_done)
        self.batch_worker.all_done.connect(self.on_struct_batch_done)
        self.batch_worker.error_occurred.connect(self.on_struct_error)
        self.batch_worker.start()

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

    def on_struct_track_done(self, track_id, result):
        # Update the sample with the results from this track
        for sample in self.dataset["samples"]:
            if sample["id"] == track_id:
                self.record_snapshot()
                sample["caption"] = result["final_caption"]
                sample["structural_segments"] = result["sections"]
                sample["stem_paths"] = result["stem_paths"]
                sample["chunk_paths"] = result["chunk_paths"]
                tags = result.get("tags") or {}
                if tags:
                    sample["tags"] = tags
                    if tags.get("bpm"):
                        sample["bpm"] = tags["bpm"]
                    if tags.get("key"):
                        sample["keyscale"] = tags["key"]
                break
        self.refresh_table()
        self.on_table_selection_changed()

    def on_struct_batch_done(self):
        self.run_struct_btn.setEnabled(True)
        self.struct_progress.setVisible(False)
        self.struct_status.setText("Structural pipeline completed for all tracks.")
        QMessageBox.information(self, "Pipeline Complete", "All selected tracks have been processed.")

    def detect_instruments_for_separation(self):
        """Detect instruments (tagger-based, no API key) and recommend MVSEP models."""
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "No Track Selected", "Please select a track first.")
            return
        audio_path = selected.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.warning(self, "Missing Audio", "The selected track's audio file is missing on disk.")
            return

        from workers.instrument_detect import recommend_from_tagger
        try:
            models, instruments = recommend_from_tagger(audio_path, self.config)
        except Exception as e:  # noqa: BLE001
            models, instruments = [], []
            print(f"tagger recommend failed: {e}")

        if instruments:
            lines = [f"• {i}" for i in instruments]
            lines.append("")
            lines.append("Recommended models:")
            if models:
                lines.extend(f"  {m}" for m in models)
            else:
                lines.append("  (no instrument-specific model matched)")
            self.detected_instruments_list.setText("\n".join(lines))
            self.extra_models_input.setText(", ".join(models))
            self.status_label.setText("Instruments detected — review the recommended models before running.")
            return

        # Fallback: keyword-match against the existing caption.
        caption = selected.get("caption", "")
        if not caption:
            self.detected_instruments_list.setText(
                "No instruments matched. Run the AI captioner first, then try again."
            )
            return
        try:
            from stem_separator import StemSeparator
            separator = StemSeparator(self.config)
            instrument_map = separator._instrument_to_model_map()
            detected = []
            caption_lower = caption.lower()
            for keyword, model_name in instrument_map.items():
                if keyword in caption_lower:
                    detected.append(f"{keyword} → {model_name}")
            if detected:
                self.detected_instruments_list.setText("\n".join(detected))
            else:
                self.detected_instruments_list.setText("No known instruments detected in caption.")
        except Exception as e:  # noqa: BLE001
            self.detected_instruments_list.setText(f"Error: {e}")

    def load_band_profiles(self):
        """Load band profiles from band_profiles.json."""
        path = os.path.join(os.path.dirname(__file__), "band_profiles.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading band profiles: {e}")
        return {}

    def refresh_band_profiles(self):
        """Populate the band combo box with loaded profiles."""
        profiles = self.load_band_profiles()
        self.band_combo.clear()
        self.band_combo.addItem("None")
        for band in profiles.keys():
            self.band_combo.addItem(band)

    def on_band_changed(self, band_name):
        """Update the era combo when a band is selected."""
        self.era_combo.clear()
        self.era_combo.addItem("None")
        if band_name == "None":
            return
        profiles = self.load_band_profiles()
        band_data = profiles.get(band_name, {})
        for era in band_data.get("eras", []):
            self.era_combo.addItem(era["name"])

    def on_struct_scope_changed(self, scope_text):
        show = scope_text == "Selected Tracks (from list)"
        self.track_list_widget.setVisible(show)
        self.track_numbers_input.setVisible(show)
        if show:
            self.refresh_track_list()

    def refresh_track_list(self):
        """Populate the track list with numbered filenames."""
        self.track_list_widget.clear()
        samples = self.dataset.get("samples", [])
        for i, s in enumerate(samples, start=1):
            name = s.get("filename", f"Track {i}")
            self.track_list_widget.addItem(f"{i} - {name}")

    def _parse_track_numbers(self, text):
        """
        Parse user input like "1,3,5" or "1-3,5" or "1 3 5" into a list of ints.
        Returns list of ints or None if invalid.
        """
        indices = []
        parts = text.replace(',', ' ').split()
        for part in parts:
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start = int(start.strip())
                    end = int(end.strip())
                    if start > end:
                        start, end = end, start
                    indices.extend(range(start, end + 1))
                except ValueError:
                    return None
            else:
                try:
                    indices.append(int(part))
                except ValueError:
                    return None
        return sorted(set(indices))

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

    def _browse_stem_dir(self):
        start = self.stem_out_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Choose Stem Output Folder", start)
        if d:
            self.stem_out_edit.setText(d)

    def _remembered_secret_keys(self):
        """Secret keys the user asked to persist (checked 'Remember' boxes)."""
        return {
            key
            for key, checked in [
                ("kaggle_key", self.remember_kaggle.isChecked()),
                ("custom_key", self.remember_custom.isChecked()),
                ("mvsep_api_key", self.remember_mvsep.isChecked()),
                ("gemini_api_key", self.remember_gemini.isChecked()),
                ("hf_token", self.remember_hf.isChecked()),
                ("openrouter_key", self.remember_openrouter.isChecked()),
                ("groq_key", self.remember_groq.isChecked()),
            ]
            if checked
        }

    def _caption_backend_value(self):
        text = self.caption_backend_combo.currentText()
        for key in ("ace_step", "gemini", "deepseek", "custom"):
            if text.startswith(key):
                return key
        return "ace_step"

    def _on_llm_provider_changed(self, *_):
        """Auto-fill model + base URL + note when the LLM provider changes."""
        from modules.llm_client import PROVIDERS

        name = self.llm_provider_combo.currentText().split(" ")[0]
        info = PROVIDERS.get(name, PROVIDERS["deepseek"])
        self.llm_model_combo.clear()
        self.llm_model_combo.addItems([
            info["model"], "deepseek-chat", "gemini-2.5-flash", "gemini-3.7-flash",
            "llama-3.3-70b-versatile", "meta-llama/llama-3.3-70b-instruct:free",
        ])
        cfg_model = (self.config.get("llm_model") or "").strip()
        self.llm_model_combo.setCurrentText(cfg_model or info["model"])
        self.llm_base_url_edit.setText(
            (self.config.get("llm_base_url") or "").strip() or info["base_url"]
        )
        if info.get("free") is True:
            self.llm_note.setText("💡 Free provider. " + info.get("note", ""))
        elif info.get("free") is False:
            self.llm_note.setText(info.get("note", ""))
        else:
            self.llm_note.setText(
                "Point the base URL at any OpenAI-compatible server (vLLM / Ollama / llama.cpp / rented GPU)."
            )

    def save_cloud_config(self):
        self.config["kaggle_user"] = self.k_user.text().strip()
        self.config["kaggle_key"] = self.k_key.text().strip()
        self.config["custom_url"] = self.custom_url.text().strip()
        self.config["custom_key"] = self.custom_key.text().strip()
        self.config["mvsep_api_key"] = self.mvsep_key.text().strip()
        self.config["remember_kaggle_key"] = self.remember_kaggle.isChecked()
        self.config["remember_custom_key"] = self.remember_custom.isChecked()
        self.config["remember_mvsep_api_key"] = self.remember_mvsep.isChecked()
        self.config["caption_backend"] = self._caption_backend_value()
        self.config["gemini_api_key"] = self.gemini_key.text().strip()
        self.config["gemini_model"] = self.gemini_model_combo.currentText().strip()
        self.config["custom_caption_url"] = self.custom_url_edit.text().strip()
        self.config["custom_caption_model"] = self.custom_model_edit.text().strip()
        self.config["custom_caption_audio"] = self.custom_audio_check.isChecked()
        self.config["remember_gemini_key"] = self.remember_gemini.isChecked()
        self.config["model_download_source"] = (
            "github" if self.model_source_combo.currentText().startswith("git") else "hf"
        )
        self.config["hf_token"] = self.hf_token_edit.text().strip()
        self.config["remember_hf_token"] = self.remember_hf.isChecked()
        self.config["model_dir"] = self.model_dir_edit.text().strip() or "models"
        self.config["llm_provider"] = self.llm_provider_combo.currentText().split(" ")[0]
        self.config["llm_model"] = self.llm_model_combo.currentText().strip()
        self.config["llm_base_url"] = self.llm_base_url_edit.text().strip()
        self.config["openrouter_key"] = self.openrouter_key.text().strip()
        self.config["remember_openrouter_key"] = self.remember_openrouter.isChecked()
        self.config["groq_key"] = self.groq_key.text().strip()
        self.config["remember_groq_key"] = self.remember_groq.isChecked()
        remember = self._remembered_secret_keys()
        try:
            save_config(self.config, remember=remember)
            self.status_label.setText("Cloud credentials saved.")
        except Exception as e:  # noqa: BLE001
            self.status_label.setText("Cloud credentials kept in memory only.")
            print(f"save_config failed: {e}")

    def save_pipeline_defaults(self):
        self.config["caption_prompt"] = self.prompt_edit.toPlainText().strip()
        self.config["caption_max_tokens"] = self.max_tokens_spin.value()
        self.config["caption_max_audio_duration"] = self.max_dur_spin.value()
        self.config["caption_batch_size"] = self.batch_size_spin.value()
        self.config["tag_caption_ratio"] = self.tag_ratio_spin.value()
        self.config["use_clap_tagger"] = {
            "auto (use CLAP if installed)": "auto",
            "on": "on",
            "off": "off",
        }.get(self.clap_tagger_combo.currentText(), "auto")
        self.config["auto_recommend_models"] = self.auto_recommend_check.isChecked()
        self.config["lead_vocal_splitter"] = {
            "off": "off",
            "mvsep (backing-vocal model)": "mvsep",
            "heuristic (experimental)": "heuristic",
        }.get(self.lead_vocal_combo.currentText(), "off")
        self.config["segment_min_sec"] = self.min_sec_spin.value()
        self.config["segment_max_k"] = self.max_k_spin.value()
        self.config["structure_backend"] = (
            "songformer" if self.structure_backend_combo.currentIndex() == 1 else "librosa"
        )
        self.config["kaggle_stem_model"] = self.stem_model_combo.currentText()
        self.config["stem_output_dir"] = self.stem_out_edit.text().strip()
        self.config["dsp_target_lufs"] = self.lufs_spin.value()
        self.config["dsp_target_sr"] = self.sr_spin.value()
        try:
            save_config(self.config, remember=self._remembered_secret_keys())
            self.status_label.setText("Pipeline defaults saved.")
        except Exception as e:  # noqa: BLE001
            self.status_label.setText("Pipeline defaults kept in memory only.")
            print(f"save_config failed: {e}")

    # -----------------------------------------------------------------------
    # Model Manager
    # -----------------------------------------------------------------------
    def _populate_model_picker(self):
        self.model_pick_combo.clear()
        for m in load_catalog().get("models", []):
            self.model_pick_combo.addItem(f"{m['id']} — {m.get('name', '')}", m["id"])

    def _selected_model_entry(self):
        model_id = self.model_pick_combo.currentData()
        return find_model(model_id) if model_id else None

    def refresh_model_status(self):
        entry = self._selected_model_entry()
        if not entry:
            self.model_status.setText("Select a model from the list.")
            return
        source = str(self.config.get("model_download_source", "hf") or "hf").lower()
        if not (entry.get("hf_repo") or entry.get("github_url")):
            note = entry.get("note", "")
            self.model_status.setText(
                f"'{entry['id']}' is API-only (preferred backend: {entry.get('preferred_backend', '?')}). "
                f"{note}"
            )
            return
        if is_downloaded(self.config, entry):
            self.model_status.setText(
                f"✅ Downloaded to {self.config.get('model_dir', 'models')}/{entry['id']}."
            )
        else:
            self.model_status.setText(
                f"⬇ Not downloaded. Source: {'GitHub' if source.startswith('git') else 'Hugging Face'}. "
                f"{entry.get('note', '')}"
            )

    def download_selected_model(self):
        entry = self._selected_model_entry()
        if not entry:
            QMessageBox.information(self, "Model Manager", "Select a model from the list first.")
            return
        source = str(self.config.get("model_download_source", "hf") or "hf").lower()
        if source.startswith("git"):
            if not entry.get("github_url"):
                QMessageBox.warning(self, "No GitHub URL", f"'{entry['id']}' has no GitHub URL.")
                return
        else:
            if not entry.get("hf_repo"):
                QMessageBox.warning(
                    self,
                    "No HF Repo",
                    f"'{entry['id']}' has no Hugging Face repo. Switch the download source to GitHub, "
                    "or it may be an MVSEP API-only model.",
                )
                return
        self.model_status.setText(f"Downloading {entry['id']}...")
        self.model_worker = ModelDownloadWorker(self.config, entry["id"], parent=self)
        self.model_worker.progress.connect(
            lambda p, m: self.model_status.setText(f"({p}%) {m}")
        )
        self.model_worker.finished_ok.connect(self.on_model_downloaded)
        self.model_worker.failed.connect(self.on_model_download_failed)
        self.model_worker.start()

    def on_model_downloaded(self, model_id):
        self.model_status.setText(f"✅ Downloaded {model_id}.")
        self.refresh_model_status()

    def on_model_download_failed(self, err):
        self.model_status.setText(f"❌ Download failed: {err}")

    def remove_selected_model(self):
        entry = self._selected_model_entry()
        if not entry:
            return
        if not is_downloaded(self.config, entry):
            self.model_status.setText(f"'{entry['id']}' is not downloaded.")
            return
        reply = QMessageBox.question(
            self, "Remove Model",
            f"Delete the local files for '{entry['id']}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            remove_model(self.config, entry)
            self.refresh_model_status()

    def open_selected_leaderboard(self):
        url = self.leaderboard_combo.currentData()
        if url:
            QDesktopServices.openUrl(QUrl(url))

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
            target_sr=int(self.config.get("dsp_target_sr", 44100)),
            target_lufs=float(self.config.get("dsp_target_lufs", -14.0))
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

        # Backend is user-configurable in ⚙ Settings; resolve() falls back
        # gracefully (ace_step without Kaggle creds -> DeepSeek -> local).
        backend = resolve_backend(self.config)

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
        model_id = self.config.get("caption_backend", "ace_step")
        for sample in self.dataset["samples"]:
            if sample.get("id") == sid:
                self.save_ai_caption_result(sample, caption, model_id=model_id, prompt="Detailed ACE-Step caption request")
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

    def on_struct_batch_done(self):
        self.run_struct_btn.setEnabled(True)
        self.struct_progress.setVisible(False)
        self.struct_status.setText("Structural pipeline completed for all tracks.")
        QMessageBox.information(self, "Pipeline Complete", "All selected tracks have been processed.")

    def detect_instruments_for_separation(self):
        QMessageBox.information(
            self,
            "Detect Instruments",
            "This feature is not yet fully implemented.\n"
            "Please run the AI captioner on the track first,\n"
            "then click 'Detect via Captioner' again."
        )

# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())


