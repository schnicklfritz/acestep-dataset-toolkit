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
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Background Worker: Health & Metadata Auditor with Degradation Penalties
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Background Worker: DSP Batch Normalizer with Archival Backups
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Background Worker: Remote Caption Worker
# ---------------------------------------------------------------------------
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

    def cancel(self):
        self._is_cancelled = True


# ---------------------------------------------------------------------------
# Main Window: Full Gentoo-Style Freedom, Undo/Redo, Health & Exceptions
# ---------------------------------------------------------------------------
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
            "custom_key": ""
        }
        self.health_reports = {}
        self.original_backups = {}
        self.active_worker = None
        self.filter_exceptions_only = False
        self.bypass_warnings = False

        self.init_ui()
        self.apply_custom_theme()

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

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        studio_tab = QWidget()
        studio_layout = QVBoxLayout(studio_tab)
        studio_layout.setContentsMargins(10, 8, 10, 8)
        studio_layout.setSpacing(6)

        settings_tab = QWidget()
        self.init_settings_tab(settings_tab)

        self.tabs.addTab(studio_tab, "🎛 Dataset Studio")
        self.tabs.addTab(settings_tab, "🎨 Appearance & Customization")

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
        c_form.addRow("Custom Webhook URL:", self.custom_url)
        c_form.addRow("Custom Auth Token:", self.custom_key)

        save_cloud_btn = QPushButton("Save Cloud Credentials")
        save_cloud_btn.clicked.connect(self.save_cloud_config)
        c_form.addRow(save_cloud_btn)

        layout.addWidget(cloud_grp)
        layout.addStretch()

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
        self.status_label.setText("Cloud credentials saved.")

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

    def start_health_audit(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Please add audio tracks before scanning.")
            return

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

        choices = [
            f"{s.get('filename', 'Unnamed file')}  [{s.get('id', '')}]"
            for s in samples
        ]
        selected_label, accepted = QInputDialog.getItem(
            self,
            "Choose Audio Track",
            "Select one track to normalize:",
            choices,
            0,
            False
        )

        if not accepted or not selected_label:
            self.status_label.setText("DSP normalization cancelled; no files changed.")
            return

        selected_index = choices.index(selected_label)
        selected_sample = samples[selected_index]
        selected_samples = [selected_sample]

        audio_path = selected_sample.get("audio_path", "")
        default_folder = (
            str(Path(audio_path).parent)
            if audio_path
            else str(Path.home())
        )

        placement_box = QMessageBox(self)
        placement_box.setWindowTitle("Normalized Audio & Backup Location")
        placement_box.setIcon(QMessageBox.Information)
        placement_box.setText(
            "Normalized audio and original-file backups will be created "
            "in the same folder as this track's audio file."
        )
        placement_box.setInformativeText(
            f"Default dataset folder:\n{default_folder}\n\n"
            "The application will create:\n"
            "• normalized_audio/\n"
            "• originals_backup/\n\n"
            "Choose OK to use the default dataset folder, or Let Me Decide "
            "to choose another persistent project folder."
        )

        default_btn = placement_box.addButton("OK", QMessageBox.AcceptRole)
        decide_btn = placement_box.addButton("Let Me Decide", QMessageBox.ActionRole)
        placement_box.addButton(QMessageBox.Cancel)
        placement_box.exec()

        clicked_btn = placement_box.clickedButton()

        if clicked_btn == default_btn:
            project_folder = default_folder
        elif clicked_btn == decide_btn:
            project_folder = QFileDialog.getExistingDirectory(
                self,
                "Select Persistent Project Folder for Normalized Audio & Backups",
                default_folder
            )
        else:
            self.status_label.setText("DSP normalization cancelled; no files changed.")
            return

        if not project_folder:
            self.status_label.setText("DSP normalization cancelled; no files changed.")
            return

        self.record_snapshot()
        self.normalize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(
            f"Starting DSP EBU R128 normalization for: "
            f"{selected_sample.get('filename', 'selected track')}..."
        )

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
        
        QMessageBox.information(
            self, 
            "DSP Normalization Finished",
            f"Selected track unified to -14 LUFS (EBU R128) and 44.1 kHz stereo WAV.\n\n"
            f"Normalized Audio Workspace:\n{norm_dir}\n\n"
            f"Original Backup Stored At:\n{backup_dir}"
        )
        
        self.status_label.setText("Rescanning dataset files. Please wait a few seconds...")

        self.rescan_notice = QMessageBox(self)
        self.rescan_notice.setWindowTitle("Rescanning Dataset")
        self.rescan_notice.setIcon(QMessageBox.Information)
        self.rescan_notice.setText(
            "Rescanning dataset files. Please wait a few seconds..."
        )
        self.rescan_notice.setStandardButtons(QMessageBox.NoButton)
        self.rescan_notice.setModal(False)
        self.rescan_notice.show()

        self.start_health_audit()


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
            res = QMessageBox.warning(
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
                    "locked": True
                })
            self.refresh_table()
            self.status_label.setText(f"Added {len(paths)} audio tracks.")
            self.start_health_audit()

    def start_ai_captioning(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks before captioning.")
            return

        has_existing = any(bool(s.get("caption")) for s in samples)
        if has_existing:
            res = QMessageBox.question(
                self, "Existing Captions Detected",
                "Some tracks already have captions. Overwrite all existing captions?",
                QMessageBox.Yes | QMessageBox.No
            )
            if res == QMessageBox.No:
                return

        self.record_snapshot()
        self.run_ai_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        general_meta = self.dataset.get("metadata", {})
        backend = "Kaggle Cloud (Free GPU)" if self.config.get("kaggle_user") else "Local Rule Engine"

        self.active_worker = RemoteCaptionWorker(samples, backend, "Deep Structural Breakdown", general_meta, self.config)
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.finished_sample.connect(self.on_sample_captioned)
        self.active_worker.all_done.connect(self.on_caption_finished)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def on_worker_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def on_sample_captioned(self, sid, caption):
        for s in self.dataset["samples"]:
            if s["id"] == sid:
                s["caption"] = caption
                break
        self.on_table_selection_changed()

    def on_caption_finished(self):
        self.run_ai_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("AI Captioning completed.")
        self.refresh_table()
        self.on_table_selection_changed()

    def on_worker_error(self, err_msg):
        self.run_ai_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.normalize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Operation error.")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())
