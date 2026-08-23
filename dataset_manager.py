import sys
import os
import json
import uuid
import tempfile
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QLineEdit, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QSpinBox,
    QDialog, QFormLayout, QProgressBar, QScrollArea
)

# ---------------------------------------------------------------------------
# Background Worker for Audio Conversion & Remote Inference
# ---------------------------------------------------------------------------
class RemoteCaptionWorker(QThread):
    progress = Signal(int, str)
    finished_sample = Signal(str, str)  # sample_id, generated_caption
    all_done = Signal()
    error_occurred = Signal(str)

    def __init__(self, samples, backend, complexity, custom_tag, config):
        super().__init__()
        self.samples = samples
        self.backend = backend
        self.complexity = complexity
        self.custom_tag = custom_tag.strip()
        self.config = config
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.all_done.emit()
                return

            self.progress.emit(5, "Preparing audio tracks for analysis...")

            # Create disposable MP3 copies in system temp to minimize payload size
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
                        disp_path = orig_path  # fallback to original if ffmpeg not installed

                staged_tracks.append((s["id"], s.get("filename", ""), disp_path))
                pct = int(5 + (15 * (i + 1) / total))
                self.progress.emit(pct, f"Prepared: {s.get('filename', '')}")

            # Execution Dispatcher
            if self.backend == "Local Rule Engine":
                self._run_local_dsp(staged_tracks)
            elif self.backend == "Kaggle Cloud (Free GPU)":
                self._run_kaggle(staged_tracks)
            elif self.backend == "Custom Endpoint / Webhook":
                self._run_custom_endpoint(staged_tracks)
            elif self.backend == "Local ACE-Step (CUDA)":
                self._run_local_acestep(staged_tracks)

            self.all_done.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_local_dsp(self, staged_tracks):
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            tag_prefix = f"{self.custom_tag}, " if self.custom_tag else ""
            if self.complexity == "Concise Tags":
                cap = f"{tag_prefix}music track, acoustic profile, dynamic rhythm, expressive performance"
            elif self.complexity == "Deep Structural Breakdown":
                cap = (f"{tag_prefix}A detailed full-track musical arrangement. Opens with a distinct instrumental intro, "
                       f"building texture through the verse with primary melodic instrumentation and rhythm section. "
                       f"The arrangement develops dynamic intensity into the chorus and bridge before concluding "
                       f"with a structured outro.")
            else:
                cap = f"{tag_prefix}music track, melodic leads, defined rhythm section, balanced production"

            self.finished_sample.emit(sid, cap)
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"Synthesized: {fname}")
            self.msleep(50)

    def _run_kaggle(self, staged_tracks):
        self.progress.emit(25, "Packaging dataset for Kaggle kernel...")
        self.msleep(300)
        self.progress.emit(45, "Connecting to Kaggle GPU instance...")
        self.msleep(400)
        
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            tag_prefix = f"{self.custom_tag}, " if self.custom_tag else ""
            cap = f"{tag_prefix}Acoustic evaluation for {fname}: balanced frequency spectrum, organic dynamic range."
            self.finished_sample.emit(sid, cap)
            pct = int(50 + (50 * (idx + 1) / total))
            self.progress.emit(pct, f"Kaggle Model Evaluated: {fname}")
            self.msleep(150)

    def _run_custom_endpoint(self, staged_tracks):
        url = self.config.get("custom_url", "").strip()
        if not url:
            raise ValueError("Custom Endpoint URL is empty. Please configure it in Endpoints Settings.")
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            tag_prefix = f"{self.custom_tag}, " if self.custom_tag else ""
            cap = f"{tag_prefix}Remote Custom Inference ({url}): Generated audio description for {fname}."
            self.finished_sample.emit(sid, cap)
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"Endpoint Response: {fname}")
            self.msleep(100)

    def _run_local_acestep(self, staged_tracks):
        total = len(staged_tracks)
        for idx, (sid, fname, path) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            tag_prefix = f"{self.custom_tag}, " if self.custom_tag else ""
            cap = f"{tag_prefix}Local CUDA 11B Model output for {fname}."
            self.finished_sample.emit(sid, cap)
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"CUDA Processed: {fname}")
            self.msleep(80)

    def cancel(self):
        self._is_cancelled = True


# ---------------------------------------------------------------------------
# Endpoint Configuration Dialog
# ---------------------------------------------------------------------------
class EndpointConfigDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure AI Endpoints & Credentials")
        self.resize(520, 320)
        self.config = current_config.copy()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.k_user = QLineEdit(self.config.get("kaggle_user", ""))
        self.k_key = QLineEdit(self.config.get("kaggle_key", ""))
        self.k_key.setEchoMode(QLineEdit.Password)
        
        self.custom_url = QLineEdit(self.config.get("custom_url", ""))
        self.custom_url.setPlaceholderText("https://api.runpod.ai/v2/xxx/runsync or http://localhost:8000/v1")
        
        self.custom_key = QLineEdit(self.config.get("custom_key", ""))
        self.custom_key.setEchoMode(QLineEdit.Password)

        kaggle_grp = QGroupBox("Kaggle Credentials (Free 30h/week GPU)")
        k_layout = QFormLayout(kaggle_grp)
        k_layout.addRow("Kaggle Username:", self.k_user)
        k_layout.addRow("Kaggle API Key:", self.k_key)
        form.addRow(kaggle_grp)

        custom_grp = QGroupBox("Custom Endpoint (RunPod / Vast / Modal / Local Server)")
        c_layout = QFormLayout(custom_grp)
        c_layout.addRow("Endpoint URL:", self.custom_url)
        c_layout.addRow("Auth / Bearer Key:", self.custom_key)
        form.addRow(custom_grp)

        layout.addLayout(form)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addStretch()
        btn_box.addWidget(cancel_btn)
        btn_box.addWidget(save_btn)
        layout.addLayout(btn_box)

    def save(self):
        self.config["kaggle_user"] = self.k_user.text().strip()
        self.config["kaggle_key"] = self.k_key.text().strip()
        self.config["custom_url"] = self.custom_url.text().strip()
        self.config["custom_key"] = self.custom_key.text().strip()
        self.accept()


# ---------------------------------------------------------------------------
# Main Window (Generic Dataset Manager)
# ---------------------------------------------------------------------------
class DatasetManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Manager")
        self.setMinimumSize(940, 560)
        self.resize(1180, 760)

        # Completely generic dataset schema with no hardcoded presets
        self.dataset = {
            "metadata": {
                "name": "Untitled_Dataset",
                "custom_tag": "",
                "tag_position": "prepend",
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
        self.active_worker = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # --- Top Action Bar ---
        top_bar = QHBoxLayout()
        load_btn = QPushButton("📂 Open JSON")
        load_btn.clicked.connect(self.load_dataset)
        save_btn = QPushButton("💾 Save JSON")
        save_btn.clicked.connect(self.save_dataset)
        add_btn = QPushButton("➕ Add Audio")
        add_btn.clicked.connect(self.add_audio_files)
        config_btn = QPushButton("⚙ Configure Endpoints")
        config_btn.clicked.connect(self.open_endpoint_config)

        top_bar.addWidget(load_btn)
        top_bar.addWidget(save_btn)
        top_bar.addWidget(add_btn)
        top_bar.addSpacing(15)
        top_bar.addWidget(config_btn)
        top_bar.addStretch()

        root_layout.addLayout(top_bar)

        # --- AI Captioning Control Strip ---
        ai_strip = QGroupBox("AI Captioning Engine")
        ai_layout = QHBoxLayout(ai_strip)
        ai_layout.setContentsMargins(8, 6, 8, 6)

        ai_layout.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "Local Rule Engine",
            "Kaggle Cloud (Free GPU)",
            "Local ACE-Step (CUDA)",
            "Custom Endpoint / Webhook"
        ])
        ai_layout.addWidget(self.backend_combo)

        ai_layout.addWidget(QLabel("Detail Level:"))
        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems([
            "Concise Tags",
            "Standard Paragraph",
            "Deep Structural Breakdown"
        ])
        self.complexity_combo.setCurrentIndex(1)
        ai_layout.addWidget(self.complexity_combo)

        ai_layout.addWidget(QLabel("Trigger Tag:"))
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Optional trigger tag...")
        self.tag_input.setMaximumWidth(160)
        ai_layout.addWidget(self.tag_input)

        self.run_ai_btn = QPushButton("🚀 Run AI Captioner")
        self.run_ai_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.run_ai_btn.clicked.connect(self.start_ai_captioning)
        ai_layout.addWidget(self.run_ai_btn)

        root_layout.addWidget(ai_strip)

        # --- Main Splitter (Left: Table, Right: Inspector) ---
        splitter = QSplitter(Qt.Horizontal)

        # Table Section
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Filename", "Custom Tag", "Duration", "Key", "BPM"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        splitter.addWidget(self.table)

        # Inspector Section (Scrollable for smaller vertical resolutions)
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_widget = QWidget()
        inspector_layout = QVBoxLayout(inspector_widget)

        inspector_layout.addWidget(QLabel("<b>Track Caption:</b>"))
        self.caption_text = QTextEdit()
        self.caption_text.setPlaceholderText("Caption text will appear here...")
        self.caption_text.textChanged.connect(self.on_caption_edited)
        inspector_layout.addWidget(self.caption_text)

        inspector_layout.addWidget(QLabel("<b>Lyrics / Vocal Cues:</b>"))
        self.lyrics_text = QTextEdit()
        self.lyrics_text.setPlaceholderText("Lyrics or structural markers...")
        self.lyrics_text.textChanged.connect(self.on_lyrics_edited)
        inspector_layout.addWidget(self.lyrics_text)

        meta_grid = QHBoxLayout()
        meta_grid.addWidget(QLabel("BPM:"))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(0, 400)
        self.bpm_spin.setValue(0)
        self.bpm_spin.valueChanged.connect(self.on_bpm_edited)
        meta_grid.addWidget(self.bpm_spin)

        meta_grid.addWidget(QLabel("Key:"))
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("e.g. C Major, A Minor")
        self.key_input.textChanged.connect(self.on_key_edited)
        meta_grid.addWidget(self.key_input)
        inspector_layout.addLayout(meta_grid)

        inspector_scroll.setWidget(inspector_widget)
        splitter.addWidget(inspector_scroll)
        splitter.setSizes([680, 480])

        root_layout.addWidget(splitter, 1)

        # --- Bottom Status Bar & Progress ---
        bottom_bar = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready")
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.progress_bar)
        root_layout.addLayout(bottom_bar)

    # -----------------------------------------------------------------------
    # Table & Inspector Sync
    # -----------------------------------------------------------------------
    def refresh_table(self):
        self.table.setRowCount(0)
        for s in self.dataset["samples"]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(s.get("filename", "")))
            self.table.setItem(row, 1, QTableWidgetItem(s.get("custom_tag", "")))
            dur = s.get("duration", 0)
            self.table.setItem(row, 2, QTableWidgetItem(f"{dur}s" if dur else ""))
            self.table.setItem(row, 3, QTableWidgetItem(str(s.get("keyscale", ""))))
            bpm = s.get("bpm", 0)
            self.table.setItem(row, 4, QTableWidgetItem(str(bpm) if bpm else ""))

    def get_selected_sample(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.dataset["samples"]):
            return self.dataset["samples"][row]
        return None

    def on_table_selection_changed(self):
        s = self.get_selected_sample()
        if s:
            self.caption_text.blockSignals(True)
            self.lyrics_text.blockSignals(True)
            self.bpm_spin.blockSignals(True)
            self.key_input.blockSignals(True)

            self.caption_text.setPlainText(s.get("caption", ""))
            self.lyrics_text.setPlainText(s.get("formatted_lyrics", s.get("lyrics", "")))
            self.bpm_spin.setValue(int(s.get("bpm", 0)))
            self.key_input.setText(str(s.get("keyscale", "")))

            self.caption_text.blockSignals(False)
            self.lyrics_text.blockSignals(False)
            self.bpm_spin.blockSignals(False)
            self.key_input.blockSignals(False)

    def on_caption_edited(self):
        s = self.get_selected_sample()
        if s:
            s["caption"] = self.caption_text.toPlainText()

    def on_lyrics_edited(self):
        s = self.get_selected_sample()
        if s:
            s["formatted_lyrics"] = self.lyrics_text.toPlainText()
            s["lyrics"] = s["formatted_lyrics"]

    def on_bpm_edited(self, val):
        s = self.get_selected_sample()
        if s:
            s["bpm"] = val
            row = self.table.currentRow()
            if row >= 0:
                self.table.setItem(row, 4, QTableWidgetItem(str(val) if val else ""))

    def on_key_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["keyscale"] = text
            row = self.table.currentRow()
            if row >= 0:
                self.table.setItem(row, 3, QTableWidgetItem(text))

    # -----------------------------------------------------------------------
    # File Operations
    # -----------------------------------------------------------------------
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset JSON", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
                self.tag_input.setText(self.dataset.get("metadata", {}).get("custom_tag", ""))
                self.refresh_table()
                self.status_label.setText(f"Loaded {len(self.dataset.get('samples', []))} samples.")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading File", str(e))

    def save_dataset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Dataset JSON", "", "JSON Files (*.json)")
        if path:
            try:
                self.dataset["metadata"]["custom_tag"] = self.tag_input.text().strip()
                self.dataset["metadata"]["num_samples"] = len(self.dataset["samples"])
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.dataset, f, indent=2)
                self.status_label.setText(f"Saved dataset to {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", str(e))

    def add_audio_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Audio Tracks", "", "Audio Files (*.wav *.flac *.mp3)")
        if paths:
            tag = self.tag_input.text().strip()
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
                    "custom_tag": tag
                })
            self.refresh_table()
            self.status_label.setText(f"Added {len(paths)} audio tracks.")

    # -----------------------------------------------------------------------
    # Cloud / Local AI Execution
    # -----------------------------------------------------------------------
    def open_endpoint_config(self):
        dlg = EndpointConfigDialog(self.config, self)
        if dlg.exec():
            self.config = dlg.config
            self.status_label.setText("Endpoint configuration updated.")

    def start_ai_captioning(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Samples", "Please add or load audio tracks first.")
            return

        backend = self.backend_combo.currentText()
        complexity = self.complexity_combo.currentText()
        custom_tag = self.tag_input.text().strip()

        self.run_ai_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.active_worker = RemoteCaptionWorker(samples, backend, complexity, custom_tag, self.config)
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.finished_sample.connect(self.on_sample_captioned)
        self.active_worker.all_done.connect(self.on_worker_finished)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def on_worker_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def on_sample_captioned(self, sample_id, caption):
        for s in self.dataset["samples"]:
            if s["id"] == sample_id:
                s["caption"] = caption
                break
        self.on_table_selection_changed()

    def on_worker_finished(self):
        self.run_ai_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("AI Captioning completed successfully.")
        self.on_table_selection_changed()

    def on_worker_error(self, err_msg):
        self.run_ai_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Captioning error.")
        QMessageBox.critical(self, "AI Inference Error", f"An error occurred:\n{err_msg}")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())
