"""MVSEP / Kaggle stem-separation tab.

A working MVSEP client GUI (ported from the official ``mvsep_client_gui``
PyQt6 reference app) integrated with the ACE-Step Dataset Toolkit:

* loads the LIVE algorithm list from MVSEP — the newest models are always listed,
* stores / reads the API token from the shared app config (settings.json),
* and can add finished stems straight into the Dataset Studio.
"""
import json
import os
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QMimeData
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from modules.mvsep_api import (
    create_separation,
    download_file,
    get_algorithms,
    poll_until_done,
    resolve_algorithm_id,
    resolve_default_first_stage,
    run_full_separation,
)
from workers.kaggle_stems import KaggleStemSeparator


class AlgorithmLoader(QThread):
    loaded = Signal(dict, dict)   # by_id, fields_by_id
    failed = Signal(str)

    def run(self):
        try:
            by_id, fields = get_algorithms()
            self.loaded.emit(by_id, fields)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class SepPollThread(QThread):
    finished_ok = Signal(str, list)   # job hash -> saved stem paths
    failed = Signal(str)

    def __init__(self, job_hash, api_token, save_path, parent=None):
        super().__init__(parent)
        self.job_hash = job_hash
        self.api_token = api_token
        self.save_path = save_path

    def run(self):
        try:
            data = poll_until_done(self.job_hash, self.api_token)
            saved = []
            for item in ((data.get("data") or {}).get("files") or []):
                url = item.get("url", "").replace("\\/", "/")
                if not url:
                    continue
                fname = item.get("download") or os.path.basename(url.split("?")[0])
                saved.append(download_file(url, fname, self.save_path))
            self.finished_ok.emit(self.job_hash, saved)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class FullSepThread(QThread):
    """Chained full separation: BS PolarFormer (124-band) -> selected model."""

    finished_ok = Signal(str, list)   # placeholder hash -> saved stem paths
    failed = Signal(str)

    def __init__(self, audio_path, api_token, multi_sep_type, output_dir,
                 first_sep_type=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.api_token = api_token
        self.multi_sep_type = multi_sep_type
        self.output_dir = output_dir
        self.first_sep_type = first_sep_type

    def run(self):
        try:
            paths = run_full_separation(
                self.audio_path,
                self.api_token,
                self.multi_sep_type,
                self.output_dir,
                first_sep_type=self.first_sep_type,
                progress=lambda m: None,
            )
            self.finished_ok.emit("full", paths)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class DragButton(QPushButton):
    """QPushButton that also accepts dropped audio files."""

    dragged = Signal()

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, event):
        self.selected_file = ""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                self.selected_file = urls[0].toLocalFile()
            event.accept()
            self.dragged.emit()
        else:
            event.ignore()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            drag.setMimeData(QMimeData())
            drag.exec(Qt.DropAction.MoveAction)


class MVSepTab(QWidget):
    """Separation tab: MVSEP Cloud API or Kaggle GPU (Demucs)."""

    stems_ready = Signal(list)   # list of local stem paths ready for the dataset

    def __init__(self, config, settings_path, parent_window=None):
        super().__init__()
        self.config = config
        self.settings_path = settings_path
        self.main = parent_window

        self.by_id = {}
        self.fields_by_id = {}
        self.alg_opt = [{}, {}, {}]
        self.selected_opt = [0, 0, 0]
        self.selected_file = None
        self.output_dir = os.path.join(str(Path.home()), "mvsep_stems")
        self.job_threads = []
        self._last_stems = []
        self._last_backend = "MVSEP Cloud API"
        self._full_chain = bool(self.config.get("mvsep_full_chain", True))
        self._first_stage_id = str(self.config.get("mvsep_first_stage", "") or "")

        self._build_ui()
        self._reload_algorithms()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        title = QLabel("🔊 MVSEP / Kaggle Stem Separator")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Models are loaded live from MVSEP — the newest algorithms are always "
            "listed. For dataset use, finished stems can be added straight to the "
            "Dataset Studio."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa;")
        layout.addWidget(info)

        # ---- backend selector ----
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["MVSEP Cloud API", "Kaggle GPU (Demucs)"])
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        backend_row.addWidget(self.backend_combo)
        backend_row.addStretch()
        layout.addLayout(backend_row)

        # ---- MVSEP panel ----
        self.mvsep_group = QGroupBox("MVSEP Cloud API")
        mv_form = QFormLayout(self.mvsep_group)

        alg_row = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.currentIndexChanged.connect(self.on_selection_change)
        alg_row.addWidget(self.type_combo, 1)
        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self._reload_algorithms)
        alg_row.addWidget(self.refresh_btn)
        mv_form.addRow("Separation type:", alg_row)

        self.alg_count_label = QLabel("Loading algorithm list…")
        self.alg_count_label.setStyleSheet("color: #888; font-size: 10px;")
        mv_form.addRow("", self.alg_count_label)

        self.token_input = QLineEdit(self.config.get("mvsep_api_key", ""))
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("Paste MVSEP API token")
        mv_form.addRow("API Token:", self.token_input)

        token_link = QLabel("<a href='https://mvsep.com/ru/full_api'>Get an MVSEP token</a>")
        token_link.setOpenExternalLinks(True)
        mv_form.addRow("", token_link)

        self.full_chain_check = QCheckBox(
            "Full separation: first stage, then selected model on the instrumental"
        )
        self.full_chain_check.setChecked(self._full_chain)
        self.full_chain_check.toggled.connect(self._on_full_chain_toggled)
        mv_form.addRow("", self.full_chain_check)

        self.first_stage_combo = QComboBox()
        self.first_stage_combo.setToolTip(
            "First stage of a full separation (default: BS PolarFormer 124-band — "
            "it separates vocals + instrumental and re-synthesizes the instrumental "
            "frequencies, preventing artifacts and clipping). Pick any algorithm "
            "from the live list to override."
        )
        self.first_stage_combo.currentIndexChanged.connect(self._on_first_stage_changed)
        self.first_stage_combo.setEnabled(self._full_chain)
        mv_form.addRow("First step:", self.first_stage_combo)

        self.opt_labels = []
        self.opt_combos = []
        for i in range(3):
            lbl = QLabel(f"Additional Option {i+1}")
            cmb = QComboBox()
            cmb.currentIndexChanged.connect(lambda idx, k=i: self.on_option_changed(k, idx))
            self.opt_labels.append(lbl)
            self.opt_combos.append(cmb)
            mv_form.addRow(lbl, cmb)

        layout.addWidget(self.mvsep_group)

        # ---- Kaggle panel ----
        self.kaggle_group = QGroupBox("Kaggle GPU (Demucs)")
        kg_form = QFormLayout(self.kaggle_group)
        self.kaggle_model_combo = QComboBox()
        self.kaggle_model_combo.addItems([
            "htdemucs_ft (4-stem, recommended)",
            "htdemucs (4-stem)",
            "htdemucs_6s (6-stem: + guitar, piano)",
        ])
        kg_form.addRow("Model:", self.kaggle_model_combo)
        self.two_stems_combo = QComboBox()
        self.two_stems_combo.addItems([
            "Full multi-stem",
            "Vocals only",
            "Instrumental (no vocals)",
        ])
        kg_form.addRow("Output:", self.two_stems_combo)
        creds_note = QLabel("Requires Kaggle credentials saved in ⚙ Settings.")
        creds_note.setStyleSheet("color: #888; font-size: 10px;")
        kg_form.addRow("", creds_note)
        self.kaggle_group.setVisible(False)
        layout.addWidget(self.kaggle_group)

        # ---- file + output ----
        self.file_button = DragButton("🎵 Select Audio File (or drop it here)")
        self.file_button.clicked.connect(self.select_file)
        self.file_button.dragged.connect(self.select_drag_file)
        layout.addWidget(self.file_button)

        self.filename_label = QLabel("No file selected")
        self.filename_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.filename_label)

        out_row = QHBoxLayout()
        self.output_dir_label = QLabel(f"Output: {self.output_dir}")
        self.output_dir_label.setStyleSheet("color: #aaa;")
        out_row.addWidget(self.output_dir_label, 1)
        out_btn = QPushButton("Choose Folder…")
        out_btn.clicked.connect(self.select_output_dir)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        self.run_button = QPushButton("▶ Run Separation")
        self.run_button.setStyleSheet(
            "font-weight: bold; padding: 10px; background-color: #0e639c;"
        )
        self.run_button.clicked.connect(self.process_separation)
        layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.status_label)

        self.add_to_dataset_btn = QPushButton("📥 Add stems to Dataset Studio")
        self.add_to_dataset_btn.clicked.connect(self._add_stems_to_dataset)
        self.add_to_dataset_btn.setEnabled(False)
        layout.addWidget(self.add_to_dataset_btn)

        layout.addStretch()

    # ---------------------------------------------------------------- logic
    def _reload_algorithms(self):
        self.alg_count_label.setText("Loading algorithm list…")
        self.type_combo.clear()
        self.loader = AlgorithmLoader(self)
        self.loader.loaded.connect(self._on_algorithms_loaded)
        self.loader.failed.connect(self._on_algorithms_failed)
        self.loader.start()

    def _on_algorithms_loaded(self, by_id, fields_by_id):
        self.by_id = by_id
        self.fields_by_id = fields_by_id
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for render_id in sorted(by_id, key=lambda r: by_id[r].lower()):
            self.type_combo.addItem(by_id[render_id], render_id)
        self.type_combo.blockSignals(False)
        self.alg_count_label.setText(f"{len(by_id)} algorithms available (live from MVSEP)")
        if self.type_combo.count():
            self.on_selection_change(0)

        # First-stage selector: default = saved choice, else BS PolarFormer.
        self.first_stage_combo.blockSignals(True)
        self.first_stage_combo.clear()
        for render_id in sorted(by_id, key=lambda r: by_id[r].lower()):
            self.first_stage_combo.addItem(by_id[render_id], render_id)
        default_id = self._first_stage_id or resolve_default_first_stage(by_id)
        idx = self.first_stage_combo.findData(default_id)
        self.first_stage_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.first_stage_combo.blockSignals(False)

    def _on_algorithms_failed(self, err):
        self.alg_count_label.setText(f"Could not load algorithm list: {err}")
        self.status_label.setText(f"MVSEP algorithm list unavailable: {err}")

    def on_backend_changed(self, name):
        self._last_backend = name
        self.mvsep_group.setVisible(name == "MVSEP Cloud API")
        self.kaggle_group.setVisible(name == "Kaggle GPU (Demucs)")

    def on_selection_change(self, index):
        render_id = self.type_combo.itemData(index)
        if not render_id:
            return
        fields = self.fields_by_id.get(render_id, [])
        for k in range(3):
            self.opt_combos[k].blockSignals(True)
            self.opt_combos[k].clear()
            self.opt_combos[k].blockSignals(False)
            self.selected_opt[k] = 0
            self.alg_opt[k] = {}
            if k < len(fields):
                self.opt_labels[k].setText(
                    f"Additional Option {k+1}: {fields[k].get('text', '')}"
                )
                opts = fields[k].get("options")
                if opts:
                    try:
                        self.alg_opt[k] = json.loads(opts)
                    except Exception:  # noqa: BLE001
                        self.alg_opt[k] = {}
                    self.opt_combos[k].addItems(
                        [v for _, v in sorted(self.alg_opt[k].items())]
                    )
            else:
                self.opt_labels[k].setText(f"Additional Option {k+1}")

    def on_option_changed(self, k, index):
        items = sorted(self.alg_opt[k].items())
        if 0 <= index < len(items):
            self.selected_opt[k] = items[index][0]

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.flac *.ogg)"
        )
        if path:
            self._set_file(path)

    def select_drag_file(self):
        if getattr(self.file_button, "selected_file", None):
            self._set_file(self.file_button.selected_file)

    def _set_file(self, path):
        self.selected_file = path
        self.filename_label.setText(f"Selected: {os.path.basename(path)}")

    def select_output_dir(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_dir)
        if chosen:
            self.output_dir = chosen
            self.output_dir_label.setText(f"Output: {self.output_dir}")

    # ------------------------------------------------------------- execution
    def process_separation(self):
        if not self.selected_file or not os.path.exists(self.selected_file):
            QMessageBox.warning(self, "No Audio File", "Select an audio file first.")
            return

        backend = self.backend_combo.currentText()
        if backend == "Kaggle GPU (Demucs)":
            self._run_kaggle()
        else:
            self._run_mvsep()

    def _run_mvsep(self):
        api_token = self.token_input.text().strip()
        if not api_token:
            QMessageBox.warning(self, "API Token", "Enter your MVSEP API token first.")
            return
        self._save_token(api_token)

        render_id = self.type_combo.itemData(self.type_combo.currentIndex())
        selected_name = self.type_combo.currentText()
        if not render_id:
            QMessageBox.warning(self, "No Algorithm", "Pick a separation type first.")
            return

        self.run_button.setEnabled(False)

        # Full stem separation: BS PolarFormer (124-band) first, then the
        # selected model on the instrumental.
        use_full_chain = self.full_chain_check.isChecked()
        if use_full_chain and "polarformer" not in selected_name.lower():
            self.status_label.setText("Full separation: PolarFormer → selected model…")
            thread = FullSepThread(
                self.selected_file,
                api_token,
                render_id,
                self.output_dir,
                first_sep_type=self.first_stage_combo.currentData(),
                parent=self,
            )
            thread.finished_ok.connect(self._on_sep_done)
            thread.failed.connect(self._on_sep_failed)
            self.job_threads.append(thread)
            thread.start()
            return

        # Single-algorithm separation.
        self.status_label.setText("Creating MVSEP separation job…")
        job_hash, status = create_separation(
            self.selected_file,
            api_token,
            render_id,
            self.selected_opt[0],
            self.selected_opt[1],
            self.selected_opt[2],
        )
        if status != 200:
            self.run_button.setEnabled(True)
            self.status_label.setText(f"MVSEP job creation failed (HTTP {status}).")
            return

        poller = SepPollThread(job_hash, api_token, self.output_dir, self)
        poller.finished_ok.connect(self._on_sep_done)
        poller.failed.connect(self._on_sep_failed)
        self.job_threads.append(poller)
        poller.start()
        self.status_label.setText(f"Separation queued. Job: {job_hash}. Polling…")

    def _run_kaggle(self):
        user = self.config.get("kaggle_user", "").strip()
        key = self.config.get("kaggle_key", "").strip()
        if not user or not key:
            QMessageBox.warning(
                self,
                "Kaggle Credentials",
                "Save your Kaggle Username & API Key in ⚙ Settings first.",
            )
            return

        model_label = self.kaggle_model_combo.currentText()
        model = {
            "htdemucs_ft (4-stem, recommended)": "htdemucs_ft",
            "htdemucs (4-stem)": "htdemucs",
            "htdemucs_6s (6-stem: + guitar, piano)": "htdemucs_6s",
        }.get(model_label, "htdemucs_ft")

        two_stems = None
        idx = self.two_stems_combo.currentIndex()
        if idx == 1:
            two_stems = "vocals"       # vocals + no_vocals
        elif idx == 2:
            two_stems = "vocals"       # keep only no_vocals (filtered below)
            self._kaggle_two_stems_mode = "instrumental"
        else:
            self._kaggle_two_stems_mode = "full"
        if idx == 1:
            self._kaggle_two_stems_mode = "vocals"

        self.status_label.setText("Pushing Demucs kernel to Kaggle GPU…")
        self.run_button.setEnabled(False)
        worker = KaggleStemSeparator(
            self.selected_file,
            self.config,
            model=model,
            two_stems=two_stems,
            output_dir=self.output_dir,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_kaggle_done)
        worker.failed.connect(self._on_sep_failed)
        self.job_threads.append(worker)
        worker.start()

    def _on_progress(self, pct, msg):
        self.status_label.setText(msg)

    def _on_sep_done(self, job_hash, paths):
        self.run_button.setEnabled(True)
        self._last_stems = paths
        self.add_to_dataset_btn.setEnabled(bool(paths))
        self.status_label.setText(
            f"Done — {len(paths)} stem file(s) saved to {self.output_dir}"
        )

    def _on_kaggle_done(self, paths):
        mode = getattr(self, "_kaggle_two_stems_mode", "full")
        if mode == "vocals":
            paths = [p for p in paths if "vocals" in os.path.basename(p)]
        elif mode == "instrumental":
            paths = [p for p in paths if "no_vocals" in os.path.basename(p)]
        self._on_sep_done(None, paths)

    def _on_sep_failed(self, err):
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Separation failed: {err}")
        QMessageBox.critical(self, "Separation Error", str(err))

    def _add_stems_to_dataset(self):
        if self._last_stems:
            self.stems_ready.emit(list(self._last_stems))

    def _persist_settings(self):
        try:
            from modules.config_store import save_config
            save_config(self.config)
        except OSError as e:
            self.status_label.setText(f"Could not persist settings: {e}")

    def _save_token(self, token):
        self.config["mvsep_api_key"] = token
        self._persist_settings()

    def _on_full_chain_toggled(self, checked):
        self._full_chain = bool(checked)
        self.first_stage_combo.setEnabled(checked)
        self.config["mvsep_full_chain"] = bool(checked)
        self._persist_settings()

    def _on_first_stage_changed(self, index):
        rid = self.first_stage_combo.itemData(index)
        if rid is not None:
            self.config["mvsep_first_stage"] = str(rid)
            self._persist_settings()



