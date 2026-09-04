import json
import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)

from modules.mvsep_api import (
    create_separation,
    get_algorithms,
    get_result_files,
    poll_until_done,
)


class MVSEPSeparationWorker(QThread):
    progress = Signal(str)
    completed = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        audio_path,
        api_token,
        algorithm_id,
        option_1,
        option_2,
        option_3,
        output_dir,
        parent=None,
    ):
        super().__init__(parent)

        self.audio_path = audio_path
        self.api_token = api_token
        self.algorithm_id = algorithm_id
        self.option_1 = option_1
        self.option_2 = option_2
        self.option_3 = option_3
        self.output_dir = output_dir

    def run(self):
        try:
            self.progress.emit("Submitting MVSEP separation job...")

            job_hash, status_code = create_separation(
                path_to_file=self.audio_path,
                api_token=self.api_token,
                sep_type=self.algorithm_id,
                add_opt1=self.option_1,
                add_opt2=self.option_2,
                add_opt3=self.option_3,
            )

            if status_code != 200:
                raise RuntimeError(
                    f"MVSEP rejected the separation request "
                    f"(HTTP {status_code}): {job_hash!r}"
                )

            self.progress.emit(
                f"MVSEP job submitted successfully. Job hash: {job_hash}"
            )

            self.progress.emit(
                "Waiting for MVSEP to finish separation. "
                "This can take several minutes..."
            )

            result = poll_until_done(
                job_hash=job_hash,
                api_token=self.api_token,
                max_wait=900,
                poll_interval=5,
            )

            self.progress.emit("Downloading MVSEP stem files...")

            downloaded_files = get_result_files(
                data=result,
                save_path=self.output_dir,
            )

            if not downloaded_files:
                raise RuntimeError(
                    "MVSEP reported completion but returned no downloadable files."
                )

            self.completed.emit(downloaded_files)

        except Exception as exc:
            self.failed.emit(str(exc))


class MVSEPDialog(QDialog):
    """Model chooser and one-track MVSEP test dialog."""

    separation_finished = Signal(list)

    def __init__(self, config, audio_path="", parent=None):
        super().__init__(parent)

        self.config = config
        self.audio_path = audio_path
        self.algorithm_fields = {}
        self.worker = None

        self.setWindowTitle("MVSEP Separation")
        self.setMinimumWidth(650)

        self._build_ui()
        self._load_saved_values()
        self.refresh_algorithms()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.audio_path_edit = QLineEdit(self.audio_path)
        self.audio_path_edit.setReadOnly(True)
        form.addRow("Audio file:", self.audio_path_edit)

        audio_buttons = QHBoxLayout()

        self.choose_audio_button = QPushButton("Choose Audio File")
        self.choose_audio_button.clicked.connect(self.choose_audio_file)
        audio_buttons.addWidget(self.choose_audio_button)

        self.choose_output_button = QPushButton("Choose Output Folder")
        self.choose_output_button.clicked.connect(self.choose_output_folder)
        audio_buttons.addWidget(self.choose_output_button)

        layout.addLayout(audio_buttons)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        form.addRow("Output folder:", self.output_dir_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Your MVSEP API token")
        form.addRow("MVSEP API token:", self.api_key_edit)

        algorithm_layout = QHBoxLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.currentIndexChanged.connect(
            self.on_algorithm_changed
        )
        algorithm_layout.addWidget(self.algorithm_combo)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_algorithms)
        algorithm_layout.addWidget(self.refresh_button)

        form.addRow("Separation algorithm:", algorithm_layout)

        self.option_1_label = QLabel("Additional Option 1:")
        self.option_1_combo = QComboBox()
        form.addRow(self.option_1_label, self.option_1_combo)

        self.option_2_label = QLabel("Additional Option 2:")
        self.option_2_combo = QComboBox()
        form.addRow(self.option_2_label, self.option_2_combo)

        self.option_3_label = QLabel("Additional Option 3:")
        self.option_3_combo = QComboBox()
        form.addRow(self.option_3_label, self.option_3_combo)

        layout.addLayout(form)

        self.status_label = QLabel("Loading MVSEP algorithms...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save MVSEP Settings")
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)

        self.run_button = QPushButton("Test Separation")
        self.run_button.setStyleSheet(
            "font-weight: bold; background-color: #0e639c; padding: 8px;"
        )
        self.run_button.clicked.connect(self.start_separation)
        button_layout.addWidget(self.run_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _load_saved_values(self):
        self.api_key_edit.setText(
            str(self.config.get("mvsep_api_key", "") or "")
        )

        output_dir = self.config.get("mvsep_output_dir", "")

        if not output_dir and self.audio_path:
            output_dir = os.path.join(
                os.path.dirname(self.audio_path),
                "mvsep_output",
            )

        self.output_dir_edit.setText(str(output_dir or ""))

    def choose_audio_file(self):
        audio_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Audio File",
            os.path.dirname(self.audio_path) if self.audio_path else "",
            "Audio files (*.wav *.flac *.mp3 *.ogg *.m4a)",
        )

        if not audio_path:
            return

        self.audio_path = audio_path
        self.audio_path_edit.setText(audio_path)

        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(
                os.path.join(os.path.dirname(audio_path), "mvsep_output")
            )

    def choose_output_folder(self):
        start_dir = self.output_dir_edit.text().strip()

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose MVSEP Output Folder",
            start_dir,
        )

        if output_dir:
            self.output_dir_edit.setText(output_dir)

    def refresh_algorithms(self):
        self.status_label.setText("Loading live MVSEP algorithm list...")
        self.algorithm_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)

        try:
            algorithms, fields_by_id = get_algorithms()

            self.algorithm_fields = fields_by_id
            self.algorithm_combo.blockSignals(True)
            self.algorithm_combo.clear()

            for algorithm_id, algorithm_name in sorted(
                algorithms.items(),
                key=lambda item: item[1].lower(),
            ):
                self.algorithm_combo.addItem(
                    algorithm_name,
                    algorithm_id,
                )

            saved_algorithm_id = str(
                self.config.get("mvsep_algorithm_id", "") or ""
            )

            if saved_algorithm_id:
                saved_index = self.algorithm_combo.findData(
                    saved_algorithm_id
                )

                if saved_index >= 0:
                    self.algorithm_combo.setCurrentIndex(saved_index)

            self.algorithm_combo.blockSignals(False)
            self.on_algorithm_changed()

            self.status_label.setText(
                f"Loaded {self.algorithm_combo.count()} MVSEP algorithms."
            )

        except Exception as exc:
            self.status_label.setText(
                f"Could not load MVSEP algorithms: {exc}"
            )
            QMessageBox.warning(
                self,
                "MVSEP Connection Error",
                f"Could not load MVSEP algorithms.\n\n{exc}",
            )

        finally:
            self.algorithm_combo.setEnabled(True)
            self.refresh_button.setEnabled(True)

    def on_algorithm_changed(self):
        algorithm_id = self.algorithm_combo.currentData()

        fields = self.algorithm_fields.get(
            str(algorithm_id or ""),
            [],
        )

        self._set_option_combo(
            label=self.option_1_label,
            combo=self.option_1_combo,
            field=fields[0] if len(fields) > 0 else None,
            default_value=self.config.get("mvsep_add_opt1", 0),
        )

        self._set_option_combo(
            label=self.option_2_label,
            combo=self.option_2_combo,
            field=fields[1] if len(fields) > 1 else None,
            default_value=self.config.get("mvsep_add_opt2", 0),
        )

        self._set_option_combo(
            label=self.option_3_label,
            combo=self.option_3_combo,
            field=fields[2] if len(fields) > 2 else None,
            default_value=self.config.get("mvsep_add_opt3", 0),
        )

    def _set_option_combo(self, label, combo, field, default_value):
        combo.clear()

        if not field:
            label.setVisible(False)
            combo.setVisible(False)
            return

        field_text = str(field.get("text", "Additional Option"))
        label.setText(field_text)
        label.setVisible(True)
        combo.setVisible(True)

        raw_options = field.get("options", {})

        if isinstance(raw_options, str):
            try:
                raw_options = json.loads(raw_options)
            except json.JSONDecodeError:
                raw_options = {}

        if not isinstance(raw_options, dict):
            raw_options = {}

        for value, display_text in sorted(
            raw_options.items(),
            key=lambda item: str(item[1]).lower(),
        ):
            combo.addItem(str(display_text), str(value))

        default_index = combo.findData(str(default_value))

        if default_index >= 0:
            combo.setCurrentIndex(default_index)

        elif combo.count() > 0:
            combo.setCurrentIndex(0)

    def save_settings(self):
        api_key = self.api_key_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        algorithm_id = self.algorithm_combo.currentData()

        if not api_key:
            QMessageBox.warning(
                self,
                "MVSEP API Token Required",
                "Enter your MVSEP API token before saving.",
            )
            return

        if not algorithm_id:
            QMessageBox.warning(
                self,
                "MVSEP Algorithm Required",
                "Choose an MVSEP separation algorithm.",
            )
            return

        self.config["mvsep_api_key"] = api_key
        self.config["mvsep_output_dir"] = output_dir
        self.config["mvsep_algorithm_id"] = str(algorithm_id)
        self.config["mvsep_add_opt1"] = (
            self.option_1_combo.currentData() or "0"
        )
        self.config["mvsep_add_opt2"] = (
            self.option_2_combo.currentData() or "0"
        )
        self.config["mvsep_add_opt3"] = (
            self.option_3_combo.currentData() or "0"
        )

        QMessageBox.information(
            self,
            "MVSEP Settings Saved",
            "The selected MVSEP algorithm and options have been saved "
            "for this application session.",
        )

    def start_separation(self):
        self.save_settings()

        api_key = self.api_key_edit.text().strip()
        audio_path = self.audio_path_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        algorithm_id = self.algorithm_combo.currentData()

        if not api_key or not audio_path or not output_dir or not algorithm_id:
            return

        if not os.path.isfile(audio_path):
            QMessageBox.warning(
                self,
                "Audio File Missing",
                f"Cannot find:\n{audio_path}",
            )
            return

        os.makedirs(output_dir, exist_ok=True)

        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Preparing MVSEP separation...")

        self.worker = MVSEPSeparationWorker(
            audio_path=audio_path,
            api_token=api_key,
            algorithm_id=str(algorithm_id),
            option_1=str(self.option_1_combo.currentData() or "0"),
            option_2=str(self.option_2_combo.currentData() or "0"),
            option_3=str(self.option_3_combo.currentData() or "0"),
            output_dir=output_dir,
            parent=self,
        )

        self.worker.progress.connect(self.on_worker_progress)
        self.worker.completed.connect(self.on_worker_completed)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()

    def on_worker_progress(self, message):
        self.status_label.setText(message)

    def on_worker_completed(self, files):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)

        file_names = "\n".join(
            f"• {os.path.basename(path)}"
            for path in files
        )

        self.status_label.setText(
            f"MVSEP complete. Downloaded {len(files)} file(s)."
        )

        QMessageBox.information(
            self,
            "MVSEP Separation Complete",
            f"Downloaded {len(files)} file(s):\n\n{file_names}",
        )

        self.separation_finished.emit(files)

    def on_worker_failed(self, error_message):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.status_label.setText(f"MVSEP error: {error_message}")

        QMessageBox.critical(
            self,
            "MVSEP Separation Failed",
            error_message,
        )
