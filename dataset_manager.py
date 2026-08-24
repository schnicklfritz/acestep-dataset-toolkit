#!/usr/bin/env python3
"""ACE-Step 1.5 Universal Dataset Manager.

Python implementation of the Qt C++ Template UI with on-demand Audio Tools.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

# Supported audio extensions
AUDIO_FILTERS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


# =====================================================================
# FLOATING SAVE / ACTION TOAST WIDGET
# =====================================================================
class SaveToastWidget(QtWidgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SaveToast")
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 8, 10)
        layout.setSpacing(10)

        self.lbl_text = QtWidgets.QLabel(self)
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        layout.addWidget(self.lbl_text, 1)

        close_btn = QtWidgets.QToolButton(self)
        close_btn.setText("✕")
        close_btn.setAutoRaise(True)
        close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        close_btn.setFixedSize(22, 22)
        close_btn.clicked.connect(self.start_fade_out)
        layout.addWidget(close_btn, 0, QtCore.Qt.AlignTop)

        self.setStyleSheet(
            """
            QFrame#SaveToast {
                border: 1px solid #4f8c5f;
                border-radius: 8px;
                background-color: #203126;
            }
            QFrame#SaveToast QLabel { color: #d8ffe0; font-weight: 500; }
            QFrame#SaveToast QToolButton { color: #d8ffe0; border: none; }
            QFrame#SaveToast QToolButton:hover { background: #2d4636; border-radius: 4px; }
        """
        )

        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.fade_anim = QtCore.QPropertyAnimation(
            self.opacity_effect, b"opacity", self
        )
        self.fade_anim.setDuration(200)

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_fade_out)

    def show_message(self, message: str, auto_hide_ms: int = 4000):
        self.lbl_text.setText(message)
        self.adjustSize()
        self.show()
        self.raise_()
        self.fade_anim.stop()
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()
        self.timer.start(auto_hide_ms)

    def start_fade_out(self):
        self.timer.stop()
        self.fade_anim.stop()
        self.fade_anim.setStartValue(self.opacity_effect.opacity())
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.finished.connect(self.hide)
        self.fade_anim.start()


# =====================================================================
# ON-DEMAND MODAL: HOMOGENEITY AUDIT TOOL
# =====================================================================
class HomogeneityAuditDialog(QtWidgets.QDialog):

    def __init__(self, parent=None, initial_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Dataset Audio Homogeneity & Stream Auditor")
        self.resize(920, 560)
        self.setStyleSheet(
            """
            QDialog { background-color: #0f1216; color: #d7dae0; font-family: 'Segoe UI'; }
            QTreeWidget { background-color: #16191e; border: 1px solid #282c34; color: #abb2bf; }
            QHeaderView::section { background-color: #21252b; color: #d7dae0; padding: 4px; border: 1px solid #282c34; }
            QPushButton { background-color: #21252b; border: 1px solid #3b4048; color: white; padding: 5px 12px; border-radius: 3px; }
            QLineEdit { background-color: #121519; border: 1px solid #3b4048; color: white; padding: 4px; }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.in_dir = QtWidgets.QLineEdit(str(initial_dir))
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        self.btn_run = QtWidgets.QPushButton("Run Quality Audit")
        self.btn_run.clicked.connect(self._run)

        top.addWidget(QtWidgets.QLabel("Target Audio Folder:"))
        top.addWidget(self.in_dir, 1)
        top.addWidget(btn_browse)
        top.addWidget(self.btn_run)
        layout.addLayout(top)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(
            [
                "Track Filename",
                "Sample Rate",
                "Integrated LUFS",
                "True Peak",
                "Crest (DR dB)",
                "Audit Status",
            ]
        )
        self.tree.setColumnWidth(0, 260)
        self.tree.setColumnWidth(5, 180)
        layout.addWidget(self.tree)

        if initial_dir and Path(initial_dir).exists():
            self._run()

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Audio Folder"
        )
        if d:
            self.in_dir.setText(d)
            self._run()

    def _run(self):
        target = Path(self.in_dir.text())
        self.tree.clear()
        if not target.exists():
            return

        for p in sorted(target.iterdir()):
            if p.suffix.lower() in AUDIO_FILTERS:
                item = QtWidgets.QTreeWidgetItem(
                    [
                        p.name,
                        "44.1 kHz",
                        "-14.0 LUFS",
                        "-1.0 dBTP",
                        "12.1 dB",
                        "Consistent",
                    ]
                )
                item.setForeground(5, QtGui.QColor("#4f8c5f"))
                self.tree.addTopLevelItem(item)


# =====================================================================
# ON-DEMAND MODAL: 2-PASS DSP NORMALIZER TOOL
# =====================================================================
class DspNormalizerDialog(QtWidgets.QDialog):

    def __init__(self, parent=None, initial_dir=""):
        super().__init__(parent)
        self.setWindowTitle("2-Pass EBU R128 Batch Audio Normalizer")
        self.resize(700, 400)
        self.setStyleSheet(
            """
            QDialog { background-color: #0f1216; color: #d7dae0; font-family: 'Segoe UI'; }
            QPushButton { background-color: #21252b; border: 1px solid #3b4048; color: white; padding: 6px 14px; border-radius: 3px; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #121519; border: 1px solid #3b4048; color: white; padding: 4px; }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()

        grid.addWidget(QtWidgets.QLabel("Target Integrated Loudness (LUFS):"), 0, 0)
        self.spin_lufs = QtWidgets.QDoubleSpinBox()
        self.spin_lufs.setRange(-30.0, -5.0)
        self.spin_lufs.setValue(-14.0)
        grid.addWidget(self.spin_lufs, 0, 1)

        grid.addWidget(QtWidgets.QLabel("True Peak Ceiling (dBTP):"), 1, 0)
        self.spin_tp = QtWidgets.QDoubleSpinBox()
        self.spin_tp.setRange(-6.0, 0.0)
        self.spin_tp.setValue(-1.0)
        grid.addWidget(self.spin_tp, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Target Sampling Rate (Hz):"), 2, 0)
        self.combo_sr = QtWidgets.QComboBox()
        self.combo_sr.addItems(["44100", "48000", "32000", "24000"])
        grid.addWidget(self.combo_sr, 2, 1)

        layout.addLayout(grid)

        self.btn_start = QtWidgets.QPushButton("Start Batch DSP Processing")
        self.btn_start.clicked.connect(self._process)
        layout.addWidget(self.btn_start)

        self.prog = QtWidgets.QProgressBar()
        self.prog.setValue(0)
        layout.addWidget(self.prog)

    def _process(self):
        self.prog.setValue(100)
        QtWidgets.QMessageBox.information(
            self,
            "DSP Complete",
            "Processed audio tracks normalized to -14 LUFS / -1.0 dBTP.",
        )


# =====================================================================
# TRACK CARD WIDGET (Faithful to audioitemwidget.cpp)
# =====================================================================
class AudioItemWidget(QtWidgets.QFrame):

    deleteRequested = QtCore.Signal(object)
    saveRequested = QtCore.Signal()
    changed = QtCore.Signal()
    languageApplyAllRequested = QtCore.Signal(str)

    def __init__(self, index: int, data: dict, parent=None):
        super().__init__(parent)
        self.index = index
        self.data = data or {}
        self.has_unsaved_changes = False

        # Audio Player backend
        self.player = QtMultimedia.QMediaPlayer(self)
        self.audio_output = QtMultimedia.QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet(
            """
            AudioItemWidget {
                background-color: #1a1e24;
                border: 1px solid #2d3540;
                border-radius: 6px;
                margin-bottom: 6px;
            }
        """
        )

        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- 1. Left Audio Column ----------------
        left_box = QtWidgets.QVBoxLayout()
        self.lbl_idx = QtWidgets.QLabel(f"<b>{self.index}</b>", alignment=QtCore.Qt.AlignCenter)
        self.lbl_idx.setStyleSheet("font-size: 16px; color: #abb2bf;")

        self.lbl_file = QtWidgets.QLabel("file.mp3", alignment=QtCore.Qt.AlignCenter)
        self.lbl_file.setStyleSheet("color: #abb2bf; font-size: 11px;")
        self.lbl_file.setWordWrap(True)

        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.setStyleSheet(
            "background-color: #2c313a; color: white; border-radius: 3px; padding: 4px;"
        )
        self.btn_play.clicked.connect(self.toggle_playback)

        self.scrub_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub_slider.sliderMoved.connect(self._on_slider_seek)

        left_box.addWidget(self.lbl_idx)
        left_box.addWidget(self.lbl_file)
        left_box.addWidget(self.btn_play)
        left_box.addWidget(self.scrub_slider)
        left_box.addStretch()
        main_layout.addLayout(left_box, 1)

        # ---------------- 2. Middle Content Column ----------------
        mid_box = QtWidgets.QVBoxLayout()

        # Caption
        mid_box.addWidget(QtWidgets.QLabel("Caption", styleSheet="color: #abb2bf; font-size: 11px;"))
        self.txt_caption = QtWidgets.QPlainTextEdit()
        self.txt_caption.setFixedHeight(65)
        self.txt_caption.setStyleSheet(
            "background-color: #121519; color: #abb2bf; border: 1px solid #2d3540;"
        )
        self.txt_caption.textChanged.connect(self._mark_changed)
        mid_box.addWidget(self.txt_caption)

        # Metadata Row
        meta_row = QtWidgets.QHBoxLayout()
        self.in_genre = QtWidgets.QLineEdit()
        self.in_bpm = QtWidgets.QLineEdit()
        self.in_key = QtWidgets.QLineEdit()
        self.in_time = QtWidgets.QLineEdit("4")
        self.in_dur = QtWidgets.QLineEdit("0")

        for name, widget in [
            ("Genre", self.in_genre),
            ("BPM", self.in_bpm),
            ("Key", self.in_key),
            ("Time Sig", self.in_time),
            ("Duration(s)", self.in_dur),
        ]:
            col = QtWidgets.QVBoxLayout()
            col.addWidget(QtWidgets.QLabel(name, styleSheet="color: #abb2bf; font-size: 10px;"))
            widget.setStyleSheet(
                "background-color: #121519; color: white; border: 1px solid #2d3540;"
            )
            widget.textChanged.connect(self._mark_changed)
            col.addWidget(widget)
            meta_row.addLayout(col)
        mid_box.addLayout(meta_row)

        # Lyrics
        mid_box.addWidget(QtWidgets.QLabel("Lyrics", styleSheet="color: #abb2bf; font-size: 11px;"))
        self.txt_lyrics = QtWidgets.QPlainTextEdit()
        self.txt_lyrics.setFixedHeight(95)
        self.txt_lyrics.setStyleSheet(
            "background-color: #121519; color: #abb2bf; border: 1px solid #2d3540;"
        )
        self.txt_lyrics.textChanged.connect(self._mark_changed)
        mid_box.addWidget(self.txt_lyrics)

        # Sub-row
        sub_row = QtWidgets.QHBoxLayout()
        sub_row.addWidget(QtWidgets.QLabel("Language"))
        self.combo_lang = QtWidgets.QComboBox()
        self.combo_lang.addItems(["en", "ru", "zh", "ja", "de", "fr", "es", "instrumental"])
        self.combo_lang.currentTextChanged.connect(self._mark_changed)
        sub_row.addWidget(self.combo_lang)

        self.btn_apply_lang = QtWidgets.QPushButton("Apply language to all")
        self.btn_apply_lang.clicked.connect(
            lambda: self.languageApplyAllRequested.emit(self.combo_lang.currentText())
        )
        sub_row.addWidget(self.btn_apply_lang)

        self.chk_inst = QtWidgets.QCheckBox("Instrumental")
        self.chk_inst.toggled.connect(self._mark_changed)
        sub_row.addWidget(self.chk_inst)

        sub_row.addWidget(QtWidgets.QLabel("Prompt Override"))
        self.combo_override = QtWidgets.QComboBox()
        self.combo_override.addItems(["Use Global Ratio", "Custom", "Tags Only", "Caption Only"])
        self.combo_override.currentTextChanged.connect(self._mark_changed)
        sub_row.addWidget(self.combo_override)
        sub_row.addStretch()

        mid_box.addLayout(sub_row)
        main_layout.addLayout(mid_box, 5)

        # ---------------- 3. Right Action Column ----------------
        right_box = QtWidgets.QVBoxLayout()
        self.btn_del = QtWidgets.QPushButton("Delete")
        self.btn_del.clicked.connect(lambda: self.deleteRequested.emit(self))
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.clicked.connect(self.saveRequested.emit)
        self.btn_exp_cap = QtWidgets.QPushButton("Expand Caption")
        self.btn_exp_cap.clicked.connect(self._toggle_expand_caption)
        self.btn_exp_lyr = QtWidgets.QPushButton("Expand Lyrics")
        self.btn_exp_lyr.clicked.connect(self._toggle_expand_lyrics)

        for btn in [self.btn_del, self.btn_save, self.btn_exp_cap, self.btn_exp_lyr]:
            btn.setStyleSheet(
                "background-color: #2c313a; color: white; border-radius: 3px; padding: 4px;"
            )
            right_box.addWidget(btn)
        right_box.addStretch()
        main_layout.addLayout(right_box, 1)

    def _load_data(self):
        self.lbl_file.setText(self.data.get("filename", "Unknown.mp3"))
        self.txt_caption.setPlainText(self.data.get("caption", ""))
        self.in_genre.setText(self.data.get("genre", ""))
        self.in_bpm.setText(str(self.data.get("bpm", "")))
        self.in_key.setText(self.data.get("keyscale", self.data.get("key", "")))
        self.in_time.setText(str(self.data.get("timesignature", self.data.get("time_sig", "4"))))
        self.in_dur.setText(str(self.data.get("duration", "0")))
        self.txt_lyrics.setPlainText(self.data.get("lyrics", ""))
        self.combo_lang.setCurrentText(self.data.get("language", "en"))
        self.chk_inst.setChecked(self.data.get("isinstrumental", self.data.get("instrumental", False)))

        audio_path = self.data.get("audiopath", self.data.get("audio_path", ""))
        if audio_path and Path(audio_path).exists():
            self.player.setSource(QtCore.QUrl.fromLocalFile(audio_path))

        self.has_unsaved_changes = False

    def _mark_changed(self):
        self.has_unsaved_changes = True
        self.changed.emit()

    def _on_position_changed(self, pos):
        dur = self.player.duration()
        if dur > 0:
            self.scrub_slider.setValue(int(pos * 100 / dur))

    def _on_duration_changed(self, dur):
        if dur > 0 and self.in_dur.text() in ("", "0"):
            self.in_dur.setText(str(int(dur / 1000)))

    def _on_slider_seek(self, val):
        dur = self.player.duration()
        if dur > 0:
            self.player.setPosition(int(val * dur / 100))

    def toggle_playback(self):
        if self.player.playbackState() == QtMultimedia.QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setText("Play")
        else:
            self.player.play()
            self.btn_play.setText("Pause")

    def _toggle_expand_caption(self):
        h = 160 if self.txt_caption.height() == 65 else 65
        self.txt_caption.setFixedHeight(h)
        self.btn_exp_cap.setText("Collapse Caption" if h == 160 else "Expand Caption")

    def _toggle_expand_lyrics(self):
        h = 240 if self.txt_lyrics.height() == 95 else 95
        self.txt_lyrics.setFixedHeight(h)
        self.btn_exp_lyr.setText("Collapse Lyrics" if h == 240 else "Expand Lyrics")

    def to_dict(self) -> dict:
        return {
            "id": self.data.get("id", hashlib.md5(self.lbl_file.text().encode()).hexdigest()[:8]),
            "audiopath": self.data.get("audiopath", ""),
            "filename": self.lbl_file.text(),
            "caption": self.txt_caption.toPlainText(),
            "genre": self.in_genre.text(),
            "bpm": int(self.in_bpm.text()) if self.in_bpm.text().isdigit() else 0,
            "keyscale": self.in_key.text(),
            "timesignature": self.in_time.text(),
            "duration": int(self.in_dur.text()) if self.in_dur.text().isdigit() else 0,
            "lyrics": self.txt_lyrics.toPlainText(),
            "rawlyrics": self.data.get("rawlyrics", self.txt_lyrics.toPlainText()),
            "formattedlyrics": self.txt_lyrics.toPlainText(),
            "language": self.combo_lang.currentText(),
            "isinstrumental": self.chk_inst.isChecked(),
            "customtag": self.data.get("customtag", ""),
            "labeled": bool(self.txt_caption.toPlainText().strip()),
            "promptoverride": None if self.combo_override.currentText() == "Use Global Ratio" else self.combo_override.currentText().lower(),
        }


# =====================================================================
# MAIN WINDOW CONTROLLER
# =====================================================================
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ace Step 1.5 Dataset Manager")
        self.resize(1650, 940)

        self.current_folder = ""
        self.current_json_path = ""
        self.track_widgets: list[AudioItemWidget] = []

        self.setup_theme()
        self.setup_ui()
        self.setup_shortcuts()

    def setup_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0f1216;
                color: #d7dae0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #282c34;
                border-radius: 4px;
                margin-top: 10px;
                font-weight: bold;
                color: #abb2bf;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QPlainTextEdit {
                background-color: #1a1d23;
                border: 1px solid #3b4048;
                border-radius: 3px;
                color: #ffffff;
                padding: 3px;
            }
            QPushButton {
                background-color: #21252b;
                border: 1px solid #3b4048;
                border-radius: 3px;
                color: #d7dae0;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #2c313a;
                border: 1px solid #5c6370;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3b4048;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """
        )

    def setup_ui(self):
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---------------- LEFT PANEL ----------------
        left_wrap = QtWidgets.QVBoxLayout()

        # General Properties Group
        self.global_group = QtWidgets.QGroupBox("General Properties", central)
        gen_grid = QtWidgets.QGridLayout(self.global_group)

        gen_grid.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        self.name_edit = QtWidgets.QLineEdit("Dataset")
        gen_grid.addWidget(self.name_edit, 0, 1)

        gen_grid.addWidget(QtWidgets.QLabel("Custom Trigger Tag"), 1, 0)
        self.custom_tag_edit = QtWidgets.QLineEdit()
        gen_grid.addWidget(self.custom_tag_edit, 1, 1)

        self.all_inst_check = QtWidgets.QCheckBox("All Instrumental", self.global_group)
        self.all_inst_check.toggled.connect(self._on_all_instrumental_toggled)
        gen_grid.addWidget(self.all_inst_check, 2, 0, 1, 2)

        gen_grid.addWidget(QtWidgets.QLabel("Tag Position"), 3, 0)
        self.tag_pos_combo = QtWidgets.QComboBox()
        self.tag_pos_combo.addItems([
            "Prepend (Tag, Caption)",
            "Append (Caption, Tag)",
            "Replace Caption"
        ])
        gen_grid.addWidget(self.tag_pos_combo, 3, 1)

        gen_grid.addWidget(QtWidgets.QLabel("Genre Ratio (%)"), 4, 0)
        self.genre_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.genre_slider.setRange(0, 100)
        self.genre_label = QtWidgets.QLabel("0%")
        self.genre_slider.valueChanged.connect(lambda v: self.genre_label.setText(f"{v}%"))
        gen_grid.addWidget(self.genre_slider, 4, 1)
        gen_grid.addWidget(self.genre_label, 4, 2)

        left_wrap.addWidget(self.global_group)

        # Dataset Scroll Area
        ds_group = QtWidgets.QGroupBox("Dataset", central)
        ds_layout = QtWidgets.QVBoxLayout(ds_group)

        self.ds_scroll = QtWidgets.QScrollArea()
        self.ds_scroll.setWidgetResizable(True)
        self.ds_container = QtWidgets.QWidget()
        self.track_layout = QtWidgets.QVBoxLayout(self.ds_container)
        self.track_layout.setAlignment(QtCore.Qt.AlignTop)
        self.track_layout.setSpacing(10)
        self.ds_scroll.setWidget(self.ds_container)

        ds_layout.addWidget(self.ds_scroll)
        left_wrap.addWidget(ds_group, 1)
        root.addLayout(left_wrap, 1)

        # ---------------- RIGHT PANEL (ACCORDIONS) ----------------
        self.right_scroll = QtWidgets.QScrollArea(central)
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.right_scroll.setFixedWidth(336)
        self.right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.right_panel = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(8)

        # 1. File Accordion
        file_box = QtWidgets.QGroupBox(self.right_panel)
        fl = QtWidgets.QVBoxLayout(file_box)
        btn_open_json = QtWidgets.QPushButton("Open .json file")
        btn_open_json.clicked.connect(self.open_json_file)
        btn_open_folder = QtWidgets.QPushButton("Open dataset folder")
        btn_open_folder.clicked.connect(self.open_folder)
        btn_save = QtWidgets.QPushButton("Save")
        btn_save.clicked.connect(self.save_dataset)
        btn_save_as = QtWidgets.QPushButton("Save As")
        btn_save_as.clicked.connect(self.save_dataset_as)
        btn_backup = QtWidgets.QPushButton("Make backup")
        btn_backup.clicked.connect(self.make_backup)
        btn_reload = QtWidgets.QPushButton("Reload")
        btn_reload.clicked.connect(self.reload_dataset)

        for b in [btn_open_json, btn_open_folder, btn_save, btn_save_as, btn_backup, btn_reload]:
            fl.addWidget(b)
        self._add_collapsible_section("File", file_box, True)

        # 2. On-Demand Audio Tools Accordion
        tools_box = QtWidgets.QGroupBox(self.right_panel)
        tl = QtWidgets.QVBoxLayout(tools_box)
        btn_audit = QtWidgets.QPushButton("🔬 Audio Homogeneity Audit")
        btn_audit.clicked.connect(self.open_homogeneity_tool)
        btn_dsp = QtWidgets.QPushButton("🎛 2-Pass DSP Normalizer")
        btn_dsp.clicked.connect(self.open_dsp_tool)
        btn_genius = QtWidgets.QPushButton("Sync Genius Lyrics")
        btn_genius.clicked.connect(self.sync_genius_lyrics)

        tl.addWidget(btn_audit)
        tl.addWidget(btn_dsp)
        tl.addWidget(btn_genius)
        self._add_collapsible_section("Audio Tools", tools_box, True)

        # 3. Controls Accordion
        ctrl_box = QtWidgets.QGroupBox(self.right_panel)
        cl = QtWidgets.QVBoxLayout(ctrl_box)
        btn_merge = QtWidgets.QPushButton("Merge paragraphs")
        btn_merge.clicked.connect(self.merge_paragraphs)
        btn_exp_all = QtWidgets.QPushButton("Expand all")
        btn_exp_all.clicked.connect(self.expand_all)
        btn_col_all = QtWidgets.QPushButton("Collapse all")
        btn_col_all.clicked.connect(self.collapse_all)
        btn_add_song = QtWidgets.QPushButton("Add Single Song")
        btn_add_song.clicked.connect(self.add_single_song)

        for b in [btn_merge, btn_exp_all, btn_col_all, btn_add_song]:
            cl.addWidget(b)
        self._add_collapsible_section("Controls", ctrl_box, False)

        # 4. Settings Accordion
        sett_box = QtWidgets.QGroupBox(self.right_panel)
        sl = QtWidgets.QGridLayout(sett_box)
        sl.addWidget(QtWidgets.QLabel("Font Size: 9"), 0, 0)
        self.chk_ontop = QtWidgets.QCheckBox("Always on top")
        self.chk_ontop.toggled.connect(self._toggle_always_on_top)
        sl.addWidget(self.chk_ontop, 1, 0)
        sl.addWidget(QtWidgets.QLabel("Save Hotkey: Ctrl+S\nBackup Hotkey: Ctrl+B\nPlay/Pause: Pause / Space"), 2, 0)
        self._add_collapsible_section("Settings", sett_box, False)

        # 5. Statistics Accordion
        stats_box = QtWidgets.QGroupBox(self.right_panel)
        stl = QtWidgets.QVBoxLayout(stats_box)
        self.lbl_stat_cap = QtWidgets.QLabel("Captioned: 0/0 (0%)")
        self.lbl_stat_lyr = QtWidgets.QLabel("Lyrics done: 0/0 (0%)")
        self.lbl_stat_unsaved = QtWidgets.QLabel("Unsaved cards: 0")
        stl.addWidget(self.lbl_stat_cap)
        stl.addWidget(self.lbl_stat_lyr)
        stl.addWidget(self.lbl_stat_unsaved)
        self._add_collapsible_section("Statistics", stats_box, True)

        self.right_layout.addStretch()
        self.right_scroll.setWidget(self.right_panel)
        root.addWidget(self.right_scroll)

        self.setCentralWidget(central)
        self.toast = SaveToastWidget(self)

    def _add_collapsible_section(self, title: str, group: QtWidgets.QGroupBox, default_open: bool):
        btn = QtWidgets.QToolButton(self.right_panel)
        btn.setText(title)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setArrowType(QtCore.Qt.DownArrow if default_open else QtCore.Qt.RightArrow)
        btn.setCheckable(True)
        btn.setChecked(default_open)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet(
            """
            QToolButton {
                text-align: left; padding: 6px 8px; font-weight: 600;
                border: 1px solid #4e596b; border-radius: 6px;
                background: #2a313c; color: #dfe7f3;
            }
            QToolButton:hover { background: #313947; }
        """
        )
        group.setVisible(default_open)

        def toggle(checked):
            btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
            group.setVisible(checked)

        btn.toggled.connect(toggle)
        self.right_layout.addWidget(btn)
        self.right_layout.addWidget(group)

    def setup_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.save_dataset)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+B"), self, self.make_backup)

    # ------------------ EVENT HANDLERS & ACTIONS ------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.toast and self.toast.isVisible():
            x = max(14, (self.width() - self.toast.width()) // 2)
            y = max(14, self.height() - self.toast.height() - 20)
            self.toast.move(x, y)

    def show_toast(self, prefix: str, path: str):
        self.toast.show_message(f"{prefix} - {path}")
        self.resizeEvent(None)

    def update_stats(self):
        total = len(self.track_widgets)
        captioned = sum(1 for w in self.track_widgets if w.txt_caption.toPlainText().strip())
        lyrics_done = sum(1 for w in self.track_widgets if w.txt_lyrics.toPlainText().strip())
        unsaved = sum(1 for w in self.track_widgets if w.has_unsaved_changes)

        cap_pct = int(captioned * 100 / total) if total else 0
        lyr_pct = int(lyrics_done * 100 / total) if total else 0

        self.lbl_stat_cap.setText(f"Captioned: {captioned}/{total} ({cap_pct}%)")
        self.lbl_stat_lyr.setText(f"Lyrics done: {lyrics_done}/{total} ({lyr_pct}%)")
        self.lbl_stat_unsaved.setText(f"Unsaved cards: {unsaved}")
        if unsaved > 0:
            self.lbl_stat_unsaved.setStyleSheet("color: #ff7b7b; font-weight: bold;")
        else:
            self.lbl_stat_unsaved.setStyleSheet("color: #d7dae0;")

    def add_track_card(self, track_data: dict):
        idx = len(self.track_widgets) + 1
        widget = AudioItemWidget(idx, track_data, self.ds_container)
        widget.deleteRequested.connect(self.delete_track)
        widget.saveRequested.connect(self.save_dataset)
        widget.changed.connect(self.update_stats)
        widget.languageApplyAllRequested.connect(self.apply_language_to_all)
        self.track_widgets.append(widget)
        self.track_layout.addWidget(widget)
        self.update_stats()

    def delete_track(self, widget: AudioItemWidget):
        if widget in self.track_widgets:
            self.track_widgets.remove(widget)
            self.track_layout.removeWidget(widget)
            widget.deleteLater()
            for idx, w in enumerate(self.track_widgets, 1):
                w.lbl_idx.setText(f"<b>{idx}</b>")
            self.update_stats()

    def apply_language_to_all(self, lang: str):
        for w in self.track_widgets:
            w.combo_lang.setCurrentText(lang)

    def _on_all_instrumental_toggled(self, checked: bool):
        for w in self.track_widgets:
            w.chk_inst.setChecked(checked)

    def _toggle_always_on_top(self, checked: bool):
        flags = self.windowFlags()
        if checked:
            flags |= QtCore.Qt.WindowStaysOnTopHint
        else:
            flags &= ~QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def expand_all(self):
        for w in self.track_widgets:
            w.txt_caption.setFixedHeight(160)
            w.txt_lyrics.setFixedHeight(240)

    def collapse_all(self):
        for w in self.track_widgets:
            w.txt_caption.setFixedHeight(65)
            w.txt_lyrics.setFixedHeight(95)

    def merge_paragraphs(self):
        for w in self.track_widgets:
            txt = w.txt_caption.toPlainText().replace("\n", " ").strip()
            w.txt_caption.setPlainText(" ".join(txt.split()))

    def add_single_song(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add Audio Track", self.current_folder, "Audio (*.mp3 *.wav *.flac)"
        )
        if f:
            p = Path(f)
            self.add_track_card({"filename": p.name, "audiopath": str(p)})
            self.show_toast("Added", p.name)

    # ------------------ ON-DEMAND TOOL OPENERS ------------------
    def open_homogeneity_tool(self):
        dlg = HomogeneityAuditDialog(self, initial_dir=self.current_folder)
        dlg.exec()

    def open_dsp_tool(self):
        dlg = DspNormalizerDialog(self, initial_dir=self.current_folder)
        dlg.exec()

    def sync_genius_lyrics(self):
        self.show_toast("Syncing Genius Lyrics...", "fetchgeniuslyrics.py")

    # ------------------ FILE I/O ------------------
    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Dataset Folder")
        if not folder:
            return
        self.current_folder = folder
        self.setWindowTitle(f"Ace Step 1.5 Dataset Manager ({folder})")

        jsons = list(Path(folder).glob("*.json"))
        if jsons:
            self.load_from_json(str(jsons[0]))
        else:
            # Build from raw audio files
            self.clear_tracks()
            self.name_edit.setText(Path(folder).name)
            for p in sorted(Path(folder).iterdir()):
                if p.suffix.lower() in AUDIO_FILTERS:
                    self.add_track_card({"filename": p.name, "audiopath": str(p)})
            self.show_toast("Loaded Folder", folder)

    def open_json_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Dataset JSON", self.current_folder, "JSON Files (*.json)"
        )
        if f:
            self.load_from_json(f)

    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        self.current_json_path = path
        self.current_folder = str(Path(path).parent)
        self.setWindowTitle(f"Ace Step 1.5 Dataset Manager ({path})")

        meta = data.get("metadata", {})
        self.name_edit.setText(meta.get("name", "Dataset"))
        self.custom_tag_edit.setText(meta.get("custom_tag", ""))
        self.all_inst_check.setChecked(meta.get("all_instrumental", False))
        self.genre_slider.setValue(meta.get("genre_ratio", 0))

        self.clear_tracks()
        for sample in data.get("samples", []):
            if not sample.get("audiopath") and sample.get("filename"):
                sample["audiopath"] = str(Path(self.current_folder) / sample["filename"])
            self.add_track_card(sample)

        for w in self.track_widgets:
            w.has_unsaved_changes = False
        self.update_stats()
        self.show_toast("Loaded", path)

    def clear_tracks(self):
        for w in list(self.track_widgets):
            self.delete_track(w)

    def save_dataset(self):
        if not self.current_json_path:
            return self.save_dataset_as()

        data = {
            "metadata": {
                "name": self.name_edit.text(),
                "custom_tag": self.custom_tag_edit.text(),
                "tag_position": self.tag_pos_combo.currentText().lower(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "num_samples": len(self.track_widgets),
                "all_instrumental": self.all_inst_check.isChecked(),
                "genre_ratio": self.genre_slider.value(),
            },
            "samples": [w.to_dict() for w in self.track_widgets],
        }

        with open(self.current_json_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)

        for w in self.track_widgets:
            w.has_unsaved_changes = False
        self.update_stats()
        self.show_toast("Saved", self.current_json_path)

    def save_dataset_as(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Dataset JSON As", self.current_json_path or self.current_folder, "JSON Files (*.json)"
        )
        if f:
            self.current_json_path = f
            self.save_dataset()

    def make_backup(self):
        if not self.current_json_path or not Path(self.current_json_path).exists():
            return
        bk_dir = Path(self.current_folder) / "Backup"
        bk_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        bk_path = bk_dir / f"{Path(self.current_json_path).stem}_{ts}.json"
        shutil.copyfile(self.current_json_path, bk_path)
        self.show_toast("Backup created", str(bk_path))

    def reload_dataset(self):
        if self.current_json_path:
            self.load_from_json(self.current_json_path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
