"""Main window: DatasetManager (QMainWindow)."""
import sys, os, json, uuid, tempfile, subprocess, time, math, struct, wave, shutil, zipfile, html, re
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
    QInputDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QCheckBox, QDialog, QFormLayout, QProgressBar, QScrollArea,
    QTabWidget, QFontComboBox, QSlider, QRadioButton, QButtonGroup,
    QFrame, QListWidget, QTextBrowser
)
from PySide6.QtGui import QFont, QColor, QDesktopServices
from stem_separator import StemSeparator
from config import DEFAULT_CONFIG, SETTINGS_PATH
from workers.deepseek import DeepSeekMusicOrchestrator
from workers.advanced import AdvancedDatasetOrchestratorWorker
from workers.spatial import SpatialPipelineWorker
from workers.health import HealthAuditorWorker
from workers.dsp import DspNormalizerWorker
from workers.caption import RemoteCaptionWorker, INSTRUMENT_ONLY_PROMPT
from workers.instrument_detect import InstrumentRecommendThread
from workers.structural import StructuralPipelineWorker, StructuralPipelineBatchWorker
from ui.mvsep_tab import MVSepTab
from workers.assistant import (
    AssistantWorker, APP_HELP_TEXT, ASSISTANT_TOOLS, build_system_prompt,
    summarize_dataset,
)

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
        self.config = dict(DEFAULT_CONFIG)
        self._load_settings()
        self.health_reports = {}
        self.original_backups = {}
        self.active_worker = None
        self.filter_exceptions_only = False
        self.bypass_warnings = False
        self.startup_scan_notice_shown = False
        self.kaggle_notebook_unlocked = False  # NEW
        self.recommended_instrument_models = []  # DeepSeek picks for instrument extraction
        self._section_captions = []              # per-section instrument captions

        self.init_ui()
        self.apply_custom_theme()

    def _load_settings(self):
        """Load settings (plain) + secrets (encrypted store)."""
        try:
            from modules.config_store import load_config
            self.config.update(load_config(DEFAULT_CONFIG))
        except Exception as e:  # noqa: BLE001
            print(f"Could not load settings: {e}")

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
        self.mvsep = MVSepTab(self.config, SETTINGS_PATH, parent_window=self)
        self.mvsep.stems_ready.connect(self.add_stem_files_to_dataset)
        self.tabs.addTab(self.mvsep, "🔊 MVSEP / Kaggle Separator")

        assistant_tab = QWidget()
        self.init_assistant_tab(assistant_tab)
        self.tabs.addTab(assistant_tab, "🤖 AI Assistant")

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

        gen_layout.addSpacing(12)
        gen_layout.addWidget(QLabel("Tags↔Captions:"))
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(int(self.config.get("tag_caption_ratio", 0)))
        self.ratio_slider.setFixedWidth(140)
        self.ratio_slider.setToolTip("Percentage of tracks using tag-style prompts vs prose captions. 0% = all captions, 100% = all tags.")
        self.ratio_slider.valueChanged.connect(self.on_ratio_changed)
        gen_layout.addWidget(self.ratio_slider)
        self.ratio_label = QLabel(f"{int(self.config.get('tag_caption_ratio', 0))}% tags")
        self.ratio_label.setStyleSheet("color: #aaa; font-size: 10px;")
        gen_layout.addWidget(self.ratio_label)

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

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Filename", "Health", "Tag", "Genre", "Duration", "Key", "BPM", "Caption"])
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)   
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 8):
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

        lang_row = QHBoxLayout()
        self.language_combo = QComboBox()
        self.language_combo.setEditable(True)
        self.language_combo.addItems(["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"])
        self.language_combo.currentTextChanged.connect(self.on_language_edited)
        self.language_lock = QCheckBox("Lock")
        lang_row.addWidget(self.language_combo, 1)
        lang_row.addWidget(self.language_lock)
        form.addRow("Language:", lang_row)

        ts_row = QHBoxLayout()
        self.timesig_combo = QComboBox()
        self.timesig_combo.setEditable(True)
        self.timesig_combo.addItems(["4/4", "3/4", "6/8", "5/4", "7/8", "12/8", "2/4", "9/8"])
        self.timesig_combo.currentTextChanged.connect(self.on_timesig_edited)
        self.timesig_lock = QCheckBox("Lock")
        ts_row.addWidget(self.timesig_combo, 1)
        ts_row.addWidget(self.timesig_lock)
        form.addRow("Time Signature:", ts_row)

        style_row = QHBoxLayout()
        self.prompt_style_combo = QComboBox()
        self.prompt_style_combo.addItems(["Use global ratio", "Caption only", "Tag only"])
        self.prompt_style_combo.currentTextChanged.connect(self.on_prompt_style_edited)
        style_row.addWidget(self.prompt_style_combo, 1)
        form.addRow("Prompt Override:", style_row)

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
    def init_assistant_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(16, 12, 16, 12)

        title = QLabel("🤖 AI Assistant (live help)")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        note = QLabel(
            "Ask how to use the app, or about the instruments in your dataset — "
            "powered by DeepSeek with the app's built-in help file."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaa;")
        layout.addWidget(note)

        self.assistant_history = QTextBrowser()
        self.assistant_history.setOpenExternalLinks(True)
        layout.addWidget(self.assistant_history, 1)

        input_row = QHBoxLayout()
        self.assistant_input = QLineEdit()
        self.assistant_input.setPlaceholderText(
            "e.g., How do I run the structural pipeline?  What instruments are in my dataset?"
        )
        self.assistant_input.returnPressed.connect(self.on_assistant_send)
        input_row.addWidget(self.assistant_input, 1)
        self.assistant_send_btn = QPushButton("Ask")
        self.assistant_send_btn.clicked.connect(self.on_assistant_send)
        input_row.addWidget(self.assistant_send_btn)
        layout.addLayout(input_row)

        self.assistant_status = QLabel("Ready.")
        self.assistant_status.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.assistant_status)

    def on_assistant_send(self):
        question = self.assistant_input.text().strip()
        if not question:
            return
        api_key = self.config.get("custom_key", "").strip()
        if not api_key:
            api_key = self.prompt_for_secret(
                "custom_key", "DeepSeek API Key",
                "Enter your DeepSeek API key for the AI assistant:",
            )
            if not api_key:
                return

        summary = summarize_dataset(self.dataset, self.health_reports)
        self.assistant_history.append(f"<b>You:</b> {html.escape(question)}")
        self.assistant_input.clear()
        self._set_assistant_busy(True)
        self.assistant_status.setText("Asking DeepSeek…")

        self._assistant_messages = [
            {"role": "system", "content": build_system_prompt(APP_HELP_TEXT, summary)},
            {"role": "user", "content": question},
        ]
        self._start_assistant()

    def _start_assistant(self):
        api_key = self.config.get("custom_key", "").strip()
        if not api_key:
            api_key = self.prompt_for_secret(
                "custom_key", "DeepSeek API Key",
                "Enter your DeepSeek API key for the AI assistant:",
            )
            if not api_key:
                self._set_assistant_busy(False)
                return
        self.assistant_worker = AssistantWorker(
            api_key, self._assistant_messages, tools=ASSISTANT_TOOLS, parent=self,
            config=self.config,
        )
        self.assistant_worker.answer_ready.connect(self.on_assistant_answer)
        self.assistant_worker.tool_requested.connect(self.on_assistant_tool)
        self.assistant_worker.failed.connect(self.on_assistant_error)
        self.assistant_worker.start()

    def _set_assistant_busy(self, busy):
        self.assistant_send_btn.setEnabled(not busy)
        self.assistant_status.setText("Thinking…" if busy else "Ready.")

    def on_assistant_tool(self, name, args_json, call_id):
        try:
            args = json.loads(args_json or "{}")
        except Exception:  # noqa: BLE001
            args = {}
        result = self.execute_assistant_tool(name, args)
        self.assistant_history.append(f"<i>⚙ tool: {html.escape(name)} → {html.escape(result[:200])}</i>")
        self._assistant_messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": args_json or "{}"}}],
        })
        self._assistant_messages.append({
            "role": "tool", "tool_call_id": call_id, "content": result,
        })
        self._start_assistant()

    def execute_assistant_tool(self, name, args):
        try:
            if name == "get_dataset_summary":
                return summarize_dataset(self.dataset, self.health_reports) or "(dataset empty)"
            if name == "list_tracks":
                lines = []
                for i, s in enumerate(self.dataset.get("samples", []), start=1):
                    cap = (s.get("caption") or "").strip().replace("\n", " ")[:80]
                    lines.append(f"{i}. {s.get('filename', '?')} — {cap or '(no caption)'}")
                return "\n".join(lines) or "(no tracks)"
            if name == "lookup_instruments":
                from modules.instruments_db import lookup_instruments
                found = lookup_instruments(args.get("filename", ""))
                return ", ".join(found) if found else "(no match)"
            if name == "audit_captions":
                from modules.caption_audit import audit_captions
                return "\n".join(audit_captions(self.dataset))
            if name == "validate_manifest":
                from modules.manifest_validation import validate_manifest
                issues = validate_manifest(self.dataset)
                return "\n".join(issues) if issues else "Manifest is valid."
            if name == "scan_health":
                self.start_health_audit()
                return "Started the health audit (Scan & Fill). Ask again after it finishes."
            if name == "detect_instruments":
                self.detect_instruments_for_separation()
                return "Started instrument detection on the selected track."
            return f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001
            return f"Tool error: {e}"

    def on_assistant_answer(self, answer):
        self._set_assistant_busy(False)
        self.assistant_history.append(f"<b>AI:</b><br>{html.escape(answer)}<hr>")

    def on_assistant_error(self, err):
        self._set_assistant_busy(False)
        self.assistant_status.setText(f"Error: {err}")
        QMessageBox.warning(self, "AI Assistant", str(err))

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
        inner.setContentsMargins(8, 22, 8, 8)

        info = QLabel(
            "This pipeline separates stems (import or MVSEP), finds structural "
            "boundaries, captions each section per stem, and aggregates via DeepSeek "
            "to produce a master caption for the whole track. No spatial L/R "
            "processing – suitable for general LoRA training."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; padding: 4px;")
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

        # Humanization preset
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Humanization Preset:"))
        self.humanize_preset_combo = QComboBox()
        self.humanize_preset_combo.setEditable(True)
        self.humanize_preset_combo.addItem("None")
        self.humanize_preset_combo.setInsertPolicy(QComboBox.NoInsert)
        for preset in self.config.get("humanize_presets", []):
            if preset and self.humanize_preset_combo.findText(preset) < 0:
                self.humanize_preset_combo.addItem(preset)
        self.humanize_preset_combo.setToolTip(
            "Free-form humanization preset — type any artist/style or pick a "
            "previously used one. Nothing is hardcoded."
        )
        preset_layout.addWidget(self.humanize_preset_combo)
        inner.addLayout(preset_layout)

        self.humanize_check = QCheckBox("Apply humanization")
        self.humanize_check.setChecked(True)
        inner.addWidget(self.humanize_check)

        # ---- Instrument extraction group ----
        sep_group = QGroupBox("Instrument‑Specific Stem Extraction")
        sep_layout2 = QVBoxLayout(sep_group)
        sep_layout2.setContentsMargins(8, 22, 8, 8)

        self.instrument_extraction_check = QCheckBox("Enable instrument‑specific extraction (recommended)")
        self.instrument_extraction_check.setChecked(True)
        sep_layout2.addWidget(self.instrument_extraction_check)

        disclaimer = QLabel(
            "⚠️ Disclaimer: The song‑specific recommendation may not be perfect. "
            "If instruments are not removed by the recommended options, you must "
            "experiment with other models that may or may not be on the list."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #ffcc80; font-size: 10px; padding: 4px;")
        sep_layout2.addWidget(disclaimer)

        detect_layout = QHBoxLayout()
        detect_layout.addWidget(QLabel("Detected instruments:"))
        self.detect_instruments_btn = QPushButton("Detect via Captioner")
        self.detect_instruments_btn.clicked.connect(self.detect_instruments_for_separation)
        detect_layout.addWidget(self.detect_instruments_btn)
        self.lookup_instruments_btn = QPushButton("🔍 Lookup")
        self.lookup_instruments_btn.setToolTip("Look up instruments from the local database (filename match) — no captioner needed.")
        self.lookup_instruments_btn.clicked.connect(self.on_instruments_lookup)
        detect_layout.addWidget(self.lookup_instruments_btn)
        self.audit_captions_btn = QPushButton("🧪 Audit Captions")
        self.audit_captions_btn.setToolTip("Check caption consistency (instruments + naming) across the dataset.")
        self.audit_captions_btn.clicked.connect(self.on_audit_captions)
        detect_layout.addWidget(self.audit_captions_btn)
        detect_layout.addStretch()
        sep_layout2.addLayout(detect_layout)

        self.section_cut_check = QCheckBox(
            "🔪 Cut at structural tags before instrument detection (recommended — "
            "short chunks make the captioner name instruments precisely)"
        )
        self.section_cut_check.setChecked(True)
        sep_layout2.addWidget(self.section_cut_check)

        self.detected_instruments_list = QTextEdit()
        self.detected_instruments_list.setPlaceholderText(
            "Instruments / MVSEP models — auto-filled by Detect/Lookup, or type your own."
        )
        self.detected_instruments_list.setMaximumHeight(80)
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
                key = self.prompt_for_secret("mvsep_api_key", "MVSEP API Key", "Enter your MVSEP API key:")
                if key:
                    self.config["mvsep_api_key"] = key
                    self.mvsep_key.setText(key)
                else:
                    return

        if self.use_deepseek_check.isChecked() and not self.config.get("custom_key"):
            key = self.prompt_for_secret("custom_key", "DeepSeek API Key", "Enter your DeepSeek API key:")
            if key:
                self.config["custom_key"] = key
                self.custom_key.setText(key)
            else:
                return

        if not self.config.get("kaggle_user"):
            user, ok1 = QInputDialog.getText(self, "Kaggle Username", "Enter Kaggle username:")
            if ok1 and user.strip():
                self.config["kaggle_user"] = user.strip()
                self.k_user.setText(user.strip())
                self._save_settings()
            else:
                return
        if not self.config.get("kaggle_key"):
            key = self.prompt_for_secret("kaggle_key", "Kaggle API Key", "Enter Kaggle API key:")
            if key:
                self.config["kaggle_key"] = key
                self.k_key.setText(key)
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
                key = self.prompt_for_secret("mvsep_api_key", "MVSEP API Key", "Enter your MVSEP API key:")
                if key:
                    self.config["mvsep_api_key"] = key
                    self.mvsep_key.setText(key)
                else:
                    return

        if self.struct_deepseek_check.isChecked() and not self.config.get("custom_key"):
            key = self.prompt_for_secret("custom_key", "DeepSeek API Key", "Enter your DeepSeek API key:")
            if key:
                self.config["custom_key"] = key
                self.custom_key.setText(key)
            else:
                return

        if not self.config.get("kaggle_user"):
            user, ok1 = QInputDialog.getText(self, "Kaggle Username", "Enter Kaggle username:")
            if ok1 and user.strip():
                self.config["kaggle_user"] = user.strip()
                self.k_user.setText(user.strip())
                self._save_settings()
            else:
                return
        if not self.config.get("kaggle_key"):
            key = self.prompt_for_secret("kaggle_key", "Kaggle API Key", "Enter Kaggle API key:")
            if key:
                self.config["kaggle_key"] = key
                self.k_key.setText(key)
            else:
                return

        # ---- Humanization preset (user-entered, free-form) ----
        humanize_preset = self.humanize_preset_combo.currentText().strip()
        if humanize_preset and self.humanize_preset_combo.findText(humanize_preset) < 0:
            self.humanize_preset_combo.addItem(humanize_preset)
        presets = list(self.config.get("humanize_presets", []))
        if humanize_preset and humanize_preset not in presets:
            presets.append(humanize_preset)
            self.config["humanize_presets"] = presets
            self._save_settings()

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
            "stem_source": "mvsep" if stem_source == "Separate via MVSEP" else "import",
            "use_deepseek": self.struct_deepseek_check.isChecked(),
            "use_lyrics": self.struct_seg_combo.currentText() == "Lyrics tags",
            "humanize": humanize,
            "humanize_preset": humanize_preset,
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
                key = self.prompt_for_secret("mvsep_api_key", "MVSEP API Key", "Enter your MVSEP API key:")
                if key:
                    self.config["mvsep_api_key"] = key
                    self.mvsep_key.setText(key)
                else:
                    return

        if self.struct_deepseek_check.isChecked() and not self.config.get("custom_key"):
            key = self.prompt_for_secret("custom_key", "DeepSeek API Key", "Enter your DeepSeek API key:")
            if key:
                self.config["custom_key"] = key
                self.custom_key.setText(key)
            else:
                return

        if not self.config.get("kaggle_user"):
            user, ok1 = QInputDialog.getText(self, "Kaggle Username", "Enter Kaggle username:")
            if ok1 and user.strip():
                self.config["kaggle_user"] = user.strip()
                self.k_user.setText(user.strip())
                self._save_settings()
            else:
                return
        if not self.config.get("kaggle_key"):
            key = self.prompt_for_secret("kaggle_key", "Kaggle API Key", "Enter Kaggle API key:")
            if key:
                self.config["kaggle_key"] = key
                self.k_key.setText(key)
            else:
                return

        # ---- Humanization preset (user-entered, free-form) ----
        humanize_preset = self.humanize_preset_combo.currentText().strip()
        if humanize_preset and self.humanize_preset_combo.findText(humanize_preset) < 0:
            self.humanize_preset_combo.addItem(humanize_preset)
        presets = list(self.config.get("humanize_presets", []))
        if humanize_preset and humanize_preset not in presets:
            presets.append(humanize_preset)
            self.config["humanize_presets"] = presets
            self._save_settings()

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
            models = [m.strip() for m in extra_models.split(',') if m.strip()] if extra_models else []
            # Prefer the DeepSeek-recommended models from 'Detect via Captioner'.
            recommended = getattr(self, "recommended_instrument_models", None) or []
            # Any instruments typed into the box also flow into the pipeline.
            box_text = self.detected_instruments_list.toPlainText().strip()
            box_models = [m.strip() for m in re.split(r"[\n,]+", box_text) if m.strip()] if box_text else []
            models = list(dict.fromkeys(recommended + box_models + models))
            if not models:
                QMessageBox.information(
                    self, "Instruments Not Detected",
                    "Instrument-specific extraction is enabled but no instruments are "
                    "listed yet. Run 'Detect via Captioner', '🔍 Lookup', or type "
                    "instruments/MVSEP models into the box. Continuing with defaults.",
                )
            if models:
                stem_options['instrument_models'] = models

        options = {
            "stem_source": "mvsep" if stem_source == "Separate via MVSEP" else "import",
            "use_deepseek": self.struct_deepseek_check.isChecked(),
            "use_lyrics": self.struct_seg_combo.currentText() == "Lyrics tags",
            "humanize": humanize,
            "humanize_preset": humanize_preset,
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
                break
        self.refresh_table()
        self.on_table_selection_changed()

    def on_struct_batch_done(self):
        self.run_struct_btn.setEnabled(True)
        self.struct_progress.setVisible(False)
        self.struct_status.setText("Structural pipeline completed for all tracks.")
        QMessageBox.information(self, "Pipeline Complete", "All selected tracks have been processed.")

    def detect_instruments_for_separation(self):
        """Detect instruments for the selected track.

        With "Cut at structural tags" enabled (default), the track is split at
        its structural boundaries and each section is captioned with the
        instruments-only prompt — short chunks make the captioner name the
        instruments precisely. The aggregated instrument list is then sent to
        DeepSeek, which recommends the instrument-specific MVSEP models to run.
        """
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "No Track Selected", "Please select a track first.")
            return

        self._section_captions = []

        if self.section_cut_check.isChecked():
            self._detect_instruments_by_sections(selected)
            return

        caption = selected.get("caption", "").strip()
        if caption:
            # Whole-track caption already exists — skip straight to DeepSeek.
            self._recommend_instrument_models(caption)
            return

        reply = QMessageBox.question(
            self,
            "Run Captioner First",
            "This track has no caption yet. Run the AI captioner now with an "
            "instruments-only prompt to detect the instruments?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_instrument_caption([selected])

    def _detect_instruments_by_sections(self, selected):
        """Cut the track at structural tags, caption each section, aggregate."""
        audio_path = selected.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.warning(self, "File Missing", f"Audio file not found:\n{audio_path}")
            return

        self.detect_instruments_btn.setEnabled(False)
        self.struct_status.setText("Cutting track at structural tags…")
        try:
            import tempfile
            from modules.audio_analysis import (
                find_structural_sections,
                slice_sections_to_wav,
            )
            temp_dir = tempfile.mkdtemp(prefix="ace_sections_")
            sections = find_structural_sections(audio_path)
            chunks = slice_sections_to_wav(audio_path, sections, temp_dir)
        except Exception as e:  # noqa: BLE001
            self.detect_instruments_btn.setEnabled(True)
            self.struct_status.setText(f"Section cutting failed: {e}")
            QMessageBox.warning(self, "Section Cut", str(e))
            return

        pseudo = []
        for i, c in enumerate(chunks, start=1):
            pseudo.append({
                "id": f"{selected.get('id', 'x')}_sec{i}",
                "audio_path": c["path"],
                "filename": os.path.basename(c["path"]),
            })
        self.struct_status.setText(
            f"Captioning {len(pseudo)} section(s) with the instruments-only prompt…"
        )
        self._run_instrument_caption(pseudo, aggregate=True)

    def _run_instrument_caption(self, samples, aggregate=False):
        backend = "Local Rule Engine"
        if self.config.get("kaggle_user") and self.config.get("kaggle_key"):
            backend = "Kaggle Cloud (Free GPU)"
        elif self.config.get("custom_key"):
            backend = "DeepSeek Cloud"

        worker = RemoteCaptionWorker(
            samples,
            backend,
            "Concise Tags",
            self.dataset.get("metadata", {}),
            self.config,
            caption_prompt=INSTRUMENT_ONLY_PROMPT,
        )
        worker.finished_sample.connect(self._on_instrument_caption_done)
        worker.all_done.connect(lambda: self._maybe_aggregate_sections(aggregate))
        worker.error_occurred.connect(self._on_instrument_caption_failed)
        self.detect_worker = worker
        worker.start()

    def _on_instrument_caption_done(self, sid, caption):
        self._section_captions.append(caption)

    def _maybe_aggregate_sections(self, aggregate):
        self.detect_instruments_btn.setEnabled(True)
        if not self._section_captions:
            self.struct_status.setText("No instruments found in the caption.")
            return
        if aggregate:
            combined = " | ".join(self._section_captions)
            self.struct_status.setText(
                "Sections captioned — asking DeepSeek for model recommendations…"
            )
            self._recommend_instrument_models(combined)
        else:
            self.struct_status.setText(
                "Instruments detected — asking DeepSeek for model recommendations…"
            )
            self._recommend_instrument_models(self._section_captions[-1])

    def _on_instrument_caption_failed(self, err):
        self.detect_instruments_btn.setEnabled(True)
        self.struct_status.setText(f"Instrument detection failed: {err}")
        QMessageBox.warning(self, "Instrument Detection", str(err))

    def _recommend_instrument_models(self, instruments_text):
        if not instruments_text.strip():
            self.struct_status.setText("No instruments found in the caption.")
            return
        self.struct_status.setText("DeepSeek is recommending instrument-specific MVSEP models…")
        self.detect_instruments_btn.setEnabled(False)
        self.rec_thread = InstrumentRecommendThread(self.config, instruments_text)
        self.rec_thread.finished_ok.connect(self._on_models_recommended)
        self.rec_thread.failed.connect(self._on_recommend_failed)
        self.rec_thread.start()

    def _on_models_recommended(self, recommended):
        self.detect_instruments_btn.setEnabled(True)
        self.recommended_instrument_models = recommended
        self.detected_instruments_list.setPlainText("\n".join(recommended))
        self.struct_status.setText(
            f"Recommended {len(recommended)} instrument-specific model(s). "
            "They will be used by the pipeline."
        )

    def _on_recommend_failed(self, err):
        self.detect_instruments_btn.setEnabled(True)
        self.struct_status.setText(f"Model recommendation failed: {err}")
        QMessageBox.warning(self, "Model Recommendation", str(err))

    def on_instruments_lookup(self):
        """Populate the instruments box from the local database (no captioner)."""
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "No Track Selected", "Please select a track first.")
            return
        from modules.instruments_db import lookup_instruments
        instruments = lookup_instruments(selected.get("filename", ""))
        if not instruments:
            self.struct_status.setText(
                "No database match for this track — type instruments/models manually."
            )
            QMessageBox.information(
                self, "Lookup",
                "No instrument entry found for this filename in the database. "
                "You can add one to instruments_db.json, or type instruments/models "
                "into the box manually.",
            )
            return
        self.recommended_instrument_models = instruments
        self.detected_instruments_list.setPlainText("\n".join(instruments))
        self.struct_status.setText(f"Looked up {len(instruments)} instrument(s) from the database.")

    def on_audit_captions(self):
        """Run the local caption-consistency audit and show the report."""
        from modules.caption_audit import audit_captions
        report = audit_captions(self.dataset)
        QMessageBox.information(
            self, "Caption Audit",
            "\n".join(report),
        )
        self.struct_status.setText(
            f"Caption audit: {sum(1 for r in report if 'NO CAPTION' in r)} track(s) missing captions."
        )

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
            api_key = self.prompt_for_secret("custom_key", "DeepSeek API Key", "Enter DeepSeek API key:")
            if api_key:
                self.config["custom_key"] = api_key
                self.custom_key.setText(api_key)
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
        self._save_settings()
        self.status_label.setText("Cloud credentials saved.")

    def add_stem_files_to_dataset(self, paths):
        """Add separated stem files (from the MVSEP tab) as new dataset samples."""
        added = 0
        for p in paths:
            if not p or not os.path.exists(p):
                continue
            self.dataset["samples"].append({
                "id": uuid.uuid4().hex[:8],
                "audio_path": p,
                "filename": os.path.basename(p),
                "caption": "",
                "genre": "",
                "lyrics": "",
                "formatted_lyrics": "",
                "bpm": 0,
                "keyscale": "",
                "timesignature": "4/4",
                "duration": 0,
                "language": "en",
                "is_instrumental": False,
                "custom_tag": "",
                "prompt_style": "use_global",
                "locked": True,
            })
            added += 1
        self.dataset["metadata"]["num_samples"] = len(self.dataset["samples"])
        if added:
            self.refresh_table()
            self.on_table_selection_changed()
            self.status_label.setText(f"Added {added} stem file(s) to the dataset.")
        else:
            self.status_label.setText("No stem files added (paths missing).")

    def _save_settings(self):
        """Persist settings; secrets go to the encrypted store, never settings.json."""
        try:
            from modules.config_store import save_config
            save_config(self.config)
        except OSError as e:
            self.status_label.setText(f"Could not save settings: {e}")

    def prompt_for_secret(self, key, title, label):
        """Return a secret, prompting the user when it isn't stored.

        Two modes: save securely to the encrypted store, or send it to the API
        from the popup and never persist it.
        """
        from modules.secrets_manager import get_secret, set_secret
        from ui.secret_prompt import SecretPromptDialog

        val = get_secret(key)
        if val:
            return val
        dlg = SecretPromptDialog(title, label, self)
        if dlg.exec() != SecretPromptDialog.Accepted:
            return ""
        val = dlg.value()
        if val:
            self.config[key] = val
            set_secret(key, val, persist=dlg.persist_choice())
        return val

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
            cap = (s.get("caption", "") or "").replace("\n", " ")[:60]
            self.table.setItem(row, 7, QTableWidgetItem(cap))

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
            self.language_combo.blockSignals(True)
            self.timesig_combo.blockSignals(True)
            self.prompt_style_combo.blockSignals(True)

            self.caption_text.setPlainText(s.get("caption", ""))
            self.lyrics_text.setPlainText(s.get("formatted_lyrics", s.get("lyrics", "")))
            self.track_tag_input.setText(s.get("custom_tag", ""))
            self.genre_input.setText(s.get("genre", ""))
            self.key_input.setText(str(s.get("keyscale", "")))
            self.bpm_spin.setValue(int(s.get("bpm", 0)))
            self.inst_check.setChecked(bool(s.get("is_instrumental", False)))
            self.language_combo.setCurrentText(s.get("language", "en") or "en")
            self.timesig_combo.setCurrentText(s.get("timesignature", "4/4") or "4/4")
            self.prompt_style_combo.setCurrentText({"use_global": "Use global ratio", "caption": "Caption only", "tag": "Tag only"}.get(s.get("prompt_style", "use_global"), "Use global ratio"))

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
            self.language_combo.blockSignals(False)
            self.timesig_combo.blockSignals(False)
            self.prompt_style_combo.blockSignals(False)

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
                    s["bpm"] = rep.get("bpm_detected", 0)
                    s["keyscale"] = rep.get("key_detected", "")
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

    def on_language_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["language"] = text.strip() or "en"

    def on_timesig_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["timesignature"] = text.strip() or "4/4"

    def on_prompt_style_edited(self, text):
        s = self.get_selected_sample()
        if s:
            s["prompt_style"] = {"Use global ratio": "use_global", "Caption only": "caption", "Tag only": "tag"}.get(text, "use_global")

    def on_ratio_changed(self, val):
        self.config["tag_caption_ratio"] = int(val)
        self.ratio_label.setText(f"{int(val)}% tags")
        self._save_settings()

    def resolve_prompt_style(self, sample):
        """Resolve a track's caption style: per-track override, else global ratio."""
        override = sample.get("prompt_style", "use_global")
        if override == "caption":
            return "caption"
        if override == "tag":
            return "tag"
        ratio = int(self.config.get("tag_caption_ratio", 0))
        idx = int(sample.get("id", "0") or "0", 16) if sample.get("id") else 0
        return "tag" if (idx % 100) < ratio else "caption"

    def _caption_complexity_resolver(self, sample):
        return "Concise Tags" if self.resolve_prompt_style(sample) == "tag" else "Deep Structural Breakdown"

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
        ratio = int(self.config.get("tag_caption_ratio", 0))
        self.ratio_slider.setValue(ratio)
        self.ratio_label.setText(f"{ratio}% tags")
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
                    s["bpm"] = rep.get("bpm_detected", 0)
                if not s.get("keyscale"):
                    s["keyscale"] = rep.get("key_detected", "")
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
                from modules.manifest_validation import validate_manifest
                issues = validate_manifest(self.dataset)
                if issues and not self.bypass_warnings:
                    QMessageBox.warning(
                        self, "Manifest Issues",
                        "The manifest has issues that may affect training:\n• "
                        + "\n• ".join(issues[:12]),
                    )
                    return
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
                    "prompt_style": "use_global",
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
            self._caption_complexity_resolver,
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


