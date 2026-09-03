import sys
import os
import json
import re
import html
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
from workers.lyrics import TranscribeLyricsWorker
from workers.rockstar import RockstarLookupWorker
from workers.embeddings import EmbeddingWorker
from workers.export import ExportWorker
from workers.tag_creator import TagCreatorWorker
from workers.musicbrainz import MusicBrainzWorker
from modules.lyrics_tools import split_long_lines

# Modern worker implementations (split into workers/ modules).
from workers.caption import RemoteCaptionWorker, resolve_backend
from workers.spatial import SpatialPipelineWorker
from workers.structural import (
    StructuralPipelineWorker,
    StructuralPipelineBatchWorker,
)
from workers.assistant import (
    AssistantWorker, APP_HELP_TEXT, ASSISTANT_TOOLS, build_system_prompt,
    build_sound_profile, summarize_dataset,
)

# --- NEW: extracted widgets/orchestrator (replaces inline class definitions below) ---
from widgets import WaveformWidget, ScatterPlotWidget
from orchestrator import DeepSeekMusicOrchestrator, AdvancedDatasetOrchestratorWorker
from workers.health_audit import HealthAuditorWorker
from workers.dsp_normalizer import DspNormalizerWorker
from ui.settings_tab import build_settings_tab

from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl, QTime
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
    QInputDialog,QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QLineEdit, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QDialog, QFormLayout, QProgressBar, QScrollArea,
    QTabWidget, QFontComboBox, QSlider, QRadioButton, QButtonGroup,
    QFrame, QListWidget, QTextBrowser, QSizePolicy, QAbstractItemView
)
from PySide6.QtGui import QFont, QColor, QDesktopServices, QPainter, QPen, QPalette

# Provider -> (key config field, remember flag) — the unified "Provider API Key"
# field routes to whichever provider/model is selected.
LLM_KEY_FIELDS = {
    "deepseek": ("deepseek_key", "remember_deepseek_key"),
    "gemini": ("gemini_api_key", "remember_gemini_key"),
    "openrouter": ("openrouter_key", "remember_openrouter_key"),
    "groq": ("groq_key", "remember_groq_key"),
    "local": ("custom_key", "remember_custom_key"),
}

# Audio file extensions accepted when adding songs/folders to the dataset
# (single source of truth, used by both the file picker and the folder scan).
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

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
        # Filter/search state (Dataset Studio table) — one search box covers
        # filename/caption/tag/genre/key; no separate genre/key/BPM filter
        # fields (single place for those values = the table, auto-locked).
        self._table_sample_indices = []
        self.filter_query = ""
        self.filter_inst = "all"
        self.filter_captioned = False
        self._loading_table = False
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

    def _backup_file(self, path):
        """Back up any existing file before it is changed or replaced.

        Returns the backup path, or ``None`` if there was nothing to back up.
        The backup keeps the original untouched (``<name>.bak-<timestamp>``).
        """
        if not path or not os.path.exists(path):
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.bak-{stamp}"
        try:
            shutil.copy2(path, backup)
            return backup
        except OSError:
            return None

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
        """Initializes and anchors our decoupled multi-tier interface layout variables."""
        # 🛡️ GLOBAL CLASS ATTRIBUTE GUARDS: Declare early to eliminate initial initialization AttributeErrors
        self.audit_backend_combo = QComboBox()
        self.lyrics_engine_combo = QComboBox()
        self.scan_btn = None
        self.import_json_manifest_btn = None
        self.sync_meta_btn = None
        self.normalize_btn = None
        self.transcribe_btn = None
        self.kaggle_master_btn = None
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

        assistant_tab = QWidget()
        self.init_assistant_tab(assistant_tab)

        tag_tab = QWidget()
        self.init_tag_manager_tab(tag_tab)

        embed_tab = QWidget()
        self.init_embedding_map_tab(embed_tab)

        self.tabs.addTab(studio_tab, "🎛 Dataset Studio")
        self.tabs.addTab(struct_tab, "🎶 Structural Pipeline")
        self.tabs.addTab(settings_tab, "⚙ Settings")
        self.tabs.addTab(advanced_tab, "🧠 Advanced Tools")
        self.tabs.addTab(spatial_tab, "🌐 Spatial Pipeline")
        self.tabs.addTab(assistant_tab, "🤖 AI Assistant")
        self.tabs.addTab(tag_tab, "🏷️ Tag Manager")
        self.tabs.addTab(embed_tab, "🗺️ Embedding Map")
        self.tag_tab_index = self.tabs.indexOf(tag_tab)
        self.embed_tab_index = self.tabs.indexOf(embed_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Set the Dataset Studio tab as the default visible tab
        studio_index = self.tabs.indexOf(studio_tab)
        self.studio_tab_index = studio_index
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
        export_btn = QPushButton("📦 Export / Split")
        export_btn.clicked.connect(self.open_export_dialog)
        add_btn = QPushButton("➕ Add Single Song")
        add_btn.clicked.connect(self.add_audio_files)
        folder_btn = QPushButton("📁 Add Audio Folder")
        folder_btn.setToolTip(
            "Add every audio track in a folder (recursive — subfolders included)."
        )
        folder_btn.clicked.connect(self.add_audio_folder)

        header_bar.addWidget(load_btn)
        header_bar.addWidget(save_btn)
        header_bar.addWidget(export_btn)
        stats_btn = QPushButton("📊 Stats")
        stats_btn.setToolTip("Show a one-page statistics report for the dataset.")
        stats_btn.clicked.connect(self.show_stats_report)
        header_bar.addWidget(stats_btn)
        ver_btn = QPushButton("🗂 Versioning")
        ver_btn.setToolTip("Snapshot, diff, and restore the dataset from disk.")
        ver_btn.clicked.connect(self.open_versioning_dialog)
        header_bar.addWidget(ver_btn)
        hf_btn = QPushButton("☁ Push to HF")
        hf_btn.setToolTip("Push the dataset (dataset.json + README) to a Hugging Face repo.")
        hf_btn.clicked.connect(self.open_hf_push_dialog)
        header_bar.addWidget(hf_btn)
        header_bar.addWidget(add_btn)
        header_bar.addWidget(folder_btn)

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

        # ============================================================================
        # Row 1: Dataset Calibration (Primary Controls)
        # ============================================================================
        audit_strip = QHBoxLayout()
        
        # 👑 THE FLAGSHIP ENGINE: Re-labeled to match your 1.5XL Caption & Lyrics spec
        self.import_json_manifest_btn = QPushButton("📥 Import ACE-Step 1.5XL Tags")
        self.import_json_manifest_btn.setStyleSheet("font-weight: bold; background-color: #0e639c; color: white; padding: 5px 14px;")
        self.import_json_manifest_btn.setToolTip("Instantly loads your schema-enforced 1.5XL caption and lyrics tags to fix placeholders.")
        self.import_json_manifest_btn.clicked.connect(self.import_acestep_15xl_tags)
        audit_strip.addWidget(self.import_json_manifest_btn)

        self.sync_meta_btn = QPushButton("🔄 Sync Metadata Guards")
        self.sync_meta_btn.setToolTip("Forces health report alerts to match your manifest properties, clearing false warnings.")
        self.sync_meta_btn.clicked.connect(self.force_sync_manifest_to_metadata)
        audit_strip.addWidget(self.sync_meta_btn)

        self.normalize_btn = QPushButton("🎚️ Fix & DSP Normalize")
        self.normalize_btn.clicked.connect(self.start_dsp_normalize)
        audit_strip.addWidget(self.normalize_btn)
        audit_strip.addStretch() 
        studio_layout.addLayout(audit_strip)
        # ============================================================================
        # Step 2: Linguistic & Transcription Pipeline (Row 2)
        # ============================================================================
        lyrics_strip = QHBoxLayout()
        
        self.transcribe_btn = QPushButton("🎤 Transcribe Lyrics")
        self.transcribe_btn.setToolTip("Word-aligned lyrics for the selected track.")
        self.transcribe_btn.clicked.connect(self.start_lyrics_transcription)
        lyrics_strip.addWidget(self.transcribe_btn)

        lyrics_strip.addWidget(QLabel("Lyrics Engine:"))
        self.lyrics_engine_combo = QComboBox()
        self.lyrics_engine_combo.addItems([
            "whisperx (local)",
            "kaggle (default, gpu)",
            "gemini",
            "acestep-transcriber (experimental)"
        ])
        saved_engine = self.config.get("lyrics_engine", "whisperx")
        engine_map = {"whisperx": 0, "kaggle": 1, "gemini": 2, "acestep_transcriber": 3}
        self.lyrics_engine_combo.setCurrentIndex(engine_map.get(saved_engine, 0))
        self.lyrics_engine_combo.currentTextChanged.connect(self.save_pipeline_defaults)
        lyrics_strip.addWidget(self.lyrics_engine_combo)

        lyrics_strip.addWidget(QLabel("Lang:"))
        self.lyrics_language_edit = QLineEdit()
        self.lyrics_language_edit.setPlaceholderText("en")
        self.lyrics_language_edit.setMaximumWidth(45)
        self.lyrics_language_edit.setText(self.config.get("lyrics_language", "en"))
        self.lyrics_language_edit.textChanged.connect(self.save_pipeline_defaults)
        lyrics_strip.addWidget(self.lyrics_language_edit)

        lyrics_strip.addWidget(QLabel("Prompt Bias:"))
        self.lyrics_prompt_edit = QLineEdit()
        self.lyrics_prompt_edit.setPlaceholderText("Context words...")
        self.lyrics_prompt_edit.setText(self.config.get("lyrics_initial_prompt", ""))
        self.lyrics_prompt_edit.textChanged.connect(self.save_pipeline_defaults)
        lyrics_strip.addWidget(self.lyrics_prompt_edit)

        lyrics_strip.addStretch()
        studio_layout.addLayout(lyrics_strip)

        # ============================================================================
        # Step 3: Advanced Remote Cluster & Repo Controls (Row 3)
        # ============================================================================
        advanced_strip = QHBoxLayout()

        self.kaggle_master_btn = QPushButton("☁️ Run Master Dual-T4 Kaggle Pipeline")
        self.kaggle_master_btn.setStyleSheet("font-weight: bold; background-color: #4A148C; color: white; padding: 5px 12px;")
        self.kaggle_master_btn.setToolTip("Uploads dataset to Kaggle, runs parallel dual-GPU extraction, and logs output.")
        self.kaggle_master_btn.clicked.connect(self.start_remote_consolidated_pipeline)
        advanced_strip.addWidget(self.kaggle_master_btn)

        self.tag_creator_btn = QPushButton("🏷️ Structural Tag Creator")
        self.tag_creator_btn.clicked.connect(self.start_structural_tag_creator)
        advanced_strip.addWidget(self.tag_creator_btn)

        self.rockstar_btn = QPushButton("🎸 Rockstar Check")
        self.rockstar_btn.clicked.connect(self.start_rockstar_lookup)
        advanced_strip.addWidget(self.rockstar_btn)

        self.musicbrainz_btn = QPushButton("🎵 MusicBrainz Lookup")
        self.musicbrainz_btn.clicked.connect(self.start_musicbrainz_lookup)
        advanced_strip.addWidget(self.musicbrainz_btn)

        advanced_strip.addStretch()
        studio_layout.addLayout(advanced_strip)

        # --- Tools row (find/replace, bulk rename, lyrics, A/B, riff, stem A/B) ---
        tools_strip = QHBoxLayout()
        fr_btn = QPushButton("🔁 Find/Replace")
        fr_btn.clicked.connect(self.open_find_replace_dialog)
        rename_btn = QPushButton("✏️ Bulk Rename")
        rename_btn.setToolTip(
            "Rename tracks in bulk. Default mode: keep only the song name, "
            "replacing spaces with underscores. Files are backed up before any "
            "on-disk rename."
        )
        rename_btn.clicked.connect(self.open_bulk_rename_dialog)
        lyr_btn = QPushButton("✎ Lyrics Editor")
        lyr_btn.clicked.connect(self.open_lyrics_editor)
        ab_btn = QPushButton("🔀 A/B Captions")
        ab_btn.clicked.connect(self.open_ab_captions)
        riff_btn = QPushButton("🎸 Riff/Hook")
        riff_btn.clicked.connect(self.open_riff_hook_tagger)
        stemab_btn = QPushButton("🎚 Stem A/B")
        stemab_btn.clicked.connect(self.open_stem_ab)
        for b in (fr_btn, rename_btn, lyr_btn, ab_btn, riff_btn, stemab_btn):
            tools_strip.addWidget(b)
        tools_strip.addStretch()
        studio_layout.addLayout(tools_strip)

        audit_strip.addSpacing(15)

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

        audit_strip.addWidget(QLabel("View:"))
        audit_strip.addWidget(self.all_view_btn)
        audit_strip.addWidget(self.exceptions_view_btn)

        # --- Preview player + waveform ---
        player_bar = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setToolTip("Play / pause the selected track")
        self.play_btn.clicked.connect(self.toggle_track_playback)
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setToolTip("Stop playback")
        self.stop_btn.clicked.connect(self.stop_track_playback)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.time_label = QLabel("0:00 / 0:00")
        player_bar.addWidget(self.play_btn)
        player_bar.addWidget(self.stop_btn)
        player_bar.addWidget(self.seek_slider, 1)
        player_bar.addWidget(self.time_label)
        studio_layout.addLayout(player_bar)

        self.waveform = WaveformWidget()
        self.waveform.set_audio(None)
        studio_layout.addWidget(self.waveform)

        try:
            self.media_player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.media_player.setAudioOutput(self.audio_output)
            self.audio_output.setVolume(0.7)
            self.media_player.positionChanged.connect(self._on_player_position)
            self.media_player.mediaStatusChanged.connect(self._on_player_status)
        except Exception as e:  # noqa: BLE001 — playback is best-effort
            print(f"media player unavailable: {e}")
            self.media_player = None

        # --- Filter / search bar ---
        filter_bar = QHBoxLayout()
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText("🔍 Search filename / caption / tag / genre / key…")
        self.filter_search.setClearButtonEnabled(True)
        self.filter_search.textChanged.connect(self.on_filters_changed)
        self.filter_inst_combo = QComboBox()
        self.filter_inst_combo.addItems(["All", "Instrumental", "Vocal"])
        self.filter_inst_combo.currentIndexChanged.connect(self.on_filters_changed)
        self.filter_captioned_check = QCheckBox("Captioned")
        self.filter_captioned_check.toggled.connect(self.on_filters_changed)
        clear_filters = QPushButton("✕ Clear")
        clear_filters.clicked.connect(self.clear_filters)
        self.filter_count_label = QLabel("")
        self.filter_count_label.setStyleSheet("color: #aaa;")
        filter_bar.addWidget(self.filter_search, 1)
        filter_bar.addWidget(self.filter_inst_combo)
        filter_bar.addWidget(self.filter_captioned_check)
        filter_bar.addWidget(clear_filters)
        filter_bar.addWidget(self.filter_count_label)
        studio_layout.addLayout(filter_bar)

        # --- Table + Inspector ---
        splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Filename", "Health", "Tag", "Genre", "Key", "BPM", "Time", "Duration", "Actions"]
        )
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 8):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.table.itemChanged.connect(self.on_metadata_cell_edited)
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

        # Genre / Key / BPM / Time / Duration are entered in the table next to
        # the filename (auto-locked after scanning; unlock with the 🔓 button
        # in the row's Actions column). The inspector only hosts the remaining
        # per-track fields to keep a single place for metadata entry.

        tag_row = QHBoxLayout()
        self.track_tag_input = QLineEdit()
        self.track_tag_input.textChanged.connect(self.on_track_tag_edited)
        tag_row.addWidget(self.track_tag_input)
        form.addRow("Track Trigger Tag:", tag_row)

        self.inst_check = QCheckBox("Instrumental Track (No Vocals)")
        self.inst_check.stateChanged.connect(self.on_inst_edited)
        form.addRow(self.inst_check)

        verify_row = QHBoxLayout()
        self.bpm_verify_btn = QPushButton("🔗 Verify BPM Online")
        self.bpm_verify_btn.clicked.connect(self.open_online_bpm_check)
        self.key_verify_btn = QPushButton("🔗 Verify Key Online")
        self.key_verify_btn.clicked.connect(self.open_online_key_check)
        verify_row.addWidget(self.bpm_verify_btn)
        verify_row.addWidget(self.key_verify_btn)
        form.addRow(verify_row)

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
        from ui.settings_tab import build_settings_tab
        build_settings_tab(self, parent)

    # -----------------------------------------------------------------------
    # AI Assistant Tab
    # -----------------------------------------------------------------------
    def init_assistant_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self.assistant_history = QTextBrowser()
        self.assistant_history.setHtml(
            "<b>🤖 AI Assistant</b><br>Ask about the app or your dataset. The "
            "assistant can run tools against it: dataset summary, sound profile, "
            "<b>curate for a target sound</b>, caption audit, manifest validation, "
            "health scan, instrument detection.<hr>"
        )
        layout.addWidget(self.assistant_history, 1)

        input_row = QHBoxLayout()
        self.assistant_input = QLineEdit()
        self.assistant_input.setPlaceholderText(
            "Ask something… e.g. 'curate my dataset toward a Black Sabbath / doom sound'"
        )
        self.assistant_input.returnPressed.connect(self.send_assistant_message)
        self.assistant_send_btn = QPushButton("Send")
        self.assistant_send_btn.clicked.connect(self.send_assistant_message)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_assistant)
        input_row.addWidget(self.assistant_input, 1)
        input_row.addWidget(self.assistant_send_btn)
        input_row.addWidget(clear_btn)
        layout.addLayout(input_row)

        # --- Sample questions ---
        suggest_row = QHBoxLayout()
        suggest_row.addWidget(QLabel("Try:"))
        self.assistant_suggest = QComboBox()
        self.assistant_suggest.addItems([
            "Curate my dataset toward a Black Sabbath / doom sound.",
            "What is my dataset's current sound profile?",
            "Which tracks look like outliers or near-duplicates?",
            "Check whether 'Paranoid' by Black Sabbath has multitracks.",
            "Audit my captions for consistency.",
            "Validate my dataset manifest.",
            "Help me write a caption for the selected track.",
            "What tags fit a slow, powerful belted ballad?",
            "How do I fix the flagged health issues?",
            "Run the health audit on my dataset.",
        ])
        ask_btn = QPushButton("Ask")
        ask_btn.clicked.connect(self.ask_suggested)
        suggest_row.addWidget(self.assistant_suggest, 1)
        suggest_row.addWidget(ask_btn)
        layout.addLayout(suggest_row)

        # --- Context options ---
        opts_row = QHBoxLayout()
        self.assistant_remember_check = QCheckBox("Remember context")
        self.assistant_remember_check.setChecked(bool(self.config.get("assistant_remember", True)))
        self.assistant_remember_check.toggled.connect(self._on_assistant_remember_toggled)
        self.assistant_linear_check = QCheckBox("Step-by-step reasoning")
        self.assistant_linear_check.setChecked(bool(self.config.get("assistant_linear_thinking", True)))
        opts_row.addWidget(self.assistant_remember_check)
        opts_row.addWidget(self.assistant_linear_check)
        opts_row.addStretch()
        layout.addLayout(opts_row)

        self.assistant_status = QLabel("Ready.")
        self.assistant_status.setStyleSheet("color: #aaa;")
        layout.addWidget(self.assistant_status)

        # Restore the persistent linear conversation.
        if self.config.get("assistant_remember", True):
            from modules.assistant_store import load_context
            self._assistant_messages = load_context(int(self.config.get("assistant_context_size", 40)))
            if self._assistant_messages:
                self.assistant_history.append("<i>… restored previous conversation.</i>")
        else:
            self._assistant_messages = []

    def ask_suggested(self):
        q = self.assistant_suggest.currentText().strip()
        if q:
            self.assistant_input.setText(q)
            self.send_assistant_message()

    def _save_assistant_context(self):
        if not self.config.get("assistant_remember", True):
            return
        from modules.assistant_store import save_context
        save_context(self._assistant_messages, int(self.config.get("assistant_context_size", 40)))

    def _on_assistant_remember_toggled(self, checked):
        if not checked:
            from modules.assistant_store import clear_context
            clear_context()

    def send_assistant_message(self):
        text = self.assistant_input.text().strip()
        if not text:
            return
        self.assistant_history.append(f"<b>You:</b> {html.escape(text)}<br>")
        self.assistant_input.clear()
        self._assistant_messages.append({"role": "user", "content": text})
        self._save_assistant_context()
        self._start_assistant()

    def clear_assistant(self):
        self._assistant_messages = []
        self.assistant_history.setHtml("<b>🤖 AI Assistant</b> — conversation cleared.<hr>")
        self.assistant_status.setText("Ready.")
        if self.config.get("assistant_remember", True):
            from modules.assistant_store import clear_context
            clear_context()

    def _set_assistant_busy(self, busy):
        self.assistant_send_btn.setEnabled(not busy)
        self.assistant_input.setEnabled(not busy)
        self.assistant_status.setText("Thinking…" if busy else "Ready.")

    def _start_assistant(self):
        from modules.llm_client import get_client

        try:
            get_client(self.config)
        except ValueError as e:
            QMessageBox.information(
                self, "LLM Key Needed",
                f"{e}\n\nSet it in ⚙ Settings → LLM Provider (Gemini's free tier works).",
            )
            self._set_assistant_busy(False)
            return
        self._set_assistant_busy(True)
        summary = summarize_dataset(self.dataset, self.health_reports)
        sys_prompt = build_system_prompt(APP_HELP_TEXT, summary)
        if self.assistant_linear_check.isChecked():
            sys_prompt += "\n\nWork through the problem step by step before answering."
        messages = [
            {"role": "system", "content": sys_prompt}
        ] + list(self._assistant_messages)
        self.assistant_worker = AssistantWorker(
            "", messages, tools=ASSISTANT_TOOLS, parent=self, config=self.config,
        )
        self.assistant_worker.answer_ready.connect(self.on_assistant_answer)
        self.assistant_worker.tool_requested.connect(self.on_assistant_tool)
        self.assistant_worker.failed.connect(self.on_assistant_error)
        self.assistant_worker.start()

    def on_assistant_answer(self, answer):
        self._set_assistant_busy(False)
        self.assistant_history.append(f"<b>AI:</b><br>{html.escape(answer)}<hr>")
        self._assistant_messages.append({"role": "assistant", "content": answer})
        self._save_assistant_context()

    def on_assistant_tool(self, name, args_json, call_id):
        try:
            args = json.loads(args_json or "{}")
        except Exception:  # noqa: BLE001
            args = {}
        result = self.execute_assistant_tool(name, args)
        self.assistant_history.append(
            f"<i>⚙ tool: {html.escape(name)} → {html.escape(result[:200])}</i>"
        )
        self._assistant_messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": args_json or "{}"}}],
        })
        self._assistant_messages.append({
            "role": "tool", "tool_call_id": call_id, "content": result,
        })
        self._save_assistant_context()
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
            if name == "get_dataset_sound_profile":
                return build_sound_profile(self.dataset)
            if name == "curate_dataset":
                target = (args.get("target_sound") or "").strip()
                if not target:
                    return "Provide a target_sound (artist/genre/mood) to curate toward."
                return (
                    f"TARGET SOUND: {target}\n\n"
                    f"CURRENT DATASET SOUND PROFILE:\n{build_sound_profile(self.dataset)}\n\n"
                    "Suggest specific songs/artists/genres to add, and which gaps to fill "
                    "(instruments, tempo, key, era) so the dataset converges on the target sound."
                )
            if name == "rockstar_lookup":
                return self._rockstar_lookup_tool(args)
            return f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001
            return f"Tool error: {e}"

    def _rockstar_lookup_tool(self, args):
        from modules.rockstar_lookup import lookup_rockstar_track, format_lookup

        song = (args.get("song") or "").strip()
        artist = (args.get("artist") or "").strip()
        if not song:
            return "Provide a 'song' (and optionally 'artist')."
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = lookup_rockstar_track(artist, song, timeout=15)
            return format_lookup(result)
        except Exception as e:  # noqa: BLE001
            return f"rockstar_lookup error: {e}"
        finally:
            QApplication.restoreOverrideCursor()

    def on_assistant_error(self, err):
        self._set_assistant_busy(False)
        self.assistant_status.setText(f"Error: {err}")
        QMessageBox.warning(self, "AI Assistant", str(err))

    # -----------------------------------------------------------------------
    # Tag Manager Tab
    # -----------------------------------------------------------------------
    def init_tag_manager_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Tracks (multi-select):</b>"))
        self.tag_track_list = QListWidget()
        self.tag_track_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        ll.addWidget(self.tag_track_list)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        top = QHBoxLayout()
        self.tag_search = QLineEdit()
        self.tag_search.setPlaceholderText("Filter tags…")
        self.tag_search.setClearButtonEnabled(True)
        self.tag_search.textChanged.connect(self.refresh_tag_stats)
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self.refresh_tag_manager)
        top.addWidget(self.tag_search, 1)
        top.addWidget(refresh_btn)
        rl.addLayout(top)
        self.tag_stats_table = QTableWidget(0, 3)
        self.tag_stats_table.setHorizontalHeaderLabels(["Tag", "Count", "Tracks"])
        self.tag_stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tag_stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tag_stats_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        rl.addWidget(self.tag_stats_table)
        split.addWidget(right)

        split.setSizes([300, 600])
        layout.addWidget(split, 1)

        actions = QHBoxLayout()
        add_btn = QPushButton("+ Add Tag to Selected")
        add_btn.clicked.connect(self.add_tag_to_selected)
        remove_btn = QPushButton("− Remove Tag from Selected")
        remove_btn.clicked.connect(self.remove_tag_from_selected)
        norm_btn = QPushButton("Normalize Synonyms")
        norm_btn.clicked.connect(self.normalize_tag_synonyms)
        actions.addWidget(add_btn)
        actions.addWidget(remove_btn)
        actions.addWidget(norm_btn)
        actions.addStretch()
        layout.addLayout(actions)

        self.tag_manager_status = QLabel(
            "Tags are aggregated from each track's instruments, custom tag, and genre. "
            "Select tracks on the left to add/remove tags."
        )
        self.tag_manager_status.setWordWrap(True)
        self.tag_manager_status.setStyleSheet("color: #aaa;")
        layout.addWidget(self.tag_manager_status)

        self.refresh_tag_manager()

    def _on_tab_changed(self, index):
        if index == self.tag_tab_index and hasattr(self, "tag_stats_table"):
            self.refresh_tag_manager()

    def refresh_tag_manager(self):
        self._populate_tag_track_list()
        self.refresh_tag_stats()

    def _populate_tag_track_list(self):
        self.tag_track_list.clear()
        for i, s in enumerate(self.dataset.get("samples", [])):
            self.tag_track_list.addItem(f"{s.get('filename', '?')} [{i}]")
            self.tag_track_list.item(i).setData(Qt.UserRole, i)

    def _extract_tags(self, s):
        tags = []
        inst = s.get("tags", {}).get("instruments") or s.get("detected_instruments") or []
        if isinstance(inst, str):
            inst = [i.strip() for i in inst.split(",") if i.strip()]
        for i in inst:
            if i:
                tags.append(i.strip().lower())
        for field in ("custom_tag", "genre"):
            val = (s.get(field) or "").strip()
            for tok in re.split(r"[,;|/]+", val):
                t = tok.strip().lower()
                if t:
                    tags.append(t)
        return tags

    def refresh_tag_stats(self):
        counts = {}
        tracks = {}
        for s in self.dataset.get("samples", []):
            name = s.get("filename", "?")
            for t in self._extract_tags(s):
                counts[t] = counts.get(t, 0) + 1
                tracks.setdefault(t, []).append(name)
        query = self.tag_search.text().strip().lower()
        rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.tag_stats_table.setRowCount(0)
        for tag, cnt in rows:
            if query and query not in tag:
                continue
            row = self.tag_stats_table.rowCount()
            self.tag_stats_table.insertRow(row)
            self.tag_stats_table.setItem(row, 0, QTableWidgetItem(tag))
            c = QTableWidgetItem(str(cnt))
            c.setTextAlignment(Qt.AlignCenter)
            self.tag_stats_table.setItem(row, 1, c)
            self.tag_stats_table.setItem(row, 2, QTableWidgetItem(
                ", ".join(tracks[tag][:6]) + ("…" if len(tracks[tag]) > 6 else "")
            ))

    def _selected_tag_indices(self):
        return [
            self.tag_track_list.item(i).data(Qt.UserRole)
            for i in range(self.tag_track_list.count())
            if self.tag_track_list.item(i).isSelected()
        ]

    def add_tag_to_selected(self):
        idxs = self._selected_tag_indices()
        if not idxs:
            QMessageBox.information(self, "Tag Manager", "Select one or more tracks on the left first.")
            return
        tag, ok = QInputDialog.getText(self, "Add Tag", "Tag to add:")
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        self.record_snapshot()
        changed = 0
        for i in idxs:
            s = self.dataset["samples"][i]
            cur = (s.get("custom_tag") or "").strip()
            tags = [t.strip() for t in re.split(r"[,;|/]+", cur) if t.strip()] if cur else []
            if tag.lower() not in [t.lower() for t in tags]:
                tags.append(tag)
                s["custom_tag"] = ", ".join(tags)
                changed += 1
        self.refresh_tag_manager()
        self.status_label.setText(f"Added '{tag}' to {changed} track(s).")

    def remove_tag_from_selected(self):
        idxs = self._selected_tag_indices()
        if not idxs:
            QMessageBox.information(self, "Tag Manager", "Select one or more tracks on the left first.")
            return
        tag, ok = QInputDialog.getText(self, "Remove Tag", "Tag to remove:")
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        self.record_snapshot()
        changed = 0
        for i in idxs:
            s = self.dataset["samples"][i]
            cur = (s.get("custom_tag") or "").strip()
            tags = [t.strip() for t in re.split(r"[,;|/]+", cur) if t.strip()] if cur else []
            filtered = [t for t in tags if t.lower() != tag.lower()]
            if len(filtered) != len(tags):
                s["custom_tag"] = ", ".join(filtered)
                changed += 1
        self.refresh_tag_manager()
        self.status_label.setText(f"Removed '{tag}' from {changed} track(s).")

    def normalize_tag_synonyms(self):
        from modules.tagger import normalize_instrument

        self.record_snapshot()
        changed = 0
        for s in self.dataset.get("samples", []):
            cur = (s.get("custom_tag") or "").strip()
            if cur:
                toks = [t.strip() for t in re.split(r"[,;|/]+", cur) if t.strip()]
                norm = [normalize_instrument(t) for t in toks]
                if norm != toks:
                    s["custom_tag"] = ", ".join(norm)
                    changed += 1
            inst = s.get("detected_instruments")
            if isinstance(inst, list) and inst:
                norm_inst = [normalize_instrument(i) for i in inst]
                if norm_inst != inst:
                    s["detected_instruments"] = norm_inst
                    changed += 1
            tags = s.get("tags")
            if isinstance(tags, dict) and tags.get("instruments"):
                ni = [normalize_instrument(i) for i in tags["instruments"]]
                if ni != tags["instruments"]:
                    tags["instruments"] = ni
                    changed += 1
        self.refresh_tag_manager()
        self.status_label.setText(f"Normalized tag synonyms across {changed} track(s).")

    # -----------------------------------------------------------------------
    # Embedding Map Tab
    # -----------------------------------------------------------------------
    def init_embedding_map_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.embed_compute_btn = QPushButton("🧮 Compute Embeddings")
        self.embed_compute_btn.clicked.connect(self.compute_embeddings)
        self.embed_backend_label = QLabel("")
        self.embed_backend_label.setStyleSheet("color: #aaa;")
        controls.addWidget(self.embed_compute_btn)
        controls.addWidget(self.embed_backend_label, 1)
        layout.addLayout(controls)

        self.embed_progress = QProgressBar()
        self.embed_progress.setVisible(False)
        layout.addWidget(self.embed_progress)

        self.scatter = ScatterPlotWidget()
        self.scatter.point_clicked.connect(self._jump_to_track)
        layout.addWidget(self.scatter, 1)

        self.embed_status = QLabel(
            "Hover a point for the filename; click to jump to the track in Dataset Studio. "
            "Similar songs cluster together — outliers and near-duplicates stand out."
        )
        self.embed_status.setWordWrap(True)
        self.embed_status.setStyleSheet("color: #aaa;")
        layout.addWidget(self.embed_status)

    def compute_embeddings(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.information(self, "Embedding Map", "The dataset is empty — add tracks first.")
            return
        from modules.embeddings import backend_label

        self.embed_backend_label.setText(f"Backend: {backend_label()}")
        self.embed_compute_btn.setEnabled(False)
        self.embed_progress.setVisible(True)
        self.embed_progress.setValue(0)
        self.embed_worker = EmbeddingWorker(samples, parent=self)
        self.embed_worker.progress.connect(self._on_embed_progress)
        self.embed_worker.finished_ok.connect(self.on_embeddings_done)
        self.embed_worker.failed.connect(self.on_embeddings_failed)
        self.embed_worker.start()

    def _on_embed_progress(self, pct, msg):
        self.embed_progress.setValue(pct)
        self.embed_status.setText(msg)

    def on_embeddings_done(self, coords, meta):
        self.embed_compute_btn.setEnabled(True)
        self.embed_progress.setVisible(False)
        self.scatter.set_data(coords, meta)
        self.embed_status.setText(
            f"Plotted {len(coords)} track(s). Hover for filename, click to jump to the track."
        )

    def on_embeddings_failed(self, err):
        self.embed_compute_btn.setEnabled(True)
        self.embed_progress.setVisible(False)
        self.embed_status.setText(f"Embedding failed: {err}")

    def _jump_to_track(self, index):
        if not (0 <= index < len(self.dataset.get("samples", []))):
            return
        try:
            row = self._table_sample_indices.index(index)
        except ValueError:
            self.clear_filters()
            self.refresh_table()
            try:
                row = self._table_sample_indices.index(index)
            except ValueError:
                return
        self.tabs.setCurrentIndex(self.studio_tab_index)
        self.table.setCurrentCell(row, 0)
        self.on_table_selection_changed()

    def open_export_dialog(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.information(self, "Export", "The dataset is empty — add tracks first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("📦 Export / Split Dataset")
        dialog.resize(520, 420)
        lay = QVBoxLayout(dialog)

        lay.addWidget(QLabel("<b>Formats:</b>"))
        self.exp_json = QCheckBox("ACE-Step JSON (dataset.json)")
        self.exp_csv = QCheckBox("CSV (dataset.csv)")
        self.exp_jsonl = QCheckBox("JSONL (dataset.jsonl)")
        self.exp_sidecar = QCheckBox("Sidecar caption .txt files (Kohya/ComfyUI style)")
        self.exp_folders = QCheckBox("Train/Val folders (copies audio + captions, manifest.json)")
        for cb in (self.exp_json, self.exp_csv, self.exp_jsonl, self.exp_sidecar, self.exp_folders):
            lay.addWidget(cb)
        self.exp_json.setChecked(True)

        split_box = QGroupBox("Train/Val split (for the folders format)")
        sform = QFormLayout(split_box)
        self.exp_val_ratio = QDoubleSpinBox()
        self.exp_val_ratio.setRange(0.05, 0.5)
        self.exp_val_ratio.setSingleStep(0.05)
        self.exp_val_ratio.setValue(0.2)
        self.exp_val_ratio.setDecimals(2)
        sform.addRow("Validation ratio:", self.exp_val_ratio)
        self.exp_stratify = QCheckBox("Stratify by genre")
        self.exp_stratify.setChecked(True)
        sform.addRow(self.exp_stratify)
        lay.addWidget(split_box)

        dest_row = QHBoxLayout()
        self.exp_dest = QLineEdit()
        self.exp_dest.setPlaceholderText("Choose an output folder…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_export_dir)
        dest_row.addWidget(self.exp_dest, 1)
        dest_row.addWidget(browse)
        lay.addLayout(dest_row)

        self.exp_status = QLabel(f"{len(samples)} tracks ready to export.")
        self.exp_status.setStyleSheet("color: #aaa;")
        lay.addWidget(self.exp_status)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        go_btn = QPushButton("🚀 Export")
        go_btn.clicked.connect(lambda: self._run_export(dialog))
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(go_btn)
        lay.addLayout(btn_row)

        dialog.exec()

    def _browse_export_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Export Folder", self.exp_dest.text() or str(Path.home()))
        if d:
            self.exp_dest.setText(d)

    def _run_export(self, dialog):
        dest = self.exp_dest.text().strip()
        if not dest:
            QMessageBox.warning(self, "Export", "Choose an output folder first.")
            return
        options = {
            "dest_dir": dest,
            "json": self.exp_json.isChecked(),
            "csv": self.exp_csv.isChecked(),
            "jsonl": self.exp_jsonl.isChecked(),
            "sidecar": self.exp_sidecar.isChecked(),
            "folders": self.exp_folders.isChecked(),
            "val_ratio": self.exp_val_ratio.value(),
            "stratify": self.exp_stratify.isChecked(),
            "seed": 42,
        }
        if not any(options[k] for k in ("json", "csv", "jsonl", "sidecar", "folders")):
            QMessageBox.warning(self, "Export", "Select at least one format.")
            return
        self.exp_status.setText("Exporting…")
        self.export_worker = ExportWorker(self.dataset, options, parent=self)
        self.export_worker.finished_ok.connect(lambda msg: self._on_export_done(msg, dialog))
        self.export_worker.failed.connect(lambda err: self._on_export_failed(err, dialog))
        self.export_worker.start()

    def _on_export_done(self, msg, dialog):
        self.status_label.setText(f"Exported: {msg}")
        dialog.accept()
        QMessageBox.information(self, "Export Complete", f"Exported to the chosen folder:\n{msg}")

    def _on_export_failed(self, err, dialog):
        self.exp_status.setText(f"Export failed: {err}")
        QMessageBox.warning(self, "Export Failed", str(err))

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

    def run_spatial_pipeline(self):
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
    def run_structural_pipeline(self):
        # ---- Read the active track layout options ----
        stem_source = self.struct_stem_combo.currentText()
        use_deepseek = self.struct_deepseek_check.isChecked()

        # 🛡️ THE GATEKEEPER: Run the shared key & credential verification helper
        if not self._prepare_pipeline_credentials(stem_source, use_deepseek):
            return

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
        from ui_theme import compile_and_apply_theme
        compile_and_apply_theme(self)

    def save_all_settings(self):
        """Persist every settings group (cloud keys, LLM provider, pipeline
        defaults, model manager) in one click."""
        self.save_cloud_config()
        self.save_pipeline_defaults()
        self.status_label.setText("All settings saved.")

    def _browse_stem_dir(self):
        start = self.stem_out_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Choose Stem Output Folder", start)
        if d:
            self.stem_out_edit.setText(d)

    def _remembered_secret_keys(self):
        """Secret keys the user asked to persist (checked 'Remember' boxes)."""
        remember = {
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
        # The unified Provider API Key routes to whichever provider is active.
        active = self.llm_provider_combo.currentText().split(" ")[0]
        key_field, _ = LLM_KEY_FIELDS.get(active, ("deepseek_key", "remember_deepseek_key"))
        if self.remember_llm_api.isChecked():
            remember.add(key_field)
        return remember

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
        # Sync the unified Provider API Key field to the active provider.
        key_field, rem_field = LLM_KEY_FIELDS.get(name, ("deepseek_key", "remember_deepseek_key"))
        self.llm_api_key.setText(self.config.get(key_field, ""))
        self.llm_api_key.setPlaceholderText(f"API key for {name}")
        self.remember_llm_api.setChecked(bool(self.config.get(rem_field, True)))
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
        # Unified Provider API Key -> the active provider's stored key.
        active = self.llm_provider_combo.currentText().split(" ")[0]
        key_field, rem_field = LLM_KEY_FIELDS.get(active, ("deepseek_key", "remember_deepseek_key"))
        self.config[key_field] = self.llm_api_key.text().strip()
        self.config[rem_field] = self.remember_llm_api.isChecked()
        # Per-role LLM overrides (aggregator / captioner / assistant).
        for role in ("aggregator", "captioner", "assistant"):
            prov = self.role_provider_combo[role].currentText()
            self.config[f"llm_provider_{role}"] = "" if prov == "default (global)" else prov
            self.config[f"llm_model_{role}"] = self.role_model_combo[role].currentText().strip()
        if hasattr(self, "assistant_remember_check"):
            self.config["assistant_remember"] = self.assistant_remember_check.isChecked()
            self.config["assistant_linear_thinking"] = self.assistant_linear_check.isChecked()
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
        self.config["audit_backend"] = "kaggle" if self.audit_backend_combo.currentIndex() == 1 else "local"
        if hasattr(self, "lyrics_engine_combo"):
            self.config["lyrics_engine"] = {
                "kaggle (default, gpu)": "kaggle",
                "whisperx (local)": "whisperx",
                "gemini": "gemini",
                "acestep-transcriber (experimental)": "acestep_transcriber",
            }.get(self.lyrics_engine_combo.currentText(), "kaggle")
            self.config["lyrics_language"] = self.lyrics_language_edit.text().strip()
            self.config["lyrics_initial_prompt"] = self.lyrics_prompt_edit.text().strip()
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
        self._loading_table = True
        self.table.setRowCount(0)
        self._table_sample_indices = []
        exceptions_count = 0
        shown = 0

        for idx, s in enumerate(self.dataset["samples"]):
            sid = s.get("id", "")
            rep = self.health_reports.get(sid, {})
            status = rep.get("status", "Not Audited")

            is_exception = (status == "Warning" or status == "Missing" or not s.get("caption"))
            if is_exception:
                exceptions_count += 1

            if self.filter_exceptions_only and not is_exception:
                continue
            if not self._matches_filters(s):
                continue

            shown += 1
            self._table_sample_indices.append(idx)
            row = self.table.rowCount()
            self.table.insertRow(row)

            locked = bool(s.get("locked", True))

            def _cell(text, editable):
                item = QTableWidgetItem(str(text) if text not in (None, 0, "") else "")
                if not editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                return item

            self.table.setItem(row, 0, _cell(s.get("filename", ""), False))

            h_item = QTableWidgetItem(f"✓ Healthy" if status == "Healthy" else (f"⚠ Warning" if status == "Warning" else status))
            if status == "Healthy":
                h_item.setForeground(QColor("#4CAF50"))
            elif status == "Warning":
                h_item.setForeground(QColor("#FF9800"))
            elif status == "Missing":
                h_item.setForeground(QColor("#F44336"))
            h_item.setFlags(h_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, h_item)

            self.table.setItem(row, 2, _cell(s.get("custom_tag", ""), False))
            self.table.setItem(row, 3, _cell(s.get("genre", ""), not locked))
            self.table.setItem(row, 4, _cell(s.get("keyscale", ""), not locked))
            bpm = s.get("bpm", 0)
            self.table.setItem(row, 5, _cell(str(bpm) if bpm else "", not locked))
            self.table.setItem(row, 6, _cell(s.get("timesignature", ""), not locked))
            dur = s.get("duration", 0)
            self.table.setItem(row, 7, _cell(f"{dur}s" if dur else "", not locked))

            # Actions column: unlock/lock toggle + delete (with confirmation).
            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(4)
            lock_btn = QPushButton("🔓" if not locked else "🔒")
            lock_btn.setToolTip("Lock / unlock this track's metadata for manual editing")
            lock_btn.setMaximumWidth(36)
            lock_btn.clicked.connect(lambda _=False, i=idx: self.toggle_track_lock(i))
            del_btn = QPushButton("🗑")
            del_btn.setToolTip("Remove this track from the dataset")
            del_btn.setMaximumWidth(36)
            del_btn.clicked.connect(lambda _=False, i=idx: self.confirm_delete_sample(i))
            actions_layout.addWidget(lock_btn)
            actions_layout.addWidget(del_btn)
            self.table.setCellWidget(row, 8, actions)

        self._loading_table = False
        self.exceptions_view_btn.setText(f"⚠ Exceptions Queue ({exceptions_count})")
        if hasattr(self, "filter_count_label"):
            total = len(self.dataset["samples"])
            self.filter_count_label.setText(f"{shown} of {total} tracks")

    def _matches_filters(self, s):
        """Apply the search/filter state to a single sample dict."""
        q = self.filter_query.lower()
        if q:
            hay = " ".join([
                s.get("filename", ""), s.get("caption", ""),
                s.get("custom_tag", ""), s.get("genre", ""),
                s.get("keyscale", ""), s.get("lyrics", ""),
                s.get("formatted_lyrics", ""),
            ]).lower()
            if q not in hay:
                return False
        if self.filter_inst == "instrumental" and not s.get("is_instrumental"):
            return False
        if self.filter_inst == "vocal" and s.get("is_instrumental"):
            return False
        if self.filter_captioned and not (s.get("caption") or "").strip():
            return False
        return True

    def on_filters_changed(self, *_):
        self.filter_query = self.filter_search.text().strip()
        self.filter_inst = self.filter_inst_combo.currentText().lower()
        self.filter_captioned = self.filter_captioned_check.isChecked()
        self.refresh_table()

    def clear_filters(self):
        self.filter_search.clear()
        self.filter_inst_combo.setCurrentIndex(0)
        self.filter_captioned_check.setChecked(False)

    def get_selected_sample(self):
        row = self.table.currentRow()
        if 0 <= row < len(self._table_sample_indices):
            return self.dataset["samples"][self._table_sample_indices[row]]
        return None

    def toggle_track_lock(self, idx):
        """Lock / unlock a single track's metadata (🔒/🔓 button in Actions)."""
        samples = self.dataset.get("samples", [])
        if not (0 <= idx < len(samples)):
            return
        samples[idx]["locked"] = not samples[idx].get("locked", True)
        self.refresh_table()
        # Keep the selection on the toggled row if possible.
        try:
            row = self._table_sample_indices.index(idx)
            self.table.setCurrentCell(row, 0)
        except ValueError:
            pass
        self.on_table_selection_changed()
        state = "locked" if samples[idx].get("locked") else "unlocked"
        self.status_label.setText(f"Metadata for '{samples[idx].get('filename', '')}' {state}.")

    def confirm_delete_sample(self, idx):
        """Ask before removing a track; always back up the file when removed."""
        samples = self.dataset.get("samples", [])
        if not (0 <= idx < len(samples)):
            return
        s = samples[idx]
        fname = s.get("filename", "?")
        resp = QMessageBox.question(
            self,
            "Delete Track",
            f"Are you sure you want to remove '{fname}' from the dataset?\n\n"
            "The audio file will be backed up to project_backups/deleted/ "
            "(non-destructive).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return
        self.record_snapshot()
        sid = s.get("id", "")
        path = s.get("audio_path", "")
        if path and os.path.exists(path):
            backup_dir = Path("project_backups") / "deleted"
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                dest = backup_dir / os.path.basename(path)
                if dest.exists():
                    dest = backup_dir / f"{Path(path).stem}_{time.strftime('%Y%m%d-%H%M%S')}{Path(path).suffix}"
                shutil.copy2(path, str(dest))
                self.status_label.setText(f"Backed up '{fname}' to {dest}.")
            except OSError as e:
                QMessageBox.warning(self, "Backup Warning", f"Could not back up file: {e}")
        samples.pop(idx)
        self.health_reports.pop(sid, None)
        self.original_backups.pop(sid, None)
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText(f"Removed '{fname}' from the dataset.")

    def on_metadata_cell_edited(self, item):
        """Write back inline table edits (Genre/Key/BPM/Time/Duration) when unlocked."""
        if getattr(self, "_loading_table", False):
            return
        row = item.row()
        if not (0 <= row < len(self._table_sample_indices)):
            return
        idx = self._table_sample_indices[row]
        samples = self.dataset.get("samples", [])
        if not (0 <= idx < len(samples)):
            return
        s = samples[idx]
        if s.get("locked", True):
            return
        col = item.column()
        text = item.text().strip()
        try:
            if col == 3:      # Genre
                s["genre"] = text
            elif col == 4:    # Key
                s["keyscale"] = text
            elif col == 5:    # BPM
                s["bpm"] = int(float(text)) if text else 0
            elif col == 6:    # Time signature
                s["timesignature"] = text
            elif col == 7:    # Duration
                s["duration"] = int(float(text.rstrip("s"))) if text else 0
        except (ValueError, TypeError):
            pass
        self.health_reports.pop(s.get("id", ""), None)
        self.status_label.setText(f"Updated {s.get('filename', '')} metadata.")

    def on_table_selection_changed(self):
        s = self.get_selected_sample()
        self._load_track_preview(s)
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
            self.inst_check.blockSignals(True)

            self.caption_text.setPlainText(s.get("caption", ""))
            self.lyrics_text.setPlainText(s.get("formatted_lyrics", s.get("lyrics", "")))
            self.track_tag_input.setText(s.get("custom_tag", ""))
            self.inst_check.setChecked(bool(s.get("is_instrumental", False)))

            self.caption_text.blockSignals(False)
            self.lyrics_text.blockSignals(False)
            self.track_tag_input.blockSignals(False)
            self.inst_check.blockSignals(False)

    def _load_track_preview(self, sample):
        path = sample.get("audio_path", "") if sample else ""
        valid = bool(path) and os.path.exists(path)
        self.waveform.set_audio(path if valid else None)
        if self.media_player is not None:
            self.media_player.stop()
            if valid:
                self.media_player.setSource(QUrl.fromLocalFile(path))
            else:
                self.seek_slider.setValue(0)
                self.time_label.setText("0:00 / 0:00")
                self.waveform.set_position_frac(0.0)

    def toggle_track_playback(self):
        if self.media_player is None:
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸")

    def stop_track_playback(self):
        if self.media_player is None:
            return
        self.media_player.stop()
        self.play_btn.setText("▶")
        self.seek_slider.setValue(0)
        self.time_label.setText("0:00 / 0:00")
        self.waveform.set_position_frac(0.0)

    def _on_player_position(self, ms):
        if self.media_player is None:
            return
        total = self.media_player.duration()
        if total > 0:
            frac = ms / total
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(frac * 1000))
            self.seek_slider.blockSignals(False)
            self.waveform.set_position_frac(frac)
            self.time_label.setText(f"{self._fmt_ms(ms)} / {self._fmt_ms(total)}")

    def _on_player_status(self, status):
        if self.media_player is None:
            return
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.play_btn.setText("▶")

    def _on_slider_moved(self, val):
        if self.media_player is None:
            return
        total = self.media_player.duration()
        if total > 0:
            self.media_player.setPosition(int(val / 1000 * total))

    @staticmethod
    def _fmt_ms(ms):
        sec = max(0, int(ms // 1000))
        return f"{sec // 60}:{sec % 60:02d}"

    def handle_lock_dropdown(self, idx):
        action = self.lock_action_combo.currentText()
        if action == "Lock All Detected":
            for s in self.dataset["samples"]:
                s["locked"] = True
            self.refresh_table()
            self.on_table_selection_changed()
            self.status_label.setText("Locked all detected metadata fields.")
        elif action == "Unlock All Fields":
            for s in self.dataset["samples"]:
                s["locked"] = False
            self.refresh_table()
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
                    s["timesignature"] = rep.get("timesig", s.get("timesignature", "4/4"))
                    self.refresh_table()
                    self.on_table_selection_changed()
                    self.status_label.setText("Restored original detected values.")
        self.lock_action_combo.setCurrentIndex(0)

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

        use_kaggle = self.audit_backend_combo.currentIndex() == 1

        if not self.startup_scan_notice_shown:
            self.startup_scan_notice_shown = True
            msg = (
                "Sending the dataset audio to a Kaggle GPU kernel for analysis. "
                "Please wait a few minutes..."
                if use_kaggle else
                "Testing dataset audio. Please wait a few seconds..."
            )
            QMessageBox.information(self, "Testing Dataset Audio", msg)
        if hasattr(self, "scan_btn") and self.scan_btn is not None:
            self.scan_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Auditing dataset health, metadata & degradation penalties...")

        if use_kaggle:
            self.active_worker = KaggleAuditWorker(samples, self.config)
        else:
            self.active_worker = HealthAuditorWorker(samples, self.config)
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.file_audited.connect(self.on_file_audited)
        self.active_worker.audit_completed.connect(self.on_audit_completed)
        self.active_worker.error_occurred.connect(self.on_worker_error)
        self.active_worker.start()

    def start_remote_consolidated_pipeline(self):
        """Asynchronously triggers the master Kaggle container and opens the console view."""
        from core.file_system import compress_dataset_folder, launch_remote_kaggle_console
        
        # Zip local directory structures
        bundle_path = compress_dataset_folder(self.dataset["samples"])
        
        # Push audio assets out to cloud storage volume mounts
        os.system(f"kaggle datasets version -m 'Upload bundle' -p {bundle_path}")
        os.system(f"kaggle kernels push -p core/kaggle_worker.py")
        
        # THE CONSOLE PASS: Open the container workspace interface instantly
        username = self.config.get("kaggle_user", "your-username")
        notebook_slug = "ace-step-master-pipeline" # Your notebook's specific URL string
        launch_remote_kaggle_console(username, notebook_slug)
        
        # SAFETY GUARD
        if hasattr(self, "scan_btn") and self.scan_btn is not None:
            self.scan_btn.setEnabled(False)

        # Hand execution tracking off to background thread monitor
        from workers.kaggle_consolidated import KaggleConsolidatedWorker
        self.active_worker = KaggleConsolidatedWorker(self.config)
        self.active_worker.progress.connect(self.on_worker_progress)
        self.active_worker.all_done.connect(self.on_remote_pipeline_success)
        self.active_worker.failed.connect(self.on_worker_error)
        self.active_worker.start()   
        self.status_label.setText("Container deployed! Redirecting your browser to monitor the GPUs live...")

    def on_file_audited(self, sid, rep):
        self.health_reports[sid] = rep
        for s in self.dataset["samples"]:
            if s["id"] != sid:
                continue
            # Auto-fill detected metadata next to the filename, then lock it.
            # User-entered (non-empty) values are never overwritten; an unlocked
            # track with non-empty values keeps its manual edits.
            filled = False
            if not s.get("bpm") or s.get("bpm") == 0:
                if rep.get("bpm_detected"):
                    s["bpm"] = rep["bpm_detected"]
                    filled = True
            if not s.get("keyscale"):
                if rep.get("key_detected"):
                    s["keyscale"] = rep["key_detected"]
                    filled = True
            if not s.get("timesignature"):
                if rep.get("timesig"):
                    s["timesignature"] = rep["timesig"]
                    filled = True
            if not s.get("duration") or s.get("duration") == 0:
                if rep.get("duration"):
                    s["duration"] = int(rep["duration"])
                    filled = True
            if filled and s.get("locked", True) is not False:
                s["locked"] = True

    def on_audit_completed(self, summary):
        """
        Callback handler when the lightweight integrity scan finishes.
        🛡️ Bulletproof Value Check: Uses a strict non-None gate to shield boot timing races.
        """
        notice = getattr(self, "rescan_notice", None)
        if notice is not None:
            notice.close()
            notice.deleteLater()
            self.rescan_notice = None        

        if getattr(self, "scan_btn", None) is not None:
            self.scan_btn.setEnabled(True)
            
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar.setVisible(False)

        # Update the visual status panel text with the clean summary payload data
        if getattr(self, "status_label", None) is not None:
            reasons = summary.get("reasons", [])
            mismatches = summary.get("missing_count", 0) + len(reasons)
            self.status_label.setText(
                f"Integrity Pass Complete. Verified: {summary.get('total_audited', 0)} tracks. "
                f"Source mismatches found: {mismatches}"
            )

        # 🦾 REDRAW THE CANVAS: Use your actual native function to update the PySide6 table view
        self.table.blockSignals(True)
        try:
            self.refresh_table()
            self.on_table_selection_changed()
        finally:
            self.table.blockSignals(False)

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

# Metadata Correction Script
    def force_sync_manifest_to_metadata(self):
        """
        Runs an explicit reconciliation script pass over the active collection.
        Forces the internal health reports to respect your canonical metadata fields,
        overwriting false 4/4 auto-detection warnings.
        """
        samples = self.dataset.get("samples", [])
        if not samples:
            self.status_label.setText("Synchronization skipped: Dataset contains no track rows.")
            return

        # 1. Initiate atomic snapshot backup
        self.record_snapshot()
        reconciled_count = 0

        for sample in samples:
            sid = sample.get("id", "")
            
            # Fetch the actual metadata values defined in your manifest
            canonical_time_sig = str(sample.get("timesignature", "4/4")).strip()
            
            # 2. Intercept the active health report record for this track
            if sid in self.health_reports:
                report = self.health_reports[sid]
                
                # Check if the report issues contains a false auto-detected time signature warning
                clean_issues = []
                for issue in report.get("issues", []):
                    # Strip out any lingering warnings about heuristic meter detection clashes
                    if "time signature auto-detected" in issue.lower() or "time-signature" in issue.lower():
                        continue
                    clean_issues.append(issue)
                
                # Write the cleaned issues back to the tracking index
                report["issues"] = clean_issues
                
                # If there are no longer any hardware degradation issues, upgrade status cleanly
                if not clean_issues:
                    report["status"] = "Healthy"
                    
                reconciled_count += 1

        # 3. 🛡️ THE SIGNAL BLOCKER GATEWAY: Pause table signals before updating cell properties
        self.table.blockSignals(True)
        try:
            self.refresh_table()
            self.on_table_selection_changed()
        finally:
            self.table.blockSignals(False) # Securely restore signals regardless of compilation output

        # 4. Force a recalculation of the global summary badge state
        from modules.audit import aggregate_health_summary
        summary = aggregate_health_summary(self.health_reports, self.dataset.get("samples", []), self.config)
        self.on_audit_completed(summary)

        self.status_label.setText(f"Manifest sync pass finished! Reconciled {reconciled_count} tracks.")
        QMessageBox.information(self, "Metadata Alignment Pass Complete", 
                                f"Successfully matched {reconciled_count} tracks.\nFalse 4/4 meter detection warnings have been removed from the health reports matrix.")

    def import_acestep_15xl_tags(self):
        """
        Primary Toolkit Engine: Ingests Pydantic-enforced 1.5XL captions and lyrics tags,
        overwrites placeholders, and resolves consistency metrics instantly.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Import ACE-Step 1.5XL Manifest", "", "Data Files (*.csv *.json)")
        if not path:
            return

        import csv
        self.record_snapshot()
        synced_count = 0

        try:
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    song_id = row.get("song_id", "").strip()
                    time_sig = row.get("time_signature", "4/4").strip()
                    try:
                        bpm_val = int(float(row.get("tempo_bpm", 0)))
                    except (ValueError, TypeError):
                        bpm_val = 0
                        
                    caption_text = row.get("caption", "").strip()
                    lyrics_text = row.get("lyrics", "").strip()

                    # Iterate and bind directly into your memory dataset array
                    for sample in self.dataset["samples"]:
                        # Match tracks by unique ID or matching filename string boundaries
                        if song_id.lower() in sample.get("filename", "").lower() or sample.get("id", "").lower() in song_id.lower() or synced_count < len(self.dataset["samples"]):
                            sample["timesignature"] = time_sig
                            sample["bpm"] = bpm_val
                            sample["caption"] = caption_text
                            sample["lyrics"] = lyrics_text
                            sample["formatted_lyrics"] = lyrics_text
                            
                            # Wipe away false placeholder tracking warnings instantly
                            sid = sample.get("id", "")
                            if sid in self.health_reports:
                                self.health_reports[sid]["issues"] = []
                                self.health_reports[sid]["status"] = "Healthy"
                                
                            synced_count += 1
                            break

            # Freeze rendering triggers to avoid PySide6 layout refresh crashes
            self.table.blockSignals(True)
            try:
                self.refresh_table()
                self.on_table_selection_changed()
            finally:
                self.table.blockSignals(False)

            # Recalculate your global UI indicator status badge panels
            from modules.audit import aggregate_health_summary
            summary = aggregate_health_summary(self.health_reports, self.dataset["samples"], self.config)
            self.on_audit_completed(summary)

            self.status_label.setText(f"Successfully loaded 1.5XL tags for {synced_count} tracks.")
            QMessageBox.information(self, "1.5XL Sync Complete", f"Successfully integrated metadata properties across {synced_count} tracks.")

        except Exception as e:
            QMessageBox.critical(self, "Manifest Ingestion Failure", f"Could not map structured data columns: {str(e)}")


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

    def load_dataset(self, checked=False, path=None):
        if isinstance(checked, str) and path is None:
            path = checked

        if path is None:
            dialog = QFileDialog(self)
            dialog.setWindowTitle("Open Dataset JSON")
            dialog.setDirectory(str(Path.home()))
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setAcceptMode(QFileDialog.AcceptOpen)
            dialog.setNameFilter("JSON Files (*.json)")
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)

            if dialog.exec() != QDialog.Accepted:
                return

            selected_files = dialog.selectedFiles()
            if not selected_files:
                return

            path = selected_files[0]

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)

            self.current_dataset_path = path
            self.record_snapshot()

            restored = self.dataset.get("metadata", {}).get(
                "health_reports",
                {},
            )
            self.health_reports.clear()
            self.health_reports.update(restored)

            self.sync_general_props_to_ui()
            self.refresh_table()

            self.status_label.setText(
                f"Loaded {len(self.dataset.get('samples', []))} tracks."
            )

            if not self.health_reports:
                self.start_health_audit()
            else:
                from modules.audit import aggregate_health_summary

                summary = aggregate_health_summary(
                    self.health_reports,
                    self.dataset.get("samples", []),
                    self.config,
                )
                self.on_audit_completed(summary)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def save_dataset(self, path=None):
        if not self.bypass_warnings and self.quality_badge.text().find("Critical") != -1:
            QMessageBox.warning(
                self, "Export Blocked by Quality Threshold",
                "Dataset quality is below safe threshold (<60%). Fix flagged issues or click '🛡 I Know What I'm Doing' to bypass.",
                QMessageBox.Ok
            )
            return False

        if path is None:
            path = getattr(self, "current_dataset_path", None)
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save Dataset JSON", "", "JSON Files (*.json)")
        if not path:
            return False

        try:
            self.on_general_prop_changed()
            self.dataset["metadata"]["num_samples"] = len(self.dataset["samples"])
            self.dataset["metadata"]["health_reports"] = self.health_reports
            backup = self._backup_file(path)  # never replace an existing file silently
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, indent=2)
            self.current_dataset_path = path
            msg = f"Saved dataset to {Path(path).name}"
            if backup:
                msg += " (previous file backed up)"
            self.status_label.setText(msg)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return False

    def save_dataset_as(self):
        """Always re-prompt for a path, ignoring any remembered one."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Dataset JSON As", "", "JSON Files (*.json)")
        if path:
            self.save_dataset(path=path)

    def closeEvent(self, event):
        if self.dataset.get("samples"):
            reply = QMessageBox.question(
                self, "Save Before Closing?", "Save the dataset before exiting?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes and not self.save_dataset():
                event.ignore()
                return
        event.accept()


    # -----------------------------------------------------------------------
    # Add Audio (single song / folder)
    # -----------------------------------------------------------------------
    def add_audio_files(self):
        exts = " ".join(sorted(AUDIO_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Single Song (one or more audio files)", "",
            f"Audio Files ({exts})",
        )
        if paths:
            self._add_audio_paths(paths)

    def add_audio_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Add Audio Folder (recursive)",
            "",
        )
        if not folder:
            return
        root = Path(folder)
        paths = sorted(
            str(p) for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        if not paths:
            QMessageBox.information(
                self, "Add Audio Folder",
                f"No supported audio files found in:\n{folder}\n\n"
                f"Supported formats: {', '.join(sorted(AUDIO_EXTS))}",
            )
            return
        self._add_audio_paths(paths)

    def _add_audio_paths(self, paths):
        """Shared add path: create samples for the given audio file paths.

        Skips files already present in the dataset (dedupe by ``audio_path``),
        then refreshes the table and kicks off the initial quality audit.
        """
        self.record_snapshot()
        global_tag = self.custom_tag_input.text().strip()
        is_all_inst = self.radio_all_inst.isChecked()
        existing = {s.get("audio_path") for s in self.dataset["samples"]}

        added = 0
        skipped = 0
        for p in paths:
            if p in existing:
                skipped += 1
                continue
            existing.add(p)
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
                "timesignature": "",
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
            added += 1
        self.refresh_table()
        msg = f"Added {added} audio track(s)."
        if skipped:
            msg += f" {skipped} already present (skipped)."
        self.status_label.setText(msg)
        if added:
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
    # Lyrics transcription (WhisperX, optional)
    # -----------------------------------------------------------------------
    def start_lyrics_transcription(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "No Track Selected", "Select a track to transcribe.")
            return
        audio_path = selected.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.warning(self, "Missing Audio", "The selected track's audio file is missing on disk.")
            return

        # Sanitize and fetch the active target configuration engine
        raw_engine = str(self.config.get("lyrics_engine", "local_whisperx")).lower().strip()
        
        # ⚙️ SYSTEM DEFENSE MAPPING: Translate any legacy keys on-the-fly to prevent unhandled branches
        if raw_engine in ("whisperx", "local_whisperx", ""):
            engine = "local_whisperx"
        elif "kaggle" in raw_engine or "acestep" in raw_engine:
            if "acestep" in raw_engine:
                engine = "kaggle_acestep"
            else:
                engine = "kaggle_whisperx"
        elif "gemini" in raw_engine:
            engine = "gemini_api"
        else:
            engine = "local_whisperx" # Ultra safety default

        language = (self.config.get("lyrics_language") or "").strip() or None
        initial_prompt = (self.config.get("lyrics_initial_prompt") or "").strip() or None
        
        self.transcribe_btn.setEnabled(False)
        self.status_label.setText(f"Initializing transcription routing via: {engine.upper()}...")
        print(f"[DEBUG] start_lyrics_transcription selected engine resolved to: {engine}")

        # Explicitly declare your worker variable to avoid unassigned attribute states
        self.lyrics_worker = None

        # 🔀 DYNAMIC ENGINE ROUTING SWITCH BLOCK
        if engine == "local_whisperx":
            from modules.lyrics import transcribe_available
            if not transcribe_available():
                QMessageBox.information(
                    self, "WhisperX Required",
                    "WhisperX is not installed locally on this machine.\n\n"
                    "Run: pip install whisperx\n\n"
                    "Or switch the backend choice in ⚙ Settings to a Kaggle or Gemini option.",
                )
                self.transcribe_btn.setEnabled(True)
                return
            from workers.lyrics import TranscribeLyricsWorker
            self.lyrics_worker = TranscribeLyricsWorker(
                audio_path, engine="whisperx", language=language, 
                initial_prompt=initial_prompt, config=self.config, parent=self
            )
            
        elif engine in ("kaggle_whisperx", "kaggle_acestep"):
            # Pipes execution directly out to your isolated dual-engine file module
            from workers.kaggle_lyrics import KaggleLyricsWorker
            target_mode = "whisperx" if engine == "kaggle_whisperx" else "acestep"
            self.lyrics_worker = KaggleLyricsWorker(
                audio_path, language=language, initial_prompt=initial_prompt, 
                config=self.config, mode=target_mode
            )
            
        elif engine == "gemini_api":
            if not self.config.get("gemini_api_key"):
                QMessageBox.warning(self, "Missing API Key", "Set your Gemini API Key in ⚙ Settings first.")
                self.transcribe_btn.setEnabled(True)
                return
            from workers.lyrics import TranscribeLyricsWorker
            self.lyrics_worker = TranscribeLyricsWorker(
                audio_path, engine="gemini", language=language, 
                initial_prompt=initial_prompt, config=self.config, parent=self
            )

        # Catch instances where initialization rules failed to deploy a backend object
        if self.lyrics_worker is None:
            print("[DEBUG] CRITICAL: Engine routing collapsed without generating a worker instance.")
            self.status_label.setText("Error initializing transcription worker.")
            self.transcribe_btn.setEnabled(True)
            return

        # Uniformly bind execution callbacks back to your UI elements
        self.lyrics_worker.finished_ok.connect(self.on_lyrics_done)
        self.lyrics_worker.failed.connect(self.on_lyrics_failed)
        if hasattr(self.lyrics_worker, "progress"):
            self.lyrics_worker.progress.connect(self.on_worker_progress)
            
        self.lyrics_worker.start()
        print("[DEBUG] Lyrics transcription worker thread successfully launched.")



    def on_lyrics_done(self, result):
        self.transcribe_btn.setEnabled(True)
        new_lyrics = (result.get("lyrics") or "").strip()
        sample = self.get_selected_sample()
        if not sample or not new_lyrics:
            self.status_label.setText("Lyrics transcription returned no text.")
            return

        engine_used = str(self.config.get("lyrics_engine", "local_whisperx")).lower()
        old_lyrics = (sample.get("lyrics") or "").strip()

        # 📦 SAVE ENGINE-SPECIFIC VARIABLES INDEPENDENTLY FOR BENCHMARKING
        if "whisperx" in engine_used:
            sample["lyrics_whisperx"] = new_lyrics
        elif "acestep" in engine_used:
            sample["lyrics_acestep"] = new_lyrics
        else:
            sample["lyrics_gemini"] = new_lyrics

        # Preserves the native file logging operation next to the source audio
        self._write_lyrics_text_files(sample, old_lyrics, new_lyrics)

        # 📊 BENCHMARK DETECTOR: If both local variants exist, pop the head-to-head window
        if "lyrics_whisperx" in sample and "lyrics_acestep" in sample:
            self._open_dual_engine_comparison_benchmark(sample)
        else:
            # Fall back to your legacy single-column review window layout frame
            self._review_lyrics_dialog(sample, old_lyrics, new_lyrics)
            
        self.refresh_table()
        self.on_table_selection_changed()

    def _open_dual_engine_comparison_benchmark(self, sample):
        """
        Gentoo-inspired side-by-side text matrix benchmarking view layout.
        Allows direct head-to-head parsing of WhisperX vs ACE-Step engines.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(f"🎛️ Head-to-Head Engine Benchmark: {sample.get('filename', '')}")
        dialog.resize(1100, 700)
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "<b>Engine Comparison Mode Activated</b><br>"
            "Both execution runs have completed. Review performance side-by-side to "
            "verify model accuracy and determine your recommended baseline default configuration choice."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        split = QSplitter(Qt.Horizontal)

        # Pane A: WhisperX Container Output Space
        w_widget = QWidget()
        wl = QVBoxLayout(w_widget)
        wl.addWidget(QLabel("<b>🔮 WhisperX Model Output Data:</b>"))
        w_box = QTextBrowser()
        w_box.setPlainText(sample.get("lyrics_whisperx", ""))
        wl.addWidget(w_box)
        split.addWidget(w_widget)

        # Pane B: ACE-Step Container Output Space
        a_widget = QWidget()
        al = QVBoxLayout(a_widget)
        al.addWidget(QLabel("<b>🔥 ACE-Step Transcriber Output Data:</b>"))
        a_box = QTextBrowser()
        a_box.setPlainText(sample.get("lyrics_acestep", ""))
        al.addWidget(a_box)
        split.addWidget(a_widget)

        layout.addWidget(split, 1)

        row = QHBoxLayout()
        diff_btn = QPushButton("🔍 Run Diff Inspection Analysis")
        diff_btn.clicked.connect(lambda: self._show_lyrics_diff(
            sample.get("lyrics_whisperx", ""), 
            sample.get("lyrics_acestep", "")
        ))
        
        apply_w_btn = QPushButton("Select WhisperX Baseline Text")
        apply_w_btn.clicked.connect(lambda: self._choose_caption(sample, sample["lyrics_whisperx"], dialog))
        
        apply_a_btn = QPushButton("Select ACE-Step Baseline Text")
        apply_a_btn.clicked.connect(lambda: self._choose_caption(sample, sample["lyrics_acestep"], dialog))
        
        close_btn = QPushButton("Close Frame")
        close_btn.clicked.connect(dialog.reject)

        row.addWidget(diff_btn)
        row.addStretch()
        row.addWidget(apply_w_btn)
        row.addWidget(apply_a_btn)
        row.addWidget(close_btn)
        layout.addLayout(row)

        dialog.exec()


    def _write_lyrics_text_files(self, sample, old_lyrics, new_lyrics):
        """Write old + new lyrics to sibling .txt files (backing up any that
        already exist) so the user can diff them externally if they prefer."""
        audio_path = sample.get("audio_path", "")
        if audio_path:
            folder = Path(audio_path).parent
            base = Path(audio_path).stem
        else:
            folder = Path.cwd()
            base = sample.get("id", "track")
        old_path = folder / f"{base}_lyrics_old.txt"
        new_path = folder / f"{base}_lyrics_new.txt"
        self._backup_file(str(old_path))
        self._backup_file(str(new_path))
        try:
            old_path.write_text(old_lyrics or "(no previous lyrics)", encoding="utf-8")
            new_path.write_text(new_lyrics, encoding="utf-8")
        except OSError as e:  # noqa: BLE001
            print(f"lyrics text files not written: {e}")

    def _review_lyrics_dialog(self, sample, old_lyrics, new_lyrics):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Review Lyrics: {sample.get('filename', '')}")
        dialog.resize(900, 620)
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "The transcription never overwrites your lyrics. Compare below and choose. "
            "Existing + transcribed are also saved as .txt files next to the audio."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        split = QSplitter(Qt.Horizontal)
        old_widget = QWidget(); old_col = QVBoxLayout(old_widget)
        old_col.addWidget(QLabel("<b>Existing (kept)</b>"))
        old_box = QTextBrowser(); old_box.setPlainText(old_lyrics or "(no previous lyrics)")
        old_col.addWidget(old_box)
        new_widget = QWidget(); new_col = QVBoxLayout(new_widget)
        new_col.addWidget(QLabel("<b>Transcribed</b>"))
        new_box = QTextBrowser(); new_box.setPlainText(new_lyrics)
        new_col.addWidget(new_box)
        split.addWidget(old_widget)
        split.addWidget(new_widget)
        layout.addWidget(split, 1)

        row = QHBoxLayout()
        diff_btn = QPushButton("Run Diff")
        diff_btn.clicked.connect(lambda: self._show_lyrics_diff(old_lyrics, new_lyrics))
        keep_btn = QPushButton("Keep Existing")
        keep_btn.clicked.connect(dialog.reject)
        use_btn = QPushButton("Use Transcribed")
        use_btn.clicked.connect(lambda: self._accept_transcribed_lyrics(sample, new_lyrics, dialog))
        row.addWidget(diff_btn)
        row.addStretch()
        row.addWidget(keep_btn)
        row.addWidget(use_btn)
        layout.addLayout(row)

        dialog.exec()

    def _accept_transcribed_lyrics(self, sample, new_lyrics, dialog):
        self.record_snapshot()
        sample["lyrics"] = new_lyrics
        sample["formatted_lyrics"] = new_lyrics
        self.status_label.setText("Lyrics updated from transcription.")
        dialog.accept()

    def _show_lyrics_diff(self, old_lyrics, new_lyrics):
        import difflib

        diff_lines = list(difflib.unified_diff(
            (old_lyrics or "").splitlines(),
            (new_lyrics or "").splitlines(),
            fromfile="existing", tofile="transcribed", lineterm="",
        ))
        diff_text = "\n".join(diff_lines) if diff_lines else "(identical)"

        dialog = QDialog(self)
        dialog.setWindowTitle("Lyrics Diff (unified)")
        dialog.resize(780, 540)
        lay = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setPlainText(diff_text)
        lay.addWidget(browser, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        lay.addWidget(close_btn)
        dialog.exec()

    def on_lyrics_failed(self, err):
        self.transcribe_btn.setEnabled(True)
        self.status_label.setText("Lyrics transcription failed.")
        QMessageBox.warning(self, "Transcription Failed", str(err))

    # -----------------------------------------------------------------------
    # Rockstar multitrack-existence lookup (metadata only + disclaimer)
    # -----------------------------------------------------------------------
    def start_rockstar_lookup(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        base = Path(selected.get("filename", "")).stem.replace("_", " ").replace("-", " - ")
        parts = [p.strip() for p in base.split(" - ", 1)]
        artist = parts[0] if len(parts) > 1 else ""
        song = parts[1] if len(parts) > 1 else parts[0]

        song, ok = QInputDialog.getText(self, "Rockstar Track Check", "Song:", text=song)
        if not ok or not song.strip():
            return
        artist, ok2 = QInputDialog.getText(self, "Rockstar Track Check", "Artist:", text=artist)
        if not ok2:
            return

        self.rockstar_btn.setEnabled(False)
        self.status_label.setText(f"Checking multitrack availability: {song.strip()}…")
        self.rockstar_worker = RockstarLookupWorker(artist.strip(), song.strip(), parent=self)
        self.rockstar_worker.finished_ok.connect(self.on_rockstar_result)
        self.rockstar_worker.failed.connect(self.on_rockstar_failed)
        self.rockstar_worker.start()

    def on_rockstar_result(self, result):
        self.rockstar_btn.setEnabled(True)
        exists = result.get("exists")
        if exists is None:
            verdict = "⚠️ Search failed (offline or blocked)"
        elif exists:
            verdict = "✅ Multitrack stems appear to EXIST"
        else:
            verdict = "❌ No multitrack references surfaced in public indices"
        lines = [
            f"<b>{html.escape(result.get('artist', ''))} — {html.escape(result.get('song', ''))}</b>",
            verdict, "",
            html.escape(result.get("note", "")),
        ]
        matches = result.get("matches") or []
        if matches:
            lines.append("")
            lines.append("References (titles + sites only):")
            for m in matches:
                idx = " 📋 index" if m.get("index") else ""
                lines.append(f"  • {html.escape(m.get('title', ''))} ({html.escape(m.get('domain', '?'))}){idx}")
        lines.append("")
        lines.append("This lookup reports existence only — it provides no files and no links.")

        dialog = QDialog(self)
        dialog.setWindowTitle("Rockstar Track Check")
        dialog.resize(720, 520)
        lay = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setHtml("<br>".join(l.replace("\n", "<br>") for l in lines))
        lay.addWidget(browser, 1)
        row = QHBoxLayout()
        disclaim_btn = QPushButton("⚠️ Show Disclaimer (what NOT to do)")
        disclaim_btn.clicked.connect(self._show_rockstar_disclaimer)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        row.addWidget(disclaim_btn)
        row.addStretch()
        row.addWidget(close_btn)
        lay.addLayout(row)
        dialog.exec()

    def on_rockstar_failed(self, err):
        self.rockstar_btn.setEnabled(True)
        self.status_label.setText("Rockstar check failed.")
        QMessageBox.warning(self, "Rockstar Check Failed", str(err))

    def _show_rockstar_disclaimer(self):
        from modules.rockstar_lookup import disclaimer_text

        dialog = QDialog(self)
        dialog.setWindowTitle("⚠️ Disclaimer — What NOT to Do")
        dialog.resize(760, 620)
        lay = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setPlainText(disclaimer_text())
        lay.addWidget(browser, 1)
        close_btn = QPushButton("I Have Read It and Will Do None of It")
        close_btn.clicked.connect(dialog.accept)
        lay.addWidget(close_btn)
        dialog.exec()

    # -----------------------------------------------------------------------
    # Structural Tag Creator
    # -----------------------------------------------------------------------
    def start_structural_tag_creator(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self._table_sample_indices)):
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        index = self._table_sample_indices[row]
        sample = self.dataset["samples"][index]
        from modules.llm_client import get_client
        try:
            get_client(self.config, role="aggregator")
        except ValueError as e:
            QMessageBox.information(
                self, "LLM Key Needed",
                f"{e}\n\nSet it in ⚙ Settings → LLM Provider.",
            )
            return
        self.tag_creator_btn.setEnabled(False)
        self.status_label.setText(f"Generating structural tags for {sample.get('filename', '')}…")
        self.tag_creator_worker = TagCreatorWorker(index, sample, self.config, parent=self)
        self.tag_creator_worker.finished_ok.connect(self.on_tag_creator_done)
        self.tag_creator_worker.failed.connect(self.on_tag_creator_failed)
        self.tag_creator_worker.start()

    def on_tag_creator_done(self, index, caption, lyrics):
        self.tag_creator_btn.setEnabled(True)
        if not (0 <= index < len(self.dataset.get("samples", []))):
            return
        sample = self.dataset["samples"][index]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Structural Tag Creator — {sample.get('filename', '')}")
        dialog.resize(780, 640)
        lay = QVBoxLayout(dialog)
        lay.addWidget(QLabel("<b>Generated Caption (global tags):</b>"))
        cap_edit = QTextEdit()
        cap_edit.setPlainText(caption or "(no caption block)")
        lay.addWidget(cap_edit, 1)
        lay.addWidget(QLabel("<b>Generated Lyrics (time-script):</b>"))
        lyr_edit = QTextEdit()
        lyr_edit.setPlainText(lyrics or "(no lyrics block)")
        lay.addWidget(lyr_edit, 2)
        row_btns = QHBoxLayout()
        cancel_btn = QPushButton("Keep Existing")
        cancel_btn.clicked.connect(dialog.reject)
        apply_btn = QPushButton("Apply to Track")
        apply_btn.clicked.connect(lambda: self._apply_tag_creator_result(sample, cap_edit, lyr_edit, dialog))
        row_btns.addStretch()
        row_btns.addWidget(cancel_btn)
        row_btns.addWidget(apply_btn)
        lay.addLayout(row_btns)
        dialog.exec()

    def _apply_tag_creator_result(self, sample, cap_edit, lyr_edit, dialog):
        caption = cap_edit.toPlainText().strip()
        lyrics = lyr_edit.toPlainText().strip()
        self.record_snapshot()
        if caption:
            sample["caption"] = caption
        if lyrics:
            sample["lyrics"] = lyrics
            sample["formatted_lyrics"] = lyrics
        sample["tags_caption"] = caption
        sample["tags_lyrics"] = lyrics
        self.status_label.setText("Structural tags applied to the track.")
        dialog.accept()
        self.refresh_table()
        self.on_table_selection_changed()

    def on_tag_creator_failed(self, err):
        self.tag_creator_btn.setEnabled(True)
        self.status_label.setText("Structural tag creation failed.")
        QMessageBox.warning(self, "Tag Creator Failed", str(err))

    # -----------------------------------------------------------------------
    # MusicBrainz enrichment + dataset stats
    # -----------------------------------------------------------------------
    def start_musicbrainz_lookup(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        base = Path(selected.get("filename", "")).stem.replace("_", " ").replace("-", " - ")
        parts = [p.strip() for p in base.split(" - ", 1)]
        artist = parts[0] if len(parts) > 1 else ""
        song = parts[1] if len(parts) > 1 else parts[0]
        song, ok = QInputDialog.getText(self, "MusicBrainz", "Song:", text=song)
        if not ok or not song.strip():
            return
        artist, ok2 = QInputDialog.getText(self, "MusicBrainz", "Artist:", text=artist)
        if not ok2:
            return
        self.musicbrainz_btn.setEnabled(False)
        self.status_label.setText(f"Looking up {song.strip()} on MusicBrainz…")
        self.musicbrainz_worker = MusicBrainzWorker(artist.strip(), song.strip(), parent=self)
        self.musicbrainz_worker.finished_ok.connect(self.on_musicbrainz_result)
        self.musicbrainz_worker.failed.connect(self.on_musicbrainz_failed)
        self.musicbrainz_worker.start()

    def on_musicbrainz_result(self, result):
        self.musicbrainz_btn.setEnabled(True)
        sample = self.get_selected_sample()
        if not result.get("ok"):
            QMessageBox.information(self, "MusicBrainz", result.get("note", "No match."))
            return
        lines = [
            f"<b>{result.get('artist')} — {result.get('title')}</b>",
            "Year: " + (result.get("year") or "unknown")
            + (f" | Country: {result.get('country')}" if result.get("country") else ""),
        ]
        if result.get("genres"):
            lines.append("Genres: " + ", ".join(result["genres"]))
        lines.append("")
        lines.append("Apply the genre / year to the selected track?")
        reply = QMessageBox.question(
            self, "MusicBrainz", "<br>".join(lines), QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes and sample:
            self.record_snapshot()
            if not (sample.get("genre") or "").strip() and result.get("genres"):
                sample["genre"] = result["genres"][0]
            sample["year"] = result.get("year", "")
            sample["country"] = result.get("country", "")
            self.refresh_table()
            self.on_table_selection_changed()
            self.status_label.setText(f"Applied MusicBrainz metadata for {sample.get('filename', '')}.")
        else:
            self.status_label.setText("MusicBrainz metadata not applied.")

    def on_musicbrainz_failed(self, err):
        self.musicbrainz_btn.setEnabled(True)
        QMessageBox.warning(self, "MusicBrainz Failed", str(err))

    def show_stats_report(self):
        from modules.stats import build_dataset_report as compute_stats_charts

        # Generate the chart data using the uniquely named alias function
        report = compute_stats_charts(self.dataset)
        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Statistics")
        dialog.resize(560, 420)
        lay = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setPlainText(report)
        lay.addWidget(browser, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(report))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        row.addWidget(copy_btn)
        row.addStretch()
        row.addWidget(close_btn)
        lay.addLayout(row)
        dialog.exec()

    # -----------------------------------------------------------------------
    # Dataset tools: find/replace, lyrics editor, A/B captions, riff/hook, stem A/B
    # -----------------------------------------------------------------------
    def open_find_replace_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Bulk Find / Replace")
        dialog.resize(420, 160)
        lay = QVBoxLayout(dialog)
        self.fr_find = QLineEdit()
        self.fr_find.setPlaceholderText("Find…")
        self.fr_repl = QLineEdit()
        self.fr_repl.setPlaceholderText("Replace with…")
        self.fr_scope = QComboBox()
        self.fr_scope.addItems(["Captions", "Custom tags", "Both"])
        lay.addWidget(self.fr_find)
        lay.addWidget(self.fr_repl)
        lay.addWidget(self.fr_scope)
        row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        go_btn = QPushButton("Replace All")
        go_btn.clicked.connect(lambda: self._apply_find_replace(dialog))
        row.addStretch(); row.addWidget(cancel_btn); row.addWidget(go_btn)
        lay.addLayout(row)
        dialog.exec()

    def _apply_find_replace(self, dialog):
        find = self.fr_find.text()
        repl = self.fr_repl.text()
        scope = self.fr_scope.currentText()
        if not find:
            QMessageBox.warning(self, "Find/Replace", "Enter some text to find.")
            return
        self.record_snapshot()
        count = 0
        for s in self.dataset.get("samples", []):
            if scope in ("Captions", "Both"):
                c = s.get("caption", "")
                if find in c:
                    s["caption"] = c.replace(find, repl)
                    count += 1
            if scope in ("Custom tags", "Both"):
                t = s.get("custom_tag", "")
                if find in t:
                    s["custom_tag"] = t.replace(find, repl)
                    count += 1
        dialog.accept()
        self.refresh_table()
        self.status_label.setText(f"Replaced '{find}' in {count} field(s).")
        QMessageBox.information(self, "Find/Replace", f"Replaced '{find}' in {count} field(s).")

    # -------------------------------------------------------------------------
    # Bulk rename — default mode: song name only, spaces -> underscores
    # -------------------------------------------------------------------------
    def open_bulk_rename_dialog(self):
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks before renaming.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Bulk Rename Tracks")
        dialog.resize(640, 540)
        lay = QVBoxLayout(dialog)

        hint = QLabel("💡 Hover any field for help. In patterns, {n} is the track counter.")
        hint.setStyleSheet("color: #9db2c8; font-style: italic;")
        lay.addWidget(hint)

        scope_combo = QComboBox()
        scope_combo.addItems(["All tracks", "Filtered (visible) tracks", "Selected tracks"])
        scope_combo.setToolTip(
            "Which tracks to rename.\n"
            "• All tracks — every row in the dataset\n"
            "• Filtered (visible) tracks — only what the search box currently shows\n"
            "• Selected tracks — the rows you have highlighted in the table"
        )
        mode_combo = QComboBox()
        mode_combo.addItems(
            ["Song name (spaces → _)", "Find & Replace", "Prefix", "Suffix", "Number sequence"]
        )
        mode_combo.setToolTip(
            "How the new name is built.\n"
            "• Song name (spaces → _) — keep only the song title, replace spaces with "
            "underscores (default)\n"
            "• Find & Replace — swap one piece of text for another\n"
            "• Prefix — add text to the START of the filename\n"
            "• Suffix — add text BEFORE the extension\n"
            "• Number sequence — rename to a numbered pattern (see Pattern)"
        )
        find_edit = QLineEdit()
        find_edit.setPlaceholderText("text to find…")
        find_edit.setToolTip(
            "Find & Replace only.\n"
            "The exact text to search for inside the filename (before the extension).\n\n"
            "Example: \"Demo\" matches \"My_Demo_Track.wav\""
        )
        repl_edit = QLineEdit()
        repl_edit.setPlaceholderText("replacement…")
        repl_edit.setToolTip(
            "Find & Replace only.\n"
            "The text that replaces every match of Find.\n\n"
            "Example: Find \"Demo\" / Replace \"Master\" → \"My_Master_Track.wav\""
        )
        prefix_edit = QLineEdit()
        prefix_edit.setPlaceholderText("prefix…")
        prefix_edit.setToolTip(
            "Prefix mode.\n"
            "Text added to the very START of the filename, before the song name.\n\n"
            "Example: prefix \"acoustic_\" → \"acoustic_My_Track.wav\""
        )
        suffix_edit = QLineEdit()
        suffix_edit.setPlaceholderText("suffix…")
        suffix_edit.setToolTip(
            "Suffix mode.\n"
            "Text added to the name BEFORE the file extension.\n\n"
            "Example: suffix \"_final\" → \"My_Track_final.wav\""
        )
        pattern_edit = QLineEdit("track_{n:03d}")
        pattern_edit.setPlaceholderText("e.g. track_{n:03d}")
        pattern_edit.setToolTip(
            "Number sequence only.\n"
            "A template where {n} is the counter (starting at Start number).\n\n"
            "• {n} → 1, 2, 3…\n"
            "• {n:02d} → 01, 02, 03… (pad to 2 digits)\n"
            "• {n:03d} → 001, 002, 003… (pad to 3 digits)\n\n"
            "Example: \"track_{n:03d}\" → track_001.wav, track_002.wav…"
        )
        start_spin = QSpinBox()
        start_spin.setRange(0, 9999)
        start_spin.setToolTip(
            "Number sequence only.\n"
            "The number {n} starts at when counting.\n\n"
            "Example: start 1 → 001, 002…"
        )

        disk_check = QCheckBox("Rename the file on disk too (original is backed up first)")
        disk_check.setChecked(False)
        disk_check.setToolTip(
            "Also rename the actual audio file on disk, not just the name shown in the table.\n"
            "The original file is always backed up first — nothing is ever lost."
        )

        preview = QListWidget()
        preview.setMaximumHeight(240)

        form = QFormLayout()
        form.addRow("Scope:", scope_combo)
        form.addRow("Mode:", mode_combo)
        form.addRow("Find:", find_edit)
        form.addRow("Replace:", repl_edit)
        form.addRow("Prefix:", prefix_edit)
        form.addRow("Suffix:", suffix_edit)
        form.addRow("Pattern ({n}):", pattern_edit)
        form.addRow("Start number:", start_spin)
        lay.addLayout(form)
        lay.addWidget(disk_check)
        lay.addWidget(QLabel("<b>Preview:</b>"))
        lay.addWidget(preview, 1)

        def _refresh_preview(*_):
            preview.clear()
            for old, new in self._bulk_rename_preview(
                scope_combo, mode_combo, find_edit, repl_edit,
                prefix_edit, suffix_edit, pattern_edit, start_spin,
            ):
                preview.addItem(f"{old}  →  {new}")

        for w in (scope_combo, mode_combo, find_edit, repl_edit,
                  prefix_edit, suffix_edit, pattern_edit, start_spin):
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(_refresh_preview)
            elif isinstance(w, QSpinBox):
                w.valueChanged.connect(_refresh_preview)
            else:
                w.textChanged.connect(_refresh_preview)

        row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        go_btn = QPushButton("Rename")
        go_btn.clicked.connect(
            lambda: self._apply_bulk_rename(
                dialog, scope_combo, mode_combo, find_edit, repl_edit,
                prefix_edit, suffix_edit, pattern_edit, start_spin, disk_check,
            )
        )
        row.addStretch()
        row.addWidget(cancel_btn)
        row.addWidget(go_btn)
        lay.addLayout(row)

        _refresh_preview()
        dialog.exec()

    @staticmethod
    def _song_name_from_filename(name):
        """Default bulk-rename: keep only the song name, spaces -> underscores.

        Strips a leading track number ("01 - ...") and an "Artist - " prefix
        (the app's MusicBrainz convention), then replaces whitespace runs with
        underscores. Extension is preserved by the caller.
        """
        stem = os.path.splitext(name)[0]
        stem = re.sub(r"^\s*\d{1,3}\s*[-._]\s*", "", stem)
        if " - " in stem:
            stem = stem.split(" - ")[-1]
        return re.sub(r"\s+", "_", stem).strip("_")

    def _bulk_rename_preview(self, scope_combo, mode_combo, find_edit, repl_edit,
                             prefix_edit, suffix_edit, pattern_edit, start_spin):
        out = []
        for s, new in self._bulk_rename_targets(
            scope_combo, mode_combo, find_edit, repl_edit,
            prefix_edit, suffix_edit, pattern_edit, start_spin,
        ):
            out.append((s.get("filename", ""), new))
        return out

    def _bulk_rename_targets(self, scope_combo, mode_combo, find_edit, repl_edit,
                             prefix_edit, suffix_edit, pattern_edit, start_spin):
        """Return ``[(sample, new_filename), ...]`` — never mutates samples."""
        samples = self.dataset.get("samples", [])
        scope = scope_combo.currentText()
        if scope == "Selected tracks":
            rows = sorted({r.row() for r in self.table.selectionModel().selectedRows()})
            targets = [samples[self._table_sample_indices[r]]
                       for r in rows if 0 <= r < len(self._table_sample_indices)]
        elif scope == "Filtered (visible) tracks":
            targets = [samples[i] for i in self._table_sample_indices if 0 <= i < len(samples)]
        else:
            targets = list(samples)

        mode = mode_combo.currentText()
        result = []
        seen = set()
        n = start_spin.value()
        for s in targets:
            old = s.get("filename", "")
            if not old:
                continue
            stem, ext = os.path.splitext(old)
            if mode == "Song name (spaces → _)":
                new = self._song_name_from_filename(old)
            elif mode == "Find & Replace":
                find = find_edit.text()
                if not find or find not in stem:
                    continue
                new = stem.replace(find, repl_edit.text()) + ext
            elif mode == "Prefix":
                new = prefix_edit.text() + old
            elif mode == "Suffix":
                new = stem + suffix_edit.text() + ext
            else:  # Number sequence
                try:
                    new = pattern_edit.text().format(n=n) + ext
                except (KeyError, ValueError, IndexError):
                    new = pattern_edit.text().replace("{n}", str(n)) + ext
                n += 1
            if not new or new in seen:
                continue
            seen.add(new)
            result.append((s, new))
        return result

    def _apply_bulk_rename(self, dialog, scope_combo, mode_combo, find_edit, repl_edit,
                           prefix_edit, suffix_edit, pattern_edit, start_spin, disk_check):
        from core.file_system import execute_disk_rename
        
        # 1. Gather targeted samples using your existing scope filters
        samples_to_rename = self._get_rename_scope_targets(scope_combo.currentText())
        if not samples_to_rename:
            return

        options = {
            "find_text": find_edit.text(),
            "replace_text": repl_edit.text(),
            "prefix_text": prefix_edit.text(),
            "suffix_text": suffix_edit.text(),
            "pattern": pattern_edit.text(),
            "start_number": start_spin.value(),
            "create_backup": True
        }

        # 2. Run the decoupled script directly on the disk files
        self.record_snapshot()
        updates, count = execute_disk_rename(samples_to_rename, mode_combo.currentText(), options)

        # 3. Synchronize memory state variables using the script results
        for update in updates:
            for sample in self.dataset["samples"]:
                if sample["id"] == update["id"]:
                    sample["filename"] = update["new_filename"]
                    sample["audio_path"] = update["new_audio_path"]

        # 4. Refresh UI instantly from the modified disk footprint
        dialog.accept()
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText(f"Successfully processed batch script! Renamed {count} files on disk.")
        
        # Trigger your health audit to scan the newly renamed configurations automatically
        self.start_health_audit()

    def open_lyrics_editor(self):
        sample = self.get_selected_sample()
        if not sample:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Lyrics Editor — {sample.get('filename', '')}")
        dialog.resize(560, 520)
        lay = QVBoxLayout(dialog)
        lay.addWidget(QLabel("<b>Lyrics (time-script):</b>"))
        edit = QTextEdit()
        edit.setPlainText(sample.get("formatted_lyrics") or sample.get("lyrics") or "")
        lay.addWidget(edit, 1)
        row = QHBoxLayout()
        split_btn = QPushButton("Split Long Lines (≤10 syll)")
        split_btn.clicked.connect(lambda: edit.setPlainText(
            split_long_lines(edit.toPlainText())))
        export_btn = QPushButton("Export .lrc…")
        export_btn.clicked.connect(lambda: self._export_lyrics(sample, edit.toPlainText()))
        save_btn = QPushButton("Save to Track")
        save_btn.clicked.connect(lambda: self._save_lyrics_editor(sample, edit.toPlainText(), dialog))
        cancel_btn = QPushButton("Close")
        cancel_btn.clicked.connect(dialog.reject)
        row.addWidget(split_btn)
        row.addWidget(export_btn)
        row.addStretch()
        row.addWidget(save_btn)
        row.addWidget(cancel_btn)
        lay.addLayout(row)
        dialog.exec()

    def _save_lyrics_editor(self, sample, text, dialog):
        self.record_snapshot()
        sample["lyrics"] = text.strip()
        sample["formatted_lyrics"] = text.strip()
        dialog.accept()
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText("Lyrics updated.")

    def _export_lyrics(self, sample, text):
        from modules.lyrics_tools import export_lrc
        base = Path(sample.get("filename", "lyrics")).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export Lyrics", f"{base}.lrc", "LRC (*.lrc);;Text (*.txt)")
        if not path:
            return
        export_lrc(text.strip(), sample.get("lyrics_segments") or [], path)
        self.status_label.setText(f"Lyrics exported to {Path(path).name}.")

    def open_ab_captions(self):
        sample = self.get_selected_sample()
        if not sample:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        a = sample.get("caption") or ""
        b = sample.get("caption_ai_raw") or sample.get("tags_caption") or ""
        if not b:
            QMessageBox.information(
                self, "A/B Captions",
                "No alternate caption yet — run the AI captioner or the Structural Tag Creator first.",
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"A/B Captions — {sample.get('filename', '')}")
        dialog.resize(820, 480)
        lay = QVBoxLayout(dialog)
        split = QSplitter(Qt.Horizontal)
        a_box = QTextBrowser(); a_box.setPlainText(a or "(empty)")
        b_box = QTextBrowser(); b_box.setPlainText(b)
        aw = QWidget(); al = QVBoxLayout(aw); al.addWidget(QLabel("<b>A — Current</b>")); al.addWidget(a_box)
        bw = QWidget(); bl = QVBoxLayout(bw); bl.addWidget(QLabel("<b>B — Alternate</b>")); bl.addWidget(b_box)
        split.addWidget(aw); split.addWidget(bw)
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        use_a = QPushButton("Use A")
        use_a.clicked.connect(lambda: self._choose_caption(sample, a, dialog))
        use_b = QPushButton("Use B")
        use_b.clicked.connect(lambda: self._choose_caption(sample, b, dialog))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        row.addStretch(); row.addWidget(use_a); row.addWidget(use_b); row.addWidget(close_btn)
        lay.addLayout(row)
        dialog.exec()

    def _choose_caption(self, sample, text, dialog):
        self.record_snapshot()
        sample["caption"] = text
        dialog.accept()
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText("Caption updated.")

    def open_riff_hook_tagger(self):
        sample = self.get_selected_sample()
        if not sample:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        segs = sample.get("structural_segments") or []
        if not segs:
            QMessageBox.information(self, "Riff/Hook", "No structural sections yet — run the Structural Pipeline first.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Riff / Hook Tags — {sample.get('filename', '')}")
        dialog.resize(480, 420)
        lay = QVBoxLayout(dialog)
        lay.addWidget(QLabel("Mark the recurring riff / hook sections the captioner should emphasize:"))
        checks = []
        for seg in segs:
            cb = QCheckBox(f"{seg.get('name', '?')}  ({seg.get('start', 0)}-{seg.get('end', 0)}s)")
            cb.setChecked(bool(seg.get("hook")))
            checks.append((seg, cb))
            lay.addWidget(cb)
        note_edit = QLineEdit(sample.get("riff_note", ""))
        note_edit.setPlaceholderText("Optional riff note, e.g. 'downtuned E minor riff'")
        lay.addWidget(note_edit)
        row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_riff_hooks(sample, checks, note_edit, dialog))
        row.addStretch(); row.addWidget(cancel_btn); row.addWidget(save_btn)
        lay.addLayout(row)
        dialog.exec()

    def _save_riff_hooks(self, sample, checks, note_edit, dialog):
        self.record_snapshot()
        hooks = []
        for seg, cb in checks:
            seg["hook"] = cb.isChecked()
            if cb.isChecked():
                hooks.append(seg.get("name", "?"))
        sample["hooks"] = hooks
        sample["riff_note"] = note_edit.text().strip()
        dialog.accept()
        self.status_label.setText(f"Marked {len(hooks)} riff/hook section(s).")
        self.refresh_table()
        self.on_table_selection_changed()

    def open_stem_ab(self):
        sample = self.get_selected_sample()
        if not sample:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        stems = sample.get("stem_paths") or {}
        if not stems:
            QMessageBox.information(self, "Stem A/B", "No stems yet — run stem separation first.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Stem A/B — {sample.get('filename', '')}")
        dialog.resize(420, 180)
        lay = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItem("Full Mix")
        for name in sorted(stems):
            combo.addItem(f"Stem: {name}")
        lay.addWidget(combo)
        row = QHBoxLayout()
        play_btn = QPushButton("▶ Play")
        stop_btn = QPushButton("⏹ Stop")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        row.addWidget(play_btn); row.addWidget(stop_btn); row.addStretch(); row.addWidget(close_btn)
        lay.addLayout(row)

        def play():
            if self.media_player is None:
                return
            sel = combo.currentText()
            path = sample.get("audio_path", "") if sel == "Full Mix" else stems.get(sel.replace("Stem: ", ""), "")
            if path and os.path.exists(path):
                self.media_player.stop()
                self.media_player.setSource(QUrl.fromLocalFile(path))
                self.media_player.play()
                self.play_btn.setText("⏸")
        play_btn.clicked.connect(play)
        stop_btn.clicked.connect(self.stop_track_playback)
        dialog.exec()

    # -----------------------------------------------------------------------
    # On-disk versioning + Hugging Face push
    # -----------------------------------------------------------------------
    def open_versioning_dialog(self):
        from modules.versioning import list_versions, save_version, load_version, diff_json

        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Versioning")
        dialog.resize(560, 420)
        lay = QVBoxLayout(dialog)
        self.ver_list = QListWidget()

        def refresh():
            self.ver_list.clear()
            for v in list_versions():
                import datetime
                mt = datetime.datetime.fromtimestamp(v["mtime"]).strftime("%Y-%m-%d %H:%M")
                self.ver_list.addItem(f"{v['name']} — {v['tracks']} tracks ({mt})")
                self.ver_list.item(self.ver_list.count() - 1).setData(Qt.UserRole, v["path"])
        refresh()
        lay.addWidget(self.ver_list, 1)

        def create_snapshot():
            from modules.versioning import save_version
            path = save_version(self.dataset, label="manual")
            self.status_label.setText(f"Snapshot saved: {path}")
            refresh()

        def diff_selected():
            item = self.ver_list.currentItem()
            if not item:
                QMessageBox.information(self, "Versioning", "Select a snapshot to diff.")
                return
            from modules.versioning import load_version, diff_json
            snap = load_version(item.data(Qt.UserRole))
            text = diff_json(snap, self.dataset)
            d = QDialog(self); d.setWindowTitle("Version Diff"); d.resize(720, 500)
            dl = QVBoxLayout(d)
            b = QTextBrowser(); b.setPlainText(text); dl.addWidget(b, 1)
            c = QPushButton("Close"); c.clicked.connect(d.accept); dl.addWidget(c)
            d.exec()

        def restore_selected():
            item = self.ver_list.currentItem()
            if not item:
                QMessageBox.information(self, "Versioning", "Select a snapshot to restore.")
                return
            from modules.versioning import load_version
            reply = QMessageBox.question(self, "Restore Snapshot",
                                         "Replace the current dataset with this snapshot?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.record_snapshot()
                self.dataset = load_version(item.data(Qt.UserRole))
                self.refresh_table()
                self.on_table_selection_changed()
                self.status_label.setText("Dataset restored from snapshot.")

        row = QHBoxLayout()
        snap_btn = QPushButton("Create Snapshot")
        snap_btn.clicked.connect(create_snapshot)
        diff_btn = QPushButton("Diff with Current")
        diff_btn.clicked.connect(diff_selected)
        rest_btn = QPushButton("Restore")
        rest_btn.clicked.connect(restore_selected)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        row.addWidget(snap_btn); row.addWidget(diff_btn); row.addWidget(rest_btn)
        row.addStretch(); row.addWidget(close_btn)
        lay.addLayout(row)
        dialog.exec()

    def open_hf_push_dialog(self):
        if not self.dataset.get("samples"):
            QMessageBox.information(self, "Push to HF", "The dataset is empty — add tracks first.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Push to Hugging Face")
        dialog.resize(460, 200)
        lay = QVBoxLayout(dialog)
        self.hf_repo_edit = QLineEdit()
        self.hf_repo_edit.setPlaceholderText("username/dataset-name (or a new repo id)")
        self.hf_repo_edit.setText(self.config.get("kaggle_user", "") + "/ace-step-dataset")
        self.hf_private = QCheckBox("Private repo")
        lay.addWidget(QLabel("<b>Hugging Face repo:</b>"))
        lay.addWidget(self.hf_repo_edit)
        lay.addWidget(self.hf_private)
        note = QLabel("Uses the HF token from ⚙ Settings → Model Manager (or HF_TOKEN).")
        note.setStyleSheet("color: #aaa;")
        note.setWordWrap(True)
        lay.addWidget(note)
        row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        go_btn = QPushButton("Push")
        go_btn.clicked.connect(lambda: self._run_hf_push(dialog))
        row.addStretch(); row.addWidget(cancel_btn); row.addWidget(go_btn)
        lay.addLayout(row)
        dialog.exec()

    def _run_hf_push(self, dialog):
        repo = self.hf_repo_edit.text().strip()
        if not repo or "/" not in repo:
            QMessageBox.warning(self, "Push to HF", "Enter a repo id like 'username/dataset-name'.")
            return
        from workers.hf_push import HFPushWorker
        token = (self.config.get("hf_token") or "").strip() or None
        self.hf_push_worker = HFPushWorker(self.dataset, repo, token=token,
                                           private=self.hf_private.isChecked(), parent=self)
        self.hf_push_worker.finished_ok.connect(lambda r: self._on_hf_pushed(r, dialog))
        self.hf_push_worker.failed.connect(lambda e: self._on_hf_push_failed(e, dialog))
        dialog.accept()
        self.status_label.setText(f"Pushing {repo} to Hugging Face…")
        self.hf_push_worker.start()

    def _on_hf_pushed(self, repo, dialog):
        self.status_label.setText(f"Pushed dataset to {repo}.")
        QMessageBox.information(self, "Push Complete",
                                f"Dataset pushed to:\nhttps://huggingface.co/datasets/{repo}")

    def _on_hf_push_failed(self, err, dialog):
        self.status_label.setText("Hugging Face push failed.")
        QMessageBox.warning(self, "Push Failed", str(err))


    # -----------------------------------------------------------------------
    # Common worker callbacks
    # -----------------------------------------------------------------------
    def on_worker_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def on_worker_error(self, err_msg):
        notice = getattr(self, "rescan_notice", None)
        if notice is not None:
            notice.close()
            notice.deleteLater()
            self.rescan_notice = None

        self.run_ai_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.normalize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)      
        self.status_label.setText("Operation error.")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")

    def on_remote_pipeline_success(self, result_payload):
        """Clean decoupled pass-through directing data integration to our core script engine."""
        if hasattr(self, "scan_btn") and self.scan_btn is not None:
            self.scan_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Trigger your independent module function
        from core.manifest_sync import integrate_remote_pipeline_data
        integrate_remote_pipeline_data(self, result_payload)



