import os
import json
import re
import html
import uuid
import time
import shutil
from pathlib import Path

from config import DEFAULT_CONFIG
from modules.mvsep_gui import MVSEPDialog
from modules.config_store import load_config, save_config
from modules.model_manager import load_catalog, find_model, is_downloaded, remove_model
from workers.model_manager import ModelDownloadWorker
from workers.rockstar import RockstarLookupWorker
from workers.embeddings import EmbeddingWorker
from workers.export import ExportWorker
from workers.tag_creator import TagCreatorWorker
from workers.musicbrainz import MusicBrainzWorker
from modules.lyrics_tools import split_long_lines
from modules.dataset_schema import (
    derive_instrumental_mode,
    new_sample,
    normalize_dataset,
)

# Modern worker implementations (split into workers/ modules).
from workers.caption import RemoteCaptionWorker, resolve_backend
from workers.structural import (
    StructuralPipelineBatchWorker,
)
from workers.assistant import (
    AssistantWorker, APP_HELP_TEXT, ASSISTANT_TOOLS, build_system_prompt,
    build_sound_profile, summarize_dataset,
)

# --- NEW: extracted widgets/orchestrator (replaces inline class definitions below) ---
from widgets import WaveformWidget, ScatterPlotWidget
from workers.advanced import AdvancedDatasetOrchestratorWorker
from workers.dsp_normalizer import DspNormalizerWorker

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QLabel, QLineEdit, QTextEdit, QFileDialog,
    QMessageBox, QSplitter, QGroupBox,
    QInputDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QCheckBox, QDialog, QFormLayout, QProgressBar, QScrollArea,
    QTabWidget, QRadioButton, QButtonGroup, QToolButton, QMenu,
    QListWidget, QListWidgetItem, QTextBrowser, QAbstractItemView
)
# Scroll-wheel-guarded value widgets: the wheel only changes these after the
# control has been clicked, so scrolling the dataset past a combo/spin/slider
# can no longer silently alter a value. Aliased to the plain Qt names so every
# existing construction site below is guarded with no further edits.
from modules.wheel_guard import (
    GuardedComboBox as QComboBox,
    GuardedDoubleSpinBox as QDoubleSpinBox,
    GuardedSlider as QSlider,
    GuardedSpinBox as QSpinBox,
)
from PySide6.QtGui import QDesktopServices

from ui.themes import repolish

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
# Deliberately permissive: users bring whatever they have. Lossless formats are
# recommended for training quality (see the add-track warnings dialog), but the
# app will accept lossy/container formats rather than block the workflow.
AUDIO_EXTS = {
    # Lossless (recommended)
    ".wav", ".flac", ".aiff", ".aif", ".aifc", ".alac", ".ape", ".wv", ".tta", ".au",
    # Lossy
    ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".ac3", ".amr", ".mp2",
    # Containers / misc
    ".webm", ".mkv", ".mp4", ".caf", ".rf64", ".bwf",
}
# Same set as a deterministic, human-ordered tuple for building the file-dialog
# filter string (a set's iteration order would make the filter jump around).
AUDIO_EXTS_ORDERED = (
    ".wav", ".flac", ".aiff", ".aif", ".aifc", ".alac", ".ape", ".wv", ".tta", ".au",
    ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".ac3", ".amr", ".mp2",
    ".webm", ".mkv", ".mp4", ".caf", ".rf64", ".bwf",
)
# Formats we treat as lossless when warning about dataset quality.
LOSSLESS_EXTS = {
    ".wav", ".flac", ".aiff", ".aif", ".aifc", ".alac", ".ape", ".wv", ".tta", ".au",
    ".caf", ".rf64", ".bwf",
}
# Formats known to be lossy-compressed (quality penalty for LoRA training).
LOSSY_EXTS = {
    ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".ac3", ".amr", ".mp2",
}
# Settings key remembering that the user dismissed the add-track warnings.
ADD_WARNINGS_SETTINGS_KEY = "suppress_add_track_warnings"

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


        self.dataset = {
            "metadata": {
                "name": "",
                "custom_tag": "",
                "tag_position": "prepend",
                "created_at": "",
                "num_samples": 0,
                "all_instrumental": False,
                "genre_ratio": 0,
                # Legacy 3-state string, kept in sync from all_instrumental.
                "instrumental_mode": "mixed",
            },
            "samples": []
        }
        # Config: DEFAULT_CONFIG (best-working defaults) + settings.json + the
        # encrypted secrets store. Each secret has a per-key "remember on this
        # device" policy — when unchecked it is kept for the session only.
        self.config = load_config(DEFAULT_CONFIG)
        self.original_backups = {}
        self.active_worker = None
        self.filter_exceptions_only = False
        self.bypass_warnings = False
        # Filter/search state (Dataset Studio table) — one search box covers
        # filename/caption/tag/genre/key; no separate genre/key/BPM filter
        # fields (single place for those values = the table, auto-locked).
        self._table_sample_indices = []
        self._last_selected_row = -1
        self.filter_query = ""
        self.filter_inst = "all"
        self.filter_captioned = False
        self._loading_table = False
        self.kaggle_notebook_unlocked = False  # NEW
        # RETAINMENT: every top-level tab page and grouped sub-tab container is
        # appended here. PySide6 frees a C++ widget when its last Python
        # reference goes away, and freeing a parent frees its children — so a
        # local-only tab variable would silently delete the widgets inside it.
        self._tab_pages = []
        self._grouped_tabs = []
        self._grouped_tab_inner = []
        # Tracks the LAST caption run covered, so the end-of-run diff review can
        # tell this run's proposals apart from the previous run's.
        self._caption_scope_ids = []
        # The backend the CURRENT run is using, so a caption produced without
        # hearing the audio can be stamped as a placeholder.
        self._active_caption_backend = ""
        # True while a caption worker is in flight. This drives the step buttons'
        # enabled state (see update_ace_actions) and `active_worker` cannot: it is
        # set on every run and never cleared, so it says nothing about NOW.
        self._caption_busy = False

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

    def open_mvsep_dialog(self):
        selected = self.get_selected_sample()

        audio_path = ""

        if selected:
            audio_path = selected.get("audio_path", "")

        dialog = MVSEPDialog(
            config=self.config,
            audio_path=audio_path,
            parent=self,
        )

        dialog.exec()

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
        self.lyrics_engine_combo = QComboBox()
        self.import_json_manifest_btn = None
        self.sync_meta_btn = None
        self.normalize_btn = None
        self.transcribe_btn = None
        # The window is assembled at the end of init_ui by ui.shell.install_shell
        # from the pieces collected here (the approved dock layout).
        parts = {}

        def _wrap(layout):
            w = QWidget()
            layout.setContentsMargins(0, 0, 0, 0)
            w.setLayout(layout)
            return w

        settings_tab = QWidget()
        self.init_settings_tab(settings_tab)

        advanced_tab = QWidget()
        self.init_advanced_tab(advanced_tab)

        struct_tab = QWidget()
        self.init_structural_tab(struct_tab)

        assistant_tab = QWidget()
        self.init_assistant_tab(assistant_tab)

        tag_tab = QWidget()
        self.init_tag_manager_tab(tag_tab)

        embed_tab = QWidget()
        self.init_embedding_map_tab(embed_tab)

        caption_tab = QWidget()
        self.init_caption_tab(caption_tab)

        ace_step_tab = QWidget()
        self.init_ace_step_tab(ace_step_tab)

        # --- Grouped tabs: keep the top level to a short, scannable list ----
        # Pipelines (Structural / Advanced) share one tab with inner
        # sub-tabs; Organize (Tag Manager / Embedding Map) likewise.
        pipelines_tab = self._build_grouped_tab(
            "Pipeline",
            [
                ("🎶 Structural", struct_tab),
                ("🧠 Advanced", advanced_tab),
            ],
        )
        organize_tab = self._build_grouped_tab(
            "Organize",
            [
                ("🏷️ Tag Manager", tag_tab),
                ("🗺️ Embedding Map", embed_tab),
            ],
        )

        lyrics_tab = QWidget()
        self.init_lyrics_tab(lyrics_tab)

        # Caption is grouped too: the ACE-Step Kaggle pipeline (staging folder,
        # dataset identity, output folder, run controls) has its own page instead
        # of sharing space with four other backends.
        caption_group_tab = self._build_grouped_tab(
            "Caption",
            [
                ("🅰 ACE-Step (Kaggle)", ace_step_tab),
                ("🎤 Other backends && MOSS", caption_tab),
            ],
        )

        shell_pages = {
            "caption": caption_group_tab, "lyrics": lyrics_tab,
            "structure": pipelines_tab, "organize": organize_tab,
            "assistant": assistant_tab, "settings": settings_tab,
        }

        # --- Header Bar ---
        # Compact primary actions; one-off utilities live behind the ⋯ menu so
        # the bar stays scannable. Every action is still one click away.
        header_bar = QHBoxLayout()

        load_btn = QPushButton("📂 Open")
        load_btn.setToolTip("Load a dataset JSON file.")
        load_btn.clicked.connect(self.load_dataset)

        save_btn = QPushButton("💾 Save")
        save_btn.setToolTip("Save the dataset JSON (previous file is backed up).")
        save_btn.clicked.connect(self.save_dataset)

        self.add_menu_btn = QToolButton()
        self.add_menu_btn.setText("➕ Add")
        self.add_menu_btn.setToolTip("Add audio to the dataset.")
        self.add_menu_btn.setPopupMode(QToolButton.InstantPopup)
        add_menu = QMenu(self.add_menu_btn)
        add_menu.addAction("Single Song…", self.add_audio_files)
        add_menu.addAction("Audio Folder (recursive)…", self.add_audio_folder)
        self.add_menu_btn.setMenu(add_menu)

        self.export_menu_btn = QToolButton()
        self.export_menu_btn.setText("📦 Export")
        self.export_menu_btn.setToolTip("Export, split, or publish the dataset.")
        self.export_menu_btn.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(self.export_menu_btn)
        export_menu.addAction("Export / Split…", self.open_export_dialog)
        export_menu.addAction("Push to Hugging Face…", self.open_hf_push_dialog)
        self.export_menu_btn.setMenu(export_menu)

        self.more_menu_btn = QToolButton()
        self.more_menu_btn.setText("⋯")
        self.more_menu_btn.setToolTip("Statistics, versioning, and warning controls.")
        self.more_menu_btn.setPopupMode(QToolButton.InstantPopup)
        more_menu = QMenu(self.more_menu_btn)
        more_menu.addAction("📊 Dataset Statistics", self.show_stats_report)
        more_menu.addAction("🗂 Versioning…", self.open_versioning_dialog)
        more_menu.addSeparator()
        self.bypass_btn = QPushButton("🛡 Bypass Warnings")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.clicked.connect(self.toggle_bypass)
        self.bypass_action = more_menu.addAction("🛡 Bypass Warnings")
        self.bypass_action.setCheckable(True)
        self.bypass_action.toggled.connect(self.bypass_btn.setChecked)
        self.bypass_btn.toggled.connect(self.bypass_action.setChecked)
        self.more_menu_btn.setMenu(more_menu)

        header_bar.addWidget(load_btn)
        header_bar.addWidget(save_btn)
        header_bar.addWidget(self.add_menu_btn)
        header_bar.addWidget(self.export_menu_btn)

        header_bar.addStretch()

        self.undo_btn = QPushButton("↩")
        self.undo_btn.setToolTip("Undo the last change.")
        self.undo_btn.setMaximumWidth(36)
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo)
        header_bar.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("↪")
        self.redo_btn.setToolTip("Redo the last undone change.")
        self.redo_btn.setMaximumWidth(36)
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self.redo)
        header_bar.addWidget(self.redo_btn)

        header_bar.addWidget(self.more_menu_btn)

        parts["header"] = _wrap(header_bar)

        # --- General Properties ---
        gen_box = QGroupBox("Dataset")
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

        parts["dataset_box"] = gen_box

        # "Set All Tracks" — the deliberate alternative to per-row editing.
        # Collapsed by default; nothing changes until Apply + confirm.
        bulk_holder = QVBoxLayout()
        self.init_bulk_edit_panel(bulk_holder)
        parts["bulk"] = _wrap(bulk_holder)

        # ============================================================================
        # Row 1: Dataset Calibration (Primary Controls)
        # ============================================================================
        audit_strip = QHBoxLayout()
        
        # 👑 THE FLAGSHIP ENGINE: Re-labeled to match your 1.5XL Caption & Lyrics spec
        self.import_json_manifest_btn = QPushButton("📥 Import ACE-Step 1.5XL Tags")
        self.import_json_manifest_btn.setProperty("role", "primary")
        self.import_json_manifest_btn.setToolTip("Instantly loads your schema-enforced 1.5XL caption and lyrics tags to fix placeholders.")
        self.import_json_manifest_btn.clicked.connect(self.import_acestep_15xl_tags)
        audit_strip.addWidget(self.import_json_manifest_btn)

        self.sync_meta_btn = QPushButton("🔄 Sync Metadata Guards")
        self.sync_meta_btn.setToolTip("Forces health report alerts to match your manifest properties, clearing false warnings.")
        self.sync_meta_btn.clicked.connect(self.force_sync_manifest_to_metadata)
        audit_strip.addWidget(self.sync_meta_btn)

        self.normalize_btn = QPushButton("🎚️ Fix && DSP Normalize")
        self.normalize_btn.clicked.connect(self.start_dsp_normalize)
        audit_strip.addWidget(self.normalize_btn)
        audit_strip.addStretch() 
        parts["audit_strip"] = audit_strip
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
        parts["lyrics_strip"] = lyrics_strip

        # ============================================================================
        # Step 3: Advanced Remote Cluster & Repo Controls (Row 3)
        # ============================================================================
        advanced_strip = QHBoxLayout()

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
        parts["advanced_strip"] = advanced_strip

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
        parts["tools_strip"] = tools_strip


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

        view_row = QHBoxLayout()
        view_row.addWidget(self.all_view_btn)
        view_row.addWidget(self.exceptions_view_btn)
        view_row.addStretch()
        parts["view_row"] = _wrap(view_row)

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
        parts["player"] = _wrap(player_bar)

        self.waveform = WaveformWidget()
        self.waveform.set_audio(None)
        parts["waveform"] = self.waveform

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
        self.filter_count_label.setProperty("muted", True)
        filter_bar.addWidget(self.filter_search, 1)
        filter_bar.addWidget(self.filter_inst_combo)
        filter_bar.addWidget(self.filter_captioned_check)
        filter_bar.addWidget(clear_filters)
        filter_bar.addWidget(self.filter_count_label)
        parts["filter"] = _wrap(filter_bar)

        # --- Table + Inspector (placed side by side by ui.shell) ---

        # Column schema (index -> field): 0 Filename (read-only),
        # 1 Tag, 2 Genre, 3 Language, 4 Key, 5 BPM, 6 Time signature,
        # 7 Duration, 8 Actions. All metadata columns are editable inline;
        # new tracks are NOT locked by default so you can type values straight in.
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Filename", "Tag", "Genre", "Language", "Key", "BPM", "Time", "Duration", "Actions"]
        )
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 8):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.table.itemChanged.connect(self.on_metadata_cell_edited)
        parts["table"] = self.table

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inspector = QWidget()
        insp_layout = QVBoxLayout(inspector)
        insp_layout.setContentsMargins(8, 8, 8, 8)

        self.sample_health_alert = QLabel("Select a track to edit its metadata and caption.")
        self.sample_health_alert.setWordWrap(True)
        self.sample_health_alert.setProperty("health", "neutral")
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
        parts["inspector"] = scroll

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self._install_shell(parts, shell_pages)

    # -----------------------------------------------------------------------
    # Settings Tab
    # -----------------------------------------------------------------------
    def _add_tab(self, page, label):
        """Add a top-level tab and retain its page on ``self``.

        See ``self._tab_pages`` — without an explicit reference PySide6 may free
        the page (and every widget inside it) once the local name goes away.
        """
        self.tabs.addTab(page, label)
        self._tab_pages.append(page)
        return page

    def _build_grouped_tab(self, label, pages):
        """Wrap several pages in one top-level tab with an inner sub-tab strip.

        Keeps the main tab bar short without hiding functionality: each page is
        the same QWidget the old top-level tab used, so every ``manager.<attr>``
        reference and handler still resolves unchanged.

        RETAINMENT: both the container and the inner QTabWidget are stored on
        ``self``. PySide6 destroys a C++ object once its last Python reference
        goes away, and destroying a parent destroys its children — a local-only
        container silently takes every widget inside it (and the manager
        attributes pointing at them) down with it.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        inner = QTabWidget()
        for page_label, page in pages:
            inner.addTab(page, page_label)
        layout.addWidget(inner)
        container.setAccessibleName(label)

        self._grouped_tabs.append(container)
        self._grouped_tab_inner.append(inner)
        return container

    def init_lyrics_tab(self, parent):
        """Build the 🎵 Lyrics tab and connect its actions."""
        from ui.lyrics_tab import build_lyrics_tab
        build_lyrics_tab(self, parent)

        self.lyrics_preview_btn.clicked.connect(self.preview_lyrics_tidy)
        self.lyrics_apply_btn.clicked.connect(self.apply_lyrics_tidy)
        self.lyrics_manual_edit_btn.clicked.connect(self.open_lyrics_editor)
        self.lyrics_add_row_btn.clicked.connect(self._lyrics_add_contract_row)
        self.lyrics_del_row_btn.clicked.connect(self._lyrics_remove_contract_row)
        self.lyrics_reset_btn.clicked.connect(self._lyrics_reset_contracts)
        self.lyrics_profile_load_btn.clicked.connect(self.load_lyrics_profile)
        self.lyrics_profile_save_btn.clicked.connect(self.save_lyrics_profile)
        self.lyrics_profile_delete_btn.clicked.connect(self.delete_lyrics_profile)
        self.lyrics_load_all_btn.clicked.connect(self.load_all_lyrics)
        self.lyrics_tidy_all_btn.clicked.connect(self.tidy_all_lyrics_block)
        self.lyrics_write_back_btn.clicked.connect(self.write_back_all_lyrics)
        self.refresh_lyrics_profiles()

    # --- Profiles: master always on, a profile specialises it -------------

    def refresh_lyrics_profiles(self, select=None):
        """Repopulate the profile combo from the local lyrics_profiles/ folder."""
        from modules.lyrics_profiles import list_profiles

        names = list_profiles()
        combo = self.lyrics_profile_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("— master rules only —")
        combo.addItems(names)
        if select and select in names:
            combo.setCurrentText(select)
        combo.blockSignals(False)
        return names

    def _lyrics_master_rules(self):
        """Master (always-on) rules: settings override, else shipped defaults."""
        from modules.lyrics_normalizer import DEFAULT_CONTRACTIONS, DEFAULT_ING_EXCEPTIONS

        master_c = self.config.get("lyrics_contractions") or dict(DEFAULT_CONTRACTIONS)
        master_i = self.config.get("lyrics_ing_exceptions") or list(DEFAULT_ING_EXCEPTIONS)
        return dict(master_c), list(master_i)

    def load_lyrics_profile(self):
        """Load the selected profile's rules into the tab's table + options."""
        from modules.lyrics_profiles import load_profile

        name = self.lyrics_profile_combo.currentText()
        if not name or name.startswith("\u2014"):
            available = self.refresh_lyrics_profiles()
            if not available:
                QMessageBox.information(
                    self, "No Profiles Yet",
                    "You have not saved any band profiles yet.\n\n"
                    "Edit the contraction table below the way this band needs it, "
                    "then press “Save As…” to store it as a named profile. "
                    "Profiles live in the local lyrics_profiles/ folder.",
                )
            else:
                QMessageBox.information(
                    self, "Pick a Profile",
                    "Choose a profile from the drop-down first, then press Load.\n\n"
                    "Available: " + ", ".join(available),
                )
            self.lyrics_profile_status.setText(
                "Master rules only — no profile selected."
            )
            return
        profile = load_profile(name)
        if not profile:
            QMessageBox.warning(self, "Profile Not Found", f"Could not read profile '{name}'.")
            self.refresh_lyrics_profiles()
            return

        self.lyrics_contract_table.setRowCount(0)
        for word in sorted(profile["contractions"], key=lambda w: (len(w), w)):
            r = self.lyrics_contract_table.rowCount()
            self.lyrics_contract_table.insertRow(r)
            self.lyrics_contract_table.setItem(r, 0, QTableWidgetItem(word))
            self.lyrics_contract_table.setItem(r, 1, QTableWidgetItem(profile["contractions"][word]))

        self.lyrics_ing_exceptions_edit.setText(", ".join(sorted(profile["ing_exceptions"])))
        self.lyrics_ing_check.setChecked(profile["ing_to_in"])
        self.lyrics_tags_check.setChecked(profile["capitalize_tags"])
        self.lyrics_punct_check.setChecked(profile["strip_punctuation"])
        self.lyrics_profile_status.setText(
            f"Profile <b>{name}</b> loaded — layered on top of the master rules."
        )
        self.status_label.setText(f"Lyrics profile '{name}' loaded.")

    def save_lyrics_profile(self):
        """Save the current table + options as a named profile."""
        from modules.lyrics_profiles import save_profile
        from ui.lyrics_tab import read_contractions_from_table, read_ing_exceptions

        name, ok = QInputDialog.getText(
            self, "Save Lyrics Profile",
            "Profile name (e.g. the band or artist):",
            text=self.lyrics_profile_combo.currentText().strip("— "),
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        path = save_profile(
            name,
            read_contractions_from_table(self),
            read_ing_exceptions(self),
            ing_to_in=self.lyrics_ing_check.isChecked(),
            capitalize_tags=self.lyrics_tags_check.isChecked(),
            strip_punctuation=self.lyrics_punct_check.isChecked(),
        )
        self.refresh_lyrics_profiles(select=name)
        self.lyrics_profile_status.setText(f"Profile <b>{name}</b> saved to {path}")
        self.status_label.setText(f"Lyrics profile '{name}' saved.")

    def delete_lyrics_profile(self):
        """Delete the selected profile file."""
        from modules.lyrics_profiles import delete_profile

        name = self.lyrics_profile_combo.currentText()
        if name.startswith("—"):
            self.status_label.setText("Select a profile to delete.")
            return
        if QMessageBox.question(
            self, "Delete Profile",
            f"Delete the lyrics profile '{name}'?\n\nThis only removes the local "
            "profile file; the master rules are unaffected.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        if delete_profile(name):
            self.refresh_lyrics_profiles()
            self.status_label.setText(f"Lyrics profile '{name}' deleted.")
        else:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete '{name}'.")

    def _lyrics_options(self):
        """Effective tidy rules: master (always on) + the loaded profile.

        Master = the settings override or the shipped defaults; a profile layers
        on top, overriding master for any word it defines and adding its own.
        """
        from modules.lyrics_profiles import load_profile, merge_rules
        from ui.lyrics_tab import read_contractions_from_table, read_ing_exceptions

        master_c, master_ing = self._lyrics_master_rules()
        table_c = read_contractions_from_table(self)
        table_ing = read_ing_exceptions(self)

        profile_name = self.lyrics_profile_combo.currentText()
        profile = None
        if not profile_name.startswith("—"):
            profile = load_profile(profile_name)

        if profile:
            # The tab's table IS the profile's view, so merge master under it.
            merged_c, merged_ing = merge_rules(
                master_c, table_c, master_ing, table_ing
            )
        else:
            merged_c, merged_ing = table_c, table_ing

        return {
            "contractions": merged_c,
            "ing_to_in": self.lyrics_ing_check.isChecked(),
            "ing_exceptions": merged_ing,
            "do_capitalize_tags": self.lyrics_tags_check.isChecked(),
            "do_capitalize_lines": self.lyrics_lines_check.isChecked(),
            "do_strip_punctuation": self.lyrics_punct_check.isChecked(),
        }

    def _lyrics_scope_samples(self):
        """Tracks the tidy pass should touch, per the scope combo."""
        samples = self.dataset.get("samples", [])
        scope = self.lyrics_scope_combo.currentText()
        if scope.startswith("Selected"):
            s = self.get_selected_sample()
            return [s] if s else []
        if scope.startswith("All Tracks ("):  # instrumentals excluded
            return [s for s in samples if not s.get("is_instrumental")]
        return list(samples)

    def preview_lyrics_tidy(self, silent=False):
        """Show the tidy result for the selected track.

        ``silent`` suppresses warnings (used when auto-loading on tab switch).
        The Before pane always shows the selected track's current lyrics; the
        After pane shows the tidy result, so both populate together.
        """
        from modules.lyrics_normalizer import normalize_lyrics

        s = self.get_selected_sample()
        if not s:
            if not silent:
                QMessageBox.warning(
                    self, "No Track Selected",
                    "Select a track in the Dataset Studio table first, then come "
                    "back to this tab.",
                )
            self.lyrics_before.setPlainText("")
            self.lyrics_after.setPlainText("")
            self.lyrics_report_label.setText(
                "Select a track in the Dataset Studio tab to preview it here."
            )
            return
        source = s.get("raw_lyrics") or s.get("formatted_lyrics") or s.get("lyrics") or ""
        if not source.strip():
            self.lyrics_before.setPlainText("")
            self.lyrics_after.setPlainText("")
            self.lyrics_report_label.setText(
                f"'{s.get('filename', '?')}' has no lyrics to tidy yet."
            )
            return
        new_text, report = normalize_lyrics(source, **self._lyrics_options())
        self.lyrics_before.setPlainText(source)
        self.lyrics_after.setPlainText(new_text)

        # Diff view: line-level changes, so you can scan what the tidy did.
        from ui.lyrics_tab import unified_diff
        self.lyrics_diff.setPlainText(unified_diff(source, new_text) or "(no changes)")

        self.lyrics_report_label.setText(
            f"{s.get('filename', '?')} — {report['lines_changed']} line(s) changed | "
            f"{len(report['contractions'])} contraction(s) mapped | "
            f"{report['apostrophes']} apostrophe(s) stripped | "
            f"{report['ing']} \u2011ing word(s) shortened | "
            f"{report['capitalized']} line(s) capitalized | "
            f"{report['tags']} tag(s) capitalized"
        )

    def apply_lyrics_tidy(self):
        """Rewrite lyrics on the tracks in scope (undoable)."""
        from modules.lyrics_normalizer import normalize_lyrics

        targets = self._lyrics_scope_samples()
        if not targets:
            QMessageBox.warning(self, "Nothing To Tidy", "No tracks match the selected scope.")
            return

        opts = self._lyrics_options()
        replace = self.lyrics_replace_check.isChecked()
        touched = 0
        total_lines = 0

        self.record_snapshot()
        for s in targets:
            source = s.get("raw_lyrics") or s.get("formatted_lyrics") or s.get("lyrics") or ""
            if not source.strip():
                continue
            if not replace and (s.get("formatted_lyrics") or "").strip():
                continue  # fill-blanks-only mode: keep existing lyrics

            new_text, report = normalize_lyrics(source, **opts)
            if new_text == source:
                continue
            # Preserve the pre-tidy text so a re-run is idempotent and reversible.
            if not s.get("raw_lyrics"):
                s["raw_lyrics"] = source
            s["formatted_lyrics"] = new_text
            s["lyrics"] = new_text
            touched += 1
            total_lines += report["lines_changed"]

        # Persist the (possibly edited) tables so they survive a restart.
        self.config["lyrics_contractions"] = opts["contractions"]
        self.config["lyrics_ing_exceptions"] = sorted(opts["ing_exceptions"])

        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText(
            f"Tidied lyrics on {touched} track(s) ({total_lines} line(s) changed)."
        )
        if touched:
            self.preview_lyrics_tidy()

    def _lyrics_add_contract_row(self):
        r = self.lyrics_contract_table.rowCount()
        self.lyrics_contract_table.insertRow(r)
        self.lyrics_contract_table.setItem(r, 0, QTableWidgetItem(""))
        self.lyrics_contract_table.setItem(r, 1, QTableWidgetItem(""))

    def _lyrics_remove_contract_row(self):
        row = self.lyrics_contract_table.currentRow()
        if row >= 0:
            self.lyrics_contract_table.removeRow(row)

    def _lyrics_reset_contracts(self):
        """Restore the master table to the shipped defaults."""
        from modules.lyrics_normalizer import DEFAULT_CONTRACTIONS, DEFAULT_ING_EXCEPTIONS
        from ui.lyrics_tab import _fill_contract_table

        self.config["lyrics_contractions"] = dict(DEFAULT_CONTRACTIONS)
        self.config["lyrics_ing_exceptions"] = sorted(DEFAULT_ING_EXCEPTIONS)
        _fill_contract_table(self)
        self.lyrics_ing_exceptions_edit.setText(", ".join(sorted(DEFAULT_ING_EXCEPTIONS)))
        self.lyrics_profile_combo.setCurrentIndex(0)
        self.lyrics_profile_status.setText(
            "Master table reset to the shipped defaults; no profile loaded."
        )
        self.status_label.setText("Master lyrics rules reset to defaults.")

    def init_bulk_edit_panel(self, parent):
        """Add the "Set All Tracks" bulk-edit panel to the Studio layout."""
        from ui.bulk_edit_panel import build_bulk_edit_panel
        build_bulk_edit_panel(self, parent)
        self.bulk_apply_btn.clicked.connect(self.apply_bulk_edits)
        self.bulk_language_apply_btn.clicked.connect(self.apply_language_to_all)

    def apply_language_to_all(self):
        """One-click: set the chosen language on every track."""
        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks first.")
            return
        lang = self.bulk_language_combo.currentText().strip()
        if not lang or lang.startswith("\u2014"):
            QMessageBox.information(
                self, "Pick a Language",
                "Choose a language in the box first, then press Apply language to ALL.",
            )
            return
        if QMessageBox.question(
            self, "Apply Language",
            f"Set language = '{lang}' on all {len(samples)} track(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.record_snapshot()
        for s in samples:
            s["language"] = lang
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText(f"Language set to '{lang}' on {len(samples)} track(s).")

    def _bulk_update_pending(self, *_):
        """Keep the pending-field count and Apply button state in sync."""
        ticked = [k for k, box in self.bulk_include.items() if box.isChecked()]
        if not ticked:
            self.bulk_pending_label.setText("No fields ticked.")
        else:
            self.bulk_pending_label.setText(
                "Will apply: " + ", ".join(t.replace("_", " ") for t in ticked)
            )
        self.bulk_apply_btn.setEnabled(bool(ticked))

    def _bulk_untick_all(self):
        for box in self.bulk_include.values():
            box.setChecked(False)
        self._bulk_update_pending()

    def apply_bulk_edits(self):
        """Apply the ticked bulk fields to every track, after confirmation."""
        from ui.bulk_edit_panel import read_bulk_edits

        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks first.")
            return

        values, meta, rewrite = read_bulk_edits(self)
        if not values and not meta and not rewrite:
            QMessageBox.information(
                self, "Nothing To Apply",
                "Tick at least one field and give it a value, then press Apply.",
            )
            return

        # Build a plain-language summary so the confirmation is unambiguous.
        lines = []
        for field, value in values.items():
            shown = "clear" if value in (None, "") else repr(value)
            lines.append(f"&bull; <b>{field.replace('_', ' ')}</b> → {shown}")
        for field, value in meta.items():
            lines.append(f"&bull; <b>{field.replace('_', ' ')}</b> (dataset) → {value}")
        if rewrite:
            lines.append(
                f"&bull; <b>audio path</b> → replace {rewrite[0]!r} with {rewrite[1]!r}"
            )

        confirm = QMessageBox(self)
        confirm.setWindowTitle("Confirm Bulk Edit")
        confirm.setIcon(QMessageBox.Warning)
        confirm.setText(
            f"Apply {len(lines)} change(s) to <b>all {len(samples)} track(s)</b>?"
        )
        confirm.setInformativeText("<br>".join(lines))
        confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        confirm.setDefaultButton(QMessageBox.Cancel)  # deliberate: default is cancel
        if confirm.exec() != QMessageBox.Yes:
            self.status_label.setText("Bulk edit cancelled; nothing changed.")
            return

        self.record_snapshot()
        changed = 0
        rewrites = 0

        for s in samples:
            for field, value in values.items():
                s[field] = value
            if rewrite:
                path = s.get("audio_path", "") or ""
                if rewrite[0] in path:
                    s["audio_path"] = path.replace(rewrite[0], rewrite[1])
                    rewrites += 1
            changed += 1

        if meta:
            md = self.dataset.setdefault("metadata", {})
            md.update(meta)
            # Keep the legacy 3-state string in step with the boolean/mode.
            if "all_instrumental" in values or "is_instrumental" in values:
                any_vocal = any(not t.get("is_instrumental") for t in samples)
                md["all_instrumental"] = not any_vocal

        self.refresh_table()
        self.on_table_selection_changed()
        note = f"Bulk edit applied to {changed} track(s)."
        if rewrites:
            note += f" {rewrites} audio path(s) rewritten."
        self.status_label.setText(note)

    # --- All-lyrics block (whole dataset in one editable view) -----------

    def load_all_lyrics(self, silent=False):
        """Pull every track's lyrics into the editable block."""
        from ui.lyrics_tab import build_all_lyrics_block

        samples = self.dataset.get("samples", [])
        if not samples:
            if not silent:
                QMessageBox.warning(self, "No Tracks", "Add audio tracks first.")
            return
        self.lyrics_all_edit.setPlainText(build_all_lyrics_block(samples))
        with_lyrics = sum(
            1 for s in samples
            if (s.get("raw_lyrics") or s.get("formatted_lyrics") or s.get("lyrics") or "").strip()
        )
        if not silent:
            self.status_label.setText(
                f"Loaded {len(samples)} track(s) into the lyrics block "
                f"({with_lyrics} with lyrics)."
            )

    def tidy_all_lyrics_block(self):
        """Run the tidy options over the whole editable block (tags untouched)."""
        from modules.lyrics_normalizer import normalize_lyrics

        text = self.lyrics_all_edit.toPlainText()
        if not text.strip():
            self.status_label.setText("Nothing to tidy — load the lyrics block first.")
            return
        new_text, report = normalize_lyrics(text, **self._lyrics_options())
        self.lyrics_all_edit.setPlainText(new_text)
        self.status_label.setText(
            f"Tidied the block: {report['lines_changed']} line(s) changed, "
            f"{len(report['contractions'])} contraction(s) mapped."
        )

    def write_back_all_lyrics(self):
        """Split the block on its filename markers and save to each track."""
        from ui.lyrics_tab import parse_all_lyrics_block

        text = self.lyrics_all_edit.toPlainText()
        if not text.strip():
            self.status_label.setText("Nothing to write back — load the block first.")
            return
        by_name = parse_all_lyrics_block(text)
        if not by_name:
            QMessageBox.warning(
                self, "No Markers Found",
                "Could not find any “---- filename ----” markers in the block.\n\n"
                "Press “Load All Lyrics” first so the markers are present.",
            )
            return

        samples = self.dataset.get("samples", [])
        self.record_snapshot()
        written = 0
        unmatched = []
        for s in samples:
            name = s.get("filename", "?")
            if name not in by_name:
                unmatched.append(name)
                continue
            new_text = by_name[name]
            if not s.get("raw_lyrics"):
                prev = s.get("formatted_lyrics") or s.get("lyrics") or ""
                if prev:
                    s["raw_lyrics"] = prev
            s["formatted_lyrics"] = new_text
            s["lyrics"] = new_text
            written += 1

        self.refresh_table()
        self.on_table_selection_changed()
        note = f"Wrote lyrics back to {written} track(s)."
        if unmatched:
            note += f" {len(unmatched)} track(s) had no marker: " + ", ".join(unmatched[:3])
            if len(unmatched) > 3:
                note += "…"
        self.status_label.setText(note)

    def init_settings_tab(self, parent):
        from ui.settings_tab import build_settings_tab
        build_settings_tab(self, parent)

    def init_caption_tab(self, parent):
        """Build the 🎤 Caption tab's backend page and connect its actions.

        The RUN buttons and the limits live in the ACE-Step page
        (``init_ace_step_tab``): this page owns the backend choice, the
        prose/tags blend and the MOSS-Audio group.
        """
        from ui.caption_tab import build_caption_tab
        build_caption_tab(self, parent)
        self.caption_blend_slider.valueChanged.connect(self.on_caption_blend_changed)
        self.caption_override_btn.clicked.connect(self.set_track_caption_override)
        self.caption_clear_override_btn.clicked.connect(self.clear_track_caption_override)
        # MOSS-Audio (open model on a Kaggle GPU)
        self.moss_run_btn.clicked.connect(self.run_moss_captioning)
        self.moss_open_btn.clicked.connect(self.open_last_moss_kernel)
        self.refresh_moss_track_picker()

    def init_ace_step_tab(self, parent):
        """Build the 🅰 ACE-Step (Kaggle) page and connect its actions."""
        from ui.ace_step_tab import build_ace_step_tab
        build_ace_step_tab(self, parent)

        self.caption_selected_btn.clicked.connect(self.caption_selected_track)
        self.caption_missing_btn.clicked.connect(self.caption_missing_tracks)
        self.caption_all_btn.clicked.connect(self.caption_all_tracks)
        self.caption_edit_btn.clicked.connect(self.open_caption_editor)
        self.caption_recaption_bad_btn.clicked.connect(self.caption_recaption_bad_tracks)
        self.caption_diff_btn.clicked.connect(self.show_caption_diff)
        self.caption_import_btn.clicked.connect(self.import_captions_json)

        self.caption_staging_browse_btn.clicked.connect(self.browse_caption_staging)
        self.caption_output_browse_btn.clicked.connect(self.browse_caption_output)
        self.staging_add_btn.clicked.connect(self.staging_add_ticked)
        self.staging_remove_btn.clicked.connect(self.staging_remove_ticked)
        self.staging_clean_btn.clicked.connect(self.staging_clean_unusable)
        self.staging_refresh_btn.clicked.connect(self.refresh_staging_list)

        # The page's ONLY track selector: ticks drive staging, captioning and
        # editing, so the user never has to leave this tab to choose tracks.
        self.ace_track_picker.selection_changed.connect(self.update_ace_tick_status)
        self.refresh_ace_track_picker()

        # Kaggle credentials: stored / session-only / not set, and the controls
        # that change it. The prompt itself lives in _ensure_kaggle_credentials so
        # the run path and this row cannot diverge.
        self.ace_cred_test_btn.clicked.connect(self.test_kaggle_connection)
        self.ace_cred_setup_btn.clicked.connect(self.configure_kaggle_credentials)
        self.ace_cred_forget_btn.clicked.connect(self.forget_stored_kaggle_key)
        self.ace_cred_reset_btn.clicked.connect(self.reset_caption_backend_prompts)
        self.refresh_kaggle_cred_status()

        # Persist the page's settings the same way the other tabs do: one shared
        # handler reads every widget by name.
        for widget, signal in (
            (self.caption_addendum_edit, "textChanged"),
            (self.caption_staging_edit, "textChanged"),
            (self.caption_audio_dataset_edit, "textChanged"),
            (self.caption_model_dataset_edit, "textChanged"),
            (self.caption_output_edit, "textChanged"),
            (self.max_tokens_spin, "valueChanged"),
            (self.max_dur_spin, "valueChanged"),
            (self.batch_size_spin, "valueChanged"),
            (self.caption_bitrate_combo, "currentTextChanged"),
            (self.caption_convert_check, "toggled"),
            (self.caption_batch_review_check, "toggled"),
            (self.caption_whole_song_check, "toggled"),
        ):
            getattr(widget, signal).connect(self.save_pipeline_defaults)

        self.refresh_staging_list()
        # Decide the initial enabled state of every step: with nothing ticked, the
        # staging and caption steps start disabled, with the reason in their
        # tooltips instead of a dialog after the click.
        self.update_ace_actions()

    # -----------------------------------------------------------------------
    # MOSS-Audio (open model) on Kaggle
    # -----------------------------------------------------------------------
    def refresh_moss_track_picker(self):
        """Keep the MOSS track picker in step with the dataset.

        Ticks are keyed by filename and survive the rebuild, so a refresh after
        an add/delete/load does not silently clear the user's selection.
        """
        picker = getattr(self, "moss_track_picker", None)
        if picker is not None:
            picker.set_tracks(self.dataset.get("samples", []))

    def run_moss_captioning(self):
        """Send the ticked tracks to MOSS-Audio on a Kaggle GPU."""
        from workers.kaggle_moss import KaggleMossWorker

        samples = self.dataset.get("samples", [])
        if not samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks first.")
            return

        picked = self.moss_track_picker.selected_filenames()
        if not picked:
            QMessageBox.information(
                self, "No Tracks Selected",
                "Tick the tracks to send to MOSS using the “Tracks” dropdown.",
            )
            return

        by_name = {s.get("filename", ""): s for s in samples}
        paths = []
        missing = []
        for name in picked:
            sample = by_name.get(name)
            path = (sample or {}).get("audio_path", "")
            if path and os.path.exists(path):
                paths.append(path)
            else:
                missing.append(name)
        if not paths:
            QMessageBox.warning(
                self, "Files Not Found",
                "None of the selected tracks exist on disk:\n\n"
                + "\n".join(missing[:10]),
            )
            return
        if missing:
            resp = QMessageBox.question(
                self, "Some Files Missing",
                f"{len(missing)} selected track(s) are missing on disk and will "
                "be skipped:\n\n" + "\n".join(missing[:10])
                + "\n\nContinue with the rest?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                return

        # Persist the backend choices the user just made in the tab.
        self.config["moss_model_id"] = self.moss_model_edit.text().strip()
        self.config["moss_model_dataset"] = \
            self.moss_weights_dataset_edit.text().strip()

        self.moss_run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.moss_status.setText(f"Starting MOSS on {len(paths)} track(s)...")
        self.status_label.setText("Uploading tracks to Kaggle...")

        self.moss_worker = KaggleMossWorker(paths, self.config)
        self.moss_worker.progress.connect(self.on_moss_progress)
        self.moss_worker.finished_ok.connect(self.on_moss_finished)
        self.moss_worker.failed.connect(self.on_moss_failed)
        self.moss_worker.start()

    def on_moss_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.moss_status.setText(msg)
        self.status_label.setText(msg)

    def on_moss_finished(self, results):
        """Write the raw MOSS output into the dataset (backups preserved)."""
        from modules.moss_import import apply_moss_output

        self.moss_run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if not results:
            self.moss_status.setText("MOSS returned no results.")
            return

        report = apply_moss_output(self.dataset, results)
        self.refresh_table()
        self.on_table_selection_changed()
        self.refresh_moss_track_picker()

        if report["matched"] == 0:
            # Say WHY, with names. A silent "0 written" is what made this take
            # several rounds to diagnose: the results were for a different set
            # of tracks than the dataset currently loaded.
            got = list(results)[:3]
            have = [s.get("filename", "?")
                    for s in self.dataset.get("samples", [])][:3]
            QMessageBox.warning(
                self, "MOSS Returned Nothing Usable",
                f"MOSS returned {len(results)} result(s), but none matched the "
                "currently-loaded dataset.\n\n"
                f"MOSS returned:\n    {chr(10) + '    '.join(got)}\n\n"
                f"Dataset has:\n    {chr(10) + '    '.join(have)}\n\n"
                "If those look like different songs, a different dataset was "
                "loaded while the run was in progress — re-run MOSS with the "
                "right dataset open.",
            )
            self.moss_status.setText(
                f"0 of {len(results)} result(s) matched this dataset — see the "
                "dialog for the names."
            )
            return

        self.moss_status.setText(
            f"Done — {report['matched']} track(s) written "
            f"({report['overwritten']} replaced, previous kept in "
            f"caption_before_moss). {report['prompt_override_fixed']} "
            f"prompt_override value(s) normalised to bool."
            + (f" No result for {len(report['no_result'])} track(s)."
               if report["no_result"] else "")
            + (f" {len(report['unknown_results'])} result(s) matched no track."
               if report["unknown_results"] else "")
        )
        self.status_label.setText(
            "MOSS finished. Next: 🎶 Structural Pipeline tab → "
            "✨ Structural Tag Creator."
        )
        QMessageBox.information(
            self, "MOSS Captioning Complete",
            f"Wrote raw style + lyrics for {report['matched']} track(s).\n\n"
            "Next step — format them into the ACE-Step caption and "
            "[Section]-tagged lyrics:\n\n"
            "    🎶 Structural Pipeline tab → ✨ Structural Tag Creator\n\n"
            "That step uses an LLM. If none is configured, open "
            "⚙ Settings → LLM Provider and pick one (Groq, Gemini and "
            "OpenRouter all have free tiers).\n\n"
            "Nothing was destroyed — previous values are in "
            "caption_before_moss / raw_lyrics_before_moss.",
        )

    def on_moss_failed(self, err):
        self.moss_run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.moss_status.setText(f"Failed: {err}")
        self.status_label.setText("MOSS captioning failed.")
        QMessageBox.critical(self, "MOSS Captioning Failed", str(err))

    def open_last_moss_kernel(self):
        """Open the most recent MOSS kernel's output page in Kaggle."""
        import webbrowser
        from workers.kaggle_moss import LAST_KERNEL_REF

        if not LAST_KERNEL_REF:
            self.moss_status.setText(
                "No MOSS run yet this session — nothing to open."
            )
            return
        webbrowser.open_new_tab(f"https://www.kaggle.com/code/{LAST_KERNEL_REF}/output")

    def on_caption_blend_changed(self, value):
        """Dataset-wide prose/tags blend ratio (persisted with other defaults)."""
        self.config["tag_caption_ratio"] = int(value)
        self.status_label.setText(
            f"Dataset caption blend: {100 - int(value)}% prose / {int(value)}% tags."
        )

    def caption_selected_track(self):
        """Caption the tracks ticked in this page's Tracks dropdown."""
        if not self._ticked_samples():
            self._no_tracks_ticked()
            return
        self.start_ai_captioning(scope="ticked")

    def caption_missing_tracks(self):
        """Run the captioner on every track that has no caption yet."""
        self.start_ai_captioning(scope="missing")

    def caption_all_tracks(self):
        """Re-caption every track after confirmation."""
        self.start_ai_captioning(scope="all")

    def open_caption_editor(self):
        """Edit the ticked track's caption / lyrics without leaving the tab.

        Several ticked → the dataset-wide diff table, because opening one editor
        for the "first" of a multi-track selection silently ignores the rest.
        """
        ticked = self._ticked_samples()
        if not ticked:
            self._no_tracks_ticked()
            return
        if len(ticked) > 1:
            self.show_caption_diff()
            return
        self.review_ai_caption_result(ticked[0])

    # -----------------------------------------------------------------------
    # ACE-Step (Kaggle): staging folder, output folder, diff review
    # -----------------------------------------------------------------------
    def caption_batch_review_enabled(self):
        """True when a run's captions are held as proposals for ONE diff review."""
        box = getattr(self, "caption_batch_review_check", None)
        if box is not None:
            return box.isChecked()
        return bool(self.config.get("caption_batch_review", True))

    def _ticked_samples(self):
        """The samples ticked in the ACE-Step page's *Tracks ▾* dropdown.

        This is the ONLY selection source for that page. The Studio table's ROW
        selection is not a track picker — telling the user to go and set one there
        for work that happens in the Caption tab was the wrong UI (and row
        selection is not "a place to add tracks" at all). Ticks are keyed by
        FILENAME, so a track removed from the dataset simply drops out instead of
        shifting every later tick onto the wrong song.
        """
        picker = getattr(self, "ace_track_picker", None)
        if picker is None:
            return []
        wanted = set(picker.selected_filenames())
        if not wanted:
            return []
        return [
            sample for sample in self.dataset.get("samples", [])
            if (sample.get("filename") or "").strip() in wanted
        ]

    def refresh_ace_track_picker(self):
        """Keep the ACE-Step track picker in step with the dataset."""
        picker = getattr(self, "ace_track_picker", None)
        if picker is not None:
            picker.set_tracks(self.dataset.get("samples", []))
        self.update_ace_tick_status()

    def update_ace_tick_status(self):
        """Always say how many tracks are ticked: an empty selection must be visible."""
        # Ticks are an INPUT to staging/captioning/editing, so every refresh of this
        # label also re-decides which step buttons are live. Guarded by NAME: a
        # partial manager (the tests' stubs, a future extraction) may not carry the
        # updater, and reaching for it unguarded is how AttributeError reached the
        # user on a click.
        updater = getattr(self, "update_ace_actions", None)
        if updater is not None:
            updater()
        label = getattr(self, "ace_tick_status", None)
        if label is None:
            return
        total = len(self.dataset.get("samples", []))
        ticked = len(self._ticked_samples())
        if not total:
            label.setText(
                "No tracks loaded yet — open your dataset with 📂 Open, or "
                "add songs in 🎛 Dataset Studio. "
                "(The app starts empty: it does not reopen the dataset you had "
                "open last time.)"
            )
        elif not ticked:
            label.setText(
                "Nothing ticked — use the “Tracks ▾” dropdown above to choose which "
                "tracks to stage and caption."
            )
        else:
            label.setText(
                f"{ticked} of {total} ticked — staging, captioning and editing act "
                "on these tracks."
            )

    def _no_tracks_ticked(self):
        """Ask for a tick, in place. No modal sending the user to another tab."""
        self.update_ace_tick_status()
        QMessageBox.information(
            self, "No Tracks Ticked",
            "Tick the tracks to work on with the “Tracks ▾” dropdown on this page.\n\n"
            "“Select tracks missing captions” ticks only the ones still to do.",
        )

    def _caption_run_samples(self):
        """The samples the last caption run covered."""
        wanted = set(self._caption_scope_ids)
        samples = self.dataset.get("samples", [])
        if not wanted:
            return samples
        return [s for s in samples if s.get("id") in wanted]

    def _proposal_rows(self):
        """Diff rows built from the proposals already stored on the samples.

        ``caption_ai_raw`` is where a proposal lives until it is accepted, so the
        table can be reopened (or rebuilt after a restart) without the worker.
        """
        from modules import caption_kaggle_run as ckr

        samples = self._caption_run_samples()
        results = {}
        for sample in samples:
            raw = (sample.get("caption_ai_raw") or "").strip()
            if raw:
                results[os.path.basename(sample.get("filename", ""))] = raw
        return ckr.diff_captions(
            samples, results,
            convert_mp3=bool(self.config.get("caption_convert_mp3", True)),
        )

    def _bad_caption_samples(self):
        """Tracks that need captioning again: blank, errored, or never returned.

        Three separate failure modes that all look like "done" in the table unless
        they are collected together:
          * no caption at all;
          * the kernel reported ERROR for the track;
          * the last run returned NOTHING for it, which leaves the caption blank
            and marks nothing.
        """
        from modules import caption_kaggle_run as ckr

        bad_ids = set()
        if self._caption_scope_ids:
            bad_ids = {
                row["id"] for row in self._proposal_rows()
                if row["status"] in (ckr.STATUS_MISSING, ckr.STATUS_ERROR)
            }
        return [
            sample for sample in self.dataset.get("samples", [])
            if not (sample.get("caption") or "").strip()
            or (sample.get("caption_ai_raw") or "").strip().upper().startswith("ERROR:")
            or sample.get("id") in bad_ids
        ]

    def caption_recaption_bad_tracks(self):
        """Re-caption only the tracks that need it again."""
        bad = self._bad_caption_samples()
        if not bad:
            QMessageBox.information(
                self, "Nothing To Redo",
                "Every track has a caption and the last run reported no errors.",
            )
            return
        self.status_label.setText(f"Re-captioning {len(bad)} track(s) that need it…")
        self.start_ai_captioning(scope="bad")

    # -- Kaggle credentials -------------------------------------------------
    def refresh_kaggle_cred_status(self):
        """Show where the Kaggle key actually lives, in the ACE-Step page.

        Two places, on purpose: the full sentence in ⚙ Settings, and the SHORT
        state on the strip chip, so "which key is this?" is answerable without
        opening anything.
        """
        from modules.secrets_manager import get_secret

        user = (self.config.get("kaggle_user") or "").strip()
        key = (self.config.get("kaggle_key") or "").strip()
        if not user or not key:
            short = "✗ not set"
            long = "Kaggle credentials: <b>not set</b> — a run will ask for them."
        elif get_secret("kaggle_key"):
            short = "✓"
            long = (
                f"Kaggle credentials: <b>stored</b> for <b>{user}</b> "
                "(OS keyring / encrypted store)."
            )
        else:
            short = "session"
            long = (
                f"Kaggle credentials: <b>this session only</b> for <b>{user}</b> "
                "— nothing was written to disk."
            )
        label = getattr(self, "ace_cred_status", None)
        if label is not None:
            label.setText(long)
        chip = getattr(self, "ace_cred_test_btn", None)
        if chip is not None:
            chip.setText(f"🔌 Kaggle {short}")

        # The last PROBE verdict, when one has been run. Where the key lives and
        # whether Kaggle accepts it are different questions, and only the second
        # one predicts whether a run will work -- so the row says both instead of
        # implying health from the presence of two strings.
        verdict = getattr(self, "_kaggle_probe_verdict", "")
        if verdict:
            label.setText(label.text() + "<br>" + verdict)

    def _ensure_kaggle_credentials(self):
        """Ask for Kaggle credentials. Returns True when they are set.

        A dialog rather than two chained ``QInputDialog``s, because the
        "remember on this device" choice has to be made WHERE THE KEY IS ENTERED:
        a user who does not want the key stored should not have to hunt through
        ⚙ Settings afterwards to untick something. Unticked means the key stays in
        this session's config and is never written anywhere.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Kaggle Credentials")
        dialog.resize(520, 260)
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "The ACE-Step captioner runs on a free Kaggle GPU, so it needs your "
            "Kaggle username and API key (kaggle.com → Settings → API)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        user_edit = QLineEdit((self.config.get("kaggle_user") or "").strip())
        key_edit = QLineEdit((self.config.get("kaggle_key") or "").strip())
        key_edit.setEchoMode(QLineEdit.Password)
        key_edit.setToolTip("Shown as dots. The value is never logged.")
        remember = QCheckBox("Remember the key on this device (encrypted store)")
        remember.setChecked(bool(self.config.get("remember_kaggle_key", True)))
        remember.setToolTip(
            "Ticked: stored in the OS keyring (or the Fernet-encrypted secrets "
            "file).\nUnticked: used for this session only — nothing is written to "
            "disk, and any previously stored Kaggle key is deleted."
        )
        form.addRow("Kaggle username:", user_edit)
        form.addRow("Kaggle API key:", key_edit)
        form.addRow("", remember)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("✅ Use these credentials")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addStretch()
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.Accepted:
            return False
        user = user_edit.text().strip()
        key = key_edit.text().strip()
        if not user or not key:
            QMessageBox.warning(
                self, "Both Fields Needed",
                "A Kaggle username AND API key are required.",
            )
            return False

        self.config["kaggle_user"] = user
        self.config["kaggle_key"] = key
        self.config["remember_kaggle_key"] = remember.isChecked()
        # Keep ⚙ Settings in step so the two views cannot disagree.
        k_user = getattr(self, "k_user", None)
        k_key = getattr(self, "k_key", None)
        remember_box = getattr(self, "remember_kaggle", None)
        if k_user is not None:
            k_user.setText(user)
        if k_key is not None:
            k_key.setText(key)
        if remember_box is not None:
            remember_box.setChecked(remember.isChecked())
        try:
            save_config(self.config, remember=self._remembered_secret_keys())
        except Exception as e:  # noqa: BLE001
            # Never lose the run over a storage failure: the key is already in the
            # in-memory config the worker reads.
            print(f"save_config failed for Kaggle credentials: {e}")
        self.refresh_kaggle_cred_status()
        return True

    def configure_kaggle_credentials(self):
        """The credentials row's "Set up / change…" button."""
        if self._ensure_kaggle_credentials():
            self.status_label.setText("Kaggle credentials ready.")

    # -- Kaggle connectivity -------------------------------------------------
    def test_kaggle_connection(self):
        """The credentials row's "Test connection" button.

        Runs on a worker thread because it makes a real network call: token
        introspection plus one authenticated API call. The verdict is written
        back into the credentials row, so it does not vanish with a dialog.
        """
        from workers.kaggle_probe import KaggleProbeWorker

        user = (self.config.get("kaggle_user") or "").strip()
        key = (self.config.get("kaggle_key") or "").strip()
        if not user or not key:
            QMessageBox.warning(
                self, "Kaggle Credentials Needed",
                "Enter your Kaggle username and API key first "
                "(🔑 Set up / change…).",
            )
            return

        self.ace_cred_test_btn.setEnabled(False)
        self._kaggle_probe_verdict = "Testing the Kaggle connection…"
        self.refresh_kaggle_cred_status()
        self.kaggle_probe_worker = KaggleProbeWorker(self.config)
        self.kaggle_probe_worker.done.connect(self._on_kaggle_probe_done)
        self.kaggle_probe_worker.failed.connect(self._on_kaggle_probe_failed)
        self.kaggle_probe_worker.start()

    def _kaggle_probe_verdict_text(self, result):
        """One-line summary of a probe result, for the credentials row."""
        if result.get("ok"):
            method = (result.get("auth_method") or "").strip()
            return (
                f"<b>Connected</b> as <b>{result.get('username') or '?'}</b>"
                + (f" (auth: {method})" if method else "")
            )
        detail = (result.get("detail") or "").strip()
        return f"<b>Not connected</b> — {detail or 'no reason reported'}"

    def _on_kaggle_probe_done(self, result):
        """Report the verdict in the row, and explain any failure in a dialog."""
        self.ace_cred_test_btn.setEnabled(True)
        self._kaggle_probe_verdict = self._kaggle_probe_verdict_text(result)
        self.refresh_kaggle_cred_status()

        problems = [str(row) for row in (result.get("problems") or [])]
        if result.get("ok") and not problems:
            self.status_label.setText(
                f"Kaggle connection OK (user {result.get('username')})."
            )
            return

        lines = []
        if result.get("detail"):
            lines.append(str(result["detail"]))
        lines.extend(problems)
        if not lines:
            lines.append("Kaggle did not accept these credentials.")
        QMessageBox.warning(self, "Kaggle Connection Failed", "\n\n".join(lines))

    def _on_kaggle_probe_failed(self, err_msg):
        """The probe could not even run — surface it instead of staying silent."""
        self.ace_cred_test_btn.setEnabled(True)
        self._kaggle_probe_verdict = f"<b>Not connected</b> — {err_msg}"
        self.refresh_kaggle_cred_status()
        QMessageBox.warning(self, "Kaggle Connection Failed", err_msg)

    def forget_stored_kaggle_key(self):
        """Delete the stored Kaggle key and fall back to session-only."""
        from modules.secrets_manager import delete_secret, get_secret

        if not get_secret("kaggle_key") and not (self.config.get("kaggle_key") or "").strip():
            QMessageBox.information(
                self, "Nothing Stored", "No Kaggle key is stored on this device."
            )
            return
        confirm = QMessageBox.question(
            self, "Forget Stored Kaggle Key",
            "Delete the stored Kaggle API key from this device?\n\n"
            "The key stays usable for this session; after a restart you will be "
            "asked for it again.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        delete_secret("kaggle_key")
        self.config["remember_kaggle_key"] = False
        try:
            save_config(self.config, remember=self._remembered_secret_keys())
        except Exception as e:  # noqa: BLE001
            print(f"save_config failed while forgetting the Kaggle key: {e}")
        self.refresh_kaggle_cred_status()
        self.status_label.setText(
            "Stored Kaggle key deleted — it remains in use for this session only."
        )

    def reset_caption_backend_prompts(self):
        """Forget the once-per-backend credential decision, so it asks again."""
        self.config["caption_cred_prompt_seen"] = []
        self.config["caption_fallback_backend"] = ""
        try:
            save_config(self.config, remember=self._remembered_secret_keys())
        except Exception as e:  # noqa: BLE001
            print(f"save_config failed while resetting the prompt memory: {e}")
        self.status_label.setText(
            "The captioner will ask about Kaggle credentials again on the next run."
        )

    # Backends that never HEAR the audio. Their output is a placeholder, not a
    # caption, and is stamped as one (see _stamp_placeholder_caption).
    PLACEHOLDER_CAPTION_BACKENDS = ("Local Rule Engine", "DeepSeek Cloud")

    def _fallback_backend_options(self):
        """The fallbacks to offer, each with whether it can actually run."""
        try:
            from modules.llm_client import provider_key_present
            llm_key = provider_key_present(self.config)
        except Exception:  # noqa: BLE001
            llm_key = False
        return [
            ("DeepSeek LLM — text-only draft, NO audio is sent",
             "DeepSeek Cloud", bool(llm_key)),
            ("Google Gemini — audio-native (needs a Gemini key)",
             "Gemini", bool(self.config.get("gemini_api_key"))),
            ("Custom endpoint — audio only if it supports it",
             "Custom Endpoint / Webhook",
             bool((self.config.get("custom_caption_url") or "").strip())),
            ("Local rule engine — canned template text, no model",
             "Local Rule Engine", True),
        ]

    def _prompt_fallback_backend(self):
        """Let the user CHOOSE the fallback. Returns the backend, or '' to abort.

        Offered as a LIST rather than picked silently: "ACE-Step is unavailable"
        says nothing about which of four very different engines should run
        instead, and each one produces a different KIND of text.
        """
        options = self._fallback_backend_options()
        labels = [
            label + ("" if usable else "   (not configured)")
            for label, _backend, usable in options
        ]
        choice, accepted = QInputDialog.getItem(
            self, "Which Backend Should Run Instead?",
            "No Kaggle credentials, so the ACE-Step captioner cannot run.\n\n"
            "Pick what should caption these tracks instead. Only the audio-native "
            "options produce real captions; the others are stamped as placeholders:",
            labels, 0, False,
        )
        if not accepted or not choice:
            return ""
        return options[labels.index(choice)][1]

    def _resolve_caption_backend(self):
        """Pick the caption backend, asking about credentials ONCE per backend.

        ``resolve_backend()`` on its own is a SILENT degradation: ``ace_step``
        with no Kaggle key falls through to DeepSeek (text-only, no audio) and
        then to the local rule engine (canned template text). A page titled
        "ACE-Step (Kaggle)" could therefore write placeholder captions that look
        entirely real. This asks first, and never picks a fallback on the user's
        behalf.

        Credentials always win over the memory: if a key is present — now or later
        — Kaggle is used without asking again.
        """
        configured = (self.config.get("caption_backend") or "ace_step").strip().lower()
        if configured not in ("ace_step", "moss"):
            return resolve_backend(self.config)      # an explicit choice: honour it

        user = (self.config.get("kaggle_user") or "").strip()
        key = (self.config.get("kaggle_key") or "").strip()
        if user and key:
            return "Kaggle Cloud (Free GPU)"

        seen = [str(name) for name in (self.config.get("caption_cred_prompt_seen") or [])]
        if configured in seen:
            remembered = (self.config.get("caption_fallback_backend") or "").strip()
            if remembered:
                return remembered
            # "Declined" was recorded without a backend: ask again rather than
            # guess which engine the user wants.

        if self._ensure_kaggle_credentials():
            return "Kaggle Cloud (Free GPU)"

        chosen = self._prompt_fallback_backend()
        if not chosen:
            return ""                                # cancelled: the caller aborts
        seen.append(configured)
        self.config["caption_cred_prompt_seen"] = sorted(set(seen))
        self.config["caption_fallback_backend"] = chosen
        try:
            save_config(self.config, remember=self._remembered_secret_keys())
        except Exception as e:  # noqa: BLE001
            print(f"save_config failed while remembering the caption fallback: {e}")
        return chosen

    def _stamp_placeholder_caption(self, sample, backend):
        """Mark a caption that was produced WITHOUT hearing the audio.

        Exported as metadata, never as a prefix inside the caption text: a stamp
        inside the text would become training data.
        """
        if not self.config.get("caption_stamp_placeholders", True):
            return
        if backend not in self.PLACEHOLDER_CAPTION_BACKENDS:
            return
        why = ("no audio was heard — canned template text"
               if backend == "Local Rule Engine"
               else "no audio was sent — filename-only draft")
        sample["caption_is_placeholder"] = True
        sample["caption_ai_model"] = f"{backend} (PLACEHOLDER — {why})"

    # -- folders -----------------------------------------------------------
    def browse_caption_staging(self):
        """Pick the local folder whose contents are uploaded to Kaggle."""
        from modules import caption_kaggle_run as ckr

        current = self.caption_staging_edit.text().strip() or ckr.default_staging_dir()
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Caption Staging Folder", current
        )
        if chosen:
            self.caption_staging_edit.setText(chosen)
            self.save_pipeline_defaults()
            self.refresh_staging_list()

    def browse_caption_output(self):
        """Pick the local folder the downloaded captions land in."""
        from modules import caption_kaggle_run as ckr

        current = self.caption_output_edit.text().strip() or ckr.default_output_dir()
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Caption Output Folder", current
        )
        if chosen:
            self.caption_output_edit.setText(chosen)
            self.save_pipeline_defaults()

    # -- staging folder contents (add / remove songs) -----------------------
    def _staging_rows(self):
        """One row per staged file: ``(path, staged name, dataset filename or "")``.

        The two names differ whenever a track is transcoded — the dataset holds
        ``aint_no_fun.flac``, the staging folder holds ``aint_no_fun.mp3`` — so the
        same song appeared under two different names, in two different places, with
        nothing connecting them. That is what made "I ticked it, why is it not
        ticked?" the inevitable question.
        """
        from modules import caption_kaggle_run as ckr

        convert = bool(self.config.get("caption_convert_mp3", True))
        source_of = {}
        for sample in self.dataset.get("samples", []):
            name = (sample.get("filename") or "").strip()
            if name:
                source_of.setdefault(ckr.staged_name(name, convert), name)
        rows = []
        for path in ckr.staged_files(ckr.staging_dir(self.config)):
            staged = os.path.basename(path)
            rows.append((path, staged, source_of.get(staged, "")))
        return rows

    def _selected_staging_names(self):
        """Staged filenames selected in the list — the names the FOLDER knows."""
        names = []
        for item in self.staging_list.selectedItems():
            staged = item.data(Qt.UserRole) or item.text()
            if staged and staged not in names:
                names.append(staged)
        return names

    def refresh_staging_list(self):
        """Show what the next run would upload, and what it would leave behind."""
        from modules import caption_kaggle_run as ckr

        folder = ckr.staging_dir(self.config)
        report = ckr.staging_report(folder)
        self.staging_list.clear()
        for path, staged, source in self._staging_rows():
            # The DISPLAYED text names the dataset track this file came from; the
            # name the folder needs is stored as item data, because handing the
            # label to the filesystem is one rename away from deleting the wrong
            # file.
            label = (f"{staged}   ←   {source}" if source
                     else f"{staged}   (not in this dataset)")
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, staged)
            try:
                item.setToolTip(f"{path}\n{os.path.getsize(path) / 1e6:.1f} MB")
            except OSError:
                item.setToolTip(path)
            self.staging_list.addItem(item)
        usable = len(report["usable"])
        ignored = report["ignored"]
        if usable:
            text = (f"{usable} file(s) staged in {folder} — all of them are "
                    "uploaded on the next run.")
        else:
            text = f"Nothing staged yet in {folder}. Use “➕ Stage”."
        if ignored:
            # The list filters unusable files OUT, so without this line the folder
            # and the list disagree silently — which is how 32 zero-byte files sat
            # in this folder while the page said "1 song".
            text += (f" ⚠ {len(ignored)} unusable file(s) ignored: "
                     + ", ".join(f"{name} ({why})" for name, why in ignored[:3]))
            if len(ignored) > 3:
                text += f" …+{len(ignored) - 3} more"
        self.staging_count_label.setText(text)
        self.update_ace_actions()

    def staging_add_ticked(self):
        """Copy (or transcode) the TICKED dataset tracks into the staging folder."""
        from modules import caption_kaggle_run as ckr

        samples = self._ticked_samples()
        if not samples:
            self._no_tracks_ticked()
            return
        convert = self.caption_convert_check.isChecked()
        if convert and not ckr.ffmpeg_available():
            QMessageBox.information(
                self, "ffmpeg Not Found",
                "ffmpeg is not on PATH, so the files will be staged at their "
                "original quality instead of MP3.",
            )
        items = [
            (s.get("id"), s.get("filename", ""), s.get("audio_path", ""))
            for s in samples
        ]
        staged, skipped = ckr.stage_tracks(
            items, ckr.staging_dir(self.config),
            convert_mp3=convert, bitrate=self.caption_bitrate_combo.currentText(),
        )
        self.refresh_staging_list()
        note = f"Staged {len(staged)} track(s) for upload."
        if skipped:
            note += f" {skipped} had no usable audio file."
        self.status_label.setText(note)

    def staging_remove_ticked(self):
        """Delete the TICKED tracks from the staging folder (= from the upload).

        "Ticked" now means the same thing here as it does on ➕ Stage: the ticks in
        the “Tracks ▾” dropdown. Before, Stage used the ticks while Remove used the
        staging list's ROW selection — so ticking a song and pressing Remove said
        "Nothing Ticked" about a song that was plainly ticked. A real dataset's
        filenames (.flac) also differ from the staged names (.mp3), which made the
        two views impossible to line up by eye.

        The list selection still works too, because a staged file can belong to no
        track at all (a leftover from an older dataset) and otherwise there would be
        no way to remove it.
        """
        from modules import caption_kaggle_run as ckr

        convert = bool(self.config.get("caption_convert_mp3", True))
        names = self._selected_staging_names()
        for sample in self._ticked_samples():
            staged = ckr.staged_name(sample.get("filename", ""), convert)
            if staged not in names:
                names.append(staged)
        if not names:
            QMessageBox.warning(
                self, "Nothing Ticked",
                "Tick tracks in “Tracks ▾” (or click a file in the staged list) "
                "first — Remove deletes what is selected there.",
            )
            return
        removed = ckr.remove_staged(ckr.staging_dir(self.config), names)
        self.refresh_staging_list()
        self.status_label.setText(
            f"Removed {removed} file(s) from the staging folder. The next run "
            "uploads a new version of the Kaggle dataset without them."
        )

    # -- diff review (existing vs proposed) ---------------------------------
    def _sample_by_id(self, sid):
        for sample in self.dataset.get("samples", []):
            if sample.get("id") == sid:
                return sample
        return None

    def show_caption_diff(self, rows=None):
        """Diff existing vs proposed captions for a whole run and apply choices.

        This is the "run a diff on an existing caption, then choose which one to
        use" step. Nothing is written to ``caption`` until Apply: a track with no
        existing caption is ADDED by "Use new", and a replaced caption is kept in
        ``caption_before_kaggle`` so an approved caption is recoverable.
        """
        from modules import caption_kaggle_run as ckr

        rows = self._proposal_rows() if rows is None else rows
        if not rows:
            QMessageBox.information(
                self, "Nothing To Review",
                "There are no caption proposals yet. Run the ACE-Step captioner, "
                "or import a captions_out.json.",
            )
            return
        if not self._caption_scope_ids:
            # An imported file: the shown rows ARE the run's scope, so a follow-up
            # "Re-caption bad / failed" targets the right tracks.
            self._caption_scope_ids = [row["id"] for row in rows]

        counts = ckr.count_by_status(rows)
        dialog = QDialog(self)
        dialog.setWindowTitle("Review Captions — existing vs new")
        dialog.resize(1180, 680)
        layout = QVBoxLayout(dialog)

        summary = QLabel(
            f"{len(rows)} track(s): "
            + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
            + ". Nothing is written until you press Apply. “new” rows have no "
            "caption at all, so using one ADDS it."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableWidget(len(rows), 5)
        table.setHorizontalHeaderLabels(
            ["Track", "Status", "Existing caption", "New caption", "Decision"]
        )
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        edited = {}
        combos = []
        for index, row in enumerate(rows):
            table.setItem(index, 0, QTableWidgetItem(row["filename"]))
            table.setItem(index, 1, QTableWidgetItem(row["status"]))
            table.setItem(index, 2, QTableWidgetItem(row["existing"] or "(no caption)"))
            table.setItem(index, 3, QTableWidgetItem(row["proposed"] or "(nothing returned)"))
            combo = QComboBox()
            for label, value in (
                ("Keep existing", "keep"),
                ("Use new", "use"),
                ("Edit…", "edit"),
                ("Skip", "skip"),
            ):
                combo.addItem(label, value)
            # Defaults that need no thought: nothing to use -> keep; a proposal
            # with no existing caption -> use (that IS the add case).
            if not row["proposed"]:
                combo.setCurrentIndex(0)
                combo.setEnabled(False)
            elif row["status"] == ckr.STATUS_NEW:
                combo.setCurrentIndex(1)
            combo.currentIndexChanged.connect(
                lambda _i, r=index: self._on_diff_decision_changed(
                    dialog, rows[r], table.cellWidget(r, 4), edited
                )
            )
            table.setCellWidget(index, 4, combo)
            combos.append(combo)
        layout.addWidget(table)

        def set_all(value):
            for combo in combos:
                if not combo.isEnabled():
                    continue
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)

        def apply_choices():
            self.record_snapshot()
            applied, skipped = 0, 0
            for row, combo in zip(rows, combos):
                decision = combo.currentData()
                if decision == "skip" or not combo.isEnabled():
                    skipped += 1
                    continue
                sample = self._sample_by_id(row["id"])
                if sample is None:
                    continue
                ckr.apply_decision(sample, row, decision, edited.get(row["id"]))
                applied += 1
            self.refresh_table()
            self.on_table_selection_changed()
            self.status_label.setText(
                f"Applied {applied} caption decision(s); {skipped} left unchanged."
            )
            dialog.accept()

        buttons = QHBoxLayout()
        use_all = QPushButton("Use new for all")
        use_all.setToolTip("Take the new caption everywhere one was proposed.")
        use_all.clicked.connect(lambda: set_all("use"))
        keep_all = QPushButton("Keep all existing")
        keep_all.setToolTip(
            "Change nothing — the proposals stay on the tracks as caption_ai_raw."
        )
        keep_all.clicked.connect(lambda: set_all("keep"))
        apply_btn = QPushButton("✅ Apply")
        apply_btn.setToolTip(
            "Write the chosen captions. A replaced caption is kept in "
            "caption_before_kaggle, and a snapshot is recorded first, so this is "
            "undoable."
        )
        apply_btn.clicked.connect(apply_choices)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(use_all)
        buttons.addWidget(keep_all)
        buttons.addStretch()
        buttons.addWidget(apply_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        dialog.exec()

    def _on_diff_decision_changed(self, dialog, row, combo, edited):
        """Handle “Edit…” in the diff table: collect the text, keep the rest."""
        if combo is None or combo.currentData() != "edit":
            return
        seed = edited.get(row["id"]) or row["proposed"] or row["existing"]
        text, ok = QInputDialog.getMultiLineText(
            dialog, "Edit Caption", f"{row['filename']} — caption to apply:", seed
        )
        if ok and text.strip():
            edited[row["id"]] = text.strip()
        else:
            # Cancelled or emptied: fall back to a decision that cannot lose the
            # caption, rather than leaving "edit" selected with nothing to write.
            index = combo.findData("keep" if row["existing"] else "use")
            if index >= 0:
                combo.setCurrentIndex(index)

    def import_captions_json(self):
        """Diff a captions_out.json without running the kernel again."""
        from modules import caption_kaggle_run as ckr

        start = self.caption_output_edit.text().strip() or ckr.default_output_dir()
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import captions_out.json", start, "Caption results (*.json)"
        )
        if not path:
            return
        try:
            results = ckr.load_results(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Could Not Read Captions", str(exc))
            return
        samples = self.dataset.get("samples", [])
        rows = ckr.diff_captions(
            samples, results,
            convert_mp3=bool(self.config.get("caption_convert_mp3", True)),
        )
        self._caption_scope_ids = [row["id"] for row in rows]
        unmatched = ckr.unmatched_results(samples, results)
        note = f"Imported {len(results)} caption(s) from {os.path.basename(path)}."
        if unmatched:
            note += (f" {len(unmatched)} matched no track in this dataset: "
                     + ", ".join(unmatched[:3]))
        self.ace_status_label.setText(note)
        self.show_caption_diff(rows)

    def set_track_caption_override(self):
        """Store a per-track blend ratio on the selected sample."""
        s = self.get_selected_sample()
        if not s:
            QMessageBox.warning(self, "No Track Selected", "Select a track in the Dataset Studio table first.")
            return
        current = s.get("prompt_override")
        default = self.caption_blend_slider.value() if current is None else int(current)
        value, ok = QInputDialog.getInt(
            self, "Track Caption Blend",
            f"Tags ratio for '{s.get('filename', '?')}' (0 = prose, 100 = tags):",
            default, 0, 100,
        )
        if not ok:
            return
        self.record_snapshot()
        s["prompt_override"] = int(value)
        self.status_label.setText(
            f"Track override set: {100 - int(value)}% prose / {int(value)}% tags "
            f"for '{s.get('filename', '?')}'."
        )

    def clear_track_caption_override(self):
        """Drop the per-track override so the track follows the dataset value."""
        s = self.get_selected_sample()
        if not s:
            QMessageBox.warning(self, "No Track Selected", "Select a track in the Dataset Studio table first.")
            return
        if s.get("prompt_override") is None:
            self.status_label.setText("This track already follows the dataset caption blend.")
            return
        self.record_snapshot()
        s["prompt_override"] = None
        self.status_label.setText(f"Cleared caption override for '{s.get('filename', '?')}'.")

    # -----------------------------------------------------------------------
    # AI Assistant Tab
    # -----------------------------------------------------------------------
    def init_assistant_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        from ui.assistant_provider import build_provider_row

        prov_row, prov_state = build_provider_row(self)
        layout.addLayout(prov_row)
        layout.addWidget(prov_state)

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
        self.assistant_status.setProperty("muted", True)
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
        summary = summarize_dataset(self.dataset)
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
                return summarize_dataset(self.dataset) or "(dataset empty)"
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
                # The health audit was removed (can be re-added later as a module).
                return (
                    "The health audit (Scan & Fill) is not available in this build — "
                    "it was removed and can be re-added later as a module."
                )
            if name == "detect_instruments":
                # Instrument detection runs through the Structural pipeline
                # (workers/tagger.py). There is no standalone trigger on the
                # live DatasetManager, so point the user at the real control
                # instead of claiming detection already started.
                return (
                    "Instrument detection runs as part of the Structural pipeline. "
                    "Select a track and run 'Structural' (or use Detect via Captioner) "
                    "to populate instrument tags."
                )
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
        self.tag_manager_status.setProperty("muted", True)
        layout.addWidget(self.tag_manager_status)

        self.refresh_tag_manager()

    def _install_shell(self, parts, pages):
        """Assemble the dock layout (see ui/shell.py)."""
        from ui.shell import flow_group, install_shell

        lyrics_page = QWidget()
        lv = QVBoxLayout(lyrics_page)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(flow_group("Transcribe", parts["lyrics_strip"]))
        lv.addWidget(pages["lyrics"], 1)

        tags_page = QWidget()
        tv = QVBoxLayout(tags_page)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(flow_group("Import && fix", parts["audit_strip"]))
        tv.addWidget(flow_group("Look up", parts["advanced_strip"]))
        tv.addWidget(flow_group("Edit tools", parts["tools_strip"]))
        tv.addWidget(pages["organize"], 1)
        self._shell_extra_pages = [lyrics_page, tags_page]

        tool_pages = [
            ("Caption", pages["caption"]),
            ("Lyrics", lyrics_page),
            ("Structure", pages["structure"]),
            ("Tags && checks", tags_page),
        ]
        install_shell(self, parts, tool_pages, pages["assistant"], pages["settings"])
        # _on_tab_changed compares against these; they now index the Tools dock.
        self.lyrics_tab_index = 1
        self.tag_tab_index = 3
        self.embed_tab_index = 3

    def _on_tab_changed(self, index):
        if index == self.tag_tab_index and hasattr(self, "tag_stats_table"):
            self.refresh_tag_manager()
        # Auto-populate the Lyrics tab so it shows the selected track on arrival
        # instead of requiring a Preview click. Uses a remembered row, so the
        # selection survives the tab switch.
        if hasattr(self, "lyrics_tab_index") and index == self.lyrics_tab_index:
            self.preview_lyrics_tidy(silent=True)
            self.load_all_lyrics(silent=True)

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
        self.embed_backend_label.setProperty("muted", True)
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
        self.embed_status.setProperty("muted", True)
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
        self.exp_status.setProperty("muted", True)
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

        group = QGroupBox("🧠 Advanced Structural Pipeline (LLM + Librosa)")
        inner = QVBoxLayout(group)

        self.advanced_pipeline_btn = QPushButton("🚀 Run Advanced Pipeline on Selected Track")
        self.advanced_pipeline_btn.clicked.connect(self.trigger_advanced_ai_pipeline)
        inner.addWidget(self.advanced_pipeline_btn)

        info = QLabel(
            "Uses Librosa to segment the selected track into ~9 macro sections,\n"
            "exports each as WAV to a 'structural_slices' folder, and then\n"
            "aggregates via the configured LLM (Groq by default) into a master caption."
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        inner.addWidget(info)

        layout.addWidget(group)
        layout.addStretch()

    # -----------------------------------------------------------------------
    # Structural Pipeline Tab
    # -----------------------------------------------------------------------
    def init_structural_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox("🎶 Structural Pipeline (Standard)")
        inner = QVBoxLayout(group)

        info = QLabel(
            "This pipeline separates stems (import or MVSEP), finds structural boundaries,\n"
            "captions each section per stem, and aggregates via the configured LLM into a\n"
            "master caption for the whole track."
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
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
        self.mvsep_gui_button = QPushButton(
            "🎛 Configure / Test MVSEP"
        )
        self.mvsep_gui_button.clicked.connect(self.open_mvsep_dialog)
        inner.addWidget(self.mvsep_gui_button)
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
        disclaimer.setProperty("tone", "warning"); disclaimer.setProperty("small", True)
        sep_layout2.addWidget(disclaimer)




        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Additional models to run:"))
        self.extra_models_input = QLineEdit()
        self.extra_models_input.setPlaceholderText("e.g., MVSep Organ, MVSep Harpsichord")
        model_layout.addWidget(self.extra_models_input)
        sep_layout2.addLayout(model_layout)

        note = QLabel("You can also manually type additional MVSEP model names above.")
        note.setProperty("muted", True); note.setProperty("small", True)
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

        # LLM aggregation toggle (attribute name kept for compatibility)
        self.struct_deepseek_check = QCheckBox("Use the LLM for aggregation")
        self.struct_deepseek_check.setChecked(True)
        inner.addWidget(self.struct_deepseek_check)

        # Run button
        self.run_struct_btn = QPushButton("🚀 Run Structural Pipeline")
        self.run_struct_btn.setProperty("role", "primary")
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
    # Structural Pipeline Methods
    # -----------------------------------------------------------------------
    def run_structural_pipeline(self):
        # ---- Read the active track layout options ----
        stem_source = self.struct_stem_combo.currentText()
        use_deepseek = self.struct_deepseek_check.isChecked()

        
        # ---- Determine which tracks to process based on scope ----
        scope = self.struct_scope_combo.currentText().strip()
        all_samples = self.dataset.get("samples", [])

        if scope == "All Tracks":
            tracks = list(all_samples)

            if not tracks:
                QMessageBox.warning(
                    self,
                    "No Tracks",
                    "The dataset is empty.",
                )
                return

        elif scope == "Tracks Missing Captions":
            tracks = [
                sample
                for sample in all_samples
                if not (sample.get("caption") or "").strip()
            ]

            if not tracks:
                QMessageBox.information(
                    self,
                    "Nothing to Process",
                    "All tracks already have captions.",
                )
                return

        else:
            # The only remaining Structural Pipeline scope is:
            # "Selected Tracks (from list)".
            selected_indices = []

            numbers_text = self.track_numbers_input.text().strip()

            if numbers_text:
                parsed_indices = self._parse_track_numbers(numbers_text)

                if parsed_indices is None:
                    QMessageBox.warning(
                        self,
                        "Invalid Input",
                        "Use track numbers or ranges, for example: "
                        "1,3,5 or 1-3,5.",
                    )
                    return

                selected_indices.extend(parsed_indices)

            for item in self.track_list_widget.selectedItems():
                try:
                    track_number = int(item.text().split(" - ", 1)[0])
                    selected_indices.append(track_number)
                except (ValueError, IndexError):
                    continue

            selected_indices = sorted(set(selected_indices))

            valid_indices = [
                index
                for index in selected_indices
                if 1 <= index <= len(all_samples)
            ]

            if not valid_indices:
                QMessageBox.warning(
                    self,
                    "No Tracks Selected",
                    "For 'Selected Tracks (from list)', select one or more "
                    "tracks from the list or enter numbers such as 1,3,5.",
                )
                return

            tracks = [
                all_samples[index - 1]
                for index in valid_indices
            ]
        
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

        if self.struct_deepseek_check.isChecked() and not self._ensure_llm_key("aggregator"):
            return

        # ONE shared prompt (it also carries the "remember on this device"
        # choice) instead of three copies that could drift apart.
        if not self._ensure_kaggle_credentials():
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

    def on_struct_batch_done(self):
        """Called when the structural BATCH worker finishes every track.

        ``StructuralPipelineBatchWorker.all_done`` is ``Signal()`` -- no
        arguments. This handler was connected in two places but never defined,
        and because ``.connect(self.on_struct_batch_done)`` evaluates the
        attribute immediately, starting a batch structural run raised
        AttributeError before the worker even began.
        """
        self.struct_progress.setVisible(False)
        self.struct_status.setText("Structural pipeline completed for all tracks.")
        self.status_label.setText(
            "Structural pipeline finished for the batch. Review the captions "
            "and lyrics."
        )
        self.refresh_table()
        self.on_table_selection_changed()

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

    def get_structural_scope_samples(self):
        """Return tracks selected by the Structural Pipeline scope pulldown."""
        samples = self.dataset.get("samples", [])
        scope = self.struct_scope_combo.currentText()

        if scope == "All Tracks":
            return list(samples)

        if scope == "Tracks Missing Captions":
            return [
                sample
                for sample in samples
                if not (sample.get("caption") or "").strip()
            ]

        if scope == "Selected Tracks":
            selected = self.get_selected_sample()

            if selected is None:
                raise ValueError(
                    "Choose a track in Dataset Studio first, then run the "
                    "Structural Pipeline with scope set to 'Selected Tracks'."
                )

            return [selected]

        raise ValueError(f"Unknown Structural Pipeline scope: {scope}")


    def on_struct_scope_changed(self, scope_text):
        """Show manual selection controls only for the selected-tracks scope."""
        show_selected_controls = scope_text == "Selected Tracks (from list)"

        self.track_list_widget.setVisible(show_selected_controls)
        self.track_numbers_input.setVisible(show_selected_controls)

        if show_selected_controls:
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
    def _ensure_llm_key(self, role):
        """Make sure the LLM provider in effect for ``role`` has a key.

        Asks for the ACTIVE provider's key (Groq by default, free) instead of
        assuming DeepSeek. Returns False if the user cancels.
        """
        from modules.llm_client import provider_info, provider_key_present

        if provider_key_present(self.config, role=role):
            return True
        name, info = provider_info(self.config, role=role)
        if name == "local":
            QMessageBox.warning(
                self, "LLM endpoint missing",
                "The 'local' LLM provider needs a Custom Endpoint URL in ⚙ Settings.",
            )
            return False
        hint = f"\nFree key: {info['signup_url']}" if info.get("free") and info.get("signup_url") else ""
        key, ok = QInputDialog.getText(
            self, f"{info['label']} API key",
            f"Enter your {info['label']} API key.{hint}\n"
            "Change provider in the Assistant panel or ⚙ Settings.",
            QLineEdit.Password,
        )
        if not (ok and key.strip()):
            return False
        self.config[info["key"]] = key.strip()
        return True

    def trigger_advanced_ai_pipeline(self):
        selected = self.get_selected_sample()
        if not selected:
            QMessageBox.warning(self, "Selection Missing", "Please pick an active audio track first.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting advanced structural segmentation and captioning...")

        if not self._ensure_llm_key("aggregator"):
            self.progress_bar.setVisible(False)
            return

        target_genre = self.custom_tag_input.text().strip() or "Alternative Rock Production"

        self.active_worker = AdvancedDatasetOrchestratorWorker(
            track_id=selected["id"],
            file_path=selected["audio_path"],
            target_genre=target_genre,
            config=self.config,
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
        """Apply the configured theme app-wide (see ui/themes.py)."""
        from PySide6.QtWidgets import QApplication

        from ui.themes import apply_theme

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self.config)
        if hasattr(self, "theme_swatches"):
            from ui.appearance_panel import refresh_swatches

            refresh_swatches(self)

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
        key_field, _ = LLM_KEY_FIELDS.get(active, ("groq_key", "remember_groq_key"))
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

        from modules.llm_client import DEFAULT_PROVIDER, KNOWN_MODELS

        name = self.llm_provider_combo.currentText().split(" ")[0]
        info = PROVIDERS.get(name, PROVIDERS[DEFAULT_PROVIDER])
        self.llm_model_combo.clear()
        # Empty first = "use the provider's default model". Only this
        # provider's verified models are offered.
        self.llm_model_combo.addItems([""] + KNOWN_MODELS.get(name, []))
        # Show the stored value, or the provider default as a hint — but do NOT
        # write the hint back (that would pin one provider's model name and
        # break after switching provider; see save_cloud_config).
        cfg_model = (self.config.get("llm_model") or "").strip()
        self.llm_model_combo.setCurrentText(cfg_model or "")
        self.llm_base_url_edit.setText(
            (self.config.get("llm_base_url") or "").strip() or info["base_url"]
        )
        # Sync the unified Provider API Key field to the active provider.
        key_field, rem_field = LLM_KEY_FIELDS.get(name, ("groq_key", "remember_groq_key"))
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
        key_field, rem_field = LLM_KEY_FIELDS.get(active, ("groq_key", "remember_groq_key"))
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
        # Appended to the built-in ACE-Step 1.5XL schema (modules/caption_spec.py).
        # getattr-guarded: the caption tab is built with the UI, but this method can
        # run before it exists in a headless/test construction.
        system_editor = getattr(self, "system_prompt_edit", None)
        if system_editor is not None:
            self.config["caption_system_prompt"] = system_editor.toPlainText().strip()
        self.config["caption_max_tokens"] = self.max_tokens_spin.value()
        self.config["caption_max_audio_duration"] = self.max_dur_spin.value()
        # Guarded: this page may not be built yet if another tab triggers a save
        # first, and a missing attribute here would crash the save of every other
        # setting too.
        whole_song = getattr(self, "caption_whole_song_check", None)
        if whole_song is not None:
            self.config["caption_whole_song"] = whole_song.isChecked()
        self.config["caption_batch_size"] = self.batch_size_spin.value()
        # ACE-Step page: the prompt add-on and the run paths. getattr-guarded for
        # the same reason as the system prompt above — this method can run before
        # the page exists in a headless/test construction.
        addendum = getattr(self, "caption_addendum_edit", None)
        if addendum is not None:
            self.config["caption_prompt_addendum"] = addendum.toPlainText().strip()
        for attr, key in (
            ("caption_staging_edit", "caption_staging_dir"),
            ("caption_audio_dataset_edit", "caption_audio_dataset"),
            ("caption_model_dataset_edit", "kaggle_model_dataset"),
            ("caption_output_edit", "caption_output_dir"),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                self.config[key] = widget.text().strip()
        bitrate = getattr(self, "caption_bitrate_combo", None)
        if bitrate is not None:
            self.config["caption_mp3_bitrate"] = bitrate.currentText().strip() or "192k"
        for attr, key in (
            ("caption_convert_check", "caption_convert_mp3"),
            ("caption_batch_review_check", "caption_batch_review"),
        ):
            box = getattr(self, attr, None)
            if box is not None:
                self.config[key] = box.isChecked()
        # tag_caption_ratio is owned by the 🎤 Caption tab's blend slider
        # (on_caption_blend_changed writes it); no spin box to read here.
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
            is_exception = not s.get("caption")
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

            def _cell(text, editable=True):
                item = QTableWidgetItem(str(text) if text not in (None, 0, "") else "")
                if not editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                return item

            # 0 Filename (read-only)
            self.table.setItem(row, 0, _cell(s.get("filename", ""), False))
            # 1 Tag, 2 Genre, 3 Language, 4 Key — free text, always editable
            self.table.setItem(row, 1, _cell(s.get("custom_tag", "")))
            self.table.setItem(row, 2, _cell(s.get("genre", "")))
            self.table.setItem(row, 3, _cell(s.get("language", "")))
            self.table.setItem(row, 4, _cell(s.get("keyscale", "")))
            # 5 BPM
            bpm = s.get("bpm", 0)
            self.table.setItem(row, 5, _cell(str(bpm) if bpm else ""))
            # 6 Time signature
            self.table.setItem(row, 6, _cell(s.get("timesignature", "")))
            # 7 Duration (seconds)
            dur = s.get("duration", 0)
            self.table.setItem(row, 7, _cell(f"{dur}s" if dur else ""))

            # 8 Actions: edit-metadata dialog + delete.
            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(4)
            edit_btn = QPushButton("✏️")
            edit_btn.setToolTip("Edit this track's tag / genre / language / key / BPM / time / duration")
            edit_btn.setMaximumWidth(36)
            edit_btn.clicked.connect(lambda _=False, i=idx: self.open_metadata_editor(i))
            del_btn = QPushButton("🗑")
            del_btn.setToolTip("Remove this track from the dataset")
            del_btn.setMaximumWidth(36)
            del_btn.clicked.connect(lambda _=False, i=idx: self.confirm_delete_sample(i))
            actions_layout.addWidget(edit_btn)
            actions_layout.addWidget(del_btn)
            self.table.setCellWidget(row, 8, actions)

        self._loading_table = False
        self.exceptions_view_btn.setText(f"⚠ Missing Captions ({exceptions_count})")
        if hasattr(self, "filter_count_label"):
            total = len(self.dataset["samples"])
            self.filter_count_label.setText(f"{shown} of {total} tracks")
        # Keep the track pickers in step with the dataset. Cheap when the track
        # list is unchanged (see TrackPickerButton.set_tracks).
        self.refresh_moss_track_picker()
        self.refresh_ace_track_picker()

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
        """Return the currently selected sample.

        Uses a remembered selection index rather than the table's live
        ``currentRow()``: switching to another top-level tab clears the table's
        current row, which previously made every other tab (Lyrics, Caption)
        report "no track selected" even though a track was clearly highlighted.
        """
        row = self.table.currentRow()
        if row < 0:
            row = getattr(self, "_last_selected_row", -1)
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
        self.original_backups.pop(sid, None)
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText(f"Removed '{fname}' from the dataset.")

    # Column index -> ("field", parser) for inline manual edits in the table.
    # Matches the header set in init_ui: 0 Filename, 1 Tag, 2 Genre, 3 Language,
    # 4 Key, 5 BPM, 6 Time signature, 7 Duration, 8 Actions.
    _MANUAL_COLS = {
        1: ("custom_tag", str),
        2: ("genre", str),
        3: ("language", str),
        4: ("keyscale", str),
        5: ("bpm", int),
        6: ("timesignature", str),
        7: ("duration", int),
    }

    def _parse_manual_value(self, field, text):
        """Parse raw cell text for a manual metadata field. Returns value or None."""
        text = (text or "").strip()
        if not text:
            return 0 if field in ("bpm", "duration") else ""
        if field in ("bpm", "duration"):
            text = text.rstrip("s").strip()
            # Allow mm:ss (and h:mm:ss) style durations, e.g. "2:15" or "1:02:33".
            if field == "duration" and ":" in text and text.count(":") <= 2:
                parts = text.split(":")
                try:
                    secs = 0
                    for p in parts:
                        secs = secs * 60 + int(float(p))
                    return secs
                except (ValueError, TypeError):
                    return None
            try:
                return int(round(float(text)))
            except (ValueError, TypeError):
                return None
        return text

    def _apply_manual_metadata(self, s, field, raw_value):
        """Validate + write a single manual field; returns True on success."""
        value = self._parse_manual_value(field, raw_value)
        if value is None:
            return False
        s[field] = value
        return True

    def on_metadata_cell_edited(self, item):
        """Write back inline table edits (Tag/Genre/Key/BPM/Time/Duration)."""
        if getattr(self, "_loading_table", False):
            return
        row = item.row()
        if not (0 <= row < len(self._table_sample_indices)):
            return
        idx = self._table_sample_indices[row]
        samples = self.dataset.get("samples", [])
        if not (0 <= idx < len(samples)):
            return
        col = item.column()
        if col not in self._MANUAL_COLS:
            return
        field = self._MANUAL_COLS[col][0]
        s = samples[idx]
        text = (item.text() or "").strip()

        # Keep the display normalized: revert the cell on parse failure.
        parsed = self._parse_manual_value(field, text)
        if parsed is None:
            self.table.blockSignals(True)
            item.setText("")
            self.table.blockSignals(False)
            return
        s[field] = parsed
        self.status_label.setText(f"Updated {s.get('filename', '')} ({self._MANUAL_COLS[col][0]}).")

    def open_metadata_editor(self, idx):
        """A clear dialog to input Tag / Genre / Key / BPM / Time / Duration manually."""
        samples = self.dataset.get("samples", [])
        if not (0 <= idx < len(samples)):
            return
        s = samples[idx]
        fname = s.get("filename", "")
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Metadata — {fname}")
        form = QFormLayout(dialog)
        dialog.setMinimumWidth(360)

        def _txt(init):
            e = QLineEdit(str(init) if init not in (None, 0, "") else "")
            return e

        tag_edit = _txt(s.get("custom_tag", ""))
        genre_edit = _txt(s.get("genre", ""))
        lang_edit = _txt(s.get("language", ""))
        key_edit = _txt(s.get("keyscale", ""))
        bpm_edit = _txt(s.get("bpm", 0))
        time_edit = _txt(s.get("timesignature", ""))
        dur_edit = _txt(s.get("duration", 0))

        form.addRow("Trigger Tag:", tag_edit)
        form.addRow("Genre:", genre_edit)
        form.addRow("Language (e.g. en):", lang_edit)
        form.addRow("Key (e.g. C, Gm):", key_edit)
        form.addRow("BPM:", bpm_edit)
        form.addRow("Time signature (e.g. 3/4):", time_edit)
        form.addRow("Duration (sec):", dur_edit)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        form.addRow(btn_row)

        cancel_btn.clicked.connect(dialog.reject)

        def _on_save():
            dirty = False
            for field, widget in (
                ("custom_tag", tag_edit),
                ("genre", genre_edit),
                ("language", lang_edit),
                ("keyscale", key_edit),
                ("bpm", bpm_edit),
                ("timesignature", time_edit),
                ("duration", dur_edit),
            ):
                if self._apply_manual_metadata(s, field, widget.text()):
                    dirty = True
            if dirty:
                self.record_snapshot()
            dialog.accept()

        save_btn.clicked.connect(_on_save)
        dialog.exec()
        self.refresh_table()
        self.on_table_selection_changed()
        self.status_label.setText(f"Saved metadata for '{fname}'.")

    def on_table_selection_changed(self):
        # Remember the row so other tabs (Lyrics / Caption) keep working after a
        # tab switch clears the table's current row.
        row = self.table.currentRow()
        if row >= 0:
            self._last_selected_row = row
        s = self.get_selected_sample()
        self._load_track_preview(s)
        if s:
            if not s.get("caption"):
                self.sample_health_alert.setText("No caption yet — add a detailed description below.")
                self.sample_health_alert.setProperty("health", "warn"); repolish(self.sample_health_alert)
            else:
                self.sample_health_alert.setText("Track loaded — edit Tag / Genre / Key / BPM / Time / Duration in the table or via the ✏️ button.")
                self.sample_health_alert.setProperty("health", "ok"); repolish(self.sample_health_alert)

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
            # Health-audit auto-detection has been removed (it can be re-added
            # later as a module). Nothing to restore from until then.
            self.status_label.setText(
                "Restore Detected Values is unavailable — the health audit was removed."
            )
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
            self.bypass_btn.setProperty("role", "danger"); repolish(self.bypass_btn)
            self.status_label.setText("Warning bypass ENABLED: Export unlocked regardless of quality penalties.")
        else:
            self.bypass_btn.setProperty("role", ""); repolish(self.bypass_btn)
            self.status_label.setText("Warning bypass DISABLED.")

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

        self.status_label.setText("DSP normalization finished. Dataset is ready to edit.")

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

        # 3. 🛡️ THE SIGNAL BLOCKER GATEWAY: Pause table signals before updating cell properties
        self.table.blockSignals(True)
        try:
            self.refresh_table()
            self.on_table_selection_changed()
        finally:
            self.table.blockSignals(False) # Securely restore signals regardless of compilation output

        self.status_label.setText(f"Manifest sync pass finished! Reconciled {reconciled_count} tracks.")
        QMessageBox.information(self, "Metadata Alignment Pass Complete",
                                f"Successfully matched {reconciled_count} tracks.")

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

                            synced_count += 1
                            break

            # Freeze rendering triggers to avoid PySide6 layout refresh crashes
            self.table.blockSignals(True)
            try:
                self.refresh_table()
                self.on_table_selection_changed()
            finally:
                self.table.blockSignals(False)

            self.status_label.setText(f"Successfully loaded 1.5XL tags for {synced_count} tracks.")
            QMessageBox.information(self, "1.5XL Sync Complete", f"Successfully integrated metadata properties across {synced_count} tracks.")

        except Exception as e:
            QMessageBox.critical(self, "Manifest Ingestion Failure", f"Could not map structured data columns: {str(e)}")


    # -----------------------------------------------------------------------
    # Load / Save Dataset
    # -----------------------------------------------------------------------
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

            # Backfill any fields this file predates (never overwrites values).
            normalize_dataset(self.dataset)

            self.current_dataset_path = path
            self.record_snapshot()

            self.sync_general_props_to_ui()
            self.refresh_table()

            self.status_label.setText(
                f"Loaded {len(self.dataset.get('samples', []))} tracks."
            )

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def save_dataset(self, path=None):
        if path is None:
            path = getattr(self, "current_dataset_path", None)
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save Dataset JSON", "", "JSON Files (*.json)")
        if not path:
            return False

        try:
            self.on_general_prop_changed()
            meta = self.dataset.setdefault("metadata", {})
            meta["num_samples"] = len(self.dataset["samples"])
            # Stamp created_at for a dataset that predates the field.
            if not meta.get("created_at"):
                normalize_dataset(self.dataset)
            # Keep the derived/legacy fields consistent with the source of truth.
            meta["all_instrumental"] = self.radio_all_inst.isChecked()
            meta["instrumental_mode"] = derive_instrumental_mode(
                meta["all_instrumental"], self.dataset["samples"]
            )
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
        try:
            from ui.shell import save_layout

            save_layout(self)
        except Exception as e:  # noqa: BLE001 -- never block closing over layout
            print(f"could not save panel layout: {e}")
        event.accept()


    # -----------------------------------------------------------------------
    # Add Audio (single song / folder)
    # -----------------------------------------------------------------------
    def add_audio_files(self):
        # Qt filter grammar needs globs ("*.wav"), not bare extensions (".wav"):
        # a bare extension matches nothing and hides every audio file in the
        # picker. Lossless formats are listed first as the recommended choice.
        globs = " ".join(f"*{e}" for e in AUDIO_EXTS_ORDERED)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Single Song (one or more audio files)", "",
            f"Audio Files ({globs});;All Files (*)",
        )
        if paths:
            if self._show_add_track_warnings(len(paths)):
                self._add_audio_paths(paths)
            else:
                self.status_label.setText("Add cancelled — no files were imported.")

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
                f"Supported formats: {', '.join(AUDIO_EXTS_ORDERED)}",
            )
            return
        if self._show_add_track_warnings(len(paths)):
            self._add_audio_paths(paths)
        else:
            self.status_label.setText("Add cancelled — no files were imported.")

    def _show_add_track_warnings(self, count):
        """Warn about dataset-quality risks before importing tracks.

        Shown once per session unless the user ticks "Don't show this again"
        (persisted via the ADD_WARNINGS_SETTINGS_KEY settings flag).
        Returns True to proceed with the import, False if the user cancelled.
        """
        if self.config.get(ADD_WARNINGS_SETTINGS_KEY):
            return True

        box = QMessageBox(self)
        box.setWindowTitle("Before You Add Tracks — Read This")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"About to add {count} audio file(s). A few things strongly affect "
            "how well your dataset trains:"
        )
        box.setInformativeText(
            "<b>1. Prefer lossless sources (recommended).</b><br>"
            "<code>.wav</code> / <code>.flac</code> / <code>.aiff</code> / "
            "<code>.alac</code> are the recommended formats. Lossy formats "
            "(<code>.mp3</code>, <code>.m4a</code>, <code>.aac</code>, "
            "<code>.opus</code>, <code>.wma</code>) permanently discard audio "
            "detail — a 128 kbps MP3 bakes compression artifacts straight into "
            "your LoRA weights. Any lossy track caps your achievable quality. "
            "If you only have a lossy file, try to find a lossless master first.<br><br>"

            "<b>2. Keep your sources consistent.</b><br>"
            "Mixing mismatched sources — different sample rates, mono vs stereo, "
            "different loudness, vinyl rips beside studio masters, or one album "
            "ripped from a different edition — teaches the model those differences "
            "instead of the music. Normalize loudness/sample rate (DSP Normalize) "
            "and avoid blending eras, releases, or transfer qualities.<br><br>"

            "<b>3. Watch out for these common problems.</b><br>"
            "&bull; <b>Clipping / brickwalled masters</b> — distortion the model will learn.<br>"
            "&bull; <b>Different sample rates</b> (44.1k vs 48k) — resample to one rate.<br>"
            "&bull; <b>Mono files mixed with stereo</b> — inconsistent stereo image.<br>"
            "&bull; <b>Fake lossless</b> (a 128 kbps MP3 re-saved as WAV) — still lossy.<br>"
            "&bull; <b>Duplicates / re-rips</b> of the same song — skewed weighting.<br>"
            "&bull; <b>Short clips vs full songs</b> — inconsistent structure learning.<br>"
            "&bull; <b>Silence, count-ins, or talking</b> at the start — trim it.<br>"
            "&bull; <b>Inconsistent loudness</b> — normalize to one target LUFS.<br><br>"

            "<b>4. Aim for a coherent, single-sound dataset.</b><br>"
            "A smaller, consistent dataset (same genre/era/production, lossless, "
            "one sample rate) trains far better than a large, heterogeneous one."
        )
        dont_show = QCheckBox("Don't show this warning again")
        box.setCheckBox(dont_show)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Ok)
        if box.exec() != QMessageBox.Ok:
            return False
        if dont_show.isChecked():
            self.config[ADD_WARNINGS_SETTINGS_KEY] = True
            save_config(self.config)
        return True

    def _add_audio_paths(self, paths):
        """Shared add path: create samples for the given audio file paths.

        Skips files already present in the dataset (dedupe by ``audio_path``),
        then refreshes the table. Tracks are added instantly with blank metadata
        (BPM/key/time/duration) and are left unlocked/editable so you can type
        values inline. No health audit runs on add (the audit module was removed).
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
            self.dataset["samples"].append(new_sample(
                id=uuid.uuid4().hex[:8],
                audio_path=p,
                filename=fname,
                language="en",
                is_instrumental=is_all_inst,
                custom_tag=global_tag,
            ))
            added += 1
        self.refresh_table()
        msg = f"Added {added} audio track(s)."
        if skipped:
            msg += f" {skipped} already present (skipped)."
        self.status_label.setText(msg)


    # -----------------------------------------------------------------------
    # AI Captioning (with DeepSeek backend option)
    # -----------------------------------------------------------------------
    def _set_caption_busy(self, busy):
        """Enable/disable the captioning triggers while a run is in flight.

        ``run_ai_btn`` (the Studio's old "Run AI Captioner" button) no longer
        exists -- it was removed in 15520af when the Caption tab took over, but
        three toggles were left behind referencing it. Reaching any of them
        raised AttributeError, and nothing did until the Caption tab started
        calling start_ai_captioning().

        Toggling by NAME means the state applies to whichever triggers actually
        exist, so a future extraction or rename cannot reintroduce the crash.
        That is why this now only RECORDS the state: ``update_ace_actions()``
        resolves every action by name and decides what is live.
        """
        self._caption_busy = bool(busy)
        self.update_ace_actions()

    def _set_ace_action(self, button, enabled, reason=""):
        """Enable/disable one step button, saying WHY when it is disabled.

        A greyed-out control with no explanation makes the user hunt for the rule
        — which is the dialog this replaces. The pristine tooltip is remembered
        the first time it is needed, so the reason can be added and removed
        without the text drifting.
        """
        if button is None:
            return
        tip = getattr(button, "_ace_pristine_tip", None)
        if tip is None:
            tip = button.toolTip()
            button._ace_pristine_tip = tip
        button.setEnabled(bool(enabled))
        button.setToolTip(tip if enabled else f"{reason}\n\n{tip}")

    def update_ace_actions(self):
        """One place decides which steps are live, and why the rest are not.

        Every desktop app with a linear flow does this — GitHub Desktop greys out
        Push until there is something to push, VS Code greys out Sync — and the
        rule it encodes is that the next action is ENABLED while anything
        downstream of a missing input is DISABLED WITH A REASON. The alternative,
        which this page used to be, is a row of always-live buttons each popping a
        dialog to explain itself.

        Called from everything that can change one of the inputs: the tick list,
        the dataset refresh, the staging list, and the start/end/error of a run.
        """
        from modules import caption_kaggle_run as ckr

        busy = getattr(self, "_caption_busy", False)
        samples = self.dataset.get("samples", [])
        ticked = bool(self._ticked_samples())
        report = ckr.staging_report(ckr.staging_dir(self.config))
        junk = [name for name, why in report["ignored"] if why in ckr.JUNK_REASONS]
        rows = self._proposal_rows() if self._caption_scope_ids else []
        pending = sum(1 for row in rows if row["status"] != ckr.STATUS_SAME)
        bad = len(self._bad_caption_samples())

        needs_ticks = ("Tick tracks in “Tracks ▾” first." if not ticked
                       else "A caption run is already in flight.")
        for name in ("staging_add_btn", "caption_selected_btn", "caption_edit_btn"):
            self._set_ace_action(
                getattr(self, name, None), ticked and not busy, needs_ticks,
            )
        for name in ("caption_missing_btn", "caption_all_btn"):
            self._set_ace_action(
                getattr(self, name, None), bool(samples) and not busy,
                "Add songs to the dataset first (🎛 Dataset Studio)."
                if not samples else "A caption run is already in flight.",
            )
        self._set_ace_action(
            getattr(self, "caption_diff_btn", None), pending > 0,
            "Nothing to review yet — run the captioner, or load a downloaded "
            "captions_out.json with 📥 Import.",
        )
        self._set_ace_action(
            getattr(self, "caption_recaption_bad_btn", None), bad > 0 and not busy,
            "Nothing needs re-captioning: every track has a caption and the last "
            "run reported no errors." if not bad
            else "A caption run is already in flight.",
        )
        self._set_ace_action(
            getattr(self, "staging_clean_btn", None), bool(junk) and not busy,
            "No junk files in the staging folder." if not junk
            else "A caption run is already in flight.",
        )

        # State ON the control, not in a floating label: the count is what makes
        # "is there anything to review?" answerable without clicking anything.
        review_btn = getattr(self, "caption_diff_btn", None)
        if review_btn is not None:
            review_btn.setText(f"🔍 Review · {pending}" if pending else "🔍 Review")
        redo_btn = getattr(self, "caption_recaption_bad_btn", None)
        if redo_btn is not None:
            redo_btn.setText(f"♻ Re-caption · {bad}" if bad else "♻ Re-caption")
        run_btn = getattr(self, "caption_selected_btn", None)
        if run_btn is not None:
            run_btn.setText("⏳ Captioning…" if busy else "🚀 Caption")

    def staging_clean_unusable(self):
        """Delete the provably-junk files from the staging folder.

        These are the files the upload would NOT take: 0-byte corpses from an
        interrupted transcode, ``.part`` scratch, and our own metadata file. They
        are invisible in the list — which filters them out — which is exactly why
        the folder and the list could silently disagree.
        """
        from modules import caption_kaggle_run as ckr

        folder = ckr.staging_dir(self.config)
        junk = [(name, why) for name, why in ckr.unusable_staged(folder)
                if why in ckr.JUNK_REASONS]
        if not junk:
            self.status_label.setText("No junk files in the staging folder.")
            return
        removed = ckr.clean_unusable_staged(folder)
        self.refresh_staging_list()
        self.status_label.setText(
            f"Removed {removed} unusable file(s) from the staging folder: "
            + ", ".join(f"{name} ({why})" for name, why in junk[:4])
            + ("…" if len(junk) > 4 else "")
        )

    def start_ai_captioning(self, checked=False, scope=None):
        """Run the captioner.

        ``scope`` may be passed programmatically ("selected" / "missing" / "all")
        by the 🎤 Caption tab; otherwise the user is asked.
        """
        all_samples = self.dataset.get("samples", [])
        if not all_samples:
            QMessageBox.warning(self, "No Tracks", "Add audio tracks before captioning.")
            return

        if scope is None:
            scope_choices = ["Ticked Tracks (Caption page)", "Tracks Missing Captions", "All Tracks — Review Every Result", "Selected Track (dataset table)"]
            scope, accepted = QInputDialog.getItem(self, "Choose Captioning Scope", "Which tracks should ACE-Step Captioner process?", scope_choices, 0, False)
            if not accepted or not scope:
                self.status_label.setText("AI captioning cancelled.")
                return

        # Normalise programmatic scope values onto the dialog's labels.
        scope = {
            "ticked": "Ticked Tracks",
            "missing": "Tracks Missing Captions",
            "all": "All Tracks — Review Every Result",
            "bad": "Tracks Needing Re-caption",
            "selected": "Selected Track",
        }.get(scope, scope)

        if scope == "Ticked Tracks":
            samples = self._ticked_samples()
            if not samples:
                self._no_tracks_ticked()
                return
        elif scope == "Selected Track":
            selected_sample = self.get_selected_sample()
            if not selected_sample:
                QMessageBox.warning(self, "No Track Selected", "Select one track in the dataset table first.")
                return
            samples = [selected_sample]
        elif scope == "Tracks Missing Captions":
            samples = [s for s in all_samples if not s.get("caption", "").strip()]
        elif scope == "Tracks Needing Re-caption":
            samples = self._bad_caption_samples()
        else:
            samples = list(all_samples)

        if not samples:
            QMessageBox.information(self, "Nothing To Caption", "No tracks match the selected scope.")
            return

        # Remembered so the end-of-run diff review knows which proposals belong to
        # THIS run: caption_ai_raw keeps the previous run's proposal otherwise.
        self._caption_scope_ids = [s.get("id") for s in samples]

        self.record_snapshot()
        self._set_caption_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        general_meta = self.dataset.get("metadata", {})

        # Backend is user-configurable in the Caption page; this ASKS about
        # credentials rather than silently degrading to an engine that never hears
        # the audio (see _resolve_caption_backend).
        backend = self._resolve_caption_backend()
        if not backend:
            self._set_caption_busy(False)
            self.progress_bar.setVisible(False)
            self.status_label.setText("AI captioning cancelled — no backend chosen.")
            return
        self._active_caption_backend = backend
        if hasattr(self, "ace_status_label"):
            self.ace_status_label.setText(f"Running: {backend}")
        if backend in self.PLACEHOLDER_CAPTION_BACKENDS:
            # Say it BEFORE the run, not only in the stamp afterwards: the user
            # must know what they are about to write into the dataset.
            self.status_label.setText(
                "⚠ Placeholder backend: it never hears the audio. Captions are "
                "stamped in caption_ai_model so they cannot be mistaken for real."
            )
            if hasattr(self, "ace_status_label"):
                self.ace_status_label.setText(
                    f"Running: {backend} — PLACEHOLDER output"
                )

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
        # Persist WHICH dataset the audio went to. The worker knows it first (it
        # creates the dataset), so it tells the window rather than writing config
        # from a thread.
        if hasattr(self.active_worker, "dataset_slug_ready"):
            self.active_worker.dataset_slug_ready.connect(
                self.on_audio_dataset_ready
            )
        self.active_worker.start()

    def on_audio_dataset_ready(self, slug):
        """Remember the Kaggle dataset that holds the uploaded audio.

        WHY THIS EXISTS: the slug was set on the config dict in memory and nothing
        ever saved it, so settings.json kept ``caption_audio_dataset=""`` while
        every run created a NEW dataset (the logs show ace-audio-80d01a, then
        ace-audio-d66edb). The "re-runs push a new VERSION of the same dataset"
        promise therefore never engaged, the account accumulated orphaned datasets,
        and the page could not say where the audio had gone.
        """
        slug = (slug or "").strip()
        if not slug:
            return
        self.config["caption_audio_dataset"] = slug
        edit = getattr(self, "caption_audio_dataset_edit", None)
        if edit is not None and edit.text().strip() != slug:
            # textChanged -> save_pipeline_defaults(), which persists the config.
            edit.setText(slug)
        else:
            from modules.config_store import save_config

            save_config(self.config)
        self.status_label.setText(f"Kaggle audio dataset: {slug}")

    def on_sample_captioned(self, sid, caption):
        model_id = self.config.get("caption_backend", "ace_step")
        for sample in self.dataset["samples"]:
            if sample.get("id") == sid:
                self.save_ai_caption_result(sample, caption, model_id=model_id, prompt="Detailed ACE-Step caption request")
                self._stamp_placeholder_caption(
                    sample, getattr(self, "_active_caption_backend", "")
                )
                if self.caption_batch_review_enabled():
                    # HOLD the proposal. A per-track modal over a 200-track run is
                    # 200 dialogs; the diff table shows every proposal at once and
                    # the caption is not touched until a decision is applied.
                    self.status_label.setText(
                        f"Captioned {sample.get('filename', sid)} — held for review."
                    )
                else:
                    self.review_ai_caption_result(sample)
                break

    def on_caption_finished(self):
        self._set_caption_busy(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("AI Captioning completed.")
        self.refresh_table()
        self.on_table_selection_changed()
        placeholders = [
            s for s in self._caption_run_samples() if s.get("caption_is_placeholder")
        ]
        if placeholders:
            # Louder than the stamp on its own: a dataset full of template text
            # that LOOKS like captions is the failure this guards against.
            self.status_label.setText(
                f"AI Captioning completed — {len(placeholders)} caption(s) are "
                "PLACEHOLDERS: that backend never heard the audio. They are "
                "marked in caption_ai_model and caption_is_placeholder."
            )
        if self.caption_batch_review_enabled() and self._caption_scope_ids:
            rows = self._proposal_rows()
            if rows:
                from modules import caption_kaggle_run as ckr
                counts = ckr.count_by_status(rows)
                self.ace_status_label.setText(
                    "Last run: "
                    + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
                    + ". Nothing has been changed yet — apply the diff to decide."
                )
                self.show_caption_diff(rows)
        # The proposals now exist, so 🔍 Review must come alive and show its count.
        self.update_ace_actions()

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
    def _tag_creator_artist(self):
        """Which artist vocabulary the tag creator should offer.

        An explicit ``tag_creator_artist`` setting wins; otherwise the dataset
        name is used as a loose hint ("sabbath" resolves to Black Sabbath,
        "Doorsdata" to The Doors). Empty or unrecognised means the cross-artist
        fundamentals only -- never another artist's signature terms.
        """
        explicit = (self.config.get("tag_creator_artist") or "").strip()
        if explicit:
            return explicit
        meta = self.dataset.get("metadata") or {}
        return (meta.get("name") or "").strip()

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
        self.tag_creator_worker = TagCreatorWorker(
            index, sample, self.config, artist=self._tag_creator_artist(),
            parent=self,
        )
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
        hint.setProperty("muted", True)
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

    def _get_rename_scope_targets(self, scope):
        """Resolve a bulk-rename scope string to the list of target samples.

        `_apply_bulk_rename` called this by NAME while `_bulk_rename_targets`
        inlined the same logic -- and the method did not exist, so actually
        running a bulk rename raised AttributeError. Extracted here so both
        paths share one implementation.
        """
        samples = self.dataset.get("samples", [])
        if scope == "Selected tracks":
            rows = sorted({r.row() for r in self.table.selectionModel().selectedRows()})
            return [samples[self._table_sample_indices[r]]
                    for r in rows if 0 <= r < len(self._table_sample_indices)]
        if scope == "Filtered (visible) tracks":
            return [samples[i] for i in self._table_sample_indices
                    if 0 <= i < len(samples)]
        return list(samples)

    def _bulk_rename_targets(self, scope_combo, mode_combo, find_edit, repl_edit,
                             prefix_edit, suffix_edit, pattern_edit, start_spin):
        """Return ``[(sample, new_filename), ...]`` — never mutates samples."""
        targets = self._get_rename_scope_targets(scope_combo.currentText())

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
            path = save_version(self.dataset, label="manual")
            self.status_label.setText(f"Snapshot saved: {path}")
            refresh()

        def diff_selected():
            item = self.ver_list.currentItem()
            if not item:
                QMessageBox.information(self, "Versioning", "Select a snapshot to diff.")
                return
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
        note.setProperty("muted", True)
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
        self._set_caption_busy(False)
        self.normalize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)      
        self.status_label.setText("Operation error.")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")
