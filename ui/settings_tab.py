"""Settings tab builder for DatasetManager.

Extracted from dataset_manager.py's init_settings_tab(). This module holds
one function, build_settings_tab(manager, parent), which builds every widget
in the Settings tab directly onto `manager` (the DatasetManager instance) --
exactly as the original inline method did. Every widget is still stored as
manager.<name> (e.g. manager.k_user, manager.llm_provider_combo) because
other DatasetManager methods (save_cloud_config, save_pipeline_defaults,
save_all_settings, _on_llm_provider_changed, _populate_model_picker, etc.)
read those same attributes by name. This extraction changes *where* the
widget-building code lives, not the object model it builds.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea, QFrame, QWidget,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QTextEdit,
    QSpinBox, QDoubleSpinBox, QSlider, QFontComboBox, QPushButton,
)

from modules.model_manager import leaderboards


def build_settings_tab(manager, parent):
    outer = QVBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)

    # Title clearance: keep the group-box titles from being overlapped by
    # the first form row (labels/inputs on the left).
    scroll.setStyleSheet("QGroupBox { padding-top: 0.9em; }")

    save_all_btn = QPushButton("💾 Save All Settings")
    save_all_btn.setStyleSheet("font-weight: bold; padding: 8px;")
    save_all_btn.clicked.connect(manager.save_all_settings)
    layout.addWidget(save_all_btn)

    theme_grp = QGroupBox("🎨 Visual Appearance & UI Themes")
    form = QFormLayout(theme_grp)
    form.setContentsMargins(8, 18, 8, 8)

    manager.font_picker = QFontComboBox()
    manager.font_picker.setToolTip("App font.")
    manager.font_picker.setCurrentFont(QFont(manager.custom_theme["font_family"]))
    manager.font_picker.currentFontChanged.connect(manager.on_font_changed)
    form.addRow("Installed System Font:", manager.font_picker)

    zoom_row = QHBoxLayout()
    manager.zoom_slider = QSlider(Qt.Horizontal)
    manager.zoom_slider.setToolTip("UI zoom level.")
    manager.zoom_slider.setRange(75, 175)
    manager.zoom_slider.setValue(100)
    manager.zoom_label = QLabel("100%")
    manager.zoom_slider.valueChanged.connect(manager.on_zoom_changed)
    zoom_row.addWidget(manager.zoom_slider)
    zoom_row.addWidget(manager.zoom_label)
    form.addRow("UI Zoom Factor:", zoom_row)

    manager.theme_preset_combo = QComboBox()
    manager.theme_preset_combo.setToolTip("Color theme preset.")
    manager.theme_preset_combo.addItems(["Dark Modern (Default)", "OLED Pure Black", "Gentoo Purple Slate", "Solarized Dark", "High Contrast Light"])
    manager.theme_preset_combo.currentTextChanged.connect(manager.on_theme_preset_changed)
    form.addRow("Theme Preset:", manager.theme_preset_combo)

    layout.addWidget(theme_grp)

    cloud_grp = QGroupBox("⚙ Cloud & Execution Endpoints")
    c_form = QFormLayout(cloud_grp)
    c_form.setContentsMargins(8, 18, 8, 8)

    manager.k_user = QLineEdit(manager.config.get("kaggle_user", ""))
    manager.k_user.setToolTip("Your Kaggle username (from kaggle.json).")
    manager.k_key = QLineEdit(manager.config.get("kaggle_key", ""))
    manager.k_key.setToolTip("Your Kaggle API key (from kaggle.json).")
    manager.k_key.setEchoMode(QLineEdit.Password)
    c_form.addRow("Kaggle Username:", manager.k_user)
    c_form.addRow("Kaggle API Key:", manager.k_key)
    manager.remember_kaggle = QCheckBox("Remember on this device (encrypted)")
    manager.remember_kaggle.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
    manager.remember_kaggle.setChecked(bool(manager.config.get("remember_kaggle_key", True)))
    c_form.addRow("", manager.remember_kaggle)

    manager.custom_url = QLineEdit(manager.config.get("custom_url", ""))
    manager.custom_url.setToolTip("Base URL for a custom OpenAI-compatible endpoint.")
    manager.custom_url.setPlaceholderText("https://api.runpod.ai/... or http://localhost:8000/v1")
    manager.custom_key = QLineEdit(manager.config.get("custom_key", ""))
    manager.custom_key.setEchoMode(QLineEdit.Password)
    c_form.addRow("Custom Auth Token (custom endpoints):", manager.custom_key)
    manager.remember_custom = QCheckBox("Remember on this device (encrypted)")
    manager.remember_custom.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
    manager.remember_custom.setChecked(bool(manager.config.get("remember_custom_key", True)))
    c_form.addRow("", manager.remember_custom)

    # NEW: MVSEP key
    manager.mvsep_key = QLineEdit(manager.config.get("mvsep_api_key", ""))
    manager.mvsep_key.setEchoMode(QLineEdit.Password)
    c_form.addRow("MVSEP API Key:", manager.mvsep_key)
    manager.remember_mvsep = QCheckBox("Remember on this device (encrypted)")
    manager.remember_mvsep.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
    manager.remember_mvsep.setChecked(bool(manager.config.get("remember_mvsep_api_key", True)))
    c_form.addRow("", manager.remember_mvsep)

    sec_note = QLabel(
        "Secrets are stored encrypted (OS keyring, or an encrypted file as a fallback) — "
        "never in settings.json. Uncheck 'Remember' to keep a key for the current session only."
    )
    sec_note.setWordWrap(True)
    sec_note.setStyleSheet("color: #aaa; font-size: 9px; padding: 2px;")
    c_form.addRow(sec_note)

    # ---- Caption backend (pluggable providers) ----
    manager.caption_backend_combo = QComboBox()
    manager.caption_backend_combo.setToolTip("Which model captions the audio.")
    manager.caption_backend_combo.addItems([
        "ace_step — ACE-Step captioner (Kaggle GPU, default)",
        "gemini — Google Gemini (audio-native)",
        "deepseek — DeepSeek LLM",
        "custom — OpenAI-compatible endpoint (local/rented GPU)",
    ])
    cur_backend = (manager.config.get("caption_backend") or "ace_step").strip().lower()
    for i in range(manager.caption_backend_combo.count()):
        if manager.caption_backend_combo.itemText(i).startswith(cur_backend):
            manager.caption_backend_combo.setCurrentIndex(i)
            break
    c_form.addRow("Caption Backend:", manager.caption_backend_combo)

    manager.gemini_key = QLineEdit(manager.config.get("gemini_api_key", ""))
    manager.gemini_key.setEchoMode(QLineEdit.Password)
    c_form.addRow("Gemini API Key:", manager.gemini_key)
    manager.remember_gemini = QCheckBox("Remember on this device (encrypted)")
    manager.remember_gemini.setToolTip("Save the key in the OS keyring / encrypted store. Uncheck to use it for this session only.")
    manager.remember_gemini.setChecked(bool(manager.config.get("remember_gemini_key", True)))
    c_form.addRow("", manager.remember_gemini)

    manager.gemini_model_combo = QComboBox()
    manager.gemini_model_combo.setToolTip("Gemini model used for audio captioning.")
    manager.gemini_model_combo.setEditable(True)
    manager.gemini_model_combo.addItems(["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"])
    cur_model = (manager.config.get("gemini_model") or "gemini-2.5-flash").strip()
    idx = manager.gemini_model_combo.findText(cur_model)
    if idx >= 0:
        manager.gemini_model_combo.setCurrentIndex(idx)
    else:
        manager.gemini_model_combo.setEditText(cur_model)
    c_form.addRow("Gemini Model:", manager.gemini_model_combo)

    manager.custom_url_edit = QLineEdit(manager.config.get("custom_caption_url", ""))
    manager.custom_url_edit.setToolTip("OpenAI-compatible base URL for the custom caption backend.")
    manager.custom_url_edit.setPlaceholderText("e.g. http://localhost:8000/v1 (vLLM/Ollama/runpod)")
    c_form.addRow("Custom Endpoint Base URL:", manager.custom_url_edit)

    manager.custom_model_edit = QLineEdit(manager.config.get("custom_caption_model", ""))
    manager.custom_model_edit.setToolTip("Model name served by the custom endpoint.")
    manager.custom_model_edit.setPlaceholderText("model name served by the endpoint")
    c_form.addRow("Custom Endpoint Model:", manager.custom_model_edit)

    manager.custom_audio_check = QCheckBox("Send audio to the endpoint (OpenAI input_audio)")
    manager.custom_audio_check.setChecked(bool(manager.config.get("custom_caption_audio", False)))
    c_form.addRow("", manager.custom_audio_check)

    save_cloud_btn = QPushButton("Save Cloud Credentials")
    save_cloud_btn.clicked.connect(manager.save_cloud_config)
    c_form.addRow(save_cloud_btn)

    layout.addWidget(cloud_grp)

    llm_grp = QGroupBox("🧠 LLM Provider")
    llm_form = QFormLayout(llm_grp)
    llm_form.setContentsMargins(8, 18, 8, 8)

    manager.llm_provider_combo = QComboBox()
    manager.llm_provider_combo.addItems([
        "deepseek (paid, cheap — default)",
        "gemini (free tier)",
        "groq (free tier)",
        "openrouter (free models)",
        "local (custom endpoint)",
    ])
    cur_prov = str(manager.config.get("llm_provider", "deepseek") or "deepseek").lower()
    for i in range(manager.llm_provider_combo.count()):
        if manager.llm_provider_combo.itemText(i).startswith(cur_prov):
            manager.llm_provider_combo.setCurrentIndex(i)
            break
    manager.llm_provider_combo.currentIndexChanged.connect(manager._on_llm_provider_changed)
    llm_form.addRow("Provider:", manager.llm_provider_combo)

    manager.llm_api_key = QLineEdit(manager.config.get("deepseek_key", ""))
    manager.llm_api_key.setEchoMode(QLineEdit.Password)
    manager.llm_api_key.setPlaceholderText("auto-routes to the selected provider/model")
    manager.llm_api_key.setToolTip("One key field that routes to whichever provider is selected (DeepSeek / Gemini / Groq / OpenRouter). The app knows which API from the model chosen.")
    llm_form.addRow("Provider API Key:", manager.llm_api_key)
    manager.remember_llm_api = QCheckBox("Remember this key (encrypted)")
    manager.remember_llm_api.setToolTip("Save the active provider's key in the encrypted store. Uncheck for session-only.")
    manager.remember_llm_api.setChecked(bool(manager.config.get("remember_deepseek_key", True)))
    llm_form.addRow("", manager.remember_llm_api)

    manager.llm_model_combo = QComboBox()
    manager.llm_model_combo.setEditable(True)
    llm_form.addRow("Model:", manager.llm_model_combo)

    manager.llm_base_url_edit = QLineEdit()
    manager.llm_base_url_edit.setPlaceholderText("auto-filled from the provider (editable)")
    llm_form.addRow("Base URL:", manager.llm_base_url_edit)

    manager.openrouter_key = QLineEdit(manager.config.get("openrouter_key", ""))
    manager.openrouter_key.setEchoMode(QLineEdit.Password)
    llm_form.addRow("OpenRouter Key:", manager.openrouter_key)
    manager.remember_openrouter = QCheckBox("Remember on this device (encrypted)")
    manager.remember_openrouter.setChecked(bool(manager.config.get("remember_openrouter_key", True)))
    llm_form.addRow("", manager.remember_openrouter)

    manager.groq_key = QLineEdit(manager.config.get("groq_key", ""))
    manager.groq_key.setEchoMode(QLineEdit.Password)
    llm_form.addRow("Groq Key:", manager.groq_key)
    manager.remember_groq = QCheckBox("Remember on this device (encrypted)")
    manager.remember_groq.setChecked(bool(manager.config.get("remember_groq_key", True)))
    llm_form.addRow("", manager.remember_groq)

    # ---- Per-role overrides (aggregator / captioner / assistant) ----
    roles_box = QGroupBox("Per-role overrides (empty = use the global provider/model)")
    rform = QFormLayout(roles_box)
    rform.setContentsMargins(8, 14, 8, 8)
    manager.role_provider_combo = {}
    manager.role_model_combo = {}
    _role_providers = ["default (global)", "deepseek", "gemini", "groq", "openrouter", "local"]
    for role in ("aggregator", "captioner", "assistant"):
        row = QHBoxLayout()
        prov = QComboBox()
        prov.addItems(_role_providers)
        cur = (manager.config.get(f"llm_provider_{role}") or "").strip()
        prov.setCurrentText(cur if cur in _role_providers[1:] else "default (global)")
        prov.setToolTip(f"Provider used by the {role} (the master-caption aggregator, the LLM captioner, or the AI assistant).")
        mod = QComboBox()
        mod.setEditable(True)
        mod.addItems(["", "deepseek-chat", "gemini-2.5-flash", "gemini-3.7-flash",
                      "llama-3.3-70b-versatile", "meta-llama/llama-3.3-70b-instruct:free"])
        mod.setCurrentText(manager.config.get(f"llm_model_{role}", ""))
        mod.setToolTip(f"Model used by the {role}; empty = the provider default.")
        manager.role_provider_combo[role] = prov
        manager.role_model_combo[role] = mod
        row.addWidget(prov, 1)
        row.addWidget(mod, 2)
        rform.addRow(f"{role.capitalize()}:", row)
    llm_form.addRow(roles_box)

    manager.llm_note = QLabel("")
    manager.llm_note.setWordWrap(True)
    manager.llm_note.setStyleSheet("color: #aaa; font-size: 9px;")
    llm_form.addRow(manager.llm_note)
    manager._on_llm_provider_changed()

    layout.addWidget(llm_grp)

    pipe_grp = QGroupBox("🎛 Pipeline & Model Defaults")
    p_form = QFormLayout(pipe_grp)
    p_form.setContentsMargins(8, 18, 8, 8)

    manager.prompt_edit = QTextEdit()
    manager.prompt_edit.setToolTip("The instruction given to the caption model for every chunk. Edit to steer descriptions.")
    manager.prompt_edit.setPlainText(manager.config.get("caption_prompt", ""))
    manager.prompt_edit.setMaximumHeight(120)
    manager.prompt_edit.setPlaceholderText("Prompt used by the caption backends (ACE-Step / Gemini / custom).")
    p_form.addRow("Caption Prompt:", manager.prompt_edit)

    manager.max_tokens_spin = QSpinBox()
    manager.max_tokens_spin.setToolTip("Maximum tokens the captioner may generate per chunk.")
    manager.max_tokens_spin.setRange(32, 2048)
    manager.max_tokens_spin.setValue(int(manager.config.get("caption_max_tokens", 512)))
    p_form.addRow("Caption Max Tokens:", manager.max_tokens_spin)

    manager.max_dur_spin = QSpinBox()
    manager.max_dur_spin.setRange(0, 600)
    manager.max_dur_spin.setValue(int(manager.config.get("caption_max_audio_duration", 120)))
    manager.max_dur_spin.setToolTip("Max audio length fed to the captioner in seconds (0 = whole file).")
    p_form.addRow("Max Audio Duration (s):", manager.max_dur_spin)

    manager.batch_size_spin = QSpinBox()
    manager.batch_size_spin.setRange(1, 8)
    manager.batch_size_spin.setValue(int(manager.config.get("caption_batch_size", 1)))
    manager.batch_size_spin.setToolTip("Chunks processed per captioner forward pass on the Kaggle GPU. 1 is safest; 2-4 is faster when VRAM allows (32 GB total across both T4s).")
    p_form.addRow("Caption Batch Size:", manager.batch_size_spin)

    manager.tag_ratio_spin = QSpinBox()
    manager.tag_ratio_spin.setRange(0, 100)
    manager.tag_ratio_spin.setSuffix("%")
    manager.tag_ratio_spin.setValue(int(manager.config.get("tag_caption_ratio", 0)))
    manager.tag_ratio_spin.setToolTip("Hybrid captions: 0% = prose only (default), 100% = tag block only, in between = both.")
    p_form.addRow("Hybrid Tag Ratio:", manager.tag_ratio_spin)

    manager.clap_tagger_combo = QComboBox()
    manager.clap_tagger_combo.addItems(["auto (use CLAP if installed)", "on", "off"])
    cur_clap = str(manager.config.get("use_clap_tagger", "auto") or "auto").lower()
    for i in range(manager.clap_tagger_combo.count()):
        if manager.clap_tagger_combo.itemText(i).lower().startswith(cur_clap):
            manager.clap_tagger_combo.setCurrentIndex(i)
            break
    manager.clap_tagger_combo.setToolTip("CLAP zero-shot tagging names specific instruments (needs torch + transformers). 'auto' uses it when available.")
    p_form.addRow("Instrument Tagger:", manager.clap_tagger_combo)

    manager.auto_recommend_check = QCheckBox("Auto-recommend instrument models from detected tags")
    manager.auto_recommend_check.setChecked(bool(manager.config.get("auto_recommend_models", True)))
    p_form.addRow(manager.auto_recommend_check)

    manager.lead_vocal_combo = QComboBox()
    manager.lead_vocal_combo.addItems(["off", "mvsep (backing-vocal model)", "heuristic (experimental)"])
    cur_lv = str(manager.config.get("lead_vocal_splitter", "off") or "off").lower()
    if cur_lv == "mvsep":
        manager.lead_vocal_combo.setCurrentIndex(1)
    elif cur_lv == "heuristic":
        manager.lead_vocal_combo.setCurrentIndex(2)
    p_form.addRow("Lead/Backing Vocal Split:", manager.lead_vocal_combo)

    manager.min_sec_spin = QDoubleSpinBox()
    manager.min_sec_spin.setToolTip("Minimum section length in seconds (shorter sections are merged).")
    manager.min_sec_spin.setRange(1.0, 120.0)
    manager.min_sec_spin.setSingleStep(0.5)
    manager.min_sec_spin.setDecimals(1)
    manager.min_sec_spin.setValue(float(manager.config.get("segment_min_sec", 12.0)))
    p_form.addRow("Min Section Length (s):", manager.min_sec_spin)

    manager.max_k_spin = QSpinBox()
    manager.max_k_spin.setToolTip("Maximum number of structural sections.")
    manager.max_k_spin.setRange(2, 40)
    manager.max_k_spin.setValue(int(manager.config.get("segment_max_k", 20)))
    p_form.addRow("Max Sections:", manager.max_k_spin)

    manager.structure_backend_combo = QComboBox()
    manager.structure_backend_combo.addItems([
        "librosa (default)",
        "songformer (functional labels, Kaggle)",
    ])
    cur_struct = str(manager.config.get("structure_backend", "librosa") or "librosa").lower()
    manager.structure_backend_combo.setCurrentIndex(1 if cur_struct.startswith("song") else 0)
    manager.structure_backend_combo.setToolTip(
        "songformer splits sections into real labels (intro/verse/chorus/bridge/solo/outro) "
        "via a Kaggle GPU kernel; falls back to librosa automatically."
    )
    p_form.addRow("Structure Backend:", manager.structure_backend_combo)

    manager.stem_model_combo = QComboBox()
    manager.stem_model_combo.setToolTip("Demucs model for Kaggle stem separation (htdemucs_6s adds guitar + piano).")
    manager.stem_model_combo.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"])
    cur_stem = manager.config.get("kaggle_stem_model", "htdemucs_ft")
    for i in range(manager.stem_model_combo.count()):
        if manager.stem_model_combo.itemText(i) == cur_stem:
            manager.stem_model_combo.setCurrentIndex(i)
            break
    p_form.addRow("Kaggle Stem Model:", manager.stem_model_combo)

    stem_out_row = QHBoxLayout()
    manager.stem_out_edit = QLineEdit(manager.config.get("stem_output_dir", ""))
    manager.stem_out_edit.setToolTip("Folder where separated stems are written (empty = default ~/mvsep_stems).")
    manager.stem_out_edit.setPlaceholderText("Default: ~/mvsep_stems")
    browse_btn = QPushButton("Browse…")
    browse_btn.clicked.connect(manager._browse_stem_dir)
    stem_out_row.addWidget(manager.stem_out_edit)
    stem_out_row.addWidget(browse_btn)
    p_form.addRow("Stem Output Folder:", stem_out_row)

    manager.lufs_spin = QDoubleSpinBox()
    manager.lufs_spin.setToolTip("Target loudness for DSP normalization (EBU R128).")
    manager.lufs_spin.setRange(-30.0, 0.0)
    manager.lufs_spin.setDecimals(1)
    manager.lufs_spin.setValue(float(manager.config.get("dsp_target_lufs", -14.0)))
    p_form.addRow("Normalize Target (LUFS):", manager.lufs_spin)

    manager.sr_spin = QSpinBox()
    manager.sr_spin.setToolTip("Target sample rate for DSP normalization.")
    manager.sr_spin.setRange(8000, 192000)
    manager.sr_spin.setSingleStep(1000)
    manager.sr_spin.setValue(int(manager.config.get("dsp_target_sr", 44100)))
    p_form.addRow("Normalize Target SR (Hz):", manager.sr_spin)

    save_pipe_btn = QPushButton("Save Pipeline Defaults")
    save_pipe_btn.clicked.connect(manager.save_pipeline_defaults)
    p_form.addRow(save_pipe_btn)

    layout.addWidget(pipe_grp)

    lyrics_grp = QGroupBox("🎤 Lyrics Transcription")
    lf = QFormLayout(lyrics_grp)
    lf.setContentsMargins(8, 18, 8, 8)
    manager.lyrics_engine_combo = QComboBox()
    manager.lyrics_engine_combo.addItems([
        "kaggle (default, GPU)", "whisperx (local)", "gemini", "acestep-transcriber (experimental)",
    ])
    cur_eng = str(manager.config.get("lyrics_engine", "kaggle") or "kaggle").lower()
    for i in range(manager.lyrics_engine_combo.count()):
        if manager.lyrics_engine_combo.itemText(i).lower().startswith(cur_eng):
            manager.lyrics_engine_combo.setCurrentIndex(i)
            break
    manager.lyrics_engine_combo.setToolTip("Which engine transcribes lyrics. Gemini uses the audio-native Gemini backend; ace-step-transcriber is experimental.")
    lf.addRow("Engine:", manager.lyrics_engine_combo)
    manager.lyrics_language_edit = QLineEdit(manager.config.get("lyrics_language", ""))
    manager.lyrics_language_edit.setPlaceholderText("e.g. en (empty = auto-detect)")
    manager.lyrics_language_edit.setToolTip("Force the transcription language (ISO code) or leave empty to auto-detect.")
    lf.addRow("Language:", manager.lyrics_language_edit)
    manager.lyrics_prompt_edit = QLineEdit(manager.config.get("lyrics_initial_prompt", ""))
    manager.lyrics_prompt_edit.setPlaceholderText("e.g. 1970s hard rock by Black Sabbath")
    manager.lyrics_prompt_edit.setToolTip("Biases the transcriber toward context: artist, genre, known words (improves accuracy).")
    lf.addRow("Initial Prompt:", manager.lyrics_prompt_edit)
    layout.addWidget(lyrics_grp)

    mm_grp = QGroupBox("🧰 Model Manager")
    mm_form = QFormLayout(mm_grp)
    mm_form.setContentsMargins(8, 18, 8, 8)

    manager.model_source_combo = QComboBox()
    manager.model_source_combo.setToolTip("Where to download catalog models from.")
    manager.model_source_combo.addItems(["hf (Hugging Face)", "github (my repo)"])
    cur_src = str(manager.config.get("model_download_source", "hf") or "hf").lower()
    manager.model_source_combo.setCurrentIndex(1 if cur_src.startswith("git") else 0)
    mm_form.addRow("Download Source:", manager.model_source_combo)

    manager.model_pick_combo = QComboBox()
    manager.model_pick_combo.setToolTip("The model to download / remove.")
    manager.model_pick_combo.setMinimumWidth(340)
    manager._populate_model_picker()
    mm_form.addRow("Model:", manager.model_pick_combo)

    manager.model_status = QLabel("Select a model to see its status.")
    manager.model_status.setWordWrap(True)
    manager.model_status.setStyleSheet("color: #aaa; font-size: 9px;")
    mm_form.addRow(manager.model_status)

    dl_row = QHBoxLayout()
    download_btn = QPushButton("⬇ Download")
    download_btn.clicked.connect(manager.download_selected_model)
    remove_btn = QPushButton("🗑 Remove")
    remove_btn.clicked.connect(manager.remove_selected_model)
    refresh_btn = QPushButton("↻ Status")
    refresh_btn.clicked.connect(manager.refresh_model_status)
    dl_row.addWidget(download_btn)
    dl_row.addWidget(remove_btn)
    dl_row.addWidget(refresh_btn)
    mm_form.addRow(dl_row)

    manager.leaderboard_combo = QComboBox()
    for item in leaderboards():
        manager.leaderboard_combo.addItem(item["name"], item["url"])
    open_lb = QPushButton("Open")
    open_lb.clicked.connect(manager.open_selected_leaderboard)
    lb_row = QHBoxLayout()
    lb_row.addWidget(manager.leaderboard_combo, 1)
    lb_row.addWidget(open_lb)
    mm_form.addRow("Leaderboards:", lb_row)

    manager.hf_token_edit = QLineEdit(manager.config.get("hf_token", ""))
    manager.hf_token_edit.setToolTip("Hugging Face token for gated model downloads.")
    manager.hf_token_edit.setEchoMode(QLineEdit.Password)
    mm_form.addRow("Hugging Face Token:", manager.hf_token_edit)
    manager.remember_hf = QCheckBox("Remember on this device (encrypted)")
    manager.remember_hf.setChecked(bool(manager.config.get("remember_hf_token", True)))
    mm_form.addRow("", manager.remember_hf)

    manager.model_dir_edit = QLineEdit(manager.config.get("model_dir", "models"))
    manager.model_dir_edit.setToolTip("Folder for downloaded models.")
    manager.model_dir_edit.setToolTip("Folder where downloaded models are stored (gitignored).")
    mm_form.addRow("Models Folder:", manager.model_dir_edit)

    layout.addWidget(mm_grp)
    layout.addStretch()
    scroll.setWidget(inner)
    outer.addWidget(scroll)
