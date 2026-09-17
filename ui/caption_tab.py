"""Caption tab builder for DatasetManager.

Extracted from the ⚙ Settings tab (backend, prompt, limits) plus the Dataset
Studio inspector (caption/lyrics editors) so captioning has one coherent home
and the Studio regains its width.

Follows the ui/settings_tab.py pattern: one function, ``build_caption_tab``,
which builds widgets directly onto ``manager`` and stores them as
``manager.<name>`` so existing DatasetManager methods (save_pipeline_defaults,
start_ai_captioning, review_ai_caption_result, ...) keep working by name.

The style <-> tag blend is exposed as a **slider** (0 = prose only,
100 = tags only). It is a guarded slider, so scrolling past it cannot move it.
A dataset-wide value lives in config["tag_caption_ratio"]; a per-track override
is stored on the sample as ``prompt_override`` (None = follow the dataset).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
# Wheel-guarded slider: scrolling past it cannot move the blend ratio.
from modules.wheel_guard import GuardedSlider as QSlider

# Backend labels -> config value in config["caption_backend"].
CAPTION_BACKENDS = (
    ("ace_step", "ACE-Step captioner (Kaggle GPU)"),
    ("gemini", "Google Gemini (audio-native)"),
    ("deepseek", "DeepSeek LLM (text-only synthesis)"),
    ("custom", "Custom OpenAI-compatible endpoint"),
)


def _blend_labels(slider, left, right, value_label=None):
    """Show a live prose/tags description plus a percentage for a blend slider."""
    def _update(value):
        if value <= 0:
            name = f"{left} only (0%)"
        elif value >= 100:
            name = f"{right} only (100%)"
        else:
            name = f"{100 - value}% {left} / {value}% {right}"
        if value_label is not None:
            value_label.setText(name)
    slider.valueChanged.connect(_update)
    _update(slider.value())


def _spin(low, high, value):
    """Small guarded spin box (see modules/wheel_guard.py)."""
    from modules.wheel_guard import GuardedSpinBox

    box = GuardedSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    return box



def build_caption_tab(manager, parent):
    outer = QVBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    outer.addWidget(scroll)

    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(10)
    scroll.setWidget(inner)

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------
    backend_grp = QGroupBox("Caption Backend")
    b_form = QFormLayout(backend_grp)

    manager.caption_backend_combo = QComboBox()
    manager.caption_backend_combo.setToolTip(
        "Which model describes the audio. ACE-Step runs on a free Kaggle GPU; "
        "Gemini understands audio natively; DeepSeek and Custom are text-only."
    )
    for _value, label in CAPTION_BACKENDS:
        manager.caption_backend_combo.addItem(label, _value)
    cur_backend = (manager.config.get("caption_backend") or "ace_step").strip().lower()
    for i in range(manager.caption_backend_combo.count()):
        if manager.caption_backend_combo.itemData(i) == cur_backend:
            manager.caption_backend_combo.setCurrentIndex(i)
            break
    b_form.addRow("Backend:", manager.caption_backend_combo)

    manager.gemini_model_combo = QComboBox()
    manager.gemini_model_combo.setEditable(True)
    manager.gemini_model_combo.addItems(
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    )
    manager.gemini_model_combo.setCurrentText(
        manager.config.get("gemini_model", "gemini-2.5-flash")
    )
    manager.gemini_model_combo.setToolTip("Gemini model used when the Gemini backend is active.")
    b_form.addRow("Gemini model:", manager.gemini_model_combo)

    manager.custom_url_edit = QLineEdit(manager.config.get("custom_caption_url", ""))
    manager.custom_url_edit.setToolTip(
        "OpenAI-compatible base URL for the custom backend (e.g. http://localhost:8000/v1)."
    )
    b_form.addRow("Custom URL:", manager.custom_url_edit)

    manager.custom_model_edit = QLineEdit(manager.config.get("custom_caption_model", ""))
    manager.custom_model_edit.setToolTip("Model name served by the custom endpoint.")
    b_form.addRow("Custom model:", manager.custom_model_edit)

    manager.custom_audio_check = QCheckBox("Send audio (requires model audio support)")
    manager.custom_audio_check.setChecked(bool(manager.config.get("custom_caption_audio", False)))
    manager.custom_audio_check.setToolTip(
        "Send audio via the OpenAI input_audio field when the endpoint supports it."
    )
    b_form.addRow("", manager.custom_audio_check)

    # ------------------------------------------------------------------
    # Style <-> Tags blend (the slider)
    # ------------------------------------------------------------------
    blend_grp = QGroupBox("Caption Style — Prose vs Tags")
    blend_layout = QVBoxLayout(blend_grp)

    manager.caption_blend_label = QLabel()
    manager.caption_blend_label.setStyleSheet("font-weight: bold;")
    blend_layout.addWidget(manager.caption_blend_label)

    blend_row = QHBoxLayout()
    manager.caption_blend_slider = QSlider(Qt.Horizontal)
    manager.caption_blend_slider.setRange(0, 100)
    manager.caption_blend_slider.setTickInterval(10)
    manager.caption_blend_slider.setTickPosition(QSlider.TicksBelow)
    manager.caption_blend_slider.setToolTip(
        "0% = prose descriptions only (default). 100% = a tag block only. "
        "In between = both, tags first. Scrolling past cannot move this slider — "
        "click it first."
    )
    manager.caption_blend_slider.setValue(int(manager.config.get("tag_caption_ratio", 0) or 0))
    blend_row.addWidget(QLabel("Prose"))
    blend_row.addWidget(manager.caption_blend_slider, 1)
    blend_row.addWidget(QLabel("Tags"))
    blend_layout.addLayout(blend_row)
    _blend_labels(manager.caption_blend_slider, "prose", "tags", manager.caption_blend_label)

    note = QLabel(
        "Applies to the whole dataset. Select a track and use "
        "“Override for This Track” for a per-track setting."
    )
    note.setStyleSheet("color: #999;")
    note.setWordWrap(True)
    blend_layout.addWidget(note)

    override_row = QHBoxLayout()
    manager.caption_override_btn = QPushButton("Override for This Track…")
    manager.caption_override_btn.setToolTip(
        "Store a blend ratio on just the selected track (sample['prompt_override'])."
    )
    manager.caption_clear_override_btn = QPushButton("Clear Track Override")
    manager.caption_clear_override_btn.setToolTip(
        "Remove the per-track override so the track follows the dataset value again."
    )
    override_row.addWidget(manager.caption_override_btn)
    override_row.addWidget(manager.caption_clear_override_btn)
    override_row.addStretch()
    blend_layout.addLayout(override_row)
    layout.addWidget(blend_grp)

    # ------------------------------------------------------------------
    # Prompt + limits
    # ------------------------------------------------------------------
    prompt_grp = QGroupBox("Caption Prompt & Limits")
    p_form = QFormLayout(prompt_grp)

    manager.prompt_edit = QTextEdit()
    manager.prompt_edit.setPlainText(manager.config.get("caption_prompt", ""))
    manager.prompt_edit.setMaximumHeight(140)
    manager.prompt_edit.setPlaceholderText("Instruction given to the caption model for every chunk.")
    manager.prompt_edit.setToolTip("Edit to steer how descriptions are written.")
    p_form.addRow("Caption Prompt:", manager.prompt_edit)

    manager.max_tokens_spin = _spin(64, 4096, int(manager.config.get("caption_max_tokens", 512)))
    manager.max_tokens_spin.setToolTip("Maximum tokens the captioner may generate per chunk.")
    p_form.addRow("Max tokens:", manager.max_tokens_spin)

    manager.max_dur_spin = _spin(0, 3600, int(manager.config.get("caption_max_audio_duration", 120)))
    manager.max_dur_spin.setToolTip(
        "Max audio length fed to the captioner in seconds (0 = whole file)."
    )
    p_form.addRow("Max audio (sec):", manager.max_dur_spin)

    manager.batch_size_spin = _spin(1, 64, int(manager.config.get("caption_batch_size", 1)))
    manager.batch_size_spin.setToolTip("Chunks per forward pass on the captioning GPU.")
    p_form.addRow("Batch size:", manager.batch_size_spin)
    layout.addWidget(prompt_grp)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    actions_grp = QGroupBox("Run")
    a_layout = QHBoxLayout(actions_grp)

    manager.caption_selected_btn = QPushButton("🚀 Caption Selected")
    manager.caption_selected_btn.setToolTip("Caption the track(s) selected in the Dataset Studio table.")
    manager.caption_missing_btn = QPushButton("Caption Missing")
    manager.caption_missing_btn.setToolTip("Caption every track that has no caption yet.")
    manager.caption_all_btn = QPushButton("🔁 Re-caption All")
    manager.caption_all_btn.setToolTip("Re-run the captioner over every track (asks for confirmation).")
    manager.caption_edit_btn = QPushButton("📝 Edit Caption…")
    manager.caption_edit_btn.setToolTip("Open the caption / lyrics editor for the selected track.")
    for btn in (
        manager.caption_selected_btn,
        manager.caption_missing_btn,
        manager.caption_all_btn,
        manager.caption_edit_btn,
    ):
        a_layout.addWidget(btn)
    a_layout.addStretch()
    layout.addWidget(actions_grp)

    layout.addStretch()
    return inner

    layout.addWidget(backend_grp)
