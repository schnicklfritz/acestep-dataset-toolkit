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
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
# Wheel-guarded slider: scrolling past it cannot move the blend ratio.
from modules.wheel_guard import GuardedSlider as QSlider


class TrackPickerButton(QToolButton):
    """Dropdown listing the dataset's tracks with checkboxes.

    Lets the user tick any subset of tracks for a run. Implemented as a
    QToolButton + InstantPopup menu rather than a QComboBox, because a Qt combo
    box with per-item checkboxes needs a custom item model and a delegate, and
    its popup misbehaves with many rows. A menu scrolls on its own and is far
    less code for the same UX.

    Ticks are keyed by FILENAME (not row index) so they survive table refreshes,
    dataset reloads, and re-ordering.
    """

    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Tracks ▾ (0 selected)")
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolTip(
            "Tick the tracks to process. Selections survive table refreshes and "
            "are matched by filename, not row."
        )
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._checked = set()          # filenames that are ticked
        self._actions = {}             # filename -> QAction
        self._known = []               # filenames currently in the dataset
        self._samples = []             # last dataset passed to set_tracks
        self._build_static_actions()
        self.menu().aboutToShow.connect(self._sync_menu_from_state)

    # -- static (quick) actions -------------------------------------------
    def _build_static_actions(self):
        for label, slot in (
            ("Select all", self.select_all),
            ("Select none", self.select_none),
            ("Select tracks missing captions", self.select_missing_captions),
        ):
            act = QAction(label, self)
            act.triggered.connect(slot)
            self._menu.addAction(act)
        self._menu.addSeparator()

    # -- population --------------------------------------------------------
    def set_tracks(self, samples):
        """(Re)build the per-track list from dataset samples.

        Existing ticks survive: a filename that was ticked stays ticked as long
        as it is still present.

        Called from refresh_table(), which fires often, so it short-circuits when
        the track list has not actually changed -- rebuilding N QActions on every
        table refresh would be wasteful on a large dataset.
        """
        samples = list(samples or [])
        new_names = [(s.get("filename") or "").strip() for s in samples]
        new_names = [n for n in new_names if n]

        if new_names == self._known:
            # Same tracks: just keep the samples current so
            # select_missing_captions() sees fresh captions. No menu rebuild.
            self._samples = samples
            return

        self._samples = samples
        self._known = new_names

        # Drop ticks for tracks that no longer exist.
        self._checked &= set(self._known)

        # Rebuild the per-track section, keeping the quick actions at the top.
        for act in self._actions.values():
            self._menu.removeAction(act)
        self._actions.clear()

        for name in self._known:
            act = QAction(name, self)
            act.setCheckable(True)
            act.setChecked(name in self._checked)
            act.toggled.connect(
                lambda checked, n=name: self._on_track_toggled(n, checked)
            )
            self._menu.addAction(act)
            self._actions[name] = act

        self._update_label()

    def refresh(self, samples=None):
        """Alias kept for callers that just want the list rebuilt."""
        if samples is None:
            return
        self.set_tracks(samples)

    def _sync_menu_from_state(self):
        for name, act in self._actions.items():
            act.blockSignals(True)
            act.setChecked(name in self._checked)
            act.blockSignals(False)

    # -- selection ---------------------------------------------------------
    def _on_track_toggled(self, name, checked):
        if checked:
            self._checked.add(name)
        else:
            self._checked.discard(name)
        self._update_label()
        self.selection_changed.emit()

    def select_missing_captions(self, *_):
        """Tick tracks whose caption is blank.

        Takes ``*_`` because QAction.triggered passes a ``checked`` bool that
        must NOT be swallowed as a real argument.
        """
        missing = {
            (s.get("filename") or "").strip()
            for s in self._samples
            if not (s.get("caption") or "").strip()
        }
        self._checked = {n for n in self._known if n in missing}
        self._sync_menu_from_state()
        self._update_label()
        self.selection_changed.emit()

    def select_all(self, *_):
        self._checked = set(self._known)
        self._sync_menu_from_state()
        self._update_label()
        self.selection_changed.emit()

    def select_none(self, *_):
        self._checked.clear()
        self._sync_menu_from_state()
        self._update_label()
        self.selection_changed.emit()

    def selected_filenames(self):
        """Ticked filenames, in dataset order (not click order)."""
        return [n for n in self._known if n in self._checked]

    def is_empty(self):
        return not self._checked

    # -- label -------------------------------------------------------------
    def _update_label(self):
        total = len(self._known)
        self.setText(f"Tracks ▾ ({len(self._checked)} of {total} selected)")


# Backend labels -> config value in config["caption_backend"].
CAPTION_BACKENDS = (
    ("ace_step", "ACE-Step captioner (Kaggle GPU)"),
    ("moss", "MOSS-Audio on Kaggle (open model, raw style + lyrics)"),
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
    note.setProperty("muted", True)
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
    # Prompt
    # ------------------------------------------------------------------
    prompt_grp = QGroupBox("Caption Prompt")
    p_form = QFormLayout(prompt_grp)

    manager.prompt_edit = QTextEdit()
    manager.prompt_edit.setPlainText(manager.config.get("caption_prompt", ""))
    manager.prompt_edit.setMaximumHeight(140)
    manager.prompt_edit.setPlaceholderText("Instruction given to the caption model for every chunk.")
    manager.prompt_edit.setToolTip(
        "The user turn: what to do with this clip. The annotation schema itself is "
        "built in and sent as the system prompt."
    )
    p_form.addRow("Caption Prompt:", manager.prompt_edit)

    manager.system_prompt_edit = QTextEdit()
    manager.system_prompt_edit.setPlainText(
        manager.config.get("caption_system_prompt", "")
    )
    manager.system_prompt_edit.setMaximumHeight(90)
    manager.system_prompt_edit.setPlaceholderText(
        "Extra instructions, appended to the built-in ACE-Step 1.5XL schema..."
    )
    manager.system_prompt_edit.setToolTip(
        "The ACE-Step 1.5XL caption schema is built in (modules/caption_spec.py) and "
        "cannot be replaced — this text is APPENDED to it. Use it for house style or "
        "per-artist emphasis."
    )
    p_form.addRow("System Prompt (added):", manager.system_prompt_edit)
    layout.addWidget(prompt_grp)

    # ------------------------------------------------------------------
    # MOSS-Audio on Kaggle (open model)
    # ------------------------------------------------------------------
    moss_grp = QGroupBox("MOSS-Audio on Kaggle (open model)")
    m_layout = QVBoxLayout(moss_grp)

    moss_note = QLabel(
        "Runs the open <b>MOSS-Audio</b> model on a free Kaggle GPU. It writes raw "
        "style text into <i>caption</i> and raw lyrics into <i>raw_lyrics</i>, then "
        "you format them with the Structural Tag Creator. "
        "Requires Kaggle credentials in ⚙ Settings."
    )
    moss_note.setProperty("muted", True)
    moss_note.setWordWrap(True)
    m_layout.addWidget(moss_note)

    m_row = QHBoxLayout()
    manager.moss_track_picker = TrackPickerButton()
    manager.moss_track_picker.setToolTip(
        "Tick the tracks to send to MOSS. 'Select tracks missing captions' is a "
        "quick way to only process what needs it."
    )
    m_row.addWidget(manager.moss_track_picker, 1)
    m_layout.addLayout(m_row)

    m_row = QHBoxLayout()
    manager.moss_model_edit = QLineEdit(
        manager.config.get("moss_model_id", "")
        or "OpenMOSS-Team/MOSS-Music-8B-Instruct"
    )
    manager.moss_model_edit.setToolTip(
        "Hugging Face repo id for the MOSS weights.\n\n"
        "MOSS-Music-8B-Instruct (recommended) is music-specialised: its tags are "
        "music-captioning / lyrics-asr / chord-recognition. "
        "MOSS-Audio-8B-Instruct is the general speech+environment+music model.\n\n"
        "Both are ~17 GiB and shard across Kaggle's two T4s. The kernel derives "
        "the repo and class names from this id, so either works."
    )
    m_row.addWidget(QLabel("Model:"))
    m_row.addWidget(manager.moss_model_edit, 1)
    m_layout.addLayout(m_row)

    m_row2 = QHBoxLayout()
    manager.moss_weights_dataset_edit = QLineEdit(
        manager.config.get("moss_model_dataset", "")
    )
    manager.moss_weights_dataset_edit.setToolTip(
        "Optional: a private Kaggle dataset holding the MOSS weights, e.g. "
        "'you/moss-audio-8b'. Set this to skip a ~17 GiB download inside the "
        "kernel on every run."
    )
    m_row2.addWidget(QLabel("Cached weights (optional):"))
    m_row2.addWidget(manager.moss_weights_dataset_edit, 1)
    m_layout.addLayout(m_row2)

    m_run = QHBoxLayout()
    manager.moss_run_btn = QPushButton("🚀 Caption via MOSS (Kaggle)")
    manager.moss_run_btn.setToolTip(
        "Uploads the selected tracks, runs MOSS-Audio on a Kaggle GPU, then writes "
        "the raw style + lyrics back into the dataset. Nothing is overwritten: "
        "previous values are kept in caption_before_moss / raw_lyrics_before_moss."
    )
    manager.moss_open_btn = QPushButton("Open last Kaggle run")
    manager.moss_open_btn.setToolTip(
        "Open the most recent MOSS kernel's output page in Kaggle to inspect the log."
    )
    m_run.addWidget(manager.moss_run_btn)
    m_run.addWidget(manager.moss_open_btn)
    m_run.addStretch()
    m_layout.addLayout(m_run)

    manager.moss_status = QLabel("Not run yet.")
    manager.moss_status.setProperty("muted", True)
    manager.moss_status.setWordWrap(True)
    m_layout.addWidget(manager.moss_status)
    layout.addWidget(moss_grp)

    layout.addStretch()
    return inner
