"""Bulk-edit panel ("Set All Tracks") for the Dataset Studio.

WHY THIS EXISTS
---------------
Per-row controls are too easy to change by accident — scrolling the dataset
moves the cursor across a combo and the wheel silently rewrites one track's
value. The fix is a place where dataset-wide values are set *deliberately*:

* the panel is **collapsed by default**, so it costs no space once configured;
* every field has its own **☑ include**, so only ticked fields are applied;
* **Apply is disabled** until at least one field is ticked;
* Apply shows a **confirmation naming the field and the track count**.

Nothing touches the dataset until Apply is pressed and confirmed.

Built by ``build_bulk_edit_panel(manager, parent)`` following the
ui/settings_tab.py pattern; widgets are stored as ``manager.<name>``.
"""
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from modules.dataset_schema import LANGS, TAG_POSITIONS, TIME_SIGNATURES

# Sentinel meaning "leave this field alone" for the combo-based fields.
LEAVE = "— leave unchanged —"


def _combo(items, editable=False):
    from modules.wheel_guard import GuardedComboBox
    box = GuardedComboBox()
    box.addItems(items)
    box.setEditable(editable)
    return box


def _spin(low, high, value, suffix=""):
    from modules.wheel_guard import GuardedSpinBox
    box = GuardedSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


def build_bulk_edit_panel(manager, parent):
    """Add the collapsible "Set All Tracks" panel to ``parent``'s layout."""
    group = QGroupBox("Set All Tracks")
    group.setCheckable(True)
    group.setChecked(False)  # collapsed by default
    group.setToolTip(
        "Apply a value to every track at once. Nothing changes until you press "
        "Apply and confirm — this is the deliberate alternative to editing row "
        "by row, where a stray click or scroll can change a single track."
    )
    outer = QVBoxLayout(group)
    outer.setContentsMargins(8, 18, 8, 8)
    outer.setSpacing(6)

    manager.bulk_pending_label = QLabel("No fields ticked.")
    manager.bulk_pending_label.setProperty("muted", True)
    outer.addWidget(manager.bulk_pending_label)

    # ---- Text fields -------------------------------------------------
    manager.bulk_include = {}

    row = QHBoxLayout()
    manager.bulk_include["genre"] = QCheckBox()
    manager.bulk_genre_edit = QLineEdit()
    manager.bulk_genre_edit.setPlaceholderText("Genre for every track…")
    manager.bulk_include["genre"].setToolTip("Set the same genre on every track.")
    row.addWidget(manager.bulk_include["genre"])
    row.addWidget(QLabel("Genre:"))
    row.addWidget(manager.bulk_genre_edit, 1)

    manager.bulk_include["custom_tag"] = QCheckBox()
    manager.bulk_custom_tag_edit = QLineEdit()
    manager.bulk_custom_tag_edit.setPlaceholderText("Trigger tag…")
    row.addWidget(manager.bulk_include["custom_tag"])
    row.addWidget(QLabel("Tag:"))
    row.addWidget(manager.bulk_custom_tag_edit, 1)
    outer.addLayout(row)

    row = QHBoxLayout()
    manager.bulk_include["prompt_override"] = QCheckBox()
    manager.bulk_prompt_override_check = QCheckBox(
        "Use my handwritten genre / caption instead of auto-generated labels"
    )
    manager.bulk_prompt_override_check.setToolTip(
        "ACE-Step's prompt_override flag. When ticked, the training loader ignores "
        "the auto-generated acestep-captioner strings for these tracks and forces "
        "your own genre / caption / lyrics into the text encoder instead."
    )
    row.addWidget(manager.bulk_include["prompt_override"])
    row.addWidget(QLabel("Prompt override:"))
    row.addWidget(manager.bulk_prompt_override_check, 1)
    outer.addLayout(row)

    # ---- Combo fields ------------------------------------------------
    row = QHBoxLayout()
    manager.bulk_include["language"] = QCheckBox()
    manager.bulk_language_combo = _combo([LEAVE] + list(LANGS), editable=True)
    manager.bulk_language_combo.setToolTip(
        "Language for every track (ISO code). A controlled list avoids typos "
        "like 'englsh'; you can still type a code that is not listed."
    )
    row.addWidget(manager.bulk_include["language"])
    row.addWidget(QLabel("Language:"))
    row.addWidget(manager.bulk_language_combo, 1)
    manager.bulk_language_apply_btn = QPushButton("Apply language to ALL")
    manager.bulk_language_apply_btn.setToolTip(
        "One click: sets the chosen language on every track (asks for confirmation). "
        "Equivalent to ticking Language and pressing Apply to All Tracks."
    )
    row.addWidget(manager.bulk_language_apply_btn)

    manager.bulk_include["timesignature"] = QCheckBox()
    manager.bulk_time_combo = _combo([LEAVE] + list(TIME_SIGNATURES), editable=True)
    row.addWidget(manager.bulk_include["timesignature"])
    row.addWidget(QLabel("Time sig:"))
    row.addWidget(manager.bulk_time_combo, 1)

    manager.bulk_include["instrumental"] = QCheckBox()
    manager.bulk_inst_combo = _combo([LEAVE, "All Instrumental", "All Vocal (No Instrumentals)"])
    manager.bulk_inst_combo.setToolTip("Mark every track instrumental, or every track vocal.")
    row.addWidget(manager.bulk_include["instrumental"])
    row.addWidget(QLabel("Instrumental:"))
    row.addWidget(manager.bulk_inst_combo, 1)

    # ---- Audio path rewrite (the real 47-tracks-by-hand use case) -----
    row = QHBoxLayout()
    manager.bulk_include["audio_path"] = QCheckBox()
    manager.bulk_audio_find_edit = QLineEdit()
    manager.bulk_audio_find_edit.setPlaceholderText("Replace this path fragment…")
    manager.bulk_audio_replace_edit = QLineEdit()
    manager.bulk_audio_replace_edit.setPlaceholderText("…with this")
    manager.bulk_audio_find_edit.setToolTip(
        "Rewrites audio_path on every track. Example: find 'D:/ripped/' and "
        "replace with '/home/me/music/' after moving the files."
    )
    row.addWidget(manager.bulk_include["audio_path"])
    row.addWidget(QLabel("Audio path:"))
    row.addWidget(manager.bulk_audio_find_edit, 1)
    row.addWidget(QLabel("→"))
    row.addWidget(manager.bulk_audio_replace_edit, 1)
    outer.addLayout(row)

    # ---- Dataset-wide values ----------------------------------------
    row = QHBoxLayout()

    manager.bulk_include["genre_ratio"] = QCheckBox()
    manager.bulk_genre_ratio_spin = _spin(0, 100, 0, "%")
    manager.bulk_genre_ratio_spin.setToolTip(
        "TRAINING STABILITY KNOB (dataset-wide).\n\n"
        "0%  = the loader always uses the detailed caption.\n"
        "100% = the loader always uses the short genre tag instead.\n\n"
        "This is a probability gate applied across the whole training run, not a "
        "style preference. Raising it makes the LoRA generalise to short prompts "
        "instead of overfitting to your specific caption wording."
    )
    row.addWidget(manager.bulk_include["genre_ratio"])
    row.addWidget(QLabel("Genre ratio:"))
    row.addWidget(manager.bulk_genre_ratio_spin)

    manager.bulk_include["tag_position"] = QCheckBox()
    manager.bulk_tag_pos_combo = _combo([LEAVE] + list(TAG_POSITIONS))
    manager.bulk_tag_pos_combo.setToolTip("Where the trigger tag is placed in a prompt.")
    row.addWidget(manager.bulk_include["tag_position"])
    row.addWidget(QLabel("Tag position:"))
    row.addWidget(manager.bulk_tag_pos_combo)

    manager.bulk_include["duration"] = QCheckBox()
    manager.bulk_duration_spin = _spin(0, 7200, 0, " s")
    manager.bulk_duration_spin.setToolTip(
        "Set the same duration on every track. 0 means 'clear to unknown'."
    )
    row.addWidget(manager.bulk_include["duration"])
    row.addWidget(QLabel("Duration:"))
    row.addWidget(manager.bulk_duration_spin)
    row.addStretch()
    outer.addLayout(row)

    # ---- Apply ------------------------------------------------------
    apply_row = QHBoxLayout()
    manager.bulk_apply_btn = QPushButton("Apply to All Tracks")
    manager.bulk_apply_btn.setEnabled(False)  # nothing ticked yet
    manager.bulk_apply_btn.setToolTip("Disabled until at least one field is ticked.")
    manager.bulk_clear_btn = QPushButton("Untick All")
    apply_row.addWidget(manager.bulk_apply_btn)
    apply_row.addWidget(manager.bulk_clear_btn)
    apply_row.addStretch()
    outer.addLayout(apply_row)

    # Keep the pending-count label and the Apply button state in sync.
    for box in manager.bulk_include.values():
        box.stateChanged.connect(manager._bulk_update_pending)
    manager.bulk_clear_btn.clicked.connect(manager._bulk_untick_all)
    manager._bulk_update_pending()

    parent.addWidget(group)
    manager.bulk_group = group
    return group


# ---------------------------------------------------------------------------
# Reading / applying
# ---------------------------------------------------------------------------
def read_bulk_edits(manager):
    """Return ``{field: value}`` for every ticked field.

    Combo fields left on the LEAVE sentinel are treated as "no value supplied"
    and reported as empty strings, so the caller can clear rather than skip.
    """
    inc = manager.bulk_include
    out = {}

    if inc["genre"].isChecked():
        out["genre"] = manager.bulk_genre_edit.text().strip()
    if inc["custom_tag"].isChecked():
        out["custom_tag"] = manager.bulk_custom_tag_edit.text().strip()
    if inc["prompt_override"].isChecked():
        # Boolean data-source switch (ACE-Step `prompt_override`).
        out["prompt_override"] = manager.bulk_prompt_override_check.isChecked()
    if inc["language"].isChecked():
        out["language"] = manager.bulk_language_combo.currentText().strip()
    if inc["timesignature"].isChecked():
        out["timesignature"] = manager.bulk_time_combo.currentText().strip()
    if inc["duration"].isChecked():
        out["duration"] = int(manager.bulk_duration_spin.value())
    if inc["instrumental"].isChecked():
        mode = manager.bulk_inst_combo.currentText()
        if mode.startswith("All Instrumental"):
            out["is_instrumental"] = True
        elif mode.startswith("All Vocal"):
            out["is_instrumental"] = False

    meta = {}
    if inc["genre_ratio"].isChecked():
        meta["genre_ratio"] = int(manager.bulk_genre_ratio_spin.value())
    if inc["tag_position"].isChecked():
        pos = manager.bulk_tag_pos_combo.currentText()
        if not pos.startswith(LEAVE[:1]):
            meta["tag_position"] = pos

    rewrite = None
    if inc["audio_path"].isChecked():
        find = manager.bulk_audio_find_edit.text()
        repl = manager.bulk_audio_replace_edit.text()
        if find:
            rewrite = (find, repl)

    return out, meta, rewrite
