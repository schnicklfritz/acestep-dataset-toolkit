"""Lyrics tab builder for DatasetManager.

Two jobs, one home:
  * **Tidy** — normalize lyrics in place (contractions -> phonetic spelling,
    optional -ing -> -in, capitalize structure tags, strip trailing punctuation).
  * **Edit** — the selected track's lyrics, with a before/after preview.

The contraction table and the -ing exception list are editable here and
persisted to settings, because the right spelling is singer-dependent.

Follows the ui/settings_tab.py pattern: ``build_lyrics_tab(manager, parent)``
stores every widget as ``manager.<name>``.
"""
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.lyrics_normalizer import (
    DEFAULT_CONTRACTIONS,
    DEFAULT_ING_EXCEPTIONS,
)

# Always-visible warning about why this tool exists.
APOSTROPHE_WARNING = (
    "<b>⚠ Apostrophes are unpredictable tokens.</b> A singing/TTS model may read "
    "<code>'</code> as a literal character, a glottal stop, or the wrong vowel — "
    "<code>she'll</code> can come out as \u201cshell\u201d, \u201cshe-ull\u201d or "
    "\u201csheel\u201d with no way to predict which. There is no universal fix, so "
    "this tool replaces contractions with deliberately spelled-out phonetics. "
    "<b>Review the preview before applying.</b>"
)


def _read_contractions(manager):
    """Contraction table from settings, falling back to the shipped default."""
    table = manager.config.get("lyrics_contractions")
    if not isinstance(table, dict) or not table:
        return dict(DEFAULT_CONTRACTIONS)
    return dict(table)


def _read_ing_exceptions(manager):
    raw = manager.config.get("lyrics_ing_exceptions")
    if isinstance(raw, list) and raw:
        return set(raw)
    return set(DEFAULT_ING_EXCEPTIONS)


def build_lyrics_tab(manager, parent):
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
    # Warning + scope
    # ------------------------------------------------------------------
    warn = QLabel(APOSTROPHE_WARNING)
    warn.setWordWrap(True)
    warn.setProperty("health", "warn")
    layout.addWidget(warn)

    scope_row = QHBoxLayout()
    manager.lyrics_scope_combo = _combo(
        ["Selected Track", "All Tracks", "All Tracks (instrumentals excluded)"]
    )
    manager.lyrics_scope_combo.setToolTip(
        "Which tracks the tidy pass rewrites. Instrumental tracks are skipped in "
        "the last option because they have no lyrics."
    )
    scope_row.addWidget(QLabel("Apply to:"))
    scope_row.addWidget(manager.lyrics_scope_combo)
    scope_row.addStretch()
    layout.addLayout(scope_row)

    # ------------------------------------------------------------------
    # Profile selector (master is always on; a profile specialises it)
    # ------------------------------------------------------------------
    prof_row = QHBoxLayout()
    manager.lyrics_profile_combo = _combo(["— master rules only —"])
    manager.lyrics_profile_combo.setToolTip(
        "Master rules always apply. Pick a profile to layer a band's own rules on "
        "top: earlier/specific artists often need their own spellings."
    )
    manager.lyrics_profile_load_btn = QPushButton("Load")
    manager.lyrics_profile_save_btn = QPushButton("Save As…")
    manager.lyrics_profile_save_btn.setToolTip(
        "Save the current table + options as a named profile in lyrics_profiles/."
    )
    manager.lyrics_profile_delete_btn = QPushButton("Delete")
    for w_ in (
        manager.lyrics_profile_load_btn,
        manager.lyrics_profile_save_btn,
        manager.lyrics_profile_delete_btn,
    ):
        w_.setMaximumWidth(90)
    prof_row.addWidget(QLabel("Profile:"))
    prof_row.addWidget(manager.lyrics_profile_combo, 1)
    prof_row.addWidget(manager.lyrics_profile_load_btn)
    prof_row.addWidget(manager.lyrics_profile_save_btn)
    prof_row.addWidget(manager.lyrics_profile_delete_btn)
    layout.addLayout(prof_row)

    manager.lyrics_profile_status = QLabel(
        "Master rules are always applied. Profiles are stored locally in "
        "<code>lyrics_profiles/</code> (gitignored)."
    )
    manager.lyrics_profile_status.setProperty("muted", True)
    manager.lyrics_profile_status.setWordWrap(True)
    layout.addWidget(manager.lyrics_profile_status)

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------
    opts = QGroupBox("Tidy Options")
    o_form = QFormLayout(opts)

    manager.lyrics_ing_check = QCheckBox("Shorten -ing to -in  (running \u2192 runnin)")
    manager.lyrics_ing_check.setToolTip(
        "Singer-dependent: Ozzy Osbourne enunciates '-ing', many singers do not. "
        "Off by default; turn it on to match how the track is actually sung."
    )
    o_form.addRow("", manager.lyrics_ing_check)

    manager.lyrics_tags_check = QCheckBox("Capitalize structure tags  ([verse] \u2192 [Verse])")
    manager.lyrics_tags_check.setChecked(True)
    o_form.addRow("", manager.lyrics_tags_check)

    manager.lyrics_lines_check = QCheckBox("Capitalize the first word of each line  (rising \u2192 Rising)")
    manager.lyrics_lines_check.setChecked(True)
    manager.lyrics_lines_check.setToolTip(
        "Lyric lines are normally written as sentences, so a lowercase start "
        "reads as a typo in a training caption. Only the first character is "
        "touched — the rest of the line keeps its case, so an ALL-CAPS line and "
        "words like iPhone/eBay are left alone."
    )
    o_form.addRow("", manager.lyrics_lines_check)

    manager.lyrics_punct_check = QCheckBox('Strip trailing punctuation  (except " and ))')
    manager.lyrics_punct_check.setChecked(True)
    manager.lyrics_punct_check.setToolTip(
        "Removes . , ! ? ; : - … from the end of each lyric line. A closing "
        'quote " or paren ) is preserved.'
    )
    o_form.addRow("", manager.lyrics_punct_check)

    manager.lyrics_replace_check = QCheckBox("Replace existing lyrics (otherwise fill blanks only)")
    manager.lyrics_replace_check.setChecked(True)
    o_form.addRow("", manager.lyrics_replace_check)
    layout.addWidget(opts)

    # ------------------------------------------------------------------
    # Preview: before | after
    # ------------------------------------------------------------------
    preview = QGroupBox("Preview (selected track)")
    p_layout = QVBoxLayout(preview)
    cols = QHBoxLayout()
    before_col = QVBoxLayout()
    before_col.addWidget(QLabel("<b>Before</b>"))
    manager.lyrics_before = QTextEdit()
    manager.lyrics_before.setReadOnly(True)
    manager.lyrics_before.setMinimumHeight(170)
    before_col.addWidget(manager.lyrics_before)
    after_col = QVBoxLayout()
    after_col.addWidget(QLabel("<b>After</b>"))
    manager.lyrics_after = QTextEdit()
    manager.lyrics_after.setReadOnly(True)
    manager.lyrics_after.setMinimumHeight(170)
    after_col.addWidget(manager.lyrics_after)
    cols.addLayout(before_col)
    cols.addLayout(after_col)
    p_layout.addLayout(cols)

    manager.lyrics_report_label = QLabel("No preview yet.")
    manager.lyrics_report_label.setProperty("muted", True)
    manager.lyrics_report_label.setWordWrap(True)
    p_layout.addWidget(manager.lyrics_report_label)

    # Diff view: the simplest way to see exactly what changed.
    diff_label = QLabel("<b>Changes (diff)</b> — <span style='color:#c33'>− old</span> / "
                        "<span style='color:#3a3'>+ new</span>")
    p_layout.addWidget(diff_label)
    manager.lyrics_diff = QTextEdit()
    manager.lyrics_diff.setReadOnly(True)
    manager.lyrics_diff.setMinimumHeight(120)
    manager.lyrics_diff.setStyleSheet("font-family: monospace;")
    p_layout.addWidget(manager.lyrics_diff)

    preview_row = QHBoxLayout()
    manager.lyrics_preview_btn = QPushButton("🔍 Preview")
    manager.lyrics_preview_btn.setToolTip("Show the tidy result for the selected track without saving.")
    manager.lyrics_apply_btn = QPushButton("✅ Apply Tidy")
    manager.lyrics_apply_btn.setToolTip("Rewrite lyrics on the tracks in scope (undoable).")
    manager.lyrics_manual_edit_btn = QPushButton("📝 Edit Raw Lyrics…")
    manager.lyrics_manual_edit_btn.setToolTip("Open the selected track's lyrics for manual editing.")
    preview_row.addWidget(manager.lyrics_preview_btn)
    preview_row.addWidget(manager.lyrics_apply_btn)
    preview_row.addWidget(manager.lyrics_manual_edit_btn)
    preview_row.addStretch()
    p_layout.addLayout(preview_row)
    layout.addWidget(preview)

    # ------------------------------------------------------------------
    # Editable contraction table
    # ------------------------------------------------------------------
    table_grp = QGroupBox("Contraction \u2192 Phonetic Table (editable)")
    t_layout = QVBoxLayout(table_grp)

    hint = QLabel(
        "Add any word a singer needs mapped; the table is saved to settings. "
        "Matching is whole-word and case-insensitive, preserving capitalization."
    )
    hint.setProperty("muted", True)
    hint.setWordWrap(True)
    t_layout.addWidget(hint)

    manager.lyrics_contract_table = QTableWidget(0, 2)
    manager.lyrics_contract_table.setHorizontalHeaderLabels(["Word", "Say it as"])
    manager.lyrics_contract_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    manager.lyrics_contract_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    manager.lyrics_contract_table.setMinimumHeight(160)
    _fill_contract_table(manager)
    t_layout.addWidget(manager.lyrics_contract_table)

    t_buttons = QHBoxLayout()
    manager.lyrics_add_row_btn = QPushButton("+ Add")
    manager.lyrics_del_row_btn = QPushButton("− Remove Selected")
    manager.lyrics_reset_btn = QPushButton("↺ Reset to Defaults")
    for b in (manager.lyrics_add_row_btn, manager.lyrics_del_row_btn, manager.lyrics_reset_btn):
        t_buttons.addWidget(b)
    t_buttons.addStretch()
    t_layout.addLayout(t_buttons)

    ing_row = QHBoxLayout()
    ing_row.addWidget(QLabel("Never shorten these -ing words:"))
    manager.lyrics_ing_exceptions_edit = QLineEdit(
        ", ".join(sorted(_read_ing_exceptions(manager)))
    )
    manager.lyrics_ing_exceptions_edit.setToolTip(
        "Comma-separated. Words whose -ing is not a verb suffix (ring, thing, king)."
    )
    ing_row.addWidget(manager.lyrics_ing_exceptions_edit, 1)
    t_layout.addLayout(ing_row)
    layout.addWidget(table_grp)

    # ------------------------------------------------------------------
    # All-lyrics view: the whole dataset's lyrics in one editable block
    # ------------------------------------------------------------------
    all_grp = QGroupBox("All Lyrics in the Dataset")
    all_layout = QVBoxLayout(all_grp)

    all_hint = QLabel(
        "Every track's lyrics in one place, prefixed by its filename so you can "
        "see what belongs to what. Tidy this whole block with the options above, "
        "or edit by hand and press <b>Write Back to Tracks</b>."
    )
    all_hint.setProperty("muted", True)
    all_hint.setWordWrap(True)
    all_layout.addWidget(all_hint)

    manager.lyrics_all_edit = QTextEdit()
    manager.lyrics_all_edit.setMinimumHeight(240)
    manager.lyrics_all_edit.setPlaceholderText(
        "Press “Load All Lyrics” to pull in every track's lyrics…"
    )
    all_layout.addWidget(manager.lyrics_all_edit)

    all_row = QHBoxLayout()
    manager.lyrics_load_all_btn = QPushButton("⬇ Load All Lyrics")
    manager.lyrics_load_all_btn.setToolTip(
        "Pull every track's lyrics into the box above, tagged by filename."
    )
    manager.lyrics_tidy_all_btn = QPushButton("✨ Tidy This Block")
    manager.lyrics_tidy_all_btn.setToolTip(
        "Run the tidy options over the whole block (tags are left intact)."
    )
    manager.lyrics_write_back_btn = QPushButton("⬆ Write Back to Tracks")
    manager.lyrics_write_back_btn.setToolTip(
        "Split the block back on the ---- filename ---- markers and save each "
        "track's lyrics. Tracks are matched by filename."
    )
    for b in (
        manager.lyrics_load_all_btn,
        manager.lyrics_tidy_all_btn,
        manager.lyrics_write_back_btn,
    ):
        all_row.addWidget(b)
    all_row.addStretch()
    all_layout.addLayout(all_row)
    layout.addWidget(all_grp)

    layout.addStretch()
    return inner


# Marker used to delimit one track's lyrics inside the all-lyrics block.
TRACK_MARKER = "---- {name} ----"


def build_all_lyrics_block(samples):
    """Render every track's lyrics into one editable block."""
    parts = []
    for s in samples:
        text = s.get("raw_lyrics") or s.get("formatted_lyrics") or s.get("lyrics") or ""
        parts.append(TRACK_MARKER.format(name=s.get("filename", "?")))
        parts.append(text.rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def parse_all_lyrics_block(text):
    """Split the block back into ``{filename: lyrics}``.

    Filenames are used rather than indices so a hand-edited or reordered block
    still maps onto the right tracks.
    """
    out = {}
    current = None
    buf = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("----") and stripped.endswith("----"):
            if current is not None:
                out[current] = "\n".join(buf).strip("\n")
            current = stripped[4:-4].strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip("\n")
    return out


def unified_diff(before, after, context=0):
    """Return a simple line diff: unchanged lines plain, changes prefixed.

    Deliberately minimal — ``-`` marks a removed line, ``+`` an added one. This
    is far simpler than inline per-word highlighting, and easier to read: you
    scan down the block and see exactly what the tidy did.
    """
    import difflib

    old = (before or "").splitlines()
    new = (after or "").splitlines()
    out = []
    for line in difflib.unified_diff(old, new, lineterm="", n=context):
        # Skip the --- / +++ file headers; this is not a real file diff.
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            continue
        out.append(line)
    return "\n".join(out)


def diff_pairs(before, after):
    """Return ``[(old_line, new_line)]`` for lines that differ (None = absent)."""
    import difflib

    old = (before or "").splitlines()
    new = (after or "").splitlines()
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if tag == "equal":
            continue
        o = old[i1:i2]
        n = new[j1:j2]
        for k in range(max(len(o), len(n))):
            pairs.append((o[k] if k < len(o) else None,
                          n[k] if k < len(n) else None))
    return pairs


def _combo(items):
    from modules.wheel_guard import GuardedComboBox
    box = GuardedComboBox()
    box.addItems(items)
    return box


def _fill_contract_table(manager):
    """(Re)populate the contraction table from settings/defaults."""
    table = _read_contractions(manager)
    grid = manager.lyrics_contract_table
    grid.setRowCount(0)
    for word in sorted(table, key=lambda w: (len(w), w)):
        r = grid.rowCount()
        grid.insertRow(r)
        grid.setItem(r, 0, QTableWidgetItem(word))
        grid.setItem(r, 1, QTableWidgetItem(table[word]))


def read_contractions_from_table(manager):
    """Return the contraction dict currently shown in the table."""
    grid = manager.lyrics_contract_table
    out = {}
    for r in range(grid.rowCount()):
        w = grid.item(r, 0)
        v = grid.item(r, 1)
        word = (w.text() if w else "").strip()
        said = (v.text() if v else "").strip()
        if word:
            out[word] = said
    return out


def read_ing_exceptions(manager):
    """Return the -ing exception set from the comma-separated field."""
    raw = manager.lyrics_ing_exceptions_edit.text()
    return {w.strip().lower() for w in raw.split(",") if w.strip()}
