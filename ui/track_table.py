"""Track-table column policy: ONE definition of the columns and their widths.

WHY THIS MODULE EXISTS
----------------------
The column set used to be declared in three places that had to agree: the
``setHorizontalHeaderLabels`` list and the ``setSectionResizeMode`` loop in
``dataset_manager.init_ui``, the ``_MANUAL_COLS`` index table used to write
edits back onto a sample, and ``ui.shell.install_column_menu``, which re-set
``Stretch`` on column 0 behind the first two. Nothing checked that they agreed.

They stopped agreeing. An ``Instr`` column was inserted at index 8 while
``refresh_table`` kept placing the Actions button widget at index 8 too, so the
buttons were painted over the checkbox: a click on the checkbox indicator left
``checkState`` untouched, and the real Actions column rendered empty. That is a
silent-wrong-answer bug produced purely by duplicated indexing, so the indices
now live in exactly one place and every consumer imports them.

WHY THE SECTIONS ARE ``Interactive``
-----------------------------------
``ResizeToContents`` and ``Stretch`` both mean the size "cannot be changed by
the user or programmatically" (Qt, QHeaderView::ResizeMode). With the header in
those modes the widths are chosen for the user, and the Filename column — the
one worth reading — is the ``Stretch`` column that absorbs whatever the
content-sized columns (Genre, and formerly the two removed ones) leave behind.
It collapsed to a few characters. ``Interactive`` gives every column a readable
default AND a draggable edge; the result is persisted by name (see
``save_column_widths``) so the user's adjustment survives a restart.
"""
from PySide6.QtCore import Qt

from modules.dataset_schema import LANGS, LANG_INSTRUMENTAL

# ---------------------------------------------------------------------------
# The column set. (header, default width in px)
# ---------------------------------------------------------------------------
# Widths are defaults, not policy: every column is user-resizable. ``Language``
# is sized for an ISO code (2 characters); the "instrumental" entry is a
# selector option, not text that has to fit — see ``language_display``.
COLUMNS = (
    ("Filename", 300),   # the one column a user actually reads; not in _MANUAL_COLS
    ("Tag", 60),
    ("Genre", 110),      # elided + full text in the tooltip; draggable
    ("Language", 28),
    ("Key", 70),
    ("BPM", 48),
    ("Time", 40),
    ("Duration", 60),
)

HEADERS = tuple(name for name, _ in COLUMNS)
DEFAULT_WIDTHS = dict(COLUMNS)

# Columns hidden by default in the narrow track list; every one is still a
# right-click on the header away. ``Language`` is deliberately NOT hidden: it
# carries the instrumental state too (blank == instrumental), so it belongs in
# the default view.
DEFAULT_HIDDEN_COLUMNS = ["Time", "Duration"]

# Narrower than Qt's 36px default, which would silently override the Language
# column's 28px default and waste space on every other column too.
MIN_SECTION_SIZE = 24

# Item data role holding the canonical Language choice for a cell. The Display
# text cannot carry it: the column is 28px wide, so "instrumental" would be
# elided to something unrecognisable. The role keeps the real value, the text
# stays short, and both are written by the editor in one place.
LANG_ROLE = Qt.UserRole + 1

# What the editor offers. A blank row first so "no language" (which means
# instrumental — see modules.dataset_schema.LANG_INSTRUMENTAL) is reachable
# without knowing the convention.
NO_LANGUAGE = ""
LANGUAGE_CHOICES = (NO_LANGUAGE, LANG_INSTRUMENTAL) + tuple(
    code for code in LANGS if code != LANG_INSTRUMENTAL
)


def choice_from_sample(sample):
    """Canonical Language choice for a sample (what the editor shows)."""
    if sample.get("is_instrumental") or sample.get("language", "") == LANG_INSTRUMENTAL:
        return LANG_INSTRUMENTAL
    return (sample.get("language") or "").strip()


def language_display(choice):
    """Compact cell text for a Language choice.

    A real code is shown verbatim (it fits). ``instrumental`` becomes a mark
    because the word cannot fit a 2-character column; the tooltip carries it.
    """
    if choice == LANG_INSTRUMENTAL:
        return "🎸"
    return choice or ""


def language_tooltip(choice):
    if choice == LANG_INSTRUMENTAL:
        return "Instrumental — no vocals. Exported as instrumental: true."
    if not choice:
        return "No language set. A track with no language is an instrumental."
    return f"Language: {choice}"


def column_index(table, name):
    """Index of the column whose header is ``name``; -1 when absent.

    Looked up by NAME rather than hard-coded: a hard-coded index is exactly
    what allowed the Actions widget and the Instr checkbox to collide.
    """
    for i in range(table.columnCount()):
        item = table.horizontalHeaderItem(i)
        if item is not None and item.text() == name:
            return i
    return -1



def install_table_policy(manager, table):
    """Apply the column set, resize modes, widths and hidden columns.

    Called from ``dataset_manager.init_ui`` before the layout is assembled, so
    the table is consistent even when ``ui.shell`` is not involved (as in the
    tests). ``ui.shell.install_column_menu`` only adds the header menu on top.
    """
    from PySide6.QtWidgets import QHeaderView

    header = table.horizontalHeader()
    # Interactive, not Stretch: only this mode lets the user drag a column edge.
    header.setStretchLastSection(False)
    header.setMinimumSectionSize(MIN_SECTION_SIZE)
    for i in range(table.columnCount()):
        header.setSectionResizeMode(i, QHeaderView.Interactive)
    apply_column_widths(manager, table)
    apply_hidden_columns(manager, table)

    # The Language cell is chosen from a list, never typed: a free-text ISO code
    # invites 'englsh' and cannot offer the instrumental option at all.
    lang_col = column_index(table, "Language")
    if lang_col >= 0:
        table.setItemDelegateForColumn(lang_col, _language_delegate(table))


# ---------------------------------------------------------------------------
# Widths / visibility persistence
# ---------------------------------------------------------------------------
def _saved_widths(manager):
    saved = manager.config.get("ui_column_widths")
    return saved if isinstance(saved, dict) else {}


def apply_column_widths(manager, table):
    """Set defaults, then overlay any saved widths (by header name)."""
    saved = _saved_widths(manager)
    header = table.horizontalHeader()
    for i in range(table.columnCount()):
        item = table.horizontalHeaderItem(i)
        name = item.text() if item is not None else ""
        width = saved.get(name)
        if not isinstance(width, int) or width < MIN_SECTION_SIZE:
            width = DEFAULT_WIDTHS.get(name)
        if width:
            header.resizeSection(i, width)


def apply_hidden_columns(manager, table):
    hidden = manager.config.get("ui_hidden_columns")
    if hidden is None or not isinstance(hidden, list):
        hidden = list(DEFAULT_HIDDEN_COLUMNS)
    for i in range(table.columnCount()):
        item = table.horizontalHeaderItem(i)
        name = item.text() if item is not None else ""
        # Column 0 is never hidden: it identifies the row.
        table.setColumnHidden(i, name in hidden and i != 0)


def save_column_widths(manager):
    """Persist the current widths by header name (called on close).

    NAME-keyed on purpose. Index-keyed widths are applied to whatever column
    later occupies that index, which is the same class of bug this module was
    created to remove.
    """
    from modules.config_store import save_plain_keys

    table = getattr(manager, "table", None)
    if table is None:
        return
    header = table.horizontalHeader()
    widths = {}
    for i in range(table.columnCount()):
        item = table.horizontalHeaderItem(i)
        if item is None:
            continue
        # A hidden section is not drawn and reports a junk width; skip it so a
        # hide/show cycle cannot overwrite a saved size with 0.
        if table.isColumnHidden(i):
            continue
        widths[item.text()] = int(header.sectionSize(i))
    manager.config["ui_column_widths"] = widths
    save_plain_keys(manager.config, ["ui_column_widths"])


# ---------------------------------------------------------------------------
# Editors
# ---------------------------------------------------------------------------
def _language_delegate(table):
    from PySide6.QtWidgets import QStyledItemDelegate

    from modules.wheel_guard import GuardedComboBox

    class _LanguageDelegate(QStyledItemDelegate):
        """Combo editor for the Language column.

        Guarded (not a bare QComboBox) because the table sits in a scroll area:
        scrolling past a cell being edited must not cycle its value, which is
        the same accident modules/wheel_guard.py exists to prevent.
        """

        def createEditor(self, parent, option, index):
            box = GuardedComboBox(parent)
            box.addItems(list(LANGUAGE_CHOICES))
            return box

        def setEditorData(self, editor, index):
            choice = index.data(LANG_ROLE)
            if not choice:
                # Fall back to the Display text for rows refreshed before the
                # role existed (or any item built by hand).
                text = index.data(Qt.DisplayRole) or ""
                mark = language_display(LANG_INSTRUMENTAL)
                choice = LANG_INSTRUMENTAL if text == mark else text
            pos = editor.findText(choice)
            editor.setCurrentIndex(pos if pos >= 0 else 0)

        def setModelData(self, editor, model, index):
            choice = (editor.currentText() or "").strip()
            # Role first: the edit handler reads it, and it must be in place
            # before the DisplayRole change raises dataChanged.
            model.setData(index, choice, LANG_ROLE)
            model.setData(index, language_display(choice), Qt.DisplayRole)
            model.setData(index, language_tooltip(choice), Qt.ToolTipRole)

    return _LanguageDelegate(table)
