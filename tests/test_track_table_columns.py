"""The Studio track table: which columns show, how they are sized, and the
Language column that carries the instrumental state.

Regressions pinned here:

  * ``Language`` was hidden by default even though every exported sample
    carries it — the column read as "missing" because it was one right-click
    away.
  * ``Instr`` and ``Actions`` were removed. ``instrumental`` is now an option
    on the Language column (no language ⇒ instrumental), so the checkbox cell
    and its separate column are gone.
  * Column indices were duplicated between ``init_ui``, the row/cell write-back
    table and ``ui.shell``. They collided: the Actions button widget stayed on
    index 8 while ``Instr`` moved into it, so the buttons were painted over the
    checkbox and a click left it unchecked. Everything now resolves a column by
    HEADER NAME (``_col()`` below, ``dataset_manager._manual_field``); no test
    here depends on a hard-coded index.
  * No column was user-resizable (``ResizeToContents``/``Stretch`` are
    documented as sizes the user cannot change), and the ``Stretch`` Filename
    column collapsed to a few characters at a narrow width.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QHeaderView  # noqa: E402

from modules.dataset_schema import (  # noqa: E402
    LANG_INSTRUMENTAL,
    new_dataset,
    new_sample,
    to_export_sample,
)
from ui.track_table import (  # noqa: E402
    COLUMNS,
    DEFAULT_HIDDEN_COLUMNS,
    DEFAULT_WIDTHS,
    LANG_ROLE,
)


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One window for the whole module, as in tests/test_lyrics_inline_editor.py.

    Building a ``DatasetManager`` applies the theme to every page, which costs
    tens of seconds here; a per-test window would make this file the slowest in
    the suite for no extra coverage. Every test below reloads the dataset (and
    any config value it touches) before asserting, so state cannot leak.
    """
    import config
    import modules.config_store as cs

    settings = tmp_path_factory.mktemp("settings") / "settings.json"
    saved = (config.SETTINGS_PATH, cs.SETTINGS_PATH)
    config.SETTINGS_PATH = settings
    cs.SETTINGS_PATH = settings
    try:
        import dataset_manager

        w = dataset_manager.DatasetManager()
    finally:
        # Restore the module globals: other test modules monkeypatch these and
        # must not inherit a path into this module's temporary directory.
        config.SETTINGS_PATH, cs.SETTINGS_PATH = saved
    yield w
    w.hide()
    w.deleteLater()


@pytest.fixture(autouse=True)
def _settings_sandbox(tmp_path, monkeypatch):
    """Keep width persistence out of the developer's real settings.json.

    ``save_column_widths`` writes ``ui_column_widths`` immediately, so a test
    that drags a column would otherwise rewrite the real file.
    """
    import modules.config_store as cs

    monkeypatch.setattr(cs, "SETTINGS_PATH", tmp_path / "settings.json")


def _load(manager, samples):
    """Put ``samples`` in the dataset and refresh the table."""
    # ``refresh_table`` calls ``setRowCount(0)``, which emits a selection
    # signal; a leftover remembered row would index the PREVIOUS (longer)
    # dataset's row map. That ordering hazard is pre-existing and recorded in
    # ``.agent_notes.md``; clearing it here keeps this file out of it.
    manager._last_selected_row = -1
    manager.dataset = new_dataset(name="cols")
    manager.dataset["samples"] = samples
    manager.config["ui_column_widths"] = None
    manager.refresh_table()
    return manager.dataset["samples"]


def _headers(manager):
    return [
        manager.table.horizontalHeaderItem(i).text()
        for i in range(manager.table.columnCount())
    ]


def _col(manager, name):
    """Column index looked up by header name, never hard-coded.

    Hard-coded indices are what allowed the Actions widget and the Instr
    checkbox to occupy the same column; a test that hard-codes one would not
    have caught it either.
    """
    return _headers(manager).index(name)


def _lang_item(manager, row=0):
    """The Language cell's item, located by header name."""
    return manager.table.item(row, _col(manager, "Language"))


# ---------------------------------------------------------------------------
# The column set
# ---------------------------------------------------------------------------
def test_the_header_row_is_exactly_the_shared_column_list(manager):
    assert _headers(manager) == [name for name, _ in COLUMNS]


def test_the_removed_columns_are_gone(manager):
    headers = _headers(manager)
    assert "Instr" not in headers
    assert "Actions" not in headers
    # ...and nothing else shrank their slots: the survivors are unchanged.
    assert headers == [
        "Filename", "Tag", "Genre", "Language", "Key", "BPM", "Time", "Duration",
    ]


def test_no_column_holds_both_an_item_and_a_cell_widget(manager):
    """The bug shape: two things on one column means one paints over the other."""
    _load(manager, [new_sample(filename="a.mp3")])
    for col in range(manager.table.columnCount()):
        assert not (
            manager.table.item(0, col) is not None
            and manager.table.cellWidget(0, col) is not None
        ), f"column {col} has both an item and a widget"


def test_language_is_visible_by_default(manager):
    assert not manager.table.isColumnHidden(_col(manager, "Language"))


def test_only_time_and_duration_stay_hidden_by_default(manager):
    from ui.shell import DEFAULT_HIDDEN_COLUMNS as SHELL_HIDDEN

    assert "Language" not in SHELL_HIDDEN
    # One list, re-exported — the shell must not carry a second opinion.
    assert SHELL_HIDDEN == DEFAULT_HIDDEN_COLUMNS == ["Time", "Duration"]


def test_new_tracks_default_to_english(manager):
    assert new_sample()["language"] == "en"
    samples = _load(manager, [new_sample(filename="a.mp3")])
    item = _lang_item(manager)
    assert item.text() == "en"
    assert item.data(LANG_ROLE) == "en"
    assert not samples[0]["is_instrumental"]


# ---------------------------------------------------------------------------
# Widths: user-adjustable, with readable defaults
# ---------------------------------------------------------------------------
def test_every_column_is_user_resizable(manager):
    """Only Interactive can be dragged; Stretch/ResizeToContents cannot."""
    header = manager.table.horizontalHeader()
    modes = [header.sectionResizeMode(i) for i in range(manager.table.columnCount())]
    assert modes == [QHeaderView.Interactive] * len(modes)
    assert not header.stretchLastSection()


def test_the_default_widths_come_from_the_shared_table(manager):
    header = manager.table.horizontalHeader()
    for i, name in enumerate(_headers(manager)):
        # A hidden section reports 0 (measured); unhide it to read the real size.
        was_hidden = manager.table.isColumnHidden(i)
        if was_hidden:
            manager.table.setColumnHidden(i, False)
        assert header.sectionSize(i) == DEFAULT_WIDTHS[name], name
        if was_hidden:
            manager.table.setColumnHidden(i, True)
    # The one column a user reads must not be squeezed below its content.
    assert DEFAULT_WIDTHS["Filename"] >= 200
    assert DEFAULT_WIDTHS["Language"] <= 32


def test_saved_widths_are_applied_by_name(manager):
    """A width is remembered against the header NAME, so it cannot land on a
    different column after a column is added or reordered."""
    from ui.track_table import apply_column_widths

    manager.config["ui_column_widths"] = {"Filename": 411, "Language": 44}
    apply_column_widths(manager, manager.table)
    header = manager.table.horizontalHeader()
    assert header.sectionSize(_col(manager, "Filename")) == 411
    assert header.sectionSize(_col(manager, "Language")) == 44


def test_garbage_saved_widths_fall_back_to_the_defaults(manager):
    from ui.track_table import apply_column_widths

    manager.config["ui_column_widths"] = {"Filename": "wide", "Tag": -5}
    apply_column_widths(manager, manager.table)
    header = manager.table.horizontalHeader()
    assert header.sectionSize(_col(manager, "Filename")) == DEFAULT_WIDTHS["Filename"]
    assert header.sectionSize(_col(manager, "Tag")) == DEFAULT_WIDTHS["Tag"]


def test_widths_round_trip_through_the_config(manager):
    """Dragging a column and closing the window must survive a restart."""
    from ui.track_table import apply_column_widths, save_column_widths

    header = manager.table.horizontalHeader()
    header.resizeSection(_col(manager, "Filename"), 377)
    save_column_widths(manager)
    assert manager.config["ui_column_widths"]["Filename"] == 377

    # A later startup applies the saved width instead of the default.
    header.resizeSection(_col(manager, "Filename"), DEFAULT_WIDTHS["Filename"])
    manager.config["ui_column_widths"] = {"Filename": 377}
    apply_column_widths(manager, manager.table)
    assert header.sectionSize(_col(manager, "Filename")) == 377
    manager.config["ui_column_widths"] = None


# ---------------------------------------------------------------------------
# Language => instrumental
# ---------------------------------------------------------------------------
def test_marking_a_track_instrumental_clears_the_language(manager):
    samples = _load(manager, [new_sample(filename="a.mp3", language="en")])
    manager.table.selectRow(0)
    manager.on_table_selection_changed()
    manager.inst_check.setChecked(True)
    assert samples[0]["is_instrumental"] is True
    assert samples[0]["language"] == ""
    # And the cell is re-rendered, so it cannot keep showing "en".
    assert _lang_item(manager).data(LANG_ROLE) == LANG_INSTRUMENTAL
    # The boundary name the training loader reads.
    assert to_export_sample(samples[0])["instrumental"] is True


def test_choosing_the_instrumental_language_option_sets_the_flag(manager):
    samples = _load(manager, [new_sample(filename="a.mp3", language="en")])
    item = _lang_item(manager)
    item.setData(LANG_ROLE, LANG_INSTRUMENTAL)
    item.setText(LANG_INSTRUMENTAL)
    assert samples[0]["is_instrumental"] is True
    assert samples[0]["language"] == ""
    assert to_export_sample(samples[0])["instrumental"] is True


def test_choosing_a_real_language_clears_the_instrumental_flag(manager):
    samples = _load(
        manager, [new_sample(filename="a.mp3", is_instrumental=True, language="")]
    )
    item = _lang_item(manager)
    item.setData(LANG_ROLE, "ja")
    item.setText("ja")
    assert samples[0]["language"] == "ja"
    assert samples[0]["is_instrumental"] is False
    assert to_export_sample(samples[0])["instrumental"] is False


def test_the_language_cell_keeps_the_inspector_checkbox_in_step(manager):
    _load(manager, [new_sample(filename="a.mp3", language="en")])
    manager.table.selectRow(0)
    manager.on_table_selection_changed()
    assert manager.inst_check.isChecked() is False
    item = _lang_item(manager)
    item.setData(LANG_ROLE, LANG_INSTRUMENTAL)
    item.setText(LANG_INSTRUMENTAL)
    assert manager.inst_check.isChecked() is True
    assert manager.dataset["samples"][0]["is_instrumental"] is True


def test_the_inspector_checkbox_writes_back_to_the_language_cell(manager):
    """The other direction: the checkbox must not be a dead-end control."""
    _load(manager, [new_sample(filename="a.mp3", language="en")])
    manager.table.selectRow(0)
    manager.on_table_selection_changed()
    manager.inst_check.setChecked(True)
    assert _lang_item(manager).data(LANG_ROLE) == LANG_INSTRUMENTAL
    manager.inst_check.setChecked(False)
    sample = manager.dataset["samples"][0]
    assert sample["is_instrumental"] is False
    assert _lang_item(manager).data(LANG_ROLE) != LANG_INSTRUMENTAL


def test_every_value_lands_in_the_column_that_names_it(manager):
    """The end-to-end shape of the original bug: a value written to the wrong
    column. Address-by-name means the assertions below are the contract."""
    s = new_sample(
        filename="a.mp3",
        custom_tag="mytag",
        genre="Hard Rock",
        language="en",
        keyscale="C maj",
        bpm=128,
        timesignature="4/4",
        duration=95,
    )
    _load(manager, [s])
    got = {
        name: (manager.table.item(0, i).text() if manager.table.item(0, i) else "")
        for i, name in enumerate(_headers(manager))
    }
    assert got["Filename"] == "a.mp3"
    assert got["Tag"] == "mytag"
    assert got["Genre"] == "Hard Rock"
    assert got["Language"] == "en"
    assert got["Key"] == "C maj"
    assert got["BPM"] == "128"
    assert got["Time"] == "4/4"
    assert got["Duration"] == "95s"
    # Filename stays read-only; the rest are editable inline.
    assert not (manager.table.item(0, _col(manager, "Filename")).flags() & Qt.ItemIsEditable)
    assert manager.table.item(0, _col(manager, "Tag")).flags() & Qt.ItemIsEditable


def test_the_editable_column_map_agrees_with_the_header_list(manager):
    """``_MANUAL_FIELDS`` names columns; every one must actually exist. A name
    that no longer matches silently stops accepting edits."""
    names = {
        entry[0]
        for entry in manager._MANUAL_FIELDS
        if entry[0] != "Filename"
    }
    assert names == set(_headers(manager)) - {"Filename"}
    for i in range(manager.table.columnCount()):
        entry = manager._manual_field(i)
        if _headers(manager)[i] == "Filename":
            assert entry is None, "Filename must not be inline-editable"
        else:
            assert entry is not None


def test_editing_a_tag_cell_writes_the_sample_field(manager):
    """The write-back path resolves the field from the header name."""
    samples = _load(manager, [new_sample(filename="a.mp3")])
    item = manager.table.item(0, _col(manager, "Tag"))
    item.setText("newtag")
    assert samples[0]["custom_tag"] == "newtag"


def test_a_bad_bpm_reverts_the_cell_instead_of_writing(manager):
    samples = _load(manager, [new_sample(filename="a.mp3", bpm=120)])
    item = manager.table.item(0, _col(manager, "BPM"))
    item.setText("not-a-number")
    assert samples[0]["bpm"] == 120
    assert item.text() == ""


def test_the_exported_sample_always_carries_an_explicit_boolean(manager):
    """``instrumental`` is the only carrier of "no vocals" now that ``language``
    may legitimately be blank, so it must never be absent from the export."""
    assert to_export_sample(new_sample(filename="a.mp3"))["instrumental"] is False
    assert to_export_sample(
        new_sample(filename="b.mp3", is_instrumental=True)
    )["instrumental"] is True


def test_english_is_not_treated_as_instrumental(manager):
    """The pinned convention is blank == instrumental; "en" is a language."""
    _load(manager, [new_sample(filename="a.mp3", language="en")])
    assert _lang_item(manager).data(LANG_ROLE) == "en"
    assert manager.dataset["samples"][0]["is_instrumental"] is False
