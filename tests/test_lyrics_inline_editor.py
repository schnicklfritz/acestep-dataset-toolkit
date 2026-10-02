"""The main editor's lyrics field: debounced auto-save, text size, expand.

WHY THIS EXISTS
---------------
The field has been through both failure modes, and this file pins the contract
that avoids both:

1. It once wrote to the dataset on EVERY keystroke with no undo snapshot::

       def on_lyrics_edited(self):
           s = self.get_selected_sample()
           if s:
               s["formatted_lyrics"] = self.lyrics_text.toPlainText()

   Clicking into the box and typing one character silently rewrote a track's
   lyrics and nothing could put it back.

2. "✔ Commit Lyrics" was added to stop that, which traded the corruption for a
   different annoyance: the field was read-only until "✏ Edit mode" was ticked,
   and text left uncommitted was silently discarded on the next selection.

The button is gone. The field is always editable and every change is written
once, 400 ms after typing stops. What replaces the button is not a smaller
button but four guarantees, one test group each below:

  * the write is DEBOUNCED — a burst of keystrokes is one save, not one per key;
  * every save is UNDOABLE, and the snapshot holds the OLD text;
  * each song is BACKED UP once, before the first change;
  * a write is REFUSED when the field no longer belongs to the selected track,
    which is the only way an auto-save could reach the wrong track.

Also pinned here: the lyrics text size steppers (A−/A+) and the numeric "Size:"
readout drive ONE setting (``lyrics_font_size``), which the Settings row shares;
and "Expand" hides what is below the field, without ever hiding the field.
"""
import pytest

pytest.importorskip("PySide6")

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from modules.dataset_schema import new_dataset, new_sample  # noqa: E402

# The debounce interval the manager arms. Kept as a name here so a test that
# waits for the save is not silently waiting too little if the interval grows.
SAVE_MS = 400


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One window for the whole module.

    Building a ``DatasetManager`` applies the theme to every page, which costs
    tens of seconds here, so a per-test window would make this file the slowest
    in the suite for no extra coverage. ``_load`` resets the fields each test
    needs, so shared state cannot leak between cases.
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
    """Keep appearance changes out of the developer's real settings.json.

    The module ``manager`` fixture restores ``SETTINGS_PATH`` as soon as the
    window is built (so other modules cannot inherit its temp dir), but the
    size controls save on every change — without this, running the suite would
    rewrite the appearance section of the real file.
    """
    import modules.config_store as cs

    monkeypatch.setattr(cs, "SETTINGS_PATH", tmp_path / "settings.json")


def _load(manager, samples, row=0):
    """Put ``samples`` in the dataset, select ``row``, and reset the switches."""
    # Reset first: these are the state a previous test may have flipped.
    manager.lyrics_expand_check.setChecked(False)
    # A save left armed by a previous test would fire against THIS dataset.
    manager._lyrics_save_timer.stop()
    manager._songs_backed_up.clear()
    # Point the per-song backups at a temp dir: without this the suite would
    # write into the developer's real _Backup/songs (no dataset is open in a
    # test, so ``_songs_backup_dir`` falls back to ``project_backups/songs``
    # under the CWD).
    tmp = tempfile.mkdtemp(prefix="lyrics-backups-")
    manager._songs_backup_dir = lambda: Path(tmp)
    # The size preference lives in module-scoped config; a test that leaves 26px
    # behind would change what the next one measures. Guarded so the common case
    # does not pay for a theme re-apply.
    if manager.config.get("lyrics_font_size"):
        manager.lyrics_font_spin_inline.setValue(0)
    manager.undo_stack.clear()
    manager.redo_stack.clear()
    # MUST be cleared BEFORE refresh_table. ``refresh_table`` calls
    # ``setRowCount(0)``, which clears the table's current row and emits a
    # selection signal; ``get_selected_sample`` then falls back to this
    # remembered row, and a leftover row would index the PREVIOUS (longer)
    # dataset's row map. That ordering hazard is pre-existing and recorded in
    # ``.agent_notes.md``; clearing the row here keeps this file out of it.
    manager._last_selected_row = -1

    manager.dataset = new_dataset(name="lyrics")
    manager.dataset["samples"] = samples
    manager.refresh_table()
    manager.table.selectRow(row)
    manager.on_table_selection_changed()
    return manager.dataset["samples"][row]


def _settle(qapp, manager, ms=SAVE_MS + 100):
    """Let the debounced save fire, then run it.

    ``QTimer`` needs the event loop to turn; under the offscreen platform in a
    test that means pumping events until the interval has passed. Looping keeps
    this robust on a loaded machine, where one long sleep may not be enough.
    """
    from PySide6.QtCore import QElapsedTimer

    clock = QElapsedTimer()
    clock.start()
    while clock.elapsed() < ms:
        qapp.processEvents()
    qapp.processEvents()
    return manager


def _track(filename, lyrics=""):
    s = new_sample(filename=filename)
    s["formatted_lyrics"] = lyrics
    s["lyrics"] = lyrics
    return s


# ---------------------------------------------------------------------------
# The field is live: no lock, no mode switch, no Commit button
# ---------------------------------------------------------------------------

def test_the_commit_button_and_its_gate_are_gone(manager):
    """The tools this file used to pin must not come back by accident.

    A leftover ``lyrics_commit_btn`` or ``lyrics_edit_check`` would mean a second
    write path still exists beside the auto-save, which is how the field would
    silently drift back to "text on screen that was never saved".
    """
    assert not hasattr(manager, "lyrics_commit_btn")
    assert not hasattr(manager, "lyrics_revert_btn")
    assert not hasattr(manager, "lyrics_edit_check")
    assert not hasattr(manager, "_lyrics_edit_active")
    assert not hasattr(manager, "on_lyrics_commit")
    assert not hasattr(manager, "on_lyrics_edit_mode_toggled")


def test_the_field_is_editable_with_no_mode_switch(manager):
    """A read-only field is the thing the button existed to unlock; with the
    button gone the field has to be writable the moment it is shown."""
    _load(manager, [_track("a.mp3", "one")])
    assert manager.lyrics_text.isReadOnly() is False


def test_the_field_shows_the_selected_track(manager):
    _load(manager, [_track("a.mp3", "[Verse]\nONE two")])
    assert manager.lyrics_text.toPlainText() == "[Verse]\nONE two"
    # Loading shows what is saved; it is not an unsaved edit.
    assert manager.lyrics_save_status.text() == "Saved"


def test_loading_a_track_does_not_arm_a_save(manager):
    """Populating the field must not look like typing.

    If it did, every selection change would snapshot the undo stack and back the
    song up 400 ms later — the field would be forever dirty for no edit.
    """
    _load(manager, [_track("a.mp3", "one")])
    assert manager._lyrics_save_timer.isActive() is False
    assert not manager.undo_stack


def test_switching_tracks_reloads_the_field(manager):
    _load(manager, [_track("a.mp3", "first"), _track("b.mp3", "second")])
    assert manager.lyrics_text.toPlainText() == "first"
    manager.table.selectRow(1)
    manager.on_table_selection_changed()
    assert manager.lyrics_text.toPlainText() == "second"


# ---------------------------------------------------------------------------
# The save is debounced
# ---------------------------------------------------------------------------

def test_typing_marks_the_field_unsaved_without_writing(manager):
    sample = _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_text.setPlainText("two")
    # The dataset is not written on the keystroke...
    assert sample["formatted_lyrics"] == "one"
    assert not manager.undo_stack
    # ...but the field says so, and a save is now armed.
    assert manager._lyrics_save_timer.isActive() is True
    assert "Unsaved" in manager.lyrics_save_status.text()


def test_a_burst_of_keystrokes_is_one_save(qapp, manager):
    """The whole point of the debounce: ten characters, one snapshot, one write.

    Without it each character would push an undo snapshot and rewrite the song,
    which is the per-keystroke corruption this field started with.
    """
    sample = _load(manager, [_track("a.mp3", "")])
    for i in range(1, 11):
        manager.lyrics_text.setPlainText("x" * i)
    _settle(qapp, manager)
    assert sample["formatted_lyrics"] == "x" * 10
    assert len(manager.undo_stack) == 1


def test_the_save_lands_after_the_debounce(qapp, manager):
    sample = _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_text.setPlainText("two")
    _settle(qapp, manager)
    assert sample["formatted_lyrics"] == "two"
    assert sample["lyrics"] == "two"
    assert manager.lyrics_save_status.text().startswith("Saved")


def test_an_auto_save_is_undoable_and_holds_the_old_text(qapp, manager):
    """Undo must restore what was there BEFORE the save.

    The snapshot is taken inside the flush, before the write. If it were taken
    after (or if the writer ran before it), Undo would "restore" the text the
    user just typed and the button would look broken.
    """
    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_text.setPlainText("two")
    _settle(qapp, manager)
    assert len(manager.undo_stack) == 1
    snapshot = json.loads(manager.undo_stack[-1])
    assert snapshot["samples"][0]["formatted_lyrics"] == "one"

    manager.undo()
    assert manager.dataset["samples"][0]["formatted_lyrics"] == "one"


def test_a_save_preserves_the_original_as_raw_lyrics(qapp, manager):
    """``raw_lyrics`` is what the tidy pass starts from, so the auto-save must
    put the PRE-edit text there — not the text it is saving."""
    sample = _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_text.setPlainText("two")
    _settle(qapp, manager)
    assert sample["raw_lyrics"] == "one"


def test_an_unchanged_field_does_not_create_a_snapshot(qapp, manager):
    """Selecting a track and "editing" it to the same text is not an edit."""
    _load(manager, [_track("a.mp3", "one")])
    manager.on_lyrics_edited()
    _settle(qapp, manager)
    assert not manager.undo_stack
    assert manager.lyrics_save_status.text() == "Saved"


# ---------------------------------------------------------------------------
# Each song is backed up once, before the first change
# ---------------------------------------------------------------------------

def test_the_first_save_backs_the_song_up(qapp, manager):
    """The backup is taken BEFORE the write, so it holds the pre-edit text."""
    backup_dir = Path(tempfile.mkdtemp(prefix="lyrics-backup-check-"))
    _load(manager, [_track("a.mp3", "original")])
    manager._songs_backup_dir = lambda: backup_dir
    manager.lyrics_text.setPlainText("edited")
    _settle(qapp, manager)

    saved = backup_dir / "a.mp3.json"
    assert saved.exists(), "the first change to a song must back it up"
    assert json.loads(saved.read_text())["formatted_lyrics"] == "original"


def test_a_song_is_backed_up_only_once(qapp, manager):
    """A second save must not overwrite the backup with the first edit.

    Re-backing-up on every save would leave you holding the state you just
    replaced, which is the same as having no backup.
    """
    backup_dir = Path(tempfile.mkdtemp(prefix="lyrics-backup-once-"))
    _load(manager, [_track("a.mp3", "original")])
    manager._songs_backup_dir = lambda: backup_dir
    manager.lyrics_text.setPlainText("first edit")
    _settle(qapp, manager)
    manager.lyrics_text.setPlainText("second edit")
    _settle(qapp, manager)

    saved = json.loads((backup_dir / "a.mp3.json").read_text())
    assert saved["formatted_lyrics"] == "original"


# ---------------------------------------------------------------------------
# A stale field never writes onto another track
# ---------------------------------------------------------------------------

def test_revert_reloads_the_saved_lyrics(qapp, manager):
    """The "↺ Revert to saved" button refreshes the field. It is not an undo —
    the auto-save has already written, so undoing is Ctrl+Z's job."""
    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_text.setPlainText("two")
    _settle(qapp, manager)
    assert manager.dataset["samples"][0]["formatted_lyrics"] == "two"

    manager.lyrics_discard_btn.click()
    assert manager.lyrics_text.toPlainText() == "two"


def test_the_save_refuses_when_the_field_shows_another_track(qapp, manager):
    """Defence in depth: the flush compares the field's track against the
    selection, so a stale buffer can never be written onto a different track.

    The normal selection path reloads the field (see
    ``test_switching_tracks_reloads_the_field``), so the stale state is built
    here by moving the selection with the table's signals blocked — that is the
    "selection moved but the field was not repopulated" case the guard exists
    for.
    """
    samples = [_track("a.mp3", "first"), _track("b.mp3", "second")]
    _load(manager, samples)
    manager.lyrics_text.setPlainText("edited for a.mp3")

    # Select row 1 without letting on_table_selection_changed repopulate.
    manager.table.blockSignals(True)
    manager.table.selectRow(1)
    manager.table.blockSignals(False)
    assert manager._lyrics_selected_filename == "a.mp3"
    assert manager.table.currentRow() == 1

    _settle(qapp, manager)
    # Neither track was touched, and the field says why.
    assert samples[0]["formatted_lyrics"] == "first"
    assert samples[1]["formatted_lyrics"] == "second"
    assert "Not saved" in manager.lyrics_save_status.text()


def test_switching_track_mid_edit_stops_the_pending_save(qapp, manager):
    """The plain case of the guard: type, then click another track.

    ``on_table_selection_changed`` refills the field and cancels the timer, so
    the pending text never lands on the track the user just left.
    """
    samples = [_track("a.mp3", "first"), _track("b.mp3", "second")]
    _load(manager, samples)
    manager.lyrics_text.setPlainText("typed for a")
    manager.table.selectRow(1)
    manager.on_table_selection_changed()
    _settle(qapp, manager)
    assert samples[0]["formatted_lyrics"] == "first"
    assert samples[1]["formatted_lyrics"] == "second"
    assert manager.lyrics_text.toPlainText() == "second"


# ---------------------------------------------------------------------------
# Text-size steppers (the old "Capitalize mode" checkbox used to live here)
# ---------------------------------------------------------------------------

def test_the_double_click_gesture_needs_no_mode_switch(manager):
    """The inline field is the same widget the popup uses and has no switch.

    The "🔠 Capitalize mode" checkbox that used to sit in the header gated the
    gestures on this field. Double-click is now always on, so the checkbox must
    be gone — a lingering ``lyrics_caps_check`` is how "double-click does
    nothing here" could silently return.
    """
    _load(manager, [_track("a.mp3", "one")])
    assert not hasattr(manager, "lyrics_caps_check")
    assert not hasattr(manager.lyrics_text, "set_gestures_enabled")


def test_the_field_is_a_lyric_text_edit(manager):
    """The inline host must be the SAME widget the popup uses, or the gestures
    would have to be maintained in two places."""
    from ui.lyrics_studio import LyricTextEdit

    _load(manager, [_track("a.mp3", "one")])
    assert isinstance(manager.lyrics_text, LyricTextEdit)


def test_plus_raises_the_lyrics_size_and_minus_lowers_it(manager):
    """A+/A− step one px, and they are the SAME setting the spinner owns.

    The buttons replaced the capitalize checkbox 1:1, so this is the control
    that has to work from the header. Each press goes through the shared config
    key, so the numeric "Size:" readout beside them moves with it.
    """
    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_font_spin_inline.setValue(20)
    manager.lyrics_text_larger_btn.click()
    assert manager.config["lyrics_font_size"] == 21
    assert manager.lyrics_font_spin_inline.value() == 21
    manager.lyrics_text_smaller_btn.click()
    assert manager.config["lyrics_font_size"] == 20


def test_plus_starts_from_the_inherited_size_when_auto(manager):
    """Auto means "the app's size", not 0 px, so the first A+ must be 14 (13+1).

    Stepping from the raw stored value (0) would target 1 px, get clamped, and
    feel like the button did nothing.
    """
    _load(manager, [_track("a.mp3", "one")])
    manager.config["lyrics_font_size"] = 0
    manager.config["ui_font_size"] = 0
    manager.lyrics_text_larger_btn.click()
    assert manager.config["lyrics_font_size"] == 14      # base 13 + 1


def test_the_steppers_reach_the_lyrics_ceiling_not_the_app_one(manager):
    """A+ must run all the way to ``LYRICS_FONT_SIZE_MAX`` (72), past the app's
    32 px cap — that is the whole reason the lyrics cap exists."""
    from ui.themes import FONT_SIZE_MAX, LYRICS_FONT_SIZE_MAX

    _load(manager, [_track("a.mp3", "one")])
    assert LYRICS_FONT_SIZE_MAX > FONT_SIZE_MAX
    manager.lyrics_font_spin_inline.setValue(LYRICS_FONT_SIZE_MAX)
    manager.lyrics_text_larger_btn.click()      # already at the ceiling
    assert manager.config["lyrics_font_size"] == LYRICS_FONT_SIZE_MAX
    assert manager.lyrics_font_spin_inline.value() == LYRICS_FONT_SIZE_MAX


# ---------------------------------------------------------------------------
# Expand
# ---------------------------------------------------------------------------

def _visible_in_inspector(manager, widget):
    """True if ``widget`` would be visible were the window shown.

    ``isVisibleTo`` ignores whether an ANCESTOR is on screen, which is what this
    test needs — the fixture never shows the window.
    """
    return widget.isVisibleTo(manager)


def test_expand_hides_what_is_below_the_field(manager):
    _load(manager, [_track("a.mp3", "one")])
    assert _visible_in_inspector(manager, manager.caption_text)
    assert _visible_in_inspector(manager, manager.track_tag_input)

    manager.lyrics_expand_check.setChecked(True)
    assert not _visible_in_inspector(manager, manager.caption_text)
    assert not _visible_in_inspector(manager, manager.track_tag_input)
    # The field itself is still there — that is the whole point.
    assert _visible_in_inspector(manager, manager.lyrics_text)


def test_expand_grows_the_field(manager):
    _load(manager, [_track("a.mp3", "one")])
    normal = manager.lyrics_text.minimumHeight()
    manager.lyrics_expand_check.setChecked(True)
    assert manager.lyrics_text.minimumHeight() > normal


def test_collapsing_expand_brings_the_fields_back(manager):
    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_expand_check.setChecked(True)
    manager.lyrics_expand_check.setChecked(False)
    assert _visible_in_inspector(manager, manager.caption_text)
    assert _visible_in_inspector(manager, manager.track_tag_input)


# ---------------------------------------------------------------------------
# Text size: the field's own control and the global one are one setting
# ---------------------------------------------------------------------------

def test_the_inline_size_control_defaults_to_auto(manager):
    """Auto = follow the app. A default of any other number would silently
    resize everyone's lyrics on upgrade."""
    assert manager.lyrics_font_spin_inline.value() == 0
    assert manager.lyrics_font_spin_inline.text() == "Auto"


def test_the_inline_control_carries_the_size_in_the_tooltip(manager):
    """The range and the meaning of Auto come from the shared builder, so a
    tooltip is the only part the two copies own separately."""
    assert "lyrics" in manager.lyrics_font_spin_inline.toolTip().lower()


def test_marking_the_inline_field_as_a_lyrics_region(manager):
    """The size reaches the field through this property, not through setFont —
    the app stylesheet overrides a widget font on every theme re-apply."""
    from ui.themes import FONT_REGION_LYRICS, FONT_REGION_PROPERTY

    assert manager.lyrics_text.property(FONT_REGION_PROPERTY) == FONT_REGION_LYRICS


def test_the_inline_control_writes_the_one_shared_config_key(manager):
    """There is one setting, not two: the field's spinner and the Settings row
    both drive ``lyrics_font_size``."""
    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_font_spin_inline.setValue(24)
    assert manager.config["lyrics_font_size"] == 24


def test_the_two_copies_of_the_size_control_do_not_disagree(manager):
    """A stale second copy is worse than no second copy: it would show a size
    the app is not using."""
    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_font_spin.setValue(21)
    assert manager.lyrics_font_spin_inline.value() == 21
    # ...and the reverse direction, which is the one the field's header owns.
    manager.lyrics_font_spin_inline.setValue(0)
    assert manager.lyrics_font_spin.value() == 0


def test_a_lyrics_size_change_persists(manager, tmp_path):
    """It is a preference, so it has to survive a restart — ``save_plain_keys``
    is the only path to settings.json (sandboxed by ``_settings_sandbox``)."""
    import json

    _load(manager, [_track("a.mp3", "one")])
    manager.lyrics_font_spin_inline.setValue(26)
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["lyrics_font_size"] == 26


def test_auto_keeps_the_lyrics_in_step_with_the_global_size(manager):
    """The setting is only useful if Auto really means "the same as the app"."""
    from ui.themes import resolve_font_sizes

    manager.config["ui_font_size"] = 18
    manager.config["lyrics_font_size"] = 0
    assert resolve_font_sizes(manager.config) == (18, None)


# ---------------------------------------------------------------------------
# Tools-dock pages addressed by name, not by hardcoded index
# ---------------------------------------------------------------------------

def test_tool_pages_are_looked_up_by_name(manager):
    """``_on_tab_changed`` used to compare against literal indices (1 and 3),
    which broke silently the moment a page was inserted or reordered."""
    assert manager._tool_page("Lyrics") >= 0
    assert manager._tool_page("Tags & checks") >= 0
    # The two names must not collide, or the lyric hook would fire on the wrong
    # page (this is the bug the literals hid).
    assert manager._tool_page("Lyrics") != manager._tool_page("Tags & checks")


def test_the_lookup_matches_the_actual_tab_order(manager):
    labels = [
        manager.tabs.tabText(i).replace("&&", "&")
        for i in range(manager.tabs.count())
    ]
    for name in ("Lyrics", "Tags & checks", "Caption", "Structure"):
        assert labels[manager._tool_page(name)] == name


def test_an_unknown_page_looks_up_to_minus_one(manager):
    """``-1`` can never match a real tab index, so a renamed page degrades to
    "no hook fires" instead of silently firing on the wrong tab."""
    assert manager._tool_page("Nope") == -1


def test_the_lyrics_page_hook_runs_the_shared_lyrics_loader(manager, monkeypatch):
    """Arriving on the Lyrics page must populate it — the behaviour the old
    hardcoded index was there for."""
    calls = []
    monkeypatch.setattr(manager, "preview_lyrics_tidy",
                        lambda silent=False: calls.append("preview"))
    monkeypatch.setattr(manager, "load_all_lyrics",
                        lambda silent=False: calls.append("block"))
    manager._on_tab_changed(manager._tool_page("Lyrics"))
    assert calls == ["preview", "block"]


def test_the_tag_page_hook_refreshes_the_tag_manager(manager, monkeypatch):
    calls = []
    monkeypatch.setattr(manager, "refresh_tag_manager", lambda: calls.append("tags"))
    manager._on_tab_changed(manager._tool_page("Tags & checks"))
    assert calls == ["tags"]


def test_the_dead_embed_index_is_gone(manager):
    """``embed_tab_index`` was written and never read; a stale name like that is
    how the next page insert gets pointed at the wrong tab."""
    assert not hasattr(manager, "embed_tab_index")
    assert not hasattr(manager, "lyrics_tab_index")
    assert not hasattr(manager, "tag_tab_index")
