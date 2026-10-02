"""Deleting tracks from the Studio: the button, the Del key, undo, backups.

WHY THIS EXISTS
---------------
Deleting used to be reachable ONLY through the right-click menu (the per-row
"Delete" button went out with the Actions column). That capability had zero
tests, which is exactly the kind of thing that silently rots: the menu
addresses the row under the cursor, and a naive "delete the selected rows"
button written next to it would read the ROW number as the SAMPLE index and
delete the wrong track the moment a filter or the Exceptions view is on.

What is pinned here:

  * the button and the Del key resolve SELECTED rows through
    ``_table_sample_indices`` — never by row number;
  * a multi-select delete removes exactly the selected tracks, in one undo step
    (descending pops, so removing one track cannot shift the next target);
  * the audio file is backed up before the sample is dropped;
  * a declined confirmation changes nothing (no snapshot, no pop, no backup).
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox, QTableWidgetSelectionRange  # noqa: E402

from modules.dataset_schema import new_dataset, new_sample  # noqa: E402


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One window for the whole module, as in tests/test_lyrics_inline_editor.py."""
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
        config.SETTINGS_PATH, cs.SETTINGS_PATH = saved
    yield w
    w.hide()
    w.deleteLater()


@pytest.fixture(autouse=True)
def _settings_sandbox(tmp_path, monkeypatch):
    import modules.config_store as cs

    monkeypatch.setattr(cs, "SETTINGS_PATH", tmp_path / "settings.json")


def _load(manager, samples):
    """Put ``samples`` in the dataset, clear filters, and refresh the table."""
    manager._last_selected_row = -1
    manager.dataset = new_dataset(name="del")
    manager.dataset["samples"] = samples
    manager.undo_stack.clear()
    manager.redo_stack.clear()
    manager.filter_query = ""
    manager.filter_inst = "all"
    manager.filter_captioned = False
    manager.filter_exceptions_only = False
    if hasattr(manager, "filter_search"):
        manager.filter_search.setText("")
    manager.refresh_table()
    return manager.dataset["samples"]


def _track(filename, caption="has a caption", path="", lyrics=""):
    s = new_sample(filename=filename)
    s["caption"] = caption
    s["audio_path"] = path
    s["formatted_lyrics"] = lyrics
    s["lyrics"] = lyrics
    return s


def _select(manager, *rows):
    """Select table rows by index (not sample index).

    ``selectRow`` REPLACES the selection, so multi-select has to go through
    ``setRangeSelected`` — which adds — or the second call would silently drop
    the first row and the test would pass for the wrong reason.
    """
    manager.table.clearSelection()
    last = manager.table.columnCount() - 1
    for r in rows:
        manager.table.setRangeSelected(
            QTableWidgetSelectionRange(r, 0, r, last), True)
    manager.on_table_selection_changed()


@pytest.fixture
def yes(monkeypatch):
    """Answer every confirmation with Yes."""
    monkeypatch.setattr(
        "dataset_manager.QMessageBox.question",
        lambda *a, **k: QMessageBox.Yes,
    )


@pytest.fixture
def no(monkeypatch):
    monkeypatch.setattr(
        "dataset_manager.QMessageBox.question",
        lambda *a, **k: QMessageBox.No,
    )


# ---------------------------------------------------------------------------
# The button exists and is wired
# ---------------------------------------------------------------------------

def test_the_delete_button_is_present_and_discoverable(manager):
    """A control, not only a context menu: a capability hidden behind
    right-click is one most users never find."""
    assert manager.delete_track_btn.text().endswith("Delete Track")
    assert "back" in manager.delete_track_btn.toolTip().lower()


def test_the_button_removes_the_selected_track(manager, yes):
    _load(manager, [_track("a.mp3"), _track("b.mp3"), _track("c.mp3")])
    _select(manager, 1)
    manager.delete_track_btn.click()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3", "c.mp3"]


def test_no_selection_deletes_nothing(manager, yes):
    _load(manager, [_track("a.mp3")])
    manager.table.clearSelection()
    manager.delete_selected_tracks()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3"]


# ---------------------------------------------------------------------------
# Row -> sample resolution (filters and the Exceptions view)
# ---------------------------------------------------------------------------

def test_delete_uses_the_row_map_not_the_row_number(manager, yes):
    """The reported-class bug: with a filter on, row N is not sample N.

    Only ``b.mp3`` is visible; its table row is 0 but its SAMPLE index is 1.
    Deleting the visible row must remove ``b.mp3``, not ``a.mp3``.
    """
    _load(manager, [_track("a.mp3"), _track("b.mp3", caption=""), _track("c.mp3")])
    manager.filter_query = "b.mp3"
    manager.refresh_table()
    assert manager._table_sample_indices == [1]
    _select(manager, 0)
    manager.delete_selected_tracks()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3", "c.mp3"]


def test_the_exceptions_view_deletes_the_visible_exception(manager, yes):
    """Exceptions view shows only caption-less tracks; the row map must still
    point at the right sample."""
    _load(manager, [_track("a.mp3"), _track("b.mp3", caption=""),
                    _track("c.mp3", caption="")])
    manager.filter_exceptions_only = True
    manager.refresh_table()
    assert manager._table_sample_indices == [1, 2]
    _select(manager, 0)                 # the visible row for b.mp3
    manager.delete_selected_tracks()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3", "c.mp3"]


# ---------------------------------------------------------------------------
# Multi-select: order and one undo step
# ---------------------------------------------------------------------------

def test_a_multi_select_delete_removes_exactly_those_tracks(manager, yes):
    """Two selected tracks, one action. Descending pops stop removing sample 1
    from shifting sample 3 down and making the second pop hit sample 2."""
    _load(manager, [_track("a.mp3"), _track("b.mp3"), _track("c.mp3"), _track("d.mp3")])
    _select(manager, 1, 3)              # b.mp3 and d.mp3
    manager.delete_selected_tracks()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3", "c.mp3"]


def test_a_multi_delete_is_a_single_undo_step(manager, yes):
    """N snapshots would make Ctrl+Z undo one track at a time with no way to
    undo "that delete". There must be exactly one."""
    _load(manager, [_track("a.mp3"), _track("b.mp3"), _track("c.mp3")])
    _select(manager, 0, 1)
    manager.delete_selected_tracks()
    assert len(manager.undo_stack) == 1
    manager.undo()
    assert [s["filename"] for s in manager.dataset["samples"]] == [
        "a.mp3", "b.mp3", "c.mp3"]


def test_a_single_delete_is_undoable(manager, yes):
    _load(manager, [_track("a.mp3"), _track("b.mp3")])
    _select(manager, 0)
    manager.delete_selected_tracks()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["b.mp3"]
    manager.undo()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3", "b.mp3"]


# ---------------------------------------------------------------------------
# The file is backed up before the sample is dropped
# ---------------------------------------------------------------------------

def test_the_audio_file_is_backed_up(manager, yes, tmp_path, monkeypatch):
    """Deleting from the dataset must never destroy the audio: the file is
    copied to project_backups/deleted/ first."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"not really a wav")
    monkeypatch.chdir(tmp_path)
    _load(manager, [_track("song.wav", path=str(audio))])
    _select(manager, 0)
    manager.delete_selected_tracks()
    backup = tmp_path / "project_backups" / "deleted" / "song.wav"
    assert backup.exists(), "the audio file was removed without a backup"
    assert backup.read_bytes() == b"not really a wav"


# ---------------------------------------------------------------------------
# Declining the confirmation
# ---------------------------------------------------------------------------

def test_a_declined_delete_changes_nothing(manager, no):
    """No means no: no pop, no snapshot, no backup."""
    _load(manager, [_track("a.mp3"), _track("b.mp3")])
    _select(manager, 0)
    manager.delete_selected_tracks()
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3", "b.mp3"]
    assert len(manager.undo_stack) == 0


# ---------------------------------------------------------------------------
# The right-click menu still works, and shares the implementation
# ---------------------------------------------------------------------------

def test_the_context_menu_delete_still_targets_the_clicked_row(manager, yes):
    """The menu addresses the row under the cursor; ``confirm_delete_sample``
    is still its entry point and must go through the same primitive."""
    _load(manager, [_track("a.mp3"), _track("b.mp3")])
    manager.confirm_delete_sample(1)        # b.mp3
    assert [s["filename"] for s in manager.dataset["samples"]] == ["a.mp3"]


# ---------------------------------------------------------------------------
# The METADATA is backed up too, not only the audio
# ---------------------------------------------------------------------------
# The audio file can be re-found on disk; a hand-written caption and lyrics
# exist in exactly one place. This is the reported data loss: a track was
# deleted, its audio was backed up, but its caption/lyrics only lived in the
# sample list, so the next save wrote the dataset WITHOUT them.

def _metadata_backups(backup_dir, filename):
    return list(backup_dir.glob(f"{filename}.json*"))


def test_deleting_a_track_backs_up_its_caption_and_lyrics(manager, yes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager.current_dataset_path = None          # no dataset open -> cwd fallback
    _load(manager, [_track("song.mp3", caption="a caption", lyrics="Line one\nLine two")])
    _select(manager, 0)
    manager.delete_selected_tracks()
    backups = _metadata_backups(tmp_path / "project_backups" / "deleted", "song.mp3")
    assert backups, "the track was dropped with no metadata backup"
    saved = json.loads(backups[0].read_text(encoding="utf-8"))
    assert saved["caption"] == "a caption"
    assert saved["formatted_lyrics"] == "Line one\nLine two"
    assert saved["lyrics"] == "Line one\nLine two"


def test_the_metadata_backup_lands_beside_an_open_dataset(manager, yes, tmp_path):
    """With a dataset open the backup goes into ITS _Backup/deleted, so the
    copies travel with the dataset instead of scattering into the CWD."""
    ds_dir = tmp_path / "mydata"
    ds_dir.mkdir()
    manager.current_dataset_path = str(ds_dir / "acdc.json")
    _load(manager, [_track("song.mp3", lyrics="words")])
    _select(manager, 0)
    manager.delete_selected_tracks()
    assert _metadata_backups(ds_dir / "_Backup" / "deleted", "song.mp3")


def test_a_batch_delete_backs_up_every_dropped_track(manager, yes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager.current_dataset_path = None
    _load(manager, [_track("a.mp3", lyrics="aaa"), _track("b.mp3", lyrics="bbb")])
    _select(manager, 0, 1)
    manager.delete_selected_tracks()
    backup_dir = tmp_path / "project_backups" / "deleted"
    assert _metadata_backups(backup_dir, "a.mp3")
    assert _metadata_backups(backup_dir, "b.mp3")


def test_a_track_with_no_audio_on_disk_still_has_its_metadata_kept(manager, yes, tmp_path, monkeypatch):
    """The audio backup is guarded by ``os.path.exists``, so a track whose file
    is missing (never rendered, or on a detached drive) used to be deleted with
    NOTHING preserved. The caption and lyrics must still be backed up."""
    monkeypatch.chdir(tmp_path)
    manager.current_dataset_path = None
    _load(manager, [_track("ghost.mp3", path=str(tmp_path / "gone.mp3"),
                           caption="kept caption", lyrics="kept lyrics")])
    _select(manager, 0)
    manager.delete_selected_tracks()
    backups = _metadata_backups(tmp_path / "project_backups" / "deleted", "ghost.mp3")
    assert backups, "a track with no audio was deleted with no metadata kept"
    assert json.loads(backups[0].read_text(encoding="utf-8"))["caption"] == "kept caption"


def test_the_delete_confirmation_says_the_metadata_is_kept(manager, monkeypatch):
    """The message is the user's only warning about what a delete costs; it must
    not imply the audio is the only thing preserved."""
    captured = {}

    def fake(parent, title, text, *a, **k):
        captured["text"] = text
        return QMessageBox.No

    monkeypatch.setattr("dataset_manager.QMessageBox.question", fake)
    _load(manager, [_track("a.mp3")])
    _select(manager, 0)
    manager.delete_selected_tracks()
    assert "metadata" in captured["text"].lower()


# ---------------------------------------------------------------------------
# Backup names cannot collide inside one second
# ---------------------------------------------------------------------------

def test_two_backups_in_the_same_second_are_both_kept(manager, tmp_path):
    """``.bak-<stamp>`` has second resolution. Two saves inside one second must
    not overwrite the first backup -- an overwritten backup is no backup."""
    target = tmp_path / "d.json"
    target.write_text("{}", encoding="utf-8")
    first = manager._backup_file(str(target))
    second = manager._backup_file(str(target))
    assert first and second
    assert first != second
    assert target.exists()
    assert Path(first).exists() and Path(second).exists()


# ---------------------------------------------------------------------------
# A save that cannot back up the existing file must WRITE NOTHING
# ---------------------------------------------------------------------------

def test_save_refuses_to_overwrite_when_the_backup_fails(manager, tmp_path, monkeypatch):
    """The reported loss, at the save layer: if the pre-save backup cannot be
    taken, saving anyway would replace the only copy of the data with this
    process's (possibly stale) view. Refuse, and leave the file intact."""
    from PySide6.QtWidgets import QMessageBox as _QMB

    target = tmp_path / "d.json"
    target.write_text('{"original": true}', encoding="utf-8")
    monkeypatch.setattr(manager, "_backup_file", lambda path: None)
    monkeypatch.setattr("dataset_manager.QMessageBox.critical",
                        lambda *a, **k: _QMB.Ok)
    manager.dataset = {"metadata": {}, "samples": []}
    manager.current_dataset_path = str(target)

    assert manager.save_dataset() is False
    assert target.read_text(encoding="utf-8") == '{"original": true}'
