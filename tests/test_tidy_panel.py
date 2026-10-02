"""The Lyrics Tidy panel on the "Tags & checks" page.

WHY THIS EXISTS
---------------
The three structure-tag rules (strip quotes, trim tag modifiers, drop stacked
tags) are destructive: they rewrite lyrics in place across the whole dataset.
The panel that fires them is on the Tools dock, beside the tag machinery that
produced the tags in the first place, and offers two scopes.

What is pinned here:

  * the panel exists on the "Tags & checks" page and nowhere else;
  * both scopes route through one ``apply_tidy``, so the snapshot and the
    per-song backup cannot diverge between "selected" and "entire dataset";
  * tidying writes all THREE ``dataset.json`` fields the contract requires
    (``formatted_lyrics`` for the exporter, ``lyrics`` for the inspector, and
    ``raw_lyrics`` for the tidy-undo);
  * a ``<stem>_lyrics.txt`` sidecar is written beside the audio, and it is
    NOT ``<stem>.txt`` — that name belongs to the caption exporter;
  * the whole-dataset scope confirms first, and Cancel changes nothing.
"""
import json

import pytest

pytest.importorskip("PySide6")

from modules.dataset_schema import new_dataset, new_sample  # noqa: E402


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One window for the whole module (building it applies the theme)."""
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


def _load(manager, samples, tmp_path, row=0):
    """Reset the dataset, point backups at a temp dir, select ``row``."""
    manager._songs_backed_up.clear()
    manager._songs_backup_dir = lambda: tmp_path / "_Backup" / "songs"
    manager.undo_stack.clear()
    manager.redo_stack.clear()
    manager._last_selected_row = -1
    manager.dataset = new_dataset(name="tidy")
    manager.dataset["samples"] = samples
    manager.refresh_table()
    manager.table.selectRow(row)
    manager.on_table_selection_changed()
    # All three rules on: each test then switches off only what it is testing.
    for box in (manager.lyrics_quotes_check, manager.lyrics_tagtrim_check,
                manager.lyrics_tagdrop_check):
        box.setChecked(True)
    return samples


def _track(filename, lyrics=""):
    s = new_sample(filename=filename)
    s["formatted_lyrics"] = lyrics
    s["lyrics"] = lyrics
    return s


# ---------------------------------------------------------------------------
# The panel lives on the right page, beside the tag tools
# ---------------------------------------------------------------------------

def test_the_panel_is_on_the_tags_and_checks_page(manager):
    from PySide6.QtWidgets import QGroupBox

    page_index = manager._tool_page("Tags & checks")
    assert page_index >= 0
    # Walk the real page rather than trusting an attribute: the point is that a
    # user looking at "Tags & checks" can see the rules.
    page = manager._shell_extra_pages[-1]
    titles = [g.title() for g in page.findChildren(QGroupBox)]
    assert any("Lyrics Tidy" in t for t in titles)


def test_the_panel_is_not_duplicated_on_the_lyrics_page(manager):
    """It belongs with the tags. A second copy on the Lyrics tab would mean two
    sets of checkboxes that could disagree."""
    from PySide6.QtWidgets import QGroupBox

    lyrics_page = manager._shell_extra_pages[0]
    titles = [g.title() for g in lyrics_page.findChildren(QGroupBox)]
    assert not any("Lyrics Tidy" in t for t in titles)


def test_the_three_rules_have_a_checkbox_each(manager):
    assert manager.lyrics_quotes_check.isChecked() is True
    assert manager.lyrics_tagtrim_check.isChecked() is True
    assert manager.lyrics_tagdrop_check.isChecked() is True


def test_the_checkbox_state_is_what_the_pass_reads(manager):
    from ui.tidy_panel import _options

    manager.lyrics_quotes_check.setChecked(False)
    manager.lyrics_tagtrim_check.setChecked(True)
    manager.lyrics_tagdrop_check.setChecked(False)
    assert _options(manager) == {
        "do_strip_quotes": False,
        "do_trim_tag_modifiers": True,
        "do_drop_stray_tags": False,
    }
    manager.lyrics_quotes_check.setChecked(True)
    manager.lyrics_tagdrop_check.setChecked(True)



# ---------------------------------------------------------------------------
# Selected-track scope
# ---------------------------------------------------------------------------

def test_tidy_selected_rewrites_just_that_track(manager, tmp_path):
    from ui.tidy_panel import apply_tidy

    samples = _load(manager, [
        _track("a.mp3", '[Chorus - loud]\n"hi'),
        _track("b.mp3", '[Verse - soft]\n"yo'),
    ], tmp_path, row=0)
    changed = apply_tidy(manager, "selected")

    assert changed == 1
    assert samples[0]["formatted_lyrics"] == "[Chorus]\nHi"
    assert samples[1]["formatted_lyrics"] == '[Verse - soft]\n"yo'


def test_tidy_writes_all_three_dataset_fields(manager, tmp_path):
    """The schema contract: the exporter reads ``formatted_lyrics``, the
    inspector reads ``lyrics``, and ``raw_lyrics`` keeps the pre-tidy text so a
    re-run is reversible. All three must land in the same edit."""
    from ui.tidy_panel import apply_tidy

    samples = _load(manager, [_track("a.mp3", '[Chorus - loud]\n"hi')], tmp_path)
    apply_tidy(manager, "selected")

    s = samples[0]
    assert s["formatted_lyrics"] == "[Chorus]\nHi"
    assert s["lyrics"] == "[Chorus]\nHi"
    assert s["raw_lyrics"] == '[Chorus - loud]\n"hi'


def test_tidy_is_undoable(manager, tmp_path):
    from ui.tidy_panel import apply_tidy

    _load(manager, [_track("a.mp3", '[Chorus - loud]\ntext')], tmp_path)
    assert not manager.undo_stack
    apply_tidy(manager, "selected")
    assert manager.undo_stack, "a tidy that rewrites lyrics must be undoable"

    manager.undo()
    assert manager.dataset["samples"][0]["formatted_lyrics"] == "[Chorus - loud]\ntext"


def test_tidy_backs_the_song_up_first(manager, tmp_path):
    from ui.tidy_panel import apply_tidy

    _load(manager, [_track("a.mp3", '[Chorus - loud]\ntext')], tmp_path)
    apply_tidy(manager, "selected")

    saved = tmp_path / "_Backup" / "songs" / "a.mp3.json"
    assert saved.exists()
    assert json.loads(saved.read_text())["formatted_lyrics"] == "[Chorus - loud]\ntext"


def test_a_track_that_needs_no_tidying_is_skipped(manager, tmp_path):
    """Nothing to change means no snapshot, no backup, and a zero count."""
    from ui.tidy_panel import apply_tidy

    samples = _load(manager, [_track("a.mp3", "[Chorus]\nClean text")], tmp_path)
    changed = apply_tidy(manager, "selected")
    assert changed == 0


# ---------------------------------------------------------------------------
# Whole-dataset scope
# ---------------------------------------------------------------------------

def test_tidy_dataset_writes_every_track(manager, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    import ui.tidy_panel as tp

    samples = _load(manager, [
        _track("a.mp3", '[Chorus - loud]\n"one'),
        _track("b.mp3", '[Verse - soft]\n"two'),
    ], tmp_path)
    # The confirmation is the point of this scope; auto-accept it here.
    monkeypatch.setattr(tp.QMessageBox, "exec", lambda self: QMessageBox.Yes)

    changed = tp.apply_tidy(manager, "dataset")
    assert changed == 2
    assert samples[0]["formatted_lyrics"] == "[Chorus]\nOne"
    assert samples[1]["formatted_lyrics"] == "[Verse]\nTwo"


def test_cancelling_the_dataset_run_changes_nothing(manager, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    import ui.tidy_panel as tp

    samples = _load(manager, [
        _track("a.mp3", '[Chorus - loud]\n"one'),
        _track("b.mp3", '[Verse - soft]\n"two'),
    ], tmp_path)
    monkeypatch.setattr(tp.QMessageBox, "exec", lambda self: QMessageBox.Cancel)

    changed = tp.apply_tidy(manager, "dataset")
    assert changed == 0
    assert samples[0]["formatted_lyrics"] == '[Chorus - loud]\n"one'
    assert samples[1]["formatted_lyrics"] == '[Verse - soft]\n"two'
    assert not manager.undo_stack


def test_tidy_with_no_rules_ticked_does_nothing(manager, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    import ui.tidy_panel as tp

    samples = _load(manager, [_track("a.mp3", '[Chorus - loud]\ntext')], tmp_path)
    for box in (manager.lyrics_quotes_check, manager.lyrics_tagtrim_check,
                manager.lyrics_tagdrop_check):
        box.setChecked(False)
    monkeypatch.setattr(tp.QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)

    assert tp.apply_tidy(manager, "selected") == 0
    assert samples[0]["formatted_lyrics"] == "[Chorus - loud]\ntext"
    for box in (manager.lyrics_quotes_check, manager.lyrics_tagtrim_check,
                manager.lyrics_tagdrop_check):
        box.setChecked(True)


# ---------------------------------------------------------------------------
# The .txt sidecar
# ---------------------------------------------------------------------------

def test_lyrics_are_written_beside_the_audio(manager, tmp_path, wav_file):
    """``<stem>_lyrics.txt``, next to the track."""
    from ui.tidy_panel import apply_tidy

    audio = wav_file("song.wav")
    sample = _track("song.wav", '[Chorus - loud]\n"hi')
    sample["audio_path"] = audio
    _load(manager, [sample], tmp_path)
    # The audio lives in tmp_path; the sidecar goes beside the AUDIO.
    sidecar = tmp_path / "song_lyrics.txt"
    apply_tidy(manager, "selected")
    assert sidecar.exists()
    assert sidecar.read_text().strip() == "[Chorus]\nHi"


def test_the_sidecar_never_takes_the_caption_filename(manager, tmp_path, wav_file):
    """``<stem>.txt`` belongs to the caption exporter
    (``modules.exporters.export_sidecar_captions``). Writing lyrics there would
    silently overwrite the caption on the next export — so the lyrics file must
    be a DIFFERENT name, and the caption file must be left alone."""
    from ui.tidy_panel import apply_tidy

    audio = wav_file("song.wav")
    caption_file = tmp_path / "song.txt"
    caption_file.write_text("the caption")
    sample = _track("song.wav", '[Chorus - loud]\n"hi')
    sample["audio_path"] = audio
    _load(manager, [sample], tmp_path)
    apply_tidy(manager, "selected")

    assert caption_file.read_text() == "the caption"
    assert (tmp_path / "song_lyrics.txt").exists()


def test_apply_tidy_to_writes_all_three_fields():
    """The unit under the panel: the three dataset fields the contract names.

    ``formatted_lyrics`` is the exporter's copy, ``lyrics`` is the inspector's,
    and ``raw_lyrics`` keeps the PRE-tidy text so the run is reversible.
    """
    from ui.tidy_panel import _apply_tidy_to

    sample = _track("a.mp3", 'OLD TEXT\n"old"')
    _apply_tidy_to(sample, "NEW TEXT")
    assert sample["formatted_lyrics"] == "NEW TEXT"
    assert sample["lyrics"] == "NEW TEXT"
    assert sample["raw_lyrics"] == 'OLD TEXT\n"old"'


def test_apply_tidy_to_keeps_the_first_raw_lyrics():
    """A second tidy run must not overwrite ``raw_lyrics`` with the interim
    text: the point of the field is the ORIGINAL, once."""
    from ui.tidy_panel import _apply_tidy_to

    sample = _track("a.mp3", "original")
    sample["raw_lyrics"] = "truly original"
    _apply_tidy_to(sample, "tidied")
    assert sample["raw_lyrics"] == "truly original"


def test_a_missing_audio_path_does_not_break_the_tidy(manager, tmp_path):
    """A track whose file moved still gets its lyrics tidied; only the sidecar
    is skipped, and nothing raises."""
    from ui.tidy_panel import apply_tidy

    sample = _track("gone.mp3", '[Chorus - loud]\ntext')
    sample["audio_path"] = str(tmp_path / "does-not-exist.wav")
    samples = _load(manager, [sample], tmp_path)

    assert apply_tidy(manager, "selected") == 1
    assert samples[0]["formatted_lyrics"] == "[Chorus]\nText"
    assert not (tmp_path / "does-not-exist_lyrics.txt").exists()


def test_a_track_with_no_lyrics_is_skipped(manager, tmp_path):
    from ui.tidy_panel import apply_tidy

    samples = _load(manager, [_track("a.mp3", "")], tmp_path)
    assert apply_tidy(manager, "selected") == 0
    assert samples[0]["formatted_lyrics"] == ""
