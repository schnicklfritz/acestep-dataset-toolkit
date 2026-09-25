"""Contract tests for the 🅰 ACE-Step (Kaggle) page and its wiring.

The page is connected by NAME: ``init_ace_step_tab`` looks up
``manager.caption_addendum_edit``, ``manager.max_tokens_spin``, ... and connects
them to slots like ``self.show_caption_diff``. A rename on either side is not
caught by importing anything — it fails at click time, in front of the user, with
an AttributeError. These tests pin both halves of that contract without building
the whole window.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt                                       # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget                     # noqa: E402

import dataset_manager                                                  # noqa: E402
from ui.ace_step_tab import build_ace_step_tab                          # noqa: E402


class _FakeManager:
    """Just enough manager for the builder: a config dict and free attributes."""

    def __init__(self, config=None):
        self.config = dict(config or {})


@pytest.fixture(autouse=True, scope="module")
def qapp():
    """A QApplication for the whole module: widgets cannot exist without one."""
    app = QApplication.instance() or QApplication([])
    yield app


def _built(config=None):
    manager = _FakeManager(config)
    page = QWidget()
    build_ace_step_tab(manager, page)
    # RETAIN the page. PySide6 destroys a C++ object once its last Python
    # reference is gone, and destroying a parent destroys its children — a
    # local-only page deletes every widget the builder just created, and the
    # manager then holds dangling wrappers ("Internal C++ object already
    # deleted"). This is the same trap `_build_grouped_tab` documents.
    manager._page = page
    return manager


def test_the_page_builds_the_settings_it_owns(qapp):
    manager = _built()
    for attr in ("caption_addendum_edit", "caption_staging_edit",
                 "caption_audio_dataset_edit", "caption_model_dataset_edit",
                 "caption_output_edit",
                 "caption_bitrate_combo", "caption_convert_check"):
        assert hasattr(manager, attr), attr


def test_the_weights_dataset_field_defaults_to_the_cached_export(qapp):
    from config import DEFAULT_CONFIG

    manager = _built({"kaggle_model_dataset": DEFAULT_CONFIG["kaggle_model_dataset"]})
    assert manager.caption_model_dataset_edit.text() == \
        "michelmoalem9b/acestep-captioner-model"


def test_the_weights_dataset_field_can_be_left_empty(qapp):
    """Empty = the kernel downloads from Hugging Face. The UI must allow it."""
    manager = _built({"kaggle_model_dataset": ""})
    assert manager.caption_model_dataset_edit.text() == ""


def test_the_limit_spins_keep_the_names_the_rest_of_the_app_reads(qapp):
    """save_pipeline_defaults() reads these three by name, unguarded."""
    manager = _built()
    assert manager.max_tokens_spin.value() == 512
    assert manager.max_dur_spin.value() == 120
    assert manager.batch_size_spin.value() == 1


def test_the_run_buttons_are_created_here_not_in_the_other_page(qapp):
    manager = _built()
    for attr in ("caption_selected_btn", "caption_missing_btn",
                 "caption_all_btn", "caption_edit_btn",
                 "caption_recaption_bad_btn", "caption_diff_btn",
                 "caption_import_btn"):
        assert hasattr(manager, attr), attr


def test_the_page_has_its_own_track_picker(qapp):
    """The page must not depend on the dataset table's row selection."""
    manager = _built()
    for attr in ("ace_track_picker", "ace_tick_status"):
        assert hasattr(manager, attr), attr
    assert manager.ace_track_picker.selected_filenames() == []


def test_no_ace_step_control_tells_the_user_to_go_to_another_tab():
    """Ticking belongs on this page; the Studio is only where SONGS are added."""
    with open(os.path.join(ROOT, "ui", "ace_step_tab.py"), encoding="utf-8") as fh:
        source = fh.read()
    for phrase in ("selected in the Dataset Studio table",
                   "Select the track(s) to add in the Dataset Studio"):
        assert phrase not in source, phrase
    # The manager keeps the one legitimate cross-reference: adding SONGS to the
    # dataset is still Dataset Studio's job.
    with open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8") as fh:
        assert "add songs in 🎛 Dataset Studio" in fh.read()


def test_the_credentials_row_offers_a_real_connection_test(qapp):
    """Where the key lives and whether Kaggle ACCEPTS it are different questions.

    The row used to answer only the first one (two non-empty strings), which is
    how a bad token looked like a working setup. The button must exist, and the
    manager must expose the slots it is connected to BY NAME.
    """
    manager = _built()
    assert hasattr(manager, "ace_cred_test_btn"), "ace_cred_test_btn"
    for slot in ("test_kaggle_connection", "_on_kaggle_probe_done",
                 "_on_kaggle_probe_failed"):
        assert hasattr(dataset_manager.DatasetManager, slot), slot


def test_the_credentials_row_exists(qapp):
    manager = _built()
    for attr in ("ace_cred_status", "ace_cred_setup_btn",
                 "ace_cred_forget_btn", "ace_cred_reset_btn"):
        assert hasattr(manager, attr), attr


def test_the_user_turn_prompt_editor_is_NOT_duplicated_here(qapp):
    """Two editors writing caption_prompt would be a silent conflict."""
    manager = _built()
    assert not hasattr(manager, "prompt_edit")


def test_settings_are_loaded_from_config(qapp):
    manager = _built({
        "caption_prompt_addendum": "live bootleg",
        "caption_staging_dir": "/tmp/stage",
        "caption_output_dir": "/tmp/out",
        "caption_audio_dataset": "me/ace-audio",
        "kaggle_model_dataset": "me/captioner-weights",
        "caption_convert_mp3": False,
        "caption_batch_review": False,
        "caption_mp3_bitrate": "320k",
    })
    assert manager.caption_addendum_edit.toPlainText() == "live bootleg"
    assert manager.caption_staging_edit.text() == "/tmp/stage"
    assert manager.caption_output_edit.text() == "/tmp/out"
    assert manager.caption_audio_dataset_edit.text() == "me/ace-audio"
    assert manager.caption_model_dataset_edit.text() == "me/captioner-weights"
    assert manager.caption_convert_check.isChecked() is False
    assert manager.caption_batch_review_check.isChecked() is False
    assert manager.caption_bitrate_combo.currentText() == "320k"


def test_folder_fields_fall_back_to_a_real_path_not_an_empty_string(qapp):
    """An empty field would make a run stage into the current directory."""
    manager = _built()
    assert manager.caption_staging_edit.text()
    assert manager.caption_output_edit.text()


def test_every_slot_init_ace_step_tab_connects_exists():
    """The wiring is by name; a renamed slot fails only when the user clicks."""
    for method in ("caption_selected_track", "caption_missing_tracks",
                   "caption_all_tracks", "open_caption_editor",
                   "caption_recaption_bad_tracks", "show_caption_diff",
                   "import_captions_json", "browse_caption_staging",
                   "browse_caption_output", "refresh_staging_list",
                   "staging_add_ticked", "staging_remove_ticked",
                   # the page's ONLY track selector
                   "_ticked_samples", "refresh_ace_track_picker",
                   "update_ace_tick_status", "_no_tracks_ticked",
                   "save_pipeline_defaults", "caption_batch_review_enabled",
                   "_bad_caption_samples", "_proposal_rows", "_sample_by_id",
                   "_on_diff_decision_changed",
                   # credential path (wired by name from this page)
                   "configure_kaggle_credentials", "forget_stored_kaggle_key",
                   "reset_caption_backend_prompts", "refresh_kaggle_cred_status",
                   "_ensure_kaggle_credentials", "_resolve_caption_backend",
                   "_fallback_backend_options", "_prompt_fallback_backend",
                   "_stamp_placeholder_caption"):
        assert hasattr(dataset_manager.DatasetManager, method), method


def test_the_new_config_keys_all_have_defaults():
    """A missing default is a KeyError on the first run, not a UI nit."""
    from config import DEFAULT_CONFIG

    for key in ("caption_prompt_addendum", "caption_output_dir",
                "caption_staging_dir", "caption_audio_dataset",
                "caption_convert_mp3", "caption_mp3_bitrate",
                "caption_batch_review", "caption_cred_prompt_seen",
                "caption_fallback_backend", "caption_stamp_placeholders"):
        assert key in DEFAULT_CONFIG, key


# ---------------------------------------------------------------------------
# the tick list: the page's only track selector
# ---------------------------------------------------------------------------

class _PickerStub:
    """A TrackPickerButton stand-in: just the ticks and the last track list."""

    def __init__(self, ticks=()):
        self._ticks = list(ticks)
        self.tracks = []

    def selected_filenames(self):
        return list(self._ticks)

    def set_tracks(self, samples):
        self.tracks = samples


class _LabelStub:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _TickManager:
    """The REAL tick methods on a plain object (they only touch config/dataset)."""

    _ticked_samples = dataset_manager.DatasetManager._ticked_samples
    refresh_ace_track_picker = dataset_manager.DatasetManager.refresh_ace_track_picker
    update_ace_tick_status = dataset_manager.DatasetManager.update_ace_tick_status

    def __init__(self, samples, ticks=(), picker=None):
        self.dataset = {"samples": samples}
        self.ace_track_picker = picker if picker is not None else _PickerStub(ticks)
        self.ace_tick_status = _LabelStub()


def _samples(*names):
    return [{"id": f"id-{name}", "filename": name} for name in names]


def test_ticked_samples_maps_ticks_to_samples_in_dataset_order():
    manager = _TickManager(_samples("a.mp3", "b.mp3", "c.mp3"),
                           ticks=["c.mp3", "a.mp3"])
    assert [s["filename"] for s in manager._ticked_samples()] == ["a.mp3", "c.mp3"]


def test_ticked_samples_ignores_ticks_for_removed_tracks():
    """Ticks are filenames, so a deleted track must drop out, not shift onto another."""
    manager = _TickManager(_samples("b.mp3"), ticks=["a.mp3", "b.mp3"])
    assert [s["filename"] for s in manager._ticked_samples()] == ["b.mp3"]


def test_no_ticks_means_no_samples():
    assert _TickManager(_samples("a.mp3"), ticks=[])._ticked_samples() == []


def test_tick_status_points_at_the_dropdown_when_nothing_is_ticked():
    manager = _TickManager(_samples("a.mp3"), ticks=[])
    manager.update_ace_tick_status()
    assert "Nothing ticked" in manager.ace_tick_status.text
    assert "Tracks ▾" in manager.ace_tick_status.text


def test_tick_status_counts_the_ticks():
    manager = _TickManager(_samples("a.mp3", "b.mp3"), ticks=["a.mp3"])
    manager.update_ace_tick_status()
    assert "1 of 2 ticked" in manager.ace_tick_status.text


def test_tick_status_names_both_ways_to_get_tracks():
    """An empty dataset is not the same as an absent one.

    The hint used to say only "add songs in 🎛 Dataset Studio", so a user who had
    simply restarted the app (it never reopens the last dataset) was told to ADD
    songs, and reported their dataset as GONE. It must name the Load button too.
    """
    manager = _TickManager([], ticks=[])
    manager.update_ace_tick_status()
    text = manager.ace_tick_status.text
    assert "📂 Open" in text
    assert "Dataset Studio" in text
    assert "starts empty" in text


def test_refresh_keeps_the_picker_in_step_with_the_dataset():
    manager = _TickManager(_samples("a.mp3"), ticks=["a.mp3"])
    manager.refresh_ace_track_picker()
    assert [s["filename"] for s in manager.ace_track_picker.tracks] == ["a.mp3"]
    manager.dataset["samples"] = _samples("a.mp3", "b.mp3")
    manager.refresh_ace_track_picker()
    assert len(manager.ace_track_picker.tracks) == 2


# ---------------------------------------------------------------------------
# the strip: one line, in workflow order, with only the next step live
# ---------------------------------------------------------------------------

class _PageManager(_FakeManager):
    """The REAL page plus the real gating/tick methods, without a QMainWindow.

    ``build_ace_step_tab`` only needs ``config``, and ``update_ace_actions`` needs
    a dataset, a scope list and the busy flag — so the layout AND the gating rules
    can be tested without constructing the whole app.
    """

    _ticked_samples = dataset_manager.DatasetManager._ticked_samples
    _caption_run_samples = dataset_manager.DatasetManager._caption_run_samples
    _proposal_rows = dataset_manager.DatasetManager._proposal_rows
    _bad_caption_samples = dataset_manager.DatasetManager._bad_caption_samples
    caption_batch_review_enabled = \
        dataset_manager.DatasetManager.caption_batch_review_enabled
    _set_ace_action = dataset_manager.DatasetManager._set_ace_action
    update_ace_actions = dataset_manager.DatasetManager.update_ace_actions
    update_ace_tick_status = dataset_manager.DatasetManager.update_ace_tick_status
    _staging_rows = dataset_manager.DatasetManager._staging_rows
    _selected_staging_names = dataset_manager.DatasetManager._selected_staging_names
    refresh_staging_list = dataset_manager.DatasetManager.refresh_staging_list
    staging_remove_ticked = dataset_manager.DatasetManager.staging_remove_ticked
    on_audio_dataset_ready = dataset_manager.DatasetManager.on_audio_dataset_ready

    def __init__(self, config=None, samples=()):
        super().__init__(config)
        self.dataset = {"samples": list(samples)}
        self._caption_scope_ids = []
        self._caption_busy = False
        self.status_label = _LabelStub()


def _built_page(config=None, samples=()):
    manager = _PageManager(config, samples)
    page = QWidget()
    build_ace_step_tab(manager, page)
    # RETAIN the page: see the note in _built() about PySide6 freeing a C++ object
    # whose last Python reference is dropped.
    manager._page = page
    return manager


def _strip_widgets(manager):
    """The strip's widgets, in order, excluding the stretchy status label."""
    layout = manager.ace_strip.layout()
    out = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if widget is not None and widget is not manager.ace_tick_status:
            out.append(widget)
    return out


def test_the_strip_is_in_workflow_order(qapp):
    """The order IS the documentation: tick, stage, caption, then review."""
    manager = _built_page()
    assert _strip_widgets(manager) == [
        manager.ace_cred_test_btn,
        manager.ace_track_picker,
        manager.staging_add_btn,
        manager.caption_selected_btn,
        manager.caption_missing_btn,
        manager.caption_all_btn,
        manager.caption_edit_btn,
        manager.caption_diff_btn,
        manager.caption_recaption_bad_btn,
        manager.caption_import_btn,
    ]


def test_set_once_controls_stay_off_the_strip(qapp):
    """Folders, limits and credential management belong in collapsed Settings."""
    manager = _built_page()
    on_strip = set(_strip_widgets(manager))
    assert manager.ace_cred_test_btn in on_strip
    for attr in ("caption_staging_edit", "caption_output_edit",
                 "caption_audio_dataset_edit", "caption_model_dataset_edit",
                 "caption_addendum_edit", "caption_bitrate_combo",
                 "max_tokens_spin", "max_dur_spin", "batch_size_spin",
                 "caption_convert_check", "caption_batch_review_check",
                 "ace_cred_setup_btn", "ace_cred_forget_btn",
                 "ace_cred_reset_btn"):
        assert getattr(manager, attr) not in on_strip, attr


def test_the_next_step_is_disabled_until_its_input_exists(qapp, tmp_path):
    manager = _built_page({"caption_staging_dir": str(tmp_path)})
    manager.update_ace_actions()
    assert not manager.staging_add_btn.isEnabled()
    assert not manager.caption_selected_btn.isEnabled()
    assert not manager.caption_edit_btn.isEnabled()
    # Nothing has run, so there is nothing to review and nothing to re-caption.
    assert not manager.caption_diff_btn.isEnabled()
    # ...and an empty staging folder has no junk to clean.
    assert not manager.staging_clean_btn.isEnabled()
    # Import stays live: it is the manual entry point for a hand-run result.
    assert manager.caption_import_btn.isEnabled()


def test_a_disabled_step_says_why_in_its_tooltip(qapp, tmp_path):
    manager = _built_page({"caption_staging_dir": str(tmp_path)})
    manager.update_ace_actions()
    assert "Tick tracks" in manager.staging_add_btn.toolTip()
    assert "Tick tracks" in manager.caption_selected_btn.toolTip()
    assert "Nothing to review" in manager.caption_diff_btn.toolTip()


def test_ticking_a_track_turns_the_stage_and_caption_steps_on(qapp, tmp_path):
    samples = [{"id": "1", "filename": "a.mp3"}]
    manager = _built_page({"caption_staging_dir": str(tmp_path)}, samples)
    manager.ace_track_picker.set_tracks(samples)
    manager.ace_track_picker.select_all()
    manager.update_ace_actions()
    assert manager.staging_add_btn.isEnabled()
    assert manager.caption_selected_btn.isEnabled()
    assert manager.caption_edit_btn.isEnabled()


def test_a_run_in_flight_says_so_and_blocks_the_other_steps(qapp, tmp_path):
    samples = [{"id": "1", "filename": "a.mp3"}]
    manager = _built_page({"caption_staging_dir": str(tmp_path)}, samples)
    manager.ace_track_picker.set_tracks(samples)
    manager.ace_track_picker.select_all()
    manager._caption_busy = True
    manager.update_ace_actions()
    assert not manager.caption_selected_btn.isEnabled()
    assert not manager.staging_add_btn.isEnabled()
    assert not manager.caption_missing_btn.isEnabled()
    assert manager.caption_selected_btn.text() == "⏳ Captioning…"
    assert "in flight" in manager.caption_selected_btn.toolTip()


def test_the_review_chip_counts_what_is_waiting(qapp, tmp_path):
    """State ON the control: the count is what makes "anything to review?"
    answerable without clicking anything."""
    samples = [{"id": "1", "filename": "a.mp3", "caption": "",
                "caption_ai_raw": "A brand new caption."}]
    manager = _built_page({"caption_staging_dir": str(tmp_path)}, samples)
    manager._caption_scope_ids = ["1"]
    manager.update_ace_actions()
    assert "Review · 1" in manager.caption_diff_btn.text()
    assert manager.caption_diff_btn.isEnabled()
    # That same track has no caption, so re-caption is live too.
    assert "Re-caption · 1" in manager.caption_recaption_bad_btn.text()


# ---------------------------------------------------------------------------
# staging: one meaning for "ticked", and the two names of one song
# ---------------------------------------------------------------------------

def test_removing_by_tick_deletes_the_staged_file_for_that_track(qapp, tmp_path):
    """THE BUG: ➕ Stage used the “Tracks ▾” ticks while ➖ Remove used the staging
    list's ROW selection — one word, two selections, two buttons."""
    (tmp_path / "aint_no_fun.mp3").write_bytes(b"audio")
    samples = [{"id": "1", "filename": "aint_no_fun.flac"}]
    manager = _built_page({"caption_staging_dir": str(tmp_path)}, samples)
    manager.ace_track_picker.set_tracks(samples)
    manager.ace_track_picker.select_all()
    manager.refresh_staging_list()
    assert [row[1] for row in manager._staging_rows()] == ["aint_no_fun.mp3"]

    manager.staging_remove_ticked()          # ticked in the dropdown only
    assert manager._staging_rows() == []


def test_the_staged_list_names_the_dataset_track_behind_each_file(qapp, tmp_path):
    """The dataset says .flac and the upload says .mp3, so without the mapping the
    same song appears under two names with nothing connecting them."""
    (tmp_path / "aint_no_fun.mp3").write_bytes(b"audio")
    (tmp_path / "orphan.mp3").write_bytes(b"audio")
    samples = [{"id": "1", "filename": "aint_no_fun.flac"}]
    manager = _built_page({"caption_staging_dir": str(tmp_path)}, samples)
    manager.refresh_staging_list()

    labels = [manager.staging_list.item(i).text()
              for i in range(manager.staging_list.count())]
    assert any("aint_no_fun.mp3" in t and "aint_no_fun.flac" in t for t in labels)
    assert any("orphan.mp3" in t and "not in this dataset" in t for t in labels)
    # The name handed to the filesystem is item data, NOT the decorated label.
    assert manager.staging_list.item(0).data(Qt.UserRole) == "aint_no_fun.mp3"


def test_a_stranded_staged_file_can_still_be_removed_from_the_list(qapp, tmp_path):
    """A staged file with no dataset track has no tick that could reach it."""
    (tmp_path / "orphan.mp3").write_bytes(b"audio")
    manager = _built_page({"caption_staging_dir": str(tmp_path)})
    manager.refresh_staging_list()
    manager.staging_list.item(0).setSelected(True)
    manager.staging_remove_ticked()
    assert manager._staging_rows() == []


def test_the_uploaded_dataset_slug_is_remembered_and_shown(qapp, tmp_path, monkeypatch):
    """settings.json kept caption_audio_dataset="" while every run created a NEW
    dataset (ace-audio-80d01a, then ace-audio-d66edb), so the "new version of the
    SAME dataset" promise never engaged and the page could not say where the audio
    went."""
    from modules import config_store

    monkeypatch.setattr(config_store, "save_config", lambda *a, **k: None)
    manager = _built_page({"caption_staging_dir": str(tmp_path)})
    manager.on_audio_dataset_ready("akronohio/ace-audio-d66edb")
    assert manager.config["caption_audio_dataset"] == "akronohio/ace-audio-d66edb"
    assert manager.caption_audio_dataset_edit.text() == "akronohio/ace-audio-d66edb"


def test_the_whole_song_option_is_offered_and_defaults_on(qapp):
    """The DEFAULT must be the correct caption (whole song), not the cheap one.

    Reported by the user as "120 seconds is not good, that is less than half": a
    single pass discarded the rest of every song, so the caption described the
    first two minutes of a four-minute track.
    """
    from config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["caption_whole_song"] is True
    manager = _built_page()
    assert manager.caption_whole_song_check.isChecked() is True
    assert "whole song" in manager.caption_whole_song_check.text()
    # ...and it is respected when the user turns it off.
    assert _built_page({"caption_whole_song": False}).caption_whole_song_check.isChecked() is False


def test_the_pass_length_is_labelled_as_a_pass_not_a_limit(qapp):
    """The spin sets the CHUNK size; coverage comes from the option above."""
    manager = _built_page({"caption_max_audio_duration": 90})
    assert manager.max_dur_spin.value() == 90
    assert "SECONDS PER PASS" in manager.max_dur_spin.toolTip()

