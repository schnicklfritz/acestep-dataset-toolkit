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


def test_tick_status_names_where_songs_are_added_on_an_empty_dataset():
    manager = _TickManager([], ticks=[])
    manager.update_ace_tick_status()
    assert "Dataset Studio" in manager.ace_tick_status.text


def test_refresh_keeps_the_picker_in_step_with_the_dataset():
    manager = _TickManager(_samples("a.mp3"), ticks=["a.mp3"])
    manager.refresh_ace_track_picker()
    assert [s["filename"] for s in manager.ace_track_picker.tracks] == ["a.mp3"]
    manager.dataset["samples"] = _samples("a.mp3", "b.mp3")
    manager.refresh_ace_track_picker()
    assert len(manager.ace_track_picker.tracks) == 2
