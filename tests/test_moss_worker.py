"""App-side MOSS plumbing: prompt defaults, placeholder filling, track picker.

These never touch Kaggle. The push path itself can only be exercised for real,
so everything reachable without network access is pinned here — especially the
placeholder quote convention, which was wrong once already.
"""
import ast
import json

import pytest

from ui.caption_tab import TrackPickerButton
from workers.kaggle_moss import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_MODEL_ID,
    _fill_placeholders,
    _moss_prompts,
)


class TestMossPrompts:
    def test_defaults_when_config_is_empty(self):
        p = _moss_prompts({})
        assert p["model_id"] == DEFAULT_MODEL_ID
        assert p["chunk_seconds"] == DEFAULT_CHUNK_SECONDS
        assert p["style"] and p["lyrics"]

    def test_config_overrides_win(self):
        p = _moss_prompts({
            "moss_model_id": "someone/MOSS-Audio-4B-Instruct",
            "moss_style_prompt": "custom style",
            "moss_lyrics_prompt": "custom lyrics",
            "moss_max_tokens": 256,
            "moss_chunk_seconds": 60,
        })
        assert p["model_id"] == "someone/MOSS-Audio-4B-Instruct"
        assert p["style"] == "custom style"
        assert p["lyrics"] == "custom lyrics"
        assert p["max_tokens"] == 256
        assert p["chunk_seconds"] == 60

    def test_chunk_seconds_stays_under_the_encoder_limit(self):
        # max_source_positions=1500 / audio_tokens_per_second=12.5 = 120 s cap.
        assert DEFAULT_CHUNK_SECONDS < 120

    def test_blank_strings_fall_back_to_defaults(self):
        # Settings round-trip blank fields as "", not None.
        p = _moss_prompts({"moss_model_id": "", "moss_style_prompt": ""})
        assert p["model_id"] == DEFAULT_MODEL_ID
        assert p["style"]


class TestFillPlaceholders:
    SCRIPT = (
        'AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"\n'
        "MODEL_ID = {{MODEL_ID}}\n"
        "STYLE_PROMPT = {{STYLE_PROMPT}}\n"
        "LYRICS_PROMPT = {{LYRICS_PROMPT}}\n"
        "MAX_NEW_TOKENS = {{MAX_NEW_TOKENS}}\n"
        "CHUNK_SECONDS = {{CHUNK_SECONDS}}\n"
        "CUSTOM_TAG = {{CUSTOM_TAG}}\n"
    )

    def _fill(self, **overrides):
        prompts = {
            "model_id": DEFAULT_MODEL_ID,
            "style": "Describe this music.",
            "lyrics": "Transcribe these lyrics.",
            "max_tokens": 1024,
            "chunk_seconds": 110,
        }
        prompts.update(overrides)
        return _fill_placeholders(self.SCRIPT, "/kaggle/input/audio", prompts, "")

    def test_no_placeholders_remain(self):
        out = self._fill()
        assert "{{" not in out and "}}" not in out

    def test_path_is_substituted_without_adding_quotes(self):
        # AUDIO_DATASET_PATH sits INSIDE quotes in the kernel, so the value must
        # be bare. Quoting it here would produce `""/kaggle/input/audio""`.
        assert 'AUDIO_FOLDER = "/kaggle/input/audio"' in self._fill()

    def test_prompts_are_substituted_as_json_string_literals(self):
        out = self._fill()
        assert 'MODEL_ID = "OpenMOSS-Team/MOSS-Audio-8B-Instruct"' in out
        assert 'STYLE_PROMPT = "Describe this music."' in out

    def test_substituted_script_is_valid_python(self):
        ast.parse(self._fill())

    def test_prompt_containing_quotes_is_escaped(self):
        # A prompt with " must not break out of the string literal.
        out = self._fill(style='Say "loud" please')
        ast.parse(out)
        assert "{{" not in out

    def test_numeric_placeholders_are_bare_ints(self):
        out = self._fill()
        assert "MAX_NEW_TOKENS = 1024" in out
        assert "CHUNK_SECONDS = 110" in out

    def test_custom_tag_is_included(self):
        out = _fill_placeholders(
            self.SCRIPT, "/kaggle/input/audio",
            {"model_id": "m", "style": "s", "lyrics": "l",
             "max_tokens": 1, "chunk_seconds": 1},
            "sabbath",
        )
        assert 'CUSTOM_TAG = "sabbath"' in out


@pytest.fixture
def picker(qapp):
    return TrackPickerButton()


def _samples(*pairs):
    """``_samples(("a.mp3", "cap"), ("b.mp3", ""))`` -> dataset samples."""
    return [{"filename": n, "caption": c} for n, c in pairs]


class TestTrackPicker:
    def test_starts_empty(self, picker):
        assert picker.selected_filenames() == []
        assert picker.is_empty()

    def test_set_tracks_reports_zero_ticked(self, picker):
        picker.set_tracks(_samples(("a.mp3", "x"), ("b.wav", "y")))
        assert "0 of 2 selected" in picker.text()

    def test_select_all_ticks_everything(self, picker):
        picker.set_tracks(_samples(("a.mp3", "x"), ("b.wav", "y")))
        picker.select_all()
        assert picker.selected_filenames() == ["a.mp3", "b.wav"]
        assert "2 of 2 selected" in picker.text()

    def test_select_none_clears(self, picker):
        picker.set_tracks(_samples(("a.mp3", "x"), ("b.wav", "y")))
        picker.select_all()
        picker.select_none()
        assert picker.selected_filenames() == []

    def test_select_missing_captions(self, picker):
        picker.set_tracks(
            _samples(("a.mp3", "has one"), ("b.wav", ""), ("c.mp3", "  "))
        )
        picker.select_missing_captions()
        assert picker.selected_filenames() == ["b.wav", "c.mp3"]

    def test_ticks_survive_a_reorder(self, picker):
        picker.set_tracks(_samples(("a.mp3", "x"), ("b.wav", "y")))
        picker.select_all()
        # A reorder forces a rebuild; ticks are keyed by filename, not row.
        picker.set_tracks(_samples(("b.wav", "y"), ("a.mp3", "x")))
        assert picker.selected_filenames() == ["b.wav", "a.mp3"]

    def test_ticks_drop_when_a_track_disappears(self, picker):
        picker.set_tracks(_samples(("a.mp3", "x"), ("b.wav", "y")))
        picker.select_all()
        picker.set_tracks(_samples(("a.mp3", "x")))
        assert picker.selected_filenames() == ["a.mp3"]

    def test_unchanged_list_does_not_rebuild_the_menu(self, picker):
        # refresh_table() fires constantly; this guard is what keeps it cheap.
        picker.set_tracks(_samples(("a.mp3", "x")))
        picker.select_all()
        before = dict(picker._actions)
        picker.set_tracks(_samples(("a.mp3", "x")))
        assert picker._actions == before, "menu was needlessly rebuilt"
        assert picker.selected_filenames() == ["a.mp3"]

    def test_unchanged_list_still_refreshes_stored_samples(self, picker):
        # Captions change without the track list changing, and
        # select_missing_captions() reads the stored samples.
        picker.set_tracks(_samples(("a.mp3", "")))
        picker.select_missing_captions()
        assert picker.selected_filenames() == ["a.mp3"]
        picker.set_tracks(_samples(("a.mp3", "now captioned")))
        picker.select_missing_captions()
        assert picker.selected_filenames() == []

    def test_blank_filenames_are_ignored(self, picker):
        picker.set_tracks([{"filename": "  "}, {"filename": "a.mp3"}])
        assert picker._known == ["a.mp3"]

    def test_selection_changed_signal_fires(self, picker):
        seen = []
        picker.selection_changed.connect(lambda: seen.append(1))
        picker.set_tracks(_samples(("a.mp3", "x")))
        picker.select_all()
        picker.select_none()
        assert len(seen) == 2

    def test_action_triggered_bool_is_not_swallowed_as_an_argument(self, picker):
        # QAction.triggered emits a `checked` bool; the slots take *_, so it
        # must not be mistaken for a real parameter.
        picker.set_tracks(_samples(("a.mp3", "")))
        picker.select_all(False)
        assert picker.selected_filenames() == ["a.mp3"]
        picker.select_missing_captions(True)
        assert picker.selected_filenames() == ["a.mp3"]
        picker.select_none(False)
        assert picker.selected_filenames() == []

