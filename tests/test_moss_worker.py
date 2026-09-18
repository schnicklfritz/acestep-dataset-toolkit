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
        "ATTN_IMPL = {{ATTN_IMPL}}\n"
    )

    def _fill(self, **overrides):
        prompts = {
            "model_id": DEFAULT_MODEL_ID,
            "style": "Describe this music.",
            "lyrics": "Transcribe these lyrics.",
            "max_tokens": 1024,
            "chunk_seconds": 110,
            "attn_impl": "",
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
             "max_tokens": 1, "chunk_seconds": 1, "attn_impl": ""},
            "sabbath",
        )
        assert 'CUSTOM_TAG = "sabbath"' in out

    def test_attention_backend_defaults_to_empty(self):
        # Empty means "leave it to the model": MOSS's audio encoder pins eager,
        # and forcing a backend at the top level is untested.
        assert 'ATTN_IMPL = ""' in self._fill()

    def test_attention_backend_can_be_requested(self):
        out = self._fill(attn_impl="sdpa")
        assert 'ATTN_IMPL = "sdpa"' in out
        ast.parse(out)

    def test_unset_attention_backend_is_omitted_from_the_default_prompts(self):
        # An empty config value must become "", never the string "None".
        assert _moss_prompts({})["attn_impl"] == ""
        assert _moss_prompts({"moss_attn_implementation": "   "})["attn_impl"] == ""

    def test_attention_backend_config_is_passed_through(self):
        assert _moss_prompts({"moss_attn_implementation": "sdpa"})["attn_impl"] == "sdpa"


class TestPlaceholderGuard:
    """A stale app must fail instantly, not after a Kaggle round-trip.

    The real failure: the kernel file gained `{{ATTN_IMPL}}` while the running
    app still had the OLD substitution function in memory (Python caches
    modules). The placeholder reached Kaggle unsubstituted and died 52 seconds
    later with `NameError: name 'ATTN_IMPL' is not defined` -- because `{{X}}`
    is valid Python (a set containing a set), it was not a syntax error.
    """

    def test_unknown_placeholder_raises(self):
        script = "X = {{SOMETHING_NEW}}\n"
        with pytest.raises(RuntimeError) as exc:
            _fill_placeholders(script, "/kaggle/input/a",
                               _moss_prompts({}), "")
        assert "SOMETHING_NEW" in str(exc.value)

    def test_error_message_names_the_remedy(self):
        script = "X = {{SOMETHING_NEW}}\n"
        with pytest.raises(RuntimeError) as exc:
            _fill_placeholders(script, "/kaggle/input/a", _moss_prompts({}), "")
        assert "restart" in str(exc.value).lower(), (
            "the message must say what to DO, not just what is wrong"
        )

    def test_all_leftover_placeholders_are_reported_together(self):
        script = "A = {{ONE}}\nB = {{TWO}}\n"
        with pytest.raises(RuntimeError) as exc:
            _fill_placeholders(script, "/kaggle/input/a", _moss_prompts({}), "")
        message = str(exc.value)
        assert "ONE" in message and "TWO" in message

    def test_a_fully_known_script_does_not_raise(self):
        script = "A = {{MODEL_ID}}\nB = {{CHUNK_SECONDS}}\n"
        out = _fill_placeholders(script, "/kaggle/input/a", _moss_prompts({}), "")
        assert "{{" not in out

    def test_the_real_kernel_file_fills_completely(self):
        """The CI invariant that was missing.

        If someone adds a placeholder to the kernel without teaching
        _fill_placeholders about it, this fails at commit time instead of on
        Kaggle. Running the REAL file (not a fixture) is the point.
        """
        from workers.kaggle_moss import KERNEL_SCRIPT, _moss_prompts

        script = KERNEL_SCRIPT.read_text(encoding="utf-8")
        filled = _fill_placeholders(script, "/kaggle/input/probe",
                                    _moss_prompts({}), "")
        assert "{{" not in filled and "}}" not in filled

    def test_the_real_kernel_file_is_valid_python_once_filled(self):
        from workers.kaggle_moss import KERNEL_SCRIPT, _moss_prompts

        script = KERNEL_SCRIPT.read_text(encoding="utf-8")
        filled = _fill_placeholders(script, "/kaggle/input/probe",
                                    _moss_prompts({}), "")
        ast.parse(filled)

    def test_validation_runs_before_the_audio_upload(self):
        # Otherwise a placeholder mismatch still costs an upload.
        #
        # Match the CALL, not the bare name: "upload_audio_dataset" first
        # appears in the function's import block, so a naive index() comparison
        # would compare against the import rather than the call site.
        import inspect
        from workers import kaggle_moss
        src = inspect.getsource(kaggle_moss.run_kaggle_moss)
        assert src.index("_fill_placeholders(") < \
            src.index("upload_audio_dataset(config"), (
                "placeholder validation must happen BEFORE upload_audio_dataset"
            )
    """A failed run must report WHY, not point at Kaggle's web UI."""

    def test_tail_is_returned(self, monkeypatch):
        from modules import kaggle

        class FakeApi:
            def kernels_logs(self, _slug):
                return "line1\nline2\nline3"

        monkeypatch.setattr(kaggle, "_get_api", lambda _c: (FakeApi(), "u"))
        assert kaggle.fetch_kernel_logs({}, "k") == "line1\nline2\nline3"

    def test_long_logs_are_truncated_to_the_tail(self, monkeypatch):
        # Failures are at the END, so a truncated log must keep the tail.
        from modules import kaggle

        body = "START\n" + ("x" * 10000) + "\nTHE ACTUAL ERROR"

        class FakeApi:
            def kernels_logs(self, _slug):
                return body

        monkeypatch.setattr(kaggle, "_get_api", lambda _c: (FakeApi(), "u"))
        out = kaggle.fetch_kernel_logs({}, "k", max_chars=100)
        assert out.endswith("THE ACTUAL ERROR")
        assert len(out) <= 101        # 100 + the ellipsis marker
        assert not out.startswith("START")

    def test_unavailable_log_returns_empty_not_an_exception(self, monkeypatch):
        # Diagnostics must never mask the original failure.
        from modules import kaggle

        def boom(_c):
            raise RuntimeError("no creds")

        monkeypatch.setattr(kaggle, "_get_api", boom)
        assert kaggle.fetch_kernel_logs({}, "k") == ""

    def test_empty_log_returns_empty(self, monkeypatch):
        from modules import kaggle

        class FakeApi:
            def kernels_logs(self, _slug):
                return None

        monkeypatch.setattr(kaggle, "_get_api", lambda _c: (FakeApi(), "u"))
        assert kaggle.fetch_kernel_logs({}, "k") == ""

    def test_status_text_returns_empty_on_error(self, monkeypatch):
        from modules import kaggle

        def boom(_c):
            raise RuntimeError("nope")

        monkeypatch.setattr(kaggle, "_get_api", boom)
        assert kaggle.kernel_status_text({}, "k") == ""

    def test_failure_message_includes_the_log_tail(self):
        # The worker must embed the log, not tell the user to go and read it.
        import inspect
        from workers import kaggle_moss
        src = inspect.getsource(kaggle_moss.run_kaggle_moss)
        assert "fetch_kernel_logs" in src
        assert "log tail" in src
        assert "Open the run log in Kaggle" not in src


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

