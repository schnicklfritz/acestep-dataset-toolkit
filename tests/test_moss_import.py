"""Importing MOSS kernel output into a dataset."""
import json
import os

import pytest

from modules.moss_import import (
    LYRICS_BACKUP,
    STYLE_BACKUP,
    apply_moss_output,
    backup_dataset,
    load_moss_output,
    normalize_prompt_override,
)


@pytest.fixture
def moss_results():
    return {
        "song_a.mp3": {"style": "doom metal, nasal vocal", "lyrics": "Oh no", "error": False},
        "song_b.wav": {"style": "synthwave", "lyrics": "", "error": False},
    }


@pytest.fixture
def kernel_json(tmp_path):
    payload = {"results": [
        {"file": "song_a.mp3", "style": "doom metal", "lyrics": "Oh no"},
        {"file": "song_b.wav", "style": "synthwave", "lyrics": ""},
        {"file": "song_c.wav", "style": "ERROR: cuda oom", "lyrics": ""},
    ]}
    path = tmp_path / "moss_out.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestLoadMossOutput:
    def test_maps_by_filename(self, kernel_json):
        out = load_moss_output(kernel_json)
        assert set(out) == {"song_a.mp3", "song_b.wav", "song_c.wav"}
        assert out["song_a.mp3"]["style"] == "doom metal"

    def test_error_entries_are_flagged_not_written(self, kernel_json):
        out = load_moss_output(kernel_json)
        assert out["song_c.wav"]["error"] is True
        assert out["song_c.wav"]["style"] == ""

    def test_blank_filename_is_skipped(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text(json.dumps({"results": [{"file": "  ", "style": "x"}]}),
                        encoding="utf-8")
        assert load_moss_output(str(path)) == {}


class TestPromptOverrideNormalisation:
    def test_string_caption_becomes_true(self):
        # The real sabbath.json stores the string "caption" here.
        ds = {"samples": [{"prompt_override": "caption"}]}
        assert normalize_prompt_override(ds) == 1
        assert ds["samples"][0]["prompt_override"] is True

    def test_none_becomes_false(self):
        ds = {"samples": [{"prompt_override": None}]}
        normalize_prompt_override(ds)
        assert ds["samples"][0]["prompt_override"] is False

    def test_real_booleans_are_left_alone(self):
        ds = {"samples": [{"prompt_override": True}, {"prompt_override": False}]}
        assert normalize_prompt_override(ds) == 0

    def test_missing_key_is_written_as_false(self):
        # The schema requires the field and the loader expects a bool, so an
        # absent key is filled rather than left as None.
        ds = {"samples": [{}]}
        assert normalize_prompt_override(ds) == 1
        assert ds["samples"][0]["prompt_override"] is False


class TestApplyMossOutput:
    def test_writes_style_to_caption(self, dataset, moss_results):
        apply_moss_output(dataset, moss_results)
        s = [x for x in dataset["samples"] if x["filename"] == "song_b.wav"][0]
        assert s["caption"] == "synthwave"

    def test_writes_lyrics_to_raw_lyrics(self, dataset, moss_results):
        apply_moss_output(dataset, moss_results)
        s = [x for x in dataset["samples"] if x["filename"] == "song_a.mp3"][0]
        assert s["raw_lyrics"] == "Oh no"

    def test_sets_prompt_override_true(self, dataset, moss_results):
        apply_moss_output(dataset, moss_results)
        for s in dataset["samples"]:
            assert s["prompt_override"] is True

    def test_previous_values_are_preserved_in_backup_fields(self, dataset, moss_results):
        s = [x for x in dataset["samples"] if x["filename"] == "song_a.mp3"][0]
        original_caption = s["caption"]
        apply_moss_output(dataset, moss_results)
        assert s[STYLE_BACKUP] == original_caption
        assert s["caption"] != original_caption

    def test_previous_lyrics_are_backed_up(self, dataset, moss_results):
        apply_moss_output(dataset, moss_results)
        s = [x for x in dataset["samples"] if x["filename"] == "song_a.mp3"][0]
        assert LYRICS_BACKUP in s

    def test_unmatched_tracks_are_reported_not_blanked(self, dataset):
        before = [dict(s) for s in dataset["samples"]]
        report = apply_moss_output(dataset, {"nonexistent.mp3": {"style": "x"}})
        assert "song_a.mp3" in report["unmatched"]
        assert dataset["samples"][0]["caption"] == before[0]["caption"]

    def test_matched_counts_only_real_results(self, dataset, moss_results):
        report = apply_moss_output(dataset, moss_results)
        assert report["matched"] == 2

    def test_empty_style_does_not_blank_an_existing_caption(self, dataset):
        original = dataset["samples"][0]["caption"]
        apply_moss_output(dataset, {"song_a.mp3": {"style": "", "lyrics": "x"}})
        assert dataset["samples"][0]["caption"] == original

    def test_dry_run_writes_nothing(self, dataset, moss_results):
        before = json.dumps(dataset, sort_keys=True)
        report = apply_moss_output(dataset, moss_results, dry_run=True)
        assert json.dumps(dataset, sort_keys=True) == before
        assert report["matched"] == 2

    def test_dry_run_reports_the_same_counts_as_a_real_run(self, dataset, moss_results):
        # A preview whose numbers differ from the real run is worse than no
        # preview: it was reporting every track as "filled" because the
        # overwrite branch was gated on `not dry_run`.
        import copy

        preview_ds = copy.deepcopy(dataset)
        real_ds = copy.deepcopy(dataset)
        preview = apply_moss_output(preview_ds, moss_results, dry_run=True)
        real = apply_moss_output(real_ds, moss_results, dry_run=False)
        for key in ("matched", "overwritten", "filled", "prompt_override_fixed"):
            assert preview[key] == real[key], f"{key} differs between dry and real run"

    def test_dry_run_counts_prompt_override_fixes(self, dataset):
        # The real sabbath.json stores the string "caption" here.
        for s in dataset["samples"]:
            s["prompt_override"] = "caption"
        before = json.dumps(dataset, sort_keys=True)
        report = apply_moss_output(dataset, {}, dry_run=True)
        assert report["prompt_override_fixed"] == len(dataset["samples"])
        assert json.dumps(dataset, sort_keys=True) == before

    def test_error_entries_do_not_touch_the_track(self, dataset):
        before = json.dumps(dataset, sort_keys=True)
        apply_moss_output(dataset, {"song_a.mp3": {"style": "", "lyrics": "", "error": True}})
        assert json.dumps(dataset, sort_keys=True) == before


class TestBackup:
    def test_backup_creates_a_copy(self, tmp_path):
        original = tmp_path / "ds.json"
        original.write_text('{"samples": []}', encoding="utf-8")
        backup = backup_dataset(str(original))
        assert os.path.exists(backup)
        assert open(backup).read() == '{"samples": []}'

    def test_backup_name_is_timestamped(self, tmp_path):
        original = tmp_path / "ds.json"
        original.write_text("{}", encoding="utf-8")
        backup = backup_dataset(str(original))
        assert ".bak-" in os.path.basename(backup)
