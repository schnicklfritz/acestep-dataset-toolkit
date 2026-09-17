"""ACE-Step export mapping and schema normalization.

The export field names are load-bearing: emitting the app's internal names
would make ACE-Step's training preprocessor silently mis-parse them, so these
are the highest-value assertions in the suite.
"""
import pytest

from modules.dataset_schema import (
    EXPORT_FIELD_MAP,
    METADATA_DEFAULTS,
    SAMPLE_DEFAULTS,
    normalize_dataset,
    new_dataset,
    new_sample,
    to_export_dataset,
    to_export_sample,
)


class TestExportFieldNames:
    def test_audio_path_becomes_file_name(self):
        s = new_sample(filename="a.mp3", audio_path="./songs/a.mp3")
        assert to_export_sample(s)["file_name"] == "./songs/a.mp3"

    def test_filename_is_the_fallback_when_no_path(self):
        s = new_sample(filename="a.mp3", audio_path="")
        assert to_export_sample(s)["file_name"] == "a.mp3"

    def test_is_instrumental_becomes_instrumental(self):
        s = new_sample(is_instrumental=True)
        out = to_export_sample(s)
        assert out["instrumental"] is True

    @pytest.mark.parametrize("internal", ["audio_path", "is_instrumental", "filename"])
    def test_internal_names_never_leak(self, internal):
        out = to_export_sample(new_sample(filename="a.mp3", audio_path="a.mp3"))
        assert internal not in out

    def test_prompt_override_is_a_boolean(self):
        assert to_export_sample(new_sample())["prompt_override"] is False
        assert to_export_sample(new_sample(prompt_override=True))["prompt_override"] is True

    def test_prompt_override_coerced_from_truthy(self):
        # Legacy files may hold a non-bool; the loader needs a real bool.
        out = to_export_sample(new_sample(prompt_override="yes"))
        assert out["prompt_override"] is True

    def test_app_internal_bookkeeping_is_dropped(self):
        s = new_sample(filename="a.mp3", audio_path="a.mp3")
        s["locked"] = True
        s["spatial_tokens"] = {"x": 1}
        s["structural_segments"] = [{"start": 0}]
        out = to_export_sample(s)
        for dropped in ("locked", "spatial_tokens", "structural_segments",
                        "stem_paths", "chunk_paths", "labeled", "raw_lyrics"):
            assert dropped not in out

    def test_training_fields_survive(self, dataset):
        out = to_export_dataset(dataset)["samples"][0]
        for key in ("genre", "caption", "lyrics", "bpm", "keyscale",
                    "timesignature", "language", "instrumental", "file_name"):
            assert key in out

    def test_mapping_table_is_the_documented_one(self):
        assert EXPORT_FIELD_MAP["audio_path"] == "file_name"
        assert EXPORT_FIELD_MAP["is_instrumental"] == "instrumental"


class TestNormalizeDataset:
    def test_backfills_missing_sample_fields(self):
        ds = {"metadata": {}, "samples": [{"id": "x", "filename": "x.mp3"}]}
        normalize_dataset(ds)
        for key in SAMPLE_DEFAULTS:
            assert key in ds["samples"][0]

    def test_backfills_missing_metadata_fields(self):
        ds = {"metadata": {}, "samples": []}
        normalize_dataset(ds)
        for key in METADATA_DEFAULTS:
            assert key in ds["metadata"]

    def test_never_overwrites_existing_values(self):
        ds = {"metadata": {"name": "keepme"}, "samples": [{"genre": "Rock"}]}
        normalize_dataset(ds)
        assert ds["metadata"]["name"] == "keepme"
        assert ds["samples"][0]["genre"] == "Rock"

    def test_legacy_instrumental_mode_is_translated(self):
        ds = {
            "metadata": {"instrumental_mode": "all_instrumental"},
            "samples": [{"is_instrumental": True}],
        }
        normalize_dataset(ds)
        assert ds["metadata"]["all_instrumental"] is True

    def test_created_at_is_stamped(self):
        ds = {"metadata": {}, "samples": []}
        normalize_dataset(ds)
        assert ds["metadata"]["created_at"]

    def test_num_samples_is_recomputed(self):
        ds = {"metadata": {"num_samples": 99}, "samples": [{"id": "a"}, {"id": "b"}]}
        normalize_dataset(ds)
        assert ds["metadata"]["num_samples"] == 2

    def test_lyrics_make_a_track_non_instrumental(self):
        # A track with lyrics cannot be instrumental; the inconsistency would
        # confuse the loader's text conditioning.
        ds = {"metadata": {}, "samples": [{"lyrics": "la la", "is_instrumental": True}]}
        normalize_dataset(ds)
        assert ds["samples"][0]["is_instrumental"] is False

    def test_tolerates_non_dict_input(self):
        assert normalize_dataset(None) is None
        assert normalize_dataset("nope") == "nope"


class TestConstructors:
    def test_new_sample_mutable_defaults_are_not_shared(self):
        a, b = new_sample(), new_sample()
        a["structural_segments"].append({"x": 1})
        a["stem_paths"]["v"] = "p"
        assert b["structural_segments"] == []
        assert b["stem_paths"] == {}

    def test_new_dataset_has_metadata_and_samples(self):
        ds = new_dataset(name="x")
        assert ds["metadata"]["name"] == "x"
        assert ds["samples"] == []
        assert ds["metadata"]["created_at"]
