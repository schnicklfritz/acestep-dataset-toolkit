"""ACE-Step export mapping and schema normalization.

The exported shape is load-bearing: ACE-Step 1.5 reads ``audio_path``,
``filename``, ``is_instrumental``, ``duration``, ``prompt_override`` and a
top-level ``metadata`` block, so these are the highest-value assertions in the
suite. They are pinned against the trainer source rather than a doc -- see the
mapping comment in ``modules/dataset_schema.py`` for the file:line evidence.
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
    """The exported JSON is ACE-Step 1.5's canonical sample shape.

    The internal names ARE canonical, so the export must pass them through
    untouched. The regression these guard: an earlier revision renamed them to
    ``file_name`` / ``instrumental`` (ACE-Step *v1* names, which the v2 trainer
    does not read) and turned ``prompt_override`` into a bool.
    """

    def test_audio_path_survives_verbatim(self):
        s = new_sample(filename="a.mp3", audio_path="./songs/a.mp3")
        assert to_export_sample(s)["audio_path"] == "./songs/a.mp3"

    def test_audio_path_falls_back_to_filename(self):
        s = new_sample(filename="a.mp3", audio_path="")
        assert to_export_sample(s)["audio_path"] == "a.mp3"

    def test_filename_is_a_bare_basename(self):
        s = new_sample(filename="", audio_path="./songs/deep/a.mp3")
        assert to_export_sample(s)["filename"] == "a.mp3"

    def test_is_instrumental_keeps_its_name_and_type(self):
        out = to_export_sample(new_sample(is_instrumental=True))
        assert out["is_instrumental"] is True
        assert "instrumental" not in out

    @pytest.mark.parametrize("v1_name", ["file_name", "instrumental"])
    def test_v1_field_names_never_appear(self, v1_name):
        out = to_export_sample(new_sample(filename="a.mp3", audio_path="a.mp3"))
        assert v1_name not in out

    def test_duration_is_exported(self):
        assert to_export_sample(new_sample(duration=197))["duration"] == 197

    def test_duration_defaults_to_zero(self):
        s = new_sample()
        del s["duration"]
        assert to_export_sample(s)["duration"] == 0

    def test_prompt_override_passes_the_two_canonical_strings(self):
        for value in ("caption", "genre"):
            out = to_export_sample(new_sample(prompt_override=value))
            assert out["prompt_override"] == value

    @pytest.mark.parametrize("legacy", [False, True, "yes", 0, 1, ""])
    def test_prompt_override_non_canonical_becomes_null(self, legacy):
        # preprocess_prompt.py:47-52 compares against "caption"/"genre" only;
        # every other value -- including the bool an earlier revision emitted --
        # means "no override".
        out = to_export_sample(new_sample(prompt_override=legacy))
        assert out["prompt_override"] is None

    def test_app_internal_bookkeeping_is_dropped(self):
        s = new_sample(filename="a.mp3", audio_path="a.mp3")
        s["locked"] = True
        s["spatial_tokens"] = {"x": 1}
        s["structural_segments"] = [{"start": 0}]
        out = to_export_sample(s)
        for dropped in ("locked", "spatial_tokens", "structural_segments",
                        "stem_paths", "chunk_paths", "virtual"):
            assert dropped not in out

    def test_training_fields_survive(self, dataset):
        out = to_export_dataset(dataset)["samples"][0]
        for key in ("id", "audio_path", "filename", "caption", "genre", "lyrics",
                    "raw_lyrics", "formatted_lyrics", "bpm", "keyscale",
                    "timesignature", "duration", "language", "is_instrumental",
                    "custom_tag", "labeled", "prompt_override"):
            assert key in out

    def test_mapping_table_is_identity(self):
        # Kept as a seam; nothing needs renaming against ACE-Step 1.5.
        assert EXPORT_FIELD_MAP == {}


class TestExportMetadataBlock:
    """The dataset-level ``metadata`` block must actually reach the JSON.

    ACE-Step reads ``custom_tag`` / ``tag_position`` / ``genre_ratio`` from it
    (``preprocess_discovery.py:145-177``); emitting ``"metadata": {}`` -- which
    ``exporters.export_json`` used to hard-code -- silently dropped the trigger
    word and the caption/genre policy for every training run.
    """

    def test_metadata_is_carried_through(self, dataset):
        dataset["metadata"]["custom_tag"] = "mytrigger"
        dataset["metadata"]["genre_ratio"] = 0
        meta = to_export_dataset(dataset)["metadata"]
        assert meta["custom_tag"] == "mytrigger"
        assert meta["genre_ratio"] == 0
        assert meta["tag_position"] == "prepend"
        assert meta["name"] == "test_dataset"

    def test_legacy_metadata_key_is_not_exported(self, dataset):
        assert "instrumental_mode" in dataset["metadata"]
        assert "instrumental_mode" not in to_export_dataset(dataset)["metadata"]

    def test_missing_metadata_block_does_not_crash(self):
        # A dataset dict with no metadata still yields a valid, self-describing
        # block rather than raising or emitting None.
        assert to_export_dataset({"samples": []})["metadata"] == {"num_samples": 0}

    def test_num_samples_matches_the_exported_samples(self, dataset):
        # The manifest must describe itself even if the caller never ran
        # normalize_dataset (which is what normally recomputes this).
        dataset["metadata"]["num_samples"] = 0
        doc = to_export_dataset(dataset)
        assert doc["metadata"]["num_samples"] == len(doc["samples"]) == 2

    def test_canonical_metadata_key_order_is_preserved(self, dataset):
        # The doc's "Full ACE-Step format" example lists metadata in this order;
        # a consumer diffing the two should see no reordering.
        keys = list(to_export_dataset(dataset)["metadata"])
        assert keys == ["name", "custom_tag", "tag_position", "created_at",
                        "num_samples", "all_instrumental", "genre_ratio"]

    def test_canonical_sample_key_order_is_preserved(self, dataset):
        keys = list(to_export_dataset(dataset)["samples"][0])
        assert keys == ["id", "audio_path", "filename", "caption", "genre",
                        "lyrics", "raw_lyrics", "formatted_lyrics", "bpm",
                        "keyscale", "timesignature", "duration", "language",
                        "is_instrumental", "custom_tag", "labeled",
                        "prompt_override"]


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
