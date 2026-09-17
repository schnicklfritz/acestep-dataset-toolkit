"""Exporter output: the file a training run actually reads."""
import json

import pytest

from modules.exporters import export_csv, export_json, export_jsonl, split_dataset


class TestExportJson:
    @pytest.fixture
    def written(self, dataset, tmp_path):
        path = tmp_path / "train.json"
        export_json(dataset["samples"], str(path))
        return json.loads(path.read_text(encoding="utf-8"))

    def test_uses_ace_step_field_names(self, written):
        s = written["samples"][0]
        assert "file_name" in s
        assert "instrumental" in s

    @pytest.mark.parametrize("internal", ["audio_path", "is_instrumental", "filename"])
    def test_no_internal_underscore_names_leak(self, written, internal):
        assert internal not in written["samples"][0]

    def test_instrumental_flag_is_a_bool(self, written):
        flags = [s["instrumental"] for s in written["samples"]]
        assert all(isinstance(f, bool) for f in flags)
        assert flags == [False, True]

    def test_trainable_fields_present(self, written):
        for key in ("genre", "caption", "lyrics", "bpm", "keyscale",
                    "language", "prompt_override"):
            assert key in written["samples"][0]

    def test_dropped_bookkeeping_absent(self, written):
        for key in ("locked", "spatial_tokens", "chunk_paths"):
            assert key not in written["samples"][0]


class TestOtherExporters:
    def test_jsonl_writes_one_object_per_line(self, dataset, tmp_path):
        path = tmp_path / "out.jsonl"
        export_jsonl(dataset["samples"], str(path))
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            json.loads(line)

    def test_csv_has_a_header_and_rows(self, dataset, tmp_path):
        path = tmp_path / "out.csv"
        export_csv(dataset["samples"], str(path))
        rows = path.read_text(encoding="utf-8").splitlines()
        assert rows[0].startswith("id,")
        assert len(rows) == 3

    def test_split_is_deterministic_for_a_seed(self, dataset, tmp_path):
        samples = dataset["samples"] * 5
        a = split_dataset(samples, val_ratio=0.2, seed=42)
        b = split_dataset(samples, val_ratio=0.2, seed=42)
        assert [s["id"] for s in a[0]] == [s["id"] for s in b[0]]

    def test_split_covers_all_samples(self, dataset, tmp_path):
        samples = dataset["samples"] * 5
        train, val = split_dataset(samples, val_ratio=0.2, seed=1)
        assert len(train) + len(val) == len(samples)
