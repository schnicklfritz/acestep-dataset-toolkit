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


class TestVirtualTracksNeverExport:
    """A concept-only placeholder must not leave the app as data.

    Regression: ``virtual`` samples were filtered nowhere. ``to_export_sample``
    strips the flag but still emits a row, so ``dataset.json``/``.csv``/``.jsonl``
    all contained a sample pointing at a ``virtual_01_*.wav`` with no audio. The
    suite passed because the only assertion checked flag-stripping.
    """

    def _dataset(self):
        from modules.dataset_schema import new_dataset, new_sample

        ds = new_dataset(name="virtual_leak")
        ds["samples"] = [
            new_sample(filename="real.wav", audio_path="./real.wav",
                       genre="Rock", caption="real"),
            new_sample(filename="virtual_01_doom.wav", audio_path="",
                       genre="Doom", virtual=True),
        ]
        return ds

    def test_for_export_drops_virtual_placeholders(self):
        from modules.exporters import for_export

        kept = for_export(self._dataset()["samples"])
        assert [s["filename"] for s in kept] == ["real.wav"]

    def test_worker_writes_no_virtual_row_in_any_format(self, qapp, tmp_path):
        from modules.exporters import for_export
        from workers.export import ExportWorker

        ds = self._dataset()
        # The worker is the boundary that shipped the leak, so drive it directly.
        worker = ExportWorker(ds, {
            "dest_dir": str(tmp_path), "json": True, "csv": True,
            "jsonl": True, "folders": False,
        })
        errors = []
        worker.failed.connect(errors.append)
        worker.run()
        assert errors == []

        manifest = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))
        # ``to_export_sample`` maps ``file_name`` from audio_path (preferred) or
        # filename, so the real track appears under its path. What matters here
        # is that the placeholder is ABSENT, not which name the real one got.
        names = [s["file_name"] for s in manifest["samples"]]
        assert names == ["./real.wav"]
        assert "virtual_01_doom.wav" not in names

        csv_text = (tmp_path / "dataset.csv").read_text(encoding="utf-8")
        assert "virtual" not in csv_text

        jsonl = [json.loads(l) for l in
                 (tmp_path / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        assert [s.get("audio_path") or s["filename"] for s in jsonl] == ["./real.wav"]

        # Sanity: the filter is what did it, not an empty dataset.
        assert len(ds["samples"]) == 2
        assert len(for_export(ds["samples"])) == 1


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
