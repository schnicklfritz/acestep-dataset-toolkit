"""Exporter output: the file a training run actually reads."""
import json
from pathlib import Path

import pytest

from modules.dataset_schema import normalize_dataset
from modules.exporters import export_csv, export_json, export_jsonl, split_dataset


class TestExportJson:
    @pytest.fixture
    def written(self, dataset, tmp_path):
        path = tmp_path / "train.json"
        export_json(dataset, str(path))
        return json.loads(path.read_text(encoding="utf-8"))

    def test_uses_ace_step_1_5_field_names(self, written):
        s = written["samples"][0]
        for key in ("id", "audio_path", "filename", "caption", "genre",
                    "lyrics", "raw_lyrics", "formatted_lyrics", "bpm",
                    "keyscale", "timesignature", "duration", "language",
                    "is_instrumental", "custom_tag", "labeled",
                    "prompt_override"):
            assert key in s

    @pytest.mark.parametrize("v1_name", ["file_name", "instrumental"])
    def test_ace_step_v1_names_never_leak(self, written, v1_name):
        # file_name/instrumental are ACE-Step v1 names; the v2 trainer reads
        # audio_path/filename and is_instrumental instead.
        assert v1_name not in written["samples"][0]

    def test_is_instrumental_flag_is_a_bool(self, written):
        flags = [s["is_instrumental"] for s in written["samples"]]
        assert all(isinstance(f, bool) for f in flags)
        assert flags == [False, True]

    def test_trainable_fields_present(self, written):
        for key in ("genre", "caption", "lyrics", "bpm", "keyscale",
                    "timesignature", "duration", "language",
                    "prompt_override"):
            assert key in written["samples"][0]

    def test_prompt_override_is_null_not_a_bool(self, written):
        # The trainer compares prompt_override against the strings
        # "caption"/"genre"; null (not false) is the canonical "no override".
        assert all(s["prompt_override"] is None for s in written["samples"])

    def test_metadata_block_reaches_the_file(self, dataset, tmp_path):
        """Regression: export_json hard-coded {"metadata": {}}.

        ACE-Step reads custom_tag / tag_position / genre_ratio from it, so an
        empty block meant no trigger word and no caption/genre policy ever
        reached a training run.
        """
        dataset["metadata"]["custom_tag"] = "mytrigger"
        path = tmp_path / "meta.json"
        export_json(dataset, str(path))
        meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        assert meta["custom_tag"] == "mytrigger"
        assert meta["tag_position"] == "prepend"
        assert meta["genre_ratio"] == 0
        assert meta["all_instrumental"] is False

    def test_audio_path_is_relative_or_bare_not_renamed(self, written):
        assert written["samples"][0]["audio_path"] == "./songs/song_a.mp3"
        assert written["samples"][0]["filename"] == "song_a.mp3"

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
        # ``audio_path`` is emitted verbatim, so the real track appears under
        # its path. What matters here is that the placeholder is ABSENT.
        names = [s["audio_path"] for s in manifest["samples"]]
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

    def test_the_worker_says_how_many_placeholders_it_skipped(self, qapp, tmp_path):
        """The drop is deliberate, so it must not look like data loss.

        A user who counted 12 tracks in the app and then counted 11 rows in
        ``dataset.json`` has no way to tell a filter from a bug unless the export
        reports it. The completion message is the surface the dialog shows.
        """
        from workers.export import ExportWorker

        worker = ExportWorker(self._dataset(), {"dest_dir": str(tmp_path), "json": True})
        messages = []
        worker.finished_ok.connect(messages.append)
        worker.run()

        assert len(messages) == 1
        assert "JSON" in messages[0]
        assert "1 virtual placeholder(s) excluded" in messages[0]

    def test_a_dataset_without_placeholders_gets_no_note(self, qapp, tmp_path):
        """The note has to mean something: no placeholders, nothing said."""
        from modules.dataset_schema import new_dataset, new_sample
        from workers.export import ExportWorker

        ds = new_dataset(name="plain")
        ds["samples"] = [new_sample(filename="real.wav", audio_path="./real.wav")]
        worker = ExportWorker(ds, {"dest_dir": str(tmp_path), "json": True})
        messages = []
        worker.finished_ok.connect(messages.append)
        worker.run()

        assert messages == ["JSON"]


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


FIXTURES = Path(__file__).parent / "fixtures"


class TestOfficialManifestRoundTrip:
    """The owner's own ACE-Step 1.5XL manifest, exported back character-for-character.

    The fixture is the owner's paste, verbatim, under ``tests/fixtures/``. Loading
    it as the app's in-memory dataset and exporting it must reproduce that file
    exactly — field order, nesting, value types — apart from ``num_samples``. The
    paste reads ``num_samples: 2`` beside a single ``samples`` entry, and both the
    export and the trainer count the array, so 2 becomes 1. Nothing else may move:
    a renamed or retyped field here is a training run that silently ignores it.
    """

    @staticmethod
    def _golden():
        return (FIXTURES / "official_acestep_1_5xl.json").read_text(encoding="utf-8")

    def test_the_fixture_is_the_official_shape(self):
        # A guard on the guard: if the fixture stops carrying all seven metadata
        # keys and all seventeen sample keys, the comparison below proves less
        # than it claims.
        doc = json.loads(self._golden())
        assert set(doc["metadata"]) == {
            "name", "custom_tag", "tag_position", "created_at", "num_samples",
            "all_instrumental", "genre_ratio",
        }
        assert set(doc["samples"][0]) == {
            "id", "audio_path", "filename", "caption", "genre", "lyrics",
            "raw_lyrics", "formatted_lyrics", "bpm", "keyscale", "timesignature",
            "duration", "language", "is_instrumental", "custom_tag", "labeled",
            "prompt_override",
        }

    def test_export_reproduces_the_official_file(self, tmp_path):
        golden = self._golden()
        out = tmp_path / "dataset.json"

        export_json(normalize_dataset(json.loads(golden)), str(out))

        written = out.read_text(encoding="utf-8")
        # The one permitted difference, plus the trailing newline the paste has
        # and ``json.dump`` does not write.
        expected = golden.replace('"num_samples": 2', '"num_samples": 1').rstrip("\n")
        assert written.rstrip("\n") == expected


class TestProjectFileVersusTrainingManifest:
    """The two documents differ on purpose, and the difference is pinned here.

    ``save_dataset`` writes ``self.dataset`` as it stands: internal bookkeeping
    (``structural_segments``, ``spatial_tokens``, ``stem_paths``, ``chunk_paths``,
    the legacy ``instrumental_mode``) and ``virtual`` placeholders included,
    because the app needs all of it back on the next load. The training manifest
    is the EXPORT, and it carries neither. Stripping the internal keys from the
    saved file would break the app; exporting them would hand the trainer a row it
    cannot resolve. What must not happen is the difference being invisible, so it
    is asserted rather than described.
    """

    INTERNAL_SAMPLE_KEYS = ("virtual", "locked", "structural_segments",
                            "spatial_tokens", "stem_paths", "chunk_paths")

    def _app_shaped(self):
        from modules.dataset_schema import new_dataset, new_sample

        ds = new_dataset(name="project_file")
        real = new_sample(filename="real.wav", audio_path="./real.wav",
                          genre="Rock", caption="real")
        real["structural_segments"] = [{"label": "intro"}]
        real["stem_paths"] = {"vocals": "./stems/real_vocals.wav"}
        real["chunk_paths"] = ["./chunks/real_01.wav"]
        ds["samples"] = [real, new_sample(filename="virtual_01_doom.wav",
                                          genre="Doom", virtual=True)]
        return ds

    def test_the_export_carries_no_internal_key_and_no_placeholder(self, tmp_path):
        out = tmp_path / "dataset.json"
        export_json(self._app_shaped(), str(out))
        doc = json.loads(out.read_text(encoding="utf-8"))

        assert len(doc["samples"]) == 1
        for key in self.INTERNAL_SAMPLE_KEYS:
            assert key not in doc["samples"][0]
        assert "instrumental_mode" not in doc["metadata"]
        # Counted the way the trainer counts: the rows it can resolve, not the
        # rows the project file holds.
        assert doc["metadata"]["num_samples"] == 1

    def test_a_saved_project_file_reloads_with_its_internal_keys(self, tmp_path):
        """The other half of the contract — and why the fix is NOT "strip them".

        The app's own file is what the DSP stages, the stem features and the gap
        audit read back, so the internal keys have to survive the round trip. The
        serialize step is the one ``save_dataset`` uses (``json.dumps`` of the live
        dataset) and the reload is ``normalize_dataset`` — the function every load
        path puts a parsed document through, so a build that stripped these fields
        "to match the manifest" is caught here rather than by a broken dataset at
        runtime.
        """
        ds = self._app_shaped()
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(ds, indent=2), encoding="utf-8")

        reloaded = normalize_dataset(json.loads(path.read_text(encoding="utf-8")))

        s = reloaded["samples"][0]
        assert s["stem_paths"] == {"vocals": "./stems/real_vocals.wav"}
        assert s["structural_segments"] == [{"label": "intro"}]
        assert s["chunk_paths"] == ["./chunks/real_01.wav"]
        assert reloaded["samples"][1]["virtual"] is True
        # The project file counts its own rows; the manifest counts the rows the
        # trainer can use. Both counts are correct for their own document.
        assert reloaded["metadata"]["num_samples"] == 2
