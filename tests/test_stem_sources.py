"""Stem sources after Kaggle Demucs was removed: MVSEP and import only.

``stem_source`` used to be one of ``kaggle_demucs`` / ``mvsep`` / ``import``.
With the Kaggle worker gone the pipeline must still separate through MVSEP and
must still pick up stems that already sit beside the audio, and the UI must not
offer a third source it can no longer honour.
"""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture()
def manager(qapp, tmp_path, monkeypatch):
    import config
    import modules.config_store as cs

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(cs, "SETTINGS_PATH", tmp_path / "settings.json")
    import dataset_manager

    w = dataset_manager.DatasetManager()
    yield w
    w.hide()


def _worker(tmp_path, source, stems=()):
    from workers.structural import StructuralPipelineWorker

    audio = tmp_path / "Song.wav"
    audio.write_bytes(b"not really audio")
    for name in stems:
        (tmp_path / f"Song_{name}.wav").write_bytes(b"stem")
    return StructuralPipelineWorker("track-1", str(audio), {}, {"stem_source": source})


def test_the_combo_offers_only_mvsep_and_import(manager):
    combo = manager.struct_stem_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Import existing stems", "Separate via MVSEP"]


def test_import_existing_stems_still_finds_them_beside_the_audio(qapp, tmp_path):
    worker = _worker(tmp_path, "import", stems=("vocals", "drums"))
    found = worker._find_imported_stems(str(tmp_path / "Song.wav"))
    assert set(found) == {"vocals", "drums"}
    assert found["vocals"].endswith("Song_vocals.wav")


def test_the_import_source_never_reaches_a_service(qapp, tmp_path):
    worker = _worker(tmp_path, "import", stems=("vocals",))
    assert set(worker._call_mvsep(str(tmp_path / "Song.wav"))) == {"vocals"}


def test_there_is_no_kaggle_stem_path_left():
    from workers.structural import StructuralPipelineWorker

    assert not hasattr(StructuralPipelineWorker, "_call_kaggle_stems")
