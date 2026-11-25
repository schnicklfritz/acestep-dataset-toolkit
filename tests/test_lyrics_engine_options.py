"""The lyrics-engine choice offers only the two engines that still exist.

Kaggle transcription (``workers/kaggle_lyrics.py`` + ``kernels/lyrics_kernel.py``)
is gone, so the combo must offer WhisperX and Gemini and nothing else. Two
things must keep working, and both are silent if they break:

* a saved ``gemini`` still selects the Gemini item (the settings tab matches by
  ``startswith`` on the item text, and the index mapping is positional);
* a saved ``kaggle`` / ``acestep_transcriber`` from an older settings.json
  degrades to WhisperX instead of selecting nothing at all.
"""
import json

import pytest

pytest.importorskip("PySide6")

ITEMS = ["whisperx (local)", "gemini"]


def _build(qapp, tmp_path, monkeypatch, stored=None):
    import config
    import modules.config_store as cs

    settings = tmp_path / "settings.json"
    if stored is not None:
        settings.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.setattr(cs, "SETTINGS_PATH", settings)
    import dataset_manager

    w = dataset_manager.DatasetManager()
    qapp.processEvents()
    return w


@pytest.fixture()
def manager(qapp, tmp_path, monkeypatch):
    w = _build(qapp, tmp_path, monkeypatch)
    yield w
    w.hide()


def test_the_combo_offers_only_the_surviving_engines(manager):
    combo = manager.lyrics_engine_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ITEMS


def test_a_gemini_choice_is_saved_and_kept(manager):
    combo = manager.lyrics_engine_combo
    combo.setCurrentText("gemini")
    manager.save_pipeline_defaults()
    assert manager.config["lyrics_engine"] == "gemini"


def test_a_whisperx_choice_is_saved(manager):
    combo = manager.lyrics_engine_combo
    combo.setCurrentText("gemini")
    combo.setCurrentText("whisperx (local)")
    manager.save_pipeline_defaults()
    assert manager.config["lyrics_engine"] == "whisperx"


@pytest.mark.parametrize("stored", ["kaggle", "acestep_transcriber"])
def test_an_older_kaggle_value_degrades_to_whisperx(qapp, tmp_path, monkeypatch, stored):
    w = _build(qapp, tmp_path, monkeypatch, {"lyrics_engine": stored})
    try:
        assert w.lyrics_engine_combo.currentText() == "whisperx (local)"
    finally:
        w.hide()
