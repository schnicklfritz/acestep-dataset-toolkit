"""MVSEP separation survives the removal of the Kaggle structural pipeline.

``stem_source`` (Import existing stems / MVSEP / Kaggle Demucs) lived on the
Structural page and went with it. What must still work is the MVSEP path
itself: the dialog that separates a track through mvsep.com keeps an entry
point in Settings, and ``StemSeparator`` still talks to the MVSEP API.
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


def test_the_mvsep_dialog_keeps_a_button(manager):
    """It used to be opened from the Structural page; it must still exist."""
    assert manager.mvsep_gui_button.text() == "🎛 Configure / Test MVSEP"
    assert manager.mvsep_gui_button.receivers("2clicked(bool)") >= 1


def test_the_button_opens_the_mvsep_dialog(manager, monkeypatch):
    opened = []
    manager.mvsep_gui_button.clicked.disconnect()
    monkeypatch.setattr(manager, "open_mvsep_dialog", lambda: opened.append(True))
    manager.mvsep_gui_button.clicked.connect(manager.open_mvsep_dialog)
    manager.mvsep_gui_button.click()
    assert opened == [True]


def test_the_structural_page_and_its_stem_ladder_are_gone(manager):
    for name in ("struct_stem_combo", "run_struct_btn"):
        assert not hasattr(manager, name), f"{name} survived the page removal"
    assert not hasattr(type(manager), "init_structural_tab")


def test_stem_separator_keeps_the_configured_mvsep_token():
    import stem_separator

    assert stem_separator.StemSeparator({"mvsep_api_key": " tok "}).api_token == "tok"
    assert stem_separator.StemSeparator({}).api_token == ""


def test_stem_separator_still_looks_up_the_live_mvsep_algorithms(monkeypatch):
    import stem_separator

    monkeypatch.setattr(stem_separator, "get_algorithms", lambda: ({1: "MDX23C"}, []))
    separator = stem_separator.StemSeparator({})
    assert separator.refresh_algorithms() == {"1": "MDX23C"}
