"""The dock layout: center track list + editor, Tools and Assistant docks."""
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
    w.resize(1440, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.hide()


def test_docks_and_center(manager):
    assert manager.tools_dock.isVisible()
    assert manager.centralWidget().isAncestorOf(manager.table)
    labels = [manager.tabs.tabText(i).replace("&&", "&") for i in range(manager.tabs.count())]
    assert labels == ["Caption", "Lyrics", "Structure", "Tags & checks"]


def test_settings_is_a_window(manager):
    assert not manager.settings_dialog.isVisible()
    manager.settings_action.trigger()
    assert manager.settings_dialog.isVisible()
    assert manager.settings_dialog.isAncestorOf(manager.theme_combo)


def test_theme_switch_and_save(manager, tmp_path):
    import json

    manager.theme_combo.setCurrentText("Paper Teal")
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["theme_name"] == "Paper Teal"
    assert "k" not in saved  # only the appearance keys were written


def test_layout_round_trip(manager, tmp_path):
    from ui.shell import restore_layout, save_layout

    save_layout(manager)
    assert restore_layout(manager)


def test_bulk_edit_really_collapses(manager):
    from PySide6.QtWidgets import QGroupBox

    grp = [g for g in manager.findChildren(QGroupBox) if g.title() == "Set All Tracks"][0]
    assert not grp.isChecked()
    assert not manager.bulk_apply_btn.isVisible()
    grp.setChecked(True)
    assert manager.bulk_apply_btn.isVisible()


def test_no_tool_page_is_wider_than_its_panel(manager, qapp):
    from PySide6.QtWidgets import QScrollArea, QTabWidget

    for i in range(manager.tabs.count()):
        manager.tabs.setCurrentIndex(i)
        qapp.processEvents()
        page = manager.tabs.widget(i)
        for tabs in [None] + page.findChildren(QTabWidget):
            for j in range(tabs.count() if tabs else 1):
                if tabs:
                    tabs.setCurrentIndex(j)
                    qapp.processEvents()
                areas = ([page] if isinstance(page, QScrollArea) else []) + page.findChildren(QScrollArea)
                for a in areas:
                    if a.isVisible() and a.widget() is not None:
                        assert a.widget().minimumSizeHint().width() <= a.viewport().width(), (i, j)


def test_restore_never_leaves_the_tools_dock_hidden(manager, monkeypatch):
    """A saved state that hides the tools dock is repaired, not honoured.

    If the right column comes back hidden there is nothing left to click, so
    ``restore_layout`` shows the dock again. (The Assistant dock this guard
    used to cover was removed along with the assistant feature.)
    """
    import base64

    from ui.shell import restore_layout

    manager.tools_dock.hide()
    # Pretend a state was saved AND restored while the dock was hidden.
    manager.config["ui_dock_state"] = base64.b64encode(b"state").decode("ascii")
    monkeypatch.setattr(manager, "restoreState", lambda *a, **k: True)
    assert restore_layout(manager) is True
    assert manager.tools_dock.isVisible()

