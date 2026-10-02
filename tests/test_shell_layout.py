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
    assert manager.assistant_dock.isVisible()
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

    manager.assistant_dock.hide()
    save_layout(manager)
    manager.assistant_dock.show()
    assert restore_layout(manager)
    assert not manager.assistant_dock.isVisible()


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


def test_assistant_dock_is_stationary(manager):
    """The Assistant is pinned to the right column and cannot be floated.

    Floating is how it used to get dragged off the screen; the feature flag is
    cleared and the allowed areas are the right column only.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget

    flag = QDockWidget.DockWidgetFeature.DockWidgetFloatable
    assert not (manager.assistant_dock.features() & flag)
    assert manager.assistant_dock.allowedAreas() == Qt.RightDockWidgetArea


def test_assistant_dock_collapses_and_restores(manager):
    """The − / ▾ title-bar button tucks the panel away without moving it.

    Collapsing caps the dock to its title bar; restoring lifts the cap. The dock
    itself stays put and stays visible, so it can never be lost.
    """
    dock = manager.assistant_dock
    toggle = dock._collapse_button
    assert toggle is not None

    toggle.click()
    assert dock.maximumHeight() < 100          # capped to the bar
    assert dock.isVisible()                    # tucked away, not closed
    assert not dock.isFloating()               # never moved

    toggle.click()
    assert dock.maximumHeight() > 100000       # cap lifted again


def test_restore_never_leaves_both_docks_hidden(manager, monkeypatch):
    """A state that hides BOTH right-column docks is repaired, not honoured.

    The whole column vanishing leaves nothing to click; the guard shows both
    again. Hiding only ONE dock is a legitimate preference and is left alone
    (``test_layout_round_trip`` covers that).
    """
    import base64

    from ui.shell import restore_layout

    manager.tools_dock.hide()
    manager.assistant_dock.hide()
    # Pretend a state was saved AND restored while both were hidden.
    manager.config["ui_dock_state"] = base64.b64encode(b"state").decode("ascii")
    monkeypatch.setattr(manager, "restoreState", lambda *a, **k: True)
    assert restore_layout(manager) is True
    assert manager.tools_dock.isVisible()
    assert manager.assistant_dock.isVisible()


def test_assistant_panel_defaults_to_the_free_provider(manager):
    assert manager.assistant_provider_combo.currentData() == "groq"
    assert "needs a key" in manager.assistant_provider_state.text()
    assert manager.assistant_free_link.isVisibleTo(manager.assistant_dock)


def test_assistant_provider_choice_is_saved_and_synced(manager, tmp_path):
    import json

    i = manager.assistant_provider_combo.findData("gemini")
    manager.assistant_provider_combo.setCurrentIndex(i)
    assert manager.config["llm_provider_assistant"] == "gemini"
    assert json.loads((tmp_path / "settings.json").read_text())["llm_provider_assistant"] == "gemini"
    assert manager.role_provider_combo["assistant"].currentText() == "gemini"
