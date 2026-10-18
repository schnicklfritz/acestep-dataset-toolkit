"""The unified Lyrics panel — one surface for the tidy rules and the budget.

WHY THIS EXISTS
---------------
The tidy options (``ui.lyrics_tab``), the three structure-tag rules
(``ui.tidy_panel``), the token-budget line and a separate Lyrics Studio window
used to be reachable from four unrelated places. The Studio window is gone; the
panel now has two inner tabs and a footer that shows the budget and a one-line
summary of the rules a tidy pass will apply.

What is pinned here:

  * the panel has its TWO inner tabs, reachable BY NAME;
  * both tidy surfaces are inside it, and each is built exactly ONCE — calling
    ``build_lyrics_panel`` *and* ``build_lyrics_tab`` would leave two sets of
    checkboxes where only one is what the pass reads;
  * the footer's budget line is a VIEW of the same measurement the waveform's
    line shows — ``update_lyric_limit_label`` writes both from one ``measure()``
    result, so the two cannot drift;
  * the "rules in effect" summary stays in step with the checkboxes;
  * the token-limits help is reachable from the panel.
"""
import pytest

pytest.importorskip("PySide6")

from ui.lyrics_panel import PANEL_TABS  # noqa: E402


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One window for the whole module (building it applies the theme)."""
    import config
    import modules.config_store as cs

    settings = tmp_path_factory.mktemp("settings") / "settings.json"
    saved = (config.SETTINGS_PATH, cs.SETTINGS_PATH)
    config.SETTINGS_PATH = settings
    cs.SETTINGS_PATH = settings
    try:
        import dataset_manager

        w = dataset_manager.DatasetManager()
    finally:
        config.SETTINGS_PATH, cs.SETTINGS_PATH = saved
    yield w
    w.hide()
    w.deleteLater()


def _tab_titles(manager):
    tabs = manager.lyrics_panel_tabs
    return [tabs.tabText(i) for i in range(tabs.count())]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_the_panel_has_its_two_inner_tabs(manager):
    assert _tab_titles(manager) == list(PANEL_TABS)


def test_the_panel_is_stored_on_the_manager(manager):
    """Built on the Lyrics page, and named — not just added and forgotten."""
    from PySide6.QtWidgets import QWidget

    assert isinstance(manager.lyrics_panel, QWidget)
    assert manager.lyrics_panel.isAncestorOf(manager.lyrics_panel_tabs)
    lyrics_page = manager._shell_extra_pages[0]
    assert lyrics_page.isAncestorOf(manager.lyrics_panel)


def test_both_tidy_surfaces_are_inside_the_one_panel(manager):
    """The contraction rules (tab 0) and the tag rules (tab 1) are two tabs of
    one panel instead of two different tool tabs."""
    panel = manager.lyrics_panel
    assert panel.isAncestorOf(manager.lyrics_ing_check)      # ui/lyrics_tab.py
    assert panel.isAncestorOf(manager.lyrics_lines_check)    # ui/lyrics_tab.py
    assert panel.isAncestorOf(manager.lyrics_punct_check)    # ui/lyrics_tab.py
    assert panel.isAncestorOf(manager.lyrics_quotes_check)   # ui/tidy_panel.py


# ---------------------------------------------------------------------------
# The budget line and the rules summary
# ---------------------------------------------------------------------------

def test_the_panel_has_its_own_budget_line(manager):
    """It is a separate label from the waveform's — the waveform one is created
    in ``init_ui`` long AFTER the panel is built, so adopting it would depend on
    construction order (and moving it would take the counter out of the panel
    you type the lyrics into)."""
    assert manager.lyrics_panel.isAncestorOf(manager.lyrics_panel_limit_label)
    assert manager.lyrics_panel_limit_label is not manager.lyrics_limit_label


def test_both_budget_lines_are_written_from_one_measurement(manager):
    """One writer, two views: ``update_lyric_limit_label`` drives both, so they
    cannot drift the way two independent counters would."""
    manager.update_lyric_limit_label(lyrics_text="hello there", caption_text="")
    text = manager.lyrics_panel_limit_label.text()
    assert "tokens" in text
    assert "Lyrics" in text
    assert text == manager.lyrics_limit_label.text()


def test_the_summary_names_the_rules_the_tidy_button_will_apply(manager):
    from ui.lyrics_panel import refresh_panel_summary

    manager.lyrics_lines_check.setChecked(True)
    refresh_panel_summary(manager)
    assert "line starts" in manager.lyrics_panel_summary.text()


def test_switching_tabs_refreshes_the_summary(manager):
    """The options live on tab 0 and the summary in the footer, so a stale line
    would describe rules the user has since changed."""
    tabs = manager.lyrics_panel_tabs
    manager.lyrics_lines_check.setChecked(True)
    tabs.setCurrentIndex(0)
    manager.lyrics_lines_check.setChecked(False)
    tabs.setCurrentIndex(_tab_titles(manager).index(PANEL_TABS[1]))
    assert "line starts" not in manager.lyrics_panel_summary.text()
    manager.lyrics_lines_check.setChecked(True)
    tabs.setCurrentIndex(0)


def test_the_token_limits_help_is_on_the_panel(manager, monkeypatch):
    calls = []
    import ui.lyric_limit_notice as notice

    monkeypatch.setattr(notice, "acknowledge_lyrics_limit",
                        lambda mgr: calls.append(mgr))
    manager.lyrics_panel_limits_btn.click()
    assert calls == [manager]


def test_the_lrc_export_survives_on_the_lyrics_page(manager, monkeypatch):
    """The Studio's one standalone file tool is still reachable: the Lyrics tab
    exports the SAVED lyrics of the selected track, not a half-typed buffer."""
    sample = {"filename": "a.mp3", "formatted_lyrics": "one two"}
    monkeypatch.setattr(manager, "get_selected_sample", lambda: sample)
    calls = []
    monkeypatch.setattr(manager, "_export_lyrics",
                        lambda s, t: calls.append((s, t)))
    manager.lyrics_export_lrc_btn.click()
    assert calls == [(sample, "one two")]

