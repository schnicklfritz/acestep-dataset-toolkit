"""The unified Lyrics panel — one surface for four formerly separate places.

WHY THIS EXISTS
---------------
The tidy options (``ui.lyrics_tab``), the three structure-tag rules
(``ui.tidy_panel``), the token-budget line (``manager.lyrics_limit_label``) and
the editor window (``ui.lyrics_studio``) were reachable from four unrelated
places: the Lyrics tool tab, the bottom of the "Tags & checks" page, a tooltip
under the waveform, and two buttons on the Lyrics tab. You could not see the
rule you were about to apply at the same time as the rule that decided where
its output went.

What is pinned here:

  * the panel has its three inner tabs, reachable BY NAME;
  * both tidy surfaces are inside it, and each is built exactly ONCE — calling
    ``build_lyrics_panel`` *and* ``build_lyrics_tab`` would leave two sets of
    checkboxes where only one is what the pass reads;
  * the budget line on the Studio tab is a VIEW of the same measurement the
    waveform's line shows — ``update_lyric_limit_label`` writes both from one
    ``measure()`` result, so the two cannot drift;
  * the Studio button routes through ``manager.open_lyrics_editor`` (the seam
    ``.agent_notes`` pins by name), and still works on a manager that has no
    such handler;
  * switching tabs refreshes the "rules in effect" summary, which is the only
    thing that tells you what the Studio's tidy button will do.
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


def _studio_page(manager):
    tabs = manager.lyrics_panel_tabs
    return tabs.widget(_tab_titles(manager).index(PANEL_TABS[2]))


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_the_panel_has_its_three_inner_tabs(manager):
    assert _tab_titles(manager) == list(PANEL_TABS)


def test_the_panel_is_stored_on_the_manager(manager):
    """Built on the Lyrics page, and named — not just added and forgotten."""
    from PySide6.QtWidgets import QWidget

    assert isinstance(manager.lyrics_panel, QWidget)
    assert manager.lyrics_panel.isAncestorOf(manager.lyrics_panel_tabs)
    lyrics_page = manager._shell_extra_pages[0]
    assert lyrics_page.isAncestorOf(manager.lyrics_panel)


def test_both_tidy_surfaces_are_inside_the_one_panel(manager):
    """The contraction rules (tab 0) and the tag rules (tab 1) are now two tabs
    of one panel instead of two different tool tabs."""
    panel = manager.lyrics_panel
    assert panel.isAncestorOf(manager.lyrics_ing_check)      # ui/lyrics_tab.py
    assert panel.isAncestorOf(manager.lyrics_lines_check)    # ui/lyrics_tab.py
    assert panel.isAncestorOf(manager.lyrics_quotes_check)   # ui/tidy_panel.py
    assert panel.isAncestorOf(manager.lyrics_tagdrop_check)  # ui/tidy_panel.py


def test_each_surface_is_built_exactly_once(manager):
    """The "BUILD IT ONCE" contract, measured rather than trusted.

    ``build_lyrics_panel`` calls ``build_lyrics_tab`` and ``build_tidy_panel``.
    If a call site ALSO called either of them, the second run would overwrite
    ``manager.<name>`` while the first set stayed on screen and connected — two
    visible copies, only one of them live. Counting the groups is the cheap way
    to catch that.
    """
    from PySide6.QtWidgets import QGroupBox

    titles = [g.title() for g in manager.findChildren(QGroupBox)]
    assert titles.count("Preview (selected track)") == 1
    assert sum(1 for t in titles if "Lyrics Tidy" in t) == 1


# ---------------------------------------------------------------------------
# The budget line: two views, ONE writer
# ---------------------------------------------------------------------------

def test_the_panel_has_its_own_budget_line(manager):
    """It is a separate label from the waveform's — the waveform one is created
    in ``init_ui`` long AFTER the panel is built, so adopting it would depend on
    construction order (and moving it would take the counter out of the panel
    you type the lyrics into)."""
    studio = _studio_page(manager)
    assert studio.isAncestorOf(manager.lyrics_panel_limit_label)
    assert manager.lyrics_panel_limit_label is not manager.lyrics_limit_label


def test_both_budget_lines_are_written_from_one_measurement(manager):
    """One writer, two views: ``update_lyric_limit_label`` drives both, so they
    cannot drift the way two independent counters would."""
    manager.update_lyric_limit_label(lyrics_text="hello there", caption_text="")
    text = manager.lyrics_panel_limit_label.text()
    assert "tokens" in text
    assert "Lyrics" in text
    assert text == manager.lyrics_limit_label.text()


# ---------------------------------------------------------------------------
# The Studio button routes through the manager's own entry point
# ---------------------------------------------------------------------------

def test_the_studio_button_calls_the_managers_entry_point(manager, monkeypatch):
    """``manager.open_lyrics_editor`` is the seam the rest of the app uses (the
    Lyrics tab's older button, and ``.agent_notes`` by name). The panel must
    route through it rather than opening a window of its own, or the two entry
    points could diverge."""
    calls = []
    monkeypatch.setattr(manager, "open_lyrics_editor", lambda: calls.append(1))
    manager.lyrics_panel_studio_btn.click()
    assert calls == [1]


def test_the_studio_button_works_without_a_manager_handler(monkeypatch):
    """A test/tooling manager has no ``open_lyrics_editor``; the lazy import is
    the fallback, and it must not raise."""
    from PySide6.QtWidgets import QWidget

    import ui.lyrics_studio as studio
    import ui.lyrics_panel as panel

    calls = []
    monkeypatch.setattr(studio, "open_lyrics_studio",
                        lambda mgr: calls.append(mgr))

    class _Stub:
        def __init__(self):
            self.config = {}

    stub = _Stub()
    host = QWidget()
    try:
        panel.build_lyrics_panel(stub, host)
        stub.lyrics_panel_studio_btn.click()
    finally:
        host.deleteLater()
    assert calls == [stub]


# ---------------------------------------------------------------------------
# The "rules in effect" summary
# ---------------------------------------------------------------------------

def test_the_summary_names_the_rules_the_tidy_button_will_apply(manager):
    from ui.lyrics_panel import refresh_panel_summary

    manager.lyrics_lines_check.setChecked(True)
    refresh_panel_summary(manager)
    assert "line starts" in manager.lyrics_panel_summary.text()


def test_switching_tabs_refreshes_the_summary(manager):
    """The options live on tab 0 and the summary on tab 2, so a stale line would
    describe rules the user has since changed."""
    tabs = manager.lyrics_panel_tabs
    manager.lyrics_lines_check.setChecked(True)
    tabs.setCurrentIndex(0)
    manager.lyrics_lines_check.setChecked(False)
    tabs.setCurrentIndex(_tab_titles(manager).index(PANEL_TABS[2]))
    assert "line starts" not in manager.lyrics_panel_summary.text()
    manager.lyrics_lines_check.setChecked(True)
    tabs.setCurrentIndex(0)
