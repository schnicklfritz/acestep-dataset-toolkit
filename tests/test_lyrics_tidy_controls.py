"""The tidy rules: declared once, visible while you edit, one implementation.

WHY THIS EXISTS
---------------
The owner's report was concrete: the tidy rules were only on the Tools window's
Lyrics tab, inside a ``QScrollArea``, and the Lyrics Studio's "Tidy" button
applied rules you could not see or change from the window you were editing in.

The fix is not a second set of rules. It is:

  * ONE declaration of every rule's wording (``RULE_SPECS``), which the Lyrics tab
    builds its checkboxes from — so the two surfaces cannot describe the same
    rule differently, and a test here proves the tab holds no copy of the text;
  * an always-visible panel in the Studio that is a VIEW of those checkboxes, so
    ticking it in the Studio ticks the tab's box and ``_lyrics_options()`` keeps
    its single source;
  * one tidy pass (``manager.apply_lyrics_tidy``), which the Studio reaches with
    the scope its own picker shows.

What is pinned, and why each would be easy to lose:

  * every rule key is a real ``normalize_lyrics`` option — a typo'd key would
    tick a box that silently does nothing;
  * the Studio's panel is NOT inside a scroll area (the actual complaint), and is
    a direct child of the window's own layout rather than buried in a page;
  * ticking either surface moves the other, in both directions;
  * "Tidy this text" obeys the ticks on screen, measured through the editor;
  * "Apply to dataset" passes the panel's scope, not the Tools tab's combo.
"""
import pytest

pytest.importorskip("PySide6")

import os  # noqa: E402

from ui.lyrics_tidy_controls import (  # noqa: E402
    RULE_SPECS,
    SCOPE_CHOICES,
    TidyRulesPanel,
    build_rule_checkboxes,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One real window for the module (building it applies the theme).

    The binding under test is between the Studio's panel and the Lyrics tab's
    checkboxes, so a stand-in manager cannot exercise it: the widgets have to
    exist.
    """
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


def _studio(manager):
    """A Lyrics Studio window on the manager's selected track."""
    from ui.lyrics_studio import LyricsStudioWindow

    sample = manager.get_selected_sample() or {
        "filename": "a.mp3", "caption": "", "lyrics": "", "formatted_lyrics": "",
    }
    win = LyricsStudioWindow(manager, sample)
    win.show()
    return win


# ---------------------------------------------------------------------------
# One declaration
# ---------------------------------------------------------------------------

def test_every_rule_key_is_a_real_normalize_option():
    """A key that is not an option would tick a box that does nothing."""
    import inspect

    from modules.lyrics_normalizer import normalize_lyrics

    params = set(inspect.signature(normalize_lyrics).parameters)
    missing = [spec["key"] for spec in RULE_SPECS if spec["key"] not in params]
    assert not missing, f"not normalize_lyrics options: {missing}"


def test_the_tabs_wording_is_built_from_this_module():
    """The Lyrics tab must not hold a second copy of any rule's text.

    This is the "cannot drift" claim, measured: if someone re-inlines the label
    on the tab, the two surfaces can start describing the same rule differently,
    and only one of them would be the one the user reads.
    """
    src = open(os.path.join(ROOT, "ui", "lyrics_tab.py"), encoding="utf-8").read()
    assert "build_rule_checkboxes" in src
    for spec in RULE_SPECS:
        assert spec["label"] not in src, spec["label"]


def test_build_rule_checkboxes_names_and_defaults_them(qapp):
    """The attribute names are the app's tidy state; ``_lyrics_options`` reads them."""
    from PySide6.QtWidgets import QFormLayout, QWidget

    class _M:
        pass

    mgr = _M()
    host = QWidget()
    form = QFormLayout(host)
    boxes = build_rule_checkboxes(mgr, form)
    try:
        for spec in RULE_SPECS:
            widget = getattr(mgr, spec["attr"])
            assert boxes[spec["key"]] is widget
            assert widget.isChecked() is spec["default"]
    finally:
        host.deleteLater()


# ---------------------------------------------------------------------------
# The Studio's panel is a VIEW of the one set of rules
# ---------------------------------------------------------------------------

def test_the_studio_shows_the_rules_outside_any_scroll_area(manager):
    """The complaint, taken literally: not scrolled off, not on another tab."""
    from PySide6.QtWidgets import QScrollArea

    win = _studio(manager)
    try:
        panel = win.tidy_panel
        assert isinstance(panel, TidyRulesPanel)
        assert win.layout().indexOf(panel) != -1      # a direct child of the window
        node, chain = panel, []
        while node is not None:
            chain.append(node)
            assert not isinstance(node, QScrollArea)
            node = node.parentWidget()
        assert win in chain                           # and it is in THIS window
        # Every rule is on screen, matching the tab's group rule for rule.
        assert set(panel.boxes) == {spec["key"] for spec in RULE_SPECS}
    finally:
        win.close()


def test_the_panel_follows_the_lyrics_tab(manager):
    """Tab -> Studio, live: the Studio's view cannot go stale while it is open."""
    win = _studio(manager)
    try:
        panel = win.tidy_panel
        manager.lyrics_lines_check.setChecked(False)
        assert panel.boxes["do_capitalize_lines"].isChecked() is False
        manager.lyrics_lines_check.setChecked(True)
        assert panel.boxes["do_capitalize_lines"].isChecked() is True
    finally:
        win.close()


def test_ticking_the_panel_toggles_the_apps_own_state(manager):
    """Studio -> tab: there is one tidy state, and the panel is one view of it."""
    win = _studio(manager)
    try:
        panel = win.tidy_panel
        panel.boxes["do_strip_punctuation"].setChecked(False)
        assert manager.lyrics_punct_check.isChecked() is False
        assert manager._lyrics_options()["do_strip_punctuation"] is False
        panel.boxes["do_strip_punctuation"].setChecked(True)
        assert manager.lyrics_punct_check.isChecked() is True
        assert manager._lyrics_options()["do_strip_punctuation"] is True
    finally:
        win.close()


# ---------------------------------------------------------------------------
# The Studio's two tidy actions
# ---------------------------------------------------------------------------

def test_tidy_this_text_obeys_the_ticks_on_screen(manager):
    """The editor is the evidence: same text, two rule sets, two results."""
    win = _studio(manager)
    try:
        text = "hello there.\nsecond line."
        manager.lyrics_lines_check.setChecked(False)

        manager.lyrics_punct_check.setChecked(True)
        win.text.setPlainText(text)
        win.tidy()
        assert win.text.toPlainText() == "hello there\nsecond line"

        manager.lyrics_punct_check.setChecked(False)
        win.text.setPlainText(text)
        win.tidy()
        assert win.text.toPlainText() == text

        manager.lyrics_lines_check.setChecked(True)
    finally:
        win.close()


def test_apply_to_dataset_passes_the_panels_scope(manager, monkeypatch):
    """The scope the user picked in the Studio, not the Tools tab's leftover."""
    win = _studio(manager)
    try:
        seen = []
        monkeypatch.setattr(
            manager, "apply_lyrics_tidy", lambda scope=None: seen.append(scope)
        )
        panel = win.tidy_panel
        assert panel.scope_text() == SCOPE_CHOICES[0]
        panel.scope_combo.setCurrentText(SCOPE_CHOICES[1])
        panel.apply_btn.click()
        assert seen == [SCOPE_CHOICES[1]]
        # The pass reports itself in the window rather than silently.
        assert SCOPE_CHOICES[1] in win.report_label.text()
    finally:
        win.close()


def test_the_panel_has_no_apply_button_of_its_own_without_a_scope(manager):
    """``with_scope`` is what adds the apply control; without it there is none."""
    panel = TidyRulesPanel(manager)
    try:
        assert panel.apply_btn is None
        assert panel.scope_text() == SCOPE_CHOICES[0]
    finally:
        panel.deleteLater()


def test_a_manager_without_a_tidy_options_method_still_ticks(qapp):
    """A stand-in manager gets working ticks from the shipped defaults."""
    class _M:
        pass

    panel = TidyRulesPanel(_M())
    try:
        opts = panel.options()
        assert opts["do_capitalize_lines"] is True
        assert opts["ing_to_in"] is False
        panel.boxes["ing_to_in"].setChecked(True)
        assert panel.options()["ing_to_in"] is True
        assert isinstance(opts["contractions"], dict) and opts["contractions"]
    finally:
        panel.deleteLater()


def test_the_panel_is_re_read_when_it_reappears(manager):
    """The fallback for a signal that never arrived.

    The live path is the tab checkbox's ``stateChanged`` (tested above), so the
    panel cannot normally be stale. ``refresh``/``showEvent`` exist for the case
    where a change happened with signals blocked -- which is exactly how this is
    reproduced -- and a window that is re-shown must not display a tick the app
    no longer holds.
    """
    win = _studio(manager)
    try:
        panel = win.tidy_panel
        box = manager.lyrics_ing_check
        box.blockSignals(True)
        box.setChecked(True)
        box.blockSignals(False)
        assert panel.boxes["ing_to_in"].isChecked() is False   # view is behind
        panel.refresh()
        assert panel.boxes["ing_to_in"].isChecked() is True    # caught up
        box.setChecked(False)
    finally:
        win.close()
