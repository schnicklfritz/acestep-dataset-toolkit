"""The Lyrics Studio's diff pane (``ui.lyrics_diff``): collapse, render, debounce.

Spec item 6's "minimizable diff field". The pane was lifted out of
``ui/lyrics_studio.py`` so the studio keeps a report it can collapse, and the
things worth pinning are the ones that were true before and must stay true:

* **The report stays read-only.** It is a report; the editor is the input.
* **Minimizing HIDES the view and never removes it from the layout.** The studio
  pins ``layout().stretch(layout().indexOf(win.diff)) == 2``, and ``indexOf``
  returns ``-1`` for a widget that has been removed — so a collapse implemented
  by taking the widget out would silently change how the window spends its height.
* **The collapsed state survives.** It is a preference, remembered under
  ``lyrics_studio_diff_minimized`` through the same injectable ``saver`` seam
  ``ui/window_chrome.py`` uses, so this test proves the write WITHOUT touching
  the user's ``settings.json``.
* **The refresh is debounced, and still happens.** The pane owns the timer, but
  the window owns what a refresh means, so the pane emits ``changed`` and the
  window connects it to ``refresh_diff``.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.lyrics_diff import (  # noqa: E402
    COLLAPSED_MARKER,
    DEBOUNCE_MS,
    EXPANDED_MARKER,
    MATCHES_TEXT,
    MINIMIZED_KEY,
    NO_LINE_CHANGE,
    LyricsDiffPane,
    attach,
    header_text,
)


def _editor(text, qapp):
    """A shown ``LyricTextEdit``, the widget the pane listens to."""
    from ui.lyrics_studio import LyricTextEdit

    edit = LyricTextEdit()
    edit.setPlainText(text)
    edit.resize(400, 240)
    edit.show()
    qapp.processEvents()
    return edit


def _pane(text, qapp, config=None, saver=None):
    """``(editor, pane)`` with both pane widgets inside a SHOWN host.

    ``isVisible()`` is False for any child of a window that was never shown, so a
    parentless pane would make every visibility assertion below pass for the wrong
    reason. The host mirrors what the studio does — header and report in one
    vertical layout — and is stashed on the editor so Qt's parent chain keeps it
    referenced.
    """
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    edit = _editor(text, qapp)
    pane = LyricsDiffPane(edit, config, saver=saver)
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.addWidget(pane.header)
    lay.addWidget(pane.view, 2)
    host.resize(400, 400)
    host.show()
    qapp.processEvents()

    edit._test_pane = pane
    edit._test_host = host
    return edit, pane


def _click(widget):
    """A synthetic mouse press, the event the header's filter reacts to."""
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(1.0, 1.0),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(widget, event)


class TestTheReport:
    """What the pane is allowed to be: a read-only report."""

    def test_the_report_is_read_only(self, qapp):
        """The one pane that must not accept text is the report."""
        _edit, pane = _pane("track line", qapp)
        assert pane.view.isReadOnly() is True

    def test_matching_text_says_so_instead_of_an_empty_diff(self, qapp):
        """An empty report reads as "no data"; the wording says "nothing to do"."""
        _edit, pane = _pane("same line", qapp)
        pane.render("same line", "same line")
        assert pane.view.toPlainText() == MATCHES_TEXT
        assert "no-op" in pane.view.toPlainText()

    def test_trailing_whitespace_is_not_a_difference(self, qapp):
        """The studio compared stripped text before; the pane must keep doing it.

        A diff caused by a trailing space is noise the user cannot act on.
        """
        _edit, pane = _pane("line", qapp)
        pane.render("line\n", "  line  ")
        assert pane.view.toPlainText() == MATCHES_TEXT

    def test_a_changed_line_is_shown_as_a_unified_diff(self, qapp):
        _edit, pane = _pane("old line", qapp)
        pane.render("old line", "new line")
        text = pane.view.toPlainText()
        assert "-old line" in text and "+new line" in text

    def test_both_sides_empty_is_reported_as_a_match(self, qapp):
        """``unified_diff`` has no line to show here; the pane must still say
        something — an empty pane reads as a bug rather than as "no change"."""
        _edit, pane = _pane("", qapp)
        pane.render("", "")
        assert pane.view.toPlainText() == MATCHES_TEXT

    def test_clear_empties_the_report(self, qapp):
        """The no-track-selected case deliberately shows NOTHING to compare."""
        _edit, pane = _pane("line", qapp)
        pane.render("a", "b")
        pane.clear()
        assert pane.view.toPlainText() == ""

    def test_the_fallback_wording_is_the_documented_one(self):
        """Guards the constant against a rename going unnoticed."""
        assert NO_LINE_CHANGE == "(no line-level change)"


class TestCollapse:
    """Minimizing hides the view; it must never remove it from the layout."""

    def test_the_pane_starts_expanded(self, qapp):
        _edit, pane = _pane("line", qapp)
        assert pane.minimized is False
        assert pane.view.isVisible() is True
        assert EXPANDED_MARKER in pane.header.text()

    def test_minimizing_hides_the_view(self, qapp):
        _edit, pane = _pane("line", qapp)
        pane.set_minimized(True)
        assert pane.minimized is True
        assert pane.view.isVisible() is False

    def test_minimizing_swaps_the_marker(self, qapp):
        """The marker is the only sign the caption is a control."""
        _edit, pane = _pane("line", qapp)
        assert EXPANDED_MARKER in pane.header.text()

        pane.set_minimized(True)
        assert COLLAPSED_MARKER in pane.header.text()
        assert EXPANDED_MARKER not in pane.header.text()

    def test_restoring_brings_the_view_back(self, qapp):
        _edit, pane = _pane("line", qapp)
        pane.set_minimized(True)
        pane.set_minimized(False)
        assert pane.minimized is False
        assert pane.view.isVisible() is True

    def test_the_header_stays_visible_when_collapsed(self, qapp):
        """It is the control that brings the pane back, so it cannot hide too."""
        _edit, pane = _pane("line", qapp)
        pane.set_minimized(True)
        assert pane.header.isVisible() is True

    def test_a_click_on_the_header_toggles(self, qapp):
        _edit, pane = _pane("line", qapp)
        _click(pane.header)
        assert pane.minimized is True

        _click(pane.header)
        assert pane.minimized is False

    def test_toggle_flips_the_state(self, qapp):
        _edit, pane = _pane("line", qapp)
        pane.toggle()
        assert pane.minimized is True
        pane.toggle()
        assert pane.minimized is False

    def test_the_header_text_helper_carries_both_markers(self):
        """One place decides the caption, so the two states cannot drift."""
        assert COLLAPSED_MARKER in header_text(True)
        assert EXPANDED_MARKER in header_text(False)
        assert "Changes (diff)" in header_text(False)


class TestPersistence:
    """The collapsed state is a preference, and it is remembered.

    ``saver`` is injected so these tests prove the WRITE without writing to the
    real ``settings.json`` — the same seam ``ui/window_chrome.py`` exposes for its
    geometry.
    """

    def test_the_collapsed_state_is_saved_under_the_documented_key(self, qapp):
        saved = []
        config = {}

        def saver(cfg, keys):
            saved.append((dict(cfg), list(keys)))

        _edit, pane = _pane("line", qapp, config=config, saver=saver)
        pane.set_minimized(True)

        assert saved, "the preference was never handed to the saver"
        cfg, keys = saved[-1]
        assert keys == [MINIMIZED_KEY]
        assert cfg[MINIMIZED_KEY] is True

    def test_restoring_is_saved_too(self, qapp):
        """Collapsing then restoring must persist the RESTORED value, not stick."""
        saved = []
        config = {}

        def saver(cfg, keys):
            saved.append(dict(cfg))

        _edit, pane = _pane("line", qapp, config=config, saver=saver)
        pane.set_minimized(True)
        pane.set_minimized(False)

        assert saved[-1][MINIMIZED_KEY] is False
        assert config[MINIMIZED_KEY] is False

    def test_a_remembered_collapse_is_restored_on_open(self, qapp):
        """Reopening the studio must come back the way the user left it."""
        config = {MINIMIZED_KEY: True}
        _edit, pane = _pane("line", qapp, config=config)

        assert pane.minimized is True
        assert pane.view.isVisible() is False
        assert COLLAPSED_MARKER in pane.header.text()

    def test_an_absent_preference_opens_expanded(self, qapp):
        """First run: no key in settings.json, so nothing is collapsed."""
        _edit, pane = _pane("line", qapp, config={})
        assert pane.minimized is False
        assert pane.view.isVisible() is True

    def test_the_config_receives_the_value_even_without_a_saver(self, qapp):
        """The in-memory dict is updated either way, so a later save_config()
        carries the preference. The saver is stubbed only so this test cannot
        write the developer's real ``settings.json``."""
        config = {}
        _edit, pane = _pane(
            "line", qapp, config=config, saver=lambda cfg, keys: None
        )
        pane.set_minimized(True)
        assert config[MINIMIZED_KEY] is True

    def test_a_failing_saver_is_not_fatal(self, qapp):
        """A preference must never take the window down with it."""
        def saver(cfg, keys):
            raise OSError("read-only settings.json")

        _edit, pane = _pane("line", qapp, config={}, saver=saver)
        pane.set_minimized(True)

        assert pane.minimized is True


class TestDebounce:
    """The pane owns the timer; the window owns what a refresh means."""

    def test_an_edit_emits_changed_after_the_debounce(self, qapp):
        edit, pane = _pane("line", qapp)
        fired = []
        pane.changed.connect(lambda: fired.append(1))

        edit.setPlainText("edited")
        QTest.qWait(DEBOUNCE_MS + 150)

        assert fired, "an edit never reached the pane's changed signal"

    def test_the_debounce_interval_is_the_documented_one(self, qapp):
        """Pinned because 220ms is a deliberate feel choice, not a default."""
        assert DEBOUNCE_MS == 220

    def test_attach_returns_a_pane_with_both_widgets(self, qapp):
        """The window adds ``header`` and ``view`` itself; both must exist."""
        edit = _editor("line", qapp)
        pane = attach(edit, None)
        edit._test_pane = pane

        assert isinstance(pane, LyricsDiffPane)
        assert pane.header is not None
        assert pane.view is not None
        assert pane.view.isReadOnly() is True
