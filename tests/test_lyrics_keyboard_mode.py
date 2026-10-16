"""Keyboard mode for the Lyrics Studio: the word walk, brackets, and the keys.

Spec item 6 — ``Enter`` enters the mode and highlights the first word of the
selected song, ``←``/``→`` step one word, ``↑``/``↓`` step one line, ``Enter``
capitalizes the highlighted word, and anything in ``[...]`` is skipped.

What is pinned, and why each is easy to lose in a refactor:

* **The word model is shared with the double-click gesture.** ``join_apostrophes``
  is what makes ``don't`` ONE word instead of the three Qt sees (``don``, ``'``,
  ``t``). The double-click needs it and so does every move here, so the module
  exposes it for both. Two implementations would eventually disagree and the key
  walk would visit fragments.
* **Brackets are structure, not lyrics.** A tag is never highlighted and never
  toggled. ``word_spans`` drops any word starting inside ``[...]``, so a tag is
  not even a stopping point on the walk.
* **The ``length()`` trap.** ``KeyboardMode._word_near`` bounds a line with
  ``len(line.text())``, never Qt's ``line.length()``, which counts the trailing
  paragraph separator. With ``length()`` a word starting at the next line's first
  character is offered as "on this line" for BOTH lines, and Up/Down return to
  the word they came from — a key that looks stuck. The Down-then-Up test is the
  regression guard.
* **Enter is consumed only by the mode.** Bare ``Enter`` is not allowed to insert
  a newline while the mode is running (``Shift+Enter`` is), which is why the mode
  is an event FILTER on the studio's editor and not a property of
  ``LyricTextEdit``: the inline field is the same class and must keep bare
  ``Enter`` as a newline. That scope decision is tested at the bottom.
* **Retention.** PySide6 frees a ``QObject`` whose last reference goes away, and
  a freed filter silently stops receiving keys with no error, so the mode is
  installed on both the editor and its viewport and the module owns a reference.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.lyrics_keyboard_mode import (  # noqa: E402
    MAX_LINE_WALK,
    MODE_ACTIVE_TEXT,
    MODE_EMPTY_TEXT,
    MODE_IDLE_TEXT,
    KeyboardMode,
    bracket_spans,
    first_word,
    in_brackets,
    join_apostrophes,
    next_word,
    prev_word,
    word_spans,
)


def _editor(text, qapp):
    """A shown ``LyricTextEdit`` at a realistic size.

    Shown rather than parentless because ``ensureCursorVisible`` and the viewport
    filter are part of what is under test, and a hidden widget has no meaningful
    viewport geometry.
    """
    from ui.lyrics_studio import LyricTextEdit

    edit = LyricTextEdit()
    edit.setPlainText(text)
    edit.resize(400, 240)
    edit.show()
    qapp.processEvents()
    return edit


def _mode(text, qapp):
    """A hosted editor with keyboard mode installed.

    Returns ``(editor, mode, label)``. The mode and label are stashed on the
    editor so the parent chain keeps them referenced — an unreferenced filter is
    freed and then silently receives nothing, which is the trap the module
    docstring documents.
    """
    from PySide6.QtWidgets import QLabel

    edit = _editor(text, qapp)
    label = QLabel("")
    mode = KeyboardMode(edit, label)
    edit._test_mode = mode
    edit._test_label = label
    return edit, mode, label


def _word(edit, mode):
    """The text of the word the walk currently highlights, or ``None``."""
    if mode.span is None:
        return None
    return edit.toPlainText()[mode.span[0]:mode.span[1]]


def _press(key, mods=Qt.NoModifier):
    """A synthetic ``KeyPress`` QEvent, for asserting the consumed/not answer."""
    return QKeyEvent(QEvent.Type.KeyPress, key, mods)


def _viewport_press():
    """A synthetic mouse press, as the viewport delivers one."""
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(1.0, 1.0),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )


class TestWordModel:
    """``join_apostrophes`` / ``word_spans``: what counts as ONE word."""

    def test_an_internal_apostrophe_makes_one_word(self):
        """``don't`` must be a single span, not ``don`` + ``'`` + ``t``.

        Qt splits all three. If the walk used Qt's boundaries a right-arrow press
        would land on the apostrophe and the taps would drift one step out of step
        with every contraction in the song.
        """
        assert word_spans("don't stop") == [(0, 5), (6, 10)]

    def test_a_contraction_is_discovered_once_not_twice(self):
        """The widened span is found from ``don`` and from ``t``; merged to one.

        Without the merge a right-arrow press would stop on the same contraction
        twice and appear to do nothing the first time.
        """
        assert word_spans("don't") == [(0, 5)]

    def test_a_trailing_apostrophe_is_not_joined(self):
        """``believin'`` keeps the apostrophe OUT of the word.

        The rule is "an apostrophe joins only with alphanumerics on BOTH sides",
        and there is nothing after the trailing one — so it stays punctuation.
        """
        assert word_spans("believin'") == [(0, 8)]

    def test_a_quoted_letter_is_not_joined_backwards(self):
        """``rock 'n' roll`` stays three words.

        The apostrophe after ``rock`` has a SPACE on its left, so it must not pull
        ``rock`` into the ``n``.
        """
        assert word_spans("rock 'n' roll") == [(0, 4), (6, 7), (9, 13)]

    def test_typographic_apostrophes_join_too(self):
        """A pasted caption uses U+2019; Qt splits that the same way."""
        assert word_spans("don\u2019t stop") == [(0, 5), (6, 10)]

    def test_punctuation_is_not_a_word(self):
        """``dog.`` must not contribute a ``.`` span of its own.

        Qt treats a trailing ``.``/``,``/``?``/``!`` as a word, and
        ``'.'.upper() == '.'`` — which is how a capitalize on ``dog.`` used to do
        nothing at all while ``dog dog`` worked.
        """
        assert word_spans("dog. dog") == [(0, 3), (5, 8)]

    def test_an_underscore_run_is_not_a_lyric_word(self):
        """``[^\\W_]`` rather than ``\\w``: the ``_`` is a separator, not a letter.

        So a ``foo_bar`` is three words, not one — an identifier-ish run is never
        treated as a single lyric word.
        """
        assert word_spans("foo_bar baz") == [(0, 3), (4, 7), (8, 11)]

    def test_spans_are_ordered(self):
        """Every walk helper assumes ascending order."""
        spans = word_spans("one two three four")
        assert spans == sorted(spans)

    def test_join_apostrophes_is_a_width_not_a_replacement(self):
        """It returns a span, so callers can keep the original text."""
        assert join_apostrophes("o'clock now", 0, 1) == (0, 7)


class TestBrackets:
    """``[...]`` is structure: never a word, never a stopping point."""

    def test_bracket_spans_are_found_and_ordered(self):
        assert bracket_spans("[A] x [B]") == [(0, 3), (6, 9)]

    def test_a_bracket_does_not_span_a_line_break(self):
        """``[^\\]]`` cannot cross a newline, so an unclosed tag is not a tag.

        An unterminated ``[`` must not swallow the rest of the song and make every
        word after it un-walkable.
        """
        assert bracket_spans("[Verse 1\nhello") == []

    def test_in_brackets_is_start_inclusive_end_exclusive(self):
        assert in_brackets("[A] x", 2) is True
        assert in_brackets("[A] x", 3) is False
        assert in_brackets("[A] x", 5) is False

    def test_words_inside_a_tag_are_dropped(self):
        """``Verse`` and ``1`` are never walkable words."""
        assert word_spans("[Verse 1]\nhello world") == [(10, 15), (16, 21)]

    def test_a_tag_only_line_has_no_words_at_all(self):
        assert word_spans("[Verse 1]") == []

    def test_first_word_skips_a_leading_tag(self):
        """Enter on a song that starts with a tag lands on the first REAL word."""
        assert first_word("[Verse 1]\nreal words") == (10, 14)


class TestWalkEdges:
    """``first_word`` / ``next_word`` / ``prev_word``: the absolute bounds."""

    def test_first_word_is_none_on_empty_text(self):
        assert first_word("") is None

    def test_first_word_is_none_on_whitespace_only(self):
        assert first_word("   \n  ") is None

    def test_next_word_stops_at_the_end(self):
        """Past the last word there is no target — the walk must NOT wrap.

        Wrapping to the top of the song is how a user marks a word twice without
        noticing, so the edge is a stop.
        """
        assert next_word("a b", 3) is None

    def test_prev_word_stops_at_the_start(self):
        assert prev_word("a b", 0) is None

    def test_next_word_takes_the_first_word_at_or_after(self):
        assert next_word("aa bb cc", 3) == (3, 5)

    def test_prev_word_takes_the_last_word_ending_at_or_before(self):
        assert prev_word("aa bb cc", 3) == (0, 2)

    def test_max_line_walk_is_a_positive_bound(self):
        """The bound is what stops the blank-line skip being an unbounded scan."""
        assert isinstance(MAX_LINE_WALK, int) and MAX_LINE_WALK > 0


class TestKeys:
    """The key table, driven through real key events on a hosted editor."""

    def test_bare_enter_enters_the_mode_on_the_first_word(self, qapp):
        edit, mode, _ = _mode("aaa bbb\nccc ddd", qapp)
        QTest.keyClick(edit, Qt.Key_Return)

        assert mode.active is True
        assert _word(edit, mode) == "aaa"

    def test_enter_is_consumed_so_it_inserts_no_newline(self, qapp):
        """The mode owns bare ``Enter``; the text must be untouched.

        A newline here would silently split the line the user is marking up.
        """
        text = "aaa bbb\nccc ddd"
        edit, mode, _ = _mode(text, qapp)
        QTest.keyClick(edit, Qt.Key_Return)

        assert edit.toPlainText() == text

    def test_enter_again_capitalizes_the_highlighted_word(self, qapp):
        edit, mode, _ = _mode("hello world", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Return)

        assert edit.toPlainText() == "HELLO world"

    def test_capitalize_takes_the_whole_contraction(self, qapp):
        """One Enter on ``don't`` yields ``DON'T``.

        It reuses the widget's own ``toggle_word_at_cursor``, so the contraction
        widening is the double-click's, by construction.
        """
        edit, mode, _ = _mode("don't stop", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Return)

        assert edit.toPlainText() == "DON'T stop"

    def test_capitalize_is_a_toggle_not_a_one_way_trip(self, qapp):
        """A second Enter on an UPPERCASE word brings it back down."""
        edit, mode, _ = _mode("hello world", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Return)

        assert edit.toPlainText() == "hello world"

    def test_right_steps_one_word(self, qapp):
        edit, mode, _ = _mode("aaa bbb ccc", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)

        assert _word(edit, mode) == "bbb"

    def test_left_steps_one_word_back(self, qapp):
        edit, mode, _ = _mode("aaa bbb ccc", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)
        QTest.keyClick(edit, Qt.Key_Left)

        assert _word(edit, mode) == "aaa"

    def test_down_and_up_step_a_line_at_the_same_column(self, qapp):
        """The regression guard for the ``length()`` trap.

        Down must land on the next line's word, and Up straight back must return
        to the word it came FROM. When ``_word_near`` used Qt's ``length()`` the
        next line's first word was offered for both lines, so Up returned to the
        word it had just left — a key that appeared stuck.
        """
        edit, mode, _ = _mode("aaa bbb\nccc ddd\neee fff", qapp)
        QTest.keyClick(edit, Qt.Key_Return)          # aaa
        QTest.keyClick(edit, Qt.Key_Right)           # bbb
        QTest.keyClick(edit, Qt.Key_Down)
        assert _word(edit, mode) == "ddd"

        QTest.keyClick(edit, Qt.Key_Up)
        assert _word(edit, mode) == "bbb"

    def test_a_line_walk_steps_over_a_blank_line(self, qapp):
        """A blank line between verses is normal and must not be a wall."""
        edit, mode, _ = _mode("aaa bbb\n\nccc ddd", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)           # bbb
        QTest.keyClick(edit, Qt.Key_Down)

        assert _word(edit, mode) == "ddd"

    def test_a_line_walk_steps_over_a_tag_only_line(self, qapp):
        """``[Chorus]`` is structure, not a place the walk stops."""
        edit, mode, _ = _mode("aaa bbb\n[Chorus]\nccc ddd", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)           # bbb
        QTest.keyClick(edit, Qt.Key_Down)

        assert _word(edit, mode) == "ddd"

    def test_the_line_walk_is_bounded(self, qapp):
        """More than ``MAX_LINE_WALK`` blank lines: give up, do not scan forever.

        The blank-line skip is why this bound has to exist at all.
        """
        text = "aaa\n" + ("\n" * (MAX_LINE_WALK + 20)) + "bbb"
        edit, mode, _ = _mode(text, qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        assert _word(edit, mode) == "aaa"

        QTest.keyClick(edit, Qt.Key_Down)
        assert _word(edit, mode) == "aaa", "the walk ran past its bound"

    def test_the_walk_stops_at_the_end_of_the_song(self, qapp):
        edit, mode, _ = _mode("aaa bbb", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)           # bbb, the last word
        QTest.keyClick(edit, Qt.Key_Right)

        assert _word(edit, mode) == "bbb", "the walk wrapped instead of stopping"

    def test_the_walk_stops_at_the_start_of_the_song(self, qapp):
        edit, mode, _ = _mode("aaa bbb", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Left)

        assert _word(edit, mode) == "aaa"

    def test_escape_leaves_the_mode_but_keeps_the_highlight(self, qapp):
        """The highlight records where the walk stopped, so Enter can resume.

        Clearing it would send the next Enter back to the top of the song.
        """
        edit, mode, _ = _mode("aaa bbb ccc", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)
        QTest.keyClick(edit, Qt.Key_Escape)

        assert mode.active is False
        assert edit.textCursor().selectedText() == "bbb"

    def test_a_step_down_uses_the_visual_column_not_the_index(self, qapp):
        """A word mid-line keeps its column when the line below is walked to."""
        edit, mode, _ = _mode("aa bb cc dd\n11 22 33 44", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)
        QTest.keyClick(edit, Qt.Key_Right)           # cc
        QTest.keyClick(edit, Qt.Key_Down)

        assert _word(edit, mode) == "33"

    def test_the_label_reports_the_mode_state(self, qapp):
        edit, mode, label = _mode("aaa bbb", qapp)
        assert label.text() == MODE_IDLE_TEXT

        QTest.keyClick(edit, Qt.Key_Return)
        assert label.text() == MODE_ACTIVE_TEXT

        QTest.keyClick(edit, Qt.Key_Escape)
        assert label.text() == MODE_IDLE_TEXT

    def test_a_modifier_press_does_not_end_the_mode(self, qapp):
        """Pressing Shift on the way to Shift+Enter is a preparation, not an edit.

        If it dropped the mode, the Shift+Enter that followed would do something
        different from what the user intended.
        """
        edit, mode, _ = _mode("aaa bbb", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        assert mode.active is True

        assert mode.handle_key(_press(Qt.Key_Shift, Qt.ShiftModifier)) is False
        assert mode.active is True

    def test_shift_enter_is_handed_back_to_the_widget(self, qapp):
        """``Shift+Enter`` is the documented way to add a newline.

        Only the CONTRACT is pinned: the mode exits and the key is not consumed,
        so it reaches the widget and behaves exactly as it does when the mode is
        off. What STILL holds the selection at that moment is a known wart, noted
        in `.agent_notes.md`, not a behaviour this test blesses.
        """
        edit, mode, _ = _mode("aaa bbb", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        assert mode.active is True

        assert mode.handle_key(_press(Qt.Key_Return, Qt.ShiftModifier)) is False
        assert mode.active is False

    def test_a_plain_letter_is_handed_back_to_the_widget(self, qapp):
        """Typing must stay possible: any other key exits and is not consumed."""
        edit, mode, _ = _mode("aaa bbb", qapp)
        QTest.keyClick(edit, Qt.Key_Return)

        assert mode.handle_key(_press(Qt.Key_X)) is False
        assert mode.active is False

    def test_a_click_leaves_the_mode(self, qapp):
        """A click moves the caret, which would strand the recorded span.

        The next Enter would then capitalize a word the caret is no longer on, so
        a mouse press on the viewport exits the mode.
        """
        edit, mode, _ = _mode("aaa bbb", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        assert mode.active is True

        QTest.mouseClick(edit.viewport(), Qt.LeftButton)
        assert mode.active is False

    def test_enter_resumes_from_where_the_walk_stopped(self, qapp):
        """Because the highlight survives Escape, Enter picks up from it.

        Restarting at the top of a 40-line song every time would make marking a
        repeated word near the end unnecessarily painful.
        """
        edit, mode, _ = _mode("aaa bbb ccc", qapp)
        QTest.keyClick(edit, Qt.Key_Return)
        QTest.keyClick(edit, Qt.Key_Right)           # bbb
        QTest.keyClick(edit, Qt.Key_Escape)
        QTest.keyClick(edit, Qt.Key_Return)

        assert mode.active is True
        assert _word(edit, mode) == "bbb"

    def test_nothing_to_walk_reports_and_stays_inactive(self, qapp):
        """A tag-only document has no words, so Enter must not pretend otherwise."""
        edit, mode, label = _mode("[Verse 1]", qapp)
        QTest.keyClick(edit, Qt.Key_Return)

        assert mode.active is False
        assert label.text() == MODE_EMPTY_TEXT


class TestInstallation:
    """The filter has to actually be installed on BOTH objects.

    PySide6 does not expose Qt's ``eventFilters()`` accessor (PyQt does), so
    installation is proven the only way that cannot lie: deliver the event only
    the filter can see, and watch the mode react to it.
    """

    def test_a_key_press_reaches_the_mode_on_the_editor(self, qapp):
        edit, mode, _ = _mode("aaa bbb", qapp)

        QApplication.sendEvent(edit, _press(Qt.Key_Return))

        assert mode.active is True

    def test_a_click_reaches_the_mode_on_the_viewport(self, qapp):
        """The viewport half is where a click-only bug hides.

        A ``QAbstractScrollArea`` hands clicks to its viewport, so a filter on the
        editor alone never sees this event and the mode would stay active with a
        stale word.
        """
        edit, mode, _ = _mode("aaa bbb", qapp)
        mode.activate()
        assert mode.active is True

        QApplication.sendEvent(edit.viewport(), _viewport_press())

        assert mode.active is False

    def test_the_label_is_optional(self, qapp):
        """``label=None`` must not crash: the mode is usable without a status line."""
        edit, mode, _ = _mode("aaa bbb", qapp)
        bare = KeyboardMode(edit, None)
        edit._test_bare = bare

        assert bare.handle_key(_press(Qt.Key_Return)) is True
        assert bare.active is True


class TestScope:
    """Keyboard mode is STUDIO-ONLY, and that is the point of the design."""

    def test_the_shared_editor_installs_no_filter_by_itself(self, qapp):
        """A bare ``LyricTextEdit`` — the inline field's class — has no mode.

        The mode consumes bare ``Enter``. If it were a property of the widget, the
        Track Inspector's inline lyrics field would lose the newline it needs. A
        ``KeyboardMode`` parents itself to the widget it filters, so the widget's
        own child list is the honest place to look.
        """
        edit = _editor("aaa bbb", qapp)
        assert edit.findChildren(KeyboardMode) == []

    def test_bare_enter_still_inserts_a_newline_in_the_bare_editor(self, qapp):
        """The contract the event-filter design exists to protect.

        This is the inline field's ``Enter``: no mode installed, so the key
        reaches the widget untouched and inserts a newline.
        """
        edit = _editor("aaa", qapp)
        edit.setFocus()
        QTest.keyClick(edit, Qt.Key_Return)

        assert "\n" in edit.toPlainText()

    def test_the_shared_editor_has_no_key_press_override(self, qapp):
        """The absence is the contract.

        The module deliberately does not touch the widget class, so a future edit
        cannot take bare ``Enter`` away from the inline field by accident. If this
        fails, the mode has been moved onto the class and the inline field is now
        affected.
        """
        from ui.lyrics_studio import LyricTextEdit

        assert "keyPressEvent" not in LyricTextEdit.__dict__
