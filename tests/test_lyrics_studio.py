"""The Lyrics Studio: the double-click gesture, the diff, and where an export lands.

This replaces two test modules that covered two windows — the Lyric Capitalizer
(``tests/test_lyric_capitalizer.py``) and the Review & Fix dialog
(``tests/test_lyrics_review.py``). Both windows are now ``ui.lyrics_studio``, so
their contracts live together here.

What is pinned, and why each is easy to lose in a refactor:

* **The one gesture.** Double-click toggles a word's case, and a word is matched
  by its letters, not Qt's idea of a word boundary, so ``dog.`` and ``don't``
  capitalize from a click on any letter. Tab and Ctrl+U are deliberately NOT
  gestures: their ABSENCE is pinned, because a lingering intercept is exactly how
  "Tab uppercased every word" or "Ctrl+U ate the selection" would come back.
* **The editor stays editable.** Pasting a whole song is the main way text gets
  in. A read-only pane would look harmless and silently break it — the diff box
  below IS read-only, and the two must not be confused.
* **The tidy rules are the Lyrics tab's.** The window does not own a second copy
  of the options or the contraction table; it calls ``manager._lyrics_options()``.
* **The export target.** ``formatted_lyrics`` is what the training export reads;
  a write that only filled ``lyrics`` would look right in the app and ship
  unmarked lyrics to training.
* **The window is resizable.** An initial size and a floor, no fixed size — a
  lyrics surface that cannot be grown to a full screen of text is the bug the
  merged window exists to avoid.
"""
import ast
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

import ui.lyrics_studio as ls  # noqa: E402
from ui.lyrics_studio import (  # noqa: E402
    BOTH_TARGET,
    CAPTION_TARGET,
    LYRICS_TARGET,
    LyricTextEdit,
    LyricsStudioWindow,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Manager:
    """The smallest stand-in that satisfies the window (no QMainWindow).

    Building a real ``DatasetManager`` applies the theme to every page, which is
    slow enough to matter, and none of the behaviour under test needs one.
    """

    def __init__(self, sample, options=None):
        from PySide6.QtWidgets import QLabel

        self.sample = sample
        self.snapshots = 0
        self.last_snapshot = None
        self.refreshed = 0
        self.selected = 0
        self.status_label = QLabel("")
        self._options = options if options is not None else {
            "contractions": {},
            "ing_to_in": False,
            "ing_exceptions": set(),
            "do_capitalize_tags": True,
            "do_capitalize_lines": True,
            "do_strip_punctuation": True,
        }

    def get_selected_sample(self):
        self.selected += 1
        return self.sample

    def record_snapshot(self):
        self.snapshots += 1
        # Mirror the real manager: the snapshot is the dataset BEFORE the write,
        # so a test can prove the write happens after the snapshot.
        self.last_snapshot = dict(self.sample) if self.sample else None

    def refresh_table(self):
        self.refreshed += 1

    def on_table_selection_changed(self):
        pass

    def _export_lyrics(self, sample, text):
        self.exported = text

    def _lyrics_options(self):
        return dict(self._options)


def _track(lyrics="", caption=""):
    return {"filename": "a.mp3", "caption": caption, "formatted_lyrics": lyrics,
            "lyrics": lyrics}


def _window(qapp, lyrics="", options=None, sample=None):
    sample = sample if sample is not None else _track(lyrics)
    manager = _Manager(sample, options)
    return manager, sample, LyricsStudioWindow(manager, sample)


def _editor(text, qapp):
    edit = LyricTextEdit()
    edit.setPlainText(text)
    return edit


def _hosted_editor(text, qapp):
    """A ``LyricTextEdit`` in a real window, beside a focusable sibling.

    Needed for the Tab tests: ``tabChangesFocus(True)`` moves focus to the NEXT
    focusable widget and inserts nothing — but if the editor is the only
    focusable widget in its window there is nowhere to move and Qt falls back to
    inserting a literal tab. The real hosts always have siblings, so the fixture
    mirrors that rather than testing a parentless widget. Returns the editor;
    the holder is stashed on it so Qt's parent chain keeps it alive.
    """
    from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

    holder = QWidget()
    lay = QVBoxLayout(holder)
    edit = LyricTextEdit()
    edit.setPlainText(text)
    sibling = QPushButton("next")
    lay.addWidget(edit)
    lay.addWidget(sibling)
    holder.show()
    edit.setFocus()
    qapp.processEvents()
    edit._test_holder = holder          # keep the parent chain referenced
    return edit


def _double_click_char(edit, index):
    """Really double-click the glyph at ``index`` of the editor's text.

    A real mouse event, not ``toggle_word_at_cursor()``, because the bug this
    guards only exists in the real path: Qt's double-click leaves the cursor at
    the word's END, and the gesture used to re-derive the word from there. A
    direct call with a position inside the word passes even when the gesture is
    broken, so it cannot catch this. The x offset targets the middle of the glyph
    (``fm.horizontalAdvance(text[:index])`` is its left edge) and y the middle of
    the first line.
    """
    text = edit.toPlainText()
    fm = edit.fontMetrics()
    x = fm.horizontalAdvance(text[:index])
    x += max(1, fm.horizontalAdvance(text[index]) // 2)
    QTest.mouseDClick(
        edit.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(x, fm.height() // 2)
    )




# ---------------------------------------------------------------------------
# The one gesture: double-click toggles a word's case
# ---------------------------------------------------------------------------

# Every punctuation mark Qt treats as a word of its own. A double-click on the
# letters of ``dog<char>`` must still uppercase ``dog``.
TRAILING_MARKS = [".", ",", "-", '"', "?", "!"]


@pytest.mark.parametrize("mark", TRAILING_MARKS)
def test_a_word_ending_in_a_special_character_capitalizes(mark, qapp):
    """``dog.`` must behave like ``dog`` — the reported bug.

    Qt reports the trailing mark as its own "word", so the gesture used to
    uppercase ``.`` (a no-op) and leave the text untouched.
    """
    edit = _editor(f"dog{mark}", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 1)          # a letter of "dog", not the mark
    assert edit.toPlainText() == f"DOG{mark}"


@pytest.mark.parametrize("mark", TRAILING_MARKS)
def test_the_same_word_toggles_back_with_a_trailing_character(mark, qapp):
    edit = _editor(f"DOG{mark}", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 1)
    assert edit.toPlainText() == f"dog{mark}"


def test_every_word_in_a_punctuated_line_is_clickable(qapp):
    """The original report, end to end: one click per word, mark by mark."""
    text = 'dog. dog, dog- dog" dog? dog!'
    want = 'DOG. DOG, DOG- DOG" DOG? DOG!'
    # Each word on its own editor: clicking the word must uppercase it whatever
    # mark follows.
    for start in range(0, len(text), 5):
        edit = _editor(text, qapp)
        edit.resize(900, 120)
        edit.show()
        _double_click_char(edit, start + 1)
        assert edit.toPlainText() == text[:start] + "DOG" + text[start + 3:], (
            f"clicking the word at {start} ({text[start:start + 3]!r}) did nothing"
        )
    # And one editor clicked word by word: uppercasing never changes a word's
    # length, so each click lands on its word and the whole line comes out.
    edit = _editor(text, qapp)
    edit.resize(900, 120)
    edit.show()
    for start in range(0, len(text), 5):
        _double_click_char(edit, start + 1)
    assert edit.toPlainText() == want


def test_an_apostrophe_word_is_clickable_from_any_letter(qapp):
    """``don't`` has the same trailing-mark problem, on a mark inside the word.

    Same root cause, different shape: Qt puts the ``'`` on its own, so
    ``WordUnderCursor`` at the word's end finds ``'`` and the click did nothing.
    """
    edit = _editor("don't stop", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 1)          # a letter of "don't"
    assert edit.toPlainText() == "DON'T stop"


def test_clicking_the_punctuation_itself_is_a_no_op(qapp):
    """Qt hands over the mark alone, and uppercasing it changes nothing.

    Pinned deliberately: reaching BACKWARDS to the word before the mark would
    make ``dog.`` capitalise from a click on the ``.``, which is a behaviour the
    gesture never had and nobody asked for. Clicking the word's own letters is
    the intended gesture.
    """
    edit = _editor("dog. dog", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 3)          # precisely on the "."
    assert edit.toPlainText() == "dog. dog"


def test_the_contraction_tail_is_joined_not_merely_skipped(qapp):
    """A click that lands mid-contraction still takes the WHOLE word.

    ``_word_under_or_before`` / ``_join_apostrophes`` join the ``t`` back onto
    ``don`` instead of stopping at the apostrophe, so a double-click on the left
    half of ``don't`` yields ``DON'T``, not ``DON'stop``. Driven through the real
    double-click path because that is where Qt's fragment selection is fixed up.
    """
    edit = _editor("don't stop", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 1)          # inside "don", before the apostrophe
    assert edit.toPlainText() == "DON'T stop"


def test_an_internal_apostrophe_comes_along_with_the_word(qapp):
    """``o'clock`` is one word, not ``o`` + ``clock``.

    Qt splits it, so a click used to capitalize only half the word.
    """
    edit = _editor("o'clock now", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 3)          # a letter of "clock"
    assert edit.toPlainText() == "O'CLOCK now"


def test_a_possessive_apostrophe_is_not_swallowed(qapp):
    """``dogs'`` must come out ``DOGS'``, not ``DOGS'`` grabbed whole.

    The apostrophe joins two letters, never a word and the space after it, so a
    possessive (and a quoted ``'n'``) keeps its mark outside the word.
    """
    edit = _editor("dogs' toys", qapp)
    edit.resize(400, 120)
    edit.show()
    _double_click_char(edit, 1)
    assert edit.toPlainText() == "DOGS' toys"


def test_double_click_gesture_uppercases_the_word(qapp):
    edit = _editor("she said stop right there", qapp)
    cur = edit.textCursor()
    cur.setPosition(9)          # inside "stop"
    edit.setTextCursor(cur)
    edit.toggle_word_at_cursor()
    assert edit.toPlainText() == "she said STOP right there"


def test_the_same_word_toggles_back_to_lowercase(qapp):
    edit = _editor("she said STOP right there", qapp)
    cur = edit.textCursor()
    cur.setPosition(11)
    edit.setTextCursor(cur)
    edit.toggle_word_at_cursor()
    assert edit.toPlainText() == "she said stop right there"


def test_there_is_no_gesture_switch_any_more(qapp):
    """Double-click is always on, so the switch that gated it is gone.

    A ``set_gestures_enabled``/``gestures_enabled`` pair would be a second way
    to enable the one gesture, and a stale ``False`` on the inline field is
    exactly how "double-click does nothing" would come back. The absence is the
    contract: the widget must NOT expose either name.
    """
    edit = _editor("stop now", qapp)
    assert not hasattr(edit, "set_gestures_enabled")
    assert not hasattr(edit, "gestures_enabled")


def test_double_click_toggles_through_the_real_event_path(qapp):
    """The gesture reaches the text via ``mouseDoubleClickEvent``, not only
    ``toggle_word_at_cursor`` — the wiring is what a refactor breaks."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    edit = _editor("stop now", qapp)
    pos = QPointF(5.0, 10.0)
    edit.mouseDoubleClickEvent(QMouseEvent(
        QMouseEvent.MouseButtonDblClick, pos, pos,
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    ))
    assert edit.toPlainText() == "STOP now"


def test_ctrl_u_does_not_uppercase_the_selection(qapp):
    """Ctrl+U is no longer a gesture; it falls through to Qt.

    The tool used to intercept Ctrl+U to toggle the selection's case. That was
    only ever safe while this widget swallowed the key — removing the
    interception restores Qt's own Ctrl+U, so the branch had to be deleted
    rather than left dangling. The selected text must therefore be UNCHANGED by
    the keystroke (whatever Qt's own binding does to the cursor, it does not
    uppercase the letters).
    """
    edit = _editor("hold me now", qapp)
    cur = edit.textCursor()
    cur.setPosition(0)
    cur.setPosition(4, QTextCursor.KeepAnchor)
    edit.setTextCursor(cur)
    QTest.keyClick(edit, Qt.Key_U, Qt.ControlModifier)
    assert edit.toPlainText() == "hold me now"


def test_tab_is_a_focus_key_not_a_capitalize_gesture(qapp):
    """Tab leaves the text alone and moves focus instead of marking a word.

    Tab used to capitalize the current word and advance. Now it is Qt's focus
    key again (``setTabChangesFocus(True)``), so the text is untouched and no
    literal ``\\t`` is inserted — a tab in a lyric line would be meaningless and
    would reach the export.
    """
    edit = _hosted_editor("run away tonight", qapp)
    cur = edit.textCursor()
    cur.setPosition(0)
    edit.setTextCursor(cur)
    QTest.keyClick(edit, Qt.Key_Tab)
    assert edit.toPlainText() == "run away tonight"
    assert "\t" not in edit.toPlainText()


def test_tab_does_not_insert_a_tab_character(qapp):
    edit = _hosted_editor("one two", qapp)
    QTest.keyClick(edit, Qt.Key_Tab)
    assert "\t" not in edit.toPlainText()


def test_shift_tab_still_moves_focus(qapp):
    """Shift+Tab must be left alone so the widget can be navigated out of."""
    edit = _hosted_editor("one two", qapp)
    QTest.keyClick(edit, Qt.Key_Tab, Qt.ShiftModifier)
    assert edit.toPlainText() == "one two"


def test_ctrl_v_pastes_from_the_clipboard(qapp):
    """The editor must stay a real editor: pasting lyrics in is the main way to
    get them in there, and an over-eager read-only would silently kill it."""
    from PySide6.QtWidgets import QApplication

    QApplication.clipboard().setText("PASTED LYRICS")
    edit = _editor("", qapp)
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key_V, Qt.ControlModifier)
    assert edit.toPlainText() == "PASTED LYRICS"


def test_open_onto_the_track_being_edited(qapp):
    sample = {"filename": "a.mp3", "formatted_lyrics": "[Verse]\nRISING up"}
    win = LyricsStudioWindow(_Manager(sample), sample)
    assert "RISING up" in win.text.toPlainText()
    win.deleteLater()



# ---------------------------------------------------------------------------
# Where an export lands (apply_target is the single field mapping)
# ---------------------------------------------------------------------------

def _export(qapp, monkeypatch, target, lyrics=""):
    """Drive ``add_to_track`` with the target given and no dialog shown."""
    manager, sample, win = _window(qapp, lyrics)
    win.text.setPlainText("RUN AWAY")
    monkeypatch.setattr(win, "_confirm", lambda *a, **k: True)
    win.add_to_track(target)
    return manager, sample


def test_export_into_the_main_caption_field(qapp, monkeypatch):
    mgr, sample = _export(qapp, monkeypatch, CAPTION_TARGET)
    assert sample["caption"] == "RUN AWAY"
    assert sample["formatted_lyrics"] == ""   # the lyrics were not touched
    assert mgr.snapshots == 1                  # undoable


def test_export_into_the_lyrics_field_writes_what_the_exporter_reads(qapp, monkeypatch):
    _mgr, sample = _export(qapp, monkeypatch, LYRICS_TARGET)
    assert sample["formatted_lyrics"] == "RUN AWAY"
    assert sample["lyrics"] == "RUN AWAY"      # the two never drift apart
    assert sample["caption"] == ""


def test_export_both(qapp, monkeypatch):
    _mgr, sample = _export(qapp, monkeypatch, BOTH_TARGET)
    assert sample["caption"] == "RUN AWAY"
    assert sample["formatted_lyrics"] == "RUN AWAY"


def test_a_none_target_asks_the_picker(qapp, monkeypatch):
    """``add_to_track(None)`` is the ▾ button: it asks where the text goes."""
    manager, sample, win = _window(qapp)
    win.text.setPlainText("RUN AWAY")
    monkeypatch.setattr(win, "ask_target", lambda: CAPTION_TARGET)
    monkeypatch.setattr(win, "_confirm", lambda *a, **k: True)
    win.add_to_track(None)
    assert sample["caption"] == "RUN AWAY"
    assert manager.snapshots == 1


def test_a_cancelled_picker_does_not_export(qapp, monkeypatch):
    manager, sample, win = _window(qapp)
    win.text.setPlainText("RUN AWAY")
    monkeypatch.setattr(win, "ask_target", lambda: None)
    win.add_to_track(None)
    assert sample["formatted_lyrics"] == "" and manager.snapshots == 0


def test_add_to_lyrics_writes_the_lyrics_field(qapp, monkeypatch):
    """The one-click path must land in ``formatted_lyrics`` — the field the
    export reads — without asking anything."""
    manager, sample, win = _window(qapp)
    win.text.setPlainText("RUN AWAY")
    monkeypatch.setattr(win, "_confirm", lambda *a, **k: True)
    win.add_to_lyrics()
    assert sample["formatted_lyrics"] == "RUN AWAY"
    assert sample["lyrics"] == "RUN AWAY"
    assert sample["caption"] == ""
    assert manager.snapshots == 1


def test_the_primary_button_is_wired_to_the_no_arg_wrapper(qapp):
    """``clicked`` emits a bool; connected straight to ``add_to_track`` that
    bool would arrive as ``target`` and silently match no branch."""
    _manager, sample, win = _window(qapp)
    assert win.add_btn.text().endswith("Add to Lyrics")
    win.text.setPlainText("RUN")
    win._confirm = lambda *a, **k: True
    # QPushButton.click() replays the real signal, so this exercises the wiring.
    win.add_btn.click()
    assert sample["formatted_lyrics"] == "RUN"


def test_an_empty_editor_is_not_added(qapp, monkeypatch):
    manager, sample, win = _window(qapp, "keep me")
    win.text.setPlainText("   ")
    monkeypatch.setattr(
        "ui.lyrics_studio.QMessageBox.information", lambda *a, **k: None
    )
    win.add_to_lyrics()
    assert sample["formatted_lyrics"] == "keep me"
    assert manager.snapshots == 0


def test_no_track_means_no_write(qapp, monkeypatch):
    manager = _Manager(None)
    win = LyricsStudioWindow(manager, None)
    monkeypatch.setattr(
        "ui.lyrics_studio.QMessageBox.warning", lambda *a, **k: None
    )
    win.text.setPlainText("words")
    win.add_to_lyrics()
    assert manager.snapshots == 0


def test_a_declined_overwrite_changes_nothing(qapp, monkeypatch):
    """``_confirm`` must actually gate the write, not just return True."""
    sample = {"filename": "a.mp3", "caption": "old caption", "formatted_lyrics": ""}
    manager = _Manager(sample)
    win = LyricsStudioWindow(manager, sample)
    monkeypatch.setattr(
        "ui.lyrics_studio.QMessageBox.question",
        lambda *a, **k: __import__(
            "PySide6.QtWidgets", fromlist=["QMessageBox"]
        ).QMessageBox.No,
    )
    assert win._confirm(sample, CAPTION_TARGET) is False
    assert sample["caption"] == "old caption"


def test_confirm_says_yes_when_the_field_is_empty(qapp):
    sample = {"filename": "a.mp3", "caption": "", "formatted_lyrics": ""}
    win = LyricsStudioWindow(_Manager(sample), sample)
    # No prompt at all when there is nothing to lose.
    assert win._confirm(sample, BOTH_TARGET) is True


def test_the_picker_still_reaches_the_caption(qapp, monkeypatch):
    sample = {"filename": "a.mp3", "caption": "", "formatted_lyrics": ""}
    manager = _Manager(sample)
    win = LyricsStudioWindow(manager, sample)
    win.text.setPlainText("SHOUTED")
    monkeypatch.setattr(win, "ask_target", lambda: CAPTION_TARGET)
    monkeypatch.setattr(win, "_confirm", lambda *a, **k: True)
    win.choose_target_then_export()
    assert sample["caption"] == "SHOUTED"
    assert sample["formatted_lyrics"] == ""



# ---------------------------------------------------------------------------
# The editor stays editable, the diff does not (paste is the point)
# ---------------------------------------------------------------------------

def test_the_editor_is_never_read_only(qapp):
    _m, _s, win = _window(qapp, "one")
    assert win.text.isReadOnly() is False


def test_the_diff_pane_is_read_only(qapp):
    """The one pane that must NOT accept text is the report."""
    _m, _s, win = _window(qapp, "one")
    assert win.diff.isReadOnly() is True


def test_ctrl_v_pastes_into_the_editor(qapp):
    from PySide6.QtWidgets import QApplication

    _m, _s, win = _window(qapp, "")
    QApplication.clipboard().setText("[Verse]\nPASTED LINE")
    win.text.setFocus()
    QTest.keyClick(win.text, Qt.Key_V, Qt.ControlModifier)
    assert win.text.toPlainText() == "[Verse]\nPASTED LINE"


def test_the_double_click_gesture_is_available_here(qapp):
    """Same widget as the inline field, so a double-click marks a word."""
    _m, _s, win = _window(qapp, "run away tonight")
    assert not hasattr(win, "capitalize_check")   # the mode checkbox is gone
    cur = win.text.textCursor()
    cur.setPosition(9)          # inside "tonight"
    win.text.setTextCursor(cur)
    win.text.toggle_word_at_cursor()
    assert win.text.toPlainText() == "run away TONIGHT"


def test_tab_is_a_focus_key_here_too(qapp):
    """Tab must not capitalize and must not drop a literal ``\\t`` into a lyric.

    This window used to carry a "🔠 Capitalize mode" checkbox because the widget
    needed a switch; with the switch gone, Tab is Qt's focus key everywhere and
    no tab character can reach the track.
    """
    _m, _s, win = _window(qapp, "one two")
    QTest.keyClick(win.text, Qt.Key_Tab)
    assert "\t" not in win.text.toPlainText()
    assert win.text.toPlainText() == "one two"


# ---------------------------------------------------------------------------
# Resizable: initial size + floor, no fixed size (the whole point of the merge)
# ---------------------------------------------------------------------------

def test_the_window_is_resizable_not_fixed(qapp):
    """A lyrics surface that cannot be grown is the bug the merge fixes.

    Pinned structurally: the initial size is the initial size, the minimum is a
    real floor, and the maximum is left open. A ``setFixedSize``/``setMaximumSize``
    creeping back in would pass every other test here and re-introduce the problem.
    """
    _m, _s, win = _window(qapp, "")
    assert (win.width(), win.height()) == ls.INITIAL_SIZE
    assert (win.minimumWidth(), win.minimumHeight()) == ls.MINIMUM_SIZE
    assert win.maximumWidth() > 100000 and win.maximumHeight() > 100000

    win.resize(1200, 1000)
    assert (win.width(), win.height()) == (1200, 1000)


def test_the_editor_absorbs_extra_height(qapp):
    """The editor's layout stretch is what makes resizing spend the space well."""
    _m, _s, win = _window(qapp, "")
    assert win.layout().stretch(win.layout().indexOf(win.text)) == 1
    assert win.layout().stretch(win.layout().indexOf(win.diff)) == 2



# ---------------------------------------------------------------------------
# The diff is against the track
# ---------------------------------------------------------------------------

def test_the_editor_is_seeded_from_the_track(qapp):
    _m, _s, win = _window(qapp, "[Verse]\nhello")
    assert win.text.toPlainText() == "[Verse]\nhello"


def test_an_unchanged_editor_says_so_instead_of_an_empty_diff(qapp):
    _m, _s, win = _window(qapp, "hello")
    assert "matches" in win.diff.toPlainText()


def test_editing_shows_the_line_delta_against_the_track(qapp):
    _m, _s, win = _window(qapp, "old line")
    win.text.setPlainText("new line")
    win.refresh_diff()
    diff = win.diff.toPlainText()
    assert "-old line" in diff and "+new line" in diff


def test_the_report_names_the_track_and_the_rules_in_effect(qapp):
    _m, _s, win = _window(qapp, "hello")
    report = win.report_label.text()
    assert "a.mp3" in report
    assert "Tidy rules in effect" in report


def test_with_no_track_the_add_button_is_disabled(qapp):
    manager = _Manager(None)
    win = LyricsStudioWindow(manager, None)
    assert win.add_btn.isEnabled() is False
    assert "Select a track" in win.warn_label.text()


# ---------------------------------------------------------------------------
# Tidy uses the TAB's rules, not a private copy
# ---------------------------------------------------------------------------

def _opts(**overrides):
    opts = {
        "contractions": {},
        "ing_to_in": False,
        "ing_exceptions": set(),
        "do_capitalize_tags": True,
        "do_capitalize_lines": True,
        "do_strip_punctuation": True,
    }
    opts.update(overrides)
    return opts


def test_tidy_reads_the_managers_options(qapp):
    """The window must not carry its own options — the Lyrics tab owns them."""
    _m, _s, win = _window(
        qapp, "", _opts(do_capitalize_tags=True, do_capitalize_lines=False,
                        do_strip_punctuation=False)
    )
    win.text.setPlainText("[verse]\nshouted")
    win.tidy()
    assert "[Verse]" in win.text.toPlainText()
    # The option that was OFF must not have been applied.
    assert win.text.toPlainText().splitlines()[1] == "shouted"


def test_tidy_respects_an_option_being_off(qapp):
    _m, _s, win = _window(
        qapp, "", _opts(do_capitalize_tags=False, do_capitalize_lines=False,
                        do_strip_punctuation=True)
    )
    win.text.setPlainText("hello.")
    win.tidy()
    assert win.text.toPlainText() == "hello"


def test_tidy_does_not_touch_the_track(qapp):
    _m, sample, win = _window(qapp, "hello")
    win.text.setPlainText("[verse]\nhello.")
    win.tidy()
    assert sample["formatted_lyrics"] == "hello"


def test_the_rule_summary_reads_the_options(qapp):
    manager, _s, _win = _window(qapp, "")
    summary = ls.rule_summary(manager)
    assert "tags" in summary and "contraction(s)" in summary


def test_the_rule_summary_survives_a_broken_manager(qapp):
    """The summary is decoration; it must never break the window."""
    class Broken:
        def _lyrics_options(self):
            raise RuntimeError("boom")

    assert ls.rule_summary(Broken()) == "unavailable"



# ---------------------------------------------------------------------------
# The old dialogs' tools must not vanish with them
# ---------------------------------------------------------------------------

def test_the_lrc_export_is_still_reachable(qapp):
    manager, _s, win = _window(qapp, "line one")
    win.lrc_btn.click()
    assert manager.exported == "line one"


def test_split_long_lines_is_still_reachable(qapp):
    _m, _s, win = _window(qapp, "")
    win.text.setPlainText(
        "this is a very long line that should be broken into pieces by the tool"
    )
    win.split_btn.click()
    assert len(win.text.toPlainText().splitlines()) > 1


# ---------------------------------------------------------------------------
# Cross-module contract + entry points
# ---------------------------------------------------------------------------

def test_there_is_one_field_mapping_for_an_export_target():
    """Both old dialogs wrote the SAME fields; the merged one must too.

    ``formatted_lyrics`` is what the training export reads; a write that only
    filled ``lyrics`` would look right in the app and ship unmarked lyrics.
    """
    lyrics = {}
    ls.apply_target(lyrics, "words", LYRICS_TARGET)
    assert lyrics == {"formatted_lyrics": "words", "lyrics": "words"}

    both = {}
    ls.apply_target(both, "words", BOTH_TARGET)
    assert both == {"caption": "words", "formatted_lyrics": "words",
                    "lyrics": "words"}

    caption = {}
    ls.apply_target(caption, "words", CAPTION_TARGET)
    assert caption == {"caption": "words"}


def test_the_entry_points_still_exist_and_open_the_studio():
    """``open_lyrics_editor`` and ``open_lyric_capitalizer`` are what the Lyrics
    tab and the toolbar call; the names are kept even though the window behind
    them changed. Both must open the ONE merged studio."""
    with open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    for name in ("open_lyrics_editor", "open_lyric_capitalizer"):
        bodies = [
            ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert bodies, f"{name} disappeared — its callers would break"
        assert "open_lyrics_studio" in bodies[0], (
            f"{name} no longer opens the Lyrics Studio"
        )


def test_open_lyrics_studio_is_modal_and_returns_the_dialog_result(qapp, monkeypatch):
    """The merged window stays modal, like the two it replaced.

    ``exec()`` is what keeps the table from moving under a half-written song.
    Driven here without a real event loop by intercepting ``exec``.
    """
    manager = _Manager(_track("hello"))
    seen = {}

    def fake_exec(self):
        seen["title"] = self.windowTitle()
        seen["editor"] = self.text.toPlainText()
        return 0

    monkeypatch.setattr(LyricsStudioWindow, "exec", fake_exec)
    result = ls.open_lyrics_studio(manager)
    assert result == 0
    assert "a.mp3" in seen["title"]
    assert seen["editor"] == "hello"

