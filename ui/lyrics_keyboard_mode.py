"""Keyboard mode for the Lyrics Studio: walk the words and mark them shouted.

Why this is its own module
--------------------------
The owner's architecture rule is that new UI is a MODULE, and the Lyrics Studio
window lives in ``ui/lyrics_studio.py``. Keyboard mode is a self-contained
interaction: a word walk, a bracket filter, and a key table. None of it needs to
know about the window's diff pane, its tidy rules or its write-back row, so it
sits here and the window installs it in one line.

It is STUDIO-ONLY, and that is a deliberate scope decision
----------------------------------------------------------
``LyricTextEdit`` is shared with the Track Inspector's inline lyrics field
(``ui/lyrics_inspector.py``), and there bare ``Enter`` is how a user adds a new
line. Keyboard mode consumes bare ``Enter``, so making it a property of the
widget would have taken the newline away from a field that must keep it.

Hence this module does NOT touch the widget class at all. It is an EVENT FILTER
that ``LyricsStudioWindow`` installs on its own editor, so:

* the inline field cannot be affected even by a careless future edit, because it
  never installs the filter;
* ``LyricTextEdit`` keeps its documented shape -- notably the absence of a
  ``keyPressEvent`` override and of the removed per-host gesture switch
  (``set_gestures_enabled`` / ``gestures_enabled``), both of which tests pin.

The one trap: RETENTION
-----------------------
``ui/lyrics_studio.py`` already records this for the window chrome, and it
applies here unchanged -- PySide6 frees a QObject whose last reference goes
away, and a freed object takes its event filter with it. An unretained
``KeyboardMode`` would simply stop receiving keys, with no error. The window
therefore stores it on ``self`` (``self.keyboard_mode``).

The word model is SHARED with the double-click gesture, not re-derived
---------------------------------------------------------------------
``join_apostrophes`` is the logic that makes ``don't`` ONE word instead of the
three Qt sees (``don``, ``'``, ``t``). The double-click gesture needs it, and so
does every move here: without it a word walk through ``don't stop believin'``
visits fragments, and the taps drift out of step with the syllables the singer
actually sings. Two implementations would eventually disagree, so the logic
lives HERE and ``ui/lyrics_studio.py`` imports it back. The import is one-way
(``lyrics_studio`` -> ``lyrics_keyboard_mode``), so there is no cycle.

Brackets are skipped
--------------------
``[Verse 1]`` and friends are structure, not lyrics: they are never highlighted
and never toggled, and the walk steps over them. ``BRACKET_SPAN`` is the same
pattern ``modules/lyrics_normalizer.py`` uses to decide what a tag is, so
"a bracket is a tag" has one meaning in the app.
"""

import re

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QTextCursor

# What counts as a structure tag. Deliberately the same shape as
# ``modules/lyrics_normalizer._TAG_BRACKETS``: whatever the tidy rules treat as
# a tag, the keyboard walk refuses to touch. A tag never spans a line break,
# because ``[^\\]]`` cannot cross one.
BRACKET_SPAN = re.compile(r"\[[^\]]*\]")

# A word, in the loose sense this module needs: a run of letters/digits. The
# underscore is excluded (``[^\\W_]`` rather than ``\\w``) so an identifier-ish
# run is not silently treated as a lyric word. Unicode-aware, so accented
# letters count.
_WORD_RUN = re.compile(r"[^\W_]+", re.UNICODE)

# How many lines a single Up/Down press may walk before giving up. A blank line
# between verses is normal, so the walk skips empty lines instead of refusing to
# move; this bound is what stops that from being an unbounded scan.
MAX_LINE_WALK = 200

# The two apostrophes a lyric line can carry: ASCII and the typographic one a
# pasted caption often uses. Both split a word in Qt's eyes.
APOSTROPHES = "'\u2019"


def bracket_spans(text):
    """Ordered ``(start, end)`` spans of every ``[...]`` in ``text``."""
    return [(m.start(), m.end()) for m in BRACKET_SPAN.finditer(text)]


def in_brackets(text, pos):
    """True when ``pos`` falls inside a ``[...]`` span."""
    return any(start <= pos < end for start, end in bracket_spans(text))


# ---------------------------------------------------------------------------
# The word model. Moved here from ``ui/lyrics_studio.py`` unchanged in
# behaviour, so that the double-click gesture and the key walk cannot drift:
# ``ui/lyrics_studio.py`` imports these back.
# ---------------------------------------------------------------------------
def join_apostrophes(text, start, end):
    """Widen ``[start, end)`` across apostrophes that sit *inside* a word.

    Qt splits ``don't`` into three words (``don``, ``'``, ``t``) and
    ``o'clock`` into two, so a click or a Tab would act on a fragment and the
    taps would drift out of step with the syllables. An apostrophe joins only
    when there is an alphanumeric character on BOTH sides, which is what keeps a
    trailing one (``believin'``) and a quoted one (``rock 'n' roll``) left alone.
    """
    while (
        end + 1 < len(text)
        and text[end] in APOSTROPHES
        and text[end + 1].isalnum()
    ):
        end += 1
        while end < len(text) and text[end].isalnum():
            end += 1
    while (
        start >= 2
        and text[start - 1] in APOSTROPHES
        and text[start - 2].isalnum()
    ):
        start -= 1
        while start > 0 and text[start - 1].isalnum():
            start -= 1
    return start, end


def words_cursor(doc, start, end):
    """A cursor selecting ``[start, end)``, widened to a full contraction."""
    start, end = join_apostrophes(doc.toPlainText(), start, end)
    cursor = QTextCursor(doc)
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.KeepAnchor)
    return cursor


def word_under_or_before(cursor):
    """The word at ``cursor``, or the one it sits immediately after.

    ``QTextCursor.WordUnderCursor`` answers "the word this position is IN",
    which is the wrong question once the position sits one past a word's last
    letter. Qt also treats a trailing ``-``/``.``/``,``/``"``/``?``/``!`` as a
    word of its own, so at position 3 of ``dog.`` it returns ``'.'`` — and
    ``'.'.upper() == '.'``, which is how a double-click on ``dog.`` came to do
    nothing at all while ``dog dog`` worked. The fallback probes
    ``position - 1`` and accepts the first selection that actually contains an
    alphanumeric character, widened to a whole contraction.

    Returns ``None`` when neither position yields a word. That is a blank line,
    a lone ``'``, or punctuation clicked directly, and all three must stay a
    no-op rather than reaching for a neighbouring word.
    """
    doc = cursor.document()
    pos = cursor.position()
    for probe in (pos, pos - 1):
        if probe < 0:
            continue
        candidate = QTextCursor(doc)
        candidate.setPosition(probe)
        candidate.select(QTextCursor.WordUnderCursor)
        if any(ch.isalnum() for ch in candidate.selectedText()):
            start = candidate.selectionStart()
            end = candidate.selectionEnd()
            return words_cursor(doc, start, end)
    return None


def word_spans(text):
    """Every lyric word in ``text`` as an ordered ``(start, end)`` span.

    Derived from the text rather than from Qt's word boundaries, because Qt's
    boundaries are exactly what this module exists to correct: a run of
    letters/digits is found first, then widened across an internal apostrophe by
    the SAME ``join_apostrophes`` the double-click gesture uses. Overlapping
    results are merged (a widened contraction is discovered twice — once from
    ``don`` and once from ``t``), and any word starting inside ``[...]`` is
    dropped, so a tag is never a stopping point.
    """
    found = []
    for match in _WORD_RUN.finditer(text):
        found.append(join_apostrophes(text, match.start(), match.end()))
    found.sort()

    merged = []
    for start, end in found:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    tags = bracket_spans(text)
    return [
        (start, end)
        for start, end in merged
        if not any(tag_start <= start < tag_end for tag_start, tag_end in tags)
    ]


def first_word(text):
    """The first navigable word of ``text``, or ``None`` if there is none."""
    spans = word_spans(text)
    return spans[0] if spans else None


def next_word(text, pos):
    """The first word starting at or after ``pos``, or ``None``."""
    for span in word_spans(text):
        if span[0] >= pos:
            return span
    return None


def prev_word(text, pos):
    """The last word ending at or before ``pos``, or ``None``."""
    previous = None
    for span in word_spans(text):
        if span[1] <= pos:
            previous = span
        else:
            break
    return previous


# ---------------------------------------------------------------------------
# What the window displays, and what the keys do.
# ---------------------------------------------------------------------------
MODE_IDLE_TEXT = "⌨ Keyboard mode: press Enter to start"
MODE_ACTIVE_TEXT = (
    "⌨ Keyboard mode — ←/→ word · ↑/↓ line · Enter capitalize · Esc to finish"
)
MODE_EMPTY_TEXT = "⌨ Nothing to walk: there are no words to capitalize here."

# A bare modifier press is a preparation, not an edit, so it must not end the
# mode: pressing Shift on the way to Shift+Enter would otherwise drop the mode
# and change what Enter then does.
_MODIFIER_KEYS = frozenset(
    (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta)
)
_ENTER_KEYS = (Qt.Key_Return, Qt.Key_Enter)


class KeyboardMode(QObject):
    """Walks the words of ONE lyrics editor with the keyboard.

    ``Enter`` starts the walk on the first word (or on a word the caret has
    already selected, so finishing and restarting does not lose your place),
    ``Enter`` again capitalizes the highlighted word, ``←``/``→`` step one word,
    ``↑``/``↓`` step one line at the same visual column, and ``Esc`` finishes
    WITHOUT clearing the highlight. Any other key — a letter, a shortcut,
    ``Shift+Enter`` for a new line — leaves the mode and then does exactly what
    it has always done, because this class returns ``False`` from the filter and
    lets the event reach the widget untouched.

    It is an event filter rather than a property of the editor on purpose: the
    editor is shared with the Track Inspector's inline field, where bare
    ``Enter`` must keep inserting a newline. See this module's docstring.

    A ``QObject``, and parented to the editor it filters, because Qt will only
    accept a ``QObject`` as an event filter and because the parent is what keeps
    it alive. The window still keeps its own reference (``self.keyboard_mode``)
    for the other half of that belt-and-braces: the chrome has already been
    bitten once by a freed collaborator silently taking its event filter with
    it.
    """

    # Class-level defaults, NOT merely instance assignments in __init__.
    # PySide6 CLEARS an object's instance attributes once its C++ side is
    # destroyed, while Qt can still deliver events to an event filter that has not
    # been removed yet. With no default the first such event raised
    # ``AttributeError: 'KeyboardMode' object has no attribute '_widget'`` from
    # inside eventFilter(), and PySide6 printed a traceback to stderr FOR EACH
    # ONE -- reproduced simply by destroying an editor that had the mode
    # installed, which is what closing the Lyrics Studio window does. Falling back
    # to these values makes the filter a harmless no-op during teardown.
    _widget = None
    _view = None
    _label = None
    _active = False
    _span = None

    def __init__(self, widget, label=None, parent=None):
        super().__init__(parent if parent is not None else widget)
        self._widget = widget
        # A QAbstractScrollArea hands mouse presses to its VIEWPORT, so a filter
        # on the editor alone never sees a click — the click would leave the mode
        # running with a stale word. The viewport is filtered for that one event
        # and nothing else.
        self._view = widget.viewport() if hasattr(widget, "viewport") else None
        self._label = label
        self._active = False
        self._span = None
        widget.installEventFilter(self)
        if self._view is not None:
            self._view.installEventFilter(self)
        self._set_label(MODE_IDLE_TEXT)

    # ---- state ----------------------------------------------------------
    @property
    def active(self):
        """True while the walk is running."""
        return self._active

    @property
    def span(self):
        """The highlighted word as ``(start, end)``, or ``None``."""
        return self._span

    # ---- filter ---------------------------------------------------------
    def eventFilter(self, obj, event):
        """Handle the mode's keys; let every other event through untouched."""
        if obj is not self._widget:
            if obj is self._view and event.type() == QEvent.Type.MouseButtonPress:
                # Any click is an editing action: it moves the caret, which
                # would leave ``self._span`` naming a word the caret is no
                # longer on — and the next Enter would capitalize that stale
                # word. The click itself is then delivered normally.
                self.deactivate()
            return False
        kind = event.type()
        if kind == QEvent.Type.KeyPress:
            return self.handle_key(event)
        return False

    def handle_key(self, event):
        """True when the key belonged to the mode and must not reach the widget."""
        key = event.key()
        mods = event.modifiers()
        if key in _MODIFIER_KEYS:
            return False
        if not self._active:
            # Enter starts the walk. It cannot also insert a newline — that is
            # what Shift+Enter is for in this window, and the hint says so.
            if key in _ENTER_KEYS and mods == Qt.NoModifier:
                self.activate()
                return True
            return False
        if key in _ENTER_KEYS and mods == Qt.NoModifier:
            self.toggle_current()
            return True
        if key == Qt.Key_Escape:
            self.deactivate()
            return True
        if mods == Qt.NoModifier:
            if key == Qt.Key_Right:
                self.move(1)
                return True
            if key == Qt.Key_Left:
                self.move(-1)
                return True
            if key == Qt.Key_Down:
                self.move_line(1)
                return True
            if key == Qt.Key_Up:
                self.move_line(-1)
                return True
        # Anything else is an edit or a shortcut (Shift+Enter included): get out
        # of the way so the key behaves exactly as it does when the mode is off.
        self.deactivate()
        return False

    # ---- actions --------------------------------------------------------
    def activate(self):
        """Start the walk. Returns False when there is nothing to walk."""
        text = self._widget.toPlainText()
        cursor = self._widget.textCursor()
        span = None
        if cursor.hasSelection() and any(
            ch.isalnum() for ch in cursor.selectedText()
        ):
            candidate = join_apostrophes(
                text, cursor.selectionStart(), cursor.selectionEnd()
            )
            if not in_brackets(text, candidate[0]):
                span = candidate
        if span is None:
            span = first_word(text)
        if span is None:
            self._active = False
            self._set_label(MODE_EMPTY_TEXT)
            return False
        self._active = True
        self._highlight(span)
        return True

    def deactivate(self):
        """Leave the walk, LEAVING the last word highlighted.

        The highlight is the record of where the walk stopped, and because
        ``activate`` resumes from a selection, pressing Enter again picks up
        there instead of jumping back to the top of the song.
        """
        self._active = False
        self._span = None
        self._set_label(MODE_IDLE_TEXT)

    def move(self, delta):
        """Step ``delta`` words (``+1`` right, ``-1`` left). Stops at the end."""
        if self._span is None:
            return
        text = self._widget.toPlainText()
        if delta > 0:
            target = next_word(text, self._span[1])
        else:
            target = prev_word(text, self._span[0])
        if target is None:
            return                      # the song's edge: stop, never wrap
        self._highlight(target)

    def move_line(self, delta):
        """Step to the next line's word nearest the current visual column.

        Blank lines and tag-only lines are stepped OVER rather than acting as a
        wall — a line between verses is normal, and refusing to move there reads
        as a broken key. ``MAX_LINE_WALK`` bounds the scan.
        """
        if self._span is None:
            return
        doc = self._widget.document()
        text = self._widget.toPlainText()
        anchor = QTextCursor(doc)
        anchor.setPosition(self._span[0])
        line = anchor.block()
        column = self._span[0] - line.position()
        for _ in range(MAX_LINE_WALK):
            line = doc.findBlockByNumber(line.blockNumber() + delta)
            if not line.isValid():
                return                  # ran off the top/bottom of the song
            target = self._word_near(text, line, column)
            if target is not None:
                self._highlight(target)
                return

    def toggle_current(self):
        """Capitalize the highlighted word, reusing the gesture's own toggle.

        Deliberately NOT a second implementation of "uppercase this word": it
        puts the selection on the word and calls the widget's
        ``toggle_word_at_cursor``, the very method the double-click reaches. One
        implementation means the contraction widening, the
        lower-it-if-already-upper rule and the "keep it selected so a repeat
        works" behaviour are identical in both paths by construction.
        """
        if not self._active or self._span is None:
            return
        self._highlight(self._span)
        self._widget.toggle_word_at_cursor()

    # ---- helpers --------------------------------------------------------
    def _highlight(self, span):
        """Select ``span``, scroll it into view and record it."""
        cursor = words_cursor(self._widget.document(), span[0], span[1])
        self._widget.setTextCursor(cursor)
        # Without this a walk down a 40-line song navigates off-screen, and the
        # user is capitalizing words they cannot see.
        self._widget.ensureCursorVisible()
        self._span = (cursor.selectionStart(), cursor.selectionEnd())
        self._set_label(MODE_ACTIVE_TEXT)

    @staticmethod
    def _word_near(text, line, column):
        """The word on ``line`` whose start is closest to ``column``.

        The upper bound is ``len(line.text())``, NOT ``line.length()``: Qt's
        ``length()`` counts the trailing paragraph separator, so ``start +
        length()`` is the first character of the NEXT line — and a word starting
        there would be offered as "on this line" for both lines. That is not
        hypothetical: it made Down from one line land on a word and the Up
        straight back from it return to the same word, a key that appeared stuck.
        """
        start = line.position()
        line_end = start + len(line.text())
        on_line = [
            span
            for span in word_spans(text)
            if start <= span[0] < line_end
        ]
        if not on_line:
            return None
        return min(
            on_line, key=lambda span: (abs((span[0] - start) - column), span[0])
        )

    def _set_label(self, text):
        if self._label is not None:
            self._label.setText(text)
