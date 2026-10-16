"""The Lyrics Studio's diff pane: its header, its read-only view, its debounce.

Why this is its own module
--------------------------
The owner's architecture rule is that new UI is a MODULE. The diff is a
self-contained surface — a caption, a read-only report, and a debounced refresh
— and none of it needs to know about the studio's tidy rules, its write-back row
or its undo snapshot, so it lives here and the window installs it.

Why the window places the pieces instead of getting ONE widget
--------------------------------------------------------------
This is the part that is easy to get wrong. ``tests/test_lyrics_studio.py`` pins
the window's own layout::

    win.layout().stretch(win.layout().indexOf(win.diff)) == 2

``QBoxLayout.indexOf`` only finds a widget that is a DIRECT child of that layout.
Handing the window a container widget with the report inside it would put the
report one level down, ``indexOf`` would return ``-1``, and the stretch that
spends the window's extra height on the diff would silently stop applying. So
the pane exposes ``header`` and ``view`` separately and the window adds BOTH
itself. That is why ``attach`` returns a controller, not a widget.

Minimizing hides, it does not remove
------------------------------------
Collapsing the pane calls ``setVisible(False)`` on the view rather than taking it
out of the layout. Removing it would break the same ``indexOf`` contract above,
and ``setVisible`` is enough: a hidden widget keeps its layout slot but is given
no space, so the editor absorbs the height. The header stays visible, because it
is the control that brings the pane back.

The collapsed state is a PREFERENCE, so it is persisted under
``lyrics_studio_diff_minimized`` through ``modules.config_store.save_plain_keys``
— the same route (and the same ``saver`` injection seam) ``ui/window_chrome.py``
uses for its geometry, so a test can assert the write without touching
``settings.json``.
"""

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QTextEdit

# The settings.json key the collapsed state is remembered under. Namespaced to
# the studio so it cannot collide with a main-window preference.
MINIMIZED_KEY = "lyrics_studio_diff_minimized"

# A diff on every keystroke of a 60-line song is wasted work; the pane is there
# to be read, not to keep up with the cursor.
DEBOUNCE_MS = 220

EXPANDED_MARKER = "▾"
COLLAPSED_MARKER = "▸"

# The caption body, unchanged from the studio's inline version: the colours are
# the legend for the unified diff below it.
_HEADER_BODY = (
    "<b>Changes (diff)</b> — <span style='color:#c33'>− track</span> / "
    "<span style='color:#3a3'>+ editor</span>"
)

# Said instead of an empty report when there is nothing to show. The "would be a
# no-op" wording is what tells the user Add to Lyrics is not needed.
MATCHES_TEXT = (
    "(editor matches the track's saved lyrics — Add to Lyrics would be a no-op)"
)
NO_LINE_CHANGE = "(no line-level change)"


def header_text(minimized):
    """The caption for either state, with its toggle marker in front."""
    marker = COLLAPSED_MARKER if minimized else EXPANDED_MARKER
    return f"{marker} {_HEADER_BODY}"


def _default_saver():
    """The real persistence path, imported lazily so importing this module does
    not drag in the config/secret store (same seam as ``window_chrome``)."""
    from modules.config_store import save_plain_keys

    return save_plain_keys


class LyricsDiffPane(QObject):
    """Owns the diff caption, the read-only report and the refresh debounce.

    A ``QObject`` because it is the header's event filter and because it parents
    the debounce timer; the window keeps its own reference on ``self`` for the
    same retention reason ``ui/lyrics_studio.py`` documents for the chrome and
    for keyboard mode.
    """

    # Emitted DEBOUNCE_MS after the last edit. The window connects this to its
    # own ``refresh_diff``: the pane renders text, but only the window knows the
    # selected track, the tidy rules and the buttons that go with them.
    changed = Signal()

    # Class-level defaults, for the PySide6 teardown trap documented on
    # ``KeyboardMode``: instance attributes are CLEARED when the C++ object is
    # destroyed, while Qt can still deliver events to a filter that is still
    # installed. Without a default that raised AttributeError from inside
    # ``eventFilter`` and printed a traceback per event.
    _config = None
    _saver = None
    _minimized = False
    header = None
    view = None

    def __init__(self, editor, config=None, saver=None, parent=None):
        super().__init__(parent)
        # A throwaway dict when there is no real config (the window passes None
        # for a manager built by tests), matching ``WindowChrome``.
        self._config = config if config is not None else {}
        self._saver = saver
        self._minimized = bool(self._config.get(MINIMIZED_KEY, False))

        self.header = QLabel()
        self.header.setTextFormat(Qt.RichText)
        self.header.setToolTip("Collapse or expand the diff")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.installEventFilter(self)

        self.view = QTextEdit()
        # This one IS read-only: it is a report, not an input. The editor is
        # where text goes.
        self.view.setReadOnly(True)
        self.view.setMinimumHeight(120)
        self.view.setStyleSheet("font-family: monospace;")
        self.view.setVisible(not self._minimized)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.changed)
        editor.textChanged.connect(self._timer.start)

        self._apply_header()

    # ---- state ----------------------------------------------------------
    @property
    def minimized(self):
        """True while the report is hidden."""
        return self._minimized

    def set_minimized(self, flag):
        """Hide or show the report, and remember the choice."""
        self._minimized = bool(flag)
        if self.view is not None:
            self.view.setVisible(not self._minimized)
        self._apply_header()
        self._persist()

    def toggle(self):
        """Flip the collapsed state (what a click on the header does)."""
        self.set_minimized(not self._minimized)

    def _apply_header(self):
        if self.header is not None:
            self.header.setText(header_text(self._minimized))

    def _persist(self):
        config = self._config
        if config is None:
            return
        config[MINIMIZED_KEY] = self._minimized
        saver = self._saver or _default_saver()
        try:
            saver(config, [MINIMIZED_KEY])
        except Exception:  # noqa: BLE001 -- a preference must never be fatal
            pass

    # ---- filter ---------------------------------------------------------
    def eventFilter(self, obj, event):
        """A click on the caption collapses or restores the report."""
        header = self.header
        if header is not None and obj is header:
            if event.type() == QEvent.Type.MouseButtonPress:
                self.toggle()
                return True
        return False

    # ---- content --------------------------------------------------------
    def render(self, track_text, editor_text):
        """Show ``editor_text`` against ``track_text`` as a unified diff.

        The import is local because ``ui.lyrics_tab`` imports Qt widgets and is
        only needed once there is something to diff.
        """
        track_text = (track_text or "").strip()
        editor_text = editor_text or ""
        if editor_text.strip() == track_text:
            self.view.setPlainText(MATCHES_TEXT)
            return
        from ui.lyrics_tab import unified_diff

        self.view.setPlainText(
            unified_diff(track_text, editor_text) or NO_LINE_CHANGE
        )

    def clear(self):
        """Empty the report — the no-track-selected case."""
        self.view.setPlainText("")


def attach(editor, config, saver=None):
    """Build a pane for ``editor``. The window adds ``header`` and ``view``."""
    return LyricsDiffPane(editor, config, saver=saver)
