"""Lyrics Studio — one resizable window for editing a track's lyrics.

What this replaces
------------------
There used to be TWO lyrics windows that shared one editor but split the work:

* the **Lyric Capitalizer** (``ui.lyric_capitalizer``) — a scratch editor whose
  only feature was the double-click UPPERCASE gesture, plus an "Add to Lyrics"
  export; and
* the **Review & Fix** dialog (``ui.lyrics_review``) — the same editor with a
  live diff against the track, the Lyrics tab's tidy rules, and file/LRC tools.

Splitting them meant the two things you actually do together — paste a song,
mark the shouted words, tidy it, write it back — were in two windows you had to
close and reopen to move between. They are now ONE window: the studio. Its
editor is a real editor (paste, undo/redo, cut/copy all work), the diff sits
directly under the text you are editing, and the mark-shouted gesture and the
tidy/write-back tools are all in the same place. Both old entry points
(``manager.open_lyrics_editor`` and ``manager.open_lyric_capitalizer``) open
this window, so no caller and no muscle memory is broken.

Why one module
--------------
``LyricTextEdit`` is the shared widget — it is ALSO the Track Inspector's inline
lyrics field (see ``dataset_manager``). Keeping the gesture defined once, here,
is the whole reason the inline field and this window cannot drift: there is no
per-host gesture switch and this is the only ``mouseDoubleClickEvent`` that
knows how to widen a contraction selection.

Resizable, deliberately
-----------------------
The two old dialogs called ``resize(...)`` as their initial size and nothing
else, which is correct — a QDialog stays user-resizable. This window keeps that
contract explicitly: ``resize()`` is initial-only, a real ``minimumSize`` is the
only floor, and the editor expands with the window (``stretch=1``). There is
deliberately NO ``setFixedSize`` / ``setMaximumSize``; a lyrics surface that
cannot be grown to a full screen of text is the bug this window exists to avoid.

Independent, not modal
----------------------
``open_lyrics_studio`` returns a live window and never calls ``exec()``. The
studio is one of the app's three REAL windows (with the main window and the
Tools window), so it is a plain framed top-level window: it can be moved,
resized, minimized, maximized and pushed always-on-top/bottom, and it stays open
beside the main window instead of blocking it. That is what makes it the MAIN
place lyrics get changed rather than a modal interruption you dismiss.

Two consequences follow, and both are deliberate:

* the window is CACHED on the manager (``manager._lyrics_studio_window``), so
  reopening re-targets the one window instead of stacking a second copy -- "one
  window, one track" still holds, without modal bookkeeping;
* the caller gets the WINDOW back, not an ``exec()`` code, so tests drive the
  window directly instead of spinning a nested event loop.

Chrome (always-on-top/bottom, and geometry that is clamped onto a real screen so
it survives a monitor being unplugged) comes from ``ui.window_chrome`` -- the
same module the main and Tools windows use, so there is exactly one
implementation of both.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import lyrics_diff, studio_player, window_chrome
from ui.lyrics_keyboard_mode import (
    KeyboardMode,
    word_under_or_before,
    words_cursor,
)

# Spec item 4, in the owner's own words: "double clicking a word capitalizes it
# by default, to enter keyboard mode, press enter". Kept verbatim in meaning --
# a hint that paraphrases the gesture is how the gesture gets forgotten.
HINT = (
    "Double-clicking a word capitalizes it by default. To enter keyboard mode, "
    "press Enter. Shift+Enter adds a new line."
)

# Where an export can land. Order matters: the first entry is the default.
CAPTION_TARGET = "Main caption field"
LYRICS_TARGET = "Lyrics field"
BOTH_TARGET = "Both (caption + lyrics)"
EXPORT_TARGETS = [CAPTION_TARGET, LYRICS_TARGET, BOTH_TARGET]

# Initial size only — the window is freely resizable from here up. ``MINIMUM``
# is the one floor (below it the diff and the editor stop being readable); it is
# NOT a fixed size.
INITIAL_SIZE = (860, 700)
MINIMUM_SIZE = (560, 420)

# settings.json key for this window's saved geometry. The z-order preference is
# stored by the chrome under ``<key>_z``. One key per window, like ui_dock_state.
GEOMETRY_KEY = "lyrics_studio_geometry"


def apply_target(sample, text, target):
    """Write ``text`` into ``sample`` for one of the ``EXPORT_TARGETS``.

    The single place that knows the field mapping. ``formatted_lyrics`` is what
    the export reads (the training loader never looks at anything else), and
    ``lyrics`` is written in step with it so every other reader in the app sees
    the same words. The capitalizer export and the review write-back both call
    this, which is exactly why it is not inlined in either.
    """
    if target in (CAPTION_TARGET, BOTH_TARGET):
        sample["caption"] = text
    if target in (LYRICS_TARGET, BOTH_TARGET):
        sample["formatted_lyrics"] = text
        sample["lyrics"] = text
    return sample


# The word model — ``join_apostrophes`` / ``words_cursor`` /
# ``word_under_or_before``, and the key walk that shares them — now lives in
# ``ui.lyrics_keyboard_mode`` and is imported at the top of this module. It moved
# so that the double-click gesture and the keyboard walk use ONE
# contraction-widening rule instead of two that could drift apart; see that
# module's docstring for why ``don't`` has to be one word.
class LyricTextEdit(QPlainTextEdit):
    """The editor with the capitalization gestures wired in.

    Shared by this window and the Track Inspector's inline lyrics field, so the
    gesture is defined once. There is no per-host switch: double-click is always
    on.

    Keyboard mode is NOT part of this widget, and that is deliberate. Bare
    ``Enter`` has to keep inserting a newline in the inline field, so the mode is
    an event filter that the Lyrics Studio installs on its own editor
    (``ui.lyrics_keyboard_mode``). Nothing on this class knows it exists — which
    is what keeps the inline field out of a keys fight it never opted into.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(13)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        # Tab is always Qt's focus key, never a gesture. This is the permanent
        # setting rather than a switch. In every real host (this window, the
        # inline field) there is a focusable sibling, so Tab moves focus in
        # either direction and no literal tab is inserted. Note the one Qt
        # fallback: if the widget were the ONLY focusable widget in its window,
        # ``tabChangesFocus`` has nowhere to move to and Qt inserts a tab
        # character instead — which is why the gesture tests host the editor
        # beside a sibling rather than leaving it parentless.
        self.setTabChangesFocus(True)
        self.setPlaceholderText("Open a .txt / .lrc file, or paste lyrics here…")

    # ---- gestures -------------------------------------------------------
    def toggle_case(self, cursor):
        """UPPERCASE the cursor's selection, or lowercase it if already upper."""
        if not cursor.hasSelection():
            return
        text = cursor.selectedText()
        if not text.strip():
            return
        new_text = text.lower() if text.isupper() else text.upper()
        start = cursor.selectionStart()
        cursor.beginEditBlock()
        cursor.insertText(new_text)
        cursor.endEditBlock()
        # Keep the same word selected so repeated toggles work on one word.
        again = QTextCursor(self.document())
        again.setPosition(start)
        again.setPosition(start + len(new_text), QTextCursor.KeepAnchor)
        self.setTextCursor(again)

    def toggle_word_at_cursor(self):
        """UPPERCASE the word under the cursor, or the one it sits after.

        The double-click path arrives here with Qt's own ``dog`` selection
        already in place (``mouseDoubleClickEvent`` calls ``super()`` first), so
        that selection is the answer and is used as-is. Re-deriving the word from
        the cursor's position instead is what made ``dog.`` unclickable: the
        double-click leaves the position *after* the word, where
        ``WordUnderCursor`` finds the ``.``.
        """
        cursor = self.textCursor()
        if cursor.hasSelection():
            # Qt's selection is the answer — but for a contraction it hands over
            # only a fragment (``don`` of ``don't``), so widen it to the whole
            # word before toggling. A punctuation-only selection is left exactly
            # as Qt gave it: uppercasing it is a no-op, which is the intended
            # behaviour for a click on the mark itself.
            if any(ch.isalnum() for ch in cursor.selectedText()):
                cursor = words_cursor(
                    self.document(), cursor.selectionStart(), cursor.selectionEnd()
                )
            self.toggle_case(cursor)
            return
        word = word_under_or_before(cursor)
        if word is not None:
            self.toggle_case(word)

    # ---- events ---------------------------------------------------------
    def mouseDoubleClickEvent(self, event):
        # Let the base class do the word selection, then toggle it. Doing it
        # here (not on a timer) keeps the selection and the toggle atomic.
        # No feature gate: double-click is THE gesture now, and the widget is
        # the only thing that knows how to widen a contraction selection.
        super().mouseDoubleClickEvent(event)
        self.toggle_word_at_cursor()

    # There is no ``keyPressEvent`` any more. It used to carry three branches —
    # Tab-to-capitalize-and-advance, Shift+Tab-to-go-back, and Ctrl+U-to-toggle
    # — all of which are gone with the gesture switch. The base class handles
    # every remaining key, including Ctrl+V paste, and ``setTabChangesFocus``
    # (set in __init__) makes Tab a focus key in every real host (where a
    # focusable sibling exists) instead of inserting a literal tab. Leaving a
    # no-op override here would only be somewhere for the deleted behaviour to
    # creep back in.
    #
    # Keyboard mode did NOT change that, deliberately. It needs bare ``Enter``,
    # which the inline field must keep as a newline, so the mode is an event
    # filter that the Lyrics Studio installs on its own editor
    # (``ui.lyrics_keyboard_mode``) rather than a hook on this shared class. If
    # you came here to make Enter do something for BOTH hosts, that is the
    # mistake this paragraph exists to stop.

    # ---- file helpers ---------------------------------------------------
    def load_file(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            self.setPlainText(fh.read())

    def save_file(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.toPlainText().strip())

class LyricsStudioWindow(QDialog):
    """The one lyrics window: edit, mark shouted, diff, tidy, write back.

    ``manager`` supplies the selected track, the undo snapshot and the tidy
    rules; ``sample`` is the track the window opened on. The window is modal
    (``open_lyrics_studio`` calls ``exec()``), so the table cannot move under a
    half-written song.
    """

    def __init__(self, manager, sample):
        # ``manager`` is a QMainWindow in the app, but the window is also built
        # from tests and tooling with a plain stand-in, so only accept it as a
        # parent when Qt can actually use it.
        parent = manager if isinstance(manager, QWidget) else None
        super().__init__(parent)
        self.manager = manager
        self.sample = sample
        # The track this window is aimed at. Set when a caller names one, and
        # re-set whenever the main table's selection moves; see ``_selected_track``.
        self._pinned = None
        # True only while this window is itself writing a track back. That write
        # goes through ``manager.on_table_selection_changed``, which notifies this
        # window again -- and a write-back is not a user changing songs, so the
        # follow logic must not read it as one. Without this the (correct) guard
        # against losing text would fire from inside its own answer and re-enter.
        self._writing = False
        name = (sample or {}).get("filename", "")
        self.setWindowTitle(f"Lyrics Studio — {name}" if name else "Lyrics Studio")

        # Resizable, deliberately: an initial size and a floor, nothing that
        # pins it. ``setMinimumSize`` is the only size constraint; there is no
        # setFixedSize/setMaximumSize, so the window grows to whatever the user
        # drags it to. Min/Max hints give the maximize button something to do.
        self.resize(*INITIAL_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        # A REAL window, not a modal sheet: full native frame (minimize,
        # maximize, close, system menu) and no modality, so the studio lives
        # beside the main window instead of blocking it. ``Qt.Window`` is what
        # gives it its own entry in the taskbar/window list.
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setModal(False)

        # Shared chrome: the Window menu (always-on-top/bottom, minimize,
        # maximize, reset position) plus geometry clamped onto a real screen on
        # every show and saved on a debounce. It is kept on ``self`` on purpose:
        # PySide6 frees a QObject whose last reference goes away, and a freed
        # chrome takes its event filter with it, so the window would silently
        # stop saving its geometry.
        self.chrome = window_chrome.attach(
            self, getattr(manager, "config", None), GEOMETRY_KEY,
            default_size=INITIAL_SIZE,
        )

        lay = QVBoxLayout(self)

        # The chrome as a menu bar, so the controls are discoverable without
        # knowing to right-click the title bar.
        self.menu_bar = QMenuBar(self)
        self.menu_bar.addMenu(self.chrome.menu())
        lay.setMenuBar(self.menu_bar)

        # ---- player (spec item 7: the waveform sits at the TOP) -------------
        # Its own ``QMediaPlayer``, in its own module (``ui/studio_player``):
        # the main window's player is a *view of its selection* and would be
        # stolen from, and steal from, this window -- see that module's
        # docstring. Kept on ``self`` like the chrome, the diff pane and
        # keyboard mode: PySide6 frees a QObject whose last reference goes away,
        # and a freed transport would silently stop answering its keys.
        self.player = studio_player.attach(getattr(manager, "config", None))
        lay.addWidget(self.player)
        # The transport keys are claimed on the WINDOW -- not on the editor and
        # not on the player -- and only while no text input has focus, so `d`
        # stays the letter d while a lyric is being typed. See that module.
        self.player.attach_keys(self)

        hint = QLabel(HINT)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addWidget(QLabel(
            "<b>UPPERCASE = belting / shouted delivery.</b> Paste or edit the "
            "song, double-click a word to mark it shouted, review the diff, then "
            "add it to the track. Nothing is saved until you do."
        ))

        # ---- editor ---------------------------------------------------------
        self.text = LyricTextEdit()
        # A lyrics surface, so it follows ``lyrics_font_size`` like the inline
        # field (see ui/themes.FONT_REGION_PROPERTY). Not QWidget.setFont(): the
        # app stylesheet would override a per-widget font on every theme
        # re-apply.
        from ui.themes import FONT_REGION_LYRICS, FONT_REGION_PROPERTY

        self.text.setProperty(FONT_REGION_PROPERTY, FONT_REGION_LYRICS)
        self.text.setPlaceholderText(
            "Paste the song here (Ctrl+V), double-click a word to mark it shouted "
            "(UPPERCASE), then review the diff below and add it to the track."
        )
        self.text.setMinimumHeight(220)
        # stretch=1 is what makes the window genuinely resizable: the editor
        # absorbs extra height instead of the layout leaving a blank band.
        lay.addWidget(self.text, 1)

        # ---- keyboard mode ---------------------------------------------------
        # The mode's own state line, plus the mode itself. Its label is separate
        # from ``report_label`` below on purpose: that one reports the RESULT of
        # Save/Copy/Tidy/Split, and a walk in progress would overwrite those
        # messages the moment the first key arrived.
        self.mode_label = QLabel("")
        self.mode_label.setProperty("muted", True)
        self.mode_label.setWordWrap(True)
        lay.addWidget(self.mode_label)
        # Stored on ``self`` rather than handed to the layout alone: PySide6
        # frees a QObject whose last reference goes away, and a freed object
        # takes its event filter with it — keyboard mode would then silently
        # stop receiving keys, exactly as the chrome above documents for the
        # geometry. Parenting (inside KeyboardMode) is the second half of that
        # belt-and-braces.
        self.keyboard_mode = KeyboardMode(self.text, self.mode_label)

        # ---- diff -----------------------------------------------------------
        # The caption, the report and its debounce are a module now
        # (``ui.lyrics_diff``). Its two widgets are added HERE rather than
        # arriving as one container, because ``test_lyrics_studio`` pins
        # ``layout().indexOf(win.diff)`` — and ``indexOf`` only sees a DIRECT
        # child of this layout, so nesting the report would break the stretch
        # below. Kept on ``self`` like the chrome and keyboard mode: PySide6
        # frees a QObject whose last reference goes away, and a freed pane would
        # silently stop taking clicks on its caption.
        self.diff_pane = lyrics_diff.attach(
            self.text, getattr(manager, "config", None)
        )
        lay.addWidget(self.diff_pane.header)
        # ``self.diff`` stays the report widget itself: it is what the window
        # tests reach for, and it is read-only because it reports, not accepts.
        self.diff = self.diff_pane.view
        lay.addWidget(self.diff, 2)

        self.report_label = QLabel("")
        self.report_label.setProperty("muted", True)
        self.report_label.setWordWrap(True)
        lay.addWidget(self.report_label)


        # ---- tools row ------------------------------------------------------
        row = QHBoxLayout()
        self.open_btn = QPushButton("📂 Open…")
        self.open_btn.setToolTip("Load lyrics from a .txt / .lrc file.")
        self.save_btn = QPushButton("💾 Save…")
        self.save_btn.setToolTip("Save this text to a file.")
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setToolTip("Copy the whole text to the clipboard.")
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.setToolTip("Undo the last edit in the editor.")
        self.redo_btn = QPushButton("↷ Redo")
        self.redo_btn.setToolTip("Redo the last undone edit.")
        self.tidy_btn = QPushButton("✨ Tidy")
        self.tidy_btn.setToolTip(
            "Apply the Lyrics tab's tidy rules to this text. The rules live in "
            "one place — see the summary below — so the tab and this window "
            "cannot drift."
        )
        self.track_btn = QPushButton("↺ Use the track's lyrics")
        self.track_btn.setToolTip(
            "Replace the editor with the selected track's saved lyrics."
        )
        self.split_btn = QPushButton("✂ Split long lines")
        self.split_btn.setToolTip(
            "Break over-long lines at word boundaries so a line stays singable "
            "(<= 10 syllables). Carried over from the old Lyrics Editor."
        )
        self.lrc_btn = QPushButton("⬇ Export .lrc…")
        self.lrc_btn.setToolTip("Write this text out as an .lrc file.")
        for w in (self.open_btn, self.save_btn, self.copy_btn,
                  self.undo_btn, self.redo_btn, self.track_btn,
                  self.split_btn, self.lrc_btn, self.tidy_btn):
            row.addWidget(w)
        row.addStretch()
        lay.addLayout(row)

        # ---- write-back row -------------------------------------------------
        actions = QHBoxLayout()
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setProperty("health", "warn")
        actions.addWidget(self.warn_label, 1)
        self.close_btn = QPushButton("Close")
        self.close_btn.setToolTip("Close without writing anything to the track.")
        self.pick_btn = QPushButton("▾")
        self.pick_btn.setToolTip("Send it to the caption instead, or to both.")
        self.pick_btn.setFixedWidth(30)
        self.add_btn = QPushButton("➕ Add to Lyrics")
        self.add_btn.setProperty("role", "primary")
        self.add_btn.setToolTip(
            "Append-or-replace this text in the selected track's lyrics field. "
            "Hold the target picker open with the ▾ button to send it to the "
            "caption instead, or to both."
        )
        actions.addWidget(self.close_btn)
        actions.addWidget(self.pick_btn)
        actions.addWidget(self.add_btn)
        lay.addLayout(actions)

        # ---- wiring ---------------------------------------------------------
        self.open_btn.clicked.connect(self.open_file)
        self.save_btn.clicked.connect(self.save_file)
        self.copy_btn.clicked.connect(self.copy_text)
        self.undo_btn.clicked.connect(self.text.undo)
        self.redo_btn.clicked.connect(self.text.redo)
        self.track_btn.clicked.connect(self.load_track_lyrics)
        self.split_btn.clicked.connect(self.split_long_lines_in_editor)
        self.lrc_btn.clicked.connect(self.export_lrc)
        self.tidy_btn.clicked.connect(self.tidy)
        self.close_btn.clicked.connect(self.reject)
        # ``clicked`` emits a bool; connected straight to the methods that take
        # the destination these wrappers keep that bool out of ``target``.
        self.pick_btn.clicked.connect(self.choose_target_then_export)
        self.add_btn.clicked.connect(self.add_to_lyrics)

        # Live diff, debounced: a diff on every keystroke of a 60-line song is
        # wasted work, and the panel is there to be read, not to keep up with
        # the cursor. The pane owns the timer and its interval; the window owns
        # what a refresh MEANS (which track, which tidy rules, which buttons), so
        # the pane's debounced signal is wired to this window's handler.
        self.diff_pane.changed.connect(self.refresh_diff)

        # Seed from the track so the window opens on the lyrics it will edit. An
        # explicitly-passed sample wins -- the caller named that track -- and
        # otherwise the window adopts whatever the table has selected.
        if sample is not None:
            self._pinned = sample
            self._fill_from(sample)
        else:
            self.load_track_lyrics()

    def retarget(self, sample):
        """Point this window at ``sample``.

        The window is cached and reused, so opening the studio for a different
        track must re-aim the one window rather than build a second. Reloading
        the editor from the new track is the whole point: otherwise the previous
        track's words would be sitting there ready to be written onto this one.
        """
        if sample is not None and sample is self._pinned:
            # Already aimed at this exact track, so leave the editor alone and
            # just re-surface the window. Reloading would throw away text the
            # user has typed but not yet written back, and this is now reachable
            # from a button in the Inspector that is easy to press twice.
            return
        self.sample = sample
        self._pinned = sample
        name = (sample or {}).get("filename", "")
        self.setWindowTitle(f"Lyrics Studio — {name}" if name else "Lyrics Studio")
        self._fill_from(sample)


    # ---- file actions ---------------------------------------------------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Lyrics", "",
            "Text & Lyric Files (*.txt *.lrc);;All Files (*)",
        )
        if not path:
            return
        try:
            self.text.load_file(path)
        except OSError as e:  # noqa: BLE001 — report, never crash the window
            QMessageBox.warning(self, "Could not open", str(e))

    def save_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Lyrics", "", "Text Files (*.txt);;LRC Files (*.lrc)"
        )
        if not path:
            return
        try:
            self.text.save_file(path)
        except OSError as e:  # noqa: BLE001
            QMessageBox.warning(self, "Could not save", str(e))
            return
        self.report_label.setText(f"Lyrics saved to {path}.")
        if self.manager is not None and hasattr(self.manager, "status_label"):
            self.manager.status_label.setText(f"Lyrics saved to {path}.")

    def copy_text(self):
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.text.toPlainText())
        self.report_label.setText("Copied to the clipboard.")

    def split_long_lines_in_editor(self):
        """Break over-long lines at word boundaries (from the old Lyrics Editor)."""
        from modules.lyrics_tools import split_long_lines

        before = self.text.toPlainText()
        after = split_long_lines(before)
        if after == before:
            self.report_label.setText("No line was long enough to split.")
            return
        self.text.setPlainText(after)
        self.report_label.setText(
            "Split long lines. Nothing is saved until you add it."
        )

    def export_lrc(self):
        """Write the editor's text out as an .lrc file."""
        manager = self.manager
        if manager is None or not hasattr(manager, "_export_lyrics"):
            QMessageBox.warning(self, "Not available", "This needs a dataset window.")
            return
        sample = manager.get_selected_sample()
        if sample is None:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        manager._export_lyrics(sample, self.text.toPlainText())

    def tidy(self):
        """Apply the Lyrics tab's tidy rules to the editor buffer (not the track)."""
        from modules.lyrics_normalizer import normalize_lyrics

        source = self.text.toPlainText()
        if not source.strip():
            QMessageBox.information(self, "Nothing to tidy", "The editor is empty.")
            return
        try:
            opts = self.manager._lyrics_options() if self.manager else {}
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Tidy unavailable", str(e))
            return
        new_text, report = normalize_lyrics(source, **opts)
        if new_text == source:
            self.report_label.setText("Tidy changed nothing.")
            return
        self.text.setPlainText(new_text)
        self.refresh_diff()
        self.report_label.setText(
            f"Tidied: {report['lines_changed']} line(s) changed | "
            f"{len(report['contractions'])} contraction(s) mapped | "
            f"{report['apostrophes']} apostrophe(s) stripped | "
            f"{report['tags']} tag(s) capitalized. "
            "Nothing is saved until you use Add to Lyrics."
        )


    # ---- diff + seeding -------------------------------------------------
    def _selected_track(self):
        """The track this window is EDITING -- and therefore writes to.

        The single source of truth for "which song are these words for", used by
        every path: the diff compares against it, the write-back targets it, and
        the transport loads its audio. Resolving the target from
        ``manager.get_selected_sample()`` at the moment of the write instead is
        how one song's words get written onto another -- the main table stays
        live while this non-modal window is open, so the row can move out from
        under the editor between the paste and the click.

        ``_pinned`` is re-set whenever the table's selection really changes
        (``on_manager_selection_changed``), which asks first if the editor holds
        text that has not been added. So the window DOES follow the main panel,
        without ever silently re-aiming a half-written song.
        """
        if self._pinned is not None:
            return self._pinned
        return self.sample

    def _fill_from(self, sample):
        """Load ``sample``'s lyrics into the editor and refresh the diff.

        The single place the editor is seeded, so every path -- first open,
        retarget, and the "use the track's lyrics" button -- goes through the
        same blockSignals dance and cannot drift.
        """
        text = ""
        if sample:
            text = sample.get("formatted_lyrics", sample.get("lyrics", "")) or ""
        self.text.blockSignals(True)
        self.text.setPlainText(text)
        self.text.blockSignals(False)
        # The transport is seeded HERE, in the one place the editor is seeded, so
        # the waveform and the lyrics can never be showing two different songs.
        self.player.set_audio((sample or {}).get("audio_path", ""))
        self.refresh_diff()

    def load_track_lyrics(self):
        """Re-aim the window at the table's CURRENT selection and load it.

        The window follows the selection on its own now
        (``on_manager_selection_changed``), so this button is the manual form of
        the same move: the user's explicit "I mean the track I have selected
        now", and the escape hatch after answering a switch prompt "Keep
        editing" -- at which point the table has moved on and the editor has not.
        """
        sample = self.manager.get_selected_sample() if self.manager else None
        if sample is not None:
            self._pinned = sample
            self.sample = sample
        self._fill_from(self._selected_track())

    # ---- following the main table ---------------------------------------------
    def on_manager_selection_changed(self, sample):
        """Follow the main table's selection, the way the main panel does.

        ``DatasetManager.on_table_selection_changed`` calls this, so the studio
        tracks the row the user clicked -- the same behaviour the Inspector's
        inline lyrics field has, and what makes the window always be showing the
        song being worked on. A non-modal studio could not do this silently: re-
        aiming the editor is how a half-written song gets written onto the wrong
        track, so the window used to stay put until a button was pressed.

        The one difference from the main panel is the editor's buffer, which the
        inline field does not have. If these words have not been added to their
        track yet, the switch ASKS first (``_confirm_switch``) rather than
        dropping them: save-then-switch, discard-and-switch, and stay are all
        offered, which is the same bargain the main window makes when it is
        closed with unsaved work.

        While the window is HIDDEN the re-aim is silent. A modal prompt for a
        window the user cannot see would be a dialog about nothing, and "the
        studio is open on the selected track" is the only sensible meaning of a
        studio that is not open. Nothing is stranded by being silent either:
        unwritten editor text was never attached to a track, so no song is left
        half-saved.
        """
        if self._writing:
            return
        current = self._selected_track()
        if sample is None:
            # Nothing is selected: stop claiming the editor's words belong to a
            # track, so "Add to Lyrics" cannot write them anywhere.
            self._pinned = None
            self.sample = None
            self.player.set_audio("")
            self.refresh_diff()
            return
        if sample is current or (
            (sample or {}).get("filename") == (current or {}).get("filename")
        ):
            return
        if self.isVisible() and self._is_dirty() and not self._confirm_switch(sample):
            self.warn_label.setText(
                f"Still editing '{current.get('filename', '?')}'. The table is on "
                f"'{sample.get('filename', '?')}' — press “↺ Use the track's "
                "lyrics” to move to it (discarding this text), or Add to Lyrics "
                "to keep it."
            )
            return
        self.retarget(sample)

    def _is_dirty(self):
        """True when the editor holds text the aimed track does not have yet.

        Stops two kinds of false alarm: reopening the studio onto the same track
        must ask nothing, and a completed "Add to Lyrics" must leave the window
        clean. Hence the stripped comparison on both sides -- ``apply_target``
        writes the stripped text -- which also mirrors how ``ui.lyrics_diff``
        decides there is nothing to show.
        """
        sample = self._selected_track() or {}
        track = sample.get("formatted_lyrics", sample.get("lyrics", "")) or ""
        return self.text.toPlainText().strip() != track.strip()

    def _confirm_switch(self, sample):
        """Ask before a change of song replaces unwritten text.

        True means go ahead and switch (the caller re-aims the window); False
        means stay on the current song. Saving is offered first and is the
        default because it is the only answer that cannot lose the text. "Keep
        editing" leaves the table and the editor out of step, and the caller says
        so in the warning line rather than pretending the two agree.
        """
        current = self._selected_track()
        can_add = current is not None
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved lyrics")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"'{current.get('filename', '?')}' is still in the editor and has not "
            "been added to that track."
            if can_add
            else "The editor holds text that is not on any track."
        )
        box.setInformativeText(
            f"What should happen before '{sample.get('filename', '?')}' is loaded?"
        )
        add_btn = None
        if can_add:
            add_btn = box.addButton(
                "Add to Lyrics, then switch", QMessageBox.AcceptRole
            )
            box.setDefaultButton(add_btn)
        box.addButton("Discard and switch", QMessageBox.DestructiveRole)
        stay_btn = box.addButton("Keep editing", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if add_btn is not None and clicked is add_btn:
            if self.write_back(LYRICS_TARGET) is False:
                # The dataset file was not written, so the editor is the only
                # copy of these words. Staying put is the safe answer.
                self.warn_label.setText(
                    "The dataset file was not written, so the editor was left "
                    "alone. Use File ▸ Save Dataset, then switch again."
                )
                return False
            return True
        if clicked is None or clicked is stay_btn:
            # Escape or the close box means STAY, never "discard".
            return False
        return True

    def refresh_diff(self):
        """Show editor vs. track, plus a summary of the tidy rules in effect.

        The rendering is the pane's (``ui.lyrics_diff``); the surrounding state
        is the window's, because only the window knows whether a track is
        selected and which of its buttons that enables.
        """
        # The track this window is EDITING, not the table's live row -- the same
        # rule the write-back follows, so the report can never describe a song
        # other than the one the text is about to be written to.
        sample = self._selected_track() if self.manager else None
        if sample is None:
            self.diff_pane.clear()
            self.report_label.setText(
                "No track selected — pick one to compare against."
            )
            self.warn_label.setText("Select a track in the table first.")
            self.add_btn.setEnabled(False)
            self.pick_btn.setEnabled(False)
            return
        self.add_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.warn_label.setText("")

        track_text = sample.get("formatted_lyrics", sample.get("lyrics", "")) or ""
        editor_text = self.text.toPlainText()
        self.diff_pane.render(track_text, editor_text)

        lines = len([ln for ln in editor_text.splitlines() if ln.strip()])
        self.report_label.setText(
            f"{sample.get('filename', '?')} — {lines} non-empty line(s) in the "
            f"editor. Tidy rules in effect: {rule_summary(self.manager)}."
        )

    # ---- write-back actions ---------------------------------------------
    def add_to_lyrics(self):
        """The primary button: straight into the lyrics field, no picker."""
        self.add_to_track(LYRICS_TARGET)

    def choose_target_then_export(self):
        """The ▾ button: ask where it should go, then write."""
        self.add_to_track(None)

    def add_to_track(self, target=None):
        """Write the editor's text into the selected track (undoable).

        ``target`` of ``None`` means "ask": that is the ▾ button. The primary
        button passes ``LYRICS_TARGET`` explicitly, so no picker stands between
        the user and the field the export actually reads.
        """
        manager = self.manager
        if manager is None:
            return
        # The track this window is EDITING, never the table's live selection:
        # ``get_selected_sample()`` can name a different row while a non-modal
        # studio is open, which is how one song's words get written onto another.
        # See ``_selected_track``.
        sample = self._selected_track()
        if sample is None:
            QMessageBox.warning(self, "No Track Selected", "Select a track first.")
            return
        text = self.text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Nothing to add", "The editor is empty.")
            return
        if target is None:
            target = self.ask_target()
            if target is None:
                return
        if not self._confirm(sample, target):
            return

        saved = self.write_back(target)
        if saved is False:
            # In memory, but not on disk. The window stays open and says so:
            # closing it here would leave the user believing the track was saved
            # because the button appeared to work.
            self.warn_label.setText(
                "Added to the track in memory, but the dataset file was not "
                "written (the save was cancelled or failed). Use File ▸ Save "
                "Dataset, then Close."
            )
            return
        name = sample.get("filename", "")
        manager.status_label.setText(
            f"Lyrics added to {target.lower()} for '{name}' "
            + ("— dataset saved." if saved else "(in memory only).")
        )
        self.accept()

    def write_back(self, target):
        """Put the editor's text into the aimed track and save the dataset.

        Split out of ``add_to_track`` because the unsaved-text prompt needs the
        same write WITHOUT the button's closing behaviour: "Add to Lyrics, then
        switch" must write these words and leave the window open on the next
        track.

        Returns True when the dataset file was written, False when a writer
        refused or the save was cancelled, and None when this manager has no
        writer at all (the test stand-in, tooling) -- the caller decides how loud
        to be about each, which is why this shows no dialog of its own.
        """
        manager = self.manager
        sample = self._selected_track()
        text = self.text.toPlainText().strip()
        manager.record_snapshot()
        # ``on_table_selection_changed`` notifies this window back through
        # ``on_manager_selection_changed``, so the follow logic is suppressed for
        # the duration: a write-back is not the user changing songs, and reading
        # it as one would re-enter the unsaved-text prompt from inside its answer.
        self._writing = True
        try:
            apply_target(sample, text, target)
            manager.refresh_table()
            manager.on_table_selection_changed()
        finally:
            self._writing = False
        return self._persist_dataset()

    def _persist_dataset(self):
        """Write the dataset out, so an Add survives closing the app.

        Routed through the manager's OWN ``save_dataset`` -- the one that backs
        the existing file up and renames the new one into place atomically --
        rather than through a second writer here. Two savers would eventually
        disagree about backups, and that backup is the whole reason a bad save is
        survivable. A manager without one returns None: the change is real and
        still in memory, and the caller says so rather than pretending it is on
        disk.
        """
        save = getattr(self.manager, "save_dataset", None)
        if not callable(save):
            return None
        try:
            return bool(save())
        except Exception as e:  # noqa: BLE001 -- the write is already in memory
            QMessageBox.warning(self, "Save Error", str(e))
            return False

    def ask_target(self):
        """The destination picker, behind the ▾ button (was on every export)."""
        choice, ok = QInputDialog.getItem(
            self, "Add to Dataset",
            "Where should this text go?",
            EXPORT_TARGETS, 0, False,
        )
        return choice if ok else None

    def _confirm(self, sample, target):
        """Never silently overwrite a field that already has text in it."""
        clobbered = []
        if target in (CAPTION_TARGET, BOTH_TARGET) and (
            sample.get("caption") or ""
        ).strip():
            clobbered.append("main caption")
        if target in (LYRICS_TARGET, BOTH_TARGET) and (
            (sample.get("formatted_lyrics") or sample.get("lyrics") or "").strip()
        ):
            clobbered.append("lyrics")
        if not clobbered:
            return True
        resp = QMessageBox.question(
            self, "Overwrite?",
            f"'{sample.get('filename', '')}' already has "
            f"{' and '.join(clobbered)}.\n\nReplace it? (Undo is available.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return resp == QMessageBox.Yes


def open_lyrics_studio(manager, sample=None):
    """Open (or re-target) the ONE Lyrics Studio window and return it.

    NOT modal: the studio is one of the app's three real windows, so it stays
    open beside the main window and ``exec()`` is gone. What comes back is the
    WINDOW itself, so a test drives it directly instead of spinning a nested
    event loop.

    The window is cached on the manager, so opening the studio again re-targets
    that one window -- there is never a second copy holding a second track. A
    track is not required: the window doubles as a scratchpad for pasting lyrics
    before there is anything to attach them to, and the write-back buttons
    disable themselves in that case.
    """
    if sample is None and manager is not None:
        sample = manager.get_selected_sample()

    window = getattr(manager, "_lyrics_studio_window", None) if manager is not None else None
    if window is None:
        window = LyricsStudioWindow(manager, sample)
        if manager is not None:
            try:
                manager._lyrics_studio_window = window
            except Exception:  # noqa: BLE001 -- a stand-in manager may refuse it
                pass
    else:
        window.retarget(sample)

    # ``restore()`` re-applies the saved geometry and z-order before the first
    # show; the chrome clamps it onto an AVAILABLE screen, so a window saved on a
    # monitor that has since been unplugged still comes back somewhere reachable.
    window.chrome.restore()
    window.show()
    window.raise_()
    window.activateWindow()
    # Spec item 6 wants ``Enter`` to start keyboard mode, and a key only reaches
    # the editor when the editor has focus. The window is a real OS window now,
    # not a modal sheet that takes focus on open, so relying on "focus follows
    # the click" left keyboard mode looking absent: its hint line was on screen
    # (``KeyboardMode`` sets one on construction) while ``Enter`` did nothing
    # until the user happened to click into the text. This is that other half.
    window.text.setFocus()
    return window


def open_studio_from(manager):
    """Open the Studio, preferring the manager's own entry point.

    ``manager.open_lyrics_editor`` is the established seam -- the Lyrics tab, the
    Edit-tools strip and the Track Inspector all call it, and ``.agent_notes``
    pins that name. Resolving it HERE, at call time, is what keeps those openers
    from drifting into three different windows; a bound method captured at
    construction time would freeze whichever handler existed first.

    The lazy import below is the fallback for a manager built by a test or by
    tooling, which has no such handler.
    """
    opener = getattr(manager, "open_lyrics_editor", None)
    if callable(opener):
        return opener()
    return open_lyrics_studio(manager)


STUDIO_BUTTON_LABEL = "🎼 Open in Lyrics Studio"
STUDIO_BUTTON_TOOLTIP = (
    "Edit this track's lyrics in the Lyrics Studio — a separate window you "
    "can keep open beside this one, with the live diff against the track, "
    "the tidy rules, and the .txt/.lrc import and export tools.\n\n"
    "The field below still works for quick edits."
)


def studio_button(manager, label=None, tooltip=None, attr=None):
    """Build a way in to the Studio, and return it.

    ONE factory for the two surfaces that need a way in — the Track Inspector's
    button and the Lyrics panel's — so the two cannot drift apart in what they
    do. They differ only in wording, which is why ``label`` and ``tooltip`` are
    parameters.

    ``attr`` names the manager attribute the button is stored on, defaulting to
    ``open_studio_btn`` (the Inspector's name). Both buttons stay reachable by
    name, per the ``ui/lyrics_panel.py`` / ``ui/lyrics_tab.py`` convention, and
    they need distinct names because one manager is shared by both pages.

    The button is built by THIS module rather than inline in ``dataset_manager``
    for the owner's architecture rule: new UI is a module, and that file is at
    its 7,390-line budget (``.agent_notes``, spec item 10). The call site is then
    a single ``addWidget``.

    It is ADDITIVE: the Inspector's own ``lyrics_text`` field stays, because a
    one-word fix should not cost a window. Both paths resolve the track through
    the same selection, so neither can write one song's words onto another.
    """
    btn = QPushButton(label or STUDIO_BUTTON_LABEL)
    btn.setToolTip(tooltip or STUDIO_BUTTON_TOOLTIP)
    btn.clicked.connect(lambda: open_studio_from(manager))
    setattr(manager, attr or "open_studio_btn", btn)
    return btn



def rule_summary(manager):
    """One line describing the tab's tidy options, so they cannot be guessed."""
    try:
        opts = manager._lyrics_options()
    except Exception:  # noqa: BLE001 — the summary must never break the window
        return "unavailable"
    on = []
    if opts.get("ing_to_in"):
        on.append("-ing→-in")
    if opts.get("do_capitalize_tags"):
        on.append("tags")
    if opts.get("do_capitalize_lines"):
        on.append("line starts")
    if opts.get("do_strip_punctuation"):
        on.append("punctuation")
    count = len(opts.get("contractions") or {})
    return (
        (", ".join(on) if on else "no optional rules")
        + f", {count} contraction(s)"
    )

