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

Modal, like its predecessors
----------------------------
``open_lyrics_studio`` calls ``exec()``: the window is modal, so the table
cannot change under a half-written song. That matches the two dialogs it
replaces and keeps "one window, one track" true without any single-instance
bookkeeping.
"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

HINT = ("[Double-Click]: UPPERCASE word  |  double-click it again to undo")

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


# The two apostrophes a lyric line can carry: ASCII and the typographic one a
# pasted caption often uses. Both split a word in Qt's eyes.
_APOSTROPHES = "'\u2019"


def _join_apostrophes(doc, start, end):
    """Widen ``[start, end)`` across apostrophes that sit *inside* a word.

    Qt splits ``don't`` into three words (``don``, ``'``, ``t``) and
    ``o'clock`` into two, so a click or a Tab would act on a fragment and the
    taps would drift out of step with the syllables. An apostrophe joins only
    when there is an alphanumeric character on BOTH sides, which is what keeps a
    trailing one (``believin'``) and a quoted one (``rock 'n' roll``) left alone.
    """
    text = doc.toPlainText()
    while (
        end + 1 < len(text)
        and text[end] in _APOSTROPHES
        and text[end + 1].isalnum()
    ):
        end += 1
        while end < len(text) and text[end].isalnum():
            end += 1
    while (
        start >= 2
        and text[start - 1] in _APOSTROPHES
        and text[start - 2].isalnum()
    ):
        start -= 1
        while start > 0 and text[start - 1].isalnum():
            start -= 1
    return start, end


def _words_cursor(doc, start, end):
    """A cursor selecting ``[start, end)``, widened to a full contraction."""
    start, end = _join_apostrophes(doc, start, end)
    cursor = QTextCursor(doc)
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.KeepAnchor)
    return cursor


def _word_under_or_before(cursor):
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
            return _words_cursor(doc, start, end)
    return None

class LyricTextEdit(QPlainTextEdit):
    """The editor with the capitalization gestures wired in.

    Shared by this window and the Track Inspector's inline lyrics field, so the
    gesture is defined once. There is no per-host switch: double-click is always
    on.
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
                cursor = _words_cursor(
                    self.document(), cursor.selectionStart(), cursor.selectionEnd()
                )
            self.toggle_case(cursor)
            return
        word = _word_under_or_before(cursor)
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
        name = (sample or {}).get("filename", "")
        self.setWindowTitle(f"Lyrics Studio — {name}" if name else "Lyrics Studio")

        # Resizable, deliberately: an initial size and a floor, nothing that
        # pins it. ``setMinimumSize`` is the only size constraint; there is no
        # setFixedSize/setMaximumSize, so the window grows to whatever the user
        # drags it to. Min/Max hints give the maximize button something to do.
        self.resize(*INITIAL_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )

        lay = QVBoxLayout(self)

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

        # ---- diff -----------------------------------------------------------
        lay.addWidget(QLabel(
            "<b>Changes (diff)</b> — <span style='color:#c33'>− track</span> / "
            "<span style='color:#3a3'>+ editor</span>"
        ))
        self.diff = QTextEdit()
        # This one IS read-only: it is a report, not an input. The editor above
        # is where text goes.
        self.diff.setReadOnly(True)
        self.diff.setMinimumHeight(120)
        self.diff.setStyleSheet("font-family: monospace;")
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
        # the cursor.
        self._diff_timer = QTimer(self)
        self._diff_timer.setSingleShot(True)
        self._diff_timer.setInterval(220)
        self._diff_timer.timeout.connect(self.refresh_diff)
        self.text.textChanged.connect(self._diff_timer.start)

        # Seed from the track so the window opens on the lyrics it will edit.
        self.load_track_lyrics()


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
        """The track to write to — re-resolved, since selection can change."""
        current = self.manager.get_selected_sample() if self.manager else None
        return current or self.sample

    def load_track_lyrics(self):
        """Seed the editor from the selected track and refresh the diff."""
        sample = self.manager.get_selected_sample() if self.manager else None
        text = ""
        if sample:
            text = sample.get("formatted_lyrics", sample.get("lyrics", "")) or ""
        self.text.blockSignals(True)
        self.text.setPlainText(text)
        self.text.blockSignals(False)
        self.refresh_diff()

    def refresh_diff(self):
        """Show editor vs. track, plus a summary of the tidy rules in effect."""
        from ui.lyrics_tab import unified_diff

        sample = self.manager.get_selected_sample() if self.manager else None
        if sample is None:
            self.diff.setPlainText("")
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
        if editor_text.strip() == track_text.strip():
            self.diff.setPlainText(
                "(editor matches the track's saved lyrics — Add to Lyrics would "
                "be a no-op)"
            )
        else:
            self.diff.setPlainText(
                unified_diff(track_text, editor_text) or "(no line-level change)"
            )

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
        sample = manager.get_selected_sample()
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

        manager.record_snapshot()
        apply_target(sample, text, target)
        manager.refresh_table()
        manager.on_table_selection_changed()
        manager.status_label.setText(
            f"Lyrics added to {target.lower()} for "
            f"'{sample.get('filename', '')}' (undoable)."
        )
        self.accept()

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


def open_lyrics_studio(manager):
    """Open the Lyrics Studio for the selected track, modally.

    Modal on purpose: the table must not change under a half-written song, which
    is exactly what both windows this replaces did. A track is not required — the
    window is also a scratchpad for pasting lyrics before there is anything to
    attach them to, and the write-back buttons disable themselves in that case.
    Returns the dialog result from ``exec()`` so a test can drive it.
    """
    sample = manager.get_selected_sample() if manager else None
    window = LyricsStudioWindow(manager, sample)
    name = (sample or {}).get("filename", "")
    if name:
        window.setWindowTitle(f"Lyrics Studio — {name}")
    return window.exec()



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

