"""The Track Inspector's lyrics surface: the field, its header, and the auto-save.

This was ~320 lines of ``dataset_manager.py`` -- the build block, the debounced
auto-save, the reload path, and the A-/A+ and Expand controls. They moved here
because the owner's rule is that new UI is a MODULE and that file is at its line
budget. The behaviour did not change: the subsystem moved, and every name the
tests and the rest of the manager reach for is still an attribute ON THE MANAGER,
because :func:`build_lyrics_surface` assigns them there rather than keeping them
private to a class. That is the convention ``ui/lyrics_panel.py`` already uses
(``manager.lyrics_panel_summary``), and it is why the extraction needed no shim
layer and changed no call site.

The four guarantees this surface owes the user
----------------------------------------------
The old field was gated behind an "Edit mode" tick and a Commit button. Removing
that gate did not remove the safety it bought; it moved it here, into four
guarantees that are all stronger than "you pressed Commit":

1. **No write while a selection change is in flight.** Loading a track refills
   the field; a debounce timer that survived the refill would write the OLD
   track's text onto the NEW one.
2. **The field must still belong to the track it was loaded for.**
   ``manager._lyrics_selected_filename`` is recorded at load time and compared at
   write time. If they disagree the write is DROPPED, not resolved -- the user's
   text is still on screen, and re-selecting the original track saves it.
3. **Snapshot before the write**, so Ctrl+Z restores the previous lyrics, and the
   song is backed up once per session before its first change.
4. **Nothing happens if the text is unchanged**, so selecting a track and clicking
   away creates no snapshot, no backup and no dirty flag.

The debounce is 400 ms: every save snapshots the undo stack and writes the
dataset, so it must not fire mid-burst.

State lives on the MANAGER, not in a class here
-----------------------------------------------
``manager._lyrics_save_timer``, ``manager._lyrics_selected_filename`` and
``manager._suppress_lyrics_save`` are created here but stored on the manager, with
no property wrapping them. ``tests/test_lyrics_inline_editor.py`` reads all three
by name; the manager is their single writer either way, so a property would buy
nothing and cost a layer.

What stays in the manager
-------------------------
``manager._lyrics_hidden_on_expand`` and ``manager.on_lyrics_expand_toggled`` stay in
``dataset_manager``: that list also holds the caption widgets, so "Expand" hides
the caption block too and is an inspector-layout concern, not a lyrics one. The
handler delegates only the field's height and fill (:meth:`LyricsSurface.set_expanded`).
``manager._lyrics_gate_open`` and ``manager.update_lyric_limit_label`` stay as
well -- the token-limit gate guards the field, and the limit label is shared with
the Lyrics panel and the caption.
"""

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
)

# The debounce for the auto-save: every save snapshots the undo stack and writes
# the dataset, so it should not fire mid-burst.
SAVE_DEBOUNCE_MS = 400

# The field's height when "Expand" is ticked / unticked.
EXPANDED_HEIGHT = 520
COLLAPSED_HEIGHT = 170


def set_layout_item_visible(item, visible):
    """Hide/show a widget, a layout, or a plain layout object.

    ``QLayout`` has no ``setVisible``, so a nested layout is hidden by hiding the
    widgets it owns. Lives here rather than as a static method on the manager
    because the only caller is the Expand toggle, which is a visibility control.
    """
    if hasattr(item, "setVisible"):
        item.setVisible(visible)
    elif hasattr(item, "count"):
        for i in range(item.count()):
            sub = item.itemAt(i)
            if sub is None:
                continue
            inner = sub.widget() or sub.layout()
            if inner is not None:
                set_layout_item_visible(inner, visible)


# The two apostrophes a lyric line can carry: ASCII and the typographic one a
# pasted caption often uses. Both split a word in Qt's eyes.
APOSTROPHES = "'\u2019"


def join_apostrophes(text, start, end):
    """Widen ``[start, end)`` across apostrophes that sit *inside* a word.

    Qt splits ``don't`` into three words (``don``, ``'``, ``t``) and
    ``o'clock`` into two, so a double-click would act on a fragment. An
    apostrophe joins only when there is an alphanumeric character on BOTH sides,
    which is what keeps a trailing one (``believin'``) and a quoted one
    (``rock 'n' roll``) left alone.
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


class LyricTextEdit(QPlainTextEdit):
    """The lyrics field, with the shouted-word gesture wired in.

    **Double-click a word to UPPERCASE it** (the belting / shouted delivery
    mark); double-click it again to undo. There is no mode switch and no second
    host: this is the field on the main page, and the gesture is defined once
    here.

    ``Tab`` is always Qt's focus key, never a literal tab. In a real host there
    is a focusable sibling, so Tab moves focus instead of inserting whitespace.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(13)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
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
        super().mouseDoubleClickEvent(event)
        self.toggle_word_at_cursor()

    # ---- file helpers ---------------------------------------------------
    def load_file(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            self.setPlainText(fh.read())

    def save_file(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.toPlainText().strip())


def build_lyrics_surface(manager, insp_layout):
    """Build the lyrics field, its header, its status row and its auto-save.

    Adds everything to ``insp_layout``, assigns the widgets and the save state
    onto ``manager`` under the names the rest of the app already uses, and
    returns the :class:`LyricsSurface` controller (also stored as
    ``manager._lyrics_field``), which the manager's methods delegate to.
    """
    from ui.appearance_panel import build_font_size_spin
    from ui.themes import (
        FONT_REGION_LYRICS,
        FONT_REGION_PROPERTY,
        LYRICS_FONT_SIZE_MAX,
    )

    lyrics_head = QHBoxLayout()
    lyrics_head.addWidget(QLabel("<b>Formatted Lyrics / Vocal Markers:</b>"))
    # Text-size steppers, one notch per click. They replace the old "Capitalize
    # mode" checkbox 1:1 -- that switch is gone because double-click now always
    # marks a word shouted (see LyricTextEdit above), so the space it occupied is
    # free for the control users actually asked for here: making the lyrics
    # readable at a glance. They are a coarse adjustment of the SAME
    # ``lyrics_font_size`` key the Settings row and the numeric "Size:" spinbox
    # own; ``ui.appearance_panel.sync_font_size_controls`` keeps every copy
    # showing one number, so there is still one source of truth. The header is
    # never hidden by "Expand" (the Expand checkbox itself lives here), so both
    # the value and the steppers stay reachable in the mode where the text is
    # biggest.
    manager.lyrics_text_smaller_btn = QPushButton("A−")
    manager.lyrics_text_smaller_btn.setToolTip("Make the lyrics text smaller.")
    manager.lyrics_text_smaller_btn.setFixedWidth(30)
    manager.lyrics_text_smaller_btn.clicked.connect(
        lambda: manager.on_lyrics_font_step(-1))
    manager.lyrics_text_larger_btn = QPushButton("A+")
    manager.lyrics_text_larger_btn.setToolTip(
        "Make the lyrics text larger — up to "
        f"{LYRICS_FONT_SIZE_MAX} px, then the text wraps in this box. "
        "\"⤢ Expand\" gives it more height to wrap into."
    )
    manager.lyrics_text_larger_btn.setFixedWidth(30)
    manager.lyrics_text_larger_btn.clicked.connect(
        lambda: manager.on_lyrics_font_step(1))
    manager.lyrics_expand_check = QCheckBox("⤢ Expand")
    manager.lyrics_expand_check.setToolTip(
        "Grow the field to most of the inspector and hide the fields below it."
    )
    lyrics_head.addStretch()
    lyrics_head.addWidget(QLabel("Size:"))
    lyrics_head.addWidget(manager.lyrics_text_smaller_btn)
    lyrics_head.addWidget(manager.lyrics_text_larger_btn)
    lyrics_head.addWidget(manager.lyrics_expand_check)
    insp_layout.addLayout(lyrics_head)

    manager.lyrics_text = LyricTextEdit()
    # Puts the field under build_stylesheet()'s lyrics-size rule, so it can grow
    # independently of the rest of the app. Not QWidget.setFont(): the app
    # stylesheet overrides a per-widget font on every theme re-apply (measured —
    # see ui/themes.FONT_REGION_PROPERTY).
    manager.lyrics_text.setProperty(FONT_REGION_PROPERTY, FONT_REGION_LYRICS)
    manager.lyrics_text.setPlaceholderText(
        "[Intro]\n[Verse 1]\nLyrics...\n[Chorus]...")
    # Always editable, and the write is debounced rather than gated behind a
    # Commit button (see the module docstring). ``textChanged`` only restarts the
    # timer; ``_flush_lyrics_edit`` does the actual write after typing stops.
    manager.lyrics_text.setReadOnly(False)
    manager.lyrics_text.textChanged.connect(manager.on_lyrics_edited)
    insp_layout.addWidget(manager.lyrics_text)

    # Which track the field is showing, so an auto-save cannot write the text of
    # one track onto another after the selection moves.
    manager._lyrics_selected_filename = None
    # True only while ``load_from_selection`` is refilling the box. The load
    # suppresses the signal too, but a repaint can still flush a queued timer;
    # this flag makes the debounced write a no-op during a load no matter how it
    # was reached.
    manager._suppress_lyrics_save = False
    # The debounced auto-save. Single-shot and restarted on every keystroke, so it
    # fires once, SAVE_DEBOUNCE_MS after typing stops.
    manager._lyrics_save_timer = QTimer(manager)
    manager._lyrics_save_timer.setSingleShot(True)
    manager._lyrics_save_timer.setInterval(SAVE_DEBOUNCE_MS)
    manager._lyrics_save_timer.timeout.connect(manager._flush_lyrics_edit)

    lyrics_actions = QHBoxLayout()
    # The Commit button is gone; so is the old "Revert" that discarded an
    # uncommitted buffer, because a live-save field has no buffer to discard. What
    # is left is a reload — put the dataset's saved copy back in the field, which
    # is the one gesture the auto-save cannot express (Ctrl+Z steps back one edit,
    # not all of them).
    manager.lyrics_save_status = QLabel("")
    manager.lyrics_save_status.setProperty("muted", True)
    manager.lyrics_save_status.setToolTip(
        "Lyrics save automatically, 400 ms after you stop typing. Every save "
        "is undoable (Ctrl+Z) and the pre-edit copy of each song is kept in "
        "<dataset>/_Backup/songs/."
    )
    manager.lyrics_discard_btn = QPushButton("↺ Revert to saved")
    manager.lyrics_discard_btn.setToolTip(
        "Reload this track's saved lyrics from the dataset, discarding the "
        "text in the field."
    )
    manager.lyrics_discard_btn.clicked.connect(manager.on_lyrics_revert)
    lyrics_actions.addWidget(manager.lyrics_save_status)
    lyrics_actions.addStretch()
    lyrics_actions.addWidget(manager.lyrics_discard_btn)
    # Placed after the stretch so it is the rightmost control on this row.
    # "Expand" hides ``_lyrics_hidden_on_expand`` and this row is not in that list,
    # so the size control stays available while the field is tall.
    manager.lyrics_font_spin_inline = build_font_size_spin(
        "Text size for the lyrics fields. Auto matches the rest of the app; "
        "raise it to make this box easier to read at a distance.",
        manager.config.get("lyrics_font_size"),
        maximum=LYRICS_FONT_SIZE_MAX,
    )
    manager.lyrics_font_spin_inline.setFixedWidth(84)
    manager.lyrics_font_spin_inline.valueChanged.connect(
        manager.on_lyrics_font_size_changed)
    lyrics_actions.addWidget(QLabel("Size:"))
    lyrics_actions.addWidget(manager.lyrics_font_spin_inline)
    insp_layout.addLayout(lyrics_actions)

    surface = LyricsSurface(manager)
    manager._lyrics_field = surface
    return surface


class LyricsSurface:
    """The behaviour behind the Inspector's lyrics field.

    Deliberately thin: every widget and every piece of save state lives on the
    ``manager`` (see the module docstring), so this class holds no attributes
    except its manager handle and exists to give the manager's methods one
    delegate each instead of 300 lines of inline body.
    """

    def __init__(self, manager):
        self.manager = manager

    # -- loading ------------------------------------------------------------

    def load_from_selection(self):
        """Refill the field from the selected track, with signals suppressed.

        The save timer is stopped and a suppression flag set for the duration:
        without them, the ``setPlainText`` below would arm the debounced write and
        400 ms later "save" the text we are loading — harmless-looking, but it
        would snapshot the undo stack on every mere selection change.
        """
        m = self.manager
        if m._lyrics_save_timer.isActive():
            m._lyrics_save_timer.stop()
        s = m.get_selected_sample()
        text = ""
        if s:
            text = s.get("formatted_lyrics", s.get("lyrics", "")) or ""
            m._lyrics_selected_filename = s.get("filename")
        else:
            m._lyrics_selected_filename = None
        m._suppress_lyrics_save = True
        try:
            m.lyrics_text.blockSignals(True)
            m.lyrics_text.setPlainText(text)
            m.lyrics_text.blockSignals(False)
        finally:
            m._suppress_lyrics_save = False
        m.lyrics_save_status.setText("Saved" if text else "")
        m.update_lyric_limit_label()

    def revert(self):
        """Put the field back to the selected track's saved lyrics.

        Called when the token-limit notice is dismissed, so a declined gate
        leaves no half-typed text that a later keystroke could flush.
        """
        m = self.manager
        sample = m.get_selected_sample()
        text = ""
        if sample:
            text = sample.get("formatted_lyrics", sample.get("lyrics", "")) or ""
        if m._lyrics_save_timer.isActive():
            m._lyrics_save_timer.stop()
        m._suppress_lyrics_save = True
        try:
            m.lyrics_text.blockSignals(True)
            m.lyrics_text.setPlainText(text)
            m.lyrics_text.blockSignals(False)
        finally:
            m._suppress_lyrics_save = False
        m.lyrics_save_status.setText(
            "Not saved — the token-limit notice was closed."
        )
        m.update_lyric_limit_label()

    def reload(self):
        """Discard the field's text and reload the track's saved lyrics.

        Everything is already committed by the time this runs — the auto-save
        writes 400 ms after each pause — so this only refreshes the field, it does
        not undo anything. To undo a save, use Ctrl+Z.
        """
        self.load_from_selection()
        self.manager.status_label.setText(
            "Lyrics reloaded from the selected track.")

    # -- editing and the debounced write ------------------------------------

    def on_edited(self):
        """Restart the auto-save clock. Does NOT write the dataset.

        Writing on every keystroke would snapshot the undo stack (and back up the
        song) once per character, which is the bug the old Commit button was added
        to fix. Instead each change just re-arms the single-shot timer; 400 ms
        after the typing stops, :meth:`flush` writes once.

        The signals are still suppressed while the field is being refilled by
        :meth:`load_from_selection`, so loading a track never counts as an edit.
        """
        m = self.manager
        if m._suppress_lyrics_save:
            return
        # Gate the first edit behind the token-limit notice. Declining restores
        # the track's saved text instead of leaving an unflushed buffer that a
        # later keystroke could write.
        if not m._lyrics_gate_open():
            self.revert()
            return
        m.update_lyric_limit_label()
        if m._lyrics_save_timer.isActive():
            m._lyrics_save_timer.stop()
        m._lyrics_save_timer.start()
        m.lyrics_save_status.setText("Unsaved changes…")

    def flush(self):
        """Write the field to its track. The debounced half of :meth:`on_edited`.

        Four guards, in order, all of them cheap and all of them load-bearing:

        1. **No dataset write while a selection change is in flight.** The table
           re-selecting a track refills this field; a timer that survived that
           would write the OLD track's text onto the NEW one.
        2. **The field must still belong to the selected track.** Recorded at load
           time in ``_lyrics_selected_filename``. If they disagree, drop the
           pending write rather than pick a winner — the user's text is still on
           screen, and re-selecting the original track re-arms and saves it.
        3. **Nothing to do if the text is unchanged.** Selecting a track and
           clicking away must not create a snapshot, a backup, or a dirty flag.
        4. **Snapshot before the write**, so Ctrl+Z restores the old lyrics, and
           back the song up once, before the first change of the session.
        """
        m = self.manager
        if m._suppress_lyrics_save:
            return
        sample = m.get_selected_sample()
        if not sample:
            m.lyrics_save_status.setText("No track selected — not saved.")
            return
        current = sample.get("filename")
        if (m._lyrics_selected_filename is not None
                and current != m._lyrics_selected_filename):
            # The selection moved on. Refuse, exactly as Commit used to.
            m.lyrics_save_status.setText(
                f"Not saved — this is '{m._lyrics_selected_filename}', but "
                f"'{current}' is selected."
            )
            return
        text = m.lyrics_text.toPlainText()
        previous = sample.get("formatted_lyrics") or ""
        if text == previous:
            m.lyrics_save_status.setText("Saved")
            return
        m._backup_song_once(sample)
        m.record_snapshot()
        sample["formatted_lyrics"] = text
        sample["lyrics"] = text
        # The table caches each track's token count for the red "over limit"
        # mark; this edit invalidates it so the next refresh_table recomputes.
        sample.pop("_lyric_tokens", None)
        # Preserve what was there BEFORE this edit, once. ``raw_lyrics`` is the
        # pre-tidy text the Lyrics tab tidies from, so filling it with the old
        # value (not the new one) is what keeps a tidy pass re-runnable and the
        # original recoverable after an auto-save has already overwritten it.
        if not sample.get("raw_lyrics") and previous:
            sample["raw_lyrics"] = previous
        m._write_lyrics_sidecar_for(sample)
        m.lyrics_save_status.setText("Saved ✓")
        m.update_lyric_limit_label()

    # -- size and layout ----------------------------------------------------

    def step_font(self, delta):
        """Nudge the lyrics size one notch (the A−/A+ buttons on the header).

        They step the SAME ``lyrics_font_size`` key the numeric "Size:" readout
        and the Settings "Lyrics size" row own — the value is written through
        ``set_appearance``, which re-applies the theme and re-syncs every copy, so
        the buttons cannot disagree with the spinner beside them. Two details that
        would otherwise make the buttons feel broken:

        * The starting point is the size the field is ACTUALLY using, not the raw
          stored value. While the setting is Auto the stored value is 0, so
          stepping from it would try to set 1 px; Auto means "inherit the app
          size", so the first press starts from that inherited size (13 -> 14).
        * The clamp uses the lyrics ceiling, ``LYRICS_FONT_SIZE_MAX``, so the
          buttons reach the whole range the spinner offers instead of stopping at
          the app-wide 32.
        """
        from ui.appearance_panel import set_appearance
        from ui.themes import FONT_SIZE_MIN, LYRICS_FONT_SIZE_MAX, resolve_font_sizes

        m = self.manager
        base, lyrics = resolve_font_sizes(m.config)
        current = lyrics or base
        value = max(FONT_SIZE_MIN, min(LYRICS_FONT_SIZE_MAX, current + delta))
        set_appearance(m, "lyrics_font_size", value)

    def font_size_changed(self, value):
        """Apply the lyrics size chosen by the field's own Size control.

        The label on that control is "Size:"; it writes the SAME config key the
        Settings "Lyrics size" row owns, so there is one setting and no second
        source of truth. Re-applying the theme is what moves the text — the size
        reaches the field through the stylesheet, not through setFont.
        """
        from ui.appearance_panel import set_appearance

        set_appearance(self.manager, "lyrics_font_size", value)

    def set_expanded(self, checked):
        """Let the field take the inspector's height, or put it back.

        Hiding the widgets that sit above the field is the manager's
        ``on_lyrics_expand_toggled``: that list holds the CAPTION block, the
        health line and the lock bar, so it is inspector-layout state, not
        lyrics state. What is left here is the field itself:

        * **checked** — growing policy plus the tall minimum, so the field fills
          the space the caption and the fields below used to occupy and its first
          line lands at the TOP of the inspector, directly under the waveform's
          column chrome. The inspector is scrolled to the top for the same
          reason: a stale scroll offset would leave the field looking unmoved.
        * **unchecked** — the normal height and policy, so the caption and the
          metadata fields come back exactly as they were.

        No second window: the field already lives inside a scroll area, so
        giving it height and hiding its neighbours is all "expanding" needs.
        """
        field = self.manager.lyrics_text
        if checked:
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            field.setMinimumHeight(EXPANDED_HEIGHT)
        else:
            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            field.setMinimumHeight(COLLAPSED_HEIGHT)
        self._scroll_inspector_to_top()

    def _scroll_inspector_to_top(self):
        """Put the inspector's scroll area back at the top.

        The lyrics field and the caption live in the same scroll area. After
        Expand hides the caption, keeping the previous scroll offset would leave
        the field visually where it was, defeating "move up to just under the
        waveform". Climbing to the nearest ``QScrollArea`` keeps this module from
        needing a reference the manager would have to hold.
        """
        from PySide6.QtWidgets import QScrollArea

        parent = self.manager.lyrics_text.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            parent.verticalScrollBar().setValue(0)

    # -- sidecar ------------------------------------------------------------

    def write_sidecar(self, sample):
        """Write ``<stem>_lyrics.txt`` beside the track's audio, if it has one."""
        from ui.tidy_panel import _write_lyrics_sidecar

        return _write_lyrics_sidecar(self.manager, sample)
