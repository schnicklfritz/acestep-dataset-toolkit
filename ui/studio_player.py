"""The Lyrics Studio's transport: the player and its waveform, at the top.

Spec (the owner's, recorded in ``.agent_notes.md`` -> "The Lyrics Studio spec"):
item 7 is *"the player with waveform sits at the top of the window"*, and item 8
is the key table --

    ``d`` play (and stop), ``f`` ahead, ``a`` back, ``e`` marker, ``=`` volume
    up, ``-`` volume down, ``s`` pause. Pause resets the marker to the pause
    point; stop returns to the last marker (Reaper style). ONE marker.

Why a SECOND media player, not the Dataset Studio's
---------------------------------------------------
The main window's player is not a transport, it is a *view of the selection*:
``DatasetManager._load_track_preview`` re-points it every time the track table
changes rows, and its play/seek/time widgets are wired into that window's own
layout. Two surfaces sharing one player would fight -- whichever window last
changed the selection would steal the source from the other, and a nudge here
would be undone by the next selection change in the main window. The Studio is a
non-modal second window (it stays open beside the main one), so it owns its own
player and its own output device. Same reasoning that put the diff pane and
keyboard mode in their own modules instead of borrowing the main window's.

Reaper semantics, and why the keys are deliberately asymmetric
--------------------------------------------------------------
* ``d`` -- play if stopped, **stop** if already playing. A stop *returns the
  playhead to the last marker*, which is what makes ``e`` + ``d`` a loop over one
  section.
* ``s`` -- **pause**, and pausing *moves the marker to the pause point*. So ``s``
  then ``d`` replays from exactly where you stopped listening: the "hear that bit
  again" workflow. Note this is the opposite of what a plain pause does.
* ``e`` -- set the marker at the playhead, or clear it if one is already set.
* ``f`` / ``a`` -- nudge the playhead forward / back by ``NUDGE_MS``.
* ``=`` / ``-`` -- output **volume**, not the playhead.
ONE marker, so "the last marker" always means the one ``e``/``s`` last set; there
is no marker list to disambiguate.

Where the keys apply, and where they deliberately do not
-------------------------------------------------------
The Studio's editor is a text field and ``d``/``f``/``a``/``s``/``e`` are
letters. A globally-grabbed ``d`` would make the letter ``d`` untypeable, which
is a non-starter in a lyrics editor -- so the key table is claimed by an event
filter on the WINDOW (``attach_keys``) and ONLY while the focused widget is not a
text input. Type in the editor and the letters are letters; click the waveform,
a transport button, or the track table and the transport keys work. This is the
same shape of filter, for the same reason, as
``ui.lyrics_keyboard_mode.KeyboardMode`` -- which is also why the editor keeps
plain ``Enter`` and the transport never claims it.
"""

import os

from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from widgets import WaveformWidget

# One f/a press moves the playhead this far. Reaper nudges by a configurable
# step; a second is the smallest step that is useful for finding a phrase by ear
# without overshooting it, and coarse enough that the move is audible.
NUDGE_MS = 1000

# Volume is adjusted in these steps and starts where the main window's player
# starts (``dataset_manager`` calls ``setVolume(0.7)``), so the two do not
# disagree about what "normal" sounds like.
VOLUME_STEP = 0.05
DEFAULT_VOLUME = 0.7

# settings.json key for the remembered volume. One key, written through the same
# ``save_plain_keys`` path the diff pane's collapsed state and the window
# geometry use, so a preference never triggers a full config save.
VOLUME_KEY = "lyrics_studio_volume"

# What counts as "the user is typing". ``QPlainTextEdit`` is NOT a ``QTextEdit``
# subclass (both derive from ``QAbstractScrollArea``), so it has to be named
# separately or the editor would lose its letters to the transport.
TEXT_INPUTS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)

TRANSPORT_HINT = (
    "🎧 Click the player, then: <b>d</b> play/stop · <b>s</b> pause · "
    "<b>f</b>/<b>a</b> nudge · <b>e</b> marker · <b>=</b>/<b>−</b> volume. "
    "Stop returns to the marker; pause sets it here. "
    "(In the editor these keys stay letters.)"
)


class MarkerWaveform(WaveformWidget):
    """The main window's waveform plus this window's ONE marker line.

    A subclass rather than an edit to ``widgets.WaveformWidget``: the marker
    belongs to the Studio's transport, and the main window's waveform -- which
    draws only a playhead -- must not grow a second line it never sets. The
    playhead itself is the inherited one: ``set_position_frac``.
    """

    # Class-level default, not merely an ``__init__`` assignment. PySide6 clears
    # an object's instance attributes once its C++ side is destroyed while Qt can
    # still paint it, and ``None`` here makes that a harmless no-op. (The same
    # teardown trap ``ui/lyrics_keyboard_mode`` documents for its event filter,
    # and ``ui/window_chrome`` for its geometry handler.)
    marker_frac = None

    def set_marker_frac(self, frac):
        """Place the marker at ``frac`` of the track, or ``None`` to clear it."""
        self.marker_frac = None if frac is None else max(0.0, min(1.0, float(frac)))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.marker_frac is None:
            return
        p = QPainter(self)
        x = int(self.marker_frac * self.width())
        p.setPen(QPen(QColor(255, 196, 0), 2))
        p.drawLine(x, 0, x, self.height())
        p.end()


def _default_saver():
    from modules.config_store import save_plain_keys

    return save_plain_keys


class StudioPlayer(QWidget):
    """Waveform + transport for the Lyrics Studio's own track.

    Owns the one marker, the output volume and the key table. Knows nothing about
    the dataset: the window hands it an audio path (``set_audio``) and that is the
    whole interface.
    """

    # Class-level defaults for the same teardown reason as ``MarkerWaveform``:
    # a key event can still arrive after the C++ side is gone.
    media_player = None
    audio_output = None
    marker_ms = None
    duration_ms = 0
    _volume = DEFAULT_VOLUME
    _path = ""
    _config = None
    _saver = None

    def __init__(self, config=None, saver=None, parent=None):
        super().__init__(parent)
        # A throwaway dict when there is no config (a manager built by tests),
        # matching ``ui.lyrics_diff.LyricsDiffPane`` and ``WindowChrome``.
        self._config = config if config is not None else {}
        self._saver = saver
        self.marker_ms = None
        self.duration_ms = 0
        self._path = ""

        saved = self._config.get(VOLUME_KEY)
        self._volume = (
            float(saved)
            if isinstance(saved, (int, float)) and 0.0 <= float(saved) <= 1.0
            else DEFAULT_VOLUME
        )

        # Focusable, and the only widget here that claims the keys directly: the
        # buttons below are NoFocus so a click on one does not silently take the
        # keys away from the transport (the next press would then go nowhere).
        self.setFocusPolicy(Qt.StrongFocus)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ d")
        self.play_btn.setToolTip(
            "Play, or STOP if already playing. A stop returns the playhead to "
            "the marker — so `e` then `d` loops one section.  [d]"
        )
        self.play_btn.clicked.connect(self.toggle_play)
        self.pause_btn = QPushButton("⏸ s")
        self.pause_btn.setToolTip(
            "Pause and REMEMBER the spot: pausing moves the marker here, so "
            "play then resumes from it.  [s]"
        )
        self.pause_btn.clicked.connect(self.pause)
        self.back_btn = QPushButton("⏪ a")
        self.back_btn.setToolTip("Nudge the playhead back one second.  [a]")
        self.back_btn.clicked.connect(lambda: self.nudge(-NUDGE_MS))
        self.fwd_btn = QPushButton("⏩ f")
        self.fwd_btn.setToolTip("Nudge the playhead forward one second.  [f]")
        self.fwd_btn.clicked.connect(lambda: self.nudge(NUDGE_MS))
        self.marker_btn = QPushButton("◆ e")
        self.marker_btn.setToolTip(
            "Set the marker at the playhead, or clear it if one is already set. "
            "ONE marker only — a second press moves it.  [e]"
        )
        self.marker_btn.clicked.connect(self.toggle_marker)

        self.vol_down_btn = QPushButton("−")
        self.vol_down_btn.setToolTip("Volume down  [−]")
        self.vol_down_btn.clicked.connect(lambda: self.change_volume(-VOLUME_STEP))
        self.vol_up_btn = QPushButton("=")
        self.vol_up_btn.setToolTip("Volume up  [=]")
        self.vol_up_btn.clicked.connect(lambda: self.change_volume(VOLUME_STEP))

        for b in (self.play_btn, self.pause_btn, self.back_btn, self.fwd_btn,
                  self.marker_btn, self.vol_down_btn, self.vol_up_btn):
            b.setFocusPolicy(Qt.NoFocus)
            row.addWidget(b)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setFocusPolicy(Qt.NoFocus)
        self.seek_slider.setToolTip("Scrub the track.")
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        row.addWidget(self.seek_slider, 1)

        self.time_label = QLabel("0:00 / 0:00")
        self.volume_label = QLabel(f"🔊 {int(round(self._volume * 100))}%")
        self.marker_label = QLabel("◆ —")
        self.marker_label.setToolTip("The marker's position, or — when none is set.")
        row.addWidget(self.time_label)
        row.addWidget(self.volume_label)
        row.addWidget(self.marker_label)
        lay.addLayout(row)

        self.waveform = MarkerWaveform()
        self.waveform.set_audio(None)
        lay.addWidget(self.waveform)

        self.hint = QLabel(TRANSPORT_HINT)
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        # Playback is best-effort, exactly as in the main window: a build without
        # QtMultimedia must still open the Studio (the editor, the diff and the
        # tidy tools are the window's job; the player is a convenience), so the
        # failure is reported in the hint instead of raised.
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self.media_player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.media_player.setAudioOutput(self.audio_output)
            self.audio_output.setVolume(self._volume)
            self.media_player.positionChanged.connect(self._on_position)
            self.media_player.mediaStatusChanged.connect(self._on_status)
        except Exception as e:  # noqa: BLE001 -- playback is best-effort
            print(f"studio media player unavailable: {e}")
            self.media_player = None
            self.audio_output = None
            self.hint.setText(
                "🎧 Playback is unavailable in this build "
                f"({e}). The waveform and the editor still work."
            )

    # ---- loading ---------------------------------------------------------
    def set_audio(self, path):
        """Point the player at ``path`` (or nothing) and clear the marker.

        Called from the window's single seeding path, so following the selection
        and opening the window cannot disagree about which track is loaded.
        """
        self._path = path or ""
        valid = bool(self._path) and os.path.exists(self._path)
        # A new track means a new timeline: the old marker would be naming a
        # second that is unrelated to the audio now loaded.
        self.marker_ms = None
        self.duration_ms = 0
        self.waveform.set_audio(self._path if valid else None)
        self.waveform.set_marker_frac(None)
        self._refresh_marker_label()
        self.play_btn.setText("▶ d")
        if self.media_player is None:
            return
        self.media_player.stop()
        if valid:
            self.media_player.setSource(QUrl.fromLocalFile(self._path))
        else:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(0)
            self.seek_slider.blockSignals(False)
            self.time_label.setText("0:00 / 0:00")
            self.waveform.set_position_frac(0.0)

    # ---- transport (what both the buttons and the keys call) -------------
    def toggle_play(self):
        """``d``: play, or stop-and-return-to-the-marker when already playing."""
        if self.media_player is None:
            self._no_playback()
            return
        playing = (
            self.media_player.playbackState()
            == self._playback_enum().PlayingState
        )
        if playing:
            self.stop()
        else:
            self.media_player.play()
            self.play_btn.setText("⏹ d")

    def stop(self):
        """Stop and put the playhead back on the marker (Reaper style)."""
        if self.media_player is None:
            return
        self.media_player.stop()
        self.play_btn.setText("▶ d")
        self.seek_to(self.marker_ms if self.marker_ms is not None else 0)

    def pause(self):
        """``s``: pause AND remember the spot — the marker moves here."""
        if self.media_player is None:
            self._no_playback()
            return
        paused_at = self._position_ms()
        self.media_player.pause()
        self.play_btn.setText("▶ d")
        # Reaper's "pause resets the marker to the pause point": the marker
        # becomes a bookmark you did not have to ask for, so play resumes from
        # where you stopped listening.
        self._set_marker(paused_at)

    def nudge(self, delta_ms):
        """``f``/``a``: move the playhead without touching the marker."""
        if self.media_player is None:
            self._no_playback()
            return
        self.seek_to(self._position_ms() + delta_ms)

    def toggle_marker(self):
        """``e``: set the marker here, or clear it if one is already set."""
        if self.marker_ms is None:
            self._set_marker(self._position_ms())
        else:
            self._set_marker(None)

    def change_volume(self, delta):
        """``=``/``-``: output volume, remembered across sessions."""
        self._volume = max(0.0, min(1.0, self._volume + delta))
        if self.audio_output is not None:
            self.audio_output.setVolume(self._volume)
        self.volume_label.setText(f"🔊 {int(round(self._volume * 100))}%")
        self._persist_volume()

    def _no_playback(self):
        self.hint.setText(
            "🎧 No audio output available for this track/build, so there is "
            "nothing to move. The waveform and the editor still work."
        )

    @staticmethod
    def _playback_enum():
        """``QMediaPlayer.PlaybackState`` lazily.

        The import is local for the same reason ``ui.lyrics_diff`` imports
        ``unified_diff`` locally and the media player itself is built in a
        ``try``: a build with no QtMultimedia must still import this module and
        open the window.
        """
        from PySide6.QtMultimedia import QMediaPlayer

        return QMediaPlayer.PlaybackState

    # ---- marker ----------------------------------------------------------
    def _set_marker(self, ms):
        self.marker_ms = None if ms is None else max(0, int(ms))
        self._refresh_marker_label()

    def _refresh_marker_label(self):
        if self.marker_ms is None:
            self.marker_label.setText("◆ —")
            self.marker_btn.setText("◆ e")
            self.waveform.set_marker_frac(None)
            return
        total = self.duration_ms or self._duration_ms()
        frac = (self.marker_ms / total) if total > 0 else None
        self.marker_label.setText(f"◆ {self._fmt_ms(self.marker_ms)}")
        self.marker_btn.setText("◆✕ e")
        self.waveform.set_marker_frac(frac)

    # ---- seeking ---------------------------------------------------------
    def seek_to(self, ms):
        """Move the playhead to ``ms``, clamped to the track."""
        total = self._duration_ms()
        ms = int(ms)
        if total > 0:
            ms = max(0, min(ms, total))
        else:
            ms = max(0, ms)
        if self.media_player is not None:
            self.media_player.setPosition(ms)
        # Update the readouts even with no player: the marker and the playhead
        # are the window's own state, and a missing device must not freeze the
        # waveform at zero.
        self._on_position(ms)

    def _position_ms(self):
        if self.media_player is None:
            return 0
        return int(self.media_player.position())

    def _duration_ms(self):
        if self.media_player is None:
            return 0
        return int(self.media_player.duration())

    def _on_slider_moved(self, val):
        total = self._duration_ms()
        if total > 0:
            self.seek_to(int(val / 1000 * total))

    def _on_position(self, ms):
        total = self._duration_ms()
        self.duration_ms = total
        if total > 0:
            frac = ms / total
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(frac * 1000))
            self.seek_slider.blockSignals(False)
            self.waveform.set_position_frac(frac)
            if self.marker_ms is not None:
                self.waveform.set_marker_frac(self.marker_ms / total)
        self.time_label.setText(f"{self._fmt_ms(ms)} / {self._fmt_ms(total)}")

    def _on_status(self, status):
        from PySide6.QtMultimedia import QMediaPlayer

        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Reaching the end IS a stop, so it follows a stop's rule and returns
            # to the marker rather than leaving the playhead parked at the end.
            self.stop()

    @staticmethod
    def _fmt_ms(ms):
        sec = max(0, int(ms // 1000))
        return f"{sec // 60}:{sec % 60:02d}"

    # ---- volume persistence ---------------------------------------------
    def _persist_volume(self):
        config = self._config
        if config is None:
            return
        config[VOLUME_KEY] = self._volume
        saver = self._saver or _default_saver()
        try:
            saver(config, [VOLUME_KEY])
        except Exception:  # noqa: BLE001 — a preference must never be fatal
            pass

    # ---- keys ------------------------------------------------------------
    def attach_keys(self, window):
        """Claim the transport keys on ``window`` (see this module's docstring).

        Installed on the WINDOW, not on the player, so the keys work while any
        non-text widget has focus -- the track table, a button, the slider -- and
        stop being claimed the moment the user is typing. One line at the call
        site, like ``KeyboardMode`` on the editor.
        """
        window.installEventFilter(self)
        # Kept on ``self`` for the teardown reason the module docstring records:
        # PySide6 frees a QObject whose last reference goes away, and a freed
        # filter silently stops receiving keys.
        self._key_window = window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        # A text input owns its letters: this is what keeps `d` a `d` while the
        # lyrics are being typed, and why the spec's key table cannot be global.
        if isinstance(QApplication.focusWidget(), TEXT_INPUTS):
            return False
        return self.handle_key(event)

    def keyPressEvent(self, event):
        if not self.handle_key(event):
            super().keyPressEvent(event)

    def handle_key(self, event):
        """True when ``event`` was a transport key and must not travel further."""
        # Bare keys only: Ctrl/Alt/Meta combinations belong to the menus and to
        # the editor, so Ctrl+O must stay "open a file".
        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        key = event.key()
        if key == Qt.Key_D:
            self.toggle_play()
        elif key == Qt.Key_S:
            self.pause()
        elif key == Qt.Key_F:
            self.nudge(NUDGE_MS)
        elif key == Qt.Key_A:
            self.nudge(-NUDGE_MS)
        elif key == Qt.Key_E:
            self.toggle_marker()
        elif key in (Qt.Key_Equal, Qt.Key_Plus):
            self.change_volume(VOLUME_STEP)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.change_volume(-VOLUME_STEP)
        else:
            return False
        event.accept()
        return True


def attach(config=None, saver=None, parent=None):
    """Build a transport. The window adds it at the top of its layout."""
    return StudioPlayer(config=config, saver=saver, parent=parent)
