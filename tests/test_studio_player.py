"""The Lyrics Studio's transport (``ui.studio_player``).

Spec item 7 -- *"the player with waveform sits at the top of the window"* -- and
spec item 8, the transport key table:

    ``d`` play (and stop), ``f`` ahead, ``a`` back, ``e`` marker, ``=`` volume
    up, ``-`` volume down, ``s`` pause. Pause resets the marker to the pause
    point; stop returns to the last marker (Reaper style). ONE marker.

What is worth pinning here, and why each one can be got wrong silently:

* **The Reaper asymmetry.** ``d`` STOPS when already playing and a stop returns
  the playhead to the marker; ``s`` PAUSES and moves the marker to the pause
  point. Swap those two and the transport still feels plausible while the
  ``e`` + ``d`` loop stops working.
* **ONE marker.** ``e`` sets it, a second ``e`` clears it, and a new track
  clears it. Nothing may accumulate a marker per press.
* **The letters belong to the editor.** The whole key table is a letter set, so
  the filter must consult the focus widget: typing ``d`` into the lyrics must
  not start playback. This is the failure a user would hit first.
* **Bare keys only.** ``Ctrl+O``/``Alt+...`` belong to the menus and the editor.
* **Volume is a preference.** It is written through the injectable ``saver``
  seam (the same one ``ui/window_chrome`` and ``ui/lyrics_diff`` use), so these
  tests prove the write WITHOUT touching the user's ``settings.json``. Every
  test that can move the volume passes its own ``saver`` -- ``_persist_volume``
  would otherwise call the real ``save_plain_keys``.
* **No QtMultimedia is not a crash.** The window must open, the editor and the
  diff must work, and the transport must say so in its hint.

The player is driven through a ``QMediaPlayer`` stand-in
(:class:`_FakePlayer`). A real one would need a codec and a device; what this
module owns is the DECISION logic, which is exactly what the fake exposes.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.studio_player import (  # noqa: E402
    DEFAULT_VOLUME,
    NUDGE_MS,
    VOLUME_KEY,
    VOLUME_STEP,
    MarkerWaveform,
    StudioPlayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _press(key, mods=Qt.NoModifier):
    """A synthetic ``KeyPress``, carrying its own ``isAccepted()`` answer."""
    return QKeyEvent(QEvent.Type.KeyPress, key, mods)


def _recorder():
    """``(calls, saver)`` -- a saver that records instead of writing JSON."""
    calls = []

    def saver(config, keys):
        calls.append((dict(config), list(keys)))

    return calls, saver


def _hosted(qapp, config=None, saver=None, attach=True):
    """``(host, player, editor)``, the player inside a SHOWN window.

    ``isVisible()`` is False for any child of a window that was never shown, so a
    parentless player would make a visibility assertion pass for the wrong
    reason. The editor is in the same host on purpose: it is what takes focus in
    the guard tests, and it is the widget the user is actually typing into.

    The filter is installed by default, because a key-delivery test that forgets
    it passes for the worst possible reason -- nothing was listening at all.
    ``attach=False`` is for ``TestInstallation``, which proves that installing is
    what makes the keys work.
    """
    _, default_saver = _recorder()
    player = StudioPlayer(config, saver=saver if saver is not None else default_saver)
    editor = QTextEdit()
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.addWidget(player)
    lay.addWidget(editor)
    host.resize(640, 480)
    host.show()
    qapp.processEvents()
    if attach:
        player.attach_keys(host)
    # Stashed on the host so the parent chain keeps every object referenced: a
    # collected player or editor would silently stop answering.
    host._test_player = player
    host._test_editor = editor
    return host, player, editor


class _FakePlayer:
    """A ``QMediaPlayer`` stand-in: the transport's logic, no codec, no device.

    Its playback state is a REAL ``QMediaPlayer.PlaybackState`` member, because
    ``StudioPlayer.toggle_play`` compares against that enum -- a bare string here
    would make the "already playing" branch unreachable and the Reaper tests
    would pass while proving nothing.
    """

    def __init__(self, duration_ms=60000):
        from PySide6.QtMultimedia import QMediaPlayer

        self._enum = QMediaPlayer.PlaybackState
        self._state = self._enum.StoppedState
        self._pos = 0
        self._dur = duration_ms
        self.calls = []
        self.source = None

    # ---- what the transport calls -------------------------------------
    def playbackState(self):
        return self._state

    def play(self):
        self._state = self._enum.PlayingState
        self.calls.append("play")

    def stop(self):
        self._state = self._enum.StoppedState
        self.calls.append("stop")

    def pause(self):
        self._state = self._enum.PausedState
        self.calls.append("pause")

    def position(self):
        return self._pos

    def duration(self):
        return self._dur

    def setPosition(self, ms):
        self._pos = int(ms)
        self.calls.append(("setPosition", int(ms)))

    def setSource(self, url):
        self.source = url
        self.calls.append("setSource")

    # ---- test conveniences -------------------------------------------
    def playing(self):
        self._state = self._enum.PlayingState
        return self

    def at(self, ms):
        self._pos = int(ms)
        return self


def _driven(qapp, **kwargs):
    """A hosted player on a fake device: the transport's logic, deterministically."""
    host, player, editor = _hosted(qapp, **kwargs)
    fake = _FakePlayer()
    player.media_player = fake
    # A real QAudioOutput would talk to a sound device; the volume logic does not
    # need it, and ``change_volume`` already tolerates ``None``.
    player.audio_output = None
    return host, player, editor, fake


# ---------------------------------------------------------------------------
# The marker and the waveform
# ---------------------------------------------------------------------------

class TestMarkerWaveform:
    """``MarkerWaveform`` adds ONE marker line to the main window's waveform."""

    def test_the_marker_starts_unset(self, qapp):
        wave = MarkerWaveform()
        assert wave.marker_frac is None

    def test_a_marker_can_be_placed_and_cleared(self, qapp):
        wave = MarkerWaveform()
        wave.set_marker_frac(0.25)
        assert wave.marker_frac == 0.25
        wave.set_marker_frac(None)
        assert wave.marker_frac is None

    def test_a_marker_is_clamped_to_the_track(self, qapp):
        """A marker off the end of the waveform would be drawn outside it."""
        wave = MarkerWaveform()
        wave.set_marker_frac(1.8)
        assert wave.marker_frac == 1.0
        wave.set_marker_frac(-3)
        assert wave.marker_frac == 0.0

    def test_it_paints_without_a_marker(self, qapp):
        """The no-marker path must not raise: it is the state on every open."""
        wave = MarkerWaveform()
        wave.resize(200, 80)
        wave.repaint()
        assert wave.marker_frac is None


# ---------------------------------------------------------------------------
# The key table (spec item 8, and its two guards)
# ---------------------------------------------------------------------------

class TestKeyTable:
    """Every documented key does its documented job -- and only while it should."""

    def _spy(self, player, monkeypatch):
        """Replace the transport methods with recorders, returning the log."""
        log = []

        def note(name):
            def _call(*args, **kwargs):
                log.append(name)

            return _call

        for name in ("toggle_play", "pause", "nudge", "toggle_marker", "change_volume"):
            monkeypatch.setattr(player, name, note(name))
        return log

    @pytest.mark.parametrize("key,expected", [
        (Qt.Key_D, "toggle_play"),
        (Qt.Key_S, "pause"),
        (Qt.Key_F, "nudge"),
        (Qt.Key_A, "nudge"),
        (Qt.Key_E, "toggle_marker"),
        (Qt.Key_Equal, "change_volume"),
        (Qt.Key_Minus, "change_volume"),
    ])
    def test_each_documented_key_reaches_its_action(self, qapp, monkeypatch, key, expected):
        host, player, _, _ = _driven(qapp)
        log = self._spy(player, monkeypatch)

        assert player.handle_key(_press(key)) is True
        assert log == [expected]

    def test_the_shifted_forms_of_the_two_volume_keys_work(self, qapp, monkeypatch):
        """``=`` and ``-`` on a keyboard need Shift; the shifted codes must count."""
        host, player, _, _ = _driven(qapp)
        log = self._spy(player, monkeypatch)

        assert player.handle_key(_press(Qt.Key_Plus)) is True
        assert player.handle_key(_press(Qt.Key_Underscore)) is True
        assert log == ["change_volume", "change_volume"]

    def test_the_volume_keys_move_in_opposite_directions(self, qapp):
        """The shared action name in the table above would hide a swapped sign."""
        host, player, _, _ = _driven(qapp, config={})

        player.handle_key(_press(Qt.Key_Equal))
        loud = player._volume
        player.handle_key(_press(Qt.Key_Minus))
        quieter = player._volume

        assert loud > DEFAULT_VOLUME
        assert quieter < loud

    def test_an_unrelated_letter_is_not_claimed(self, qapp):
        """A stray claim would eat keystrokes the rest of the window wants."""
        host, player, _, fake = _driven(qapp)
        player.setFocus()
        qapp.processEvents()
        event = _press(Qt.Key_Z)

        QApplication.sendEvent(host, event)

        # Delivered and left ignored: the filter said no and nothing else wanted
        # it either. (A freshly built QKeyEvent is accepted by default, so
        # asserting on the object without delivering it would prove nothing.)
        assert not event.isAccepted()
        assert fake.calls == []

    def test_control_combinations_stay_with_the_menus(self, qapp, monkeypatch):
        """``Ctrl+D`` is a menu shortcut, not a transport key."""
        host, player, _, _ = _driven(qapp)
        log = self._spy(player, monkeypatch)

        for mods in (Qt.ControlModifier, Qt.AltModifier, Qt.MetaModifier):
            assert player.handle_key(_press(Qt.Key_D, mods)) is False
        assert log == []

    def test_a_focused_text_input_keeps_its_letters(self, qapp):
        """The editor is a text field and ``d`` is a letter.

        This is why the table cannot be grabbed globally: a user typing the word
        "down" would start and stop playback twice.
        """
        # The filter IS installed on the host here (``_driven`` does it), so an
        # empty call log below means the focus guard refused the key -- not that
        # nothing was listening.
        host, player, editor, fake = _driven(qapp)
        assert player._key_window is host
        editor.setFocus()
        qapp.processEvents()
        # The precondition is asserted, not assumed: offscreen Qt does deliver
        # focus, and if that ever stops being true this test must fail loudly
        # rather than pass because ``focusWidget()`` returned ``None``.
        assert QApplication.focusWidget() is editor
        event = _press(Qt.Key_D)

        QApplication.sendEvent(host, event)

        assert not event.isAccepted()
        assert fake.calls == []
        assert player.play_btn.text() == "▶ d"

    def test_a_non_text_widget_does_not_hold_the_keys_back(self, qapp):
        """Click the player rather than the editor and the transport answers."""
        host, player, _, fake = _driven(qapp)
        player.setFocus()
        qapp.processEvents()
        assert not isinstance(QApplication.focusWidget(), (QTextEdit, QLineEdit))
        event = _press(Qt.Key_D)

        QApplication.sendEvent(host, event)

        assert event.isAccepted()
        assert fake.calls == ["play"]


# ---------------------------------------------------------------------------
# Reaper semantics: the four transport rules that are easy to swap
# ---------------------------------------------------------------------------

class TestReaperSemantics:
    """``d`` stops to the marker, ``s`` pauses onto the marker, ``e`` is one."""

    def test_d_plays_when_stopped(self, qapp):
        host, player, _, fake = _driven(qapp)

        player.handle_key(_press(Qt.Key_D))

        assert fake.calls == ["play"]
        assert player.play_btn.text() == "⏹ d"

    def test_d_stops_when_already_playing(self, qapp):
        """The second press is a STOP, not a pause -- that is what makes a loop."""
        host, player, _, fake = _driven(qapp)
        fake.playing()

        player.handle_key(_press(Qt.Key_D))

        assert "stop" in fake.calls
        assert player.play_btn.text() == "▶ d"

    def test_a_stop_returns_the_playhead_to_the_marker(self, qapp):
        """Reaper style: stop is a jump back to the marker, not to zero."""
        host, player, _, fake = _driven(qapp)
        player._set_marker(5000)
        fake.playing().at(20000)

        player.stop()

        assert fake._pos == 5000

    def test_a_stop_without_a_marker_returns_to_the_start(self, qapp):
        host, player, _, fake = _driven(qapp)
        fake.playing().at(20000)

        player.stop()

        assert fake._pos == 0

    def test_pause_moves_the_marker_to_the_pause_point(self, qapp):
        """THE asymmetry: ``s`` is not a plain pause, it bookmarks the spot."""
        host, player, _, fake = _driven(qapp)
        fake.playing().at(7000)

        player.handle_key(_press(Qt.Key_S))

        assert "pause" in fake.calls
        assert player.marker_ms == 7000
        assert player.marker_label.text() == "◆ 0:07"

    def test_e_sets_the_marker_at_the_playhead(self, qapp):
        host, player, _, fake = _driven(qapp)
        fake.at(3000)

        player.handle_key(_press(Qt.Key_E))

        assert player.marker_ms == 3000
        assert player.waveform.marker_frac == 3000 / 60000

    def test_a_second_e_clears_the_one_marker(self, qapp):
        """``e`` toggles. A marker list would make "the last marker" ambiguous."""
        host, player, _, fake = _driven(qapp)
        fake.at(3000)
        player.handle_key(_press(Qt.Key_E))
        assert player.marker_ms == 3000

        player.handle_key(_press(Qt.Key_E))

        assert player.marker_ms is None
        assert player.waveform.marker_frac is None

    def test_the_marker_is_exactly_one_position(self, qapp):
        """Whatever the internal shape, it must not accumulate markers."""
        host, player, _, fake = _driven(qapp)
        fake.at(1000)
        player.handle_key(_press(Qt.Key_E))
        fake.at(2000)
        player.handle_key(_press(Qt.Key_E))
        player.handle_key(_press(Qt.Key_E))

        assert player.marker_ms is None or isinstance(player.marker_ms, int)
        assert not isinstance(player.marker_ms, (list, tuple, set, dict))

    def test_the_nudges_do_not_move_the_marker(self, qapp):
        """Nudging is how you COMPARE to the marker; moving it defeats the point."""
        host, player, _, fake = _driven(qapp)
        player._set_marker(5000)
        fake.at(5000)

        player.handle_key(_press(Qt.Key_F))
        after_forward = (fake._pos, player.marker_ms)
        player.handle_key(_press(Qt.Key_A))

        assert after_forward == (5000 + NUDGE_MS, 5000)
        assert fake._pos == 5000
        assert player.marker_ms == 5000

    def test_the_nudges_are_clamped_to_the_track(self, qapp):
        """A nudge back from 0 would seek to a negative position."""
        host, player, _, fake = _driven(qapp)
        fake.at(0)

        player.handle_key(_press(Qt.Key_A))

        assert fake._pos == 0

    def test_the_end_of_the_track_is_a_stop_not_a_park(self, qapp):
        """Playing out IS a stop, so it follows a stop's rule and returns."""
        from PySide6.QtMultimedia import QMediaPlayer

        host, player, _, fake = _driven(qapp)
        player._set_marker(2000)
        fake.playing().at(60000)

        player._on_status(QMediaPlayer.MediaStatus.EndOfMedia)

        assert player.marker_ms == 2000
        assert fake._pos == 2000

    def test_the_time_readout_is_minutes_and_seconds(self, qapp):
        host, player, _, fake = _driven(qapp)
        fake.at(61000)

        player._on_position(61000)

        assert player.time_label.text() == "1:01 / 1:00"


# ---------------------------------------------------------------------------
# Volume: a preference, written through the injected saver
# ---------------------------------------------------------------------------

class TestVolume:
    """``=``/``-`` move the output volume and remember it across sessions."""

    def test_volume_moves_in_steps(self, qapp):
        calls, saver = _recorder()
        host, player, _, _ = _driven(qapp, config={}, saver=saver)

        player.change_volume(VOLUME_STEP)

        assert player._volume == pytest.approx(DEFAULT_VOLUME + VOLUME_STEP)

    def test_volume_is_clamped_at_both_ends(self, qapp):
        """Unclamped, ``=`` would hand Qt a gain above 1.0 and ``-`` a negative."""
        calls, saver = _recorder()
        host, player, _, _ = _driven(qapp, config={}, saver=saver)

        player.change_volume(5.0)
        assert player._volume == 1.0

        player.change_volume(-5.0)
        assert player._volume == 0.0

    def test_volume_is_persisted_under_its_own_key(self, qapp):
        """The saver seam is the point: no test here may write settings.json."""
        calls, saver = _recorder()
        host, player, _, _ = _driven(qapp, config={}, saver=saver)

        player.change_volume(VOLUME_STEP)

        written = calls[-1][0]
        assert written[VOLUME_KEY] == pytest.approx(player._volume)
        # The key list matters: ``save_plain_keys`` merges exactly these, so a
        # wrong list would either drop the preference or rewrite the whole file.
        assert calls[-1][1] == [VOLUME_KEY]
        assert list(written) == [VOLUME_KEY]

    def test_a_remembered_volume_is_restored(self, qapp):
        calls, saver = _recorder()
        host, player, _, _ = _driven(qapp, config={VOLUME_KEY: 0.25}, saver=saver)

        assert player._volume == pytest.approx(0.25)
        assert player.volume_label.text() == "🔊 25%"

    @pytest.mark.parametrize("junk", ["loud", 5.0, -1, None, [0.5]])
    def test_a_nonsense_saved_volume_falls_back(self, qapp, junk):
        """A hand-edited settings.json must not silence the transport."""
        calls, saver = _recorder()
        host, player, _, _ = _driven(qapp, config={VOLUME_KEY: junk}, saver=saver)

        assert player._volume == pytest.approx(DEFAULT_VOLUME)

    def test_the_label_follows_the_volume(self, qapp):
        calls, saver = _recorder()
        host, player, _, _ = _driven(qapp, config={}, saver=saver)

        player.change_volume(-4 * VOLUME_STEP)

        expected = int(round((DEFAULT_VOLUME - 4 * VOLUME_STEP) * 100))
        assert player.volume_label.text() == f"🔊 {expected}%"


# ---------------------------------------------------------------------------
# No audio device / no QtMultimedia: degrade, do not crash
# ---------------------------------------------------------------------------

class TestNoPlayback:
    """The Studio is an editor first. A missing player must not take it down."""

    def _bare(self, qapp):
        host, player, editor = _hosted(qapp, config={})
        player.media_player = None
        player.audio_output = None
        return host, player, editor

    def test_the_transport_explains_instead_of_raising(self, qapp):
        host, player, _ = self._bare(qapp)

        player.handle_key(_press(Qt.Key_D))
        player.handle_key(_press(Qt.Key_S))
        player.handle_key(_press(Qt.Key_F))
        player.handle_key(_press(Qt.Key_A))

        assert "nothing to move" in player.hint.text()

    def test_the_waveform_and_the_readout_still_move(self, qapp):
        """The marker and the playhead are the window's own state, not the codec's."""
        host, player, _ = self._bare(qapp)

        player.seek_to(2000)

        assert player.waveform.position_frac == 0.0  # no duration to take a fraction of
        assert player.time_label.text() == "0:02 / 0:00"

    def test_the_marker_still_works_without_a_device(self, qapp):
        host, player, _ = self._bare(qapp)

        player.handle_key(_press(Qt.Key_E))

        assert player.marker_ms == 0
        player.handle_key(_press(Qt.Key_E))
        assert player.marker_ms is None

    def test_volume_still_works_without_a_device(self, qapp):
        calls, saver = _recorder()
        host, player, _ = _hosted(qapp, config={}, saver=saver)
        player.media_player = None
        player.audio_output = None

        player.handle_key(_press(Qt.Key_Equal))

        assert player._volume == pytest.approx(DEFAULT_VOLUME + VOLUME_STEP)
        assert calls[-1][1] == [VOLUME_KEY]


# ---------------------------------------------------------------------------
# Loading a track
# ---------------------------------------------------------------------------

class TestSetAudio:
    """``set_audio`` is the player's whole interface to the dataset."""

    def test_a_new_track_clears_the_marker(self, qapp, wav_file):
        """The old marker names a second of the OLD track -- meaningless now."""
        host, player, _, fake = _driven(qapp)
        fake.at(5000)
        player.handle_key(_press(Qt.Key_E))
        assert player.marker_ms == 5000

        player.set_audio(wav_file("song.wav", seconds=0.2))

        assert player.marker_ms is None
        assert player.waveform.marker_frac is None
        assert player.marker_label.text() == "◆ —"

    def test_a_real_file_reaches_both_the_player_and_the_waveform(self, qapp, wav_file):
        host, player, _, fake = _driven(qapp)

        player.set_audio(wav_file("song.wav", seconds=0.2))

        assert fake.source is not None
        assert player.waveform.min_buckets is not None

    def test_a_new_track_resets_the_timeline(self, qapp, wav_file):
        """Keeping the old duration would draw the marker against the wrong length."""
        host, player, _, fake = _driven(qapp)
        fake.at(4000)
        player._on_position(4000)

        player.set_audio(wav_file("song.wav", seconds=0.2))

        assert player.duration_ms == 0
        assert player.play_btn.text() == "▶ d"

    def test_a_missing_file_is_dropped_not_raised(self, qapp):
        """A dataset entry can point at a file that was moved or renamed."""
        host, player, _, fake = _driven(qapp)

        player.set_audio("/definitely/not/here.wav")

        assert player.waveform.min_buckets is None
        assert "stop" in fake.calls
        assert player.time_label.text() == "0:00 / 0:00"
        assert player.seek_slider.value() == 0

    def test_dropping_the_track_empties_the_player(self, qapp, wav_file):
        host, player, _, fake = _driven(qapp)
        player.set_audio(wav_file("song.wav", seconds=0.2))

        player.set_audio(None)

        assert player.waveform.min_buckets is None
        assert player.marker_ms is None
        assert player.time_label.text() == "0:00 / 0:00"


# ---------------------------------------------------------------------------
# Installation, the buttons, and the teardown trap
# ---------------------------------------------------------------------------

class TestInstallation:
    """The filter must be on the WINDOW, and the window must be kept alive."""

    def test_without_the_installation_the_key_does_nothing(self, qapp):
        """The baseline the next test needs: nothing is claimed until attached."""
        host, player, _, fake = _driven(qapp, attach=False)
        player.setFocus()
        qapp.processEvents()

        QApplication.sendEvent(host, _press(Qt.Key_D))

        assert fake.calls == []

    def test_a_key_on_the_window_reaches_the_transport(self, qapp):
        """PySide6 has no ``eventFilters()`` accessor, so this is the only proof."""
        host, player, _, fake = _driven(qapp, attach=False)
        player.setFocus()
        qapp.processEvents()

        player.attach_keys(host)
        QApplication.sendEvent(host, _press(Qt.Key_D))

        assert fake.calls == ["play"]

    def test_the_window_reference_is_kept(self, qapp):
        """A collected filter silently stops receiving keys, so it is held."""
        host, player, _, _ = _driven(qapp, attach=False)

        player.attach_keys(host)

        assert player._key_window is host

    def test_the_player_answers_its_own_key_press(self, qapp):
        """Clicking the player (StrongFocus) must not dead-end the transport."""
        host, player, _, fake = _driven(qapp)

        QApplication.sendEvent(player, _press(Qt.Key_D))

        assert fake.calls == ["play"]

    def test_the_buttons_do_the_same_jobs_as_the_keys(self, qapp):
        """Both routes call the same methods; a divergence would be a bug."""
        host, player, _, fake = _driven(qapp)

        player.play_btn.click()
        assert fake.calls == ["play"]

        fake.at(10000)
        player.back_btn.click()
        assert fake._pos == 10000 - NUDGE_MS

        player.fwd_btn.click()
        assert fake._pos == 10000

        player.marker_btn.click()
        assert player.marker_ms == 10000
        player.marker_btn.click()
        assert player.marker_ms is None

    def test_the_volume_buttons_do_the_same_jobs_as_the_keys(self, qapp):
        calls, saver = _recorder()
        host, player, _ = _hosted(qapp, config={}, saver=saver)

        player.vol_up_btn.click()
        assert player._volume == pytest.approx(DEFAULT_VOLUME + VOLUME_STEP)
        player.vol_down_btn.click()
        assert player._volume == pytest.approx(DEFAULT_VOLUME)

    def test_the_pause_button_pauses(self, qapp):
        host, player, _, fake = _driven(qapp)
        fake.playing().at(9000)

        player.pause_btn.click()

        assert "pause" in fake.calls
        assert player.marker_ms == 9000

    def test_the_hint_names_the_whole_key_table(self, qapp):
        """The hint is the only place a user learns the table exists."""
        host, player, _, _ = _driven(qapp)

        for key in ("d", "s", "f", "a", "e", "=", "−"):
            assert key in player.hint.text()

    def test_the_buttons_stay_out_of_the_focus_chain(self, qapp):
        """A button that takes focus would swallow the next key press."""
        host, player, _, _ = _driven(qapp)

        for btn in (player.play_btn, player.pause_btn, player.back_btn,
                    player.fwd_btn, player.marker_btn,
                    player.vol_up_btn, player.vol_down_btn, player.seek_slider):
            assert btn.focusPolicy() == Qt.NoFocus


class TestTeardownDefaults:
    """Class-level defaults, because a key can arrive after the C++ side is gone.

    PySide6 clears instance attributes once the C++ object is destroyed while Qt
    can still deliver a last event, so every attribute the key path reads is
    declared on the class. This is the same trap ``ui/lyrics_keyboard_mode`` and
    ``ui/window_chrome`` document.
    """

    def test_the_player_declares_its_key_path_attributes(self):
        assert StudioPlayer.media_player is None
        assert StudioPlayer.audio_output is None
        assert StudioPlayer.marker_ms is None
        assert StudioPlayer.duration_ms == 0
        assert StudioPlayer._volume == DEFAULT_VOLUME
        assert StudioPlayer._config is None
        assert StudioPlayer._saver is None

    def test_the_waveform_declares_its_marker(self):
        assert MarkerWaveform.marker_frac is None


class TestFactory:
    """``attach()`` is what the window calls; it must not need a config."""

    def test_it_builds_a_player_without_a_config(self, qapp, monkeypatch):
        """A manager built by tests has no config store, so there is no write."""
        import modules.config_store as cs

        def _explode(*args, **kwargs):
            raise AssertionError("a config-less player must not write settings")

        monkeypatch.setattr(cs, "save_plain_keys", _explode)
        from ui.studio_player import attach

        player = attach()

        assert isinstance(player, StudioPlayer)
