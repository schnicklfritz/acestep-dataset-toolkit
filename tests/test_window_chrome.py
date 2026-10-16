"""Shared window chrome: z-order and the geometry that must never get lost.

The load-bearing assertion in this file is that a window restored or moved to a
position no monitor covers is brought BACK. This project has already lost a
window off the edge of the screen once (``ui/shell.py`` made the Assistant dock
non-floatable because of it), and an always-on-top window stuck off-screen is
worse than a normal one: it is invisible and still intercepting input.

Geometry is also the part that is easy to get wrong quietly, because a wrong
position is only obvious on a machine with the same monitor layout it was saved
on. These tests drive it headlessly, where the available area is a known 800x800.
"""
import base64

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from ui.window_chrome import (
    MIN_VISIBLE_H,
    MIN_VISIBLE_W,
    WindowChrome,
    attach,
    clamp_to_screen,
    restore_geometry,
    save_geometry,
    set_always_on_top,
)

# A position no monitor on the offscreen platform covers.
FAR_AWAY = (99999, 99999)


def _available():
    return QGuiApplication.primaryScreen().availableGeometry()


def _window(size=(400, 300), pos=None):
    window = QWidget()
    window.resize(*size)
    if pos is not None:
        window.move(*pos)
    window.show()
    return window


def _frame_offset(window):
    """Platform offset between ``move()`` and the resulting geometry.

    The offscreen platform shifts a shown window by a couple of pixels, so
    arithmetic on the REQUESTED position lands a few pixels away from the real
    one. Measuring it keeps the threshold test below exact.
    """
    window.move(100, 100)
    return window.geometry().x() - 100, window.geometry().y() - 100


class TestClampToScreen:
    def test_a_reachable_window_is_left_alone(self, qapp):
        available = _available()
        window = _window(pos=(available.x() + 10, available.y() + 10))
        before = window.geometry()
        assert clamp_to_screen(window) is False
        assert window.geometry() == before
        window.close()

    def test_an_off_screen_window_is_brought_back(self, qapp):
        window = _window(pos=FAR_AWAY)
        assert clamp_to_screen(window) is True
        assert any(
            window.geometry().intersected(s.availableGeometry()).width() >= MIN_VISIBLE_W
            for s in QGuiApplication.screens()
        )
        window.close()

    def test_an_off_screen_window_lands_on_the_primary_screen(self, qapp):
        window = _window(pos=FAR_AWAY)
        clamp_to_screen(window)
        available = _available()
        assert available.contains(window.geometry().topLeft())
        window.close()

    def test_an_oversized_window_is_shrunk_to_fit(self, qapp):
        window = _window(size=(5000, 5000), pos=FAR_AWAY)
        clamp_to_screen(window)
        available = _available()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        window.close()

    def test_the_reachability_threshold_is_exact(self, qapp):
        """Exactly MIN_VISIBLE_W on screen is grabbable; one pixel less is not.

        This is the boundary that decides "leave the user's window alone" versus
        "rescue it", so it is pinned from both sides rather than approximated.
        """
        available = _available()
        edge = available.x() + available.width()
        window = _window()
        offset_x, _offset_y = _frame_offset(window)

        window.move(edge - MIN_VISIBLE_W - offset_x, available.y())
        overlap = window.geometry().intersected(available).width()
        assert overlap == MIN_VISIBLE_W
        assert clamp_to_screen(window) is False

        window.move(edge - MIN_VISIBLE_W + 1 - offset_x, available.y())
        assert window.geometry().intersected(available).width() == MIN_VISIBLE_W - 1
        assert clamp_to_screen(window) is True
        window.close()

    def test_a_maximized_window_is_not_moved(self, qapp):
        """Qt sizes a maximized window to its screen; moving it fights the WM."""
        window = _window()
        window.showMaximized()
        if not window.isMaximized():
            window.close()
            return  # platform cannot maximize (headless) -- nothing to assert
        assert clamp_to_screen(window) is False
        window.close()



class TestGeometryPersistence:
    def test_save_writes_base64_under_the_key(self, qapp):
        window = _window()
        config = {}
        save_geometry(window, config, "k", saver=lambda *_a: None)
        assert isinstance(config["k"], str)
        # Must survive a JSON round-trip, because settings.json is JSON.
        assert base64.b64decode(config["k"]) == bytes(window.saveGeometry())
        window.close()

    def test_save_calls_the_saver_once_with_just_that_key(self, qapp):
        window = _window()
        calls = []
        save_geometry(window, {}, "k", saver=lambda cfg, keys: calls.append(keys))
        assert calls == [["k"]]
        window.close()

    def test_save_without_a_config_is_a_no_op(self, qapp):
        window = _window()
        save_geometry(window, None, "k")  # must not raise
        window.close()

    def test_a_saver_that_raises_does_not_escape(self, qapp):
        """A preference write must never take the app down with it."""
        window = _window()

        def boom(*_a):
            raise OSError("disk full")

        save_geometry(window, {}, "k", saver=boom)
        window.close()

    def test_restore_with_no_saved_geometry_uses_the_default_size(self, qapp):
        window = _window()
        assert restore_geometry(window, {}, "k", default_size=(321, 234)) is False
        assert (window.width(), window.height()) == (321, 234)
        window.close()

    def test_a_corrupt_blob_falls_back_instead_of_raising(self, qapp):
        window = _window()
        restored = restore_geometry(window, {"k": "not-base64!!"}, "k",
                                    default_size=(321, 234))
        assert restored is False
        assert (window.width(), window.height()) == (321, 234)
        window.close()

    def test_a_saved_geometry_round_trips(self, qapp):
        window = _window(size=(400, 300))
        config = {}
        save_geometry(window, config, "k", saver=lambda *_a: None)

        other = _window(size=(100, 100))
        assert restore_geometry(other, config, "k") is True
        assert (other.width(), other.height()) == (400, 300)
        window.close()
        other.close()

    def test_a_restore_from_a_vanished_monitor_is_corrected(self, qapp):
        """The whole point: a saved position is a bet the monitor still exists."""
        window = _window(size=(400, 300))
        config = {}
        save_geometry(window, config, "k", saver=lambda *_a: None)
        window.close()

        stranded = _window(size=(400, 300), pos=FAR_AWAY)
        restore_geometry(stranded, config, "k")
        assert any(
            stranded.geometry().intersected(s.availableGeometry()).width() >= MIN_VISIBLE_W
            for s in QGuiApplication.screens()
        )
        stranded.close()


class TestZOrder:
    def test_always_on_top_sets_the_hint(self, qapp):
        from PySide6.QtCore import Qt

        window = _window()
        set_always_on_top(window, True)
        assert bool(window.windowFlags() & Qt.WindowStaysOnTopHint)
        assert not bool(window.windowFlags() & Qt.WindowStaysOnBottomHint)
        window.close()

    def test_always_on_bottom_sets_only_the_bottom_hint(self, qapp):
        from PySide6.QtCore import Qt

        window = _window()
        set_always_on_top(window, True, bottom=True)
        assert bool(window.windowFlags() & Qt.WindowStaysOnBottomHint)
        assert not bool(window.windowFlags() & Qt.WindowStaysOnTopHint)
        window.close()

    def test_turning_it_off_clears_both_hints(self, qapp):
        from PySide6.QtCore import Qt

        window = _window()
        set_always_on_top(window, True)
        set_always_on_top(window, False)
        assert not bool(window.windowFlags() & Qt.WindowStaysOnTopHint)
        assert not bool(window.windowFlags() & Qt.WindowStaysOnBottomHint)
        window.close()

    def test_a_visible_window_stays_visible(self, qapp):
        """setWindowFlags HIDES a window; the toggle must re-show it."""
        window = _window()
        set_always_on_top(window, True)
        assert window.isVisible()
        set_always_on_top(window, False)
        assert window.isVisible()
        window.close()


class TestWindowChrome:
    def test_the_menu_offers_every_chrome_action(self, qapp):
        chrome = attach(_window(), {}, "k", saver=lambda *_a: None)
        labels = [a.text() for a in chrome.menu().actions() if a.text()]
        assert labels == [
            "Always on top", "Always on bottom", "Minimize", "Maximize",
            "Reset window position",
        ]
        chrome.window.close()

    def test_top_and_bottom_are_mutually_exclusive(self, qapp):
        """Qt would otherwise apply the last one and leave both checkmarks set."""
        chrome = attach(_window(), {}, "k", saver=lambda *_a: None)
        chrome.on_top.setChecked(True)
        chrome.on_bottom.setChecked(True)
        assert not chrome.on_top.isChecked()
        assert chrome.on_bottom.isChecked()
        chrome.window.close()

    def test_unchecking_one_leaves_neither_checked(self, qapp):
        from PySide6.QtCore import Qt

        chrome = attach(_window(), {}, "k", saver=lambda *_a: None)
        chrome.on_top.setChecked(True)
        chrome.on_top.setChecked(False)
        assert not chrome.on_top.isChecked()
        assert not chrome.on_bottom.isChecked()
        assert not bool(chrome.window.windowFlags() & Qt.WindowStaysOnTopHint)
        chrome.window.close()

    def test_save_persists_geometry_and_z_order_in_one_write(self, qapp):
        calls = []
        chrome = attach(_window(), {}, "k", saver=lambda cfg, keys: calls.append(list(keys)))
        chrome.on_top.setChecked(True)
        assert calls == [["k", "k_z"]]
        assert chrome.config["k_z"] == "top"
        assert chrome.config["k"]
        chrome.window.close()

    def test_restore_reapplies_the_saved_z_order(self, qapp):
        from PySide6.QtCore import Qt

        chrome = attach(_window(), {"k": "", "k_z": "top"}, "k", saver=lambda *_a: None)
        chrome.restore()
        assert chrome.on_top.isChecked()
        assert bool(chrome.window.windowFlags() & Qt.WindowStaysOnTopHint)
        chrome.window.close()

    def test_restore_ignores_an_unknown_z_order_value(self, qapp):
        chrome = attach(_window(), {"k_z": "sideways"}, "k", saver=lambda *_a: None)
        chrome.restore()
        assert not chrome.on_top.isChecked()
        assert not chrome.on_bottom.isChecked()
        chrome.window.close()

    def test_reset_position_centres_at_the_default_size(self, qapp):
        window = _window(size=(100, 100), pos=FAR_AWAY)
        chrome = attach(window, {}, "k", default_size=(400, 300), saver=lambda *_a: None)
        chrome.reset_position()
        assert (window.width(), window.height()) == (400, 300)
        available = _available()
        assert available.contains(window.geometry().topLeft())
        window.close()

    def test_reset_position_rescues_a_stranded_window(self, qapp):
        """The documented recovery path for a window lost by an older build."""
        window = _window(pos=FAR_AWAY)
        chrome = attach(window, {}, "k", default_size=(400, 300), saver=lambda *_a: None)
        chrome.reset_position()
        assert any(
            window.geometry().intersected(s.availableGeometry()).width() >= MIN_VISIBLE_W
            for s in QGuiApplication.screens()
        )
        window.close()

    def test_maximize_text_follows_the_window_state(self, qapp):
        window = _window()
        chrome = attach(window, {}, "k", saver=lambda *_a: None)
        window.showMaximized()
        qapp.processEvents()
        expected = "Restore" if window.isMaximized() else "Maximize"
        assert chrome.maximize_action.text() == expected
        window.close()

    def test_a_move_schedules_a_debounced_save(self, qapp):
        """Wiring check: the event filter actually persists a drag."""
        chrome = attach(_window(), {}, "k", saver=lambda *_a: None)
        chrome._save_timer.setInterval(0)
        chrome.window.move(30, 40)
        qapp.processEvents()
        assert chrome.config.get("k")
        assert chrome.config.get("k_z") == ""
        chrome.window.close()

    def test_attach_installs_an_event_filter_that_survives(self, qapp):
        """Retainment: a collected chrome object would stop saving, silently."""
        window = _window()
        chrome = attach(window, {}, "k", saver=lambda *_a: None)
        assert isinstance(chrome, WindowChrome)
        assert chrome in window.findChildren(WindowChrome)
        assert chrome.parent() is window
        window.close()
