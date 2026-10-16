"""Shared chrome for the app's real (non-docked) windows.

WHAT THIS IS FOR
----------------
Three windows have to behave the same way: the main dataset manager, the Tools
window, and the Lyrics window. All three must be resizable, minimizable,
maximizable and always-on-top/bottom, and they must survive being moved between
monitors with different DPI.

Resize, minimize and maximize are what a normal top-level window already does,
so the module does not reinvent them -- :meth:`WindowChrome.menu` just exposes
them next to the two things Qt genuinely does not get right on its own:

1. **Z-order.** A window cannot be made to float above others by a style sheet or
   a layout; it needs ``Qt.WindowStaysOnTopHint`` on the window flags.
2. **Geometry restore.** A saved position is a bet that the monitor it was saved
   on still exists. See below.

WHY GEOMETRY IS A MODULE AND NOT THREE COPIES
---------------------------------------------
This failure has already happened once in this project: ``ui/shell.py`` had to
make the Assistant dock deliberately **non-floatable**, with the note that
floating "is how it got lost off the edge of the screen". A window restored onto
a monitor that is no longer plugged in is a window the user cannot reach, cannot
move, and -- if it is always-on-top -- cannot even see past. Mixing different DPI
scales across monitors makes the saved coordinates wrong in a second way, because
the saved rect is in the pixels of the screen it was captured on.

So every restore goes through :func:`clamp_to_screen`, which guarantees a
reachable rectangle on an *available* screen, and every window re-clamps when
Qt tells it that its screen changed. One implementation, three call sites.

WHAT IS NOT HERE
----------------
No maximise animation, no snap-to-edge, no custom title bar: the native frame
already provides those, and a hand-drawn frame would have to re-implement
dragging, resizing and system menu behaviour on three platforms. These windows
are plain framed windows on purpose.
"""
import base64

from PySide6.QtCore import QEvent, QObject, QRect, QTimer, Qt
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QMenu

# Used when a window has no saved geometry yet, or the saved blob is unusable.
DEFAULT_SIZE = (900, 700)

# How much of a window must sit on a screen for it to count as reachable.
# Deliberately small: enough to grab the title bar with a mouse and drag it back
# into view. Anything larger would refuse to honour a legitimate saved position
# that is mostly on a second monitor.
MIN_VISIBLE_W = 160
MIN_VISIBLE_H = 48

# Settings-file writes are debounced by this much. A window emits a move/resize
# for every pixel of a drag, and writing settings.json on each one would be
# hundreds of file writes per gesture.
SAVE_DEBOUNCE_MS = 400


def _available_rects():
    return [s.availableGeometry() for s in QGuiApplication.screens()]


def _primary_rect():
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        return screen.availableGeometry()
    rects = _available_rects()
    return rects[0] if rects else None


def _reachable(rect, avail):
    """True when enough of ``rect`` overlaps ``avail`` to be grabbed."""
    inter = avail.intersected(rect)
    return inter.width() >= MIN_VISIBLE_W and inter.height() >= MIN_VISIBLE_H


def clamp_to_screen(window):
    """Bring ``window`` back somewhere the user can actually reach it.

    Returns True when the window had to be moved, which is the signal that a
    restore landed off-screen. A maximized window is left alone: Qt sizes it to
    whichever screen it is on, so there is nothing to correct and moving it would
    fight the window manager.
    """
    rects = _available_rects()
    if not rects or window.isMaximized() or window.isFullScreen():
        return False

    current = window.geometry()
    if any(_reachable(current, avail) for avail in rects):
        return False

    target = _primary_rect() or rects[0]
    width = min(max(current.width(), MIN_VISIBLE_W), target.width())
    height = min(max(current.height(), MIN_VISIBLE_H), target.height())
    window.setGeometry(QRect(
        target.x() + (target.width() - width) // 2,
        target.y() + (target.height() - height) // 2,
        width,
        height,
    ))
    return True


def restore_geometry(window, config, key, default_size=DEFAULT_SIZE):
    """Restore ``window``'s saved geometry, then guarantee it is reachable.

    Returns True when a saved blob was applied. ``restoreGeometry`` itself is
    trusted for the size but NOT for the position -- it is a bet that the monitor
    still exists, which is exactly the bet that loses.
    """
    raw = ""
    if config:
        raw = config.get(key) or ""
    restored = False
    if raw:
        try:
            restored = bool(window.restoreGeometry(base64.b64decode(raw)))
        except Exception:  # noqa: BLE001 -- a corrupt blob is not fatal
            restored = False
    if not restored:
        window.resize(*default_size)
    clamp_to_screen(window)
    return restored


def save_geometry(window, config, key, saver=None):
    """Persist ``window``'s geometry under ``key`` as base64 in ``config``.

    base64 because ``saveGeometry`` returns a QByteArray and settings.json is
    JSON; this matches how ``ui_dock_state`` is stored in ``ui/shell.py``.
    """
    if config is None:
        return
    try:
        blob = bytes(window.saveGeometry())
    except Exception:  # noqa: BLE001 -- a window being torn down has no geometry
        return
    if not blob:
        return
    config[key] = base64.b64encode(blob).decode("ascii")
    if saver is None:
        from modules.config_store import save_plain_keys

        saver = save_plain_keys
    try:
        saver(config, [key])
    except Exception:  # noqa: BLE001 -- never let a preference write crash the app
        pass


def set_always_on_top(window, on, bottom=False):
    """Put ``window`` above (or below) other windows, or clear that.

    ``setWindowFlags`` HIDES a window -- Qt recreates the native window because
    the hint is a creation-time property on some platforms. So visibility and the
    maximized state are captured first and re-applied after, otherwise toggling
    this would make the window vanish (or un-maximize) in front of the user.
    """
    was_visible = window.isVisible()
    was_maximized = window.isMaximized()

    flags = window.windowFlags()
    flags &= ~Qt.WindowStaysOnTopHint
    flags &= ~Qt.WindowStaysOnBottomHint
    if on:
        flags |= Qt.WindowStaysOnBottomHint if bottom else Qt.WindowStaysOnTopHint
    window.setWindowFlags(flags)

    if was_visible:
        window.show()
        if was_maximized:
            window.showMaximized()
    return on


def _default_saver():
    from modules.config_store import save_plain_keys

    return save_plain_keys


class WindowChrome(QObject):
    """Always-on-top/bottom plus safe geometry for ONE real window.

    Create one per window and KEEP A REFERENCE TO IT (store it on the manager).
    PySide6 frees a QObject whose last Python reference goes away, and a freed
    chrome object takes its event filter with it -- the window would then stop
    saving its geometry, silently, and the bug would only show up as "my window
    size doesn't stick".

    ``key`` is the settings.json key for the geometry blob; the z-order is stored
    under ``<key>_z`` so both preferences come back on the next launch.
    """

    def __init__(self, window, config, key, default_size=DEFAULT_SIZE, saver=None):
        super().__init__(window)
        self.window = window
        self.config = config if config is not None else {}
        self.key = key
        self._z_key = f"{key}_z"
        self.default_size = tuple(default_size)
        self._saver = saver
        self._screen_hooked = False

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self.save)

        self.on_top = QAction("Always on top", window)
        self.on_top.setCheckable(True)
        self.on_top.toggled.connect(lambda on: self._set_z_order(on, bottom=False))

        self.on_bottom = QAction("Always on bottom", window)
        self.on_bottom.setCheckable(True)
        self.on_bottom.toggled.connect(lambda on: self._set_z_order(on, bottom=True))

        self.minimize_action = QAction("Minimize", window)
        self.minimize_action.triggered.connect(window.showMinimized)

        self.maximize_action = QAction("Maximize", window)
        self.maximize_action.triggered.connect(self.toggle_maximized)

        self.reset_action = QAction("Reset window position", window)
        self.reset_action.triggered.connect(self.reset_position)

        self.window.installEventFilter(self)

    # ---- user-facing surface ------------------------------------------------
    def menu(self, title="Window"):
        """The chrome as a QMenu, for a toolbar/menu-bar/context-menu host."""
        menu = QMenu(title, self.window)
        menu.addAction(self.on_top)
        menu.addAction(self.on_bottom)
        menu.addSeparator()
        menu.addAction(self.minimize_action)
        menu.addAction(self.maximize_action)
        menu.addSeparator()
        menu.addAction(self.reset_action)
        return menu

    def toggle_maximized(self):
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()

    def reset_position(self):
        """Put the window back on the primary screen at its default size.

        The escape hatch for a window that ended up somewhere unusable. It
        CENTRES rather than merely clamping, because clamping is a no-op when a
        window is technically reachable but, say, four pixels onto the edge of a
        screen. It is also the recovery path for a window that was lost
        off-screen by an older build, which is how this project lost one before.
        """
        if self.window.isMaximized() or self.window.isFullScreen():
            self.window.showNormal()
        width, height = self.default_size
        target = _primary_rect()
        self.window.resize(width, height)
        if target is not None:
            self.window.move(
                target.x() + max(0, (target.width() - width) // 2),
                target.y() + max(0, (target.height() - height) // 2),
            )
        self.save()

    # ---- persistence --------------------------------------------------------
    def restore(self):
        """Apply the saved geometry and z-order. Call before ``show()``."""
        restored = restore_geometry(self.window, self.config, self.key, self.default_size)
        state = (self.config.get(self._z_key) or "").strip().lower()
        if state in ("top", "bottom"):
            action = self.on_top if state == "top" else self.on_bottom
            action.setChecked(True)  # its handler applies the flag
        return restored

    def save(self):
        if self.config is None:
            return
        # Geometry is encoded through save_geometry but written with the z-order
        # in ONE settings.json write, to avoid two file writes per gesture.
        save_geometry(self.window, self.config, self.key, saver=lambda *_a, **_k: None)
        self.config[self._z_key] = self._z_state()
        saver = self._saver or _default_saver()
        try:
            saver(self.config, [self.key, self._z_key])
        except Exception:  # noqa: BLE001 -- a preference write must never be fatal
            pass

    def _z_state(self):
        if self.on_top.isChecked():
            return "top"
        if self.on_bottom.isChecked():
            return "bottom"
        return ""

    # ---- event plumbing -----------------------------------------------------
    def eventFilter(self, obj, event):
        """Clamp on show, re-clamp on screen change, debounce-save on move/resize.

        An event filter rather than signal connections because Qt emits no signal
        for "this window just became visible", and a window that is shown without
        ever being clamped is exactly the lost-off-screen case.
        """
        if obj is self.window:
            kind = event.type()
            if kind == QEvent.Type.Show:
                self._hook_screen()
                clamp_to_screen(self.window)
            elif kind == QEvent.Type.WindowStateChange:
                self._sync_maximize_text()
            elif kind in (QEvent.Type.Move, QEvent.Type.Resize):
                self._save_timer.start()
            elif kind == QEvent.Type.Close:
                self.save()
        return super().eventFilter(obj, event)

    def _sync_maximize_text(self):
        self.maximize_action.setText("Restore" if self.window.isMaximized() else "Maximize")

    def _set_z_order(self, on, bottom=False):
        """Apply one z-order action and keep the other one honest.

        The two hints are mutually exclusive, but Qt applies whichever was set
        last and leaves both checkmarks in place -- so the menu would claim the
        window is both above and below everything. Unchecking the sibling here is
        what keeps the menu truthful.
        """
        sibling = self.on_bottom if not bottom else self.on_top
        if on and sibling.isChecked():
            sibling.blockSignals(True)
            sibling.setChecked(False)
            sibling.blockSignals(False)
        set_always_on_top(self.window, on, bottom=bottom)
        self.save()

    def _hook_screen(self):
        """Re-clamp when Qt moves this window onto a different screen.

        Monitors can differ in DPI scale and in available area, so a rect that
        was fine on one is not automatically fine on the other -- this is the
        spec's mixed-DPI requirement. It can only be hooked once a windowHandle
        exists, which is at the first Show, hence the call from eventFilter.
        """
        handle = self.window.windowHandle()
        if handle is None or self._screen_hooked:
            return
        handle.screenChanged.connect(self._on_screen_changed)
        self._screen_hooked = True

    def _on_screen_changed(self, _screen=None):
        clamp_to_screen(self.window)
        self.save()


def attach(window, config, key, default_size=DEFAULT_SIZE, saver=None):
    """Build a :class:`WindowChrome` for ``window`` and return it.

    **Store the return value** somewhere that lives as long as the window (e.g.
    ``manager.lyrics_chrome``). Returning it is the point: PySide6 frees an object
    whose last Python reference goes away, and a freed chrome object takes its
    event filter with it, so the window would silently stop saving its geometry.
    """
    return WindowChrome(window, config, key, default_size=default_size, saver=saver)



