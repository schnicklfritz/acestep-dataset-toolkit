# main.py
# 🚀 Pure Assembly & Execution Layer for the ACE-Step Toolkit

import sys
from PySide6.QtCore import QLockFile, QStandardPaths, Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from dataset_manager import DatasetManager
from modules.wheel_guard import install_wheel_guard

# Two live windows are two independent writers onto the same dataset file. The
# save path can only make each write atomic; it cannot make two whole copies of
# the in-memory dataset agree, so the last save silently wins and the other
# window's captions/lyrics are gone. Refusing a second instance is the only
# place that race can be closed, and it is cheap.
_SINGLE_INSTANCE_KEY = "acestep_dataset_toolkit"
# Fallback for the first startup message loop: unless the primary answers within
# this window, this process takes over rather than making the app unreachable.
_ACTIVATION_TIMEOUT_MS = 700


def _instance_socket_path():
    """A per-user, per-machine path to name the single-instance lock and socket."""
    runtime = QStandardPaths.writableLocation(QStandardPaths.RuntimeLocation)
    return f"{runtime or '.'}/{_SINGLE_INSTANCE_KEY}"


def _activate_running_instance(name):
    """Ask the already-running window to come forward.

    Returns True when an instance answered, so the caller exits instead of
    opening a second window.
    """
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(_ACTIVATION_TIMEOUT_MS):
        return False
    socket.write(b"activate\n")
    socket.flush()
    socket.waitForBytesWritten(_ACTIVATION_TIMEOUT_MS)
    socket.disconnectFromServer()
    return True


def _single_instance_guard(app):
    """Ensure only one window edits the dataset; returns the guard, or None.

    ``activate`` is the slot a second launch calls on the running window. The
    returned object owns the QLocalServer, so the caller must keep a reference
    for as long as the app runs.
    """
    name = _instance_socket_path()
    if _activate_running_instance(name):
        return None
    # The activation socket is gone but a lock may linger after a crash; a stale
    # lock is removed and retried once rather than blocking every later launch.
    lock = QLockFile(f"{name}.lock")
    lock.setStaleLockTime(0)
    if not lock.tryLock(100) and not (lock.removeStaleLockFile() and lock.tryLock(100)):
        return None
    QLocalServer.removeServer(name)
    server = QLocalServer()
    if not server.listen(name):
        return None

    class _SingleInstance:
        def __init__(self, lock_file, local_server):
            self.lock_file = lock_file
            self.server = local_server
            self.window = None
            local_server.newConnection.connect(self._on_connection)

        def bind(self, window):
            self.window = window

        def _on_connection(self):
            conn = self.server.nextPendingConnection()
            if conn is None:
                return
            conn.readyRead.connect(lambda: self._activate(conn))
            conn.disconnected.connect(conn.deleteLater)
            # Some clients send the byte before the connection is delivered.
            self._activate(conn)

        def _activate(self, conn):
            # Drain so the client's flush finishes; the payload is not read.
            conn.readAll()
            w = self.window
            if w is None:
                return
            for call in (w.showNormal, w.raise_, w.activateWindow):
                call()
            w.setWindowState(w.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)

    guard = _SingleInstance(lock, server)
    app.aboutToQuit.connect(lock.unlock)
    return guard


if __name__ == "__main__":
    # Handle high-DPI font scaling and layout adjustments across multiple displays natively
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    # Stop the scroll wheel from silently changing combos/spinboxes/sliders the
    # cursor passes over while scrolling the dataset. Focus unlocks them.
    install_wheel_guard(app)
    # A second launch raises the running window instead of opening a second one
    # that would edit the same dataset file and overwrite the first's work.
    guard = _single_instance_guard(app)
    if guard is None:
        sys.exit(0)
    window = DatasetManager()
    guard.bind(window)
    window.show()
    sys.exit(app.exec())

