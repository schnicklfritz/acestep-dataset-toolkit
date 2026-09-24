"""Background Kaggle connectivity probe.

WHY A THREAD: ``probe_kaggle`` makes a real network round trip — token
introspection plus one authenticated API call. Running that on the GUI thread
freezes the window for as long as a stalled request takes, which is precisely
what a "Test connection" button must not do.

WHY THIS CATCHES ``SystemExit`` AS WELL AS ``Exception``: the SDK's
``authenticate()`` ends with ``exit(1)`` when it cannot authenticate at all, and
that raises ``SystemExit``. ``SystemExit`` derives from ``BaseException``, so a
plain ``except Exception`` lets it escape — and a QThread that raises out of
``run()`` dies WITHOUT emitting ``failed``, leaving the UI waiting forever with
nothing on screen to explain it. That silent death is the most likely reason a
Kaggle run appears to "never connect".

``KeyboardInterrupt`` and ``GeneratorExit`` are deliberately NOT caught.
"""
from PySide6.QtCore import QThread, Signal

from modules.kaggle import probe_kaggle


class KaggleProbeWorker(QThread):
    """Runs ``probe_kaggle`` off the GUI thread and always reports back.

    Emits exactly one of ``done`` (the probe's result dict, which carries its own
    ``ok`` flag) or ``failed`` (a short reason when even the probe could not
    run). It never raises across the thread boundary.
    """

    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            result = probe_kaggle(self.config)
        except (Exception, SystemExit) as e:  # noqa: BLE001 — see module docstring
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.done.emit(result)
