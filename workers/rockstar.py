"""Rockstar multitrack-existence lookup worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.rockstar_lookup import lookup_rockstar_track


class RockstarLookupWorker(QThread):
    finished_ok = Signal(dict)   # lookup result dict
    failed = Signal(str)

    def __init__(self, artist, song, parent=None):
        super().__init__(parent)
        self.artist = artist
        self.song = song

    def run(self):
        try:
            result = lookup_rockstar_track(self.artist, self.song)
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))