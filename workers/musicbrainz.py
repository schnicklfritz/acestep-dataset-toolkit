"""MusicBrainz metadata lookup worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.musicbrainz import lookup_recording


class MusicBrainzWorker(QThread):
    finished_ok = Signal(object)   # result dict
    failed = Signal(str)

    def __init__(self, artist, song, parent=None):
        super().__init__(parent)
        self.artist = artist
        self.song = song

    def run(self):
        try:
            result = lookup_recording(self.artist, self.song)
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))