"""WhisperX lyrics transcription worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.lyrics import transcribe


class TranscribeLyricsWorker(QThread):
    finished_ok = Signal(dict)   # {"lyrics", "segments"}
    failed = Signal(str)

    def __init__(self, audio_path, model_size="small", language=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language

    def run(self):
        try:
            result = transcribe(self.audio_path, model_size=self.model_size,
                                language=self.language)
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))