"""Lyrics transcription worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.lyrics import transcribe_lyrics_engine


class TranscribeLyricsWorker(QThread):
    finished_ok = Signal(object)   # {"lyrics", "segments"}
    failed = Signal(str)

    def __init__(self, audio_path, engine="whisperx", model_size="small",
                 language=None, initial_prompt=None, config=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.engine = engine
        self.model_size = model_size
        self.language = language
        self.initial_prompt = initial_prompt
        self.config = config or {}

    def run(self):
        try:
            result = transcribe_lyrics_engine(
                self.audio_path, engine=self.engine, config=self.config,
                model_size=self.model_size, language=self.language,
                initial_prompt=self.initial_prompt,
            )
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))