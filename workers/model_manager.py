"""Model download worker thread (QThread wrapper around modules.model_manager)."""
from PySide6.QtCore import QThread, Signal

from modules.model_manager import download_model, find_model


class ModelDownloadWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)   # model id
    failed = Signal(str)

    def __init__(self, config, model_id, parent=None):
        super().__init__(parent)
        self.config = config
        self.model_id = model_id

    def run(self):
        try:
            entry = find_model(self.model_id)
            if entry is None:
                raise ValueError(f"Unknown model id: {self.model_id}")
            download_model(
                self.config, entry,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(self.model_id)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))