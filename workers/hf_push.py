"""Hugging Face dataset push worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.hf_push import push_dataset


class HFPushWorker(QThread):
    finished_ok = Signal(str)   # repo_id
    failed = Signal(str)

    def __init__(self, dataset, repo_id, token=None, private=False, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.repo_id = repo_id
        self.token = token
        self.private = private

    def run(self):
        try:
            repo = push_dataset(self.dataset, self.repo_id, token=self.token,
                                private=self.private)
            self.finished_ok.emit(repo)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))