"""Dataset embedding + projection worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.embeddings import compute_embedding, reduce_2d


class EmbeddingWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(object, object)   # coords {index: (x, y)}, meta {index: {filename, genre}}
    failed = Signal(str)

    def __init__(self, samples, parent=None):
        super().__init__(parent)
        self.samples = samples

    def run(self):
        try:
            vecs = {}
            meta = {}
            total = len(self.samples)
            for i, s in enumerate(self.samples):
                if total:
                    self.progress.emit(int(100 * i / total), f"Embedding {s.get('filename', '?')}…")
                v = compute_embedding(s.get("audio_path", ""))
                if v is not None:
                    vecs[i] = v
                    meta[i] = {"filename": s.get("filename", "?"), "genre": s.get("genre", "")}
            coords = reduce_2d(vecs)
            self.finished_ok.emit(coords, meta)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))