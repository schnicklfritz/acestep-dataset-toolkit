"""Dataset export worker thread."""
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from modules import exporters


class ExportWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, dataset, options, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.options = options   # dict of export options

    def run(self):
        try:
            samples = self.dataset.get("samples", [])
            dest = self.options.get("dest_dir", "")
            Path(dest).mkdir(parents=True, exist_ok=True)
            done = []

            if self.options.get("json"):
                exporters.export_json(samples, Path(dest) / "dataset.json")
                done.append("JSON")
            if self.options.get("csv"):
                exporters.export_csv(samples, Path(dest) / "dataset.csv")
                done.append("CSV")
            if self.options.get("jsonl"):
                exporters.export_jsonl(samples, Path(dest) / "dataset.jsonl")
                done.append("JSONL")
            if self.options.get("sidecar"):
                n = exporters.export_sidecar_captions(samples, Path(dest) / "captions")
                done.append(f"sidecar .txt ({n})")
            if self.options.get("folders"):
                self.progress.emit(5, "Copying audio into train/val folders…")
                counts = exporters.export_folders(
                    samples, Path(dest),
                    val_ratio=self.options.get("val_ratio", 0.2),
                    seed=self.options.get("seed", 42),
                    stratify=self.options.get("stratify", True),
                )
                done.append(f"train/val ({counts['train']}/{counts['val']})")

            self.finished_ok.emit(", ".join(done) if done else "nothing selected")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))