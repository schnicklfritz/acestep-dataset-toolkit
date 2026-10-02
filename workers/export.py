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

            # THE contract point for "virtual never leaves as data".
            #
            # A virtual track is a concept-only placeholder (``virtual: True``,
            # no ``audio_path``). It is legitimate inside the app — the gap audit
            # and the captioner both use it — but it is NOT training data, and
            # every consumer below is fed from this single list. Filtering here
            # covers all five formats at once; filtering inside ``export_json``
            # alone would still leak through CSV, JSONL, sidecars and the
            # train/val folders. The rule itself lives in
            # ``modules.exporters.for_export`` so it can be tested on its own.
            samples = exporters.for_export(samples)

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