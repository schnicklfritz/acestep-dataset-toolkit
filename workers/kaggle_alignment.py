"""Kaggle GPU word-level forced alignment (MMS_FA).

Pushes ``kernels/alignment_kernel.py`` to a private Kaggle GPU kernel, waits for
completion, downloads the output and returns per-word timings plus line-level
segments. Mirrors ``workers/kaggle_lyrics.py`` (same push/poll/download path).

The result is what the app was missing: ``modules.lyrics_tools.export_lrc``
reads ``sample["lyrics_segments"]``, and until now nothing wrote that field, so
"Export .lrc" always fell back to untimed text.
"""
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal

KERNEL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "kernels" / "alignment_kernel.py"
)


def run_kaggle_alignment(audio_path, lyrics, config, progress_cb=None):
    """Align ``lyrics`` against ``audio_path`` on a Kaggle GPU; return the result.

    ``progress_cb(pct, msg)`` is optional. The returned dict is the kernel's
    per-track record: ``{"segments": [...], "words": [...], "lyrics": str,
    "shout_ratio": float}``. ``segments`` is exactly the ``[{"start", "text"}]``
    shape :func:`modules.lyrics_tools.export_lrc` consumes.

    Raises ``RuntimeError`` with the kernel's own reason when a file could not be
    aligned -- a silent empty result would look like "no words were shouted".
    """
    if progress_cb is None:
        progress_cb = lambda p, m: None
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not (lyrics or "").strip():
        raise ValueError("This track has no lyrics to align.")

    from modules.kaggle import (
        upload_audio_dataset, push_kernel, wait_kernel_done,
        wait_dataset_ready, download_kernel_output, preflight_kaggle,
    )

    # PREFLIGHT: verify credentials before uploading anything large.
    preflight_kaggle(config)

    audio_name = os.path.basename(audio_path)
    temp_dir = tempfile.mkdtemp(prefix="ace_align_")
    try:
        stage_dir = os.path.join(temp_dir, "audio")
        os.makedirs(stage_dir, exist_ok=True)
        shutil.copy2(audio_path, os.path.join(stage_dir, audio_name))
        # Keyed by filename AND by stem: the kernel prefers a vocal stem when one
        # is staged, and that file's name is not the one we were handed.
        stem = os.path.splitext(audio_name)[0]
        with open(os.path.join(stage_dir, "lyrics.json"), "w", encoding="utf-8") as f:
            json.dump({audio_name: lyrics, stem: lyrics}, f, indent=1)

        progress_cb(8, "Uploading audio + lyrics to a private Kaggle dataset...")
        audio_slug = upload_audio_dataset(config, stage_dir)
        # dataset_create_new returns before the version is mounted; pushing the
        # kernel early yields an EMPTY /kaggle/input/<slug>.
        progress_cb(12, "Waiting for the Kaggle dataset to finish processing...")
        reason = []
        if not wait_dataset_ready(config, audio_slug, reason=reason):
            raise RuntimeError(
                f"Kaggle dataset {audio_slug} did not become ready in time; "
                "the kernel would mount an EMPTY folder. "
                + " ".join(reason) + " Re-run."
            )
        dataset_name = audio_slug.split("/")[-1]

        script = KERNEL_SCRIPT.read_text(encoding="utf-8")
        script = script.replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{dataset_name}")

        kernel_slug = f"ace-align-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kernel")
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w", encoding="utf-8") as f:
            f.write(script)

        user = config.get("kaggle_user", "").strip()
        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [audio_slug],
            "competition_sources": [],
            "kernel_sources": [],
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        progress_cb(10, "Pushing forced-alignment kernel to Kaggle...")
        push_kernel(config, kernel_dir, kernel_slug)
        progress_cb(25, "Kaggle GPU job queued (MMS_FA downloads ~1 GB of weights)...")
        wait_kernel_done(config, kernel_slug)

        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(config, kernel_slug, out_dir)

        result_path = os.path.join(out_dir, "alignment_out.json")
        if not os.path.exists(result_path):
            raise RuntimeError(
                "Kaggle job finished but produced no alignment_out.json -- "
                "check the kernel output."
            )
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        tracks = data.get("tracks") or {}
        record = tracks.get(audio_name)
        if record is None and len(tracks) == 1:
            record = next(iter(tracks.values()))
        if record is None:
            reason = (data.get("errors") or {}).get(audio_name, "no result for this file")
            raise RuntimeError(f"Kaggle alignment failed: {reason}")

        progress_cb(100, "Alignment complete.")
        return record
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class KaggleAlignmentWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(dict)   # the track record from run_kaggle_alignment
    failed = Signal(str)

    def __init__(self, audio_path, lyrics, config=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.lyrics = lyrics
        self.config = config or {}
        self._is_cancelled = False

    def run(self):
        try:
            result = run_kaggle_alignment(
                self.audio_path, self.lyrics, self.config,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

    def cancel(self):
        self._is_cancelled = True

