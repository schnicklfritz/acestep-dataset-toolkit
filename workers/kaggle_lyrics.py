"""Kaggle GPU lyrics transcription via WhisperX (or the ACE-Step transcriber).

Pushes the ``kernels/lyrics_kernel.py`` script (with the chosen engine baked
in) to a private Kaggle GPU kernel, waits for completion, downloads the
output and returns the transcribed lyrics text.

Mirrors the Kaggle push/poll/download pattern already used by
``workers/kaggle_stems.py`` for Demucs stem separation, and by
``workers/caption.py`` for the AI captioner.
"""
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal

KERNEL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "kernels" / "lyrics_kernel.py"
)


def run_kaggle_lyrics(audio_path, config, mode="whisperx", language=None,
                       initial_prompt=None, progress_cb=None):
    """Run WhisperX (or the ACE-Step transcriber) on a Kaggle GPU kernel and
    return the transcribed lyrics text.

    This is the reusable, headless core of :class:`KaggleLyricsWorker` so any
    caller can use the same "Kaggle GPU" lyrics backend without touching Qt.
    ``progress_cb(pct, msg)`` is optional.

    Returns the transcribed lyrics as a single string.
    """
    if progress_cb is None:
        progress_cb = lambda p, m: None
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    from modules.kaggle import (
        upload_audio_dataset, push_kernel, wait_kernel_done,
        download_kernel_output,
    )

    mode = str(mode or "whisperx").lower().strip()
    if mode not in ("whisperx", "acestep"):
        mode = "whisperx"
    language = (language or "").strip() or "en"
    initial_prompt = (initial_prompt or "").strip()

    temp_dir = tempfile.mkdtemp(prefix="ace_lyrics_")
    try:
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        shutil.copy2(audio_path, os.path.join(audio_dir, os.path.basename(audio_path)))

        # ---- upload audio as a private Kaggle dataset ----
        progress_cb(8, "Uploading audio to a private Kaggle dataset...")
        audio_slug = upload_audio_dataset(config, audio_dir)
        audio_name = audio_slug.split("/")[-1]

        # ---- kernel script with engine + input baked in ----
        script = KERNEL_SCRIPT.read_text(encoding="utf-8")
        script = script.replace("{{MODE}}", mode)
        script = script.replace("{{LANGUAGE}}", language)
        script = script.replace("{{INITIAL_PROMPT}}", initial_prompt)
        script = script.replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{audio_name}")

        kernel_slug = f"ace-lyrics-{uuid.uuid4().hex[:6]}"
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

        progress_cb(10, f"Pushing lyrics-transcription kernel to Kaggle ({mode})...")
        push_kernel(config, kernel_dir, kernel_slug)
        progress_cb(25, "Kaggle GPU job queued...")
        wait_kernel_done(config, kernel_slug)

        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(config, kernel_slug, out_dir)

        result_path = os.path.join(out_dir, "lyrics_output.json")
        if not os.path.exists(result_path):
            raise RuntimeError(
                "Kaggle job finished but produced no lyrics_output.json -- "
                "check the kernel output."
            )
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        lyrics = (data.get("lyrics") or "").strip()
        if not lyrics:
            raise RuntimeError("Kaggle job produced an empty transcript.")
        progress_cb(100, "Lyrics transcription complete.")
        return lyrics
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class KaggleLyricsWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(dict)  # {"lyrics": str} — matches TranscribeLyricsWorker's shape
    failed = Signal(str)

    def __init__(self, audio_path, language=None, initial_prompt=None,
                 config=None, mode="whisperx", parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.language = language
        self.initial_prompt = initial_prompt
        self.config = config or {}
        self.mode = mode
        self._is_cancelled = False

    def run(self):
        try:
            lyrics = run_kaggle_lyrics(
                self.audio_path, self.config, mode=self.mode,
                language=self.language, initial_prompt=self.initial_prompt,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit({"lyrics": lyrics})
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

    def cancel(self):
        self._is_cancelled = True
