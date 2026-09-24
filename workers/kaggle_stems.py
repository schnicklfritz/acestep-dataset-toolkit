"""Kaggle GPU stem separation via Meta Demucs.

Pushes the ``kernels/stem_separation_kernel.py`` script (with the chosen
model baked in) to a private Kaggle GPU kernel, waits for completion, downloads
the output and copies the separated stems into the requested output directory.

Mirrors the Kaggle push/poll/download pattern already used by
``workers/caption.py`` for the AI captioner.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal

KERNEL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "kernels" / "stem_separation_kernel.py"
)

# Populated by run_kaggle_stems() so a caller can deep-link the finished kernel
# (and its downloadable output artifacts) in Kaggle's own web GUI for manual
# review — the app stays the control surface; Kaggle stays the inspection UI.
LAST_KERNEL_REF = ""     # e.g. "akronohio/ace-stems-XXXXXX"
LAST_DATASET_REF = ""    # e.g. "akronohio/ace-audio-XXXXXX"


def run_kaggle_stems(audio_path, config, model=None, two_stems=None,
                     output_dir=None, progress_cb=None):
    """Run Demucs on a Kaggle GPU kernel and return the local stem paths.

    This is the reusable, headless core of :class:`KaggleStemSeparator` so the
    Structural/Spatial pipelines can use the same "Separate via Kaggle (Demucs)"
    backend. ``progress_cb(pct, msg)`` is optional.

    Returns the list of local stem file paths (``<output_dir>/<track>__<stem>.wav``).
    """
    if progress_cb is None:
        progress_cb = lambda p, m: None
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    from modules.kaggle import (
        upload_audio_dataset, push_kernel, wait_kernel_done,
        download_kernel_output, preflight_kaggle,
    )

    # PREFLIGHT: verify the credentials before uploading the audio.
    preflight_kaggle(config)

    model = model or config.get("kaggle_stem_model", "htdemucs_ft")
    output_dir = output_dir or config.get("stem_output_dir") \
        or os.path.join(str(Path.home()), "mvsep_stems")
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="ace_stems_")
    try:
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        shutil.copy2(audio_path, os.path.join(audio_dir, os.path.basename(audio_path)))

        # ---- upload audio as a private Kaggle dataset ----
        progress_cb(8, "Uploading audio to a private Kaggle dataset...")
        audio_slug = upload_audio_dataset(config, audio_dir)
        audio_name = audio_slug.split("/")[-1]

        # Wait for the dataset to be fully versioned/mounted before pushing a
        # kernel that references it — otherwise /kaggle/input/<slug> is empty.
        progress_cb(9, "Waiting for dataset to be ready...")
        from modules.kaggle import wait_dataset_ready
        ready_reason = []
        if not wait_dataset_ready(config, audio_slug, reason=ready_reason):
            raise RuntimeError(
                f"Kaggle dataset {audio_slug} never became ready before timeout. "
                + " ".join(ready_reason)
            )

        # ---- kernel script with model + input baked in ----
        script = KERNEL_SCRIPT.read_text(encoding="utf-8")
        script = script.replace("{{MODEL}}", model)
        script = script.replace("{{TWO_STEMS}}", two_stems or "")
        script = script.replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{audio_name}")

        kernel_slug = f"ace-stems-{uuid.uuid4().hex[:6]}"
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

        progress_cb(10, "Pushing stem-separation kernel to Kaggle...")
        kernel_ref = push_kernel(config, kernel_dir, kernel_slug)
        global LAST_KERNEL_REF, LAST_DATASET_REF
        LAST_KERNEL_REF = kernel_ref
        LAST_DATASET_REF = audio_slug
        progress_cb(25, "Kaggle GPU job queued...")
        wait_kernel_done(config, kernel_slug)

        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(config, kernel_slug, out_dir)

        manifest = os.path.join(out_dir, "stems_manifest.json")
        if not os.path.exists(manifest):
            raise RuntimeError(
                "Kaggle job finished but produced no stems_manifest.json -- "
                "check the kernel output."
            )
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)

        stems = []
        for track, stem_map in data.items():
            for _stem_name, rel in stem_map.items():
                # The kernel writes stems under a `stems/` subfolder of its
                # /kaggle/working dir; the manifest paths (e.g.
                # "test_tone_0/drums.wav") are relative to that `stems/` folder.
                # Search the whole downloaded tree for the rel path so we find
                # the file whether it lands at <out>/stems/<rel> or <out>/<rel>.
                src = None
                for base in (out_dir, os.path.join(out_dir, "stems")):
                    cand = os.path.join(base, rel)
                    if os.path.exists(cand):
                        src = cand
                        break
                if src is None:
                    continue
                dest = os.path.join(output_dir, f"{track}__{os.path.basename(rel)}")
                shutil.copy2(src, dest)
                stems.append(dest)

        if not stems:
            raise RuntimeError("Kaggle job produced no stem files.")
        progress_cb(100, f"Stem separation complete: {len(stems)} file(s).")
        return stems
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def open_last_kernel_in_kaggle(open_dataset_too=False):
    """Open the most recent stem kernel (and optionally its dataset) in Kaggle.

    Returns nothing. When the push has already happened it opens the finish
    kernel page so the user can review the run log + downloadable stem files in
    Kaggle's own web GUI. Call after ``run_kaggle_stems`` so the URL is known.
    """
    import webbrowser
    opened = []
    if LAST_KERNEL_REF:
        url = f"https://www.kaggle.com/code/{LAST_KERNEL_REF}/output"
        webbrowser.open_new_tab(url)
        opened.append(url)
    if open_dataset_too and LAST_DATASET_REF:
        url = f"https://www.kaggle.com/datasets/{LAST_DATASET_REF}"
        webbrowser.open_new_tab(url)
        opened.append(url)
    return opened


def build_default_stem_plan(method="demucs"):
    """Return a recommended stem plan (the power-user 'which splitter' surface).

    Represents which stems to produce and, for MVSEP, the note/recommendation.
    The Kaggle/Demucs backend:
      - "full"      -> vocals + drums + bass + other (htdemucs_ft, MIT)
    The MVSEP backend is chosen by the caller via the live algorithm list.
    """
    return {
        "backend": "kaggle",          # "kaggle" | "mvsep"
        "model": "htdemucs_ft",       # ignored by mvsep (uses its own list)
        "split": "full",              # "full" | "vocals" | "instrumental" | "custom"
        "instruments": None,          # when split == "custom": list of stem names
        "note": (
            "Demucs htdemucs_ft (MIT) is fully open-source and runs free on "
            "Kaggle. For a cleaner instrumental rebuild consider the MVSEP "
            "PolarFormer first-stage instead (rebuilds instrumental to avoid "
            "artifacts), then this as the multi-stem split of the instrumental."
        ),
    }


class KaggleStemSeparator(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(list)   # list of local stem paths
    failed = Signal(str)

    def __init__(self, audio_path, config, model=None,
                 two_stems=None, output_dir=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.config = config
        self.model = model or config.get("kaggle_stem_model", "htdemucs_ft")
        self.two_stems = two_stems  # "vocals" -> vocals + no_vocals; None -> full
        self.output_dir = output_dir or config.get("stem_output_dir") \
            or os.path.join(str(Path.home()), "mvsep_stems")
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        try:
            stems = run_kaggle_stems(
                self.audio_path, self.config, model=self.model,
                two_stems=self.two_stems, output_dir=self.output_dir,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(stems)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
