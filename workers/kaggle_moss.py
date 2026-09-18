"""MOSS-Audio captioning on a Kaggle GPU, driven from the app.

Pushes ``kernels/moss_caption_kernel.py`` (placeholders filled in) to a private
Kaggle GPU kernel, waits, downloads ``moss_out.json`` and returns raw style +
lyrics text per track. The caller writes that into the dataset via
``modules.moss_import``; formatting is the existing tag_creator LLM stage.

Mirrors the push/poll/download pattern in ``workers/kaggle_stems.py``.

INTERNET IS REQUIRED
--------------------
The kernel git-clones OpenMOSS/MOSS-Audio (the model class is not on the Hub)
and pip-installs transformers==4.57.1, so ``enable_internet`` must be true.
"""
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal

KERNEL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "kernels" / "moss_caption_kernel.py"
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}

# MOSS-Audio-8B-Instruct: ~17 GiB, fits sharded across Kaggle's two T4s.
DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Music-8B-Instruct"

# The encoder caps a single pass at ~120 s (max_source_positions=1500 /
# audio_tokens_per_second=12.5), so 110 s leaves headroom.
DEFAULT_CHUNK_SECONDS = 110

DEFAULT_STYLE_PROMPT = (
    "Describe this music in detail for a training dataset. Cover the genre and "
    "subgenre, the mood and energy, every instrument you can identify, the "
    "vocal style and delivery, the production and mix character, and the era. "
    "Also describe how the music develops from start to finish. "
    "Do not mention BPM, key, or time signature."
)

DEFAULT_LYRICS_PROMPT = (
    "Transcribe the lyrics of this song exactly as sung, line by line. "
    "If a section is instrumental, write [Instrumental]. "
    "Do not translate, summarise, or add commentary."
)

# Populated by run_kaggle_moss so the caller can deep-link the finished kernel.
LAST_KERNEL_REF = ""
LAST_DATASET_REF = ""


def _moss_prompts(config):
    """Prompts + limits, all overridable from settings."""
    return {
        "model_id": (config.get("moss_model_id") or DEFAULT_MODEL_ID).strip(),
        "style": (config.get("moss_style_prompt") or DEFAULT_STYLE_PROMPT).strip(),
        "lyrics": (config.get("moss_lyrics_prompt") or DEFAULT_LYRICS_PROMPT).strip(),
        "max_tokens": int(config.get("moss_max_tokens", 1024) or 1024),
        "chunk_seconds": int(
            config.get("moss_chunk_seconds", DEFAULT_CHUNK_SECONDS)
            or DEFAULT_CHUNK_SECONDS
        ),
        # "" = leave the attention backend to the model. See config.py for why
        # this is not forced, and note flash-attn has no Turing (T4) support.
        "attn_impl": (config.get("moss_attn_implementation") or "").strip(),
    }


def _fill_placeholders(script, audio_input_path, prompts, custom_tag):
    """Substitute the kernel's placeholders.

    `{{AUDIO_DATASET_PATH}}` is written INSIDE quotes in the kernel, so it takes
    a bare value. The prompt/tag placeholders stand alone and take a JSON string
    literal (quotes included).
    """
    out = script
    out = out.replace("{{AUDIO_DATASET_PATH}}", audio_input_path)
    out = out.replace("{{MODEL_ID}}", json.dumps(prompts["model_id"]))
    out = out.replace("{{STYLE_PROMPT}}", json.dumps(prompts["style"]))
    out = out.replace("{{LYRICS_PROMPT}}", json.dumps(prompts["lyrics"]))
    out = out.replace("{{MAX_NEW_TOKENS}}", str(prompts["max_tokens"]))
    out = out.replace("{{CHUNK_SECONDS}}", str(prompts["chunk_seconds"]))
    out = out.replace("{{CUSTOM_TAG}}", json.dumps(custom_tag or ""))
    out = out.replace("{{ATTN_IMPL}}", json.dumps(prompts.get("attn_impl", "")))

    # REFUSE to return a script with placeholders this function cannot fill.
    #
    # This is how a real failure happened: the kernel file gained
    # `{{ATTN_IMPL}}` while the running app still had the OLD function in memory
    # (Python caches modules, so editing a file does not update a live process).
    # The placeholder went through unsubstituted, and because `{{X}}` is VALID
    # Python -- a set containing a set -- it failed on Kaggle 52 seconds later
    # with a bare `NameError: name 'ATTN_IMPL' is not defined`.
    #
    # A placeholder whose name never lands in the pushed script is a bug; a
    # placeholder that lands unsubstituted is a mystery. Fail here instead.
    leftover = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", out)))
    if leftover:
        raise RuntimeError(
            "The MOSS kernel contains placeholders this version cannot fill: "
            + ", ".join(leftover)
            + "\n\nThe app is running stale code: it reads the kernel from disk "
              "but substitutes using the function loaded at startup. Restart "
              "the app and try again."
        )
    return out


def run_kaggle_moss(audio_paths, config, custom_tag="", progress_cb=None):
    """Caption tracks with MOSS-Audio on a Kaggle GPU.

    ``audio_paths`` is the list of files to caption. Returns
    ``{filename: {"style":..., "lyrics":...}}`` — the same shape
    ``modules.moss_import.load_moss_output`` produces, so the caller can pass it
    straight to ``apply_moss_output``.
    """
    if progress_cb is None:
        progress_cb = lambda p, m: None
    if not audio_paths:
        raise ValueError("No audio files to caption.")

    from modules.kaggle import (
        download_kernel_output,
        push_kernel,
        upload_audio_dataset,
        wait_dataset_ready,
        wait_kernel_done,
    )

    prompts = _moss_prompts(config)
    temp_dir = tempfile.mkdtemp(prefix="ace_moss_")
    try:
        # ------------------------------------------------------------------
        # Validate the kernel's placeholders BEFORE any Kaggle round-trip.
        #
        # Filling with throwaway values runs the real substitution (and its
        # leftover guard), so a kernel file that has drifted ahead of this
        # build fails here -- instantly and for free -- rather than after an
        # upload, a kernel push and 52 seconds of Kaggle time.
        # ------------------------------------------------------------------
        script_template = KERNEL_SCRIPT.read_text(encoding="utf-8")
        _fill_placeholders(
            script_template, "/kaggle/input/probe",
            {"model_id": "", "style": "", "lyrics": "", "max_tokens": 1,
             "chunk_seconds": 1, "attn_impl": ""},
            "",
        )

        # ---- stage every track into one upload folder -------------------
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        staged = 0
        for path in audio_paths:
            if not os.path.exists(path):
                continue
            if Path(path).suffix.lower() not in AUDIO_EXTS:
                continue
            shutil.copy2(path, os.path.join(audio_dir, os.path.basename(path)))
            staged += 1
        if not staged:
            raise RuntimeError("None of the given files are supported audio.")

        progress_cb(5, f"Uploading {staged} track(s) to a private Kaggle dataset...")
        audio_slug = upload_audio_dataset(config, audio_dir)
        audio_name = audio_slug.split("/")[-1]

        progress_cb(8, "Waiting for the dataset to be ready...")
        if not wait_dataset_ready(config, audio_slug):
            raise RuntimeError(
                f"Kaggle dataset {audio_slug} never became ready before timeout."
            )

        # ---- build the kernel ------------------------------------------
        script = _fill_placeholders(
            script_template, f"/kaggle/input/{audio_name}", prompts, custom_tag
        )

        kernel_slug = f"ace-moss-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kernel")
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w",
                  encoding="utf-8") as f:
            f.write(script)


        user = config.get("kaggle_user", "").strip()
        sources = [audio_slug]
        # Optional: a cached weights dataset avoids a ~17 GiB download INSIDE
        # the kernel. The kernel finds it via its /kaggle/input scan.
        cached_weights = (config.get("moss_model_dataset") or "").strip()
        if cached_weights:
            sources.append(cached_weights)

        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            # Required: the kernel git-clones MOSS-Audio and pip-installs.
            "enable_internet": "true",
            "dataset_sources": sources,
            "competition_sources": [],
            "kernel_sources": [],
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        progress_cb(12, "Pushing the MOSS-Audio kernel to Kaggle...")
        kernel_ref = push_kernel(config, kernel_dir, kernel_slug)
        global LAST_KERNEL_REF, LAST_DATASET_REF
        LAST_KERNEL_REF = kernel_ref
        LAST_DATASET_REF = audio_slug

        progress_cb(20, "Kaggle GPU job queued (install + weight load takes a "
                        "while; ~17 GiB on the first run)...")
        # Generous timeout: clone, pip install, model download and two passes
        # per track all happen inside this window.
        if not wait_kernel_done(config, kernel_slug, timeout=10800):
            # Do not just tell the user to go read Kaggle -- fetch the reason.
            from modules.kaggle import fetch_kernel_logs, kernel_status_text
            status = kernel_status_text(config, kernel_slug)
            tail = fetch_kernel_logs(config, kernel_slug)
            raise RuntimeError(
                "Kaggle kernel did not succeed.\n\n"
                f"Kernel: {kernel_ref}\n"
                f"Final status: {status or '(unknown)'}\n\n"
                "Common causes:\n"
                "  - ran past the 3h wait window (17 GiB weight download + two\n"
                "    passes per track is slow on a T4)\n"
                "  - out of disk while downloading the weights (Kaggle gives a\n"
                "    few tens of GB; use the 'Cached weights' dataset instead)\n"
                "  - pip could not install the pinned transformers\n"
                "  - out of VRAM loading the 8B model\n\n"
                f"--- log tail ---\n{tail or '(log unavailable)'}"
            )

        progress_cb(85, "Downloading results...")
        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(config, kernel_slug, out_dir)

        result_path = None
        for base, _dirs, names in os.walk(out_dir):
            if "moss_out.json" in names:
                result_path = os.path.join(base, "moss_out.json")
                break
        if not result_path:
            # List what WAS downloaded: "no output" alone gives nothing to act
            # on, and this failure has already happened once unexplained.
            found = sorted(
                os.path.relpath(os.path.join(base, n), out_dir)
                for base, _dirs, names in os.walk(out_dir)
                for n in names
            )
            raise RuntimeError(
                "Kaggle job finished but no moss_out.json was downloaded.\n\n"
                f"Files actually downloaded ({len(found)}):\n  "
                + ("\n  ".join(found[:40]) or "(none)")
                + "\n\nThe kernel normally writes /kaggle/working/moss_out.json. "
                  "Check the run log for the real cause."
            )
        with open(result_path, encoding="utf-8") as f:
            payload = json.load(f)

        results = {}
        for entry in payload.get("results", []):
            name = (entry.get("file") or "").strip()
            if not name:
                continue
            results[name] = {
                "style": (entry.get("style") or "").strip(),
                "lyrics": (entry.get("lyrics") or "").strip(),
                "error": (entry.get("style") or "").startswith("ERROR:"),
            }
        progress_cb(100, f"MOSS finished: {len(results)} track(s).")
        return results
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class KaggleMossWorker(QThread):
    """Background wrapper so the UI stays responsive during a Kaggle run."""

    progress = Signal(int, str)
    finished_ok = Signal(dict)   # {filename: {style, lyrics, error}}
    failed = Signal(str)

    def __init__(self, audio_paths, config, custom_tag="", parent=None):
        super().__init__(parent)
        self.audio_paths = list(audio_paths)
        self.config = config
        self.custom_tag = custom_tag

    def run(self):
        try:
            results = run_kaggle_moss(
                self.audio_paths, self.config, custom_tag=self.custom_tag,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(results)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

