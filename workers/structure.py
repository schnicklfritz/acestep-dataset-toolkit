"""SongFormer music-structure analysis via a Kaggle kernel.

Pushes ``kernels/structure_kernel.py`` (audio uploaded as a private Kaggle
dataset), waits for completion, downloads ``structure.json`` and returns
per-file labeled sections: ``[{"start", "end", "label"}]``.
"""
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

KERNEL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "kernels" / "structure_kernel.py"
)


def run_structure_analysis(audio_path, config, min_segment_sec=0, progress_cb=None):
    """Return labeled sections ``[{"start", "end", "label"}]`` for one audio file.

    Raises ``RuntimeError`` when the kernel finishes without structure.json;
    returns ``[]`` when the file wasn't in the results.
    """
    progress_cb = progress_cb or (lambda p, m: None)
    from modules.kaggle import (
        upload_audio_dataset, push_kernel, wait_kernel_done,
        download_kernel_output,
    )

    temp_dir = tempfile.mkdtemp(prefix="ace_structure_")
    try:
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        shutil.copy2(audio_path, os.path.join(audio_dir, os.path.basename(audio_path)))

        progress_cb(10, "Uploading audio for structure analysis...")
        audio_slug = upload_audio_dataset(config, audio_dir)
        audio_name = audio_slug.split("/")[-1]

        script = KERNEL_SCRIPT.read_text(encoding="utf-8")
        script = script.replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{audio_name}")
        script = script.replace("{{MIN_SEGMENT_SEC}}", str(int(min_segment_sec or 0)))

        kernel_slug = f"ace-structure-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kernel")
        os.makedirs(kernel_dir, exist_ok=True)
        Path(os.path.join(kernel_dir, "kernel_worker.py")).write_text(script, encoding="utf-8")

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
        Path(os.path.join(kernel_dir, "kernel-metadata.json")).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        progress_cb(25, "Pushing structure-analysis kernel to Kaggle...")
        push_kernel(config, kernel_dir, kernel_slug)
        progress_cb(35, "SongFormer analyzing structure...")
        wait_kernel_done(config, kernel_slug)

        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(config, kernel_slug, out_dir)

        struct_path = os.path.join(out_dir, "structure.json")
        if not os.path.exists(struct_path):
            raise RuntimeError(
                "Structure job finished without structure.json -- check the kernel logs."
            )
        with open(struct_path, encoding="utf-8") as f:
            data = json.load(f)
        target = os.path.basename(audio_path)
        for item in data.get("results", []):
            if item.get("file") == target:
                return item.get("segments", [])
        return []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)