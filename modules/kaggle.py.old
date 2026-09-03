"""Shared Kaggle helpers: private audio-dataset upload + kernel push/wait/download.

The correct way to feed audio to a Kaggle kernel is as a **dataset source** —
files copied into the pushed kernel directory are NOT reliably included. These
helpers upload the staged audio as a private Kaggle dataset and attach it via
``dataset_sources`` in the kernel metadata.
"""
import json
import os
import subprocess
import time
import uuid


def ensure_kaggle_creds(config):
    """Validate Kaggle credentials and set env vars. Returns the username."""
    user = config.get("kaggle_user", "").strip()
    key = config.get("kaggle_key", "").strip()
    if not user or not key:
        raise ValueError(
            "Kaggle credentials not configured. Open ⚙ Settings to enter "
            "your Username & Key."
        )
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key
    return user


def upload_audio_dataset(config, audio_dir, title_prefix="ace-audio"):
    """Upload a directory of audio as a private Kaggle dataset.

    Returns the ``user/slug`` of the new dataset.
    """
    user = ensure_kaggle_creds(config)
    slug = f"{title_prefix}-{uuid.uuid4().hex[:6]}"
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "isPrivate": True,
        "licenses": [{"name": "unknown"}],
    }
    with open(os.path.join(audio_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    res = subprocess.run(
        ["kaggle", "datasets", "create", "-p", audio_dir, "--dir-mode", "skip"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Kaggle dataset upload failed: {res.stderr[:300] or res.stdout[:300]}"
        )
    return f"{user}/{slug}"


def push_kernel(config, kernel_dir, kernel_slug):
    """Push a kernel directory. Returns ``user/kernel_slug``."""
    user = ensure_kaggle_creds(config)
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", kernel_dir],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Kaggle kernel push failed: {res.stderr[:300] or res.stdout[:300]}"
        )
    return f"{user}/{kernel_slug}"


def wait_kernel_done(config, kernel_slug, timeout=1200, poll_seconds=20):
    """Poll the kernel status until complete/failed. Returns bool success."""
    user = ensure_kaggle_creds(config)
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        res = subprocess.run(
            ["kaggle", "kernels", "status", f"{user}/{kernel_slug}"],
            capture_output=True, text=True,
        )
        out = (res.stdout + res.stderr).lower()
        if "complete" in out:
            return True
        if "error" in out or "failed" in out or "cancelled" in out:
            return False
    return False


def download_kernel_output(config, kernel_slug, out_dir):
    """Download the kernel output folder. Returns out_dir."""
    user = ensure_kaggle_creds(config)
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", f"{user}/{kernel_slug}", "-p", out_dir],
        capture_output=True,
    )
    return out_dir
