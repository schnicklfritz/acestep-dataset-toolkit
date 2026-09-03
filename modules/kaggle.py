"""Shared Kaggle helpers: private audio-dataset upload + kernel push/wait/download.

The correct way to feed audio to a Kaggle kernel is as a **dataset source** —
files copied into the pushed kernel directory are NOT reliably included. These
helpers upload the staged audio as a private Kaggle dataset and attach it via
``dataset_sources`` in the kernel metadata.

This module talks to Kaggle through the official ``kaggle`` Python package's
API client (``KaggleApi``) instead of shelling out to the ``kaggle`` console
script via ``subprocess``. Calling the API directly means this only requires
``kaggle`` to be *importable* (guaranteed by ``pip install kaggle``) — it does
not depend on a ``kaggle`` executable being resolvable on the OS ``PATH``,
which is what caused ``[Errno 2] No such file or directory: 'kaggle'`` under
the old subprocess-based implementation.
"""
import json
import os
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


def kaggle_available():
    """True if the ``kaggle`` package is importable in this environment."""
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _get_api(config):
    """Authenticate and return an authenticated ``(KaggleApi, username)`` pair.

    The import is deliberately local to this function (not module-level) so
    that ``modules.kaggle`` stays importable even when the ``kaggle`` package
    is not installed — only code paths that actually need Kaggle pay the
    price of a missing dependency, and they get a clear ``ImportError``
    instead of the rest of the app failing to start.
    """
    user = ensure_kaggle_creds(config)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError(
            "The 'kaggle' package is not installed. Run: pip install kaggle "
            "(or switch the engine/backend to a non-Kaggle option in Settings)."
        ) from e
    api = KaggleApi()
    api.authenticate()
    return api, user


def upload_audio_dataset(config, audio_dir, title_prefix="ace-audio"):
    """Upload a directory of audio as a private Kaggle dataset.

    Returns the ``user/slug`` of the new dataset.
    """
    api, user = _get_api(config)
    slug = f"{title_prefix}-{uuid.uuid4().hex[:6]}"
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "isPrivate": True,
        "licenses": [{"name": "unknown"}],
    }
    with open(os.path.join(audio_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    try:
        api.dataset_create_new(folder=audio_dir, dir_mode="skip")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle dataset upload failed: {e}") from e

    return f"{user}/{slug}"


def push_kernel(config, kernel_dir, kernel_slug):
    """Push a kernel directory. Returns ``user/kernel_slug``."""
    api, user = _get_api(config)
    try:
        api.kernels_push(kernel_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle kernel push failed: {e}") from e

    return f"{user}/{kernel_slug}"


def wait_kernel_done(config, kernel_slug, timeout=1200, poll_seconds=20):
    """Poll the kernel status until complete/failed. Returns bool success."""
    api, user = _get_api(config)
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        try:
            status = api.kernels_status(f"{user}/{kernel_slug}")
        except Exception:  # noqa: BLE001 — transient network hiccup, keep polling
            continue

        out = str(status).lower()
        if "complete" in out:
            return True
        if "error" in out or "failed" in out or "cancelled" in out:
            return False

    return False


def download_kernel_output(config, kernel_slug, out_dir):
    """Download the kernel output folder. Returns out_dir."""
    api, user = _get_api(config)
    os.makedirs(out_dir, exist_ok=True)
    try:
        api.kernels_output(f"{user}/{kernel_slug}", path=out_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle kernel output download failed: {e}") from e

    return out_dir
