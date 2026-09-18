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


ACCESS_TOKEN_FILE = "~/.kaggle/access_token"


def ensure_kaggle_creds(config):
    """Validate Kaggle credentials and set env vars. Returns the username.

    Kaggle's modern ``kaggle`` SDK (>= 2.x) authenticates FIRST via a
    bearer access token (``KAGGLE_API_TOKEN`` / ``~/.kaggle/access_token``)
    before falling back to the legacy ``KAGGLE_USERNAME``/``KAGGLE_KEY``
    pair. The token issued from Settings > API (``KGAT_...``) is one of
    these new-style access tokens, so we must present it in the new fields
    or the API rejects it with ``401 Unauthorized``.

    ``kaggle_user`` is kept for building dataset/kernel slugs only (the SDK
    re-derives the real username from the token at auth time).
    """
    user = config.get("kaggle_user", "").strip()
    key = config.get("kaggle_key", "").strip()
    if not user or not key:
        raise ValueError(
            "Kaggle credentials not configured. Open ⚙ Settings to enter "
            "your Username & Key."
        )
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key
    # New access-token auth (takes precedence in the modern SDK).
    os.environ["KAGGLE_API_TOKEN"] = key
    try:
        token_path = os.path.expanduser(ACCESS_TOKEN_FILE)
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as f:
            f.write(key)
        os.chmod(token_path, 0o600)
    except OSError:
        pass  # env var alone is enough; file is a convenience fallback
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


def wait_dataset_ready(config, audio_slug, timeout=300, poll_seconds=10):
    """Poll a just-created private dataset until Kaggle reports it ``ready``.

    ``dataset_create_new`` returns before the dataset version is fully
    processed/mounted. Pushing a kernel that references the dataset before it
    is ready results in an empty ``/kaggle/input/<slug>`` mount and a silent
    empty manifest. Returns True when ready; False on timeout.
    """
    api, user = _get_api(config)
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        try:
            status = api.dataset_status(audio_slug)
            if isinstance(status, dict):
                status = status.get("status") or sorted(status.values())[-1] if status else ""
            if str(status).lower() in {"ready", "complete"}:
                return True
        except Exception:  # noqa: BLE001 — keep polling on transient errors
            continue
    return False


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


def fetch_kernel_logs(config, kernel_slug, max_chars=6000):
    """Return the TAIL of a kernel's execution log, or '' if unavailable.

    Failures surface at the end of the log, so the tail is what matters. The
    returned text is meant to be embedded in an error message: without it the
    user is told to go read Kaggle's web UI, which is a poor trade when the API
    can hand the reason over directly.
    """
    try:
        api, user = _get_api(config)
        text = api.kernels_logs(f"{user}/{kernel_slug}")
    except Exception:  # noqa: BLE001 — diagnostics must never mask the real error
        return ""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= max_chars else "…" + text[-max_chars:]


def kernel_status_text(config, kernel_slug):
    """Human-readable kernel status, or '' when it cannot be read."""
    try:
        api, user = _get_api(config)
        return str(api.kernels_status(f"{user}/{kernel_slug}"))
    except Exception:  # noqa: BLE001
        return ""


def kernel_stdout(config, kernel_slug):
    """Return what a kernel PRINTED to stdout, parsed from its log.

    ``kernels_logs`` returns a JSON array of stream events::

        [{"stream_name": "stdout", "time": 7.9, "data": "..."}, ...]

    Joining the ``stdout`` events gives the kernel's actual console output,
    which is how results are retrieved. This exists because ``kernels_output``
    turned out to be unreliable: it HUNG outright on a completed kernel (a 90s
    timeout with no output), which is what made an earlier run report "no
    moss_out.json" when the file had in fact been written.

    Falls back to the raw log text when it is not parseable as events.
    """
    raw = fetch_kernel_logs(config, kernel_slug, max_chars=10 ** 9)
    if not raw:
        return ""
    try:
        events = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(events, list):
        return raw
    return "".join(
        event.get("data", "")
        for event in events
        if isinstance(event, dict) and event.get("stream_name") == "stdout"
    )


def extract_marked_json(text, begin, end):
    """Return the JSON payload printed between two markers, or None.

    Markers beat scraping a log for a filename: the payload survives reordering,
    progress bars and unrelated output.
    """
    if not text:
        return None
    start = text.find(begin)
    if start < 0:
        return None
    start += len(begin)
    stop = text.find(end, start)
    if stop < 0:
        return None
    try:
        return json.loads(text[start:stop].strip())
    except ValueError:
        return None


def download_kernel_output(config, kernel_slug, out_dir):
    """Download the kernel output folder. Returns out_dir."""
    api, user = _get_api(config)
    os.makedirs(out_dir, exist_ok=True)
    try:
        api.kernels_output(f"{user}/{kernel_slug}", path=out_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle kernel output download failed: {e}") from e

    return out_dir
