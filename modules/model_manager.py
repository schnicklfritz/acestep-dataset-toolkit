"""Model download manager for the curated catalog (``models.json``).

Download sources:
  * **Hugging Face** (default) — ``huggingface_hub.snapshot_download``; gated
    models use the configurable ``hf_token``.
  * **GitHub repo** — a release-asset URL downloads that file; a plain repo URL
    downloads the repo archive (codeload) and extracts it.

Local downloads live in the configurable ``model_dir`` (default ``models/``,
gitignored). MVSEP API-only models (e.g. BS PolarFormer) are intentionally not
downloadable — they run via the MVSEP service.
"""
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

CATALOG_PATH = Path(__file__).resolve().parent.parent / "models.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_catalog():
    """Return the parsed ``models.json`` catalog."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_model(model_id):
    """Return the catalog entry for ``model_id`` or ``None``."""
    catalog = load_catalog()
    for m in catalog.get("models", []):
        if m.get("id") == model_id:
            return m
    return None


def leaderboards():
    """Return the catalog's leaderboard list (``[{"name", "url"}]``)."""
    return load_catalog().get("leaderboards", [])


def model_dir(config):
    """Resolve the local model directory (relative to the project by default)."""
    d = str(config.get("model_dir") or "models").strip() or "models"
    p = Path(d)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def is_downloaded(config, model_entry):
    """True when the model's local directory exists and is non-empty."""
    local = model_dir(config) / model_entry["id"]
    return local.exists() and any(local.iterdir())


def download_model(config, model_entry, progress_cb=None):
    """Download a catalog model into the local model dir. Returns the local path."""
    progress_cb = progress_cb or (lambda p, m: None)
    source = str(config.get("model_download_source") or "hf").strip().lower()
    local = model_dir(config) / model_entry["id"]
    local.mkdir(parents=True, exist_ok=True)
    if source == "github":
        return _download_github(config, model_entry, local, progress_cb)
    return _download_hf(config, model_entry, local, progress_cb)


def remove_model(config, model_entry):
    """Delete the model's local directory. Returns True if anything was removed."""
    local = model_dir(config) / model_entry["id"]
    if local.exists():
        shutil.rmtree(local, ignore_errors=True)
        return True
    return False


def _download_hf(config, model_entry, local, progress_cb):
    repo = (model_entry.get("hf_repo") or "").strip()
    if not repo:
        raise RuntimeError(
            f"'{model_entry['id']}' has no Hugging Face repo — switch the "
            "download source to GitHub, or run it via its preferred backend."
        )
    from huggingface_hub import snapshot_download

    progress_cb(10, f"Downloading {repo} from Hugging Face...")
    snapshot_download(
        repo_id=repo,
        token=(config.get("hf_token") or "").strip() or None,
        local_dir=str(local),
    )
    progress_cb(100, f"Downloaded {model_entry['id']}")
    return str(local)


def _repo_archive_url(url):
    """Turn ``https://github.com/owner/repo`` into a codeload tarball URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = [p for p in url.split("/") if p]
    try:
        owner, repo = parts[-2], parts[-1]
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Could not parse GitHub repo URL: {url}") from e
    return f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/main"


def _download_and_extract_archive(url, dest):
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with tempfile.TemporaryDirectory(prefix="model_dl_") as tmp:
        archive = os.path.join(tmp, "model.bin")
        with open(archive, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        extracted = os.path.join(tmp, "extracted")
        os.makedirs(extracted, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(extracted)
        # If extraction produced a single top-level directory, use its contents.
        entries = [e for e in os.listdir(extracted)]
        if len(entries) == 1 and os.path.isdir(os.path.join(extracted, entries[0])):
            src = os.path.join(extracted, entries[0])
        else:
            src = extracted
        for name in os.listdir(src):
            shutil.move(os.path.join(src, name), os.path.join(dest, name))


def _download_github(config, model_entry, local, progress_cb):
    url = (model_entry.get("github_url") or "").strip()
    if not url:
        raise RuntimeError(f"'{model_entry['id']}' has no GitHub URL.")
    progress_cb(10, f"Downloading {model_entry['id']} from GitHub...")
    if url.endswith(".zip") or url.endswith(".tar.gz"):
        _download_and_extract_archive(url, local)
    else:
        _download_and_extract_archive(_repo_archive_url(url), local)
    progress_cb(100, f"Downloaded {model_entry['id']}")
    return str(local)