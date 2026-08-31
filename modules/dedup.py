"""Near-duplicate + exact-duplicate audio detection.

Near-duplicates use librosa mel fingerprints (catches same-song-different-
encode). Exact duplicates use file MD5 (identical files). No heavy model.
"""
import hashlib
import os

import librosa
import numpy as np

FP_SR = 16000
FP_N_MELS = 24
FP_HOP = 1024
FP_MAX_SEC = 60
FP_DIM = 32


def fingerprint(path, max_sec=FP_MAX_SEC):
    """Return a normalized fingerprint vector, or ``None`` if unreadable."""
    y, sr = librosa.load(path, sr=FP_SR, mono=True, duration=max_sec)
    if len(y) < sr:  # too short to judge
        return None
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=FP_N_MELS, hop_length=FP_HOP
    )
    logmel = librosa.power_to_db(mel)
    vec = librosa.util.fix_length(logmel.mean(axis=1), size=FP_DIM)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return vec / norm


def find_near_duplicates(paths, threshold=0.95, progress_cb=None):
    """Return a list of ``(path_a, path_b, similarity)`` for near-duplicates.

    ``threshold`` is a cosine similarity in [0, 1] (default 0.95).
    """
    progress_cb = progress_cb or (lambda p, m: None)
    vecs = {}
    paths = [p for p in paths if p and os.path.exists(p)]
    for i, p in enumerate(paths):
        if not _probe_ok(p):   # skip corrupt/unreadable files (no hang)
            continue
        try:
            v = fingerprint(p)
            if v is not None:
                vecs[p] = v
        except Exception:  # noqa: BLE001
            continue
        progress_cb(int(100 * (i + 1) / max(1, len(paths))), f"Fingerprinting {os.path.basename(p)}")

    pairs = []
    items = list(vecs.items())
    for i in range(len(items)):
        pa, va = items[i]
        for j in range(i + 1, len(items)):
            pb, vb = items[j]
            sim = float(np.dot(va, vb))
            if sim >= threshold:
                pairs.append((pa, pb, sim))
    return pairs


def find_exact_duplicates(paths):
    """Return groups of byte-identical files (by MD5). Each group has >= 2 paths."""
    groups = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            h = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            groups.setdefault(h.hexdigest(), []).append(p)
        except Exception:  # noqa: BLE001
            continue
    return [g for g in groups.values() if len(g) > 1]


def _probe_ok(path, timeout=10):
    """Fast, non-hanging check that a file decodes (ffprobe)."""
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001 — if ffprobe is missing, don't over-flag
        return True