"""Dataset export + train/val split helpers.

Formats:
  * ACE-Step JSON   — the manifest itself
  * CSV             — flat table (id, filename, genre, bpm, key, duration, caption…)
  * JSONL           — one sample object per line
  * Sidecar .txt    — a caption file per track (Kohya/ComfyUI convention)
  * Train/Val folders — ``train/`` + ``val/`` with audio copies, caption
    sidecars, and a ``manifest.json`` — ready to hand to a training run.
"""
import csv
import json
import os
import random
import shutil
from pathlib import Path


def split_dataset(samples, val_ratio=0.2, seed=42, stratify=True):
    """Split into (train, val), optionally stratified by genre."""
    if val_ratio <= 0:
        return list(samples), []
    rng = random.Random(seed)
    train, val = [], []
    if stratify:
        groups = {}
        for s in samples:
            g = (s.get("genre") or "").strip() or "?"
            groups.setdefault(g, []).append(s)
        for group in groups.values():
            rng.shuffle(group)
            n_val = max(0, int(round(len(group) * val_ratio)))
            val.extend(group[:n_val])
            train.extend(group[n_val:])
    else:
        pool = list(samples)
        rng.shuffle(pool)
        n_val = max(0, int(round(len(pool) * val_ratio)))
        val = pool[:n_val]
        train = pool[n_val:]
    return train, val


def for_export(samples):
    """The samples an export may write: everything except virtual placeholders.

    A virtual track (``virtual: True``) is a concept-only placeholder with no
    ``audio_path``. It is real inside the app — the gap audit and the captioner
    use it — but it is not training data, so it must never appear in a manifest,
    a CSV row, a JSONL line, a sidecar, or a train/val folder.

    This is the ONE place that decision is made for CSV / JSONL / sidecars /
    folders. ``to_export_sample`` cannot make it: it strips the ``virtual`` key
    but still emits a row, whose ``audio_path``/``filename`` would point at a
    ``virtual_01_*.wav`` that does not exist on disk. (``export_json`` applies
    it again to the dataset it is handed, so it is correct standalone too.)
    """
    return [s for s in samples if not s.get("virtual")]


_CSV_FIELDS = ["id", "filename", "audio_path", "genre", "bpm", "keyscale",
               "duration", "is_instrumental", "custom_tag", "caption"]


def export_csv(samples, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s.get(k, "") for k in _CSV_FIELDS})


def export_jsonl(samples, path):
    """One canonical sample object per line.

    Routed through ``to_export_sample`` so a JSONL line carries the same field
    names as the manifest, and no app-internal keys (``virtual``,
    ``structural_segments``, ...) leak into a downstream tool that reads it.
    """
    from modules.dataset_schema import to_export_sample

    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(to_export_sample(s), ensure_ascii=False) + "\n")


def export_sidecar_captions(samples, dest_dir):
    """Write ``<stem>.txt`` next to the track (or in ``dest_dir`` when copying).

    The text is the sample's caption, falling back to its tag.
    Returns the number written.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for s in samples:
        audio = s.get("audio_path", "")
        if not audio:
            continue
        text = (s.get("caption") or "").strip() or (s.get("custom_tag") or "").strip()
        stem = Path(audio).stem
        (dest_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        written += 1
    return written


def export_folders(samples, dest_dir, val_ratio=0.2, seed=42, stratify=True):
    """Copy audio into ``train/`` + ``val/`` with caption sidecars + manifest."""
    dest_dir = Path(dest_dir)
    train, val = split_dataset(samples, val_ratio=val_ratio, seed=seed, stratify=stratify)
    manifest = []
    for split_name, group in (("train", train), ("val", val)):
        out = dest_dir / split_name
        out.mkdir(parents=True, exist_ok=True)
        for s in group:
            audio = s.get("audio_path", "")
            if not audio or not os.path.exists(audio):
                continue
            fname = Path(audio).name
            shutil.copy2(audio, out / fname)
            cap = (s.get("caption") or "").strip() or (s.get("custom_tag") or "").strip()
            (out / f"{Path(audio).stem}.txt").write_text(cap, encoding="utf-8")
            manifest.append({
                "split": split_name,
                "filename": fname,
                "genre": s.get("genre", ""),
                "bpm": s.get("bpm", 0),
                "keyscale": s.get("keyscale", ""),
                "duration": s.get("duration", 0),
                "caption": cap,
            })
    (dest_dir / "manifest.json").write_text(
        json.dumps({"val_ratio": val_ratio, "stratify": stratify, "tracks": manifest},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return {"train": len(train), "val": len(val)}


def export_json(dataset, path):
    """Write the dataset as ACE-Step 1.5 training JSON.

    Takes the *whole dataset*, not a bare sample list: ACE-Step reads a
    dataset-level ``metadata`` block (``custom_tag``, ``tag_position``,
    ``genre_ratio``) that a sample list cannot supply
    (``acestep/training_v2/preprocess_discovery.py:145-177``). The field names
    come from ``modules.dataset_schema.to_export_dataset`` so the canonical
    mapping stays in exactly one place.

    Regression this guards: this function used to hand-roll
    ``{"metadata": {}, ...}``, which discarded the entire metadata block -- the
    trigger word and the caption/genre policy never reached training.
    """
    from modules.dataset_schema import to_export_dataset

    document = dict(dataset or {})
    document["samples"] = for_export(document.get("samples", []) or [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_export_dataset(document), f, indent=2, ensure_ascii=False)