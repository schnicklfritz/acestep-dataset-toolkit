"""Dataset export + train/val split helpers.

Formats:
  * ACE-Step JSON   — the manifest itself
  * CSV             — flat table (id, filename, genre, bpm, key, duration, caption…)
  * JSONL           — one sample object per line
  * Sidecar         — a ``<stem>.caption.txt`` file per track (see
    ``export_sidecar_captions`` for why it is not a bare ``<stem>.txt``)
  * JSON sidecar    — a ``<stem>.json`` per track carrying the five fields
    ACE-Step's Dataset Builder reads off the audio (see
    ``export_track_sidecar_json`` for why it holds only those five)
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
    """Write ``<stem>.caption.txt`` next to the track (or in ``dest_dir``).

    The text is the sample's caption, falling back to its tag.
    Returns the number written.

    The name is ``.caption.txt`` and deliberately NOT a bare ``<stem>.txt``.
    ACE-Step reads a bare ``<stem>.txt`` as **lyrics**, not as a caption, and a
    lyrics file present for a track also forces ``is_instrumental=false``
    (``acestep/training/dataset_builder_modules/scan.py:66-68``, and the lyric
    lookup at ``audio_io.py:72-85``). The caption is read only from
    ``<stem>.caption.txt`` (``audio_io.py:30-38``). A caption written to
    ``<stem>.txt`` therefore arrives as the LYRIC string and its real caption is
    silently dropped, so the sample trains on lyrics it never had.
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
        (dest_dir / f"{stem}.caption.txt").write_text(text, encoding="utf-8")
        written += 1
    return written


#: The keys ACE-Step reads out of a ``<stem>.json`` sidecar -- and ONLY these.
#: ``acestep/training/dataset_builder_modules/scan.py:84-97`` looks up each of
#: these with a truthiness test, so a blank value is skipped rather than fatal,
#: and no other key is read at all.
SIDECAR_JSON_FIELDS = ("caption", "bpm", "keyscale", "timesignature", "language")


def track_sidecar_json_path(audio_path):
    """The path ACE-Step looks for a sidecar at: the audio's ``<stem>.json``.

    ``Path.with_suffix`` is the same split ACE-Step performs and the toolkit
    already uses for caption sidecars, so ``song.v2.mp3`` becomes
    ``song.v2.json`` on both sides and a basename containing dots cannot drift.
    """
    return str(Path(audio_path).with_suffix(".json")) if audio_path else ""


def export_track_sidecar_json(sample, path=None):
    """Write ONE track's ACE-Step sidecar JSON. Returns the path, or ``None``.

    This is the file ``acestep/training/dataset_builder_modules/scan.py:84-97``
    reads next to the audio, under the audio's own basename. Exactly the five
    ``SIDECAR_JSON_FIELDS`` are written; every other key on the sample is
    dropped, because ACE-Step reads no others here -- a stray key would look
    like it did something and silently do nothing.

    Three traps, recorded because each one is invisible at the call site:

    * **Location is the contract.** The file is read only when it sits beside
      the audio under the audio's basename. A copy anywhere else is dead weight,
      which is why the default is ``track_sidecar_json_path``.
    * **No ``lyrics`` key, deliberately.** ACE-Step takes lyrics from
      ``<stem>.lyrics.txt`` (or the legacy ``<stem>.txt``) and never from the
      JSON sidecar (``audio_io.py:72-85``), so a lyrics key written here would
      be ignored while reading like a feature.
    * **``language`` has a side effect.** Any value other than
      ``instrumental`` forces ``is_instrumental=false`` on that path
      (``scan.py:94-97``), so blanking it for an instrumental track is not
      cosmetic.

    The v2 JSON trainer (``--dataset-json``) does not read sidecars at all --
    this format is for the Dataset Builder / Gradio / API route.
    """
    sample = sample or {}
    audio = sample.get("audio_path", "")
    if not audio:
        return None
    target = Path(path or track_sidecar_json_path(audio))
    payload = {}
    for key in SIDECAR_JSON_FIELDS:
        value = sample.get(key, "")
        # Blank stays an empty string rather than null: scan.py's truthiness
        # test skips both, and "" keeps the file readable if it is opened.
        payload[key] = "" if value is None else value
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)


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
            # ``.caption.txt``, not ``.txt``: a bare ``<stem>.txt`` is read by
            # ACE-Step as lyrics (see ``export_sidecar_captions``).
            (out / f"{Path(audio).stem}.caption.txt").write_text(cap, encoding="utf-8")
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