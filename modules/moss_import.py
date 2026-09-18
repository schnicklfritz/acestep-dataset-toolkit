"""Import MOSS-Audio kernel output into a dataset JSON.

The MOSS kernel (``kernels/moss_caption_kernel.py``) emits RAW text and
deliberately does not format it:

    {"results": [{"file": "War_Pigs.flac", "style": "...", "lyrics": "..."}]}

This module maps that back onto tracks **by filename** and writes it into the
two fields the LLM formatting stage reads (``modules/tag_creator.py`` builds its
prompt from ``sample["caption"]`` and ``sample["lyrics"]``):

    style  -> caption      (+ caption_before_moss keeps the previous value)
    lyrics -> raw_lyrics   (+ raw_lyrics_before_moss keeps the previous value)
    prompt_override -> True

WHY prompt_override IS SET
--------------------------
ACE-Step's ``prompt_override`` is a BOOLEAN data-source switch:

    False -> the loader may auto-label the track with the captioner LLM
    True  -> bypass auto-labelling; use the handwritten genre/caption/lyrics

Hand-annotated text only reaches the encoder when this is ``True``, so leaving
it off would silently discard the work MOSS just did.

SAFETY
------
Nothing is overwritten destructively: the previous caption and lyrics are
copied into ``*_before_moss`` fields, and the dataset file itself is backed up
before the first write. ``--dry-run`` reports what would change and writes
nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

# Field names written by apply_moss_output.
STYLE_FIELD = "caption"
LYRICS_FIELD = "raw_lyrics"       # pre-formatting lyrics; tag_creator reads it
STYLE_BACKUP = "caption_before_moss"
LYRICS_BACKUP = "raw_lyrics_before_moss"


def load_moss_output(path):
    """Read the kernel's JSON into ``{filename: {"style":..., "lyrics":...}}``.

    An entry whose value looks like an error string is dropped rather than
    written into the dataset -- a failed pass must not become training data.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    out = {}
    for entry in results or []:
        name = (entry.get("file") or "").strip()
        if not name:
            continue
        style = (entry.get("style") or "").strip()
        lyrics = (entry.get("lyrics") or "").strip()
        if style.startswith("ERROR:") or lyrics.startswith("ERROR:"):
            # Keep the entry so the caller can report it, but mark it bad.
            out[name] = {"style": "", "lyrics": "", "error": True}
            continue
        out[name] = {"style": style, "lyrics": lyrics, "error": False}
    return out


def normalize_prompt_override(dataset, dry_run=False):
    """Coerce ``prompt_override`` to a real boolean on every sample.

    Two problems this fixes:

    * Older files stored the *string* ``"caption"`` here. That is truthy, so it
      happens to behave like True -- but the training loader expects a bool, and
      a string is mis-parsed at the boundary.
    * A sample with the key missing entirely yields ``None`` from ``.get()``,
      which is neither True nor False. The schema requires the field, so it is
      written explicitly.

    ``dry_run`` counts without mutating, so a preview reports the same number a
    real run would change.

    Returns the number of samples that need changing.
    """
    changed = 0
    for sample in dataset.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if "prompt_override" not in sample:
            if not dry_run:
                sample["prompt_override"] = False
            changed += 1
            continue
        current = sample["prompt_override"]
        if not isinstance(current, bool):
            if not dry_run:
                sample["prompt_override"] = bool(current)
            changed += 1
    return changed


def apply_moss_output(dataset, results, dry_run=False):
    """Write MOSS results into ``dataset`` in place.

    Returns a report dict:
        matched, unmatched, overwritten, filled, errored, prompt_override_fixed

    Matching keys on BOTH ``filename`` and ``basename(audio_path)``, because the
    kernel names each result after the staged file (``basename(audio_path)``)
    while a dataset may store a different ``filename``. Keying on only one of
    them silently produced "0 tracks written" with no clue why.

    A track with no result is left untouched -- never silently blanked.
    """
    report = {
        "matched": 0,
        "unknown_results": [],   # MOSS returned files this dataset does not have
        "no_result": [],         # dataset tracks MOSS returned nothing for
        "overwritten": 0, "filled": 0, "errored": [],
        "prompt_override_fixed": 0,
    }

    by_name = {}
    for sample in dataset.get("samples", []):
        if not isinstance(sample, dict):
            continue
        name = (sample.get("filename") or "").strip()
        if name:
            by_name.setdefault(name, sample)
        # Also index the on-disk basename: the kernel stages files by
        # basename(audio_path), so this is the name MOSS reports back.
        base = os.path.basename((sample.get("audio_path") or "").strip())
        if base:
            by_name.setdefault(base, sample)

    consumed = set()
    for filename, payload in results.items():
        sample = by_name.get(filename)
        if sample is None:
            report["unknown_results"].append(filename)
            continue
        consumed.add(id(sample))
        if payload.get("error"):
            report["errored"].append(filename)
            continue

        report["matched"] += 1
        style = payload.get("style") or ""
        lyrics = payload.get("lyrics") or ""

        # Classify by CONTENT, not by whether we are writing: a dry run must
        # report the same overwrite/fill split as a real run, or it is useless.
        if style:
            existing = (sample.get(STYLE_FIELD) or "").strip()
            if existing:
                report["overwritten"] += 1
                if not dry_run:
                    sample.setdefault(STYLE_BACKUP, sample[STYLE_FIELD])
            else:
                report["filled"] += 1
            if not dry_run:
                sample[STYLE_FIELD] = style

        if lyrics:
            previous = (sample.get(LYRICS_FIELD) or "").strip() or \
                       (sample.get("formatted_lyrics") or "").strip()
            if previous and not dry_run:
                sample.setdefault(LYRICS_BACKUP, previous)
            if not dry_run:
                sample[LYRICS_FIELD] = lyrics

        # Hand-annotated text only reaches the encoder when this is True.
        if not dry_run:
            sample["prompt_override"] = True

    # Dataset tracks MOSS returned nothing for: report, do not touch.
    for sample in dataset.get("samples", []):
        if not isinstance(sample, dict) or id(sample) in consumed:
            continue
        name = (sample.get("filename") or "").strip()
        if name:
            report["no_result"].append(name)

    report["prompt_override_fixed"] = normalize_prompt_override(
        dataset, dry_run=dry_run
    )

    return report


def backup_dataset(path):
    """Copy ``path`` aside before it is rewritten. Returns the backup path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.bak-{stamp}"
    shutil.copy2(path, backup)
    return backup


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="dataset JSON to update (backed up first)")
    parser.add_argument("--results", required=True,
                        help="moss_out.json from the Kaggle kernel")
    parser.add_argument("--out", default=None,
                        help="write here instead of updating --dataset in place")
    parser.add_argument("--dry-run", action="store_true",
                        help="report changes without writing anything")
    args = parser.parse_args(argv)

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    results = load_moss_output(args.results)

    report = apply_moss_output(dataset, results, dry_run=args.dry_run)

    print(f"matched          : {report['matched']}")
    print(f"overwritten      : {report['overwritten']} (previous kept in {STYLE_BACKUP})")
    print(f"filled (was blank): {report['filled']}")
    print(f"errored entries  : {len(report['errored'])}")
    print(f"prompt_override  : {report['prompt_override_fixed']} coerced to bool")
    if report["unknown_results"]:
        print(f"results with no matching track ({len(report['unknown_results'])}): "
              + ", ".join(report["unknown_results"][:10]))
    if report["no_result"]:
        print(f"tracks with no result ({len(report['no_result'])}): "
              + ", ".join(report["no_result"][:10]))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    target = args.out or args.dataset
    if target == args.dataset:
        print(f"backup           : {backup_dataset(args.dataset)}")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"wrote            : {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

