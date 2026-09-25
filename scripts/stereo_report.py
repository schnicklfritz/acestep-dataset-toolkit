#!/usr/bin/env python3
"""Report mono vs stereo per dataset folder, using the tagger's detector.

    python scripts/stereo_report.py /home/fritz/42 [--csv out.csv]

Each immediate subfolder of ROOT is treated as one dataset (audio directly in
ROOT is reported as "."). For every audio file it prints mono/stereo and the
side/mid ratio in dB, then a per-dataset summary. A dataset that contains BOTH
is flagged MIXED, and files within 10 dB of the threshold are flagged
BORDERLINE -- e.g. a needle-drop of a mono record. Exit status is 1 if any
dataset is MIXED, so it can gate a script.

Read-only: nothing is written except the optional CSV.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa  # noqa: E402

from modules.tagger import MAX_ANALYSIS_SEC, MONO_SIDE_MID_DB, detect_stereo  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aif", ".aiff"}
BORDER_DB = 10.0


def _load(path):
    # Skip a possible silent/fade-in intro, but only when the file is long
    # enough: seeking past the end raises instead of returning nothing.
    total = librosa.get_duration(path=path)
    offset = 30.0 if total > 30.0 + MAX_ANALYSIS_SEC / 3 else 0.0
    y, _ = librosa.load(path, sr=None, mono=False, offset=offset,
                        duration=MAX_ANALYSIS_SEC)
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--csv", help="also write per-file rows to this CSV")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rows, by_ds = [], defaultdict(lambda: {"mono": 0, "stereo": 0, "error": 0})
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        rel = os.path.relpath(dirpath, root)
        dataset = "." if rel == "." else rel.split(os.sep)[0]
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in AUDIO_EXT:
                continue
            path = os.path.join(dirpath, name)
            try:
                y = _load(path)
                label, db = detect_stereo(y)
                ch = 1 if y.ndim == 1 else y.shape[0]
            except Exception as e:  # noqa: BLE001 -- report and keep going
                label, db, ch = "error", None, 0
                print(f"  ERROR {os.path.relpath(path, root)}: {e}", file=sys.stderr)
            by_ds[dataset][label] += 1
            border = db is not None and db != float("-inf") and abs(db - MONO_SIDE_MID_DB) < BORDER_DB
            rows.append({"dataset": dataset, "file": os.path.relpath(path, root),
                         "channels": ch, "label": label,
                         "side_mid_db": "" if db is None else f"{db:.1f}",
                         "borderline": "yes" if border else ""})
            flag = "  BORDERLINE" if border else ""
            dbs = "   n/a" if db is None else f"{db:6.1f}"
            print(f"{label:6} {dbs} dB  ch={ch}  {rows[-1]['file']}{flag}")

    if not rows:
        print(f"no audio files under {root}", file=sys.stderr)
        return 2
    print("\n=== per dataset ===")
    mixed = False
    for ds in sorted(by_ds):
        c = by_ds[ds]
        is_mixed = c["mono"] and c["stereo"]
        mixed |= bool(is_mixed)
        verdict = "MIXED" if is_mixed else ("mono" if c["mono"] else "stereo" if c["stereo"] else "-")
        err = f", {c['error']} error(s)" if c["error"] else ""
        print(f"{ds:30} {verdict:7} mono={c['mono']} stereo={c['stereo']}{err}")
    nb = sum(1 for r in rows if r["borderline"])
    if nb:
        print(f"\n{nb} file(s) within {BORDER_DB:.0f} dB of the {MONO_SIDE_MID_DB:.0f} dB threshold: listen to them.")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 1 if mixed else 0


if __name__ == "__main__":
    sys.exit(main())
