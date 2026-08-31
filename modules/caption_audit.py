"""Audit caption naming consistency across the dataset.

Helps the "consistent instrument naming" requirement for LoRA/LoKR training —
finds tracks with missing captions, per-track instrument coverage, and
inconsistent casing/spelling of the same instrument across captions.
"""
import re

COMMON_INSTRUMENTS = [
    "vocals", "lead vocal", "backing vocals", "drums", "drum kit", "bass",
    "electric guitar", "lead guitar", "rhythm guitar", "acoustic guitar",
    "organ", "piano", "keys", "keyboard", "rhodes", "synth", "hammond",
    "fiddle", "steel guitar", "harmonica", "saxophone", "trumpet", "trombone",
    "percussion", "congas", "tambourine", "cello", "violin", "harp",
    "mandolin", "banjo", "flute", "clarinet",
]


def audit_captions(dataset):
    """Return a list of report lines about the dataset's captions."""
    reports = []
    samples = dataset.get("samples", []) or []
    if not samples:
        return ["No samples in dataset."]

    for i, s in enumerate(samples, start=1):
        cap = (s.get("caption") or "").strip()
        name = s.get("filename", f"Track {i}")
        if not cap:
            reports.append(f"  {i}. {name} — NO CAPTION")
            continue
        low = cap.lower()
        found = [inst for inst in COMMON_INSTRUMENTS if inst in low]
        reports.append(
            f"  {i}. {name} — instruments: "
            + (", ".join(found) if found else "(none found)")
        )

    # Casing/spelling consistency of known instruments across all captions
    casing = {}
    for s in samples:
        cap = s.get("caption") or ""
        for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9' -]{2,}\b", cap):
            tok = m.group(0)
            key = tok.lower()
            if key in COMMON_INSTRUMENTS:
                casing.setdefault(key, set()).add(tok)
    inconsistent = {k: sorted(v) for k, v in casing.items() if len(v) > 1}
    if inconsistent:
        reports.append("Inconsistent instrument naming across captions:")
        for key, variants in inconsistent.items():
            reports.append(f"  '{key}' appears as: {', '.join(variants)}")
    else:
        reports.append("Instrument naming is consistent across captions.")
    return reports
