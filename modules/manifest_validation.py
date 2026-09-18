"""Validate the ACE-Step manifest before export.

Checks the schema ACE-Step 1.5XL training expects (per-track required fields
and sane values) and returns a list of human-readable issues.

The field lists live in ``modules/dataset_schema.py`` so the schema has a single
source of truth shared with the app's own dataset construction. Caption TEXT is
checked against the annotation standard in ``modules/caption_quality.py``.
"""
from modules.caption_quality import check_caption
from modules.dataset_schema import (
    REQUIRED_METADATA_FIELDS as REQUIRED_META_FIELDS,
    REQUIRED_SAMPLE_FIELDS,
)


def validate_manifest(dataset):
    """Return a list of issue strings (empty = valid)."""
    issues = []
    if not isinstance(dataset, dict):
        return ["Dataset is not a dict."]

    meta = dataset.get("metadata", {}) or {}
    for field in REQUIRED_META_FIELDS:
        if field not in meta:
            issues.append(f"metadata missing '{field}'")

    samples = dataset.get("samples", []) or []
    if not samples:
        issues.append("no samples in dataset")

    for i, s in enumerate(samples, start=1):
        for field in REQUIRED_SAMPLE_FIELDS:
            if field not in s:
                issues.append(f"sample {i} missing '{field}'")
        if s.get("is_instrumental") and (s.get("lyrics") or s.get("formatted_lyrics")):
            issues.append(f"sample {i} is instrumental but has lyrics")
        caption = (s.get("caption") or "").strip()
        if not caption:
            issues.append(f"sample {i} has no caption")
        else:
            # ACE-Step 1.5XL caption schema (front-loaded tags, vocal descriptor,
            # no BPM/key/time-signature, no headings, no decoding loops).
            for issue in check_caption(caption):
                issues.append(f"sample {i} caption: {issue}")
        bpm = s.get("bpm")
        if bpm and not (0 < bpm <= 300):
            issues.append(f"sample {i} BPM out of range: {bpm}")
    return issues
