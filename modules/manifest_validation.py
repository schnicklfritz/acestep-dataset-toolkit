"""Validate the ACE-Step manifest before export.

Checks the schema ACE-Step 1.5XL training expects (per-track required fields
and sane values) and returns a list of human-readable issues.
"""
REQUIRED_META_FIELDS = ["name", "custom_tag", "tag_position", "instrumental_mode", "num_samples"]

REQUIRED_SAMPLE_FIELDS = [
    "id", "audio_path", "filename", "caption", "genre", "lyrics",
    "formatted_lyrics", "bpm", "keyscale", "timesignature", "duration",
    "language", "is_instrumental", "custom_tag",
]


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
        if not (s.get("caption") or "").strip():
            issues.append(f"sample {i} has no caption")
        bpm = s.get("bpm")
        if bpm and not (0 < bpm <= 300):
            issues.append(f"sample {i} BPM out of range: {bpm}")
    return issues
