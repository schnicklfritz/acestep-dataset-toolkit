"""Dataset sound-profile summarizer (headless — no Qt dependency).

Shared by the AI assistant's curation skill and the MCP server.
"""


def build_sound_profile(dataset):
    """Summarize the dataset's current *sound* — the input to the
    'curate for a target sound' skill."""
    from collections import Counter

    samples = dataset.get("samples", []) or []
    if not samples:
        return "(dataset empty — nothing to profile yet)"
    genres = Counter()
    keys = Counter()
    instruments = Counter()
    bpm_lo = bpm_hi = None
    vocals = instrumentals = captioned = 0
    for s in samples:
        g = (s.get("genre") or "").strip()
        if g:
            genres[g] += 1
        b = s.get("bpm")
        if b:
            bpm_lo = b if bpm_lo is None else min(bpm_lo, int(b))
            bpm_hi = b if bpm_hi is None else max(bpm_hi, int(b))
        k = (s.get("keyscale") or "").strip()
        if k:
            keys[k] += 1
        inst = s.get("tags", {}).get("instruments") or s.get("detected_instruments") or []
        if isinstance(inst, str):
            inst = [i.strip() for i in inst.split(",") if i.strip()]
        for i in inst:
            instruments[i.strip()] += 1
        if s.get("is_instrumental"):
            instrumentals += 1
        else:
            vocals += 1
        if (s.get("caption") or "").strip():
            captioned += 1
    total = len(samples)
    lines = [
        f"Tracks: {total}",
        f"Vocal: {vocals} | Instrumental: {instrumentals} | Captioned: {captioned}/{total}",
    ]
    if genres:
        lines.append("Genres: " + ", ".join(f"{g} ({c})" for g, c in genres.most_common(6)))
    if bpm_lo:
        lines.append(f"BPM range: {bpm_lo}-{bpm_hi}")
    if keys:
        lines.append("Keys: " + ", ".join(f"{k} ({c})" for k, c in keys.most_common(5)))
    if instruments:
        lines.append("Instruments: " + ", ".join(f"{i} ({c})" for i, c in instruments.most_common(8)))
    return "\n".join(lines)