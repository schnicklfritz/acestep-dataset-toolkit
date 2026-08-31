"""Dataset statistics report (one-page summary)."""
from collections import Counter


def build_dataset_report(dataset):
    """Return a human-readable dataset summary."""
    samples = dataset.get("samples", []) or []
    genres = Counter()
    keys = Counter()
    instruments = Counter()
    bpms = []
    durs = []
    caption_words = []
    vocals = instrumentals = captioned = 0

    for s in samples:
        g = (s.get("genre") or "").strip()
        if g:
            genres[g] += 1
        k = (s.get("keyscale") or "").strip()
        if k:
            keys[k] += 1
        b = s.get("bpm")
        if b:
            bpms.append(int(b))
        d = s.get("duration")
        if d:
            durs.append(float(d))
        cap = (s.get("caption") or "").strip()
        if cap:
            captioned += 1
            caption_words.append(len(cap.split()))
        inst = s.get("tags", {}).get("instruments") or s.get("detected_instruments") or []
        if isinstance(inst, str):
            inst = [i.strip() for i in inst.split(",") if i.strip()]
        for i in inst:
            if i:
                instruments[i.strip()] += 1
        if s.get("is_instrumental"):
            instrumentals += 1
        else:
            vocals += 1

    n = len(samples)
    if not n:
        return "(dataset empty)"
    lines = [
        f"Tracks: {n} | Vocals: {vocals} | Instrumental: {instrumentals} | "
        f"Captioned: {captioned}/{n}",
    ]
    if genres:
        lines.append("Genres: " + ", ".join(f"{g} ({c})" for g, c in genres.most_common(8)))
    if bpms:
        lines.append(f"BPM: {min(bpms)}-{max(bpms)} (mean {int(sum(bpms) / len(bpms))})")
    if durs:
        lines.append(f"Duration: {min(durs):.0f}s-{max(durs):.0f}s (mean {sum(durs) / len(durs):.0f}s)")
    if keys:
        lines.append("Keys: " + ", ".join(f"{k} ({c})" for k, c in keys.most_common(6)))
    if instruments:
        lines.append("Instruments: " + ", ".join(f"{i} ({c})" for i, c in instruments.most_common(8)))
    if caption_words:
        lines.append(
            f"Caption length: {min(caption_words)}-{max(caption_words)} words "
            f"(mean {sum(caption_words) / len(caption_words):.0f})"
        )
    return "\n".join(lines)