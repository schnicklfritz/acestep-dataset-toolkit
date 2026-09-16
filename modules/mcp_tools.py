"""Headless tools for the MCP server (no Qt dependency).

Operate on a dataset JSON file (the app's save format) plus the app's own
headless modules (tagger, sound_profile).
"""
import json


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tool_list_tracks(dataset_path):
    data = load_dataset(dataset_path)
    samples = data.get("samples", []) or []
    if not samples:
        return "(no tracks)"
    lines = []
    for i, s in enumerate(samples, start=1):
        cap = (s.get("caption") or "").strip().replace("\n", " ")[:80]
        lines.append(f"{i}. {s.get('filename', '?')} — {cap or '(no caption)'}")
    return "\n".join(lines)


def tool_dataset_summary(dataset_path):
    data = load_dataset(dataset_path)
    samples = data.get("samples", []) or []
    captioned = sum(1 for s in samples if (s.get("caption") or "").strip())
    vocal = sum(1 for s in samples if not s.get("is_instrumental"))
    return (
        f"Tracks: {len(samples)} | Vocals: {vocal} | Instrumental: {len(samples) - vocal} | "
        f"Captioned: {captioned}/{len(samples)}"
    )


def tool_tag_track(audio_path):
    from modules.tagger import analyze_audio

    tags = analyze_audio(audio_path)
    return (
        f"BPM: {tags.get('bpm')} | Key: {tags.get('key') or '?'} | "
        f"Instruments: {', '.join(tags.get('instruments') or []) or '?'}"
    )


def tool_curate(dataset_path, target_sound):
    from modules.sound_profile import build_sound_profile

    data = load_dataset(dataset_path)
    target = (target_sound or "").strip()
    if not target:
        return "Provide a target_sound (artist/genre/mood)."
    return (
        f"TARGET SOUND: {target}\n\n"
        f"CURRENT SOUND PROFILE:\n{build_sound_profile(data)}\n\n"
        "Suggest songs/artists/genres to add and which gaps to fill "
        "(instruments, tempo, key, era) so the dataset converges on the target."
    )