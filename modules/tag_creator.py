"""Structural Tag Creator — maps a track onto the ACE-Step tag vocabulary.

Produces the two blocks of the two-layer model:
  * **Caption (global)**: genre, mood/energy, instruments, vocal style.
  * **Lyrics (time-script)**: the lyrics with ``[Section]`` markers and
    per-moment vocal delivery notes (preserving the existing lyric text).
"""
ACE_STEP_VOCABULARY = """VOCAL TIMBRE: bright, dark, warm, cold, breathy, nasal, gritty, smooth, husky,
metallic, whispery, resonant, airy, smoky, sultry, light, clear, high-pitched,
raspy, powerful, ethereal, flute-like, hollow, velvety, shrill, hoarse, mellow,
thin, thick, reedy, silvery, twangy
VOCAL DELIVERY: whispered, shouted, spoken word, narration, singing, falsetto,
powerful belting, harmonies, call and response, ad-libs
RAP STYLES: mumble rap, chopper rap, melodic rap, lyrical rap, trap flow, double-time rap
VOCAL FX: auto-tune, reverb, delay, distortion
ENERGY/MOOD: high energy, low energy, building energy, explosive, melancholic,
euphoric, dreamy, aggressive
GLOBAL ENERGY: high (energetic, aggressive, intense, powerful, explosive,
stadium-sized, driving), moderate (mid-tempo, moderate energy, relaxed groove,
laid-back, steady beat), low (calm, intimate, restrained, minimalist, quiet
tension, subtle, lo-fi), emotional (uplifting, euphoric, melancholic, passionate,
triumphant, defiant)
STRUCTURE: [Intro] [Verse] [Verse 1] [Pre-Chorus] [Chorus] [Bridge] [Outro]
[Build] [Drop] [Breakdown] [Instrumental] [Guitar Solo] [Piano Interlude]
[Fade Out] [Silence]"""


def build_track_context(sample):
    lines = [f"Track: {sample.get('filename', '?')}"]
    if sample.get("genre"):
        lines.append(f"Genre: {sample['genre']}")
    if sample.get("bpm"):
        lines.append(f"BPM: {sample['bpm']}")
    if sample.get("keyscale"):
        lines.append(f"Key: {sample['keyscale']}")
    inst = sample.get("tags", {}).get("instruments") or sample.get("detected_instruments") or []
    if inst:
        lines.append("Instruments: " + ", ".join(inst))
    if sample.get("caption"):
        lines.append("Existing caption: " + str(sample["caption"])[:300])
    segs = sample.get("structural_segments") or []
    if segs:
        parts = [f"{seg.get('name', '?')} {seg.get('start', 0)}-{seg.get('end', 0)}s" for seg in segs]
        lines.append("Sections: " + ", ".join(parts))
    if sample.get("hooks"):
        lines.append("Hooks/riffs to emphasize: " + ", ".join(sample["hooks"]))
    if sample.get("riff_note"):
        lines.append("Riff note: " + sample["riff_note"])
    ly = (sample.get("lyrics") or sample.get("formatted_lyrics") or "").strip()
    if ly:
        lines.append("LYRICS:\n" + ly[:2000])
    return "\n".join(lines)


def tag_creator_messages(sample):
    """Messages that produce the Caption + Lyrics blocks."""
    sys_prompt = (
        "You are ACE-Step's Structural Tag Creator. Using ONLY the ACE-Step "
        "vocabulary below, produce two blocks for this track.\n\n"
        + ACE_STEP_VOCABULARY + "\n\n"
        "Rules:\n"
        "- Choose tags ONLY from the vocabulary (timbre, delivery, energy, structure).\n"
        "- Keep the existing lyrics text verbatim; only add [Section] markers and "
        "(vocal delivery) notes to it.\n"
        "- Map the detected sections onto structure tags ([Verse], [Chorus], [Guitar Solo]...).\n"
        "- Output exactly two blocks:\n"
        "CAPTION:\n<comma-separated: genre, mood/energy tier, instruments, vocal style>\n\n"
        "LYRICS:\n<the structured lyrics with [Section] markers and (vocal delivery) notes>"
    )
    return [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": build_track_context(sample)}]


def parse_output(text):
    caption = ""
    lyrics = ""
    if "CAPTION:" in text:
        caption = text.split("CAPTION:", 1)[1]
        if "LYRICS:" in caption:
            caption = caption.split("LYRICS:", 1)[0]
    if "LYRICS:" in text:
        lyrics = text.split("LYRICS:", 1)[1]
    return caption.strip(), lyrics.strip()