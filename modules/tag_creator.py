"""Structural Tag Creator — maps a track onto the ACE-Step tag vocabulary.

Produces the two blocks of the two-layer model:
  * **Caption (global)**: genre, mood/energy, instruments, vocal style.
  * **Lyrics (time-script)**: the lyrics with ``[Section]`` markers and
    per-moment vocal delivery notes (preserving the existing lyric text).

WHERE THE VOCABULARY COMES FROM
-------------------------------
Not from this file. It is generated from ``docs/descriptor_reference.md`` —
the project's authoritative annotation vocabulary — by
``scripts/build_lexicon.py``, and loaded at prompt time:

    docs/vocabulary.json      artist-scoped facets (preferred)
    docs/vocabulary.txt       flat term list (fallback)
    docs/section_markers.txt  [Marker] names for the lyrics block

WHY ARTIST SCOPING MATTERS
--------------------------
A choice set is only useful if it is narrow AND correct. One hardcoded generic
list used to serve every track; measured against the authoritative doc it shared
just 14 of its 34 terms and offered "mumble rap" / "trap flow" for a Black
Sabbath record while omitting "downtuned guitar". Now a Sabbath track is given
Sabbath's own facets plus the cross-artist fundamentals, and is never offered
steel guitar or Vox Continental organ.

An empty or unmatched artist falls back to the cross-artist fundamentals only —
never to some other artist's signature terms.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, "docs")
GROUPS_PATH = os.path.join(DOCS_DIR, "vocabulary.json")
FLAT_PATH = os.path.join(DOCS_DIR, "vocabulary.txt")
MARKERS_PATH = os.path.join(DOCS_DIR, "section_markers.txt")


def _format_facets(facets):
    return "\n".join(f"{facet}: {', '.join(terms)}"
                     for facet, terms in facets.items() if terms)


def load_vocabulary(artist=""):
    """Return ``(vocabulary_text, resolved_artist_name)`` for the prompt.

    Prefers the artist-scoped groups; falls back to the flat list if the JSON is
    absent. Raises FileNotFoundError with the fix, rather than silently sending
    the model an empty vocabulary (which would make it invent descriptors).
    """
    from modules.research_tools import (
        load_vocabulary_groups, load_vocabulary as load_flat, terms_for_artist,
    )

    if os.path.isfile(GROUPS_PATH):
        groups = load_vocabulary_groups(GROUPS_PATH)
        artist_facets, general_facets, resolved = terms_for_artist(groups, artist)
        parts = []
        if artist_facets:
            parts.append(f"ARTIST VOCABULARY — {resolved} "
                         "(prefer these):\n" + _format_facets(artist_facets))
        parts.append("CROSS-ARTIST FUNDAMENTALS (use freely):\n"
                     + _format_facets(general_facets))
        return "\n\n".join(parts), resolved

    if os.path.isfile(FLAT_PATH):
        terms = load_flat(FLAT_PATH)
        return ("VOCABULARY (one list for all artists):\n"
                + ", ".join(terms)), None

    raise FileNotFoundError(
        "No vocabulary found. Generate it with:\n"
        "    .venv/bin/python scripts/build_lexicon.py"
    )


def load_markers():
    """Section markers for the lyrics block, or a small built-in fallback."""
    if os.path.isfile(MARKERS_PATH):
        with open(MARKERS_PATH, encoding="utf-8") as f:
            return [ln.strip() for ln in f
                    if ln.strip() and not ln.startswith("#")]
    return ["[Intro]", "[Verse]", "[Chorus]", "[Bridge]", "[Outro]"]


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


def tag_creator_messages(sample, artist=""):
    """Messages that produce the Caption + Lyrics blocks.

    ``artist`` is an artist name or a loose dataset name (``"sabbath"``), and
    scopes the offered vocabulary. Empty means cross-artist fundamentals only.
    """
    vocabulary, _resolved = load_vocabulary(artist)
    markers = " ".join(load_markers())
    sys_prompt = (
        "You are ACE-Step's Structural Tag Creator. Choose descriptors ONLY "
        "from the vocabulary below, preferring the artist's own terms, and "
        "produce two blocks for this track.\n\n"
        + vocabulary + "\n\n"
        "SECTION MARKERS (for the lyrics block):\n" + markers + "\n\n"
        "Rules:\n"
        "- Choose tags ONLY from the vocabulary above.\n"
        "- Keep the existing lyrics text verbatim; only add [Section] markers and "
        "(vocal delivery) notes to it.\n"
        "- Map the detected sections onto the section markers above.\n"
        "- Max 3 descriptors per bracket. More risks the model singing tag names.\n"
        "- NEVER put BPM, key, or time signature in the caption or section tags; "
        "those belong in dedicated metadata fields, not in the text.\n"
        "- Caption shape: 5-12 comma-separated keywords, then 2-3 sentences "
        "describing how the energy moves through the track.\n"
        "- Output exactly two blocks:\n"
        "CAPTION:\n<comma-separated keywords. Then 2-3 sentences.>\n\n"
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