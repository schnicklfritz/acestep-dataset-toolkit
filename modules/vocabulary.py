"""Vocabulary loader — the artist-scoped *invocation* vocabulary.

WHAT THIS IS
------------
The vocabulary is the invocation interface: descriptors you paste into a
generation prompt to steer ACE-Step toward the sound the trained weights
already encode. It belongs to the prompt, not to the dataset.

WHAT THIS IS NOT
----------------
This is not a validation list. It does not gate training annotations and it
must never be used to decide what belongs in a caption bracket or a lyric
bracket. A trained LoKR records what *varies* between its tracks, so a term
that appears in every track is dead weight there no matter how often it
appears here — see ``docs/Comprehensive_LoKR_Dataset_Annotation_Guide.txt``
§1 (the Omission Principle).

File formats
------------
    vocabulary.txt      one term per line (flat fallback)
    vocabulary.json     artist-scoped facets (preferred)

Both are hand-maintained now. The corpus-mining pipeline that used to
generate them from a prose reference doc has been retired.
"""
import json
import os
import re

GENERAL_GROUP = "general"


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def load_vocabulary(path):
    """Read a one-term-per-line vocabulary file.

    Blank lines and ``#`` comments are ignored; duplicates are removed
    case-insensitively while keeping the first spelling seen.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"vocabulary file not found: {path}")
    terms = []
    seen = set()
    for raw in _read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(line)
    return terms


def resolve_artist(groups, artist):
    """Match a loose dataset/artist name onto a vocabulary artist, or ``None``.

    Word-based rather than string-prefix based, because real dataset folders are
    not named after the artist: ``Doorsdata`` -> "The Doors", ``hank_sr`` ->
    "Hank Williams Sr.", ``sabbath`` -> "Black Sabbath".
    """
    if not artist:
        return None
    key = re.sub(r"[^a-z0-9]", "", artist.lower())
    if not key:
        return None
    fallback = None
    for name in groups.get("artists", {}):
        if re.sub(r"[^a-z0-9]", "", name.lower()) == key:
            return name                      # exact after normalisation
        # Significant words only; "the"/"sr" carry no identity.
        for word in re.findall(r"[a-z0-9]+", name.lower()):
            if len(word) >= 4 and word in key:
                fallback = fallback or name
    return fallback


def terms_for_artist(groups, artist):
    """Return ``(artist_facets, general_facets, resolved_name)``.

    The two buckets stay separate because they mean different things: the
    artist's own facets are the narrow, preferred choice set, while the general
    facets are cross-artist fundamentals (vocal timbre, energy, production) that
    apply to any track. Keeping them apart lets a prompt say "prefer these"
    without hiding the fundamentals.

    An empty or unmatched artist yields an empty artist bucket and the general
    vocabulary only -- the safe fallback, so one artist's signature terms are
    never offered for another.
    """
    selected = resolve_artist(groups, artist)
    artists = groups.get("artists", {})
    return (
        dict(artists.get(selected, {})) if selected else {},
        dict(groups.get(GENERAL_GROUP, {})),
        selected,
    )


def load_vocabulary_groups(path):
    """Read the artist-grouped vocabulary JSON."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"grouped vocabulary not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_source", None)
    data.pop("_note", None)
    return data
