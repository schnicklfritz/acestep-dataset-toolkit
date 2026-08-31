"""Local, user-extensible instrument database.

Seeded from Rock Band / Guitar Hero MOGG channel metadata (the channel lists
published on isolated-tracks.com) for classic artists. Users can add their own
entries in ``instruments_db.json`` (project root); keys are matched against the
track filename (lowercase). This gives the "data lookup of instruments" path so
the app never depends on the captioner alone.
"""
import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "instruments_db.json"

# Seeded defaults — extend via instruments_db.json ({"song keyword": [instruments]})
DEFAULT_DB = {
    "iron man": ["drum kit", "bass", "electric guitar 1", "electric guitar 2", "lead vocal"],
    "war pigs": ["drum kit", "tambourine", "bass", "electric guitar left", "electric guitar right", "lead electric guitar", "siren", "lead vocal"],
    "paranoid": ["drum kit", "bass", "rhythm electric guitar left", "rhythm electric guitar right", "lead electric guitar left", "lead electric guitar right", "lead vocal"],
    "fairies wear boots": ["drum kit", "bass", "electric guitar left", "electric guitar right", "lead vocal"],
    "the wizard": ["drum kit", "bass", "electric guitar 1", "electric guitar 2", "harmonica", "lead vocal"],
    "heaven and hell": ["drum kit", "bass", "acoustic guitar", "electric guitar", "lead electric guitar", "backing vocals", "lead vocal"],
    "highway to hell": ["drum kit", "bass", "electric guitar theme left", "electric guitar theme right", "rhythm electric guitar", "lead electric guitar", "backing vocals", "lead vocal"],
    "thunderstruck": ["drum kit", "bass", "rhythm electric guitar left", "rhythm electric guitar right", "lead electric guitar", "distorted electric guitar", "backing vocals", "lead vocal"],
    "tnt": ["drum kit", "bass", "rhythm electric guitar left", "rhythm electric guitar right", "lead electric guitar", "backing vocals", "lead vocal"],
    "t.n.t.": ["drum kit", "bass", "rhythm electric guitar left", "rhythm electric guitar right", "lead electric guitar", "backing vocals", "lead vocal"],
    "hells bells": ["electronic drum kit", "bass", "electric guitar left", "electric guitar right", "lead electric guitar", "bells", "backing vocals", "lead vocal"],
    "break on through": ["drums", "bass keyboard", "organ", "electric guitar", "lead vocal"],
    "light my fire": ["drums", "bass keyboard", "organ", "electric guitar", "lead vocal"],
    "riders on the storm": ["drums", "bass keyboard", "organ", "electric guitar", "lead vocal"],
    "purple haze": ["drums", "bass", "lead electric guitar", "lead vocal"],
    "little wing": ["drums", "bass", "lead electric guitar", "lead vocal"],
    "foxey lady": ["drums", "bass", "lead electric guitar", "lead vocal"],
    "voodoo child": ["drums", "bass", "lead electric guitar", "lead vocal"],
}


def load_db():
    """Return the merged instrument database (defaults + user instruments_db.json)."""
    db = dict(DEFAULT_DB)
    try:
        if DB_PATH.exists():
            extra = json.loads(DB_PATH.read_text(encoding="utf-8"))
            db.update({str(k).lower(): list(v) for k, v in extra.items()})
    except Exception:  # noqa: BLE001
        pass
    return db


def lookup_instruments(filename, artist_hint=""):
    """Find instruments for a track by keyword-matching filename (+ optional artist)."""
    name = f"{filename} {artist_hint}".lower().replace("_", " ")
    for key, instruments in load_db().items():
        if key in name:
            return list(instruments)
    return []
