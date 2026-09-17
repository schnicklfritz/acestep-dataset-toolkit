"""Lyrics normalization profiles — named, per-band rules.

THREE LAYERS
------------
1. **Master** (always on) — the shipped defaults in ``lyrics_normalizer`` plus
   any global edits. Always applied; cannot be switched off.
2. **Profile** (per band / per dataset) — a named set of rules stored as its own
   small JSON file in ``lyrics_profiles/``. Layered on top of master, so a
   profile can add words or override what master says for a specific band.
3. **Per-track** — handled elsewhere (``prompt_override``).

Each profile is a plain JSON file, which makes it trivially shareable and
diffable::

    lyrics_profiles/
        black_sabbath.json
        my_band.json

STORAGE
-------
The folder lives beside the app and is **gitignored** — profiles are local user
data (like ``settings.json``), not source. A profile is small text, so copying a
file is all it takes to move rules to another machine.
"""
import json
import os
import re

PROFILE_DIRNAME = "lyrics_profiles"


def profile_dir(base_dir=None):
    """Absolute path to the profiles folder (created on demand by callers)."""
    root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, PROFILE_DIRNAME)


def _safe_name(name):
    """Filename-safe slug for a profile name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return slug or "profile"


def list_profiles(base_dir=None):
    """Return the available profile names (without the .json extension)."""
    folder = profile_dir(base_dir)
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".json"):
            out.append(fn[:-5])
    return out


def load_profile(name, base_dir=None):
    """Load one profile. Returns ``None`` when it does not exist."""
    if not name:
        return None
    path = os.path.join(profile_dir(base_dir), f"{_safe_name(name)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "name": data.get("name") or name,
        "contractions": data.get("contractions") or {},
        "ing_exceptions": data.get("ing_exceptions") or [],
        "ing_to_in": bool(data.get("ing_to_in", False)),
        "capitalize_tags": bool(data.get("capitalize_tags", True)),
        "strip_punctuation": bool(data.get("strip_punctuation", True)),
        "notes": data.get("notes", ""),
    }


def save_profile(name, contractions, ing_exceptions, ing_to_in=False,
                 capitalize_tags=True, strip_punctuation=True, notes="",
                 base_dir=None):
    """Write a profile file. Returns the path written."""
    folder = profile_dir(base_dir)
    os.makedirs(folder, exist_ok=True)
    payload = {
        "name": name,
        "contractions": dict(sorted((contractions or {}).items())),
        "ing_exceptions": sorted(ing_exceptions or []),
        "ing_to_in": bool(ing_to_in),
        "capitalize_tags": bool(capitalize_tags),
        "strip_punctuation": bool(strip_punctuation),
        "notes": notes,
    }
    path = os.path.join(folder, f"{_safe_name(name)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def delete_profile(name, base_dir=None):
    """Delete a profile file. Returns True when something was removed."""
    path = os.path.join(profile_dir(base_dir), f"{_safe_name(name)}.json")
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except OSError:
            return False
    return False


def merge_rules(master_contractions, profile_contractions,
                master_ing=None, profile_ing=None):
    """Layer a profile over master rules.

    Merge semantics: the profile **overrides** master for any word it defines
    and **adds** its own words; master entries the profile does not mention
    still apply. That matches "master is always on, profiles specialise it".

    Returns ``(contractions, ing_exceptions)``.
    """
    merged = dict(master_contractions or {})
    merged.update(profile_contractions or {})
    ing = set(master_ing or ())
    ing |= set(profile_ing or ())
    return merged, ing
