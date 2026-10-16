"""Re-point dataset entries at audio files whose name or format changed.

THE PROBLEM
-----------
``audio_path`` is stored absolute and verbatim from the file picker
(``dataset_manager._add_audio_paths`` stores the picker's path unchanged and
dedupes on that exact string). Playback then hard-requires it:
``_load_track_preview`` does ``os.path.exists(audio_path)`` and gives up silently
otherwise, so a track whose file was renamed or re-encoded **outside** the app
shows an empty waveform and will not play. The Inspector's ``Filename`` cell is
read-only, so the entry cannot be repaired by typing either.

That is exactly what happened to the owner's Doors dataset: every ``.wav`` became
``.flac`` on disk, and every entry kept pointing at ``.wav``.

WHY THIS IS NOT ``core.file_system.execute_disk_rename``
-------------------------------------------------------
That function MOVES files on disk (and backs them up). Here the files are
*already* renamed; running it would try to rename ``.wav`` sources that no longer
exist. Relinking is a different operation with a different risk profile, so it
lives in its own module and **never touches the filesystem**. Everything below is
pure logic over paths plus an in-memory index, so a caller can build a plan, show
it to a human, and only then apply it.

WHAT MUST CHANGE TOGETHER
-------------------------
Both ``filename`` and ``audio_path``. ACE-Step indexes per-sample metadata by
``filename`` and only falls back to ``basename(audio_path)`` (see the file:line
evidence in ``modules/dataset_schema.py``), so updating one alone points the
trainer's metadata at the wrong name.
"""
import os

# Mirrors the app's canonical set (``dataset_manager.AUDIO_EXTS_ORDERED``) so this
# module is usable - and testable - without importing the 7,000-line manager,
# which would also be a circular import (the manager imports this). Callers that
# want the two in lockstep pass their own set explicitly.
DEFAULT_AUDIO_EXTS = (
    ".wav", ".flac", ".aiff", ".aif", ".aifc", ".alac", ".ape", ".wv", ".tta", ".au",
    ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".ac3", ".amr", ".mp2",
    ".webm", ".mkv", ".mp4", ".caf", ".rf64", ".bwf",
)

# Confidence, best first. Carried into the plan table so a wrong guess is visible
# BEFORE it is applied rather than after.
EXACT = "exact"    # same stem, different extension, SAME folder - the .wav->.flac case
STEM = "stem"      # same stem, found elsewhere in the scanned tree
FUZZY = "fuzzy"    # stem matches once case/punctuation/spacing are ignored
_CONFIDENCE_ORDER = {EXACT: 0, STEM: 1, FUZZY: 2}


def _norm_stem(name):
    """Case/punctuation/space-insensitive stem, for MATCHING only.

    Never written back anywhere: the file on disk is the truth. This is only how
    two names are decided to be "the same song" when the text differs slightly.
    """
    stem = os.path.splitext(os.path.basename(name or ""))[0].lower()
    return "".join(ch for ch in stem if ch.isalnum())


def resolved_path(sample, base_dir=""):
    """The path a sample points at, or ``""`` when it names no file at all.

    A relative ``audio_path`` resolves against the dataset JSON's own folder,
    which is what ACE-Step does too (``dataset_schema.py`` documents it). An
    empty ``audio_path`` falls back to ``filename``, matching the trainer.
    """
    raw = (sample.get("audio_path") or "").strip()
    if not raw:
        raw = (sample.get("filename") or "").strip()
    if not raw:
        return ""
    if base_dir and not os.path.isabs(raw):
        return os.path.join(base_dir, raw)
    return raw


def index_folder(folder, exts=DEFAULT_AUDIO_EXTS, recursive=True):
    """``{normalised stem: [paths]}`` for the audio files under ``folder``.

    Built ONCE per rescan and reused for every track: matching stems by listing
    the folder per track would be O(tracks x files).
    """
    index = {}
    if not folder or not os.path.isdir(folder):
        return index
    if recursive:
        for root, _dirs, files in os.walk(folder):
            _index_into(index, root, files, exts)
    else:
        try:
            names = os.listdir(folder)
        except OSError:
            return index
        _index_into(index, folder, names, exts)
    return index


def _index_into(index, root, names, exts):
    wanted = {str(e).lower() for e in exts}
    for name in names:
        if os.path.splitext(name)[1].lower() not in wanted:
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path):
            index.setdefault(_norm_stem(name), []).append(path)


def find_missing(samples, base_dir=""):
    """``[(index, sample, path)]`` for every entry whose file is not on disk.

    ``virtual`` samples are skipped by design: a concept-only placeholder has no
    audio on disk yet (``dataset_schema.py`` documents it), so reporting one as
    "missing" would be pure noise. A sample that names no file at all is skipped
    too - there is nothing to relink it to.
    """
    missing = []
    for i, s in enumerate(samples or []):
        if s.get("virtual"):
            continue
        path = resolved_path(s, base_dir)
        if not path:
            continue
        if not os.path.exists(path):
            missing.append((i, s, path))
    return missing


def match_candidate(sample, index, base_dir="", old_path=None, allow_fuzzy=True):
    """Best replacement for a missing sample: ``(path, confidence)`` or ``None``.

    ``index`` comes from :func:`index_folder`. Tried in confidence order:

    * ``EXACT`` - same stem in the SAME folder. The format-conversion case, and
      the only one that is safe to apply in bulk without looking.
    * ``STEM`` - same stem found elsewhere under the scanned folder.
    * ``FUZZY`` - stem containment, offered ONLY when exactly one candidate
      matches. A near-miss stem that matches several files is more likely to be a
      different song than the same one, so it is suggested to nobody rather than
      guessed at.
    """
    old = old_path or resolved_path(sample, base_dir)
    old_name = os.path.basename(old) or (sample.get("filename") or "")
    key = _norm_stem(old_name)
    if not key:
        return None

    old_dir = os.path.abspath(os.path.dirname(old)) if old else ""
    candidates = sorted(p for p in index.get(key, []) if os.path.exists(p))
    if candidates:
        if old_dir:
            for path in candidates:
                if os.path.abspath(os.path.dirname(path)) == old_dir:
                    return (path, EXACT)
        return (candidates[0], STEM)

    if not allow_fuzzy:
        return None

    loose = []
    for other, paths in index.items():
        if other == key or len(key) < 4 or len(other) < 4:
            continue
        if key in other or other in key:
            loose.extend(p for p in paths if os.path.exists(p))
    if len(loose) == 1:
        return (loose[0], FUZZY)
    return None


def plan_relink(samples, index, base_dir="", allow_fuzzy=True):
    """``[(i, sample, old_path, new_path, confidence)]`` for the matched entries.

    Builds a plan and writes NOTHING. A caller can render this, let a human untick
    rows, and only then call :func:`apply_relink` - which is the whole point of
    keeping this module free of filesystem writes.
    """
    plan = []
    for i, sample, old in find_missing(samples, base_dir):
        match = match_candidate(sample, index, base_dir, old_path=old,
                                allow_fuzzy=allow_fuzzy)
        if match:
            new, confidence = match
            plan.append((i, sample, old, new, confidence))
    plan.sort(key=lambda row: (_CONFIDENCE_ORDER.get(row[4], 9), row[0]))
    return plan


def apply_relink(sample, new_path, base_dir=""):
    """Point ``sample`` at ``new_path``. Returns the two values that changed.

    Both keys move together - see "WHAT MUST CHANGE TOGETHER" above.

    The relative/absolute STYLE of the original is preserved: a dataset that
    stored relative paths keeps storing relative ones. Without this a relink would
    silently bake this machine's absolute paths into an otherwise portable
    dataset, which is a change nobody asked for.
    """
    old_raw = (sample.get("audio_path") or "").strip()
    wants_relative = bool(old_raw) and not os.path.isabs(old_raw)
    if wants_relative and base_dir:
        stored = os.path.relpath(new_path, base_dir)
    else:
        stored = new_path
    sample["audio_path"] = stored
    sample["filename"] = os.path.basename(new_path)
    return {"audio_path": stored, "filename": sample["filename"]}


def apply_plan(plan, base_dir=""):
    """Apply every row of a :func:`plan_relink` result. Returns the rows applied.

    Kept here rather than in the dialog so the "apply" step is testable without a
    single Qt import.
    """
    applied = []
    for _i, sample, _old, new, confidence in plan:
        apply_relink(sample, new, base_dir)
        applied.append((sample, new, confidence))
    return applied
