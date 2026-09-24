"""Local side of the ACE-Step Kaggle caption run.

WHY THIS MODULE EXISTS
----------------------
The caption worker used to stage audio into a ``tempfile.mkdtemp()`` folder that
died with the run, and to upload it as a brand-new Kaggle dataset with a random
slug on EVERY run. Both choices are invisible until the user wants the two things
the tempfile/random-slug design cannot express:

  * "add or remove songs from the uploaded dataset" — needs a dataset whose
    IDENTITY survives and whose CONTENTS the user controls;
  * "send the captions to a local folder I define" — needs a download
    destination that is not a temp dir.

So the audio that gets uploaded now lives in a PERSISTENT local staging folder
that the user owns (default ``~/acestep_kaggle_staging``), the Kaggle dataset
slug is remembered in config so later uploads push a new VERSION in place, and
the downloaded ``captions_out.json`` lands in a user-chosen folder (default
``~/acestep_captions``).

Everything here is deliberately Qt-free and pure where possible, so the staging
and the diff/merge rules can be tested without a GPU, a Kaggle account or a GUI.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# Same set the kernel scans for. Keep the two in step: the kernel lists files on
# the MOUNTED dataset, and a suffix it does not know is silently skipped.
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}

STAGING_SUBDIR = "acestep_kaggle_staging"
CAPTIONS_SUBDIR = "acestep_captions"

# Characters ffmpeg/Kaggle dataset paths handle poorly. Replaced, not stripped, so
# two different titles cannot collapse onto the same staged filename.
_UNSAFE = re.compile(r"[^A-Za-z0-9._ ()-]+")


def default_staging_dir():
    """The staging folder used when config has none."""
    return os.path.join(str(Path.home()), STAGING_SUBDIR)


def staging_dir(config):
    """The persistent folder whose CONTENTS are the uploaded Kaggle dataset."""
    configured = (config.get("caption_staging_dir") or "").strip()
    return configured or default_staging_dir()


def default_output_dir():
    """Where downloaded captions go when config has no folder."""
    return os.path.join(str(Path.home()), CAPTIONS_SUBDIR)


def output_dir(config):
    """The LOCAL folder the kernel's output is downloaded into.

    Kaggle only persists ``/kaggle/working``, so a path inside the kernel is not
    something a user can meaningfully choose — this is the knob that matters.
    """
    configured = (config.get("caption_output_dir") or "").strip()
    return configured or default_output_dir()


def staged_name(filename, convert_mp3=True):
    """The name a track gets inside the staging folder / on the Kaggle mount.

    The STEM is preserved so a result can be matched back to its track even after
    a restart, and only the extension is changed when transcoding.
    """
    name = os.path.basename(filename or "")
    stem = _UNSAFE.sub("_", os.path.splitext(name)[0]).strip() or "track"
    if convert_mp3:
        return stem + ".mp3"
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in SUPPORTED_FORMATS:
        suffix = ".mp3"
    return stem + suffix


def _is_usable(path):
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def ffmpeg_available():
    """True when ffmpeg is on PATH (the MP3 transcode needs it)."""
    return bool(shutil.which("ffmpeg"))


def convert_to_mp3(src, dst, bitrate="192k"):
    """Transcode ``src`` to MP3 at ``dst``. Returns True on success."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-vn", "-map_metadata", "-1",
             "-c:a", "libmp3lame", "-b:a", str(bitrate or "192k"), str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0 and _is_usable(dst)


def stage_track(audio_path, staging, convert_mp3=True, bitrate="192k", name=None):
    """Put one track into the staging folder. Returns the staged path, or ''.

    MP3 is attempted with ffmpeg; when ffmpeg is missing or refuses the input the
    ORIGINAL file is copied instead, so a run never silently drops a track (an
    empty staging folder is what produced a captioning run with 0 results).
    """
    if not _is_usable(audio_path):
        return ""
    os.makedirs(staging, exist_ok=True)
    target = os.path.join(staging, name or staged_name(audio_path, convert_mp3))
    if convert_mp3 and convert_to_mp3(audio_path, target, bitrate):
        return target
    # Transcode unavailable -> keep the ORIGINAL extension so the kernel's
    # SUPPORTED_FORMATS scan still picks the file up.
    fallback = os.path.join(staging, staged_name(name or audio_path, False))
    try:
        if os.path.abspath(audio_path) != os.path.abspath(fallback):
            shutil.copy2(audio_path, fallback)
        return fallback
    except OSError:
        return ""


def stage_tracks(items, staging, convert_mp3=True, bitrate="192k", progress=None):
    """Stage ``(key, filename, audio_path)`` triples.

    Returns ``(staged, skipped)`` where ``staged`` is a list of
    ``(key, filename, staged_path)``. ``skipped`` counts tracks whose audio is
    missing/unreadable — reported rather than ignored, because a run that uploads
    nothing must not look like a run that works.
    """
    staged, skipped = [], 0
    for key, filename, audio_path in items:
        target = stage_track(audio_path, staging, convert_mp3=convert_mp3,
                             bitrate=bitrate)
        if target:
            staged.append((key, filename, target))
        else:
            skipped += 1
        if progress is not None:
            progress(key, filename, bool(target))
    return staged, skipped


def staged_files(staging):
    """Audio files currently in the staging folder, sorted."""
    if not os.path.isdir(staging):
        return []
    return sorted(
        os.path.join(staging, name) for name in os.listdir(staging)
        if os.path.splitext(name)[1].lower() in SUPPORTED_FORMATS
        and _is_usable(os.path.join(staging, name))
    )


def remove_staged(staging, names):
    """Delete staged files by name. Returns the count actually removed."""
    removed = 0
    for name in names:
        path = os.path.join(staging, os.path.basename(name))
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# diff / merge — the "run a diff on an existing caption" step
# ---------------------------------------------------------------------------

STATUS_NEW = "new"          # no existing caption -> the proposed one is ADDED
STATUS_CHANGED = "changed"  # both exist and differ -> the user chooses
STATUS_SAME = "same"        # identical -> nothing to decide
STATUS_MISSING = "missing"  # the kernel returned nothing for this track
STATUS_ERROR = "error"      # the kernel reported an error for this track


def load_results(path):
    """Read a kernel ``captions_out.json``. Returns a ``{basename: caption}`` map.

    Raises ``ValueError`` with a readable message instead of letting a JSON
    traceback surface: the file is user-supplied (it can be downloaded and moved
    anywhere), so "not JSON" is an expected input, not a bug.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read captions from {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(
            f"{path} is not a captions_out.json (expected {{'results': [...]}})"
        )
    out = {}
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        name = os.path.basename(str(item.get("file", "")))
        if name:
            out[name] = str(item.get("caption", "") or "")
    return out


def _key_for(filename, convert_mp3=True):
    """Every name a result file could legitimately carry for this track."""
    base = os.path.basename(filename or "")
    stem = os.path.splitext(base)[0]
    return {
        base,
        stem,
        staged_name(base, convert_mp3),
        os.path.splitext(staged_name(base, convert_mp3))[0],
    }


def diff_captions(samples, results, convert_mp3=True):
    """Pair kernel results with dataset samples for the review table.

    ``results`` maps a downloaded filename to the caption the kernel produced.
    Returns rows of ``{id, filename, existing, proposed, status}``; matching is by
    staged name / stem (NOT by position), so a run that returned a subset, or
    returned files out of order, still pairs correctly — and a track with no
    result is reported as MISSING instead of silently keeping its old caption.
    """
    rows = []
    for sample in samples:
        filename = sample.get("filename", "")
        existing = (sample.get("caption") or "").strip()
        proposed = ""
        matched = False
        for candidate in _key_for(filename, convert_mp3):
            if candidate in results:
                proposed = (results[candidate] or "").strip()
                matched = True
                break
        if proposed.upper().startswith("ERROR:"):
            status = STATUS_ERROR
        elif not matched or not proposed:
            status = STATUS_MISSING
        elif not existing:
            status = STATUS_NEW
        elif proposed == existing:
            status = STATUS_SAME
        else:
            status = STATUS_CHANGED
        rows.append({
            "id": sample.get("id", ""),
            "filename": filename,
            "existing": existing,
            "proposed": proposed,
            "status": status,
        })
    return rows


def unmatched_results(samples, results, convert_mp3=True):
    """Result filenames that no sample claimed (usually a stale staging file)."""
    claimed = set()
    for sample in samples:
        claimed |= _key_for(sample.get("filename", ""), convert_mp3)
    return sorted(name for name in results if name not in claimed)


def count_by_status(rows):
    """``{status: n}`` for a row list, used by the tab's summary label."""
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def apply_decision(sample, row, decision, edited_text=None):
    """Write the chosen caption onto ``sample``. Returns the text written.

    ``decision`` is one of:
      * ``keep`` — leave the existing caption alone (still records the proposal
        in ``caption_ai_raw`` so the audit trail shows what was offered);
      * ``use``  — take the kernel's proposal;
      * ``edit`` — take ``edited_text``.

    The previous caption is moved to ``caption_before_kaggle`` before it is
    overwritten, matching the ``caption_before_moss`` convention: an approved
    caption must never be destroyed by a re-run. A blank proposal is refused
    outright, because writing "" would ERASE a caption and look like success.
    """
    existing = (row.get("existing") or "").strip()
    proposed = (row.get("proposed") or "").strip()

    if decision == "keep":
        if proposed:
            sample["caption_ai_raw"] = proposed
        return existing

    text = (edited_text or "").strip() if decision == "edit" else proposed
    if not text or text.upper().startswith("ERROR:"):
        return existing

    if existing and existing != text:
        sample["caption_before_kaggle"] = existing
    sample["caption"] = text
    sample["caption_ai_raw"] = proposed
    return text
