"""What the assistant is actually allowed to do to a dataset.

WHY THIS MODULE EXISTS
----------------------
"Ask the app in English and it does the task" moves a lot of trust onto a model.
The model proposes an action; this module decides whether that action is
*something that exists* and *something safe*, and then performs it. Keeping that
decision here rather than inside the tool-call handler is what makes it
reviewable:

  * Every function is **pure** — it takes the dataset dict and returns a short
    human-readable result. No Qt, no network, no threads. So the whole
    "assistant did it" surface is unit-testable without an LLM, which is
    precisely the part that must not silently do the wrong thing.
  * The writable-field set is an **allowlist**. A model must never be able to
    point ``audio_path`` at another file, rewrite an ``id`` (it is the stable
    key) or set ``labeled`` to claim metadata was reviewed. Those are one bad
    hallucination away from a corrupt dataset.
  * Destructive/expensive work (ffmpeg, Kaggle, transcription) is expressed as
    a **command or a plan**, not as a side effect, so the caller can show it,
    run it in a worker, or refuse it.

Reported back to the model are *facts about the dataset* (what is missing, what
was written) — never an instruction the model has to take on faith.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Field allowlists
# ---------------------------------------------------------------------------
# Fields an assistant may write directly. Deliberately excludes:
#   id, audio_path, filename  -> identity/paths: the app's own import + rename
#                                flows own these, and a wrong value breaks
#                                every filename-matched import downstream.
#   raw_lyrics                -> only written by the lyrics importer, which
#                                keeps formatted_lyrics in step.
#   labeled, locked           -> provenance flags; a model must not claim work
#                                was human-reviewed.
WRITABLE_FIELDS = (
    "caption",
    "genre",
    "custom_tag",
    "language",
    "keyscale",
    "timesignature",
    "bpm",
    "duration",
    "is_instrumental",
    "lyrics",
    "formatted_lyrics",
)

# Fields a track needs before it is worth exporting for training. Mirrors the
# check in modules/manifest_validation.py; kept here as data so the gap report
# can name them one by one instead of saying "invalid".
REQUIRED_FOR_TRAINING = (
    "caption",
    "genre",
    "bpm",
    "keyscale",
    "language",
)

# Language the app ships as the default, so a blank field is not a gap.
DEFAULT_LANGUAGE = "en"


def _samples(dataset):
    return dataset.setdefault("samples", [])


def resolve_indices(dataset, refs):
    """Map what the model said onto real indices.

    Accepts 1-based positions (the numbering ``list_tracks`` prints), 0-based
    positions, or filenames. Returns ``(indices, unresolved)`` so the caller can
    report a miss instead of silently editing the wrong track.
    """
    samples = _samples(dataset)
    resolved, unresolved = [], []
    for ref in refs or []:
        found = None
        text = str(ref).strip()
        # Filename first: a name is unambiguous, an index is a guess.
        for i, s in enumerate(samples):
            if (s.get("filename") or "") == text:
                found = i
                break
        if found is None and text.isdigit():
            n = int(text)
            if 1 <= n <= len(samples):
                found = n - 1            # 1-based, as list_tracks prints
            elif 0 <= n < len(samples):
                found = n                # tolerate 0-based
        if found is None:
            unresolved.append(ref)
        elif found not in resolved:
            resolved.append(found)
    return resolved, unresolved


def _format_value(field, value):
    """Parse a JSON value into the type the schema stores for ``field``."""
    if field == "is_instrumental":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "instrumental")
    if field in ("bpm", "duration"):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0
    return "" if value is None else str(value)


def set_track_field(dataset, index, field, value):
    """Write one allowlisted field on one track (0-based ``index``)."""
    samples = _samples(dataset)
    if field not in WRITABLE_FIELDS:
        return f"Refused: '{field}' is not a field the assistant may write."
    if not (0 <= index < len(samples)):
        return f"No such track: index {index}."
    s = samples[index]
    parsed = _format_value(field, value)
    s[field] = parsed
    shown = "true" if parsed is True else "false" if parsed is False else parsed
    return f"{s.get('filename', '?')}: {field} = {shown}"


def set_field_many(dataset, refs, field, value):
    """Write one field across many tracks, reporting each one."""
    indices, unresolved = resolve_indices(dataset, refs)
    if not indices:
        return f"No tracks matched {refs!r}."
    lines = [set_track_field(dataset, i, field, value) for i in indices]
    if unresolved:
        lines.append(f"(not found: {', '.join(str(u) for u in unresolved)})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lyrics import
# ---------------------------------------------------------------------------
def import_lyrics(dataset, refs, text):
    """Put ``text`` on the matched tracks as their lyrics.

    All three lyric fields are written together on purpose: ``formatted_lyrics``
    is what the exporter reads, ``lyrics`` is what the inspector shows, and
    ``raw_lyrics`` is the pre-formatting copy the tidy pass keeps for undo. A
    tool that filled only one of them would look right and still ship the
    wrong text.
    """
    indices, unresolved = resolve_indices(dataset, refs)
    if not indices:
        return f"No tracks matched {refs!r}."
    body = (text or "").strip()
    if not body:
        return "Refused: the lyrics text was empty."
    samples = _samples(dataset)
    for i in indices:
        s = samples[i]
        s["raw_lyrics"] = body
        s["formatted_lyrics"] = body
        s["lyrics"] = body
        # Lyrics on a track marked instrumental is the contradiction
        # manifest_validation flags, so clear the flag rather than create it.
        if s.get("is_instrumental"):
            s["is_instrumental"] = False
    names = ", ".join(samples[i].get("filename", "?") for i in indices)
    out = [f"Imported {len(body)} characters of lyrics into: {names}"]
    if unresolved:
        out.append(f"(not found: {', '.join(str(u) for u in unresolved)})")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# ffmpeg: normalization and temp-MP3 staging
# ---------------------------------------------------------------------------
def build_normalize_command(src, dst, target_lufs=-14.0, target_sr=44100):
    """argv for EBU R128 loudness normalization.

    Mirrors workers/dsp_normalizer.py exactly (loudnorm I=<lufs>:TP=-1.0:LRA=11,
    forced sample rate and stereo) so the assistant and the 🎚 DSP Normalize
    button cannot produce different audio for the same settings.
    """
    return [
        "ffmpeg", "-y", "-i", str(src),
        "-af", f"loudnorm=I={float(target_lufs)}:TP=-1.0:LRA=11",
        "-ar", str(int(target_sr)), "-ac", "2", str(dst),
    ]


def build_temp_mp3_command(src, dst, bitrate="192k"):
    """argv for the scratch MP3 conversion.

    Why MP3 at all: the structural/transcription kernels take a small, uniform
    upload, and Kaggle dataset sources are size-limited. This output is
    *staging only* — it never replaces the training audio, which stays lossless
    (see the lossy-format warning in dataset_manager.AUDIO_EXTS).
    """
    return [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ar", "44100", "-ac", "2",
        "-b:a", bitrate, str(dst),
    ]


def normalize_tracks(dataset, refs, target_dir, target_lufs=-14.0,
                     target_sr=44100, runner=subprocess.run):
    """Normalize the matched tracks into ``<target_dir>/normalized_audio``.

    ``runner`` is injectable so a test can assert the ffmpeg argv without
    running ffmpeg. Originals are copied to ``<target_dir>/originals_backup``
    first — same contract as the DSP Normalize button.
    """
    indices, unresolved = resolve_indices(dataset, refs)
    samples = _samples(dataset)
    norm_dir = os.path.join(str(target_dir), "normalized_audio")
    backup_dir = os.path.join(str(target_dir), "originals_backup")
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    lines = []
    for i in indices:
        s = samples[i]
        src = s.get("audio_path") or ""
        if not src or not os.path.exists(src):
            lines.append(f"{s.get('filename', '?')}: SKIPPED (no audio on disk)")
            continue
        fname = s.get("filename") or os.path.basename(src)
        backup = os.path.join(backup_dir, fname)
        if not os.path.exists(backup):
            shutil.copy2(src, backup)
        dst = os.path.join(norm_dir, f"norm_{Path(fname).stem}.wav")
        try:
            res = runner(build_normalize_command(src, dst, target_lufs, target_sr),
                         capture_output=True, text=True, check=False)
        except OSError as e:
            lines.append(f"{fname}: ffmpeg not runnable ({e})")
            continue
        if getattr(res, "returncode", 1) == 0 and os.path.exists(dst):
            lines.append(
                f"{fname}: normalized to {target_lufs} LUFS / {target_sr} Hz -> {dst}"
            )
        else:
            lines.append(f"{fname}: FAILED (ffmpeg returned {res.returncode})")
    if unresolved:
        lines.append(f"(not found: {', '.join(str(u) for u in unresolved)})")
    return "\n".join(lines) if lines else "No tracks matched."




# ---------------------------------------------------------------------------
# Gap audit — "what is missing or omitted?"
# ---------------------------------------------------------------------------
def _caption_issues(caption):
    """Reuse the app's own caption backstop instead of a second opinion."""
    try:
        from modules.caption_quality import check_caption

        return list(check_caption(caption))
    except Exception:  # noqa: BLE001 — a report must never crash the assistant
        return []


def find_gaps(dataset):
    """Report what each track is missing, and what the dataset as a whole is.

    This is the "attempt a virtual dataset creation and find missing or omitted
    features" half of the job: it is the checklist the model composes its plan
    from, and the thing a human reads to decide whether to trust the plan.
    """
    samples = _samples(dataset)
    if not samples:
        return "Dataset is empty — nothing to audit yet."

    lines = []
    clean = 0
    seen = {}
    for i, s in enumerate(samples, start=1):
        name = s.get("filename") or f"track {i}"
        missing = []
        for f in REQUIRED_FOR_TRAINING:
            if f == "bpm":
                if not s.get("bpm"):
                    missing.append(f)
            elif not str(s.get(f) or "").strip():
                missing.append(f)
        mine = []
        if missing:
            mine.append(f"missing {'/'.join(missing)}")
        if not str(s.get("language") or "").strip():
            mine.append(f"no language (defaults to '{DEFAULT_LANGUAGE}')")
        if s.get("is_instrumental") and str(s.get("lyrics") or "").strip():
            mine.append("marked instrumental but has lyrics")
        if not s.get("is_instrumental") and not str(s.get("lyrics") or "").strip():
            mine.append("has no lyrics and is not marked instrumental")
        body = str(s.get("lyrics") or "").strip()
        if body and "[" not in body:
            mine.append("lyrics carry no [Section] markers")
        cap = str(s.get("caption") or "").strip()
        if cap:
            for issue in _caption_issues(cap):
                mine.append(f"caption: {issue}")
        path = s.get("audio_path") or ""
        if s.get("virtual"):
            mine.append("virtual concept — no audio by design")
        elif not path or not os.path.exists(path):
            mine.append("audio file not found on disk")
        if not mine:
            clean += 1
        seen.setdefault(name, []).append(i)
        lines.append(f"{i}. {name}: " + ("; ".join(mine) if mine else "ok"))

    for name, idx in seen.items():
        if len(idx) > 1:
            lines.append(
                f"DATASET: duplicate filename {name} x{len(idx)} — imports "
                "match by filename, so these would collide."
            )
    lines.append(
        f"SUMMARY: {len(samples)} track(s); {clean} fully specified; "
        f"{len(samples) - clean} with gaps."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Virtual dataset creation
# ---------------------------------------------------------------------------
def create_virtual_dataset(dataset, specs, name=None):
    """Draft tracks as *concepts* so gaps can be found before audio exists.

    A virtual track is a normal sample with ``virtual=True`` and no
    ``audio_path``: it takes part in the gap audit, its caption can be written
    and reviewed, and it is skipped by everything that needs a file. It never
    reaches training data — ``workers/export.py`` drops virtual samples before
    handing the list to any exporter (all five formats).

    ``specs`` is a list of dicts; only allowlisted metadata is honoured.
    """
    from modules.dataset_schema import new_sample

    if not specs:
        return "Refused: no tracks specified."
    if name:
        dataset.setdefault("metadata", {})["name"] = str(name)

    added = []
    for spec in specs:
        spec = dict(spec or {})
        filename = str(spec.get("filename") or "").strip()
        if not filename:
            # Derive a stable, readable placeholder so gap reports make sense.
            slug = "".join(
                ch if ch.isalnum() else "_"
                for ch in str(spec.get("genre") or "track").lower()
            ).strip("_")
            filename = f"virtual_{len(_samples(dataset)) + 1:02d}_{slug or 'track'}.wav"
        sample = new_sample(filename=filename, audio_path="")
        sample["virtual"] = True
        sample["language"] = DEFAULT_LANGUAGE
        for field in WRITABLE_FIELDS:
            if field in spec and spec[field] not in (None, ""):
                sample[field] = _format_value(field, spec[field])
        _samples(dataset).append(sample)
        added.append(filename)

    return (
        f"Created {len(added)} virtual track(s): {', '.join(added)}\n"
        "These have no audio yet, so normalization, transcription and export "
        "skip them until a file is attached. Run the gap audit to see what each "
        "one still needs."
    )
