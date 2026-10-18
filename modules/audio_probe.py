"""Song length for the track table, without adding a dependency.

``soundfile`` already ships with the app (``modules/lead_vocals.py`` uses it),
and ``sf.info`` reads the container HEADER only -- it does not decode -- so
probing a folder of tracks is cheap. ``soxi -D`` is the second opinion (sox
opens a few containers libsndfile does not, and answers with one bare number),
and ``ffprobe`` is the last resort for what neither can open (some ``.m4a``/AAC
files). Both are optional: the app already requires ffmpeg for caption staging,
but a machine with neither tool must still be able to measure a wav.

WHY THE PROCESS-BASED PROBES SIT *AFTER* ``soundfile``
-----------------------------------------------------
``sf.info`` needs no fork. Importing a folder of a few hundred tracks runs the
probe once per file, and spawning two subprocesses per track is the difference
between an instant add and a visible pause -- so the in-process header read is
asked first and the external tools only see files it could not read.

A file that cannot be read returns ``0.0``. The table renders that as a blank
cell; the probe never guesses a length.
"""
from __future__ import annotations

import json
import shutil
import subprocess


def _soxi_seconds(path):
    """``soxi -D``: the length sox reports, in seconds, or ``0.0``.

    soxi prints a single bare float (``180.000000``), so there is no output to
    parse beyond ``float()``. It exits non-zero and writes to stderr for a file
    it cannot open -- including a missing one -- which is the same "cannot
    measure" case as everything else here, not an error to raise.
    """
    soxi = shutil.which("soxi")
    if not soxi:
        return 0.0
    try:
        result = subprocess.run(
            [soxi, "-D", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return 0.0
        return max(0.0, float((result.stdout or "").strip() or 0.0))
    except Exception:                                        # noqa: BLE001
        return 0.0


def audio_duration_seconds(path):
    """Length of ``path`` in seconds, or ``0.0`` when it cannot be measured."""
    try:
        import soundfile as sf  # noqa: PLC0415 -- optional at import time

        info = sf.info(str(path))
        samplerate = float(info.samplerate or 0)
        if samplerate <= 0:
            return 0.0
        return float(info.frames) / samplerate
    except Exception:                                        # noqa: BLE001
        pass

    by_sox = _soxi_seconds(path)
    if by_sox > 0:
        return by_sox

    probe = shutil.which("ffprobe")
    if not probe:
        return 0.0
    try:
        result = subprocess.run(
            [
                probe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return 0.0
        duration = json.loads(result.stdout or "{}").get("format", {}).get("duration")
        return max(0.0, float(duration or 0.0))
    except Exception:                                        # noqa: BLE001
        return 0.0


def duration_seconds_rounded(path):
    """Whole seconds for the ``duration`` field, or ``0`` when unknown."""
    seconds = audio_duration_seconds(path)
    return int(round(seconds)) if seconds > 0 else 0


def backfill_durations(dataset):
    """Fill ``duration`` on samples that have none; returns how many changed.

    Only FALSY values are touched. A length the user typed inline, or one a
    previous probe found, is never recomputed -- the file may have been moved
    or replaced since, and silently rewriting a hand-entered value is the kind
    of edit that cannot be undone. A sample whose file is missing stays at 0
    rather than raising: a dataset is routinely opened on a machine that does
    not hold every track, and one absent file must not stop the load.
    """
    changed = 0
    for sample in (dataset or {}).get("samples") or []:
        if not isinstance(sample, dict) or sample.get("duration"):
            continue
        path = sample.get("audio_path") or ""
        if not path:
            continue
        seconds = duration_seconds_rounded(path)
        if seconds:
            sample["duration"] = seconds
            changed += 1
    return changed
