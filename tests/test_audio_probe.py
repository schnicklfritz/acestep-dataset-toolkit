"""The Duration/Time cells and the audio probe behind them.

Two things read as "the column is missing" and neither is a missing column:

  * ``Time``/``Duration`` were hidden in ``DEFAULT_HIDDEN_COLUMNS``;
  * nothing ever wrote ``duration`` on import, so even a visible column was
    blank for every track the app added itself.

These tests pin the display format (``m:ss``), the blank-Time default (4/4 is
shown, never written), and the probe's refusal to invent a length.
"""
from __future__ import annotations

import wave

import pytest

from modules.audio_probe import (
    audio_duration_seconds,
    backfill_durations,
    duration_seconds_rounded,
)
from ui.track_table import DEFAULT_TIMESIGNATURE, format_duration


def _write_wav(path, seconds, samplerate=8000):
    """A silent mono wav -- the header is all the probe reads."""
    frames = int(seconds * samplerate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, ""),          # unknown is blank, never "0:00"
        (None, ""),
        ("", ""),
        (9, "0:09"),
        (95, "1:35"),
        (600, "10:00"),
        (3723, "1:02:03"),
    ],
)
def test_duration_is_displayed_as_a_clock_value(seconds, expected):
    assert format_duration(seconds) == expected


def test_the_display_format_round_trips_through_the_cell_parser():
    """``m:ss`` is what the user edits, and the editor must read it back.

    The stored field is whole seconds; if the display format were not parseable
    the cell would silently store 0 the first time it was edited.
    """
    from dataset_manager import DatasetManager
    assert DatasetManager._parse_manual_value(None, "duration", "1:35") == 95
    assert DatasetManager._parse_manual_value(None, "duration", "1:02:03") == 3723


def test_a_blank_time_signature_shows_four_four():
    assert DEFAULT_TIMESIGNATURE == "4/4"


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def test_a_wav_is_measured_from_its_header(tmp_path):
    path = _write_wav(tmp_path / "two_and_a_half.wav", 2.5)
    assert audio_duration_seconds(path) == pytest.approx(2.5, abs=0.01)
    assert duration_seconds_rounded(path) == 2


def test_an_unreadable_file_is_unknown_not_zero_length(tmp_path):
    """0.0 is the sentinel for "cannot measure"; the table renders it blank."""
    missing = tmp_path / "nope.wav"
    assert audio_duration_seconds(missing) == 0.0
    assert duration_seconds_rounded(missing) == 0

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")
    assert audio_duration_seconds(junk) == 0.0


def test_backfill_fills_only_blank_durations(tmp_path):
    measured = _write_wav(tmp_path / "measured.wav", 3.0)
    dataset = {
        "samples": [
            {"audio_path": str(measured), "duration": 0},
            {"audio_path": str(measured), "duration": 250},   # typed by hand
            {"audio_path": str(tmp_path / "gone.wav"), "duration": 0},
            {"audio_path": "", "duration": 0},
        ]
    }
    assert backfill_durations(dataset) == 1
    assert dataset["samples"][0]["duration"] == 3
    assert dataset["samples"][1]["duration"] == 250
    assert dataset["samples"][2]["duration"] == 0
    assert dataset["samples"][3]["duration"] == 0


def test_backfill_tolerates_a_dataset_with_no_samples():
    assert backfill_durations({}) == 0
    assert backfill_durations({"samples": None}) == 0
