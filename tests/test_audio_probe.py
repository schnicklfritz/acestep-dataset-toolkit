"""The Duration/Time cells and the audio probe behind them.

Two things read as "the column is missing" and neither is a missing column:

  * ``Time``/``Duration`` were hidden in ``DEFAULT_HIDDEN_COLUMNS``;
  * nothing ever wrote ``duration`` on import, so even a visible column was
    blank for every track the app added itself.

These tests pin the display format (``m:ss``), the blank-Time default (4/4 is
shown, never written), the probe's refusal to invent a length, and the ORDER of
the probe chain: the in-process header read first, then ``soxi``, then
``ffprobe``. The order is asserted, not just assumed, because it is what keeps a
bulk import from paying two forks per track.
"""
from __future__ import annotations

import shlex
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


# ---------------------------------------------------------------------------
# The probe chain: soundfile -> soxi -> ffprobe
# ---------------------------------------------------------------------------

def _isolated_bin(tmp_path, monkeypatch):
    """An empty ``PATH`` holding only the fake tools a test installs.

    ``PATH`` is replaced rather than extended so the real ``soxi``/``ffprobe``
    this machine happens to have can never answer a test that meant to exercise
    a fake -- and so "tool not installed" is testable at all.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


def _fake_tool(bin_dir, name, *lines):
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + "\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_the_header_read_is_asked_before_any_subprocess(tmp_path, monkeypatch):
    """A readable wav must not fork: that is what keeps a bulk import quick.

    The fakes leave a marker file when invoked, so a chain that merely *looks*
    right but shells out first fails here instead of in the UI.
    """
    bin_dir = _isolated_bin(tmp_path, monkeypatch)
    soxi_marker = tmp_path / "soxi-ran"
    ffprobe_marker = tmp_path / "ffprobe-ran"
    _fake_tool(bin_dir, "soxi", f"touch {shlex.quote(str(soxi_marker))}", "echo 999")
    _fake_tool(bin_dir, "ffprobe", f"touch {shlex.quote(str(ffprobe_marker))}",
               "echo '{}'")

    wav = _write_wav(tmp_path / "song.wav", 2.0)
    assert audio_duration_seconds(wav) == pytest.approx(2.0, abs=0.01)
    assert not soxi_marker.exists()
    assert not ffprobe_marker.exists()


def test_soxi_measures_what_the_header_read_cannot(tmp_path, monkeypatch):
    """sox opens containers libsndfile cannot; its bare float is the answer."""
    bin_dir = _isolated_bin(tmp_path, monkeypatch)
    _fake_tool(bin_dir, "soxi", "echo 180.600000")
    junk = tmp_path / "song.m4a"
    junk.write_bytes(b"not audio for libsndfile")

    assert audio_duration_seconds(junk) == pytest.approx(180.6)
    # The field is whole seconds, as the documented example shows (``180``).
    assert duration_seconds_rounded(junk) == 181


def test_a_failing_soxi_falls_through_to_ffprobe(tmp_path, monkeypatch):
    """soxi exits non-zero and writes to stderr for a file it cannot open."""
    bin_dir = _isolated_bin(tmp_path, monkeypatch)
    _fake_tool(bin_dir, "soxi", "echo 'soxi FAIL formats: no handler' >&2", "exit 1")
    _fake_tool(bin_dir, "ffprobe", 'printf \'{"format": {"duration": "95.5"}}\'')

    junk = tmp_path / "song.m4a"
    junk.write_bytes(b"not audio for libsndfile")
    assert audio_duration_seconds(junk) == pytest.approx(95.5)


def test_unparseable_soxi_output_is_unknown_not_a_crash(tmp_path, monkeypatch):
    """``float("N/A")`` must not escape as an exception during a dataset load."""
    bin_dir = _isolated_bin(tmp_path, monkeypatch)
    _fake_tool(bin_dir, "soxi", "echo N/A")
    junk = tmp_path / "song.m4a"
    junk.write_bytes(b"not audio for libsndfile")

    assert audio_duration_seconds(junk) == 0.0
    assert duration_seconds_rounded(junk) == 0


def test_a_missing_file_is_unknown_even_with_soxi_installed(tmp_path, monkeypatch):
    """The real soxi is on this machine; a nonexistent path must still be 0."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    _fake_tool(bin_dir, "soxi", 'echo "soxi FAIL formats: can\'t open" >&2', "exit 1")

    assert audio_duration_seconds(tmp_path / "gone.wav") == 0.0
