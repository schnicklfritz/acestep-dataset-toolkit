"""Shared fixtures for the ACE-Step dataset toolkit test suite.

Run with:  .venv/bin/python -m pytest -q

Tests must never touch the network. Any LLM provider (Groq / DeepSeek /
Gemini / OpenRouter) is mocked — see the ``no_network`` autouse fixture.
"""
import os
import struct
import sys
import wave

import pytest

# The app is a flat-layout repo (modules/, ui/, workers/ at the root), so make
# sure the root is importable regardless of where pytest is invoked from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Qt must run headless in CI / terminals without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test tries to open a real HTTP connection.

    The LLM client is OpenAI-compatible, so a stray live call would silently
    burn quota and make tests non-deterministic.
    """
    import socket

    real_connect = socket.socket.connect

    def _blocked(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address, *args, **kwargs)
        raise RuntimeError(
            f"Test tried to reach the network: {address!r}. "
            "Mock the provider instead of calling it live."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=False)


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session (Qt forbids more than one)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def wav_file(tmp_path):
    """Factory for a small, valid mono WAV on disk.

    Returns a callable so a test can make several:

        def test_x(wav_file):
            p = wav_file("song.wav", seconds=0.1)
    """
    def _make(name="sample.wav", seconds=0.1, sr=44100, channels=1):
        path = tmp_path / name
        frames = int(sr * seconds)
        with wave.open(str(path), "w") as f:
            f.setnchannels(channels)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))
        return str(path)

    return _make


@pytest.fixture
def dataset():
    """A small in-memory dataset in the app's internal schema."""
    from modules.dataset_schema import new_dataset, new_sample

    ds = new_dataset(name="test_dataset")
    ds["samples"] = [
        new_sample(
            id="aaa111",
            filename="song_a.mp3",
            audio_path="./songs/song_a.mp3",
            genre="Rock",
            caption="An energetic rock track.",
            lyrics="[verse]\nShe'll be running, can't you see!",
            formatted_lyrics="[verse]\nShe'll be running, can't you see!",
            bpm=140,
            keyscale="E minor",
            language="en",
        ),
        new_sample(
            id="bbb222",
            filename="song_b.wav",
            audio_path="./songs/song_b.wav",
            genre="Synthwave",
            caption="112 bpm analog synth track.",
            is_instrumental=True,
            bpm=112,
            keyscale="F# minor",
            language="en",
        ),
    ]
    return ds
