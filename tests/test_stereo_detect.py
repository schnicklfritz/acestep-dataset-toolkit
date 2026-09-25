"""detect_stereo: side/mid ratio on the full mix, measured on real encodes."""
import shutil
import subprocess

import librosa
import numpy as np
import pytest
import soundfile as sf

from modules.tagger import MONO_SIDE_MID_DB, detect_stereo, format_tags

SR = 22050


def _signal(seed=0, freq=220.0):
    t = np.arange(SR * 4) / SR
    rng = np.random.default_rng(seed)
    return 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(t.size)


def _cases():
    m = _signal()
    rng = np.random.default_rng(1)
    return {
        "dual_mono": (np.stack([m, m]), "mono"),
        "narrow_stereo": (
            np.stack([m + 0.05 * rng.standard_normal(m.size),
                      m + 0.05 * rng.standard_normal(m.size)]),
            "stereo",
        ),
        "hard_pan": (np.stack([m, _signal(2, 330.0)]), "stereo"),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_arrays(name):
    y, want = _cases()[name]
    assert detect_stereo(y)[0] == want


def test_single_channel_is_mono():
    assert detect_stereo(_signal()) == ("mono", None)


def test_silence_does_not_crash():
    assert detect_stereo(np.zeros((2, 1000)))[0] == "mono"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("ext,args", [("mp3", ["-b:a", "128k"]), ("ogg", ["-q:a", "3"])])
@pytest.mark.parametrize("name", list(_cases()))
def test_survives_lossy_encoding(tmp_path, name, ext, args):
    y, want = _cases()[name]
    wav = tmp_path / f"{name}.wav"
    out = tmp_path / f"{name}.{ext}"
    sf.write(wav, y.T, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), *args, str(out)],
                   check=True)
    y2, _ = librosa.load(out, sr=None, mono=False)
    label, db = detect_stereo(y2)
    assert label == want, (label, db)
    if want == "stereo":
        assert db > MONO_SIDE_MID_DB


def test_only_mono_is_written_into_the_tag_block():
    # Stereo is ACE-Step's default (its VAE is stereo), so only the exception
    # is worth caption space.
    assert "mono recording" in format_tags({"stereo": "mono"})
    assert "stereo" not in format_tags({"stereo": "stereo"}).lower()
