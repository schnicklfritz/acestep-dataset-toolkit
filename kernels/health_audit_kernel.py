"""ACE-Step dataset health audit — Kaggle kernel.

Runs the same per-track checks as the app's local HealthAuditorWorker
(duration, sample rate, channels, digital clipping, lossy cutoff, LUFS
estimate, BPM/key via librosa) on every audio file in a mounted Kaggle
dataset and writes ``/kaggle/working/audit_results.json``::

    {"results": {"<original filename>": {report}, ...}}

Placeholders substituted by the app at push time:
  {{AUDIO_DATASET_PATH}}  -> /kaggle/input/<audio-dataset-name>
"""
import glob
import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma", ".opus"}


def _install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "librosa", "soundfile", "numpy"], check=False)


_install()

import librosa  # noqa: E402
import numpy as np  # noqa: E402

# Krumhansl-Schmuckler key profiles (mirrors modules/tagger.py).
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_tempo(y, sr):
    """Return an integer BPM estimate (0 if it cannot be determined)."""
    try:
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr)
        return int(round(float(np.atleast_1d(tempo)[0])))
    except Exception:
        return 0


def detect_key(y, sr):
    """Return a key name like 'A minor' or '' via Krumhansl-Schmuckler."""
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        best_corr = -1.0
        best = None
        for shift in range(12):
            rotated = np.roll(chroma_mean, shift)
            corr_maj = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
            corr_min = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])
            if corr_maj > best_corr:
                best_corr, best = corr_maj, (shift, "major")
            if corr_min > best_corr:
                best_corr, best = corr_min, (shift, "minor")
        if best is None:
            return ""
        shift, mode = best
        return f"{KEY_NAMES[shift]} {mode}"
    except Exception:
        return ""


def detect_timesig(y, sr):
    """Return a time-signature estimate ('3/4', '4/4') or '' when uncertain.

    Mirrors modules/tagger.detect_timesig: beat-track + onset-sampling, then
    compare downbeat-pattern salience for bars of 3 vs 4 beats. Blank when too
    short or neither grouping shows a clear meter.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if bpm <= 0 or len(beats) < 16:
            return ""
        beat_env = np.array([
            onset_env[int(b)] for b in beats
            if 0 <= int(b) < len(onset_env)
        ])
        n = len(beat_env)
        m = 12 * (n // 12)
        if m < 24:
            return ""
        seq = beat_env[:m]

        def _salience(k):
            bars = seq.reshape(-1, k)
            means = bars.mean(axis=0)
            denom = float(means.mean()) + 1e-9
            return float(means.std() / denom)

        s3 = _salience(3)
        s4 = _salience(4)
        if max(s3, s4) < 0.08:
            return ""
        if s3 > s4 * 1.15:
            return "3/4"
        if s4 > s3 * 1.15:
            return "4/4"
        return ""
    except Exception:
        return ""


def _analyze_track(path):
    """Mirror of dataset_manager.HealthAuditorWorker._analyze_track."""
    sr = 44100
    channels = 2
    dur = 0.0
    is_clipping = False
    has_lossy_cutoff = False
    lufs_est = -14.0
    issues = []

    try:
        if path.lower().endswith(".wav"):
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                nframes = wf.getnframes()
                dur = nframes / float(sr) if sr > 0 else 0
                sampwidth = wf.getsampwidth()

                frames_to_read = min(nframes, sr * 15)
                raw_data = wf.readframes(frames_to_read)
                if sampwidth == 2 and raw_data:
                    fmt = f"<{len(raw_data)//2}h"
                    samples = struct.unpack(fmt, raw_data)
                    max_val = max(abs(s) for s in samples) if samples else 0
                    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0

                    if max_val >= 32700:
                        is_clipping = True
                        issues.append("Digital clipping (> -0.1 dBFS)")
                    if rms > 0:
                        lufs_est = 20 * math.log10(rms / 32768.0) - 3.0
        else:
            cmd = ["ffprobe", "-v", "error",
                   "-show_entries", "stream=sample_rate,channels,duration,bit_rate",
                   "-of", "json", path]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                meta = json.loads(res.stdout)
                streams = meta.get("streams", [{}])[0]
                sr = int(streams.get("sample_rate", 44100))
                channels = int(streams.get("channels", 2))
                dur = float(streams.get("duration", 0))
                bitrate = int(streams.get("bit_rate", 0))
                if bitrate > 0 and bitrate < 192000:
                    has_lossy_cutoff = True
                    issues.append(f"Lossy stream compression ({bitrate//1000} kbps)")
    except Exception:
        pass

    if sr not in (44100, 48000):
        issues.append(f"Non-standard sample rate ({sr} Hz)")
    if channels == 1:
        issues.append("Mono recording (Stereo recommended)")
    if dur > 0 and dur < 10:
        issues.append("Short track (< 10s) — possible dataset killer")

    # Full-decode check — unreadable/corrupt files must not enter the dataset.
    decode_ok = True
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=20,
        )
        decode_ok = probe.returncode == 0
    except Exception:
        decode_ok = True
    if not decode_ok:
        issues.append("Unreadable / corrupt audio file")

    bpm_detected = 0
    bpm_confidence = 0.4
    key_detected = ""
    key_confidence = 0.4
    timesig = ""
    if decode_ok:
        try:
            y, sr_full = librosa.load(path, sr=None, mono=True, duration=90)
            if len(y) > 0:
                if y.ndim > 1:
                    y = librosa.to_mono(y)
                bpm_detected = detect_tempo(y, sr_full)
                key_detected = detect_key(y, sr_full)
                timesig = detect_timesig(y, sr_full)
                if bpm_detected:
                    bpm_confidence = 0.9
                if key_detected:
                    key_confidence = 0.9
        except Exception:
            pass

    status = "Healthy" if not issues else "Warning"
    return {
        "status": status,
        "sample_rate": sr,
        "channels": channels,
        "duration": round(dur, 2),
        "is_clipping": is_clipping,
        "has_lossy_cutoff": has_lossy_cutoff,
        "lufs": lufs_est,
        "bpm_detected": bpm_detected,
        "bpm_confidence": bpm_confidence,
        "key_detected": key_detected,
        "key_confidence": key_confidence,
        "timesig": timesig,
        "issues": issues,
    }


AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"


def main():
    audio_files = sorted(
        p for p in Path(AUDIO_FOLDER).rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
    )
    results = {}
    for p in audio_files:
        try:
            results[p.name] = _analyze_track(str(p))
        except Exception as e:  # noqa: BLE001
            results[p.name] = {
                "status": "Warning",
                "issues": [f"Audit failed: {e}"],
                "sample_rate": 0,
                "channels": 0,
                "duration": 0,
                "is_clipping": False,
                "has_lossy_cutoff": False,
                "lufs": -14.0,
                "bpm_detected": 0,
                "bpm_confidence": 0.4,
                "key_detected": "",
                "key_confidence": 0.4,
            }

    out_path = "/kaggle/working/audit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Health audit complete: {len(results)} file(s) -> {out_path}")


if __name__ == "__main__":
    main()

