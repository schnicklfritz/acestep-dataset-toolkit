"""Shared audio-analysis helpers: structural section detection + slicing.

The captioner names specific instruments far more reliably on short, focused
audio chunks than on a whole song. These helpers cut a track at its structural
boundaries (the same MFCC agglomerative clustering the structural pipeline
uses) so instrument detection can caption each section independently.
"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def find_structural_sections(audio_path, min_sec=12.0, max_k=20):
    """Find structural section boundaries via MFCC agglomerative clustering.

    Returns a list of ``{"name", "start", "end"}`` sections (seconds), where
    ``name`` is e.g. ``Section_01``.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    duration = librosa.get_duration(y=y_mono, sr=sr)

    # ---- Adaptive k ----
    if duration < 60:          # less than 1 minute
        k = max(2, int(duration / 15))
    else:                      # 1 minute or longer
        k = max(4, int(duration / 30))
    k = min(k, max_k)

    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
    bounds = librosa.segment.agglomerative(mfcc, k=k)
    bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]

    # Merge sections shorter than min_sec
    filtered = [bound_times[0]]
    for t in bound_times[1:]:
        if t - filtered[-1] >= min_sec:
            filtered.append(t)
    if filtered[-1] < duration:
        filtered[-1] = duration
    bound_times = filtered

    sections = []
    for i in range(len(bound_times) - 1):
        sections.append({
            "name": f"Section_{i+1:02d}",
            "start": bound_times[i],
            "end": bound_times[i + 1],
        })
    return sections


def slice_sections_to_wav(audio_path, sections, out_dir):
    """Slice the audio into per-section WAV chunks.

    Returns a list of ``{"name", "start", "end", "path"}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    base = Path(audio_path).stem
    chunks = []
    for sec in sections:
        start_s = int(sec["start"] * sr)
        end_s = int(sec["end"] * sr)
        chunk = y[:, start_s:end_s] if y.ndim > 1 else y[start_s:end_s]
        out_path = out_dir / f"{base}_{sec['name']}.wav"
        if y.ndim > 1:
            sf.write(str(out_path), chunk.T, sr)
        else:
            sf.write(str(out_path), chunk, sr)
        chunks.append({
            "name": sec["name"],
            "start": sec["start"],
            "end": sec["end"],
            "path": str(out_path),
        })
    return chunks
