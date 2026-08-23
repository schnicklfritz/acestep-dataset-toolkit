"""Music Dataset Homogeneity Engine.

Audits sample rates, dynamic range (Crest Factor), LUFS variance, and
spectral centroids to ensure dataset consistency.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

logger = logging.getLogger("homogeneity")


@dataclass
class TrackMetrics:
    file_name: str
    sample_rate: int
    channels: int
    duration_sec: float
    lufs: float
    peak_db: float
    crest_factor_db: float
    spectral_centroid_hz: float
    is_outlier: bool = False
    outlier_reasons: List[str] = None


@dataclass
class HomogeneityReport:
    score: float  # 0.0 to 100.0
    total_tracks: int
    sample_rates: List[int]
    lufs_mean: float
    lufs_std: float
    crest_mean_db: float
    crest_std_db: float
    centroid_mean_hz: float
    centroid_std_hz: float
    outliers_count: int
    track_details: List[TrackMetrics]
    action_plan: List[str]


class HomogeneityEngine:

    def __init__(self, target_sr: int = 44100, target_lufs: float = -14.0):
        self.target_sr = target_sr
        self.target_lufs = target_lufs

    def _calc_spectral_centroid(self, audio: np.ndarray, sr: int) -> float:
        """Compute the average spectral centroid (brightness) in Hz."""
        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        # Fast FFT on a representative 30s middle chunk to keep audit snappy
        mid_start = len(mono) // 4
        chunk = mono[mid_start : mid_start + (sr * 30)]
        if len(chunk) < sr:
            chunk = mono

        fft_mag = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

        sum_mag = np.sum(fft_mag)
        if sum_mag == 0:
            return 0.0
        return float(np.sum(freqs * fft_mag) / sum_mag)

    def analyze_directory(self, audio_dir: Path) -> HomogeneityReport:
        """Analyze all compatible audio files in a directory."""
        valid_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        audio_files = sorted(
            [f for f in audio_dir.rglob("*") if f.suffix.lower() in valid_exts]
        )

        if not audio_files:
            return HomogeneityReport(
                score=0.0,
                total_tracks=0,
                sample_rates=[],
                lufs_mean=0.0,
                lufs_std=0.0,
                crest_mean_db=0.0,
                crest_std_db=0.0,
                centroid_mean_hz=0.0,
                centroid_std_hz=0.0,
                outliers_count=0,
                track_details=[],
                action_plan=["No audio files found in directory."],
            )

        metrics_list: List[TrackMetrics] = []

        for f in audio_files:
            data, sr = sf.read(str(f), always_2d=True)
            duration = len(data) / sr

            # 1. Loudness via EBU R128
            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(data)

            # 2. Crest Factor (Peak to RMS ratio in dB)
            peak_val = np.max(np.abs(data))
            peak_db = (
                20.0 * np.log10(peak_val) if peak_val > 0 else -100.0
            )
            rms = np.sqrt(np.mean(data**2))
            crest_factor_db = (
                20.0 * np.log10(peak_val / (rms + 1e-9))
                if rms > 0
                else 0.0
            )

            # 3. Spectral Centroid
            centroid = self._calc_spectral_centroid(data, sr)

            metrics_list.append(
                TrackMetrics(
                    file_name=f.name,
                    sample_rate=sr,
                    channels=data.shape[1],
                    duration_sec=round(duration, 2),
                    lufs=round(float(lufs), 2),
                    peak_db=round(float(peak_db), 2),
                    crest_factor_db=round(float(crest_factor_db), 2),
                    spectral_centroid_hz=round(float(centroid), 1),
                    outlier_reasons=[],
                )
            )

        # Statistical baseline calculations
        lufs_vals = [m.lufs for m in metrics_list if np.isfinite(m.lufs)]
        crest_vals = [m.crest_factor_db for m in metrics_list]
        centroid_vals = [m.spectral_centroid_hz for m in metrics_list]
        sr_set = list(set(m.sample_rate for m in metrics_list))

        lufs_mean = float(np.mean(lufs_vals))
        lufs_std = float(np.std(lufs_vals))
        crest_mean = float(np.mean(crest_vals))
        crest_std = float(np.std(crest_vals))
        centroid_mean = float(np.mean(centroid_vals))
        centroid_std = float(np.std(centroid_vals))

        # Outlier Detection (Z-score > 1.75 or sample rate divergence)
        outlier_count = 0
        action_plan = []

        for m in metrics_list:
            reasons = []
            if m.sample_rate != self.target_sr:
                reasons.append(
                    f"Sample rate {m.sample_rate}Hz != target {self.target_sr}Hz"
                )

            if lufs_std > 1.0 and abs(m.lufs - lufs_mean) > (1.75 * lufs_std):
                reasons.append(
                    f"LUFS outlier ({m.lufs} vs mean {lufs_mean:.1f})"
                )

            if crest_std > 1.5 and abs(m.crest_factor_db - crest_mean) > (
                1.75 * crest_std
            ):
                reasons.append(
                    f"Dynamic range outlier (Crest {m.crest_factor_db}dB vs mean {crest_mean:.1f}dB)"
                )

            if centroid_std > 300 and abs(
                m.spectral_centroid_hz - centroid_mean
            ) > (1.75 * centroid_std):
                reasons.append(
                    f"Timbre brightness anomaly ({m.spectral_centroid_hz}Hz)"
                )

            if reasons:
                m.is_outlier = True
                m.outlier_reasons = reasons
                outlier_count += 1

        # Calculate Overall Homogeneity Score (0-100)
        score = 100.0
        score -= min(30.0, lufs_std * 6.0)  # Penalize loudness variance
        score -= min(
            30.0, crest_std * 5.0
        )  # Penalize master dynamic range clash
        score -= min(20.0, (len(sr_set) - 1) * 15.0)  # Penalize mixed clocks
        score = max(0.0, round(score, 1))

        # Build Harmonization Action Plan
        if len(sr_set) > 1:
            action_plan.append(
                f"Resample {len(audio_files) - sr_set.count(self.target_sr)} tracks to uniform {self.target_sr} Hz."
            )
        if lufs_std > 1.5:
            action_plan.append(
                f"Run 2-pass EBU R128 normalizer to collapse {lufs_std:.1f} LUFS spread down to uniform {self.target_lufs} LUFS."
            )
        if crest_std > 2.5:
            action_plan.append(
                "Significant dynamic range clash detected (mixed quiet vinyl & squashed remasters). Homogenize with soft limiter ceiling."
            )
        if not action_plan:
            action_plan.append(
                "Dataset is highly homogeneous. Ready for zero-crossing slicing."
            )

        return HomogeneityReport(
            score=score,
            total_tracks=len(audio_files),
            sample_rates=sr_set,
            lufs_mean=round(lufs_mean, 2),
            lufs_std=round(lufs_std, 2),
            crest_mean_db=round(crest_mean, 2),
            crest_std_db=round(crest_std, 2),
            centroid_mean_hz=round(centroid_mean, 1),
            centroid_std_hz=round(centroid_std, 1),
            outliers_count=outlier_count,
            track_details=metrics_list,
            action_plan=action_plan,
        )
