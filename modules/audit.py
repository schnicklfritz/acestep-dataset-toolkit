"""
Shared dataset quality-audit aggregation.

Both the local HealthAuditorWorker and the optional Kaggle GPU audit backend
produce per-track reports. This module turns those reports into a dataset-level
quality score, recommendations, and conservative duplicate warnings.
"""

import math
import os


def _canonical_audio_path(value):
    """Return a normalized real file path, or an empty string."""
    if not value:
        return ""

    try:
        return os.path.normcase(
            os.path.realpath(
                os.path.abspath(
                    os.path.expanduser(str(value))
                )
            )
        )
    except (OSError, TypeError, ValueError):
        return ""

def aggregate_health_summary(reports, samples, config=None, progress_cb=None):
    """
    Aggregate per-track health reports into a dataset health summary.

    `reports` maps a sample ID to its per-track audit report.
    `samples` is the list of top-level dataset manifest sample dictionaries.
    """
    config = config or {}
    progress_cb = progress_cb or (lambda _progress, _message: None)

    sample_rates = []
    channels_list = []
    lufs_list = []
    clipping_files = []
    lossy_files = []
    missing_files = []
    uncertain_bpm = []

    for sample in samples:
        sample_id = sample.get("id", "")
        report = reports.get(sample_id, {})
        filename = sample.get("filename", "")

        if report.get("status") == "Missing":
            missing_files.append(filename)
            continue

        if report.get("sample_rate"):
            sample_rates.append(report["sample_rate"])

        if report.get("channels"):
            channels_list.append(report["channels"])

        if report.get("lufs") is not None:
            lufs_list.append(report["lufs"])

        if report.get("is_clipping"):
            clipping_files.append(filename)

        if report.get("has_lossy_cutoff"):
            lossy_files.append(filename)

        if report.get("bpm_confidence", 1.0) < 0.65:
            uncertain_bpm.append(filename)

    total = len(samples)
    quality_score = 100
    reasons = []

    if missing_files:
        penalty = min(40, len(missing_files) * 20)
        quality_score -= penalty
        reasons.append(
            f"Missing Files (-{penalty}%): {len(missing_files)} track(s) "
            "cannot be read on disk."
        )

    if clipping_files:
        penalty = min(20, len(clipping_files) * 10)
        quality_score -= penalty
        reasons.append(
            f"Digital Clipping (-{penalty}%): {len(clipping_files)} track(s) "
            "exceed the -0.1 dBFS ceiling."
        )

    if lossy_files:
        penalty = min(20, len(lossy_files) * 8)
        quality_score -= penalty
        reasons.append(
            f"Lossy Source Inconsistency (-{penalty}%): "
            f"{len(lossy_files)} track(s) have <192 kbps streams or "
            "high-frequency cutoffs."
        )

    unique_sample_rates = sorted(set(sample_rates))

    if len(unique_sample_rates) > 1:
        quality_score -= 10
        reasons.append(
            f"Mixed Sample Rates (-10%): Dataset mixes "
            f"{unique_sample_rates} Hz."
        )

    unique_channels = sorted(set(channels_list))

    if len(unique_channels) > 1:
        quality_score -= 10
        reasons.append(
            f"Mismatched Channels (-10%): Dataset mixes "
            f"{unique_channels} channels."
        )

    lufs_spread = 0.0

    if lufs_list:
        lufs_spread = max(lufs_list) - min(lufs_list)

    if lufs_spread > 5.0:
        quality_score -= 15
        reasons.append(
            f"Loudness Spread (-15%): Volume variation across tracks is "
            f"{lufs_spread:.1f} dB."
        )

    if total < 10:
        quality_score -= 15
        reasons.append(
            f"Small Dataset (-15%): Current size ({total} tracks) is under "
            "the recommended 10+ samples."
        )

    # ---- Near-duplicate detection: unique top-level song pairs only ----
    near_duplicates = []
    duplicate_candidate_count = 0

    if near_duplicates:
        affected_tracks = {
            filename
            for left_name, right_name, _similarity in near_duplicates
            for filename in (left_name, right_name)
        }

        affected_fraction = (
            len(affected_tracks) / total
            if total
            else 0.0
        )

        # Duplicate detection is informative but must never dominate quality.
        penalty = min(15, max(1, round(affected_fraction * 15)))
        quality_score -= penalty

        reasons.append(
            f"Near-Duplicates (-{penalty}%): "
            f"{len(near_duplicates)} unique pair(s) affect "
            f"{len(affected_tracks)} of {total} top-level track(s)."
        )

    # ---- Exact duplicates: byte-identical source files ----
    exact_duplicates = []


    if exact_duplicates:
        redundant_file_count = sum(
            len(group) - 1
            for group in exact_duplicates
        )

        penalty = min(20, redundant_file_count * 10)
        quality_score -= penalty

        reasons.append(
            f"Exact Duplicates (-{penalty}%): "
            f"{len(exact_duplicates)} group(s) of byte-identical files."
        )

    quality_score = max(5, min(100, quality_score))

    recommendations = []

    if missing_files:
        recommendations.append(
            "Restore or re-add the missing audio files before exporting."
        )

    if clipping_files:
        recommendations.append(
            "Re-export clipping tracks with a lower ceiling, such as -1 dBTP."
        )

    if lossy_files:
        recommendations.append(
            "Replace lossy sources with lossless masters for cleaner training."
        )

    if len(unique_sample_rates) > 1:
        recommendations.append(
            "Run DSP Normalize to resample every track to one sample rate."
        )

    if lufs_spread > 5.0:
        recommendations.append(
            "Run DSP Normalize (EBU R128) to reduce loudness variation."
        )

    if near_duplicates:
        recommendations.append(
            "Review the listed unique song pairs and remove only confirmed "
            "duplicate recordings or alternate masters."
        )

    if exact_duplicates:
        recommendations.append(
            "Remove exact byte-identical files to avoid redundant training data."
        )

    if uncertain_bpm:
        recommendations.append(
            "Verify BPM and key on tracks whose detection confidence was low."
        )

    if total < 10:
        recommendations.append(
            "Aim for 10 or more tracks; very small datasets limit LoRA "
            "generalization."
        )

    if not recommendations:
        recommendations.append(
            "Dataset is healthy — ready to caption and export."
        )

    return {
        "quality_score": quality_score,
        "healthy": quality_score >= 80,
        "reasons": reasons,
        "recommendations": recommendations,
        "near_duplicates": near_duplicates,
        "exact_duplicates": exact_duplicates,
        "total_audited": total,
        "duplicate_candidate_count": duplicate_candidate_count,
        "unique_sample_rates": unique_sample_rates,
        "unique_channels": unique_channels,
        "lufs_spread": lufs_spread,
        "clipping_count": len(clipping_files),
        "lossy_count": len(lossy_files),
        "missing_count": len(missing_files),
    }
