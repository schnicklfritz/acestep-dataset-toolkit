"""Audio analysis toolkit: BPM, key, duration, and meter estimation.

Four independent, confidence-scored analyzers over a single loaded audio
buffer, sharing one entry point (analyze_track) so a caller only has to load
the audio once. Each analyzer is also exposed standalone for callers that
only need one value.

Design notes
------------
- BPM and key detection are well-established, reliable techniques (beat
  tracking via onset periodicity; key via chroma/Krumhansl-Schmuckler key
  profile correlation). Both return a confidence-like score, but these are
  generally trustworthy on real music.
- Meter (time signature) estimation is fundamentally different: it is an
  unsolved MIR problem. Published CNN classifiers trained on labeled
  datasets (METER2800) only reach ~69% accuracy on curated test sets. The
  estimator here is a from-scratch accent-grouping heuristic, validated only
  against synthetic click tracks with a known accent pattern. Treat its
  output as a *suggestion for manual review*, never as ground truth --
  see integrate_into_sample() for the non-destructive, lock-respecting
  way to apply it.
- Duration is exact and requires no estimation; it is included here purely
  so a caller can get everything from one module/one audio load.

None of these functions raise on failure -- they degrade to empty/zero
values with an "error" field set, mirroring the existing analyze_track /
on_file_audited convention already used elsewhere in this app (confidence
scores, never crashing the health-audit pass on a single bad file).
"""
from __future__ import annotations

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:  # noqa: BLE001
    _LIBROSA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles (major/minor), used by detect_key().
# ---------------------------------------------------------------------------
_KEY_PROFILE_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KEY_PROFILE_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Tempogram-ratio-style grouping periods used by estimate_meter(). Kept as a
# module constant so callers/tests can widen it (e.g. add 6 for 6/8) without
# touching function internals.
_METER_LABELS = {3: "3/4", 4: "4/4", 6: "6/8"}


def audio_analysis_available():
    """True if librosa is importable in this environment."""
    return _LIBROSA_AVAILABLE


# ---------------------------------------------------------------------------
# BPM
# ---------------------------------------------------------------------------
def detect_bpm(y, sr):
    """Estimate tempo in beats per minute from a loaded audio buffer.

    Returns (bpm: float, confidence: float). Confidence is a simple
    proxy (not from librosa) based on how strongly peaked the onset
    envelope's autocorrelation is around the detected tempo -- higher
    means a clearer, more regular beat.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo[0]) if len(tempo) else 0.0

        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        confidence = 0.5
        if ac.size > 1 and ac[0] > 1e-9:
            confidence = float(np.clip(ac[1:].max() / ac[0], 0.0, 1.0))

        return bpm, confidence
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------
def detect_key(y, sr):
    """Estimate musical key via chroma + Krumhansl-Schmuckler key-profile
    correlation.

    Returns (key: str like "C major"/"A minor", confidence: float 0-1).
    Confidence is the winning profile's correlation coefficient, clipped
    to [0, 1] -- higher means the audio's pitch-class distribution matches
    that key's expected profile more closely.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        best_score, best_key = -1e9, ""
        for i in range(12):
            major_profile = np.roll(_KEY_PROFILE_MAJOR, i)
            minor_profile = np.roll(_KEY_PROFILE_MINOR, i)
            major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
            if major_corr > best_score:
                best_score, best_key = major_corr, f"{_NOTE_NAMES[i]} major"
            if minor_corr > best_score:
                best_score, best_key = minor_corr, f"{_NOTE_NAMES[i]} minor"

        confidence = float(np.clip(best_score, 0.0, 1.0))
        return best_key, confidence
    except Exception:  # noqa: BLE001
        return "", 0.0


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------
def get_duration(y, sr):
    """Exact duration in seconds. No estimation involved -- included so
    callers can get bpm/key/duration/meter from one loaded buffer.
    """
    try:
        return float(librosa.get_duration(y=y, sr=sr))
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Meter / time signature (heuristic -- see module docstring)
# ---------------------------------------------------------------------------
def estimate_meter(y, sr, margin=1.10, candidate_periods=(3, 4)):
    """Estimate whether a track's accent pattern groups in 3s or 4s.

    This measures *subjective accenting* directly, rather than assuming
    meter shows up as a timing irregularity (it usually doesn't -- a 3/4
    waltz can have perfectly evenly-spaced beats, distinguished from 4/4
    only by which beat is emphasized). Method: detect onsets, take each
    onset's *strength* (not just its timestamp), group onsets by
    position-mod-N for each candidate N, and compare how much the loudest
    group in each grouping stands out from that grouping's average.
    Whichever grouping shows the larger relative standout wins.

    Returns a dict:
        time_signature_guess : str  -- e.g. "3/4", "4/4", or "" if ambiguous/failed
        confidence : float          -- 0.0-1.0, how separated the two best scores were
        scores : dict[str, float]   -- raw grouping score per candidate period
        n_onsets : int
        error : str or None
    """
    result = {
        "time_signature_guess": "",
        "confidence": 0.0,
        "scores": {},
        "n_onsets": 0,
        "error": None,
    }
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, backtrack=False
        )

        min_onsets = max(candidate_periods) * 3
        if len(onset_frames) < min_onsets:
            result["error"] = f"too few onsets detected ({len(onset_frames)})"
            return result

        strengths = onset_env[onset_frames]
        n = len(strengths)
        result["n_onsets"] = int(n)

        def grouping_score(period):
            groups = [strengths[i::period] for i in range(period)]
            means = np.array([g.mean() for g in groups if len(g) > 0])
            if len(means) < period or means.mean() <= 1e-9:
                return 0.0
            return float((means.max() - means.mean()) / (means.mean() + 1e-9))

        scores = {p: grouping_score(p) for p in candidate_periods}
        result["scores"] = {str(p): round(v, 4) for p, v in scores.items()}

        p_low, p_high = min(candidate_periods), max(candidate_periods)
        score_low, score_high = scores[p_low], scores[p_high]
        total = score_low + score_high
        if total <= 1e-9:
            return result

        if score_low > score_high * margin:
            result["time_signature_guess"] = _METER_LABELS.get(p_low, f"{p_low}-grouping")
            result["confidence"] = round(min(1.0, (score_low - score_high) / total), 3)
        elif score_high > score_low * margin:
            result["time_signature_guess"] = _METER_LABELS.get(p_high, f"{p_high}-grouping")
            result["confidence"] = round(min(1.0, (score_high - score_low) / total), 3)

        return result
    except Exception as e:  # noqa: BLE001 -- must never crash the audit pass
        result["error"] = str(e)
        return result


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------
def analyze_track(audio_path, sr=22050, include_meter=True):
    """Load audio once and run bpm/key/duration (+ optional meter) on it.

    Returns a dict:
        bpm, bpm_confidence,
        key, key_confidence,
        duration,
        time_signature_guess, time_signature_confidence  (only if include_meter),
        error : str or None (top-level load failure only; per-analyzer
                              failures degrade to 0/"" individually and do
                              not set this field)
    """
    out = {
        "bpm": 0.0, "bpm_confidence": 0.0,
        "key": "", "key_confidence": 0.0,
        "duration": 0.0,
        "error": None,
    }
    if include_meter:
        out["time_signature_guess"] = ""
        out["time_signature_confidence"] = 0.0

    if not _LIBROSA_AVAILABLE:
        out["error"] = "librosa is not installed"
        return out

    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if y is None or len(y) == 0:
            out["error"] = "empty or unreadable audio"
            return out
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        return out

    out["bpm"], out["bpm_confidence"] = detect_bpm(y, sr)
    out["key"], out["key_confidence"] = detect_key(y, sr)
    out["duration"] = get_duration(y, sr)

    if include_meter:
        meter = estimate_meter(y, sr)
        out["time_signature_guess"] = meter["time_signature_guess"]
        out["time_signature_confidence"] = meter["confidence"]

    return out


# ---------------------------------------------------------------------------
# Non-destructive sample integration (mirrors on_file_audited() convention)
# ---------------------------------------------------------------------------
def integrate_into_sample(sample, min_meter_confidence=0.35, apply_meter=True):
    """Fill bpm/key/duration/time_signature on a dataset sample dict,
    following the same lock-respecting, non-destructive auto-fill
    convention already used by HealthAuditorWorker.on_file_audited() in
    dataset_manager.py:

      - never overwrites a field that already has a value
      - bpm/key/duration are filled whenever detected (these are reliable
        enough to auto-fill, matching existing app behavior)
      - time_signature is ONLY filled when apply_meter is True AND the
        heuristic's confidence clears min_meter_confidence -- and it always
        stamps its own confidence + source so the UI can flag it for
        manual review rather than treating it as equal to a locked value

    Returns (sample, filled_fields: list[str]).
    """
    audio_path = sample.get("audio_path", "")
    filled_fields = []
    if not audio_path:
        return sample, filled_fields

    result = analyze_track(audio_path, include_meter=apply_meter)
    if result["error"]:
        return sample, filled_fields

    if not sample.get("bpm"):
        sample["bpm"] = result["bpm"]
        sample["bpm_confidence"] = result["bpm_confidence"]
        filled_fields.append("bpm")

    if not sample.get("key_scale"):
        sample["key_scale"] = result["key"]
        sample["key_confidence"] = result["key_confidence"]
        filled_fields.append("key_scale")

    if not sample.get("duration"):
        sample["duration"] = round(result["duration"], 2)
        filled_fields.append("duration")

    if apply_meter and not sample.get("time_signature"):
        guess = result.get("time_signature_guess", "")
        conf = result.get("time_signature_confidence", 0.0)
        if guess and conf >= min_meter_confidence:
            sample["time_signature"] = guess
            sample["time_signature_confidence"] = conf
            sample["time_signature_source"] = "heuristic_accent_grouping"
            filled_fields.append("time_signature")

    return sample, filled_fields
