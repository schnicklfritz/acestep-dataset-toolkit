"""Meter (time-signature) estimation from raw audio.

This is a heuristic, confidence-scored *estimator*, not an authoritative
detector. Time-signature detection from audio is a genuinely unsolved
problem in music information retrieval -- published CNN classifiers trained
on labeled datasets (e.g. METER2800) only reach ~69% accuracy on curated
test sets. This module exists to feed a *suggestion* into a manual-review
workflow (e.g. the app's Exceptions Queue). It must never silently auto-lock
time_signature -- see integrate_into_sample() below, which only ever writes
to an empty field and always records its own confidence alongside the guess.

Mechanism
---------
This measures subjective accenting directly, rather than assuming meter
shows up as a timing irregularity (it usually doesn't -- a 3/4 waltz can
have perfectly evenly-spaced beats, distinguished from 4/4 only by which
beat is emphasized). The method:

  1. Detect onsets (percussive/attack transients) via librosa's standard
     spectral-flux onset strength envelope.
  2. Take the onset *strength* (not just its timestamp) at each detected
     onset -- this is the "how loud/sharp was this hit" signal.
  3. Group onsets by position-mod-3 and position-mod-4 and compare how much
     the loudest group in each grouping stands out from that grouping's
     average. A real triple meter (accent every 3rd beat) will show one
     mod-3 group much louder than the other two; a real quadruple meter
     shows the same effect at mod-4. Whichever grouping shows the larger
     relative standout wins.

This was validated against synthetic click tracks with a programmatically
known accent-every-Nth-beat pattern -- it correctly recovered 3/4 vs 4/4 on
both directions of that test. Real-world accuracy will be lower than on
clean synthetic clicks, especially on quieter or heavily-mixed recordings
where the accenting instrument is not the dominant transient source in the
mix.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:  # noqa: BLE001
    _LIBROSA_AVAILABLE = False


def meter_estimation_available():
    """True if librosa is importable in this environment."""
    return _LIBROSA_AVAILABLE


def estimate_meter(audio_path, sr=22050, margin=1.10, candidate_periods=(3, 4)):
    """Estimate whether a track's accent pattern groups in 3s or 4s.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    sr : int
        Sample rate to load audio at. 22050 (librosa default) is sufficient
        for rhythm analysis -- this is not a pitch-sensitive task.
    margin : float
        How much stronger one grouping's score must be over the other
        before it is reported instead of "" (ambiguous). 1.10 = >10% higher.
    candidate_periods : tuple[int, int]
        The two beat-groupings to compare. Defaults to (3, 4) for the
        3/4-vs-4/4 case this module was built for; can be widened later
        (e.g. adding 6 for 6/8) without changing the calling convention.

    Returns
    -------
    dict with keys:
        time_signature_guess : str   -- "3/4", "4/4", or "" if ambiguous/failed
        confidence : float           -- 0.0-1.0, how separated the two scores were
        scores : dict[str, float]    -- raw grouping score per candidate period
        n_onsets : int                -- onsets detected (0 if it failed early)
        error : str or None
    """
    result = {
        "time_signature_guess": "",
        "confidence": 0.0,
        "scores": {},
        "n_onsets": 0,
        "error": None,
    }

    if not _LIBROSA_AVAILABLE:
        result["error"] = "librosa is not installed"
        return result

    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if y is None or len(y) == 0:
            result["error"] = "empty or unreadable audio"
            return result

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

        label_map = {3: "3/4", 4: "4/4", 6: "6/8"}
        if score_low > score_high * margin:
            result["time_signature_guess"] = label_map.get(p_low, f"{p_low}-grouping")
            result["confidence"] = round(min(1.0, (score_low - score_high) / total), 3)
        elif score_high > score_low * margin:
            result["time_signature_guess"] = label_map.get(p_high, f"{p_high}-grouping")
            result["confidence"] = round(min(1.0, (score_high - score_low) / total), 3)

        return result

    except Exception as e:  # noqa: BLE001 -- must never crash the audit pass
        result["error"] = str(e)
        return result


def integrate_into_sample(sample, min_confidence=0.35):
    """Apply estimate_meter() to a dataset sample dict, following the same
    lock-respecting, non-destructive auto-fill convention used elsewhere in
    the app for bpm/key/time_signature (see HealthAuditorWorker.analyze_track
    and on_file_audited in dataset_manager.py).

    Only writes sample["time_signature"] when:
      - the field is currently empty (never overwrites an existing/manual/
        locked value)
      - the estimate was not ambiguous
      - the estimate's confidence clears ``min_confidence``

    Always records the estimator's own confidence and source alongside any
    value it writes, so the UI/exceptions queue can distinguish a heuristic
    guess from a locked, human-confirmed value.

    Returns (sample, filled) -- ``filled`` mirrors the boolean pattern already
    used by on_file_audited() for bpm/key/time_signature.
    """
    filled = False
    audio_path = sample.get("audio_path", "")
    if not audio_path or sample.get("time_signature"):
        return sample, filled

    result = estimate_meter(audio_path)
    if result["time_signature_guess"] and result["confidence"] >= min_confidence:
        sample["time_signature"] = result["time_signature_guess"]
        sample["time_signature_confidence"] = result["confidence"]
        sample["time_signature_source"] = "heuristic_accent_grouping"
        filled = True

    return sample, filled
