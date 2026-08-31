"""Lead vs. backing vocal separation (experimental).

No dominant open-source model exists for lead/backing separation yet, so this
is a fallback chain:

  1. Prefer a backing-vocal / harmony / lead-vocal model from MVSEP's *live*
     catalog (they add separation models frequently).
  2. Fall back to a clearly-experimental DSP heuristic on the vocal stem.

Both are opt-in via config ``lead_vocal_splitter``: ``off | mvsep | heuristic``.
The split is reviewable output — backing vocals are frequently subtle (or the
engineer doubled the lead on key words), so treat it as a suggestion.
"""
import os

import numpy as np


def find_backing_vocal_model():
    """Return ``(render_id, name)`` for a lead/backing-vocal MVSEP model.

    Returns ``(None, None)`` when no suitable model is in the live catalog.
    """
    try:
        from modules.mvsep_api import get_algorithms

        by_id, _ = get_algorithms()
        for rid, name in by_id.items():
            n = name.lower()
            if ("backing" in n and "vocal" in n) or "harmony" in n or (
                "lead" in n and "vocal" in n
            ):
                return rid, name
    except Exception:  # noqa: BLE001
        pass
    return None, None


def split_lead_backing(vocal_path, out_dir, sr=None):
    """Experimental DSP split of a vocal stem into lead and backing WAVs.

    Lead-salience = per-frame voicing (``librosa.pyin``) weighted by harmonic
    energy; ``lead = vocal * mask``, ``backing = vocal - lead``. Both files
    recombine to the original stem exactly.

    Returns ``(lead_path, backing_path)``.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(vocal_path, sr=sr, mono=False)
    mono = librosa.to_mono(y)

    f0, voiced, _ = librosa.pyin(mono, fmin=80, fmax=1000, sr=sr)
    voiced_prob = np.asarray(voiced if voiced is not None else np.zeros_like(f0), dtype=float)

    harmonic = librosa.effects.harmonic(mono)
    hop = 512
    h_energy = librosa.feature.rms(y=harmonic, hop_length=hop)[0]
    v_energy = librosa.feature.rms(y=mono, hop_length=hop)[0]
    v_energy = np.where(v_energy < 1e-6, 1e-6, v_energy)

    sal = np.nan_to_num(voiced_prob) * np.nan_to_num(h_energy)
    if sal.size:
        sal = sal / (sal.max() + 1e-9)
    # Soft mask: frames with a sustained, prominent voice = lead.
    mask = 1.0 / (1.0 + np.exp(-8.0 * (sal - 0.5)))
    mask = np.repeat(mask, hop)[: len(mono)]
    if len(mask) < len(mono):
        mask = np.pad(mask, (0, len(mono) - len(mask)))

    if y.ndim > 1:
        lead = y * mask[np.newaxis, :]
        backing = y * (1.0 - mask)[np.newaxis, :]
    else:
        lead = y * mask
        backing = y * (1.0 - mask)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(vocal_path))[0]
    lead_path = os.path.join(out_dir, f"{base}_lead.wav")
    backing_path = os.path.join(out_dir, f"{base}_backing.wav")
    if y.ndim > 1:
        sf.write(lead_path, lead.T, sr)
        sf.write(backing_path, backing.T, sr)
    else:
        sf.write(lead_path, lead, sr)
        sf.write(backing_path, backing, sr)
    return lead_path, backing_path