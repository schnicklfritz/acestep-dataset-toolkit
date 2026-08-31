"""Content-aware stem-separation recommendations.

Recommendations are driven by the instruments the tagger actually detected
(spectral + optional CLAP) rather than a static default list — so a
country/folk track (acoustic guitar, fiddle, steel guitar, upright bass)
never gets drums/synth models recommended, while a metal track does.
"""
from modules.tagger import normalize_instrument
from stem_separator import StemSeparator


def mvsep_instrument_map():
    """Canonical instrument keyword -> MVSEP instrument-specific model name."""
    try:
        return StemSeparator({})._instrument_to_model_map()
    except Exception:  # noqa: BLE001
        return {}


def recommend_mvsep_models(instruments):
    """Ordered MVSEP instrument-model recommendations for detected instruments.

    Returns a list of ``{"instrument", "model", "backend"}`` (deduplicated,
    in detection order).
    """
    ivmap = mvsep_instrument_map()
    seen = set()
    out = []
    for inst in instruments or []:
        key = normalize_instrument(inst)
        model = ivmap.get(key)
        if model is None and key.endswith("s"):
            # e.g. "electric guitars" -> "electric guitar"
            model = ivmap.get(key[:-1])
        if model and model not in seen:
            seen.add(model)
            out.append({"instrument": inst, "model": model, "backend": "mvsep"})
    return out


def recommend_all(instruments):
    """All recommendations for the detected instruments (MVSEP + OSS hints).

    For now OSS multi-stem downloads are genre-agnostic, so only the MVSEP
    instrument-specific models are content-driven; the general multi-stem
    recommendation is returned separately via :func:`general_multi_stem`.
    """
    return recommend_mvsep_models(instruments)


def general_multi_stem():
    """The always-safe multi-stem split plan, independent of detected tags."""
    return [
        {"role": "vocal/instrumental", "backend": "mvsep", "model": "BS PolarFormer (124-band)"},
        {"role": "multi-stem", "backend": "mvsep", "model": "BS RoFormer SW (6 stems)"},
        {"role": "multi-stem (OSS)", "backend": "kaggle", "model": "htdemucs_6s"},
    ]