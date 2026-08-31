"""Instrument detection + DeepSeek model recommendation.

Flow used by the Structural Pipeline's "Detect via Captioner" button:

1. The audio captioner runs first with an *instruments-only* prompt
   (:data:`workers.caption.INSTRUMENT_ONLY_PROMPT`) and returns the instrument list.
2. :class:`InstrumentRecommendThread` sends that list to DeepSeek, which picks
   instrument-specific MVSEP models **only from the live MVSEP catalog**.
3. The recommended model names are shown in the UI and used by the pipeline.
"""
import re

from PySide6.QtCore import QThread, Signal

from modules.mvsep_api import get_algorithms
from workers.deepseek import DeepSeekMusicOrchestrator


def available_instrument_models():
    """Return the live MVSEP algorithm names (the catalog DeepSeek may pick from)."""
    try:
        by_id, _ = get_algorithms()
        return list(by_id.values())
    except Exception:  # noqa: BLE001 — offline fallback
        return []


def parse_instruments(text):
    """Split a captioner / DeepSeek response into a clean instrument/model list."""
    if not text:
        return []
    items = [i.strip().strip('.-*"') for i in re.split(r"[,;\n]+", text)]
    return [i for i in items if i]


class InstrumentRecommendThread(QThread):
    """Ask DeepSeek which instrument-specific MVSEP models to run."""

    finished_ok = Signal(list)   # recommended MVSEP model names
    failed = Signal(str)

    def __init__(self, config, instruments_text, available=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.instruments_text = instruments_text
        self.available = available if available is not None else available_instrument_models()

    def run(self):
        try:
            api_key = self.config.get("custom_key", "").strip()
            if not api_key:
                raise ValueError("DeepSeek API key missing — add it in ⚙ Settings.")
            orch = DeepSeekMusicOrchestrator(api_key=api_key)
            raw = orch.recommend_instrument_models(self.instruments_text, self.available)
            recommended = parse_instruments(raw)
            # Keep only names that actually exist in the live MVSEP catalog.
            avail_lower = [n.lower() for n in self.available]
            recommended = [r for r in recommended if r.lower() in avail_lower]
            if not recommended:
                raise ValueError(
                    "DeepSeek returned no usable instrument-specific models."
                )
            self.finished_ok.emit(recommended)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


def recommend_from_tagger(audio_path, config):
    """Deterministic instrument detection + model recommendations (no API key).

    Runs the spectral tagger (plus CLAP when available) on the audio and maps
    the detected instruments to MVSEP instrument-specific models. Returns
    ``(recommended_model_names, detected_instruments)``.
    """
    from modules.tagger import analyze_audio
    from modules.recommender import recommend_mvsep_models

    tags = analyze_audio(
        audio_path,
        use_clap=str(config.get("use_clap_tagger", "auto")).lower() != "off",
    )
    instruments = tags.get("instruments") or []
    models = [r["model"] for r in recommend_mvsep_models(instruments)]
    return models, instruments


class TaggerRecommendThread(QThread):
    """Recommend instrument-specific models from the tagger (deterministic)."""

    finished_ok = Signal(list, list)   # recommended models, detected instruments
    failed = Signal(str)

    def __init__(self, config, audio_path, parent=None):
        super().__init__(parent)
        self.config = config
        self.audio_path = audio_path

    def run(self):
        try:
            models, instruments = recommend_from_tagger(self.audio_path, self.config)
            if not models:
                raise ValueError(
                    "No instrument-specific models matched the detected instruments "
                    "instruments: %s" % (instruments or [])
                )
            self.finished_ok.emit(models, instruments)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
