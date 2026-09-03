"""Deterministic audio tagging (BPM, key, instrument estimates) + hybrid captions.

The tagger is deliberately dependency-light (librosa + numpy). It replaces the
app's old hardcoded BPM/key placeholders (120 BPM / "A minor") with real
estimates and produces a spectral instrument guess. When the LLM captioner's
instrument list is available it can be merged in via ``llm_instruments``.

Hybrid captions blend a deterministic tag block with the LLM prose caption.
``tag_caption_ratio`` (0-100):
  * 0       -> prose only (default, best-working)
  * 100     -> tag block only
  * between -> tag block + prose
"""
import librosa
import numpy as np

# Krumhansl-Schmuckler major/minor key profiles (pitch-class correlation).
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAX_ANALYSIS_SEC = 90  # analyze a representative window, not the whole file


def _window(y, sr):
    """Return a mono, time-bounded representative window for fast analysis."""
    if y.ndim > 1:
        y = librosa.to_mono(y)
    n = int(sr * MAX_ANALYSIS_SEC)
    if len(y) > n:
        start = (len(y) - n) // 2
        y = y[start : start + n]
    return y


def detect_tempo(y, sr):
    """Return an integer BPM estimate (0 if it cannot be determined)."""
    try:
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr)
        return int(round(float(np.atleast_1d(tempo)[0])))
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        return ""


def detect_timesig(y, sr):
    """Return a time-signature estimate ('3/4', '4/4') or '' when uncertain.

    Beat-tracks the audio, samples the onset envelope at each beat frame, then
    measures how "peaked" the downbeat pattern is when the beats are grouped
    into bars of 3 vs 4. The grouping with the stronger per-position contrast
    wins. Shift-invariant (doesn't need to know which beat is the downbeat).
    Returns '' when the clip is too short or neither grouping shows a clear
    meter — so we never assert a meter we can't hear.
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
        # Use a multiple of 12 beats so both 3- and 4-beat grouping is exact.
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
        # Both the absolute contrast and a clear margin over the loser are
        # required, so we never guess on flat/unclear grooves.
        if max(s3, s4) < 0.08:
            return ""
        if s3 > s4 * 1.15:
            return "3/4"
        if s4 > s3 * 1.15:
            return "4/4"
        return ""
    except Exception:  # noqa: BLE001
        return ""


def detect_instruments(y, sr, llm_instruments=None):
    """Spectral instrument estimate, merged with the LLM instrument list.

    The spectral pass is a *coarse* estimate (bass / drums / harmonic mid /
    vocal-band presence). The optional ``llm_instruments`` list (from the
    captioner) is treated as authoritative and appended when it adds
    something the heuristics did not already cover.
    """
    est = []
    try:
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        h_energy = float(np.mean(harmonic**2))
        p_energy = float(np.mean(percussive**2))
        total = max(1e-9, h_energy + p_energy)

        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        def band(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            return float(np.mean(S[mask] ** 2)) if mask.any() else 0.0

        bass = band(50, 250)
        mid = band(250, 2000)
        high = band(2000, 8000)
        band_total = max(1e-9, bass + mid + high)

        if bass / band_total > 0.35:
            est.append("bass")
        if p_energy / total > 0.45:
            est.append("drums")
        if mid / band_total > 0.3:
            est.append("rhythm instrument")  # guitar / keys / piano family
        if high / band_total > 0.22 and h_energy / total > 0.6:
            est.append("vocals or bright lead")
    except Exception:  # noqa: BLE001
        est = []

    merged = list(est)
    if llm_instruments:
        for name in llm_instruments:
            nm = name.lower()
            if not any(nm in m.lower() or m.lower() in nm for m in merged):
                merged.append(name)
    return merged


def analyze_audio(audio_path, llm_instruments=None, use_clap=True):
    """Analyze an audio file and return a tags dict for captions/metadata.

    Returns ``{"bpm", "key", "timesig", "instruments", "instrumental"}``.
    ``llm_instruments`` is an optional authoritative instrument list (e.g. from
    a captioner); ``use_clap`` enables zero-shot CLAP tagging when available.
    """
    tags = {
        "bpm": 0,
        "key": "",
        "timesig": "",
        "instruments": [],
        "instrumental": False,
    }
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        y_w = _window(y, sr)
        tags["bpm"] = detect_tempo(y_w, sr)
        tags["key"] = detect_key(y_w, sr)
        tags["timesig"] = detect_timesig(y_w, sr)
        source_names = list(llm_instruments) if llm_instruments else []
        if use_clap:
            hits = tag_instruments_clap(audio_path)
            source_names = [h["instrument"] for h in hits] + source_names
        tags["instruments"] = detect_instruments(
            y_w, sr, llm_instruments=source_names or None
        )
        low_inst = " ".join(i.lower() for i in tags["instruments"])
        tags["instrumental"] = not (
            "vocal" in low_inst or "sing" in low_inst or "voice" in low_inst
        )
    except Exception as e:  # noqa: BLE001
        # Never crash the pipeline over tagging; return what we have.
        tags["error"] = str(e)
    return tags


def format_tags(tags):
    """Render a tags dict as a compact, caption-safe tag block."""
    if not tags:
        return ""
    parts = []
    if tags.get("bpm"):
        parts.append(f"{int(tags['bpm'])} BPM")
    if tags.get("key"):
        parts.append(tags["key"])
    if tags.get("timesig"):
        parts.append(tags["timesig"])
    inst = tags.get("instruments") or []
    if inst:
        parts.append("Instruments: " + ", ".join(i for i in inst if i))
    if tags.get("instrumental"):
        parts.append("Instrumental")
    return "; ".join(parts)


def compose_caption(tags, prose, ratio=0):
    """Blend a tag block with LLM prose according to ``tag_caption_ratio``.

    ``ratio`` semantics: 0 -> prose only, 100 -> tags only, otherwise both.
    """
    if ratio is None:
        ratio = 0
    ratio = max(0.0, min(100.0, float(ratio)))
    tag_block = format_tags(tags) if tags else ""
    if ratio >= 100:
        return tag_block or (prose or "")
    if ratio <= 0:
        return prose or ""
    combined = []
    if tag_block:
        combined.append(tag_block)
    if prose:
        combined.append(prose)
    return "\n\n".join(combined)


# ---------------------------------------------------------------------------
# Instrument aliases -> canonical names used by the model catalog
# ---------------------------------------------------------------------------
INSTRUMENT_ALIASES = {
    "fiddle": "violin",
    "upright bass": "double bass",
    "standup bass": "double bass",
    "string bass": "double bass",
    "acoustic bass": "double bass",
    "lap steel": "pedal steel guitar",
    "steel guitar": "pedal steel guitar",
    "pedal steel": "pedal steel guitar",
    "acoustic bass guitar": "bass",
    "bass guitar": "bass",
    "backing vocal": "backing vocals",
    "backing vocals": "backing vocals",
    "harmony vocal": "backing vocals",
    "harmony vocals": "backing vocals",
    "male vocal": "lead vocal",
    "female vocal": "lead vocal",
    "male vocals": "lead vocal",
    "female vocals": "lead vocal",
    "singer": "lead vocal",
    "vocalist": "lead vocal",
    "vocals": "lead vocal",
    "voice": "lead vocal",
    "electric bass": "bass",
    "rhythm guitar": "guitar",
    "lead guitar": "guitar",
    "acoustic steel guitar": "acoustic guitar",
    "classical guitar": "acoustic guitar",
    "drum kit": "drums",
    "keyboard": "piano",
    "synths": "synth",
    "electric piano": "rhodes",
}


def normalize_instrument(name):
    """Map common synonyms/plurals to canonical catalog names."""
    if not name:
        return name
    n = name.lower().strip()
    if n in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[n]
    if n.endswith("s") and n[:-1] in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[n[:-1]]
    if n.endswith("s") and n[:-1] in CANONICAL_INSTRUMENTS:
        return n[:-1]
    return n


# Canonical instrument names (the catalog's own vocabulary), so plurals like
# "guitars" collapse to "guitar" even though "guitar" isn't an alias.
CANONICAL_INSTRUMENTS = frozenset({
    "organ", "harpsichord", "accordion", "vibraphone", "rhodes", "piano",
    "guitar", "acoustic guitar", "electric guitar", "pedal steel guitar",
    "steel guitar", "harp", "mandolin", "banjo", "sitar", "ukulele", "dobro",
    "violin", "viola", "cello", "double bass", "bass", "drums", "synth",
    "saxophone", "flute", "trumpet", "trombone", "clarinet", "harmonica",
    "brass", "woodwind", "percussion", "tambourine", "congas", "marimba",
    "xylophone", "bells", "cowbell",
})


# ---------------------------------------------------------------------------
# CLAP zero-shot instrument tagging (optional — needs torch + transformers)
# ---------------------------------------------------------------------------
CLAP_MODEL_ID = "laion/larger_clap_music"

CLAP_INSTRUMENT_VOCABULARY = [
    "acoustic guitar", "electric guitar", "lead guitar", "rhythm guitar",
    "bass guitar", "double bass", "drums", "drum kit", "percussion",
    "piano", "electric piano", "rhodes", "organ", "hammond organ", "synth",
    "violin", "fiddle", "viola", "cello", "acoustic bass",
    "pedal steel guitar", "steel guitar", "dobro", "banjo", "mandolin", "harp",
    "saxophone", "trumpet", "trombone", "french horn", "tuba",
    "flute", "clarinet", "oboe", "bassoon", "harmonica", "accordion",
    "lead vocal", "male vocals", "female vocals", "backing vocals", "choir",
    "cowbell", "tambourine", "congas", "marimba", "xylophone", "bells",
]

_clap_cache = {"model": None, "processor": None, "device": None}


def clap_available():
    """True when transformers + torch are importable (CLAP can run)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_clap():
    """Lazily load the CLAP model + processor (cached). Returns (m, p, dev)."""
    if _clap_cache["model"] is not None:
        return _clap_cache["model"], _clap_cache["processor"], _clap_cache["device"]
    if not clap_available():
        return None, None, None
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL_ID)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _clap_cache.update(model=model, processor=processor, device=device)
    return model, processor, device


def tag_instruments_clap(audio_path, candidates=None, top_k=6):
    """Zero-shot instrument tagging with LAION-CLAP (music variant).

    Returns a list of ``{"instrument", "score"}`` for the top-k candidates.
    Empty list when CLAP is unavailable or the call fails.
    """
    model, processor, device = _load_clap()
    if model is None or processor is None:
        return []
    import torch

    candidates = list(candidates) if candidates else CLAP_INSTRUMENT_VOCABULARY
    try:
        inputs = processor(
            audios=[audio_path], text=candidates, return_tensors="pt",
            sampling_rate=48000,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits_per_audio  # (n_audio, n_text)
        probs = torch.softmax(logits, dim=-1)[0]
        k = min(top_k, len(candidates))
        top = torch.topk(probs, k=k)
        return [
            {"instrument": candidates[i], "score": float(s)}
            for s, i in zip(top.values.cpu().tolist(), top.indices.cpu().tolist())
        ]
    except Exception:  # noqa: BLE001
        return []
