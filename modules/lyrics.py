"""Word-aligned lyrics transcription via WhisperX (optional).

WhisperX needs ``torch`` + ``transformers`` + ``faster-whisper`` (install:
``pip install whisperx``). This is a guarded module — if it isn't installed,
:func:`transcribe` raises a clear, actionable message.
"""
import os


def transcribe_available():
    try:
        import whisperx  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def transcribe(audio_path, model_size="small", language=None, device=None):
    """Transcribe lyrics with WhisperX.

    Returns ``{"lyrics", "segments"}`` where ``segments`` is the word-aligned
    WhisperX segment list (start/end/text per segment).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    if not transcribe_available():
        raise RuntimeError(
            "WhisperX is not installed. Install it with:  pip install whisperx  "
            "(requires torch + transformers)."
        )
    import torch
    import whisperx

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device.startswith("cuda") else "float32"
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language=language)

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language"), device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception:  # noqa: BLE001 — alignment is best-effort
        pass

    segments = result.get("segments", [])
    lyrics = "\n".join((seg.get("text") or "").strip() for seg in segments).strip()
    return {"lyrics": lyrics, "segments": segments}