"""Kaggle kernel script template for GPU lyrics transcription.

This file is never run directly on your machine — ``workers/kaggle_lyrics.py``
reads it, string-replaces the {{...}} placeholders below, and pushes the
result to a private Kaggle GPU kernel. Keep the placeholder names in sync
with ``run_kaggle_lyrics()``.

Audio-only (no video/moviepy path) since the app only ever hands this
kernel .wav/.flac/.mp3/.ogg/.m4a files.

Placeholders:
    {{MODE}}                whisperx | acestep
    {{LANGUAGE}}             ISO language code, e.g. "en"
    {{INITIAL_PROMPT}}       optional biasing prompt (may be empty)
    {{AUDIO_DATASET_PATH}}   /kaggle/input/<slug> — the uploaded audio dataset
"""
import json
import os
import subprocess
import sys

MODE = "{{MODE}}"
LANGUAGE = "{{LANGUAGE}}"
INITIAL_PROMPT = "{{INITIAL_PROMPT}}"
AUDIO_DATASET_PATH = "{{AUDIO_DATASET_PATH}}"

OUTPUT_PATH = "/kaggle/working/lyrics_output.json"


def _sh(cmd):
    print(f"LOG: running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _find_audio_file():
    for name in os.listdir(AUDIO_DATASET_PATH):
        if name.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
            return os.path.join(AUDIO_DATASET_PATH, name)
    raise FileNotFoundError(f"No audio file found in {AUDIO_DATASET_PATH}")


def transcribe_whisperx(audio_file):
    print("LOG: Installing WhisperX (PyTorch/CUDA already provided by the Kaggle image)...")
    _sh("apt-get install -y ffmpeg")
    # Pin to the official upstream repo directly — matches the confirmed
    # working install command from the reference notebook.
    _sh(f"{sys.executable} -m pip install -q git+https://github.com/m-bain/whisperx.git")

    import torch
    import whisperx
    from whisperx.utils import get_writer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"LOG: Loading WhisperX large-v3 on {device} ({compute_type})...")
    model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type,
        asr_options={"initial_prompt": INITIAL_PROMPT} if INITIAL_PROMPT else None,
    )

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16, language=LANGUAGE or None)

    # Word-level alignment for cleaner line breaks (skips diarization —
    # not needed for solo song lyrics, and it avoids a Hugging Face token
    # dependency).
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=result.get("language", LANGUAGE or "en"), device=device,
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception as e:  # noqa: BLE001 — alignment is best-effort
        print(f"LOG: word-level alignment skipped ({e}); using raw segments.")

    # Use WhisperX's own writer for the plain-text dump (matches how the
    # reference notebook exports), then also build our own joined string
    # for the unified JSON output the app expects.
    txt_writer = get_writer("txt", "/kaggle/working")
    txt_writer(result, audio_file, {"max_line_width": None, "max_line_count": None, "highlight_words": False})

    lines = [seg["text"].strip() for seg in result.get("segments", []) if seg.get("text", "").strip()]
    return "\n".join(lines)


def transcribe_acestep(audio_file):
    """ACE-Step/acestep-transcriber — a Qwen2.5-Omni fine-tune specialized
    for singing-voice transcription with musical structure annotation
    (verse/chorus/bridge tags), unlike WhisperX which is speech-first.

    Loaded via `transformers` like any Qwen2.5-Omni model — there is no
    separate pip package for this; it's a Hugging Face model checkpoint.
    """
    print("LOG: Installing transformers + audio deps for ACE-Step Transcriber...")
    _sh("apt-get install -y ffmpeg")
    _sh(f"{sys.executable} -m pip install -q "
        "\"transformers>=4.52.0\" accelerate soundfile librosa qwen-omni-utils")

    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    model_id = "ACE-Step/acestep-transcriber"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"LOG: Loading {model_id} on {device} ({dtype})...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    prompt = "*Task* Transcribe this audio in detail"
    if INITIAL_PROMPT:
        prompt += f"\nContext: {INITIAL_PROMPT}"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_file},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True,
    ).to(device)

    print("LOG: Generating transcription...")
    output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    output_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

    # Extract just the "# Lyrics" section body per the model's documented
    # output format; fall back to the raw output if the tag is missing.
    if "# Lyrics" in output_text:
        lyrics = output_text.split("# Lyrics", 1)[1].strip()
    else:
        lyrics = output_text
    return lyrics


def main():
    audio_file = _find_audio_file()
    print(f"LOG: transcribing {audio_file} with engine={MODE}")

    if MODE == "acestep":
        lyrics = transcribe_acestep(audio_file)
    else:
        lyrics = transcribe_whisperx(audio_file)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"lyrics": lyrics}, f)
    print("LOG: Processing completed successfully.")


if __name__ == "__main__":
    main()
