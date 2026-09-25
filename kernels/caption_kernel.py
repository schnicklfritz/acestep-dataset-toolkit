"""ACE-Step audio captioner — Kaggle kernel.

Reads audio from a mounted Kaggle dataset, runs the Qwen2.5-Omni captioner with
the proper chat template, writes ``/kaggle/working/captions_out.json``::

    {"results": [{"file": "<staged filename>", "caption": "..."}, ...]}

The dataset is a private Kaggle dataset the app uploaded from the user's local
STAGING folder (``modules/caption_kaggle_run.py``). Re-runs push a new VERSION of
that same dataset, so songs can be added or removed without the dataset losing
its identity.

Nothing is truncated on the way out: the full caption is printed for every track
and every caption is written to the JSON. (An earlier version printed only the
first 100 characters of each caption, which made a good caption look cut off in
the log and hid where a bad one started to go wrong.) The length control is ``MAX_AUDIO_DURATION`` — one PASS over the audio, in
seconds (0 = whole file in a single pass) — which is a GPU-memory limit, not a
reporting one. With ``WHOLE_SONG`` (the default) a track longer than one pass is
covered by passes that together span the WHOLE file, and their captions are merged
into a single one by a text-only pass: the schema asks how the track develops from
beginning to end, which no single pass can answer.

Placeholders substituted by the app at push time:
  {{AUDIO_DATASET_PATH}}  -> /kaggle/input/<audio-dataset-name>
  {{CAPTION_PROMPT}}      -> the user turn, as a JSON string literal
  {{SYSTEM_PROMPT}}       -> the ACE-Step 1.5XL annotation schema (system turn),
                             as a JSON string literal. Appended to the model's
                             own identity line, never substituted for it.
  {{MAX_NEW_TOKENS}}      -> int (Concise Tags ~64, else ~512)
  {{MAX_AUDIO_DURATION}}  -> int seconds per pass, 0 = whole file in one pass
  {{WHOLE_SONG}}          -> bool: cover the whole file in several passes
  {{BATCH_SIZE}}          -> int (single-pass mode only)
  {{CUSTOM_TAG}}          -> trigger tag as a JSON string literal
  {{REPETITION_PENALTY}}  -> float, 1.0 = off
  {{NO_REPEAT_NGRAM}}     -> int, 0 = off
"""
import os
import sys
import json
import glob
import tempfile
import time
import subprocess
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def _preflight():
    """Print the environment BEFORE anything slow or silent happens.

    This kernel used to be silent for its first ~5 minutes (pip install, then a
    multi-GB model download), so "nothing is happening" and "it is working" looked
    identical from the outside. Worse, a failed pip install was swallowed by
    check=False -- which is exactly how an Internet-off session fails with NO
    error at all.
    """
    print(f"[caption] python       : {sys.version.split()[0]}", flush=True)
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True).stdout
        gpus = [ln for ln in out.splitlines() if ln.strip()]
        print(f"[caption] gpu          : {len(gpus)} device(s)", flush=True)
        for line in gpus:
            print(f"[caption]   {line.strip()}", flush=True)
        if not gpus:
            print("[caption] !! NO GPU -- Session options -> Accelerator = GPU T4 x2",
                  flush=True)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[caption] gpu          : nvidia-smi unavailable ({exc})", flush=True)

    import socket  # noqa: PLC0415
    try:
        socket.create_connection(("huggingface.co", 443), timeout=8).close()
        print("[caption] internet     : reachable", flush=True)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[caption] internet     : NOT REACHABLE ({exc})", flush=True)
        print("[caption] !! Turn Internet ON in Session options, or pip AND the "
              "model download both fail silently.", flush=True)

    try:
        stat = os.statvfs("/kaggle/working")
        print(f"[caption] free disk    : "
              f"{stat.f_bavail * stat.f_frsize / 1e9:.0f} GB", flush=True)
    except Exception:                                            # noqa: BLE001
        pass


def _install():
    """pip install, REPORTING failures instead of swallowing them."""
    steps = (
        ["accelerate", "huggingface_hub", "hf-transfer", "soundfile",
         "librosa", "numba", "tinytag", "tqdm"],
        ["git+https://github.com/huggingface/transformers"
         "@v4.51.3-Qwen2.5-Omni-preview", "qwen-omni-utils[decord]"],
    )
    for index, packages in enumerate(steps):
        label = "base packages" if index == 0 else "transformers fork + qwen-omni-utils"
        print(f"[caption] pip install  : {label} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages], check=False
        )
        if proc.returncode != 0:
            print(f"[caption] !! pip FAILED (rc={proc.returncode}) for {label} -- "
                  f"is Internet ON in Session options?", flush=True)
        else:
            print(f"[caption] pip ok       : {label}", flush=True)


_preflight()
_install()

import torch  # noqa: E402
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

import transformers as _transformers  # noqa: E402

print(f"[caption] torch        : {torch.__version__} "
      f"(cuda {torch.version.cuda}, available={torch.cuda.is_available()})",
      flush=True)
print(f"[caption] transformers : {_transformers.__version__}", flush=True)

AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"
CAPTION_PROMPT = {{CAPTION_PROMPT}}
SYSTEM_PROMPT = {{SYSTEM_PROMPT}}
MAX_NEW_TOKENS = {{MAX_NEW_TOKENS}}
BATCH_SIZE = {{BATCH_SIZE}}
CUSTOM_TAG = {{CUSTOM_TAG}}
REPETITION_PENALTY = {{REPETITION_PENALTY}}
NO_REPEAT_NGRAM = {{NO_REPEAT_NGRAM}}
# Caption the WHOLE file: a track longer than one pass is covered by several
# passes and their captions are merged into one. OFF = a single pass over the
# first MAX_AUDIO_DURATION seconds, i.e. the rest of the song is discarded.
WHOLE_SONG = {{WHOLE_SONG}}
# The pass length as a REAL NAME. The app substitutes a LITERAL for
# {{MAX_AUDIO_DURATION}}, so it is a value wherever it appears -- referencing
# MAX_AUDIO_DURATION without this binding is a NameError that surfaces only on
# Kaggle, minutes in, which is exactly how the first whole-song run died at its
# first track. 0 = the whole file in one pass.
MAX_AUDIO_DURATION = {{MAX_AUDIO_DURATION}}

# The model's own identity line, kept verbatim (set in stone). The ACE-Step
# annotation SCHEMA in SYSTEM_PROMPT is APPENDED to it, never substituted for it.
QWEN_IDENTITY = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

print("[caption] limits       : max_new_tokens=" + str(MAX_NEW_TOKENS)
      + " max_audio_s=" + str(MAX_AUDIO_DURATION)
      + " whole_song=" + str(WHOLE_SONG)
      + " batch=" + str(BATCH_SIZE)
      + " rep_penalty=" + str(REPETITION_PENALTY)
      + " no_repeat_ngram=" + str(NO_REPEAT_NGRAM), flush=True)
print("[caption] schema       : "
      + ("ON (ACE-Step 1.5XL system prompt)" if SYSTEM_PROMPT else "OFF"), flush=True)
print(f"[caption] audio folder : {AUDIO_FOLDER}", flush=True)


def is_valid_json(path):
    try:
        json.load(open(path))
        return True
    except Exception:
        return False


# ---- Model source: prefer a cached weights dataset under /kaggle/input ----
MODEL_SOURCE = None
for candidate in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
    d = os.path.dirname(candidate)
    if is_valid_json(candidate) and any(
        f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(d)
    ):
        MODEL_SOURCE = d
        break

if MODEL_SOURCE is None:
    MODEL_SOURCE = "ACE-Step/acestep-captioner"
    hf = os.environ.get("HF_TOKEN")
    if hf:
        from huggingface_hub import login
        login(token=hf, add_to_git_credential=False)

print(f"[caption] model source : {MODEL_SOURCE}", flush=True)
print("[caption] loading model -- download + shard load, this is the slow part",
      flush=True)
_load_started = time.time()

torch_dtype = torch.float16
load_kwargs = {
    "device_map": "balanced",
    "max_memory": {0: "10GiB", 1: "10GiB"},
    "offload_folder": "/kaggle/working/offload",
    "trust_remote_code": True,
    "torch_dtype": torch_dtype,
}
try:
    import flash_attn  # noqa: F401
    load_kwargs["attn_implementation"] = "flash_attention_2"
except ImportError:
    load_kwargs["attn_implementation"] = "sdpa"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_SOURCE, **load_kwargs)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_SOURCE, trust_remote_code=True)
print(f"[caption] model loaded : {time.time() - _load_started:.0f}s", flush=True)

def _walk_for_audio(base):
    """Every supported audio file under ``base``, recursively.

    Kaggle mounts private datasets inconsistently: sometimes at
    ``/kaggle/input/<slug>``, other times nested under
    ``/kaggle/input/datasets/<owner>/<slug>``. AUDIO_FOLDER names the expected
    location, but the mount is what it is -- so when that folder turns up
    nothing, the whole input tree is searched instead of guessing at a path.

    This is why every app run of this kernel wrote a perfectly valid-looking
    ``{"results": []}``: the model loaded, the folder was searched, and the audio
    was simply not where the path said it was. Mirrors
    kernels/moss_caption_kernel.py, which credits
    kernels/stem_separation_kernel.py for finding this first.

    Returns ``Path`` objects, NOT strings: every caller below uses the Path API
    (``batch[0].name`` for the progress line, ``f.name`` as the key written into
    captions_out.json). The MOSS kernel's twin returns STRINGS because its caller
    uses ``os.path.basename`` -- copying that function without its caller's
    contract crashed this kernel once, four minutes into a real run, right after
    the model had loaded and the mount fallback had correctly found the audio.
    """
    return sorted(
        Path(os.path.join(root, name))
        for root, _dirs, names in os.walk(base)
        for name in names
        if os.path.splitext(name)[1].lower() in SUPPORTED_FORMATS
    )


audio_files = _walk_for_audio(AUDIO_FOLDER)
searched = AUDIO_FOLDER
if not audio_files and os.path.isdir("/kaggle/input"):
    audio_files = _walk_for_audio("/kaggle/input")
    if audio_files:
        searched = "/kaggle/input"
        print(f"[caption] {AUDIO_FOLDER} was empty; found audio elsewhere under "
              "/kaggle/input instead.", flush=True)

# Say which folder actually supplied the files. The count line used to claim the
# expected folder even when the fallback had found them somewhere else, so the log
# contradicted itself one line apart.
print(f"[caption] audio files  : {len(audio_files)} (searched {searched})", flush=True)
if not audio_files:
    # FAIL LOUDLY. A zero-track result surfaced in the app as "no captions came
    # back" instead of "there was no audio to read", which is what sent this
    # chasing the model, the prompt and the upload for days.
    print("[caption] NO AUDIO FOUND. Input tree:", flush=True)
    for root, _dirs, names in os.walk("/kaggle/input"):
        print("  DIR:", root, flush=True)
        for name in names[:50]:
            print("  FILE:", os.path.join(root, name), flush=True)
    raise SystemExit(
        f"No supported audio found under {AUDIO_FOLDER} or /kaggle/input. "
        f"Supported extensions: {sorted(SUPPORTED_FORMATS)}"
    )
elif len(audio_files) == 1:
    print("[caption] !! only ONE file found -- if you expected more, the dataset "
          "mount is wrong.", flush=True)


def _duration_seconds(path):
    """Length of an audio file in seconds (0.0 when it cannot be read)."""
    try:
        import soundfile as sf  # noqa: PLC0415

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:                                        # noqa: BLE001
        return 0.0


def _windows_for(path, pass_sec):
    """``[(offset, length), ...]`` covering the WHOLE file, or ``[(0, 0)]``.

    ``(0, 0)`` means "one pass over the whole file", which is what this kernel
    always did. A pass is capped by GPU memory (~120 s on two T4s), so a full song
    is captioned in several passes whose captions are then merged -- the same shape
    ``kernels/moss_caption_kernel.py`` uses (``for offset, window in windows``) and
    for the same reason: a caption has to describe the whole track, not its first
    two minutes. The final window is clamped to the real end, and a sub-second tail
    is folded into the window before it rather than becoming its own pass.
    """
    if not pass_sec or pass_sec <= 0:
        return [(0, 0)]
    total = _duration_seconds(path)
    if not total or total <= pass_sec:
        return [(0, 0)]
    spans, offset = [], 0.0
    while offset < total - 0.5:
        spans.append((offset, min(float(pass_sec), total - offset)))
        offset += pass_sec
    return spans


def truncate_audio(audio_path, max_seconds=MAX_AUDIO_DURATION, offset=0.0):
    """One PASS of ``audio_path``: ``max_seconds`` starting at ``offset``.

    ``0`` / ``None`` for ``max_seconds`` means the whole file from ``offset``, and
    when there is nothing to cut the ORIGINAL path is returned so the common case
    copies no audio at all. Otherwise the pass is written to a fresh temp file,
    because the processor reads the path it is handed.
    """
    if not max_seconds or max_seconds <= 0:
        if not offset:
            return audio_path
        max_seconds = None
    import librosa  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False,
                             offset=offset, duration=max_seconds)
        if not len(y):
            return audio_path
        ext = os.path.splitext(audio_path)[1]
        fd, tmp = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        if y.ndim > 1:
            sf.write(tmp, y.T, sr)
        else:
            sf.write(tmp, y, sr)
        return tmp
    except Exception:
        return audio_path


def extract_reply(text):
    if "assistant\\n" in text:
        return text.split("assistant\\n")[-1].strip()
    if "assistant" in text:
        return text.split("assistant")[-1].strip()
    return text.strip()


def _caption_audio_clips(clip_paths):
    """One generated caption per clip path, in order — a single model call."""
    conversations = [
        [
            {"role": "system", "content": [{"type": "text", "text": (
                QWEN_IDENTITY + "\n\n" + SYSTEM_PROMPT)}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": t},
                {"type": "text", "text": CAPTION_PROMPT},
            ]},
        ]
        for t in clip_paths
    ]
    text_input = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(
        text=text_input, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    ).to(model.device).to(model.dtype)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, use_audio_in_video=False, return_audio=False,
            max_new_tokens=MAX_NEW_TOKENS,
            # Greedy decoding with no penalty is what let a caption loop on a
            # repeated lyric phrase ~200 times until the token cap.
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
    full_texts = processor.batch_decode(
        output_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [extract_reply(t) for t in full_texts]


def _pass_batches(clips):
    """One forward pass PER clip: never batch the windows of a single track.

    WHY THIS EXISTS: sending the two 120 s windows of one song through the model in
    a single pass asked the audio tower for 32.61 GiB on a 14.56 GiB T4 (its
    attention cost is quadratic in the audio tokens, so two clips is far more than
    twice one) and killed the run with an OOM. One clip per pass is the profile that
    has always worked here: the single-pass kernel captioned 34 tracks at 120 s each
    on the same two T4s.

    It is a separate function, rather than a comment, so that a future "let us batch
    this for speed" change has to break a TEST to reintroduce the OOM.
    """
    return [[clip] for clip in clips]


def _merge_part_captions(name, spans, parts):
    """Merge per-pass captions into ONE caption in the same schema — text only.

    WHY A SECOND CALL: the schema demands how the track develops "from beginning
    to end", and no single pass can see that. The merge is a TEXT conversation with
    the SAME system prompt, so the result is schema-shaped by construction and
    costs seconds — the audio is not re-read.
    """
    listing = "\n".join(
        f"Part {i + 1} (from {int(offset)}s): {text}"
        for i, ((offset, _length), text) in enumerate(zip(spans, parts))
    )
    instruction = (
        f"These are descriptions of {len(parts)} consecutive parts of ONE song "
        f"({name}). Merge them into a SINGLE caption for the whole track, in the "
        "exact schema above: the front-loaded comma-separated keyword list first, "
        "then 2-3 sentences describing how the track develops from beginning to "
        "end. Do not mention that the song was split into parts, do not repeat the "
        "keyword list, and output ONLY the merged caption text.\n\n" + listing
    )
    conversation = [[
        {"role": "system", "content": [{"type": "text", "text": (
            QWEN_IDENTITY + "\n\n" + SYSTEM_PROMPT)}]},
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]]
    text_input = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=text_input, return_tensors="pt").to(
        model.device).to(model.dtype)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, return_audio=False,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
    return extract_reply(processor.batch_decode(
        output_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0])


results = []
if WHOLE_SONG:
    # ONE track at a time, several passes each, then a merge. Batching is
    # deliberately off in this mode: the passes already fill the GPU, and batching
    # multi-pass tracks is exactly how a T4 runs out of memory.
    for index, path in enumerate(audio_files, start=1):
        spans = _windows_for(path, MAX_AUDIO_DURATION)
        print(f"[caption] ({index}/{len(audio_files)}) {path.name} — "
              f"{len(spans)} pass(es)", flush=True)
        temps = []
        try:
            clips = []
            for pass_no, (offset, length) in enumerate(spans, start=1):
                if len(spans) > 1:
                    print(f"[caption]   pass {pass_no}/{len(spans)} "
                          f"[{int(offset)}-{int(offset + length)}s]", flush=True)
                clip = truncate_audio(str(path), length, offset=offset)
                if clip != str(path):
                    temps.append(clip)
                clips.append(clip)

            # ONE clip per forward pass — see _pass_batches for the 32.61 GiB OOM
            # that batching the windows caused.
            parts = []
            for pass_no, batch_clips in enumerate(_pass_batches(clips), start=1):
                parts.append(_caption_audio_clips(batch_clips)[0])
                # Release the finished pass's activations before the next one. The
                # model is spread over two 14.56 GiB cards, so a stale cache is what
                # turns a tight-but-fine pass into an OOM.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[caption]   pass {pass_no}/{len(clips)} done "
                      f"({len(parts[-1])} chars)", flush=True)
            if len(parts) == 1:
                caption = parts[0]
            else:
                try:
                    caption = _merge_part_captions(path.name, spans, parts)
                except Exception:                            # noqa: BLE001
                    import traceback                    # noqa: PLC0415
                    traceback.print_exc()
                    caption = ""
                if not caption.strip():
                    # Never lose a track to a failed merge: the first pass alone is
                    # a valid caption for the opening of the song.
                    caption = parts[0]
            if CUSTOM_TAG:
                caption = f"{CUSTOM_TAG}, {caption}"
            results.append({"file": path.name, "caption": caption})
            print("OK", path.name, caption, flush=True)
        except Exception as e:
            import traceback                                # noqa: PLC0415
            traceback.print_exc()
            results.append({"file": path.name, "caption": f"ERROR: {e}"})
        finally:
            for t in temps:
                if t.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(t)
                    except Exception:
                        pass
else:
    for i in range(0, len(audio_files), BATCH_SIZE):
        batch = audio_files[i:i + BATCH_SIZE]
        print(f"[caption] captioning {i + 1}/{len(audio_files)}: {batch[0].name} "
              f"(~30-120s, no output until the first one finishes)", flush=True)
        try:
            truncated = []
            for f in batch:
                truncated.append(truncate_audio(str(f)))

            full_texts = _caption_audio_clips(truncated)
            for f, t, ft in zip(batch, truncated, full_texts):
                caption = ft
                if CUSTOM_TAG:
                    caption = f"{CUSTOM_TAG}, {caption}"
                results.append({"file": f.name, "caption": caption})
                # FULL caption, never a character-limited preview: the preview is
                # the only place a caption can be checked from the log, and
                # silently cutting it off is how a truncated-looking caption gets
                # blamed on the model.
                print("OK", f.name, caption, flush=True)
                if t != str(f) and t.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(t)
                    except Exception:
                        pass
        except Exception as e:
            import traceback                                # noqa: PLC0415
            traceback.print_exc()
            for f in batch:
                results.append({"file": f.name, "caption": f"ERROR: {e}"})

with open("/kaggle/working/captions_out.json", "w") as out_f:
    json.dump({"results": results}, out_f, indent=2)
print("DONE", len(results))

