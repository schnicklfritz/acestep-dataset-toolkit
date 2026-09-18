"""MOSS-Audio captioner — Kaggle kernel.

Runs OpenMOSS MOSS-Audio (8B-Instruct by default) over a mounted Kaggle audio
dataset and writes RAW text for two fields per track:

    /kaggle/working/moss_out.json
    {"results": [{"file": "<filename>", "style": "...", "lyrics": "..."}, ...]}

This kernel deliberately does NOT format captions. It emits raw MOSS output;
the app's existing tag_creator LLM stage turns that into
``[Section - descriptors]`` and the hybrid caption.

Placeholders substituted by the app at push time:
  {{AUDIO_DATASET_PATH}}  -> /kaggle/input/<audio-dataset-name>
  {{MODEL_ID}}            -> Hugging Face repo id, e.g.
                             "OpenMOSS-Team/MOSS-Audio-8B-Instruct"
  {{STYLE_PROMPT}}        -> JSON string literal
  {{LYRICS_PROMPT}}       -> JSON string literal
  {{MAX_NEW_TOKENS}}      -> int
  {{CHUNK_SECONDS}}       -> int, audio window per pass (see AUDIO LIMIT below)
  {{CUSTOM_TAG}}          -> JSON string literal ("" for none)
  {{ATTN_IMPL}}           -> JSON string literal ("" = leave the model default).
                             "sdpa" / "flash_attention_2" / "eager". flash-attn
                             does NOT support Turing (T4, sm_75).

--------------------------------------------------------------------------
WHY THE GITHUB CLONE IS MANDATORY
--------------------------------------------------------------------------
The Hugging Face repo ships ``configuration_moss_audio.py`` and
``processing_moss_audio.py`` but NOT ``modeling_moss_audio.py``, and its
config.json ``auto_map`` has no ``AutoModel`` entry. So
``MossAudioModel.from_pretrained("<hf-repo>")`` cannot resolve the class.
The model class exists only in the GitHub repo under ``src/``.

--------------------------------------------------------------------------
AUDIO LIMIT: ~120 SECONDS PER PASS
--------------------------------------------------------------------------
``audio_tokens_per_second = 12.5`` and the encoder's positional embedding is
fixed at ``max_source_positions = 1500``:

    1500 / 12.5 = 120 s

So one forward pass handles at most ~120 s of audio. Tracks longer than that
must be chunked (done below), or the model silently sees only the head.

--------------------------------------------------------------------------
SILENT-DEFAULT TRAPS (why values are asserted, not assumed)
--------------------------------------------------------------------------
* ``MossAudioProcessor.from_pretrained`` defaults ``enable_time_marker=False``
  while ``__init__`` defaults it to True. Omit it and you lose timestamps.
* Every processor kwarg is read with ``kwargs.pop(..., default)``, so a TYPO
  is silently ignored rather than raising.
* ``from_pretrained`` may accept ``dtype=`` (transformers >= 4.56) or only
  ``torch_dtype=`` (older). ``config.json`` says bfloat16, and a T4 has no
  native bfloat16 -- so an ignored dtype argument means a failed run.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}


def _pip(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages],
                   check=False)


def _install():
    """Install runtime deps.

    transformers is pinned deliberately: MOSS imports
    ``transformers.utils.auto_docstring`` and ``models.qwen3``, and the
    ``dtype=`` kwarg needs >= 4.56. Kaggle's preinstalled transformers is
    usually older, which would make the fp16 request a silent no-op.

    torch and torchaudio are NOT touched -- Kaggle's CUDA build is correct and
    reinstalling them risks breaking the GPU stack.
    """
    _pip("safetensors", "numpy", "soundfile", "tiktoken", "einops",
         "scipy", "tqdm", "accelerate")
    _pip("transformers==4.57.1")


def _clone_moss():
    """Fetch the MOSS-Audio repo (provides the missing model class)."""
    dst = "/kaggle/working/MOSS-Audio"
    if not os.path.isdir(dst):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/OpenMOSS/MOSS-Audio.git", dst],
            check=False,
        )
    return dst


_install()
REPO_DIR = _clone_moss()

# `src/` is a package (it has __init__.py) and uses absolute `from src.x import`
# internally, so the REPO ROOT must be importable -- not src/ itself.
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import shutil  # noqa: E402
import torch  # noqa: E402
import transformers as _transformers  # noqa: E402

# ---------------------------------------------------------------------------
# Environment diagnostics
# ---------------------------------------------------------------------------
# Printed BEFORE the heavy work so a failed run's log explains itself. The _pip
# calls above use check=False (one failing package must not abort the whole
# kernel), which means an install problem would otherwise surface much later as
# a confusing ImportError with no hint about the cause.
print(f"[moss] python       : {sys.version.split()[0]}", flush=True)
print(f"[moss] torch        : {torch.__version__} "
      f"(cuda {torch.version.cuda}, available={torch.cuda.is_available()})",
      flush=True)
if torch.cuda.is_available():
    for _i in range(torch.cuda.device_count()):
        _cap = torch.cuda.get_device_capability(_i)
        _free = torch.cuda.mem_get_info(_i)[0] / 1e9
        print(f"[moss] gpu[{_i}]       : {torch.cuda.get_device_name(_i)} "
              f"sm_{_cap[0]}{_cap[1]} {_free:.1f} GB free", flush=True)
print(f"[moss] transformers : {_transformers.__version__}", flush=True)
print(f"[moss] free disk /  : {shutil.disk_usage('/').free / 1e9:.1f} GB "
      f"(8B weights are ~17 GB)", flush=True)

# Fail HERE with a clear reason instead of at a confusing line later.
try:
    from transformers.utils.auto_docstring import auto_docstring  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        f"transformers is too old for MOSS (found {_transformers.__version__}); "
        f"the pinned install did not take effect. Missing: {exc}"
    )

try:
    from src.modeling_moss_audio import MossAudioModel  # noqa: E402
    from src.processing_moss_audio import MossAudioProcessor  # noqa: E402
    from src.audio_io import load_audio  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"Could not import the MOSS source from {REPO_DIR}. The git clone most "
        f"likely failed -- internet must be enabled for this kernel. "
        f"Missing: {exc}"
    )

AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"
MODEL_ID = {{MODEL_ID}}
STYLE_PROMPT = {{STYLE_PROMPT}}
LYRICS_PROMPT = {{LYRICS_PROMPT}}
MAX_NEW_TOKENS = {{MAX_NEW_TOKENS}}
CHUNK_SECONDS = {{CHUNK_SECONDS}}
CUSTOM_TAG = {{CUSTOM_TAG}}
ATTN_IMPL = {{ATTN_IMPL}}


# ---------------------------------------------------------------------------
# Model weights: prefer a cached Kaggle dataset, else download from HF
# ---------------------------------------------------------------------------
def _is_model_dir(path):
    """A real weights dir has config.json AND at least one weight shard."""
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg) as f:
            json.load(f)
    except (OSError, ValueError):
        return False
    return any(
        n.startswith("model") and n.endswith((".safetensors", ".bin", ".pt"))
        for n in os.listdir(path)
    )


def _find_cached_weights():
    for candidate in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(candidate)
        if _is_model_dir(d):
            return d
    return None


MODEL_SOURCE = _find_cached_weights()
if MODEL_SOURCE is None:
    # No cached copy: pull from Hugging Face. Expects the token in a Kaggle
    # secret / env var. The 8B repo is ~17 GiB, so a cached dataset is far
    # faster on repeat runs.
    MODEL_SOURCE = MODEL_ID
    print(f"[moss] no cached weights found; downloading {MODEL_ID}", flush=True)
    _pip("huggingface_hub", "hf_transfer")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf:
        from huggingface_hub import login  # noqa: PLC0415
        login(token=hf, add_to_git_credential=False)
else:
    print(f"[moss] using cached weights: {MODEL_SOURCE}", flush=True)


# ---------------------------------------------------------------------------
# Load model + processor
# ---------------------------------------------------------------------------
# fp16, not bf16: config.json declares bfloat16, but Kaggle T4s (Turing) have
# no native bfloat16. fp16 is native and also what makes the 8B fit.
#
# attn_implementation is only passed when explicitly requested. MOSS's audio
# encoder config pins "_attn_implementation": "eager" for the Whisper layers and
# injects deepstack features through forward hooks, so forcing a different
# backend is untested. Left empty, the model's own setting wins.
#
# Note flash-attn does NOT support Turing (T4, sm_75): its README sends Turing
# users to a separate fork with only a subset of features. It is only worth
# requesting on an Ampere+ allocation.
_load_kwargs = {"trust_remote_code": True, "device_map": "balanced"}
if ATTN_IMPL:
    _load_kwargs["attn_implementation"] = ATTN_IMPL
try:
    model = MossAudioModel.from_pretrained(
        MODEL_SOURCE, dtype=torch.float16, **_load_kwargs
    )
except TypeError:
    # transformers < 4.56 uses torch_dtype=
    model = MossAudioModel.from_pretrained(
        MODEL_SOURCE, torch_dtype=torch.float16, **_load_kwargs
    )
model.eval()
print(f"[moss] attn_implementation={ATTN_IMPL or '(model default)'}", flush=True)

# enable_time_marker must be EXPLICIT: from_pretrained defaults it to False
# even though __init__ defaults to True. Timestamps are the reason we want
# MOSS for lyrics at all, so verify it actually stuck rather than trusting it.
processor = MossAudioProcessor.from_pretrained(
    MODEL_SOURCE, enable_time_marker=True
)
assert getattr(processor, "enable_time_marker", False) is True, (
    "enable_time_marker did not apply -- timestamps would be missing"
)
MEL_SR = int(processor.config.mel_sr)
print(f"[moss] loaded. mel_sr={MEL_SR} "
      f"audio_token_id={processor.audio_token_id} "
      f"model.dtype={model.dtype}", flush=True)



# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------
def _load_mono(path):
    """Load a file as 1-D float32 mono at MEL_SR.

    Uses MOSS's own loader so the array matches exactly what the processor's
    WhisperFeatureExtractor expects.
    """
    return load_audio(path, sample_rate=MEL_SR)


def _chunk(audio, seconds):
    """Split a 1-D array into non-overlapping windows of ``seconds``."""
    size = int(seconds * MEL_SR)
    if size <= 0 or audio.shape[0] <= size:
        return [(0, audio)]
    return [(i, audio[i:i + size]) for i in range(0, audio.shape[0], size)]


def _mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


# ---------------------------------------------------------------------------
# One generation pass over one audio window
# ---------------------------------------------------------------------------
def _generate(audio, prompt):
    """Run MOSS on a single window. Returns decoded text."""
    inputs = processor(text=prompt, audios=[audio], return_tensors="pt")
    inputs = inputs.to(model.device)

    # audio_data is built as bfloat16 by MelConfig.mel_dtype; cast it to the
    # model's dtype or a T4 will choke on bfloat16.
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)

    inputs["audio_input_mask"] = (
        inputs["input_ids"] == processor.audio_token_id
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,      # deterministic: this is annotation, not art
            num_beams=1,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(
        generated[0, input_len:], skip_special_tokens=True
    ).strip()


def _strip_think(text):
    """Drop any <think>...</think> preamble a Thinking variant may emit."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# ---------------------------------------------------------------------------
# Two passes per track
# ---------------------------------------------------------------------------
# Both passes take the already-loaded audio so a track is decoded and resampled
# ONCE. Loading twice (the original shape of this code) meant two
# torchaudio.load + two resamples per track for no reason.
def style_for_track(path, audio=None):
    """Style pass: the first window is representative of the whole song."""
    if audio is None:
        audio = _load_mono(path)
    if audio.shape[0] == 0:
        return ""
    _, first = _chunk(audio, CHUNK_SECONDS)[0]
    text = _strip_think(_generate(first, STYLE_PROMPT))
    if CUSTOM_TAG:
        text = f"{CUSTOM_TAG}, {text}"
    return text


def lyrics_for_track(path, audio=None):
    """Lyrics pass: transcribe every window in order, prefixing timestamps.

    Non-overlapping windows are deliberate -- overlapping would duplicate
    words across the join, and the model emits its own time markers.
    """
    if audio is None:
        audio = _load_mono(path)
    if audio.shape[0] == 0:
        return ""
    windows = _chunk(audio, CHUNK_SECONDS)
    if len(windows) == 1:
        return _strip_think(_generate(windows[0][1], LYRICS_PROMPT))

    parts = []
    for offset, window in windows:
        text = _strip_think(_generate(window, LYRICS_PROMPT))
        if not text:
            continue
        parts.append(f"[{_mmss(offset / MEL_SR)}]\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Discover audio
# ---------------------------------------------------------------------------
audio_files = sorted(
    str(p) for p in Path(AUDIO_FOLDER).rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
)
print(f"[moss] {len(audio_files)} audio file(s) under {AUDIO_FOLDER}", flush=True)
if not audio_files:
    for root, _dirs, names in os.walk("/kaggle/input"):
        print("  DIR:", root, flush=True)
        for n in names[:20]:
            print("  FILE:", os.path.join(root, n), flush=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
results = []
for index, path in enumerate(audio_files, start=1):
    name = os.path.basename(path)
    print(f"[moss] ({index}/{len(audio_files)}) {name}", flush=True)
    entry = {"file": name, "style": "", "lyrics": ""}
    # Decode + resample ONCE and share it between both passes. Loading inside
    # each pass doubled the audio I/O per track for no benefit.
    audio = None
    try:
        audio = _load_mono(path)
    except Exception as exc:                           # noqa: BLE001
        import traceback                            # noqa: PLC0415
        traceback.print_exc()
        entry["style"] = f"ERROR: {exc}"
        entry["lyrics"] = f"ERROR: {exc}"
        results.append(entry)
        with open("/kaggle/working/moss_out.json", "w", encoding="utf-8") as out:
            json.dump({"results": results}, out, indent=2, ensure_ascii=False)
        continue

    try:
        entry["style"] = style_for_track(path, audio)
        print("   style :", entry["style"][:120], flush=True)
    except Exception as exc:                       # noqa: BLE001
        import traceback                        # noqa: PLC0415
        traceback.print_exc()
        entry["style"] = f"ERROR: {exc}"
    try:
        entry["lyrics"] = lyrics_for_track(path, audio)
        print("   lyrics:", entry["lyrics"][:120].replace("\n", " "), flush=True)
    except Exception as exc:                       # noqa: BLE001
        import traceback                        # noqa: PLC0415
        traceback.print_exc()
        entry["lyrics"] = f"ERROR: {exc}"
    del audio          # free before the next track
    results.append(entry)

    # Persist after every track so a later crash still leaves usable work.
    with open("/kaggle/working/moss_out.json", "w", encoding="utf-8") as out:
        json.dump({"results": results}, out, indent=2, ensure_ascii=False)

print(f"DONE {len(results)} track(s) -> /kaggle/working/moss_out.json", flush=True)

