"""Kaggle kernel — music source separation with Meta Demucs.

This script runs on a Kaggle GPU kernel. Audio files placed in ``./input/``
(the app copies the selected track there before pushing the kernel) are
separated with Demucs (default ``htdemucs_ft`` — the fine-tuned hybrid
transformer, currently the best 4-stem model; ``htdemucs_6s`` adds guitar
and piano stems).

Outputs:
  * ``./stems/<track>/<stem>.wav``  — one WAV per source (vocals/drums/bass/other…)
  * ``./stems_manifest.json``       — ``{track: {stem: relpath}}``

Two placeholders are substituted by the app at push time:
  ``{{MODEL}}``     e.g. ``htdemucs_ft``
  ``{{TWO_STEMS}}`` e.g. ``vocals`` (or empty for full multi-stem output)
"""
import json
import os
import subprocess
import sys


def _install():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "demucs", "soundfile"],
        check=False,
    )


_install()

MODEL = "{{MODEL}}" or "htdemucs_ft"
TWO_STEMS = ("{{TWO_STEMS}}" or "").strip() or None
INPUT_DIR = "{{AUDIO_DATASET_PATH}}" or "input"

import torch  # noqa: E402
import torchaudio  # noqa: E402
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

model = get_model(MODEL)

# --- Device selection -------------------------------------------------------
# Free Kaggle tiers often allocate older GPUs (Tesla P100, sm_60; K80, sm_37)
# that the default torch build may NOT support (torch >= 2.x ships for
# sm_70+). Calling `model.cuda()` / `device="cuda"` on such a box silently
# produces no stems. So pick a genuinely usable device — CUDA only when the
# running torch reports a compatible CUDA device, otherwise CPU.
def _pick_device():
    if torch.cuda.is_available():
        # torch.cuda.is_available() can still be true for an unsupported
        # older GPU; verify the capability is within this torch build's range.
        cap = torch.cuda.get_device_capability(0)
        if cap is not None:
            major, minor = int(cap[0]), int(cap[1])
            # torch caps the computed supported list; treat >=7.0 as usable,
            # matching modern PyTorch builds (sm_70+).
            if (major, minor) >= (7, 0):
                return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


DEVICE_NAME, DEVICE = _pick_device()
print(f"Using device: {DEVICE_NAME}", flush=True)
model = model.to(DEVICE).eval()

# Audio discovery: walk the WHOLE /kaggle/input tree. Kaggle sometimes mounts
# private datasets at /kaggle/input/<slug> and other times nests them under
# /kaggle/input/datasets/<...>; searching the whole tree handles both.
def _walk_input_for_audio(base):
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted(
        os.path.join(r, n)
        for r, _d, fs in os.walk(base)
        for n in fs
        if os.path.splitext(n)[1].lower() in exts
    )


INPUT_ROOT = os.path.dirname(INPUT_DIR) or "/kaggle/input"
files = _walk_input_for_audio(INPUT_ROOT)
if not files:
    print(f"No audio under {INPUT_ROOT}. Input tree:", flush=True)
    for root, _dirs, names in os.walk(INPUT_ROOT):
        print("  DIR:", root, flush=True)
        for n in names:
            print("  FILE:", os.path.join(root, n), flush=True)

os.makedirs("stems", exist_ok=True)
manifest = {}


def _save(track, track_out, name, tensor, sr):
    rel = os.path.join(track, f"{name}.wav")
    torchaudio.save(os.path.join("stems", rel), tensor.cpu(), sr)
    manifest[track][name] = rel


for f in files:
    try:
        wav, sr = torchaudio.load(f)
    except Exception:  # noqa: BLE001 — skip unreadable files
        print("SKIP", f)
        continue

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    ref = wav.mean(0)

    with torch.no_grad():
        sources = apply_model(model, wav[None], device=DEVICE_NAME, shifts=1, split=True)[0]

    track = os.path.splitext(os.path.basename(f))[0]
    track_out = os.path.join("stems", track)
    os.makedirs(track_out, exist_ok=True)
    manifest[track] = {}

    source_names = list(model.sources)

    if TWO_STEMS == "vocals":
        # Two-stem mode: keep the vocals stem, and build 'no_vocals' as the
        # sum of every other source (matches Demucs' --two-stems vocals).
        voc_idx = source_names.index("vocals")
        no_vocals = sum(
            (sources[j] for j, n in enumerate(source_names) if n != "vocals"), 0
        )
        _save(track, track_out, "vocals", sources[voc_idx], sr)
        _save(track, track_out, "no_vocals", no_vocals, sr)
    else:
        for i, name in enumerate(source_names):
            stem = sources[i]
            if stem.shape[0] == 1:
                stem = stem.repeat(2, 1)
            # Re-normalize each stem to the input loudness (Demucs behaviour).
            if ref.abs().max() > 0 and stem.abs().max() > 0:
                stem = stem * (ref.abs().max() / stem.abs().max()) * 0.9
            _save(track, track_out, name, stem, sr)

    print("DONE", track, list(manifest[track]))

with open("stems_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

print("STEMS COMPLETE", json.dumps(manifest))
