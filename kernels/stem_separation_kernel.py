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
import glob
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
model.cuda().eval()

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
    + glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    + glob.glob(os.path.join(INPUT_DIR, "*.flac"))
    + glob.glob(os.path.join(INPUT_DIR, "*.ogg"))
)

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
        sources = apply_model(model, wav[None], device="cuda", shifts=1, split=True)[0]

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
