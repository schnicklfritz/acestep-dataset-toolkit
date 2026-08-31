"""Kaggle kernel — music structure analysis with SongFormer (functional labels).

Reads audio from the mounted Kaggle dataset, runs SongFormer (state of the art
for intro/verse/chorus/bridge/solo/outro boundaries), and writes
``/kaggle/working/structure.json``::

    {"results": [{"file": "<name>", "segments": [{"start": 0.0, "end": 15.2, "label": "verse"}, ...]}]}

Model source:
  * a cached Kaggle dataset whose path contains "songformer" (mounted via
    ``dataset_sources``), otherwise
  * downloaded from Hugging Face (``ASLP-lab/SongFormer``) — requires the
    ``HF_TOKEN`` secret for gated access.

Placeholders substituted by the app at push time:
  {{AUDIO_DATASET_PATH}}  -> /kaggle/input/<audio-dataset-name>
  {{MIN_SEGMENT_SEC}}     -> merge segments shorter than this (0 = no merge)
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def _install():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers", "accelerate", "huggingface_hub", "librosa", "soundfile"],
        check=False,
    )


_install()

AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"
MIN_SEGMENT_SEC = {{MIN_SEGMENT_SEC}}
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


def is_valid_json(path):
    try:
        json.load(open(path))
        return True
    except Exception:
        return False


# ---- model source: prefer a cached SongFormer dataset, else HF download ----
MODEL_SOURCE = None
for candidate in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
    d = os.path.dirname(candidate)
    if "songformer" in d.lower() and is_valid_json(candidate):
        MODEL_SOURCE = d
        break

if MODEL_SOURCE is None:
    from huggingface_hub import snapshot_download
    hf = os.environ.get("HF_TOKEN")
    if hf:
        from huggingface_hub import login
        login(token=hf, add_to_git_credential=False)
    MODEL_SOURCE = snapshot_download(
        repo_id="ASLP-lab/SongFormer",
        repo_type="model",
        resume_download=True,
        ignore_patterns=["SongFormer.pt", "SongFormer.safetensors"],
    )

sys.path.append(MODEL_SOURCE)
os.environ["SONGFORMER_LOCAL_DIR"] = MODEL_SOURCE

import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

songformer = AutoModel.from_pretrained(MODEL_SOURCE, trust_remote_code=True,
                                       low_cpu_mem_usage=False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
songformer.to(device)
songformer.eval()

audio_files = sorted(
    p for p in Path(AUDIO_FOLDER).rglob("*")
    if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
)


def merge_short(segments, min_sec):
    """Merge segments shorter than ``min_sec`` into their predecessor."""
    if not min_sec or min_sec <= 0 or len(segments) < 2:
        return segments
    out = []
    for seg in segments:
        short = (seg["end"] - seg["start"]) < min_sec
        if out and short and (out[-1]["end"] - out[-1]["start"]) < min_sec:
            out[-1]["end"] = seg["end"]
        else:
            out.append(dict(seg))
    return out


results = []
for f in audio_files:
    try:
        segs = songformer(str(f))
        segs = merge_short(segs, MIN_SEGMENT_SEC)
        results.append({"file": f.name, "segments": segs})
        print("OK", f.name, [s.get("label") for s in segs])
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        results.append({"file": f.name, "segments": [], "error": str(e)})

with open("/kaggle/working/structure.json", "w") as out_f:
    json.dump({"results": results}, out_f, indent=2)
print("DONE", len(results))