"""ACE-Step audio captioner — Kaggle kernel.

Based on the working ACE-Step captioner notebook. Reads audio from a mounted
Kaggle dataset, runs the Qwen2.5-Omni captioner with the proper chat template,
writes ``/kaggle/working/captions_out.json``::

    {"results": [{"file": "<original filename>", "caption": "..."}, ...]}

Placeholders substituted by the app at push time:
  {{AUDIO_DATASET_PATH}}  -> /kaggle/input/<audio-dataset-name>
  {{CAPTION_PROMPT}}      -> prompt as a JSON string literal
  {{MAX_NEW_TOKENS}}      -> int (Concise Tags ~64, else ~512)
  {{CUSTOM_TAG}}          -> trigger tag as a JSON string literal
"""
import os
import sys
import json
import glob
import tempfile
import subprocess
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def _install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "accelerate", "huggingface_hub", "hf-transfer",
                    "soundfile", "librosa", "numba", "tinytag", "tqdm"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview",
                    "qwen-omni-utils[decord]"],
                   check=False)


_install()

import torch  # noqa: E402
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"
CAPTION_PROMPT = {{CAPTION_PROMPT}}
MAX_NEW_TOKENS = {{MAX_NEW_TOKENS}}
BATCH_SIZE = {{BATCH_SIZE}}
CUSTOM_TAG = {{CUSTOM_TAG}}
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


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

audio_files = sorted(
    p for p in Path(AUDIO_FOLDER).rglob("*")
    if p.suffix.lower() in SUPPORTED_FORMATS and p.is_file()
)


def truncate_audio(audio_path, max_seconds={{MAX_AUDIO_DURATION}}):
    if not max_seconds or max_seconds <= 0:
        return audio_path
    import librosa  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False, duration=max_seconds)
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


results = []
for i in range(0, len(audio_files), BATCH_SIZE):
    batch = audio_files[i:i + BATCH_SIZE]
    try:
        truncated = []
        for f in batch:
            truncated.append(truncate_audio(str(f)))

        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech.")}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": t},
                    {"type": "text", "text": CAPTION_PROMPT},
                ]},
            ]
            for t in truncated
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
            )
        full_texts = processor.batch_decode(
            output_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for f, t, ft in zip(batch, truncated, full_texts):
            caption = extract_reply(ft)
            if CUSTOM_TAG:
                caption = f"{CUSTOM_TAG}, {caption}"
            results.append({"file": f.name, "caption": caption})
            print("OK", f.name, caption[:100])
            if t != str(f) and t.startswith(tempfile.gettempdir()):
                try:
                    os.remove(t)
                except Exception:
                    pass
    except Exception as e:
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        for f in batch:
            results.append({"file": f.name, "caption": f"ERROR: {e}"})

with open("/kaggle/working/captions_out.json", "w") as out_f:
    json.dump({"results": results}, out_f, indent=2)
print("DONE", len(results))

