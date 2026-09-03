# core/kaggle_worker.py
import os
import sys
import glob
import json
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def prepare_kaggle_runtime():
    """Natively compiles dependencies inside the cloud container shell."""
    import subprocess
    print("📦 Building container system hooks...")
    # Fast quiet installations for audio and transcription dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "pyloudnorm", "soundfile", "numpy", "librosa"], check=False)
    # Install optimized WhisperX framework for rapid parallel GPU decoding passes
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "git+https://github.com"], check=False)

try:
    _installed
except NameError:
    prepare_kaggle_runtime()
    _installed = True

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import whisperx

# Strict Kaggle Container Layout Environment Mapping
INPUT_BUNDLE_DIR = "/kaggle/input/ace-step-upload-bundle"
OUTPUT_MANIFEST_PATH = "/kaggle/working/audit_results.json"

def process_track_on_device(file_path, gpu_id):
    """Worker Process isolated directly inside an explicit CUDA visibility slot."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" 
    
    fname = os.path.basename(file_path)
    report = {"filename": fname, "status": "Healthy", "issues": []}
    
    try:
        # 1. TECHNICAL HOMOGENEITY SWEEPS (Soundfile + EBU R128)
        data, sr = sf.read(file_path, always_2d=True)
        duration = len(data) / sr
        
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(data)
        peak_val = np.max(np.abs(data))
        peak_db = 20.0 * np.log10(peak_val) if peak_val > 0 else -100.0
        
        # 2. LIGHTWEIGHT ACOUSTIC RECOVERY (BPM)
        y, sr_lib = librosa.load(file_path, sr=None, mono=True, duration=45)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr_lib)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr_lib)
        bpm_detected = int(round(float(np.atleast_1d(tempo)[0])))
        
        # 3. ACCELERATED WHISPERX TRANSCRIPTION
        model = whisperx.load_model("small", device, compute_type="float16")
        audio = whisperx.load_audio(file_path)
        transcription_result = model.transcribe(audio, batch_size=16)
        compiled_lyrics = " ".join([seg["text"] for seg in transcription_result["segments"]])
        
        report.update({
            "sample_rate": sr,
            "channels": data.shape[1],
            "duration": round(duration, 2),
            "lufs": round(float(lufs), 2),
            "is_clipping": peak_db >= -0.1,
            "bpm_detected": bpm_detected,
            "lyrics_extracted": compiled_lyrics.strip()
        })
        
    except Exception as e:
        report["status"] = "Warning"
        report["issues"].append(f"Container calculation failure: {str(e)}")
        
    return report

def run_pipeline():
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"🐳 Container verified active accelerators: {num_gpus} GPU(s).")

    audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    all_tracks = sorted([
        str(p) for p in Path(INPUT_BUNDLE_DIR).rglob("*")
        if p.suffix.lower() in audio_extensions and p.is_file()
    ])

    if not all_tracks:
        print("Error: No audio target files discovered in mounted container path.")
        return

    results = {}
    worker_tasks = []
    
    # 🦾 DUAL-GPU HARDWARE SCHEDULER
    with ProcessPoolExecutor(max_workers=max(1, num_gpus)) as executor:
        for idx, track_path in enumerate(all_tracks):
            assigned_gpu = idx % num_gpus if num_gpus > 0 else 0
            future = executor.submit(process_track_on_device, track_path, assigned_gpu)
            worker_tasks.append(future)
            
        for future in worker_tasks:
            track_report = future.result()
            results[track_report["filename"]] = track_report

    with open(OUTPUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Successfully compiled dataset metrics manifest for {len(results)} items.")

if __name__ == "__main__":
    run_pipeline()
