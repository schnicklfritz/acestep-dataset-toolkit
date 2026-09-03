# This script runs remotely on Kaggle inside the background GPU kernel.
# It automatically detects both T4 GPUs and splits the audio processing workload in parallel.

import os
import sys
import glob
import json
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Force Kaggle environment initialization routines
def _install_deps():
    import subprocess
    print("Compiling remote GPU library hooks...")
    # Fast quiet installations for audio and transcription dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "pyloudnorm", "soundfile", "numpy", "librosa"], check=False)
    # Install optimized WhisperX framework for rapid GPU decoding pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                    "git+https://github.com"], check=False)

_install_deps()

# Now safe to import audio modules on remote instance
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa

# Define target paths mapped by your app's zip package manager
AUDIO_INPUT_DIR = "/kaggle/input/ace-step-upload-bundle"
OUTPUT_MANIFEST_PATH = "/kaggle/working/consolidated_manifest.json"

def process_audio_file_on_gpu(file_path, gpu_id):
    """
    Core Worker Process: Executes on a dedicated target device (GPU 0 or GPU 1).
    Extracts Homogeneity properties, runs Librosa sweeps, and generates WhisperX text.
    """
    # Force this specific process thread to bind cleanly to its assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:0" # Maps natively to the isolated visible slot
    
    fname = os.path.basename(file_path)
    report = {"filename": fname, "status": "Healthy", "issues": []}
    
    try:
        # 1. RUN HOMOGENEITY SWEEPS (Soundfile + EBU R128)
        data, sr = sf.read(file_path, always_2d=True)
        duration = len(data) / sr
        
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(data)
        
        peak_val = np.max(np.abs(data))
        peak_db = 20.0 * np.log10(peak_val) if peak_val > 0 else -100.0
        
        # 2. RUN LIBROSA ACOUSTIC EXTRACTION (BPM & Key)
        # Load a snappy 60-second clip to keep processing lightning-fast
        y, sr_lib = librosa.load(file_path, sr=None, mono=True, duration=60)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr_lib)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr_lib)
        bpm_detected = int(round(float(np.atleast_1d(tempo)[0])))
        
        # 3. RUN BATCH WHISPERX TRANSCRIPTION
        import whisperx
        # Load lightning-fast model setup directly into this GPU's VRAM allocation space
        model = whisperx.load_model("small", device, compute_type="float16")
        audio = whisperx.load_audio(file_path)
        
        transcription_result = model.transcribe(audio, batch_size=16)
        compiled_lyrics = " ".join([seg["text"] for seg in transcription_result["segments"]])
        
        # Compile properties back to update main window mapping keys cleanly
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
        report["issues"].append(f"Remote calculation failure: {str(e)}")
        
    return report

def main():
    # Detect available GPU resources inside the Kaggle notebook node
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Verified active remote accelerators: {num_gpus} GPU(s) active.")

    audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    all_tracks = sorted([
        str(p) for p in Path(AUDIO_INPUT_DIR).rglob("*")
        if p.suffix.lower() in audio_extensions and p.is_file()
    ])

    if not all_tracks:
        print("Error: No audio source elements discovered inside upload bundle.")
        return

    results = {}
    
    # 🦾 THE DUAL-GPU HARDWARE ENGINE SPLITTER
    # Distribute tracking files alternating between GPU 0 and GPU 1
    worker_tasks = []
    with ProcessPoolExecutor(max_workers=max(1, num_gpus)) as executor:
        for idx, track_path in enumerate(all_tracks):
            # Alternates target allocation: 0, 1, 0, 1, etc.
            assigned_gpu = idx % num_gpus if num_gpus > 0 else 0
            
            future = executor.submit(process_audio_file_on_gpu, track_path, assigned_gpu)
            worker_tasks.append((track_path, future))
            
        # Collect processing payloads as they finish running asynchronously
        for track_path, future in worker_tasks:
            track_report = future.result()
            results[track_report["filename"]] = track_report

    # Output master dataset manifest record to local disk path
    with open(OUTPUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
        
    print(f"Consolidated pipeline run successfully finished! Mapped {len(results)} assets.")

if __name__ == "__main__":
    main()
