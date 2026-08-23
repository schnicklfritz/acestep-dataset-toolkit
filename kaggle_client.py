#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
import glob

def run_remote_captioning(dataset_json_path, audio_folder_path):
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json) and not os.environ.get("KAGGLE_USERNAME"):
        print("Error: Kaggle API token not found at ~/.kaggle/kaggle.json")
        sys.exit(1)

    print(f"Packaging audio files from: {audio_folder_path}")
    staging_dir = "/tmp/kaggle_acestep_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
        for f in glob.glob(os.path.join(audio_folder_path, ext)):
            shutil.copy(f, staging_dir)
            audio_files.append(os.path.basename(f))

    if not audio_files:
        print("No audio files found in directory.")
        return

    print(f"Uploading {len(audio_files)} tracks to Kaggle private dataset...")
    meta = {
        "title": "acestep-audio-staging",
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'user')}/acestep-audio-staging",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(staging_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    subprocess.run(["kaggle", "datasets", "version", "-p", staging_dir, "-m", "Batch audio", "-d"], check=False)
    print("Dispatched execution to Kaggle Dual T4 GPU Kernel.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./kaggle_client.py <dataset_json_path> <audio_directory>")
        sys.exit(1)
    run_remote_captioning(sys.argv[1], sys.argv[2])
