# ACE-Step Dataset Toolkit (Python / Qt6 Edition)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-green.svg)](https://pypi.org/project/PySide6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, multi-backend desktop utility for preparing, tagging, captioning, and formatting audio datasets for **ACE-Step (1.5 / XL / LoRA / LoKR)** fine-tuning.

---

## 🌟 Overview & Attribution

This project is a standalone Python/Qt6 application inspired by the foundational dataset management concepts originally prototyped in C++.

* **Attribution**: Full credit to the original author for the foundational data structures, workflow concept, and dataset layout.
* **Why the Python/Qt6 Rewrite?**: While native C++ is ideal for real-time digital signal processing, modern neural audio captioners (such as `ACE-Step/acestep-captioner` 11B) are natively developed and maintained within the Python ecosystem (`transformers`, `torchaudio`, `accelerate`). This rewrite provides direct access to cloud GPUs, headless workers, and REST endpoints without C++/LibTorch compilation barriers.

---

## ✨ Key Features

* **Multi-Backend AI Captioning**:
  * **Local Rule Engine (CPU)**: Instant profile-based caption generation without GPU requirements.
  * **Kaggle Cloud GPU (100% Free)**: Pre-configured headless batch worker that runs the 11B captioner using free Kaggle GPU compute (30 hours/week).
  * **Local CUDA (11B)**: Direct PyTorch GPU inference for systems with 16GB+ VRAM.
  * **Custom REST Webhook**: Connect any self-hosted or serverless inference endpoint (RunPod, Vast.ai, Modal, Ollama).
* **Disposable Bandwidth Optimization**:
  * Automatically downsamples audio tracks to temporary 16 kHz mono preview files in system scratch memory before remote upload, reducing payload sizes by up to 95% while keeping captions mapped to original lossless `.wav` files.
* **3-Tier Caption Complexity**:
  * **Concise Tags**: Focused, comma-separated acoustic tags for style adapters.
  * **Standard Paragraph**: Balanced descriptive overview of instruments, timbre, and mix.
  * **Deep Structural Breakdown**: Multi-section analysis mapping acoustic changes across intro, verses, chorus dynamics, and outro.
* **Full Metadata & Lyrics Editor**:
  * Inspect and modify musical key scale, BPM, language, custom trigger tags, and structured lyrics with section markers (`[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`).
* **Responsive Multi-Screen GUI**:
  * Adaptive layout designed to scale fluidly from low-resolution 720p laptop displays to large 4K external monitors.
* **Standard Dataset JSON Export**:
  * Outputs training manifests that plug directly into ACE-Step training pipelines.

---

## 🛠 Prerequisites & Installation

### 1. System Dependencies
The toolkit uses native audio decoders and Tkinter desktop bindings. Install the system libraries for your OS:

* **Arch Linux / CachyOS / Manjaro**:
  ```bash
  sudo pacman -S tk ffmpeg libsndfile
  ```
* **Debian / Ubuntu / Mint**:
  ```bash
  sudo apt-get update && sudo apt-get install -y python3-tk ffmpeg libsndfile1
  ```
* **Fedora / RHEL**:
  ```bash
  sudo dnf install python3-tkinter ffmpeg libsndfile
  ```
* **macOS (Homebrew)**:
  ```bash
  brew install python-tk ffmpeg libsndfile
  ```
* **Windows**:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *(Python on Windows includes Tkinter by default during standard installation.)*

### 2. Python Environment Setup
Clone the repository and install the audio DSP and pipeline dependencies:

```bash
# Clone repository
git clone https://github.com/<YOUR-USERNAME>/acestep-dataset-toolkit.git
cd acestep-dataset-toolkit

# Install required Python packages
pip install -r requirements.txt
```

### 3. Launching the App
Launch the graphical interface or use the headless CLI:

```bash
# Graphical User Interface (Default)
python dataset_manager.py

# Headless CLI - Homogeneity Audit
python dataset_manager.py audit ./raw_audio

# Headless CLI - 2-Pass Loudness Normalization
python dataset_manager.py normalize --raw-dir ./raw_audio --out-dir ./processed_dataset
```

---

## 🚀 AI Backend Configuration

| Backend | Setup Required | Latency / Speed | Cost |
| :--- | :--- | :--- | :--- |
| **Local Rule Engine** | None | Instant (<10ms) | Free |
| **Kaggle Cloud GPU** | Free Kaggle Account & API Key | ~3–4 min (Queue + 11B Batch) | Free (30h/week) |
| **Local ACE-Step** | NVIDIA GPU (16GB+ VRAM) + PyTorch | ~1–3s per song | Free |
| **Custom Webhook** | Endpoint URL & Auth Key | ~2–5s per song | User's compute |

### Setting Up Free Kaggle Cloud GPU:
1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Go to **Settings** $\rightarrow$ **API** $\rightarrow$ **Create New Token** to download `kaggle.json`.
3. In this toolkit, click **`⚙ Configure Endpoints`** and paste your Kaggle Username and API Key.

---

## 📄 Output Schema Example

Saved JSON datasets follow the standard ACE-Step manifest structure:

```json
{
  "metadata": {
    "name": "Custom_Dataset",
    "custom_tag": "YourTriggerTag",
    "tag_position": "prepend",
    "num_samples": 14
  },
  "samples": [
    {
      "id": "e03aa4d7",
      "audio_path": "/path/to/audio.wav",
      "filename": "track_01.wav",
      "caption": "YourTriggerTag, vintage psychedelic rock, Vox organ riff, driving bassline, expressive baritone vocal",
      "genre": "Psychedelic Rock",
      "lyrics": "[Verse 1]\nExample lyrics...\n[Chorus]\nMain hook...",
      "formatted_lyrics": "[Verse 1]\nExample lyrics...\n[Chorus]\nMain hook...",
      "bpm": 124,
      "keyscale": "E minor",
      "timesignature": "4/4",
      "duration": 210,
      "language": "en",
      "custom_tag": "YourTriggerTag"
    }
  ]
}
```

---

## 🤝 Contributing & Roadmap

- [ ] One-click cross-platform desktop installer script (`install.sh` / `install.bat`)
- [ ] Automatic batch BPM and key-scale audio detection
- [ ] Direct one-click dataset push to HuggingFace Datasets Hub

Pull requests and community feedback are welcome!

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
