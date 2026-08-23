# ACE-Step LoRA Dataset Studio (Python / Qt6 Edition)

A flexible dataset management and auto-captioning suite designed for preparing, labeling, and structuring audio datasets for **ACE-Step 1.5 LoRA** fine-tuning.

---

## 🌟 Overview & Attribution

This project is a Python / Qt6 fork and modern rewrite of the original C++ dataset concept. 

* **Original Idea & Architecture**: Full credit to the original author for the foundational workflow and C++ data management concept.
* **Why the Python Conversion?**: The original design was reimagined in Python/PySide6 to seamlessly bridge the gap between dataset preparation and the modern AI ecosystem (`transformers`, `PyTorch`, `HuggingFace Hub`, `Kaggle API`, `Modal`, `RunPod`).

---

## 🚀 Why This Python / Qt6 Fork?

While native C++ excels at real-time DSP audio plugins and low-latency audio playback, modern audio-language models (such as `ACE-Step/acestep-captioner` 11B) are natively implemented and updated within the Python machine learning stack. 

This conversion provides key advantages:
1. **Direct AI Integration**: Zero C++/LibTorch compilation friction; works directly with upstream PyTorch and HuggingFace models.
2. **Generic Dataset Architecture**: Completely decoupled from hardcoded templates or sample-specific presets. It operates as a blank-canvas studio for any genre, artist, or style dataset.
3. **Multi-Backend Cloud & Local AI**: Supports **Local Rule Engine**, **Kaggle Cloud (Free 30h/week GPU)**, **Local CUDA (11B)**, and **Universal REST Webhooks** (RunPod, Vast.ai, Modal, Ollama).
4. **Bandwidth-Optimized Staging**: Automatically encodes disposable lightweight previews (16 kHz mono) in temporary scratch storage to accelerate cloud uploads while linking generated captions back to original lossless `.wav` files.
5. **Multi-Monitor Responsive GUI**: Uses responsive layouts and scroll containers, dynamically scaling down to 720p external monitors or expanding to 4K displays.

---

## 🛠 Features

* **Multi-Format Ingestion**: Load `.wav`, `.flac`, and `.mp3` tracks into a unified dataset matrix.
* **Granular Complexity Control**:
  * **Concise Tags**: Short comma-separated acoustic descriptors.
  * **Standard Paragraph**: Coherent overview of instruments and dynamics.
  * **Deep Structural Breakdown**: Multi-section analysis (intro, verses, chorus dynamics, solo instruments, outro).
* **Metadata & Lyric Synchronization**: Edit BPM, musical key scale, structural section markers (`[Verse]`, `[Chorus]`, `[Solo]`), and custom trigger tags (`custom_tag`).
* **Clean In-App Credential Management**: Configure Kaggle or Custom REST Endpoints directly in the UI without modifying config files.
* **One-Click JSON Dataset Export**: Outputs structured training JSON schemas matching the official ACE-Step LoRA specifications.

---

## 📋 Prerequisites & Installation

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/your-username/acestep-dataset-studio.git
cd acestep-dataset-studio

# 2. Install dependencies
pip install PySide6

# Optional: Install ffmpeg on your system PATH for fast audio compression:
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: winget install Gyan.FFmpeg
```

### Running the Application
```bash
python dataset_manager.py
```

*(Note: An automated cross-platform installation script will be included in an upcoming release once initial testing is complete.)*

---

## ⚙️ AI Engine Backends

| Backend | Requirements | Speed / Latency | Cost |
| :--- | :--- | :--- | :--- |
| **Local Rule Engine** | None (CPU native) | Instant (<10ms) | Free |
| **Kaggle Cloud GPU** | Free Kaggle Account & API Key | ~3–4 min (Queue + 11B Inference) | Free (30h/wk) |
| **Local ACE-Step** | NVIDIA GPU (16GB+ VRAM) + PyTorch | ~1–3s per track | Free |
| **Custom Webhook** | RunPod / Modal / Vast / Local Server | ~2–5s per track | User's compute |

---

## 📄 License & Acknowledgments

* Designed for the **ACE-Step 1.5** training ecosystem.
* Gratefully builds upon the original C++ dataset tool concepts created by the community.
