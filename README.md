# ACE-Step Dataset Toolkit (Gentoo Edition)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-green.svg)](https://pypi.org/project/PySide6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, outcome-driven desktop utility designed for preparing, auditing, normalizing, and auto-captioning audio datasets for **ACE-Step 1.5 / XL / LoRA / LoKR** fine-tuning.

## 📁 Project Structure

```
dataset_manager.py        # Entry point — run with: python dataset_manager.py
config.py                 # Default configuration (API keys live in settings.json, gitignored)
stem_separator.py         # MVSEP API stem separation
modules/homogeneity.py    # DSP homogeneity engine (LUFS, crest factor, spectral centroid)
workers/                  # Background QThread workers (one per pipeline stage)
  deepseek.py             #   DeepSeek music-prompt orchestrator client
  advanced.py             #   Advanced AI pipeline orchestrator
  spatial.py              #   Spatial pipeline (stems -> sections -> captions)
  health.py               #   Health audit (metadata, clipping, lossy cutoffs, BPM/key)
  dsp.py                  #   EBU R128 normalization
  caption.py              #   Multi-backend captioning (Kaggle / DeepSeek / custom)
  structural.py           #   Structural pipeline + batch worker
ui/main_window.py         # DatasetManager QMainWindow (tabs, theming, undo/redo)
```

---

## 🌟 Philosophy: User Sovereignty (Gentoo-Style)
* **Zero Hardcoded Restrictions**: If any color, font, zoom level, path, or metadata value cannot be customized or overridden by the user, it is treated as a bug.
* **Non-Destructive Workflows**: Original audio files are never overwritten. Every normalization, bulk edit, or caption update creates automatic snapshots with full **Undo/Redo** and **A/B Fallback** support.
* **Bypass Gate**: A prominent **"I Know What I'm Doing"** mode allows intentional exports regardless of diagnostic warnings.

---

## 📊 Dataset Quality Gauge & Degradation Penalties

The toolkit displays a live **Dataset Quality Percentage** at the top of the interface:
* **🟢 Green (80% – 100%)**: Fully homogeneous, verified sample rates, lossless audio, structured captions.
* **🟡 Yellow (60% – 79%)**: Minor inconsistencies (loudness spread, unverified tempos, missing lyrics).
* **🔴 Red (<60%)**: Critical quality risks (digital clipping, missing files, lossy compression artifacts).

### Penalty Breakdown Reference

| Warning Condition | Quality Penalty | Why it Degrades LoRA Training |
| :--- | :--- | :--- |
| **Missing / Unreadable File** | `-40%` | Dataloader crashes on broken paths. |
| **Digital Clipping (> -0.1 dBFS)** | `-20%` | Flat-top square wave distortion gets permanently baked into model timbre. |
| **Lossy Compression (<192 kbps)** | `-20%` | Model learns to reproduce MP3/AAC artifacts and high-frequency shelf cutoffs. |
| **Loudness Spread (> 5 dB LUFS)** | `-15%` | Causes unstable energy distribution and dynamic pumping during generation. |
| **Dataset Size (< 10 Tracks)** | `-15%` | Insufficient acoustic diversity for generalizable LoRA representation. |
| **Mixed Sample Rates / Channels** | `-10%` | On-the-fly resampling introduces subtle phase artifacts. |

---

## 🛠 Core Workflow & Capabilities
[ ➕ Import Audio Tracks / Folder]
│
▼
[ 🔍 Scan Audio & Fill Metadata] ──► Detects Sample Rate, Duration, Channels,
│ BPM & Key Confidence + Lossy Cutoffs.
▼
[ ⚠ Exceptions Queue Filter] ────► Surfaces ONLY rows requiring human review.
│
▼
[ 🎚 DSP Normalizer (EBU R128)] ───► Unifies to -14 LUFS & 44.1kHz stereo.
│ Archives original audio in backups folder.
▼
[ 🚀 Headless Kaggle Cloud GPU] ───► Compresses temporary 16kHz previews,
│ runs 11B captioner, purges previews.
▼
[ 💾 Validate & Save JSON] ────────► Outputs standard ACE-Step manifest.

---

## 🎨 Customization & Appearance Engine

Switch to the **`🎨 Appearance & Customization`** tab to adjust:
* **System Font Picker**: Automatically queries all installed system fonts with live typeface preview.
* **UI Zoom Scale**: Dynamically adjusts interface scale from **75% to 175%** (ideal for 720p through 4K displays).
* **Theme Presets**: Switch instantly between *Dark Modern*, *OLED Pure Black*, *Gentoo Purple Slate*, *Solarized Dark*, and *High Contrast Light*.

---

## 📋 Quickstart & Installation

### 1. Prerequisites
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Winget)
winget install Gyan.FFmpeg
```

### 2. Python Setup & Launch
```bash
git clone https://github.com/<YOUR-USERNAME>/acestep-dataset-toolkit.git
cd acestep-dataset-toolkit
pip install PySide6
python dataset_manager.py
```

---

## 📄 Manifest Output Format

Outputs standard training manifests:

```json
{
  "metadata": {
    "name": "My_Dataset",
    "custom_tag": "MyTriggerTag",
    "tag_position": "prepend",
    "instrumental_mode": "mixed",
    "num_samples": 14
  },
  "samples": [
    {
      "id": "e03aa4d7",
      "audio_path": "/path/to/normalized_audio/norm_track_01.wav",
      "filename": "norm_track_01.wav",
      "caption": "MyTriggerTag, raw blues rock, Gibson guitar lead, Vox organ solo, gritty dynamic arrangement",
      "genre": "Blues Rock",
      "lyrics": "[Verse 1]\nSample lyric text...\n[Chorus]\nMain hook...",
      "formatted_lyrics": "[Verse 1]\nSample lyric text...\n[Chorus]\nMain hook...",
      "bpm": 120,
      "keyscale": "A minor",
      "timesignature": "4/4",
      "duration": 210,
      "language": "en",
      "is_instrumental": false,
      "custom_tag": "MyTriggerTag",
      "locked": true
    }
  ]
}
```

---

## 📜 License & Credits
* Distributed under the **MIT License**.
* Built for the **ACE-Step 1.5** training ecosystem.
* Gratefully credits the original dataset management concepts created by the open-source community.

