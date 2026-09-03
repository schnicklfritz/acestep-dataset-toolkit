# ACE-Step Dataset Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-green.svg)](https://pypi.org/project/PySide6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A desktop toolkit for building **high-quality ACE-Step 1.5 / XL / LoRA / LoKR**
training datasets: import audio → **audit & recommend** → **normalize** →
**stem-separate** → **structure-split (verse/chorus/solo…)** → **auto-caption**
→ export. Every stage is **configurable**; the defaults are the best-working
setup, and the app is deliberately **not dependent on any paid service**
(free LLM + Kaggle options included).

Run with:

```bash
python dataset_manager.py
```

---

## 🚀 The workflow (from scratch)

1. **Dataset Studio** → *Add Single Song* (one or more files) or *Add Audio
   Folder* (recursive — every track in a folder and its subfolders, deduped
   against what's already loaded). Tracks land in the table.
2. **🔍 Scan & Fill** — health audit: sample rates, channels, clipping, lossy
   cutoffs, **real BPM/key** (librosa + Krumhansl-Schmuckler), loudness spread,
   **near-duplicate detection**, and a quality score with **actionable
   recommendations** (lossy source → find a lossless master, mono → re-export,
   duplicates → remove, etc.).
3. **🎚 DSP Normalize** — EBU R128 to a target loudness/sample rate (defaults
   -14 LUFS / 44.1 kHz; originals are backed up, Undo/Redo available).
4. **🚀 Run AI Captioner** — pick a backend (below). Each caption is reviewed
   before it's accepted.
5. **Structural / Spatial pipelines** — per-stem, per-section captions:
   **tag → recommend → split → (lead/backing vocals) → segment → caption →
   aggregate** into a hybrid of deterministic tags + a comprehensive paragraph.
6. **Validate & Save JSON** — the ACE-Step manifest.

---

## ✨ Features

### AI Captioning — pluggable backends
| Backend | What it is | Cost |
|---|---|---|
| **ACE-Step captioner** (default) | Qwen2.5-Omni on a **Kaggle GPU** (cached model dataset, configurable batch size) | free (Kaggle) |
| **Gemini** | Google Gemini, audio-native | free tier |
| **DeepSeek** | DeepSeek LLM | paid, cheap |
| **Custom endpoint** | any OpenAI-compatible server (vLLM / Ollama / local / rented GPU) | your own |

The caption prompt, max tokens, max audio duration, and batch size are all
editable in ⚙ Settings → *Pipeline & Model Defaults*.

### Pluggable LLM provider (aggregation, recommendations, assistant)
One OpenAI-compatible client, five providers — pick in ⚙ Settings → *LLM Provider*:

* `deepseek` (default) · `gemini` (free tier) · `groq` (free tier) ·
  `openrouter` (free `:free` models) · `local` (any OpenAI-compatible server).
* Model and base URL auto-fill per provider and are overridable.
* Used by the master-caption aggregator, instrument-model recommendation, and
  the AI assistant — so **no paid dependency**.

### Instrument tagger + content-aware recommendations
* Real **BPM/key** and a spectral instrument estimate (`modules/tagger.py`).
* Optional **CLAP zero-shot tagging** (`laion/larger_clap_music`) for specific
  instrument names (fiddle, pedal steel, upright bass, …) when
  `torch`+`transformers` are installed (`use_clap_tagger: auto`).
* Recommendations are **driven by what's actually in the track** — a Hank
  Williams Sr. track recommends Acoustic Guitar / Violin / Pedal Steel / Double
  Bass, never drums/synth.

### Stem separation — 3 stages, both clouds
1. **Vocal/instrumental** — MVSEP **BS PolarFormer 124-band** (default; it
   re-synthesizes the instrumental, preventing artifacts). Fast on MVSEP —
   PolarFormer is *not* recommended on Kaggle (~10–15 min).
2. **Multi-stem on the instrumental** — MVSEP (BS RoFormer SW / Demucs) *or*
   **Kaggle Demucs** (`htdemucs_ft` / `htdemucs_6s` guitar+piano).
3. **Instrument-specific** — auto-recommended from the tagger.

### Structure segmentation — SongFormer (functional labels)
⚙ Settings → *Structure Backend* → `songformer`: pushes a Kaggle kernel
(`kernels/structure_kernel.py`) that runs ASLP-lab SongFormer and returns
**intro / verse / chorus / bridge / solo / outro** boundaries. Sections get
real names (`Verse_01`, `Solo_01`) that flow into per-stem captioning and the
final aggregation. Falls back to librosa automatically.

### Spatial pipeline — real pan/width (ILD)
The spatial pipeline measures each stem's **pan** (inter-channel level
difference) and **width** (L/R correlation) → tokens like
`guitar: hard left, wide`. Captures classic 70s hard-pan mixes
(Sabbath/Zeppelin) and feeds the LoKR spatial field.

### Lead / backing vocals
⚙ Settings → *Lead/Backing Vocal Split*: `mvsep` (prefers a backing-vocal
model from MVSEP's live catalog) or `heuristic` (experimental DSP). Off by
default; output is reviewable.

### Model manager
⚙ Settings → *🧰 Model Manager*: a curated catalog (`models.json`) of
separation/tagging/segmentation/lyrics models with **download source
(Hugging Face default, or your GitHub repo)**, per-model status, and
**leaderboard links** (MDX, MVSEP, papers-with-code). Downloads are
gitignored. An **HF token** (encrypted, optional) unlocks gated models.

### Health audit → recommendations + near-duplicates
The audit flags tracks that would drag the dataset down — lossy sources,
clipping, mono, LUFS outliers, low-confidence BPM/key, small dataset size,
and **near-duplicate pairs** (librosa fingerprints) — then gives you
**actionable, copyright-safe recommendations** (e.g. "find a lossless
master", "remove the duplicate").

The audit is the same "🔍 Scan Audio & Fill Metadata" action (there is no
separate second audit) — pick **Local (fast)** or **Kaggle GPU** as the audit
backend. It auto-fills and **locks** **duration / key / BPM / time signature**
next to each filename; unlock a row with the 🔓 button to edit those values
inline in the table.

### Dataset Studio tools — bulk rename & non-destructive delete
* **✏️ Bulk Rename** — rename any scope (all / filtered / selected tracks).
  Default mode keeps **only the song name with spaces → `_`**
  (`03 - Artist - Cold Cold Heart.wav` → `Cold_Cold_Heart.wav`); Find &
  Replace, Prefix, Suffix, and Number-sequence modes are also available, with a
  live preview. On-disk renames are optional and **always back up the original
  file first**.
* **🗑 per-track delete** — removes a track from the dataset after confirmation;
  the audio file is **backed up to `project_backups/deleted/`**, never
  destroyed. Every file the toolkit changes or removes is backed up first.

### AI assistant skills
The built-in assistant (DeepSeek or any LLM provider) exposes **tools** it can
call against your dataset: dataset summary, list tracks, instrument lookup,
caption audit, manifest validation, health scan, instrument detection, **sound
profile**, and **curate dataset for a target sound** (picks the genre/artist/
instrument gaps to fill so the dataset converges on a specific sound).

### MCP server
The app can run as a **Model Context Protocol server**
(`python mcp_server.py --dataset path/to/dataset.json`), exposing dataset
summary, health audit, near-duplicate, tagging, and curation tools to any MCP
client (Claude Desktop, Cursor, custom agents). Requires `pip install mcp`.
This is rare in dataset apps — your dataset becomes directly steerable by AI.

## 🔐 Configurability & security

* **"Anything you can't configure is a bug."** Every model, backend, threshold,
  and prompt is an overridable key in ⚙ Settings.
* **Secrets are never stored in `settings.json`.** Each key (Kaggle, DeepSeek,
  Gemini, MVSEP, HF, OpenRouter, Groq) has a **"Remember on this device
  (encrypted)"** toggle — OS keyring, or an encrypted file fallback. Uncheck it
  for a session-only key, and the stored copy is removed.
* **Non-destructive:** originals are never overwritten; normalization and edits
  keep backups with Undo/Redo.

---

## 📦 Installation

```bash
# System deps (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 python3-tk

# Python deps
pip install -r requirements.txt

# Kaggle CLI (for the GPU backends)
pip install kaggle
```

Optional extras:
```bash
pip install google-genai            # Gemini caption backend
pip install whisperx                # lyrics transcription (needs torch + transformers)
pip install "mcp[cli]"              # MCP server
pip install transformers torch      # CLAP zero-shot instrument tagging
```

### Kaggle setup
1. Create a Kaggle account and an **API key** (`kaggle.json`), put
   Username/Key in ⚙ Settings.
2. Upload the captioner weights as a private dataset
   (`kaggle_model_dataset`, default
   `michelmoalem9b/acestep-captioner-model`) so the captioner kernel finds it
   mounted under `/kaggle/input`.
3. For SongFormer, the kernel downloads from Hugging Face
   (`ASLP-lab/SongFormer`) or uses a cached `...songformer...` dataset;
   set `HF_TOKEN` if needed.

---

## 🗂 Project structure

```
dataset_manager.py          Entry point (PySide6 desktop app)
config.py                   Defaults + secret-key policy + settings.json
stem_separator.py           MVSEP 3-stage separation (PolarFormer first stage)
mcp_server.py               MCP server (optional, needs `mcp`)
models.json                 Curated model catalog (sources, backend, notes, leaderboards)
modules/
  tagger.py                 BPM/key + spectral/CLAP instrument tagging, hybrid captions
  recommender.py            Content-aware instrument → model recommendations
  lead_vocals.py            MVSEP backing-vocal lookup + experimental DSP split
  dedup.py                  Near-duplicate detection (librosa fingerprints)
  lyrics.py                 WhisperX lyrics (optional)
  llm_client.py             Pluggable LLM providers (DeepSeek/Gemini/Groq/OpenRouter/local)
  model_manager.py          Catalog downloader (HF / GitHub)
  mcp_tools.py              Headless tools for the MCP server
  kaggle.py                 Private audio-dataset upload + kernel push/wait/download
  mvsep_api.py              MVSEP live algorithm list + separation jobs
  config_store.py           settings.json + encrypted secrets
  secrets_manager.py        OS keyring / Fernet-encrypted secrets
  audio_analysis.py         Structural sections + slicing
  homogeneity.py            LUFS / crest / centroid audit
  caption_audit.py          Instrument-naming consistency across captions
  instruments_db.py         Filename-keyword instrument lookup
  manifest_validation.py    ACE-Step schema validation
kernels/                    Kaggle GPU kernel scripts
  caption_kernel.py         Qwen2.5-Omni captioner (batched)
  stem_separation_kernel.py Meta Demucs (htdemucs_ft / htdemucs_6s)
  structure_kernel.py       SongFormer structure (functional labels)
workers/                    QThread workers (caption backends, pipelines, stems, …)
ui/                         main_window.py (parallel entry point) + mvsep_tab.py
```

---

## 🛠 Troubleshooting

* **"Kaggle credentials not configured"** → ⚙ Settings → Cloud & Execution
  Endpoints → enter Username/Key → *Save Cloud Credentials*.
* **Captioner kernel fails with no weights** → attach the captioner-model
  dataset to the kernel (or set `HF_TOKEN`), and check the kernel logs.
* **CLAP isn't used** → `use_clap_tagger: auto` requires
  `torch` + `transformers`; it degrades to the spectral tagger otherwise.
* **Assistant needs a key** → set any LLM provider key (Gemini free tier
  works); the assistant uses the configured provider.
* **`pip` refuses to install (PEP 668)** → use a virtual environment
  (`python -m venv .venv && source .venv/bin/activate`).

---

## 🌟 Philosophy

* **Total configurability** — if you can't change it, it's a bug.
* **Best-working defaults** — everything works out of the box (Kaggle free /
  free LLM tiers), and scales to rented/owned GPUs.
* **No paid-service dependency** — free LLM providers and free Kaggle compute
  are first-class citizens.
* **Non-destructive** — backups + Undo/Redo everywhere.
