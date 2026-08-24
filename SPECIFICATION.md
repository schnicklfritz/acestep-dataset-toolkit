# ACE-Step Dataset Toolkit: Specification & Design Document

## 1. System Philosophy & Gentoo-Style Customizability
* **Complete User Sovereignty**: If any field, value, path, color, font, or setting cannot be altered, bypassed, or customized by the user, it is treated as a bug.
* **Non-Destructive Workflows**: Original files are never overwritten. Every edit, scan, and normalization step generates automatic backups with rollback support.
* **Outcome-Oriented**: Automate tedious, error-prone manual labor while providing explicit overrides and bypass controls.

---

## 2. Definitions: Exceptions Queue & Quality Score

### What is the "Exceptions Queue"?
The **Exceptions Queue** is a smart view/filter on the main dataset grid that isolates tracks requiring human attention. Instead of manually auditing dozens of rows:
* **All Tracks View**: Displays the complete dataset table.
* **Exceptions Queue Filter**: Displays only tracks with unresolved warnings (missing files, clipping, loudness spikes, lossy cutoffs, missing captions, or low-confidence BPM/keys).
* Resolving an issue removes the track from the queue until all items pass.

### Dataset Quality Metric & Health Gauge
Displayed at the top of the window with live color coding:
* **Green (80% – 100%)**: Optimal homogeneity, valid metadata, lossless audio, structured captions.
* **Yellow (60% – 79%)**: Minor inconsistencies (loudness spread, unverified tempos, missing lyrics).
* **Red (Below 60%)**: Critical degradation risks (digital clipping, missing files, mixed lossy rips).
* **Estimated Degradation Penalty**: Each warning displays an impact penalty (e.g., `-15% Quality: Low-bitrate 128 kbps source introduces compression artifacts into LoRA weights`).
* **Bypass Gate**: A prominent **"I Know What I'm Doing" (Bypass All)** button allows intentional export despite warnings.

---

## 3. UI Layout & Customization Engine

### Main Interface Layout
1. **Global Header & Health Strip**:
   * Live Quality Score Gauge (`0% - 100%`) with color progression (Red $\rightarrow$ Yellow $\rightarrow$ Green).
   * Dataset Name & Trigger Tag inputs.
   * "I Know What I'm Doing" (Bypass Warnings) button.
   * Global Instrumental Mode: `[ All Instrumental ]` / `[ No Instrumentals ]` / `[ Mixed Dataset ]`.
2. **Top Action Bar**:
   * `📂 Open JSON`, `💾 Save JSON`, `➕ Add Audio`, `⚙ Settings & Endpoints`, `↩ Undo`, `↪ Redo`.
3. **Exceptions Queue & View Bar**:
   * View Toggle: `[ All Tracks (N) ]` | `[ Exceptions Queue (N) ]`.
   * Quick Actions: `[ 🔍 Scan & Fill ]` | `[ 🎚 DSP Normalize ]` | `[ 🚀 Run AI Captioner ]`.
4. **Central Workspace (Split View)**:
   * **Left (60%)**: Responsive Dataset Matrix with per-track status badges, locked/unlocked state, and quick-playback compare buttons.
   * **Right (40%)**: Scrollable Track Inspector with individual lock checkboxes for every field, lyrics editor, and diagnostic report.
5. **Appearance & Customization Tab (Gentoo Philosophy)**:
   * **Theme & Colors**: Fully customizable background, surface, text, border, and accent colors.
   * **Typography**: System font selector matching installed OS fonts with real-time preview.
   * **UI Zoom**: Dynamic scale factor slider (75% to 200%) to accommodate 720p monitors through 4K displays.

---

## 4. Metadata Detection, Provenance & Locking System

### Detection Capabilities
* **Authoritative Audio Container Fields** (100% Certainty):
  * Duration, Sample Rate, Channel Count, Bit Depth, Audio Codec, File Size.
* **Probabilistic Musical Metrics** (With Confidence % & Verification Links):
  * **BPM / Tempo**: Computed with confidence percentage; flags potential half/double-time ambiguities; includes direct lookup links (e.g., SongBPM, Tunebat) for manual verification.
  * **Key Scale**: Chromatic profile estimation with major/minor confidence score.
  * **Time Signature**: Downbeat distribution estimation (default `4/4` with confidence metric).
  * **Loudness & Dynamics**: Integrated LUFS (EBU R128) and Peak dBFS clipping detection.
  * **Spectral Lossy Cutoff**: Detects high-frequency shelf roll-offs typical of lossy MP3/AAC web rips.

### Field Locking & Provenance
* Every metadata attribute maintains an independent lock toggle:
  * `[x] Lock BPM`, `[x] Lock Key`, `[x] Lock Duration`, `[x] Lock Genre`, `[x] Lock Language`.
* Top-level dropdown: `[ Lock All Detected ]`, `[ Unlock All ]`, `[ Restore Detected Values ]`.
* Changes trigger an automatic backup snapshot in the project history stack.

---

## 5. DSP Audio Normalization & Backup Pipeline

### Pre-processing & Bandwidth Optimization
* Remote operations (Kaggle, Webhooks) automatically stage lightweight 16 kHz mono preview files in `/tmp/ace_stage_...` to minimize upload bandwidth.
* Previews are purged automatically upon completion; captions link directly back to the active lossless audio tracks.

### Normalization Pipeline (Historical & Archival Support)
* Designed for both modern recordings and archival restoration (e.g., 1940s–1950s Hank Williams mono masters).
* **Two-Pass EBU R128 Normalization**: Unifies target loudness (default `-14.0 LUFS`, true-peak `-1.0 dBTP`).
* **Format Unification**: Converts sources to standardized 44.1 kHz 16-bit stereo/mono WAV.
* **Archival Backup Management**:
  * Original un-normalized files are moved to `project_backups/originals/`.
  * The dataset manifest seamlessly points to the normalized files.
  * Includes an **"A/B Compare to Original"** toggle and a **"Revert to Original"** restore button.
  * Option to compress `project_backups/` into a `.zip` archive for space management.

---

## 6. AI Captioning & Lyrics Engine

### Multi-Backend Support
1. **Kaggle Cloud GPU (Pre-configured)**: Headless batch runner using free 30h/week compute.
2. **Local ACE-Step (CUDA)**: Direct execution for local GPUs (16GB+ VRAM).
3. **Custom REST Endpoint**: RunPod, Modal Labs, Vast.ai, or self-hosted Ollama/vLLM servers.
4. **Local Profile Rule Engine**: Instant CPU-based template generation.

### Caption Architectures
* **Hybrid Structure (Recommended)**: Comma-separated acoustic/instrumentation tags followed by a structural arrangement narrative.
* **Concise Tags**: Short comma-separated tags for style/character adapters.
* **Deep Structural Breakdown**: Multi-section chronological progression (Intro $\rightarrow$ Verse $\rightarrow$ Chorus $\rightarrow$ Bridge $\rightarrow$ Outro).
* **Overwrite Safety**: Prompts the user before modifying existing captions, with automatic backup snapshots.

---

## 7. Quality Degradation Penalty Reference Table

| Diagnostic Warning | Quality Impact | Degradation Description |
| :--- | :--- | :--- |
| **Missing / Unreadable File** | `-40%` | Training dataloader will crash on missing path. |
| **Severe Digital Clipping** | `-20%` | Square-wave distortion will be learned as acoustic timbre. |
| **Lossy Spectral Cutoff (<16kHz)** | `-15%` | Model reproduces MP3 compression artifacts in generated audio. |
| **Loudness Spread > 5 dB** | `-15%` | Causes unstable energy distribution and volume pumping. |
| **Mismatched Sample Rates** | `-10%` | Dataloader resampling introduces phase inconsistencies. |
| **Unverified / Low Confidence BPM** | `-5%` | Weakens tempo conditioning precision. |
| **Dataset Size < 10 Tracks** | `-15%` | Insufficient diversity for generalizable LoRA representation. |
