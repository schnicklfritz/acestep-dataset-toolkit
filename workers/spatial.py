import os, json, uuid, tempfile, subprocess, shutil, time
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from stem_separator import StemSeparator
from modules.tagger import analyze_audio, compose_caption

class SpatialPipelineWorker(QThread):
    progress = Signal(int, str)
    step_completed = Signal(str, dict)
    pipeline_finished = Signal(str, dict)
    error_occurred = Signal(str)

    def __init__(self, track_id, file_path, config, options):
        super().__init__()
        self.track_id = track_id
        self.file_path = file_path
        self.config = config
        self.options = options
        self._is_cancelled = False
        self.tags = {}

    def run(self):
        try:
            # Step 1: Tags (spectral + optional CLAP). Drives recommendations,
            # hybrid captions, and metadata.
            self.progress.emit(5, "Analyzing tags (BPM, key, instruments)...")
            self.tags = analyze_audio(
                self.file_path,
                use_clap=str(self.config.get("use_clap_tagger", "auto")).lower() != "off",
            )
            self.step_completed.emit("tags", self.tags)

            # Step 2: Obtain stems
            source = self.options.get('stem_source')
            if source == 'mvsep':
                self.progress.emit(15, "Calling MVSEP API for stem separation...")
                stem_paths = self._call_mvsep(self.file_path)
            elif source == 'kaggle_demucs':
                self.progress.emit(15, "Running Kaggle Demucs stem separation...")
                stem_paths = self._call_kaggle_stems(self.file_path)
            else:
                self.progress.emit(15, "Looking for imported stems...")
                stem_paths = self._find_imported_stems(self.file_path)
            if not stem_paths:
                self.error_occurred.emit("No stems found.")
                return
            self.step_completed.emit("stems", stem_paths)

            # Step 3: Structural slicing
            self.progress.emit(25, "Slicing full mix into structural sections...")
            sections = self._slice_audio(self.file_path)
            if not sections:
                self.error_occurred.emit("Structural slicing failed.")
                return
            self.step_completed.emit("sections", sections)

            # Step 4: Extract L/R chunks
            self.progress.emit(35, "Extracting L/R channels per section...")
            chunks = self._extract_lr_chunks(stem_paths, sections)
            if not chunks:
                self.error_occurred.emit("L/R chunk extraction failed.")
                return
            self.step_completed.emit("chunks", chunks)

            # Step 5: Caption via Kaggle (or custom endpoint)
            self.progress.emit(45, "Running captioning on L/R chunks via Kaggle...")
            captions = self._caption_chunks(chunks)
            if not captions:
                self.error_occurred.emit("Captioning failed.")
                return
            self.step_completed.emit("captions", captions)

            # Step 5: Spatial evaluator — real pan (ILD) + width (L/R correlation)
            self.progress.emit(70, "Evaluating spatial placement (ILD/correlation)...")
            spatial_tokens = self._evaluate_spatial(chunks)
            self.step_completed.emit("spatial", spatial_tokens)

            # Step 6: DeepSeek aggregation
            if self.options.get('use_deepseek', True):
                self.progress.emit(80, "Aggregating via DeepSeek...")
                final_caption = self._aggregate_with_deepseek(sections, captions, spatial_tokens)
            else:
                final_caption = "Spatial pipeline complete (DeepSeek skipped)."

            self.step_completed.emit("aggregated", {"final_caption": final_caption})

            result = {
                "track_id": self.track_id,
                "final_caption": final_caption,
                "sections": sections,
                "spatial_tokens": spatial_tokens,
                "stem_paths": stem_paths,
                "chunk_paths": chunks,
                "tags": self.tags,
            }
            self.progress.emit(100, "Pipeline complete.")
            self.pipeline_finished.emit(self.track_id, result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _call_mvsep(self, audio_path):
        """
        Use StemSeparator to obtain stems.
        If stem_source is 'import', fall back to imported stems.
        Otherwise, run the full separation pipeline.
        """
        # If the user chose "Import existing stems", just look for them
        if self.options.get('stem_source') == 'import':
            return self._find_imported_stems(audio_path)

        # Otherwise, use the StemSeparator
        separator = StemSeparator(self.config, progress_callback=self.progress.emit)
        # Get stem options from the worker's options (they are set in run_structural_pipeline)
        stem_options = dict(self.options.get('stem_options', {}))
        # Content-aware Stage-3 recommendations from the detected instruments.
        if self.config.get('auto_recommend_models', True):
            if not stem_options.get('instrument_models') and self.tags.get('instruments'):
                from modules.recommender import recommend_mvsep_models
                models = [r["model"] for r in recommend_mvsep_models(self.tags["instruments"])]
                if models:
                    stem_options['instrument_models'] = models
                    self.step_completed.emit("recommendations", {"models": models})
        # Determine which method to use – for Spatial, we always want instrument‑specific
        method = 'polarformer+multi+instrument'
        stems = separator.separate(audio_path, method=method, options=stem_options)
        return stems

    def _call_kaggle_stems(self, audio_path):
        """Separate stems via Kaggle GPU Demucs and map files back to stem types."""
        from workers.kaggle_stems import run_kaggle_stems

        paths = run_kaggle_stems(
            audio_path, self.config,
            progress_cb=lambda p, m: self.progress.emit(p, m),
        )
        stem_paths = {}
        for p in paths:
            stem_type = os.path.basename(p).rsplit(".", 1)[0].split("__")[-1]
            stem_paths[stem_type] = p
        return stem_paths

    def _find_imported_stems(self, audio_path):
        base = Path(audio_path).stem
        dir_path = Path(audio_path).parent
        stems = {}
        for ext in ['.wav', '.flac', '.mp3']:
            for stem_type in ['vocals', 'drums', 'bass', 'other']:
                f = dir_path / f"{base}_{stem_type}{ext}"
                if f.exists():
                    stems[stem_type] = str(f)
        return stems

    def _slice_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        duration = librosa.get_duration(y=y_mono, sr=sr)
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=9)
        bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]
        min_sec = float(self.config.get("segment_min_sec", 12.0))
        filtered = [bound_times[0]]
        for t in bound_times[1:]:
            if t - filtered[-1] >= min_sec:
                filtered.append(t)
        if filtered[-1] < duration:
            filtered[-1] = duration
        sections = []
        for i in range(len(filtered)-1):
            sections.append({
                "name": f"Section_{i+1:02d}",
                "start": filtered[i],
                "end": filtered[i+1]
            })
        return sections

    def _extract_lr_chunks(self, stem_paths, sections):
        chunks = []
        base_dir = Path(self.file_path).parent / "spatial_chunks"
        base_dir.mkdir(exist_ok=True)
        for stem_type, stem_path in stem_paths.items():
            y, sr = librosa.load(stem_path, sr=None, mono=False)
            if y.ndim < 2:
                y = np.stack([y, y], axis=0)
            for sec in sections:
                start_s = int(sec['start'] * sr)
                end_s = int(sec['end'] * sr)
                left_chunk = y[0, start_s:end_s]
                right_chunk = y[1, start_s:end_s]
                base_name = f"{Path(stem_path).stem}_{sec['name']}"
                L_path = base_dir / f"{base_name}_L.wav"
                R_path = base_dir / f"{base_name}_R.wav"
                sf.write(L_path, left_chunk, sr)
                sf.write(R_path, right_chunk, sr)
                chunks.append({
                    "stem_type": stem_type,
                    "section": sec['name'],
                    "L_path": str(L_path),
                    "R_path": str(R_path)
                })
        return chunks

    def _caption_chunks(self, chunks):
        """Caption L/R section chunks via the ACE-Step captioner on a Kaggle GPU.

        Correct flow (mirrors workers/caption.py): upload the chunk WAVs as a
        private Kaggle dataset, attach it (plus the cached model dataset) via
        ``dataset_sources``, and run the shared caption kernel
        (kernels/caption_kernel.py) that the app's notebook is based on.
        Returns dict mapping L_path/R_path -> caption.
        """
        from modules.kaggle import (
            upload_audio_dataset, push_kernel, wait_kernel_done,
            download_kernel_output,
        )
        from pathlib import Path as _Path

        # 1. Stage chunk WAVs into a dataset dir (L and R sides)
        temp_dir = tempfile.mkdtemp(prefix="spatial_caption_")
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        for chunk in chunks:
            for key in ("L_path", "R_path"):
                src = chunk[key]
                shutil.copy2(src, os.path.join(audio_dir, Path(src).name))

        # 2. Upload as a private Kaggle dataset
        self.progress.emit(45, "Uploading L/R section audio to Kaggle...")
        audio_slug = upload_audio_dataset(self.config, audio_dir)
        audio_name = audio_slug.split("/")[-1]

        # 3. Build the kernel from the shared template
        prompt = self.config.get("caption_prompt", (
            "You are a professional music metadata tagger preparing training data for ACE-Step. "
            "Listen carefully to this audio clip and write a detailed description. "
            "Cover: specific instrumentation (name every instrument you hear), "
            "whether vocals are present or confirm instrumental, "
            "recording and production character, mood, and how the clip develops. "
            "Write 3 to 5 sentences. Start with A or An. "
            "Genre, BPM, key, and time signature are handled separately -- do not include them."
        ))
        max_tokens = int(self.config.get("caption_max_tokens", 512))
        max_duration = int(self.config.get("caption_max_audio_duration", 120))
        kernel_template = _Path(__file__).resolve().parent.parent / "kernels" / "caption_kernel.py"
        kernel_script = (
            kernel_template.read_text(encoding="utf-8")
            .replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{audio_name}")
            .replace("{{CAPTION_PROMPT}}", json.dumps(prompt))
            .replace("{{MAX_NEW_TOKENS}}", str(max_tokens))
            .replace("{{MAX_AUDIO_DURATION}}", str(max_duration))
            .replace("{{BATCH_SIZE}}", str(max(1, int(self.config.get("caption_batch_size", 1)))))
            .replace("{{CUSTOM_TAG}}", json.dumps(""))
        )

        kernel_slug = f"spatial-caption-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kernel")
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w", encoding="utf-8") as f:
            f.write(kernel_script)

        user = self.config.get("kaggle_user", "").strip()
        model_slug = self.config.get("kaggle_model_dataset", "michelmoalem9b/acestep-captioner-model")
        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [audio_slug, model_slug],
            "competition_sources": [],
            "kernel_sources": [],
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Push, wait, download
        self.progress.emit(50, "Pushing caption kernel to Kaggle GPU...")
        push_kernel(self.config, kernel_dir, kernel_slug)
        self.progress.emit(60, "Kaggle GPU captioning in progress...")
        wait_kernel_done(self.config, kernel_slug)
        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(self.config, kernel_slug, out_dir)

        json_path = os.path.join(out_dir, "captions_out.json")
        if not os.path.exists(json_path):
            raise RuntimeError("Kaggle job finished without captions_out.json -- check the kernel logs.")

        with open(json_path, "r") as f:
            data = json.load(f)
        caption_map = {}
        for item in data.get("results", []):
            file_name = item.get("file", "")
            for chunk in chunks:
                for key in ("L_path", "R_path"):
                    if Path(chunk[key]).name == file_name:
                        caption_map[chunk[key]] = item.get("caption", "")
                        break
        # Hybrid captions: blend deterministic tags with the LLM prose.
        ratio = float(self.config.get("tag_caption_ratio", 0) or 0)
        if ratio > 0 and self.tags:
            caption_map = {
                p: compose_caption(self.tags, c, ratio) for p, c in caption_map.items()
            }
        shutil.rmtree(temp_dir, ignore_errors=True)
        return caption_map

    def _evaluate_spatial(self, chunks):
        """Estimate per-stem pan (ILD) and width (inter-channel correlation).

        Pan is derived from the inter-channel level difference (ILD):
        ``20*log10(RMS_R / RMS_L)`` in dB. Width comes from the normalized
        cross-correlation of the L/R pair (high correlation = narrow/mono,
        low correlation = wide/stereo-spread). Classic hard-pan mixes
        (Sabbath/Zeppelin-era) show up as strongly left/right stems with low
        L/R correlation. Returns ``{stem_type: "<pos>, <width>"}``.
        """
        per_stem = {}
        for chunk in chunks or []:
            l_path = chunk.get("L_path")
            r_path = chunk.get("R_path")
            stem = chunk.get("stem_type") or "other"
            if not l_path or not r_path:
                continue
            if not os.path.exists(l_path) or not os.path.exists(r_path):
                continue
            try:
                L, sr = librosa.load(l_path, sr=None, mono=True)
                R, _ = librosa.load(r_path, sr=None, mono=True)
                n = min(len(L), len(R))
                if n < sr:  # too short to judge
                    continue
                L, R = L[:n], R[:n]
                rms_l = float(np.sqrt(np.mean(L ** 2)))
                rms_r = float(np.sqrt(np.mean(R ** 2)))
                if rms_l < 1e-7 and rms_r < 1e-7:
                    continue
                ild_db = 20.0 * np.log10((rms_r + 1e-9) / (rms_l + 1e-9))
                if np.std(L) > 0 and np.std(R) > 0:
                    corr = float(np.corrcoef(L, R)[0, 1])
                else:
                    corr = 1.0
                per_stem.setdefault(stem, []).append({"ild": ild_db, "corr": corr})
            except Exception:  # noqa: BLE001
                continue

        spatial_tokens = {}
        for stem, vals in per_stem.items():
            ild = float(np.mean([v["ild"] for v in vals]))
            corr = float(np.mean([v["corr"] for v in vals]))
            if ild > 6.0:
                pos = "hard right"
            elif ild > 2.0:
                pos = "right"
            elif ild < -6.0:
                pos = "hard left"
            elif ild < -2.0:
                pos = "left"
            else:
                pos = "center"
            if corr > 0.8:
                width = "narrow"
            elif corr < 0.4:
                width = "wide"
            else:
                width = "medium"
            spatial_tokens[stem] = f"{pos}, {width}"
        if not spatial_tokens:
            spatial_tokens["stereo"] = "balanced"
        return spatial_tokens

    def _aggregate_with_deepseek(self, sections, captions, spatial_tokens):
        return "Master caption from DeepSeek."

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL HealthAuditorWorker (from your file – keep as is)
# ============================================================================
