import os, json, uuid, tempfile, subprocess, shutil, time, struct
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI
from PySide6.QtCore import QThread, Signal
from stem_separator import StemSeparator
from workers.deepseek import DeepSeekMusicOrchestrator
from modules.tagger import analyze_audio, compose_caption

class StructuralPipelineWorker(QThread):
    progress = Signal(int, str)
    step_completed = Signal(str, dict)
    pipeline_finished = Signal(str, dict)
    error_occurred = Signal(str)

    def __init__(self, track_id, file_path, config, options):
        super().__init__()
        self.track_id = track_id
        self.file_path = file_path
        self.config = config
        self.options = options  # stem_source, use_deepseek, use_lyrics
        self._is_cancelled = False
        self.tags = {}

    def run(self):
        try:
            # Step 1: Tags (spectral + optional CLAP). Drives recommendations,
            # hybrid captions, and per-sample metadata.
            self.progress.emit(5, "Analyzing tags (BPM, key, instruments)...")
            self.tags = analyze_audio(
                self.file_path,
                use_clap=str(self.config.get("use_clap_tagger", "auto")).lower() != "off",
            )
            self.step_completed.emit("tags", self.tags)

            # Step 2: Obtain stems (auto-recommended instrument models from tags)
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

            # Step 2b: Optional lead/backing vocal split on the vocal stem
            self._split_lead_backing(stem_paths)

            # Step 3: Structural boundaries – SongFormer (functional labels) or librosa
            self.progress.emit(25, "Finding structural boundaries...")
            sections = self._find_sections(self.file_path)
            if not sections:
                self.error_occurred.emit("Could not determine structural boundaries.")
                return
            self.step_completed.emit("sections", sections)

            # Step 4: Extract stem sections (no L/R split)
            self.progress.emit(35, "Extracting stem sections...")
            chunks = self._extract_stem_sections(stem_paths, sections)
            if not chunks:
                self.error_occurred.emit("Failed to extract stem sections.")
                return
            self.step_completed.emit("chunks", chunks)

            # Step 5: Real captioning via Kaggle (Qwen2.5‑Omni)
            self.progress.emit(45, "Captioning stem sections via Kaggle (Qwen2.5‑Omni)...")
            captions = self._caption_chunks(chunks)
            if not captions:
                self.error_occurred.emit("Captioning failed.")
                return
            self.step_completed.emit("captions", captions)

            # Step 6: DeepSeek aggregation with enriched context
            if self.options.get('use_deepseek', True):
                self.progress.emit(80, "Aggregating via DeepSeek with evidence...")
                final_caption = self._aggregate_with_deepseek(sections, captions)
            else:
                final_caption = "Structural pipeline complete (DeepSeek skipped)."

            self.step_completed.emit("aggregated", {"final_caption": final_caption})

            result = {
                "track_id": self.track_id,
                "final_caption": final_caption,
                "sections": sections,
                "stem_paths": stem_paths,
                "chunk_paths": chunks,
                "tags": self.tags,
            }
            self.progress.emit(100, "Pipeline complete.")
            self.pipeline_finished.emit(self.track_id, result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    # -----------------------------------------------------------------------
    # Stems: MVSEP (placeholder) or import
    # -----------------------------------------------------------------------
    def _call_mvsep(self, audio_path):
        """
        Use StemSeparator to obtain stems.
        If stem_source is 'import', fall back to imported stems.
        Otherwise, run the full separation pipeline, auto-filling Stage-3
        instrument-specific models from the detected tags when enabled.
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

    def _split_lead_backing(self, stem_paths):
        """Optional lead/backing vocal split (MVSEP model or experimental DSP)."""
        mode = str(self.config.get("lead_vocal_splitter", "off") or "off").strip().lower()
        if mode == "off":
            return
        vocal = stem_paths.get("vocals")
        if not vocal or not os.path.exists(vocal):
            return
        out_dir = os.path.join(str(Path(self.file_path).parent), "vocal_split")
        try:
            if mode == "mvsep":
                from modules.lead_vocals import find_backing_vocal_model
                from modules.mvsep_api import create_separation, poll_until_done, get_result_files
                rid, name = find_backing_vocal_model()
                api_token = self.config.get("mvsep_api_key", "").strip()
                if rid and api_token:
                    self.progress.emit(20, f"Running lead/backing split via MVSEP ({name})...")
                    job, status = create_separation(vocal, api_token, rid)
                    if status == 200:
                        data = poll_until_done(job, api_token)
                        files = get_result_files(data, out_dir)
                        lead = backing = None
                        for f in files:
                            bn = os.path.basename(f).lower()
                            if "backing" in bn or "harmony" in bn:
                                backing = f
                            elif "lead" in bn or "main" in bn or "vocal" in bn:
                                lead = f
                        if lead and backing:
                            stem_paths["lead_vocals"] = lead
                            stem_paths["backing_vocals"] = backing
                            self.step_completed.emit("vocal_split", {"lead": lead, "backing": backing})
                            return
                # fall through to the heuristic when the MVSEP path produced nothing
            from modules.lead_vocals import split_lead_backing
            self.progress.emit(20, "Splitting lead/backing vocals (heuristic, experimental)...")
            lead, backing = split_lead_backing(vocal, out_dir)
            stem_paths["lead_vocals"] = lead
            stem_paths["backing_vocals"] = backing
            self.step_completed.emit("vocal_split", {"lead": lead, "backing": backing})
        except Exception as e:  # noqa: BLE001
            self.progress.emit(20, f"Lead/backing split skipped: {e}")

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

    # -----------------------------------------------------------------------
    # Structural boundaries – now adaptive!
    # -----------------------------------------------------------------------
    def _find_sections(self, audio_path):
        """Structure segmentation: SongFormer (functional labels) or librosa.

        When ``structure_backend`` is ``songformer`` and Kaggle credentials are
        present, runs the SongFormer kernel and maps ``[{start, end, label}]``
        into the pipeline's section format (names like ``Verse_01``). Falls back
        to the librosa agglomerative method on any failure.
        """
        backend = str(self.config.get("structure_backend", "librosa") or "librosa").strip().lower()
        if backend == "songformer":
            if not (self.config.get("kaggle_user") and self.config.get("kaggle_key")):
                self.progress.emit(28, "SongFormer needs Kaggle creds — using librosa boundaries.")
            else:
                try:
                    from workers.structure import run_structure_analysis
                    self.progress.emit(26, "Running SongFormer structure analysis (Kaggle)...")
                    segs = run_structure_analysis(
                        audio_path, self.config,
                        min_segment_sec=int(float(self.config.get("segment_min_sec", 12))),
                        progress_cb=lambda p, m: self.progress.emit(26 + int(p * 0.2), m),
                    )
                    if segs:
                        sections = []
                        counts = {}
                        for seg in segs:
                            label = (seg.get("label") or "section").strip().replace(" ", "_")
                            counts[label] = counts.get(label, 0) + 1
                            sections.append({
                                "name": f"{label.capitalize()}_{counts[label]:02d}",
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                                "label": label,
                            })
                        sections = [s for s in sections if s["end"] > s["start"]]
                        if sections:
                            return sections
                        self.progress.emit(28, "SongFormer returned no segments — using librosa boundaries.")
                    else:
                        self.progress.emit(28, "SongFormer returned no segments — using librosa boundaries.")
                except Exception as e:  # noqa: BLE001
                    self.progress.emit(28, f"SongFormer failed ({e}) — using librosa boundaries.")
        return self._find_boundaries(audio_path)

    def _find_boundaries(self, audio_path):
        # Load audio and get duration
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        duration = librosa.get_duration(y=y_mono, sr=sr)

        # ---- Adaptive k ----
        if duration < 60:          # less than 1 minute
            k = max(2, int(duration / 15))
        else:                      # 1 minute or longer
            k = max(4, int(duration / 30))
        # Cap k (configurable) to avoid too many tiny sections
        max_k = int(self.config.get("segment_max_k", 20))
        k = min(k, max_k)
        self.progress.emit(25, f"Using k={k} sections for {duration:.1f}s audio")

        # MFCC agglomerative clustering
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=k)
        bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]

        # Merge sections shorter than the configured minimum (default 12s)
        min_sec = float(self.config.get("segment_min_sec", 12.0))
        filtered = [bound_times[0]]
        for t in bound_times[1:]:
            if t - filtered[-1] >= min_sec:
                filtered.append(t)
        if filtered[-1] < duration:
            filtered[-1] = duration
        bound_times = filtered

        # Build the final sections list
        sections = []
        for i in range(len(bound_times)-1):
            sections.append({
                "name": f"Section_{i+1:02d}",
                "start": bound_times[i],
                "end": bound_times[i+1]
            })
        return sections

    # -----------------------------------------------------------------------
    # Extract sections from each stem (no L/R)
    # -----------------------------------------------------------------------
    def _extract_stem_sections(self, stem_paths, sections):
        chunks = []
        base_dir = Path(self.file_path).parent / "structural_chunks"
        base_dir.mkdir(exist_ok=True)
        for stem_type, stem_path in stem_paths.items():
            y, sr = librosa.load(stem_path, sr=None, mono=False)
            for sec in sections:
                start_s = int(sec['start'] * sr)
                end_s = int(sec['end'] * sr)
                chunk = y[:, start_s:end_s] if y.ndim > 1 else y[start_s:end_s]
                base_name = f"{Path(stem_path).stem}_{sec['name']}"
                out_path = base_dir / f"{base_name}.wav"
                if y.ndim > 1:
                    sf.write(out_path, chunk.T, sr)
                else:
                    sf.write(out_path, chunk, sr)
                chunks.append({
                    "stem_type": stem_type,
                    "section": sec['name'],
                    "path": str(out_path)
                })
        return chunks

    # -----------------------------------------------------------------------
    # REAL KAGGLE CAPTIONING using Qwen2.5‑Omni (copied from your notebook)
    # -----------------------------------------------------------------------
    def _caption_chunks(self, chunks):
        """
        Caption section chunks via the ACE-Step captioner on a Kaggle GPU.

        Correct flow: upload the chunk WAVs as a private Kaggle dataset, attach
        it (plus the cached model dataset) via dataset_sources, and run the
        shared caption kernel (kernels/caption_kernel.py).
        Returns dict mapping chunk path -> caption.
        """
        from modules.kaggle import (
            upload_audio_dataset, push_kernel, wait_kernel_done,
            download_kernel_output,
        )
        from pathlib import Path as _Path

        # 1. Stage chunk WAVs into a dataset dir
        temp_dir = tempfile.mkdtemp(prefix="struct_caption_")
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        for chunk in chunks:
            shutil.copy2(chunk['path'], os.path.join(audio_dir, Path(chunk['path']).name))

        # 2. Upload as a private Kaggle dataset
        self.progress.emit(50, "Uploading section audio to Kaggle...")
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

        kernel_slug = f"struct-caption-{uuid.uuid4().hex[:6]}"
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
        self.progress.emit(60, "Pushing caption kernel to Kaggle GPU...")
        push_kernel(self.config, kernel_dir, kernel_slug)
        ok = wait_kernel_done(self.config, kernel_slug)
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
                if Path(chunk['path']).name == file_name:
                    caption_map[chunk['path']] = item.get("caption", "")
                    break
        # Hybrid captions: blend deterministic tags with the LLM prose.
        ratio = float(self.config.get("tag_caption_ratio", 0) or 0)
        if ratio > 0 and self.tags:
            caption_map = {
                p: compose_caption(self.tags, c, ratio) for p, c in caption_map.items()
            }
        shutil.rmtree(temp_dir, ignore_errors=True)
        return caption_map
    def _aggregate_with_deepseek(self, sections, captions):
        """
        Build a rich context for DeepSeek using:
          - Segment names and timestamps.
          - The actual caption from Kaggle for each segment.
          - Additional evidence (e.g., if caption mentions vocals, etc.)
        Then call DeepSeek to synthesize a master caption.
        """
        # Build per‑segment evidence
        evidence_lines = []
        for sec in sections:
            sec_name = sec['name']
            start = sec['start']
            end = sec['end']
            # Collect all captions that belong to this section
            sec_captions = []
            for path, cap in captions.items():
                if sec_name in path:
                    sec_captions.append(cap)
            combined = " | ".join(sec_captions) if sec_captions else "(no caption)"

            # Simple evidence extraction: if "vocal" appears in caption, note it
            vocal_notes = "Vocals detected" if "vocal" in combined.lower() else "Instrumental"
            energy_notes = "High energy" if any(w in combined.lower() for w in ["loud", "intense", "driving"]) else "Moderate energy"

            evidence_lines.append(
                f"  [{sec_name}] {start:.1f}s - {end:.1f}s\n"
                f"    Caption: {combined}\n"
                f"    Evidence: {vocal_notes}, {energy_notes}"
            )

        evidence_text = "\n".join(evidence_lines)

        # Build the LLM prompt (provider-aware)
        from modules.llm_client import provider_key_present
        if not provider_key_present(self.config):
            return "No LLM API key provided for the selected provider."

        # Use real detected tags when available (tagger runs pre-stem).
        target_genre = "Alternative Rock"
        global_bpm = int(self.tags.get("bpm") or 120)

        # Use the existing orchestrator but with enriched context
        # We'll pass the evidence as a separate parameter.
        # Since the orchestrator expects `segments` with captions, we reuse that.
        # But we add the evidence to the description.
        segments_with_evidence = []
        for sec in sections:
            # Re‑create the segment with the combined caption (already have it)
            sec_name = sec['name']
            combined = ""
            for path, cap in captions.items():
                if sec_name in path:
                    combined = cap
                    break
            segments_with_evidence.append({
                "name": sec_name,
                "start_sec": sec['start'],
                "end_sec": sec['end'],
                "caption": combined,
                # Add extra fields for evidence (the orchestrator will ignore them for now)
                "evidence": evidence_lines  # we'll pass it separately
            })

        # Instantiate the provider-aware LLM orchestrator
        orchestrator = DeepSeekMusicOrchestrator(config=self.config)

        # We'll override the user context building in the orchestrator by
        # calling the method directly and passing the evidence as a string.
        # To avoid modifying the orchestrator, we'll call it with the enriched segments.
        # But the orchestrator's `generate_master_dataset_prompt` doesn't accept evidence.
        # So we'll build a custom prompt here and use the orchestrator's client.
        system_prompt = (
            "You are an elite music prompt engineer for ACE-Step. Synthesize a cohesive master prompt from structural evidence, "
            "instrumentation descriptions, and timing information. Output ONLY the final prompt, no introductory text. "
            "Structure: [Genre/Vibe], [Production Texture], [Instrumentation with descriptive details], [Dynamics/Energy], [Structural flow]."
        )
        user_context = f"TARGET GENRE: {target_genre}\nGLOBAL BPM: {global_bpm}\n\n"
        user_context += "STRUCTURAL EVIDENCE (per section):\n"
        user_context += evidence_text
        user_context += "\n\nCompile final master caption now:"

        try:
            response = orchestrator.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                temperature=0.4,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"DeepSeek error: {e}"

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# Batch Worker for Structural Pipeline (processes multiple tracks)
# ============================================================================
class StructuralPipelineBatchWorker(QThread):
    progress = Signal(int, str)
    track_done = Signal(str, dict)
    all_done = Signal()
    error_occurred = Signal(str)

    def __init__(self, tracks, config, options):
        super().__init__()
        self.tracks = tracks
        self.config = config
        self.options = options
        self._is_cancelled = False

    def run(self):
        total = len(self.tracks)
        for idx, sample in enumerate(self.tracks):
            if self._is_cancelled:
                break

            track_id = sample["id"]
            file_path = sample["audio_path"]

            # Create a single-track worker
            worker = StructuralPipelineWorker(
                track_id=track_id,
                file_path=file_path,
                config=self.config,
                options=self.options
            )

            # Connect its signals
            worker.progress.connect(lambda p, msg, idx=idx, total=total:
                self.progress.emit(int((idx/total)*100 + (p/total)),
                                   f"[{idx+1}/{total}] {msg}"))
            worker.step_completed.connect(lambda step, data: None)
            worker.pipeline_finished.connect(lambda tid, result: self.track_done.emit(tid, result))
            worker.error_occurred.connect(self.error_occurred.emit)

            worker.start()
            worker.wait()
            self.msleep(100)  # brief pause between tracks

        self.all_done.emit()

    def cancel(self):
        self._is_cancelled = True
# ============================================================================
# Batch Worker for Structural Pipeline (processes multiple tracks)
# ============================================================================
