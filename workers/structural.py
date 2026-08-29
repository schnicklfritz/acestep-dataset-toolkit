import os, json, uuid, tempfile, subprocess, shutil, time, struct
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI
from PySide6.QtCore import QThread, Signal
from stem_separator import StemSeparator
from workers.deepseek import DeepSeekMusicOrchestrator

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

    def run(self):
        try:
            # Step 1: Obtain stems
            if self.options.get('stem_source') == 'mvsep':
                self.progress.emit(10, "Calling MVSEP API for stem separation...")
                stem_paths = self._call_mvsep(self.file_path)
            else:
                self.progress.emit(10, "Looking for imported stems...")
                stem_paths = self._find_imported_stems(self.file_path)
            if not stem_paths:
                self.error_occurred.emit("No stems found.")
                return
            self.step_completed.emit("stems", stem_paths)

            # Step 2: Structural boundaries – now adaptive!
            self.progress.emit(20, "Finding structural boundaries (adaptive)...")
            sections = self._find_boundaries(self.file_path)
            if not sections:
                self.error_occurred.emit("Could not determine structural boundaries.")
                return
            self.step_completed.emit("sections", sections)

            # Step 3: Extract stem sections (no L/R split)
            self.progress.emit(30, "Extracting stem sections...")
            chunks = self._extract_stem_sections(stem_paths, sections)
            if not chunks:
                self.error_occurred.emit("Failed to extract stem sections.")
                return
            self.step_completed.emit("chunks", chunks)

            # Step 4: Real captioning via Kaggle (Qwen2.5‑Omni)
            self.progress.emit(40, "Captioning stem sections via Kaggle (Qwen2.5‑Omni)...")
            captions = self._caption_chunks(chunks)
            if not captions:
                self.error_occurred.emit("Captioning failed.")
                return
            self.step_completed.emit("captions", captions)

            # Step 5: DeepSeek aggregation with enriched context
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
                "chunk_paths": chunks
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
        Otherwise, run the full separation pipeline.
        """
        # If the user chose "Import existing stems", just look for them
        if self.options.get('stem_source') == 'import':
            return self._find_imported_stems(audio_path)

        # Otherwise, use the StemSeparator
        separator = StemSeparator(self.config, progress_callback=self.progress.emit)
        # Get stem options from the worker's options (they are set in run_structural_pipeline)
        stem_options = self.options.get('stem_options', {})
        # Determine which method to use – for Spatial, we always want instrument‑specific
        method = 'polarformer+multi+instrument'
        stems = separator.separate(audio_path, method=method, options=stem_options)
        return stems

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
        # Cap k at 20 to avoid too many tiny sections
        k = min(k, 20)
        self.progress.emit(25, f"Using k={k} sections for {duration:.1f}s audio")

        # MFCC agglomerative clustering
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=k)
        bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]

        # Merge sections shorter than 12 seconds
        filtered = [bound_times[0]]
        for t in bound_times[1:]:
            if t - filtered[-1] >= 12.0:
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
        Pushes a Kaggle kernel that:
          - Loads the Qwen2.5‑Omni captioner (ACE-Step/acestep-captioner)
          - Processes every WAV file in /kaggle/working/input/
          - Writes a captions.json with results
        Returns dict mapping chunk path -> caption.
        """
        import tempfile, shutil, subprocess, time, json, os
        from pathlib import Path

        # 1. Stage all chunk WAVs in a temporary folder
        temp_dir = tempfile.mkdtemp(prefix="struct_caption_")
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        for chunk in chunks:
            src = chunk['path']
            dst = os.path.join(input_dir, Path(src).name)
            shutil.copy2(src, dst)

        # 2. Set Kaggle credentials
        user = self.config.get("kaggle_user", "").strip()
        key = self.config.get("kaggle_key", "").strip()
        if not user or not key:
            raise ValueError("Kaggle credentials not configured.")
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key

        # 3. Build the kernel script – this is the notebook code!
        kernel_slug = f"struct-caption-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(tempfile.mkdtemp(), kernel_slug)
        os.makedirs(kernel_dir, exist_ok=True)

        # ---- The kernel script (copied from your working notebook) ----
        worker_py = f"""
import os, json, glob, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ---- Configuration ----
MODEL_ID = "ACE-Step/acestep-captioner"
PRECISION = "fp16"
BATCH_SIZE = 1

# Set device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if PRECISION == "fp16" else torch.bfloat16

# Load model with balanced device map (for multi‑GPU)
load_kwargs = {{
    "device_map": "balanced",
    "max_memory": {{0: "10GiB", 1: "10GiB"}},
    "offload_folder": "/kaggle/working/offload",
    "trust_remote_code": True,
    "torch_dtype": torch_dtype,
}}
try:
    import flash_attn
    load_kwargs["attn_implementation"] = "flash_attention_2"
except ImportError:
    load_kwargs["attn_implementation"] = "sdpa"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ---- Caption prompt (same as your notebook) ----
CAPTION_PROMPT = (
    "You are a professional music metadata tagger preparing training data for ACE-Step. "
    "Listen carefully to this audio clip and write a detailed description. "
    "Cover: specific instrumentation (name every instrument you hear), "
    "whether vocals are present (gender, register, timbre) or confirm instrumental, "
    "recording and production character, mood, and how the clip develops. "
    "Write 3 to 5 sentences. Start with A or An. "
    "Genre, BPM, key, and time signature are handled separately — do not include them."
)

def extract_reply(text):
    if "assistant\\n" in text:
        return text.split("assistant\\n")[-1].strip()
    if "assistant" in text:
        return text.split("assistant")[-1].strip()
    return text.strip()

# ---- Process all WAV files in input folder ----
input_folder = "/kaggle/working/input"
os.makedirs(input_folder, exist_ok=True)

audio_files = sorted(glob.glob(os.path.join(input_folder, "*.wav")))
results = []

for f in audio_files:
    fname = os.path.basename(f)
    # Build conversation
    conversation = [
        {{"role": "system", "content": [{{"type": "text", "text": (
            "You are Qwen, a virtual human developed by the Qwen Team, "
            "Alibaba Group, capable of perceiving auditory and visual inputs, "
            "as well as generating text and speech."
        )}}]}},
        {{"role": "user", "content": [
            {{"type": "audio", "audio": f}},
            {{"type": "text", "text": CAPTION_PROMPT}},
        ]}}
    ]
    text_input = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text_input, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False
    ).to(model.device).to(model.dtype)

    output_ids = model.generate(
        **inputs, use_audio_in_video=False, return_audio=False, max_new_tokens=512
    )
    full_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    caption = extract_reply(full_text)
    results.append({{"file": fname, "caption": caption}})

# ---- Write JSON output ----
with open("/kaggle/working/captions.json", "w") as out_f:
    json.dump({{"results": results}}, out_f, indent=2)
"""
        # Write the kernel files
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w") as f:
            f.write(worker_py)

        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Push the kernel
        self.progress.emit(50, "Pushing Kaggle kernel (Qwen2.5‑Omni)...")
        push_cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
        res = subprocess.run(push_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Kaggle push failed: {res.stderr}")

        # 5. Wait for completion and fetch output
        self.progress.emit(60, "Waiting for Kaggle job...")
        out_dir = os.path.join(tempfile.mkdtemp(), "output")
        os.makedirs(out_dir, exist_ok=True)
        for attempt in range(12):  # up to ~2 minutes
            if self._is_cancelled:
                return {}
            time.sleep(10)
            # Try to download output
            subprocess.run(
                ["kaggle", "kernels", "output", f"{user}/{kernel_slug}", "-p", out_dir],
                capture_output=True
            )
            json_path = os.path.join(out_dir, "captions.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                results = data.get("results", [])
                caption_map = {}
                for item in results:
                    file_name = item["file"]
                    for chunk in chunks:
                        if Path(chunk['path']).name == file_name:
                            caption_map[chunk['path']] = item["caption"]
                            break
                # Clean up
                shutil.rmtree(kernel_dir, ignore_errors=True)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return caption_map

        raise TimeoutError("Kaggle job did not complete in time.")

    # -----------------------------------------------------------------------
    # DEEPSEEK AGGREGATION with enriched evidence
    # -----------------------------------------------------------------------
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

        # Build the DeepSeek prompt
        api_key = self.config.get("custom_key", "").strip()
        if not api_key:
            return "No DeepSeek API key provided."

        # For now, we use defaults for genre and BPM.
        # In a future version we can read them from the sample metadata.
        target_genre = "Alternative Rock"
        global_bpm = 120

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

        # Instantiate DeepSeek orchestrator
        orchestrator = DeepSeekMusicOrchestrator(api_key=api_key)

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
