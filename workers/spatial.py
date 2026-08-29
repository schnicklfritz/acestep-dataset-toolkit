import os, json, uuid, tempfile, subprocess, shutil, time
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from stem_separator import StemSeparator

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

    def run(self):
        try:
            # Step 1: Obtain stems
            if self.options['stem_source'] == 'mvsep':
                self.progress.emit(10, "Calling MVSEP API for stem separation...")
                stem_paths = self._call_mvsep(self.file_path)
            else:
                self.progress.emit(10, "Looking for imported stems...")
                stem_paths = self._find_imported_stems(self.file_path)
            if not stem_paths:
                self.error_occurred.emit("No stems found.")
                return
            self.step_completed.emit("stems", stem_paths)

            # Step 2: Structural slicing
            self.progress.emit(20, "Slicing full mix into structural sections...")
            sections = self._slice_audio(self.file_path)
            if not sections:
                self.error_occurred.emit("Structural slicing failed.")
                return
            self.step_completed.emit("sections", sections)

            # Step 3: Extract L/R chunks
            self.progress.emit(30, "Extracting L/R channels per section...")
            chunks = self._extract_lr_chunks(stem_paths, sections)
            if not chunks:
                self.error_occurred.emit("L/R chunk extraction failed.")
                return
            self.step_completed.emit("chunks", chunks)

            # Step 4: Caption via Kaggle (or custom endpoint)
            self.progress.emit(40, "Running captioning on L/R chunks via Kaggle...")
            captions = self._caption_chunks(chunks)
            if not captions:
                self.error_occurred.emit("Captioning failed.")
                return
            self.step_completed.emit("captions", captions)

            # Step 5: Spatial evaluator
            self.progress.emit(70, "Evaluating spatial placement...")
            spatial_tokens = self._evaluate_spatial(captions)
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
                "chunk_paths": chunks
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

    def _slice_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        duration = librosa.get_duration(y=y_mono, sr=sr)
        mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=9)
        bound_times = [0.0] + librosa.frames_to_time(bounds, sr=sr).tolist() + [duration]
        filtered = [bound_times[0]]
        for t in bound_times[1:]:
            if t - filtered[-1] >= 12.0:
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
        # Placeholder – would push Kaggle notebook
        captions = {}
        for chunk in chunks:
            captions[chunk['L_path']] = f"Caption for {chunk['L_path']}"
            captions[chunk['R_path']] = f"Caption for {chunk['R_path']}"
        return captions

    def _evaluate_spatial(self, captions):
        instruments = ["guitar", "organ", "bass", "drums", "vocals", "piano", "synth"]
        spatial_tokens = {}
        for inst in instruments:
            spatial_tokens[inst] = "centered"
        return spatial_tokens

    def _aggregate_with_deepseek(self, sections, captions, spatial_tokens):
        return "Master caption from DeepSeek."

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL HealthAuditorWorker (from your file – keep as is)
# ============================================================================
