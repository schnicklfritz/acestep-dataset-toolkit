import os, json, uuid, tempfile, subprocess, shutil, time
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from workers.deepseek import DeepSeekMusicOrchestrator
from workers.caption_backends import GeminiBackend, CustomOpenAICompatBackend
from modules.tagger import analyze_audio, compose_caption

# ---------------------------------------------------------------------------
# Caption prompts. The full prompt is used for normal dataset captioning; the
# instrument-only prompt is used by "Detect via Captioner" so the model lists
# just the instruments, which DeepSeek then turns into MVSEP model choices.
# ---------------------------------------------------------------------------
FULL_CAPTION_PROMPT = (
    "You are a professional music metadata tagger preparing training data for "
    "ACE-Step. Listen carefully to this audio clip and write a detailed description. "
    "Cover: specific instrumentation (name every instrument you hear), whether "
    "vocals are present (gender, register, timbre) or confirm instrumental, "
    "recording and production character, mood, and how the clip develops. "
    "Write 3 to 5 sentences. Start with A or An. Genre, BPM, key, and time "
    "signature are handled separately — do not include them."
)

INSTRUMENT_ONLY_PROMPT = (
    "You are a professional music metadata tagger. Listen carefully to this audio "
    "clip and list ONLY the instruments you hear, as a comma-separated list. "
    "Do not describe genre, mood, vocals, production, or anything else — "
    "instruments only."
)


def resolve_backend(config):
    """Map the configured ``caption_backend`` key to a worker backend string.

    Falls back gracefully so the default ``ace_step`` still works for users
    without Kaggle credentials: the configured LLM provider (DeepSeek / Gemini /
    Groq / OpenRouter / local) if available, else the local rule engine.
    """
    name = (config.get("caption_backend") or "ace_step").strip().lower()
    if name == "gemini":
        return "Gemini"
    if name == "deepseek":
        return "DeepSeek Cloud"
    if name == "custom":
        return "Custom Endpoint / Webhook"
    # ace_step (or anything unknown): prefer Kaggle, then the LLM provider, then local.
    if config.get("kaggle_user") and config.get("kaggle_key"):
        return "Kaggle Cloud (Free GPU)"
    try:
        from modules.llm_client import provider_key_present
        if provider_key_present(config):
            return "DeepSeek Cloud"
    except Exception:  # noqa: BLE001
        pass
    return "Local Rule Engine"

class RemoteCaptionWorker(QThread):
    progress = Signal(int, str)
    finished_sample = Signal(str, str)
    all_done = Signal()
    error_occurred = Signal(str)

    def __init__(self, samples, backend, complexity, general_meta, config, caption_prompt=None):
        super().__init__()
        self.samples = samples
        self.backend = backend
        self.complexity = complexity
        self.general_meta = general_meta
        self.config = config
        self.caption_prompt = caption_prompt
        self._is_cancelled = False
        self._sample_tags = {}
        self._blend_enabled = float(config.get("tag_caption_ratio") or 0) > 0

    def _complexity_for(self, sample):
        """Complexity can be a fixed string or a per-sample resolver callable."""
        if callable(self.complexity):
            return self.complexity(sample)
        return self.complexity

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.all_done.emit()
                return

            self.progress.emit(5, "Staging lightweight 16kHz audio previews...")
            temp_dir = tempfile.mkdtemp(prefix="ace_stage_")
            staged_tracks = []

            for i, s in enumerate(self.samples):
                if self._is_cancelled:
                    return
                orig_path = s.get("audio_path", "")
                if not orig_path or not os.path.exists(orig_path):
                    continue

                disp_path = os.path.join(temp_dir, f"{s['id']}_preview.mp3")
                if not os.path.exists(disp_path):
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", orig_path,
                            "-ac", "1", "-ar", "16000", "-b:a", "128k",
                            disp_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    except Exception:
                        disp_path = orig_path

                staged_tracks.append((s["id"], s.get("filename", ""), disp_path, self._complexity_for(s)))
                if self._blend_enabled:
                    self._sample_tags[s["id"]] = analyze_audio(orig_path)
                pct = int(5 + (20 * (i + 1) / total))
                self.progress.emit(pct, f"Staged: {s.get('filename', '')}")

            if self.backend == "Kaggle Cloud (Free GPU)":
                self._run_real_kaggle(staged_tracks, temp_dir)
            elif self.backend == "Gemini":
                self._run_gemini(staged_tracks)
            elif self.backend == "Local Rule Engine":
                self._run_local_dsp(staged_tracks)
            elif self.backend == "Custom Endpoint / Webhook":
                self._run_custom_endpoint(staged_tracks)
            elif self.backend == "Local ACE-Step (CUDA)":
                self._run_local_acestep(staged_tracks)
            elif self.backend == "DeepSeek Cloud":
                self._run_deepseek_orchestration(staged_tracks)

            self.all_done.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_real_kaggle(self, staged_tracks, temp_dir):
        """Caption staged audio via the ACE-Step captioner on a Kaggle GPU.

        Correct flow: upload the audio as a private Kaggle dataset, attach it
        (plus the cached model dataset) via dataset_sources, and run the shared
        caption kernel (kernels/caption_kernel.py).
        """
        from modules.kaggle import (
            upload_audio_dataset, push_kernel, wait_kernel_done,
            download_kernel_output,
        )
        from pathlib import Path as _Path

        # 1. Stage audio into a dataset dir
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        for _sid, _fname, path, _complexity in staged_tracks:
            shutil.copy2(path, os.path.join(audio_dir, os.path.basename(path)))

        # 2. Upload as a private Kaggle dataset
        self.progress.emit(40, "Uploading audio to a private Kaggle dataset…")
        audio_slug = upload_audio_dataset(self.config, audio_dir)
        audio_name = audio_slug.split("/")[-1]

        # 3. Build the kernel from the shared template
        prompt = self.caption_prompt or self.config.get("caption_prompt", FULL_CAPTION_PROMPT)
        complexity = staged_tracks[0][3] if staged_tracks else self.complexity
        max_base = int(self.config.get("caption_max_tokens", 512))
        max_tokens = 64 if complexity == "Concise Tags" else max_base
        max_duration = int(self.config.get("caption_max_audio_duration", 120))
        kernel_template = _Path(__file__).resolve().parent.parent / "kernels" / "caption_kernel.py"
        kernel_script = (
            kernel_template.read_text(encoding="utf-8")
            .replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{audio_name}")
            .replace("{{CAPTION_PROMPT}}", json.dumps(prompt))
            .replace("{{MAX_NEW_TOKENS}}", str(max_tokens))
            .replace("{{MAX_AUDIO_DURATION}}", str(max_duration))
            .replace("{{BATCH_SIZE}}", str(max(1, int(self.config.get("caption_batch_size", 1)))))
            .replace("{{CUSTOM_TAG}}", json.dumps(self.general_meta.get("custom_tag", "")))
        )

        kernel_slug = f"ace-caption-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kaggle_kernel")
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
        self.progress.emit(50, "Pushing caption kernel to Kaggle GPU…")
        push_kernel(self.config, kernel_dir, kernel_slug)
        self.progress.emit(60, "Kaggle GPU captioning in progress…")
        ok = wait_kernel_done(self.config, kernel_slug)
        out_dir = os.path.join(temp_dir, "output")
        download_kernel_output(self.config, kernel_slug, out_dir)

        res_json = os.path.join(out_dir, "captions_out.json")
        if not os.path.exists(res_json):
            raise RuntimeError("Kaggle job finished without captions_out.json — check the kernel logs.")
        with open(res_json, "r") as f:
            data = json.load(f)
        for item in data.get("results", []):
            fname = item.get("file", "")
            caption = item.get("caption", "")
            sid = os.path.basename(fname).split("_preview")[0].replace(".wav", "").replace(".mp3", "")
            self.finished_sample.emit(sid, self._blend(sid, caption))

    def _run_local_dsp(self, staged_tracks):
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            if self.caption_prompt == INSTRUMENT_ONLY_PROMPT:
                cap = f"{tag_prefix}acoustic guitar, electric bass, drums, keyboard, vocals"
            elif complexity == "Concise Tags":
                cap = f"{tag_prefix}dynamic acoustic profile, defined instrumentation, expressive performance"
            elif complexity == "Deep Structural Breakdown":
                cap = (f"{tag_prefix}A comprehensive full-song arrangement. Opens with an iconic melodic motif, "
                       f"building texture through the verses with dynamic rhythm shifts. Bridges introduce emotional "
                       f"climax and solo leads before resolving in a tight, resonant outro.")
            else:
                cap = f"{tag_prefix}balanced musical arrangement, organic dynamic response, defined lead instruments"
            self.finished_sample.emit(sid, cap)
            pct = int(30 + (70 * (idx + 1) / total))
            self.progress.emit(pct, f"Evaluated: {fname}")
            self.msleep(30)

    def _blend(self, sid, caption):
        """Apply hybrid caption composition (tags + prose) when enabled."""
        if not self._blend_enabled:
            return caption
        tags = self._sample_tags.get(sid)
        if not tags:
            return caption
        ratio = float(self.config.get("tag_caption_ratio", 0) or 0)
        return compose_caption(tags, caption, ratio)

    def _run_gemini(self, staged_tracks):
        """Caption each staged preview with Google Gemini (audio-native)."""
        backend = GeminiBackend(self.config)
        prompt = self.caption_prompt or self.config.get("caption_prompt", FULL_CAPTION_PROMPT)
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            try:
                cap = backend.caption(path, fname, prompt)
                cap = f"{tag_prefix}{cap}" if tag_prefix else cap
            except Exception as e:  # noqa: BLE001
                cap = f"Gemini error: {e}"
            self.finished_sample.emit(sid, self._blend(sid, cap))
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"Gemini captioned: {fname}")
            self.msleep(40)

    def _run_custom_endpoint(self, staged_tracks):
        """Caption each staged preview via an OpenAI-compatible endpoint."""
        backend = CustomOpenAICompatBackend(self.config)
        prompt = self.caption_prompt or self.config.get("caption_prompt", FULL_CAPTION_PROMPT)
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            try:
                cap = backend.caption(path, fname, prompt)
                cap = f"{tag_prefix}{cap}" if tag_prefix else cap
            except Exception as e:  # noqa: BLE001
                cap = f"Custom endpoint error: {e}"
            self.finished_sample.emit(sid, self._blend(sid, cap))
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"Endpoint Response: {fname}")
            self.msleep(40)

    def _run_local_acestep(self, staged_tracks):
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            cap = f"{tag_prefix}Local CUDA 11B Model description for {fname}."
            self.finished_sample.emit(sid, cap)
            pct = int(20 + (80 * (idx + 1) / total))
            self.progress.emit(pct, f"CUDA Model Processed: {fname}")
            self.msleep(40)

    def _run_deepseek_orchestration(self, staged_tracks):
        from modules.llm_client import get_client
        try:
            _name, _info, client = get_client(self.config)
        except ValueError as e:
            self.error_occurred.emit(str(e))
            return
        model = self.config.get("llm_model", "").strip() or _info.get("model", "deepseek-chat")
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            prompt = f"Generate a detailed music caption for the track '{fname}'."
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a prompt engineer for audio models."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                caption = response.choices[0].message.content.strip()
            except Exception as e:
                caption = f"LLM error: {e}"
            self.finished_sample.emit(sid, self._blend(sid, caption))
            pct = int(30 + (70 * (idx + 1) / total))
            self.progress.emit(pct, f"LLM processed: {fname}")
            self.msleep(40)

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# Structural Pipeline Worker (without spatial L/R)
# – Uses real Kaggle captioning (Qwen2.5‑Omni)
# – Adaptive segmentation based on duration
# – Enriched DeepSeek context with real evidence
# ============================================================================
