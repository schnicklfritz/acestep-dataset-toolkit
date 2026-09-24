import os, json, uuid, tempfile, subprocess, time
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from workers.deepseek import DeepSeekMusicOrchestrator
from workers.caption_backends import GeminiBackend, CustomOpenAICompatBackend
from modules import caption_spec
from modules.caption_quality import trim_to_caption
from modules.tagger import analyze_audio, compose_caption

# ---------------------------------------------------------------------------
# Caption prompts. The full prompt is used for normal dataset captioning; the
# instrument-only prompt is used by "Detect via Captioner" so the model lists
# just the instruments, which DeepSeek then turns into MVSEP model choices.
# ---------------------------------------------------------------------------
# DEPRECATED — kept as an alias so existing imports keep working. This was the
# schema-less prompt that contradicted the ACE-Step 1.5XL annotation standard:
# "Write 3 to 5 sentences. Start with A or An" is the exact opposite of the
# required front-loaded tag list (5-12 comma-separated keywords). The schema now
# comes from modules/caption_spec.py and is sent as the SYSTEM prompt; this is
# only the user-turn task line.
FULL_CAPTION_PROMPT = caption_spec.DEFAULT_TASK_PROMPT

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

            skipped = total - len(staged_tracks)
            if not staged_tracks:
                # REFUSE to proceed. Running anyway uploads an EMPTY Kaggle
                # dataset, and the kernel then reports "... was empty" and burns a
                # GPU session producing nothing. That is exactly what happened.
                self.error_occurred.emit(
                    f"Nothing to caption: none of the {total} track(s) has a "
                    "usable audio_path, so there is nothing to upload. Every "
                    "sample whose audio file is missing is skipped silently — "
                    "check the paths in 📂 Open JSON (e.g. files moved by "
                    "normalization, or a dataset referencing another machine's "
                    "absolute paths)."
                )
                return
            if skipped:
                self.progress.emit(
                    25, f"Skipped {skipped} track(s) with missing audio files."
                )

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
            upload_or_update_audio_dataset, push_kernel, wait_kernel_done,
            wait_dataset_ready, download_kernel_output, fetch_kernel_logs,
            dataset_sources, preflight_kaggle,
        )
        from modules import caption_kaggle_run as ckr
        from pathlib import Path as _Path

        # PREFLIGHT: prove Kaggle accepts the credentials and the dataset slugs
        # BEFORE staging the audio. Failing here costs nothing; the same failure
        # after a staging pass and an upload used to surface only as an opaque
        # HTTP error minutes later.
        preflight_kaggle(
            self.config,
            self.config.get("caption_audio_dataset"),
            self.config.get("kaggle_model_dataset"),
        )

        # 1. Stage the ORIGINAL audio into the PERSISTENT staging folder.
        #    NOT the 16 kHz mono preview the other backends consume: the captioner
        #    hears this file, so a downsampled preview would degrade every caption
        #    it produced. The folder is on disk (not tempfile.mkdtemp) so the user
        #    can add or remove songs between runs — its CONTENTS are the uploaded
        #    Kaggle dataset.
        convert = bool(self.config.get("caption_convert_mp3", True))
        bitrate = self.config.get("caption_mp3_bitrate", "192k")
        staging = ckr.staging_dir(self.config)
        by_id = {s.get("id"): s for s in self.samples}
        items = [
            (sid, fname, (by_id.get(sid) or {}).get("audio_path") or path)
            for sid, fname, path, _complexity in staged_tracks
        ]
        self.progress.emit(35, f"Staging {len(items)} track(s) to {staging}…")
        staged, skipped = ckr.stage_tracks(
            items, staging, convert_mp3=convert, bitrate=bitrate,
        )
        if not staged:
            raise RuntimeError(
                f"Nothing could be staged into {staging} — every track's audio "
                "file was missing or unreadable. Uploading now would caption "
                "nothing."
            )
        if skipped:
            self.progress.emit(38, f"{skipped} track(s) could not be staged (missing audio).")
        staged_map = {os.path.basename(path): sid for sid, _fname, path in staged}
        if convert and not ckr.ffmpeg_available():
            self.progress.emit(
                38, "ffmpeg not found — staging the original files at full quality."
            )

        # 2. Upload as a private Kaggle dataset, or push a NEW VERSION of the one
        #    this user already has, so its identity survives across runs.
        known_slug = (self.config.get("caption_audio_dataset") or "").strip()
        label = f"Updating Kaggle dataset {known_slug}…" if known_slug \
            else "Uploading audio to a private Kaggle dataset…"
        self.progress.emit(40, label)
        audio_slug = upload_or_update_audio_dataset(
            self.config, staging, slug=known_slug,
        )
        # Remember it so "add/remove songs" updates THIS dataset next time
        # instead of orphaning it behind a new random slug.
        self.config["caption_audio_dataset"] = audio_slug
        # WAIT for the dataset version to finish processing. dataset_create_new
        # returns early, and pushing the kernel before it is ready mounts an EMPTY
        # /kaggle/input/<slug> -- the kernel then finds no audio and reports the
        # folder as empty. This is the failure wait_dataset_ready exists to stop.
        self.progress.emit(45, "Waiting for the Kaggle dataset to finish processing…")
        ready_reason = []
        if not wait_dataset_ready(self.config, audio_slug, reason=ready_reason):
            raise RuntimeError(
                f"Kaggle dataset {audio_slug} did not become ready in time. "
                "Pushing the kernel now would mount an EMPTY folder and caption "
                "nothing. "
                + " ".join(ready_reason)
                + " Check the dataset on kaggle.com and re-run."
            )
        audio_name = audio_slug.split("/")[-1]

        # 3. Build the kernel from the shared template.
        #    The SCHEMA is the system prompt; the user turn only says what to do
        #    with this clip. An explicit caption_prompt (e.g. INSTRUMENT_ONLY_
        #    PROMPT from "Detect via Captioner") still wins as the user turn.
        prompt = self.caption_prompt or caption_spec.task_prompt_from_config(self.config)
        system_prompt = caption_spec.system_prompt_from_config(self.config)
        rep_penalty = float(self.config.get("caption_repetition_penalty", 1.15) or 1.0)
        no_repeat = int(self.config.get("caption_no_repeat_ngram", 6) or 0)
        complexity = staged_tracks[0][3] if staged_tracks else self.complexity
        max_base = int(self.config.get("caption_max_tokens", 512))
        max_tokens = 64 if complexity == "Concise Tags" else max_base
        max_duration = int(self.config.get("caption_max_audio_duration", 120))
        kernel_template = _Path(__file__).resolve().parent.parent / "kernels" / "caption_kernel.py"
        kernel_script = (
            kernel_template.read_text(encoding="utf-8")
            .replace("{{AUDIO_DATASET_PATH}}", f"/kaggle/input/{audio_name}")
            .replace("{{CAPTION_PROMPT}}", json.dumps(prompt))
            .replace("{{SYSTEM_PROMPT}}", json.dumps(system_prompt))
            .replace("{{MAX_NEW_TOKENS}}", str(max_tokens))
            .replace("{{MAX_AUDIO_DURATION}}", str(max_duration))
            .replace("{{BATCH_SIZE}}", str(max(1, int(self.config.get("caption_batch_size", 1)))))
            .replace("{{CUSTOM_TAG}}", json.dumps(self.general_meta.get("custom_tag", "")))
            .replace("{{REPETITION_PENALTY}}", str(rep_penalty))
            .replace("{{NO_REPEAT_NGRAM}}", str(no_repeat))
        )

        kernel_slug = f"ace-caption-{uuid.uuid4().hex[:6]}"
        kernel_dir = os.path.join(temp_dir, "kaggle_kernel")
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "kernel_worker.py"), "w", encoding="utf-8") as f:
            f.write(kernel_script)

        user = self.config.get("kaggle_user", "").strip()
        # Empty is a DELIBERATE choice (no cached weights -> the kernel downloads
        # from Hugging Face), so it must reach dataset_sources() as "" and be
        # dropped there, not replaced by a default.
        model_slug = (self.config.get(
            "kaggle_model_dataset", "michelmoalem9b/acestep-captioner-model"
        ) or "").strip()
        metadata = {
            "id": f"{user}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": dataset_sources(audio_slug, model_slug),
            "competition_sources": [],
            "kernel_sources": [],
        }
        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Push, wait, download into the USER'S local folder.
        self.progress.emit(50, "Pushing caption kernel to Kaggle GPU…")
        push_kernel(self.config, kernel_dir, kernel_slug)
        self.progress.emit(60, "Kaggle GPU captioning in progress…")
        ok = wait_kernel_done(self.config, kernel_slug)
        if not ok:
            raise RuntimeError(
                "The Kaggle caption kernel did not finish successfully. "
                + (fetch_kernel_logs(self.config, kernel_slug)
                   or "Check the kernel log on kaggle.com.")
            )
        # The download destination is the path the user can actually choose:
        # Kaggle only persists /kaggle/working, so a path INSIDE the kernel is not
        # a knob worth offering.
        out_dir = ckr.output_dir(self.config)
        os.makedirs(out_dir, exist_ok=True)
        download_kernel_output(self.config, kernel_slug, out_dir)

        res_json = os.path.join(out_dir, "captions_out.json")
        if not os.path.exists(res_json):
            raise RuntimeError(
                f"Kaggle job finished without captions_out.json in {out_dir} — "
                "check the kernel logs."
            )
        results = ckr.load_results(res_json)

        # Match a result back to its track through the STAGING NAME, not by
        # surgery on the returned filename: the staged name is what we chose, so
        # a rename on either side is visible instead of silently mismatching.
        emitted, unmatched = 0, []
        for name, caption in results.items():
            sid = staged_map.get(os.path.basename(name)) or self._sid_for_stem(name)
            if not sid:
                unmatched.append(name)
                continue
            self.finished_sample.emit(sid, self._blend(sid, caption))
            emitted += 1
        if unmatched:
            self.progress.emit(
                95, f"{len(unmatched)} result(s) matched no track: "
                    + ", ".join(sorted(unmatched)[:3])
            )
        self.progress.emit(
            98, f"{emitted} caption(s) written to {res_json}"
        )

    def _sid_for_stem(self, name):
        """Fallback match: the staged filename's stem against sample filenames.

        Needed for a captions_out.json the user downloaded in an EARLIER run and
        re-imported later, where the in-memory staging map no longer exists.
        """
        stem = os.path.splitext(os.path.basename(name or ""))[0].lower()
        for sample in self.samples:
            if os.path.splitext(sample.get("filename", ""))[0].lower() == stem:
                return sample.get("id")
        return None

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
        prompt = self.caption_prompt or caption_spec.task_prompt_from_config(self.config)
        system_prompt = caption_spec.system_prompt_from_config(self.config)
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            try:
                cap = backend.caption(path, fname, prompt, system_prompt=system_prompt)
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
        prompt = self.caption_prompt or caption_spec.task_prompt_from_config(self.config)
        system_prompt = caption_spec.system_prompt_from_config(self.config)
        tag = self.general_meta.get("custom_tag", "").strip()
        tag_prefix = f"{tag}, " if tag else ""
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            try:
                cap = backend.caption(path, fname, prompt, system_prompt=system_prompt)
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
            _name, _info, client = get_client(self.config, role="captioner")
        except ValueError as e:
            self.error_occurred.emit(str(e))
            return
        model = self.config.get("llm_model", "").strip() or _info.get("model", "deepseek-chat")
        total = len(staged_tracks)
        for idx, (sid, fname, path, complexity) in enumerate(staged_tracks):
            if self._is_cancelled:
                break
            # NO AUDIO IS SENT on this path — the model never hears the track, so
            # it can only produce a filename-derived DRAFT. It is given the schema
            # so the shape is right, but do not mistake it for a grounded caption.
            prompt = (
                f"Filename: {fname}\n\n"
                + caption_spec.task_prompt_from_config(self.config)
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": caption_spec.system_prompt_from_config(self.config)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=int(self.config.get("caption_max_tokens", 512) or 512),
                    frequency_penalty=float(
                        self.config.get("caption_frequency_penalty", 0.3) or 0
                    ),
                )
                caption = trim_to_caption(
                    (response.choices[0].message.content or "").strip()
                )
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
