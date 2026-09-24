"""Default application configuration."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.json"

# Keys whose values are API credentials — stored in the encrypted store
# (OS keyring / secrets.enc), never in settings.json.
SECRET_KEYS = {"kaggle_key", "custom_key", "mvsep_api_key", "gemini_api_key", "hf_token",
               "openrouter_key", "groq_key", "deepseek_key"}

DEFAULT_CONFIG = {
    "kaggle_user": "",
    "kaggle_key": "",
    "custom_url": "",
    "custom_key": "",       # DeepSeek API key (secret)
    "mvsep_api_key": "",    # MVSEP API key (secret)
    "mvsep_first_stage": "",    # render_id of the first separation stage (empty = BS PolarFormer)
    "mvsep_full_chain": True,   # run first-stage -> multi-stem chain by default
    "humanize_presets": [],     # user-entered humanization presets (free-form)
    "tag_caption_ratio": 0,     # % of tracks using tag-style prompts (0 = all captions, 100 = all tags)
    "kaggle_model_dataset": "michelmoalem9b/acestep-captioner-model",  # cached captioner weights
    # Suppress the "Before You Add Tracks" dataset-quality warnings dialog.
    "suppress_add_track_warnings": False,
    # ---- MOSS (open model captioning on a Kaggle GPU) ----
    # MOSS-Music is the music-specialised model ("music-captioning, lyrics-asr,
    # chord-recognition"); MOSS-Audio is general audio. Both are ~17 GiB and
    # shard across Kaggle's two T4s. Swap to
    # "OpenMOSS-Team/MOSS-Audio-8B-Instruct" for the general model -- the kernel
    # derives the repo and class names from whichever id is set.
    "moss_model_id": "OpenMOSS-Team/MOSS-Music-8B-Instruct",
    # Optional: a private Kaggle dataset holding the weights. Set this to skip a
    # ~17 GiB download inside the kernel on every run.
    "moss_model_dataset": "",
    # Empty = use the built-in prompts in workers/kaggle_moss.py.
    "moss_style_prompt": "",
    "moss_lyrics_prompt": "",
    "moss_max_tokens": 1024,
    # One MOSS pass covers ~120 s (hard encoder limit); 110 leaves headroom.
    "moss_chunk_seconds": 110,
    # Structural Tag Creator: which artist's vocabulary to offer the model.
    # Empty = use the dataset name as a hint ("sabbath" -> Black Sabbath).
    # Unrecognised = cross-artist fundamentals only. See docs/vocabulary.json.
    "tag_creator_artist": "",
    # Attention backend. Empty = leave it to the model/transformers.
    #
    # WHY THIS IS EMPTY BY DEFAULT: MOSS's audio encoder config pins
    # "_attn_implementation": "eager" for the Whisper layers, and the encoder
    # does deepstack feature injection through forward hooks. Forcing a
    # different backend at the top level might not reach the encoder, and
    # changing it is untested -- so the default stays out of the way.
    #
    # Options are "sdpa" / "flash_attention_2" / "eager". Note that
    # flash-attn does NOT support Turing (T4, sm_75) at all; its README points
    # Turing users at a separate fork with only a subset of features. It is only
    # worth setting on an Ampere+ allocation (A100 / L4), and installing it
    # needs `pip install flash-attn --no-build-isolation` plus a multi-minute
    # CUDA compile.
    "moss_attn_implementation": "",
    # ---- Lyrics tidy (contraction -> phonetic table + -ing exceptions) ----
    # Empty = use the shipped defaults in modules/lyrics_normalizer.py.
    "lyrics_contractions": {},
    "lyrics_ing_exceptions": [],
    # Per-secret "remember on this device" policy (non-secret, kept in settings.json).
    # When True the secret is stored encrypted (OS keyring / secrets.enc); when
    # False it is used for the current session only and never persisted.
    "remember_kaggle_key": True,
    "remember_custom_key": True,
    "remember_mvsep_api_key": True,
    # ---- Pipeline & model defaults (all overridable in ⚙ Settings) ----
    # ---- Caption schema + prompts ----
    # The ACE-Step 1.5XL caption schema itself is BUILT IN and not editable:
    # modules/caption_spec.py encodes docs/ACE_Step_1.5_Master_Annotation_Guide.md
    # §1/§3 and docs/descriptor_reference.md §10, and every backend is given it as
    # the SYSTEM prompt.
    #
    # caption_system_prompt: ADDITIONAL user instructions, appended to the schema
    # (house style, per-artist emphasis). It can never replace the schema.
    "caption_system_prompt": "",
    # caption_prompt: the USER TURN — what to do with this clip. Kept as a key for
    # backward compatibility. The old shipped default said "Write 3 to 5 sentences.
    # Start with A or An", which CONTRADICTS the schema's front-loaded tag list, so
    # caption_spec.task_prompt_from_config() ignores that value if it is the
    # unmodified default (see LEGACY_CAPTION_PROMPT there).
    "caption_prompt": "",
    # Decoding controls for the GPU captioners. Greedy decoding with no penalty is
    # what let a caption loop on a lyric refrain ~200 times until the token cap.
    "caption_repetition_penalty": 1.15,   # 1.0 = off
    "caption_no_repeat_ngram": 6,         # tokens; 0 = off
    "caption_max_tokens": 512,
    "caption_max_audio_duration": 120,   # seconds (0 = whole file)
    "caption_batch_size": 1,             # chunks per captioner forward pass on the Kaggle GPU
    # caption_prompt_addendum: EXTRA text APPENDED to the user turn (the task
    # prompt). The schema lives in the SYSTEM prompt and cannot be replaced; this
    # only adds run-specific emphasis ("1970s live bootleg", "Bon Scott era").
    # It is ignored when an explicit caption_prompt override is passed (e.g. the
    # instrument-only prompt from "Detect via Captioner").
    "caption_prompt_addendum": "",
    # ---- ACE-Step Kaggle run plumbing ----
    # LOCAL folder the downloaded captions_out.json is written to. Kaggle only
    # persists /kaggle/working, so the path the user can actually choose is the
    # DOWNLOAD destination, not a path inside the kernel. Empty = ask with a
    # folder picker at run time.
    "caption_output_dir": "",
    # Persistent LOCAL staging folder whose contents ARE the uploaded Kaggle
    # dataset. Keeping it on disk (instead of a tempfile that dies with the run)
    # is what makes "add or remove songs from the uploaded dataset" possible.
    # Empty = ~/acestep_kaggle_staging.
    "caption_staging_dir": "",
    # The private Kaggle dataset slug used for uploads ("user/slug"). Remembered
    # so later runs push a NEW VERSION of the same dataset (dataset_create_version)
    # instead of creating a new random-slug dataset on every run. Empty = create
    # one on the next upload and remember it.
    "caption_audio_dataset": "",
    # Transcode the staged audio to MP3 before upload: smaller dataset, and one
    # codec instead of a mix of flac/wav/m4a.
    "caption_convert_mp3": True,
    "caption_mp3_bitrate": "192k",
    # Hold every caption from a run as a PROPOSAL and review them in one diff
    # table at the end, instead of opening a modal dialog per track. Off = the
    # old one-dialog-per-track behaviour.
    "caption_batch_review": True,
    # ---- Caption credentials (see dataset_manager._resolve_caption_backend) ----
    # Backends the user has ALREADY answered the credentials prompt for, so it is
    # asked once per backend instead of on every run. Credentials always override
    # this memory: if a key exists, Kaggle is used without asking again.
    "caption_cred_prompt_seen": [],
    # The backend the user picked when they declined to enter Kaggle credentials.
    # Reused for that backend's later runs instead of re-asking. Empty = ask.
    "caption_fallback_backend": "",
    # Mark captions produced by a backend that never HEARD the audio (the local
    # rule engine's canned text, DeepSeek's filename-only draft) so a placeholder
    # can never be mistaken for a grounded caption. Off = no stamp.
    "caption_stamp_placeholders": True,
    "segment_min_sec": 12.0,
    "segment_max_k": 20,
    "structure_backend": "librosa",   # librosa (default) | songformer (functional labels, Kaggle)
    "kaggle_stem_model": "htdemucs_ft",
    "stem_output_dir": "",               # empty = default location
    "dsp_target_lufs": -14.0,
    "dsp_target_sr": 44100,
    # ---- Lyrics transcription ----
    "lyrics_engine": "kaggle",   # kaggle (default, GPU) | whisperx | gemini | acestep_transcriber (experimental)
    "lyrics_language": "",            # empty = auto-detect
    "lyrics_initial_prompt": "",      # e.g. "1970s hard rock by Black Sabbath"
    # ---- Caption backend (pluggable providers) ----
    #   ace_step  = ACE-Step captioner (Qwen2.5-Omni) on a Kaggle GPU  [default]
    #   moss      = MOSS-Audio (open model) on a Kaggle GPU — raw style + lyrics
    #   gemini    = Google Gemini (audio-native)
    #   deepseek  = DeepSeek LLM (text-only synthesis)
    #   custom    = any OpenAI-compatible endpoint (vLLM / Ollama / local or rented GPU)
    "caption_backend": "ace_step",
    "gemini_api_key": "",             # (secret)
    "gemini_model": "gemini-2.5-flash",
    "custom_caption_url": "",         # OpenAI-compatible base URL, e.g. http://localhost:8000/v1
    "custom_caption_model": "",       # model name served by the endpoint
    "custom_caption_audio": False,    # send audio via OpenAI input_audio when the model supports it
    "remember_gemini_key": True,
    # ---- Instrument tagging + content-aware recommendations ----
    # "auto" = use CLAP when torch+transformers are installed, else spectral only.
    "use_clap_tagger": "auto",
    "auto_recommend_models": True,    # feed detected instruments into Stage-3 model selection
    # Lead/backing vocal split: off | mvsep | heuristic (experimental).
    "lead_vocal_splitter": "off",
    # ---- Model download manager (Piece 3) ----
    "hf_token": "",                   # (secret) Hugging Face token for model downloads
    "model_download_source": "hf",    # hf | github
    "model_dir": "models",            # local dir for downloaded models
    "remember_hf_token": True,
    # ---- Pluggable LLM provider (aggregation, recommendations, assistant) ----
    # deepseek (default, paid) | gemini (free) | groq (free) | openrouter (free) | local
    "llm_provider": "deepseek",
    "llm_model": "",                  # empty = provider default
    "llm_base_url": "",               # empty = provider default
    # Per-role overrides (empty = use the global provider/model above):
    # the aggregator, captioner, and assistant can each use a different model.
    "llm_provider_aggregator": "",
    "llm_model_aggregator": "",
    "llm_provider_captioner": "",
    "llm_model_captioner": "",
    "llm_provider_assistant": "",
    "llm_model_assistant": "",
    "deepseek_key": "",               # (secret) official DeepSeek API key
    "remember_deepseek_key": True,
    "openrouter_key": "",             # (secret)
    "groq_key": "",                   # (secret)
    "remember_openrouter_key": True,
    "remember_groq_key": True,
    # ---- AI assistant ----
    "assistant_remember": True,        # persist the conversation across sessions
    "assistant_context_size": 40,      # max messages kept in context
    "assistant_linear_thinking": True, # step-by-step reasoning
}

