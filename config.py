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
    # Per-secret "remember on this device" policy (non-secret, kept in settings.json).
    # When True the secret is stored encrypted (OS keyring / secrets.enc); when
    # False it is used for the current session only and never persisted.
    "remember_kaggle_key": True,
    "remember_custom_key": True,
    "remember_mvsep_api_key": True,
    # ---- Pipeline & model defaults (all overridable in ⚙ Settings) ----
    "caption_prompt": (
        "You are a professional music metadata tagger preparing training data for ACE-Step. "
        "Listen carefully to this audio clip and write a detailed description. "
        "Cover: specific instrumentation (name every instrument you hear), "
        "whether vocals are present (gender, register, timbre) or confirm instrumental, "
        "recording and production character, mood, and how the clip develops. "
        "Write 3 to 5 sentences. Start with A or An. "
        "Genre, BPM, key, and time signature are handled separately — do not include them."
    ),
    "caption_max_tokens": 512,
    "caption_max_audio_duration": 120,   # seconds (0 = whole file)
    "caption_batch_size": 1,             # chunks per captioner forward pass on the Kaggle GPU
    "segment_min_sec": 12.0,
    "segment_max_k": 20,
    "structure_backend": "librosa",   # librosa (default) | songformer (functional labels, Kaggle)
    "kaggle_stem_model": "htdemucs_ft",
    "stem_output_dir": "",               # empty = default location
    "dsp_target_lufs": -14.0,
    "dsp_target_sr": 44100,
    # ---- Lyrics transcription ----
    "lyrics_engine": "whisperx",      # whisperx | gemini | acestep_transcriber (experimental)
    "lyrics_language": "",            # empty = auto-detect
    "lyrics_initial_prompt": "",      # e.g. "1970s hard rock by Black Sabbath"
    # ---- Caption backend (pluggable providers) ----
    #   ace_step  = ACE-Step captioner (Qwen2.5-Omni) on a Kaggle GPU  [default]
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

