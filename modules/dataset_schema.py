"""Dataset JSON schema — single source of truth.

Both the app and the exporter/validator read the shape from here so a field is
added or renamed in exactly one place.

Two-layer model (see SPECIFICATION.md):

``metadata``  — dataset-wide properties set once.
``samples[]`` — one dict per track.

BACKWARD COMPATIBILITY
----------------------
Older files predate several fields. ``normalize_dataset()`` backfills defaults
without discarding anything, so a legacy ``dataset.json`` loads cleanly and is
written back in the current shape. A legacy ``instrumental_mode`` string is
translated *into* the boolean ``all_instrumental`` while the original key is
left in place, so nothing downstream breaks mid-migration.
"""
import datetime

# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
# Dataset-wide defaults. Order is deliberate: it is the order written to JSON.
METADATA_DEFAULTS = {
    "name": "",
    "custom_tag": "",
    "tag_position": "prepend",      # prepend | append | none
    "created_at": "",               # ISO-8601, filled on first save of a new dataset
    "num_samples": 0,
    "all_instrumental": False,      # every track is instrumental
    "genre_ratio": 0,               # 0-100: % of tracks using genre-style prompts
    # Legacy key retained for older consumers; kept in sync from all_instrumental
    # on write. Not a primary field any more.
    "instrumental_mode": "mixed",   # mixed | all_instrumental | no_instrumentals
}

# ---------------------------------------------------------------------------
# samples[] — canonical per-track fields
# ---------------------------------------------------------------------------
SAMPLE_DEFAULTS = {
    "id": "",
    "audio_path": "",               # editable; relative or absolute
    "filename": "",
    "caption": "",                  # style/production description
    "genre": "",
    "lyrics": "",                   # lyrics as given (may include [Section] tags)
    "raw_lyrics": "",               # pre-formatting lyrics (e.g. straight from ASR)
    "formatted_lyrics": "",         # [Section]-tagged lyric block for training
    "bpm": 0,
    "keyscale": "",
    "timesignature": "",
    "duration": 0,
    "language": "en",
    "is_instrumental": False,
    "custom_tag": "",
    "labeled": False,               # metadata has been reviewed/verified
    # ACE-Step data-source switch (BOOLEAN, per track):
    #   False -> the loader may auto-label this sample with the captioner LLM
    #   True  -> the loader bypasses auto-labelling and uses the handwritten
    #            genre / caption / lyrics for this track verbatim
    "prompt_override": False,
    # Structural / spatial pipelines
    "structural_segments": [],
    "spatial_tokens": {},
    "stem_paths": {},
    "chunk_paths": [],
}



# ---------------------------------------------------------------------------
# ACE-Step export mapping
# ---------------------------------------------------------------------------
# The app's internal field names differ from the names ACE-Step's training
# preprocessor expects. Emitting the internal names would make the preprocessor
# silently fail to parse those values (falling back to NaN / defaults), so the
# JSON handed to training must use the authoritative keys below.
#
#   internal            ->  ACE-Step key     type
#   audio_path          ->  file_name        str
#   is_instrumental     ->  instrumental     bool
#
# Everything else is emitted under its internal name.
EXPORT_FIELD_MAP = {
    "filename": "file_name",
    "audio_path": "file_name",
    "is_instrumental": "instrumental",
}

# Keys written by a training-export (in order). `audio_path` and `filename`
# both map to file_name, so only the first present is used.
EXPORT_SAMPLE_FIELDS = (
    "id",
    "filename",
    "genre",
    "caption",
    "prompt_override",
    "lyrics",
    "bpm",
    "keyscale",
    "timesignature",
    "language",
    "is_instrumental",
)


def to_export_sample(sample):
    """Convert one internal sample dict into ACE-Step training JSON.

    Applies ``EXPORT_FIELD_MAP`` and drops app-internal bookkeeping
    (``locked``, ``spatial_tokens``, ``structural_segments``, ``raw_lyrics``,
    ``labeled``, ``stem_paths``, ``chunk_paths``) that training does not read.

    ``file_name`` is taken from ``audio_path`` when present (the pointer the
    loader needs), otherwise from ``filename``.
    """
    out = {}
    for key in EXPORT_SAMPLE_FIELDS:
        if key not in sample:
            continue
        target = EXPORT_FIELD_MAP.get(key, key)
        if target == "file_name":
            # Prefer the real path; filename is only a fallback.
            if out.get("file_name"):
                continue
            value = sample.get("audio_path") or sample.get("filename") or ""
        else:
            value = sample[key]
        out[target] = value

    # prompt_override must be an explicit boolean for the loader.
    out["prompt_override"] = bool(sample.get("prompt_override"))
    return out


def to_export_dataset(dataset):
    """Whole-dataset conversion for training export."""
    samples = dataset.get("samples", []) or []
    meta = dict(dataset.get("metadata", {}) or {})
    return {
        "metadata": meta,
        "samples": [to_export_sample(s) for s in samples],
    }


def new_metadata(**overrides):
    """Fresh metadata block (``created_at`` stamped)."""
    meta = dict(METADATA_DEFAULTS)
    meta["created_at"] = datetime.datetime.now().isoformat()
    meta.update(overrides)
    return meta


def new_dataset(**metadata_overrides):
    """Fresh, empty dataset in the current schema."""
    return {"metadata": new_metadata(**metadata_overrides), "samples": []}


def new_sample(**overrides):
    """Fresh per-track dict with every canonical field present."""
    sample = dict(SAMPLE_DEFAULTS)
    # Mutable defaults must not be shared between samples.
    sample["structural_segments"] = []
    sample["spatial_tokens"] = {}
    sample["stem_paths"] = {}
    sample["chunk_paths"] = []
    sample.update(overrides)
    return sample


def derive_instrumental_mode(all_instrumental, samples=None):
    """Map the boolean back onto the legacy 3-state string."""
    if all_instrumental:
        return "all_instrumental"
    if samples and all(s.get("is_instrumental") is False for s in samples):
        return "no_instrumentals"
    return "mixed"


def normalize_dataset(dataset):
    """Backfill missing fields in place and return the dataset.

    Never overwrites a value that is present. Legacy ``instrumental_mode`` is
    translated into ``all_instrumental``; the legacy key is preserved.
    """
    if not isinstance(dataset, dict):
        return dataset

    dataset.setdefault("samples", [])
    meta = dataset.setdefault("metadata", {})

    # Legacy -> current translation (before defaults so we can read the old key).
    legacy_mode = meta.get("instrumental_mode")
    if "all_instrumental" not in meta and legacy_mode is not None:
        meta["all_instrumental"] = legacy_mode == "all_instrumental"

    for key, default in METADATA_DEFAULTS.items():
        if key not in meta:
            meta[key] = list(default) if isinstance(default, list) else default

    # Stamp created_at for datasets that predate the field.
    if not meta.get("created_at"):
        meta["created_at"] = datetime.datetime.now().isoformat()

    # Keep the legacy string consistent with the boolean.
    meta["instrumental_mode"] = derive_instrumental_mode(
        bool(meta.get("all_instrumental")), dataset["samples"]
    )

    if not isinstance(meta.get("num_samples"), int) or meta["num_samples"] < 0:
        meta["num_samples"] = 0
    meta["num_samples"] = len(dataset["samples"])

    for sample in dataset["samples"]:
        if not isinstance(sample, dict):
            continue
        for key, default in SAMPLE_DEFAULTS.items():
            if key not in sample:
                if isinstance(default, list):
                    sample[key] = []
                elif isinstance(default, dict):
                    sample[key] = {}
                else:
                    sample[key] = default
        # A sample with lyrics can never be instrumental.
        if (sample.get("raw_lyrics") or sample.get("lyrics")) and sample.get("is_instrumental"):
            sample["is_instrumental"] = False

    return dataset

# Fields that must be present for ACE-Step training. Used by the validator.
REQUIRED_METADATA_FIELDS = [
    "name", "custom_tag", "tag_position", "num_samples",
]
REQUIRED_SAMPLE_FIELDS = [
    "id", "audio_path", "filename", "caption", "genre", "lyrics",
    "formatted_lyrics", "bpm", "keyscale", "timesignature", "duration",
    "language", "is_instrumental", "custom_tag",
]

# Valid choices, for UI population and validation.
TAG_POSITIONS = ("prepend", "append", "none")
INSTRUMENTAL_MODES = ("mixed", "all_instrumental", "no_instrumentals")
LANGS = (
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da", "fi", "pl",
    "ru", "uk", "cs", "el", "tr", "ar", "he", "hi", "ja", "ko", "zh", "th",
    "vi", "id", "instrumental",
)
TIME_SIGNATURES = ("4/4", "3/4", "6/8", "2/4", "5/4", "7/8", "12/8", "4")
