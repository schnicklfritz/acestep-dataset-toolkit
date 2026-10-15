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
import os

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
    # True = a concept-only placeholder drafted by the assistant (no audio on
    # disk yet). It takes part in the gap audit and can be captioned, but every
    # file-based step skips it, and ``workers/export.py`` filters it out before
    # any exporter runs — so it never reaches training data in ANY format.
    # (``to_export_sample`` alone is not enough: it strips this key but still
    # emits a row, which is why the exclusion lives at the export boundary.)
    "virtual": False,
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
# The app's internal sample fields ARE ACE-Step 1.5's canonical field names.
#
# Verified against the trainer itself, not against any doc:
#   ACE-Step-1.5/docs/sidestep/Dataset Preparation.md -> the "Full ACE-Step
#     format" example lists, in order: id, audio_path, filename, caption,
#     genre, lyrics, raw_lyrics, formatted_lyrics, bpm, keyscale,
#     timesignature, duration, language, is_instrumental, custom_tag, labeled,
#     prompt_override. That is this module's SAMPLE_DEFAULTS, minus the
#     app-internal keys below.
#   ACE-Step-1.5/acestep/training_v2/preprocess_discovery.py:48-50 -> a
#     relative ``audio_path`` resolves against the dataset JSON's own folder.
#   ACE-Step-1.5/acestep/training_v2/preprocess_discovery.py:110-122 ->
#     per-sample metadata is indexed by ``filename``, falling back to the
#     basename of ``audio_path``.
#   ACE-Step-1.5/acestep/training_v2/preprocess_prompt.py:43-52 -> the
#     ``prompt_override`` comparison is against the STRINGS "caption"/"genre".
#
# So the export is a PASS-THROUGH, not a rename. An earlier revision renamed
# ``audio_path``/``filename`` -> ``file_name`` and ``is_instrumental`` ->
# ``instrumental``, and emitted ``prompt_override`` as a bool. ``file_name``
# and ``instrumental`` are ACE-Step *v1* names, not the trainer this toolkit
# feeds, and ``True`` is not a value ``preprocess_prompt`` ever compares
# against -- so each of those renames silently mis-named a real field.
#
# Kept as an (empty) seam in case a future ACE-Step renames fields again.
EXPORT_FIELD_MAP = {}

# App-internal bookkeeping that is not part of ACE-Step's sample schema.
# ``virtual`` is also filtered upstream (``exporters.for_export``); dropping it
# here means a sample can never carry it into a manifest by accident.
EXPORT_DROPPED_FIELDS = (
    "virtual",
    "structural_segments",
    "spatial_tokens",
    "stem_paths",
    "chunk_paths",
    "locked",                   # transient flag the table editor stamps on a row
)

# Keys written by a training export, in ACE-Step 1.5's documented order.
EXPORT_SAMPLE_FIELDS = (
    "id",
    "audio_path",
    "filename",
    "caption",
    "genre",
    "lyrics",
    "raw_lyrics",
    "formatted_lyrics",
    "bpm",
    "keyscale",
    "timesignature",
    "duration",
    "language",
    "is_instrumental",
    "custom_tag",
    "labeled",
    "prompt_override",
)

# Metadata keys ACE-Step does not read. ``instrumental_mode`` is the toolkit's
# legacy key; ``all_instrumental`` carries the same fact in the canonical shape.
EXPORT_METADATA_DROPPED = ("instrumental_mode",)


def _canonical_filename(sample):
    """The bare basename ACE-Step wants in ``filename``.

    ``preprocess_discovery.py:110-122`` indexes per-sample metadata by
    ``filename`` first and only falls back to the basename of ``audio_path``,
    so an unset or pathed ``filename`` is reduced here to keep that key
    unambiguous and to match the documented example (``"track1.mp3"``).
    """
    name = (sample.get("filename") or "").strip()
    if not name:
        name = (sample.get("audio_path") or "").strip()
    return os.path.basename(name)


def to_export_sample(sample):
    """Convert one internal sample dict into ACE-Step 1.5 training JSON.

    A pass-through with three jobs:

    * ``EXPORT_DROPPED_FIELDS`` never reaches training.
    * ``audio_path``, ``filename``, ``is_instrumental``, ``duration`` and
      ``prompt_override`` are always present, so the trainer's silent
      fallbacks (``preprocess.py:295,300`` -> ``duration`` 0 and
      ``is_instrumental`` True) and its string comparison
      (``preprocess_prompt.py:47-52``) see the intended value rather than a
      default.
    * ``prompt_override`` is normalised to ``None`` unless it is one of the two
      strings ACE-Step compares against. It is NOT a boolean and must never be
      coerced into one: an earlier revision emitted ``True``, and the trainer
      ignores every value except ``"caption"``/``"genre"``.
    """
    out = {}
    for key in EXPORT_SAMPLE_FIELDS:
        if key not in sample:
            continue
        out[EXPORT_FIELD_MAP.get(key, key)] = sample[key]

    for key in EXPORT_DROPPED_FIELDS:
        out.pop(key, None)

    # Canonical values, always present, with the same defaults the trainer
    # assumes when a key is missing.
    out["filename"] = _canonical_filename(sample)
    if not (out.get("audio_path") or "").strip():
        # The trainer falls back to ``filename`` itself
        # (``preprocess_discovery.py:50``); filling it here keeps the manifest
        # self-sufficient.
        out["audio_path"] = out["filename"]
    out["duration"] = sample.get("duration", 0) or 0
    out["is_instrumental"] = bool(sample.get("is_instrumental"))

    override = sample.get("prompt_override")
    out["prompt_override"] = override if override in ("caption", "genre") else None
    return out


def to_export_dataset(dataset):
    """Whole-dataset conversion for training export.

    Emits the canonical two-layer shape ACE-Step 1.5 accepts: a top-level
    ``metadata`` block plus ``samples``
    (``docs/sidestep/Dataset Preparation.md``, "Full ACE-Step format").

    Regression this guards: an earlier revision of ``exporters.export_json``
    bypassed this function and hand-rolled ``{"metadata": {}, ...}``, so
    ``custom_tag``, ``tag_position`` and ``genre_ratio`` were discarded
    entirely -- no trigger word and no caption/genre policy ever reached
    training (``preprocess_discovery.py:145-177`` reads all three from here).
    """
    samples = dataset.get("samples", []) or []
    meta = {
        key: value
        for key, value in dict(dataset.get("metadata", {}) or {}).items()
        if key not in EXPORT_METADATA_DROPPED
    }
    # The manifest must describe itself. ``normalize_dataset`` recomputes this on
    # load/save, but a caller that never normalized (a test, an export script)
    # would otherwise ship ``num_samples: 0`` next to N real samples. Assigning
    # in place keeps the canonical key order when it was already present.
    meta["num_samples"] = len(samples)
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

# ``LANGS`` entry that is NOT a language: "this track has no vocals at all".
#
# WHY IT IS AN OPTION ON ``language`` RATHER THAN A FIELD OF ITS OWN
# ---------------------------------------------------------------
# The two facts are the same fact. A track with no vocals has no language to
# declare, and a track that declares a language does have vocals — so a second
# boolean field is a second chance to disagree with the first. ``language``
# therefore carries the instrumental state: blank means instrumental, and the
# selector offers the word so nobody has to know that convention.
#
# WHAT IT DOES *NOT* DO
# ---------------------
# It is never written to ``language``. The word is a UI sentinel, not an ISO
# code, and ACE-Step's loader is only documented with real codes (docs/aceinfo.txt),
# so a pseudo-code would reach training data as an unknown value. Choosing it
# clears ``language`` and raises ``is_instrumental``, which the exporter writes
# as ``instrumental: true``.
LANG_INSTRUMENTAL = "instrumental"

# The one way the app turns a Language choice into dataset state. Every editor
# that offers the language list (the table cell, the bulk panel) routes through
# here, so they cannot disagree about what "instrumental" means.
def apply_language_choice(sample, lang):
    """Write a Language choice onto ``sample``; returns the resulting
    instrumental flag.

    ``"instrumental"`` -> ``language=""`` and ``is_instrumental=True``.
    Any other value  -> that value, and ``is_instrumental=False`` (declaring a
    language is a statement that the track has vocals).

    Both directions are written because both are the *user's explicit choice*.
    Deliberately NOT applied by ``normalize_dataset``: that would silently
    rewrite existing datasets, e.g. one holding ``language="en"`` next to
    ``is_instrumental=true``. Re-saving only normalizes what was edited.
    """
    lang = (lang or "").strip()
    if lang == LANG_INSTRUMENTAL:
        sample["language"] = ""
        sample["is_instrumental"] = True
        return True
    sample["language"] = lang
    sample["is_instrumental"] = False
    return False
