"""Token budgets for ACE-Step 1.5 text conditioning (captions and vocals).

WHY THIS EXISTS
---------------
ACE-Step truncates text without telling anyone. Every text entry point passes
``truncation=True``, so lyrics longer than the cap are quietly cut and the model
simply never sings the rest. The caps are not in any user-facing document — they
are integer literals in the source — so nothing in this app could report them.

The caps, read out of the ACE-Step 1.5 tree (measured, not estimated):

    stream                      cap    where
    --------------------------  -----  ----------------------------------------
    caption / DiT text prompt    256   core/generation/handler/conditioning_text.py
                                       (``max_length=256``, ``truncation=True``)
    lyrics — training           512    training/dataset_builder_modules/
                                       preprocess_lyrics.py (``max_length=512``)
    lyrics — inference         2048    core/generation/handler/conditioning_text.py
                                       (``max_length=2048``, ``truncation=True``)

There is NO "2 x 256 window" mechanism: the ``chunk_masks`` passed around the
generator are an AUDIO-LATENT-TIME mask and have nothing to do with text. The
lyrics are one contiguous token sequence, cut once at the cap above.

The 512 is the number this app enforces: it writes TRAINING datasets, and the
trainer's ``encode_lyrics`` is the function that reads the lyric files.

THE TOKENIZER
-------------
The caps are counted in tokens of ``Qwen3-Embedding-0.6B`` — the DiT's text
encoder. ``core/generation/handler/init_service_loader_components.py`` builds
``AutoTokenizer.from_pretrained(<checkpoint>/Qwen3-Embedding-0.6B)``, and the
checkpoint catalog (``acestep/api/model_download.py``) maps that folder to the
Hugging Face repo ``ACE-Step/Ace-Step1.5``.

We read the SAME ``tokenizer.json`` that ``transformers`` would, but with the
Rust ``tokenizers`` library: no torch, no transformers, no ~1.2 GB download and
no GPU. Because it is literally the same file, ``len(encode(text).ids)`` is the
token count the model sees, EOS token and all, so the count matches the model's
own truncation exactly.

When that file is absent the counter falls back to a character estimate and the
UI SAYS SO, rather than printing a precise-looking number that cannot be backed
up. ``download_tokenizer()`` fetches the real file on demand (one time, ~11 MB).
"""
import math
import threading
from dataclasses import dataclass
from pathlib import Path

from config import BASE_DIR

# --- The caps (source of truth: the ACE-Step 1.5 tree, see module docstring) --
CAPTION_TOKEN_LIMIT = 256
LYRIC_TRAIN_TOKEN_LIMIT = 512
LYRIC_INFER_TOKEN_LIMIT = 2048

#: What this app enforces by default: it writes training data, so the trainer's
#: cap is the one that governs whether a lyric file survives preprocessing.
DEFAULT_LYRIC_LIMIT = LYRIC_TRAIN_TOKEN_LIMIT

# --- Where the exact tokenizer lives -----------------------------------------
CHECKPOINT_DIRNAME = "Qwen3-Embedding-0.6B"
#: Same layout ACE-Step itself uses: ``<checkpoint>/Qwen3-Embedding-0.6B``.
#: ``models/`` is gitignored, so the fetched file is never committed.
DEFAULT_TOKENIZER_PATH = BASE_DIR / "models" / CHECKPOINT_DIRNAME / "tokenizer.json"
#: The exact file ``transformers.AutoTokenizer`` would read for that folder.
TOKENIZER_URL = (
    "https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/main/"
    "Qwen3-Embedding-0.6B/tokenizer.json"
)

#: Chars per token for the FALLBACK only. ~3.7 is what the toolkit's own lyrics
#: measure (structural tags, punctuation and line breaks push it under the 4.0
#: rule of thumb). Never used when the real tokenizer is present.
FALLBACK_CHARS_PER_TOKEN = 3.7


@dataclass(frozen=True)
class Measurement:
    """One text's size against a limit."""

    tokens: int
    chars: int
    #: True when ``tokens`` came from the real Qwen3 tokenizer, False when it is
    #: the character estimate. The UI must surface this, not hide it.
    exact: bool

    def over_by(self, limit):
        """How many tokens past ``limit`` this text is (0 when within budget)."""
        return max(0, self.tokens - limit)

    def is_over(self, limit):
        return self.tokens > limit


# --- tokenizer loading (cached; a failure is NOT cached, so a later download
# --- is picked up without a restart) -----------------------------------------
_cache = {}
_cache_lock = threading.Lock()


def _get_tokenizer(path):
    """Return a loaded ``tokenizers.Tokenizer`` for ``path``, or None."""
    key = str(path)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    tok = None
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415 — optional dependency

        tok = Tokenizer.from_file(key)
    except Exception:  # noqa: BLE001 — a counter must never break the editor
        tok = None
    if tok is not None:
        with _cache_lock:
            _cache[key] = tok
    return tok


def clear_cache():
    """Forget every loaded tokenizer (call after a fresh download)."""
    with _cache_lock:
        _cache.clear()


def _resolve(tokenizer_path):
    return Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH


def have_tokenizer(tokenizer_path=None):
    """True when the exact Qwen3 tokenizer is available at ``tokenizer_path``."""
    return _get_tokenizer(_resolve(tokenizer_path)) is not None


def measure(text, tokenizer_path=None):
    """Measure ``text`` in Qwen3 tokens, falling back to a char estimate.

    The returned ``Measurement.exact`` is False on the fallback path so callers
    can (and the UI does) label the number as an estimate.
    """
    text = text or ""
    tok = _get_tokenizer(_resolve(tokenizer_path))
    if not text:
        # An empty field is 0 tokens to the user. The tokenizer would report 1
        # (its EOS token), which reads as a bug in a live counter.
        return Measurement(tokens=0, chars=0, exact=tok is not None)
    if tok is not None:
        try:
            return Measurement(tokens=len(tok.encode(text).ids), chars=len(text), exact=True)
        except Exception:  # noqa: BLE001 — fall through to the estimate
            pass
    return Measurement(
        tokens=int(math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)),
        chars=len(text),
        exact=False,
    )


def download_tokenizer(tokenizer_path=None, timeout=60):
    """Fetch the Qwen3 ``tokenizer.json`` (one time, ~11 MB). Returns the path.

    Written atomically: a dropped connection must not leave a half file that
    ``Tokenizer.from_file`` would then reject on every subsequent call.
    """
    import requests  # already a dependency (see requirements.txt)

    dest = _resolve(tokenizer_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(TOKENIZER_URL, timeout=timeout)
    resp.raise_for_status()
    tmp = dest.with_suffix(".json.part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    clear_cache()
    return dest


# --- config helpers ----------------------------------------------------------
def tokenizer_path_from_config(config):
    """The tokenizer path a manager's config selects ("" = the default path)."""
    raw = (config or {}).get("lyrics_tokenizer_path") or ""
    return Path(raw) if raw else DEFAULT_TOKENIZER_PATH


def lyric_limit_from_config(config):
    """The enforced lyric token cap (0/absent = the trainer's 512).

    Configurable so a user who deliberately targets inference (2048), or wants a
    stricter personal budget, can set it without editing code.
    """
    val = (config or {}).get("lyrics_max_tokens") or 0
    try:
        val = int(val)
    except (TypeError, ValueError):
        val = 0
    return val if val > 0 else DEFAULT_LYRIC_LIMIT
