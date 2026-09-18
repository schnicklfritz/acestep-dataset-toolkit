"""Deterministic checks that a caption conforms to the ACE-Step 1.5XL schema.

Prompts are guidance; this is the backstop. A model told to "front-load 5-12
keywords" can still open with a sentence, invent a BPM, or loop on a lyric phrase
— all three have actually happened in this project, and the last one produced a
caption that repeated "don't go" roughly 200 times.

These are pure functions (no Qt, no network) so they can run at save time, inside
the caption audit, and in the export validator.

DELIBERATELY NOT APPLIED TO LYRICS. Songs repeat refrains on purpose: a
repetition check over lyric text flags legitimate material (a real Doors lyric
repeats "when you're strange" 15 times). Captions are prose ABOUT a song and must
not repeat. Pass captions only.
"""
import re

from modules.caption_spec import (
    CAPTION_MAX_CHARS,
    CAPTION_MAX_WORDS,
    TAG_HARD_MAX,
    TAG_MAX,
    TAG_MIN,
)

# A field label a schema-less model invents: "Genre & Style:", "**Tempo:**".
HEADING = re.compile(r"(?m)^\s*(?:\*\*)?[A-Z][A-Za-z /&]{2,30}(?:\*\*)?\s*:")

# Rule 5: BPM / key / time signature belong in metadata fields, never the text.
BPM_IN_TEXT = re.compile(r"\b\d{2,3}\s*(?:BPM|bpm)\b|\b(?:BPM|bpm)\b")
TIME_SIG_IN_TEXT = re.compile(r"\b\d\s*/\s*(?:4|8)\b|\btime\s*signature\b", re.I)
KEY_IN_TEXT = re.compile(r"\b(?:in\s+)?[A-G](?:#|b)?\s*(?:major|minor)\b")

# Markdown artefacts (rule 6: output only the caption text).
MARKDOWN = re.compile(r"\*\*|^#{1,6}\s|^\s*[-*]\s", re.M)

# Rule 3: a caption must always declare vocal presence and character.
VOCAL_TERMS = (
    "male vocal", "female vocal", "male baritone", "baritone", "tenor",
    "raspy vocal", "raspy vocals", "powerful belting", "clean vocal",
    "clean vocals", "screamed vocal", "screamed vocals", "whispered vocal",
    "whispered vocals", "breathy", "falsetto", "choir", "instrumental",
    "no vocals", "lead vocal", "male voice", "female voice", "spoken word",
)

# Captions are 2-3 sentences of flow narrative after the tag list.
MIN_FLOW_SENTENCES = 2
MAX_FLOW_SENTENCES = 3

# A 3-gram repeated this many times in a caption is a decoding loop, not style.
REPEAT_NGRAM = 3
REPEAT_MIN = 3
# ...or one token repeated back-to-back this many times.
CONSECUTIVE_MIN = 5


def tag_prefix(text):
    """Return the front-loaded comma-separated tag list (before the first '.').

    The schema puts the keywords first and the flow narrative after a full stop,
    so everything before the first period is the tag block.
    """
    head = (text or "").strip().split(".")[0]
    return [t.strip() for t in head.split(",") if t.strip()]


def count_sentences(text):
    """Count sentences in the flow narrative (everything after the tag block)."""
    parts = (text or "").split(".", 1)
    tail = parts[1] if len(parts) > 1 else ""
    return len([s for s in re.split(r"[.!?]+", tail) if len(s.split()) >= 3])


def max_ngram_repeat(text, n=3):
    """Return (count, phrase) for the most-repeated n-gram of words."""
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    if len(words) < n:
        return 0, ""
    counts = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    gram, count = max(counts.items(), key=lambda kv: kv[1])
    return count, " ".join(gram)


def max_consecutive_repeat(text):
    """Return (count, word) for the longest run of one word repeated."""
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    best, best_word, run = 0, "", 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
        else:
            if run > best:
                best, best_word = run, words[i - 1]
            run = 1
    if run > best:
        best, best_word = run, words[-1] if words else ""
    return best, best_word


def check_caption(text):
    """Return a list of schema issues (empty list = the caption conforms)."""
    issues = []
    body = (text or "").strip()
    if not body:
        return ["caption is empty"]

    tags = tag_prefix(body)
    if len(tags) < TAG_MIN:
        issues.append(
            f"only {len(tags)} front-loaded keyword(s); the schema requires "
            f"{TAG_MIN}-{TAG_MAX} before the first sentence"
        )
    elif len(tags) > TAG_HARD_MAX:
        issues.append(f"{len(tags)} keywords exceeds the hard maximum of {TAG_HARD_MAX}")

    if not any(term in body.lower() for term in VOCAL_TERMS):
        issues.append("no vocal descriptor (rule 3: declare vocal presence/character)")

    if HEADING.search(body):
        issues.append("field-label heading found (rule 6: output only caption text)")
    if MARKDOWN.search(body):
        issues.append("markdown found (rule 6: no bold, headings or bullets)")

    if BPM_IN_TEXT.search(body):
        issues.append("BPM in the caption (rule 5: use the metadata field)")
    if TIME_SIG_IN_TEXT.search(body):
        issues.append("time signature in the caption (rule 5)")
    if KEY_IN_TEXT.search(body):
        issues.append("key in the caption (rule 5)")

    sentences = count_sentences(body)
    if sentences < MIN_FLOW_SENTENCES:
        issues.append(
            f"{sentences} flow sentence(s); the schema requires "
            f"{MIN_FLOW_SENTENCES}-{MAX_FLOW_SENTENCES} after the tag list"
        )
    elif sentences > MAX_FLOW_SENTENCES:
        issues.append(
            f"{sentences} flow sentences; the schema allows at most "
            f"{MAX_FLOW_SENTENCES} — a caption is not an essay"
        )

    words = len(body.split())
    if words > CAPTION_MAX_WORDS:
        issues.append(
            f"{words} words; the schema allows {CAPTION_MAX_WORDS} "
            f"(keywords + 2-3 sentences)"
        )
    if len(body) > CAPTION_MAX_CHARS:
        issues.append(f"{len(body)} characters; the schema allows {CAPTION_MAX_CHARS}")

    count, phrase = max_ngram_repeat(body)
    if count >= REPEAT_MIN:
        issues.append(f"degenerate repetition: '{phrase}' x{count} (decoding loop)")
    run, word = max_consecutive_repeat(body)
    if run >= CONSECUTIVE_MIN:
        issues.append(f"degenerate repetition: '{word}' repeated {run}x in a row")

    return issues


def is_conforming(text):
    """True when a caption satisfies every schema check."""
    return not check_caption(text)


def trim_to_caption(text, max_words=CAPTION_MAX_WORDS):
    """Trim an over-long caption back to the schema shape.

    Keeps the front-loaded tag list (capped at ``TAG_MAX`` keywords) and then as
    many whole sentences as fit the word budget. Whole sentences only — never a
    cut mid-sentence — so the result is still usable training data.

    This exists because the provider's ``max_tokens`` cannot be trusted: a caption
    came back at ~2000 words from an endpoint configured with 512.
    """
    body = (text or "").strip()
    head, _, tail = body.partition(".")
    tags = [t.strip() for t in head.split(",") if t.strip()]

    # Nothing to do? Return the original string untouched, byte for byte.
    if len(body.split()) <= max_words and len(tags) <= TAG_MAX:
        return body

    keep = tags[:TAG_MAX]
    parts = [", ".join(keep) + "."] if keep else []
    budget = max_words - len(", ".join(keep).split())

    for sentence in re.split(r"(?<=[.!?])\s+", tail.strip()):
        sentence = sentence.strip()
        cost = len(sentence.split())
        if not cost:
            continue
        if cost > budget:
            # One run-on sentence larger than the whole budget: keep as many
            # words as fit rather than dropping the flow narrative entirely.
            keep_words = sentence.split()[:max(0, budget)]
            if keep_words:
                parts.append(" ".join(keep_words).rstrip(",;:") + ".")
            break
        parts.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
        budget -= cost
    return " ".join(parts).strip()
