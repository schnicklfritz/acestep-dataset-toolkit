"""Length guard for captions — deliberately NOT a conformance checker.

Caption TEXT is judged by a human or an LLM, never by this module (2026-11-18).
Captions are too varied to be scored by a script, and the rule this module used
to enforce — "front-load 5-12 tags, then 8-16 comma-separated acoustic facts" —
was a SONG-GENERATION prompt convention, not an ACE-Step LoRA training
requirement. Upstream's own LoRA recipe disagrees with it: the tutorial's caption
example (``docs/en/LoRA_Training_Tutorial.md`` in the ACE-Step source tree) is a
one-line prose description, and the official LoRA captioner script
(``scripts/lora_data_prepare/gemini_caption.py``) constrains the shape not at
all. Enforcing the generation-prompt shape on training captions therefore
flagged correct captions as broken.

What remains is ``trim_to_caption``, and it is a RUNAWAY guard, not a quality
check: a provider's ``max_tokens`` cannot be trusted (a caption came back at
~2000 words from an endpoint configured with 512), and the trainer's prompt is
capped at 256 tokens with the ``# Metas`` block AFTER the caption, so an
over-long caption silently deletes the bpm/keyscale/timesignature lines.
"""
import re

from modules.caption_spec import CAPTION_MAX_WORDS, TAG_MAX


def trim_to_caption(text, max_words=CAPTION_MAX_WORDS):
    """Trim an over-long caption back under the word budget.

    Keeps the leading comma-separated list (capped at ``TAG_MAX`` items) and
    then as many whole comma-separated units as fit. Whole units only — never a
    cut mid-unit — so the result is still usable training data.
    """
    body = (text or "").strip()
    head, _, tail = body.partition(".")
    tags = [t.strip() for t in head.split(",") if t.strip()]

    # Nothing to do? Return the original string untouched, byte for byte.
    if len(body.split()) <= max_words and len(tags) <= TAG_MAX:
        return body

    keep = tags[:TAG_MAX]
    head_text = ", ".join(keep) + "." if keep else ""
    budget = max_words - len(", ".join(keep).split())
    facts = []

    for fact in re.split(r"[,;]|(?<=[.!?])\s+", tail.strip()):
        fact = fact.strip().strip(".!?;,").strip()
        cost = len(fact.split())
        if not cost:
            continue
        if cost > budget:
            # One run-on unit larger than the whole budget: keep as many words
            # as fit rather than dropping the tail entirely.
            keep_words = fact.split()[:max(0, budget)]
            if keep_words:
                facts.append(" ".join(keep_words).rstrip(",;:"))
            break
        facts.append(fact)
        budget -= cost

    if not facts:
        return head_text
    return ((head_text + " ") if head_text else "") + ", ".join(facts) + "."
