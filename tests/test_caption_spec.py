"""Tests for the ACE-Step 1.5XL caption prompt composer and the trim guard.

Caption TEXT is deliberately NOT scored by a script any more (2026-11-18): the
"5-12 tags then 8-16 comma-separated facts" rule was a song-generation prompt
convention, not a LoRA training requirement, and upstream's own LoRA recipe uses
a one-line prose caption. The prompt below still describes the shape the captioner
aims for; judging whether a given caption is good is a human/LLM call.

The load-bearing test here is ``test_every_kernel_placeholder_is_filled``: an
unfilled ``{{X}}`` is VALID Python (a set containing a set), so a missing
substitution reaches Kaggle and dies there with a bare NameError 50+ seconds
later. That already happened once with ``{{ATTN_IMPL}}``.

The second is that the schema-less legacy prompt -- "Write 3 to 5 sentences. Start
with A or An" -- can never come back through the config.
"""
import os
import re

from modules.caption_quality import trim_to_caption
from modules.caption_spec import (
    BUILTIN_SYSTEM_PROMPT,
    CAPTION_DENSITY_RULE,
    CAPTION_FORMULA,
    CAPTION_MAX_WORDS,
    COUNTER_EXAMPLE,
    DEFAULT_TASK_PROMPT,
    GOLD_EXAMPLE,
    LEGACY_CAPTION_PROMPT,
    MAX_DESCRIPTORS,
    TAG_HARD_MAX,
    TAG_MAX,
    TAG_MIN,
    build_system_prompt,
    is_legacy_caption_prompt,
    is_narrative_caption_prompt,
    system_prompt_from_config,
    task_prompt_from_config,
    build_task_prompt,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def _placeholders(text):
    return set(re.findall(r"\{\{[A-Z_]+\}\}", text))


# --------------------------------------------------------------------------
# the composer: the schema is set in stone, the user prompt is additive
# --------------------------------------------------------------------------

def test_builtin_prompt_encodes_the_schema():
    assert CAPTION_FORMULA in BUILTIN_SYSTEM_PROMPT
    assert "5-12 comma-separated keywords" in BUILTIN_SYSTEM_PROMPT
    assert "NEVER put BPM, key, or time signature" in BUILTIN_SYSTEM_PROMPT
    # Few-shot beats instructions: the worthless example must be shown as such.
    assert COUNTER_EXAMPLE in BUILTIN_SYSTEM_PROMPT
    assert GOLD_EXAMPLE in BUILTIN_SYSTEM_PROMPT


def test_builtin_prompt_encodes_the_density_rule():
    """The rule is the schema, and the example must already obey it."""
    assert CAPTION_DENSITY_RULE in BUILTIN_SYSTEM_PROMPT
    assert "2-3 sentences" not in BUILTIN_SYSTEM_PROMPT
    # tag_prefix() splits on the first '.', so the gold example must carry
    # exactly one: the full stop that ends the tag block.
    assert GOLD_EXAMPLE.count(".") == 1


def test_empty_addendum_changes_nothing():
    assert build_system_prompt("") == BUILTIN_SYSTEM_PROMPT
    assert build_system_prompt("   \n  ") == BUILTIN_SYSTEM_PROMPT


def test_user_text_is_appended_not_substituted():
    out = build_system_prompt("House style: name the guitar amp.")
    assert out.startswith(BUILTIN_SYSTEM_PROMPT)
    assert out.endswith("House style: name the guitar amp.")


def test_legacy_schema_less_prompt_is_detected():
    assert is_legacy_caption_prompt(LEGACY_CAPTION_PROMPT)
    assert not is_legacy_caption_prompt("Prefer Black Sabbath's own terms.")


def test_legacy_default_cannot_return_through_the_config():
    """The shipped default contradicts the schema, so it must be ignored."""
    config = {"caption_prompt": LEGACY_CAPTION_PROMPT}
    prompt = system_prompt_from_config(config)
    assert "Start with A or An" not in prompt
    assert prompt == BUILTIN_SYSTEM_PROMPT
    assert task_prompt_from_config(config) == DEFAULT_TASK_PROMPT


def test_a_stored_narrative_prompt_cannot_return_through_the_config():
    """settings.json may still hold the retired "2-3 sentences" instruction."""
    stale = ("Front-load the keywords, then 2-3 sentences describing how the "
             "energy moves through the track.")
    assert is_narrative_caption_prompt(stale)
    assert system_prompt_from_config({"caption_system_prompt": stale}) == (
        BUILTIN_SYSTEM_PROMPT)


def test_a_genuine_user_edit_is_still_honoured():
    config = {"caption_prompt": "Always name the drum kit.",
              "caption_system_prompt": ""}
    assert task_prompt_from_config(config) == "Always name the drum kit."


def test_system_prompt_config_wins_over_the_legacy_key():
    config = {"caption_system_prompt": "House style.", "caption_prompt": "ignored"}
    assert system_prompt_from_config(config).endswith("House style.")


# --------------------------------------------------------------------------
# the USER TURN add-on (the "add to the caption prompt" field)
# --------------------------------------------------------------------------

def test_empty_task_addendum_changes_nothing():
    assert build_task_prompt("") == DEFAULT_TASK_PROMPT
    assert build_task_prompt("   \n ", base="") == DEFAULT_TASK_PROMPT
    assert build_task_prompt("", base="Always name the drum kit.") == \
        "Always name the drum kit."


def test_task_addendum_is_appended_and_cannot_replace_the_task_line():
    out = build_task_prompt("This is a 1970s live bootleg.")
    assert out.startswith(DEFAULT_TASK_PROMPT)
    assert out.endswith("This is a 1970s live bootleg.")


def test_task_addendum_from_config_reaches_the_user_turn():
    out = task_prompt_from_config({"caption_prompt_addendum": "Name the amp."})
    assert out.startswith(DEFAULT_TASK_PROMPT)
    assert out.endswith("Name the amp.")


def test_task_addendum_survives_the_legacy_default_being_ignored():
    """The old contradictory default is dropped, the add-on must still apply."""
    config = {"caption_prompt": LEGACY_CAPTION_PROMPT,
              "caption_prompt_addendum": "Name the amp."}
    out = task_prompt_from_config(config)
    assert "Start with A or An" not in out
    assert out.startswith(DEFAULT_TASK_PROMPT)
    assert out.endswith("Name the amp.")


def test_task_addendum_stacks_on_a_genuine_user_edit():
    config = {"caption_prompt": "Always name the drum kit.",
              "caption_prompt_addendum": "Mention the room."}
    out = task_prompt_from_config(config)
    assert out.startswith("Always name the drum kit.")
    assert out.endswith("Mention the room.")


# --------------------------------------------------------------------------
# the schema numbers the prompt and the trim guard share
# --------------------------------------------------------------------------

def test_schema_numbers_are_the_documented_ones():
    assert (TAG_MIN, TAG_MAX, TAG_HARD_MAX) == (5, 12, 15)
    assert MAX_DESCRIPTORS == 3


def test_trim_is_a_no_op_for_a_caption_under_budget():
    assert trim_to_caption(GOLD_EXAMPLE) == GOLD_EXAMPLE


TAGS = "hard rock, drums, bass, male vocal, raw production, 1970s analog"
TOME = TAGS + ". " + "The band plays with conviction and the guitars ring out. " * 40


def test_trim_brings_a_tome_back_under_the_budget():
    trimmed = trim_to_caption(TOME)
    assert len(trimmed.split()) <= CAPTION_MAX_WORDS
    assert trimmed.startswith("hard rock, drums, bass, male vocal")
    assert trimmed.endswith(".")          # whole sentences only, never a cut


def test_trim_caps_a_runaway_tag_list():
    head = ", ".join(f"tag{i}" for i in range(40))
    out = trim_to_caption(head + ". It builds to a peak and fades out now.")
    assert len(out.split(".")[0].split(",")) <= TAG_MAX


def test_trim_keeps_flow_from_a_single_run_on_sentence():
    """A model that emits one 500-word 'sentence' must still yield a caption."""
    out = trim_to_caption(TAGS + ". " + "word " * 500 + "end.")
    assert len(out.split()) <= CAPTION_MAX_WORDS
    assert out.count(".") >= 2            # tag list + at least one flow sentence


# --------------------------------------------------------------------------
# kernel <-> worker placeholder contract (the {{ATTN_IMPL}} lesson)
# --------------------------------------------------------------------------

CAPTION_KERNEL = "kernels/caption_kernel.py"
CAPTION_KERNEL_PUSHERS = (
    "workers/caption.py",      # Caption tab
    "workers/structural.py",   # structural pipeline
)


def test_every_caption_kernel_placeholder_is_filled_by_every_pusher():
    """Any {{X}} in the kernel must be substituted by EVERY worker that pushes it.

    A missed placeholder is not caught locally: {{X}} parses as valid Python (a
    set containing a set), so the failure only surfaces on Kaggle with a bare
    ``NameError: name 'X' is not defined`` after the queue wait.
    """
    todo = _placeholders(_read(CAPTION_KERNEL))
    assert todo, "expected the kernel to declare placeholders"
    for pusher in CAPTION_KERNEL_PUSHERS:
        source = _read(pusher)
        missing = sorted(p for p in todo if p not in source)
        assert not missing, f"{pusher} does not fill {missing}"


def test_caption_kernels_carry_repetition_controls():
    """Greedy decoding with no penalty is what let a caption loop ~200 times."""
    for kernel in (CAPTION_KERNEL,):
        source = _read(kernel)
        assert "repetition_penalty=REPETITION_PENALTY" in source, kernel
        assert "no_repeat_ngram_size=NO_REPEAT_NGRAM" in source, kernel


def test_caption_kernel_gives_the_schema_a_system_role():
    """The schema must arrive as the system turn, not as a competing user turn."""
    kernel = _read(CAPTION_KERNEL)
    assert '{"role": "system"' in kernel
    assert "SYSTEM_PROMPT" in kernel
