"""Tests for the ACE-Step 1.5XL caption schema, its prompt composer and validator.

The load-bearing test here is ``test_every_kernel_placeholder_is_filled``: an
unfilled ``{{X}}`` is VALID Python (a set containing a set), so a missing
substitution reaches Kaggle and dies there with a bare NameError 50+ seconds
later. That already happened once with ``{{ATTN_IMPL}}``.

The second is that the schema-less legacy prompt -- "Write 3 to 5 sentences. Start
with A or An" -- can never come back through the config, since it directly
contradicts the required front-loaded tag list.
"""
import os
import re

from modules.caption_quality import (
    check_caption,
    count_sentences,
    is_conforming,
    max_consecutive_repeat,
    tag_prefix,
    trim_to_caption,
)
from modules.caption_spec import (
    BUILTIN_SYSTEM_PROMPT,
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

def test_gold_example_conforms_and_counter_example_does_not():
    assert check_caption(GOLD_EXAMPLE) == []
    assert check_caption(COUNTER_EXAMPLE), "the real failure must be flagged"


def test_builtin_prompt_encodes_the_schema():
    assert CAPTION_FORMULA in BUILTIN_SYSTEM_PROMPT
    assert "5-12 comma-separated keywords" in BUILTIN_SYSTEM_PROMPT
    assert "NEVER put BPM, key, or time signature" in BUILTIN_SYSTEM_PROMPT
    # Few-shot beats instructions: the worthless example must be shown as such.
    assert COUNTER_EXAMPLE in BUILTIN_SYSTEM_PROMPT
    assert GOLD_EXAMPLE in BUILTIN_SYSTEM_PROMPT


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
# the validator
# --------------------------------------------------------------------------

def test_tag_prefix_stops_at_the_first_sentence():
    text = "hard rock, drums, male vocal. It builds to a peak."
    assert tag_prefix(text) == ["hard rock", "drums", "male vocal"]


def test_too_few_front_loaded_keywords_is_flagged():
    thin = "rock, drums, male vocal. It builds through the verse and peaks late on."
    assert any("front-loaded" in i for i in check_caption(thin))
    assert (TAG_MIN, TAG_MAX, TAG_HARD_MAX) == (5, 12, 15)


def test_bpm_in_the_caption_is_flagged():
    text = ("hard rock, drums, bass, male vocal, raw production, 1970s analog. "
            "Drives at 120 BPM through the verse. Peaks, then fades.")
    assert any("BPM" in i for i in check_caption(text))


def test_field_label_heading_is_flagged():
    text = ("Genre & Style: hard rock, drums, bass, male vocal, raw production, "
            "1970s analog. Builds to a peak. Fades out.")
    assert any("heading" in i for i in check_caption(text))


def test_missing_vocal_descriptor_is_flagged():
    text = ("hard rock, drums, bass, electric guitar, raw production, "
            "1970s analog. Builds to a peak. Fades out.")
    assert any("vocal" in i for i in check_caption(text))


def test_tags_only_caption_is_flagged_for_no_flow_narrative():
    text = "hard rock, drums, bass, male vocal, raw production, 1970s analog"
    assert any("flow sentence" in i for i in check_caption(text))


def test_degenerate_repetition_is_flagged():
    text = ("hard rock, drums, bass, male vocal, raw production. "
            + "don't go, " * 30 + "Fades out.")
    assert any("degenerate repetition" in i for i in check_caption(text))


def test_a_legitimate_repeated_instrument_name_is_not_flagged():
    """The repeat check must not fire on ordinary captions."""
    text = ("hard rock, distorted guitar, bass, male vocal, live drums, "
            "raw production. Guitar and drums lock in tight through the verse. "
            "The guitar solo peaks, then the drums drop out for the outro.")
    assert not any("degenerate repetition" in i for i in check_caption(text))


def test_empty_caption_is_flagged():
    assert check_caption("") == ["caption is empty"]
    assert check_caption("   ")


def test_helpers_behave():
    assert count_sentences(
        "tags, here. It builds through the verse and peaks. Then it fades out."
    ) == 2
    assert max_consecutive_repeat("go go go go go now")[0] == 5
    assert is_conforming(GOLD_EXAMPLE) is True
    assert is_conforming(COUNTER_EXAMPLE) is False
    assert MAX_DESCRIPTORS == 3


def test_trim_is_a_no_op_for_a_conforming_caption():
    assert trim_to_caption(GOLD_EXAMPLE) == GOLD_EXAMPLE


TAGS = "hard rock, drums, bass, male vocal, raw production, 1970s analog"
TOME = TAGS + ". " + "The band plays with conviction and the guitars ring out. " * 40


def test_an_over_long_caption_is_flagged():
    """A 2000-word caption is unusable; the token cap could not be trusted."""
    issues = check_caption(TOME)
    assert any("flow sentences" in i for i in issues)
    assert any("words" in i for i in issues)


def test_trim_brings_a_tome_back_to_the_schema():
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


def test_moss_kernel_placeholders_are_filled():
    todo = _placeholders(_read("kernels/moss_caption_kernel.py"))
    source = _read("workers/kaggle_moss.py")
    missing = sorted(p for p in todo if p not in source)
    assert not missing, f"workers/kaggle_moss.py does not fill {missing}"


def test_caption_kernels_carry_repetition_controls():
    """Greedy decoding with no penalty is what let a caption loop ~200 times."""
    for kernel in (CAPTION_KERNEL, "kernels/moss_caption_kernel.py"):
        source = _read(kernel)
        assert "repetition_penalty=REPETITION_PENALTY" in source, kernel
        assert "no_repeat_ngram_size=NO_REPEAT_NGRAM" in source, kernel


def test_caption_kernel_gives_the_schema_a_system_role():
    """The schema must arrive as the system turn, not as a competing user turn."""
    kernel = _read(CAPTION_KERNEL)
    assert '{"role": "system"' in kernel
    assert "SYSTEM_PROMPT" in kernel
