"""The ACE-Step 1.5XL caption schema — single source of truth.

The authoritative standard is ``docs/ACE_Step_1.5_Master_Annotation_Guide.md``
(§1 caption architecture, §2 lyrics, §3 descriptors) plus the key rules in
``docs/descriptor_reference.md`` §10. This module turns that prose standard into
constants the code cites, so the captioner, the structural pipelines, the Tag
Creator and the validators share ONE definition instead of the five drifted
copies that used to exist (config.py, workers/caption.py, workers/spatial.py,
workers/structural.py).

WHY THIS EXISTS
---------------
The captioner used to be told: "write a detailed description ... Write 3 to 5
sentences. Start with A or An." The schema says the opposite: front-load 5-12
comma-separated conditioning keywords (max 15), then a comma-separated list of
acoustic facts. A model that opens with "A high-energy Garage Rock track with a
raw, lo-fi aesthetic..." is COMPLYING with that old prompt — which is exactly why
the output was worthless as training data. The prompt was schema-less; the model
was not at fault.

WHY NO FLOW NARRATIVE (2026-10-23)
----------------------------------
The 2-3 sentence "flow narrative" that used to sit after the tag list was a
holdover from image captioning (BLIP/CLIP), where a caption describes one frame.
An audio DiT conditions on the caption through cross-attention for the whole
render, so connective tissue ("the track breathes with an unhurried...
atmosphere driven by") spends attention mass on tokens that carry no acoustic
information, and spends prompt budget the metadata block needs. Measured with
the shipped Qwen3-Embedding-0.6B tokenizer: one Black Sabbath caption costs 75
tokens as 47 words of narrative prose and 59 tokens as 35 words of dense facts.
The rule below is therefore a hard ban, and ``GOLD_EXAMPLE`` is dense for the
same reason — models imitate the example before they obey the instructions.

The built-in prompt below IS the schema and is set in stone. User text is
APPENDED to it, never substituted for it — see ``build_system_prompt``.
"""

# The §1 structure formula.
CAPTION_FORMULA = (
    "[Trigger Tag], [Primary Genre], [Subgenre/Mood], [2-3 Specific Instruments], "
    "[Vocal Style], [Production & Mix], [Era/Aesthetic]. "
    "[8-16 comma-separated acoustic facts: arrangement, transitions, timbral detail]."
)

# The density rule, verbatim. Every document that describes the caption shape
# quotes this string, so the prose cannot drift from the prompt that is actually
# sent. See the module docstring for why the narrative form was retired.
CAPTION_DENSITY_RULE = (
    "Every token in the caption must pay rent in acoustic information. Do not "
    "write full prose sentences. A dense, comma-separated list of acoustic facts, "
    "structural transitions, and timbral descriptors provides higher attention "
    "focus to the DiT cross-attention heads than narrative filler."
)

# The caption is the tag block plus the fact tail. The band is what leaves the
# metadata block room inside a 256-token prompt: a 59-word dense caption measures
# 99 tokens against 75 for a 47-word narrative one (Qwen3-Embedding-0.6B).
CAPTION_MIN_WORDS = 35
CAPTION_MAX_WORDS = 70

# §1 formatting standards + descriptor_reference §10 rules 5, 8, 9, 10, 11.
# NOTE: rule 1 keeps the exact phrase "5-12 comma-separated keywords" because
# tests/test_tag_creator.py asserts it appears in the Tag Creator prompt.
CAPTION_RULES = (
    "1. FRONT-LOAD the conditioning keywords: 5-12 comma-separated keywords "
    "(max 15) at the very start. Never open with a sentence.\n"
    "2. Use CONCRETE instruments and gear (e.g. Vox Continental organ, Gibson SG "
    "with P-90 pickups, Fender Rhodes bass, 808 sub bass, gated reverb drums).\n"
    "3. ALWAYS declare vocal presence and character: male vocal, female vocal, "
    "male baritone, raspy vocals, powerful belting, clean vocals, screamed "
    "vocals, whispered vocals, choir, or instrumental / no vocals.\n"
    "4. " + CAPTION_DENSITY_RULE + "\n"
    "5. NEVER put BPM, key, or time signature in the caption — those are "
    "dedicated numeric/categorical metadata fields.\n"
    "6. No headings, no field labels, no bullet points, no markdown. Output ONLY "
    "the caption text. The tag block ends with one full stop; the fact tail "
    "contains no full stops — facts are separated by commas.\n"
    "7. Do not transcribe, quote, or repeat lyrics — describe the vocal delivery "
    "instead. Never repeat a word or phrase for emphasis.\n"
    f"8. Keep the WHOLE caption to the keyword list plus 8-16 facts: about "
    f"{CAPTION_MIN_WORDS}-{CAPTION_MAX_WORDS} words, and NEVER more than "
    f"{CAPTION_MAX_WORDS}. A long essay is not a caption."
)

# §2 lyrics architecture + §3 descriptor limits. Consumed by the Tag Creator and
# by modules/caption_quality.py so the numbers cannot drift from the prose rules.
MAX_DESCRIPTORS = 3          # §3: more risks the model singing tag names
TAG_MIN = 5                  # §1.1 front-loaded keywords, lower bound
TAG_MAX = 12                 # §1.1 recommended upper bound
TAG_HARD_MAX = 15            # §1.1 absolute maximum
LYRIC_MIN_SYLLABLES = 6      # §2 syllable cadence
LYRIC_MAX_SYLLABLES = 10
LYRIC_FLAG_SYLLABLES = 12    # §2: flag lines exceeding this

# The schema is the tag block plus a fact tail: 35-70 words. Anything far beyond
# that is not a caption: a 2000+ word output is either a decoding runaway or a
# provider ignoring max_tokens (observed: a caption ran to ~2000 words on an
# endpoint configured with max_tokens=512), and it is unusable as training data.
# Enforced in caption_quality, and enforced by trimming in the backends because a
# provider's token cap cannot be trusted. The upper bound is the caption half of
# the 256-token prompt, not an aesthetic preference: see CAPTION_MIN_WORDS.
CAPTION_MAX_CHARS = 900

LYRICS_RULES = (
    "- Every section starts with a Capitalized marker in square brackets: "
    "[Intro], [Verse 1], [Chorus], [Bridge], [Outro].\n"
    f"- Maximum {MAX_DESCRIPTORS} descriptors per bracket, attached with a dash: "
    "[Chorus - anthemic, full].\n"
    "- Separate sections with a blank line.\n"
    f"- Keep lines to {LYRIC_MIN_SYLLABLES}-{LYRIC_MAX_SYLLABLES} syllables; flag "
    f"any line over {LYRIC_FLAG_SYLLABLES}.\n"
    "- UPPERCASE signals belting, screaming, or shouted delivery; (parentheses) "
    "signal backing vocals, echoes, or call-and-response.\n"
    "- Instrumental sections carry the marker on its own line with no lyrics below."
)

# A caption that conforms. Taken from the Black Sabbath example in
# docs/descriptor_reference.md "Complete examples", rewritten in the dense form:
# 8 tags, then 8 facts, 45 words. Models copy the shape of the example before
# they obey the rules, so this string is the real spec.
GOLD_EXAMPLE = (
    "heavy metal, doom metal, downtuned guitar, distorted bass, thunderous drums, "
    "nasal male vocal, raw 1970s analog production, heavy low-end. slow ominous "
    "riff intro, plaintive verse, wailing vocal, doubled guitar harmony, crushing "
    "chorus peak, wah solo over pedal tone, half-time tom fill, bleak atmospheric "
    "outro fade"
)

# A real captioner failure, kept as the negative example. It violates rules 1
# (opens with a sentence, no tag list), 5 (states a BPM it cannot know), 6
# (field-label headings) and 7 (echoes the lyric refrain instead of describing
# delivery). Models follow few-shot pairs far better than instructions alone.
COUNTER_EXAMPLE = (
    "Genre & Style: A high-energy Garage Rock track with a raw, lo-fi aesthetic. "
    "Tempo: Fast, driving tempo at approximately 120 BPM. "
    "Vocals & Lyrics: ... the repeated refrain: \"Baby, don't go, don't go, "
    "don't go, don't go, don't go, don't go...\""
)

BUILTIN_SYSTEM_PROMPT = (
    "You are a music annotation assistant for the ACE-Step 1.5 music generation "
    "model, producing training captions that conform to the ACE-Step 1.5XL dataset "
    "annotation schema. Output ONLY the caption text — no preamble, no headings, "
    "no markdown.\n\n"
    "CAPTION SCHEMA (exact shape):\n"
    + CAPTION_FORMULA + "\n\n"
    "RULES:\n" + CAPTION_RULES + "\n\n"
    "GOLD EXAMPLE (conforms — imitate this shape):\n"
    + GOLD_EXAMPLE + "\n\n"
    "WORTHLESS EXAMPLE (violates the schema — never produce this):\n"
    + COUNTER_EXAMPLE
)


def build_system_prompt(user_addendum=""):
    """Return the schema system prompt, with any user text APPENDED to it.

    The schema is set in stone: user text can add house style or per-artist
    emphasis, but it can never replace or precede the schema itself. An empty or
    whitespace-only addendum is ignored, so a blank field changes nothing.
    """
    extra = (user_addendum or "").strip()
    if not extra:
        return BUILTIN_SYSTEM_PROMPT
    return (
        BUILTIN_SYSTEM_PROMPT
        + "\n\nADDITIONAL USER INSTRUCTIONS (they must not contradict the schema "
        "above):\n" + extra
    )


# The schema-less prompt that shipped before this module existed. It says "Write
# 3 to 5 sentences. Start with A or An", which directly contradicts rule 1. It is
# still persisted verbatim in existing settings.json files, so it must be
# detected and treated as "no user edit" -- appending it as a user instruction
# would inject the contradiction straight back into the prompt.
LEGACY_CAPTION_PROMPT = (
    "You are a professional music metadata tagger preparing training data for "
    "ACE-Step. Listen carefully to this audio clip and write a detailed description. "
    "Cover: specific instrumentation (name every instrument you hear), whether "
    "vocals are present (gender, register, timbre) or confirm instrumental, "
    "recording and production character, mood, and how the clip develops. "
    "Write 3 to 5 sentences. Start with A or An. "
    "Genre, BPM, key, and time signature are handled separately — do not include them."
)


def is_legacy_caption_prompt(text):
    """True if ``text`` is the old schema-less shipped default (not a user edit)."""
    return (text or "").strip() == LEGACY_CAPTION_PROMPT


# The rule-4 wording that shipped before 2026-10-23: "then 2-3 sentences
# describing how the energy moves through the track". That sentence now
# contradicts rule 4 (CAPTION_DENSITY_RULE), so a stored prompt containing it is
# treated as "no user edit" rather than appended as an instruction that fights
# the schema. Substring, not equality: unlike LEGACY_CAPTION_PROMPT this wording
# was pasted into ``caption_system_prompt`` with edits around it, so there is no
# single string to compare against.
NARRATIVE_CAPTION_MARKERS = (
    "2-3 sentences",
    "3 to 5 sentences",
    "flow narrative",
    "how the energy moves",
    "energy moves through the track",
)


def is_narrative_caption_prompt(text):
    """True if ``text`` still asks for prose sentences after the tag list."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in NARRATIVE_CAPTION_MARKERS)


def user_addendum_from_config(config):
    """Pull the user's caption instructions out of a config dict.

    Returns "" when the stored value is the legacy contradictory default or still
    asks for narrative sentences, so an untouched settings.json cannot silently
    undo the schema.
    """
    stored = (config.get("caption_system_prompt") or "").strip()
    if stored:
        return "" if is_narrative_caption_prompt(stored) else stored
    legacy = (config.get("caption_prompt") or "").strip()
    if legacy and not is_legacy_caption_prompt(legacy):
        return "" if is_narrative_caption_prompt(legacy) else legacy
    return ""


def system_prompt_from_config(config):
    """The complete system prompt for a config dict — the one callers should use."""
    return build_system_prompt(user_addendum_from_config(config))


# The user turn sent alongside the system prompt. It must not restate the schema
# (the system prompt owns that) — it only says what to do with this clip.
DEFAULT_TASK_PROMPT = (
    "Annotate this audio clip as ONE caption in the exact schema above. "
    "Output only the caption text."
)


# Separator used when extra run-specific instructions are appended to the user
# turn. Worded as a constraint, like the system addendum, so the model cannot
# mistake the addition for permission to abandon the schema.
TASK_PROMPT_ADDENDUM_HEADER = (
    "\n\nADDITIONAL INSTRUCTIONS FOR THIS RUN (they must not contradict the "
    "schema above):\n"
)


def build_task_prompt(addendum="", base=""):
    """Return the user turn, with any extra text APPENDED to it.

    ``base`` overrides the default task line (used for the instrument-only
    prompt); an empty or whitespace-only ``addendum`` leaves the base untouched,
    so a blank field changes nothing.
    """
    text = (base or "").strip() or DEFAULT_TASK_PROMPT
    extra = (addendum or "").strip()
    if not extra:
        return text
    return text + TASK_PROMPT_ADDENDUM_HEADER + extra


def task_prompt_from_config(config):
    """The user-turn instruction, ignoring the legacy schema-less default.

    ``caption_prompt`` used to hold the whole instruction, including "Write 3 to 5
    sentences. Start with A or An". A stored copy of that default must not be
    sent as the user turn, or the contradiction returns through the other door.

    ``caption_prompt_addendum`` is APPENDED to whichever base prompt survives
    that filter — the schema is in the system turn, so the user turn is the only
    place run-specific emphasis can be added without competing with it.
    """
    stored = (config.get("caption_prompt") or "").strip()
    base = stored if (stored and not is_legacy_caption_prompt(stored)) else ""
    return build_task_prompt(
        addendum=config.get("caption_prompt_addendum"), base=base
    )