"""The ACE-Step 1.5XL caption schema — single source of truth.

The authoritative standard is ``docs/ACE_Step_1.5_Master_Annotation_Guide.md``
(§1 caption architecture, §2 lyrics, §3 descriptors) plus the key rules in
``docs/descriptor_reference.md`` §10. This module turns that prose standard into
constants the code cites, so the captioner, the structural pipelines, the Tag
Creator and the validators share ONE definition instead of the five drifted
copies that used to exist (config.py, workers/caption.py, workers/spatial.py,
workers/structural.py, workers/kaggle_moss.py).

WHY THIS EXISTS
---------------
The captioner used to be told: "write a detailed description ... Write 3 to 5
sentences. Start with A or An." The schema says the opposite: front-load 5-12
comma-separated conditioning keywords (max 15), then 2-3 sentences of flow
narrative. A model that opens with "A high-energy Garage Rock track with a raw,
lo-fi aesthetic..." is COMPLYING with that old prompt — which is exactly why the
output was worthless as training data. The prompt was schema-less; the model was
not at fault.

The built-in prompt below IS the schema and is set in stone. User text is
APPENDED to it, never substituted for it — see ``build_system_prompt``.
"""

# The §1 structure formula, verbatim.
CAPTION_FORMULA = (
    "[Trigger Tag], [Primary Genre], [Subgenre/Mood], [2-3 Specific Instruments], "
    "[Vocal Style], [Production & Mix], [Era/Aesthetic]. "
    "[2-3 sentences detailing dynamic build, arrangement transitions, and energy flow]."
)

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
    "4. Follow the tag list with 2-3 sentences describing how the energy moves "
    "through the track (dynamic build, arrangement transitions, energy flow).\n"
    "5. NEVER put BPM, key, or time signature in the caption — those are "
    "dedicated numeric/categorical metadata fields.\n"
    "6. No headings, no field labels, no bullet points, no markdown. Output ONLY "
    "the caption text.\n"
    "7. Do not transcribe, quote, or repeat lyrics — describe the vocal delivery "
    "instead. Never repeat a word or phrase for emphasis.\n"
    f"8. Keep the WHOLE caption to the keywords plus 2-3 sentences: about 70-90 "
    f"words, and NEVER more than {120}. A long essay is not a caption."
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

# The schema is 5-12 keywords plus 2-3 sentences, i.e. roughly 70-90 words.
# Anything far beyond that is not a caption: a 2000+ word output is either a
# decoding runaway or a provider ignoring max_tokens (observed: a caption ran to
# ~2000 words on an endpoint configured with max_tokens=512), and it is unusable
# as training data. Enforced in caption_quality, and enforced by trimming in the
# backends because a provider's token cap cannot be trusted.
CAPTION_MAX_WORDS = 120
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
# docs/descriptor_reference.md "Complete examples".
GOLD_EXAMPLE = (
    "heavy metal, doom metal, doom-laden, ominous, downtuned guitar, distorted "
    "bass, thunderous drums, nasal male vocal, raw 1970s analog production, "
    "heavy low-end. Opens with a slow, ominous riff before building into a "
    "plaintive verse with wailing vocals. Reaches a crushing peak and fades "
    "into a bleak, atmospheric outro."
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


def user_addendum_from_config(config):
    """Pull the user's caption instructions out of a config dict.

    Returns "" when the stored value is the legacy contradictory default, so an
    untouched settings.json cannot silently undo the schema.
    """
    stored = (config.get("caption_system_prompt") or "").strip()
    if stored:
        return stored
    legacy = (config.get("caption_prompt") or "").strip()
    if legacy and not is_legacy_caption_prompt(legacy):
        return legacy
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