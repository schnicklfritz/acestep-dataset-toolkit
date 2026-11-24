**Incorrect**
*Marked for deletion*



# ACE-Step 1.5 Master Annotation & Dataset Guide

> **What this file is:** the short normative *format specification* — the caption
> formula, the structural marker table, the approved descriptor vocabulary and
> the max-3-descriptors rule. This is the path cited by
> `modules/caption_spec.py:3`, `config.py:62` and `README.md`.
>
> **What it is not:** the source-verified dataset reference. Token budgets,
> ingestion-path behaviour, the Omission Principle and each claim's evidence
> status live in `docs/Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md`.
>
> **The two are not yet reconciled.** In particular this file's "maximum 3
> descriptors per bracket" (§3) and the annotation guide's descriptor count
> differ; that discrepancy is a still-open decision, not a settled one. Read
> both and do not assume they agree.

This master specification details the annotation standard for **ACE-Step 1.5** training datasets, covering prompt conditioning, structural lyric tagging, architectural descriptors, and hybrid captioning.

---

## 1. Hybrid Caption Architecture (Global Conditioning)

The Caption establishes the **global sound portrait** across the entire duration of the audio clip.

### Structure Formula
```text
[Trigger Tag], [Primary Genre], [Subgenre/Mood], [2-3 Specific Instruments], [Vocal Style], [Production & Mix], [Era/Aesthetic]. [8-16 comma-separated acoustic facts: arrangement, transitions, timbral detail].
```

### Formatting Standards
1. **Front-Loaded Tags**: High-priority conditioning keywords appear at the very start (5–12 keywords, maximum 15) separated by commas.
2. **Concrete Instruments**: Use exact, specific instruments and gear (e.g., `Vox Continental organ`, `Gibson SG with P-90 pickups`, `Fender Rhodes bass`, `808 sub bass`, `gated reverb drums`).
3. **Mandatory Vocal Descriptor**: Always declare vocal presence and character:
   - `male vocal`, `female vocal`, `male baritone`, `raspy vocals`, `powerful belting`, `clean vocals`, `screamed vocals`, `whispered vocals`, `choir`, or `instrumental` / `no vocals`.
4. **No Prose Sentences**: Every token in the caption must pay rent in acoustic information. Do not write full prose sentences. A dense, comma-separated list of acoustic facts, structural transitions, and timbral descriptors provides higher attention focus to the DiT cross-attention heads than narrative filler.
   - The tag block ends with **one full stop**; the fact tail carries no full stops — facts are separated by commas. That single period is load-bearing: `modules/caption_quality.tag_prefix` splits on it, so without it the entire caption parses as tags and trips the 15-tag hard maximum.
5. **No Parameter Contradictions**: Never place BPM, Key, or Time Signature inside the caption—those are handled as dedicated numeric/categorical metadata fields.

> **Why the narrative was retired (2026-10-23).** The 2–3 sentence "flow narrative" was a holdover from image captioning (BLIP/CLIP), where a caption describes a single frame. Here the caption conditions the DiT through cross-attention for the whole render, so connective tissue ("opens with… before building into…") spends attention mass on tokens that carry no acoustic information. Measured with the shipped `Qwen3-Embedding-0.6B` tokenizer: this guide's Black Sabbath caption is **47 words / 75 tokens** as narrative prose and **35 words / 59 tokens** as dense facts. The token counts are measured; the attention-dilution mechanism is the stated rationale, not measured on this checkpoint. The rule is defined once in `modules/caption_spec.CAPTION_DENSITY_RULE` — the text above quotes it.

---

## 2. Lyrics Architecture (The Temporal Script)

Lyrics control **when and how** elements occur over time. Every song section must begin with an explicit structural marker enclosed in square brackets.

### Structural Markers
| Category | Marker | Description |
| :--- | :--- | :--- |
| **Song Structure** | `[Intro]` | Atmospheric opening / instrumental lead-in |
| | `[Verse]` / `[Verse 1]` | Narrative progression |
| | `[Pre-Chorus]` | Tension builder before the hook |
| | `[Chorus]` | Main hook / emotional climax |
| | `[Bridge]` | Contrasting section or thematic departure |
| | `[Outro]` | Resolution and track conclusion |
| **Dynamic / EDM** | `[Build]` / `[Build-Up]`| Rising tension, filter sweeps, sidechain kick |
| | `[Drop]` | Energy release / full dynamic impact |
| | `[Breakdown]` | Stripped-back space before rebuild |
| **Instrumentals** | `[Instrumental]` / `[Inst]` | Pure instrumental section (no lyrics below) |
| | `[Guitar Solo]` | Featured guitar lead |
| | `[Piano Interlude]` | Keyboard/piano passage |
| | `[Drum Break]` | Rhythm section highlight |
| **Special** | `[Fade Out]` | Gradual volume decrease ending |
| | `[Silence]` | Complete rest / drop |

### Lyric Formatting Rules
- **Capitalization**: Section marker names are always capitalized (`[Verse]`, `[Chorus]`, `[Outro]`).
- **Section Spacing**: Always separate distinct sections with a blank line (`\n\n`).
- **Syllable Cadence**: Keep lines between **6 to 10 syllables** for optimal beat alignment. Flag lines exceeding 12 syllables.
- **Vocal Intensity**: Use `UPPERCASE` text to signal belting, screaming, or shouted delivery.
- **Harmonies & Backing**: Use `(parentheses)` to indicate backing vocals, echoes, or call-and-response lines.
- **Instrumental Songs**: For pure instrumentals, set lyrics to `[Instrumental]` only.

### Backing Vocals and Parentheses

The **Harmonies & Backing** rule above carries the whole convention: content in parentheses is
processed as background vocals or harmonies. Everything below is how to apply it without
contradicting itself.

#### What the model actually receives

> **Verified in the ACE-Step source.** The lyric channel is a bare string. It is tokenized at
> `max_length=512` with truncation (`acestep/training/dataset_builder_modules/preprocess_lyrics.py`),
> embedded by table lookup only (`text_encoder.embed_tokens`) and then contextualised by
> ACE-Step's own 8-layer `AceStepLyricEncoder`. The caption path, by contrast, runs a full
> forward pass through the Qwen3 text encoder
> (`acestep/training/dataset_builder_modules/preprocess_text.py`, `max_length=256`). A
> parenthesis is therefore never *interpreted* by a language model — it is the fixed embedding
> of token `(` (id 7) or `)` (id 8), and its meaning is learned as an association by the lyric
> encoder. The counts in the table below were reproduced directly with the `tokenizers` library
> against the shipped `models/Qwen3-Embedding-0.6B/tokenizer.json` (single-vocabulary, 151,669
> entries).

| Fact | Tokenisation | Effect |
|---|---|---|
| `(` | 1 token, id 7 | An open parenthesis is never free-standing syntax |
| `)` | 1 token, id 8 | Every parenthetical costs at least +1 token |
| `" ("` | 1 token, id 320 | A space before the parenthesis is free — the pair merges |
| `(together)` | `(to` `gether` `)` = 3 | `(` merges into the next word; `(together)` is not a vocabulary item |
| `[Verse 1]` | `[` `Verse` `Ġ` `1` `]` = 5 | Numbered markers are expensive; `[Verse]` costs 3 |

- No merged token exists for a parenthesised word (`(together)`, `(Look`, `(into` are absent
  from the 151,669-entry vocabulary), so parentheses do not change a word's own tokenisation —
  they add the `(` / `)` tokens and can re-split the word that follows (`(to` + `gether`).
- The lyric channel has **no alignment model**, so position is the only "clock": line breaks,
  word order and where the parenthesis sits are all the model is given. Placement is therefore
  a timing claim, not decoration.
- The 512-token training cap is binding. Parentheses and numbered markers spend budget
  (`)` costs +1 each, `[Verse 1]` costs +5); keep the whole lyric file under 512 tokens
  (training cap — inference allows 2048) or the tail is silently dropped. Word-level echoes are
  cheap; numbered markers are not.

> **The 5Hz LM sees the same lyrics with the same ids.** The optional 5Hz LM
> (`acestep-5Hz-lm-{0.6B,1.7B,4B}`) is a *second, independent* consumer of lyric text: it
> receives them inside its chat template (`acestep/llm_inference.py` → `# Lyric\n{lyrics}`,
> `:1705`) and tokenizes the result with its own checkpoint tokenizer (`:612`, `:1008`). That
> tokenizer's base vocabulary, merge list, pre-tokenizer and normaliser are identical to the text
> encoder's, so every id and count above is unchanged on that path. The 4B checkpoint differs
> only by a 65,535-entry `<|audio_code_N|>` block appended at ids 151669–217203 (217,204 ids
> total) and a different chat template; no added token contains a parenthesis. Whichever LM size
> is active, `(` is id 7 and `)` is id 8.

> **NOT documented upstream:** the official docs state *that* parentheses produce background
> vocals (`docs/en/Tutorial.md:540-548`, example `We rise together (together)`; restated in
> `docs/en/ace_step_musicians_guide.md:331-335` and
> `.claude/skills/acestep-songwriting/SKILL.md:131`) but say nothing about *placement*. The
> placement rules below are this project's empirical convention — apply them consistently
> within a dataset rather than treating them as a documented timing language.

#### Macro and micro

When a verse mixes vocal styles — opening solo, a gang-shouted unison phrase, an echo answer,
then an overlap — do not split the verse into a dozen tiny brackets. Use the Macro/Micro rule:

> **Macro** (the bracket header): the section's overall arrangement profile, written with the
> approved descriptors (maximum 3).
>
> **Micro** (line-by-line syntax): typography, parentheses and line placement direct the
> moment-by-moment vocal interplay.

#### Tags versus parentheses: which one do you need?

They do different jobs, and the difference is **scope**. A bracket descriptor is a commitment
about the whole section: it says what the section *is*, and the adapter then expects that
texture on its lines. Parentheses are a local annotation: they mark one echo, one answer or one
overlap, and say nothing about the rest of the section. Getting that scope wrong is the most
common error in both directions.

| Ask yourself | The answer points to | What the output then claims |
|---|---|---|
| Does the interaction span the section, on most of its lines? | A header tag on the section marker | Backing vocals throughout: `call and response`, `layered`, `gang vocals` or `harmonies` describe the whole section |
| Is it one event — one echo, one answer, one overlap? | Parentheses, header left plain | A local instruction: the backing vocal is here, at this point, and nowhere else |
| Is it rare but real — a single throwaway echo in an otherwise solo verse? | Parentheses only | The echo is notated without committing the section to a texture it does not have |
| Is the section one continuous texture — all gang-shouted, all layered, all harmonised? | Header tag plus plain text | The texture is stated once, at the header, instead of restated on every line |

How to read that table, row by row:

> Row 1 — the test is coverage. If you would have to write a parenthesis on most lines to
> describe what you hear, the section has an arrangement profile, and a profile belongs in the
> header, stated once.
>
> Row 2 — the test is locality. If the backing vocal is a single event, parentheses place it
> exactly, at the word or the line where it happens.
>
> Row 3 — the trap, and it catches people in both directions. Leaving the header plain is
> correct: the parenthesis alone carries the echo. Adding `[Verse 1 - call and response]` is not
> "more accurate", it is a different claim — that the whole verse trades phrases — and the model
> will write you one.
>
> Row 4 — the mirror trap. If every line is gang-shouted, the header already says so, and
> `(DONE DIRT CHEAP)` on each line only restates it. Write the line once, in caps, under the
> header.

**Decision list:**

1. One isolated echo in an otherwise solo verse → parentheses only; the header stays `[Verse 1]`. Coverage is one line, so there is no section profile to name.
2. An echo, trade or answer on most lines → add the header descriptor `[Verse 1 - call and response]` and keep the parentheses on the individual lines. The header states the arrangement; the parentheses still say which word is the answer.
3. The section is one continuous texture (all gang-shouted, all layered) → header descriptor plus plain text; parentheses add nothing.
4. The rhythm section physically stops or shifts structure → start a new section bracket instead of annotating the old one.

#### Line types, and how each one is written

One convention generates the whole table: **capitals carry intensity, parentheses carry "not
the lead", and a new line carries time**. Both that convention and the placements below are
this project's, not documented upstream (see the evidence note above). Applied consistently,
they are learnable within one dataset.

| Line type | How you write it | Example | What the model has to work from |
|---|---|---|---|
| Solo lead, normal sung delivery | Lowercase, no parentheses | Well me and the boys are out to have fun | One voice, mid-intensity, on the beat |
| Solo lead, shouted or belted | ALL CAPS, no parentheses | GET OUT OF MY WAY | One voice, high intensity |
| Full-band unison, same words | ALL CAPS, no parentheses; the header carries `[gang vocals]` | `[Chorus - gang vocals]` DIRTY DEEDS DONE DIRT CHEAP | One text, many voices |
| Backing echo that answers the lead line | Parentheses on their own line, directly below the line they answer | Im a LIVE WIRE / (LIVE WIRE) | Two lines = two time positions; the answer lands after the lead finishes |
| Backing lead-in, sung before the lead enters | Parentheses at the start of the lead line | (Look out) here comes the man | Backing first, lead takes over inside the same line |
| Shouted gang backing | ALL CAPS inside parentheses | (DONE DIRT CHEAP) | Backing vocal, high intensity |
| Overlap during the lead line | Parentheses at the exact word where the backing lands | You know I love you so (Please dont go) | Two voices at once; the parenthesis position is the timing |
| Layered part, different words | Parentheses around the backing words, plus a `[layered]` header | `[Bridge - layered]` Neck ties contracts (DONE DIRT CHEAP) | Two independent texts at once, which the header marks as deliberate |
| Harmony on the same words, held | Header descriptor only; write the line once, no parentheses | `[Chorus - vocal harmonies]` HIGH VOLTAGE ROCK N ROLL | One text, several voices in harmony; no second text to place |

Two details in that table are conventions, not rules, so choose them once and keep them constant:

> The gap between a header and its words (`[Chorus - gang vocals]  DIRTY DEEDS...`) is your
> own: pick one separator and use it everywhere. What matters is that the header sits as a
> separate word group at the front of the line.
>
> The mid-line overlap is the only row where a parenthesis sits inside running text instead of
> bracketing a whole phrase. When the backing part is longer than a word or two, give it its own
> line (the trailing-answer row) or promote the section to `[layered]`.

Caps and parentheses can combine — ALL CAPS inside parentheses is a shouted backing part — but
neither one implies the other. A capped line inside a plain section is one belted lead voice,
not a crowd; a crowd is a header descriptor plus plain words.

#### Background vocals in the middle of a line

Mid-line backing causes the most confusion, so here it is explicitly. There are only three
positions a backing vocal can occupy relative to the lead, and each is written differently.
Position does that much work because the lyric channel carries text only and is never aligned
to audio: line breaks, word order and the position of a parenthesis are the only "clock" the
model is given.

```text
Opening — the backing vocal starts before the lead word, then hands over to it.
The parenthesis opens the line:

    (Look out) here comes the man

Middle — the backing vocal enters while the lead is already singing. The
parenthesis sits at the exact word it lands on, and the words after it belong to
the lead again:

    You know I love you so (Please dont go)

Trailing — the backing vocal answers after the lead line has finished. Put it on
its own line below, which is what separates an answer from an overlap:

    Im a LIVE WIRE
    (LIVE WIRE)
```

Position is the whole message. Written inline — `Im a LIVE WIRE (LIVE WIRE)` — the same two
parts read as an **overlap**, because both texts share one line and therefore one moment. Split
onto two lines they read as a **call and an answer**. Same words, different timing claim, and
the model has nothing else to go on.

Two limits follow from the same idea:

> Do not stack several mid-line parentheticals onto one line. If the backing vocal needs two or
> more of them, the section is layered, and that belongs in the header descriptor with the
> backing words written out.
>
> Keep one parenthesis to one phrase. When the backing part is longer than a word or two, give
> it its own line (a trailing answer) or promote the section to `[layered]`; do not bury a whole
> sentence inside the lead line.

#### Worked examples

**Example A — one isolated echo.** Coverage is a single line, so the header stays plain:

```text
[Verse 1]
Well me and the boys are out to have fun
(Look out) here comes the man
You know I love you so
Im a LIVE WIRE
```

Read back line by line: line 1 is solo lead, normal delivery — lowercase, no parentheses, one
voice. Line 2 is the backing lead-in: the parenthesis opens the line, the backing finishes
before "here", and the lead takes over inside the same line. Line 3 is solo lead again, and note
what is absent — no parenthesis follows "love you so", because nothing answers it. Line 4 is a
solo lead that is belted: LIVE WIRE is the title phrase and is shouted, so it is capped. Caps
carry intensity only; a capped line inside a plain section is one voice, not a crowd.

Header check: one echo on one line out of four is the "rare but real" row of the decision table
above. `[Verse 1]` stays plain and the parenthesis alone marks the echo. Adding
`[Verse 1 - call and response]` would claim the whole verse trades phrases.

**Example B — an echo on most lines.** Coverage is the section, so the descriptor goes into the
header **and** the parentheses stay on the individual lines:

```text
[Verse 1 - call and response]
Shes got balls (Balls)
Likes to crawl (Crawl)
Shes got balls (Balls)
```

Every line ends with a one-word bounce, which is a section-wide arrangement, so the "spans the
section" row applies and the header states it once. The parentheses are not redundant here: the
header says the section trades phrases, the parentheses say which word each answer repeats.

**Example C — answer or overlap.** The same words, and the only difference is where the
parenthesis sits.

```text
Im a LIVE WIRE
(LIVE WIRE)
```

Two lines: the answer lands after the lead line has finished.

```text
Im a LIVE WIRE (LIVE WIRE)
```

One line: those words now read as an overlap, because the model gets one time slot for the line
and both texts sit inside it.

```text
You know I love you so (Please dont go)
```

That last form is the mid-line overlap: the parenthesis sits at the exact word where the backing
lands, and the lead continues after it. Which form is the accurate one is a decision about the
recording, not a style preference.

**Example D — one continuous texture.** When the whole section is gang-shouted or harmonised,
the header carries it and the lines stay plain:

```text
[Chorus - gang vocals]
DIRTY DEEDS DONE DIRT CHEAP
DIRTY DEEDS DONE DIRT CHEAP
```

No parentheses: with no lead/backing split left to mark, they would only restate the header on
every line. The words are written once per repetition because the repetitions are the
arrangement.

The four examples differ only in **coverage**: A is one line, B is every line, C is the timing of
a single answer, and D is the section as a whole. Working out which one you are looking at is the
whole of this convention.

#### When to break the verse into multiple sections

Only break the verse into multiple section brackets if the rhythm section physically stops or
shifts structure:

> If the drums stop and only vocals/clapping continue:

```text
[Verse 1]
Lyrics...
```

```text
[Breakdown - stripped back, call and response]
(BAD BOY BOOGIE) I said up they said down
```

If it is just one isolated word in the whole verse: **no**. Leave the bracket simple (e.g.
`[Verse 1]`). The inline parenthesis `(word)` is enough on its own. Adding
`[Verse 1 - call and response]` for a single throwaway echo will trick the model into expecting a
full choir across the entire verse.

If 1–2 word echoes happen repeatedly on every line: **yes**. If every line ends with a one-word
bounce (like `Shes got balls (Balls)` / `Likes to crawl (Crawl)`), then add the descriptor:
`[Verse 1 - call and response]`.

---

## 3. Architectural Descriptors (Per-Section Modifiers)

Descriptors attach to section markers with a dash to modify local delivery and energy without conflicting with the global caption.

```text
[Marker - descriptor 1, descriptor 2, descriptor 3]
```

### Approved Descriptor Vocabulary
- **Vocal Style**: `whispered`, `belted`, `falsetto`, `spoken word`, `layered`, `harmonized`, `call-and-response`, `ad-lib`, `raspy vocal`, `powerful belting`, `breathy`, `clean vocals`, `screamed vocals`
- **Energy**: `quiet`, `sparse`, `building`, `rising`, `peak`, `intense`, `stripped back`, `full`, `low energy`, `high energy`, `explosive`
- **Instrument Focus**: `synth lead`, `guitar solo`, `organ solo`, `drum fill`, `bass groove`, `strings swell`, `brass stabs`, `808 bass`, `acoustic rhythm`
- **Dynamics & Production**: `fade in`, `fade out`, `abrupt cut`, `swell`, `drop out`, `reverb-heavy`, `dry`, `filtered`, `tape stop`, `sidechain kick`

### Descriptor Rules
- **Maximum 3 Descriptors**: Never attach more than 3 descriptors per bracket to prevent the model from singing tag names as lyrics.
- **Non-English Songs**: Place capitalized language tags at the start of the bracket before the dash:
  `[EN - Chorus - anthemic]`, `[JA - Verse - whispered, sparse]`.

---

## 4. Master Consistency Checklist

| Area | Caption (Global) | Lyrics (Temporal) |
| :--- | :--- | :--- |
| **Instrumentation** | `Vox Continental organ, Gibson SG P-90` | `[Instrumental - Vox Continental Organ solo]` |
| **Vocal Delivery** | `raspy male baritone vocal` | `[Verse 1 - powerful belting]` |
| **Energy & Mood** | `rebellious mood, driving tempo` | `[Bridge - shouting]`, `[Chorus - high energy]` |
| **Language / Type** | `is_instrumental: false, language: "en"` | `[EN - Verse]` with English lyric text |

---

## 5. System Prompt for LLM Auto-Annotation

```text
You are a music annotation assistant for the ACE-Step 1.5 music generation model.
Process each song entry and output ONLY a JSON object containing:
1. "caption": A hybrid caption (comma-separated tags followed by 8-16 comma-separated acoustic facts; no prose sentences).
2. "lyrics": Full lyrics with Capitalized structural markers ([Verse], [Chorus]) and max 3 architectural descriptors.

Output Format:
{
  "id": "<song_id>",
  "caption": "<tag list>. <8-16 comma-separated acoustic facts>",
  "lyrics": "[Intro]\n\n[Verse 1 - delivery]\nLyrics...\n\n[Chorus - energy]\n..."
}
```

<!-- BEGIN GENERATED: caption_spec -->

## Hard limits (generated — do not edit by hand)

Rendered from `modules/caption_spec.py` by `scripts/gen_doc_numbers.py`.
If these disagree with the prose above, the prose is wrong.

| limit | value |
| --- | --- |
| Caption length | 35–70 words |
| Caption hard cap | 900 characters |
| Front-loaded tags | 5–12, hard max 15 |
| Facts after the tag block | 8–16 |
| Descriptors per bracket | max 3 |
| Lyric line syllables | 6–10, flag over 12 |

Density rule, verbatim:

> Every token in the caption must pay rent in acoustic information. Do not write full prose sentences. A dense, comma-separated list of acoustic facts, structural transitions, and timbral descriptors provides higher attention focus to the DiT cross-attention heads than narrative filler.

<!-- END GENERATED: caption_spec -->
