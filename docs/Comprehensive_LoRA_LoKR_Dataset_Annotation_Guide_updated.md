# The Complete Guide to Caption & Lyric Annotation for ACE-Step 1.5 XL (LoRA & LoKR / SFT Variant)

> **Note on Adapters:** While this guide was refined during LoKR research, the Omission Principle and annotation mechanics apply identically to standard LoRA, DoRA, and LoHA, as all PEFT adapters learn via residual loss against the text encoder.

> Scope: This guide is the first comprehensive reference for preparing annotation data (captions and lyrics) for LoKR adapter training on ACE-Step 1.5 XL using the SFT model variant. It is written for dataset creators who will use an external LLM (such as Gemini 3.8 Flash) to assist with annotation.
>
> Model Variant Note: SFT is assumed throughout. Where Base or Turbo behavior differs materially, it is explicitly flagged. Turbo's removed alignment logic makes it unsuitable for LoKR training on this material.

> **Companion documents.** This guide covers *what to annotate*. The trainer-side
> mechanics — preprocessing failure modes, trainable modules, timesteps, loss,
> optimiser, checkpoints and DoRA defaults — live in
> [`LoRA_LoKR_Creation_Guide.md`](LoRA_LoKR_Creation_Guide.md). The artist-scoped
> caption/tag template feature lives in
> [`Artist_Scoped_Caption_Tag_Generation.md`](Artist_Scoped_Caption_Tag_Generation.md).

## 1. The Fundamental Distinction: Training Annotation vs. Creation Prompting

The most important concept in LoKR dataset preparation is that training annotations and creation prompts serve opposite purposes. Confusing them is the primary cause of weak adapters.

| Dimension | Music Creation Prompting | LoKR Training Annotation |
|---|---|---|
| Purpose | Elicit generation from the model | Teach the adapter what the audio actually contains |
| Content | Desired outcome | Observed reality |
| Common variables | Describe them — you want them generated | Omit them — the trigger tag absorbs them |
| Variable elements | May be omitted or left open | Describe them — they differentiate songs |
| Language | Creative, evocative, natural | Concrete, factual, consistent |

> **Note:** The Omission Principle

> **Note:** How the two conditioning streams are actually wired (verified from source): caption and lyrics are never merged into a single prompt string. They are encoded as two independent channels, and the model cross-attends to both:

```python
def prepare_condition(                 # acestep/models/sft/modeling_acestep_v15_base.py:1657-1673
    self,
    text_hidden_states: torch.FloatTensor,        # caption stream (256 token cap)
    text_attention_mask: torch.Tensor,
    lyric_hidden_states: torch.FloatTensor,       # lyric stream (512 token cap in training)
    lyric_attention_mask: torch.Tensor,
    refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
    refer_audio_order_mask: torch.Tensor,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.Tensor,
    silence_latent: torch.FloatTensor,
    src_latents: torch.FloatTensor,
    chunk_masks: torch.Tensor,
    is_covers: torch.Tensor,
    precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
    audio_codes: torch.FloatTensor = None,
):
```

Both streams are passed to self.encoder(...) and returned as one (encoder_hidden_states, encoder_attention_mask) pair (same file, lines 1676-1683), which is what the diffusion transformer cross-attends to. So a lyric-bracket descriptor and a caption tag are parallel conditioning on the same generation, and the adapter receives gradient from both.

**The encoding stage differs by stream**, and that difference is worth knowing before you annotate:

| Stream | Inside the Qwen3-Embedding-0.6B text encoder | Then, inside ACE-Step |
|---|---|---|
| Caption | Full forward pass; `.last_hidden_state` (acestep/core/generation/handler/conditioning_embed.py:74) | A single linear projection, `text_projector`, and no further transformer before cross-attention (acestep/models/sft/modeling_acestep_v15_base.py:1528, 1547) |
| Lyrics | Embedding-table lookup only — `text_encoder.embed_tokens(...)`, with every Qwen3 transformer layer bypassed (acestep/core/generation/handler/conditioning_embed.py:79; training path: acestep/training/dataset_builder_modules/preprocess_lyrics.py:24) | A linear projection to model width, then **8 bidirectional transformer layers** — `AceStepLyricEncoder` (acestep/models/sft/modeling_acestep_v15_base.py:583-603; `num_lyric_encoder_hidden_layers=8` at acestep/models/common/configuration_acestep_v15.py:173) |

So the lyric stream is not left unencoded: it skips Qwen3's transformer and is contextualised instead by ACE-Step's own lyric encoder, which self-attends across the entire lyric sequence (bidirectional — `is_causal=False`, modeling_acestep_v15_base.py:678) before the DiT cross-attends to the packed result. Each stream therefore receives exactly one transformer pass, and it is a different transformer in each case. Where word knowledge lives then differs: caption semantics come from a language-pretrained Qwen3 model, while a lyric descriptor draws its meaning from the Qwen3 embedding table alone — the 8-layer lyric encoder is trained from scratch on audio and contributes contextual integration, not vocabulary.

Neither encoder is adapted by training. LoKR targets are collected exclusively from the DiT decoder (`get_dit_target_modules` walks `model.decoder.named_modules()` and keeps only `q_proj`/`k_proj`/`v_proj`/`o_proj`, acestep/training/lora_injection.py:128-129), and everything else is frozen, with any `encoder`/`text_encoder` prefix explicitly excluded (lora_injection.py:136-143). The lyric encoder and the text projector both live under `encoder` (`self.encoder = AceStepConditionEncoder(...)`, modeling_acestep_v15_base.py:1621). Annotation consequence: phrasing consistency is a data-side lever — training will not repair an inconsistent descriptor convention for you.

Training consequence (this guide's recommendation, not a code constraint): do not name the same feature in both streams. A descriptor written into both the caption and every lyric bracket gives the adapter two independent, always-on predictors of the same acoustic feature, so the trigger has to out-compete both and ends up as a blend instead of a clean on/off switch.

The rule that governs every annotation decision (prevalence form - see 1.1 for the thresholds, because the strict "every song" wording is misleading on its own):

> A feature that is present in essentially every song must not be named in any caption or lyric descriptor. The trigger tag is the only name for that feature. A feature present in only part of the dataset may be named - but on the songs where it actually occurs, not only on the side that lacks it.

This is not a stylistic preference. It is a property of how LoKR adapters learn. During training, the adapter adjusts its weights based on both the caption and the lyric descriptors. If a descriptor appears in nearly every solo bracket, the adapter learns that descriptor is the cause of the solo sound rather than the trigger. The trigger becomes a weighted blend instead of a clean on/off switch.

The official ACE-Step training documentation confirms that structural tags help the model learn song structure, but it does not explicitly address the omission principle for LoKR training. This principle is derived from the broader LoRA/LoKR training literature, where it is well-established that adapters learn whatever is constant across the dataset and not described in the captions.

#### What This Means in Practice

Each row below is one decision, shown twice: what you would write for music creation, and what you write for training annotation. They are different files for different jobs, and only the right-hand column belongs in a dataset.

| The feature is... | Feature | Music creation prompt (you want it generated) | Training annotation (what the audio contains) |
|---|---|---|---|
| in essentially every song | raspy vocal | Write raspy vocal so the model sings raspy | Omit it. If every track is raspy, the trigger tag is the only thing that should name it |
| in essentially every solo | explosive playing | Write explosive guitar solo so the solos come out explosive | Omit it. [Guitar Solo] alone is the full description; explosive is already the trigger's job |
| in one verse of one song | whispered delivery | Write [Verse 1 - whispered] so that verse is whispered | Keep it, because it is song-specific - but only on the songs whose verse is actually whispered |
| in essentially every song | hard rock | Write hard rock so the genre is generated | Omit it. Genre is constant, so the trigger carries it; naming it again gives the adapter a second, competing name for the same sound |

Read the two columns as opposites on purpose. "Omit it during training" does not mean the concept disappears from the dataset - it means the trigger tag becomes its only name.

### 1.1 How Much Is "Every"? (prevalence bands)

"Every song" is shorthand, not a switch. What matters is how much information a descriptor carries: a tag present on nearly all songs predicts nothing, so the adapter is free to attribute the sound to the trigger instead. Use the proportion of songs that carry the feature:

| Prevalence | Treat it as | What to write |
|---|---|---|
| about 100% (24 of 24) | A constant | Omit it from every caption and bracket. The trigger owns it |
| about 90-99% (23 of 24) | A constant in practice | Omit it from all 24 songs, then describe the exception by what genuinely differs about it. One song cannot teach a switch, and tagging only that song turns the other 23 into an accidental default |
| roughly 10-90% (12 of 24, or 4 of 24) | A real variable | Tag it on every song where it occurs, the majority side included |
| a few songs (1-3, under roughly 10%) | Usually not learnable | Do not rely on a tag for control. There are too few examples for a reliable pathway; keep it only as decoration, or fold it into the trigger and omit it |

Two questions that look like the same question but are not:

> "23 of 24 songs share a feature - do I half the database?" No, and you do not tag the 23 either. Nearly all is all: omit the descriptor from all 24, and spend the effort you saved on what actually distinguishes the 24th song.
>
> "One song names the same descriptor in both its caption and its lyric bracket - is that a prevalence problem?" No. Prevalence is about the dataset; this is one sample naming the same feature twice, which gives the adapter two independent, always-on predictors of the same acoustic feature. Fix it by writing the feature in exactly one stream (section 5, and the placement table in 6.6).

## 2. The Trigger Tag (Persona / Custom Tag)

Your persona is the trigger tag that becomes the on/off switch for your LoKR. When the trigger is present in a prompt, the adapter activates the trained style. When absent, the base model's default behavior returns.

> **Tooling Warning:** If your dataset manager or JSON uses "tag_position": "prepend", do not write the trigger tag inside your raw .caption.txt file as well, or the builder will bake a duplicate trigger (acdc_bon, acdc_bon) into the prompt.

#### How the Trigger Works in Training

The custom_tag field in the dataset JSON is your trigger. During preprocessing, it is prepended (or appended, or replaced) relative to the caption. json

{ "custom_tag": "acdc_bon", "tag_position": "prepend", "caption": "mid-tempo swagger, party anthem, ..." }

The trigger is not inserted into the sung lyrics. It belongs in the caption only. Choosing a Trigger

> Be specific enough to avoid confusion with broader styles (e.g., acdc_bon rather than just acdc).
>
> The trigger absorbs every element that is constant across your dataset: the guitar tone, vocal character, production aesthetic, arrangement style, and genre identity.
>
> The trigger must be the only thing that reliably predicts the common sound.

## 3. Caption Annotation for LoKR Training

### 3.1 What Belongs in a Training Caption

A LoKR training caption describes the song-specific variables — the elements that differ between songs in your dataset. It does not describe the constants.

#### Structure:

```text
[Trigger Tag], [Song-Specific Tempo/Feel], [Song-Specific Mood], [Song-Specific Structure], [Song-Specific Solo Character], [Song-Specific Dynamic].
```

#### Example (corrected):

acdc_bon, mid-tempo swagger, party anthem, opens with a single guitar riff before drums crash in, shouted gang vocal on the title phrase, short stabbed solo phrases with wah-wah, abrupt cut ending.

#### Example (wrong — leaks constants):

hard rock, raspy male vocal, gretsch jet firebird, gibson sg, raw analog, punchy, 1970s

Every tag in the wrong example is a constant across a Bon Scott-era dataset. The adapter would learn that hard rock and raspy male vocal are the cause of the sound, not the trigger.

### 3.2 Caption Length

There is no caption-only token limit. The caption is inserted into the fixed training template

```text
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
{your caption}

# Metas
- bpm: ...
- timesignature: ...
- keyscale: ...
- duration: ... seconds<|endoftext|>
```

(acestep/constants.py:160-168, SFT_GEN_PROMPT) and the whole assembled string is tokenized with max_length=256, truncation=True (acestep/training/dataset_builder_modules/preprocess_text.py:18-24; the v2 preprocessor uses the same 256 cap). Because the metadata block sits after the caption, an over-long caption is what silently consumes the bpm/timesignature/keyscale/duration lines, and the sample then trains on a prompt with no metadata at all. The same 256-token cap applies at inference (acestep/core/generation/handler/conditioning_text.py:127-133), so an over-long caption cannot be used for generation either.

Budget: front-load 5–12 keywords (hard maximum 15) and keep the caption to **35–70 words** — 8–16 comma-separated acoustic facts, no prose sentences. The cap itself is measurably larger — 256 tokens total, 57 of them template, so about 199 for the caption, on the order of 130–160 words — and the measurement is in §7 under "The Literal String the Text Encoder Receives". Treat 35–70 as the working target and the 199-token figure as the hard ceiling; confirm any outlier with the tokenizer rather than by eye.

### 3.3 Mono Recordings

Do not assume a classic-recordings dataset is mono. The 1970s AC/DC albums this guide uses as its running example are stereo; the earlier draft's claim that Bon Scott-era AC/DC is "all mono or near-mono" was wrong and is withdrawn. Mono has to be established per dataset, from the files themselves, before it can be annotated.

If every file is genuinely mono, mono is a constant production characteristic: omit it from every caption and let the trigger absorb it. A real example of that pattern is a dataset built from one act's early-1960s 45 rpm singles, where a single mono master served every release.

If only some files are mono and the rest are stereo, mono is a variable and may be described in the caption of the mono tracks only. Such a caption could read:

```text
[TRIGGER_TAG], mono recording, mid-tempo swagger, party anthem, ...
```

One mechanical note, verified in the training pipeline: mono does not survive as a layout. Every file is resampled to 48 kHz and duplicated to two channels, and anything beyond two channels is cut down to the first two (acestep/training/dataset_builder_modules/preprocess_audio.py:4-21; see also section 7). "Mono recording" in a caption is therefore a provenance and production descriptor - what the source sounded and felt like - never a channel-layout instruction. That is exactly why it is legitimate to describe when it varies, and legitimate to leave to the trigger when it does not.

How to tell: inspect the files rather than trusting the reputation of an era. ffprobe -show_streams, or sox --i, reports the channel count of what you actually have.

### 3.4 What to Leave Out of Captions (an example, not a universal word list)

The table below is the worked example for ONE dataset - Bon Scott-era AC/DC. Those words are that dataset's constants. They are not universal negatives. Every artist has a different constant set, so a different artist's dataset needs its own list, derived the same way: assemble the material, write down the words that would be true of every track, and omit exactly those.

The categories transfer; the words do not. A Black Sabbath dataset's constants are not raw analog rock, and a Hank Williams dataset shares neither guitar tone nor vocal character with either of them. Do not carry this table into another dataset.

Words that are constant across this particular dataset - omit them from every caption when they are true of essentially every song (see the prevalence bands in 1.1):

| Category | Words to omit in THIS dataset (Bon Scott-era AC/DC) |
|---|---|
| Genre | hard rock, blues rock, rock n roll, pub rock |
| Vocal character | raspy male vocal, gritty male vocal, sneering, snarling, swaggering, bluesy shout |
| Guitar tone & gear | Gibson SG, Marshall Plexi, Gretsch Jet Firebird, clean-to-crunchy, edge of breakup |
| Production | raw analog, 1970s analog production, dry, punchy, in-your-face |
| Mood & energy | energetic, aggressive, driving, rebellious, relentless |
| Solo | pentatonic, blues-based leads, vibrato, explosive |

If a row is not constant for your artist, those words are not negatives for your dataset - they are ordinary song-specific annotations, and leaving them out is the mistake.

## 4. Lyric Annotation for LoKR Training

Every song lyric file requires explicit structural layout and acoustic formatting. Two different limits apply, and the tighter one wins: during training the lyric string is tokenized with max_length=512, truncation=True (acestep/training/dataset_builder_modules/preprocess_lyrics.py:6-12), while inference allows 2048 tokens (acestep/core/generation/handler/conditioning_text.py:137-144). Keep every lyric file inside 512 tokens. Past that, the tail of the song is silently dropped during training while still being singable at inference — exactly the train/inference misalignment you want to avoid.

### 4.1 Structural Markers

Every song section must begin with an explicit structural marker in square brackets. These markers control when elements occur.  Separate each section with a blank line.

Supported core markers: [Intro], [Verse], [Verse 1], [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Outro], [Hook], [Refrain], [Guitar Solo], [Instrumental], [Breakdown], [Silence], [Build-Up], [Break], [Fade Out], [Spoken]

#### Marker formatting rules:

```text
Section marker names are always capitalized: [Verse], [Chorus], [Outro].

Separate distinct sections with a blank line (\n\n).

Instrumental sections have no lyrics below the tag.

For non-English songs, place language tags at the start: [EN - Chorus - anthemic].
```

### 4.2 Architectural Descriptors (The Omission Principle Applied)

#### Descriptors attach to section markers with a dash:

```text
[Marker - descriptor 1, descriptor 2, descriptor 3]
```

Hard rule: maximum 3 descriptors per bracket. More risks the model singing tag names as lyrics.

Constant descriptors (omit from every bracket). This list is dataset-specific in exactly the way the caption table in 3.4 is: these are the words that happen to be constant for a raw 1970s hard-rock guitar-and-vocal dataset. Derive your own list the same way, and do not carry these rows into a different artist's dataset as if they were universal negatives.

| Category | Examples |
|---|---|
| Vocal | raspy vocal, sneering, snarling, swaggering, bluesy shout |
| Solo | blues-based leads, pentatonic, vibrato, bend, pull-off |
| Energy | high energy, driving, relentless, anthemic, explosive |
| Production | dry, raw production, live feel, punchy, in-your-face |

#### Variable descriptors (include when they describe song-specific variation):

| Category | Examples |
|---|---|
| Vocal variation | whispered, spoken word, menacing, building, strained |
| Solo character | wah-wah engaged, slow sustained bends, short stabbed phrases |
| Structural variation | solo guitar riff intro, no drums, abrupt cut, fade out |
| Dynamic variation | sparse, stripped back, explosive, full band |

### 4.3 Lyric Text Formatting Rules

| Rule | Detail |
|---|---|
| Capitalization | Section markers capitalized. Use UPPERCASE for belting, screaming, or shouted delivery. Use lowercase for normal sung lines. |
| Syllable cadence | Keep lines between 6 to 10 syllables for optimal beat alignment. Deviation within a section should not exceed 2 syllables per line. |
| Section spacing | Always separate distinct sections with a blank line (\n\n). |
| Vocal intensity | UPPERCASE for belting, screaming, or shouted delivery. |
| Harmonies & backing | Use (parentheses) for backing vocals, echoes, and call-and-response. |
| Instrumental songs | Set lyrics to [Instrumental] only. |

### 4.4 Capitalization for Shouted Words

UPPERCASE in lyrics signals stronger vocal intensity — shouting, belting, or screaming.

#### Example:

```text
[Bridge - building intensity, ascending vocal]
NO, BABY PLEASE DONT LEAVE ME
WHY MUST YOU GO AWAY AND DO THIS TO ME, BABY
I want to suffer for you, SUFFER, SUFFER
OH BABY please dont GO
NO
```

The ALL CAPS lines are correct because the vocalist is genuinely belting. If he is singing normally, use lowercase.

### 4.5 Punctuation Removal

> **Evidence note (corrected):** punctuation does not break a 25 Hz text grid. The "25 Hz" figure in ACE-Step is a latent rate, not a text rate: the only operation prepare_condition applies to the Lyric hints is a length crop to the latent sequence (precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :], acestep/models/sft/modeling_acestep_v15_base.py:1690), and that tensor replaces src_latents only when is_covers is set (same file:1699). Nothing there constrains punctuation or line length. Punctuation also has no fixed token cost: the lyric channel is tokenized with truncation at 2048 tokens at inference (acestep/core/generation/handler/conditioning_text.py:137-144) and at 512 tokens during training (acestep/training/dataset_builder_modules/preprocess_lyrics.py:6-12).

The rule below is therefore hygiene, not a frame-alignment requirement — and it is still worth following anyway, because the tokenizer is sub-word based: punctuation is billed as its own token, or fused into an adjacent one, for no phonetic benefit. Note the correction: no stage of this repository's lyric path strips it for you (see "Why (corrected)" below).

Remove all punctuation that the model is not trained to interpret: quotation marks ("), periods (.), question marks (?), exclamation marks (!), backslashes (\), and any other non-phonetic punctuation at the end of lines.

Why (corrected): this repository has no lyric normalizer. The training lyric path tokenizes the raw string and applies no case folding, punctuation stripping or emoji removal of its own (acestep/training/dataset_builder_modules/preprocess_lyrics.py:6-12), and no later stage sanitizes lyric text. So the punctuation you leave in is the punctuation the adapter sees. Removing it is still worth doing, for a simpler reason: a sub-word tokenizer normally bills punctuation as tokens of its own (dont, for example), so stray symbols consume budget without carrying pronunciation, and they matter most in a lyric file that is already near the 512-token ceiling.

> **Note:** Commas are a special case.

There is a vital difference between grammatical punctuation and acoustic punctuation in singing models:

```text
Trailing commas at the end of a line (word,\n): Bad, by this guide's reasoning rather than by measurement. The newline (\n) already signals the phrase boundary, so the comma adds a token without adding information. The mechanism this guide used to give for why that is harmful (trailing hesitation, vocal fry, rhythmic stuttering) is plausible but unmeasured, and no upstream document states it.

Inline commas inside a line (word, word): Helpful, but only when used acoustically. This guide's convention is that an inline comma marks a brief musical breath, a micro-pause. Upstream documentation describes commas only for caption tags and never for lyric timing, so treat this as an empirical convention and apply it consistently across the dataset.
```

#### Wrong:

Baby, please don't go, baby, please dont go. You know I love you so, baby, please dont go.

#### Right:

Baby please dont go Baby please dont go Baby please dont go down to New Orleans You know I love you so Baby please dont go

### 4.6 Apostrophes and Phonetic Spelling (partly unverified - read the note)

> **Note:** Evidence note (corrected). The earlier version of this section asserted three things that ACE-Step's own code and documentation do not support:

1. That "the SFT tokenizer and alignment model treat apostrophes as ambiguous". No alignment model exists in the repository, and no apostrophe handling exists anywhere in the training or inference lyric path.

2. That dropping an apostrophe makes the model sing the wrong word, described with the Chinese term for it. That failure mode is not documented anywhere in ACE-Step, and the term appears in no ACE-Step file.

What IS verified is more modest. Lyrics reach the model as a raw string: the training path tokenizes the string as given (acestep/training/dataset_builder_modules/ preprocess_lyrics.py:6-12) and embeds it directly (same file:24), with no normalization step of its own. So an apostrophe is a token boundary, and dont, don't and do not are three different token sequences - but which one produces better pronunciation has not been measured. Note also that nothing in this repository strips punctuation, emoji or case before tokenization, contrary to what the older wording of section 4.5 claimed.

Why the rules exist at all: ACE-Step's official generation documentation does recommend plain apostrophe removal and vowel elongation in lyric text, and does document parentheses for backing vocals (docs/en/Tutorial.md:540-558; docs/en/ace_step_musicians_guide.md:331-335; .claude/skills/acestep-songwriting/SKILL.md:131-132). Those are lyric-writing tips for generation. They say nothing about apostrophes being sung wrong, and nothing about the respellings below.

The operational rule that follows: pick ONE convention for your dataset and apply it identically in every file. Consistency matters more than which form you choose, because the adapter learns the mapping you actually give it. If a form costs extra tokens you can see it with the text-encoder tokenizer, but the only real test of pronunciation is a listening A/B after training.

#### Tier 1 - apostrophe removed, no respelling

The remaining string is normally not a word, so it is unambiguous as text. This is the closest thing to an upstream-endorsed convention, because it is what the official documentation demonstrates.

| Standard | Write as |
|---|---|
| don't | dont |
| didn't | didnt |
| isn't | isnt |
| aren't | arent |
| I've | Ive |
| I'm | Im |
| you've | youve |
| you're | youre |
| that's | thats |
| it's | its |
| let's | lets |
| we've | weve |
| ain't | aint |
| you'll | youll |
| they'll | theyll |
| there'll | therell |
| who'll | wholl |
| he'd | hed |

Note the correction to the earlier table, which listed you'll -> yool: that is a respelling, not an apostrophe removal, and it is not a form ACE-Step documents.

#### Tier 2 - respelling (this guide's convention, unverified)

These forms exist because a plain removal would produce a different real word (won't -> wont, can't -> cant). Whether respelling improves pronunciation has not been measured. Treat the middle column as a placeholder you may replace with your own convention; what matters is that the same word is written the same way in every file.

| Standard | Write as | Real word if you only drop the apostrophe |
|---|---|---|
| won't | wohnt | wont (accustomed) |
| can't | kant | cant (hypocritical talk) |
| we'll | weel | well |
| he'll | heel | hell |
| she'll | sheel | shell |
| we're | weer | were |
| I'll | eyel | ill |
| I'd | eyed | id |
| we'd | wead | wed |
| 'cause | kuz | cause |
| she'd | sheed | shed |
| who're | who-er | whore |
| 'bout | bowt | bout |

#### Tier 3 - vowel elongation and melisma (a generation convention, not a training rule)

When a vocalist stretches a vowel across several beats, repeated vowels and hyphens are the conventional notation. This is a lyric-writing convention rather than a training requirement: no ACE-Step LoRA, LoKR or SFT training document mentions melisma or vowel elongation at all (checked against docs/en/LoRA_Training_Tutorial.md and docs/en/Large_Scale_SFT_Training_Guide.md, which discuss lyric files, transcription and structural tags only). The word melisma occurs once in the repository, as a genre descriptor in the caption vocabulary (acestep/genres_vocab.txt, e.g. "trap, Eastern melisma") - it is not lyric guidance. The official tutorial's own warning applies: "use cautiously - effects are unstable, sometimes ignored or mispronounced".

| Delivery | Notation |
|---|---|
| Single sustained vowel | baaaaby |
| Vowel re-articulated rhythmically | bay-aa-aa-bee |
| Vowel sustained, consonant lands late | baaaay-bee |
| Clean two-syllable delivery | bay-bee |

A melisma counts as one syllable regardless of how many vowels are written.

### 4.7 Words That Stay Dictionary Words

#### Words that should NOT be respelled phonetically (because they are already dictionary words or informal reductions that the model understands):

gonna, wanna, gotta, gimme, lemme, kinda, outta, dunno, ya, em, til, round, bout, mm, ah-uh-huh, ooh, woah, yeah

Dropped -ing endings: If the vocalist drops the final g, write the dropped form: goin, lyin, waitin, tryin, nothin, somethin. If he pronounces it, write the full form. Verify against the audio.

### 4.8 Backing Vocals and Parentheses

> **Note:** Evidence note. Verified: content in parentheses is processed as background vocals or harmonies ("Content in parentheses is processed as background vocals or harmonies", docs/en/Tutorial.md:540-548; restated in docs/en/ace_step_musicians_guide.md:331-335 and .claude/skills/acestep-songwriting/SKILL.md:131). NOT documented: placement. The official docs say nothing about whether a parenthesis at the start, middle or end of a line changes timing, so the placement rules below are this guide's empirical convention. Apply them consistently within a dataset rather than treating them as a documented timing language.

When a verse mixes vocal styles - opening solo, a gang-shouted unison phrase, an echo answer, then an overlap - do not split the verse into a dozen tiny brackets. Use the Macro/Micro rule:

> Macro (the bracket header): the section's overall arrangement profile, written with the approved descriptors (maximum 3).
>
> Micro (line-by-line syntax): typography, parentheses and line placement direct the moment-by-moment vocal interplay.

#### Tags versus parentheses: which one do you need?

They do different jobs, and the difference is scope. A bracket descriptor is a commitment about the whole section: it says what the section IS, and the adapter then expects that texture on its lines. Parentheses are a local annotation: they mark one echo, one answer or one overlap, and say nothing about the rest of the section. Getting that scope wrong is the most common 4.8 error, in both directions.

| Ask yourself | The answer points to | What the output then claims |
|---|---|---|
| Does the interaction span the section, on most of its lines? | A header tag on the section marker | Backing vocals throughout: call and response, layered, gang vocals or harmonies describe the whole section |
| Is it one event - one echo, one answer, one overlap? | Parentheses, header left plain | A local instruction: the backing vocal is here, at this point, and nowhere else |
| Is it rare but real - a single throwaway echo in an otherwise solo verse? | Parentheses only | The echo is notated without committing the section to a texture it does not have |
| Is the section one continuous texture - all gang-shouted, all layered, all harmonised? | Header tag plus plain text | The texture is stated once, at the header, instead of restated on every line |

#### How to read that table, row by row:

> Row 1 - the test is coverage. If you would have to write a parenthesis on most lines to describe what you hear, the section has an arrangement profile, and a profile belongs in the header, stated once.
>
> Row 2 - the test is locality. If the backing vocal is a single event, parentheses place it exactly, at the word or the line where it happens.
>
> Row 3 - the trap, and it catches people in both directions. Leaving the header plain is correct: the parenthesis alone carries the echo. Adding [Verse 1 - call and response] is not "more accurate", it is a different claim - that the whole verse trades phrases - and the model will write you one.
>
> Row 4 - the mirror trap. If every line is gang-shouted, the header already says so, and (DONE DIRT CHEAP) on each line only restates it. Write the line once, in caps, under the header.

#### Decision list:

1. One isolated echo in an otherwise solo verse -> parentheses only; the header stays [Verse 1]. Coverage is one line, so there is no section profile to name.

2. An echo, trade or answer on most lines -> add the header descriptor [Verse 1 - call and response] and keep the parentheses on the individual lines. The header states the arrangement; the parentheses still say which word is the answer.

3. The section is one continuous texture (all gang-shouted, all layered) -> header descriptor plus plain text; parentheses add nothing.

#### Line types, and how each one is written

One convention generates the whole table: capitals carry intensity, parentheses carry "not the lead", and a new line carries time. Both that convention and the placements below are this guide's, not documented upstream (see the evidence note above). Applied consistently, they are learnable within one dataset.

| Line type | How you write it | Example | What the model has to work from |
|---|---|---|---|
| Solo lead, normal sung delivery | Lowercase, no parentheses | Well me and the boys are out to have fun | One voice, mid-intensity, on the beat |
| Solo lead, shouted or belted | ALL CAPS, no parentheses | GET OUT OF MY WAY | One voice, high intensity |
| Full-band unison, same words | ALL CAPS, no parentheses; the header carries [gang vocals] | [Chorus - gang vocals]  DIRTY DEEDS DONE DIRT CHEAP | One text, many voices |
| Backing echo that answers the lead line | Parentheses on their own line, directly below the line they answer | Im a LIVE WIRE / (LIVE WIRE) | Two lines = two time positions; the answer lands after the lead finishes |
| Backing lead-in, sung before the lead enters | Parentheses at the start of the lead line | (Look out) here comes the man | Backing first, lead takes over inside the same line |
| Shouted gang backing | ALL CAPS inside parentheses | (DONE DIRT CHEAP) | Backing vocal, high intensity |
| Overlap during the lead line | Parentheses at the exact word where the backing lands | You know I love you so (Please dont go) | Two voices at once; the parenthesis position is the timing |
| Layered part, different words | Parentheses around the backing words, plus a [layered] header | [Bridge - layered]  Neck ties contracts (DONE DIRT CHEAP) | Two independent texts at once, which the header marks as deliberate |
| Harmony on the same words, held | Header descriptor only; write the line once, no parentheses | [Chorus - vocal harmonies]  HIGH VOLTAGE ROCK N ROLL | One text, several voices in harmony; no second text to place |

Two details in that table are conventions, not rules, so choose them once and keep them constant:

> The gap between a header and its words ("[Chorus - gang vocals]  DIRTY DEEDS...") is your own: pick one separator and use it everywhere. What matters is that the header sits as a separate word group at the front of the line.
>
> The mid-line overlap is the only row where a parenthesis sits inside running text instead of bracketing a whole phrase. When the backing part is longer than a word or two, give it its own line (the trailing-answer row) or promote the section to [layered].

Caps and parentheses can combine - ALL CAPS inside parentheses is a shouted backing part - but neither one implies the other. A capped line inside a plain section is one belted lead voice, not a crowd; a crowd is a header descriptor plus plain words.

#### Background vocals in the middle of a line

Mid-line backing causes the most confusion, so here it is explicitly. There are only three positions a backing vocal can occupy relative to the lead, and each is written differently. Position does that much work because the lyric channel carries text only: with no alignment model (see 4.5), line breaks, word order and the position of a parenthesis are the only "clock" the model is given.

```text
Opening - the backing vocal starts before the lead word, then hands over to it.
The parenthesis opens the line:

    (Look out) here comes the man

Middle - the backing vocal enters while the lead is already singing. The
parenthesis sits at the exact word it lands on, and the words after it belong to
the lead again:

    You know I love you so (Please dont go)

Trailing - the backing vocal answers after the lead line has finished. Put it on
its own line below, which is what separates an answer from an overlap:

    Im a LIVE WIRE
    (LIVE WIRE)
```

Position is the whole message. Written inline - "Im a LIVE WIRE (LIVE WIRE)" - the same two parts read as an overlap, because both texts share one line and therefore one moment. Split onto two lines they read as a call and an answer. Same words, different timing claim, and the model has nothing else to go on.

Two limits follow from the same idea:

> Do not stack several mid-line parentheticals onto one line. If the backing vocal needs two or more of them, the section is layered, and that belongs in the header descriptor with the backing words written out.
>
> Keep one parenthesis to one phrase. When the backing part is longer than a word or two, give it its own line (a trailing answer) or promote the section to [layered]; do not bury a whole sentence inside the lead line.

#### Worked examples

Example A - one isolated echo. Coverage is a single line, so the header stays plain:

```text
[Verse 1]
Well me and the boys are out to have fun
(Look out) here comes the man
You know I love you so
Im a LIVE WIRE
```

#### Read back line by line:

> Line 1 - solo lead, normal delivery: lowercase, no parentheses, one voice. Line 2 - backing lead-in: the parenthesis opens the line, the backing finishes before "here", and the lead takes over inside the same line. Line 3 - solo lead again, and note what is absent: no parenthesis follows "love you so", because nothing answers it. Line 4 - solo lead that is belted: LIVE WIRE is the title phrase and is shouted, so it is capped. Caps carry intensity only; a capped line inside a plain section is one voice, not a crowd.

Header check: one echo on one line out of four is the "rare but real" row of the decision table above. [Verse 1] stays plain and the parenthesis alone marks the echo. Adding [Verse 1 - call and response] would claim the whole verse trades phrases.

Example B - an echo on most lines. Coverage is the section, so the descriptor goes into the header AND the parentheses stay on the individual lines:

```text
[Verse 1 - call and response]
Shes got balls (Balls)
Likes to crawl (Crawl)
Shes got balls (Balls)
```

Read back: every line ends with a one-word bounce, which is a section-wide arrangement, so the "spans the section" row applies and the header states it once. The parentheses are not redundant here: the header says the section trades phrases, the parentheses say which word each answer repeats.

Example C - answer or overlap. The same words, and the only difference is where the parenthesis sits.

```text
Im a LIVE WIRE
(LIVE WIRE)
```

Two lines: the answer lands after the lead line has finished.

```text
Im a LIVE WIRE (LIVE WIRE)
```

One line: those words now read as an overlap, because the model gets one time slot for the line and both texts sit inside it.

```text
You know I love you so (Please dont go)
```

That last form is the mid-line overlap: the parenthesis sits at the exact word where the backing lands, and the lead continues after it. Which form is the accurate one is a decision about the recording, not a style preference.

Example D - one continuous texture. When the whole section is gang-shouted or harmonised, the header carries it and the lines stay plain:

```text
[Chorus - gang vocals]
DIRTY DEEDS DONE DIRT CHEAP
DIRTY DEEDS DONE DIRT CHEAP
```

No parentheses: with no lead/backing split left to mark, they would only restate the header on every line. The words are written once per repetition because the repetitions are the arrangement.

The four examples differ only in coverage: A is one line, B is every line, C is the timing of a single answer, and D is the section as a whole. Working out which one you are looking at is the whole of 4.8.

#### When SHOULD You Break It into Multiple Sections?

> **Note:** Only break the verse into multiple section brackets if the rhythm section physically stops or shifts structure:

> If the drums stop and only vocals/clapping continue:

```text
[Verse 1]
Lyrics...
```

```text
[Breakdown - stripped back, call and response]
(BAD BOY BOOGIE) I said up they said down
```

If it's just one isolated word in the whole verse: No. Leave the bracket simple (e.g., [Verse 1]). The inline parenthesis (word) is enough on its own. Adding [Verse 1 - call and response] for a single throwaway echo will trick the model into expecting a full choir across the entire verse.

If 1–2 word echoes happen repeatedly on every line: Yes. If every line ends with a 1-word bounce (like "Shes got balls (Balls) / Likes to crawl (Crawl)"), then add the descriptor: [Verse 1 - call and response]

### 4.9 The Lyric Token Budget: 512 in Training, 2048 at Inference

Two ceilings apply to the same lyric file, and they are not the same number:

| Path | Limit | Where it is enforced |
|---|---|---|
| Training | 512 tokens | acestep/training/dataset_builder_modules/preprocess_lyrics.py:6-12 |
| Inference | 2048 tokens | acestep/core/generation/handler/conditioning_text.py:137-144 |

512 tokens is the binding constraint, so treat it as your dataset limit: a 600-token lyric file still trains, but only its first 512 tokens do, so the adapter never learns the ending (while that same file sings to the end at inference). If a song cannot fit in 512 tokens, split it into two samples instead of letting the tail be dropped.

> **Do not follow the "set your lyric truncation to 2048" advice.** It circulates in the reference notes ("Modify your training data's lyric truncation length to be consistent with this 2048 limit") and it is wrong for dataset creation. No dataset-side setting can raise the training cap, because the 512 limit is hard-coded in the preprocessor as `max_length=512, truncation=True` (`acestep/training/dataset_builder_modules/preprocess_lyrics.py:6-12`) rather than exposed as a flag. 2048 is the *inference* ceiling and nothing else. Following the advice therefore produces exactly the misalignment it claims to prevent: the adapter learns the first 512 tokens of a 2048-token file and meets the remaining three-quarters for the first time at generation.

Also note what the lyric channel actually is: a bare string. Training feeds this field verbatim, with a fallback of the literal text [Instrumental] when the field is missing (acestep/training_v2/preprocess.py:262), and it reaches the model through ACE-Step's lyric encoder rather than through Qwen3's transformer: the string is embedded by table lookup (text_encoder.embed_tokens, acestep/training/dataset_builder_modules/preprocess_lyrics.py:24) and then contextualised by the 8-layer AceStepLyricEncoder (stage-by-stage detail in the wiring note at the top of the guide). The "# Languages" / "# Lyric" wrapper you see in the generation UI is added at inference only (acestep/core/generation/handler/prompt_utils.py:27-29) — do not write those headers into your dataset files.

## 5. The Relationship Between Captions and Lyrics

The caption and lyrics are two separate conditioning streams that the adapter reads during training.

| Stream | What it conditions | What the adapter learns from it |
|---|---|---|
| Caption | Global sound portrait | "When these words appear, produce this overall sound" |
| Lyrics + section descriptors | Temporal script | "When this section marker and these descriptors appear, produce this specific moment" |

Both streams are active during training. The adapter adjusts its weights based on both. If a descriptor appears in every solo bracket across your dataset, the adapter learns that descriptor is a cause of the solo sound — exactly the same leakage problem as putting "hard rock" in every caption.

#### Consistency checklist:

| Area | Caption (Global) | Lyrics (Temporal) |
|---|---|---|
| Instrumentation | Vox Continental organ, Gibson SG P-90 | [Instrumental - Vox Continental Organ solo] |
| Vocal delivery | raspy male baritone vocal | [Verse 1 - powerful belting] |
| Energy & mood | rebellious mood, driving tempo | [Bridge - shouting], [Chorus - high energy] |

## 6. Variable Vocal Styles, Novel Tags, and the Default State

### 6.1 The Core Problem: What Happens When a Style Is Untagged

The Omission Principle states that features present in every song must be omitted. But what about features present in most songs? Or some songs? This is where dataset creators most often go wrong.

When you omit a tag, you are not creating a neutral state. You are creating a default. The adapter learns that the untagged audio represents the baseline sound of the trigger. The behavior of that default depends entirely on the distribution of your training data.

Consider a dataset where the singer's vocal style varies:

> 80% of songs: raspy vocal
>
> 20% of songs: melancholic vocal

Scenario A: You omit the tag on raspy songs but add melancholic vocal to the melancholic ones.

Result: The adapter learns two pathways.

> No vocal style tag → defaults to raspy (the dominant untagged pattern)
>
> melancholic vocal tag → overrides to melancholic

This works, but the default is biased toward raspy. If you later generate an untagged song, it will be raspy.

Scenario B: You omit both tags entirely.

Result: The adapter has no conditional signal to separate the two styles. It learns a single, blended representation—vocals that are somewhere between raspy and melancholic. This averaged texture is usually unsatisfying for both styles. You lose control entirely.

Scenario C (Recommended): You tag every style consistently.

Result: The adapter learns distinct conditional pathways.

> raspy vocal → raspy
>
> melancholic vocal → melancholic
>
> Neither tag → neutral blend (the base model's default, not your dataset's average)

This is the only approach that gives you full control.

### 6.2 The Majority Trap: Why "Mostly Untagged" Fails

If 80% of your songs are raspy and untagged, the adapter's default will be raspy. You may think you have a "raspy LoKR" with a "melancholic option." In reality, you have a LoKR where raspy is the baseline and melancholic is an override. This can work, but it is fragile:

> The raspy signal is diluted because it competes with the trigger tag for credit.
>
> The trigger absorbs "raspy" as part of its identity, making it harder to generate non-raspy vocals even with the melancholic tag.
>
> If you ever want to generate a truly neutral vocal, the model cannot do it—the trigger itself now implies raspy.

Rule: Tag the majority style explicitly. Do not let it become the default by omission. Every style you want to control must be tagged.

### 6.3 Novel Style Tags: You Can Invent Them

You are not limited to tags the base model already knows. ACE-Step 1.5 uses the **Qwen3-Embedding-0.6B** text encoder (acestep/core/generation/handler/init_service_loader_components.py:90) — an embedding model, not a generative LLM, and not a 4B model. Every 1.5 variant, XL included, loads this same 0.6B encoder. It already has meaningful vectors for words like drunk, slurred, melancholic, breathy.

When you train a LoKR, you are not teaching the model a new word, and you are not adapting the encoder either — the encoder is frozen and LoKR touches only the DiT decoder's attention projections (acestep/training/lora_injection.py:128-129, 136-143). What you are teaching is an association between the vector the encoder already produces for "slurred" (the text embedding for it) and specific acoustic features in your audio.

#### Example: Tagging a drunk vocal style

Your dataset contains songs where the vocalist slurs words and has an unsteady, loose delivery. You tag them consistently: text

```text
[Verse 1 - slurred, unsteady delivery]
well I been drinkin all night long
and I cant seem to find my way back home
```

#### Caption for these songs:

text

acdc_bon, loose drunken swagger, stumbling mid-tempo groove, ...

After training, the tag slurred or drunk will activate the learned association. You can use any words that semantically capture the style—but use them consistently across your dataset. Do not use slurred on one song and drunk on another. Pick one phrasing and stick with it.

#### Combining novel and established tags:

For best results, anchor your novel tag with a more standard descriptor: text

```text
[Verse 1 - slurred, raspy, loose]
```

The model understands raspy from its base training. The LoKR teaches it how slurred modifies that sound.

### 6.4 Solo Tagging: Style vs. Duplication

You can tag solos to control where they occur, what instrument plays them, and what style they have. You cannot tag a solo to duplicate a specific, exact performance.

#### What works:

| Tag | What it teaches the adapter |
|---|---|
| [Guitar Solo - wah-wah, pentatonic] | Generate a guitar solo with wah-wah and pentatonic phrasing |
| [Instrumental - saxophone, melancholic] | Generate a melancholic saxophone solo |
| [Breakdown - drum solo, explosive] | Generate an explosive drum break |

The adapter learns the general characteristics of that type of solo and generates a new solo when prompted.

#### What does not work:

| Tag | Why it fails |
|---|---|
| [Guitar Solo - the solo from song X] | The adapter learns style features, not unique audio sequences. It cannot retrieve a specific performance. |
| [Solo - solo_01] | Arbitrary identifiers carry no semantic content; the encoder returns an arbitrary, non-generalising vector for them, so any association the adapter forms with them is arbitrary too. No reusable style feature is taught. |

The distinction: LoKR learns generalizable style features, not exact audio sequences. You are teaching the model how your singer's solos tend to sound, not what a specific solo was.

#### Practical solo tagging strategy:

> Describe the solo's character, not its identity.
>
> Use consistent descriptors for solos that share a style.
>
> Use structural markers to control placement: [Guitar Solo], [Instrumental], [Breakdown].
>
> Combine with the trigger tag in the caption to bring in the overall style.

#### Example:

```text
[Guitar Solo - slow sustained bends, wah-wah engaged]
```

#### Caption:

acdc_bon, mid-tempo swagger, party anthem, ...

The trigger brings in the artist's guitar tone and style. The solo bracket descriptor adds the specific character of this solo.

### 6.5 The Default State: What Happens When You Omit a Variable

The Omission Principle is about constants - features present in essentially every song (section 1.1 gives the bands; "essentially every" means "so nearly every that the tag carries no information", not literally 100%). For variables - features present in some songs but not others - omission creates a default that is learned from the distribution of your data.

#### The rule for variables:

> If a feature is present in a song, tag it. If it is absent, do not tag it. The adapter learns: tag present -> feature present; tag absent -> feature absent, or, when the tag is rare, the default.

The critical caveat: if the feature is present in most songs and you omit the tag on those songs, the adapter learns that the feature is the default. The tag then works only as an override for the minority case.

#### Example: raspy vs. clean vocals (12 songs)

#### Wrong approach - tag only the minority:

> Raspy songs (10): no vocal tag Clean songs (2): clean vocal tag

Result: the trigger absorbs raspy as part of its identity, clean vocal becomes an override, untagged prompts default to raspy, and two tagged songs are in any case too thin a signal to learn a reliable clean pathway from.

#### Right approach - tag every song where the style occurs:

> Raspy songs (10): raspy vocal tag Clean songs (2): clean vocal tag

Result: two distinct pathways. Untagged prompts produce the base model's default vocal, not your dataset's average, and you have full control.

#### The middle and partial cases

| Split across the dataset | What you actually have | What to do |
|---|---|---|
| 10 raspy / 2 clean | A dominant style plus a real minority | Tag both. The minority is small but the pathway is clean, because every occurrence is labelled |
| 6 raspy / 6 clean | No default state at all | Tag both. Nothing is left for the trigger to absorb, and whichever side you leave untagged becomes a phantom default for that half of the dataset |
| 11 raspy / 1 clean | A single exception | Treat raspy as a constant: omit it from all 12 songs (section 1.1), and describe the one clean song by what genuinely differs about it. A one-song tag is not learnable, and leaving raspy untagged on 11 songs teaches the trigger that raspy is its baseline anyway |
| 8 of 12 songs have harmonies on the chorus | A variable that lives inside sections, not a whole-song property | Tag it where it occurs - a bracket descriptor on the chorus brackets that have it - and leave it off the ones that do not. Do not put it in the caption, which would mark the entire song as harmony-bearing |

One prevalence question that looks like a distribution problem but is not: a single song that names the same feature in both its caption and its lyric bracket. That is redundancy inside one sample, not a majority effect - see the "name it once" row in section 6.6.

These bands are a recommendation about what an adapter can learn, not a code constraint: ACE-Step enforces no prevalence rule. The cost of tagging everything is annotation effort plus a slightly diluted trigger; the cost of tagging too little is a default state, which is worse because it stays invisible until you try to generate against it.

### 6.6 Summary Table

| Situation | Wrong approach | Right approach |
|---|---|---|
| Feature in essentially every song (about 100%, certainly above roughly 90%) | Tag it on every song, or tag it only on the few songs that lack it | Omit it everywhere - the trigger absorbs it (section 1.1). Describe the exception songs by what genuinely differs about them |
| Feature in about half the songs | Tag only the majority side, which turns it into an accidental default | Tag every song where it occurs, majority side included |
| Feature in a clear minority (roughly 10-50%) | Tag the majority style instead, leaving the minority untagged | Tag every song where it occurs, and leave it off the rest |
| Feature in one or two songs only | Invent a tag and expect it to work as a switch | Do not rely on it for control. There are too few examples to learn a pathway; keep it only if you accept that it is decorative |
| The same feature named in both the caption and the lyrics of one song | Two reminders are better than one | Name it once, in the stream that owns it (table below) |
| Novel style (drunk, slurred) | Assume the model cannot understand the wording | Tag it consistently - the model learns the association, and the wording must be identical on every song |
| Solo style | Try to duplicate one specific solo | Tag the style - the model generates new solos in that style |
| Solo identity | Use arbitrary ID tags | Use descriptive tags with semantic meaning |

#### Where a tag belongs when it is needed:

| The property is... | Write it in... | Because |
|---|---|---|
| Constant across the whole dataset | The trigger tag (inside the caption) | The trigger is the single name for that feature |
| Global to the song, and it varies between songs | The caption | Captions describe the song's overall sound |
| Local to one section, and it varies between songs | The bracket descriptor in the lyrics | Brackets describe a moment in time |
| A momentary backing vocal, echo or overlap | Parentheses on that lyric line, not a tag | It is not a property of the section, so a tag would over-promise (section 4.8) |

## 7. Dataset File Structure and Preprocessing

#### Required Files

Each training sample requires three files sharing the same basename, placed in the same directory:

```text
dataset/
├── song1.flac             # Audio (.mp3, .wav, .flac, .ogg, .opus)
├── song1.lyrics.txt       # Lyrics (recommended)
├── song1.caption.txt      # Caption (or in JSON)
├── song1.json             # Optional: bpm, keyscale, timesignature, language
```

#### File lookup priority during scanning:

```text
{filename}.lyrics.txt (recommended)

{filename}.txt (backward compatible)

{filename}.caption.txt for captions
```

#### Sample-Level Metadata: What Is Actually Read, and From Where (verified)

There are two ingestion paths, and they read different things. Only the keys listed below are consumed; any other key in your JSON is silently ignored.

#### Path A — files on disk sharing the audio basename (the original dataset builder, used when you point it at a directory):

| File | What it supplies |
|---|---|
| <basename>.caption.txt | caption (acestep/training/dataset_builder_modules/audio_io.py:30-38) |
| <basename>.lyrics.txt | lyrics; falls back to <basename>.txt for the older convention (same file:72-85) |
| <basename>.json | sidecar metadata; read keys are exactly caption, bpm, keyscale, timesignature, language (acestep/training/dataset_builder_modules/scan.py:84-97) |
| <basename>.csv | per-file rows matched by filename; read keys are bpm, key, caption (same file:99-108) |
| the audio file itself | duration is measured from the audio, never read from metadata (same file:54, audio_io.py:88-108) |

Precedence on Path A: sidecar .json overrides .caption.txt, then CSV overrides both. custom_tag, tag_position, genre_ratio and all_instrumental are dataset-level fields on this path, not per-sample ones — writing "custom_tag" into a per-sample .json does nothing here. A lyrics file present for a track also forces is_instrumental=false (scan.py:66-68), as does a sidecar language other than "instrumental" (scan.py:94-97).

#### Path B — one dataset JSON (v2 "Side-Step" preprocessing, --dataset-json):

{ "metadata": {

```json
"custom_tag": "acdc_bon",
"tag_position": "prepend",
"genre_ratio": 0
```

}, "samples": [

```json
{
  "audio_path": "audio/song1.flac",
  "filename": "song1.flac",
  "caption": "acdc_bon, mid-tempo swagger, party anthem, ...",
  "lyrics": "[Verse 1]\n...",
  "genre": "",
  "bpm": 122,
  "keyscale": "E Minor",
  "timesignature": "4",
  "duration": 197,
  "is_instrumental": false,
  "custom_tag": "acdc_bon",
  "prompt_override": "caption"
}
```

] }

#### Rules that actually bite on Path B (all in acestep/training_v2/):

```json
The top level may also be a bare JSON array of samples (preprocess_discovery.py:42). The dataset-level "metadata" block is only read when the top level is an object, so with a bare array you get tag_position=prepend, genre_ratio=0 and no default trigger (:157-175).

audio_path is used first, filename only as fallback (:50). Relative paths resolve against the directory holding the dataset JSON, not your shell's working directory (:48-55). A sample whose file cannot be found is skipped with a warning (:56-59), and if no file resolves at all the run falls back to scanning --audio-dir (:66-70).

Per-sample metadata is indexed by filename, by basename when filename contains a path, and by the basename of audio_path (:111-122); the encode-time lookup uses only the audio file's own name (preprocess.py:260). So "filename" must be either absent or exactly the real file name — a path there is tolerated, a wrong name is not.

Any audio file that is not matched receives defaults that look like real data: caption = file stem with "_" and "-" turned into spaces, lyrics = "[Instrumental]", genre = "", duration = 0, is_instrumental = true (:127-140). An unmatched dataset is the most common way to produce an adapter that trains fine and learns nothing from your captions.

prompt_override is a STRING, "caption" or "genre" — not a boolean (preprocess_prompt.py:30, 43-52). Any other value, including true/false, is ignored and the caption is used.

custom_tag placement follows tag_position: prepend yields "{tag}, {caption}", append yields "{caption}, {tag}", replace discards the caption (preprocess_prompt.py:54-61). A dataset-level custom_tag only fills samples that have no custom_tag of their own (preprocess.py:111-116).

duration must be filled in. It is written into the metadata string verbatim, and the default when the key is missing is 0 (preprocess_prompt.py:66-73), which trains the sample on the literal text "- duration: 0 seconds". Path A measures duration from the audio instead, so the two paths disagree on duration unless you supply it.

genre is only used for the samples picked by genre_ratio, or for a sample whose prompt_override is "genre" (preprocess_prompt.py:47-52). A blank genre combined with a non-zero genre_ratio trains those samples on an empty caption.

language is not read at all on Path B, and on Path A it only sets sample.language (plus the instrumental flag). In neither path does it reach the encoded prompt — it is a generation-time control. Do not rely on a "language" key to change training behaviour.
```

Interop warning — the dataset toolkit's training export does not currently match this contract. acestep-dataset-toolkit (modules/dataset_schema.py:92-96 and 129-145, modules/exporters.py:131-137) writes each sample's path as file_name and its flag as instrumental, and emits an empty top-level "metadata" block. ACE-Step reads neither name — it looks for audio_path/filename (acestep/training_v2/preprocess_discovery.py:50) and is_instrumental (acestep/training_v2/preprocess.py:300), and it takes custom_tag, tag_position and genre_ratio from "metadata". An export handed over as-is therefore resolves no audio files from the JSON (with --audio-dir present it falls back to a directory scan instead), and whatever does load is trained on default captions, default lyrics and "- duration: 0 seconds". Until the exporter is updated, remap the export before training, or the annotation work in this guide never reaches the adapter:

```text
file_name -> filename, plus audio_path if the loader should use a path
instrumental -> is_instrumental
"metadata": {} -> {"custom_tag": "<trigger>", "tag_position": "prepend", "genre_ratio": 0}
every sample: add a measured "duration" value
```

#### Silent Defaults: What a Missing Key Becomes

No field is validated and none produces a warning. An absent value is replaced by a
plausible-looking one before the prompt is built:

| Situation | What the sample trains on | Source |
|---|---|---|
| Audio file has **no metadata entry at all** | caption = file stem with `_` and `-` turned into spaces; lyrics = `[Instrumental]`; bpm/keyscale/timesignature = `N/A`; duration = 0 | `acestep/training_v2/preprocess_discovery.py:127-140` |
| Entry exists but has no `caption` key | caption = the **raw file stem**, underscores intact — not `N/A`, not blank | `acestep/training_v2/preprocess.py:261` |
| Entry exists but has no `lyrics` key | lyrics = `[Instrumental]` | `acestep/training_v2/preprocess.py:262` |
| Entry exists but has no `duration` key | `- duration: 0 seconds` in the prompt | `preprocess_prompt.py:66-73` |

The two fallbacks are different in kind and both are silent: the discovery-level default
humanises the file stem, the encode-level default does not. A track whose `lyrics` key
was misspelled is trained as an instrumental; a track whose `caption` key was misspelled
is trained on its filename. Neither failure looks like a dataset error from the training
log — it looks like the adapter ignoring your annotation. That is why §7's pre-flight
list asks you to compare the loaded-sample count against your file count rather than
trust the run's exit code.

#### The Metadata String Handed to the Text Encoder (verified)

Whichever path you use, the four metadata values end up serialized in exactly this shape and order, with the literal string N/A standing in for an empty value:

- bpm: 122
- timesignature: 4
- keyscale: E Minor
- duration: 197 seconds

(acestep/training/dataset_builder_modules/preprocess_utils.py:16-23; acestep/training_v2/preprocess_prompt.py:63-73.)

#### The Literal String the Text Encoder Receives

The caption is never encoded on its own. Both preprocessing paths wrap it in this exact
template (`acestep/constants.py:160-168`, filled by
`acestep/training_v2/preprocess_prompt.py:16-74`), and the instruction line is a fixed
string, not something you control (`acestep/constants.py:119`):

```text
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
<trigger>, <your caption>

# Metas
- bpm: 122
- timesignature: 4
- keyscale: E Minor
- duration: 197 seconds<|endoftext|>
```

Four things follow, and each of them is a decision you were probably making by feel:

1. **The caption is one field of a generated document.** `bpm: 122` hand-written inside a
   caption is the same information the `# Metas` block already emits two lines later, in
   the model's own format. It costs tokens twice and teaches the same fact twice; if the
   two disagree, the disagreement is what gets trained. Put tempo, key, time signature
   and duration in the metadata fields and leave them out of the caption text.
2. **The trigger tag is mechanically prepended, not manually typed.** With
   `tag_position: "prepend"` the wrapper emits `"{tag}, {caption}"`
   (`preprocess_prompt.py:54-61`), so a caption that also starts with the tag trains
   `tag, tag, …`. The tag belongs in the dataset's `custom_tag`/`metadata`, once.
3. **There is a hard token ceiling on the whole string.** It is tokenised with
   `max_length=256, truncation=True`
   (`acestep/training/dataset_builder_modules/preprocess_text.py:15-24`), which cuts the
   tail — the last metadata lines and the terminator — not the caption.
4. **The ceiling is measurable, so do not guess at it.** Counted with the shipped
   `tokenizer.json` (the `tokenizers` library, BPE; allow a token or two for encoder
   specials), the empty-caption skeleton is **57 tokens**, of which the four-line Metas
   block at ordinary values is **34**. That leaves roughly **199 tokens** for the
   caption — on the order of 130–160 words of descriptor-dense text, and less if the
   keyscale string is long. That is headroom, not a target: the schema caps the caption at
   70 words so that every token in it pays rent, and a caption far below that is not
   "safe" — it is simply not
   using the budget; a caption above it loses metadata, silently, before training ever
   starts.

Truncation happens at preprocessing and is not logged per sample (`truncation=True` is
the only signal), so an over-long caption is indistinguishable from a short one in the
training log. The check has to be made on the text.

#### What that means for annotation:

> There are exactly four fields, so extra metadata keys you invent never reach the model. The values are copy-pasted verbatim into the prompt: no key spelling is normalized, so "E Minor" and "E minor" become different tokens. Pick one casing for the whole dataset.
>
> A blank value is not dropped; it becomes the tokens for N/A. If you can hear the tempo but not the key, a reasonable approximation teaches more than N/A in every sample.
>
> duration is the one field you cannot leave out on the JSON path: missing means "0 seconds" (see above), a mismatch with the real audio, and the model is told the wrong length for that track.

#### Dataset-Level Prompt Controls: tag_position, genre_ratio, custom_tag

tag_position and custom_tag are covered above. The third one is the most often misused:

> genre_ratio (0-100) is the percentage of samples that are prompted with the genre field instead of the caption. The selection is a deterministic shuffle seeded with 42, so the same dataset always selects the same samples (acestep/training_v2/preprocess_discovery.py:181-201; the older builder uses the same seed in preprocess_utils.py:7-13).
>
> genre_ratio=0 (the default) means the genre field is never used, at which point a genre field in your JSON is harmless but also pointless.
>
> If you do set it above 0, every selected sample needs a real genre string, or those samples train with an empty prompt. It is also worth remembering that the samples chosen are arbitrary from the model's point of view — a 20% genre_ratio means 20% of your dataset is annotated inconsistently with the other 80%.

SFT note (corrected): bpm, keyscale and timesignature are not CFG guidance inputs. They reach the model purely as the "# Metas" text block above, inside the same 256-token prompt as the caption (acestep/constants.py:160-168), so they are prompt conditioning and nothing else. --cfg-ratio is unrelated to them: it is the probability that the caption condition is dropped during training (default 0.15, acestep/training_v2/configs.py:142, applied in fixed_lora_module.py:313-317). Accuracy still matters — an inaccurate value is taught to the adapter as fact — but no guidance mechanism reads it. Common metadata errors to correct: a Major → A Major, a mAJOR → A Major, E major → E Major.

#### Model Variant Selection for Preprocessing

Use --model-variant sft for preprocessing. The preprocessing must use the same model variant that will be used for training, because the text encoder, the VAE and the silence latent are variant-specific (the variant selects the checkpoint subfolder the silence latent is loaded from).

Audio constraints the preprocessor imposes, which are worth knowing before you master sources (acestep/training/dataset_builder_modules/preprocess_audio.py:4-21, called from acestep/training_v2/preprocess.py:243-244):

> Everything is resampled to 48 kHz (TARGET_SR = 48000, acestep/training_v2/preprocess_vae.py:18). Sample rate is handled for you; it is not a dataset-creation decision.
>
> Mono is duplicated to two channels, and audio with more than two channels is cut down to the first two (:12-15). So mono is still a real production property worth annotating (section 3.3), but it never reaches the model as a mono layout.
>
> Audio longer than --max-duration is truncated from the end, keeping the head (default 240.0 s, acestep/training_v2/cli/args.py:318-322). A six-minute track trains on its first four minutes only. Trim deliberately — keep the musically complete portion — or raise the flag and pay the VRAM cost.

The relevant preprocessing flags are --preprocess, --audio-dir, --dataset-json, --tensor-output and --max-duration (acestep/training_v2/cli/args.py:318-322). Preprocessing tokenizes every caption and lyric into frozen tensors: the finished .pt file holds target_latents, attention_mask, encoder_hidden_states, encoder_attention_mask, context_latents and the metadata copy, and no training step re-reads your JSON or your .txt files afterwards (acestep/training_v2/preprocess.py:435-445). Practical consequence: fix metadata errors in the JSON and re-run preprocessing. Editing the JSON alone changes nothing about an already-preprocessed dataset.

**`context_latents` is not adapter-agnostic.** The `.pt` contract above includes `context_latents`, the reference latents the DiT is conditioned on, and how they are built depends on how the preprocessing ran. In the legacy dataset builder the source is the track's own latents in `lokr` mode and silence otherwise — `context_src = target_latents if mode == "lokr" else None` (`acestep/training/dataset_builder_modules/preprocess.py:252`, passed at `:258`; `mode` is the `preprocess_mode` argument, normalised from `"lora"`/`"lokr"` at `:75-77`), and `None` resolves to the silence latent (`acestep/training/dataset_builder_modules/preprocess_context.py:12-13`). In v2 the context is always silence and the mode is never consulted (`acestep/training_v2/preprocess.py:418-423`). So a tensor set preprocessed as `lokr` and one preprocessed as `lora` are **not interchangeable** — they carry different `context_latents` — whereas anything v2 preprocesses is genuinely mode-independent. Because `preprocess_mode` is fixed at preprocessing time (`acestep/ui/gradio/interfaces/training_dataset_tab_save_preprocess.py:64`), the adapter you train afterwards cannot change it: preprocess for the adapter you intend to train.

This is where the bundled upstream note needs a caveat. `docs/sidestep/Dataset Preparation.md:287` states "**Preprocessing is adapter-agnostic:** The same preprocessed tensors work for both LoRA and LoKR training. You only need to preprocess once." That is true of v2, which always writes silence, but not of the legacy builder, where `preprocess_mode` selects a different context tensor. Preprocess once only if every adapter you will train comes from the same preprocessing path and the same `preprocess_mode`.

| Variant | Command Flag | When to Use |
|---|---|---|
| SFT | --model-variant sft | Your use case — LoKR training |
| Base | --model-variant base | Special tasks (lego/complete/extract) |
| Turbo | --model-variant turbo | Fast generation only — not for LoKR |

> **The default is Turbo, not SFT.** `model_variant` defaults to `"turbo"` (`acestep/training_v2/configs.py:159`), so omitting the flag runs preprocessing *and* training on Turbo with no error and no warning — the variant-match rule is then satisfied consistently wrong, which is worse than a mismatch because nothing surfaces. Pass `--model-variant sft` explicitly. Full consequences in the Creation Guide §8.

#### Verification Before Training (the checks that catch silent data loss)

1. Every audio file resolves. On Path B the loader needs audio_path or a filename equal to the real file name; unmatched files are not errors, they are defaults (preprocess_discovery.py:50, 127-140). Count the "[Side-Step] Loaded metadata for N samples" line against your file count.

2. duration is populated for every sample. A missing duration becomes "- duration: 0 seconds" in the prompt (preprocess_prompt.py:66-73).

3. Captions are not empty. An empty caption with prompt_override absent trains the sample on the wrapper text alone.

4. Lyric files are inside 512 tokens. Longer lyrics are silently truncated at preprocessing (preprocess_lyrics.py:6-12).

5. Tracks are within --max-duration, or were trimmed on purpose. The tail beyond the cap never enters training (preprocess_audio.py:17-19).

6. Captions fit the prompt budget. The caption is encoded inside a 256-token template whose empty-caption skeleton already costs 57 tokens, so roughly 199 remain before the tail — the Metas block — is silently truncated (preprocess_text.py:15-24; measurements in the section above). Check the longest caption in the dataset against that number, not against a feeling.

#### Upstream source documents (what each is for)

The upstream repository's own documents cover different slices of this ground truth. This guide is written to agree with them, and the one place it deliberately does not is called out above.

- `docs/en/Large_Scale_SFT_Training_Guide.md` — the official converter that turns a Muse-format JSONL corpus into an ACE-Step dataset (`convert_muse_jsonl`, `:118`, shipped as `convert_muse_to_acestep.py` and invoked at `:239`). Read it for the canonical "build a dataset from scratch" script and for the JSONL field names ACE-Step itself emits.
- `docs/sidestep/Dataset Preparation.md` — the field-by-field dataset JSON schema and the semantics of `custom_tag`, `tag_position` and `genre_ratio`. Its "Preprocessing is adapter-agnostic" note (`:287`) is the claim qualified in the `context_latents` paragraph above: accurate for v2, not for the legacy dataset builder.

## 8. Training Facts That Change How You Annotate (verified against source)

Every item below was read out of the ACE-Step training code rather than inferred, and
each is here because it constrains *annotation*: it decides what the adapter can and
cannot learn from the words you write. Section 7 tells you what to write; this section
tells you what happens to it afterwards.

This section used to carry the full trainer mechanics. Those have moved to the
companion guide, [`LoRA_LoKR_Creation_Guide.md`](LoRA_LoKR_Creation_Guide.md), which
covers preprocessing, target modules, timesteps, loss, optimiser, checkpoints and
resume. What remains here is only the subset with a consequence for dataset creation.
A relocation map is at the end of this section.

### 8.1 The conditioning is frozen at preprocessing — later edits are worth exactly nothing

Preprocessing encodes the caption and the lyric once and stores both as tensors in the `.pt` file (`acestep/training_v2/preprocess.py:231-245, 265, 435-445`). No training step ever re-reads your `.caption.txt`, your `.lyrics.txt` or your dataset JSON.

Consequence: a caption or lyric edit made *after* preprocessing has **zero** effect on the adapter until the preprocessor is re-run. Not a reduced effect — zero. This is also what makes the Omission Principle in section 1 a mechanism rather than a preference: because conditioning is a fixed input the adapter cannot refresh, a descriptor constant across every sample is a constant it may safely ignore, while the trigger tag is the one constant it is rewarded for retaining. Practically: settle the text, then preprocess, then train. "Tweak the caption and resume training" silently trains on the stale tensor.

*Mechanism, and the failed-preprocessing mode that leaves unusable `.tmp.pt` files behind: Creation Guide §2.*

### 8.2 `--cfg-ratio` removes the caption and the lyrics together

Caption and lyric are packed into a single `encoder_hidden_states` tensor, and the 0.15 default dropout (`acestep/training_v2/configs.py:142`) replaces the *whole* fused condition with the model's null embedding (`acestep/training_v2/fixed_lora_module.py:312-318`). There is no caption-only dropout and no lyric-only dropout anywhere in the training path.

Two dataset decisions follow:

1. **Lyrics are not a protected channel.** Information placed only in the lyric brackets disappears on those steps exactly as if it had been placed nowhere. Do not treat the lyric stream as a safe home for something the caption cannot say.
2. **You cannot buy robustness by splitting information across the two streams.** Both streams share one dropout switch, so state a feature once, in the stream it actually belongs to.

*Mechanism, the Turbo-specific version of it, and the fact that legacy runs are pinned to 0.15: Creation Guide §7.*

### 8.3 Only the attention projections are trainable, so annotation phrasing is a data-side lever

Adapter targets are the four attention projections `q_proj`/`k_proj`/`v_proj`/`o_proj` by default (`acestep/training/configs.py:25-27` for LoRA, `:52-54` for LoKR), and injection freezes every parameter outside the adapter itself (`acestep/training/lora_injection.py:216-218`). The projection list can be widened (`--target-modules`, `acestep/training_v2/cli/args.py:289`), but nothing adds an adapter to the encoders.

Consequence: the text encoder, the lyric encoder, the `text_projector` and the VAE are frozen. An annotation convention whose vocabulary lives only in the text encoder cannot be taught, because the module that would learn it is not trainable. Phrasing consistency across your dataset therefore stays a **data-side** lever, not something training will repair.

*Target-module overrides, the LoKR filter that fails open, adapter scale and DoRA defaults: Creation Guide §3 and §4.*

### 8.4 Durations must stay comparable if you raise the batch size

The flow-matching loss is an unmasked mean over the full padded tensor (`acestep/training_v2/fixed_lora_module.py:354`), so padding enters the loss as if it were signal. At the default `batch_size = 1` this is harmless; raise the batch and the wider the duration spread inside a batch, the more capacity is spent predicting noise.

Consequence for dataset curation: a dataset of tracks with wildly different lengths behaves differently at batch size 4 than at batch size 1, and the difference is not a tuning detail — it is dilution.

### 8.5 Where the rest of the training mechanics went

| Topic | Where it lives now |
|---|---|
| Preprocessing contract, the `.tmp.pt` failure mode, `--max-duration` truncation | Creation Guide §2 |
| Trainable modules, `--target-modules`, the LoKR filter that fails open, adapter scale, `attention_type` | Creation Guide §3 |
| LoRA vs LoKR, DoRA defaults per surface | Creation Guide §4 |
| The flow-matching timestep convention, the two sampling regimes, `data_proportion` | Creation Guide §5 |
| The unmasked loss | Creation Guide §6 |
| CFG dropout mechanism, Turbo train-vs-inference CFG, the `# Metas` correction | Creation Guide §7 |
| `--model-variant` defaulting to Turbo | Creation Guide §8 |
| Optimiser defaults, the warmup clamp, the two Prodigy overrides | Creation Guide §9 |
| Checkpoint layout, `--resume-from`, LoKR vs LoRA saving, `offload_encoder` | Creation Guide §10 |
| Which subcommand you are running (`fixed`, `vanilla`, `estimate`) and what each one changes | Creation Guide §1.1 |
| Rank / alpha / dropout per surface (CLI, Gradio, presets) | Creation Guide §3.3 |
| `--bias` being re-frozen by the injection filter; `estimate` not feeding back into training | Creation Guide §3.4, §3.5 |
| Flags that are parsed, stored and never read (`--sample-every-n-epochs`, `--shift`, `--num-inference-steps`) | Creation Guide §11 |
| What the generation side does with the artefact (LoKR metadata rebuild, quantized refusal, 0–1 scale) | Creation Guide §12 |

Two corrections landed with the move, both in the Creation Guide: `attention_type` is
**not** inert (it rewrites the target patterns to `self_attn.*` / `cross_attn.*`), and
the four-projection list is a default rather than a hard limit (`--target-modules`
replaces it).

Section 7's checklist remains the practical gate; §8.1–§8.4 above explain why four of
its items are load-bearing.

## 9. Master Checklist for LoKR Dataset Annotation

| Step | Action |
|---|---|
| 1 | Choose a specific trigger tag (e.g., acdc_bon) and use it in every caption |
| 2 | Identify constant features across the dataset (genre, vocal character, gear, production) |
| 2a | Check each candidate against the prevalence bands in section 1.1 before deciding to omit or keep it |
| 3 | Remove all constants from captions and lyric brackets |
| 4 | Rewrite captions to describe only song-specific variables |
| 5 | Rewrite lyric brackets to describe only song-specific variation |
| 6 | Pick ONE apostrophe convention and apply it to every lyric file (Tier 1 apostrophe removal is the documented one; the Tier 2 respellings and Tier 3 elongations are optional conventions, see 4.6) |
| 7 | Remove all punctuation from lyrics (periods, quotes, question marks, exclamation marks) |
| 8 | Remove all punctuation at the end of a line, except ) |
| 9 | Inline commas may be retained only when they intentionally represent a brief vocal pause. |
| 10 | Capitalize only sections that are genuinely shouted/belted |
| 11 | Keep lines between 6–10 syllables for normal sung lines |
| 12 | Use parentheses only for clear backing vocals in the audio |
| 13 | Verify every lyric line against the audio for accuracy |
| 14 | Normalize all metadata (keyscale casing, BPM, time signature) |
| 15 | Ensure all files have matching basenames (.flac, .lyrics.txt, .caption.txt, .json) |
| 16 | Run the dataset builder with --model-variant sft |
| 17 | Confirm that every sample resolved (each file name matched), that duration is populated for every sample, and that no lyric file exceeds 512 tokens |
| 18 | Check the longest caption against the ~199-token caption budget (256-token prompt minus the 57-token template) — over-long captions lose the Metas block silently |

## 10. The One-Sentence Rule

A descriptor that is present on essentially every song of your dataset, in the captions or in the lyric brackets, belongs to the trigger tag and must not be written again in either stream. If it is present on only part of the dataset, name it on exactly the songs where it occurs. Section 1.1 defines where "essentially every" begins.

## 11. Comprehensive Prompt for Gemini 3.8 Flash

Copy the entire prompt below and paste it into Gemini. Replace the bracketed [TRIGGER_TAG] and [ARTIST_STYLE] with your specific values.

```text
You are an expert music annotation assistant for ACE-Step 1.5 XL LoKR dataset creation. Your task is to generate captions and lyrics for a training dataset.

CRITICAL CONTEXT: This is for LoKR adapter training, NOT music creation prompting. The goal is to teach the adapter what the audio ACTUALLY CONTAINS, not what you want it to generate.

THE OMISSION PRINCIPLE (most important rule):
A feature present on essentially every song in the dataset (about 100%, and certainly above roughly 90%) must NOT appear in ANY caption or lyric descriptor. The trigger tag is the only name for that feature. A feature present on only part of the dataset must be named on every song where it actually occurs - never on the majority side only. Describe ONLY what differs between songs.

DATASET PARAMETERS:
- Trigger tag: [TRIGGER_TAG]
- Artist/style: [ARTIST_STYLE]
- Model variant: SFT
- Language: English

YOUR TASK:
For each audio file provided, generate:
1. A CAPTION (35-70 words; front-load 5-12 keywords, hard maximum 15; then 8-16 comma-separated acoustic facts, never prose sentences) that describes only song-specific variables. The caption goes into a fixed prompt template whose total budget is 256 tokens, and the metadata lines sit after the caption, so an over-long caption silently cuts off the bpm/keyscale/timesignature/duration lines. Measured with the shipped Qwen3-Embedding-0.6B tokenizer: the same track costs 75 tokens as 47 words of narrative prose and 59 tokens as 35 words of facts, so prose is what spends the metadata block's room.
2. LYRICS with structural markers and descriptors that describe only song-specific variation.

CAPTION FORMAT:
[TRIGGER_TAG], [song-specific tempo/feel], [song-specific mood], [song-specific structure], [song-specific solo character], [song-specific dynamic]. [8-16 comma-separated acoustic facts]
Do NOT include genre, vocal character, gear, or production terms that are constant across the dataset. (This omission rule is for the artist-scoped LoRA variant. A caption that goes through `modules/caption_spec.py` must still carry a vocal descriptor — rule 3 is a hard validation check.)
Do NOT include BPM, key, or time signature in the caption — these belong in metadata.

LYRICS FORMAT:
Every section must begin with a structural marker in square brackets: [Intro], [Verse 1], [Chorus], [Bridge], [Guitar Solo], [Instrumental], [Outro], [Breakdown], [Silence], [Build-Up], [Break], [Fade Out].
Maximum 3 descriptors per bracket, attached with a dash: [Verse 1 - whispered, sparse].
Do NOT use constant descriptors (e.g., raspy vocal, explosive, high energy) in brackets. Use only song-specific variation.

LYRIC TEXT RULES:
1. Remove ALL punctuation: periods, question marks, exclamation marks, quotation marks, backslashes.
2. UPPERCASE only for genuine shouted/belted lines. Lowercase for normal sung lines.
3. Keep lines between 6 and 10 syllables for optimal beat alignment.
4. Use parentheses ( ) only for clear backing vocals or call-and-response in the audio.
5. Apostrophe handling:
   - Tier 1 (safe to drop): dont, didnt, isnt, arent, ive, im, thats, its, lets
   - Tier 2 (must respell phonetically): wont→wohnt, cant→kant, well→weel, hell→heel, were→weer, ill→eyel, id→eyed, cause→kuz
   - Tier 3 (vowel elongation): ba-aa-aa-abe, baaaaby
6. Dropped -ing endings: goin, lyin, waitin, tryin, nothin, somethin
7. Words that stay as dictionary words: gonna, wanna, gotta, gimme, em, til, round, mm, ah-uh-huh, ooh, yeah
8. Separate sections with a blank line.
9. Remove all puntuation at the end of a line except )
OUTPUT FORMAT (JSON only):
{
  "caption": "<trigger_tag>, <song-specific variables>. <8-16 comma-separated acoustic facts>.",
  "lyrics": "[Intro - ...]\n\n[Verse 1 - ...]\nLyrics...\n\n[Chorus - ...]\nLyrics...",
  "bpm": <integer>,
  "keyscale": "<key>",
  "timesignature": "<time signature>",
  "language": "en"
}

Do NOT include any text outside this JSON structure.
```

Note on duration: do not guess or estimate it. duration is required for training (a missing value trains the sample with the literal metadata text "- duration: 0 seconds"), and it must be measured from the audio file itself, so fill it in programmatically after annotation rather than asking the LLM for it. The "language" field is kept for your records: it does not reach the model during training.

This guide and prompt are a working reference for ACE-Step 1.5 XL LoKR dataset annotation, not an official specification. Its claims do not all have the same status, and the difference matters when deciding what to trust:

| Claim class | Status | Section |
|---|---|---|
| Prompt template, token caps (256 assembled prompt / 512 training lyric / 2048 inference lyric), file names, JSON keys, metadata string, mono-to-stereo handling, 48 kHz resampling, max-duration truncation, genre_ratio | Verified against ACE-Step source, with file and line references | 3.2, 4, 4.9, 7 |
| `context_latents` depend on `preprocess_mode` in the legacy dataset builder (silence in v2), so legacy tensors are not interchangeable between LoRA and LoKR | Verified against ACE-Step source, with file and line references | 7 |
| Caption and lyric as two independent conditioning streams | Verified against ACE-Step source | 1 |
| Parentheses are processed as background vocals or harmonies; vowel elongation works but is unstable | Verified in official generation docs and the songwriting skill | 4.6, 4.8 |
| Omission principle and the prevalence bands | This guide's recommendation, derived from how adapter loss is distributed. ACE-Step enforces no such rule, and the exact thresholds are empirical | 1, 1.1, 6 |
| Macro/Micro bracket and parenthesis placement convention | This guide's convention. Placement semantics are not documented upstream | 4.8 |
| Comma usage in lyrics (inline comma as a breath mark; trailing comma as harmless but redundant) | This guide's convention. Upstream documents nothing about comma semantics in lyrics | 4.5 |
| Tier 2 phonetic respellings (wohnt, kant, weel, eyel, kuz, ...) | Unverified. None appear anywhere in ACE-Step documentation, and none have been measured against plain apostrophe removal | 4.6 |
| An alignment model; a lyric normalizer that strips punctuation, emoji and case; the word-swallowing failure mode | Not supported. No alignment model exists in the repository, and the training lyric path applies no normalization of its own | 4.5, 4.6 |
| Melisma or vowel elongation as a LoRA/LoKR training requirement | Not supported. No ACE-Step training document mentions it; melisma occurs only as a genre descriptor in the caption vocabulary (acestep/genres_vocab.txt) | 4.6 |
| Training-time constraints (conditioning frozen at preprocessing, CFG dropout removing the fused condition, the trainable projections, the unmasked loss) | Verified against ACE-Step source, with file and line references | 8 |
| The rest of the trainer (adapter scale, timesteps, optimiser, checkpoints, DoRA defaults, dead flags, inference-side behaviour) | Verified against ACE-Step source in the companion Creation Guide | Creation Guide §1–§12 |
| "AC/DC songs are mono" | Withdrawn. The 1970s albums are stereo; mono must be established per file | 3.3 |

Apply the verified mechanics as written. Treat the recommendation and convention rows as defaults to test on your own material, and do not spend annotation effort defending the unverified rows.

## 12. Moved: Artist-Scoped Caption & Tag Generation

The artist-scoped caption and bracket template specification has moved to its own
document: [`Artist_Scoped_Caption_Tag_Generation.md`](Artist_Scoped_Caption_Tag_Generation.md).

Reasons it is separate: it specifies a toolkit feature that does not exist yet, it is
scoped to a single-artist dataset rather than to annotation in general, and it depends
on the rules in this guide rather than changing them.

## Companion documents

- [`LoRA_LoKR_Creation_Guide.md`](LoRA_LoKR_Creation_Guide.md) — preprocessing,
  trainable modules, timesteps, loss, optimiser, checkpoints, resume, the flags that are
  parsed but never read (§11), and what the generation side does with the artefact
  (§12). It is the mechanism behind every training claim in this guide.
  The prompt the text encoder actually receives, and the token budget it leaves for a
  caption, are measured in §7 above.
- [`Artist_Scoped_Caption_Tag_Generation.md`](Artist_Scoped_Caption_Tag_Generation.md) —
  the artist-scoped caption and bracket template feature (specified, not implemented).
- [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) — how documentation in this repository is
  verified and cited.


---

## Correction (2026-11-18): caption shape is guidance, not a gate

The "35–70 words, 8–16 comma-separated acoustic facts, no prose sentences" budget
stated in §3.2 is the **captioner's prompt**, not an ACE-Step LoRA training
requirement, and the app no longer enforces it. The rule entered this guide on
2026-10-23 (commit `b545885`) from a token measurement; that measurement is real,
but the claim that LoRA training *requires* the dense shape is not supported
upstream:

- `docs/en/LoRA_Training_Tutorial.md` in the ACE-Step 1.5 source tree gives
  `"A high-energy J-pop track with synthesizer leads and fast tempo"` as its
  caption example — one prose line, 11 words.
- `scripts/lora_data_prepare/gemini_caption.py`, the official LoRA captioner,
  asks only for a "detailed caption and lyrics" and constrains no shape.

What IS an upstream constraint, and is kept: `SFT_GEN_PROMPT`
(`acestep/constants.py:160-168`) places the caption before a `# Metas` block
inside a 256-token prompt, so an over-long caption silently deletes the
bpm/keyscale/timesignature/duration lines. Hence a length guard still exists
(`modules/caption_quality.trim_to_caption`), and BPM/key/time-signature still
should not be restated inside the caption text.

Retired with this correction: `check_caption`, `count_facts`, the tag-count,
fact-count and vocal-descriptor checks, `modules/caption_audit.py`, the
export-time caption gate and `write_caption`'s self-review. Caption wording is
judged by a human or an LLM, because captions are too varied to be scored by a
script.

