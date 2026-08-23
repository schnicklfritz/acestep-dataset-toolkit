# ACE-Step 1.5 Master Annotation & Dataset Guide

This master specification details the annotation standard for **ACE-Step 1.5** training datasets, covering prompt conditioning, structural lyric tagging, architectural descriptors, and hybrid captioning.

---

## 1. Hybrid Caption Architecture (Global Conditioning)

The Caption establishes the **global sound portrait** across the entire duration of the audio clip.

### Structure Formula
```text
[Trigger Tag], [Primary Genre], [Subgenre/Mood], [2-3 Specific Instruments], [Vocal Style], [Production & Mix], [Era/Aesthetic]. [2-3 sentences detailing dynamic build, arrangement transitions, and energy flow].
```

### Formatting Standards
1. **Front-Loaded Tags**: High-priority conditioning keywords appear at the very start (5–12 keywords, maximum 15) separated by commas.
2. **Concrete Instruments**: Use exact, specific instruments and gear (e.g., `Vox Continental organ`, `Gibson SG with P-90 pickups`, `Fender Rhodes bass`, `808 sub bass`, `gated reverb drums`).
3. **Mandatory Vocal Descriptor**: Always declare vocal presence and character:
   - `male vocal`, `female vocal`, `male baritone`, `raspy vocals`, `powerful belting`, `clean vocals`, `screamed vocals`, `whispered vocals`, `choir`, or `instrumental` / `no vocals`.
4. **Flow Narrative**: Follow the tag list with 2–3 concise sentences describing the song's energy progression (e.g., *Opens with a sparse intro before building into a driven verse with layered harmonies, reaching a peak belted chorus and dynamic outro*).
5. **No Parameter Contradictions**: Never place BPM, Key, or Time Signature inside the caption—those are handled as dedicated numeric/categorical metadata fields.

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
1. "caption": A hybrid caption (comma-separated tags followed by 2-3 sentences of dynamic flow).
2. "lyrics": Full lyrics with Capitalized structural markers ([Verse], [Chorus]) and max 3 architectural descriptors.

Output Format:
{
  "id": "<song_id>",
  "caption": "<tag list>. <structural flow narrative>",
  "lyrics": "[Intro]\n\n[Verse 1 - delivery]\nLyrics...\n\n[Chorus - energy]\n..."
}
```
