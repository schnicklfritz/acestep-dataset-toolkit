# Hank Williams Sr. — Arrangement Notes & Curation Decisions

Source: Gemini 3.8 workflow-emulation pass, captured 2026-09-06.
Reference: /home/fritz/Music/hank_sr/originals_backup1/stem-label.txt

---

## 1. Arrangement template (why captioning is cheap for this material)

Hank Williams Sr. sessions (produced by Fred Rose) were deliberately formulaic
so jukebox/country-radio listeners instantly recognized the groove:

- **Bass (upright, slap technique — Cedric Rainwater / Hillous Butrum)**
  = CONSTANT. Root-fifth on beats 1&3, slap on 2&4. One GLOBAL caption needed:
    "1950s upright acoustic double bass, percussive slap bass, root-fifth
     country two-beat, driving honky-tonk timekeeping, raw tube mono"

- **Acoustic rhythm guitar (boom-chicka — Hank / Sammy Pruett / Bob McNett)**
  = MINIMAL VARIATION. One GLOBAL caption:
    "acoustic rhythm guitar, boom-chick strum, classic country backing,
     un-amplified 1950s flat-top"

- **Steel guitar (Don Helms, non-pedal Gibson Console Grande E6)**
  = 3 STATES (state-based tagging, not per-section):
    1. idle/soft:  "backing steel, subtle swells, quiet chordal background"
    2. vocal fill: "lap steel fills, E6 piercing accents, weeping tone"
    3. break/solo: "lead steel solo, melodic break, vibrato glissando"

- **Fiddle (Jerry Rivers)** = alternates with steel (intro vs break).

=> Only VOCALS + lead instruments (steel/fiddle) need dynamic per-section
   captioning. Bass + rhythm guitar can reuse a single global caption.

---

## 2. Alignment method

Use LYRIC-GUIDED FORCED ALIGNMENT (align lyric timestamps to audio) for
segmenting vocals. WhisperX already produces word-aligned segments
(see modules/lyrics.py → transcribe()).

---

## 3. Curation decisions (pending user action)

- [ ] DROP from dataset: kaw-liga, there's-a-tear-in-my-beer, ramblin-man
- [ ] VERIFY time signature = 3/4 (waltz): im-so-lonesome-i-could-cry,
      the-alabama-waltz
- [ ] REPLACE source (currently bad) for: long-gone-lonesome-blues
      (also trim the end / fade out)
- [ ] ENHANCE (find better/live source) for: wild-side-of-life
