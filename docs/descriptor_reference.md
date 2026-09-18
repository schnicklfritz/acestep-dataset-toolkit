# Master Descriptor & Tag Reference (English Only)

Datasets: **Hank Williams Sr. | The Doors | Black Sabbath | AC/DC (Bon Scott)**

This is the authoritative annotation vocabulary for this project. Where it
disagrees with the official ACE-Step songwriting skill, **this file wins** — it
is tuned for *dataset annotation*, the skill is written for *generation
prompting*. Where this file is silent, the skill fills gaps (see `## Sources`).

---

## Section 1: File Metadata Tags (LRC & ID3)

| Tag | Meaning |
| :--- | :--- |
| `[ti:]` | Title |
| `[ar:]` | Artist |
| `[al:]` | Album |
| `[by:]` | Creator of the LRC file |
| `[au:]` | Author of the lyrics |
| `[offset:]` | Time offset in ms (e.g. `[offset:+500]`) |
| `[length:]` | Song length (mm:ss) |
| `[re:]` | Tool/editor that created the file |
| `[ve:]` | LRC format version |
| `[#:]` | Comment |

Technical: ID3v2 frames `USLT` (unsynchronized) / `SYLT` (synchronized).
Vorbis comment for FLAC: `UNSYNCEDLYRICS`. Language codes `lyrics:lang:desc`.
Locales `en`, `en-US`, `en-GB`.

---

## Section 2: Structural Markers

### Core structure
`[Intro]` `[Verse]` `[Verse 1]` `[Verse 2]` `[Pre-Chorus]` `[Chorus]`
`[Post-Chorus]` `[Bridge]` `[Outro]` `[Hook]` `[Refrain]` `[Rap Verse]`
`[Solo]` `[Break]` `[Interlude]` `[Final Chorus]`

### Dynamic / EDM
`[Build]` `[Build-Up]` `[Drop]` `[Breakdown]`

### Instrumental
`[Instrumental]` `[Inst]` `[Guitar Solo]` `[Piano Solo]` `[Piano Interlude]`
`[Synth Solo]` `[Organ Solo]` `[Drum Break]` `[Drum Fill]` `[Bass Solo]`

### Special
`[Fade Out]` `[Fade In]` `[Silence]` `[Spoken]` `[End]` `[Energy: High]`

---

## Section 3: Architectural Descriptors

Format: `[Marker - descriptor 1, descriptor 2, descriptor 3]`
**Hard rule: max 3 descriptors per bracket.** More risks the model singing tag
names as lyrics.

### Vocal style
whispered, belted, falsetto, spoken word, layered, harmonized,
call-and-response, ad-lib, raspy vocal, powerful belting, breathy, clean vocals,
screamed vocals, shouted vocals, harmony layer, falsetto verse, growl, vibrato,
yodel, blue yodel, nasal, strained, wailing, crooning, sneering, snarling,
chanting, moaning, humming, spoken/shouted, operatic, theatrical,
conversational, narrative, declarative, mournful, pleading, taunting,
celebratory

### Energy
quiet, sparse, building, rising, peak, intense, stripped back, full, low energy,
high energy, explosive, driving, relentless, swaggering, menacing, hypnotic,
frenzied, furious, triumphant, somber, eerie

### Instrument focus
synth lead, guitar solo, organ solo, drum fill, bass groove, strings swell,
brass stabs, 808 bass, acoustic rhythm, riff-driven, power chords,
palm-muted riffs, wah-wah lead, slide guitar, fingerpicked, flatpicked,
strummed, arpeggiated, double-stop, bend, vibrato bar, tremolo picking,
galloping rhythm, blast beat, half-time, shuffle, straight, syncopated,
polyrhythmic

### Dynamics & production
fade in, fade out, abrupt cut, swell, drop out, reverb-heavy, dry, filtered,
tape stop, sidechain kick, analog warmth, lo-fi, hi-fi, raw production,
polished production, live feel, room reverb, plate reverb, spring reverb,
tape delay, digital delay, chorus effect, flanger, phaser, distortion,
overdrive, fuzz, compression, gated reverb, monaural, stereo, wall of sound

---

## Section 4: Country Music Descriptors

### Vocal
twangy vocals, Tennessee twang, Southern drawl, Texan twang, Appalachian nasal,
Carolina smooth, raspy male vocal, storytelling vocals, heartfelt vocals,
raw emotive vocal, conversational male vocals, powerhouse raspy male vocal,
warm female vocals, falsetto lead, close-mic male, yodel, blue yodel

### Acoustic instruments
acoustic guitar, steel guitar, pedal steel, lap steel guitar, fiddle, banjo,
mandolin, harmonica, dobro, upright bass, acoustic piano, organ, autoharp,
jug, washboard

### Electric & rhythm
electric guitar (clean), Telecaster twang, hard-hitting country drums,
brushed drums, 808 sub bass, baritone guitar, pedal steel swells,
fiddle melody, acoustic guitar strumming, acoustic guitar picking,
banjo picking, flat-picking

### Production & mix
polished Nashville production, radio-ready, radio-friendly, raw production,
live feel, DIY-rough recording, intimate acoustic production, vinyl crackle,
room reverb, warm production, nostalgic production,
polished country-radio production, monaural, era-standard production

### Mood & energy
heartfelt, nostalgic, warm, earnest, uplifting, reflective, storytelling,
honky-tonk, outlaw country, country-soul, Americana, bluegrass, country-folk,
country-pop crossover, lonesome, mournful, weary, wistful, defiant, playful,
rowdy, tender, bittersweet, haunting

### Sub-genre tag combinations
```
Classic Country Ballad:
  Classic Country, Slow Tempo, Pedal Steel, Fiddle, Male Vocals, Heartfelt, Traditional

Modern Country Pop:
  Country, Mid-Tempo, Acoustic Guitar strumming, Steel Guitar, Female Vocals,
  Uplifting, Radio-friendly Production

Upbeat Bluegrass:
  Bluegrass, Fast Tempo, Banjo Picking, Fiddle, Acoustic Guitar, Male Vocals,
  Energetic, Folksy

Outlaw Country:
  Outlaw country, Vintage production, Raspy male vocal, Electric guitar,
  Classic-rock sensibility

Folk-Country / Americana:
  Oklahoma folk-country, Raw emotive male vocal, Intimate acoustic guitar,
  Fiddle, Sparse drums

Country-Soul:
  Country-soul, Powerhouse raspy male vocal, Gospel-influenced runs,
  Vintage outlaw country
```

### Country tips
1. Regional accents matter: Tennessee twang, Texan twang, Southern drawl,
   Appalachian nasal.
2. Twang is essential — include a twang descriptor.
3. Instrument tags are reliable: pedal steel, fiddle, banjo beat genre tags alone.
4. Choose a production aesthetic: polished Nashville vs raw vs DIY-rough.
5. Avoid contradictions: no twangy vocals with auto-tune/trap drums unless intended.


---

## Section 5: Hank Williams Sr.

### Vocal & performance
sincere hillbilly vocals, high nasal strained timbre, lonesome conservative
delivery, narrow melodic range, naturally breaking half-yodel, blues-drenched
singing, sincere declarative vocals, joyful upbeat delivery, call-and-response
with backing vocals, weathered vocal, plaintive delivery, storytelling vocal

### Steel guitar (Don Helms style)
piercing high-pitched steel guitar, high whining sound, visceral mournfulness,
non-pedal steel guitar, treble-string focused, plays high on the neck,
singing almost-vocal counter-line, weeping guitar, celebratory steel fills,
shouted steel guitar interjections

### Fiddle (Jerry Rivers / Tommy Jackson style)
double-stop fiddle style, "garden seed" fiddle, short fiddle fills,
supports the harmonic movement, mournful fiddle, lively driving fiddle,
fiddle intro, peppy fiddle breaks, single-string melody,
driving melodic fiddle lead, bright declarative fiddle phrase,
Tommy Jackson-style fiddle

### Production & arrangement
sparse stripped-down arrangement, honky-tonk and country blues,
voice-steel-and-fiddle at the center, steel and fiddle trade short fills,
radio-friendly compact arrangement, acoustic guitar and bass three-beat pulse,
monaural era-standard production, simple driving chord progression,
upbeat energetic tempo, country gospel / gospel-country

### Complete examples
**Caption**
> classic honky-tonk country, sincere hillbilly vocals, high nasal strained
> timbre, piercing high-pitched non-pedal steel guitar, double-stop fiddle,
> sparse stripped-down arrangement. Sparse intro leads to a verse where voice,
> steel and fiddle are at the center, trading short fills. The arrangement stays
> compact and radio-friendly, centered on the vocal phrasing.

**Section tags**
```
[Intro - sparse, acoustic guitar]

[Verse 1 - lonesome delivery, steel and fiddle fill]
I'm so lonesome I could cry...

[Chorus - high nasal vocal, piercing steel guitar]
Did you ever see a robin weep...

[Instrumental Break - weeping steel guitar solo, double-stop fiddle fills]

[Outro - fade out, sparse arrangement]
```

---

## Section 6: The Doors

### Vocal (Jim Morrison)
deep baritone vocal, brooding delivery, poetic delivery, spoken word,
theatrical delivery, drunken slur, bluesy shout, crooning, whispered, moaning,
wailing, screaming, hypnotic repetition, lizard-king intensity, sinister,
seductive, menacing, ecstatic, desperate

### Instruments
Vox Continental organ, Fender Rhodes piano, Gibson SG guitar,
no bass guitar (organ left-hand bass), organ bass pedals,
jazz-influenced drums, brushed drums, ride cymbal, flamenco guitar,
slide guitar, wah-wah lead, feedback

### Mood & atmosphere
psychedelic, ominous, nocturnal, hypnotic, cinematic, theatrical, brooding,
mysterious, desert-like, rain-soaked, carnival-like, funereal, apocalyptic,
erotic, drug-induced

### Production & arrangement
1960s analog production, reverb-heavy, live feel, extended instrumental
sections, organ-driven, guitar-driven, dynamic shifts, tempo changes,
spoken word passages, theatrical builds

### Genre
psychedelic rock, acid rock, blues rock, art rock, garage rock, jazz rock

### Complete examples
**Caption**
> psychedelic rock, deep brooding baritone vocals, poetic spoken word passages,
> Vox Continental organ, jazz-influenced drums, 1960s analog production,
> reverb-heavy. Opens with a hypnotic organ riff before building into a
> theatrical verse with whispered and shouted dynamics, reaching an ecstatic
> peak and fading into a mysterious outro.

**Section tags**
```
[Intro - Vox Continental organ, hypnotic]

[Verse 1 - deep baritone, spoken word, sparse]
Riders on the storm...

[Chorus - brooding, building, full band]
Into this house we're born...

[Organ Solo - jazz-influenced, reverb-heavy]

[Verse 2 - whispered, menacing, building]

[Outro - fade out, rain-soaked, mysterious]
```


---

## Section 7: Black Sabbath (pre-Never Say Die)

### Vocal (Ozzy Osbourne)
nasal vocal, high-pitched wail, haunting delivery, plaintive delivery,
eerie vocal, melodic vocal, sorrowful, menacing, screaming, wailing, chanting,
moaning, haunting harmonies, double-tracked vocals, reverb-drenched vocals

### Guitar (Tony Iommi)
downtuned guitar, heavy riffing, blues-based leads, wah-wah lead,
palm-muted riffs, power chords, sustained notes, bend, vibrato, trilling,
acoustic guitar, flamenco guitar, slide guitar

### Bass & drums (Geezer Butler / Bill Ward)
distorted bass, thick bass tone, rumbling bass, jazz-influenced drums,
swinging drums, thunderous drums, dynamic drumming, tom-heavy fills,
ride cymbal, crash cymbal

### Mood & atmosphere
doom-laden, ominous, apocalyptic, sinister, occult, evil, dark, heavy,
brooding, menacing, funereal, hypnotic, psychedelic, industrial, bleak

### Production & arrangement
raw production, 1970s analog production, dry, live feel, heavy low-end,
guitar-driven, riff-based, dynamic shifts, tempo changes,
extended instrumental sections, acoustic interludes, atmospheric intros

### Genre
heavy metal, doom metal, hard rock, blues rock, proto-metal, acid rock,
progressive rock

### Complete examples
**Caption**
> heavy metal, doom-laden, nasal haunting vocals, downtuned heavy riffing,
> distorted bass, thunderous drums, raw 1970s analog production. Opens with a
> slow, ominous riff before building into a plaintive verse with wailing vocals,
> reaching a crushing peak and fading into a bleak, atmospheric outro.

**Section tags**
```
[Intro - slow, ominous, heavy riffing]

[Verse 1 - nasal vocal, plaintive, sparse]
What is this that stands before me?

[Chorus - menacing, full band, doom-laden]
Oh no, no, please God help me!

[Guitar Solo - wah-wah lead, blues-based]

[Verse 2 - wailing vocal, building, heavy]

[Outro - fade out, bleak, atmospheric]
```

---

## Section 8: AC/DC (Bon Scott years)

### Vocal (Bon Scott)
raspy vocal, gritty delivery, high-pitched snarl, sneering delivery, snarling,
bluesy shout, swaggering delivery, cheeky delivery, aggressive, raucous,
anthemic, call-and-response, shouted vocals, screamed vocals, growled vocals,
strained vocal, wailing, whooping, taunting

### Guitar (Angus & Malcolm Young)
clean-to-crunchy guitar, riff-based, power chords, open chords,
blues-based leads, vibrato, bend, pull-off, hammer-on, double-stop,
palm-muted riffs, galloping rhythm, driving rhythm guitar,
tight rhythm section, simple riffs

### Bass & drums (Cliff Williams / Phil Rudd)
simple bass lines, driving bass, tight drums, straight beat, hard-hitting
drums, snare-heavy, kick-drum driven, hi-hat work, ride cymbal, crash cymbal,
no frills

### Mood & atmosphere
energetic, aggressive, anthemic, rebellious, swaggering, party, raw, driving,
relentless, frenzied, furious, triumphant, rowdy, playful, cheeky

### Production & arrangement
raw production, 1970s analog production, dry, live feel, punchy, in-your-face,
guitar-driven, riff-based, simple verse-chorus, guitar solos, breakdowns,
call-and-response, no frills

### Genre
hard rock, blues rock, rock and roll, heavy rock, pub rock

### Complete examples
**Caption**
> hard rock, raspy sneering vocals, clean-to-crunchy guitar riffs, tight rhythm
> section, raw 1970s analog production, punchy. Opens with a driving guitar riff
> before building into an anthemic verse with snarling vocals, reaching a
> triumphant chorus and fading into a raucous outro.

**Section tags**
```
[Intro - driving guitar riff, tight drums]

[Verse 1 - raspy vocal, sneering, sparse]
Living easy, living free...

[Chorus - anthemic, full band, high energy]
Highway to hell!

[Guitar Solo - blues-based leads, vibrato]

[Verse 2 - snarling vocal, building, driving]

[Outro - fade out, raucous, relentless]
```


---

## Section 9: Master Consistency Checklist

| Area | Caption (global) | Lyrics (temporal) |
| :--- | :--- | :--- |
| Instrumentation | Vox Continental organ, Gibson SG P-90 | `[Instrumental - Vox Continental Organ solo]` |
| Vocal delivery | raspy male baritone vocal | `[Verse 1 - powerful belting]` |
| Energy & mood | rebellious mood, driving tempo | `[Bridge - shouting]`, `[Chorus - high energy]` |
| Language / type | `is_instrumental: false, language: "en"` | `[EN - Verse]` with English lyric text |

---

## Section 10: Key Rules & Best Practices

1. **Max 3 descriptors per bracket.** More risks the model singing tag names.
2. Always separate sections with a blank line (`\n\n`) for optimal parsing.
3. **UPPERCASE lyrics** signal belting, screaming, or shouted delivery.
4. **`(parentheses)`** for backing vocals, echoes, or call-and-response.
5. **Never put BPM, Key, or Time Signature** inside caption or lyric tags —
   those belong in dedicated metadata fields.
6. For instrumental sections, put the tag on its own line with no lyrics below.
7. Capitalize section marker names consistently (`[Verse]`, `[Chorus]`, `[Outro]`).
8. **Front-load conditioning keywords** in captions (5–12 keywords, max 15).
9. Use **concrete instruments and gear** (Vox Continental organ, Gibson SG with
   P-90 pickups, Fender Rhodes bass, 808 sub bass, gated reverb drums).
10. **Always declare vocal presence and character** in captions.
11. Follow the tag list with **2–3 sentences** describing energy progression.
12. For non-English songs, put capitalized language tags at the start of the
    bracket: `[EN - Chorus - anthemic]`, `[JA - Verse - whispered, sparse]`.

---

## Section 11: Quick Reference — Descriptor Count by Section Type

| Section | Recommended | Example |
| :--- | :--- | :--- |
| Intro | 1–2 | `[Intro - sparse, piano]` |
| Verse | 2–3 | `[Verse 1 - whispered, building]` |
| Pre-Chorus | 2–3 | `[Pre-Chorus - rising, strings swell]` |
| Chorus | 2–3 | `[Chorus - belted, high energy]` |
| Bridge | 2–3 | `[Bridge - stripped back, spoken word]` |
| Solo | 1–2 | `[Guitar Solo - intense]` |
| Outro | 1–2 | `[Outro - fade out, quiet]` |

---

## Section 12: Quick Reference — Marker + Descriptor Combinations

| Marker | Common combinations |
| :--- | :--- |
| `[Intro]` | sparse, piano; fade in, atmospheric; organ, hypnotic |
| `[Verse 1]` | whispered, building; raspy vocal, sparse; nasal, plaintive |
| `[Pre-Chorus]` | rising, strings swell; building, sidechain kick |
| `[Chorus]` | belted, high energy; harmonized, powerful belting; anthemic, full band |
| `[Bridge]` | stripped back, spoken word; quiet, intimate; menacing, building |
| `[Outro]` | fade out, quiet; abrupt cut, dry; bleak, atmospheric |
| `[Guitar Solo]` | intense, reverb-heavy; blues-based, wah-wah; melodic, vibrato |
| `[Organ Solo]` | jazz-influenced, reverb-heavy; hypnotic, building |
| `[Drum Break]` | thunderous, tom-heavy; tight, snare-heavy |
| `[Bass Solo]` | distorted, rumbling; simple, driving |
| `[Drop]` | explosive, 808 bass; full, sidechain kick |
| `[Breakdown]` | stripped back, heavy; doom-laden, slow |

---

## Section 13: Dataset-Specific Quick Reference

**Hank Williams Sr.**
Genre: honky-tonk country, country gospel, country blues.
Vocal: sincere hillbilly, high nasal strained, lonesome, weathered, plaintive.
Instruments: non-pedal steel guitar, double-stop fiddle, acoustic guitar, upright bass.
Mood: lonesome, mournful, heartfelt, celebratory, sincere.
Production: monaural, era-standard, sparse, stripped-down.

**The Doors**
Genre: psychedelic rock, acid rock, blues rock, art rock.
Vocal: deep baritone, brooding, poetic, spoken word, theatrical, hypnotic.
Instruments: Vox Continental organ, Fender Rhodes piano, Gibson SG guitar,
jazz-influenced drums.
Mood: ominous, nocturnal, hypnotic, cinematic, mysterious, erotic.
Production: 1960s analog, reverb-heavy, live feel, dynamic shifts.

**Black Sabbath (pre-Never Say Die)**
Genre: heavy metal, doom metal, hard rock, proto-metal.
Vocal: nasal, haunting, plaintive, eerie, wailing, melodic.
Instruments: downtuned guitar, heavy riffing, distorted bass, thunderous drums.
Mood: doom-laden, ominous, apocalyptic, sinister, occult, bleak.
Production: raw, 1970s analog, dry, live feel, heavy low-end.

**AC/DC (Bon Scott years)**
Genre: hard rock, blues rock, rock and roll.
Vocal: raspy, sneering, snarling, swaggering, cheeky, anthemic.
Instruments: clean-to-crunchy guitar, riff-based, tight rhythm section, driving bass.
Mood: energetic, aggressive, anthemic, rebellious, party, rowdy.
Production: raw, 1970s analog, dry, punchy, in-your-face.

---

## Sources & precedence

1. **This file** — authoritative for this project. Tuned for dataset annotation.
2. `docs/ACE_Step_1.5_Master_Annotation_Guide.md` — the format spec this follows.
3. The official `acestep-songwriting` skill (ace-step/ace-step-1.5) — used to fill
   gaps only: its wider vocal/energy tag list and the caption↔lyrics consistency
   checks. It permits more than 3 descriptors per bracket; **this file's max-3
   rule wins** because it guards against the model singing tag names as lyrics.

`[Energy: Low]` `[Acapella]` `[Whistling]` `[Sound Effect]`
