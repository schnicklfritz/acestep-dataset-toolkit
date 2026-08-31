# Master-Track Research: MOGG Metadata & Artist Session Info

Purpose: source **ground-truth instrument metadata** for the five focus artists
(The Doors, Black Sabbath, AC/DC, Hank Williams Sr, Jimi Hendrix) so the ACE-Step
dataset captions name the real instruments. This doc covers where the metadata
lives, the MOGG header format, and per-artist studio lineups.

> ⚠️ Licensing note: MOGG files from Rock Band/Guitar Hero are **ripped,
> copyrighted multitrack recordings**. The *metadata* (track/channel names,
> session lineups, song lists) is fine to research and use as ground truth.
> Training models on the actual ripped **audio** without a license is a legal
> risk — for released multitracks prefer licensed sources (Rock Band Network
> stems, Jammit, official box-set / session releases, ACM/Indaba contest stems).

---

## 1. MOGG files & header metadata

* **MOGG** = multi-channel Ogg Vorbis used by Rock Band / Guitar Hero. Each
  channel corresponds to a studio track (e.g. `vocals`, `drums`, `bass`,
  `lead guitar`, `rhythm guitar L`, `rhythm guitar R`, `keys`, …). The
  **header embeds the channel/track names as text** — that text is the
  "header dump" metadata used for instrument ground truth.
* Some MOGG files are **encrypted** (older C3 CON Tools); decryption tooling
  exists (see onyx below).
* **Toolkit:** [mtolly/onyx](https://github.com/mtolly/onyx) — converts/builds
  RB/GH/Clone Hero songs, reads MOGG (incl. encrypted) + Bink. The right tool
  for dumping headers locally.
* **Extraction guides:** Frets on Fire forum (the original RB/GH extraction
  hub), `multitrackdownloads.blogspot.com`, and SongStems Discord (community
  rippers with an Excel of Artist/Song/DL links).

## 2. Readable "header dump" catalog (no audio needed)

* **[isolated-tracks.com](https://isolated-tracks.com)** — catalog of
  multitracks with **exact per-song channel lists**. This is effectively the
  MOGG header metadata in text. Examples found:
  * Black Sabbath — *Iron Man*: `Drum Kit, Bass, Electric Guitar 1, Electric
    Guitar 2, Voice (I am Iron Man), Lead Vocal`; another title:
    `Metronome, Drum Kit, Tambourine, Bass, Electric Guitar (L), Electric
    Guitar (R), Lead Electric Guitar, Siren, Lead Vocal`
  * AC/DC titles: `Metronome, Drum Kit, Bass, Electric Guitar (L), Electric
    Guitar (R), Lead Electric Guitar, Rhythm Electric Guitar (C), Rhythm
    Electric Guitar (R), Backing Vocals, Lead Vocal` (10 ch); also 8-ch and
    10-ch with electronic drum kit variants
  * Great for building an instrument ground-truth table per song.
* **[multitrackmaster.com](https://multitrackmaster.com)** — streams isolated
  tracks (many are RB-origin); **no downloads** (they don't own the tracks).
  Good for *verifying* which instruments are in a master.

## 3. Clone Hero master-track spreadsheets

* The spreadsheet "where most everyone gets their custom songs" (official
  Clone Hero account): Google Sheets with every RB/GH custom/master chart.
* Public copy example:
  `docs.google.com/spreadsheets/d/13B823ukxdVMocowo1s5XnT3tzciOfruhUVePENKc01o/`
  (GH + RB songs for Clone Hero).
* r/CloneHero "I compiled all of the official Guitar Hero charts" thread —
  catalog of all official GH charts.
* [customsongscentral.com](https://customsongscentral.com) — 5,000+ charts.
* Use these to build the **master-track song list** (artist → song → source
  game/DLC) for the dataset.

## 4. Box sets & official session documentation

Anniversary/deluxe box sets print **high-res photos of tape boxes and session
track sheets** — authoritative, published documentation of who played what.

* **Jimi Hendrix**
  * *The Jimi Hendrix Experience* ("Purple Box", 2000 / expanded) — 60
    tracks, chronological 1966–70, extensive session notes (McDermott/Marsh).
  * *Electric Lady Studios: A Jimi Hendrix Vision* — 38 previously-unreleased
    1970 tracks; lineups documented (Billy Cox bass, Mitch Mitchell drums).
  * Session musicians: **Jimi Hendrix** (guitar/vocals), **Noel Redding**
    (bass), **Mitch Mitchell** (drums); later **Billy Cox** (bass),
    **Buddy Miles** (drums); guests **Larry Young** (organ), **Larry Lee**
    (rhythm guitar), **Steve Winwood** (backing vocals), **Juma Sultan /
    Jerry Velez** (percussion).
* **The Doors** — 1998 *Box Set* + deluxe/anniversary editions publish
  tape-box and session documentation.
* **Black Sabbath / AC/DC** — deluxe reissues include session notes; the
  isolated-tracks channel lists are the practical ground truth.

## 5. Per-artist instrument ground truth (for captions)

### Hank Williams Sr
* Mono-era Nashville (1946–1952). Classic session lineup — **the Drifting
  Cowboys**:
  * **Vocals**: Hank Williams
  * **Steel guitar**: Don Helms (all 1950–52 recordings); earlier: Jerry Byrd
    ("I'm So Lonesome I Could Cry", "Mansion on the Hill", "Lovesick Blues"),
    Smokey Lohman ("Honky Tonkin'", "I Saw the Light"), Don Davis ("Honky Tonk
    Blues", "Lost Highway", "Mind Your Own Business"), Herman Herron
  * **Fiddle**: Jerry Rivers
  * **Guitar**: Sammy Pruett, Chet Atkins (overdubs), Bob McNett, Zeke Turner
    (electric), Jack Shook (rhythm), Ray Edenton (rhythm)
  * **Bass**: Charles "Indian Chuck" Wright, Floyd "Lightnin'" Chance
  * **Drums**: generally **absent** on the classic sides (added later on
    transcriptions, e.g. Buddy Harman 1959)
  * **Piano**: Floyd Cramer (1959 transcriptions)
* → Captions should expect: vocals, steel guitar, fiddle, acoustic/electric
  guitar, upright bass — **no drums** on the classics.

### Jimi Hendrix
* Experience (1966–69): **guitar (Hendrix), bass (Redding), drums (Mitchell)**.
* 1969–70 (Band of Gypsys / Electric Lady): **guitar, bass (Billy Cox),
  drums (Mitch Mitchell or Buddy Miles)**; organ (Larry Young), percussion,
  backing vocals possible.
* Expect "lead/rhythm electric guitar, bass guitar, drums, vocals" plus organ/
  percussion on 1970 sessions.

### The Doors
* Core: **vocals (Jim Morrison), keyboards (Ray Manzarek — Vox Continental
  organ, Rhodes, Fender bass VI / bass keyboard), guitar (Robby Krieger),
  drums (John Densmore)**.
* No dedicated bassist — bass lines on keyboard; session bassists on later
  tracks (Douglas Lubahn, Harvey Brooks, Kerry Magness, Larry Knechtel) +
  occasional horns (Curtis Amy sax) and congas (Reinol Andino).
* Expect: organ, electric piano, guitar, drums, vocals, bass (keys), possible
  saxophone/congas.

### Black Sabbath
* Classic (1970–78): **vocals (Ozzy Osbourne), guitar (Tony Iommi), bass
  (Geezer Butler), drums (Bill Ward)**.
* Channel examples from isolated-tracks show: metronome, drum kit, bass,
  electric guitar L/R, lead electric guitar, tambourine, siren, lead vocal
  (and *Voice (I am Iron Man)* on Iron Man). Expect slight channel variety.
* → Expect: lead/rhythm electric guitar, bass, drums, vocals (+ occasional
  keyboard, tambourine, siren).

### AC/DC
* Classic: **lead guitar (Angus Young), rhythm guitar (Malcolm Young), vocals
  (Bon Scott/Brian Johnson), bass (Cliff Williams), drums (Phil Rudd)**.
* Channel examples: metronome, drum kit / electronic drum kit, bass, electric
  guitar L/R, lead electric guitar, rhythm electric guitar C/R, backing
  vocals, lead vocal.
* → Expect: dual electric guitars (lead + rhythm), bass, drums, lead + backing
  vocals; watch for the metronome channel.

## 6. Suggested workflow for the dataset builder

1. Use the **Clone Hero master list** to enumerate target songs per artist.
2. For each song, pull the **channel list** from isolated-tracks (or a MOGG
   header dump via onyx) → authoritative instrument ground truth.
3. Feed the ground-truth instrument list into the toolkit's "Detect via
   Captioner" flow (or directly as `instrument_models` recommendations).
4. Cross-check against box-set session notes (Hendrix/Doors) where available.

