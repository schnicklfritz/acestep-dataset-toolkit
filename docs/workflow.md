# ACE-Step Workflow Guide

> Process / how-to-run-a-dataset guide. For the *annotation standard itself*,
> see `docs/ACE_Step_1.5_Master_Annotation_Guide.md`. This guide answers
> "what order do I do things in?", not "what does valid output look like?".

---

## Recommended order of operations

1. **Dataset Studio** — Add single song or an audio folder (recursive, deduped). Tracks land in the table instantly with blank, editable metadata.
2. **🗣 Lyrics & Tags** — transcribe lyrics and tag the track **before** deciding anything about stems (see "Why lyrics/tags come first" below).
3. **🎚 DSP Normalize** — EBU R128 loudness / sample-rate normalization (defaults −14 LUFS / 44.1 kHz; originals backed up).
4. **🎸 Stem-split decision** — *decide*, don't auto-run. Whether/how to split is a choice informed by what steps 2–3 revealed. It is not a mandatory step for every dataset.
5. **🚀 Caption** — run the AI captioner and review each caption before accepting.
7. **Structural / Spatial** — per-section (and optionally per-stem) captions. Only meaningful if stems were created; otherwise this collapses into a single section-level caption.
8. **Validate & Save JSON** — run manifest validation, then save the ACE-Step manifest.

---

## Why lyrics/tags come before the stem-split decision

Stem splitting is a **means** to per-section captioning, not a goal in its own right. If lyrics + tags are produced first, you already know what is in the song — instruments present, whether vocals exist, the structural shape — so the stem-split decision is informed rather than arbitrary.

For many datasets the whole-song caption (steps 2–3 + 6) is enough and **no stems are needed at all**. Only add stem splitting when you genuinely need per-instrument / per-position detail across sections.

---

## Recommended lyrics / tag engine

**Gemini via Google AI Studio (free) is a strong default** because it was
trained on a very large body of sheet-music and tablature data, so it already
reasons reliably about BPM, key, timing, and lyric structure — and it
understands lyrics as a scored, time-ordered script.

To get output that matches the toolkit's expectations, feed it the master
annotation guide:

> System: *"You are a music annotation assistant conforming to the ACE-Step
> 1.5 standard. Read `docs/ACE_Step_1.5_Master_Annotation_Guide.md` first and
> produce annotations in exactly that schema."* + the guide's §5 prompt.

Backends you can still choose instead, per workflow step: local (free, private,
hardware-bound), Kaggle (free/near-free GPU, some queue/setup overhead),
MVSEP (feature-rich; free tier is limited / time-consuming), or import.

---

## Decisions you will be asked about (no wrong answers assumed)

- **Background vocals** — easy to forget when only 1–2 tracks in a set have them. Always check before finalizing stems.
- **Fullness variants** — MVSEP may return the same stem at more than one "fullness" level; pick/rename deliberately or keep both.
- **Stem naming** — the common convention is one folder per song with bare lowercase stem names (`vocals.flac`, `instrumental.flac`, …). A "simplified rename" option can normalize MVSEP's long downloaded names.
