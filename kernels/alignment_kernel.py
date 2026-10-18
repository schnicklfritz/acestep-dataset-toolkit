"""Kaggle kernel script template for word-level forced alignment (MMS_FA).

This file is never run directly on your machine -- ``workers/kaggle_alignment.py``
reads it, string-replaces the ``{{...}}`` placeholders below, and pushes the
result to a private Kaggle GPU kernel. Keep the placeholder names in sync with
``run_kaggle_alignment()``.

It is a port of the standalone MMS_FA pass the toolkit used to run by hand, with
three changes it needs in order to run over a DATASET instead of one song:

  * it aligns EVERY audio file it finds and keys its output by basename, so one
    push can cover a single track (what the Lyrics page button does) or a batch;
  * the lyrics come from ``lyrics.json`` staged beside the audio, not from text
    hard-coded in this file, and the ``[Section]`` layout is kept so aligned
    words map back onto LYRIC LINES -- which is the shape ``export_lrc`` reads;
  * a word's duration boost now reaches the decision. The hand-run script
    assigned ``raw_shout`` twice (once from the duration-boosted level, then
    again from the plain level), so the sustained-note boost was computed and
    then silently discarded. ``effective_db`` is used consistently here.

Placeholders:
    {{AUDIO_DATASET_PATH}}   /kaggle/input/<slug> -- audio + lyrics.json
"""
import glob
import json
import math
import os
import statistics

import torch
import torchaudio

AUDIO_DATASET_PATH = "{{AUDIO_DATASET_PATH}}"
LYRICS_JSON = "lyrics.json"
OUT_PATH = "/kaggle/working/alignment_out.json"

SAMPLE_RATE = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")

# A longer file is skipped rather than silently truncated: a half-aligned song
# would produce line timings that look authoritative and are wrong past the cut.
MAX_AUDIO_SEC = 900

# --- the shout classifier's constants -------------------------------------
# The threshold is self-normalising: mean + K * sigma of THIS track's word
# levels, so a quiet mix and a loud master are judged on the same scale.
THRESHOLD_K = 0.35
# Below this a word is too short for its RMS to mean anything; it is still
# aligned and emitted, it just does not set the statistics.
MIN_WORD_SEC = 0.12
# A held vowel reads as loud because it is LONG, not because it is shouted.
# These two give it the boost that the hand-run script computed and dropped.
SUSTAIN_SEC = 0.45
SUSTAIN_DB = 1.5
# Fix A: a short run of quiet words inside a shouted phrase is a breath, not a
# return to normal delivery.
BRIDGE_GAP = 2
# Fix B: three loud words in a seven-word window carry their neighbours. The
# window MUST be wider than BRIDGE_GAP, or Fix A has already claimed every word
# Fix B could reach (a single quiet word between shouts is always bridged) and
# this pass is dead code. Measured: at ±2 the pass never fired.
PHRASE_WINDOW = 7
PHRASE_MIN = 3
PHRASE_DB = 2.0
# A track with no dynamic range (silence, a constant test tone) cannot single
# any word out. Without this floor the threshold IS the mean, so every word
# would land on it and the whole lyric would come back in capitals.
MIN_SPREAD_DB = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_VOCAL_HINTS = ("vocal", "vocals", "vox", "lead_vocal")


def _group_key(path):
    """Filename with any stem suffix removed, so a song's variants group up."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for hint in _VOCAL_HINTS:
        stem = stem.replace(hint, "")
    return "".join(ch for ch in stem if ch.isalnum())


def _pick_audio(root):
    """One file per song, preferring the vocal stem when both were staged.

    MMS_FA on a full mix aligns the instruments' transients as readily as the
    voice; on a separated vocal it has one thing to listen to. When the stem
    kernel's output is uploaded alongside the mix, this is what makes it be used.
    """
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", "*" + ext), recursive=True))
    groups = {}
    for path in sorted(set(files)):
        groups.setdefault(_group_key(path), []).append(path)
    picked = []
    for paths in groups.values():
        vocal = [p for p in paths if any(h in os.path.basename(p).lower() for h in _VOCAL_HINTS)]
        picked.append(sorted(vocal or paths)[0])
    return picked


def _load_lyrics(root):
    path = os.path.join(root, LYRICS_JSON)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {LYRICS_JSON} beside the audio.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _lyric_lines(lyrics):
    """Non-empty lyric lines, in order. ``[Section]`` markers are not aligned."""
    lines = []
    for raw in (lyrics or "").splitlines():
        text = raw.strip()
        if text and not (text.startswith("[") and text.endswith("]")):
            lines.append(text)
    return lines


def _load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.to(DEVICE)


def _db(waveform, start, end):
    """RMS level of ``[start, end)`` in seconds, in dBFS (silence -> -120)."""
    first = max(0, int(start * SAMPLE_RATE))
    last = min(waveform.shape[-1], int(end * SAMPLE_RATE))
    if last <= first:
        return -120.0
    window = waveform[0, first:last]
    rms = float(torch.sqrt(torch.mean(window * window)).item())
    return 20.0 * math.log10(max(rms, 1e-6))


def _align_file(model, bundle, waveform, lyrics):
    """Return ``(words, lines)`` for one track, or raise with the reason."""
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    if waveform.shape[-1] / SAMPLE_RATE > MAX_AUDIO_SEC:
        raise RuntimeError(f"longer than {MAX_AUDIO_SEC}s; split it first")

    # Tokenize PER WORD and align the flat token stream, then split the spans by
    # count. torchaudio's tutorial maps spaces back to word boundaries; counting
    # is the same thing without depending on how the bundle encodes a space.
    words, flat, counts = [], [], []
    for line in _lyric_lines(lyrics):
        for word in line.split():
            ids = tokenizer(word, star="*") or tokenizer("*", star="*")
            words.append(word)
            flat.extend(ids)
            counts.append(len(ids))
    if not words:
        raise RuntimeError("no lyric words to align")

    with torch.inference_mode():
        emission, _ = model(waveform)
    emission = emission[0]
    spans = aligner(emission, flat)

    # MMS_FA emits at 50 Hz; derive the ratio from the real tensors instead of
    # assuming a hop length, so a bundle change cannot shift every timestamp.
    sec_per_frame = (waveform.shape[-1] / SAMPLE_RATE) / emission.shape[0]

    records, cursor = [], 0
    for word, count in zip(words, counts):
        word_spans = spans[cursor:cursor + count]
        cursor += count
        if not word_spans:
            continue
        start = word_spans[0].start * sec_per_frame
        end = word_spans[-1].end * sec_per_frame
        start, end = max(0.0, start), max(start, end)
        records.append({
            "word": word,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "db": round(_db(waveform, start, end), 2),
        })
    if not records:
        raise RuntimeError("aligner returned no spans")

    _classify(records)

    # Line-level segments for ``export_lrc``: a segment per lyric line, starting
    # where its first word was aligned. Words that failed to align are dropped
    # from the text rather than shifted onto a neighbour's timestamp.
    lines, index = [], 0
    for line in _lyric_lines(lyrics):
        count = len(line.split())
        chunk = records[index:index + count]
        index += count
        if chunk:
            lines.append({"start": round(chunk[0]["start"], 3),
                          "text": " ".join(w["final_word"] for w in chunk)})
def _classify(records):
    """Mark each record's ``raw_shout`` / ``shouted`` / ``final_word`` in place.

    Three passes, in this order: the per-track threshold, Fix A (bridge short
    quiet runs inside a shouted phrase), Fix B (phrase elevation). Each word
    keeps the intermediate flags so a surprising capitalization can be traced
    to the pass that caused it.
    """
    levels = [r["db"] + (SUSTAIN_DB if r["duration"] >= SUSTAIN_SEC else 0.0)
              for r in records]
    for record, level in zip(records, levels):
        record["effective_db"] = round(level, 2)

    sample = [lvl for r, lvl in zip(records, levels) if r["duration"] >= MIN_WORD_SEC]
    sample = sample or levels
    spread = statistics.pstdev(sample)
    threshold = statistics.fmean(sample) + THRESHOLD_K * spread

    for record, level in zip(records, levels):
        record["threshold_db"] = round(threshold, 2)
        record["delta"] = round(level - threshold, 2)
        record["raw_shout"] = spread >= MIN_SPREAD_DB and level >= threshold

    flags = [r["raw_shout"] for r in records]

    # Fix A -- bridge: a run of 1..BRIDGE_GAP quiet words between two shouted
    # ones is part of the same phrase. The run is measured strictly BETWEEN the
    # shouts; an earlier version stepped the far marker one word too far and
    # bridged runs of three.
    for i, flag in enumerate(flags):
        if flag:
            continue
        left = i - 1
        while left >= 0 and not flags[left]:
            left -= 1
        right = i + 1
        while right < len(flags) and not flags[right]:
            right += 1
        if left < 0 or right >= len(flags):
            continue
        if (right - left - 1) <= BRIDGE_GAP:
            for j in range(left + 1, right):
                flags[j] = True
    for record, flag in zip(records, flags):
        record["bridged"] = bool(flag and not record["raw_shout"])

    # Fix B -- phrase elevation: a word just under the threshold, surrounded by
    # shouted words, is shouted too. It has to land within PHRASE_DB, so the
    # window cannot drag quiet talk into capitals.
    half = PHRASE_WINDOW // 2
    elevated = []
    for i, record in enumerate(records):
        if flags[i]:
            elevated.append(True)
            continue
        lo, hi = max(0, i - half), min(len(records), i + half + 1)
        neighbours = sum(1 for j in range(lo, hi) if j != i and flags[j])
        elevated.append(neighbours >= PHRASE_MIN and record["delta"] >= -PHRASE_DB)

    for record, flag in zip(records, elevated):
        record["phrase_elevated"] = bool(
            flag and not record["raw_shout"] and not record["bridged"])
        record["shouted"] = bool(flag)
        record["final_word"] = record["word"].upper() if flag else record["word"]
        record["tag"] = "LOUD" if flag else "quiet"


def main():
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(DEVICE).eval()
    lyrics_by_track = _load_lyrics(AUDIO_DATASET_PATH)

    out = {"tracks": {}, "errors": {}}
    for path in _pick_audio(AUDIO_DATASET_PATH):
        name = os.path.basename(path)
        # Also accept the stem as the key, so a track renamed to add "_vocals"
        # still finds its lyrics.
        text = lyrics_by_track.get(name) or lyrics_by_track.get(os.path.splitext(name)[0])
        if not text:
            out["errors"][name] = "no lyrics staged for this file"
            continue
        try:
            words, lines = _align_file(model, bundle, _load_audio(path), text)
        except Exception as exc:                              # noqa: BLE001
            out["errors"][name] = f"{type(exc).__name__}: {exc}"
            continue
        out["tracks"][name] = {
            "segments": lines,
            "words": words,
            "lyrics": "\n".join(line["text"] for line in lines),
            "shout_ratio": round(sum(1 for w in words if w["shouted"]) / len(words), 3),
        }
        print(f"[align] {name}: {len(words)} words, "
              f"{out['tracks'][name]['shout_ratio']:.0%} shouted", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"[align] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

