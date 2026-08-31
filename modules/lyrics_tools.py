"""Lyrics tools: syllable-counted line splitting + timed LRC export."""
import re
from pathlib import Path

_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def count_syllables(text):
    """Rough syllable count (vowel groups per word)."""
    total = 0
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        n = len(_VOWEL_GROUP.findall(word))
        total += max(1, n)
    return total


def _split_line(s, max_syllables):
    if count_syllables(s) <= max_syllables:
        return [s]
    words = s.split()
    best = 0
    for i in range(1, len(words)):
        if count_syllables(" ".join(words[:i])) <= max_syllables:
            best = i
    if best <= 0 or best >= len(words):
        return [s]
    return [" ".join(words[:best])] + _split_line(" ".join(words[best:]), max_syllables)


def split_long_lines(text, max_syllables=10):
    """Split lines longer than ``max_syllables`` at word boundaries.

    Matches ACE-Step's guidance (6-10 syllables per line, singable in one
    breath). Section markers ([Verse], [Chorus]...) are left untouched.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        if s.startswith("["):
            out.append(s)
            continue
        out.extend(_split_line(s, max_syllables))
    return "\n".join(out)


def export_lrc(lyrics, segments, path):
    """Write timed LRC when word/segment timestamps exist, else section-tagged text."""
    if segments:
        lines = []
        for seg in segments:
            start = float(seg.get("start", 0) or 0)
            mm = int(start // 60)
            ss = int(start % 60)
            cs = int(round((start - int(start)) * 100))
            lines.append(f"[{mm:02d}:{ss:02d}.{cs:02d}]{seg.get('text', '').strip()}")
        text = "\n".join(lines)
    else:
        text = lyrics
    Path(path).write_text(text, encoding="utf-8")