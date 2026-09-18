"""Lyrics normalization — make text singable and TTS-friendly.

PROBLEM
-------
Apostrophes are unpredictable tokens. A singing/TTS model may read ``'`` as a
literal character, a glottal stop, or produce a wrong vowel — ``she'll`` can
come out as "shell", "she-ull", or "sheel" with no way to predict which. There
is no universal fix, so the approach here is to replace contractions with
**deliberately spelled-out phonetic equivalents** so pronunciation is
deterministic and reviewable.

WHAT THIS DOES
--------------
1. Contractions -> phonetic spelling from an editable table
   (``she'll`` -> ``sheel``, ``he'll`` -> ``heel``, ``I'll`` -> ``aisle``).
2. Optional ``-ing`` -> ``-in`` (singer-dependent: Ozzy enunciates it, many do
   not — hence a switch, with an exception list so ``ring``/``thing`` stay).
3. Capitalizes structure tags (``[verse]`` -> ``[Verse]``).
4. Strips trailing punctuation from lyric lines, except ``"`` and ``)``.

Nothing here touches ``[Section]`` lines except tag capitalization.
"""
import re

# ---------------------------------------------------------------------------
# Contractions -> phonetic spelling
# ---------------------------------------------------------------------------
# Editable in the 🎤 Lyrics tab; this is the shipped default.
# NOTE on 'll: the vowel is matched to the host word rather than a generic
# "'ll" removal, which is what produces "shell" from "she'll".
DEFAULT_CONTRACTIONS = {
    # --- 'll : vowel-matched, the she'll / he'll / I'll problem ---------
    "she'll": "sheel",
    "he'll": "heel",
    "i'll": "aisle",
    "we'll": "weel",
    "you'll": "yool",
    "they'll": "theyl",
    "it'll": "idle",
    "that'll": "thadle",
    "there'll": "therel",
    "who'll": "hool",
    # --- n't : keeps the syllable count / rhythm (cant, dont, wont) -----
    "can't": "cant",
    "don't": "dont",
    "won't": "wont",
    "isn't": "isnt",
    "aren't": "arent",
    "wasn't": "wasnt",
    "weren't": "werent",
    "hasn't": "hasnt",
    "haven't": "havent",
    "hadn't": "hadnt",
    "doesn't": "doesnt",
    "didn't": "didnt",
    "couldn't": "couldnt",
    "wouldn't": "wouldnt",
    "shouldn't": "shouldnt",
    "mustn't": "mustnt",
    "ain't": "aint",
    # --- 're / 've / 'd / 'm -------------------------------------------
    "we're": "weer",
    "they're": "theyre",
    "you're": "your",
    "we've": "weev",
    "i've": "ive",
    "you've": "youve",
    "they've": "theyve",
    "i'm": "im",
    "let's": "lets",
    # --- leading apostrophes (more often sung as one word) --------------
    "'cause": "cause",
    "'bout": "bout",
    "'til": "til",
    "'round": "round",
    "'em": "em",
}

# ---------------------------------------------------------------------------
# -ing -> -in
# ---------------------------------------------------------------------------
# Words ending in -ing that must NOT be shortened, because the -ing is not a
# syllabic suffix there. Compared case-insensitively.
DEFAULT_ING_EXCEPTIONS = {
    "bring", "spring", "string", "sting", "swing", "cling", "fling", "sling",
    "ring", "sing", "king", "thing", "wing", "ling", "ping", "zing", "ding",
    "bing", "everything", "something", "anything", "nothing", "morning",
    "evening", "darling", "wedding", "building", "during", "boring",
}

# Punctuation removed from the END of a lyric line. `"` and `)` are kept.
STRIP_TRAILING_PUNCT = ".,!?;:-–—…•*_~`^"

# A structure tag alone on its line, e.g. "[Verse 1]" / "[Guitar Solo]".
_TAG_LINE = re.compile(r"^\s*(\[.*?\])\s*$")
# Leading structure tags followed by lyric text on the same line.


def _title_keep_hyphens(word):
    """Title-case a single token, treating a joined hyphen as a word boundary."""
    parts = word.split("-")
    return "-".join(p[:1].upper() + p[1:].lower() for p in parts)


def capitalize_tags(text):
    """Normalize structure-tag capitalisation.

    A structural tag is everything from ``[`` up to the first ``" -"``
    (space + hyphen). That leading part is Capitalized; anything after the
    ``" -"`` is a modifier and stays lowercase.

    A *joined* hyphen is not the separator — it is part of the tag itself, so it
    title-cases both halves. Only ``" -"`` (with a space) splits tag from
    modifier::

        [verse 1]              -> [Verse 1]
        [pre-chorus]           -> [Pre-Chorus]
        [GUITAR SOLO]          -> [Guitar Solo]
        [verse - quiet]        -> [Verse - quiet]
        [chorus - loud - big]  -> [Chorus - loud - big]
    """
    splitter = " -"

    def _title(tag):
        return " ".join(_title_keep_hyphens(w) for w in tag.split(" "))

    def _fix(match):
        inner = match.group(1)
        if splitter in inner:
            head, _, tail = inner.partition(splitter)
            return "[" + _title(head) + splitter + tail.lower() + "]"
        return "[" + _title(inner) + "]"

    return re.sub(r"\[([^\]]*)\]", _fix, text)


def _match_case(replacement, source):
    """Carry ``source``'s capitalisation onto ``replacement``.

    The table maps *spelling*, never capitalisation. A word and its capitalised
    forms are the same word — one rule covers all of them, and the output keeps
    whatever case the source had:

        she'll -> sheel      SHE'LL -> SHEEL      She'll -> Sheel

    A word that is NOT in the table is left completely alone, in every case.
    """
    if not source or not replacement:
        return replacement
    letters = [c for c in source if c.isalpha()]
    if not letters:
        return replacement

    if all(c.isupper() for c in letters):
        return replacement.upper()
    if len(letters) >= 2 and all(c.islower() for c in letters):
        return replacement.lower()
    if source[:1].isupper() and all(c.islower() for c in letters[1:]):
        return replacement[:1].upper() + replacement[1:]

    # Unusual casing in the source (rare): mirror it per character where the
    # lengths line up, otherwise fall back to a leading capital.
    if len(replacement) >= len(source):
        out = []
        for i, ch in enumerate(source):
            rch = replacement[i]
            out.append(rch.upper() if ch.isupper() else rch.lower())
        return "".join(out) + replacement[len(source):]
    return replacement[:1].upper() + replacement[1:]


def apply_contractions(text, table):
    """Replace contractions using ``table`` (case-insensitive whole words).

    Capitalisation of the matched word is carried onto the replacement, so a
    single entry covers lower, Title, and ALL-CAPS forms.
    """
    if not table:
        return text
    # Longest first so "she'll" wins over any shorter overlapping entry.
    keys = sorted(table, key=len, reverse=True)
    lowered = {k.lower(): v for k, v in table.items()}
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
        re.IGNORECASE,
    )

    def _sub(match):
        found = match.group(0)
        replacement = lowered.get(found.lower())
        if replacement is None:
            return found
        return _match_case(replacement, found)

    return pattern.sub(_sub, text)


def strip_all_apostrophes(text):
    """Remove every apostrophe character, wherever it appears.

    Applied *after* the contraction table, so a mapped word (``she'll`` ->
    ``sheel``) is already resolved and only leftover apostrophes go —
    ``rock 'n' roll`` -> ``rock nr roll``, ``sun's`` -> ``suns``.
    """
    for ch in ("\u2019", "\u2018", "\u02bc", "\u2032"):  # curly, modifier, prime
        text = text.replace(ch, "")
    return text.replace("'", "")


def shorten_ing(text, exceptions):
    """Convert word-final ``-ing`` to ``-in``, skipping ``exceptions``."""
    exceptions = {w.lower() for w in (exceptions or ())}

    def _sub(match):
        word = match.group(0)
        if word.lower() in exceptions:
            return word
        if len(word) <= 4:  # "ring", "king", "sing" — too short to be a suffix
            return word
        return word[:-3] + "in"

    return re.sub(r"\b[A-Za-z]+ing\b", _sub, text)


def strip_trailing_punctuation(line):
    """Drop trailing punctuation, keeping ``"`` and ``)``.

    Also cleans punctuation sitting *before* a kept closing quote/paren, so
    ``"Save me!)`` becomes ``"Save me)``.
    """
    stripped = line.rstrip()
    if not stripped:
        return line

    keepers = {'"', ")"}
    tail_kept = ""
    while stripped and stripped[-1] in keepers:
        tail_kept = stripped[-1] + tail_kept
        stripped = stripped[:-1]

    while stripped and stripped[-1] in STRIP_TRAILING_PUNCT:
        stripped = stripped[:-1].rstrip()

    return stripped + tail_kept


def capitalize_first_word(line):
    """Capitalise the first letter of a lyric line, leaving the rest alone.

    Skips anything that has no leading letter (blank lines, a lone "'", a
    line already starting with a capital, a line starting with punctuation).
    Only the FIRST character is touched: the rest of the line keeps whatever
    case the writer used, because a line may legitimately be all-caps.

    A word that already carries intentional inner capitalisation (iPhone, eBay,
    iTunes) is left as written — turning it into IPhone would be worse than
    leaving it lowercase.
    """
    for i, ch in enumerate(line):
        if ch.isalpha():
            if i == 0 and ch.isupper():
                return line          # already capitalised
            # Reject deliberate mixed-case words like iPhone / eBay.
            word = line[i:].split(" ", 1)[0]
            if any(c.isupper() for c in word[1:]):
                return line
            return line[:i] + ch.upper() + line[i + 1:]
        if not ch.isspace():
            return line              # starts with punctuation/digit; leave it
    return line                      # blank or no letters at all


def normalize_lyrics(
    text,
    contractions=None,
    ing_to_in=False,
    ing_exceptions=None,
    do_capitalize_tags=True,
    do_strip_punctuation=True,
    do_capitalize_lines=True,
    protect_markers=True,
):
    """Normalize a lyrics block. Returns ``(new_text, report)``.

    ``report`` carries ``contractions`` (list of ``(from, to)`` applied),
    ``ing`` (count converted), ``tags`` (count touched) and ``lines_changed``.

    ``do_capitalize_lines`` capitalises the first word of every lyric line.
    Lyrics are normally written as sentences, and a lowercase line start reads
    as a typo in a training caption. Only the first character is affected — the
    rest of the line is untouched, so an intentionally ALL-CAPS line survives.

    ``protect_markers`` leaves ``---- filename ----`` separator lines (used by
    the all-lyrics view) completely untouched, so tidying a whole-dataset block
    cannot corrupt the markers that write-back depends on.
    """
    if contractions is None:
        contractions = DEFAULT_CONTRACTIONS
    if ing_exceptions is None:
        ing_exceptions = DEFAULT_ING_EXCEPTIONS

    report = {"contractions": [], "ing": 0, "tags": 0, "lines_changed": 0,
              "apostrophes": 0, "capitalized": 0, "word_changes": []}
    changed = set()
    lowered = {k.lower(): v for k, v in (contractions or {}).items()}
    out_lines = []

    for raw_line in (text or "").splitlines():
        line = raw_line

        # 0. Separator markers (---- filename ----) pass through verbatim.
        stripped = line.strip()
        if protect_markers and stripped.startswith("----") and stripped.endswith("----") and len(stripped) > 8:
            out_lines.append(line)
            continue

        # 1. Structure tags: capitalize only.
        if do_capitalize_tags and "[" in line:
            fixed = capitalize_tags(line)
            if fixed != line:
                report["tags"] += 1
                line = fixed

        # 2. A line that is only a tag needs no further processing.
        if _TAG_LINE.match(line):
            out_lines.append(line)
            if line != raw_line:
                report["lines_changed"] += 1
            continue

        # 3. Split any leading tags off so tag text is untouched by later passes.
        prefix, body = "", line
        m = _LEADING_TAGS.match(line)
        if m and m.group(2).strip():
            prefix, body = m.group(1), m.group(2)

        # 4. Contractions.
        before = body
        body = apply_contractions(body, contractions or {})
        if body != before:
            # Count apostrophes consumed by the table too: from the user's
            # point of view this line DID have apostrophes handled, and a
            # report of "0" next to "She'll -> Sheel" reads like a failure.
            report["apostrophes"] += before.count("'") - body.count("'")
            for key, val in lowered.items():
                if re.search(r"\b" + re.escape(key) + r"\b", before, re.IGNORECASE):
                    if (key, val) not in report["contractions"]:
                        report["contractions"].append((key, val))

        # 5. -ing -> -in
        if ing_to_in:
            before_words = body.split()
            body = shorten_ing(body, ing_exceptions)
            report["ing"] += sum(
                1 for a, b in zip(before_words, body.split()) if a != b
            )

        # 6. Any remaining apostrophe is removed, wherever it sits. This runs
        #    AFTER the table so mapped words are already resolved (she'll ->
        #    sheel) and only leftovers go (rock 'n' roll -> rock nr roll).
        before_apos = body
        body = strip_all_apostrophes(body)
        if body != before_apos:
            report["apostrophes"] += len(before_apos) - len(body)

        # 7. Capitalise the first word of the line. Runs after every word-level
        #    transform so it capitalises the FINAL form (e.g. "sheel", not
        #    "she'll"), and before punctuation stripping so a leading quote
        #    cannot swallow the change.
        if do_capitalize_lines:
            before_cap = body
            body = capitalize_first_word(body)
            if body != before_cap:
                report["capitalized"] += 1

        # 8. Trailing punctuation.
        if do_strip_punctuation:
            body = strip_trailing_punctuation(body)

        rebuilt = (prefix + body).rstrip() if prefix else body.rstrip()
        out_lines.append(rebuilt)

        # Record word-level differences so the UI can highlight what changed.
        if rebuilt != raw_line:
            report["lines_changed"] += 1
        for a, b in zip(raw_line.split(), rebuilt.split()):
            if a != b:
                changed.add((a, b))

    report["word_changes"] = sorted(changed, key=lambda p: p[0].lower())
    return "\n".join(out_lines), report

_LEADING_TAGS = re.compile(r"^((?:\s*\[[^\]]*\]\s*)+)(.*)$")
