"""Headless corpus-research tools (no Qt dependency).

Operate on a directory of plain-text source documents (album reviews, session
notes, interviews, fan descriptions, liner notes) plus a descriptor vocabulary
file. Used by the research MCP server, the ``scripts/build_lexicon.py`` CLI, and
the test suite.

TOKEN DISCIPLINE — this is the whole point of the module:

  * No function returns raw document text. ``corpus_stats`` returns counts only.
  * ``grep_corpus`` returns only the lines an explicit regex matched, capped,
    and each matched line is itself truncated.
  * Every list-returning function takes an explicit cap.

An agent therefore *cannot* accidentally ingest a 500 KB corpus, because no
entry point serves it. That is enforced here rather than by asking politely.

Matching is deterministic: case-insensitive, non-alphanumeric-bounded literal
terms. No embeddings, no LLM, no network, fully reproducible.
"""
import json
import os
import re
from collections import Counter, defaultdict

# Document extensions worth reading as plain text.
TEXT_EXTS = (".txt", ".md", ".text", ".rst", ".csv", ".tsv", ".log")

# Directory names never descended into. A corpus root is supposed to be prose
# about an artist, but pointing one at a data root silently ingested an entire
# virtualenv: 236 "files" / 1.9 MB of numpy test .csv whose terms then showed up
# as inducted vocabulary. That is a WRONG ANSWER, not an error, so the guard
# belongs here rather than in a docstring asking nicely.
# Matched case-insensitively against each path segment, so a nested
# ``venv/lib/python3.x/site-packages`` tree is pruned at its top.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".idea", ".vscode", "node_modules", "venv", ".venv", "env",
    ".env", "site-packages", "dist-packages", "bower_components",
}

# Every list tool is capped by construction. Callers may lower these, not raise
# them silently -- the cap is part of the contract.
DEFAULT_MAX_HITS = 50
DEFAULT_TOP_TERMS = 40
DEFAULT_MIN_SOURCES = 2
DEFAULT_NGRAM_N = 30
DEFAULT_MAX_FILES = 20

# A matched line is truncated to this many characters so one minified/huge line
# cannot dump kilobytes into a context window.
MAX_LINE_CHARS = 160

# A vocabulary "term" longer than this is almost certainly a leaked sentence
# from an example caption rather than a real descriptor. The lexicon builder
# uses it to flag parsing mistakes.
MAX_TERM_LEN = 60

# Words that should not sit at an n-gram edge when inducting unlisted terms.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "were", "will", "with", "you",
}


def _iter_corpus_files(corpus_dir):
    """Yield sorted plain-text file paths under ``corpus_dir`` (recursive).

    Raises FileNotFoundError so the caller can report a clear message instead of
    silently returning "no results" for a mistyped path.
    """
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    found = []
    for root, dirs, files in os.walk(corpus_dir):
        # Prune in place so os.walk never descends -- see SKIP_DIRS.
        dirs[:] = [d for d in dirs
                   if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in TEXT_EXTS:
                found.append(os.path.join(root, name))
    return sorted(found)


def _read_text(path):
    """Read a text file defensively -- a research corpus is full of odd encodings."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _rel(path, corpus_dir):
    """Path relative to the corpus root, for stable, short output."""
    try:
        return os.path.relpath(path, corpus_dir)
    except ValueError:  # different drive on Windows
        return os.path.basename(path)


def corpus_stats(corpus_dir):
    """Counts only: files, total characters, extension histogram, biggest files.

    Deliberately returns no document content whatsoever, so an agent can confirm
    "the corpus arrived" for a few hundred tokens regardless of corpus size.
    """
    try:
        files = _iter_corpus_files(corpus_dir)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"
    if not files:
        return f"(no text files under {corpus_dir}; looked for {', '.join(TEXT_EXTS)})"

    exts = Counter()
    total_chars = 0
    sized = []
    for path in files:
        text = _read_text(path)
        chars = len(text)
        total_chars += chars
        exts[os.path.splitext(path)[1].lower() or "(none)"] += 1
        sized.append((chars, path))

    sized.sort(reverse=True)
    lines = [
        f"Corpus: {corpus_dir}",
        f"Files: {len(files)} | Characters: {total_chars:,}",
        "By extension: " + ", ".join(f"{e} ({c})" for e, c in exts.most_common()),
    ]
    shown = sized[:DEFAULT_MAX_FILES]
    lines.append("Largest files:")
    lines.extend(f"  {c:>9,} chars  {_rel(p, corpus_dir)}" for c, p in shown)
    if len(sized) > len(shown):
        lines.append(f"  ... and {len(sized) - len(shown)} more file(s)")
    return "\n".join(lines)


def grep_corpus(corpus_dir, pattern, max_hits=DEFAULT_MAX_HITS):
    """Return matching LINES as ``file:line: text``, capped at ``max_hits``.

    This is the only way to look at corpus prose, and it can only ever return
    lines the caller explicitly asked for with a regex. Each line is truncated
    to MAX_LINE_CHARS.
    """
    try:
        files = _iter_corpus_files(corpus_dir)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"ERROR: invalid regex {pattern!r}: {exc}"

    hits = []
    truncated = False
    for path in files:
        if len(hits) >= max_hits:
            truncated = True
            break
        for lineno, line in enumerate(_read_text(path).splitlines(), start=1):
            if rx.search(line):
                text = line.strip().replace("\t", " ")
                if len(text) > MAX_LINE_CHARS:
                    text = text[:MAX_LINE_CHARS] + "..."
                hits.append(f"{_rel(path, corpus_dir)}:{lineno}: {text}")
                if len(hits) >= max_hits:
                    truncated = True
                    break

    if not hits:
        return f"(no matches for {pattern!r} in {len(files)} file(s))"
    out = [f"{len(hits)} match(es) for {pattern!r}:"]
    out.extend(hits)
    if truncated:
        out.append(f"... capped at {max_hits} hits; narrow the pattern for more.")
    return "\n".join(out)


def load_vocabulary(path):
    """Read a one-term-per-line vocabulary file.

    Blank lines and ``#`` comments are ignored; duplicates are removed
    case-insensitively while keeping the first spelling seen.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"vocabulary file not found: {path}")
    terms = []
    seen = set()
    for raw in _read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(line)
    return terms


def _term_pattern(term):
    """Non-alphanumeric-bounded, case-insensitive literal match.

    Lookarounds rather than ``\\b`` so terms containing punctuation
    (``palm-muted riffs``, ``808 sub bass``, ``[Energy: High]``) behave
    predictably.
    """
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE
    )


def _count_sources(corpus_dir, terms):
    """Map each term -> number of distinct FILES containing it.

    Source count, not raw frequency, is the ranking signal: a term repeated
    twenty times in one file is one person's idiolect, while a term appearing
    across nine files is community vocabulary. Returns (counts, texts).
    """
    files = _iter_corpus_files(corpus_dir)
    texts = [(p, _read_text(p).lower()) for p in files]
    counts = defaultdict(int)
    for term in terms:
        rx = _term_pattern(term)
        for _path, text in texts:
            if rx.search(text):
                counts[term] += 1
    return counts, texts


def induct_terms(corpus_dir, vocab_path, min_sources=DEFAULT_MIN_SOURCES,
                 top=DEFAULT_TOP_TERMS):
    """Rank the KNOWN vocabulary by how many corpus sources actually use it.

    The core tool: which descriptors are grounded in real descriptions of this
    music, and which are aspirational.
    """
    try:
        files = _iter_corpus_files(corpus_dir)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"
    try:
        terms = load_vocabulary(vocab_path)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"

    counts, _texts = _count_sources(corpus_dir, terms)
    grounded = [(t, c) for t, c in counts.items() if c >= min_sources]
    # Rank by source count desc, then case-insensitively by term for stability.
    grounded.sort(key=lambda tc: (-tc[1], tc[0].lower()))

    out = [
        f"Vocabulary: {os.path.basename(vocab_path)} ({len(terms)} terms)",
        f"Corpus: {len(files)} file(s)",
        f"Grounded in >= {min_sources} source(s): {len(grounded)} term(s)",
    ]
    if not grounded:
        out.append("(none -- lower min_sources or add corpus material)")
        return "\n".join(out)
    out.append("")
    out.extend(f"{c:>4}  {t}" for t, c in grounded[:top])
    if len(grounded) > top:
        out.append(f"... and {len(grounded) - top} more (raise `top`).")
    return "\n".join(out)


def unlisted_terms(corpus_dir, vocab_path, n=DEFAULT_NGRAM_N):
    """Find frequent 1-3 word phrases in the corpus that the vocabulary LACKS.

    The sleeper tool: the most valuable output is usually community vocabulary
    the curated list does not contain, so the list grows from evidence instead
    of guesswork. Phrases are ranked by distinct source count.
    """
    try:
        files = _iter_corpus_files(corpus_dir)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"
    try:
        terms = load_vocabulary(vocab_path)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"

    known = "\n".join(terms).lower()
    sources = defaultdict(set)
    for path in files:
        tokens = re.findall(r"[a-z0-9][a-z0-9'-]*", _read_text(path).lower())
        for size in (1, 2, 3):
            for i in range(len(tokens) - size + 1):
                gram = tokens[i:i + size]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                phrase = " ".join(gram)
                if phrase in known:          # already covered by the vocabulary
                    continue
                sources[phrase].add(_rel(path, corpus_dir))

    ranked = [(p, len(s)) for p, s in sources.items() if len(s) >= 2]
    ranked.sort(key=lambda ps: (-ps[1], ps[0]))
    out = [
        f"Phrases in corpus but NOT in {os.path.basename(vocab_path)}:",
        f"Corpus: {len(files)} file(s) | showing: {min(n, len(ranked))}",
        "",
    ]
    if not ranked:
        out.append("(none -- the vocabulary already covers the corpus)")
        return "\n".join(out)
    out.extend(f"{c:>4}  {p}" for p, c in ranked[:n])
    return "\n".join(out)


def vocab_diff(a_path, b_path):
    """Compare two vocabulary files: shared, only-in-A, only-in-B."""
    try:
        a = load_vocabulary(a_path)
        b = load_vocabulary(b_path)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"

    a_key = {t.lower(): t for t in a}
    b_key = {t.lower(): t for t in b}
    shared = sorted(set(a_key) & set(b_key))
    only_a = sorted(set(a_key) - set(b_key))
    only_b = sorted(set(b_key) - set(a_key))

    out = [
        f"A = {os.path.basename(a_path)} ({len(a)} terms)",
        f"B = {os.path.basename(b_path)} ({len(b)} terms)",
        f"Shared: {len(shared)} | Only A: {len(only_a)} | Only B: {len(only_b)}",
    ]
    if only_a:
        out.append("")
        out.append(f"Only in {os.path.basename(a_path)}:")
        out.extend(f"  {a_key[k]}" for k in only_a[:DEFAULT_TOP_TERMS])
        if len(only_a) > DEFAULT_TOP_TERMS:
            out.append(f"  ... and {len(only_a) - DEFAULT_TOP_TERMS} more")
    if only_b:
        out.append("")
        out.append(f"Only in {os.path.basename(b_path)}:")
        out.extend(f"  {b_key[k]}" for k in only_b[:DEFAULT_TOP_TERMS])
        if len(only_b) > DEFAULT_TOP_TERMS:
            out.append(f"  ... and {len(only_b) - DEFAULT_TOP_TERMS} more")
    return "\n".join(out)


def spec_coverage(spec_path, vocab_path):
    """Which vocabulary terms a caption spec actually uses, and which it misses.

    Reveals gaps in the spec: descriptors that exist in the vocabulary but are
    never used by the examples, and (via unlisted_terms) the reverse.
    """
    if not os.path.isfile(spec_path):
        return f"ERROR: spec file not found: {spec_path}"
    try:
        terms = load_vocabulary(vocab_path)
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"

    text = _read_text(spec_path).lower()
    used, unused = [], []
    for term in terms:
        (used if _term_pattern(term).search(text) else unused).append(term)

    total = len(terms) or 1
    out = [
        f"Spec: {os.path.basename(spec_path)}",
        f"Vocabulary: {os.path.basename(vocab_path)} ({len(terms)} terms)",
        f"Used: {len(used)}/{len(terms)} ({100 * len(used) // total}%)",
    ]
    if unused:
        out.append("")
        out.append("NOT used by the spec:")
        out.extend(f"  {t}" for t in unused[:DEFAULT_TOP_TERMS])
        if len(unused) > DEFAULT_TOP_TERMS:
            out.append(f"  ... and {len(unused) - DEFAULT_TOP_TERMS} more")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Lexicon building: turn the human-authored descriptor doc into a clean,
# one-term-per-line vocabulary the tools above can consume.
#
# This is the part that must not go wrong. The reference doc mixes vocabulary
# lists with *example captions* and *lyric snippets*; naively splitting it on
# commas would ingest half a sentence as a "term". The rules below therefore
# only harvest terms from `###` term-list blocks (and the labelled
# ``Genre: ...`` form), skipping blockquotes, fenced code blocks and tables.
# --------------------------------------------------------------------------

# `###` headings whose body is prose or examples, never a term list.
NON_TERM_BLOCKS = {
    "complete examples",
    "architectural descriptors",
    "sources",
    "country tips",
}

# Section 1 documents LRC/ID3 *file metadata* tags ([ti:], [ar:], [offset:]) which
# are bracketed like markers but are not structural markers. Markers drive lyric
# structure, so these are excluded from the marker lexicon.
_METADATA_TAG = re.compile(r"^\[(?:ti|ar|al|by|au|offset|length|re|ve|#):", re.I)


def _looks_like_prose(term):
    """Heuristic: is this extracted string a leaked sentence rather than a term?

    The doc's own long-but-valid descriptors ("acoustic guitar and bass
    three-beat pulse") must pass, while example-caption prose must fail, so the
    bar is set at real sentence length and finite-verb shape.
    """
    if len(term) > 45:
        return True
    if len(term.split()) > 6:
        return True
    return any(f" {verb} " in term.lower() for verb in ("is", "are", "was", "were", "be"))


# Lines that start a new construct rather than continuing a term list.
_SKIP_PREFIXES = ("|", ">", "**", "*", "-", "+", "`", "1.", "2.", "3.", "4.")


def _add_terms(out, text):
    """Split one comma-list line into clean terms, appending to ``out``.

    Deduplicates against ``out`` itself rather than a shared set: a caller that
    wants per-facet lists gets per-facet dedupe, while the flat caller (one list
    for the whole doc) still gets one entry per term.
    """
    existing = {t.lower() for t in out}
    for piece in text.split(","):
        piece = piece.strip().strip("`*_ ").rstrip(".").strip()
        if not piece:
            continue
        # A real descriptor is short, has few words and no markdown/brackets.
        if len(piece) > MAX_TERM_LEN:
            continue
        if "**" in piece or "[" in piece or "]" in piece or "`" in piece:
            continue
        if len(piece.split()) > 6:
            continue
        key = piece.lower()
        if key in existing:
            continue
        existing.add(key)
        out.append(piece)


def extract_descriptor_terms(text):
    """Harvest descriptor terms from a descriptor-reference document."""
    terms = []
    block = None          # current `###` heading, or None
    in_fence = False

    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("```"):
            in_fence = not in_fence
            block = None
            continue
        if in_fence:
            continue                      # code fences hold examples/lyrics
        if line.startswith("### "):
            block = line[4:].strip()
            continue
        if line.startswith("#") or not line:
            block = None
            continue                      # heading or blank ends a block
        if block is None:
            # The quick-reference sections label their lists instead of using
            # `###` headings, e.g. "Genre: honky-tonk country, country blues."
            match = re.match(r"^([A-Z][A-Za-z ]{2,20}):\s*(.+)$", line)
            if match and not match.group(2).startswith(("**", "`", ">")):
                _add_terms(terms, match.group(2))
            continue
        if block.lower() in NON_TERM_BLOCKS:
            continue
        if line.startswith(_SKIP_PREFIXES):
            continue
        _add_terms(terms, line)

    return terms


def extract_section_markers(text):
    """Harvest ``[Marker]`` names from anywhere in the doc.

    Markers are a different namespace from descriptors. They appear both
    backticked inline (``[Intro]`` ``[Verse]``) and as the opening token of
    example lines inside code fences (``[Chorus - anthemic, full band]``).
    """
    markers = []
    seen = set()
    for match in re.finditer(r"`(\[[^\]`]+\])`", text):
        marker = match.group(1)
        if _METADATA_TAG.match(marker):
            continue
        if marker.lower() not in seen:
            seen.add(marker.lower())
            markers.append(marker)
    for match in re.finditer(r"^\s*(\[[^\]\n]+\])", text, re.MULTILINE):
        # Keep only the marker name, dropping any "- descriptors" tail. The
        # split consumes the closing bracket along with the tail, so restore it.
        name = re.split(r"\s+[-–—]\s+", match.group(1), maxsplit=1)[0].strip()
        if not name.endswith("]"):
            name += "]"
        if _METADATA_TAG.match(name):
            continue
        if name.lower() not in seen:
            seen.add(name.lower())
            markers.append(name)
    return markers


# --------------------------------------------------------------------------
# Artist-scoped grouping.
#
# Sections 5-8 of the reference doc are artist-scoped; sections 1-4 hold the
# cross-artist vocabulary; section 13 repeats per-artist lines in a labelled
# ("Genre: ...") form. Grouping is what lets the tag creator offer a Black
# Sabbath track doom/downtuned-guitar terms WITHOUT also offering it steel
# guitar or rap styles -- a narrow, correct choice set.
# --------------------------------------------------------------------------

# `## Section 5..8: <Artist>` are the artist-scoped sections.
_ARTIST_SECTION_RANGE = (5, 8)

# Sections that are not "general" despite sitting outside the artist range.
# Section 4 is the country vocabulary; this project's only country artist is
# Hank Williams Sr., so it scopes to him rather than leaking pedal steel and
# banjo into every other artist's prompt.
_SECTION_ARTIST_OVERRIDES = {4: "Hank Williams Sr."}
_SECTION_HEADING = re.compile(r"^##\s+Section\s+(\d+)\s*:")
_ARTIST_LABEL = re.compile(r"^\*\*(.+?)\*\*\s*$")
_LABELLED_LINE = re.compile(r"^([A-Z][A-Za-z ]{2,20}):\s*(.+)$")

GENERAL_GROUP = "general"


def clean_artist_name(name):
    """'Black Sabbath (pre-Never Say Die)' -> 'Black Sabbath'."""
    return re.sub(r"\s*\(.*?\)\s*$", "", name).strip()


def extract_grouped_terms(text):
    """Harvest descriptors grouped by artist, then by facet.

    Returns ``{"general": {facet: [terms]}, "artists": {artist: {facet: [...]}}}``.
    ``general`` is the cross-artist vocabulary (sections 1-4); each artist holds
    only its own descriptors.
    """
    general = {}
    artists = {}
    facets = None            # the dict currently being filled
    block = None             # current `###` heading
    in_fence = False

    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("```"):
            in_fence = not in_fence
            continue                      # the heading persists across a fence
        if in_fence:
            continue

        heading = _SECTION_HEADING.match(line)
        if heading:
            number = int(heading.group(1))
            low, high = _ARTIST_SECTION_RANGE
            override = _SECTION_ARTIST_OVERRIDES.get(number)
            if override:
                artists.setdefault(override, {})
                facets = artists[override]
            elif low <= number <= high:
                title = line.split(":", 1)[1].strip()
                artist = clean_artist_name(title)
                artists.setdefault(artist, {})
                facets = artists[artist]
            else:
                facets = general
            block = None
            continue

        if line.startswith("### "):
            block = line[4:].strip()
            continue
        if line.startswith("#"):
            block = None
            continue
        if not line:
            # A blank line does NOT end the block: "### Complete examples"
            # spans blank lines, and clearing here lets its bold labels through
            # as phantom artists.
            continue
        if block is not None and block.lower() in NON_TERM_BLOCKS:
            continue

        # Section 13 form: a bold artist label scopes the labelled lines below.
        label = _ARTIST_LABEL.match(line)
        if label:
            artist = clean_artist_name(label.group(1))
            artists.setdefault(artist, {})
            facets = artists[artist]
            block = None
            continue

        if facets is None:
            continue
        if line.startswith(_SKIP_PREFIXES) or re.match(r"^\d+[.)]\s", line):
            continue

        labelled = _LABELLED_LINE.match(line)
        if labelled and not labelled.group(2).startswith(("**", "`", ">")):
            facet, body = labelled.group(1).strip(), labelled.group(2)
        elif block is not None:
            facet, body = block, line
        else:
            # Outside any facet block, only a labelled line is a term list --
            # otherwise section 10's prose rules get comma-split into "terms".
            continue
        # Only register a facet that actually receives a term, so a prose line
        # that parses to nothing does not leave an empty group behind.
        bucket = facets.get(facet)
        if bucket is None:
            bucket = []
            _add_terms(bucket, body)
            if bucket:
                facets[facet] = bucket
        else:
            _add_terms(bucket, body)

    return {GENERAL_GROUP: general, "artists": artists}


def resolve_artist(groups, artist):
    """Match a loose dataset/artist name onto a vocabulary artist, or ``None``.

    Word-based rather than string-prefix based, because real dataset folders are
    not named after the artist: ``Doorsdata`` -> "The Doors", ``hank_sr`` ->
    "Hank Williams Sr.", ``sabbath`` -> "Black Sabbath".
    """
    if not artist:
        return None
    key = re.sub(r"[^a-z0-9]", "", artist.lower())
    if not key:
        return None
    fallback = None
    for name in groups.get("artists", {}):
        if re.sub(r"[^a-z0-9]", "", name.lower()) == key:
            return name                      # exact after normalisation
        # Significant words only; "the"/"sr" carry no identity.
        for word in re.findall(r"[a-z0-9]+", name.lower()):
            if len(word) >= 4 and word in key:
                fallback = fallback or name
    return fallback


def terms_for_artist(groups, artist):
    """Return ``(artist_facets, general_facets, resolved_name)``.

    The two buckets stay separate because they mean different things: the
    artist's own facets are the narrow, preferred choice set, while the general
    facets are cross-artist fundamentals (vocal timbre, energy, production) that
    apply to any track. Keeping them apart lets a prompt say "prefer these"
    without hiding the fundamentals.

    An empty or unmatched artist yields an empty artist bucket and the general
    vocabulary only -- the safe fallback, so one artist's signature terms are
    never offered for another.
    """
    selected = resolve_artist(groups, artist)
    artists = groups.get("artists", {})
    return (
        dict(artists.get(selected, {})) if selected else {},
        dict(groups.get(GENERAL_GROUP, {})),
        selected,
    )


def build_lexicon(source_path, out_dir=None):
    """Extract vocabulary + markers from ``source_path`` and write them out.

    Returns a dict: {terms, markers, vocabulary_path, markers_path, report}.
    Raises FileNotFoundError if the source doc is missing.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"descriptor doc not found: {source_path}")
    text = _read_text(source_path)
    terms = extract_descriptor_terms(text)
    markers = extract_section_markers(text)
    groups = extract_grouped_terms(text)

    out_dir = out_dir or os.path.dirname(os.path.abspath(source_path))
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, "vocabulary.txt")
    markers_path = os.path.join(out_dir, "section_markers.txt")
    groups_path = os.path.join(out_dir, "vocabulary.json")

    stamp = (
        f"# Extracted from {os.path.basename(source_path)} -- do not hand-edit.\n"
        "# Regenerate with: .venv/bin/python scripts/build_lexicon.py\n"
    )
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(stamp + "# Descriptor vocabulary: one term per line.\n")
        f.write("\n".join(terms) + "\n")
    with open(markers_path, "w", encoding="utf-8") as f:
        f.write(stamp + "# Section markers: one per line.\n")
        f.write("\n".join(markers) + "\n")
    grouped = dict(groups)
    grouped["_source"] = os.path.basename(source_path)
    grouped["_note"] = "Generated by scripts/build_lexicon.py -- do not hand-edit."
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, sort_keys=False)
        f.write("\n")

    # Suspect terms are the guard against example prose leaking in.
    suspects = [t for t in terms if _looks_like_prose(t)]
    report = [
        f"Source: {source_path}",
        f"Descriptors: {len(terms)} -> {vocab_path}",
        f"Section markers: {len(markers)} -> {markers_path}",
        f"Artists: {len(groups['artists'])} -> {groups_path}",
    ]
    for artist, facets in groups["artists"].items():
        count = len({t.lower() for ts in facets.values() for t in ts})
        report.append(f"  {artist}: {count} term(s)")
    if suspects:
        report.append(f"Review {len(suspects)} suspiciously long term(s):")
        report.extend(f"  {t}" for t in suspects[:10])
    return {
        "terms": terms,
        "markers": markers,
        "groups": groups,
        "vocabulary_path": vocab_path,
        "markers_path": markers_path,
        "groups_path": groups_path,
        "report": "\n".join(report),
    }


def load_vocabulary_groups(path):
    """Read the artist-grouped vocabulary written by ``build_lexicon``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"grouped vocabulary not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_source", None)
    data.pop("_note", None)
    return data
