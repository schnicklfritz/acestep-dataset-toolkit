"""Tests for the corpus-research tools.

The load-bearing test here is ``test_no_tool_ever_returns_raw_corpus_text``: it
turns the module's central promise (an agent cannot accidentally ingest a big
corpus) into something that fails loudly if a future edit breaks it.
"""
import re

import pytest

from modules.research_tools import (
    MAX_LINE_CHARS,
    _looks_like_prose,
    build_lexicon,
    corpus_stats,
    extract_descriptor_terms,
    extract_grouped_terms,
    extract_section_markers,
    grep_corpus,
    induct_terms,
    load_vocabulary,
    load_vocabulary_groups,
    resolve_artist,
    spec_coverage,
    terms_for_artist,
    unlisted_terms,
    vocab_diff,
)

# A phrase that exists only in the corpus, used to prove nothing leaks it out.
SECRET = "zanzibar-quasar-secret-phrase"

DOC_A = f"""\
Album review: Black Sabbath - Paranoid
The band leans on doom-laden riffs and thunderous drums.
Critics call it doom metal. Ozzy's nasal vocal suits the downtuned guitar.
Raw 1970s analog production throughout. {SECRET} appears here.
"""

DOC_B = """\
Session notes: Paranoid
downtuned guitar and distorted bass dominate; thunderous drums.
Raw 1970s analog production with a nasal vocal delivery.
Some reviewers call it heavy metal, others doom metal.
"""


@pytest.fixture
def corpus(tmp_path):
    """Two documents that share most of their descriptive vocabulary."""
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "review_a.txt").write_text(DOC_A, encoding="utf-8")
    (d / "notes_b.md").write_text(DOC_B, encoding="utf-8")
    # Ignored: not a text extension.
    (d / "cover.png").write_bytes(b"\x89PNG\r\n\x1a\nbinary")
    return str(d)


@pytest.fixture
def vocab(tmp_path):
    """A vocabulary file with terms both present and absent from the corpus."""
    p = tmp_path / "vocabulary.txt"
    p.write_text(
        "# a comment line\n"
        "\n"
        "doom-laden\n"
        "thunderous drums\n"
        "nasal vocal\n"
        "downtuned guitar\n"
        "raw 1970s analog production\n"
        "gated reverb\n"        # never in the corpus
        "yodel\n",              # never in the corpus
        encoding="utf-8",
    )
    return str(p)


# --------------------------------------------------------------------------
# corpus_stats
# --------------------------------------------------------------------------

def test_corpus_stats_counts_only_text_files(corpus):
    out = corpus_stats(corpus)
    assert "Files: 2" in out              # the .png is ignored
    assert "Characters:" in out
    assert ".txt" in out and ".md" in out


def test_corpus_stats_reports_missing_dir(tmp_path):
    out = corpus_stats(str(tmp_path / "nope"))
    assert out.startswith("ERROR:")
    assert "not found" in out


# --------------------------------------------------------------------------
# grep_corpus
# --------------------------------------------------------------------------

def test_grep_corpus_returns_file_and_line_refs(corpus):
    out = grep_corpus(corpus, "thunderous")
    assert "review_a.txt:3:" in out or "review_a.txt:2:" in out
    assert "notes_b.md:" in out


def test_grep_corpus_caps_hits(corpus):
    out = grep_corpus(corpus, ".", max_hits=3)
    # Count only real result lines (file:line: text) -- the header also has a colon.
    hits = [ln for ln in out.splitlines() if re.match(r"^\S+:\d+:", ln)]
    assert len(hits) == 3
    assert "capped at 3" in out


def test_grep_corpus_truncates_long_lines(tmp_path):
    long_line = "x" * 900
    (tmp_path / "big.txt").write_text(long_line, encoding="utf-8")
    out = grep_corpus(str(tmp_path), "x")
    assert "..." in out
    longest = max(len(ln) for ln in out.splitlines())
    assert longest < MAX_LINE_CHARS + 200


def test_grep_corpus_invalid_regex_reports_error(corpus):
    out = grep_corpus(corpus, "([unclosed")
    assert out.startswith("ERROR:")
    assert "invalid regex" in out


def test_grep_corpus_no_match(corpus):
    out = grep_corpus(corpus, "no-such-token-anywhere")
    assert "no matches" in out


# --------------------------------------------------------------------------
# load_vocabulary
# --------------------------------------------------------------------------

def test_load_vocabulary_skips_comments_blanks_and_dedupes(vocab):
    terms = load_vocabulary(vocab)
    assert "doom-laden" in terms
    assert not any(t.startswith("#") for t in terms)
    assert "" not in terms
    assert len(terms) == len({t.lower() for t in terms})


def test_load_vocabulary_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vocabulary(str(tmp_path / "absent.txt"))


# --------------------------------------------------------------------------
# induct_terms
# --------------------------------------------------------------------------

def test_induct_terms_ranks_by_source_count(corpus, vocab):
    out = induct_terms(corpus, vocab, min_sources=2)
    assert "thunderous drums" in out
    assert "downtuned guitar" in out
    # Present in only one source, so below the >= 2 bar.
    assert "doom-laden" not in out
    # Absent from the corpus entirely.
    assert "gated reverb" not in out
    assert "Grounded in >= 2 source(s): 4 term(s)" in out


def test_induct_terms_min_sources_one_includes_single_source(corpus, vocab):
    out = induct_terms(corpus, vocab, min_sources=1)
    assert "doom-laden" in out


def test_induct_terms_reports_missing_vocab(corpus, tmp_path):
    out = induct_terms(corpus, str(tmp_path / "nope.txt"))
    assert out.startswith("ERROR:")


# --------------------------------------------------------------------------
# unlisted_terms
# --------------------------------------------------------------------------

def test_unlisted_terms_finds_phrase_absent_from_vocab(corpus, vocab):
    out = unlisted_terms(corpus, vocab)
    # In both documents, not in the vocabulary.
    assert "doom metal" in out
    # Already covered by the vocabulary, so must not be offered as new.
    assert "thunderous drums" not in out


def test_unlisted_terms_ignores_single_source_phrases(corpus, vocab):
    out = unlisted_terms(corpus, vocab)
    # The secret phrase occurs in one file only -> below the >= 2 sources bar.
    assert "zanzibar" not in out


# --------------------------------------------------------------------------
# vocab_diff / spec_coverage
# --------------------------------------------------------------------------

def test_vocab_diff_splits_shared_and_unique(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("doom-laden\nthunderous drums\n", encoding="utf-8")
    b.write_text("doom-laden\nyodel\n", encoding="utf-8")
    out = vocab_diff(str(a), str(b))
    assert "Shared: 1" in out
    assert "Only A: 1" in out
    assert "Only B: 1" in out
    assert "thunderous drums" in out
    assert "yodel" in out


def test_spec_coverage_reports_unused_terms(corpus, vocab):
    spec = corpus + "/review_a.txt"
    out = spec_coverage(spec, vocab)
    assert "Used:" in out
    assert "gated reverb" in out          # not used by that file


# --------------------------------------------------------------------------
# Token discipline: the promise this module exists to keep
# --------------------------------------------------------------------------

def test_no_tool_returns_raw_corpus_text(corpus, vocab):
    """No tool may hand back document prose unless a regex asked for it.

    ``grep_corpus`` is deliberately excluded -- it returns only lines matching
    an explicit pattern, which is the one sanctioned way to read prose.
    """
    outputs = {
        "corpus_stats": corpus_stats(corpus),
        "induct_terms": induct_terms(corpus, vocab),
        "unlisted_terms": unlisted_terms(corpus, vocab),
        "vocab_diff": vocab_diff(vocab, vocab),
    }
    for name, text in outputs.items():
        assert "appears here" not in text, f"{name} leaked document prose"
        assert "Session notes" not in text, f"{name} leaked a document header"


def test_corpus_stats_returns_no_terms_only_counts(corpus):
    out = corpus_stats(corpus)
    # A term inside the corpus must not surface -- only counts do.
    assert "doom-laden" not in out
    assert "thunderous" not in out


# --------------------------------------------------------------------------
# Lexicon extraction -- the parser must never ingest example prose
# --------------------------------------------------------------------------

MIXED_DOC = """\
# A Descriptor Reference

Some introductory prose that is definitely not a term list.

## Section 1: Terms

### Vocal style
whispered, belted, falsetto, raspy vocal

### Complete examples
**Caption**
> heavy metal, doom-laden, nasal haunting vocals, downtuned heavy riffing,
> reaching a crushing peak and fading into a bleak, atmospheric outro.

**Section tags**
```
[Intro - slow, ominous, heavy riffing]

[Verse 1 - nasal vocal, plaintive, sparse]
What is this that stands before me?
```

## Section 2: Quick reference

| Marker | Combinations |
| :--- | :--- |
| `[Intro]` | sparse, piano; fade in |

**Black Sabbath**
Genre: heavy metal, doom metal, hard rock.
Vocal: nasal, haunting, eerie, wailing.
"""


def test_extract_terms_keeps_term_lists():
    terms = extract_descriptor_terms(MIXED_DOC)
    assert "whispered" in terms
    assert "belted" in terms
    assert "raspy vocal" in terms


def test_extract_terms_skips_example_blockquote_caption():
    terms = extract_descriptor_terms(MIXED_DOC)
    assert not any("crushing peak" in t for t in terms)
    assert not any("haunting vocals" in t for t in terms)


def test_extract_terms_skips_code_fenced_examples():
    terms = extract_descriptor_terms(MIXED_DOC)
    assert "plaintive" not in terms           # lives inside the fence
    assert not any("stands before me" in t for t in terms)


def test_extract_terms_skips_tables_and_bold_labels():
    terms = extract_descriptor_terms(MIXED_DOC)
    assert not any("Combinations" in t for t in terms)
    assert not any("Black Sabbath" in t for t in terms)


def test_extract_terms_reads_labelled_quick_reference_lines():
    terms = extract_descriptor_terms(MIXED_DOC)
    assert "doom metal" in terms              # from "Genre: ..." line
    assert "eerie" in terms                   # from "Vocal: ..." line
    assert "hard rock" in terms


def test_extract_terms_never_contains_sentences():
    terms = extract_descriptor_terms(MIXED_DOC)
    assert not any(_looks_like_prose(t) for t in terms), terms


def test_extract_section_markers_from_backticks_and_examples():
    markers = extract_section_markers(MIXED_DOC)
    assert "[Intro]" in markers
    assert "[Verse 1]" in markers             # tail "- nasal vocal" stripped


def test_extract_section_markers_excludes_lrc_metadata_tags():
    doc = "| `[ti:]` | Title |\n| `[ar:]` | Artist |\n\n[Chorus]\n"
    markers = extract_section_markers(doc)
    assert "[Chorus]" in markers
    assert not any(m.lower().startswith(("[ti", "[ar")) for m in markers)


def test_looks_like_prose_accepts_long_but_valid_descriptors():
    assert not _looks_like_prose("acoustic guitar and bass three-beat pulse")
    assert not _looks_like_prose("sparse stripped-down arrangement")


def test_looks_like_prose_rejects_sentences():
    assert _looks_like_prose("reaching a crushing peak and fading into a bleak outro")
    assert _looks_like_prose("the arrangement is compact and radio-friendly")
    assert _looks_like_prose("voice-steel-and-fiddle at the center is the point")


def test_build_lexicon_writes_both_files(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(MIXED_DOC, encoding="utf-8")
    out = build_lexicon(str(src), str(tmp_path / "out"))
    import os
    assert os.path.isfile(out["vocabulary_path"])
    assert os.path.isfile(out["markers_path"])
    assert "Descriptors:" in out["report"]
    written = open(out["vocabulary_path"], encoding="utf-8").read()
    assert "whispered" in written
    assert "crushing peak" not in written


def test_build_lexicon_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_lexicon(str(tmp_path / "absent.md"))


# --------------------------------------------------------------------------
# Artist grouping -- sections 5-8 are per-artist, 1-4 cross-artist, 13 labelled
# --------------------------------------------------------------------------

GROUPED_DOC = """\
# Reference

## Section 3: Architectural Descriptors

### Vocal style
whispered, belted, raspy vocal

## Section 7: Black Sabbath (pre-Never Say Die)

### Guitar (Tony Iommi)
downtuned guitar, palm-muted riffs, power chords

### Complete examples
**Caption**
> heavy metal, doom-laden, downtuned heavy riffing

**Section tags**
```
[Intro - slow, ominous]
```

## Section 8: AC/DC (Bon Scott years)

### Vocal (Bon Scott)
raspy vocal, sneering, snarling

## Section 13: Quick reference

**Hank Williams Sr.**
Genre: honky-tonk country.
Instruments: steel guitar, fiddle.

## Section 10: Key Rules

1. **Max 3 descriptors per bracket.** More risks the model singing tag names.
8. **Front-load conditioning keywords** in captions (5-12 keywords, max 15).
Locales `en`, `en-US`, `en-GB`.
"""


def test_grouped_terms_split_by_artist():
    groups = extract_grouped_terms(GROUPED_DOC)
    artists = groups["artists"]
    assert "Black Sabbath" in artists             # parenthetical stripped
    assert "AC/DC" in artists
    assert "Hank Williams Sr." in artists
    assert "general" in groups


def test_grouped_example_labels_do_not_become_artists():
    artists = extract_grouped_terms(GROUPED_DOC)["artists"]
    assert "Caption" not in artists
    assert "Section tags" not in artists


def test_grouped_terms_do_not_leak_prose_or_backticks():
    groups = extract_grouped_terms(GROUPED_DOC)
    every = [t for facets in groups["artists"].values()
             for terms in facets.values() for t in terms]
    every += [t for terms in groups["general"].values() for t in terms]
    assert not [t for t in every if _looks_like_prose(t)], every
    assert not [t for t in every if "`" in t or "]" in t], every
    assert "en-US" not in every


def test_grouped_terms_no_empty_facets():
    groups = extract_grouped_terms(GROUPED_DOC)
    for facets in list(groups["artists"].values()) + [groups["general"]]:
        assert not [k for k, v in facets.items() if not v]


def test_resolve_artist_handles_real_dataset_names():
    groups = {"artists": {"Black Sabbath": {}, "The Doors": {}, "AC/DC": {},
                          "Hank Williams Sr.": {}}, "general": {}}
    assert resolve_artist(groups, "sabbath") == "Black Sabbath"
    assert resolve_artist(groups, "Doorsdata") == "The Doors"
    assert resolve_artist(groups, "hank_sr") == "Hank Williams Sr."
    assert resolve_artist(groups, "acdc") == "AC/DC"
    assert resolve_artist(groups, "Nirvana") is None
    assert resolve_artist(groups, "") is None


def test_terms_for_artist_keeps_buckets_separate():
    groups = extract_grouped_terms(GROUPED_DOC)
    artist_facets, general_facets, name = terms_for_artist(groups, "sabbath")
    assert name == "Black Sabbath"
    artist_terms = [t for ts in artist_facets.values() for t in ts]
    assert "downtuned guitar" in artist_terms
    # Another artist's signatures must not appear in the artist bucket.
    assert "sneering" not in artist_terms
    assert "steel guitar" not in artist_terms
    # Cross-artist fundamentals remain available in their own bucket.
    assert "whispered" in [t for ts in general_facets.values() for t in ts]


def test_terms_for_artist_unmatched_is_general_only():
    groups = extract_grouped_terms(GROUPED_DOC)
    artist_facets, general_facets, name = terms_for_artist(groups, "Nonexistent")
    assert name is None
    assert artist_facets == {}
    assert general_facets


def test_build_lexicon_writes_grouped_json(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(GROUPED_DOC, encoding="utf-8")
    out = build_lexicon(str(src), str(tmp_path / "out"))
    import os
    assert os.path.isfile(out["groups_path"])
    loaded = load_vocabulary_groups(out["groups_path"])
    assert "Black Sabbath" in loaded["artists"]
    assert "Artists:" in out["report"]


def test_real_descriptor_doc_extracts_cleanly():
    """Integration guard on the actual reference document."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = os.path.join(root, "docs", "descriptor_reference.md")
    if not os.path.isfile(doc):
        pytest.skip("descriptor reference doc not present")
    from modules.research_tools import load_vocabulary
    terms = extract_descriptor_terms(open(doc, encoding="utf-8").read())
    assert len(terms) > 200
    assert not [t for t in terms if _looks_like_prose(t)]
    # Terms the doc is known to define must be present.
    for expected in ("whispered", "palm-muted riffs", "thunderous drums"):
        assert expected in terms
    # Example-caption prose and lyric lines must be absent.
    assert not any("crushing peak" in t for t in terms)
    assert not any("Riders on the storm" in t for t in terms)
    # And the generated file, if committed, agrees with the parser.
    gen = os.path.join(root, "docs", "vocabulary.txt")
    if os.path.isfile(gen):
        assert load_vocabulary(gen) == terms

    # Section markers must also round-trip, and every one must be well formed --
    # an unbalanced "[Verse 1" (no closing bracket) is the failure mode of
    # splitting a marker name off its "- descriptors" tail.
    markers = extract_section_markers(open(doc, encoding="utf-8").read())
    assert markers, "expected some section markers"
    assert not [m for m in markers if not m.endswith("]")], markers
    assert "[Intro]" in markers
    gen_markers = os.path.join(root, "docs", "section_markers.txt")
    if os.path.isfile(gen_markers):
        assert load_vocabulary(gen_markers) == markers

    # Grouped vocabulary: exactly the four artists, and the generic/rap terms
    # that used to be hardcoded into the tag creator must be absent entirely.
    groups = extract_grouped_terms(open(doc, encoding="utf-8").read())
    assert set(groups["artists"]) == {
        "Hank Williams Sr.", "The Doors", "Black Sabbath", "AC/DC"
    }
    every = [t.lower() for facets in groups["artists"].values()
             for terms in facets.values() for t in terms]
    every += [t.lower() for terms in groups["general"].values() for t in terms]
    for unwanted in ("mumble rap", "chopper rap", "trap flow", "auto-tune"):
        assert unwanted not in every

    # Artist scoping must exclude other artists' signatures.
    sabbath_terms, _general, name = terms_for_artist(groups, "sabbath")
    sabbath = [t.lower() for ts in sabbath_terms.values() for t in ts]
    assert name == "Black Sabbath"
    assert "downtuned guitar" in sabbath
    assert "steel guitar" not in sabbath
    assert "vox continental organ" not in sabbath