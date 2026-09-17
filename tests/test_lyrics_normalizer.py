"""Lyrics normalization: contractions, apostrophes, case, tags, -ing.

This module has the subtlest rules in the app and I mis-stated them twice while
building it, so the behaviour is pinned down here explicitly.
"""
import pytest

from modules.lyrics_normalizer import (
    DEFAULT_CONTRACTIONS,
    DEFAULT_ING_EXCEPTIONS,
    apply_contractions,
    capitalize_tags,
    normalize_lyrics,
    shorten_ing,
    strip_all_apostrophes,
    strip_trailing_punctuation,
)

TABLE = {"she'll": "sheel", "he'll": "heel", "can't": "cant"}


class TestCaseHandling:
    """The table maps SPELLING. Case follows the source; it is never invented."""

    @pytest.mark.parametrize("src,want", [
        ("she'll", "sheel"),
        ("She'll", "Sheel"),
        ("SHE'LL", "SHEEL"),
    ])
    def test_case_is_carried_from_source(self, src, want):
        assert apply_contractions(src, TABLE) == want

    @pytest.mark.parametrize("word", ["yeah", "Yeah", "YEAH", "yEAh"])
    def test_words_not_in_the_table_are_untouched(self, word):
        assert apply_contractions(word, TABLE) == word

    def test_matches_whole_words_only(self):
        # "cant" inside "cannot"/"cantata" must not be rewritten oddly
        assert apply_contractions("cantata", TABLE) == "cantata"

    def test_longest_key_wins(self):
        table = {"ll": "X", "she'll": "sheel"}
        assert apply_contractions("she'll", table) == "sheel"


class TestApostrophes:
    @pytest.mark.parametrize("src,want", [
        ("rock 'n' roll", "rock n roll"),
        ("the sun's rays", "the suns rays"),
        ("'Save me!'", "Save me!"),
        ("don't", "dont"),
    ])
    def test_all_apostrophes_removed(self, src, want):
        assert strip_all_apostrophes(src) == want

    @pytest.mark.parametrize("curly", ["\u2019", "\u2018"])
    def test_curly_quotes_removed(self, curly):
        assert "'" not in strip_all_apostrophes(f"don{curly}t")

    def test_table_runs_before_the_blanket_strip(self):
        # she'll must become sheel, not shell
        out, _ = normalize_lyrics("She'll", contractions=TABLE)
        assert "Sheel" in out
        assert "Shell" not in out

    def test_no_apostrophe_survives(self):
        out, _ = normalize_lyrics(
            "rock 'n' roll, sun's rays, can't stop", contractions=TABLE
        )
        assert "'" not in out


class TestTagCapitalization:
    """A tag is everything before ' -'. A joined hyphen still title-cases."""

    @pytest.mark.parametrize("src,want", [
        ("[verse 1]", "[Verse 1]"),
        ("[guitar solo]", "[Guitar Solo]"),
        ("[GUITAR SOLO]", "[Guitar Solo]"),
        ("[CHORUS]", "[Chorus]"),
        ("[pre-chorus]", "[Pre-Chorus]"),
        ("[Verse 1]", "[Verse 1]"),
        ("[fade out]", "[Fade Out]"),
    ])
    def test_title_cases_the_tag(self, src, want):
        assert capitalize_tags(src) == want

    @pytest.mark.parametrize("src,want", [
        ("[verse - quiet]", "[Verse - quiet]"),
        ("[chorus - loud - big]", "[Chorus - loud - big]"),
        ("[verse 1 - spoken]", "[Verse 1 - spoken]"),
        ("[bridge - whispered]", "[Bridge - whispered]"),
    ])
    def test_spaced_hyphen_starts_a_lowercase_modifier(self, src, want):
        assert capitalize_tags(src) == want

    def test_handles_multiple_tags_on_a_line(self):
        out = capitalize_tags("[verse][chorus]")
        assert out == "[Verse][Chorus]"


class TestIngShortening:
    def test_shortens_syllabic_ing(self):
        assert shorten_ing("running", set()) == "runnin"
        assert shorten_ing("dancing", set()) == "dancin"

    @pytest.mark.parametrize("word", ["something", "ring", "thing", "king", "morning"])
    def test_never_shortens_exceptions(self, word):
        assert shorten_ing(word, DEFAULT_ING_EXCEPTIONS) == word

    def test_short_words_are_left_alone(self):
        assert shorten_ing("sing", set()) == "sing"


class TestTrailingPunctuation:
    @pytest.mark.parametrize("src,want", [
        ("Save me!", "Save me"),
        ("Save me,", "Save me"),
        ("Save me.", "Save me"),
        ('"Save me!"', '"Save me"'),
        ("(Save me!)", "(Save me)"),
    ])
    def test_strips_punctuation_but_keeps_quote_and_paren(self, src, want):
        assert strip_trailing_punctuation(src) == want

    def test_plain_text_is_unchanged(self):
        assert strip_trailing_punctuation("no punctuation here") == "no punctuation here"


class TestNormalizeLyricsReport:
    def test_report_tracks_counts(self):
        out, rep = normalize_lyrics(
            "[verse]\nShe'll be running, can't you see!",
            contractions=TABLE, ing_to_in=True,
        )
        assert rep["lines_changed"] >= 1
        assert rep["apostrophes"] >= 2
        assert rep["tags"] >= 1
        assert rep["ing"] >= 1

    def test_report_lists_per_word_changes(self):
        _, rep = normalize_lyrics("She'll run, can't stop!", contractions=TABLE)
        pairs = dict(rep["word_changes"])
        assert pairs["She'll"] == "Sheel"
        assert pairs["can't"] == "cant"

    def test_separator_markers_are_protected(self):
        # The all-lyrics block depends on these surviving a tidy.
        text = "---- song.mp3 ----\nShe'll run"
        out, _ = normalize_lyrics(text, contractions=TABLE)
        assert out.startswith("---- song.mp3 ----")

    def test_normalize_is_stable_on_second_pass(self):
        src = "[verse]\nShe'll be running, can't you see!"
        once, _ = normalize_lyrics(src, contractions=TABLE, ing_to_in=True)
        twice, _ = normalize_lyrics(once, contractions=TABLE, ing_to_in=True)
        assert once == twice

    def test_default_contraction_table_is_populated(self):
        for key in ("she'll", "he'll", "i'll", "can't", "don't"):
            assert key in DEFAULT_CONTRACTIONS
