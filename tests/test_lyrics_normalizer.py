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


class TestLineCapitalization:
    """Capitalise the first word of each lyric line; leave the rest alone."""

    @pytest.mark.parametrize("src,want", [
        ("rising up from the ashes", "Rising up from the ashes"),
        ("we're on fire", "We're on fire"),
        ("just one word", "Just one word"),
    ])
    def test_capitalizes_the_first_letter(self, src, want):
        from modules.lyrics_normalizer import capitalize_first_word
        assert capitalize_first_word(src) == want

    @pytest.mark.parametrize("src", [
        "Already Capitalised",
        "ALL CAPS LINE STAYS LOUD",
        "iPhone and eBay",          # intentional internal casing preserved
    ])
    def test_leaves_rest_of_the_line_untouched(self, src):
        from modules.lyrics_normalizer import capitalize_first_word
        assert capitalize_first_word(src) == src

    @pytest.mark.parametrize("src", ["", "   ", "123 456", "!!!", "'"])
    def test_ignores_lines_with_no_leading_letter(self, src):
        from modules.lyrics_normalizer import capitalize_first_word
        assert capitalize_first_word(src) == src

    def test_skips_leading_whitespace(self):
        from modules.lyrics_normalizer import capitalize_first_word
        assert capitalize_first_word("   rising up") == "   Rising up"

    def test_applied_through_normalize_lyrics(self):
        out, rep = normalize_lyrics(
            "rising up from the ashes\nwe're on fire",
            contractions={"we're": "weer"},
        )
        assert "Rising up" in out
        assert "Weer on fire" in out
        assert rep["capitalized"] >= 2

    def test_capitalizes_the_final_form_not_the_source(self):
        # she'll -> sheel must come out as "Sheel", not "Sheel" from "She'll"
        out, _ = normalize_lyrics("she'll be there", contractions=TABLE)
        assert out.startswith("Sheel")

    def test_tag_lines_are_not_treated_as_lyric_lines(self):
        # [verse] is handled by the tag rule, not the line rule
        out, _ = normalize_lyrics("[verse]\nrising up")
        assert "[Verse]" in out
        assert "Rising up" in out

    def test_can_be_switched_off(self):
        sentinel = object()
        out, _ = normalize_lyrics("rising up", contractions=TABLE,
                                  do_capitalize_lines=False)
        assert out.startswith("rising")

    def test_all_caps_line_survives_a_tidy(self):
        out, _ = normalize_lyrics("YEAH, dont stop")
        assert "YEAH" in out


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


class TestQuoteStripping:
    """Every double quote goes; apostrophes are a different character."""

    @pytest.mark.parametrize("src,want", [
        ('"hello', "hello"),
        ('She said "stop" now', "She said stop now"),
        ("\u201cHi\u201d", "Hi"),
        ("\u00abHi\u00bb", "Hi"),
        ("no quotes here", "no quotes here"),
    ])
    def test_quotes_are_removed(self, src, want):
        from modules.lyrics_normalizer import strip_quotes
        assert strip_quotes(src) == want

    def test_apostrophes_are_left_to_the_other_rule(self):
        from modules.lyrics_normalizer import strip_quotes
        assert strip_quotes("don't") == "don't"

    def test_it_runs_before_capitalisation(self):
        """The whole reason this pass is first: a line opening with a quote
        starts with punctuation, so ``capitalize_first_word`` skips it. Strip
        later and ``"hello`` would stay lowercase."""
        out, _ = normalize_lyrics('"hello there', do_strip_punctuation=False)
        assert out == "Hello there"

    def test_it_can_be_switched_off(self):
        out, _ = normalize_lyrics('"hello', do_strip_quotes=False)
        assert '"' in out

    def test_the_report_counts_the_characters(self):
        _, rep = normalize_lyrics('"a" and "b"', do_strip_punctuation=False)
        assert rep["quotes"] == 4


class TestTagModifierTrimming:
    """``[Chorus - Raspy Vocals]`` -> ``[Chorus]``; the tag itself survives."""

    @pytest.mark.parametrize("src,want", [
        ("[Chorus]", "[Chorus]"),
        ("[Chorus - Raspy Vocals]", "[Chorus]"),
        ("[Verse 1 - raspy vocal, low energy]", "[Verse 1]"),
        ("[Chorus - loud - big]", "[Chorus]"),
        ("[Guitar Solo - intense]", "[Guitar Solo]"),
        # A JOINED hyphen is not a separator: "Pre-Chorus" is one tag.
        ("[Pre-Chorus]", "[Pre-Chorus]"),
    ])
    def test_trimming(self, src, want):
        from modules.lyrics_normalizer import trim_tag_modifiers
        assert trim_tag_modifiers(src) == want

    @pytest.mark.parametrize("src", [
        "[EN - Verse]",                     # language, not section + modifier
        "[JA - Verse - whispered, sparse]",
        "[EN - Chorus - anthemic]",
    ])
    def test_a_language_prefix_is_left_verbatim(self, src):
        """docs/descriptor_reference.md puts the language FIRST in the bracket.

        Trimming there would delete the language declaration, so the ``" -"``
        straight after a two-letter code is not a modifier separator.
        """
        from modules.lyrics_normalizer import trim_tag_modifiers
        assert trim_tag_modifiers(src) == src

    def test_tags_on_one_line_are_all_trimmed(self):
        from modules.lyrics_normalizer import trim_tag_modifiers
        assert (trim_tag_modifiers("[Chorus - loud] [Verse - soft]")
                == "[Chorus] [Verse]")

    def test_lyric_text_around_a_tag_is_kept(self):
        from modules.lyrics_normalizer import trim_tag_modifiers
        assert (trim_tag_modifiers("sing [Chorus - loud] now")
                == "sing [Chorus] now")

    def test_it_can_be_switched_off(self):
        out, _ = normalize_lyrics("[Chorus - loud]", do_trim_tag_modifiers=False)
        assert "[Chorus - loud]" in out

    def test_the_report_counts_trimmed_tags(self):
        _, rep = normalize_lyrics("[Chorus - loud]\n[Verse - soft]")
        assert rep["tags_trimmed"] == 2


class TestStrayTagDropping:
    """Consecutive tag-only lines: keep the first, drop the rest."""

    def test_the_worked_example_from_the_request(self):
        from modules.lyrics_normalizer import drop_stray_tags
        assert (drop_stray_tags("[Verse 1]\n[Raspy Vocal] [Mid-Tempo Groove]")
                == "[Verse 1]")

    def test_a_following_lyric_line_is_kept(self):
        from modules.lyrics_normalizer import drop_stray_tags
        out = drop_stray_tags(
            "[Verse 1]\n[Raspy Vocal] [Mid-Tempo Groove]\nI walked alone")
        assert out == "[Verse 1]\nI walked alone"

    def test_a_lone_tag_is_never_dropped(self):
        from modules.lyrics_normalizer import drop_stray_tags
        assert drop_stray_tags("[Chorus]\nverse here") == "[Chorus]\nverse here"

    def test_a_blank_line_ends_the_run(self):
        """The legal two-tag shape from the annotation guide must survive: a tag
        per section, sections separated by a blank line."""
        from modules.lyrics_normalizer import drop_stray_tags
        text = "[Chorus]\n\nverse\n\n[Bridge]\nverse2"
        assert drop_stray_tags(text) == text

    def test_a_tag_after_lyric_text_is_kept(self):
        from modules.lyrics_normalizer import drop_stray_tags
        text = "[Chorus]\nverse line\n[Bridge]\nverse2"
        assert drop_stray_tags(text) == text

    def test_a_run_of_three_keeps_only_the_first(self):
        from modules.lyrics_normalizer import drop_stray_tags
        assert drop_stray_tags("[Intro]\n[Verse]\n[Chorus]\ntext") == "[Intro]\ntext"

    def test_it_can_be_switched_off(self):
        out, _ = normalize_lyrics("[Verse 1]\n[Raspy Vocal]",
                                  do_drop_stray_tags=False)
        assert "[Raspy Vocal]" in out

    def test_the_report_counts_dropped_lines(self):
        _, rep = normalize_lyrics("[Verse 1]\n[Raspy Vocal]\n[Loud]\ntext")
        assert rep["tags_dropped"] == 2


class TestTheThreeRulesTogether:
    def test_a_whole_block_is_tidied_in_one_pass(self):
        src = ('[Intro - sparse piano]\n'
               '[Chorus - Raspy Vocals]\n'
               '"hello there\n'
               '[Verse 1]\n'
               '[Raspy Vocal] [Mid-Tempo Groove]\n'
               'I said "go"')
        out, rep = normalize_lyrics(src)
        assert out == ("[Intro]\nHello there\n[Verse 1]\nI said go")
        assert rep["quotes"] == 3
        assert rep["tags_dropped"] == 2

    def test_a_marker_inside_a_quote_is_still_protected(self):
        """Marker protection runs first, so no tag/quote rule can reach into a
        separator line and break the all-lyrics write-back."""
        text = '---- a.mp3 ----\n"quoted"\n---- b.mp3 ----\nmore'
        out, _ = normalize_lyrics(text)
        assert out.startswith("---- a.mp3 ----")
        assert "---- b.mp3 ----" in out

    def test_the_result_is_stable_on_a_second_pass(self):
        src = '[Chorus - loud]\n"hi\n[Verse 1]\n[Raspy Vocal]'
        once, _ = normalize_lyrics(src)
        twice, _ = normalize_lyrics(once)
        assert once == twice

