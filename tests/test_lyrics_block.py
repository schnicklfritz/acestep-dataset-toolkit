"""The all-lyrics block: build, tidy, write back.

This round-trip is what broke in the real app (markers eaten by the punctuation
stripper), so it is pinned down explicitly. Needs Qt, so it uses the qapp fixture.
"""
import pytest

from modules.lyrics_normalizer import normalize_lyrics
from ui.lyrics_tab import (
    build_all_lyrics_block,
    diff_pairs,
    parse_all_lyrics_block,
    unified_diff,
)

TABLE = {"she'll": "sheel", "can't": "cant"}


class TestAllLyricsBlock:
    def test_build_includes_every_track_with_markers(self, dataset):
        block = build_all_lyrics_block(dataset["samples"])
        assert "---- song_a.mp3 ----" in block
        assert "---- song_b.wav ----" in block

    def test_round_trip_preserves_lyrics(self, dataset):
        block = build_all_lyrics_block(dataset["samples"])
        parsed = parse_all_lyrics_block(block)
        assert "She'll be running" in parsed["song_a.mp3"]

    def test_round_trip_is_keyed_by_filename(self, dataset):
        parsed = parse_all_lyrics_block(build_all_lyrics_block(dataset["samples"]))
        assert set(parsed) == {"song_a.mp3", "song_b.wav"}

    def test_track_without_lyrics_still_gets_a_marker(self, dataset):
        parsed = parse_all_lyrics_block(build_all_lyrics_block(dataset["samples"]))
        assert parsed["song_b.wav"] == ""

    def test_markers_survive_a_tidy(self, dataset):
        # Regression: the punctuation stripper used to eat the trailing "----",
        # which silently broke write-back.
        block = build_all_lyrics_block(dataset["samples"])
        tidied, _ = normalize_lyrics(block, contractions=TABLE, ing_to_in=True)
        assert "---- song_a.mp3 ----" in tidied
        assert "---- song_b.wav ----" in tidied

    def test_tidied_block_still_parses(self, dataset):
        block = build_all_lyrics_block(dataset["samples"])
        tidied, _ = normalize_lyrics(block, contractions=TABLE)
        parsed = parse_all_lyrics_block(tidied)
        assert set(parsed) == {"song_a.mp3", "song_b.wav"}
        assert "Sheel" in parsed["song_a.mp3"]

    def test_parse_ignores_text_before_the_first_marker(self):
        parsed = parse_all_lyrics_block("junk\n---- a.mp3 ----\nwords")
        assert parsed == {"a.mp3": "words"}

    def test_parse_of_empty_input_is_empty(self):
        assert parse_all_lyrics_block("") == {}
        assert parse_all_lyrics_block(None) == {}


class TestDiff:
    def test_unified_diff_shows_old_and_new_lines(self):
        d = unified_diff("She'll run", "Sheel run")
        assert "-She'll run" in d
        assert "+Sheel run" in d

    def test_unified_diff_has_no_file_headers(self):
        d = unified_diff("a", "b")
        assert "---" not in d
        assert "+++" not in d

    def test_identical_input_produces_no_diff(self):
        assert unified_diff("same", "same") == ""

    def test_diff_pairs_returns_structured_old_new(self):
        pairs = diff_pairs("[verse]\nShe'll run", "[Verse]\nSheel run")
        assert ("[verse]", "[Verse]") in pairs
        assert ("She'll run", "Sheel run") in pairs

    def test_diff_pairs_is_empty_when_unchanged(self):
        assert diff_pairs("same", "same") == []
