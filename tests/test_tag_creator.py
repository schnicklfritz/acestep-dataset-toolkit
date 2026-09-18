"""Tests for the Structural Tag Creator's vocabulary wiring.

The regression these guard: the prompt used to carry one hardcoded generic list
for every track (34 terms, including rap styles). Measured against the
authoritative descriptor doc it shared 14 of 34 terms, and it offered "mumble
rap" / "trap flow" for a Black Sabbath record while omitting "downtuned guitar".
"""
import pytest

from modules import tag_creator
from modules.tag_creator import (
    build_track_context,
    load_markers,
    load_vocabulary,
    parse_output,
    tag_creator_messages,
)


@pytest.fixture
def sample():
    return {
        "filename": "War_Pigs.flac",
        "genre": "heavy metal",
        "bpm": 140,
        "keyscale": "E minor",
        "caption": "",
        "lyrics": "[verse]\nGenerals gathered in their masses",
    }


def _system_prompt(sample, artist=""):
    messages = tag_creator_messages(sample, artist)
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


# --------------------------------------------------------------------------
# The real vocabulary is loaded from the generated lexicon
# --------------------------------------------------------------------------

def test_artist_scoping_offers_that_artists_terms(sample):
    prompt = _system_prompt(sample, "sabbath")
    assert "Black Sabbath" in prompt
    assert "downtuned guitar" in prompt


def test_artist_scoping_excludes_other_artists_terms(sample):
    prompt = _system_prompt(sample, "sabbath")
    assert "Vox Continental organ" not in prompt      # The Doors
    assert "steel guitar" not in prompt               # Hank Williams Sr.


def test_rap_styles_are_gone_entirely(sample):
    """The old hardcoded list offered these for 1970s hard rock."""
    for artist in ("sabbath", "acdc", "", "Doorsdata"):
        prompt = _system_prompt(sample, artist)
        for unwanted in ("mumble rap", "chopper rap", "trap flow", "double-time"):
            assert unwanted not in prompt, f"{unwanted} leaked for {artist!r}"


def test_unknown_artist_falls_back_to_fundamentals_only(sample):
    prompt = _system_prompt(sample, "Nirvana")
    # No artist heading, but the fundamentals are still offered.
    assert "ARTIST VOCABULARY" not in prompt
    assert "CROSS-ARTIST FUNDAMENTALS" in prompt
    assert "Vox Continental organ" not in prompt


def test_dataset_name_style_hint_resolves(sample):
    assert "Black Sabbath" in _system_prompt(sample, "sabbath")
    assert "The Doors" in _system_prompt(sample, "Doorsdata")
    assert "Hank Williams Sr." in _system_prompt(sample, "hank_sr")
    assert "AC/DC" in _system_prompt(sample, "acdc")


# --------------------------------------------------------------------------
# Spec rules carried into the prompt (from the authoritative doc)
# --------------------------------------------------------------------------

def test_prompt_enforces_spec_shape_rules(sample):
    prompt = _system_prompt(sample, "sabbath")
    assert "Max 3 descriptors per bracket" in prompt
    assert "NEVER put BPM, key, or time signature" in prompt
    assert "5-12 comma-separated keywords" in prompt
    assert "CAPTION:" in prompt and "LYRICS:" in prompt


def test_prompt_includes_section_markers(sample):
    prompt = _system_prompt(sample, "sabbath")
    assert "[Chorus]" in prompt
    assert "[Guitar Solo]" in prompt


def test_prompt_forbids_inventing_descriptors(sample):
    prompt = _system_prompt(sample, "sabbath")
    assert "Choose tags ONLY from the vocabulary above." in prompt
    assert "verbatim" in prompt


# --------------------------------------------------------------------------
# Loading, fallbacks and the output contract
# --------------------------------------------------------------------------

def test_load_vocabulary_returns_resolved_artist():
    text, resolved = load_vocabulary("sabbath")
    assert resolved == "Black Sabbath"
    assert "downtuned guitar" in text


def test_load_vocabulary_without_artist_returns_fundamentals():
    text, resolved = load_vocabulary("")
    assert resolved is None
    assert "CROSS-ARTIST FUNDAMENTALS" in text


def test_load_markers_returns_generated_markers():
    markers = load_markers()
    assert "[Intro]" in markers
    assert not [m for m in markers if not m.endswith("]")]


def test_missing_vocabulary_raises_with_the_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(tag_creator, "GROUPS_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setattr(tag_creator, "FLAT_PATH", str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError) as exc:
        load_vocabulary("sabbath")
    assert "build_lexicon" in str(exc.value)


def test_flat_list_is_used_when_groups_are_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(tag_creator, "GROUPS_PATH", str(tmp_path / "nope.json"))
    flat = tmp_path / "vocabulary.txt"
    flat.write_text("downtuned guitar\nthunderous drums\n", encoding="utf-8")
    monkeypatch.setattr(tag_creator, "FLAT_PATH", str(flat))
    text, resolved = load_vocabulary("sabbath")
    assert resolved is None
    assert "thunderous drums" in text


def test_build_track_context_still_includes_metadata(sample):
    context = build_track_context(sample)
    assert "War_Pigs.flac" in context
    assert "BPM: 140" in context      # context may carry it; the CAPTION may not
    assert "Generals gathered" in context


def test_parse_output_contract_unchanged():
    caption, lyrics = parse_output(
        "CAPTION:\nheavy metal, doom-laden. Opens with a slow riff.\n\n"
        "LYRICS:\n[Verse - nasal vocal]\nGenerals gathered"
    )
    assert caption.startswith("heavy metal")
    assert "Opens with" in caption
    assert lyrics.startswith("[Verse - nasal vocal]")
    assert "Generals gathered" in lyrics


def test_parse_output_handles_absent_blocks():
    caption, lyrics = parse_output("no markers here")
    assert caption == "" and lyrics == ""


# --------------------------------------------------------------------------
# The dataset-level artist hint used by the UI
# --------------------------------------------------------------------------

def test_tag_creator_artist_prefers_explicit_setting():
    from dataset_manager import DatasetManager

    class Stub:
        config = {"tag_creator_artist": "AC/DC"}
        dataset = {"metadata": {"name": "sabbath"}}

    assert DatasetManager._tag_creator_artist(Stub()) == "AC/DC"


def test_tag_creator_artist_falls_back_to_dataset_name():
    from dataset_manager import DatasetManager

    class Stub:
        config = {"tag_creator_artist": ""}
        dataset = {"metadata": {"name": "sabbath"}}

    assert DatasetManager._tag_creator_artist(Stub()) == "sabbath"


def test_tag_creator_artist_handles_missing_metadata():
    from dataset_manager import DatasetManager

    class Stub:
        config = {}
        dataset = {}

    assert DatasetManager._tag_creator_artist(Stub()) == ""
