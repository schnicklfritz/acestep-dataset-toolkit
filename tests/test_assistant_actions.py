"""The assistant's action layer: what it may write, and what it refuses.

The whole point of ``modules/assistant_actions.py`` is that a language model
proposes an action and *this* code decides whether it is real and safe. So the
tests that matter are the refusals — an identity/path field, a bogus index, a
contradiction (instrumental track with lyrics) — plus the promise that nothing
here needs an LLM or runs ffmpeg on its own.
"""
import pytest

from modules import assistant_actions as aa
from modules.dataset_schema import new_dataset, new_sample, to_export_sample


@pytest.fixture()
def ds():
    d = new_dataset(name="actions")
    d["samples"] = [
        new_sample(filename="a.mp3", genre="Rock", bpm=120, keyscale="E minor",
                   language="en", caption="ok"),
        new_sample(filename="b.wav", genre="Doom", is_instrumental=True),
    ]
    return d


# ---------------------------------------------------------------------------
# Field writes
# ---------------------------------------------------------------------------
def test_writes_an_allowlisted_field(ds):
    assert "caption = heavy" in aa.set_track_field(ds, 0, "caption", "heavy")
    assert ds["samples"][0]["caption"] == "heavy"


def test_refuses_to_repoint_the_audio_path(ds):
    """A hallucinated path would silently attach the wrong file to a track."""
    out = aa.set_track_field(ds, 0, "audio_path", "/etc/passwd")
    assert "Refused" in out
    assert ds["samples"][0]["audio_path"] == ""


@pytest.mark.parametrize("field", ["id", "filename", "labeled", "locked"])
def test_refuses_every_identity_and_provenance_field(ds, field):
    before = ds["samples"][0].get(field)
    assert "Refused" in aa.set_track_field(ds, 0, field, "hacked")
    assert ds["samples"][0].get(field) == before


def test_refuses_an_out_of_range_index(ds):
    assert "No such track" in aa.set_track_field(ds, 99, "caption", "x")


def test_coerces_types_the_schema_expects(ds):
    aa.set_track_field(ds, 0, "bpm", "137.6")
    assert ds["samples"][0]["bpm"] == 137
    aa.set_track_field(ds, 0, "is_instrumental", "yes")
    assert ds["samples"][0]["is_instrumental"] is True


def test_resolves_tracks_by_filename_and_by_printed_number(ds):
    aa.set_field_many(ds, ["b.wav"], "genre", "Sludge")
    assert ds["samples"][1]["genre"] == "Sludge"
    # 1-based: "1" is the first track as list_tracks prints it.
    aa.set_field_many(ds, [1], "genre", "Hard Rock")
    assert ds["samples"][0]["genre"] == "Hard Rock"


def test_reports_unresolved_references_instead_of_guessing(ds):
    assert "No tracks matched" in aa.set_field_many(ds, ["nope.mp3"], "genre", "X")
    out = aa.set_field_many(ds, ["a.mp3", "ghost.wav"], "genre", "X")
    assert "not found: ghost.wav" in out


# ---------------------------------------------------------------------------
# Lyrics import
# ---------------------------------------------------------------------------
def test_lyrics_import_keeps_the_three_fields_in_step(ds):
    aa.import_lyrics(ds, ["a.mp3"], "[Verse]\nRUN AWAY")
    s = ds["samples"][0]
    assert s["formatted_lyrics"] == s["lyrics"] == s["raw_lyrics"] == "[Verse]\nRUN AWAY"


def test_importing_lyrics_clears_an_instrumental_flag(ds):
    """An instrumental track with lyrics is what the validator flags."""
    aa.import_lyrics(ds, ["b.wav"], "[Verse]\nsing")
    assert ds["samples"][1]["is_instrumental"] is False


def test_an_empty_lyrics_import_is_refused(ds):
    assert "Refused" in aa.import_lyrics(ds, ["a.mp3"], "   ")
    assert ds["samples"][0]["lyrics"] == ""


# ---------------------------------------------------------------------------
# ffmpeg: commands are built, not run
# ---------------------------------------------------------------------------
def test_normalize_command_matches_the_dsp_normalize_button():
    argv = aa.build_normalize_command("in.flac", "out.wav", -14.0, 44100)
    assert argv[:4] == ["ffmpeg", "-y", "-i", "in.flac"]
    assert "loudnorm=I=-14.0:TP=-1.0:LRA=11" in argv
    assert argv[-1] == "out.wav"


def test_temp_mp3_command_is_an_audio_only_staging_copy():
    argv = aa.build_temp_mp3_command("in.flac", "/tmp/in.mp3", "192k")
    assert "-vn" in argv          # audio only: no cover art in the upload
    assert "192k" in argv
    assert argv[-1] == "/tmp/in.mp3"


def test_normalize_uses_the_injected_runner_and_backs_up_first(ds, tmp_path):
    """No ffmpeg in the test: assert the argv and the backup, not the audio."""
    src = tmp_path / "a.mp3"
    src.write_bytes(b"not really audio")
    ds["samples"][0]["audio_path"] = str(src)

    calls = []

    class _Result:
        returncode = 1          # simulate failure: nothing is written

    def fake_runner(argv, **kwargs):
        calls.append(argv)
        return _Result()

    out = aa.normalize_tracks(ds, ["a.mp3"], tmp_path / "work", runner=fake_runner)
    assert len(calls) == 1
    assert calls[0][0] == "ffmpeg"
    assert (tmp_path / "work" / "originals_backup" / "a.mp3").exists()
    assert "FAILED" in out


def test_normalize_skips_a_track_with_no_audio_on_disk(ds, tmp_path):
    out = aa.normalize_tracks(ds, ["a.mp3"], tmp_path, runner=lambda *a, **k: None)
    assert "SKIPPED" in out



# ---------------------------------------------------------------------------
# Gap audit
# ---------------------------------------------------------------------------
def test_gap_audit_names_each_missing_field(ds):
    out = aa.find_gaps(ds)
    # b.wav has genre and the schema's default language, so only these are gaps.
    assert "b.wav" in out
    assert "missing caption/bpm/keyscale" in out
    # a.mp3 has those three but no lyrics, and is not marked instrumental.
    assert "has no lyrics and is not marked instrumental" in out


def test_gap_audit_flags_a_contradiction(ds):
    ds["samples"][1]["lyrics"] = "[Verse]\noops"
    assert "instrumental but has lyrics" in aa.find_gaps(ds)


def test_gap_audit_flags_lyrics_without_section_markers(ds):
    ds["samples"][0]["lyrics"] = "run away tonight"
    assert "no [Section] markers" in aa.find_gaps(ds)


def test_gap_audit_flags_a_duplicate_filename(ds):
    ds["samples"][1]["filename"] = "a.mp3"
    assert "duplicate filename a.mp3" in aa.find_gaps(ds)


def test_gap_audit_calls_an_empty_dataset_empty():
    assert "empty" in aa.find_gaps(new_dataset())


def test_gap_audit_says_ok_for_a_complete_track(ds, tmp_path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"x")
    ds["samples"] = [new_sample(
        filename="a.mp3", audio_path=str(src), genre="Rock", bpm=120,
        keyscale="E minor", language="en", lyrics="[Verse]\nrun",
        caption=("hard rock, drums, bass, male vocal, raw production, 1970s "
                 "analog. It opens sparse and builds steadily through the "
                 "verse. The chorus hits hard and the track finishes on a peak."),
    )]
    out = aa.find_gaps(ds)
    assert "a.mp3: ok" in out
    assert "1 fully specified" in out


# ---------------------------------------------------------------------------
# Virtual dataset creation
# ---------------------------------------------------------------------------
def test_virtual_tracks_are_flagged_and_have_no_audio(ds):
    aa.create_virtual_dataset(ds, [{"filename": "v1.wav", "genre": "Doom"}])
    v = ds["samples"][-1]
    assert v["virtual"] is True
    assert v["audio_path"] == ""
    assert v["language"] == "en"


def test_a_virtual_track_never_reaches_training_data(ds):
    """The export boundary must DROP a virtual track, not merely untag it.

    Regression: ``to_export_sample`` strips the ``virtual`` key but still emits
    a row, so ``file_name`` fell back to the placeholder ``virtual_01_*.wav``
    and ``dataset.json`` pointed at audio that does not exist. Asserting
    ``"virtual" not in ...`` passed the whole time and hid the leak — the real
    assertion is that the track is ABSENT from the export.
    """
    aa.create_virtual_dataset(ds, [{"filename": "virtual_01_doom.wav",
                                    "genre": "Doom"}])
    assert "virtual" not in to_export_sample(ds["samples"][-1])

    from modules import exporters

    exported = exporters.for_export(ds["samples"])
    assert exported, "sanity: a real track must still be exported"
    assert all(not s.get("virtual") for s in exported)
    assert "virtual_01_doom.wav" not in [
        s.get("audio_path") or s.get("filename") for s in exported
    ]


def test_virtual_tracks_get_a_readable_placeholder_name(ds):
    aa.create_virtual_dataset(ds, [{"genre": "Doom Metal"}])
    assert ds["samples"][-1]["filename"].startswith("virtual_")
    assert "doom_metal" in ds["samples"][-1]["filename"]


def test_virtual_tracks_only_take_allowlisted_metadata(ds):
    aa.create_virtual_dataset(ds, [{"filename": "v.wav", "id": "stolen",
                                    "audio_path": "/etc/passwd"}])
    v = ds["samples"][-1]
    assert v["id"] != "stolen"
    assert v["audio_path"] == ""


def test_creating_a_virtual_dataset_can_name_the_dataset(ds):
    aa.create_virtual_dataset(ds, [{"filename": "v.wav"}], name="Concept Album")
    assert ds["metadata"]["name"] == "Concept Album"


def test_virtual_creation_refuses_an_empty_spec_list(ds):
    assert "Refused" in aa.create_virtual_dataset(ds, [])

