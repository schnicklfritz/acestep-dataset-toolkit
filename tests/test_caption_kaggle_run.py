"""Tests for the local side of the ACE-Step Kaggle run.

The three things this module exists to make possible are each easy to break
silently, which is why they are pinned here:

  * a PERSISTENT staging folder (its contents are the uploaded dataset, so a
    track that fails to stage must be REPORTED, not skipped quietly — an empty
    staging folder is what produced a captioning run with 0 results);
  * matching results back to tracks by STAGED NAME (a rename must not silently
    mismap captions onto the wrong songs);
  * the diff/merge rules (an approved caption must never be erased by a blank or
    ERROR proposal).
"""
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import caption_kaggle_run as ckr                             # noqa: E402


def _write(path, data=b"audio"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _sample(sid, filename, caption=""):
    return {"id": sid, "filename": filename, "caption": caption,
            "audio_path": f"/in/{filename}"}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def test_staging_and_output_default_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ckr.staging_dir({}) == str(tmp_path / ckr.STAGING_SUBDIR)
    assert ckr.output_dir({}) == str(tmp_path / ckr.CAPTIONS_SUBDIR)


def test_staging_and_output_honour_configured_folders():
    assert ckr.staging_dir({"caption_staging_dir": "/tmp/here"}) == "/tmp/here"
    assert ckr.output_dir({"caption_output_dir": "/tmp/there"}) == "/tmp/there"


def test_blank_config_values_fall_back_to_the_default(tmp_path, monkeypatch):
    """An empty string in settings.json must not mean 'the current directory'."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ckr.staging_dir({"caption_staging_dir": "   "}) == ckr.default_staging_dir()
    assert ckr.output_dir({"caption_output_dir": ""}) == ckr.default_output_dir()


# ---------------------------------------------------------------------------
# staged names
# ---------------------------------------------------------------------------

def test_staged_name_keeps_the_stem_and_swaps_the_extension():
    assert ckr.staged_name("Back In Black.flac") == "Back In Black.mp3"
    assert ckr.staged_name("Back In Black.flac", convert_mp3=False) == "Back In Black.flac"


def test_staged_name_replaces_path_hostile_characters():
    """A ':' or a slash in a title would otherwise break the upload path."""
    assert "/" not in ckr.staged_name("AC:DC / Live?.wav")
    assert ckr.staged_name("") == "track.mp3"


def test_staged_name_never_invents_an_unsupported_extension():
    assert ckr.staged_name("song.xyz", convert_mp3=False).endswith(".mp3")


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------

def test_stage_track_copies_when_conversion_is_off(tmp_path):
    src = _write(str(tmp_path / "src" / "song.flac"))
    staged = ckr.stage_track(src, str(tmp_path / "stage"), convert_mp3=False)
    assert staged == str(tmp_path / "stage" / "song.flac")
    assert os.path.getsize(staged) == os.path.getsize(src)


def test_stage_track_falls_back_to_the_original_when_ffmpeg_refuses(tmp_path, monkeypatch):
    """A failed transcode must COPY, not drop the track."""
    monkeypatch.setattr(ckr, "convert_to_mp3", lambda *a, **k: False)
    src = _write(str(tmp_path / "src" / "song.wav"))
    staged = ckr.stage_track(src, str(tmp_path / "stage"), convert_mp3=True)
    assert staged.endswith("song.wav")
    assert os.path.getsize(staged) == os.path.getsize(src)


def test_stage_track_refuses_a_missing_or_empty_file(tmp_path):
    assert ckr.stage_track("", str(tmp_path / "stage")) == ""
    empty = _write(str(tmp_path / "src" / "empty.mp3"), b"")
    assert ckr.stage_track(empty, str(tmp_path / "stage")) == ""


def test_stage_tracks_reports_skips_instead_of_hiding_them(tmp_path):
    good = _write(str(tmp_path / "src" / "good.flac"))
    items = [
        ("id-1", "good.flac", good),
        ("id-2", "gone.flac", str(tmp_path / "src" / "gone.flac")),
    ]
    staged, skipped = ckr.stage_tracks(
        items, str(tmp_path / "stage"), convert_mp3=False
    )
    assert [key for key, _f, _p in staged] == ["id-1"]
    assert skipped == 1


def test_staged_files_ignores_the_metadata_control_file_and_junk(tmp_path):
    stage = str(tmp_path / "stage")
    _write(os.path.join(stage, "one.mp3"))
    _write(os.path.join(stage, "dataset-metadata.json"), b"{}")
    _write(os.path.join(stage, "notes.txt"))
    _write(os.path.join(stage, "zero.mp3"), b"")
    assert [os.path.basename(p) for p in ckr.staged_files(stage)] == ["one.mp3"]


def test_remove_staged_deletes_only_what_it_is_given(tmp_path):
    stage = str(tmp_path / "stage")
    _write(os.path.join(stage, "a.mp3"))
    _write(os.path.join(stage, "b.mp3"))
    assert ckr.remove_staged(stage, ["a.mp3", "missing.mp3"]) == 1
    assert [os.path.basename(p) for p in ckr.staged_files(stage)] == ["b.mp3"]


def test_remove_staged_cannot_escape_the_staging_folder(tmp_path):
    """A '..' name must not delete a file outside the folder."""
    outside = _write(str(tmp_path / "outside.mp3"))
    stage = str(tmp_path / "stage")
    os.makedirs(stage, exist_ok=True)
    ckr.remove_staged(stage, [os.path.join("..", "outside.mp3")])
    assert os.path.exists(outside)


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

def test_load_results_reads_the_kernel_shape(tmp_path):
    path = tmp_path / "captions_out.json"
    path.write_text(json.dumps({"results": [
        {"file": "Song A.mp3", "caption": "tags, flow"},
        {"file": "sub/Song B.mp3", "caption": "other"},
    ]}), encoding="utf-8")
    assert ckr.load_results(str(path)) == {
        "Song A.mp3": "tags, flow", "Song B.mp3": "other",
    }


def test_load_results_rejects_a_non_captions_document(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"samples": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="captions_out.json"):
        ckr.load_results(str(path))


def test_load_results_reports_unreadable_json_as_a_value_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read"):
        ckr.load_results(str(path))


# ---------------------------------------------------------------------------
# the diff
# ---------------------------------------------------------------------------

def test_diff_classifies_new_changed_same_missing_and_error():
    samples = [
        _sample("1", "New Song.mp3"),                       # no caption -> new
        _sample("2", "Changed Song.mp3", "old text"),       # differs -> changed
        _sample("3", "Same Song.mp3", "identical"),         # same
        _sample("4", "Missing Song.mp3", "old text"),       # no result -> missing
        _sample("5", "Broken Song.mp3", "old text"),        # ERROR -> error
    ]
    results = {
        "New Song.mp3": "fresh tags, flow",
        "Changed Song.mp3": "better tags, flow",
        "Same Song.mp3": "identical",
        "Broken Song.mp3": "ERROR: out of memory",
    }
    rows = {row["id"]: row for row in ckr.diff_captions(samples, results)}
    assert rows["1"]["status"] == ckr.STATUS_NEW
    assert rows["2"]["status"] == ckr.STATUS_CHANGED
    assert rows["3"]["status"] == ckr.STATUS_SAME
    assert rows["4"]["status"] == ckr.STATUS_MISSING
    assert rows["5"]["status"] == ckr.STATUS_ERROR


def test_diff_matches_by_staged_name_not_by_position():
    """Results come back named as STAGED files; position must not matter."""
    samples = [_sample("1", "Back In Black.flac"), _sample("2", "Hells Bells.wav")]
    results = {
        "Hells Bells.mp3": "second track caption",
        "Back In Black.mp3": "first track caption",
    }
    rows = {row["id"]: row for row in ckr.diff_captions(samples, results)}
    assert rows["1"]["proposed"] == "first track caption"
    assert rows["2"]["proposed"] == "second track caption"


def test_diff_can_match_the_untouched_original_name():
    samples = [_sample("1", "song.mp3")]
    rows = ckr.diff_captions(samples, {"song.mp3": "kept the name"})
    assert rows[0]["proposed"] == "kept the name"


def test_a_renamed_result_is_reported_unmatched_not_mismapped():
    samples = [_sample("1", "Song.mp3", "existing")]
    results = {"Totally Different.mp3": "caption"}
    rows = ckr.diff_captions(samples, results)
    assert rows[0]["status"] == ckr.STATUS_MISSING
    assert ckr.unmatched_results(samples, results) == ["Totally Different.mp3"]


def test_count_by_status_summarises_a_run():
    rows = [
        {"status": ckr.STATUS_NEW}, {"status": ckr.STATUS_NEW},
        {"status": ckr.STATUS_CHANGED},
    ]
    assert ckr.count_by_status(rows) == {"new": 2, "changed": 1}


# ---------------------------------------------------------------------------
# applying a decision
# ---------------------------------------------------------------------------

def test_keep_records_the_proposal_without_touching_the_caption():
    sample = _sample("1", "Song.mp3", "approved caption")
    row = ckr.diff_captions([sample], {"Song.mp3": "offered caption"})[0]
    assert ckr.apply_decision(sample, row, "keep") == "approved caption"
    assert sample["caption"] == "approved caption"
    assert sample["caption_ai_raw"] == "offered caption"


def test_use_replaces_the_caption_and_keeps_the_old_one():
    sample = _sample("1", "Song.mp3", "approved caption")
    row = ckr.diff_captions([sample], {"Song.mp3": "much better caption"})[0]
    assert ckr.apply_decision(sample, row, "use") == "much better caption"
    assert sample["caption"] == "much better caption"
    assert sample["caption_before_kaggle"] == "approved caption"


def test_use_ADDS_the_caption_when_there_was_none():
    """'if there are none, they should be added' — the new row case."""
    sample = _sample("1", "Song.mp3")
    row = ckr.diff_captions([sample], {"Song.mp3": "brand new caption"})[0]
    assert row["status"] == ckr.STATUS_NEW
    ckr.apply_decision(sample, row, "use")
    assert sample["caption"] == "brand new caption"
    assert "caption_before_kaggle" not in sample       # nothing to preserve


def test_edit_takes_the_hand_written_text():
    sample = _sample("1", "Song.mp3", "approved caption")
    row = ckr.diff_captions([sample], {"Song.mp3": "offered"})[0]
    ckr.apply_decision(sample, row, "edit", edited_text="hand merged caption")
    assert sample["caption"] == "hand merged caption"
    assert sample["caption_before_kaggle"] == "approved caption"


def test_a_blank_proposal_cannot_erase_a_caption():
    """Writing '' would look like success while destroying an approved caption."""
    sample = _sample("1", "Song.mp3", "approved caption")
    row = ckr.diff_captions([sample], {})[0]
    assert ckr.apply_decision(sample, row, "use") == "approved caption"
    assert sample["caption"] == "approved caption"


def test_an_ERROR_proposal_is_never_committed():
    sample = _sample("1", "Song.mp3", "approved caption")
    row = ckr.diff_captions([sample], {"Song.mp3": "ERROR: CUDA out of memory"})[0]
    assert ckr.apply_decision(sample, row, "use") == "approved caption"
    assert sample["caption"] == "approved caption"


def test_accepting_the_same_text_does_not_fake_a_backup():
    sample = _sample("1", "Song.mp3", "identical caption")
    row = ckr.diff_captions([sample], {"Song.mp3": "identical caption"})[0]
    ckr.apply_decision(sample, row, "use")
    assert "caption_before_kaggle" not in sample
    assert sample["caption"] == "identical caption"


# ---------------------------------------------------------------------------
# staging must not be able to lie about what uploads
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, returncode):
        self.returncode = returncode


def test_a_failed_transcode_cannot_leave_a_zero_byte_track(tmp_path, monkeypatch):
    """THE BUG THIS PINS: ffmpeg creates its output file the moment it starts, so
    a process killed mid-encode left a 0-byte file under the track's REAL name.
    The UI filtered it out (it lists only usable files) and the Kaggle upload sent
    it anyway, where the kernel captioned it into an error row."""
    stage = tmp_path / "stage"
    stage.mkdir()
    src = _write(str(tmp_path / "src" / "song.flac"), b"x" * 64)

    def killed(cmd, **kwargs):
        with open(cmd[-1], "wb") as fh:      # the real ffmpeg behaviour...
            fh.write(b"")                    # ...a file that exists and is empty
        return _FakeProc(1)                  # ...then it dies

    monkeypatch.setattr(ckr.subprocess, "run", killed)
    staged = ckr.stage_track(src, str(stage), convert_mp3=True)

    # The designed fallback still stages the ORIGINAL, so no track is lost...
    assert staged.endswith("song.flac")
    # ...and nothing pretends to be a finished MP3, and no scratch is left.
    assert not (stage / "song.mp3").exists()
    assert not list(stage.glob("*" + ckr.PART_SUFFIX))


def test_convert_renames_into_place_only_after_ffmpeg_succeeds(tmp_path, monkeypatch):
    src = _write(str(tmp_path / "src" / "song.flac"), b"x")
    dst = tmp_path / "stage" / "song.mp3"
    dst.parent.mkdir()

    def ok(cmd, **kwargs):
        with open(cmd[-1], "wb") as fh:      # ffmpeg writes the .part file
            fh.write(b"mp3data")
        return _FakeProc(0)

    monkeypatch.setattr(ckr.subprocess, "run", ok)
    assert ckr.convert_to_mp3(src, str(dst)) is True
    assert dst.read_bytes() == b"mp3data"
    assert not list(dst.parent.glob("*" + ckr.PART_SUFFIX))


def test_convert_discards_the_scratch_file_when_ffmpeg_fails(tmp_path, monkeypatch):
    src = _write(str(tmp_path / "src" / "song.flac"), b"x")
    dst = tmp_path / "stage" / "song.mp3"
    dst.parent.mkdir()

    def bad(cmd, **kwargs):
        with open(cmd[-1], "wb") as fh:
            fh.write(b"half")
        return _FakeProc(1)

    monkeypatch.setattr(ckr.subprocess, "run", bad)
    assert ckr.convert_to_mp3(src, str(dst)) is False
    assert not dst.exists()
    assert not list(dst.parent.glob("*" + ckr.PART_SUFFIX))


def test_staging_report_names_why_a_file_is_ignored(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    _write(str(stage / "good.mp3"), b"audio")
    _write(str(stage / "zero.mp3"), b"")
    _write(str(stage / ("song.mp3" + ckr.PART_SUFFIX)), b"half")
    _write(str(stage / ckr.METADATA_NAME), b"{}")

    report = ckr.staging_report(str(stage))
    assert [os.path.basename(p) for p in report["usable"]] == ["good.mp3"]
    reasons = dict(report["ignored"])
    assert reasons["zero.mp3"] == "empty or unreadable"
    assert reasons["song.mp3" + ckr.PART_SUFFIX] == "interrupted transcode"
    assert reasons[ckr.METADATA_NAME] == "Kaggle metadata, not audio"


def test_clean_unusable_removes_junk_but_keeps_everything_else(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    _write(str(stage / "good.mp3"), b"audio")
    _write(str(stage / "zero.mp3"), b"")
    _write(str(stage / ("scratch.mp3" + ckr.PART_SUFFIX)), b"half")
    # A file of an unsupported FORMAT is NOT junk: it may be intentional, so the
    # cleanup must leave it alone even though it is not uploaded.
    _write(str(stage / "cover.png"), b"art")

    assert ckr.clean_unusable_staged(str(stage)) == 2
    assert sorted(os.listdir(stage)) == ["cover.png", "good.mp3"]


def test_the_upload_carries_only_uploadable_files(tmp_path):
    """The list and the upload must not disagree: anything the page refuses to
    list must also never reach the Kaggle dataset."""
    from modules import kaggle as kg

    stage = tmp_path / "stage"
    stage.mkdir()
    _write(str(stage / "good.mp3"), b"audio")
    _write(str(stage / "zero.mp3"), b"")
    _write(str(stage / ("interrupted.mp3" + ckr.PART_SUFFIX)), b"half")
    _write(str(stage / ckr.METADATA_NAME), b"{}")

    payload = kg._upload_payload(str(stage), {"id": "u/d", "title": "d"})
    try:
        assert sorted(os.listdir(payload)) == [ckr.METADATA_NAME, "good.mp3"]
        # The staging folder itself is untouched — the junk is left for the user
        # to clean, not deleted behind their back.
        assert sorted(os.listdir(stage)) == [
            ckr.METADATA_NAME, "good.mp3", "interrupted.mp3" + ckr.PART_SUFFIX,
            "zero.mp3",
        ]
    finally:
        shutil.rmtree(payload, ignore_errors=True)


def test_the_upload_refuses_a_folder_with_nothing_uploadable(tmp_path):
    """Uploading an empty dataset would LOOK like a successful run."""
    from modules import kaggle as kg

    stage = tmp_path / "stage"
    stage.mkdir()
    _write(str(stage / "zero.mp3"), b"")
    with pytest.raises(RuntimeError, match="Nothing uploadable"):
        kg._upload_payload(str(stage), {"id": "u/d", "title": "d"})
