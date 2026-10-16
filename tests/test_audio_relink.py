"""Relink audio entries whose file changed name or format (``modules/audio_relink``).

The load-bearing case is the owner's Doors dataset: every ``.wav`` was re-encoded
to ``.flac`` outside the app, so ``audio_path`` points at a file that no longer
exists and the track is silent with an empty waveform.

Two properties are asserted here rather than assumed, because they are the whole
reason this module exists separately from ``core.file_system``:

1. **It never touches the filesystem.** A relink must not rename, move, copy or
   delete anything - the files are already correct on disk. The tests below
   assert the bytes on disk are untouched, so a future edit that reaches for a
   rename helper fails loudly.
2. **Both ``filename`` and ``audio_path`` move together**, because ACE-Step keys
   per-sample metadata by ``filename`` and only falls back to the basename of
   ``audio_path``. Updating one alone points the trainer at the wrong name.
"""
import os

from modules.audio_relink import (
    EXACT,
    FUZZY,
    STEM,
    apply_plan,
    apply_relink,
    find_missing,
    index_folder,
    match_candidate,
    plan_relink,
    resolved_path,
)
from modules.dataset_schema import new_sample


def _touch(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


class TestResolvedPath:
    def test_absolute_path_is_used_verbatim(self, tmp_path):
        s = new_sample(audio_path=str(tmp_path / "a.flac"))
        assert resolved_path(s) == str(tmp_path / "a.flac")

    def test_relative_path_resolves_against_base_dir(self, tmp_path):
        s = new_sample(audio_path="audio/a.flac")
        assert resolved_path(s, str(tmp_path)) == os.path.join(str(tmp_path), "audio/a.flac")

    def test_relative_path_without_base_dir_is_left_alone(self):
        assert resolved_path(new_sample(audio_path="audio/a.flac")) == "audio/a.flac"

    def test_empty_audio_path_falls_back_to_filename(self):
        assert resolved_path(new_sample(filename="a.mp3", audio_path="")) == "a.mp3"

    def test_no_path_at_all_is_empty(self):
        assert resolved_path(new_sample(filename="", audio_path="")) == ""


class TestFindMissing:
    def test_existing_file_is_not_reported(self, tmp_path):
        p = _touch(str(tmp_path / "a.flac"))
        s = new_sample(audio_path=p)
        assert find_missing([s]) == []

    def test_stale_path_is_reported_with_its_index(self, tmp_path):
        gone = str(tmp_path / "a.wav")
        s = new_sample(audio_path=gone)
        assert find_missing([s]) == [(0, s, gone)]

    def test_virtual_sample_is_skipped(self, tmp_path):
        """A concept-only placeholder has no audio on disk BY DESIGN."""
        s = new_sample(audio_path=str(tmp_path / "nope.wav"), virtual=True)
        assert find_missing([s]) == []

    def test_sample_naming_no_file_is_skipped(self):
        assert find_missing([new_sample(filename="", audio_path="")]) == []

    def test_relative_path_is_checked_against_base_dir(self, tmp_path):
        s = new_sample(audio_path="audio/a.flac")
        _touch(str(tmp_path / "audio" / "a.flac"))
        assert find_missing([s], str(tmp_path)) == []


class TestIndexFolder:
    def test_groups_audio_by_normalised_stem(self, tmp_path):
        _touch(str(tmp_path / "Songs" / "Take It As It Comes.flac"))
        assert list(index_folder(str(tmp_path))) == ["takeitasitcomes"]

    def test_ignores_non_audio_files(self, tmp_path):
        _touch(str(tmp_path / "notes.txt"))
        _touch(str(tmp_path / "cover.png"))
        assert index_folder(str(tmp_path)) == {}

    def test_recursive_is_the_default(self, tmp_path):
        _touch(str(tmp_path / "a" / "b" / "song.flac"))
        assert "song" in index_folder(str(tmp_path))

    def test_non_recursive_stays_in_one_folder(self, tmp_path):
        _touch(str(tmp_path / "a" / "song.flac"))
        assert index_folder(str(tmp_path), recursive=False) == {}

    def test_missing_folder_is_an_empty_index(self, tmp_path):
        assert index_folder(str(tmp_path / "nope")) == {}
        assert index_folder("") == {}


class TestMatchCandidate:
    def test_same_stem_different_extension_in_same_folder_is_exact(self, tmp_path):
        """The Doors case: every .wav re-encoded to .flac, in place."""
        gone = str(tmp_path / "Take It As It Comes.wav")
        new = _touch(str(tmp_path / "Take It As It Comes.flac"))
        s = new_sample(filename="Take It As It Comes.wav", audio_path=gone)
        assert match_candidate(s, index_folder(str(tmp_path))) == (new, EXACT)

    def test_same_stem_in_another_folder_is_stem(self, tmp_path):
        gone = str(tmp_path / "old" / "song.wav")
        new = _touch(str(tmp_path / "new" / "song.flac"))
        s = new_sample(audio_path=gone)
        assert match_candidate(s, index_folder(str(tmp_path))) == (new, STEM)

    def test_case_and_spacing_differences_still_match(self, tmp_path):
        gone = str(tmp_path / "old" / "Take It As It Comes.wav")
        new = _touch(str(tmp_path / "new" / "take_it_as_it_comes.flac"))
        s = new_sample(audio_path=gone)
        assert match_candidate(s, index_folder(str(tmp_path))) == (new, STEM)

    def test_no_candidate_returns_none(self, tmp_path):
        _touch(str(tmp_path / "something_else.flac"))
        s = new_sample(audio_path=str(tmp_path / "unrelated.wav"))
        assert match_candidate(s, index_folder(str(tmp_path))) is None

    def test_partial_name_is_fuzzy_when_unambiguous(self, tmp_path):
        gone = str(tmp_path / "old" / "tracksong.wav")
        new = _touch(str(tmp_path / "new" / "tracksong_master.flac"))
        s = new_sample(audio_path=gone)
        assert match_candidate(s, index_folder(str(tmp_path))) == (new, FUZZY)

    def test_partial_name_is_refused_when_ambiguous(self, tmp_path):
        """Two near-misses are more likely two other songs than one match."""
        _touch(str(tmp_path / "tracksong_a.flac"))
        _touch(str(tmp_path / "tracksong_b.flac"))
        s = new_sample(audio_path=str(tmp_path / "tracksong.wav"))
        assert match_candidate(s, index_folder(str(tmp_path))) is None

    def test_fuzzy_can_be_switched_off(self, tmp_path):
        gone = str(tmp_path / "old" / "tracksong.wav")
        _touch(str(tmp_path / "new" / "tracksong_master.flac"))
        s = new_sample(audio_path=gone)
        idx = index_folder(str(tmp_path))
        assert match_candidate(s, idx, allow_fuzzy=False) is None

    def test_relative_stale_path_matches_against_base_dir(self, tmp_path):
        new = _touch(str(tmp_path / "audio" / "song.flac"))
        s = new_sample(audio_path="audio/song.wav")
        idx = index_folder(str(tmp_path))
        assert match_candidate(s, idx, base_dir=str(tmp_path)) == (new, EXACT)


class TestPlanRelink:
    def test_exact_matches_sort_before_stem_matches(self, tmp_path):
        _touch(str(tmp_path / "here" / "a.flac"))
        _touch(str(tmp_path / "else" / "b.flac"))
        exact = new_sample(audio_path=str(tmp_path / "here" / "a.wav"))
        stem = new_sample(audio_path=str(tmp_path / "gone" / "b.wav"))
        plan = plan_relink([stem, exact], index_folder(str(tmp_path)))
        assert [row[4] for row in plan] == [EXACT, STEM]

    def test_unmatched_samples_are_absent_from_the_plan(self, tmp_path):
        _touch(str(tmp_path / "other.flac"))
        s = new_sample(audio_path=str(tmp_path / "nothing.wav"))
        assert plan_relink([s], index_folder(str(tmp_path))) == []

    def test_plan_does_not_change_the_sample(self, tmp_path):
        gone = str(tmp_path / "a.wav")
        _touch(str(tmp_path / "a.flac"))
        s = new_sample(filename="a.wav", audio_path=gone)
        plan_relink([s], index_folder(str(tmp_path)))
        assert s["audio_path"] == gone
        assert s["filename"] == "a.wav"

    def test_plan_row_carries_index_old_and_new(self, tmp_path):
        gone = str(tmp_path / "a.wav")
        new = _touch(str(tmp_path / "a.flac"))
        s = new_sample(audio_path=gone)
        plan = plan_relink([s], index_folder(str(tmp_path)))
        assert plan == [(0, s, gone, new, EXACT)]


class TestApplyRelink:
    def test_updates_both_filename_and_audio_path(self, tmp_path):
        new = _touch(str(tmp_path / "a.flac"))
        s = new_sample(filename="a.wav", audio_path=str(tmp_path / "a.wav"))
        changed = apply_relink(s, new)
        assert s["audio_path"] == new
        assert s["filename"] == "a.flac"
        assert changed == {"audio_path": new, "filename": "a.flac"}

    def test_absolute_style_is_preserved(self, tmp_path):
        new = _touch(str(tmp_path / "a.flac"))
        s = new_sample(audio_path=str(tmp_path / "a.wav"))
        apply_relink(s, new)
        assert os.path.isabs(s["audio_path"])

    def test_relative_style_is_preserved(self, tmp_path):
        """A portable dataset must not get this machine's absolute paths baked in."""
        new = _touch(str(tmp_path / "audio" / "a.flac"))
        s = new_sample(audio_path="audio/a.wav")
        apply_relink(s, new, base_dir=str(tmp_path))
        assert s["audio_path"] == os.path.join("audio", "a.flac")
        assert not os.path.isabs(s["audio_path"])

    def test_relative_style_without_base_dir_stores_absolute(self, tmp_path):
        """Nothing to be relative TO, so guessing would be worse than absolute."""
        new = _touch(str(tmp_path / "audio" / "a.flac"))
        s = new_sample(audio_path="audio/a.wav")
        apply_relink(s, new)
        assert s["audio_path"] == new

    def test_captions_and_lyrics_survive_the_relink(self, tmp_path):
        """They are per-sample JSON keyed by id, so a relink cannot lose them."""
        new = _touch(str(tmp_path / "a.flac"))
        s = new_sample(id="keepme", audio_path=str(tmp_path / "a.wav"),
                       caption="A moody organ vamp.", lyrics="[verse]\nHello")
        apply_relink(s, new)
        assert s["id"] == "keepme"
        assert s["caption"] == "A moody organ vamp."
        assert s["lyrics"] == "[verse]\nHello"

    def test_apply_plan_returns_the_applied_rows(self, tmp_path):
        new = _touch(str(tmp_path / "a.flac"))
        s = new_sample(audio_path=str(tmp_path / "a.wav"))
        plan = plan_relink([s], index_folder(str(tmp_path)))
        applied = apply_plan(plan)
        assert applied == [(s, new, EXACT)]


class TestNeverTouchesTheFilesystem:
    def test_relink_leaves_disk_byte_identical(self, tmp_path):
        """The one property that separates this from execute_disk_rename()."""
        old = str(tmp_path / "a.wav")
        new = _touch(str(tmp_path / "a.flac"), data=b"flac-bytes")
        s = new_sample(filename="a.wav", audio_path=old)

        plan = plan_relink([s], index_folder(str(tmp_path)))
        apply_plan(plan)

        assert not os.path.exists(old), "relink must not create the old path"
        with open(new, "rb") as fh:
            assert fh.read() == b"flac-bytes"
        assert sorted(os.listdir(str(tmp_path))) == ["a.flac"]


