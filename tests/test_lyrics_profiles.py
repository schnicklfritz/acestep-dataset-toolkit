"""Lyrics profiles: the three-layer master/profile merge and file storage."""
import json

import pytest

from modules.lyrics_normalizer import DEFAULT_CONTRACTIONS
from modules.lyrics_profiles import (
    delete_profile,
    list_profiles,
    load_profile,
    merge_rules,
    save_profile,
)


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    """Point the profile store at a temp dir so tests never touch the repo."""
    monkeypatch.setattr(
        "modules.lyrics_profiles.profile_dir", lambda base_dir=None: str(tmp_path)
    )
    return tmp_path


class TestMergeRules:
    """master (always on) + profile: profile wins, master-only entries survive."""

    def test_profile_overrides_master(self):
        c, _ = merge_rules({"you're": "your"}, {"you're": "yore"})
        assert c["you're"] == "yore"

    def test_master_only_entries_still_apply(self):
        c, _ = merge_rules({"she'll": "sheel"}, {"gonna": "gunna"})
        assert c["she'll"] == "sheel"

    def test_profile_only_entries_are_added(self):
        c, _ = merge_rules({"she'll": "sheel"}, {"gonna": "gunna"})
        assert c["gonna"] == "gunna"

    def test_ing_exceptions_are_unioned(self):
        _, ing = merge_rules({}, {}, {"ring"}, {"morning"})
        assert ing == {"ring", "morning"}

    def test_empty_inputs_are_safe(self):
        c, ing = merge_rules(None, None, None, None)
        assert c == {} and ing == set()


class TestProfileStorage:
    def test_save_then_load_round_trips(self, profile_root):
        save_profile("test_band", {"gonna": "gunna"}, ["morning"], ing_to_in=True)
        p = load_profile("test_band")
        assert p["contractions"] == {"gonna": "gunna"}
        assert p["ing_exceptions"] == ["morning"]
        assert p["ing_to_in"] is True

    def test_list_profiles_excludes_non_json(self, profile_root):
        save_profile("band_a", {}, [])
        (profile_root / "notes.txt").write_text("ignore me")
        assert list_profiles() == ["band_a"]

    def test_missing_profile_returns_none(self, profile_root):
        assert load_profile("nope") is None

    def test_delete_removes_the_file(self, profile_root):
        save_profile("temp", {}, [])
        assert delete_profile("temp") is True
        assert list_profiles() == []

    def test_delete_missing_is_false(self, profile_root):
        assert delete_profile("nope") is False

    def test_unsafe_names_are_slugged(self, profile_root):
        # A name with path separators must not escape the profiles folder.
        save_profile("../../evil", {}, [])
        files = [p.name for p in profile_root.iterdir()]
        assert all("/" not in f and ".." not in f for f in files)
        # Leading/trailing dots and underscores are stripped by the slug.
        assert list_profiles() == ["evil"]

    def test_file_is_written_inside_the_profile_dir(self, profile_root):
        save_profile("../../evil", {}, [])
        assert (profile_root / "evil.json").exists()

    def test_corrupt_profile_is_ignored_not_crashed(self, profile_root):
        (profile_root / "broken.json").write_text("{not json")
        assert load_profile("broken") is None
        assert "broken" in list_profiles()

    def test_saved_file_is_human_readable_json(self, profile_root):
        save_profile("band", {"gonna": "gunna"}, ["morning"])
        data = json.loads((profile_root / "band.json").read_text())
        assert data["contractions"] == {"gonna": "gunna"}
        assert data["name"] == "band"


class TestDefaultTable:
    def test_master_defaults_are_available(self):
        assert DEFAULT_CONTRACTIONS["she'll"] == "sheel"
        assert DEFAULT_CONTRACTIONS["i'll"] == "aisle"
        assert DEFAULT_CONTRACTIONS["he'll"] == "heel"
