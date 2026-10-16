"""The Track Inspector's tools row: what the owner asked to be taken out.

WHY THIS EXISTS
---------------
Four buttons sat under the Inspector's fields and none of them did the thing its
label promised:

* **🔗 Verify BPM Online** and **🔗 Verify Key Online** opened songbpm.com and
  tunebat.com with the *filename* as the query. A web page's guess about a track
  name is not verification of a value measured from this audio, and the two
  wrapper methods existed only to build those two URLs — so the "services" went
  with them.
* **🎧 A/B Compare Original** did not play anything: it listed the two paths in a
  message box and told the user to open their own media player.
* **⏮ Revert to Original Audio** did write ``audio_path`` back, but there is no
  way to hear which is which from the dataset window, so it was a blind swap.

They are gone, with their handlers. What is pinned here is both halves of that:
the four buttons and their handlers are ABSENT, and — because a deletion is easy
to overdo — the neighbouring Inspector controls and the normalize backup RECORD
(``original_backups``) are still present.
"""
import pytest

pytest.importorskip("PySide6")

import os  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The widgets that were removed, by the name they were stored under.
REMOVED_WIDGETS = (
    "bpm_verify_btn",
    "key_verify_btn",
    "ab_compare_btn",
    "fallback_btn",
)

# Their handlers, and the two web-service wrappers.
REMOVED_METHODS = (
    "open_online_bpm_check",
    "open_online_key_check",
    "ab_compare_playback",
    "fallback_to_original",
)


@pytest.fixture(scope="module")
def manager(qapp, tmp_path_factory):
    """One real window for the module (building it applies the theme)."""
    import config
    import modules.config_store as cs

    settings = tmp_path_factory.mktemp("settings") / "settings.json"
    saved = (config.SETTINGS_PATH, cs.SETTINGS_PATH)
    config.SETTINGS_PATH = settings
    cs.SETTINGS_PATH = settings
    try:
        import dataset_manager

        w = dataset_manager.DatasetManager()
    finally:
        config.SETTINGS_PATH, cs.SETTINGS_PATH = saved
    yield w
    w.hide()
    w.deleteLater()


def test_the_four_unused_inspector_buttons_are_gone(manager):
    """The buttons AND their handlers: a handler with no button is dead code."""
    for name in REMOVED_WIDGETS:
        assert not hasattr(manager, name), f"{name} is still on the Inspector"
    for name in REMOVED_METHODS:
        assert not hasattr(manager, name), f"{name} is still callable"


def test_the_online_verification_services_are_gone_with_them():
    """No call site left that opens songbpm.com or tunebat.com.

    Measured on the source rather than on a widget: the point is that the app no
    longer has a code path to those sites, not merely that a button is hidden.
    The URL prefix is what is checked, not the bare host name, because the
    comment that records WHY they went names the two sites.
    """
    src = open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8").read()
    for url in ("https://songbpm", "https://tunebat"):
        assert url not in src, f"{url} is still wired up"


def test_the_inspector_still_has_its_other_controls(manager):
    """The deletion took four buttons, not their neighbours.

    The lyrics surface sits directly above the removed row and the Tag / Key
    fields below it, so an over-eager deletion would have removed a live control
    here and nothing else would have noticed.
    """
    assert hasattr(manager, "track_tag_input")
    assert hasattr(manager, "inst_check")
    assert hasattr(manager, "caption_text")
    assert hasattr(manager, "bpm_edit") or hasattr(manager, "lyrics_text")


def test_the_normalize_backup_record_is_kept(manager):
    """Removing the buttons must not delete the RECORD of what was normalized.

    ``original_backups`` is no longer read by anything (its only readers were the
    two removed buttons), and a dead-code sweep would flag it. It is deliberately
    kept: it is the app's own note of which original file each normalize wrote
    where, the normalize-done dialog still names that folder, and re-deriving it
    later is impossible. If a future pass wants it gone, the owner should be asked
    — which is what this test is for.
    """
    assert hasattr(manager, "original_backups")
    assert hasattr(manager, "on_file_normalized")
    # And it is still written by the normalize path.
    manager.on_file_normalized("sid-1", "/tmp/orig.wav", "/tmp/norm.wav", 44100, -14.0)
    assert manager.original_backups["sid-1"] == "/tmp/orig.wav"
