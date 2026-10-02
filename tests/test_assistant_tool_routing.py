"""The Assistant's action tools: spec and executor must agree.

A tool the model can see but that executes as "Unknown tool" is the worst
failure mode this feature has: the model claims the task is done and nothing
moves. So the headline test walks every declared tool name and asserts the
executor actually handles it — not just the happy path of the ones somebody
remembered to wire up.
"""
import pytest

pytest.importorskip("PySide6")

from modules.dataset_schema import new_dataset, new_sample  # noqa: E402
from workers.assistant import ACTION_TOOLS, ASSISTANT_TOOLS  # noqa: E402

BAD = "Unknown tool: "


@pytest.fixture(scope="module")
def manager():
    """One DatasetManager for the whole module.

    Constructing it costs seconds (toolbars, workers, playback, docks), and this
    file has ~20 tests — building it per test pushes the file past any sane
    timeout. The dataset and undo stacks are reset per test by ``_reset``.
    The settings path is redirected to a temp dir so a test can never touch the
    user's real settings.json.
    """
    import tempfile
    from pathlib import Path

    from PySide6.QtWidgets import QApplication

    mp = pytest.MonkeyPatch()
    tmp = Path(tempfile.mkdtemp(prefix="assistant_tools_"))
    import config
    import modules.config_store as cs

    mp.setattr(config, "SETTINGS_PATH", tmp / "settings.json")
    mp.setattr(cs, "SETTINGS_PATH", tmp / "settings.json")
    import dataset_manager

    app = QApplication.instance() or QApplication([])
    w = dataset_manager.DatasetManager()
    w._test_tmp = tmp
    yield w
    w.hide()
    w.deleteLater()
    mp.undo()
    del app


@pytest.fixture(autouse=True)
def _reset(manager, tmp_path):
    """Fresh dataset + empty undo stacks before every test."""
    manager.dataset = new_dataset(name="assistant")
    manager.dataset["samples"] = [
        new_sample(filename="a.mp3", audio_path=str(tmp_path / "a.mp3")),
        new_sample(filename="b.wav", audio_path=str(tmp_path / "b.wav")),
    ]
    manager.undo_stack.clear()
    manager.redo_stack.clear()
    manager.refresh_table()
    yield


def _names():
    return [t["function"]["name"] for t in ACTION_TOOLS]


def test_action_tools_are_declared():
    assert _names() == [
        "set_track_metadata", "write_caption", "import_lyrics", "find_gaps",
        "create_virtual_dataset", "normalize_audio", "stage_temp_mp3",
    ]


def test_action_tools_are_visible_to_the_model():
    exposed = [t["function"]["name"] for t in ASSISTANT_TOOLS]
    assert set(_names()) <= set(exposed)


@pytest.mark.parametrize("name", _names())
def test_every_declared_tool_is_actually_handled(manager, name):
    """The regression: a tool in the spec with no branch in the executor."""
    out = manager.execute_assistant_tool(name, {"tracks": ["nope.wav"]})
    assert BAD not in out, f"{name} is declared but not handled"
    assert "Unknown" not in out


def test_find_gaps_reports_the_dataset(manager):
    out = manager.execute_assistant_tool("find_gaps", {})
    assert "a.mp3" in out and "b.wav" in out
    assert "SUMMARY" in out


def test_set_track_metadata_writes_and_snapshots(manager):
    depth = len(manager.undo_stack)
    out = manager.execute_assistant_tool("set_track_metadata", {
        "tracks": ["a.mp3"], "field": "genre", "value": "Doom",
    })
    assert "genre = Doom" in out
    assert manager.dataset["samples"][0]["genre"] == "Doom"
    # Undo must exist for the change, or a model mistake is permanent.
    assert len(manager.undo_stack) == depth + 1


def test_a_refusal_does_not_push_an_undo_entry(manager):
    depth = len(manager.undo_stack)
    original = manager.dataset["samples"][0]["audio_path"]
    out = manager.execute_assistant_tool("set_track_metadata", {
        "tracks": ["a.mp3"], "field": "audio_path", "value": "/etc/passwd",
    })
    assert "Refused" in out
    assert manager.dataset["samples"][0]["audio_path"] == original
    assert len(manager.undo_stack) == depth


def test_set_track_metadata_rejects_an_unknown_track(manager):
    depth = len(manager.undo_stack)
    out = manager.execute_assistant_tool("set_track_metadata", {
        "tracks": ["ghost.mp3"], "field": "genre", "value": "x",
    })
    assert "No tracks matched" in out
    assert len(manager.undo_stack) == depth


# ---------------------------------------------------------------------------
# Write + validate in one call
# ---------------------------------------------------------------------------
GOOD_CAPTION = (
    "hard rock, drums, bass, male vocal, raw production, 1970s analog. "
    "It opens sparse and builds steadily through the verse. The chorus hits "
    "hard and the track finishes on a peak."
)


def test_write_caption_validates_what_it_just_wrote(manager):
    out = manager.execute_assistant_tool("write_caption", {
        "tracks": ["a.mp3"], "caption": GOOD_CAPTION,
    })
    assert "Schema: conforms" in out
    assert manager.dataset["samples"][0]["caption"] == GOOD_CAPTION


def test_write_caption_reports_violations_so_the_model_can_fix_them(manager):
    out = manager.execute_assistant_tool("write_caption", {
        "tracks": ["a.mp3"], "caption": "it was a track with many years.",
    })
    assert "Schema issues:" in out
    assert "front-loaded keyword" in out


def test_write_caption_refuses_empty_text(manager):
    depth = len(manager.undo_stack)
    out = manager.execute_assistant_tool("write_caption", {
        "tracks": ["a.mp3"], "caption": "  ",
    })
    assert "Refused" in out
    assert manager.dataset["samples"][0]["caption"] == ""
    assert len(manager.undo_stack) == depth


# ---------------------------------------------------------------------------
# Lyrics + virtual tracks
# ---------------------------------------------------------------------------
def test_import_lyrics_fills_the_export_field(manager):
    out = manager.execute_assistant_tool("import_lyrics", {
        "tracks": ["a.mp3"], "lyrics": "[Verse]\nRUN AWAY",
    })
    assert "Imported" in out
    assert manager.dataset["samples"][0]["formatted_lyrics"] == "[Verse]\nRUN AWAY"
    assert manager.dataset["samples"][0]["lyrics"] == "[Verse]\nRUN AWAY"


def test_create_virtual_dataset_extends_the_dataset(manager):
    depth = len(manager.undo_stack)
    out = manager.execute_assistant_tool("create_virtual_dataset", {
        "tracks": [{"genre": "Sludge"}, {"genre": "Doom"}], "name": "Concepts",
    })
    assert "Created 2 virtual track(s)" in out
    assert len(manager.dataset["samples"]) == 4
    assert manager.dataset["metadata"]["name"] == "Concepts"
    assert len(manager.undo_stack) == depth + 1


def test_create_virtual_dataset_refuses_an_empty_list(manager):
    depth = len(manager.undo_stack)
    out = manager.execute_assistant_tool("create_virtual_dataset", {"tracks": []})
    assert "Refused" in out
    assert len(manager.undo_stack) == depth


# ---------------------------------------------------------------------------
# ffmpeg-facing tools: plan or start, never block the GUI
# ---------------------------------------------------------------------------
def test_stage_temp_mp3_returns_a_plan_and_runs_nothing(manager):
    calls = []
    out = manager.execute_assistant_tool("stage_temp_mp3", {
        "tracks": ["a.mp3"], "target_dir": "/tmp/stage", "bitrate": "192k",
    })
    assert "Staging plan (not executed)" in out
    assert "ffmpeg" in out and "/tmp/stage/a.mp3" in out
    assert calls == []          # nothing ran: the GUI must never block on ffmpeg


def test_stage_temp_mp3_needs_a_target_dir(manager):
    out = manager.execute_assistant_tool("stage_temp_mp3", {"tracks": ["a.mp3"]})
    assert "Refused" in out and "target_dir" in out


def test_normalize_audio_needs_a_target_dir(manager):
    out = manager.execute_assistant_tool("normalize_audio", {"tracks": ["a.mp3"]})
    assert "Refused" in out and "target_dir" in out


def test_normalize_audio_refuses_when_there_is_no_audio_on_disk(manager):
    out = manager.execute_assistant_tool("normalize_audio", {
        "tracks": ["a.mp3"], "target_dir": "/tmp/does-not-matter",
    })
    assert "No selected track has audio on disk" in out


def test_normalize_audio_starts_the_real_worker(manager, tmp_path, monkeypatch):
    """Same worker as the 🎚 button — but no modal dialogs, which the model
    has no way to answer."""
    src = tmp_path / "a.mp3"
    src.write_bytes(b"fake audio")
    manager.dataset["samples"][0]["audio_path"] = str(src)

    class _FakeWorker:
        def __init__(self, samples, **kwargs):
            self.samples = samples
            self.kwargs = kwargs
            self.progress = _FakeWorker._sig()
            self.file_normalized = _FakeWorker._sig()
            self.all_done = _FakeWorker._sig()
            self.error_occurred = _FakeWorker._sig()
            self.started = False

        class _sig:
            def __init__(self):
                self.connected = []

            def connect(self, fn):
                self.connected.append(fn)

            def emit(self, *a):
                pass

        def start(self):
            self.started = True

    monkeypatch.setattr("dataset_manager.DspNormalizerWorker", _FakeWorker)

    target = str(tmp_path / "out")
    out = manager.execute_assistant_tool("normalize_audio", {
        "tracks": ["a.mp3"], "target_dir": target, "target_lufs": -16.0,
    })
    assert "Started EBU R128 normalization of 1 track(s) at -16.0 LUFS" in out
    assert "originals" in out
    assert "still running" in out
    worker = manager.active_worker
    assert worker.started and worker.kwargs["target_lufs"] == -16.0
    assert (tmp_path / "out").is_dir()
    # The instruction the model reads must not claim the job finished.
    assert "never overwritten" in out

