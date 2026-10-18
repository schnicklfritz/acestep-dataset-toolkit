"""Offline validation of the MMS_FA alignment kernel and its Kaggle worker.

A real alignment run costs a GPU session and ~1.1 GB of weights, so the parts
that decide what gets written are pinned here instead:

  * the shout classifier's threshold is PER TRACK (mean + K*sigma), so a quiet
    mix and a loud master are judged the same way;
  * Fix A bridges a short quiet gap inside a shouted phrase, Fix B lifts a word
    that is just under the threshold when its phrase is shouting;
  * the sustained-note boost reaches the decision. The hand-run script assigned
    ``raw_shout`` twice and threw the boosted level away; these tests fail if
    that ever comes back;
  * line-level segments are what ``export_lrc`` reads, and a section marker is
    not a lyric line;
  * the worker stages lyrics keyed by BOTH filename and stem, and turns the
    kernel's error entry into a raised message rather than an empty success.

Torch is NOT installed on this machine (the kernel runs on Kaggle), so the
script is imported with a minimal stub; nothing here touches the model.
"""
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KERNEL_PATH = os.path.join(ROOT, "kernels", "alignment_kernel.py")


@pytest.fixture(scope="module")
def kernel():
    stubbed = []
    for name in ("torch", "torchaudio"):
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            sys.modules[name] = module
            stubbed.append(name)
    torch = sys.modules["torch"]
    if not hasattr(torch, "cuda"):
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    if not hasattr(torch, "inference_mode"):
        torch.inference_mode = contextlib.nullcontext
    torchaudio = sys.modules["torchaudio"]
    if not hasattr(torchaudio, "pipelines"):
        torchaudio.pipelines = types.SimpleNamespace(MMS_FA=None)

    spec = importlib.util.spec_from_file_location(
        "alignment_kernel_under_test", KERNEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for name in stubbed:
        sys.modules.pop(name, None)


def _records(dbs, duration=0.3):
    return [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + duration,
             "duration": duration, "db": db}
            for i, db in enumerate(dbs)]


# ---------------------------------------------------------------------------
# The script itself
# ---------------------------------------------------------------------------

def test_the_worker_placeholder_matches_the_worker():
    """``run_kaggle_alignment`` substitutes the string the kernel declares."""
    with open(KERNEL_PATH, encoding="utf-8") as handle:
        source = handle.read()
    assert 'AUDIO_DATASET_PATH = "{{AUDIO_DATASET_PATH}}"' in source

    with open(os.path.join(ROOT, "workers", "kaggle_alignment.py"),
              encoding="utf-8") as handle:
        worker_source = handle.read()
    assert '"{{AUDIO_DATASET_PATH}}"' in worker_source


def test_the_kernel_writes_the_file_the_worker_downloads(kernel):
    assert kernel.OUT_PATH.endswith("alignment_out.json")


def test_section_markers_and_blank_lines_are_not_aligned(kernel):
    text = "[Verse 1]\n  first line  \n\n[Chorus]\nsecond line\n"
    assert kernel._lyric_lines(text) == ["first line", "second line"]


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

def test_the_threshold_is_relative_to_each_track(kernel):
    """Same relative pattern, 8 dB apart: the same words are shouted."""
    quiet = [-28, -28, -18, -18, -28, -28]
    loud = [-20, -20, -10, -10, -20, -20]
    a, b = _records(quiet), _records(loud)
    kernel._classify(a)
    kernel._classify(b)
    assert [r["shouted"] for r in a] == [r["shouted"] for r in b]
    assert [r["shouted"] for r in a] == [False, False, True, True, False, False]


def test_fix_a_bridges_a_one_word_gap_inside_a_phrase(kernel):
    records = _records([-17, -20, -17])
    kernel._classify(records)
    assert [r["raw_shout"] for r in records] == [True, False, True]
    assert records[1]["bridged"] is True
    assert records[1]["phrase_elevated"] is False
    assert records[1]["final_word"] == records[1]["word"].upper()


def test_fix_a_does_not_bridge_a_long_gap(kernel):
    """Three quiet words in a row is a return to normal delivery."""
    records = _records([-17, -30, -30, -30, -17])
    kernel._classify(records)
    assert [r["shouted"] for r in records[:4]] == [True, False, False, False]


def test_fix_b_lifts_a_near_threshold_word_in_a_shouted_phrase(kernel):
    """The phrase window must reach PAST the bridge gap, or this is dead code.

    A quiet word between two shouts is Fix A's job; Fix B exists for a word just
    under the threshold at the EDGE of a shouted phrase, where the gap is too
    long to bridge.
    """
    assert kernel.PHRASE_WINDOW // 2 > kernel.BRIDGE_GAP

    records = _records([-17, -17, -17, -21.5, -30, -30, -30])
    kernel._classify(records)
    middle = records[3]
    assert middle["raw_shout"] is False
    assert middle["bridged"] is False          # three quiet words: not bridgeable
    assert middle["delta"] >= -kernel.PHRASE_DB
    assert middle["phrase_elevated"] is True
    assert middle["shouted"] is True


def test_fix_b_refuses_a_word_far_below_the_threshold(kernel):
    records = _records([-17, -17, -17, -30, -30, -30, -17])
    kernel._classify(records)
    for index in (3, 4, 5):
        assert records[index]["delta"] < -kernel.PHRASE_DB
        assert records[index]["shouted"] is False
        assert records[index]["final_word"] == records[index]["word"]


def test_a_flat_track_shouts_nothing(kernel):
    """No dynamic range means nothing can be singled out -- and a relative
    threshold with zero spread would otherwise put the WHOLE lyric in capitals."""
    records = _records([-40, -39, -41, -40])
    kernel._classify(records)
    for record in records:
        assert record["shouted"] is False
        assert record["final_word"] == record["word"]
        assert record["tag"] == "quiet"


# ---------------------------------------------------------------------------
# The regression: the sustained-note boost must reach the DECISION
# ---------------------------------------------------------------------------

def test_the_sustained_note_boost_lands_in_effective_db(kernel):
    short = _records([-20], duration=0.2)[0]
    long = _records([-20], duration=1.0)[0]
    kernel._classify([short, long])
    assert short["effective_db"] == short["db"]
    assert long["effective_db"] == round(long["db"] + kernel.SUSTAIN_DB, 2)


def test_the_sustained_note_boost_can_change_the_verdict(kernel, monkeypatch):
    """The hand-run bug: compute the boost, then decide from the plain level.

    The long word is shouted WITH the boost and quiet WITHOUT it, so a
    ``raw_shout`` computed from ``r["db"]`` fails the second half.
    """
    dbs = [-17] * 5 + [-24] * 4 + [-20]
    durations = [0.3] * 9 + [1.0]

    def build():
        return [{"word": f"w{i}", "start": i, "end": i + 1,
                 "duration": durations[i], "db": dbs[i]} for i in range(len(dbs))]

    boosted = build()
    kernel._classify(boosted)
    assert boosted[-1]["shouted"] is True
    assert boosted[-1]["raw_shout"] is True      # the BOOST decided it

    monkeypatch.setattr(kernel, "SUSTAIN_DB", 0.0)
    plain = build()
    kernel._classify(plain)
    assert plain[-1]["raw_shout"] is False
    # Fix B must not rescue it either: its neighbours are quiet.
    assert plain[-1]["phrase_elevated"] is False
    assert plain[-1]["shouted"] is False


# ---------------------------------------------------------------------------
# The worker: staging, result selection, and the failure path
# ---------------------------------------------------------------------------

def _patch_kaggle(monkeypatch, payload, captured):
    """Replace the network half of modules.kaggle with recorders."""
    import modules.kaggle as kg

    def fake_upload(config, directory):
        captured["files"] = sorted(os.listdir(directory))
        with open(os.path.join(directory, "lyrics.json"), encoding="utf-8") as handle:
            captured["lyrics"] = json.load(handle)
        captured["uploaded"] = True
        return "someone/ace-audio-test"

    def fake_download(config, kernel_slug, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "alignment_out.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle)

    monkeypatch.setattr(kg, "upload_audio_dataset", fake_upload)
    monkeypatch.setattr(kg, "wait_dataset_ready", lambda *a, **k: True)
    monkeypatch.setattr(kg, "push_kernel", lambda *a, **k: None)
    monkeypatch.setattr(kg, "wait_kernel_done", lambda *a, **k: None)
    monkeypatch.setattr(kg, "download_kernel_output", fake_download)
    monkeypatch.setattr(kg, "preflight_kaggle", lambda *a, **k: None)


def test_the_worker_stages_lyrics_under_name_and_stem(tmp_path, monkeypatch):
    from workers.kaggle_alignment import run_kaggle_alignment

    audio = tmp_path / "My Song.flac"
    audio.write_bytes(b"not really audio")
    payload = {"tracks": {"My Song.flac": {
        "segments": [{"start": 1.5, "text": "HELLO world"}],
        "words": [], "lyrics": "HELLO world", "shout_ratio": 0.5}},
        "errors": {}}
    captured = {}
    _patch_kaggle(monkeypatch, payload, captured)

    result = run_kaggle_alignment(str(audio), "hello world", {})

    assert result["segments"] == [{"start": 1.5, "text": "HELLO world"}]
    assert captured["files"] == ["My Song.flac", "lyrics.json"]
    # Both keys: the kernel prefers a vocal stem, whose filename differs.
    assert captured["lyrics"]["My Song.flac"] == "hello world"
    assert captured["lyrics"]["My Song"] == "hello world"


def test_a_kernel_error_is_raised_not_returned_as_empty(tmp_path, monkeypatch):
    from workers.kaggle_alignment import run_kaggle_alignment

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"x")
    payload = {"tracks": {}, "errors": {"song.wav": "RuntimeError: no lyric words to align"}}
    _patch_kaggle(monkeypatch, payload, {})

    with pytest.raises(RuntimeError, match="no lyric words to align"):
        run_kaggle_alignment(str(audio), "la", {})


def test_nothing_is_uploaded_without_lyrics(tmp_path, monkeypatch):
    from workers.kaggle_alignment import run_kaggle_alignment

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"x")
    captured = {}
    _patch_kaggle(monkeypatch, {"tracks": {}}, captured)

    with pytest.raises(ValueError):
        run_kaggle_alignment(str(audio), "   ", {})
    assert "uploaded" not in captured


def test_the_worker_class_exposes_the_signals_the_page_connects():
    from workers.kaggle_alignment import KaggleAlignmentWorker

    for name in ("progress", "finished_ok", "failed"):
        assert hasattr(KaggleAlignmentWorker, name)



