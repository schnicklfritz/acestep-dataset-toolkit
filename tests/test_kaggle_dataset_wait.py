"""Every worker that uploads an audio dataset must WAIT for it to be ready.

``dataset_create_new`` returns before the dataset version is processed/mounted.
Pushing a kernel inside that window mounts an EMPTY ``/kaggle/input/<slug>``, and
the kernel then finds no audio and captions nothing.

That is exactly what was observed in a real run:

    [moss] /kaggle/input/ace-audio-48bd46 was empty; found audio elsewhere ...

``modules/kaggle.py`` already had ``wait_dataset_ready()`` with a docstring
describing this precise failure — but only **two** of eight workers called it.
This test pins the contract for every worker that uploads audio.
"""
import glob
import os
import re
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = os.path.join(ROOT, "workers")

# The CALL, not the import line, and BOTH upload entry points: the caption worker
# now pushes a new VERSION of an existing dataset (upload_or_update_audio_dataset)
# instead of always creating one. A detector that only knew the first name would
# have silently dropped that worker out of this contract the moment it changed.
UPLOAD_CALL = re.compile(r"upload(?:_or_update)?_audio_dataset\(\s*(?:self\.)?config")


def _uploaders():
    """Worker modules that call an audio-dataset upload (not just import one)."""
    found = []
    for path in sorted(glob.glob(os.path.join(WORKERS, "*.py"))):
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        if UPLOAD_CALL.search(source):
            found.append((os.path.basename(path), source))
    return found


def _upload_call(source):
    """Index of the CALL, not the import line."""
    match = UPLOAD_CALL.search(source)
    return match.start() if match else -1


def test_there_is_at_least_one_uploader():
    assert _uploaders(), "expected worker(s) that upload an audio dataset"


def test_every_uploader_waits_for_the_dataset_before_pushing():
    problems = []
    for name, source in _uploaders():
        upload = _upload_call(source)
        if upload == -1:
            problems.append(f"{name}: could not locate the upload call")
            continue
        wait = source.find("wait_dataset_ready(")
        push = source.find("push_kernel(")
        if wait == -1:
            problems.append(f"{name}: never calls wait_dataset_ready()")
        elif not (upload < wait < push):
            problems.append(
                f"{name}: wait_dataset_ready() is not between the upload and the push"
            )
    assert not problems, problems


def test_the_wait_failure_is_loud_not_silent():
    """An unready dataset must ABORT the run, not push an empty mount."""
    for name, source in _uploaders():
        index = source.find("if not wait_dataset_ready(")
        assert index != -1, f"{name}: no wait guard"
        following = source[index:index + 400]
        assert "raise" in following, f"{name}: wait failure does not abort the run"
        assert "ready" in following, f"{name}: message does not say what went wrong"
        assert "dataset" in following.lower(), name


# ---------------------------------------------------------------------------
# dataset_sources: an empty slug must never reach the kernel metadata
# ---------------------------------------------------------------------------

def test_dataset_sources_drops_empty_slugs():
    """A blank weights setting means 'download from Hugging Face', not ''."""
    from modules.kaggle import dataset_sources

    assert dataset_sources("me/ace-audio-abc", "") == ["me/ace-audio-abc"]
    assert dataset_sources("me/ace-audio-abc", "   ") == ["me/ace-audio-abc"]
    assert dataset_sources("me/ace-audio-abc", None) == ["me/ace-audio-abc"]
    assert dataset_sources("", "") == []
    assert dataset_sources("a/b", "c/d") == ["a/b", "c/d"]


def test_every_caption_kernel_pusher_uses_the_dataset_sources_helper():
    """A hand-built [audio, model] list breaks when the weights setting is cleared.

    MOSS builds its own list, conditionally, and is not a caption-kernel pusher —
    so it is deliberately out of scope here.
    """
    problems = []
    for path in sorted(glob.glob(os.path.join(WORKERS, "*.py"))):
        name = os.path.basename(path)
        if name not in ("caption.py", "spatial.py", "structural.py"):
            continue
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        if '"dataset_sources"' not in source:
            continue
        if "dataset_sources(audio_slug" not in source:
            problems.append(name)
    assert not problems, f"pushers building dataset_sources by hand: {problems}"


# ---------------------------------------------------------------------------
# WHY the wait failed: a permanent 404 must not look like a slow dataset
# ---------------------------------------------------------------------------

class _StatusApi:
    """A KaggleApi whose dataset_status does whatever the test needs."""

    def __init__(self, status=None, error=None):
        self.status = status
        self.error = error

    def dataset_status(self, slug):
        if self.error is not None:
            raise self.error
        return self.status


def _patch_wait(monkeypatch, api):
    from modules import kaggle as kg

    monkeypatch.setattr(kg, "_get_api", lambda config: (api, "someone"))
    # The loop polls on a wall clock; a fake keeps the test instant.
    monkeypatch.setattr(kg, "time", types.SimpleNamespace(sleep=lambda seconds: None))
    return kg


def test_the_wait_reports_why_it_gave_up(monkeypatch):
    """Confirmed against the live API: ``dataset_status`` 404s for a dataset this
    account does not own.

    That is a PERMANENT error, yet the polling loop swallows exceptions so a
    transient hiccup cannot end a 5-minute wait. Without the reason, it is
    indistinguishable from a slow dataset and the run aborts with a bare timeout.
    """
    kg = _patch_wait(monkeypatch, _StatusApi(error=RuntimeError("404 Not Found")))
    reason = []
    ok = kg.wait_dataset_ready({"kaggle_user": "u", "kaggle_key": "k"},
                               "someone/ace-audio-abc", timeout=20, poll_seconds=10,
                               reason=reason)
    assert ok is False
    assert reason == ["last error: RuntimeError: 404 Not Found"]


def test_the_wait_reports_the_last_status_it_saw(monkeypatch):
    """A dataset that stays 'pending' must say so, not just 'timed out'."""
    kg = _patch_wait(monkeypatch, _StatusApi(status="pending"))
    reason = []
    ok = kg.wait_dataset_ready({"kaggle_user": "u", "kaggle_key": "k"},
                               "someone/ace-audio-abc", timeout=20, poll_seconds=10,
                               reason=reason)
    assert ok is False
    assert reason == ["last status: 'pending'"]


def test_a_ready_dataset_returns_true_and_records_nothing(monkeypatch):
    kg = _patch_wait(monkeypatch, _StatusApi(status="ready"))
    reason = []
    ok = kg.wait_dataset_ready({"kaggle_user": "u", "kaggle_key": "k"},
                               "someone/ace-audio-abc", timeout=20, poll_seconds=10,
                               reason=reason)
    assert ok is True
    assert reason == []


def test_the_reason_argument_stays_optional(monkeypatch):
    """Every other caller passes three arguments; none of them may break."""
    kg = _patch_wait(monkeypatch, _StatusApi(error=RuntimeError("nope")))
    assert kg.wait_dataset_ready({"kaggle_user": "u", "kaggle_key": "k"},
                                 "someone/x", timeout=10, poll_seconds=10) is False
