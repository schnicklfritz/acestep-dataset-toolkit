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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = os.path.join(ROOT, "workers")


def _uploaders():
    """Worker modules that call upload_audio_dataset (not just import it)."""
    found = []
    for path in sorted(glob.glob(os.path.join(WORKERS, "*.py"))):
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        if "upload_audio_dataset(conf" in source or \
                "upload_audio_dataset(self.config" in source:
            found.append((os.path.basename(path), source))
    return found


def _upload_call(source):
    """Index of the CALL, not the import line."""
    for needle in ("upload_audio_dataset(self.config,", "upload_audio_dataset(config,"):
        index = source.find(needle)
        if index != -1:
            return index
    return -1


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
