"""Offline validation of kernels/caption_kernel.py's diagnostics.

A Kaggle run costs 10+ minutes and a multi-GB download, and the kernel used to be
SILENT for its first ~5 minutes (pip install, then the model download) — so
"nothing is happening" and "it is working" were indistinguishable from outside.
That is a real failure mode: the user reported exactly that ("it won't start, no
errors").

These tests pin the two fixes:

1. A preflight that prints python, GPU, INTERNET reachability, disk and the
   resolved audio folder BEFORE anything slow happens.
2. pip failures are REPORTED. ``check=False`` used to swallow them, which is how
   an Internet-off session failed with no error at all.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_caption_notebook import fill_kernel                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = os.path.join(ROOT, "kernels", "caption_kernel.py")


def _source():
    with open(KERNEL, encoding="utf-8") as fh:
        return fh.read()


def test_preflight_runs_before_the_install():
    """The whole point is that it prints before anything slow or silent."""
    src = _source()
    assert src.index("_preflight()") < src.index("_install()")


def test_preflight_reports_gpu_internet_and_disk():
    src = _source()
    assert "[caption] gpu" in src
    assert "[caption] internet" in src
    assert "socket.create_connection" in src
    assert "[caption] free disk" in src


def test_preflight_warns_loudly_about_a_missing_gpu_or_internet():
    src = _source()
    assert "!! NO GPU" in src
    assert "NOT REACHABLE" in src
    assert "Turn Internet ON" in src


def test_pip_failures_are_reported_not_swallowed():
    """check=False alone made an Internet-off session fail with no error."""
    src = _source()
    assert "proc.returncode != 0" in src
    assert "pip FAILED" in src


def test_audio_folder_and_file_count_are_printed():
    src = _source()
    assert "[caption] audio folder" in src
    assert "[caption] audio files" in src
    assert "NO AUDIO FOUND" in src
    assert "only ONE file found" in src


def test_model_source_and_load_time_are_printed():
    src = _source()
    assert "[caption] model source" in src
    assert "model loaded" in src
    assert "time.time() - _load_started" in src


def test_every_diagnostic_is_flushed():
    """Kaggle buffers stdout; an unflushed print looks like a hang."""
    tree = ast.parse(fill_kernel(_source()))
    diagnostics = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "print"
        and "[caption]" in ast.unparse(node)
    ]
    assert len(diagnostics) >= 12
    missing = [call for call in diagnostics if "flush=True" not in call]
    assert not missing, missing


def test_the_schema_reaches_the_kernel():
    """The system turn must carry the ACE-Step 1.5XL schema."""
    src = _source()
    assert "SYSTEM_PROMPT" in src
    assert '{"role": "system"' in src
