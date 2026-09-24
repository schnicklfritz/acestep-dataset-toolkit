"""Tests for scripts/build_caption_notebook.py.

The generated cell goes to Kaggle, where a mistake costs a GPU session. The
load-bearing checks are that no placeholder survives and that the injected
auto-resolve block sits BEFORE ``audio_files`` is built — referencing it earlier
would be a NameError that ``ast.parse`` cannot see.
"""
import ast
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.build_caption_notebook as bcn                       # noqa: E402


@pytest.fixture
def notebook(tmp_path):
    path = tmp_path / "cell.ipynb"
    bcn.build(str(path), tag="acdc")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _cell_source(nb):
    return "".join(nb["cells"][1]["source"])


def test_notebook_has_a_markdown_header_and_one_code_cell(notebook):
    assert [c["cell_type"] for c in notebook["cells"]] == ["markdown", "code"]
    assert notebook["nbformat"] == 4


def test_no_placeholder_survives(notebook):
    source = _cell_source(notebook)
    assert "{{" not in source and "}}" not in source


def test_generated_cell_is_valid_python(notebook):
    ast.parse(_cell_source(notebook))


def test_gpu_and_internet_are_enabled(notebook):
    """The kernel needs 2 GPUs (device_map balanced) and pip access to GitHub."""
    kaggle = notebook["metadata"]["kaggle"]
    assert kaggle["isGpuEnabled"] is True
    assert kaggle["isInternetEnabled"] is True
    assert kaggle["accelerator"] != "none"


def test_cell_carries_the_ace_step_schema(notebook):
    source = _cell_source(notebook)
    assert "5-12 comma-separated keywords" in source
    assert "SYSTEM_PROMPT" in source
    assert "repetition_penalty=REPETITION_PENALTY" in source


def test_auto_resolve_is_injected_before_audio_files_is_built(notebook):
    source = _cell_source(notebook)
    assert "AUTO-RESOLVED AUDIO FOLDER" in source
    # The injected block must run BEFORE the file list is built, whichever call
    # builds it. (The kernel gained its own _walk_for_audio() fallback after a real
    # run mounted the dataset at /kaggle/input/datasets/<owner>/<slug>/ and the
    # kernel found 0 files; the notebook injection is now belt-and-braces, and this
    # assertion is what keeps it ordered correctly.)
    assert (source.index("AUTO-RESOLVED")
            < source.index("audio_files = _walk_for_audio(AUDIO_FOLDER)"))


def test_the_trigger_tag_is_applied(notebook):
    assert 'CUSTOM_TAG = "acdc"' in _cell_source(notebook)


def test_build_refuses_to_emit_an_unfilled_placeholder(monkeypatch, tmp_path):
    """{{X}} is valid Python, so ast.parse alone would not catch this."""
    bad = tmp_path / "kernel.py"
    bad.write_text("SUPPORTED_FORMATS = {'.mp3'}\nX = {{UNKNOWN}}\n", encoding="utf-8")
    monkeypatch.setattr(bcn, "KERNEL", str(bad))
    with pytest.raises(SystemExit, match="unfilled placeholder"):
        bcn.build(str(tmp_path / "out.ipynb"))


def test_a_plain_py_copy_is_also_emitted(tmp_path):
    """An .ipynb is JSON; the .py is what actually gets pasted into Kaggle."""
    path = tmp_path / "cell.ipynb"
    bcn.build(str(path), tag="acdc")
    py_path = tmp_path / "cell.py"
    assert py_path.is_file()
    text = py_path.read_text(encoding="utf-8")
    ast.parse(text)
    assert "{{" not in text
    assert "Run All" in text
    assert bcn.DEFAULT_PY.endswith("kaggle_caption_cell.py")


def test_a_missing_anchor_is_reported_not_ignored(monkeypatch, tmp_path):
    bad = tmp_path / "kernel.py"
    bad.write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(bcn, "KERNEL", str(bad))
    with pytest.raises(SystemExit, match="anchor not found"):
        bcn.build(str(tmp_path / "out.ipynb"))
