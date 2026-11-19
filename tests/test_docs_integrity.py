"""The hand-written guides must not be silently truncated.

An agent once deleted a large block from the Comprehensive LoRA/LoKR guide - the
only doc in the repo about LoRA/LoKR dataset creation rather than song
generation - and .clinerules' "never delete" prose did not prevent it. A test
does. Re-baseline deliberately with:

    .venv/bin/python scripts/doc_manifest.py --update
"""
from __future__ import annotations
import importlib.util, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guides_not_truncated():
    dm = _load("doc_manifest", "scripts/doc_manifest.py")
    problems = dm.compare(dm.load())
    assert not problems, "guide content lost:\n  " + "\n  ".join(problems)
