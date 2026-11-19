"""The master annotation guide must state the numbers the code enforces.

`modules/caption_spec.py` claims to be the single source of truth, but the guide
is prose maintained by hand. It has already drifted once (a stale copy on the
Desktop still mandates the retired 2-3 sentence flow narrative). This test makes
the drift loud instead of silent.
"""
from __future__ import annotations

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "ACE_Step_1.5_Master_Annotation_Guide.md")


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc() -> str:
    with open(DOC, encoding="utf-8") as fh:
        return fh.read()


def test_generated_block_matches_constants():
    gen = _load("gen_doc_numbers", "scripts/gen_doc_numbers.py")
    text = _doc()
    assert gen.BEGIN in text, "generated block missing from the guide"
    assert gen.END in text, "generated block missing from the guide"
    start = text.index(gen.BEGIN)
    end = text.index(gen.END, start) + len(gen.END)
    assert text[start:end].strip() == gen.render().strip(), (
        "docs/ACE_Step_1.5_Master_Annotation_Guide.md drifted from "
        "modules/caption_spec.py - run .venv/bin/python scripts/gen_doc_numbers.py"
    )


def test_density_rule_quoted_verbatim():
    cs = _load("caption_spec", "modules/caption_spec.py")
    assert cs.CAPTION_DENSITY_RULE in _doc(), (
        "the guide must quote CAPTION_DENSITY_RULE verbatim so the prompt text "
        "and the prose cannot diverge"
    )


def test_word_budget_is_stated():
    cs = _load("caption_spec", "modules/caption_spec.py")
    text = _doc()
    assert str(cs.CAPTION_MAX_WORDS) in text, "caption word cap not stated in the guide"
    assert str(cs.CAPTION_MAX_CHARS) in text, "caption character cap not stated in the guide"
    assert str(cs.MAX_DESCRIPTORS) in text, "descriptor limit not stated in the guide"
