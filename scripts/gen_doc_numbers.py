#!/usr/bin/env python3
"""Regenerate the hard-limits block in the master annotation guide from
modules/caption_spec.py, so the prose cannot drift from the constants the code
actually enforces.

    .venv/bin/python scripts/gen_doc_numbers.py

Exit codes: 0 = block written or already current, 0 with --check = current,
1 with --check = drifted (used by tests/test_docs_sync.py).
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "ACE_Step_1.5_Master_Annotation_Guide.md")
BEGIN = "<!-- BEGIN GENERATED: caption_spec -->"
END = "<!-- END GENERATED: caption_spec -->"


def load_spec():
    path = os.path.join(ROOT, "modules", "caption_spec.py")
    spec = importlib.util.spec_from_file_location("caption_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render() -> str:
    cs = load_spec()
    rows = [
        ("Caption length", "%d\u2013%d words" % (cs.CAPTION_MIN_WORDS, cs.CAPTION_MAX_WORDS)),
        ("Caption hard cap", "%d characters" % cs.CAPTION_MAX_CHARS),
        ("Front-loaded tags", "%d\u2013%d, hard max %d" % (cs.TAG_MIN, cs.TAG_MAX, cs.TAG_HARD_MAX)),
        ("Facts after the tag block", "8\u201316"),
        ("Descriptors per bracket", "max %d" % cs.MAX_DESCRIPTORS),
        ("Lyric line syllables",
         "%d\u2013%d, flag over %d" % (cs.LYRIC_MIN_SYLLABLES, cs.LYRIC_MAX_SYLLABLES, cs.LYRIC_FLAG_SYLLABLES)),
    ]
    out = [
        BEGIN,
        "",
        "## Hard limits (generated \u2014 do not edit by hand)",
        "",
        "Rendered from `modules/caption_spec.py` by `scripts/gen_doc_numbers.py`.",
        "If these disagree with the prose above, the prose is wrong.",
        "",
        "| limit | value |",
        "| --- | --- |",
    ]
    out += ["| %s | %s |" % (k, v) for k, v in rows]
    out += ["", "Density rule, verbatim:", "", "> " + cs.CAPTION_DENSITY_RULE, "", END]
    return "\n".join(out)


def main(argv):
    body = render()
    with open(DOC, encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        current = BEGIN + rest.split(END, 1)[0] + END
        if current.strip() == body.strip():
            if "--check" in argv:
                print("docs sync: OK")
                return 0
            print("docs sync: already current")
            return 0
        if "--check" in argv:
            print("docs sync: DRIFT \u2014 run scripts/gen_doc_numbers.py")
            return 1
        new = head + body + tail
    else:
        if "--check" in argv:
            print("docs sync: DRIFT \u2014 generated block missing")
            return 1
        new = text.rstrip() + "\n\n" + body + "\n"
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("docs sync: block written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
