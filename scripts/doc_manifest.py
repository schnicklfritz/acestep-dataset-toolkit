#!/usr/bin/env python3
"""Manifest guard for the hand-written guides in docs/.

The Comprehensive LoRA/LoKR guide is the only document that covers LoRA/LoKR
dataset creation rather than song generation, and it has been silently gutted by
an agent before. "Never delete" in .clinerules did not stop that; this does.

    .venv/bin/python scripts/doc_manifest.py            # check
    .venv/bin/python scripts/doc_manifest.py --update   # re-baseline intentionally
"""
from __future__ import annotations
import hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
MANIFEST = os.path.join(DOCS, ".guide_manifest.json")
GUIDES = [
    "ACE_Step_1.5_Master_Annotation_Guide.md",
    "Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md",
    "LoRA_LoKR_Creation_Guide.md",
    "descriptor_reference.md",
    # The index/context protocol text. It was lost once already (recovered from
    # a paste outside the repo), so it is guarded like a guide: truncation is
    # loud instead of silent. v1 is the frozen original the living file is
    # annotated against; it must never change.
    "gitdiff-plus.md",
    "gitdiff-plus-v1-original.md",
]


def scan(name: str) -> dict:
    path = os.path.join(DOCS, name)
    text = open(path, encoding="utf-8").read()
    lines = text.splitlines()
    heads = [ln.strip() for ln in lines if re.match(r"^#{1,3} ", ln)]
    return {
        "lines": len(lines),
        "chars": len(text),
        "headings": heads,
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def load() -> dict:
    return json.load(open(MANIFEST, encoding="utf-8"))


def compare(man: dict) -> list[str]:
    problems = []
    for name in GUIDES:
        if name not in man:
            continue
        cur = scan(name)
        base = man[name]
        missing = [h for h in base["headings"] if h not in cur["headings"]]
        if missing:
            problems.append("%s: %d heading(s) gone, first: %r" % (name, len(missing), missing[0]))
        if cur["lines"] < base["lines"]:
            problems.append("%s shrank: %d -> %d lines" % (name, base["lines"], cur["lines"]))
        if len(cur["headings"]) < len(base["headings"]):
            problems.append("%s heading count: %d -> %d" % (name, len(base["headings"]), len(cur["headings"])))
    return problems


def main(argv):
    if "--update" in argv:
        man = {name: scan(name) for name in GUIDES if os.path.exists(os.path.join(DOCS, name))}
        json.dump(man, open(MANIFEST, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("manifest written:", ", ".join("%s=%d lines" % (k, v["lines"]) for k, v in man.items()))
        return 0
    problems = compare(load())
    for p in problems:
        print("DOC INTEGRITY:", p)
    if problems:
        return 1
    print("doc integrity: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
