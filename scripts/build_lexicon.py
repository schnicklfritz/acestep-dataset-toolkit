#!/usr/bin/env python
"""Build the descriptor vocabulary from the master descriptor reference doc.

The reference doc (`docs/descriptor_reference.md`) is written for humans: it
mixes term lists with example captions and lyric snippets. The research tools
need a clean, one-term-per-line vocabulary, so this script extracts it.

    .venv/bin/python scripts/build_lexicon.py

Writes:
    docs/vocabulary.txt       descriptor terms, one per line
    docs/section_markers.txt  [Marker] names, one per line

Both files are generated -- edit the reference doc, then re-run this.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.research_tools import _looks_like_prose, build_lexicon   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(ROOT, "docs", "descriptor_reference.md")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="descriptor reference doc (default: %(default)s)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the generated files (default: docs/)")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any extracted term looks like prose")
    parser.add_argument("--list", action="store_true",
                        help="print every extracted term, not just the report")
    args = parser.parse_args(argv)

    try:
        result = build_lexicon(args.source, args.out_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(result["report"])
    if args.list:
        print()
        for term in result["terms"]:
            print(term)
        print()
        print("Section markers:")
        for marker in result["markers"]:
            print(f"  {marker}")

    if args.check:
        suspects = [t for t in result["terms"] if _looks_like_prose(t)]
        if suspects:
            print(f"\nFAIL: {len(suspects)} term(s) look like prose, not descriptors.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
