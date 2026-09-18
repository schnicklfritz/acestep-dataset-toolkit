#!/usr/bin/env python3
"""Gather a local prose corpus about an artist from web sources.

Curating the URLs is a search job; fetching and cleaning them is mechanical.
This script does the mechanical half:

    search (agent/human) -> manifest -> *this* -> docs/research/<artist>/*.txt

Hard rules:
  * Corpus text is third-party prose. docs/research/ is gitignored -- regenerate
    it, never commit it.
  * Only PROSE lands in *.txt. Provenance goes to a sidecar sources.json, because
    .json is not in research_tools.TEXT_EXTS. A "# source: <url>" header would
    otherwise inject "www"/"com"/"html" into the unlisted_terms n-gram counts.
  * requests + the stdlib html.parser only. BeautifulSoup/lxml are not runtime
    deps of this app and must not become any.
  * Fetching is injected (fetcher=) so the test suite -- which fails any real
    network connection -- can exercise cleaning offline.

Usage:
    .venv/bin/python scripts/gather_corpus.py --artist black_sabbath --url URL
    .venv/bin/python scripts/gather_corpus.py --manifest docs/research/sources.json
    .venv/bin/python scripts/gather_corpus.py --manifest ... --dry-run

APPEND-ONLY MANIFESTS: filenames are ``NN-slug.txt`` by list position, so add new
URLs to the END of an artist's list. Inserting in the middle renumbers every later
file and re-fetches the lot.
"""
import argparse
import json
import os
import re
import sys
from datetime import date
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_ROOT = os.path.join(ROOT, "docs", "research")

# Containers whose text is never prose.
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form",
             "noscript", "svg", "figure", "table"}

# Block elements that imply a line break, so paragraphs do not run together.
BLOCK_TAGS = {"p", "br", "div", "li", "tr", "section", "article", "blockquote",
              "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "pre"}

# A line shorter than this is navigation chrome, not a sentence.
MIN_PROSE_CHARS = 40

# Lines that are site chrome even when long enough to look like prose.
BOILERPLATE = re.compile(
    r"^(retrieved|jump to|toggle|menu|sign in|log in|create account|donate|"
    r"privacy policy|about wikipedia|disclaimers|cookie|contents|main page|"
    r"random article|wikipedia|share|advertisement|subscribe|newsletter)",
    re.I,
)

# Headings that start reference apparatus. Everything from here on is citation
# prose, not description -- it is how a vocabulary file ends up "learning" the
# word August, the number 13, and "retrieved".
CUT_SECTIONS = {"references", "external links", "further reading",
                "bibliography", "see also", "footnotes", "works cited",
                "notes and references", "general sources", "sources"}

# Per-line citation debris that survives when the heading was not captured.
CITATION = re.compile(
    r"(retrieved\s+\d|archived from the original|\bdoi:|\bisbn\b|\bissn\b|"
    r"\bpmid\b|\boclc\b|^\s*[\u2191^])",
    re.I,
)


class _ProseExtractor(HTMLParser):
    """Collect visible text, honouring SKIP_TAGS and block-level breaks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _truncate_at_appendix(raw):
    """Drop everything from the first reference/bibliography heading onwards."""
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower() in CUT_SECTIONS:
            return "\n".join(lines[:index])
    return raw


def _clean_lines(raw):
    """Keep sentence-like lines; drop chrome, edit links and duplicates."""
    out, seen = [], set()
    for line in raw.splitlines():
        line = re.sub(r"\[\d+\]", "", line)                # citation markers
        line = re.sub(r"\[edit\]", "", line, flags=re.I)
        line = " ".join(line.split())
        if len(line) < MIN_PROSE_CHARS or len(line.split()) < 6:
            continue
        if BOILERPLATE.match(line) or CITATION.search(line):
            continue
        if not re.search(r"[A-Za-z]{3}", line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


def html_to_text(html):
    """HTML -> pure prose. Deterministic, offline, dependency-free."""
    parser = _ProseExtractor()
    parser.feed(html)
    return _clean_lines(_truncate_at_appendix("".join(parser.parts)))


def slugify(text):
    """A URL/title -> a safe, short filename stem."""
    text = re.sub(r"^https?://", "", text or "")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:60].rstrip("-") or "source"


def _write_provenance(out_dir, records):
    """Merge this run's sources into the artist's sidecar manifest."""
    path = os.path.join(out_dir, "sources.json")
    existing = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            existing = {r["file"]: r for r in json.load(fh).get("sources", [])}
    for record in records:
        existing[record["file"]] = record
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"sources": list(existing.values())}, fh, indent=2)
        fh.write("\n")


def gather(urls, artist, fetcher=None, dry_run=False, corpus_root=CORPUS_ROOT):
    """Fetch and clean ``urls`` into ``corpus_root/<artist>/``.

    Returns ``(report_lines, records)``. A fetch failure is reported, never
    raised: one dead URL must not abandon the rest of the corpus. Already-present
    files are skipped, so re-running is cheap and safe.
    """
    if fetcher is None:
        from modules.web_research import fetch_page as fetcher
    out_dir = os.path.join(corpus_root, artist)
    lines, records = [], []
    for index, url in enumerate(urls, start=1):
        name = f"{index:02d}-{slugify(url)}.txt"
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            lines.append(f"  skip   {name} (already present)")
            continue
        try:
            payload = fetcher(url)
        except Exception as exc:                           # noqa: BLE001
            lines.append(f"  FAIL   {url} ({type(exc).__name__}: {exc})")
            continue
        if not isinstance(payload, str) or payload.lstrip()[:4] == "%PDF":
            lines.append(f"  skip   {name} (not HTML)")
            continue
        text = html_to_text(payload)
        words = len(text.split())
        if words < 50:
            lines.append(f"  skip   {name} (only {words} prose words)")
            continue
        records.append({"file": name, "url": url, "words": words,
                        "fetched": date.today().isoformat()})
        if dry_run:
            lines.append(f"  would  {name} ({words} words)")
            continue
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        lines.append(f"  ok     {name} ({words} words)")
    if records and not dry_run:
        _write_provenance(out_dir, records)
    return lines, records


def _jobs_from_manifest(path):
    """Read ``{artist: [url, ...]}`` into ``[(artist, urls)]``."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [(artist, list(urls)) for artist, urls in data.items()
            if isinstance(urls, list)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artist", help="corpus subdirectory, e.g. black_sabbath")
    parser.add_argument("--url", action="append", default=[],
                        help="source URL (repeatable); requires --artist")
    parser.add_argument("--manifest", help="JSON mapping artist -> [url, ...]")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and report, but write nothing")
    args = parser.parse_args(argv)

    jobs = _jobs_from_manifest(args.manifest) if args.manifest else []
    if args.url:
        if not args.artist:
            parser.error("--url requires --artist")
        jobs.append((args.artist, args.url))
    if not jobs:
        parser.error("nothing to do: pass --manifest and/or --artist with --url")

    total = 0
    for artist, urls in jobs:
        print(f"{artist}: {len(urls)} URL(s)")
        lines, records = gather(urls, artist, dry_run=args.dry_run)
        print("\n".join(lines))
        total += len(records)
    print(f"\n{total} file(s) {'planned' if args.dry_run else 'written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())