"""Tests for scripts/gather_corpus.py.

Offline by construction: every test injects a fake fetcher, because
tests/conftest.py fails any real HTTP connection.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gather_corpus import (  # noqa: E402
    MIN_PROSE_CHARS,
    gather,
    html_to_text,
    slugify,
)

PROSE = (
    "Black Sabbath built Paranoid on downtuned guitar and thunderous drums, "
    "with Iommi's heavy riffing carrying the record through its runtime. "
    "The band tracked the album in a matter of days, leaning on raw 1970s "
    "analog production that left the guitars dry and the drums enormous. "
    "Ozzy's nasal vocal sits high in the mix, pleading and menacing at once, "
    "while Geezer Butler's distorted bass locks in with Bill Ward's swing. "
    "Critics who reached for the phrase doom metal were describing that weight."
)

HTML = f"""<html><head><style>p{{color:red}}</style>
<script>var tracking=1;</script></head>
<body><nav>Jump to content Main page Random article</nav>
<p>{PROSE}</p>
<footer>Privacy policy Disclaimers Cookie statement</footer></body></html>"""


def _fake_fetcher(pages, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        if url not in pages:
            raise RuntimeError("404")
        return pages[url]
    return fetch


# --------------------------------------------------------------------------
# html_to_text
# --------------------------------------------------------------------------

def test_html_to_text_keeps_prose_and_drops_chrome():
    out = html_to_text(HTML)
    assert "downtuned guitar" in out
    assert "tracking" not in out            # <script>
    assert "color:red" not in out            # <style>
    assert "Jump to content" not in out      # <nav>
    assert "Privacy policy" not in out       # <footer>


def test_html_to_text_drops_short_fragments():
    assert html_to_text("<p>Home</p><p>Menu</p>") == ""
    assert MIN_PROSE_CHARS > 4


def test_html_to_text_strips_citation_markers():
    out = html_to_text(f"<p>{PROSE}[12] and more text follows here.</p>")
    assert "[12]" not in out


def test_html_to_text_never_returns_markup():
    out = html_to_text(HTML)
    assert "<" not in out and "&nbsp" not in out


def test_html_to_text_truncates_reference_appendix():
    html = (
        f"<p>{PROSE}</p>"
        "<h2>References</h2>"
        "<p>Smith, John (12 August 2019). Title. Rolling Stone. "
        "Archived from the original on 5 May 2020. Retrieved 26 September 2023.</p>"
    )
    out = html_to_text(html)
    assert "downtuned guitar" in out
    assert "Retrieved" not in out
    assert "Archived" not in out


def test_html_to_text_drops_citation_lines_without_a_heading():
    html = (
        f"<p>{PROSE}</p>"
        "<p>Kimball, Duncan. Milesago: Australasian Music and Popular Culture. "
        "Ice Productions. Retrieved 26 September 2023.</p>"
    )
    out = html_to_text(html)
    assert "downtuned guitar" in out
    assert "Milesago" not in out


def test_html_to_text_drops_wikipedia_backlink_lines():
    html = (f"<p>{PROSE}</p>"
            "<p>↑ McFarlane, Ian (2017). Third Stone Press. Retrieved 16 March 2024.</p>")
    assert "McFarlane" not in html_to_text(html)


def test_html_to_text_keeps_prose_mentioning_dates():
    """The citation filter must not eat ordinary prose that names a month."""
    out = html_to_text(f"<p>{PROSE} The sessions wrapped in August 1970.</p>")
    assert "August 1970" in out


# --------------------------------------------------------------------------
# slugify
# --------------------------------------------------------------------------

def test_slugify_is_filename_safe():
    assert slugify("https://en.wikipedia.org/wiki/Paranoid_(album)") == \
        "en-wikipedia-org-wiki-paranoid-album"
    assert "/" not in slugify("a/b?c=d&e")
    assert slugify("") == "source"


# --------------------------------------------------------------------------
# gather
# --------------------------------------------------------------------------

def test_gather_writes_prose_and_provenance_sidecar(tmp_path):
    url = "https://example.com/review"
    calls = []
    lines, records = gather([url], "black_sabbath",
                            fetcher=_fake_fetcher({url: HTML}, calls),
                            corpus_root=str(tmp_path))
    out_dir = tmp_path / "black_sabbath"
    assert sorted(p.name for p in out_dir.iterdir()) == \
        ["01-example-com-review.txt", "sources.json"]
    text = (out_dir / "01-example-com-review.txt").read_text(encoding="utf-8")
    assert "downtuned guitar" in text
    assert "http" not in text                # provenance is NOT in the prose
    prov = json.loads((out_dir / "sources.json").read_text(encoding="utf-8"))
    assert prov["sources"][0]["url"] == url
    assert prov["sources"][0]["words"] == len(text.split())
    assert calls == [url]
    assert records and "ok" in "\n".join(lines)


def test_gather_is_idempotent(tmp_path):
    url = "https://example.com/review"
    calls = []
    fetcher = _fake_fetcher({url: HTML}, calls)
    gather([url], "sabbath", fetcher=fetcher, corpus_root=str(tmp_path))
    lines, records = gather([url], "sabbath", fetcher=fetcher,
                            corpus_root=str(tmp_path))
    assert calls == [url]                    # not fetched twice
    assert records == []
    assert "already present" in "\n".join(lines)


def test_gather_reports_failure_without_raising(tmp_path):
    lines, records = gather(["https://example.com/gone"], "sabbath",
                            fetcher=_fake_fetcher({}), corpus_root=str(tmp_path))
    assert records == []
    assert "FAIL" in "\n".join(lines)


def test_gather_skips_pdf_and_thin_pages(tmp_path):
    pdf = "https://example.com/x.pdf"
    thin = "https://example.com/thin"
    lines, records = gather(
        [pdf, thin], "sabbath",
        fetcher=_fake_fetcher({pdf: "%PDF-1.4 binary", thin: "<p>hi</p>"}),
        corpus_root=str(tmp_path),
    )
    assert records == []
    joined = "\n".join(lines)
    assert "not HTML" in joined
    assert "prose words" in joined


def test_gather_dry_run_writes_nothing(tmp_path):
    url = "https://example.com/review"
    lines, records = gather([url], "sabbath",
                            fetcher=_fake_fetcher({url: HTML}),
                            corpus_root=str(tmp_path), dry_run=True)
    assert len(records) == 1                 # planned, reported
    assert "would" in "\n".join(lines)
    assert not (tmp_path / "sabbath").exists()


def test_manifest_parsing_keeps_artist_to_url_mapping(tmp_path):
    import scripts.gather_corpus as gc

    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sabbath": ["https://example.com/a"]}),
                        encoding="utf-8")
    assert gc._jobs_from_manifest(str(manifest)) == \
        [("sabbath", ["https://example.com/a"])]
