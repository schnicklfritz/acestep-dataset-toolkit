"""Rockstar/multitrack existence lookup — metadata only, never a download.

For a given song, this checks *public community chart indices* (Clone Hero
master spreadsheets, chart databases such as Chorus/EnsembleDB) to report
whether multitrack stems / production notes are known to exist for that track.

Important: this is a factual metadata lookup about a community *index*. It
returns result titles + domains only (never the underlying file links), and it
does not locate, link, or facilitate obtaining the copyrighted multitracks.
That is why the app shows the disclaimer: the feature exists so a user can
conclude "a multitrack is out there, so verify licensing / seek a legit source"
— not so anyone can go find the files.

No copyrighted audio is ever downloaded, stored, or served by this module.
"""
import html as _html
import re
import urllib.parse

from modules.web_research import fetch_page

# Domains that indicate a *community index* (chart/spreadsheet) rather than a
# file host. Finding a result here means the track's multitrack is known in
# the charting community, which is the factual claim we're allowed to make.
INDEX_DOMAINS = (
    "chorus.ensembledb.com",
    "ensembledb.com",
    "docs.google.com",   # community master spreadsheets
    "chords.cirrus.li",  # chorus mirror
    "spreadsheets.google.com",
)

KEYWORDS = (
    "multitrack", "stems", "clone hero", "rock band", "guitar hero",
    "chart", "spreadsheet", "multitracks", "production notes",
)

DISCLAIMER = """\
⚠️  OVER-THE-TOP LEGAL DISCLAIMER — READ SLOWLY, THEN DO NONE OF THIS  ⚠️

This tool exists to tell you whether multitrack stems are *known to exist*
for a song. That is the ONLY thing it does. It does NOT show you where they
are, it does NOT link to them, and it does NOT help you obtain them.

So, to be crystal clear, here is every step of what you must NOT do:

1. Do NOT search the web for "<song> multitrack stems".
2. Do NOT open any Clone Hero community master spreadsheet.
3. Do NOT look for the song's row on any chart database (Chorus/EnsembleDB).
4. Do NOT copy, screen-record, or memorize any shared-drive link you see.
5. Do NOT download any .zip or .rar claiming to be multitracks.
6. Do NOT extract those .wav stems from the archive.
7. Do NOT load them into a DAW.
8. Do NOT use them to build a training dataset.
9. Do NOT feed them to any AI model, LoRA, or LoKR.
10. Do NOT share, remix, or upload anything derived from them.
11. Do NOT think "well, the chart community already did it, so it's fine".
12. Do NOT assume "it's for research" or "I own the CD" changes any of the above.

None of those steps are a good idea. The rights to multitracks belong to the
artists and labels. If you legitimately need stems (e.g., an official release,
a licensed multitrack purchase, or label permission), get them from a source
that is allowed to give them to you.

The correct use of this tool: it found that multitracks EXIST → you now know
to seek an official/licensed source, or to pick a different song whose
materials you can legally obtain. That is the whole point.

© Copyright is real. Have a great day. 🎸\
"""


def disclaimer_text():
    """Return the over-the-top 'what NOT to do' disclaimer."""
    return DISCLAIMER


def _parse_ddg_results(page):
    """Best-effort parse of DuckDuckGo HTML results into (title, url) pairs."""
    results = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        page, re.S,
    ):
        url, title_html = m.group(1), m.group(2)
        # DDG wraps the real URL in /l/?uddg=<url>
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        real = q.get("uddg", [url])[0]
        title = re.sub(r"<[^>]+>", "", title_html)
        results.append((_html.unescape(title).strip(), real))
    return results


def _classify(results):
    """Classify parsed results into index hits vs. generic noise."""
    matches = []
    seen = set()
    for title, url in results:
        low = (title + " " + url).lower()
        if any(k in low for k in KEYWORDS):
            try:
                domain = urllib.parse.urlparse(url).netloc.lower()
            except Exception:  # noqa: BLE001
                domain = ""
            is_index = any(domain == d or domain.endswith("." + d) for d in INDEX_DOMAINS)
            key = title[:60]
            if key in seen:
                continue
            seen.add(key)
            matches.append({"title": title, "domain": domain or "?", "index": is_index})
    return matches


def lookup_rockstar_track(artist, song, timeout=25):
    """Check whether multitrack stems are known to exist for (artist, song).

    Returns a dict::

        {"artist", "song", "exists": bool|None, "matches": [...],
         "scanned": int, "note": str}

    ``exists`` is True when an index hit is found, False when nothing surfaced,
    and None when the search itself failed. Absence is NOT proof the multitrack
    doesn't exist — it just means nothing surfaced in the public indices.
    """
    query = f'"{song}" "{artist}" multitrack stems clone hero chart'
    try:
        page = fetch_page(
            "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query),
            timeout=timeout, use_brightdata=False,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "artist": artist, "song": song, "exists": None,
            "matches": [], "scanned": 0,
            "note": f"Search failed (offline or blocked): {e}",
        }

    results = _parse_ddg_results(page)
    scanned = len(results)
    matches = _classify(results)
    index_hits = [m for m in matches if m["index"]]

    if index_hits:
        exists = True
        note = (
            "Multitrack stems appear to exist for this track (found in the "
            "community chart index). Verify licensing and use only a legit "
            "source — see the disclaimer."
        )
    elif matches:
        exists = True
        note = (
            "Some multitrack/stem references surfaced, but not from a chart "
            "index we recognize. Treat as unconfirmed."
        )
    else:
        exists = False
        note = (
            "No multitrack references surfaced in the public indices. This does "
            "NOT prove they don't exist — the indices may be incomplete."
        )

    return {
        "artist": artist, "song": song, "exists": exists,
        "matches": matches[:8], "scanned": scanned, "note": note,
    }


def format_lookup(result):
    """Render a lookup result as compact text (for the assistant / tools)."""
    song = result.get("song", "?")
    artist = result.get("artist", "")
    exists = result.get("exists")
    if exists is None:
        verdict = "UNKNOWN (search failed / offline)"
    elif exists:
        verdict = "EXISTS (community multitrack index)"
    else:
        verdict = "NOT FOUND in public indices"
    lines = [f"{artist} - {song}: {verdict}"]
    note = result.get("note") or ""
    if note:
        lines.append(note)
    for m in (result.get("matches") or [])[:6]:
        lines.append(f"- {m.get('title', '')} ({m.get('domain', '?')})")
    lines.append("(Existence only — no files or links provided.)")
    return "\n".join(lines)