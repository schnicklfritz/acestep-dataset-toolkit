"""MusicBrainz metadata enrichment (artist / title / year / genre).

Uses the public MusicBrainz web API (no key required). Given an artist + song,
returns the best recording match with its release year, country, and genre
tags — copyright-safe metadata only.
"""
import json
import urllib.parse

from modules.web_research import fetch_page


def lookup_recording(artist, song, timeout=15):
    """Look up a recording on MusicBrainz. Returns a result dict."""
    artist = (artist or "").strip()
    song = (song or "").strip()
    if not song:
        return {"ok": False, "note": "A song title is required."}
    q = f'"{song}"'
    if artist:
        q += f' AND artist:"{artist}"'
    url = "https://musicbrainz.org/ws/2/recording/?" + urllib.parse.urlencode(
        {"query": q, "fmt": "json", "limit": "5"}
    )
    try:
        page = fetch_page(url, use_brightdata=False)
        data = json.loads(page)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "note": f"Lookup failed (offline or blocked): {e}"}

    recordings = data.get("recordings", [])
    if not recordings:
        return {"ok": False, "note": "No MusicBrainz recording matched."}

    top = recordings[0]
    title = top.get("title", song)
    artist_name = ", ".join(
        a.get("name", "") for a in top.get("artist-credit", []) if a.get("name")
    ) or artist
    year = ""
    country = ""
    genres = []
    for rel in top.get("releases", []):
        if rel.get("date"):
            year = rel["date"][:4]
            country = rel.get("country", "")
            genres = [g.get("name", "") for g in rel.get("genres", [])]
            break
    return {
        "ok": True,
        "title": title,
        "artist": artist_name,
        "year": year,
        "country": country,
        "genres": sorted({g for g in genres if g})[:8],
        "note": f"Best match: {artist_name} - {title} ({year or 'year unknown'})",
    }