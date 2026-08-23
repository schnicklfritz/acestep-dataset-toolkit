#!/usr/bin/env python3
import sys
import os
import json
import requests
from bs4 import BeautifulSoup
import urllib.parse

GENIUS_SEARCH_URL = "https://genius.com/api/search/multi"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def search_genius(query):
    url = f"{GENIUS_SEARCH_URL}?q={urllib.parse.quote(query)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        
        sections = data.get("response", {}).get("sections", [])
        hits = []
        for sec in sections:
            if sec.get("type") == "song":
                for hit in sec.get("hits", []):
                    result = hit.get("result", {})
                    hits.append({
                        "id": result.get("id"),
                        "title": result.get("title"),
                        "artist": result.get("primary_artist", {}).get("name"),
                        "path": result.get("path"),
                        "url": f"https://genius.com{result.get('path')}",
                        "stats": result.get("stats", {})
                    })
        return hits
    except Exception as e:
        sys.stderr.write(f"Search error: {e}\n")
        return []

def scrape_lyrics(song_url):
    try:
        resp = requests.get(song_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if not containers:
            return ""
        
        lyrics_text = []
        for c in containers:
            for br in c.find_all(["br", "hr"]):
                br.replace_with("\n")
            lyrics_text.append(c.get_text())
            
        full_lyrics = "\n\n".join(lyrics_text).strip()
        return full_lyrics
    except Exception as e:
        sys.stderr.write(f"Scrape error: {e}\n")
        return ""

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: fetchgeniuslyrics.py <song_title_or_query>"}))
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    hits = search_genius(query)
    
    if not hits:
        print(json.dumps({"query": query, "hits": [], "top_lyrics": ""}))
        return

    top_hit = hits[0]
    top_lyrics = scrape_lyrics(top_hit["url"])

    output = {
        "query": query,
        "top_hit": top_hit,
        "hits": hits[:5],
        "top_lyrics": top_lyrics
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
