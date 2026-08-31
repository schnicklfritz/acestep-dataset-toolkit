"""Web research helper with optional Bright Data routing.

By default this fetches with plain requests. When Bright Data credentials are
present it routes through Bright Data, so bot-protected sites (Clone Hero
spreadsheets, multitrack catalogs, forums) can be read reliably.

Configure via environment variables OR the app's settings.json keys:

  BRIGHTDATA_TOKEN   Web Unlocker API bearer token (api.brightdata.com/request)
  BRIGHTDATA_ZONE    proxy zone name (default "web_unlocker1" for the API)
  BRIGHTDATA_USER    residential-proxy username (brd-customer-<id>-zone-<zone>)
  BRIGHTDATA_PASS    residential-proxy password

settings.json equivalents: brightdata_token / brightdata_zone /
brightdata_user / brightdata_pass.
"""
import json
import os

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _settings_config():
    cfg = {}
    for key, env, skey in [
        ("token", "BRIGHTDATA_TOKEN", "brightdata_token"),
        ("zone", "BRIGHTDATA_ZONE", "brightdata_zone"),
        ("user", "BRIGHTDATA_USER", "brightdata_user"),
        ("pass", "BRIGHTDATA_PASS", "brightdata_pass"),
        ("geo", "BRIGHTDATA_GEO", "brightdata_geo"),
    ]:
        val = os.getenv(env)
        if not val:
            try:
                with open("settings.json", encoding="utf-8") as f:
                    val = json.load(f).get(skey, "")
            except Exception:  # noqa: BLE001
                val = ""
        cfg[key] = (val or "").strip()
    return cfg


def brightdata_enabled():
    """True when either the Web Unlocker token or proxy credentials are set."""
    c = _settings_config()
    return bool(c.get("token")) or (bool(c.get("user")) and bool(c.get("pass")))


def fetch_page(url, timeout=30, use_brightdata=True):
    """Fetch a URL and return its text, routed through Bright Data when enabled."""
    if use_brightdata and brightdata_enabled():
        return _fetch_via_brightdata(url, timeout)
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def _fetch_via_brightdata(url, timeout):
    c = _settings_config()
    if c.get("token"):
        # ---- Web Unlocker API ----
        resp = requests.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {c['token']}",
                "Content-Type": "application/json",
            },
            json={"zone": c.get("zone") or "web_unlocker1", "url": url},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("html") or data.get("body") or json.dumps(data)
    # ---- Residential proxy (with optional geo-targeting) ----
    user = c["user"]
    geo = c.get("geo", "")
    if geo:
        # e.g. "us,ohio" -> -country-us-state-ohio appended to the username
        parts = [p.strip().lower() for p in geo.replace(";", ",").split(",") if p.strip()]
        if parts and parts[0] not in ("", "none"):
            user += f"-country-{parts[0]}"
            if len(parts) >= 2:
                user += f"-state-{parts[1]}"
    proxy = f"http://{user}:{c['pass']}@brd.superproxy.io:33335"
    proxies = {"http": proxy, "https": proxy}
    resp = requests.get(
        url, proxies=proxies, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    return resp.text
