"""MVSEP API client — dynamic algorithm list + separation jobs.

Ported from the official mvsep_client_gui reference (PyQt6) but framework-free,
so it can be used from Qt tabs, worker threads, or the CLI.

Key advantage over hardcoded model IDs: :func:`get_algorithms` pulls the live
algorithm list from MVSEP, so the newest separation models are always available.
"""
import json
import os
import time

import requests

BASE_URL = "https://mvsep.com/api"


def get_algorithms():
    """Fetch the live list of separation algorithms from MVSEP.

    Returns ``(by_id, fields_by_id)`` where::

        by_id: {render_id (str): algorithm display name}
        fields_by_id: {render_id: [algorithm_fields]}  # for extra options 1-3

    This endpoint is public (no API token required).
    """
    resp = requests.get("https://mvsep.com/api/app/algorithms", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    by_id = {}
    fields_by_id = {}
    if isinstance(data, list):
        for alg in data:
            if not isinstance(alg, dict):
                continue
            render_id = str(alg.get("render_id", ""))
            if not render_id:
                continue
            by_id[render_id] = alg.get("name", render_id)
            fields_by_id[render_id] = alg.get("algorithm_fields", []) or []
    return by_id, fields_by_id


def create_separation(
    path_to_file,
    api_token,
    sep_type,
    add_opt1=0,
    add_opt2=0,
    add_opt3=0,
    output_format="1",
):
    """Create a separation job. Returns ``(hash_or_error, status_code)``."""
    with open(path_to_file, "rb") as audio:
        files = {
            "audiofile": audio,
            "api_token": (None, api_token),
            "sep_type": (None, str(sep_type)),
            "add_opt1": (None, str(add_opt1)),
            "add_opt2": (None, str(add_opt2)),
            "add_opt3": (None, str(add_opt3)),
            "output_format": (None, output_format),
            "is_demo": (None, "0"),
        }
        resp = requests.post(
            f"{BASE_URL}/separation/create", files=files, timeout=120
        )
    if resp.status_code != 200:
        return resp.content, resp.status_code
    parsed = resp.json()
    return parsed["data"]["hash"], resp.status_code


def check_result(job_hash, api_token=""):
    """Check a separation job. Returns ``(success, data)``."""
    params = {"hash": job_hash}
    if api_token:
        params["api_token"] = api_token
    resp = requests.get(f"{BASE_URL}/separation/get", params=params, timeout=30)
    data = resp.json()
    return bool(data.get("success")), data


def poll_until_done(job_hash, api_token="", max_wait=900, poll_interval=5):
    """Poll /separation/get until the job finishes.

    Returns the final ``data`` dict (status == 'done').
    Raises :class:`RuntimeError` on failure and :class:`TimeoutError` on timeout.
    """
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        success, data = check_result(job_hash, api_token)
        if not success:
            raise RuntimeError(f"MVSEP status check failed: {data}")
        status = data.get("status")
        if status == "done":
            return data
        if status == "failed":
            raise RuntimeError(f"MVSEP job failed: {data}")
    raise TimeoutError("MVSEP job timed out")


def download_file(url, filename, save_path):
    """Download a result file. Returns the local path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    os.makedirs(save_path, exist_ok=True)
    dest = os.path.join(save_path, filename)
    with open(dest, "wb") as out:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                out.write(chunk)
    return dest


def get_result_files(data, save_path):
    """Extract + download every result file from a 'done' response.

    Returns a list of local file paths.
    """
    files = (data.get("data") or {}).get("files") or []
    saved = []
    for item in files:
        url = item.get("url", "").replace("\\/", "/")
        if not url:
            continue
        filename = item.get("download") or os.path.basename(url.split("?")[0])
        saved.append(download_file(url, filename, save_path))
    return saved


def separation_types_as_json():
    """Return the live algorithm list as ``{render_id: name}`` JSON (debug/CLI)."""
    by_id, _ = get_algorithms()
    return json.dumps(by_id, indent=2)


# ---------------------------------------------------------------------------
# Full stem separation — BS PolarFormer (124-band) first, then multi-stem
# ---------------------------------------------------------------------------
def resolve_algorithm_id(by_id, keywords):
    """Find the render_id whose name contains every keyword (case-insensitive)."""
    lowered = [k.lower() for k in keywords]
    for rid, name in by_id.items():
        n = name.lower()
        if all(k in n for k in lowered):
            return rid
    return None


def resolve_default_first_stage(by_id):
    """Return the render_id of BS PolarFormer (the default first stage), if present.

    PolarFormer separates vocals + instrumental and re-synthesizes the
    instrumental frequencies, which prevents artifacts and clipping in later
    multi-stem stages — hence it is the recommended default. The user can
    override it with any algorithm.
    """
    return resolve_algorithm_id(by_id, ["polarformer"])


def _find_instrumental(files):
    """Locate the instrumental stem among downloaded result files."""
    for path in files:
        if "instrumental" in os.path.basename(path).lower():
            return path
    # Fall back to the largest file (the instrumental is usually the longest).
    if files:
        return max(files, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
    return None


def run_full_separation(input_path, api_token, multi_sep_type, output_dir,
                        first_sep_type=None, progress=None):
    """Full stem separation: a first stage on the full mix, then the selected
    multi-stem algorithm applied to the instrumental.

    ``first_sep_type`` is the render_id of the first-stage algorithm. It is
    *not* hardcoded: when omitted it defaults to BS PolarFormer (124-band),
    which re-synthesizes the instrumental and prevents artifacts/clipping, but
    any algorithm from the live list can be supplied instead.

    Returns the list of downloaded local file paths (stage 1 + stage 2).
    """
    progress = progress or (lambda m: None)

    by_id, _ = get_algorithms()
    first_id = str(first_sep_type or "").strip() or resolve_default_first_stage(by_id)
    if not first_id:
        raise RuntimeError(
            "No first-stage algorithm selected and BS PolarFormer was not found "
            "in the live MVSEP algorithm list."
        )

    # ---- Step 1: first stage (default BS PolarFormer 124-band) ----
    first_name = by_id.get(first_id, "First stage")
    progress(f"Step 1/2: {first_name} — vocals + instrumental…")
    hash1, status = create_separation(input_path, api_token, first_id)
    if status != 200:
        raise RuntimeError(f"First-stage job failed (HTTP {status}).")
    data1 = poll_until_done(hash1, api_token)
    stage1 = get_result_files(data1, output_dir)

    instrumental = _find_instrumental(stage1)
    if not instrumental:
        raise RuntimeError("First-stage output contained no instrumental stem.")

    # ---- Step 2: selected multi-stem algorithm on the instrumental ----
    progress("Step 2/2: Running selected algorithm on the instrumental…")
    hash2, status = create_separation(instrumental, api_token, multi_sep_type)
    if status != 200:
        raise RuntimeError(f"Multi-stem job failed (HTTP {status}).")
    data2 = poll_until_done(hash2, api_token)
    stage2 = get_result_files(data2, output_dir)

    return stage1 + stage2

