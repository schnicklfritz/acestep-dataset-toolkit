"""On-disk dataset versioning: snapshots, diff, restore."""
import difflib
import json
import time
from pathlib import Path

VERSIONS_DIR = Path(__file__).resolve().parent.parent / "dataset_versions"


def versions_dir():
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return VERSIONS_DIR


def save_version(dataset, label=""):
    """Write a versioned snapshot of the dataset. Returns the file path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"v_{stamp}"
    if label:
        name += f"_{str(label)[:30]}"
    name += ".json"
    path = versions_dir() / name
    path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def list_versions():
    """Return metadata for each snapshot (newest first)."""
    out = []
    for p in versions_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "path": str(p),
                "name": p.stem,
                "mtime": p.stat().st_mtime,
                "tracks": len(data.get("samples", [])),
            })
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda v: v["mtime"], reverse=True)
    return out


def load_version(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diff_json(a, b):
    """Unified text diff between two dataset dicts (JSON, sorted keys)."""
    lines = difflib.unified_diff(
        json.dumps(a, indent=2, sort_keys=True).splitlines(),
        json.dumps(b, indent=2, sort_keys=True).splitlines(),
        fromfile="snapshot", tofile="current", lineterm="",
    )
    return "\n".join(lines) or "(identical)"