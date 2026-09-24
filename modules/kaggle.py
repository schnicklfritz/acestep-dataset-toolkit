"""Shared Kaggle helpers: private audio-dataset upload + kernel push/wait/download.

The correct way to feed audio to a Kaggle kernel is as a **dataset source** —
files copied into the pushed kernel directory are NOT reliably included. These
helpers upload the staged audio as a private Kaggle dataset and attach it via
``dataset_sources`` in the kernel metadata.

This module talks to Kaggle through the official ``kaggle`` Python package's
API client (``KaggleApi``) instead of shelling out to the ``kaggle`` console
script via ``subprocess``. Calling the API directly means this only requires
``kaggle`` to be *importable* (guaranteed by ``pip install kaggle``) — it does
not depend on a ``kaggle`` executable being resolvable on the OS ``PATH``,
which is what caused ``[Errno 2] No such file or directory: 'kaggle'`` under
the old subprocess-based implementation.
"""
import json
import os
import time
import uuid


ACCESS_TOKEN_FILE = "~/.kaggle/access_token"

# Kaggle's modern access tokens (kaggle.com > Settings > API) are prefixed with
# this. They are NOT legacy API keys: the SDK authenticates them through a
# different code path, and presenting one as an API key is what makes the app
# look connected while every call comes back 401.
ACCESS_TOKEN_PREFIX = "KGAT_"


def is_access_token(value):
    """True when ``value`` is a modern bearer access token, not a legacy key."""
    return str(value or "").strip().startswith(ACCESS_TOKEN_PREFIX)


def _write_access_token_file(config, token):
    """Create the SDK's convenience token file — only if the user asked to remember.

    WHY THIS IS GATED: the file is a PLAINTEXT copy of the API token, outside the
    encrypted store, so writing it unconditionally contradicted an explicit
    "don't store it on this device" choice.

    WHY IT ONLY CREATES, NEVER OVERWRITES: ``~/.kaggle/access_token`` is a SHARED
    location that other Kaggle tools use, and this app cannot tell its own stale
    copy from another tool's live token. Refusing to touch an existing file is the
    only rule that can never break something else — and nothing is lost, because
    ``KAGGLE_API_TOKEN`` is checked FIRST by
    ``kagglesdk.get_access_token_from_env()`` and carries this process's key.

    Returns True when this call created the file.
    """
    if not config.get("remember_kaggle_key", True):
        return False
    path = os.path.expanduser(ACCESS_TOKEN_FILE)
    try:
        if os.path.exists(path):
            return False                  # shared location: leave it alone
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(token)
        os.chmod(path, 0o600)
        return True
    except OSError:
        return False


def ensure_kaggle_creds(config):
    """Validate Kaggle credentials and set env vars. Returns the username.

    Kaggle's modern ``kaggle`` SDK (>= 2.x) authenticates FIRST via a
    bearer access token (``KAGGLE_API_TOKEN`` / ``~/.kaggle/access_token``)
    before falling back to the legacy ``KAGGLE_USERNAME``/``KAGGLE_KEY``
    pair. The token issued from Settings > API (``KGAT_...``) is one of
    these new-style access tokens, so we must present it in the new fields
    or the API rejects it with ``401 Unauthorized``.

    ``kaggle_user`` is kept for building dataset/kernel slugs only (the SDK
    re-derives the real username from the token at auth time).
    """
    user = config.get("kaggle_user", "").strip()
    key = config.get("kaggle_key", "").strip()
    if not user or not key:
        raise ValueError(
            "Kaggle credentials not configured. Open ⚙ Settings to enter "
            "your Username & Key."
        )
    os.environ["KAGGLE_USERNAME"] = user
    # A "KGAT_..." value is an ACCESS TOKEN, not a legacy API key — do NOT hand it
    # to KAGGLE_KEY. The installed SDK's authenticate() tries the access token
    # first and, if that introspection fails, falls back to the legacy pair by
    # PRESENCE ALONE with no network check at all, and reports success. Every
    # later call then dies with a bare 401 that points nowhere near credentials.
    # Popped, not merely skipped: a stale value set by an earlier call in this
    # process would keep the misleading legacy path alive.
    if is_access_token(key):
        os.environ.pop("KAGGLE_KEY", None)
    else:
        os.environ["KAGGLE_KEY"] = key
    # New access-token auth (takes precedence in the modern SDK).
    os.environ["KAGGLE_API_TOKEN"] = key
    # The env var above is sufficient — kagglesdk checks it FIRST — so the
    # plaintext token file is written ONLY when the user asked to remember the
    # key, and never over someone else's token.
    _write_access_token_file(config, key)
    return user


def kaggle_available():
    """True if the ``kaggle`` package is importable in this environment."""
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _get_api(config):
    """Authenticate and return an authenticated ``(KaggleApi, username)`` pair.

    The import is deliberately local to this function (not module-level) so
    that ``modules.kaggle`` stays importable even when the ``kaggle`` package
    is not installed — only code paths that actually need Kaggle pay the
    price of a missing dependency, and they get a clear ``ImportError``
    instead of the rest of the app failing to start.
    """
    user = ensure_kaggle_creds(config)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError(
            "The 'kaggle' package is not installed. Run: pip install kaggle "
            "(or switch the engine/backend to a non-Kaggle option in Settings)."
        ) from e
    api = KaggleApi()
    api.authenticate()
    return api, user


# Owners that are placeholders rather than accounts. Kaggle usernames are at
# least three characters, so "me" cannot even exist; the rest are the stand-ins
# people type into a settings box. This is deliberately a small, explicit list
# rather than a clever guess: it exists because a literal "me/my-weights" shipped
# in a real config and was only discovered when a kernel push was rejected.
PLACEHOLDER_OWNERS = {
    "me", "you", "user", "owner", "username", "your-username", "yourname",
    "someone", "example",
}


def slug_problems(*slugs):
    """Complain about Kaggle dataset slugs that cannot be a real dataset.

    WHY THIS EXISTS: an unusable slug does not fail where it is typed. It is
    written into the pushed kernel's ``dataset_sources`` and the push is then
    rejected by the API. A placeholder such as ``me/my-weights`` reached a real
    config exactly this way and looked like a Kaggle outage.

    This checks SHAPE and obvious placeholders only — it cannot tell whether a
    well-formed dataset actually exists, which is what the probe's API call is
    for.
    """
    problems = []
    for slug in slugs:
        slug = str(slug or "").strip()
        if not slug:
            continue                      # empty is a legitimate choice
        parts = slug.split("/")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            problems.append(
                f"'{slug}' is not a Kaggle dataset slug -- it must be "
                "'owner/slug'. Copy both parts from the dataset's URL."
            )
        elif parts[0].strip().lower() in PLACEHOLDER_OWNERS:
            problems.append(
                f"'{slug}' looks like a placeholder, not a real Kaggle dataset. "
                "Use the real 'owner/slug' from the dataset's URL."
            )
    return problems


def probe_kaggle(config, slugs=None):
    """Check whether Kaggle actually ACCEPTS these credentials. Never raises.

    ``slugs`` restricts the dataset-slug validation; ``None`` checks the three
    standard settings, which is what the Test-connection button wants. A caller
    that only cares about its own dataset passes just that one, so an unrelated
    bad slug cannot fail its run.

    Returns ``{"ok": bool, "username": str, "auth_method": str, "detail": str,
    "problems": [str, ...]}``.

    WHY THIS EXISTS
    ---------------
    Everything in this app used to infer "Kaggle works" from two non-empty
    strings in the config. That inference is why a bad credential surfaced as an
    unexplained 401 minutes later, or as a worker that vanished without a
    message. The installed SDK makes it worse rather than better:
    ``authenticate()`` tries the access token first (the only step that proves
    anything), then falls back to the legacy username/key pair by PRESENCE ALONE
    with no network check and reports success -- and if even that is missing it
    ends by calling ``exit(1)``, which raises ``SystemExit``. ``SystemExit`` is a
    ``BaseException``, so a worker's ``except Exception`` cannot catch it and the
    thread dies silently with no error dialog.

    This function is deliberately the opposite of that: it makes ONE real
    authenticated call, reports WHICH auth method was used, and RETURNS its
    failures instead of raising them (the same rule ``fetch_kernel_logs``
    follows -- diagnostics must never mask the real error).
    """
    result = {
        "ok": False,
        "username": "",
        "auth_method": "",
        "detail": "",
        "problems": [],
    }

    try:
        result["username"] = ensure_kaggle_creds(config)
    except Exception as e:  # noqa: BLE001 -- reported, never raised
        result["detail"] = str(e)
        return result

    if not kaggle_available():
        result["detail"] = (
            "The 'kaggle' package is not installed in this Python environment. "
            "Run: pip install kaggle"
        )
        return result

    try:
        api, user = _get_api(config)
    except SystemExit:
        result["detail"] = (
            "Kaggle rejected the credentials and the SDK exited without an "
            "explanation (SystemExit). Re-issue the token at "
            "kaggle.com > Settings > API, and re-enter it in Settings."
        )
        return result
    except Exception as e:  # noqa: BLE001
        result["detail"] = f"{type(e).__name__}: {e}"
        return result

    result["username"] = user
    config_values = getattr(api, "config_values", None) or {}
    result["auth_method"] = str(config_values.get("auth_method", "") or "")

    key = (config.get("kaggle_key") or "").strip()
    if "LEGACY" in result["auth_method"].upper() and is_access_token(key):
        result["problems"].append(
            "The stored value is an ACCESS TOKEN (KGAT_...), but the SDK "
            "authenticated it as a LEGACY API KEY. Every real call will come "
            "back 401 Unauthorized. Re-issue the token at "
            "kaggle.com > Settings > API and re-enter it."
        )

    if slugs is None:
        slugs = (
            config.get("caption_audio_dataset"),
            config.get("kaggle_model_dataset"),
            config.get("moss_model_dataset"),
        )
    result["problems"].extend(slug_problems(*slugs))

    # The actual proof: one authenticated round trip.
    try:
        api.kernels_list(mine=True, page_size=1)
    except Exception as e:  # noqa: BLE001 -- the whole point is to report it
        result["detail"] = f"{type(e).__name__}: {e}"
        return result

    result["ok"] = True
    return result


def preflight_kaggle(config, *slugs):
    """Raise RuntimeError when Kaggle is unusable, else return the probe result.

    Called at the TOP of a Kaggle run, so the user learns in seconds what
    otherwise surfaces minutes later as an opaque 401 — or never, when a worker
    thread dies silently. It runs inside the run's own worker thread, so it never
    blocks the GUI.
    """
    result = probe_kaggle(config, slugs=slugs)
    if result["ok"] and not result["problems"]:
        return result

    lines = ["Kaggle is not usable right now.", ""]
    lines.append(f"Credentials: {result['username'] or '(unknown)'}")
    if result["auth_method"]:
        lines.append(f"Auth method: {result['auth_method']}")
    if result["detail"]:
        lines += ["", result["detail"]]
    for problem in result["problems"]:
        lines += ["", problem]
    raise RuntimeError("\n".join(lines))


def dataset_sources(*slugs):
    """Dataset slugs for kernel metadata, with empties dropped.

    WHY THIS EXISTS: ``dataset_sources`` used to be built as
    ``[audio_slug, model_slug]`` unconditionally. "No cached weights dataset" is a
    LEGITIMATE choice — the kernel then downloads ``ACE-Step/acestep-captioner``
    from Hugging Face inside the session — but expressing it as a blank setting
    put an empty string in the kernel metadata, which the API rejects. Callers
    pass whatever they have; the empty ones simply are not sent.
    """
    return [str(slug).strip() for slug in slugs if str(slug or "").strip()]


def upload_audio_dataset(config, audio_dir, title_prefix="ace-audio"):
    """Upload a directory of audio as a private Kaggle dataset.

    Returns the ``user/slug`` of the new dataset.
    """
    api, user = _get_api(config)
    slug = f"{title_prefix}-{uuid.uuid4().hex[:6]}"
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "isPrivate": True,
        "licenses": [{"name": "unknown"}],
    }
    with open(os.path.join(audio_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    try:
        api.dataset_create_new(folder=audio_dir, dir_mode="skip")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle dataset upload failed: {e}") from e

    return f"{user}/{slug}"


def upload_or_update_audio_dataset(config, audio_dir, slug="", title_prefix="ace-audio"):
    """Upload ``audio_dir`` as a private Kaggle dataset, or update one in place.

    With a remembered ``slug`` this pushes a NEW VERSION of the SAME dataset
    (``dataset_create_version``), which is what makes "add or remove songs from
    the uploaded dataset" work: the dataset's contents change while its identity
    — and therefore any kernel/bookmark pointing at it — does not.

    Without a slug it creates a new dataset (the old behaviour) and returns the
    new ``user/slug`` so the caller can remember it.

    WHY NOT ALWAYS CREATE NEW: every run used to create ``ace-audio-<random6>``,
    so a re-run left the previous dataset behind, orphaned and still counting
    against the user's dataset quota, and there was no way to correct one bad
    file without a whole new dataset.
    """
    api, _user = _get_api(config)
    slug = (slug or "").strip()
    meta = {
        "id": slug or "",
        "title": slug.split("/")[-1] if slug else "",
        "isPrivate": True,
        "licenses": [{"name": "unknown"}],
    }
    if slug:
        with open(os.path.join(audio_dir, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        try:
            api.dataset_create_version(
                folder=audio_dir, version_notes="ACE-Step caption upload",
                dir_mode="skip",
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Kaggle dataset VERSION upload failed for {slug}: {e}"
            ) from e
        return slug

    return upload_audio_dataset(config, audio_dir, title_prefix=title_prefix)


def wait_dataset_ready(config, audio_slug, timeout=300, poll_seconds=10, reason=None):
    """Poll a just-created private dataset until Kaggle reports it ``ready``.

    ``dataset_create_new`` returns before the dataset version is fully
    processed/mounted. Pushing a kernel that references the dataset before it
    is ready results in an empty ``/kaggle/input/<slug>`` mount and a silent
    empty manifest. Returns True when ready; False on timeout.

    ``reason``: optional list. When given, the last observed status — or the last
    error — is appended to it, so a caller can say WHY the wait failed.

    WHY THAT MATTERS: this loop deliberately swallows exceptions, because a
    transient network error should not end a 5-minute wait. The cost is that a
    PERMANENT error (`dataset_status` returns 404 for a dataset this account does
    not own — confirmed against the live API) is indistinguishable from a slow
    dataset, and the run then aborts with a bare "did not become ready in time".
    Reporting the last error turns that into a usable clue.
    """
    api, user = _get_api(config)
    elapsed = 0
    note = f"no status was ever returned in {timeout}s"
    while elapsed < timeout:
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        try:
            status = api.dataset_status(audio_slug)
            if isinstance(status, dict):
                status = status.get("status") or sorted(status.values())[-1] if status else ""
            note = f"last status: {status!r}"
            if str(status).lower() in {"ready", "complete"}:
                return True
        except Exception as e:  # noqa: BLE001 — keep polling on transient errors
            note = f"last error: {type(e).__name__}: {e}"
            continue
    if reason is not None:
        reason.append(note)
    return False


def push_kernel(config, kernel_dir, kernel_slug):
    """Push a kernel directory. Returns ``user/kernel_slug``."""
    api, user = _get_api(config)
    try:
        api.kernels_push(kernel_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle kernel push failed: {e}") from e

    return f"{user}/{kernel_slug}"


def wait_kernel_done(config, kernel_slug, timeout=1200, poll_seconds=20):
    """Poll the kernel status until complete/failed. Returns bool success."""
    api, user = _get_api(config)
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        try:
            status = api.kernels_status(f"{user}/{kernel_slug}")
        except Exception:  # noqa: BLE001 — transient network hiccup, keep polling
            continue

        out = str(status).lower()
        if "complete" in out:
            return True
        if "error" in out or "failed" in out or "cancelled" in out:
            return False

    return False


def fetch_kernel_logs(config, kernel_slug, max_chars=6000):
    """Return the TAIL of a kernel's execution log, or '' if unavailable.

    Failures surface at the end of the log, so the tail is what matters. The
    returned text is meant to be embedded in an error message: without it the
    user is told to go read Kaggle's web UI, which is a poor trade when the API
    can hand the reason over directly.
    """
    try:
        api, user = _get_api(config)
        text = api.kernels_logs(f"{user}/{kernel_slug}")
    except Exception:  # noqa: BLE001 — diagnostics must never mask the real error
        return ""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= max_chars else "…" + text[-max_chars:]


def kernel_status_text(config, kernel_slug):
    """Human-readable kernel status, or '' when it cannot be read."""
    try:
        api, user = _get_api(config)
        return str(api.kernels_status(f"{user}/{kernel_slug}"))
    except Exception:  # noqa: BLE001
        return ""


def kernel_stdout(config, kernel_slug):
    """Return what a kernel PRINTED to stdout, parsed from its log.

    ``kernels_logs`` returns a JSON array of stream events::

        [{"stream_name": "stdout", "time": 7.9, "data": "..."}, ...]

    Joining the ``stdout`` events gives the kernel's actual console output,
    which is how results are retrieved. This exists because ``kernels_output``
    turned out to be unreliable: it HUNG outright on a completed kernel (a 90s
    timeout with no output), which is what made an earlier run report "no
    moss_out.json" when the file had in fact been written.

    Falls back to the raw log text when it is not parseable as events.
    """
    raw = fetch_kernel_logs(config, kernel_slug, max_chars=10 ** 9)
    if not raw:
        return ""
    try:
        events = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(events, list):
        return raw
    return "".join(
        event.get("data", "")
        for event in events
        if isinstance(event, dict) and event.get("stream_name") == "stdout"
    )


def extract_marked_json(text, begin, end):
    """Return the JSON payload printed between two markers, or None.

    Markers beat scraping a log for a filename: the payload survives reordering,
    progress bars and unrelated output.
    """
    if not text:
        return None
    start = text.find(begin)
    if start < 0:
        return None
    start += len(begin)
    stop = text.find(end, start)
    if stop < 0:
        return None
    try:
        return json.loads(text[start:stop].strip())
    except ValueError:
        return None


def download_kernel_output(config, kernel_slug, out_dir):
    """Download the kernel output folder. Returns out_dir."""
    api, user = _get_api(config)
    os.makedirs(out_dir, exist_ok=True)
    try:
        api.kernels_output(f"{user}/{kernel_slug}", path=out_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Kaggle kernel output download failed: {e}") from e

    return out_dir
