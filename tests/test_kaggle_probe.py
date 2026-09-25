"""Kaggle connectivity: the probe, the KAGGLE_KEY trap, and the preflight.

WHY THESE EXIST
---------------
The app used to treat "two non-empty strings in the config" as proof that Kaggle
worked. It does not: the installed SDK's ``authenticate()`` falls back to legacy
API-key auth on PRESENCE ALONE, with no network check, and reports success — so a
modern ``KGAT_`` access token presented as a legacy key "connects" and then every
real call dies with a bare 401. If even that fallback is unavailable the SDK ends
in ``exit(1)``, i.e. ``SystemExit``, which is a ``BaseException`` and therefore
invisible to a worker's ``except Exception`` — the thread dies silently and the
UI waits forever.

Each test below pins one of those, offline: the fake api is injected, so nothing
here touches the network or a real Kaggle account.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules import kaggle as kg                                       # noqa: E402


class _FakeApi:
    """Stands in for an authenticated KaggleApi."""

    def __init__(self, auth_method="ACCESS_TOKEN", error=None):
        self.config_values = {"auth_method": auth_method}
        self.error = error
        self.calls = []

    def kernels_list(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return []


def _patch(monkeypatch, api, *, available=True, creds_error=None):
    """Inject a fake SDK surface. ``api`` may be an exception to raise."""
    monkeypatch.setattr(kg, "kaggle_available", lambda: available)

    def _creds(config):
        if creds_error is not None:
            raise creds_error
        return "someone"

    monkeypatch.setattr(kg, "ensure_kaggle_creds", _creds)

    def _get_api(config):
        if isinstance(api, BaseException):
            raise api
        return api, "someone"

    monkeypatch.setattr(kg, "_get_api", _get_api)


# ---------------------------------------------------------------------------
# the KAGGLE_KEY trap
# ---------------------------------------------------------------------------

def test_an_access_token_is_never_exported_as_a_legacy_key(tmp_path, monkeypatch):
    """A KGAT_ value is an access token; offering it as KAGGLE_KEY is the trap."""
    monkeypatch.setattr(kg, "ACCESS_TOKEN_FILE", str(tmp_path / "access_token"))
    monkeypatch.setenv("KAGGLE_KEY", "STALE_FROM_AN_EARLIER_CALL")
    kg.ensure_kaggle_creds({"kaggle_user": "someone", "kaggle_key": "KGAT_abc",
                            "remember_kaggle_key": False})
    assert os.environ.get("KAGGLE_API_TOKEN") == "KGAT_abc"
    assert "KAGGLE_KEY" not in os.environ, (
        "a stale key must be POPPED, not merely left in place"
    )


def test_a_legacy_api_key_is_still_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "ACCESS_TOKEN_FILE", str(tmp_path / "access_token"))
    kg.ensure_kaggle_creds({"kaggle_user": "someone", "kaggle_key": "0123abc",
                            "remember_kaggle_key": False})
    assert os.environ.get("KAGGLE_KEY") == "0123abc"


# ---------------------------------------------------------------------------
# the probe: report, never raise
# ---------------------------------------------------------------------------

def test_a_successful_round_trip_is_reported_ok(monkeypatch):
    api = _FakeApi()
    _patch(monkeypatch, api)
    result = kg.probe_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    assert result["ok"] is True
    assert result["username"] == "someone"
    assert result["auth_method"] == "ACCESS_TOKEN"
    assert result["problems"] == []
    assert api.calls == [{"mine": True, "page_size": 1}], "must make a real call"


def test_a_rejected_call_is_reported_not_raised(monkeypatch):
    """A 401 from the API must come back as a value, not an exception."""
    _patch(monkeypatch, _FakeApi(error=RuntimeError("401 Unauthorized")))
    result = kg.probe_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    assert result["ok"] is False
    assert "401 Unauthorized" in result["detail"]


def test_the_sdk_exiting_is_survived(monkeypatch):
    """authenticate() ends in exit(1); SystemExit is NOT an Exception.

    If it escapes, a QThread dies without emitting `failed` and the UI waits
    forever -- the "Kaggle never connects" symptom with nothing on screen.
    """
    _patch(monkeypatch, SystemExit(1))
    result = kg.probe_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    assert result["ok"] is False
    assert "SystemExit" in result["detail"]


def test_the_legacy_fallback_is_named_as_the_problem(monkeypatch):
    """KGAT_ token + LEGACY_API_KEY auth = 'connected' but every call 401s."""
    _patch(monkeypatch, _FakeApi(auth_method="LEGACY_API_KEY"))
    result = kg.probe_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    assert result["ok"] is True, "the SDK does report success here -- that is the trap"
    assert result["problems"], "the trap must be named, not silently accepted"
    assert "ACCESS TOKEN" in result["problems"][0]


def test_a_missing_kaggle_package_is_explained(monkeypatch):
    _patch(monkeypatch, _FakeApi(), available=False)
    result = kg.probe_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    assert result["ok"] is False
    assert "pip install kaggle" in result["detail"]


def test_missing_credentials_are_reported_without_raising(monkeypatch):
    _patch(monkeypatch, _FakeApi(),
           creds_error=ValueError("Kaggle credentials not configured."))
    result = kg.probe_kaggle({"kaggle_user": "", "kaggle_key": ""})
    assert result["ok"] is False
    assert "not configured" in result["detail"]


# ---------------------------------------------------------------------------
# dataset slugs: a placeholder fails at CONNECT time, not at push time
# ---------------------------------------------------------------------------

def test_a_placeholder_slug_is_rejected():
    problems = kg.slug_problems("me/my-weights")
    assert problems, "a non-existent dataset must be flagged"
    assert "owner/slug" in problems[0]


def test_valid_and_blank_slugs_are_accepted():
    assert kg.slug_problems("michelmoalem9b/acestep-captioner-model") == []
    assert kg.slug_problems("") == [], "blank = 'download from HF instead'"
    assert kg.slug_problems(None) == []


def test_a_bad_unrelated_slug_does_not_fail_this_run(monkeypatch):
    """MOSS must not fail because the CAPTION weights field is wrong."""
    _patch(monkeypatch, _FakeApi())
    config = {"kaggle_user": "someone", "kaggle_key": "KGAT_abc",
              "caption_audio_dataset": "me/my-weights"}
    result = kg.probe_kaggle(config, slugs=(config.get("moss_model_dataset"),))
    assert result["problems"] == []


# ---------------------------------------------------------------------------
# the preflight every run path calls
# ---------------------------------------------------------------------------

def test_the_preflight_raises_with_the_reason(monkeypatch):
    _patch(monkeypatch, _FakeApi(error=RuntimeError("403 Forbidden")))
    with pytest.raises(RuntimeError) as excinfo:
        kg.preflight_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    message = str(excinfo.value)
    assert "403 Forbidden" in message
    assert "Kaggle is not usable" in message


def test_the_preflight_returns_quietly_when_healthy(monkeypatch):
    _patch(monkeypatch, _FakeApi())
    result = kg.preflight_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"})
    assert result["ok"] is True


def test_the_preflight_fails_on_a_bad_slug_alone(monkeypatch):
    _patch(monkeypatch, _FakeApi())          # the API itself is fine
    with pytest.raises(RuntimeError, match="owner/slug"):
        kg.preflight_kaggle({"kaggle_user": "someone", "kaggle_key": "KGAT_abc"},
                            "me/my-weights")


# ---------------------------------------------------------------------------
# wiring (source-level, the way test_kaggle_dataset_wait.py pins its callers)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "workers/caption.py",           # ACE-Step captioner
    "workers/kaggle_moss.py",       # MOSS-Audio
    "workers/kaggle_lyrics.py",     # Kaggle lyrics
    "workers/kaggle_stems.py",      # Demucs stems
    "workers/structure.py",         # SongFormer structure
    "workers/structural.py",        # section captioning
])
def test_every_kaggle_run_path_runs_the_preflight(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        source = fh.read()
    assert "preflight_kaggle," in source, f"{path} does not import the preflight"
    assert "preflight_kaggle(" in source, f"{path} never calls the preflight"


# ---------------------------------------------------------------------------
# the removed legacy CLI path stays removed
# ---------------------------------------------------------------------------

def test_the_legacy_cli_master_pipeline_stays_removed():
    """It could not work, and its failures were invisible.

    ``kaggle datasets version -p <a .zip file>`` and
    ``kaggle kernels push -p core/kaggle_worker.py`` (a file, with no
    kernel-metadata.json anywhere) are both invalid, ``os.system`` return codes
    were never checked, the output went to a terminal rather than the GUI, and
    the success slot imported ``core.manifest_sync`` -- a module that does not
    exist. This test keeps it from creeping back.
    """
    with open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8") as fh:
        source = fh.read()
    for token in ("kaggle_master_btn", "start_remote_consolidated_pipeline",
                  "KaggleConsolidatedWorker", "core.manifest_sync",
                  'os.system("kaggle'):
        assert token not in source, f"legacy path returned: {token}"
    for dead in ("workers/kaggle_consolidated.py", "core/kaggle_worker.py"):
        assert not os.path.exists(os.path.join(ROOT, dead)), dead
    # ...while the shared worker callbacks it used to share must SURVIVE.
    for kept in ("def on_worker_error", "def on_worker_progress"):
        assert kept in source, f"shared callback lost: {kept}"
