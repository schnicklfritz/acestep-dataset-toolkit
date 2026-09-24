"""Credential handling for the caption run.

Two failure modes are pinned here, both of which are invisible in a diff:

1. SILENT DEGRADATION — ``resolve_backend()`` alone turns "ace_step with no
   Kaggle key" into DeepSeek (text-only, NO audio) and then into the local rule
   engine (canned template text). A page titled "ACE-Step (Kaggle)" could
   therefore write placeholder captions that look entirely real.
   ``_resolve_caption_backend()`` must ASK instead.
2. ACCIDENTAL STORAGE — a key the user chose not to remember must not reach the
   encrypted store, and must not be copied into the plaintext
   ``~/.kaggle/access_token`` file the SDK will happily read.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataset_manager as dm                                           # noqa: E402
from modules import kaggle as kaggle_mod                               # noqa: E402


# ---------------------------------------------------------------------------
# the plaintext token file (the SDK's fallback, after KAGGLE_API_TOKEN)
# ---------------------------------------------------------------------------

def test_token_file_is_not_written_when_the_user_did_not_remember(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_mod, "ACCESS_TOKEN_FILE", str(tmp_path / "access_token"))
    written = kaggle_mod._write_access_token_file({"remember_kaggle_key": False}, "KGAT_x")
    assert written is False
    assert not (tmp_path / "access_token").exists()


def test_token_file_is_written_when_remembering(tmp_path, monkeypatch):
    path = tmp_path / "access_token"
    monkeypatch.setattr(kaggle_mod, "ACCESS_TOKEN_FILE", str(path))
    assert kaggle_mod._write_access_token_file({"remember_kaggle_key": True}, "KGAT_x") is True
    assert path.read_text(encoding="utf-8") == "KGAT_x"
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_an_existing_token_file_is_left_alone(tmp_path, monkeypatch):
    """It is a SHARED location (other Kaggle tools read it): never overwrite it."""
    path = tmp_path / "access_token"
    path.write_text("SOMEONE_ELSES_TOKEN", encoding="utf-8")
    monkeypatch.setattr(kaggle_mod, "ACCESS_TOKEN_FILE", str(path))
    assert kaggle_mod._write_access_token_file({"remember_kaggle_key": True}, "KGAT_ours") is False
    assert path.read_text(encoding="utf-8") == "SOMEONE_ELSES_TOKEN"


def test_the_env_var_carries_the_key_even_when_the_file_is_not_written(tmp_path, monkeypatch):
    """The file is only a fallback, so skipping it must not lose authentication."""
    monkeypatch.setattr(kaggle_mod, "ACCESS_TOKEN_FILE", str(tmp_path / "access_token"))
    monkeypatch.setattr(kaggle_mod, "_get_api", lambda config: (None, "someone"))
    try:
        kaggle_mod.ensure_kaggle_creds({"kaggle_user": "someone", "kaggle_key": "KGAT_x",
                                        "remember_kaggle_key": False})
    except Exception:      # noqa: BLE001 — _get_api is stubbed; only the env matters
        pass
    assert os.environ.get("KAGGLE_API_TOKEN") == "KGAT_x"
    assert os.environ.get("KAGGLE_USERNAME") == "someone"
    assert not (tmp_path / "access_token").exists()


# ---------------------------------------------------------------------------
# backend resolution: ask, never degrade silently
# ---------------------------------------------------------------------------

class _Stub:
    """A DatasetManager without the window.

    The three methods under test are the REAL ones, borrowed unbound so a plain
    object can call them (they only read ``self.config``); the two dialogs are
    stubbed so a test can say what the user chose.
    """

    _resolve_caption_backend = dm.DatasetManager._resolve_caption_backend
    _fallback_backend_options = dm.DatasetManager._fallback_backend_options
    _stamp_placeholder_caption = dm.DatasetManager._stamp_placeholder_caption
    PLACEHOLDER_CAPTION_BACKENDS = dm.DatasetManager.PLACEHOLDER_CAPTION_BACKENDS

    credential_answer = False
    fallback_answer = ""

    def _remembered_secret_keys(self):
        return {"kaggle_key"}

    def __init__(self, **overrides):
        self.config = dict(dm.DEFAULT_CONFIG)
        self.config.update(overrides)
        self.calls = []

    # Dialog stand-ins, so a test can say what the user chose.
    def _ensure_kaggle_credentials(self):
        self.calls.append("credentials")
        if self.credential_answer:
            self.config["kaggle_user"] = "someone"
            self.config["kaggle_key"] = "KGAT_key"
        return self.credential_answer

    def _prompt_fallback_backend(self):
        self.calls.append("fallback")
        return self.fallback_answer


@pytest.fixture
def stub_factory(monkeypatch):
    saved = []
    monkeypatch.setattr(dm, "save_config", lambda cfg, remember=None: saved.append(cfg))
    _Stub.saved = saved

    def make(**overrides):
        stub = _Stub(**overrides)
        stub.calls = []
        return stub
    return make


def test_credentials_present_go_straight_to_kaggle(stub_factory):
    stub = stub_factory(kaggle_user="someone", kaggle_key="KGAT_key",
                        caption_backend="ace_step")
    assert stub._resolve_caption_backend() == "Kaggle Cloud (Free GPU)"
    assert stub.calls == []          # asked nothing


def test_no_credentials_prompts_and_never_degrades_silently(stub_factory):
    stub = stub_factory(caption_backend="ace_step")
    stub.credential_answer = False
    stub.fallback_answer = ""        # user cancelled the fallback list too
    backend = stub._resolve_caption_backend()
    assert stub.calls == ["credentials", "fallback"]
    assert backend == "", "a cancelled run must not pick an engine by itself"


def test_declining_offers_the_list_and_remembers_the_choice(stub_factory):
    stub = stub_factory(caption_backend="ace_step")
    stub.credential_answer = False
    stub.fallback_answer = "Gemini"
    assert stub._resolve_caption_backend() == "Gemini"
    assert stub.config["caption_cred_prompt_seen"] == ["ace_step"]
    assert stub.config["caption_fallback_backend"] == "Gemini"


def test_the_choice_is_reused_per_backend_without_asking_again(stub_factory):
    stub = stub_factory(caption_backend="ace_step",
                        caption_cred_prompt_seen=["ace_step"],
                        caption_fallback_backend="Local Rule Engine")
    assert stub._resolve_caption_backend() == "Local Rule Engine"
    assert stub.calls == [], "the prompt must be once per backend, not per run"


def test_credentials_override_the_remembered_fallback(stub_factory):
    """A key that appears later must win — the memory only covers the silent case."""
    stub = stub_factory(kaggle_user="someone", kaggle_key="KGAT_key",
                        caption_backend="ace_step",
                        caption_cred_prompt_seen=["ace_step"],
                        caption_fallback_backend="Local Rule Engine")
    assert stub._resolve_caption_backend() == "Kaggle Cloud (Free GPU)"
    assert stub.calls == []


def test_an_explicit_backend_choice_is_honoured_without_asking(stub_factory):
    stub = stub_factory(caption_backend="gemini")
    assert stub._resolve_caption_backend() == "Gemini"
    assert stub.calls == []


def test_the_fallback_list_offers_every_engine(stub_factory):
    options = stub_factory()._fallback_backend_options()
    assert [row[1] for row in options] == [
        "DeepSeek Cloud", "Gemini", "Custom Endpoint / Webhook", "Local Rule Engine",
    ]
    labels = " ".join(row[0].lower() for row in options)
    assert "no audio" in labels      # the text-only engine says so
    assert "canned" in labels        # and so does the rule engine


# ---------------------------------------------------------------------------
# placeholder stamping — only the backends that never heard the audio
# ---------------------------------------------------------------------------

def test_backends_that_never_heard_the_audio_are_stamped(stub_factory):
    for backend in ("Local Rule Engine", "DeepSeek Cloud"):
        sample = {"caption": "plausible-looking text", "caption_ai_model": "ace_step"}
        stub_factory(caption_stamp_placeholders=True)._stamp_placeholder_caption(
            sample, backend
        )
        assert sample["caption_is_placeholder"] is True, backend
        assert "PLACEHOLDER" in sample["caption_ai_model"], backend


def test_real_backends_are_not_stamped(stub_factory):
    for backend in ("Kaggle Cloud (Free GPU)", "Gemini", "Custom Endpoint / Webhook"):
        sample = {"caption": "real caption", "caption_ai_model": "ace_step"}
        stub_factory()._stamp_placeholder_caption(sample, backend)
        assert "caption_is_placeholder" not in sample, backend
        assert sample["caption_ai_model"] == "ace_step", backend


def test_the_stamp_never_enters_the_caption_text(stub_factory):
    """A stamp inside the text would itself become training data."""
    sample = {"caption": "kept text", "caption_ai_model": ""}
    stub_factory()._stamp_placeholder_caption(sample, "Local Rule Engine")
    assert sample["caption"] == "kept text"


def test_stamping_can_be_switched_off(stub_factory):
    sample = {}
    stub_factory(caption_stamp_placeholders=False)._stamp_placeholder_caption(
        sample, "Local Rule Engine"
    )
    assert "caption_is_placeholder" not in sample


# ---------------------------------------------------------------------------
# wiring (source-level, the way tests/test_caption_spec.py pins placeholders)
# ---------------------------------------------------------------------------

def _manager_source():
    with open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8") as fh:
        return fh.read()


def test_the_caption_run_resolves_instead_of_calling_resolve_backend():
    source = _manager_source()
    assert "backend = self._resolve_caption_backend()" in source
    assert "backend = resolve_backend(self.config)" not in source, (
        "a bare resolve_backend() call is the silent degradation this guards against"
    )
    assert "self._stamp_placeholder_caption(" in source


def test_the_pipeline_prompts_share_one_helper():
    source = _manager_source()
    assert source.count("if not self._ensure_kaggle_credentials():") == 3
    assert "Enter Kaggle username" not in source, "an old inline prompt survived"
