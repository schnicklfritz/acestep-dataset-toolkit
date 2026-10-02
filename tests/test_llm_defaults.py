"""The default LLM is a free provider, and no pipeline assumes DeepSeek."""
import ast
import os

from config import DEFAULT_CONFIG, SECRET_KEYS
from modules.llm_client import DEFAULT_PROVIDER, KNOWN_MODELS, PROVIDERS, provider_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_provider_is_free():
    assert DEFAULT_PROVIDER == "groq"
    assert DEFAULT_CONFIG["llm_provider"] == DEFAULT_PROVIDER
    assert PROVIDERS[DEFAULT_PROVIDER]["free"] is True
    assert provider_info({})[0] == DEFAULT_PROVIDER


def test_unknown_provider_falls_back_to_the_free_default():
    assert provider_info({"llm_provider": "nope"})[0] == DEFAULT_PROVIDER


def test_every_provider_default_model_is_in_its_picker():
    for name, info in PROVIDERS.items():
        if info["model"]:
            assert info["model"] in KNOWN_MODELS[name], name


def test_free_providers_link_to_a_key_page():
    for name, info in PROVIDERS.items():
        if info["free"]:
            assert info["signup_url"].startswith("https://"), name


def test_no_pipeline_prompts_for_a_deepseek_key():
    with open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert '"DeepSeek API Key"' not in src
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_ensure_llm_key" in names


# ---------------------------------------------------------------------------
# OpenAI route + the Gemini 3.8 default
# ---------------------------------------------------------------------------
def test_openai_provider_is_registered_with_a_real_endpoint():
    info = PROVIDERS["openai"]
    assert info["base_url"] == "https://api.openai.com/v1"
    assert info["key"] == "openai_key"
    assert info["signup_url"].startswith("https://")


def test_the_openai_key_is_a_secret_not_a_plain_setting():
    """A key that reached settings.json would be written to disk in the clear."""
    assert "openai_key" in SECRET_KEYS
    assert DEFAULT_CONFIG["openai_key"] == ""


def test_selecting_openai_resolves_its_own_key(monkeypatch):
    from modules.llm_client import provider_key_present

    assert provider_key_present({"llm_provider": "openai"}) is False
    assert provider_key_present(
        {"llm_provider": "openai", "openai_key": "sk-test"}
    ) is True


def test_a_per_role_override_can_pick_openai_for_the_assistant():
    name, info = provider_info(
        {"llm_provider": "groq", "llm_provider_assistant": "openai"},
        role="assistant",
    )
    assert name == "openai"
    assert info["base_url"] == "https://api.openai.com/v1"


def test_gemini_defaults_to_the_3_8_flash_model():
    assert PROVIDERS["gemini"]["model"] == "gemini-3.8-flash"
    assert PROVIDERS["gemini"]["model"] in KNOWN_MODELS["gemini"]
    assert "gemini-3.8-pro" in KNOWN_MODELS["gemini"]


def test_every_provider_has_a_key_field_wired_into_the_app():
    """A provider whose key has no config field cannot ever be authenticated."""
    with open(os.path.join(ROOT, "dataset_manager.py"), encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "LLM_KEY_FIELDS" for t in node.targets
        ):
            mapping = ast.literal_eval(node.value)
    assert mapping is not None
    for name in PROVIDERS:
        assert name in mapping, name
        key_field, remember_field = mapping[name]
        assert key_field in DEFAULT_CONFIG, key_field
        assert remember_field in DEFAULT_CONFIG, remember_field
