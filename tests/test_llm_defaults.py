"""The default LLM is a free provider, and no pipeline assumes DeepSeek."""
import ast
import os

from config import DEFAULT_CONFIG
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
