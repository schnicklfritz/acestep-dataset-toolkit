"""Pluggable LLM provider resolution for the app's text-LLM needs.

Everything the app asks an LLM to do is text-based (master-caption
aggregation, instrument-model recommendation, the AI assistant), so every
provider is reached through the **OpenAI-compatible Chat Completions API** —
one client, one ``base_url``:

  * ``deepseek``  — https://api.deepseek.com/v1                       (cheap, paid)  [default]
  * ``gemini``    — https://generativelanguage.googleapis.com/v1beta/openai/  (free tier)
  * ``groq``      — https://api.groq.com/openai/v1                    (free tier)
  * ``openrouter``— https://openrouter.ai/api/v1                      (free ``:free`` models)
  * ``local``     — any OpenAI-compatible server (vLLM / Ollama / llama.cpp / rented GPU)

``llm_provider`` picks the provider; ``llm_model`` / ``llm_base_url`` override
the per-provider defaults (useful for custom gateways or self-hosted models).
"""
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key": "custom_key",
        "free": False,
        "label": "DeepSeek",
        "note": "Cheap paid API; needs a DeepSeek API key.",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key": "gemini_api_key",
        "free": True,
        "label": "Gemini (free tier)",
        "note": "Free tier key from aistudio.google.com/apikey. Flash models are very capable for aggregation and the assistant.",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key": "groq_key",
        "free": True,
        "label": "Groq (free tier)",
        "note": "Free tier key from console.groq.com/keys. Llama 3.3 70B, very fast.",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key": "openrouter_key",
        "free": True,
        "label": "OpenRouter (free models)",
        "note": "Free ':free' models (rate-limited); key from openrouter.ai/keys.",
    },
    "local": {
        "base_url": "",
        "model": "",
        "key": "custom_key",
        "free": None,
        "label": "Local / custom endpoint",
        "note": "Point the Custom Endpoint URL at any vLLM / Ollama / llama.cpp / rented-GPU server.",
    },
}


def provider_info(config, provider=None):
    """Return ``(provider_name, info)`` with per-provider defaults + overrides."""
    name = (provider or config.get("llm_provider") or "deepseek").strip().lower()
    info = dict(PROVIDERS.get(name, PROVIDERS["deepseek"]))
    if name == "local":
        info["base_url"] = (config.get("custom_url") or "").strip()
        info["model"] = (config.get("llm_model") or "").strip()
    else:
        info["base_url"] = (config.get("llm_base_url") or "").strip() or info["base_url"]
        info["model"] = (config.get("llm_model") or "").strip() or info["model"]
    return name, info


def get_client(config, provider=None):
    """Return ``(provider_name, info, OpenAI-compatible client)``.

    Raises ``ValueError`` with a clear, actionable message when the provider
    needs configuration the user hasn't provided.
    """
    name, info = provider_info(config, provider)
    if not info["base_url"]:
        raise ValueError(
            "No LLM endpoint configured — for the 'local' provider set the "
            "Custom Endpoint URL in ⚙ Settings."
        )
    key = (config.get(info["key"]) or "").strip()
    if name != "local" and not key:
        raise ValueError(
            f"{info['label']} needs an API key — set it in ⚙ Settings. "
            f"{info.get('note', '')}"
        )
    from openai import OpenAI

    return name, info, OpenAI(api_key=key or "sk-no-key", base_url=info["base_url"])


def provider_key_present(config, provider=None):
    """True when the provider's required key is present in config."""
    name, info = provider_info(config, provider)
    if name == "local":
        return bool(info["base_url"])
    return bool((config.get(info["key"]) or "").strip())