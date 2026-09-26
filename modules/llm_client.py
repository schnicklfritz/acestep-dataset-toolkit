"""Pluggable LLM provider resolution for the app's text-LLM needs.

Everything the app asks an LLM to do is text-based (master-caption
aggregation, instrument-model recommendation, the AI assistant), so every
provider is reached through the **OpenAI-compatible Chat Completions API** —
one client, one ``base_url``:

  * ``groq``      — https://api.groq.com/openai/v1                    (free tier)  [default]
  * ``gemini``    — https://generativelanguage.googleapis.com/v1beta/openai/  (free tier)
  * ``openrouter``— https://openrouter.ai/api/v1                      (free ``:free`` models)
  * ``deepseek``  — https://api.deepseek.com/v1                       (cheap, paid)

Default is Groq because it had the most generous verifiable free quota on
2026-09-25 (console.groq.com/docs/rate-limits): openai/gpt-oss-120b at 30 RPM,
1,000 requests/day, 200K tokens/day, 131K context, tool calling supported --
the assistant needs tools. Gemini's free Flash tier was down to ~20 req/day
and Google uses free-tier content for training; OpenRouter free is 50 req/day
without credits. Re-check these numbers when changing the default.
  * ``local``     — any OpenAI-compatible server (vLLM / Ollama / llama.cpp / rented GPU)

``llm_provider`` picks the provider; ``llm_model`` / ``llm_base_url`` override
the per-provider defaults (useful for custom gateways or self-hosted models).
"""
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "key": "groq_key",
        "free": True,
        "label": "Groq (free tier)",
        "signup_url": "https://console.groq.com/keys",
        "note": "Free key, no card: console.groq.com/keys. gpt-oss-120b: 1,000 requests/day, 200K tokens/day, tool calling.",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "key": "gemini_api_key",
        "free": True,
        "label": "Gemini (free tier)",
        "signup_url": "https://aistudio.google.com/apikey",
        "note": "Free key from aistudio.google.com/apikey. Flash-Lite has the larger free quota. Free-tier content is used by Google to improve its products.",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "key": "openrouter_key",
        "free": True,
        "label": "OpenRouter (free models)",
        "signup_url": "https://openrouter.ai/keys",
        "note": "Free ':free' models: 50 requests/day (1,000 after a one-time $10 credit). Key from openrouter.ai/keys.",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key": "deepseek_key",
        "free": False,
        "label": "DeepSeek (paid)",
        "signup_url": "https://platform.deepseek.com/api_keys",
        "note": "Cheap paid API; needs a DeepSeek API key.",
    },
    "local": {
        "base_url": "",
        "model": "",
        "key": "custom_key",
        "free": None,
        "label": "Local / custom endpoint",
        "signup_url": "",
        "note": "Point the Custom Endpoint URL at any vLLM / Ollama / llama.cpp / rented-GPU server.",
    },
}

DEFAULT_PROVIDER = "groq"

# Model names offered in the pickers. Only models verified as served on
# 2026-09-25; an empty entry means "the provider's default".
KNOWN_MODELS = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"],
    "gemini": ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"],
    "openrouter": ["nvidia/nemotron-3-super-120b-a12b:free", "qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"],
    "deepseek": ["deepseek-chat"],
    "local": [],
}


def provider_info(config, provider=None, role=None):
    """Return ``(provider_name, info)`` with per-provider defaults + overrides.

    ``role`` (``aggregator``/``captioner``/``assistant``) resolves a per-role
    provider and model override when ``provider`` isn't given explicitly.
    """
    if provider is None and role:
        provider = (config.get(f"llm_provider_{role}") or "").strip() or None
    name = (provider or config.get("llm_provider") or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        name = DEFAULT_PROVIDER
    info = dict(PROVIDERS[name])
    model_key = f"llm_model_{role}" if role else "llm_model"
    if name == "local":
        info["base_url"] = (config.get("custom_url") or "").strip()
        info["model"] = (config.get(model_key) or "").strip()
    else:
        info["base_url"] = (config.get("llm_base_url") or "").strip() or info["base_url"]
        info["model"] = (config.get(model_key) or "").strip() or info["model"]
    return name, info


def _key_value(config, info):
    """Resolve the provider's API key, with a backward-compat fallback for
    DeepSeek (the old ``custom_key`` field was used before ``deepseek_key``)."""
    val = (config.get(info.get("key")) or "").strip()
    if not val and info.get("key") == "deepseek_key":
        val = (config.get("custom_key") or "").strip()
    return val


def get_client(config, provider=None, role=None):
    """Return ``(provider_name, info, OpenAI-compatible client)``.

    Raises ``ValueError`` with a clear, actionable message when the provider
    needs configuration the user hasn't provided.
    """
    name, info = provider_info(config, provider, role=role)
    if not info["base_url"]:
        raise ValueError(
            "No LLM endpoint configured — for the 'local' provider set the "
            "Custom Endpoint URL in ⚙ Settings."
        )
    key = _key_value(config, info)
    if name != "local" and not key:
        raise ValueError(
            f"No LLM is configured for this step ({info['label']} has no key).\n\n"
            "Open ⚙ Settings → LLM Provider and either pick a provider or paste "
            "a key.\n\n"
            "Free tiers (no card needed):\n"
            "  • Groq      — console.groq.com/keys\n"
            "  • Gemini    — aistudio.google.com/apikey\n"
            "  • OpenRouter— openrouter.ai/keys\n\n"
            f"What this provider needs: {info.get('note', '')}"
        )
    from openai import OpenAI

    return name, info, OpenAI(api_key=key or "sk-no-key", base_url=info["base_url"])


def provider_key_present(config, provider=None, role=None):
    """True when the provider's required key is present in config."""
    name, info = provider_info(config, provider, role=role)
    if name == "local":
        return bool(info["base_url"])
    return bool(_key_value(config, info))


def effective_provider(config, role=None):
    """Resolve the provider name in effect (global or per-role) for the UI."""
    name, _info = provider_info(config, role=role)
    return name