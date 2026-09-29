"""Provider-agnostic LLM adapter.

Reads config from environment variables; the Streamlit pages bridge ``st.secrets``
into the environment at startup via :func:`bridge_secrets`. The provider SDK is
lazy-imported inside :func:`complete`, so the whole app imports and runs fine with
no key at all — every LLM feature has a rule-based fallback elsewhere.

Note: Groq's ``gpt-oss`` models are *reasoning* models that spend tokens thinking
before answering, so we always request a generous ``max_tokens``.
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_MODEL = "openai/gpt-oss-120b"


def config() -> dict:
    return {
        "provider": os.getenv("LLM_PROVIDER", "groq"),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
        "api_key": os.getenv("LLM_API_KEY", ""),
    }


def available() -> bool:
    """True if an API key is configured (so the UI can offer LLM features)."""
    return bool(config()["api_key"])


def bridge_secrets(secrets) -> None:
    """Copy LLM_* from a mapping (e.g. st.secrets) into os.environ if unset.

    Lets ``core`` stay Streamlit-free while pages still feed it secrets.
    """
    for key in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
        try:
            value = secrets.get(key)
        except Exception:
            value = None
        if value and not os.getenv(key):
            os.environ[key] = str(value)


def complete(system: str, user: str, *, max_tokens: int = 1200,
             temperature: float = 0.3) -> Optional[str]:
    """Return the model's reply, or ``None`` on any missing-key/error condition."""
    cfg = config()
    if not cfg["api_key"]:
        return None
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if cfg["provider"] == "groq":
            from groq import Groq
            client = Groq(api_key=cfg["api_key"])
            resp = client.chat.completions.create(
                model=cfg["model"], messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
        elif cfg["provider"] == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=cfg["api_key"])
            resp = client.chat.completions.create(
                model=cfg["model"], messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
        else:
            return None
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None
