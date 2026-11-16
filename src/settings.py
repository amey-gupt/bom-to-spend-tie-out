"""Keys and deployment settings. Values come from Streamlit secrets first, then the environment (or .env.local).

On a hosted copy with no API keys of its own, a visitor can supply theirs for one session. Those keys are handed in with
session_keys() for the length of one call and are never stored here, written to disk, logged or put in a message.

Nothing here prints or logs a value. Callers pass keys straight to the API clients.
"""
from __future__ import annotations

import contextvars
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

API_KEYS = {"ANTHROPIC_API_KEY": "Anthropic", "OPENAI_API_KEY": "OpenAI"}
# Per thread of execution, so one visitor's keys can never be read by another visitor's run.
_visitor_keys: contextvars.ContextVar[dict | None] = contextvars.ContextVar("visitor_keys", default=None)


class ApiKeyError(Exception):
    """An API rejected the key it was given. The message names the provider and never contains the key."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"The {provider} API key was rejected. Check that it is correct and has not expired, then try again.")


@contextmanager
def session_keys(keys: dict | None):
    """Use a visitor's own keys inside this block, for any key the server does not have itself."""
    token = _visitor_keys.set({k: v for k, v in (keys or {}).items() if k in API_KEYS and v})
    try:
        yield
    finally:
        _visitor_keys.reset(token)


def redact(text: object) -> str:
    """A message safe to show: any key in use, and anything shaped like a key, is removed."""
    out = str(text)
    for value in list((_visitor_keys.get() or {}).values()) + [server_secret(name) for name in API_KEYS]:
        if value:
            out = out.replace(value, "[key removed]")
    return re.sub(r"sk-[A-Za-z0-9_\-*.]{6,}", "[key removed]", out)


def _load_env_files():
    for name in (".env.local", ".env"):
        if Path(name).exists():
            for line in Path(name).read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def secret(name: str, default: str = "") -> str:
    """The server's own value; for an API key the server lacks, the visitor's key for this call, if one was handed in."""
    value = server_secret(name)
    if not value and name in API_KEYS:
        value = (_visitor_keys.get() or {}).get(name, "")
    return value or default


def server_secret(name: str, default: str = "") -> str:
    """Streamlit secrets when the app is running and has them; otherwise the environment."""
    st = sys.modules.get("streamlit")  # only inside the app; command-line runs never import Streamlit for this
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:  # no secrets file is the normal case on a laptop
            pass
    _load_env_files()
    return os.environ.get(name, default)


def hosted() -> bool:
    """True on a public deployment (a HOSTED secret). Turns on the access code, the limits and no-disk-writes."""
    return secret("HOSTED").strip().lower() in ("1", "true", "yes", "on")
