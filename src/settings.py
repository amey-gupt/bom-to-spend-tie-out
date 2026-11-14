"""Keys and deployment settings. Values come from Streamlit secrets first, then the environment (or .env.local).

Nothing here prints or logs a value. Callers pass keys straight to the API clients.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_env_files():
    for name in (".env.local", ".env"):
        if Path(name).exists():
            for line in Path(name).read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def secret(name: str, default: str = "") -> str:
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
