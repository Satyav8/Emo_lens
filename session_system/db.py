# session_system/db.py
"""
Supabase client helper.

- Uses environment variables SUPABASE_URL and SUPABASE_KEY.
- Returns a client or None if not configured.
- Safe to call repeatedly (returns a single client instance).
"""

import os
from typing import Optional

try:
    # modern supabase client
    from supabase import create_client, Client
except Exception:
    # graceful fallback if package missing
    create_client = None
    Client = None

_client: Optional[Client] = None

def _get_env_var(key: str):
    # Prefer environment variables (Render / Streamlit secrets mapped to env)
    val = os.getenv(key)
    return val

def get_db():
    """
    Return a Supabase client or None.
    Use this instead of creating multiple clients.
    """
    global _client
    if _client is not None:
        return _client

    url = _get_env_var("SUPABASE_URL")
    key = _get_env_var("SUPABASE_KEY")

    if not url or not key or create_client is None:
        # Not configured or supabase package not installed
        return None

    try:
        _client = create_client(url, key)
        return _client
    except Exception as e:
        # fail gracefully
        print("Supabase client init error:", e)
        return None


