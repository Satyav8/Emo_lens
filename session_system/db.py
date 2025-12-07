import os
from supabase import create_client

def get_db():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("❌ Supabase env variables not set")
        return None

    return create_client(url, key)

