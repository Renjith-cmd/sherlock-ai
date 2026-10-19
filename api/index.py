"""Vercel serverless entry point for the Sherlock investigation SPA and API."""

import os

os.environ.setdefault("SHERLOCK_DB_PATH", "/tmp/sherlock_investigations.db")

from src.api.app import create_app


api_token = os.getenv("SHERLOCK_API_TOKEN", "").strip()
if len(api_token) < 16:
    raise RuntimeError("Set SHERLOCK_API_TOKEN to a value of at least 16 characters in Vercel Project Settings.")

app = create_app(
    db_path=os.getenv("SHERLOCK_DB_PATH", "/tmp/sherlock_investigations.db"),
    backend_mode=os.getenv("SHERLOCK_BACKEND_MODE", "rule"),
    api_token=api_token,
    enforce_auth=True,
)
