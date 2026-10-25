"""Thin wrapper over the Gemini Interactions API: video parts, JSON-schema output, chained history."""
import base64
import json
import os
import time
from pathlib import Path

from google import genai

_client = None


def load_env(path=".env"):
    """Minimal KEY=VALUE loader. .env wins over the shell so a stale exported key can't shadow it."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def client():
    global _client
    if _client is None:
        load_env()
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def model(fast=False):
    load_env()
    name = os.environ.get("GEMINI_FAST_MODEL") if fast else None
    name = name or os.environ.get("GEMINI_MODEL")
    if not name:
        raise SystemExit("Set GEMINI_MODEL in .env (see .env.example).")
    return name


def text(s):
    return {"type": "text", "text": s}


def video(path, fps=None):
    """Inline video part. Clips here are a few MB, so base64 avoids a Files API round trip."""
    part = {"type": "video", "mime_type": "video/mp4",
            "data": base64.b64encode(Path(path).read_bytes()).decode()}
    fps = fps or float(os.environ.get("RC_VIDEO_FPS", "0") or 0)
    if fps:
        part["processing"] = {"type": "static", "fps": fps}
    return part


def ask(parts, schema, previous_id=None, thinking="high", fast=False, system=None):
    """Run one interaction in background mode and return (parsed_json, interaction_id)."""
    body = dict(model=model(fast), input=parts, background=True, generation_config={"thinking_level": thinking},
                response_format={"type": "text", "mime_type": "application/json", "schema": schema})
    if previous_id:
        body["previous_interaction_id"] = previous_id
    if system:
        body["system_instruction"] = system
    c = client()
    it = c.interactions.create(**body)
    while it.status in ("in_progress", "queued"):
        time.sleep(3)
        it = c.interactions.get(id=it.id)
    if it.status != "completed":
        raise RuntimeError(f"interaction {it.id} ended with status {it.status}")
    return json.loads(it.output_text), it.id
