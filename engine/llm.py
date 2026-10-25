"""Thin wrapper over the Gemini Interactions API. Nothing here runs until GEMINI_API_KEY is set."""
import base64
import json
import os
import time
from pathlib import Path

_client = None
ROOT = Path(__file__).resolve().parent.parent


class NotConfigured(RuntimeError):
    pass


def load_env(path=ROOT / ".env"):
    """Minimal KEY=VALUE loader. .env wins over the shell so a stale exported key can't shadow it."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if v:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)  # a blank entry in .env means "unset", not "inherit from the shell"


def configured() -> bool:
    load_env()
    return bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_MODEL"))


def require():
    if not configured():
        raise NotConfigured("Gemini is not configured: set GEMINI_API_KEY and GEMINI_MODEL in .env")


def client():
    global _client
    require()
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def model(kind="main"):
    require()
    env = {"main": "GEMINI_MODEL", "agent": "GEMINI_AGENT_MODEL"}[kind]
    return os.environ.get(env) or os.environ["GEMINI_MODEL"]


def text(s):
    return {"type": "text", "text": s}


INLINE_VIDEO_MAX = 15 * 1024 ** 2


def video_part(path):
    """Small clips go inline; anything larger goes through the Files API."""
    data = Path(path).read_bytes()
    if len(data) <= INLINE_VIDEO_MAX:
        return {"type": "video", "mime_type": "video/mp4", "data": base64.b64encode(data).decode()}
    return upload_video(path)


def upload_video(path, timeout_s=900):
    """Files API upload for long videos (too big to inline). Waits until the file is processed."""
    c = client()
    f = c.files.upload(file=str(path))
    deadline = time.time() + timeout_s
    state = lambda: str(getattr(f.state, "name", f.state)).upper()
    while "PROCESSING" in state():
        if time.time() > deadline:
            raise TimeoutError(f"video {f.name} still processing after {timeout_s} s")
        time.sleep(5)
        f = c.files.get(name=f.name)
    if "ACTIVE" not in state():
        raise RuntimeError(f"video upload {f.name} ended in state {state()}")
    return {"type": "video", "mime_type": f.mime_type or "video/mp4", "uri": f.uri}


def ask(parts, schema, previous_id=None, thinking="high", kind="main", system=None):
    """One background interaction. Returns (parsed_json, interaction_id)."""
    body = dict(model=model(kind), input=parts, background=True, generation_config={"thinking_level": thinking},
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
