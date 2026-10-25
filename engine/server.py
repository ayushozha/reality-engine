"""Local server for the CAD editor. Binds to 127.0.0.1 only.

    uv run python -m engine.server [--port 8765]
"""
import argparse
import json
import logging
import mimetypes
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from . import llm, pipeline, verify
from .mjcf import compile_mjcf
from .scene import Scene, load, save

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RUNS = pipeline.RUNS
MAX_UPLOAD = 8 * 1024 ** 3
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
_busy: set[str] = set()
_lock = threading.Lock()


def _safe(base: Path, rel: str) -> Path | None:
    p = (base / rel).resolve()
    return p if p.is_file() and p.is_relative_to(base.resolve()) else None


def _start(run_dir: Path, **kw) -> bool:
    with _lock:
        if run_dir.name in _busy:
            return False
        _busy.add(run_dir.name)

    def work():
        try:
            pipeline.run(run_dir, **kw)
        except Exception:
            pass  # recorded in status.json by the pipeline
        finally:
            with _lock:
                _busy.discard(run_dir.name)
    threading.Thread(target=work, daemon=True).start()
    return True


class Handler(BaseHTTPRequestHandler):
    server_version = "RealityEngine/1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path | None):
        if not path:
            return self._send(404, {"error": "not found"})
        self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def _run_dir(self, run_id):
        if not pipeline.RUN_ID.match(run_id) or not (RUNS / run_id).is_dir():
            return None
        return RUNS / run_id

    def _body(self, limit):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > limit:
            raise ValueError("missing or oversized body")
        return n

    # --- GET -----------------------------------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html")
        if path.startswith("/web/"):
            return self._file(_safe(WEB, path[5:]))
        if path == "/api/config":
            return self._send(200, {"gemini": llm.configured()})
        if path == "/api/runs":
            runs = []
            for d in sorted(RUNS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
                if d.is_dir() and pipeline.RUN_ID.match(d.name):
                    s = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
                    runs.append({"id": d.name, "stages": {k: v["state"] for k, v in s.get("stages", {}).items()},
                                 "busy": d.name in _busy})
            return self._send(200, runs)
        m = re.match(r"^/api/runs/([^/]+)/(status|scene|verify|scene\.mjcf|keyframes)$", path)
        if m:
            d = self._run_dir(m.group(1))
            if not d:
                return self._send(404, {"error": "unknown run"})
            what = m.group(2)
            if what == "scene.mjcf":
                if not (d / "scene.json").exists():
                    return self._send(404, {"error": "no scene yet"})
                return self._send(200, compile_mjcf(load(d / "scene.json"), dynamic=True), "application/xml",
                                  {"Content-Disposition": f'attachment; filename="{d.name}.xml"'})
            f = d / {"status": "status.json", "scene": "scene.json", "verify": "verify.json", "keyframes": "keyframes.json"}[what]
            if not f.exists():
                return self._send(404, {"error": f"no {what} yet"})
            data = json.loads(f.read_text())
            if what == "status":
                data["busy"] = d.name in _busy
            return self._send(200, data)
        m = re.match(r"^/runs/([^/]+)/(.+)$", path)
        if m and (d := self._run_dir(m.group(1))):
            return self._file(_safe(d, m.group(2)))
        self._send(404, {"error": "not found"})

    # --- POST / PUT ----------------------------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/runs":
                return self._upload()
            m = re.match(r"^/api/runs/([^/]+)/(verify|rebuild)$", path)
            d = m and self._run_dir(m.group(1))
            if not d:
                return self._send(404, {"error": "unknown run"})
            if m.group(2) == "rebuild":
                return self._send(202, {"started": True}) if _start(d) else self._send(409, {"error": "already running"})
            scene = load(d / "scene.json")
            return self._send(200, verify.run(scene, d, verify.load_points(d)))
        except (ValueError, ValidationError) as e:
            self._send(400, {"error": str(e)[:2000]})

    def do_PUT(self):
        m = re.match(r"^/api/runs/([^/]+)/scene$", urlparse(self.path).path)
        d = m and self._run_dir(m.group(1))
        if not d:
            return self._send(404, {"error": "unknown run"})
        try:
            n = self._body(64 * 1024 ** 2)
            scene = Scene.model_validate_json(self.rfile.read(n))
            save(scene, d / "scene.json")
            self._send(200, {"ok": True, "objects": len(scene.objects)})
        except (ValueError, ValidationError) as e:
            self._send(400, {"error": str(e)[:2000]})

    def _upload(self):
        name = Path(self.headers.get("X-Filename") or "video.mp4").name
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT:
            raise ValueError(f"unsupported video type {ext or '(none)'}")
        n = self._body(MAX_UPLOAD)
        run_id = (re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:40] or "video") + time.strftime("-%H%M%S")
        d = RUNS / run_id
        d.mkdir(parents=True, exist_ok=False)
        with open(d / f"input{ext}", "wb") as f:
            left = n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError("upload interrupted")
                f.write(chunk)
                left -= len(chunk)
        pipeline.Status(d).log(f"uploaded {name} ({n / 1e6:.1f} MB)")
        _start(d)
        self._send(201, {"id": run_id})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RUNS.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    logging.info("Reality Engine editor: http://127.0.0.1:%d", a.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
