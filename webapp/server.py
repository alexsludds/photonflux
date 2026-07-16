#!/usr/bin/env python3
"""Local web frontend for photonflux/circulax: schematic editor + simulator.

    .venv-circulax/bin/python webapp/server.py [--port 8642]

Then open http://localhost:8642 — drag photonic/electrical components onto
the canvas, wire them, attach probes, hit Run. Pure stdlib server (no extra
dependencies); simulations run through circulax in-process, serialized by a
lock (JAX solves are single-flight).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # repo root -> photonflux importable

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import simulate  # noqa: E402
from catalog import CATALOG, load_user_va  # noqa: E402

_loaded_va = load_user_va()
if _loaded_va:
    print(f"user VA models: {', '.join(_loaded_va)}")

STATIC = HERE / "static"
EXAMPLES = HERE / "examples"
_RUN_LOCK = threading.Lock()

# --- public-deployment knobs (all default to the local-dev behaviour) --------
# Verilog-A upload compiles untrusted source through the native OpenVAF
# toolchain, so a public host must turn it OFF. It defaults ON so running
# `server.py` locally is unchanged; the container image sets this to "0".
_ALLOW_VA_UPLOAD = os.environ.get("PHOTONFLUX_ALLOW_VA_UPLOAD", "1") == "1"
# Wall-clock guard on a single /api/run. 0 (the default) means "no limit" so
# local dev and the heavy showcase examples (e.g. the Vernier laser, which
# takes minutes) are unaffected. The container sets a generous ceiling to bound
# runaways without killing legitimate long solves.
try:
    _RUN_TIMEOUT_S = float(os.environ.get("PHOTONFLUX_RUN_TIMEOUT_S", "0") or "0")
except ValueError:
    _RUN_TIMEOUT_S = 0.0

_MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
         ".json": "application/json", ".svg": "image/svg+xml",
         ".png": "image/png", ".ico": "image/x-icon"}


def _examples_index() -> list[dict]:
    out = []
    for p in sorted(EXAMPLES.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            out.append({"id": p.stem, "title": data.get("title", p.stem),
                        "description": data.get("description", ""),
                        "group": data.get("group", "More")})
        except json.JSONDecodeError:
            continue
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/components":
            self._json(CATALOG)
            return
        if path == "/api/examples":
            self._json(_examples_index())
            return
        if path.startswith("/api/examples/"):
            p = EXAMPLES / (path.rsplit("/", 1)[1] + ".json")
            if p.is_file() and p.resolve().parent == EXAMPLES.resolve():
                self._send(200, p.read_bytes(), "application/json")
            else:
                self._json({"error": "no such example"}, 404)
            return
        # static files
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if target.is_file() and STATIC.resolve() in target.parents:
            self._send(200, target.read_bytes(),
                       _MIME.get(target.suffix, "application/octet-stream"))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route not in ("/api/run", "/api/upload", "/api/upload_va"):
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20_000_000:
                raise ValueError("payload too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": f"bad request: {exc}"}, 400)
            return
        if route == "/api/upload":
            self._json(self._upload(payload))
            return
        if route == "/api/upload_va":
            if not _ALLOW_VA_UPLOAD:
                self._json({"ok": False, "error": "Verilog-A upload is "
                            "disabled on this server."}, 403)
                return
            with _RUN_LOCK:
                self._json(self._upload_va(payload))
            return
        result = self._run(payload)
        self._json(result, 200 if result.get("ok") else
                   (503 if result.get("_timeout") else 422))

    @staticmethod
    def _run(payload: dict) -> dict:
        """Run one simulation, serialized by _RUN_LOCK. When a wall-clock
        limit is set, the solve runs in a worker that keeps holding the lock
        until it finishes (JAX solves can't be interrupted cleanly), so a
        timed-out run never corrupts the shared circuit caches — the next
        request simply queues behind it."""
        if _RUN_TIMEOUT_S <= 0:
            with _RUN_LOCK:
                return simulate.run(payload)
        box: dict = {}

        def _work() -> None:
            with _RUN_LOCK:
                try:
                    box["result"] = simulate.run(payload)
                except Exception as exc:  # defensive: worker must not vanish
                    box["result"] = {"ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"}

        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        worker.join(_RUN_TIMEOUT_S)
        if worker.is_alive():
            return {"ok": False, "_timeout": True,
                    "error": f"run exceeded the {_RUN_TIMEOUT_S:g}s time "
                             "limit on this server"}
        return box.get("result", {"ok": False, "error": "no result"})

    @staticmethod
    def _upload_va(payload: dict) -> dict:
        """Compile an uploaded Verilog-A model and register it as a
        placeable component (persists in webapp/models_user/)."""
        import re
        import warnings

        import catalog

        content = str(payload.get("content", ""))
        m = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", content)
        if not m:
            return {"ok": False, "error": "no `module` declaration found"}
        stem = m.group(1)
        catalog.USER_VA_DIR.mkdir(exist_ok=True)
        path = catalog.USER_VA_DIR / f"{stem}.va"
        path.write_text(content)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                entry = catalog.register_user_va(stem)
        except Exception as exc:
            path.unlink(missing_ok=True)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        simulate._MODELS_CACHE.clear()   # next run rebuilds with the model
        simulate._CIRCUIT_CACHE.clear()
        return {"ok": True, "type": f"uva_{stem}", "entry": entry,
                "warnings": [str(w.message) for w in caught]}

    @staticmethod
    def _upload(payload: dict) -> dict:
        """Store a text data file (Touchstone etc.) under webapp/uploads."""
        import hashlib

        name = str(payload.get("name", "file"))
        content = str(payload.get("content", ""))
        if not content:
            return {"ok": False, "error": "empty upload"}
        safe = "".join(c for c in name if c.isalnum() or c in "._-")[-48:]
        digest = hashlib.sha256(content.encode()).hexdigest()[:10]
        fid = f"{digest}_{safe or 'file'}"
        uploads = HERE / "uploads"
        uploads.mkdir(exist_ok=True)
        (uploads / fid).write_text(content)
        return {"ok": True, "id": fid, "name": safe}

    def log_message(self, fmt: str, *args) -> None:
        # keep the console to /api/run lines, but never let logging raise:
        # send_error() logs *before* writing the response, and args[0] there
        # is an HTTPStatus (not the request line), so a naive substring test
        # crashes the handler thread and the client gets no response at all.
        first = args[0] if args else ""
        if isinstance(first, str) and "/api/run" in first:
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # $PORT / $HOST let a container (or PaaS like HF Spaces) place the server;
    # the defaults keep local `server.py` on 127.0.0.1:8642 exactly as before.
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8642")))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"photonflux web UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
