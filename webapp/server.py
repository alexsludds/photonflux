#!/usr/bin/env python3
"""Local web frontend for photonflux/circulax: schematic editor + simulator.

    .venv-circulax/bin/python webapp/server.py [--port 8642] [--no-reload]

Then open http://localhost:8642 — drag photonic/electrical components onto
the canvas, wire them, attach probes, hit Run. Pure stdlib server (no extra
dependencies); simulations run through circulax in-process, serialized by a
lock (JAX solves are single-flight).

Static assets (index.html/app.js/style.css) and example JSON are read from
disk on every request, so editing those is already live — just refresh the
page. The Python engine (simulate.py, catalog.py, the photonflux package) is
imported once at startup, so editing *code* would otherwise need a restart.
Auto-reload is ON by default: the launched process supervises a child server
and restarts it whenever a .py source file changes, so a browser refresh
always shows the latest code. Pass --no-reload (or set PHOTONFLUX_RELOAD=0) to
run a single plain process with no supervisor — the container image does this,
so production runs one process exactly as before.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _finite(o):
    """Replace NaN/Infinity with null so the payload is valid JSON."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # repo root -> photonflux importable

# The heavy imports (jax/circulax + the model catalog) live in _load_engine()
# rather than at module scope so the --reload supervisor process — which only
# re-spawns children and never serves a request — stays light and starts
# instantly. They are bound as module globals the Handler looks up at call time.
simulate = None      # type: ignore[assignment]  # set by _load_engine()
CATALOG: dict = {}   # set by _load_engine()


def _load_engine() -> None:
    """Import the simulation stack (jax/circulax) and register user VA models.

    Called once in the process that actually serves requests. On --reload this
    is the freshly re-spawned child, so every restart re-imports the edited
    code."""
    global simulate, CATALOG
    import jax

    jax.config.update("jax_enable_x64", True)  # circuit solves need float64

    import simulate as _simulate
    from catalog import CATALOG as _CATALOG, load_user_va

    simulate = _simulate
    CATALOG = _CATALOG

    # Opt this (server) process in to live per-step transient progress, which
    # the browser polls via /api/progress. Library/CLI callers don't call this,
    # so they skip the solver's per-save-point host callback entirely.
    import progress
    progress.PROGRESS.enable()

    loaded_va = load_user_va()
    if loaded_va:
        print(f"user VA models: {', '.join(loaded_va)}")


STATIC = HERE / "static"
EXAMPLES = HERE / "examples"
_RUN_LOCK = threading.Lock()

# Live schematic mirror shared with notebook clients (photonflux.nb). Kept in
# a module the supervisor never imports; pure stdlib, so importing it here (at
# module scope, unlike the lazy engine) costs nothing.
from session import SESSION  # noqa: E402

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
# The notebook bridge (/api/schematic + its SSE feed) mirrors whatever is on
# the canvas through one process-global session — correct for the single-user
# local server, but on a shared public host it would leak one visitor's
# schematic to the next and let anyone inject documents into connected
# editors. Defaults ON for local dev; the container image sets it to "0".
_ENABLE_BRIDGE = os.environ.get("PHOTONFLUX_ENABLE_BRIDGE", "1") == "1"
# Cap concurrent SSE subscribers: each holds a handler thread for its whole
# lifetime, so an uncapped public endpoint is a trivial thread/socket DoS.
_SSE_LIMIT = 32
_SSE_COUNT = 0
_SSE_LOCK = threading.Lock()

_MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
         ".json": "application/json", ".svg": "image/svg+xml",
         ".png": "image/png", ".ico": "image/x-icon",
         ".woff2": "font/woff2"}


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
        # json.dumps emits bare NaN/Infinity by default, which is invalid JSON
        # and makes the browser's JSON.parse reject the entire response — a
        # single non-finite number anywhere loses the whole result. Try the
        # strict encoding first (fast, and the normal case), and only pay for
        # sanitising when something non-finite actually slipped in.
        try:
            body = json.dumps(obj, allow_nan=False)
        except ValueError:
            body = json.dumps(_finite(obj), allow_nan=False)
        self._send(code, body.encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/components":
            self._json(CATALOG)
            return
        if path == "/api/examples":
            self._json(_examples_index())
            return
        if path == "/api/schematic":
            # Live mirror of the browser's active tab (see session.py).
            # Lock-free w.r.t. _RUN_LOCK so a notebook can read the canvas
            # while a run is in flight.
            if not _ENABLE_BRIDGE:
                self._json(self._bridge_disabled(), 403)
                return
            self._json(SESSION.get())
            return
        if path == "/api/schematic/events":
            if not _ENABLE_BRIDGE:
                # a non-200 also tells EventSource to stop reconnecting
                self._json(self._bridge_disabled(), 403)
                return
            self._sse_events()
            return
        if path == "/api/progress":
            # Live transient-solve progress, polled by the browser while an
            # /api/run is in flight. Deliberately lock-free (it never touches
            # _RUN_LOCK), so it answers on its own handler thread while the run
            # worker holds the lock and updates the shared state.
            import progress
            self._json(progress.PROGRESS.snapshot())
            return
        if path == "/api/veriloga":
            import catalog
            from urllib.parse import parse_qs, urlsplit
            key = (parse_qs(urlsplit(self.path).query).get("type") or [""])[0]
            src = catalog.veriloga_source(key)
            if src is None:
                self._json({"ok": False,
                            "error": "no Verilog-A source for this component"},
                           404)
            else:
                self._json({"ok": True, "path": src[0], "source": src[1]})
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

    @staticmethod
    def _bridge_disabled() -> dict:
        return {"ok": False, "disabled": True,
                "error": "the notebook bridge is disabled on this server "
                         "(PHOTONFLUX_ENABLE_BRIDGE=0)"}

    def _sse_events(self) -> None:
        """Stream schematic-mirror change events (server-sent events).

        One long-lived response per subscriber (the browser's EventSource and
        each notebook ``watch()``); ThreadingHTTPServer gives each its own
        handler thread, which parks in ``SESSION.wait_change``. An event fires
        on every rev bump; a comment line every ``timeout`` keeps proxies and
        dead-peer detection honest. Lock-free w.r.t. _RUN_LOCK, so edits
        propagate while a solve is running."""
        global _SSE_COUNT
        with _SSE_LOCK:
            if _SSE_COUNT >= _SSE_LIMIT:
                self._json({"ok": False,
                            "error": "too many event subscribers"}, 503)
                return
            _SSE_COUNT += 1
        try:
            self._sse_stream()
        finally:
            with _SSE_LOCK:
                _SSE_COUNT -= 1

    def _sse_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        rev = -1  # sentinel: always send one snapshot event immediately
        try:
            while True:
                cur, source = SESSION.wait_change(rev, timeout=15.0)
                if cur == rev:
                    self.wfile.write(b": ping\n\n")     # heartbeat
                else:
                    rev = cur
                    data = json.dumps({"rev": rev, "source": source})
                    self.wfile.write(
                        f"event: change\ndata: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return  # subscriber went away — normal teardown

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/api/cancel":
            # Stop the in-flight run. Deliberately lock-free (like /api/progress)
            # so it lands on its own handler thread while the run worker holds
            # _RUN_LOCK; the solver's next per-step callback sees the flag and
            # aborts the transient loop (see webapp/progress.py).
            import progress
            progress.PROGRESS.request_cancel()
            self._json({"ok": True})
            return
        if route not in ("/api/run", "/api/upload", "/api/upload_va",
                         "/api/schematic"):
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20_000_000:
                raise ValueError("payload too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            # an unread body would desync the keep-alive connection: the next
            # request on it would parse body bytes as a request line
            self.close_connection = True
            self._json({"ok": False, "error": f"bad request: {exc}"}, 400)
            return
        if route == "/api/schematic":
            if not _ENABLE_BRIDGE:
                self._json(self._bridge_disabled(), 403)
                return
            doc = payload.get("doc")
            if not isinstance(doc, dict) or \
                    not isinstance(doc.get("schematic"), dict):
                self._json({"ok": False, "error": "doc.schematic required"},
                           400)
                return
            base_rev = payload.get("base_rev")
            if base_rev is not None and (isinstance(base_rev, bool) or
                                         not isinstance(base_rev, int)):
                self._json({"ok": False,
                            "error": "base_rev must be an integer"}, 400)
                return
            res = SESSION.put(doc, str(payload.get("source") or "unknown"),
                              base_rev)
            self._json(res, 200 if res["ok"] else 409)
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


# --- dev auto-reload ---------------------------------------------------------
# A tiny Werkzeug/uvicorn-style reloader: the process the user launches becomes
# a *supervisor* that re-spawns itself (PHOTONFLUX_RELOAD_CHILD=1) and restarts
# the child whenever it exits with code 3. The child runs the real server plus a
# background thread that watches .py source mtimes and calls os._exit(3) on any
# change (edit of a tracked file OR a newly created .py). Static files/examples
# are read per-request, so they are deliberately NOT watched — editing them
# needs no restart, only a page refresh.
_RELOAD_EXIT_CODE = 3
_WATCH_ROOTS = [HERE, HERE.parent / "photonflux"]


def _iter_source_files():
    for root in _WATCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _reload_watcher(interval: float = 1.0) -> None:
    """Poll watched .py files; on the first edit of a tracked file or the
    appearance of a new .py, exit(3) so the supervisor re-spawns a fresh child
    that re-imports the changed code. (Deletions are ignored — a removed module
    only matters once an edit to a surviving file stops importing it, which
    itself triggers a reload.)"""
    mtimes = {}
    for f in _iter_source_files():
        try:
            mtimes[f] = f.stat().st_mtime
        except OSError:
            pass
    while True:
        time.sleep(interval)
        for f in _iter_source_files():
            try:
                m = f.stat().st_mtime
            except OSError:
                continue
            prev = mtimes.get(f)
            if prev is None:  # newly created .py
                reason = "added"
            elif m != prev:   # edited in place
                reason = "changed"
            else:
                continue
            sys.stdout.write(f"\n[reload] {f.name} {reason} — restarting\n")
            sys.stdout.flush()
            os._exit(_RELOAD_EXIT_CODE)


def _run_supervisor() -> int:
    """Re-spawn the server as a child and restart it on reload-triggered exits.
    Any other exit code (clean shutdown, crash, Ctrl-C) ends the supervisor."""
    child_env = dict(os.environ, PHOTONFLUX_RELOAD_CHILD="1")
    while True:
        try:
            code = subprocess.call([sys.executable] + sys.argv, env=child_env)
        except KeyboardInterrupt:
            return 0
        if code != _RELOAD_EXIT_CODE:
            return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # $PORT / $HOST let a container (or PaaS like HF Spaces) place the server;
    # the defaults keep local `server.py` on 127.0.0.1:8642 exactly as before.
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8642")))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    # Auto-restart on .py edits. On by default so local dev "just refreshes";
    # pass --no-reload (or set PHOTONFLUX_RELOAD=0, as the container does) to
    # run a single plain process with no supervisor/watcher.
    ap.add_argument("--reload", action=argparse.BooleanOptionalAction,
                    default=os.environ.get("PHOTONFLUX_RELOAD", "1") == "1",
                    help="auto-restart when a .py source file changes "
                         "(default: on; use --no-reload to disable)")
    args = ap.parse_args()

    is_child = os.environ.get("PHOTONFLUX_RELOAD_CHILD") == "1"
    if args.reload and not is_child:
        return _run_supervisor()

    _load_engine()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.reload:
        threading.Thread(target=_reload_watcher, daemon=True).start()
        print(f"photonflux web UI: http://{args.host}:{args.port}  "
              "(auto-reload on — edit code and refresh the page)")
    else:
        print(f"photonflux web UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
