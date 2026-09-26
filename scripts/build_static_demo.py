"""Build the static GitHub Pages demo of the web editor.

The site is webapp/static plus webapp/static/demo.js (a fetch shim) and a
data/ tree of pre-computed API responses. Everything is captured from a
running photonflux server (normally the published container image, see
.github/workflows/demo-site.yml):

* GET responses: component catalog, examples index + docs, Verilog-A sources.
* One /api/run per example, recorded by driving the real editor headlessly
  (Playwright/Chromium): load the example, press Run, save the request body's
  SHA-256 and the response. demo.js replays a run whose body hashes the same,
  so the captured payload is exactly what a visitor's browser sends.

Usage:
    python scripts/build_static_demo.py --server http://localhost:7860 --out site
    python scripts/build_static_demo.py --only 01_photodiode_tia,39_chi3_fwm ...

Needs `pip install playwright && playwright install chromium`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "webapp" / "static"
BOOT_EXAMPLE = "01_photodiode_tia"   # app.js loads it on a first visit


def get(server: str, path: str) -> bytes | None:
    try:
        with urllib.request.urlopen(server + path, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError:
        return None


def copy_frontend(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out)
    html = (out / "index.html").read_text()
    # project Pages live under /<repo>/, so root-absolute asset paths must go
    html = re.sub(r'(src|href)="/(?!/)', r'\1="', html)
    html = html.replace('<script src="app.js"></script>',
                        '<script src="demo.js"></script>\n<script src="app.js"></script>')
    assert 'src="demo.js"' in html, "index.html no longer loads app.js as expected"
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")


def dump_api(server: str, data: Path, only: set[str] | None) -> list[str]:
    components = get(server, "/api/components")
    examples = get(server, "/api/examples")
    if components is None or examples is None:
        sys.exit(f"server at {server} did not answer /api/components + /api/examples")
    (data / "examples").mkdir(parents=True)
    (data / "veriloga").mkdir()
    (data / "components.json").write_bytes(components)
    (data / "examples.json").write_bytes(examples)
    ids = [e["id"] for e in json.loads(examples)]
    for ex in ids:
        (data / "examples" / f"{ex}.json").write_bytes(get(server, f"/api/examples/{ex}") or b"{}")
    for key in json.loads(components):
        q = urllib.parse.quote(key, safe="")
        src = get(server, f"/api/veriloga?type={q}")
        if src is not None:
            (data / "veriloga" / f"{q}.json").write_bytes(src)
    return [e for e in ids if only is None or e in only]


def capture_runs(server: str, data: Path, ids: list[str], timeout_s: float) -> dict:
    from playwright.sync_api import sync_playwright

    runs = data / "runs"
    runs.mkdir()
    index: dict = {"hashes": {}, "examples": []}
    failed: list[str] = []

    def is_run(r) -> bool:
        return r.url.endswith("/api/run") and r.request.method == "POST"

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        # (label, example to select or None = the first-visit auto-load)
        jobs = [(BOOT_EXAMPLE + " (first visit)", None)] if BOOT_EXAMPLE in ids else []
        jobs += [(ex, ex) for ex in ids]
        for label, ex in jobs:
            t0 = time.time()
            ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
            page = ctx.new_page()
            try:
                page.goto(server + "/", wait_until="networkidle")
                page.wait_for_function(
                    "document.querySelectorAll('#sel-example option').length > 1")
                if ex is not None:
                    page.select_option("#sel-example", ex)
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(500)
                with page.expect_response(is_run, timeout=timeout_s * 1000) as info:
                    page.evaluate("document.getElementById('btn-run').click()")
                resp = info.value
                body = resp.request.post_data or ""
                text = resp.text()
                if not json.loads(text).get("ok"):
                    raise RuntimeError(json.loads(text).get("error", "run failed"))
                name = ex or BOOT_EXAMPLE
                if ex is not None:
                    (runs / f"{name}.json").write_text(text)
                    index["examples"].append(name)
                index["hashes"][hashlib.sha256(body.encode()).hexdigest()] = name
                print(f"  ok   {label}  {time.time() - t0:5.1f}s  {len(text) / 1e6:.1f} MB",
                      flush=True)
            except Exception as exc:  # keep going; one slow example shouldn't sink the site
                failed.append(label)
                print(f"  FAIL {label}  {time.time() - t0:5.1f}s  {str(exc)[:200]}", flush=True)
            finally:
                ctx.close()
        browser.close()
    (runs / "index.json").write_text(json.dumps(index))
    index["failed"] = failed
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default="http://localhost:7860")
    ap.add_argument("--out", type=Path, default=REPO / "site")
    ap.add_argument("--only", help="comma-separated example ids (default: all)")
    ap.add_argument("--timeout", type=float, default=660, help="per-run timeout, s")
    args = ap.parse_args()
    server = args.server.rstrip("/")
    only = set(args.only.split(",")) if args.only else None

    copy_frontend(args.out)
    data = args.out / "data"
    ids = dump_api(server, data, only)
    print(f"capturing {len(ids)} example runs from {server}", flush=True)
    index = capture_runs(server, data, ids, args.timeout)
    ok = len(index["examples"])
    print(f"done: {ok}/{len(ids)} examples captured"
          + (f"; failed: {', '.join(index['failed'])}" if index["failed"] else ""))
    if ok == 0:
        sys.exit("no example runs captured")


if __name__ == "__main__":
    main()
