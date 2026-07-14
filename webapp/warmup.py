#!/usr/bin/env python3
"""Build-time warmup for the container image.

Runs once during `docker build` so the shipped image has every model cache
already populated and the native toolchain proven:

  * build_models()  -> lowers all photonic *.va to models/__jax__/*.py and
                       compiles the SKY130 FET flavors to **Linux** .osdi
                       (via the freshly-built bin/openvaf-ir).
  * a representative example set is solved to JIT-warm the circulax solvers and
    to force each FET flavor's OSDI compile through a real run.

Per-item failures are logged, not fatal: the image still ships (photonics works
even if the FET/OpenVAF path has a problem) unless WARMUP_STRICT=1. Read the
PASS/FAIL summary in the build log to see what compiled. Runs under the venv
python configured in the Dockerfile.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # webapp/  -> catalog, simulate
sys.path.insert(0, str(HERE.parent))          # repo root -> lightspice

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import simulate  # noqa: E402
import catalog  # noqa: E402

# A small set that together exercises: photonic linear + nonlinear models, and
# the nfet/pfet SKY130 OSDI flavors (04/05 curves + 22 CMOS amp). Enough to
# compile every FET flavor and prove the end-to-end path without running the
# multi-minute showcase examples.
WARM_EXAMPLES = [
    "01_photodiode_tia",
    "39_chi3_fwm",
    "04_sky130_nfet_output_curves",
    "05_sky130_pfet_output_curves",
    "22_common_source_amp",
]
STRICT = os.environ.get("WARMUP_STRICT", "") == "1"


def _load(example_id: str) -> dict:
    return json.loads((HERE / "examples" / f"{example_id}.json").read_text())


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    print("[warmup] build_models(): lowering photonic VA + compiling FET OSDI…",
          flush=True)
    t0 = time.perf_counter()
    try:
        catalog.build_models()
        print(f"[warmup] build_models OK in {time.perf_counter() - t0:.1f}s",
              flush=True)
    except Exception:
        traceback.print_exc()
        results.append(("build_models", False, "see traceback above"))

    for ex in WARM_EXAMPLES:
        t0 = time.perf_counter()
        try:
            res = simulate.run(_load(ex))
            ok = bool(res.get("ok", True)) and (
                res.get("traces") or res.get("rows") or res.get("extra_plots"))
            dt = time.perf_counter() - t0
            note = (f"{len(res.get('traces', []))} traces, {dt:.1f}s"
                    if ok else f"no output: {str(res.get('error', ''))[:80]}")
            results.append((ex, bool(ok), note))
            print(f"[warmup] {'PASS' if ok else 'FAIL'} {ex} — {note}",
                  flush=True)
        except Exception as exc:
            traceback.print_exc()
            results.append((ex, False, f"{type(exc).__name__}: {exc}"))
            print(f"[warmup] FAIL {ex} — {exc}", flush=True)

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n[warmup] summary: {n_ok}/{len(results)} ok", flush=True)
    for name, ok, note in results:
        print(f"  {'✓' if ok else '✗'} {name}: {note}", flush=True)

    failed = [n for n, ok, _ in results if not ok]
    if failed and STRICT:
        print(f"[warmup] STRICT: failing build ({', '.join(failed)})", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
