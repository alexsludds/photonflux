#!/usr/bin/env python3
"""Optimize a ring-modulator driver's parameters to maximize the inner eye.

A thin wrapper around ``ring_mod_sky130.py``: it drives that testbench's
transient + eye pipeline as a black-box objective and searches the driver
parameters you hand it (transistor widths, Miller-neutralization cap, two-stage
taper) to **maximize the sampled inner eye opening** — the same worst-case
one-vs-zero margin (``eye_h``, in mW) the testbench reports.

You pick the *topology* and the *starting point*; the optimizer tunes the knobs
that topology exposes:

    single        ->  w_p, w_n
    single-neut   ->  w_p, w_n, c_neut          (Miller-neutralization cap)
    two-stage     ->  w_p, w_n, taper           (stage-2 width = taper * stage-1)

Widths are in um, ``c_neut`` in fF, ``taper`` unitless. Everything else (ring
physics, laser bias, baud, PRBS) is inherited from ``ring_mod_sky130.py``.

Examples:

    # tune the neutralized single stage at 30 Gbd, cap in [0, 20] fF
    .venv-circulax/bin/python examples/optimize_driver.py --driver single-neut \\
        --baud 30e9 --wp 30 --wn 15 --cneut 5 --c-bounds 0 20

    # size a plain single-stage inverter for 50 Gbd
    .venv-circulax/bin/python examples/optimize_driver.py --driver single \\
        --baud 50e9 --wp 22 --wn 12 --w-bounds 4 80 --maxfev 60

    # two-stage buffer: widths + taper
    .venv-circulax/bin/python examples/optimize_driver.py --driver two-stage \\
        --baud 40e9 --wp 40 --wn 20 --taper 1.0 --taper-bounds 0.5 3

Cost: a *new* geometry is a full circuit compile + XLA + BDF2 transient
(~0.5-3 min); a geometry seen before (cache) or a repeated grid point (memo) is
near-instant. The search therefore snaps every candidate to a grid (``--w-step``
etc.) so it reuses a small, cacheable set of geometries, and it counts only
distinct evaluations against ``--maxfev``. Use a shorter ``--nbits`` and coarser
grid while exploring; refine with a longer ``--nbits``. Add ``--plot`` to render
the eye diagram at the optimum via the underlying testbench.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time

import numpy as np

# the testbench we wrap, and the importable driver builders it uses
import ring_mod_sky130 as R
import drivers as D

# a single FET wider than this trips the SKY130 model-card bin / netlist load,
# so keep the search inside a range that always compiles
MAX_W_UM = 100.0
# objective value returned when an evaluation cannot be scored (solver/netlist
# failure, or an out-of-range width) — worse than any real closed eye
FAIL_PENALTY_MW = -1e3

# which knobs each topology exposes, in optimization order
TOPOLOGY_KNOBS = {
    "single": ("w_p", "w_n"),
    "single-neut": ("w_p", "w_n", "c_neut"),
    "two-stage": ("w_p", "w_n", "taper"),
}


def make_parts(topology: str, p: dict):
    """Build the DriverParts fragment for a parameter dict (um / fF / x)."""
    l = R.L_CH_UM  # noqa: E741
    if topology == "single":
        return D.single_stage_inverter(w_p=p["w_p"], w_n=p["w_n"], l=l)
    if topology == "single-neut":
        return D.single_stage_neutralized_inverter(
            w_p=p["w_p"], w_n=p["w_n"], l=l, c_neut=p["c_neut"] * 1e-15)
    if topology == "two-stage":
        return D.two_stage_inverter(
            w_p=p["w_p"], w_n=p["w_n"], l=l, taper=p["taper"])
    raise ValueError(f"unknown topology {topology!r}")


def fet_widths(topology: str, p: dict) -> list[float]:
    """Every physical FET width the parameter set implies (for range guarding)."""
    ws = [p["w_p"], p["w_n"]]
    if topology == "two-stage":
        ws += [p["w_p"] * p["taper"], p["w_n"] * p["taper"]]
    return ws


def evaluate(topology: str, p: dict, design: dict, mdl: dict,
             baud: float, nbits: int) -> tuple[float, float, float]:
    """Run one transient + eye scoring. Returns (eye_h_mW, er_db, edge_ps).

    Raises nothing scoreable — callers treat exceptions as a failed eval.
    """
    parts = make_parts(topology, p)

    # inject this candidate's driver into the testbench (build_driver is what
    # run_transient calls; the c_neut kwarg is carried inside `parts` already)
    def _bd(design_, kind, c_neut=0.0):
        return parts

    R.build_driver = _bd
    # the testbench is chatty; keep its per-eval prints out of the search log
    with contextlib.redirect_stdout(io.StringIO()):
        out = R.run_transient(mdl, design, baud, nbits, driver=topology)
        t, vin, vdrv, p_rx, bits, t_bit, spu, t_edge, inverting = out
        _, eye_h, er_db, _ = R.eye_and_metrics(
            t, p_rx, bits, t_bit, spu, inverting=inverting)
    return float(eye_h), float(er_db), float(t_edge * 1e12)


def snap(p: dict, steps: dict, bounds: dict) -> dict:
    """Round each knob to its grid step and clip to bounds.

    Snapping keeps the search on a finite set of geometries: revisited grid
    points hit the transient/OSDI cache (~3 s) instead of triggering a fresh
    ~1-3 min XLA recompile, and sub-grid width wiggles (which barely move the
    eye) are never wasted on.
    """
    out = {}
    for k, v in p.items():
        s = steps[k]
        v = round(v / s) * s
        lo, hi = bounds[k]
        out[k] = float(min(max(v, lo), hi))
    return out


def build_objective(topology, knobs, steps, bounds, design, mdl, baud, nbits,
                    state):
    """Closure: parameter dict -> value to MINIMIZE (negative eye height).

    Snaps to the grid, memoizes on the snapped key, and tracks the best eye.
    """
    def objective(praw: dict) -> float:
        p = snap(praw, steps, bounds)
        key = tuple(round(p[k], 4) for k in knobs)
        if key in state["memo"]:
            return state["memo"][key]

        state["neval"] += 1
        n = state["neval"]
        if any(w > MAX_W_UM or w <= 0 for w in fet_widths(topology, p)):
            print(f"  [{n:3d}] {fmt_params(p):<40} width out of range -> penalty")
            val = -FAIL_PENALTY_MW
            state["memo"][key] = val
            return val

        t0 = time.perf_counter()
        try:
            eye_h, er_db, edge_ps = evaluate(
                topology, p, design, mdl, baud, nbits)
        except Exception as exc:  # solver/netlist failure -> penalize
            print(f"  [{n:3d}] {fmt_params(p):<40} FAILED ({type(exc).__name__})")
            val = -FAIL_PENALTY_MW
            state["memo"][key] = val
            return val
        dt = time.perf_counter() - t0

        best = eye_h > state["best_eye"]
        if best:
            state["best_eye"] = eye_h
            state["best_p"] = dict(p)
            state["best_extra"] = {"er_db": er_db, "edge_ps": edge_ps}
        state["history"].append({"n": n, **p, "eye_h": eye_h, "er_db": er_db})
        print(f"  [{n:3d}] {fmt_params(p):<40} eye={eye_h:+.4f} mW  "
              f"ER={er_db:4.1f} dB  edge={edge_ps:4.1f} ps  ({dt:.0f} s)"
              f"{'   <- best' if best else ''}")
        val = -eye_h
        state["memo"][key] = val
        return val
    return objective


def pattern_search(objective, x0: dict, knobs, steps: dict, bounds: dict,
                   state: dict, max_evals: int, coarse: int = 4) -> None:
    """Hooke-Jeeves coordinate pattern search, coarse-to-fine on the grid.

    Derivative-free and cache-friendly: start with a step of ``coarse`` grid
    units, probe each knob +/-, move to the first improvement, and halve the
    step once no coordinate move helps — down to one grid unit (the finest
    resolution). Big early jumps escape the start point; the final passes refine
    at grid resolution. ``objective`` handles snapping, memoization, and
    best-tracking; ``state['neval']`` counts real (uncached) evaluations, the
    budget we spend against.
    """
    def budget_left() -> bool:
        return state["neval"] < max_evals

    x = snap(x0, steps, bounds)
    objective(x)
    step = {k: coarse * steps[k] for k in knobs}

    while budget_left() and any(step[k] >= steps[k] for k in knobs):
        base = objective(x)          # cached: costs no eval
        improved = False
        for k in knobs:
            if not budget_left():
                break
            for sgn in (+1, -1):
                cand = snap({**x, k: x[k] + sgn * step[k]}, steps, bounds)
                if cand[k] == x[k]:             # clipped to a wall; no move
                    continue
                val = objective(cand)
                if val < base:                 # minimizing -> lower is better
                    x, base, improved = cand, val, True
                    break
        if not improved:
            step = {k: step[k] / 2 for k in knobs}   # refine toward the grid


def fmt_params(p: dict) -> str:
    bits = []
    for k, v in p.items():
        if k == "c_neut":
            bits.append(f"c_neut={v:.2f}fF")
        elif k == "taper":
            bits.append(f"taper={v:.2f}")
        else:
            bits.append(f"{k}={v:.1f}um")
    return " ".join(bits)


def main() -> int:
    # stream progress line-by-line even when piped to a file (each eval is ~30 s;
    # block-buffered stdout would hide the search until it finished)
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--driver", choices=tuple(TOPOLOGY_KNOBS),
                    default="single-neut", help="driver topology to optimize")
    ap.add_argument("--baud", type=float, default=30e9, help="symbol rate [baud]")
    ap.add_argument("--nbits", type=int, default=63,
                    help="PRBS length per eval (shorter = faster, noisier)")
    ap.add_argument("--kappa2", type=float, default=None,
                    help="override ring bus coupling")
    # starting point (units: um, fF, x)
    ap.add_argument("--wp", type=float, default=30.0, help="initial PMOS width [um]")
    ap.add_argument("--wn", type=float, default=15.0, help="initial NMOS width [um]")
    ap.add_argument("--cneut", type=float, default=5.0,
                    help="initial neutralization cap [fF] (single-neut)")
    ap.add_argument("--taper", type=float, default=1.0,
                    help="initial stage-2/stage-1 ratio (two-stage)")
    # bounds
    ap.add_argument("--w-bounds", type=float, nargs=2, default=(4.0, 80.0),
                    metavar=("LO", "HI"), help="width bounds [um]")
    ap.add_argument("--c-bounds", type=float, nargs=2, default=(0.0, 25.0),
                    metavar=("LO", "HI"), help="neutralization-cap bounds [fF]")
    ap.add_argument("--taper-bounds", type=float, nargs=2, default=(0.5, 3.0),
                    metavar=("LO", "HI"), help="taper bounds")
    # search grid: candidates snap to these steps so the search reuses a finite,
    # cacheable set of geometries (a new geometry costs a ~1-3 min recompile; a
    # revisited one is instant). Sub-grid width wiggles barely move the eye.
    ap.add_argument("--w-step", type=float, default=2.0, help="width grid [um]")
    ap.add_argument("--c-step", type=float, default=1.0,
                    help="neutralization-cap grid [fF]")
    ap.add_argument("--taper-step", type=float, default=0.25, help="taper grid")
    ap.add_argument("--maxfev", type=int, default=40,
                    help="max distinct (uncached) evaluations (~0.5-3 min each)")
    ap.add_argument("--plot", action="store_true",
                    help="render the eye diagram at the optimum via the testbench")
    args = ap.parse_args()

    topology = args.driver
    knobs = TOPOLOGY_KNOBS[topology]
    init_all = {"w_p": args.wp, "w_n": args.wn,
                "c_neut": args.cneut, "taper": args.taper}
    bounds = {"w_p": tuple(args.w_bounds), "w_n": tuple(args.w_bounds),
              "c_neut": tuple(args.c_bounds), "taper": tuple(args.taper_bounds)}
    steps = {"w_p": args.w_step, "w_n": args.w_step,
             "c_neut": args.c_step, "taper": args.taper_step}
    x0 = {k: init_all[k] for k in knobs}

    # the ring design + device models are independent of the driver knobs, so
    # build them once and reuse across every evaluation
    design = R.design_ring(args.baud, kappa2=args.kappa2)
    mdl = R.base_models()
    orig_build_driver = R.build_driver

    print(f"optimizing '{topology}' driver at {args.baud/1e9:g} Gbd "
          f"(PRBS-{R.PRBS_ORDER}, {args.nbits} bits) to maximize inner eye")
    print(f"knobs: {', '.join(knobs)}")
    print("start: " + fmt_params(snap(x0, steps, bounds)))
    print("bounds: " + ", ".join(f"{k}∈[{bounds[k][0]:g},{bounds[k][1]:g}]"
                                 for k in knobs))
    print("grid: " + ", ".join(f"{k}±{steps[k]:g}" for k in knobs))
    print(f"budget: up to {args.maxfev} distinct evals\n")

    state = {"neval": 0, "best_eye": -np.inf, "best_p": None,
             "best_extra": {}, "history": [], "memo": {}}
    objective = build_objective(
        topology, knobs, steps, bounds, design, mdl, args.baud, args.nbits, state)

    t_start = time.perf_counter()
    try:
        pattern_search(objective, x0, knobs, steps, bounds, state, args.maxfev)
    except KeyboardInterrupt:
        print("\ninterrupted — reporting best so far")
    finally:
        R.build_driver = orig_build_driver  # restore the testbench's own builder
    elapsed = time.perf_counter() - t_start

    if state["best_p"] is None:
        print("\nno successful evaluation — every candidate failed to score")
        return 1

    print(f"\n=== optimum after {state['neval']} evals ({elapsed:.0f} s) ===")
    print("  " + fmt_params(state["best_p"]))
    ex = state["best_extra"]
    print(f"  inner eye height = {state['best_eye']:+.4f} mW   "
          f"ER = {ex['er_db']:.1f} dB   electrode edge = {ex['edge_ps']:.1f} ps")

    # persist result + search history for later inspection
    out_json = R.OUT.parent / "optimize_driver_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "topology": topology, "baud": args.baud, "nbits": args.nbits,
        "best_params": state["best_p"], "best_eye_mW": state["best_eye"],
        **ex, "history": state["history"],
    }, indent=2))
    print(f"  wrote {out_json}")

    if args.plot:
        _render_optimum(topology, state["best_p"], args)
    return 0


def _render_optimum(topology: str, p: dict, args) -> None:
    """Re-run the testbench at the optimum (via its globals) to draw the eye."""
    R.W_P_UM, R.W_N_UM = p["w_p"], p["w_n"]          # single / single-neut path
    R.W_P2_UM, R.W_N2_UM = p["w_p"], p["w_n"]        # two-stage path
    R.TWO_STAGE_TAPER = p.get("taper", R.TWO_STAGE_TAPER)
    c_neut = p.get("c_neut", 0.0) * 1e-15
    print("\nrendering eye at optimum via ring_mod_sky130.main() ...")
    R.main(baud=args.baud, n_bits=args.nbits, kappa2=args.kappa2,
           driver=topology, c_neut=c_neut)


if __name__ == "__main__":
    raise SystemExit(main())
