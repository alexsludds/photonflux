#!/usr/bin/env python3
"""Sweep the Miller-neutralization cap and show what it does to the eyes.

Runs the ``single-neut`` driver at a fixed geometry over several ``c_neut``
values and renders, per value: the electrode (driver-output) eye — where the
Cgd kickback lives — and the optical through-port eye, plus summary curves of
electrode rail-overshoot and inner optical eye vs ``c_neut``.

The point: the cap's job is cancelling the output-FET Cgd feedthrough, which
shows up as the electrode overshooting the 0 / V_DD rails. That overshoot should
fall as ``c_neut`` rises toward Cgd_out and grow again (opposite sign) past it —
independent of whether the *optical* inner eye benefits at this baud.

Geometry defaults to the best driver in ``out/optimize_driver_result.json``.

    .venv-circulax/bin/python examples/sweep_neut.py --baud 30e9
    .venv-circulax/bin/python examples/sweep_neut.py --baud 50e9 --cneuts 0 4 8 12 20

        -> out/sweep_neut_<baud>g.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import ring_mod_sky130 as R
from optimize_driver import make_parts
from plot_best_eyes import scan_inner_eye

RESULT = R.OUT.parent / "optimize_driver_result.json"


def run_one(w_p, w_n, c_neut_fF, baud, nbits, design, mdl):
    """One transient at a given cap; returns the probes + derived metrics."""
    parts = make_parts("single-neut",
                       {"w_p": w_p, "w_n": w_n, "c_neut": c_neut_fF})
    R.build_driver = lambda d, k, c_neut=0.0: parts
    out = R.run_transient(mdl, design, baud, nbits, driver="single-neut",
                          c_neut=c_neut_fF * 1e-15)
    t, vin, vdrv, p_rx, bits, t_bit, spu, t_edge, inverting = out
    ph, inner, z_lvl, o_lvl = scan_inner_eye(p_rx, bits, spu, inverting)
    return {
        "vdrv": vdrv, "p_rx": p_rx, "bits": bits, "spu": spu, "t_bit": t_bit,
        "edge_ps": t_edge * 1e12, "inner": inner, "z": z_lvl, "o": o_lvl,
        "over_hi": float(vdrv.max()) - R.V_DD, "over_lo": -float(vdrv.min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baud", type=float, default=30e9)
    ap.add_argument("--nbits", type=int, default=63)
    ap.add_argument("--cneuts", type=float, nargs="+",
                    default=[0.0, 4.0, 8.0, 12.0, 20.0], help="cap values [fF]")
    ap.add_argument("--wp", type=float, default=None, help="PMOS width [um]")
    ap.add_argument("--wn", type=float, default=None, help="NMOS width [um]")
    args = ap.parse_args()

    if args.wp is None or args.wn is None:
        best = json.loads(RESULT.read_text())["best_params"]
        w_p = args.wp if args.wp is not None else best["w_p"]
        w_n = args.wn if args.wn is not None else best["w_n"]
    else:
        w_p, w_n = args.wp, args.wn

    caps = list(args.cneuts)
    design = R.design_ring(args.baud)
    mdl = R.base_models()
    print(f"sweeping c_neut over {caps} fF at {args.baud/1e9:g} Gbd, "
          f"inverter {w_p:g}/{w_n:g} um (PRBS-{R.PRBS_ORDER}, {args.nbits} bits)")

    runs = []
    for c in caps:
        r = run_one(w_p, w_n, c, args.baud, args.nbits, design, mdl)
        runs.append(r)
        print(f"  c_neut={c:5.1f} fF: overshoot +{r['over_hi']*1e3:4.0f}/"
              f"-{r['over_lo']*1e3:4.0f} mV  edge={r['edge_ps']:4.1f} ps  "
              f"inner eye={r['inner']:+.4f} mW")

    # ---- figure: electrode eyes (row 0), optical eyes (row 1), summary (row 2)
    n = len(caps)
    ui_ps0 = np.arange(2 * runs[0]["spu"]) / runs[0]["spu"] * runs[0]["t_bit"] * 1e12
    v_lo = min(r["vdrv"].min() for r in runs)
    v_hi = max(r["vdrv"].max() for r in runs)
    p_lo = min(r["p_rx"].min() for r in runs)
    p_hi = max(r["p_rx"].max() for r in runs)

    fig = plt.figure(figsize=(3.0 * n, 9.5))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.85], hspace=0.42,
                          wspace=0.3)

    for j, (c, r) in enumerate(zip(caps, runs)):
        ui_ps = np.arange(2 * r["spu"]) / r["spu"] * r["t_bit"] * 1e12
        # electrode eye
        axe = fig.add_subplot(gs[0, j])
        for tr in R.fold_ui2(r["vdrv"], r["spu"], len(r["bits"])):
            axe.plot(ui_ps, tr, color="tab:purple", alpha=0.16, lw=0.8)
        axe.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.5)
        axe.axhline(R.V_DD, color="k", lw=0.6, ls="--", alpha=0.5)
        axe.set_ylim(v_lo - 0.1, v_hi + 0.1)
        axe.set_title(f"c_neut = {c:g} fF\novershoot "
                      f"+{r['over_hi']*1e3:.0f}/-{r['over_lo']*1e3:.0f} mV",
                      fontsize=10)
        if j == 0:
            axe.set_ylabel("electrode\nV(electrode) [V]")
        axe.grid(alpha=0.3)
        # optical eye
        axo = fig.add_subplot(gs[1, j])
        for tr in R.fold_ui2(r["p_rx"], r["spu"], len(r["bits"])):
            axo.plot(ui_ps, tr, color="tab:orange", alpha=0.16, lw=0.8)
        axo.axhspan(r["z"], r["o"], color="tab:green", alpha=0.15)
        axo.set_ylim(p_lo - 0.02, p_hi + 0.02)
        axo.set_title(f"inner eye {r['inner']:+.3f} mW", fontsize=10)
        axo.set_xlabel("time within 2 UI [ps]")
        if j == 0:
            axo.set_ylabel("optical\nP_thru [mW]")
        axo.grid(alpha=0.3)

    # summary curves
    over_hi = [r["over_hi"] * 1e3 for r in runs]
    over_lo = [r["over_lo"] * 1e3 for r in runs]
    inner = [r["inner"] for r in runs]
    ax1 = fig.add_subplot(gs[2, : max(1, n // 2)])
    ax1.plot(caps, over_hi, "o-", color="tab:red", label="over V_DD")
    ax1.plot(caps, over_lo, "s--", color="tab:blue", label="under GND")
    ax1.set_xlabel("c_neut [fF]"); ax1.set_ylabel("rail overshoot [mV]")
    ax1.set_title("Cgd kickback vs c_neut  (the cap's job)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[2, max(1, n // 2):])
    ax2.plot(caps, inner, "D-", color="tab:green")
    jbest = int(np.argmax(inner))
    ax2.plot(caps[jbest], inner[jbest], "*", ms=16, color="k",
             label=f"max @ {caps[jbest]:g} fF")
    ax2.set_xlabel("c_neut [fF]"); ax2.set_ylabel("inner optical eye [mW]")
    ax2.set_title("inner optical eye vs c_neut  (the objective)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Neutralization-cap sweep — single-neut {w_p:g}/{w_n:g} um @ "
        f"{args.baud/1e9:g} Gbd:  electrode kickback shrinks as c_neut→Cgd, "
        f"then reverses", fontsize=12.5)
    out = R.OUT.parent / f"sweep_neut_{args.baud/1e9:g}g.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
