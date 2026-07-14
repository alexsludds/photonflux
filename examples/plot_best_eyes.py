#!/usr/bin/env python3
"""Render the three eyes — driver input, driver output (electrode), optical
through-port — for the best driver found by ``optimize_driver.py``.

Reads ``out/optimize_driver_result.json`` (topology + best params + baud), re-runs
the ``ring_mod_sky130`` transient at those parameters (with a full PRBS for a
dense eye), folds each probe onto 2 UI, and marks the sampled inner-eye opening
that the optimizer maximized on the optical panel.

    .venv-circulax/bin/python examples/plot_best_eyes.py            # uses the JSON
    .venv-circulax/bin/python examples/plot_best_eyes.py --nbits 127

        -> out/best_eyes.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import ring_mod_sky130 as R
from optimize_driver import fmt_params, make_parts

RESULT = R.OUT.parent / "optimize_driver_result.json"
OUT = R.OUT.parent / "best_eyes.png"


def scan_inner_eye(p_rx, bits, spu, inverting):
    """Repeat the testbench's phase + integer-de-skew scan for the annotation.

    Returns (phase, inner_height_mW, zero_level, one_level) at the best sampling
    point — the same worst-case one-vs-zero margin ``eye_and_metrics`` reports.
    """
    n = len(bits)
    tx = (1 - bits) if inverting else bits
    max_deskew = min(8, n - R.SETTLE_UI - 4)
    best = None
    for d in range(max_deskew + 1):
        lo = max(R.SETTLE_UI, d)
        for ph in range(spu):
            samples = p_rx[ph::spu][:n]
            sam, lab = samples[lo:], tx[lo - d:n - d]
            m = min(len(sam), len(lab))
            sam, lab = sam[:m], lab[:m]
            ones, zeros = sam[lab == 1], sam[lab == 0]
            if len(ones) == 0 or len(zeros) == 0:
                continue
            margin = ones.min() - zeros.max()
            if best is None or margin > best[0]:
                best = (margin, ph, zeros.max(), ones.min())
    return best[1], best[0], best[2], best[3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nbits", type=int, default=R.N_BITS,
                    help=f"PRBS length for the eye (default {R.N_BITS}, a full "
                         "PRBS-7 makes a denser eye than the optimizer's run)")
    ap.add_argument("--result", default=str(RESULT),
                    help="optimizer result JSON to plot")
    args = ap.parse_args()

    res = json.loads(open(args.result).read())
    topology, baud, p = res["topology"], res["baud"], res["best_params"]
    nbits = args.nbits

    print(f"best {topology} driver: {fmt_params(p)}")
    print(f"replaying at {baud/1e9:g} Gbd, PRBS-{R.PRBS_ORDER} ({nbits} bits) "
          "for the eye diagrams ...")

    design = R.design_ring(baud)
    mdl = R.base_models()
    parts = make_parts(topology, p)
    R.build_driver = lambda d, k, c_neut=0.0: parts   # inject the best driver
    # pass the real cap through so run_transient's log matches (the value is
    # already baked into `parts`; this just keeps the printout honest)
    c_neut_f = p.get("c_neut", 0.0) * 1e-15
    t, vin, vdrv, p_rx, bits, t_bit, spu, t_edge, inverting = R.run_transient(
        mdl, design, baud, nbits, driver=topology, c_neut=c_neut_f)
    _, eye_h, er_db, _ = R.eye_and_metrics(
        t, p_rx, bits, t_bit, spu, inverting=inverting)

    ph, inner, z_lvl, o_lvl = scan_inner_eye(p_rx, bits, spu, inverting)
    ui_ps = np.arange(2 * spu) / spu * t_bit * 1e12

    panels = [
        (vin, "V_in [V]", "driver input", "tab:gray"),
        (vdrv, "V(electrode) [V]", "driver output (electrode)", "tab:purple"),
        (p_rx, "P_thru [mW]", "optical output", "tab:orange"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (sig, ylabel, title, color) in zip(axes, panels):
        traces = R.fold_ui2(sig, spu, len(bits))
        for tr in traces:
            ax.plot(ui_ps, tr, color=color, alpha=0.18, lw=0.8)
        ax.set_xlabel("time within 2 UI [ps]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)

    # annotate the maximized inner eye on the optical panel
    axo = axes[2]
    for x in (ph / spu * t_bit * 1e12, (ph + spu) / spu * t_bit * 1e12):
        axo.axvline(x, color="k", ls=":", lw=1, alpha=0.7)
    axo.axhspan(z_lvl, o_lvl, color="tab:green", alpha=0.15)
    axo.axhline(z_lvl, color="tab:green", lw=1)
    axo.axhline(o_lvl, color="tab:green", lw=1)
    x_txt = (ph + spu) / spu * t_bit * 1e12
    axo.annotate("", xy=(x_txt, o_lvl), xytext=(x_txt, z_lvl),
                 arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.5))
    axo.text(x_txt + 1.5, 0.5 * (z_lvl + o_lvl),
             f"inner eye\n{inner:.3f} mW", color="tab:green", fontsize=9,
             va="center")

    fig.suptitle(
        f"Best '{topology}' driver @ {baud/1e9:g} Gbd — {fmt_params(p)}   "
        f"(inner eye {eye_h:+.3f} mW, ER {er_db:.1f} dB, "
        f"electrode edge {t_edge*1e12:.1f} ps)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"inner eye = {eye_h:+.4f} mW, ER = {er_db:.1f} dB")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
