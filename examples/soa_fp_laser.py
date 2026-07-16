#!/usr/bin/env python3
"""Fabry-Perot laser built from parts: a Verilog-A SOA between two partially
reflective mirrors, solved as one circulax system.

Nothing in this circuit "is" a laser — ``models/optical_field/soa.va`` is a bidirectional
Agrawal-Olsson gain reservoir and ``models/optical_field/mirror.va`` is a unitary partial
reflector — but close the loop

    HR mirror (R1 = 0.9)  <->  SOA (both directions, shared gain)  <->
    output coupler (R2 = 0.3)  ->  P_out

and lasing emerges: below threshold the cavity just amplifies the SOA's ASE
seed; at G_th = 1/sqrt(R1*R2) per pass the round trip closes and the gain
CLAMPS, turning excess pump linearly into light.

Analytic pins (derived from the same model equations):

    h_th   = -ln(r1*r2)/2 = 0.6547        (G_th = 1.92, 2.8 dB per pass)
    I_th   = i_tr + h_th/h_op*(i_op-i_tr) = 18.2 mA
    P_out  = (1-R2)*G_th*(h0(I)-h_th)*p_sat/((G_th-1)*(1+R2*G_th))

The testbench is self-checking:

1. **L-I staircase** — one transient with the bias stepped 5 -> 80 mA;
   the settled tail of every step is a point on the L-I curve. (A DC sweep
   cannot do this: above threshold Newton happily converges to the dark
   stationary branch — mathematically valid, dynamically unstable. An L-I
   curve IS a sequence of turn-ons.)
2. **Turn-on transient** — bias stepped through threshold; the round-trip
   gain r1*r2*G(t) visibly clamps to 1 as the field builds.
3. Threshold location, L-I slope, and the clamped gain are asserted against
   the analytic laser.

    .venv-circulax/bin/python examples/soa_fp_laser.py
    .venv-circulax/bin/python examples/soa_fp_laser.py --r2 0.5 --imax 60e-3

        -> out/soa_fp_laser.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit

from _cavity import port_power, run_transient, staircase_source, terminator
from photonflux import cx

# --- the device -------------------------------------------------------------
R1 = 0.9                   # back-mirror power reflectivity
R2 = 0.3                   # output-coupler power reflectivity
SOA = dict(g0_db=20.0, i_op_ma=80.0, i_tr_ma=8.0, p_sat=10e-3,
           tau_c=0.3e-9, tau_bw=1e-12, alpha_h=0.0, p_seed=1e-9,
           Von=1.2, Rs=3.0)

# --- the L-I staircase ------------------------------------------------------
I_MIN, I_MAX = 5e-3, 80e-3   # bias sweep [A]
N_STEP = 16                  # staircase levels
T_STEP = 4e-9                # per level: >> tau_c, settles fully
DT = 4e-12                   # BDF2 fixed step

OUT = Path(__file__).resolve().parents[1] / "out" / "soa_fp_laser.png"


# ===========================================================================
# analytic laser (same equations as the VA models)
# ===========================================================================
def analytic(r1: float, r2: float) -> dict:
    hop = SOA["g0_db"] * np.log(10) / 10
    h_th = -0.5 * np.log(r1 * r2)
    g_th = np.exp(h_th)
    i_th = (SOA["i_tr_ma"] + h_th / hop * (SOA["i_op_ma"] - SOA["i_tr_ma"])) * 1e-3
    return {"hop": hop, "h_th": h_th, "g_th": g_th, "i_th": i_th}


def p_out_analytic(i_a: np.ndarray, law: dict, r2: float) -> np.ndarray:
    h0 = law["hop"] * (i_a * 1e3 - SOA["i_tr_ma"]) / (SOA["i_op_ma"] - SOA["i_tr_ma"])
    p_fi = np.maximum(h0 - law["h_th"], 0.0) * SOA["p_sat"] / (
        (law["g_th"] - 1) * (1 + r2 * law["g_th"]))
    return (1 - r2) * law["g_th"] * p_fi


# ===========================================================================
# the cavity netlist
# ===========================================================================
def build(v_levels: np.ndarray, t_step: float, r1: float, r2: float):
    """FP cavity: M1 <-> SOA <-> M2, bias staircase on the SOA diode."""
    instances = {
        "GND": {"component": "ground"},
        "SOA": {"component": "soa", "settings": dict(SOA)},
        "M1": {"component": "mirror", "settings": {"refl": r1}},
        "M2": {"component": "mirror", "settings": {"refl": r2}},
        "VB": {"component": "stair"},
        "TBK": {"component": "term"},        # M1 back emission, unused
    }
    connections = {
        # forward path: M1 right-going -> SOA forward -> M2 left input
        "M1,ro_re": "SOA,fi_re", "M1,ro_im": "SOA,fi_im",
        "SOA,fo_re": "M2,li_re", "SOA,fo_im": "M2,li_im",
        # backward path: M2 reflection -> SOA backward -> M1 right input
        "M2,lo_re": "SOA,bi_re", "M2,lo_im": "SOA,bi_im",
        "SOA,bo_re": "M1,ri_re", "SOA,bo_im": "M1,ri_im",
        # unused driven outputs terminate at infinite impedance
        "M1,lo_re": "TBK,re", "M1,lo_im": "TBK,im",
        # bias
        "VB,p1": "SOA,an",
    }
    grounded = ["SOA,cat", "SOA,gnd", "M1,gnd", "M2,gnd", "VB,p2",
                "M1,li_re", "M1,li_im",      # dark: nothing enters from the left
                "M2,ri_re", "M2,ri_im"]      # dark: nothing enters from the right
    connections["GND,p1"] = tuple(grounded)
    net = {"instances": instances, "connections": connections,
           "ports": {"pout_re": "M2,ro_re", "pout_im": "M2,ro_im",
                     "pfi_re": "SOA,fi_re", "pfi_im": "SOA,fi_im",
                     "pfo_re": "SOA,fo_re", "pfo_im": "SOA,fo_im",
                     "vb": "SOA,an"}}
    models = {
        "ground": lambda: 0,
        "soa": cx.va("soa"),
        "mirror": cx.va("mirror"),
        "stair": staircase_source(v_levels, t_step),
        "term": terminator(),
    }
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


# ===========================================================================
def main(r1: float = R1, r2: float = R2, i_max: float = I_MAX) -> int:
    law = analytic(r1, r2)
    print(f"cavity: R1 = {r1}, R2 = {r2} -> G_th = {law['g_th']:.4f} "
          f"({10*np.log10(law['g_th']):.2f} dB/pass), "
          f"I_th = {law['i_th']*1e3:.2f} mA")

    # ---- part 1: L-I by staircase transient --------------------------------
    i_levels = np.linspace(I_MIN, i_max, N_STEP)
    v_levels = SOA["Von"] + SOA["Rs"] * i_levels
    c = build(v_levels, T_STEP, r1, r2)
    t_max = N_STEP * T_STEP
    t, sol = run_transient(c, t_max, DT, save_every=20e-12)
    p_out = port_power(c, sol.ys, "pout_re", "pout_im")

    # settled point per step: average the last 20% of each dwell
    p_li = np.array([
        p_out[(t > (k + 0.8) * T_STEP) & (t < (k + 1) * T_STEP)].mean()
        for k in range(N_STEP)])
    p_ref = p_out_analytic(i_levels, law, r2)
    above = i_levels > 1.25 * law["i_th"]
    err = np.abs(p_li[above] - p_ref[above]) / p_ref[above]
    i_on = i_levels[np.argmax(p_li > 0.05e-3)]
    print(f"L-I: first lasing step at {i_on*1e3:.1f} mA "
          f"(analytic threshold {law['i_th']*1e3:.2f} mA), "
          f"max |P - analytic|/P = {err.max():.2%} above 1.25*I_th")
    assert abs(i_on - law["i_th"]) < (i_levels[1] - i_levels[0]) + 1e-6, \
        "threshold in the wrong staircase step"
    assert err.max() < 0.02, "L-I deviates from the clamped-gain analytic line"
    # below threshold: amplified seed only
    assert p_li[i_levels < 0.9 * law["i_th"]].max() < 1e-5

    # ---- part 2: turn-on transient through threshold ------------------------
    i_step = 0.35 * i_max + 0.65 * law["i_th"]      # comfortably above I_th
    v_step = SOA["Von"] + SOA["Rs"] * np.array([5e-3, i_step])
    c2 = build(v_step, 0.5e-9, r1, r2)
    t2, sol2 = run_transient(c2, 6e-9, 1e-12, save_every=5e-12)
    p2 = port_power(c2, sol2.ys, "pout_re", "pout_im")
    # single-pass power gain G = P_fo/P_fi; the round trip is TWO passes and
    # one bounce off each mirror: round-trip power gain = R1*R2*G^2 -> 1
    g_pass = port_power(c2, sol2.ys, "pfo_re", "pfo_im") / np.maximum(
        port_power(c2, sol2.ys, "pfi_re", "pfi_im"), 1e-30)
    g_rt = r1 * r2 * g_pass**2
    p_settle = p2[t2 > 5e-9].mean()
    p_target = float(p_out_analytic(np.array([i_step]), law, r2)[0])
    g_settle = g_rt[t2 > 5e-9].mean()
    print(f"turn-on at {i_step*1e3:.1f} mA: settled P = {p_settle*1e3:.3f} mW "
          f"(analytic {p_target*1e3:.3f}), overshoot x{p2.max()/p_settle:.1f}, "
          f"round-trip gain R1*R2*G^2 = {g_settle:.5f} (clamps to 1)")
    assert abs(p_settle - p_target) / p_target < 0.02
    assert abs(g_settle - 1.0) < 5e-3, "gain did not clamp to the loop loss"
    print("ALL TESTBENCH CHECKS PASSED")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    i_fine = np.linspace(0, i_max, 300)
    ax.plot(i_fine * 1e3, p_out_analytic(i_fine, law, r2) * 1e3, "k:",
            lw=1, label="analytic clamped-gain L-I")
    ax.plot(i_levels * 1e3, p_li * 1e3, "o", c="tab:red", ms=5,
            label="settled staircase steps")
    ax.axvline(law["i_th"] * 1e3, c="gray", lw=0.8, ls="--",
               label=f"I_th = {law['i_th']*1e3:.1f} mA")
    ax.set_xlabel("bias current [mA]"); ax.set_ylabel("P_out [mW]")
    ax.set_title(f"L-I: SOA in a R1={r1}/R2={r2} cavity — lasing emerges")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t * 1e9, p_out * 1e3, c="tab:orange", lw=0.9)
    ax.set_xlabel("time [ns]"); ax.set_ylabel("P_out [mW]")
    ax.set_title("the staircase transient behind the L-I points")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t2 * 1e9, p2 * 1e3, c="tab:orange")
    ax.axhline(p_target * 1e3, c="gray", lw=0.8, ls="--", label="analytic P_out")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("P_out [mW]")
    ax.set_title(f"turn-on: 5 -> {i_step*1e3:.0f} mA step through threshold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t2 * 1e9, g_rt, c="tab:purple")
    ax.axhline(1.0, c="gray", lw=0.8, ls="--", label="clamp: R1*R2*G² = 1")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("round-trip power gain")
    ax.set_ylim(0, max(2.0, np.nanmax(g_rt[t2 > 0.4e-9]) * 1.1))
    ax.set_title("gain clamping as the field builds")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Fabry-Perot laser from a Verilog-A SOA + partial reflectors "
                 "(circulax)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r1", type=float, default=R1, help="back mirror R1")
    ap.add_argument("--r2", type=float, default=R2, help="output coupler R2")
    ap.add_argument("--imax", type=float, default=I_MAX,
                    help="top of the L-I sweep [A]")
    args = ap.parse_args()
    raise SystemExit(main(r1=args.r1, r2=args.r2, i_max=args.imax))
