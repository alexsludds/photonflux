#!/usr/bin/env python3
"""Multi-section traveling-wave Fabry-Perot laser: the segmented gain framework
that DFB/DBR are built from, validated against the lumped ``soa`` cavity.

The gain region is N cascaded ``models/optical_field/tw_gain_seg.va`` slices —
each a bidirectional directed-wave element carrying its OWN forward/backward
field state (a transit-time ddt) and its OWN carrier reservoir — sandwiched
between two ``mirror.va`` partial reflectors:

    HR mirror (R1 = 0.9)  <->  [ gain slice x N ]  <->  OC (R2 = 0.3)  ->  P_out

This is the method of lines applied to the traveling-wave rate equations. It
reproduces the SAME emergent lasing as the single-``soa`` cavity
(``soa_fp_laser.py``) — below threshold the cavity amplifies the distributed
ASE seed, at G_th = 1/sqrt(R1*R2) per pass the round trip closes and the gain
clamps — but now the reservoir is SPATIALLY RESOLVED, so two extra things fall
out that a lumped model cannot show:

1. **Threshold + slope efficiency** match the analytic clamped-gain L-I. The
   per-slice unsaturated gains sum to the same round-trip gain, so the
   threshold current and the L-I slope land on the lumped-``soa`` analytic
   line (both asserted, < 8 %).
2. **Longitudinal spatial hole burning** — the circulating power is larger
   toward the output coupler (lower R), so those slices saturate harder and
   their local gain is lower: the reservoir is NON-UNIFORM along the cavity.
   The example reads every slice's circulating power and asserts the profile
   tilts toward the output facet.

Self-checking, like ``soa_fp_laser.py``: an L-I staircase (one transient with
the bias stepped, the settled tail of each step a point on the L-I curve — a DC
sweep would sit on the unstable dark branch above threshold).

    python examples/tw_fp_laser.py            ->  out/tw_fp_laser.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit

from _cavity import port_power, run_transient, staircase_source, terminator
from photonflux import cx

C0 = 2.99792458e8

# --- the cavity -------------------------------------------------------------
R1, R2 = 0.9, 0.3            # back mirror / output coupler power reflectivity
LAM_NM = 1310.0
L_TOT = 300e-6              # gain-region length [m]
N_SEG = 6                  # number of gain slices
DZ = L_TOT / N_SEG
NG = 3.7
G0_DB = 20.0               # round-trip small-signal gain at i_op [dB]
# 2*g_unsat*L = hop = G0_DB*ln10/10 so the summed slice gain equals the lumped h
G_UNSAT = G0_DB * np.log(10) / 10 / (2 * L_TOT)
I_OP_MA, I_TR_MA = 80.0, 8.0
P_SAT = 10e-3
TAU_C = 0.3e-9

SEG = dict(lambda_nm=LAM_NM, lambda_bragg_nm=LAM_NM, n_g=NG, dz=DZ,
           g_unsat_pm=G_UNSAT, i_op_ma=I_OP_MA, i_tr_ma=I_TR_MA, p_sat=P_SAT,
           tau_c=TAU_C, alpha_h=0.0, kappa_pm=0.0, loss_pm=0.0,
           p_seed=1e-9, Von=1.2, Rs=3.0)

# --- the L-I staircase ------------------------------------------------------
I_MIN, I_MAX = 5e-3, 80e-3
N_STEP = 12
T_STEP = 2e-9              # >> tau_c: settles fully
DT = 0.3e-12              # BDF2 fixed step (< slice transit time)

OUT = Path(__file__).resolve().parents[1] / "out" / "tw_fp_laser.png"


# ===========================================================================
# analytic laser (same clamped-gain equations as soa_fp_laser.py)
# ===========================================================================
def analytic() -> dict:
    hop = G0_DB * np.log(10) / 10
    h_th = -0.5 * np.log(R1 * R2)
    g_th = np.exp(h_th)
    i_th = (I_TR_MA + h_th / hop * (I_OP_MA - I_TR_MA)) * 1e-3
    return {"hop": hop, "h_th": h_th, "g_th": g_th, "i_th": i_th}


def p_out_analytic(i_a: np.ndarray, law: dict) -> np.ndarray:
    h0 = law["hop"] * (i_a * 1e3 - I_TR_MA) / (I_OP_MA - I_TR_MA)
    p_fi = np.maximum(h0 - law["h_th"], 0.0) * P_SAT / (
        (law["g_th"] - 1) * (1 + R2 * law["g_th"]))
    return (1 - R2) * law["g_th"] * p_fi


# ===========================================================================
# the segmented cavity netlist
# ===========================================================================
def build(v_levels: np.ndarray, t_step: float):
    """M1 <-> [tw_gain_seg x N_SEG] <-> M2, bias staircase on all slice diodes."""
    instances = {
        "GND": {"component": "ground"},
        "M1": {"component": "mirror", "settings": {"refl": R1}},
        "M2": {"component": "mirror", "settings": {"refl": R2}},
        "VB": {"component": "stair"},
        "TBK": {"component": "term"},         # M1 back emission, unused
    }
    for k in range(N_SEG):
        instances[f"S{k}"] = {"component": "seg", "settings": SEG}

    connections = {
        # forward: M1 right-going -> slice 0 -> ... -> M2 left input
        "M1,ro_re": "S0,fl_re", "M1,ro_im": "S0,fl_im",
        # backward: slice 0 left-going -> M1 right input
        "S0,bl_re": "M1,ri_re", "S0,bl_im": "M1,ri_im",
        "M1,lo_re": "TBK,re", "M1,lo_im": "TBK,im",
    }
    for k in range(N_SEG - 1):
        connections[f"S{k},fr_re"] = f"S{k+1},fl_re"
        connections[f"S{k},fr_im"] = f"S{k+1},fl_im"
        connections[f"S{k+1},bl_re"] = f"S{k},br_re"
        connections[f"S{k+1},bl_im"] = f"S{k},br_im"
    connections[f"S{N_SEG-1},fr_re"] = "M2,li_re"
    connections[f"S{N_SEG-1},fr_im"] = "M2,li_im"
    connections["M2,lo_re"] = f"S{N_SEG-1},br_re"
    connections["M2,lo_im"] = f"S{N_SEG-1},br_im"
    connections["VB,p1"] = tuple(f"S{k},an" for k in range(N_SEG))  # common bias

    grounded = ["M1,gnd", "M2,gnd", "VB,p2", "M1,li_re", "M1,li_im",
                "M2,ri_re", "M2,ri_im"]
    for k in range(N_SEG):
        grounded += [f"S{k},gnd", f"S{k},cat"]
    connections["GND,p1"] = tuple(grounded)

    ports = {"pout_re": "M2,ro_re", "pout_im": "M2,ro_im"}
    for k in range(N_SEG):          # per-slice fields for the SHB profile
        for pt in ("fr_re", "fr_im", "bl_re", "bl_im"):
            ports[f"S{k},{pt}"] = f"S{k},{pt}"
    net = {"instances": instances, "connections": connections, "ports": ports}
    models = {"ground": lambda: 0, "mirror": cx.va("mirror"),
              "seg": cx.va("tw_gain_seg"),
              "stair": staircase_source(v_levels, t_step),
              "term": terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def slice_power(c, ys, t, t_from: float) -> np.ndarray:
    """Settled circulating power |R|^2 + |S|^2 in every slice."""
    out = []
    for k in range(N_SEG):
        p = (np.real(c.port(ys, f"S{k},fr_re"))**2
             + np.real(c.port(ys, f"S{k},fr_im"))**2
             + np.real(c.port(ys, f"S{k},bl_re"))**2
             + np.real(c.port(ys, f"S{k},bl_im"))**2)
        out.append(p[t > t_from].mean())
    return np.asarray(out)


# ===========================================================================
def main() -> int:
    law = analytic()
    print(f"cavity: R1={R1} R2={R2} -> G_th={law['g_th']:.3f} "
          f"({10*np.log10(law['g_th']):.2f} dB/pass), I_th={law['i_th']*1e3:.2f} mA; "
          f"{N_SEG} slices x {DZ*1e6:.0f} um, tau_s={NG*DZ/C0*1e12:.3f} ps/slice")

    i_levels = np.linspace(I_MIN, I_MAX, N_STEP)
    v_levels = 1.2 + 3.0 * i_levels
    t0 = time.time()
    c = build(v_levels, T_STEP)
    t, sol = run_transient(c, N_STEP * T_STEP, DT, save_every=20e-12)
    p_out = port_power(c, sol.ys, "pout_re", "pout_im")
    print(f"transient: {time.time()-t0:.1f}s")

    p_li = np.array([
        p_out[(t > (k + 0.8) * T_STEP) & (t < (k + 1) * T_STEP)].mean()
        for k in range(N_STEP)])
    p_ref = p_out_analytic(i_levels, law)
    above = i_levels > 1.25 * law["i_th"]
    err = np.abs(p_li[above] - p_ref[above]) / p_ref[above]
    i_on = i_levels[np.argmax(p_li > 0.05e-3)]

    # slope efficiency (linear fit above threshold) vs analytic
    slope_sim = np.polyfit(i_levels[above], p_li[above], 1)[0]
    slope_ana = np.polyfit(i_levels[above], p_ref[above], 1)[0]

    # spatial hole burning: circulating power at the top bias step
    p_slice = slice_power(c, sol.ys, t, (N_STEP - 0.2) * T_STEP)
    shb = (p_slice[-1] - p_slice[0]) / p_slice.mean()

    print(f"L-I: first lasing step {i_on*1e3:.1f} mA (analytic {law['i_th']*1e3:.2f}), "
          f"max |P-analytic|/P = {err.max():.1%} above 1.25*I_th")
    print(f"slope efficiency: sim {slope_sim*1e-3*1e3:.4f} mW/mA, "
          f"analytic {slope_ana*1e-3*1e3:.4f} mW/mA "
          f"({abs(slope_sim-slope_ana)/slope_ana:.1%})")
    print(f"spatial hole burning: circulating power {p_slice[0]*1e3:.1f} -> "
          f"{p_slice[-1]*1e3:.1f} mW back->front ({shb:+.1%} tilt toward output)")

    assert abs(i_on - law["i_th"]) < (i_levels[1] - i_levels[0]) + 1e-6, \
        "threshold in the wrong staircase step"
    assert err.max() < 0.08, "L-I deviates from the clamped-gain analytic line"
    assert abs(slope_sim - slope_ana) / slope_ana < 0.08, "slope efficiency off"
    assert p_li[i_levels < 0.9 * law["i_th"]].max() < 1e-5, "lasing below threshold"
    assert shb > 0.02 and np.all(np.diff(p_slice) > 0), \
        "no monotonic spatial hole burning toward the output facet"
    print("ALL TESTBENCH CHECKS PASSED")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    i_fine = np.linspace(0, I_MAX, 300)
    ax.plot(i_fine * 1e3, p_out_analytic(i_fine, law) * 1e3, "k:", lw=1,
            label="analytic clamped-gain L-I")
    ax.plot(i_levels * 1e3, p_li * 1e3, "o", c="tab:red", ms=5,
            label="segmented-gain staircase")
    ax.axvline(law["i_th"] * 1e3, c="gray", lw=0.8, ls="--",
               label=f"I_th = {law['i_th']*1e3:.1f} mA")
    ax.set_xlabel("bias current [mA]"); ax.set_ylabel("P_out [mW]")
    ax.set_title(f"L-I: {N_SEG}-section TW gain in a R1={R1}/R2={R2} cavity")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t * 1e9, p_out * 1e3, c="tab:orange", lw=0.9)
    ax.set_xlabel("time [ns]"); ax.set_ylabel("P_out [mW]")
    ax.set_title("L-I staircase transient")
    ax.grid(alpha=0.3)

    ax = axes[2]
    z = (np.arange(N_SEG) + 0.5) * DZ * 1e6
    ax.plot(z, p_slice * 1e3, "o-", c="tab:purple")
    ax.set_xlabel("position along cavity [um]  (back mirror -> output)")
    ax.set_ylabel("circulating power |R|²+|S|² [mW]")
    ax.set_title(f"spatial hole burning at {I_MAX*1e3:.0f} mA ({shb:+.0%} tilt)")
    ax.grid(alpha=0.3)

    fig.suptitle("Segmented traveling-wave Fabry-Perot laser (tw_gain_seg + "
                 "mirror, circulax) — threshold, slope, spatial hole burning",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    raise SystemExit(main())
