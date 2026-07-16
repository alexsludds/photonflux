#!/usr/bin/env python3
"""Thermo-optic bistability of a self-heating microring, solved with circulax.

A CW laser drives a high-Q silicon microring
(``models/optical_field/ring_selfheat.va``) whose resonance heats itself: the
fraction of the circulating power that is *absorbed* warms the ring, silicon's
``dn/dT > 0`` red-shifts the resonance, and that shift changes how much power the
ring stores — a nonlinear feedback loop. The one knob we drive is the **laser
wavelength**, ramped slowly up (blue -> red) and back down (red -> blue) in one
transient — the "waveform sweep, backwards and forward".

The through-port then traces a **hysteresis loop**:

* Sweeping to the **red**, the laser enters the resonance from the blue, heats
  the ring, and the resonance red-shifts *with* the laser — so the laser stays
  locked and the ring rides the high-circulating-power branch across an extended,
  triangular thermal-locking lineshape, until it can no longer keep up and snaps
  back (switch-down) well to the red of the cold resonance.
* Sweeping back to the **blue**, the ring is cold and stays cold (through ~= 1)
  until the laser reaches the *cold* resonance, where it drops abruptly into the
  sharp cold-cavity dip (switch-up).

The two directions take different paths through the same wavelengths —
bistability — over a window whose width grows with power (~90 pm at 50 uW here).
The asymmetric triangle (forward) versus sharp Lorentzian dip (backward) is the
classic experimental fingerprint of optical thermal bistability. The loop is
pinned to the analytic three-root thermal self-consistency (see
``tests/test_ring_selfheat.py``).

Everything is one differentiable JAX system: coherent-field optics
(``|E|^2`` = power) and the lumped thermal reservoir solved together by a single
Newton DC + implicit BDF2 transient.

    .venv-circulax/bin/python examples/ring_selfheat.py            # 50 uW default
    .venv-circulax/bin/python examples/ring_selfheat.py --power-uw 100   # wider loop

        -> out/ring_selfheat.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import diffrax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source

from _cavity import run_transient, terminator
from photonflux import cx

C0 = 2.99792458e8

# --- device (mirrors the model card set on the ring below) ------------------
RADIUS_UM = 8.0
N_G = 4.0
LOSS_DB_M = 300.0     # intrinsic (absorption + scatter) propagation loss
KAPPA2 = 0.004        # near-critical bus coupling -> deep, ~10 pm FWHM dip
HEAT_FRAC = 1.0       # all intrinsic loss taken as absorption -> heat
RTH_K_W = 3.0e4       # ring thermal resistance [K/W]
TAU_TH_S = 1.0e-6     # thermal time constant R_th*C_th [s]
DL_DT_PM = 80.0       # thermo-optic resonance shift [pm/K]
LAM_RES_NM = 1310.0   # cold resonance

# --- sweep defaults ---------------------------------------------------------
POWER_UW = 50.0       # laser power [uW]
LAM_LO_NM = 1309.980  # blue end of the wavelength ramp
LAM_HI_NM = 1310.250  # red end (a bit past the forward switch-down)
T_HALF = 60e-6        # duration of each sweep direction [s] (>> TAU_TH: quasi-static)
DT = 10e-9            # fixed BDF2 step [s] (A-stable; resolves the ~us snaps)

OUT = Path(__file__).resolve().parents[1] / "out" / "ring_selfheat.png"


# ===========================================================================
# Wavelength-sweep source: a triangle waveform on the ring's `lam_nm` node
# ===========================================================================
def wavelength_ramp(lam_lo: float, lam_hi: float, t_half: float):
    """Voltage source whose value is the laser wavelength [nm] vs time.

    Ramps ``lam_lo -> lam_hi`` over the first half of the run (forward, the
    laser sweeping to the red) and back over the second half (backward), so the
    whole forward+backward sweep is one transient. Same VCVS idiom as the
    ``_cavity.staircase_source``: the branch current ``i_src`` is the unknown
    that satisfies the prescribed node value.
    """
    @source(ports=("p1", "p2"), states=("i_src",))
    def WavelengthRamp(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
        frac = t / t_half                              # 0 -> 2 over the whole run
        up = jnp.where(frac < 1.0, frac, 2.0 - frac)   # 0 ->1 ->0 triangle
        lam = lam_lo + (lam_hi - lam_lo) * up
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - lam}, {}

    return WavelengthRamp


# ===========================================================================
# Netlist: CW laser -> field_to_ri -> ring; wavelength ramp on lam_nm
# ===========================================================================
def build(power_w: float, lam_lo: float, lam_hi: float, t_half: float):
    ring = dict(lambda_res_nm=LAM_RES_NM, radius_um=RADIUS_UM, n_g=N_G,
                loss_db_m=LOSS_DB_M, kappa2=KAPPA2, heat_frac=HEAT_FRAC,
                rth_k_w=RTH_K_W, tau_th_s=TAU_TH_S, dl_dt_pm=DL_DT_PM)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SRC": {"component": "laser", "settings": {"power": power_w}},
            "TAP": {"component": "f2ri"},
            "LAM": {"component": "lam_ramp"},
            "RG": {"component": "ring", "settings": ring},
            "T1": {"component": "term"},
        },
        "connections": {
            "SRC,p1": "TAP,c",
            "TAP,re": "RG,in_re", "TAP,im": "RG,in_im",
            "LAM,p1": "RG,lam_nm",
            "RG,out_re": "T1,re", "RG,out_im": "T1,im",
            "GND,p1": ("SRC,p2", "LAM,p2", "RG,gnd"),
        },
        "ports": {"po_re": "RG,out_re", "po_im": "RG,out_im", "lam": "RG,lam_nm"},
    }
    models = {"ground": lambda: 0, "laser": cx.cw_laser(),
              "f2ri": cx.field_to_ri(),
              "lam_ramp": wavelength_ramp(lam_lo, lam_hi, t_half),
              "ring": cx.va("ring_selfheat"), "term": terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


# ===========================================================================
# Analytic steady-state reference — the thermal self-consistency S-curve
# ===========================================================================
def _rates() -> dict:
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    v_g = C0 / N_G
    t_rt = circ / v_g
    alpha = LOSS_DB_M * np.log(10) / 10
    inv_tau_i = alpha * v_g / 2
    inv_tau_e = KAPPA2 / (2 * t_rt)
    inv_tau = inv_tau_i + inv_tau_e
    return dict(inv_tau_i=inv_tau_i, inv_tau_e=inv_tau_e, inv_tau=inv_tau,
                tau=1 / inv_tau, tk2=(1 / inv_tau) * 2 * inv_tau_e)


def steady_roots(lam_nm: float, power_w: float, r: dict) -> list[float]:
    """All steady temperature-rise roots of the self-heating loop at one wavelength."""
    coeff = RTH_K_W * HEAT_FRAC * (r["inv_tau_i"] / r["inv_tau_e"])
    lam_l = lam_nm * 1e-9

    def g(dt):
        lam_res = LAM_RES_NM * 1e-9 + DL_DT_PM * 1e-12 * dt
        d = 2 * np.pi * C0 * (1 / lam_l - 1 / lam_res)
        A2 = r["tk2"] ** 2 * power_w / (1 + (r["tau"] * d) ** 2)
        return dt - coeff * A2

    grid = np.linspace(0.0, coeff * r["tk2"] ** 2 * power_w * 1.05 + 1e-12, 6000)
    gv = g(grid)
    out = []
    for i in np.where(np.diff(np.sign(gv)) != 0)[0]:
        a, b = grid[i], grid[i + 1]
        for _ in range(60):
            m = 0.5 * (a + b)
            if g(a) * g(m) <= 0:
                b = m
            else:
                a = m
        out.append(0.5 * (a + b))
    return out


def bistable_window(lam_nm: np.ndarray, power_w: float, r: dict) -> np.ndarray:
    n = np.array([len(steady_roots(l, power_w, r)) for l in lam_nm])
    return lam_nm[n >= 3]


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="self-heating ring bistability sweep")
    ap.add_argument("--power-uw", type=float, default=POWER_UW)
    ap.add_argument("--t-half", type=float, default=T_HALF, help="s per direction")
    ap.add_argument("--dt", type=float, default=DT, help="fixed BDF2 step [s]")
    args = ap.parse_args()
    power_w = args.power_uw * 1e-6

    r = _rates()
    Q = (2 * np.pi * C0 / (LAM_RES_NM * 1e-9)) / (2 * r["inv_tau"])
    fwhm_pm = (LAM_RES_NM * 1e-9) ** 2 / (2 * np.pi * C0) * (2 * r["inv_tau"]) * 1e12
    print(f"cold ring: R = {RADIUS_UM:.0f} um, {LOSS_DB_M:.0f} dB/m, "
          f"kappa2 = {KAPPA2}: loaded Q ~ {Q:.2e}, FWHM ~ {fwhm_pm:.2f} pm, "
          f"tau_photon ~ {r['tau'] * 1e12:.0f} ps")

    lam_scan = np.linspace(LAM_LO_NM, LAM_HI_NM, 2000)
    win = bistable_window(lam_scan, power_w, r)
    if win.size:
        print(f"analytic bistable window: [{win.min():.4f}, {win.max():.4f}] nm "
              f"= {(win.max() - win.min()) * 1e3:.1f} pm at {args.power_uw:.0f} uW")
    else:
        print(f"analytic bistable window: none at {args.power_uw:.0f} uW")

    # --- one transient: forward (up) then backward (down) -----------------
    c = build(power_w, LAM_LO_NM, LAM_HI_NM, args.t_half)
    print(f"compiled: {c.sys_size} complex unknowns, {len(c.groups)} groups")
    t_max = 2 * args.t_half
    # fixed-step implicit BDF2 (A-stable): the ~180 ps photon mode is slaved to
    # the slow sweep, so dt << TAU_TH resolves the thermal snaps without needing
    # to resolve the fast mode. The _cavity runner is the shared idiom.
    t, sol = run_transient(c, t_max, args.dt, save_every=t_max / 2000.0)

    lam = np.asarray(c.port(sol.ys, "lam").real)                    # nm
    o_re = np.asarray(c.port(sol.ys, "po_re").real)
    o_im = np.asarray(c.port(sol.ys, "po_im").real)
    T = (o_re ** 2 + o_im ** 2) / power_w                           # through-port
    p_abs = power_w * (1.0 - T)                                     # all-pass energy balance
    dT = RTH_K_W * HEAT_FRAC * p_abs                                # quasi-static temp rise [K]
    fwd = t <= args.t_half

    print(f"transient OK: {len(t)} points over {t_max * 1e6:.0f} us")
    print(f"  through-port T range: [{T.min():.3f}, {T.max():.3f}]")
    print(f"  peak temperature rise DT = {dT.max():.2f} K")

    # self-check: inside the window the two directions ride different branches
    ok_hyst = True
    if win.size:
        wc = 0.5 * (win.min() + win.max())
        Tf = float(np.interp(wc, lam[fwd], T[fwd]))
        Tb = float(np.interp(wc, lam[~fwd][::-1], T[~fwd][::-1]))
        ok_hyst = (Tf < Tb - 0.3)
        print(f"  at window centre {wc:.4f} nm: T_forward={Tf:.3f} (hot/locked), "
              f"T_backward={Tb:.3f} (cold)  -> hysteresis dT = {abs(Tf - Tb):.3f}")
    assert ok_hyst, "expected the forward sweep to ride the hot (low-T) branch"

    # --- plot -------------------------------------------------------------
    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])
    ax_t = fig.add_subplot(gs[0, :])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("Self-heating microring — thermo-optic bistability "
                 f"(P_in = {args.power_uw:.0f} uW)", fontweight="bold")

    t_us = t * 1e6
    dl = (lam - LAM_RES_NM) * 1e3   # pm from cold resonance
    dlw = ((win.min() - LAM_RES_NM) * 1e3, (win.max() - LAM_RES_NM) * 1e3) if win.size else None

    # (a) the waveform sweep in time
    ax_t.plot(t_us, T, color="tab:green", lw=1.3)
    ax_t.set_ylabel("through-port T", color="tab:green")
    ax_t.set_xlabel("time [us]")
    ax_t.tick_params(axis="y", labelcolor="tab:green")
    axw = ax_t.twinx()
    axw.plot(t_us, dl, color="0.5", lw=1.0, ls="--")
    axw.set_ylabel("laser - cold res [pm]", color="0.5")
    axw.tick_params(axis="y", labelcolor="0.5")
    ax_t.axvline(args.t_half * 1e6, color="0.8", lw=1)
    ax_t.set_title("waveform sweep: forward (blue->red) then backward (red->blue)",
                   fontsize=9)
    ax_t.grid(alpha=0.3)

    # (b) hysteresis loop
    ax_h.plot(dl[fwd], T[fwd], color="tab:blue", lw=1.5, label="forward (blue->red)")
    ax_h.plot(dl[~fwd], T[~fwd], color="tab:red", lw=1.5, label="backward (red->blue)")
    if dlw:
        ax_h.axvspan(*dlw, color="0.88", label="bistable window")
    ax_h.set_xlabel("laser - cold resonance [pm]")
    ax_h.set_ylabel("through-port T")
    ax_h.set_title("hysteresis loop", fontsize=9)
    ax_h.legend(fontsize=7.5, loc="lower left")
    ax_h.grid(alpha=0.3)

    # (c) temperature-rise loop
    ax_d.plot(dl[fwd], dT[fwd] * 1e3, color="tab:blue", lw=1.5)
    ax_d.plot(dl[~fwd], dT[~fwd] * 1e3, color="tab:red", lw=1.5)
    if dlw:
        ax_d.axvspan(*dlw, color="0.88")
    ax_d.set_xlabel("laser - cold resonance [pm]")
    ax_d.set_ylabel("ring temperature rise DT [mK]")
    ax_d.set_title("self-heating (DT = R_th * P_absorbed)", fontsize=9)
    ax_d.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
