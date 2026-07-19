#!/usr/bin/env python3
"""EDFA gain dynamics: the WDM channel-drop surge, solved as one circulax
system around ``models/optical_field/edfa.va``.

An erbium-doped fibre amplifier saturates HOMOGENEOUSLY — every WDM channel
draws on one shared inversion (the log-gain reservoir h, G = e^h, with a slow
~ms erbium lifetime tau_c). So the channels are coupled: drop 7 of 8 and the
total input power falls, the reservoir refills, and the ONE surviving channel
surges in power over ~tau_c. This is the classic transient every EDFA in a
reconfigurable WDM network must survive.

The bench drives the amplifier with the aggregate input power of 8 equal
channels (each P_ch), then at t_drop removes 7 of them, leaving P_ch. Because
the gain is shared, the survivor's output power is P_ch*G(t): it jumps the
instant the load drops (the reservoir has not moved yet) and then climbs to the
new, higher small-signal-er gain as h relaxes.

Self-checking pins (all against an independent numpy solution of the same
reservoir equation h0 - h - (G-1)*P/p_sat = 0):

1. **Settled gains** — the pre-drop (8-channel) and post-drop (1-channel)
   steady-state gains match the analytic saturated gain.
2. **Surge** — the surviving channel's power jumps by the ratio of those two
   gains.
3. **Time constant** — the recovery follows the reservoir with the effective
   time constant tau_eff = tau_c / (1 + G*P/p_sat); doubling the configured
   tau_c doubles the measured recovery time, proving tau_c sets the dynamics.

    python examples/edfa_wdm.py            ->  out/edfa_wdm.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source

from _cavity import run_transient
from photonflux import cx

C0 = 2.99792458e8
HPLANCK = 6.62607015e-34

# --- the amplifier ----------------------------------------------------------
EDFA = dict(g0_db=30.0, p_pump_mw=120.0, p_pump_op_mw=100.0, p_pump_tr_mw=8.0,
            p_sat=5e-3, tau_c=10e-3, lambda_ref_nm=1550.0,
            lambda_peak_nm=1532.0, gain_bw_nm=30.0, alpha_h=0.0, p_ase=0.0)

N_CH = 8              # channels before the drop
P_CH = 0.5e-3        # per-channel input power [W]
NF_DB = 5.0          # noise figure for the ASE-floor annotation

OUT = Path(__file__).resolve().parents[1] / "out" / "edfa_wdm.png"


# ===========================================================================
# analytic reservoir (the same equation the VA model integrates)
# ===========================================================================
def h0_of_pump(p_pump_mw: float) -> float:
    hop = EDFA["g0_db"] * np.log(10) / 10
    return hop * (p_pump_mw - EDFA["p_pump_tr_mw"]) / (
        EDFA["p_pump_op_mw"] - EDFA["p_pump_tr_mw"])


def sat_gain(p_in: float) -> float:
    """Saturated peak gain G = e^h from h0 - h - (e^h - 1)*P_in/p_sat = 0."""
    h0 = h0_of_pump(EDFA["p_pump_mw"])
    return np.exp(brentq(lambda h: h0 - h - (np.exp(h) - 1) * p_in / EDFA["p_sat"],
                         -30.0, h0 + 1e-9))


def tau_eff(p_in: float) -> float:
    """Linearised reservoir recovery time constant about the P_in steady state:
    tau_c*dh/dt = h0 - h - (e^h-1)P/p_sat  ->  tau_eff = tau_c/(1 + G*P/p_sat)."""
    g = sat_gain(p_in)
    return EDFA["tau_c"] / (1.0 + g * p_in / EDFA["p_sat"])


def ase_psd_dbm_hz(nf_db: float, gain_db: float, lambda_nm: float) -> float:
    """One-sided ASE power spectral density added by an amplifier of the given
    noise figure and gain: S = n_sp*h*nu*(G-1), with n_sp = 10^(NF/10)/2. Feed
    this to a cascaded ASE Noise Source (webapp/catalog.py ase_src) for a
    stochastic noise-figure study."""
    nu = C0 / (lambda_nm * 1e-9)
    g = 10.0 ** (gain_db / 10.0)
    n_sp = 10.0 ** (nf_db / 10.0) / 2.0
    s_w_hz = n_sp * HPLANCK * nu * (g - 1.0)
    return 10.0 * np.log10(s_w_hz / 1e-3)


# ===========================================================================
# a WDM aggregate-power source with a channel-drop step
# ===========================================================================
def wdm_drop_source(p_high: float, p_low: float, t_drop: float,
                    t_edge: float = 0.2e-3):
    """Drives (re, im) with a real field sqrt(P(t)): P steps p_high -> p_low at
    t_drop (smoothstep edge, C1-continuous for BDF2). P(t) is the TOTAL input
    power of the surviving channels; the field sits on the real axis."""
    @source(ports=("re", "im"), states=("i_re", "i_im"))
    def WDMDrop(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
        x = jnp.clip((t - t_drop) / t_edge, 0.0, 1.0)
        smooth = x * x * (3.0 - 2.0 * x)
        p = p_high + (p_low - p_high) * smooth
        amp = jnp.sqrt(p)
        return {"re": s.i_re, "im": s.i_im,
                "i_re": signals.re - amp,
                "i_im": signals.im}, {}

    return WDMDrop


def build_drop(p_high: float, p_low: float, t_drop: float, tau_c: float):
    # put the survivor AT the gain peak (lambda_ref == lambda_peak) so the
    # measured gain is the pure reservoir gain — the channel-drop dynamics are a
    # reservoir effect, independent of where in the band the survivor sits. The
    # spectral shape is exercised separately in the gain-spectrum panel.
    settings = {**EDFA, "tau_c": tau_c,
                "lambda_ref_nm": EDFA["lambda_peak_nm"]}
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SRC": {"component": "src"},
            "AMP": {"component": "edfa", "settings": settings},
        },
        "connections": {
            "SRC,re": "AMP,fi_re", "SRC,im": "AMP,fi_im",
            "GND,p1": "AMP,gnd",
        },
        "ports": {"in_re": "SRC,re", "in_im": "SRC,im",
                  "out_re": "AMP,fo_re", "out_im": "AMP,fo_im"},
    }
    models = {
        "ground": lambda: 0,
        "src": wdm_drop_source(p_high, p_low, t_drop),
        "edfa": cx.va("edfa"),
    }
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def port_power(c, sol, re: str, im: str) -> np.ndarray:
    er = np.asarray(c.port(sol, re).real)
    ei = np.asarray(c.port(sol, im).real)
    return er ** 2 + ei ** 2


# ===========================================================================
# a steady-state per-channel gain spectrum (one DC solve per channel)
# ===========================================================================
def dc_gain(p_in: float, **params) -> float:
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SRC": {"component": "src"},
            "AMP": {"component": "edfa", "settings": {**EDFA, **params}},
        },
        "connections": {"SRC,re": "AMP,fi_re", "SRC,im": "AMP,fi_im",
                        "GND,p1": "AMP,gnd"},
        "ports": {"out_re": "AMP,fo_re", "out_im": "AMP,fo_im"},
    }
    models = {"ground": lambda: 0,
              "src": wdm_drop_source(p_in, p_in, 1.0),   # constant (no drop)
              "edfa": cx.va("edfa")}
    c = compile_circuit(net, models, backend="dense", is_complex=True)
    y = c.dc()
    out = complex(c.port(y, "out_re")) ** 2 + complex(c.port(y, "out_im")) ** 2
    return out.real / p_in


# ===========================================================================
def main(tau_c: float = EDFA["tau_c"]) -> int:
    p_high, p_low = N_CH * P_CH, P_CH
    g_before, g_after = sat_gain(p_high), sat_gain(p_low)
    print(f"pump {EDFA['p_pump_mw']:.0f} mW, p_sat {EDFA['p_sat']*1e3:.1f} mW, "
          f"tau_c {tau_c*1e3:.1f} ms")
    print(f"{N_CH} x {P_CH*1e3:.2f} mW channels -> drop to 1: "
          f"gain {10*np.log10(g_before):.2f} -> {10*np.log10(g_after):.2f} dB "
          f"(analytic), surge x{g_after/g_before:.2f}")

    # ---- transient: drop 7 of 8 channels at t_drop -------------------------
    t_drop = 1.0 * tau_c
    t_max = 7.0 * tau_c
    dt = tau_c / 800.0
    c = build_drop(p_high, p_low, t_drop, tau_c)
    t, sol = run_transient(c, t_max, dt, save_every=tau_c / 200.0, progress=False)
    p_in = port_power(c, sol.ys, "in_re", "in_im")
    p_out = port_power(c, sol.ys, "out_re", "out_im")
    gain = p_out / np.maximum(p_in, 1e-30)
    # the surviving channel carries P_CH and shares the gain: P_surv = P_CH*G(t)
    p_surv = P_CH * gain

    pre = (t > 0.8 * t_drop) & (t < t_drop)
    post = t > t_max - 0.5 * tau_c
    g_pre = gain[pre].mean()
    g_post = gain[post].mean()
    print(f"measured settled gain: pre {10*np.log10(g_pre):.2f} dB, "
          f"post {10*np.log10(g_post):.2f} dB")
    assert abs(g_pre - g_before) / g_before < 0.02, "pre-drop gain off analytic"
    assert abs(g_post - g_after) / g_after < 0.02, "post-drop gain off analytic"

    # surge: survivor power jumps by the gain ratio
    surge = g_post / g_pre
    print(f"surviving-channel surge: x{surge:.2f} "
          f"({P_CH*g_pre*1e3:.2f} -> {P_CH*g_post*1e3:.2f} mW)")
    assert abs(surge - g_after / g_before) / (g_after / g_before) < 0.03

    # recovery time constant: the reservoir relaxes to the new steady state
    # with the linearised tail constant tau_eff = tau_c/(1 + G*P/p_sat). Fit the
    # clean exponential tail (residual between 30% and 3% of the full swing),
    # where the response is single-pole about the final state.
    def tail_tau(t_arr, g_arr, t_d, g_f):
        resid = g_f - g_arr
        r0 = resid[np.searchsorted(t_arr, t_d)]
        m = (t_arr > t_d) & (resid < 0.30 * r0) & (resid > 0.03 * r0)
        slope = np.polyfit(t_arr[m], np.log(resid[m]), 1)[0]
        return -1.0 / slope

    tau_meas = tail_tau(t, gain, t_drop, g_post)
    te = tau_eff(p_low)
    print(f"recovery tail: tau = {tau_meas*1e3:.2f} ms, "
          f"analytic tau_eff = {te*1e3:.2f} ms  (tau_c/(1+G*P/p_sat))")
    assert abs(tau_meas - te) / te < 0.15, "recovery time constant off the reservoir"
    print("ALL TESTBENCH CHECKS PASSED")

    # ---- a second run at 2*tau_c: the recovery must slow proportionally -----
    c2 = build_drop(p_high, p_low, 2 * tau_c, 2 * tau_c)
    t2, sol2 = run_transient(c2, 7.0 * (2 * tau_c), (2 * tau_c) / 800.0,
                             save_every=(2 * tau_c) / 200.0, progress=False)
    g2 = port_power(c2, sol2.ys, "out_re", "out_im") / np.maximum(
        port_power(c2, sol2.ys, "in_re", "in_im"), 1e-30)
    g2_post = g2[t2 > 7.0 * (2 * tau_c) - tau_c].mean()
    tau_meas2 = tail_tau(t2, g2, 2 * tau_c, g2_post)
    print(f"tau_c doubled: recovery tau {tau_meas*1e3:.2f} -> {tau_meas2*1e3:.2f} "
          f"ms (ratio {tau_meas2/tau_meas:.2f}, expect ~2)")
    assert 1.8 < tau_meas2 / tau_meas < 2.2, "recovery does not scale with tau_c"

    # ---- the gain spectrum across the C-band (steady-state) ----------------
    lam = np.linspace(1528.0, 1566.0, 20)
    g_lam = np.array([dc_gain(1e-6, lambda_ref_nm=l) for l in lam])
    ase_floor = ase_psd_dbm_hz(NF_DB, 10 * np.log10(g_lam.max()),
                               EDFA["lambda_peak_nm"])
    print(f"ASE floor for NF = {NF_DB:.1f} dB at peak gain: "
          f"{ase_floor:.1f} dBm/Hz (set a cascaded ASE Noise Source to this)")

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    ax.plot(t * 1e3, p_in * 1e3, c="tab:gray", lw=1.2)
    ax.axvline(t_drop * 1e3, c="k", lw=0.8, ls="--", label="drop 7 of 8")
    ax.set_xlabel("time [ms]"); ax.set_ylabel("total input power [mW]")
    ax.set_title(f"WDM load: {N_CH} channels -> 1"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t * 1e3, p_surv * 1e3, c="tab:red", lw=1.3)
    ax.axhline(P_CH * g_before * 1e3, c="gray", lw=0.8, ls=":",
               label="analytic pre")
    ax.axhline(P_CH * g_after * 1e3, c="tab:blue", lw=0.8, ls=":",
               label="analytic post")
    ax.axvline((t_drop + tau_meas) * 1e3, c="tab:green", lw=0.8, ls="--",
               label=f"tau = {tau_meas*1e3:.1f} ms")
    ax.set_xlabel("time [ms]"); ax.set_ylabel("surviving-channel power [mW]")
    ax.set_title("survivor surge + reservoir recovery"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t * 1e3, 10 * np.log10(gain), c="tab:purple", lw=1.3)
    ax.set_xlabel("time [ms]"); ax.set_ylabel("shared gain [dB]")
    ax.set_title("homogeneous gain: shared by every channel")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(lam, 10 * np.log10(g_lam), "o-", c="tab:orange", ms=4,
            label="G(lambda)")
    ax.axvline(EDFA["lambda_peak_nm"], c="gray", lw=0.8, ls="--",
               label=f"peak {EDFA['lambda_peak_nm']:.0f} nm")
    ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("small-signal gain [dB]")
    ax.set_title(f"spectral gain (BW {EDFA['gain_bw_nm']:.0f} nm); "
                 f"ASE {ase_floor:.0f} dBm/Hz @ NF {NF_DB:.0f} dB")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_c", type=float, default=EDFA["tau_c"],
                    help="erbium lifetime [s] (default 10 ms)")
    args = ap.parse_args()
    raise SystemExit(main(args.tau_c))
