#!/usr/bin/env python3
"""Transient response of the traveling-wave MZM (``models/optical_power/mzm_tw.va``).

This is the bench that shows what the traveling-wave MZM models and a plain
lumped ``cos()`` modulator cannot: the **electro-optic bandwidth** of a real
distributed electrode. ``mzm_tw.va`` keeps the quasi-static transfer
``T = IL*(0.5 + 0.5*eta*cos(pi*V/Vpi))`` but first pushes the drive through two
cascaded single-pole roll-offs before it acts on the optical wave:

  * **electrode loss** — a pole at ``f_el`` (rising skin-effect / dielectric
    loss of the microwave line), and
  * **velocity walk-off** — optical and RF waves co-propagate at group indices
    ``n_opt`` and ``n_rf``; over the electrode the optical phase averages the
    drive across the walk-off window ``T_w = |n_rf - n_opt|*len/c``. That boxcar
    average is the textbook ``sinc(f*T_w)``, captured here by its equivalent
    ``-3 dB`` pole ``f_w = 0.443/T_w``. Velocity-matched (``n_rf == n_opt``)
    kills this pole and only ``f_el`` limits the bandwidth.

The bench holds the optical input at a constant CW power (power-domain node =
watts) and drives the RF electrode with an ideal fast source — no 50 ohm
network, so the only bandwidth limit the optical output sees is the model's
*intrinsic* EO response (the electrical RC of a matched 50||50 electrode with
``cel`` ~ 50 fF sits near 130 GHz, far above the EO poles). Two studies:

  1. **Small-signal step at quadrature.** Bias the electrode at ``Vpi/2`` and
     apply a small fast step. The optical output is a clean cascaded-pole step
     response; we measure its 10-90% rise time and check it against the analytic
     poles (``tr ~= 2.2*sqrt(sum tau_i^2)``, rise times adding in RSS).

  2. **Full-swing NRZ eye.** Drive a PRBS-like ``0 -> Vpi`` bit stream and fold
     the optical output into an eye. Velocity walk-off slows the edges and
     closes the eye that a velocity-matched electrode keeps open.

Everything is one JAX system solved by a single Newton DC + Diffrax transient.

    python examples/mzm_tw_transient.py            # -> out/mzm_tw_transient.png
    python examples/mzm_tw_transient.py --baud 40e9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.electronic import Resistor, VoltageSource

from _progress import transient_progress_meter

from photonflux import cx

C0 = 299792458.0

# --- fixed bench conditions --------------------------------------------------
P_IN = 1e-3          # constant CW input optical power [W] (power-domain node)
VPI = 1.5            # half-wave voltage [V] (mzm_tw default)
IL_DB = 3.0
ER_DB = 20.0
LEN_MM = 4.0         # electrode length [mm]
F_EL = 35e9          # electrode-loss pole [Hz]
N_OPT = 4.2          # optical group index

# The two electrodes we compare: same loss pole, differing only in whether the
# microwave line is velocity-matched to the optical wave.
CONFIGS = {
    "velocity-matched": dict(n_rf=N_OPT, color="tab:blue"),
    "walk-off": dict(n_rf=2.4, color="tab:red"),
}

OUT = Path(__file__).resolve().parents[1] / "out" / "mzm_tw_transient.png"


# ===========================================================================
# Ideal RF drive: a voltage source following an arbitrary (t, v) waveform
# ===========================================================================
def waveform_source(times: np.ndarray, vals: np.ndarray):
    """Ideal voltage source enforcing ``V(p1,p2) = interp(t; times, vals)``."""
    from circulax.components.base_component import Signals, States, source

    tt = jnp.asarray(times)
    vv = jnp.asarray(vals)

    @source(ports=("p1", "p2"), states=("i_src",))
    def WaveformSource(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
        v = jnp.interp(t, tt, vv)
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": (signals.p1 - signals.p2) - v}, {}

    return WaveformSource


def edge_waveform(levels, holds, tr):
    """Breakpoints for a piecewise-constant drive with finite ``tr`` edges.

    ``levels[i]`` is held for ``holds[i]`` seconds; consecutive levels are joined
    by a linear ramp of width ``tr``. Returns monotonic (times, vals) arrays for
    ``jnp.interp``.
    """
    times, vals = [0.0], [levels[0]]
    t = 0.0
    for i, (lvl, hold) in enumerate(zip(levels, holds)):
        # hold the current level
        t += hold
        times.append(t)
        vals.append(lvl)
        # ramp to the next level (if any)
        if i + 1 < len(levels):
            t += tr
            times.append(t)
            vals.append(levels[i + 1])
    return np.asarray(times), np.asarray(vals)


def prbs_bits(n: int, seed: int = 0x2A) -> np.ndarray:
    """Deterministic PRBS-7 bit stream (x^7 + x^6 + 1), no RNG at import time."""
    state = seed & 0x7F or 1
    bits = np.empty(n, dtype=int)
    for i in range(n):
        bit = ((state >> 6) ^ (state >> 5)) & 1
        state = ((state << 1) | bit) & 0x7F
        bits[i] = state & 1
    return bits


# ===========================================================================
# Netlist: CW power in -> mzm_tw -> pout, RF electrode driven by `drive`
# ===========================================================================
def build_net(n_rf: float) -> dict:
    return {
        "instances": {
            "GND": {"component": "ground"},
            # constant optical input power: a power-domain node held at P_IN watts
            "PIN": {"component": "vsrc", "settings": {"V": P_IN}},
            "DRV": {"component": "drive"},
            "MOD": {"component": "mzm_tw",
                    "settings": {"vpi": VPI, "vbias": 0.0, "il_db": IL_DB,
                                 "er_db": ER_DB, "len": LEN_MM * 1e-3,
                                 "n_rf": n_rf, "n_opt": N_OPT, "f_el": F_EL}},
            # keeps the driven pout node well conditioned (draws ~0 current)
            "RL": {"component": "res", "settings": {"R": 1e9}},
        },
        "connections": {
            "GND,p1": ("PIN,p2", "DRV,p2", "MOD,vn", "MOD,gnd", "RL,p2"),
            "PIN,p1": "MOD,pin",
            "DRV,p1": "MOD,vp",
            "MOD,pout": "RL,p1",
        },
        "ports": {"pout": "MOD,pout"},
    }


def run(n_rf: float, drive_cls, t_max: float, dt: float, save_every: float):
    """DC then fixed-step BDF2: the mzm_tw electrode adds internal ddt poles at
    ~4.5 ps that make the system stiff, so — like the other VA/OSDI benches
    (see examples/_cavity.py) — we march it with the implicit BDF2 solver rather
    than letting the adaptive path chase the picosecond time constants."""
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    models = {
        "ground": lambda: 0,
        "vsrc": VoltageSource,
        "res": Resistor,
        "drive": drive_cls,
        "mzm_tw": cx.va("mzm_tw"),
    }
    circuit = compile_circuit(build_net(n_rf), models, is_complex=True)
    y0 = circuit.dc()
    ts = jnp.arange(0.0, t_max, save_every)
    sol = circuit.transient(
        t0=0.0, t1=t_max, dt0=dt, y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=circuit.solver, newton_max_steps=40),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=int(t_max / dt) + 10, throw=False,
        progress_meter=transient_progress_meter(),
    )
    if sol.result != diffrax.RESULTS.successful:
        raise RuntimeError(f"transient FAILED ({n_rf=}): {sol.result}")
    t = np.asarray(sol.ts)
    # power-domain optical node: the node voltage *is* the power in watts
    p_out = np.asarray(circuit.port(sol.ys, "pout").real)
    return t, p_out


# ===========================================================================
# Analysis helpers
# ===========================================================================
def rise_time_10_90(t: np.ndarray, y: np.ndarray) -> float:
    """10-90% rise time of a monotone-ish step (works for rising or falling)."""
    y0, y1 = y[0], y[-1]
    span = y1 - y0
    lo, hi = y0 + 0.1 * span, y0 + 0.9 * span
    frac = (y - y0) / span  # normalized 0..1

    def cross(level):
        idx = np.argmax(frac >= level)
        if idx == 0:
            return t[0]
        x0, x1 = frac[idx - 1], frac[idx]
        return t[idx - 1] + (level - x0) / (x1 - x0) * (t[idx] - t[idx - 1])

    return abs(cross(0.9) - cross(0.1))


def analytic_rise_time(n_rf: float) -> float:
    """RSS of the cascaded-pole rise times, tr_i = 2.2 * tau_i."""
    taus = [1.0 / (2.0 * np.pi * F_EL)]                       # electrode loss
    tw = abs(n_rf - N_OPT) * (LEN_MM * 1e-3) / C0
    if tw > 0:
        taus.append(tw / (2.0 * np.pi * 0.443))              # walk-off
    return 2.2 * float(np.sqrt(sum(tau ** 2 for tau in taus)))


# ===========================================================================
# Study 1: small-signal step response at quadrature
# ===========================================================================
def study_step():
    t_edge = 30e-12
    tr_drive = 1e-12
    t_max = 260e-12
    v_bias = VPI / 2.0            # quadrature
    dv = 0.1 * VPI               # small signal
    levels = [v_bias - dv / 2, v_bias + dv / 2]
    holds = [t_edge, t_max]       # (second hold is trimmed by t_max)
    times, vals = edge_waveform(levels, holds, tr_drive)
    drive = waveform_source(times, vals)

    results = {}
    for name, cfg in CONFIGS.items():
        t, p_out = run(cfg["n_rf"], drive, t_max, dt=0.25e-12, save_every=0.25e-12)
        # measure only over the response after the edge starts
        m = t >= t_edge
        tr_meas = rise_time_10_90(t[m], p_out[m])
        tr_pred = analytic_rise_time(cfg["n_rf"])
        results[name] = dict(t=t, p=p_out, tr_meas=tr_meas, tr_pred=tr_pred,
                             color=cfg["color"])
        print(f"  step [{name:>16}]: rise 10-90 = {tr_meas * 1e12:6.1f} ps "
              f"(analytic {tr_pred * 1e12:5.1f} ps, "
              f"f_3dB ~= {0.35 / tr_meas / 1e9:5.1f} GHz)")
    return results, t_edge


# ===========================================================================
# Study 2: full-swing NRZ eye
# ===========================================================================
def study_eye(baud: float):
    ui = 1.0 / baud
    tr_drive = min(6e-12, 0.25 * ui)
    n_bits = 96
    bits = prbs_bits(n_bits)
    levels = [float(b) * VPI for b in bits]
    holds = [ui] * n_bits
    times, vals = edge_waveform(levels, holds, tr_drive)
    t_max = float(times[-1])

    results = {}
    for name, cfg in CONFIGS.items():
        drive = waveform_source(times, vals)
        # march and save at ui/40 -> 40 samples/UI resolves the eye
        t, p_out = run(cfg["n_rf"], drive, t_max,
                       dt=ui / 40.0, save_every=ui / 40.0)
        results[name] = dict(t=t, p=p_out, color=cfg["color"])
    return results, ui


def fold_eye(t: np.ndarray, y: np.ndarray, ui: float, span_ui: int = 2,
             sps: int = 40, skip_ui: int = 8):
    """Fold the waveform into overlaid eye traces.

    Resamples onto a uniform ``sps`` samples/UI grid (past the ``skip_ui``
    settling head) and reshapes into consecutive ``span_ui``-wide windows so
    each is drawn as a continuous line. Returns (phase [s], rows[n_windows]).
    """
    t0 = skip_ui * ui
    tgrid = np.arange(t0, t[-1], ui / sps)
    yg = np.interp(tgrid, t, y)
    win = span_ui * sps
    n = (len(yg) // win) * win
    rows = yg[:n].reshape(-1, win)
    phase = (np.arange(win) / sps) * ui
    return phase, rows


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baud", type=float, default=50e9,
                    help="NRZ symbol rate for the eye study [baud] (default 50e9)")
    args = ap.parse_args()

    print("mzm_tw traveling-wave MZM — transient EO bandwidth bench")
    print(f"  Vpi={VPI} V, len={LEN_MM} mm, n_opt={N_OPT}, f_el={F_EL/1e9:.0f} GHz")
    for name, cfg in CONFIGS.items():
        tw = abs(cfg["n_rf"] - N_OPT) * (LEN_MM * 1e-3) / C0
        fw = 0.443 / tw if tw > 0 else float("inf")
        print(f"  {name:>16}: n_rf={cfg['n_rf']}, walk-off pole f_w = "
              f"{'off' if tw == 0 else f'{fw/1e9:.1f} GHz'}")

    print("step response (small-signal, biased at quadrature):")
    step, t_edge = study_step()

    print(f"eye diagram (full-swing PRBS NRZ at {args.baud/1e9:.0f} Gbaud):")
    eye, ui = study_eye(args.baud)

    # --- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax_step = fig.add_subplot(gs[0, :])
    ax_m = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])

    # step response, normalized 0..1
    for name, r in step.items():
        m = r["t"] >= t_edge
        t = (r["t"][m] - t_edge) * 1e12
        y = r["p"][m]
        y = (y - y[0]) / (y[-1] - y[0])
        ax_step.plot(t, y, color=r["color"],
                     label=f"{name}: {r['tr_meas']*1e12:.1f} ps "
                           f"(pole {r['tr_pred']*1e12:.1f} ps)")
    ax_step.axhline(0.1, color="gray", ls=":", lw=0.8)
    ax_step.axhline(0.9, color="gray", ls=":", lw=0.8)
    ax_step.set_xlim(0, 120)
    ax_step.set_xlabel("time after edge [ps]")
    ax_step.set_ylabel("normalized optical step")
    ax_step.set_title("mzm_tw EO step response — velocity walk-off sets the rise time")
    ax_step.legend(loc="lower right", fontsize=8, title="10-90% rise")
    ax_step.grid(alpha=0.3)

    # eyes
    for ax, name in ((ax_m, "velocity-matched"), (ax_w, "walk-off")):
        r = eye[name]
        phase, rows = fold_eye(r["t"], r["p"], ui)
        for row in rows:
            ax.plot(phase * 1e12, row * 1e3, color=r["color"], alpha=0.25, lw=0.7)
        ax.set_title(f"{name} eye")
        ax.set_xlabel("time [ps]")
        ax.grid(alpha=0.3)
    ax_m.set_ylabel("optical power [mW]")
    # share the same y-scale so the eye-height difference is visible
    lo = min(ax_m.get_ylim()[0], ax_w.get_ylim()[0])
    hi = max(ax_m.get_ylim()[1], ax_w.get_ylim()[1])
    ax_m.set_ylim(lo, hi)
    ax_w.set_ylim(lo, hi)

    fig.suptitle(f"Traveling-wave MZM transient — {args.baud/1e9:.0f} Gbaud NRZ",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
