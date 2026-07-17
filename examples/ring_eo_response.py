#!/usr/bin/env python3
"""Testbench: small-signal electro-optic (EO) frequency response of the Si ring modulator.

The device under test is ``models/optical_field/ring_mod.va`` — the same
coupled-mode-theory microring used by ``ring_mod_sky130.py`` (R = 7.5 um,
n_g = 4.0, 7000 dB/m junction loss, bus power coupling kappa^2 = 10 %,
45 pm/V depletion tuning, 0.5 fF/um junction cap). This bench asks a different
question than the eye-diagram testbench: **how fast can the ring modulate?** —
i.e. its S21-style small-signal EO frequency response, |dP_thru/dV|(f).

Because the modulator is a *resonator*, its intracavity field cannot follow the
electrode faster than it can charge and discharge: the photon lifetime sets a
first-order low-pass on the modulation. That is the ``da/dt`` in the CMT model,
so the compiled circulax circuit reproduces it with no extra machinery.

Method (a swept-tone / lock-in AC measurement, since the coherent-field solver
is a time-domain ODE, not a frequency-domain S-parameter engine):

    cx.cw_laser  (blue-detuned to the max-slope point of the resonance,
                  where dT/dlambda is steepest so intensity modulation is linear)
      -> ring_mod.va   (electrode driven by a small sinusoid, V_ac sin(2*pi*f*t))
      -> |E|^2         (through-port optical power)

For each RF frequency f the electrode is driven with a small tone (default
5 mV, ~0.2 pm of resonance swing — deep in the linear regime), the circuit is
integrated to steady state, and a single-bin DFT over an integer number of
periods extracts the amplitude of the through-port power tone. Normalised to
its low-frequency value, that is the EO frequency response. It is pinned
point-by-point against the closed-form linearisation of the same CMT equations
(both modulation sidebands + |E|^2 detection), and the extracted -3 dB corner
is checked against the analytic photon-lifetime bandwidth
f_3dB = 1 / (2*pi*tau_photon) ~ 44 GHz.

Two physical effects the curve makes visible:

* **Response peaking.** Biased on the resonance slope the response rises a few
  tenths of a dB before it rolls off — the detuned cavity resonantly enhances
  one modulation sideband near the linewidth (the small-signal cousin of the
  transient overshoot the eye testbench sees).
* **The electrode RC pole.** ``--rs`` puts a source resistance in series with
  the junction capacitance; the electrode voltage then rolls off at
  f_RC = 1/(2*pi*R*Cj) *on top of* the optical cavity limit, so the real EO
  bandwidth is the smaller of the two. With the default ideal drive (rs = 0)
  the bench isolates the pure cavity/photon-lifetime response.

    .venv-circulax/bin/python examples/ring_eo_response.py
    .venv-circulax/bin/python examples/ring_eo_response.py --kappa2 0.30   # broader linewidth, faster
    .venv-circulax/bin/python examples/ring_eo_response.py --rs 2e3        # add the electrode RC pole
    .venv-circulax/bin/python examples/ring_eo_response.py --vac 2e-3 --points 28

        -> out/ring_eo_response.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source
from circulax.solvers.transient import BDF2VectorizedTransientSolver

# reuse the physical device + optics chain from the ring eye-diagram testbench
from ring_mod_sky130 import (
    C0,
    P_LASER,
    RADIUS_UM,
    WAVELENGTH_NM,
    base_models,
    design_ring,
    optics_instances,
)

# --- testbench knobs (also settable from the command line) ---------------------
V_AC = 5e-3            # small-signal drive amplitude [V] (linear regime)
F_START = 2e9         # sweep start [Hz]
F_STOP = 300e9        # sweep stop [Hz]
N_POINTS = 22         # frequencies (log-spaced)
R_SERIES = 0.0        # electrode series resistance [ohm]; 0 = ideal drive
N_SETTLE_TAU = 20.0   # photon lifetimes to settle before the lock-in window
N_WIN = 8             # integer RF periods in the lock-in window
SPP = 32              # lock-in samples per RF period

OUT = Path(__file__).resolve().parents[1] / "out" / "ring_eo_response.png"


# ===========================================================================
# Bias point: park the laser on the steepest part of the resonance
# ===========================================================================
def small_signal_bias(design: dict) -> dict:
    """Blue-detune the laser to the Lorentzian's max-slope (inflection) point.

    A Lorentzian L(x) = 1/(1+x^2) has its steepest slope at x = 1/sqrt(3) HWHM
    from line centre — the linear intensity-modulation bias, independent of the
    (vanishing) small-signal swing. Overwrites the eye testbench's swing-aware
    detune with this pure small-signal one and returns the CMT detuning delta0.
    """
    hwhm_pm = design["fwhm_pm"] / 2
    detune_pm = hwhm_pm / np.sqrt(3.0)
    design = dict(design)
    design["detune_pm"] = detune_pm
    design["lambda_light_nm"] = WAVELENGTH_NM - detune_pm * 1e-3
    lam_l, lam_res = design["lambda_light_nm"] * 1e-9, WAVELENGTH_NM * 1e-9
    design["delta0"] = 2 * np.pi * C0 * (1 / lam_l - 1 / lam_res)
    return design


# ===========================================================================
# The small-signal drive: a sinusoidal voltage source component
# ===========================================================================
def sine_source(v_bias: float, v_ac: float, freq: float):
    """Ideal voltage source V = v_bias + v_ac*sin(2*pi*freq*t) (starts at 0)."""

    @source(ports=("p1", "p2"), states=("i_src",))
    def Sine(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
        v = v_bias + v_ac * jnp.sin(2.0 * jnp.pi * freq * t)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}

    return Sine


# ===========================================================================
# Analytic small-signal EO response (linearised CMT, both sidebands + |E|^2)
# ===========================================================================
def analytic_eo_response(freqs: np.ndarray, design: dict,
                         r_series: float = 0.0) -> np.ndarray:
    """|dP_thru|(f) per unit drive from the linearised ring_mod CMT model.

    Intracavity field a: da/dt = (j*delta - 1/tau) a + j*kappa^2 s_in, output
    s_out = s_in + j*a. Linearise delta = delta0 + delta1(t) about the DC point
    a0. A cosine drive puts a sideband at +f and -f, each transferred by
    G(w) = -a0 / (1/tau + j(w - delta0)); the detected power tone is
    2*Re(s0* s_out1). An optional electrode RC = 1/(1 + j*w*R*Cj) multiplies the
    drive (the junction charges through the source resistance).
    """
    tau = design["tau"]
    inv_tau = 1.0 / tau
    delta0 = design["delta0"]
    kappa2 = design["tk2"] / tau           # = 2/tau_e (bus coupling rate)
    s_in = np.sqrt(P_LASER)                # real CW field, |s_in|^2 = P_laser
    a0 = 1j * kappa2 * s_in / (inv_tau - 1j * delta0)
    s0 = s_in + 1j * a0                     # DC through-port field

    def transfer(w):
        g = -a0 / (inv_tau + 1j * (w - delta0))     # field response to delta1
        return np.conj(s0) * g                       # -> power via 2 Re(s0* .)

    out = np.empty(len(freqs))
    for k, f in enumerate(freqs):
        w = 2.0 * np.pi * f
        # cosine drive: amplitude of the f-tone in 2 Re(s0* s_out1)
        resp = np.abs(transfer(w) + np.conj(transfer(-w)))
        rc = 1.0 / (1.0 + 1j * w * r_series * design["cj"])   # electrode pole
        out[k] = resp * np.abs(rc)
    return out


# ===========================================================================
# Simulated small-signal EO response: one lock-in transient per frequency
# ===========================================================================
def measure_response(design: dict, freqs: np.ndarray, v_ac: float,
                     r_series: float) -> np.ndarray:
    """Drive each frequency, integrate to steady state, DFT out the power tone."""
    mdl = base_models()
    amps = np.empty(len(freqs))
    tau = design["tau"]
    for k, freq in enumerate(freqs):
        inst, conn, gnd = optics_instances(design)
        inst["VS"] = {"component": "sine"}
        if r_series > 0:
            inst["RS"] = {"component": "res", "settings": {"R": r_series}}
            conn["VS,p1"] = "RS,p1"
            conn["RS,p2"] = "RING,vp"
        else:
            conn["VS,p1"] = "RING,vp"
        conn["GND,p1"] = tuple(gnd + ["VS,p2"])
        m = dict(mdl)
        m["sine"] = sine_source(0.0, v_ac, freq)
        net = {"instances": inst, "connections": conn,
               "ports": {"prx": "PD,po_p"}}
        c = compile_circuit(net, m, backend="dense", is_complex=True,
                            max_steps=300)

        # settle for many photon lifetimes, then sample exactly N_WIN periods
        # on an integer grid (spp/period): the drive tone lands on DFT bin
        # N_WIN, orthogonal to the large DC term — no spectral leakage.
        period = 1.0 / freq
        n = N_WIN * SPP
        t_lock = N_SETTLE_TAU * tau + np.arange(n) * (period / SPP)
        t_stop = float(t_lock[-1])
        dt = min(period / 64.0, tau / 8.0)     # resolve both drive and cavity
        sol = c.transient(
            t0=0.0, t1=t_stop, dt0=dt, y0=c.dc(),
            saveat=diffrax.SaveAt(ts=jnp.asarray(t_lock)),
            transient_solver=BDF2VectorizedTransientSolver(
                linear_solver=c.solver, newton_max_steps=40),
            max_steps=int(t_stop / dt) + 20, throw=False,
            stepsize_controller=diffrax.ConstantStepSize())
        assert sol.result == diffrax.RESULTS.successful, \
            f"transient failed at {freq/1e9:g} GHz: {sol.result}"
        p_thru = np.asarray(jnp.abs(c.port(sol.ys, "prx")) ** 2).real
        x = np.sum(p_thru * np.exp(-1j * 2 * np.pi * N_WIN * np.arange(n) / n))
        amps[k] = 2.0 * np.abs(x) / n          # amplitude of the f-tone [W]
        print(f"  {freq/1e9:7.2f} GHz  |dP| = {amps[k]*1e3:.4e} mW")
    return amps


def f_3db(freqs: np.ndarray, resp_db: np.ndarray) -> float:
    """First -3 dB down-crossing (log-log interpolated), or nan."""
    for k in range(1, len(freqs)):
        if resp_db[k - 1] >= -3.0 > resp_db[k]:
            f0, f1 = np.log10(freqs[k - 1]), np.log10(freqs[k])
            d0, d1 = resp_db[k - 1], resp_db[k]
            return 10 ** (f0 + (-3.0 - d0) * (f1 - f0) / (d1 - d0))
    return float("nan")


def main(kappa2: float | None = None, v_ac: float = V_AC,
         f_start: float = F_START, f_stop: float = F_STOP,
         points: int = N_POINTS, r_series: float = R_SERIES) -> int:
    design = small_signal_bias(design_ring(0.0, kappa2=kappa2))
    f_bw = design["f_bw"]
    print(f"device: R = {RADIUS_UM} um, kappa^2 = {design['kappa2']:.3f} "
          f"(critical = {design['kappa2_crit']:.3f}) -> "
          f"Q_loaded = {design['q_loaded']:.0f}, "
          f"tau_photon = {design['tau']/2*1e12:.2f} ps, "
          f"linewidth = {design['fwhm_pm']:.0f} pm")
    print(f"bias: laser blue-detuned {design['detune_pm']:.0f} pm to the "
          f"max-slope point; Cj = {design['cj']*1e15:.1f} fF, "
          f"R_series = {r_series:g} ohm, V_ac = {v_ac*1e3:g} mV")
    print(f"analytic photon-lifetime f_3dB = {f_bw/1e9:.1f} GHz "
          f"(= 1/(2*pi*tau_photon))")

    freqs = np.logspace(np.log10(f_start), np.log10(f_stop), points)
    print(f"sweeping {points} tones, {f_start/1e9:g}-{f_stop/1e9:g} GHz:")
    sim = measure_response(design, freqs, v_ac, r_series)
    ana = analytic_eo_response(freqs, design, r_series)

    sim_db = 20 * np.log10(sim / sim[0])
    ana_db = 20 * np.log10(ana / ana[0])
    err = float(np.max(np.abs(sim_db - ana_db)))
    f3_sim = f_3db(freqs, sim_db)
    # analytic corner on a fine grid (independent of the sampled sweep)
    fg = np.logspace(np.log10(f_start), np.log10(f_stop), 4000)
    ag_db = 20 * np.log10(analytic_eo_response(fg, design, r_series)
                          / analytic_eo_response(freqs[:1], design, r_series)[0])
    f3_ana = f_3db(fg, ag_db)
    peak_db = float(sim_db.max())

    print(f"response: peaking = +{peak_db:.2f} dB, "
          f"sim f_3dB = {f3_sim/1e9:.1f} GHz, analytic f_3dB = {f3_ana/1e9:.1f} GHz")
    print(f"max |sim - analytic CMT| = {err:.3f} dB")

    # --- self-checks --------------------------------------------------------
    assert err < 0.5, f"simulated EO response deviates from CMT: {err:.3f} dB"
    assert np.isfinite(f3_sim), "no -3 dB corner found in the swept band"
    # the ideal-drive corner is the photon-lifetime bandwidth; a series R only
    # lowers it (adds the RC pole), so f_3dB <= f_bw within tolerance either way
    assert f3_sim <= 1.2 * f_bw, "EO bandwidth exceeds the photon-lifetime limit"
    if r_series == 0:
        assert abs(f3_sim - f_bw) < 0.2 * f_bw, \
            f"cavity f_3dB {f3_sim/1e9:.1f} GHz != photon-lifetime {f_bw/1e9:.1f} GHz"
    assert sim_db[-1] < -6.0, "response never rolls off — not low-pass"
    print("ALL TESTBENCH CHECKS PASSED")

    # --- plot ---------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 9),
                                   gridspec_kw={"height_ratios": [1, 1.3]})

    # (1) where we bias: the through-port Lorentzian vs resonance detuning
    x = np.linspace(-4, 4, 601)                 # detuning in HWHM units
    tk2 = design["tk2"]
    T = ((1 - tk2) ** 2 + x ** 2) / (1 + x ** 2)
    x_bias = 1 / np.sqrt(3.0)                    # max-slope point
    T_bias = ((1 - tk2) ** 2 + x_bias ** 2) / (1 + x_bias ** 2)
    ax0.plot(x, T, c="tab:blue")
    ax0.plot([x_bias], [T_bias], "o", c="tab:red",
             label=f"laser bias (max slope, -{design['detune_pm']:.0f} pm)")
    ax0.axhline(1.0, c="gray", lw=0.5, ls=":")
    ax0.set_xlabel("laser detuning from resonance  [HWHM]")
    ax0.set_ylabel("through-port |H|²")
    ax0.set_title("Ring modulator bias point — steepest slope of the resonance")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # (2) the EO frequency response: simulated tones vs analytic CMT
    ax1.semilogx(fg, ag_db, "k:", lw=1.2, label="analytic CMT (linearised)")
    ax1.semilogx(freqs, sim_db, "o", c="tab:orange", ms=5,
                 label="ring_mod.va in circulax (lock-in)")
    ax1.axhline(-3.0, c="gray", lw=0.6, ls="--")
    ax1.axvline(f3_sim, c="tab:red", lw=0.8, ls="--",
                label=f"f_3dB = {f3_sim/1e9:.1f} GHz")
    rc_tag = f",  R_series = {r_series:g} Ω" if r_series > 0 else " (ideal drive)"
    ax1.set_xlabel("modulation frequency  [Hz]")
    ax1.set_ylabel("normalised EO response  [dB]")
    ax1.set_title("Small-signal electro-optic frequency response, |dP_thru/dV|"
                  + rc_tag)
    ax1.set_ylim(min(-20, sim_db.min() - 2), max(2.0, peak_db + 1))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kappa2", type=float, default=None,
                    help="override bus power coupling (default 0.10)")
    ap.add_argument("--vac", type=float, default=V_AC,
                    help=f"small-signal drive amplitude [V] (default {V_AC:g})")
    ap.add_argument("--fstart", type=float, default=F_START,
                    help=f"sweep start [Hz] (default {F_START:g})")
    ap.add_argument("--fstop", type=float, default=F_STOP,
                    help=f"sweep stop [Hz] (default {F_STOP:g})")
    ap.add_argument("--points", type=int, default=N_POINTS,
                    help=f"log-spaced frequencies (default {N_POINTS})")
    ap.add_argument("--rs", type=float, default=R_SERIES,
                    help="electrode series resistance [ohm]; adds the RC pole "
                         f"(default {R_SERIES:g} = ideal drive)")
    args = ap.parse_args()
    raise SystemExit(main(kappa2=args.kappa2, v_ac=args.vac,
                          f_start=args.fstart, f_stop=args.fstop,
                          points=args.points, r_series=args.rs))
