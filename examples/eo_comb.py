#!/usr/bin/env python3
"""Ring-modulator electro-optic frequency comb, built from optical sub-components.

Drive the depletion electrode of a silicon **microring modulator** with a strong
single RF tone and the through port sprouts a *frequency comb* — a fan of optical
lines spaced by the drive frequency ``f_RF``. This bench builds the ring the way
a photonics engineer draws it: from three physical **sub-components** wired into
a loop, rather than one monolithic coupled-mode equation. The pieces are

  * **directional coupler** — the point coupler between the bus and the ring:
    it couples the input field into the circulating cavity mode (at the external
    rate ``2/tau_e`` set by the power coupling ``kappa^2``) and taps the through
    port ``s_out = s_in + j*a``;
  * **EO phase shifter** — the depletion phase shifter *inside* the ring: the
    electrode voltage shifts the resonance (``45 pm/V``), i.e. it sets the cavity
    detuning ``delta(V)``. This is the electro-optic drive;
  * **cavity mode** — the ring loop itself: the circulating field's energy
    storage (``d/dt``) plus its round-trip propagation loss (rate ``1/tau_i``).
    This is where the **photon lifetime** lives.

Their Kirchhoff sum on the shared cavity node ``a`` is exactly the ring CMT

    dA/dt = (-1/tau + j*delta(t)) A + j*kappa^2 s_in ,   s_out = s_in + j*A,

with ``1/tau = 1/tau_i + 1/tau_e``. So the monolithic ``models/optical_field/
ring_mod.va`` used by ``ring_mod_sky130.py`` / ``ring_eo_response.py`` is
reproduced here as a block diagram — and the bench asserts the two agree to
machine precision (part 1 below). Building it from parts is not just cosmetic:
the field primitives are memoryless (an optical S-matrix has no delay), so a
literal coupler + waveguide *feedback loop* would be algebraic and would model
only the adiabatic, lifetime-free ring; localising the round trip into the
cavity-mode storage element is what keeps the photon-lifetime dynamics.

How the comb forms. The EO phase shifter moves the resonance,
``lambda_res(V) = lambda_res + 45 pm/V * V``, so a tone
``V(t) = V_ac sin(2*pi*f_RF t)`` sweeps it back and forth across a CW laser
parked on its slope. The cavity is a *linear time-varying* system (the drive
enters only through ``delta(t)``), so its periodic response is a comb at every
harmonic ``n*f_RF``. Unlike a plain phase modulator's flat Jacobi-Anger (Bessel)
comb, this comb is **shaped by the cavity**: the finite photon lifetime
low-passes the modulation, so sidebands beyond the cavity linewidth roll off.
Two regimes fall out:

  * **adiabatic** (f_RF << f_cav): the ring follows the drive quasi-statically,
    so the through-port field just traces the swept Lorentzian ``s(delta(t))`` —
    the comb is the Fourier series of that periodic waveform, and its bandwidth
    grows in step with ``f_RF``;
  * **photon-lifetime limited** (f_RF ~ f_cav): the cavity can no longer charge
    and discharge within a period, the waveform lags and rounds, and the comb
    bandwidth **saturates** near the cavity bandwidth ``f_cav = 1/(2*pi*(tau/2))``
    (~44 GHz here) — the photon-lifetime rolloff ``ring_eo_response.py`` measures
    small-signal.

The whole thing is one JAX system (CW laser -> coupler/phase-shifter/cavity ring
-> through field) solved by Newton DC + fixed-step BDF2, then a leak-free
rectangular FFT reads the teeth. Every line is pinned against an **independent
integration of the CMT ODE** (scipy, high accuracy).

Self-checks (all asserted):
  1. sub-components == ring_mod.va — the coupler+phase-shifter+cavity ring
     reproduces the monolithic Verilog-A model to machine precision
  2. golden CMT   — each comb line matches an independent high-accuracy
                    integration of the CMT ODE (scipy) to <1 dB
  3. line spacing — all comb power sits on the f_RF harmonic grid
  4. passivity    — the ring is passive and lossy, so the *cycle-averaged*
                    through-port power is <= the laser power (it may overshoot
                    instantaneously as the overcoupled cavity dumps stored energy)
  5. cavity limit — sweeping f_RF, the comb bandwidth tracks the adiabatic
                    (Fourier-series) growth at low f_RF but saturates near f_cav
                    once the photon lifetime can't follow

    .venv-circulax/bin/python examples/eo_comb.py
    .venv-circulax/bin/python examples/eo_comb.py --frf 20e9 --swing 2.5
    .venv-circulax/bin/python examples/eo_comb.py --kappa2 0.30   # broader, faster ring

        -> out/eo_comb.png
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
from scipy.integrate import solve_ivp

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component, source
from circulax.solvers.transient import BDF2VectorizedTransientSolver

from photonflux import cx

# the physical device + the monolithic ring, shared with the eye-diagram bench
from ring_mod_sky130 import (
    C0,
    DL_DV_PM,
    P_LASER,
    RADIUS_UM,
    WAVELENGTH_NM,
    design_ring,
    ring_settings,
)

# --- testbench knobs (also settable from the command line) -------------------
F_RF = 10e9           # RF drive tone / comb line spacing [Hz]
SWING_HWHM = 1.6      # resonance sweep amplitude in half-linewidths (sets V_ac)
DT = 0.25e-12         # sample / solver step [s]
NPT = 8000            # samples (2 ns): f_RF on a bin, integer periods -> leak-free
N_SETTLE_TAU = 25.0   # photon lifetimes to settle before the FFT window

OUT = Path(__file__).resolve().parents[1] / "out" / "eo_comb.png"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ===========================================================================
# Device + bias point
# ===========================================================================
# volts -> detuning-rate conversion: delta1 = (2*pi*c/lambda^2) * dl_dv * V_ac,
# and one half-linewidth (HWHM) of detuning is 1/tau, so V_ac for a given swing
# in HWHM is swing/(tau * VOLT_TO_DELTA).
VOLT_TO_DELTA = 2 * np.pi * C0 / (WAVELENGTH_NM * 1e-9) ** 2 * DL_DV_PM * 1e-12


def comb_device(kappa2: float | None = None) -> dict:
    """The CMT ring parked at its maximum-slope bias, split into CMT rates.

    ``tau`` (loaded) and ``tk2`` come from the physical device (``design_ring``);
    the coupler/cavity split follows ``ring_mod.va``'s derivation:
    ``1/tau_e = tk2/(2*tau)`` (bus coupling), ``1/tau_i = 1/tau - 1/tau_e``
    (round-trip loss), and the drive/output rate ``kappa^2 = tk2/tau``.
    """
    d = dict(design_ring(0.0, kappa2=kappa2))
    hwhm_pm = d["fwhm_pm"] / 2
    detune_pm = hwhm_pm / np.sqrt(3.0)              # Lorentzian max-slope point
    d["detune_pm"] = detune_pm
    d["lambda_light_nm"] = WAVELENGTH_NM - detune_pm * 1e-3
    lam_l, lam_res = d["lambda_light_nm"] * 1e-9, WAVELENGTH_NM * 1e-9
    d["delta0"] = 2 * np.pi * C0 * (1 / lam_l - 1 / lam_res)
    d["krate"] = d["tk2"] / d["tau"]               # kappa^2: drive/output rate
    d["inv_tau_e"] = d["tk2"] / (2 * d["tau"])     # bus coupling rate
    d["inv_tau_i"] = 1.0 / d["tau"] - d["inv_tau_e"]  # round-trip loss rate
    return d


def vac_for_swing(design: dict, swing_hwhm: float) -> float:
    return swing_hwhm / design["tau"] / VOLT_TO_DELTA


# ===========================================================================
# The ring, built from optical sub-components (temporal coupled-mode blocks)
# ===========================================================================
# The three blocks share one internal complex node — the circulating cavity
# field ``a`` — and their Kirchhoff sum there IS the ring CMT (see module
# docstring). Each is an ordinary circulax coherent-field component.
def directional_coupler(inv_tau_e: float, krate: float):
    """Bus <-> ring point coupler.

    Couples the input field into the cavity mode at the external (amplitude)
    rate ``1/tau_e`` and taps the through port ``s_out = s_in + j*a``. Contributes
    ``a/tau_e - j*kappa^2 s_in`` to the cavity node (coupling loss + drive) and
    draws nothing from the bus input (an ideal directional tap)."""

    @component(ports=("sin", "sout", "a"), states=("i_out",))
    def Coupler(signals: Signals, s: States) -> tuple[dict, dict]:
        return {
            "sin": 0.0,                                    # ideal input tap
            "a": inv_tau_e * signals.a - 1j * krate * signals.sin,
            "sout": s.i_out,
            "i_out": signals.sout - (signals.sin + 1j * signals.a),
        }, {}

    return Coupler


def eo_phase_shifter(design: dict, cj: float = 0.0):
    """Depletion phase shifter in the ring: electrode voltage -> cavity detuning.

    ``lambda_res(V) = lambda_res + dl_dv_pm*V`` moves the resonance, so the
    electrode sets the CMT detuning ``delta(V) = 2*pi*c*(1/lambda_light -
    1/lambda_res(V))`` — the same electro-optic map as ``ring_mod.va``. It
    contributes ``-j*delta(V)*a`` to the cavity node. ``cj`` is the junction
    capacitance a real driver would load (0 = ideal drive; harmless here since
    the drive is an ideal voltage source)."""
    lam_l = design["lambda_light_nm"] * 1e-9
    lam_res0 = WAVELENGTH_NM * 1e-9

    @component(ports=("a", "vp", "vn"))
    def PhaseShifter(signals: Signals, s: States) -> tuple[dict, dict]:
        v = (signals.vp - signals.vn).real
        lam_res = lam_res0 + DL_DV_PM * 1e-12 * v
        delta = 2 * np.pi * C0 * (1.0 / lam_l - 1.0 / lam_res)
        f = {"a": -1j * delta * signals.a, "vp": 0.0, "vn": 0.0}
        q_el = cj * (signals.vp - signals.vn)
        return f, {"vp": q_el, "vn": -q_el}

    return PhaseShifter


def cavity_mode(inv_tau_i: float):
    """The ring loop: circulating-field energy storage + round-trip loss.

    Contributes ``dA/dt + a/tau_i`` to the cavity node — the ``d/dt`` (photon
    storage, i.e. the photon lifetime) plus the intrinsic round-trip loss."""

    @component(ports=("a",))
    def Cavity(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"a": inv_tau_i * signals.a}, {"a": signals.a}

    return Cavity


def sine_source():
    """Ideal RF voltage source V = v_ac*sin(2*pi*freq*t) (starts at 0)."""

    @source(ports=("p1", "p2"), states=("i_src",))
    def Sine(signals: Signals, s: States, t: float,
             v_ac: float = 1.0, freq: float = F_RF) -> tuple[dict, dict]:
        v = v_ac * jnp.sin(2.0 * jnp.pi * freq * t)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}

    return Sine


def field_terminator():
    """Single complex-node termination that draws nothing (open circuit)."""

    @component(ports=("c",))
    def FieldTerminator(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"c": 0.0}, {}

    return FieldTerminator


# ===========================================================================
# Circuits: the sub-component ring, and the monolithic ring_mod.va twin
# ===========================================================================
def build(design: dict):
    """CW laser -> [coupler | EO phase shifter | cavity mode] ring -> through."""
    inst = {
        "GND": {"component": "ground"},
        "LAS": {"component": "laser",
                "settings": {"wavelength_nm": design["lambda_light_nm"],
                             "power": P_LASER}},
        "CPL": {"component": "coupler"},        # bus <-> ring coupler + tap
        "PS": {"component": "phaseshifter"},    # EO detuning drive
        "CAV": {"component": "cavity"},         # ring loop storage + loss
        "VS": {"component": "sine"},
        "TO": {"component": "term"},
    }
    conn = {
        "LAS,p1": "CPL,sin",
        "CPL,sout": "TO,c",
        "CPL,a": ("PS,a", "CAV,a"),             # shared circulating-field node
        "VS,p1": "PS,vp",
        "GND,p1": ("LAS,p2", "PS,vn", "VS,p2"),
    }
    net = {"instances": inst, "connections": conn, "ports": {"pout": "CPL,sout"}}
    models = {"ground": lambda: 0, "laser": cx.cw_laser(),
              "coupler": directional_coupler(design["inv_tau_e"], design["krate"]),
              "phaseshifter": eo_phase_shifter(design, cj=design["cj"]),
              "cavity": cavity_mode(design["inv_tau_i"]),
              "sine": sine_source(), "term": field_terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=400)


def build_monolithic(design: dict):
    """The same ring as the single ``ring_mod.va`` block — the reference twin."""
    inst = {
        "GND": {"component": "ground"},
        "LAS": {"component": "laser",
                "settings": {"wavelength_nm": design["lambda_light_nm"],
                             "power": P_LASER}},
        "TAP": {"component": "f2ri"},
        "RING": {"component": "ring", "settings": ring_settings(design)},
        "JOIN": {"component": "ri2f"},
        "VS": {"component": "sine"},
        "TO": {"component": "term"},
    }
    conn = {
        "LAS,p1": "TAP,c",
        "TAP,re": "RING,in_re", "TAP,im": "RING,in_im",
        "RING,out_re": "JOIN,re", "RING,out_im": "JOIN,im",
        "JOIN,c": "TO,c",
        "VS,p1": "RING,vp",
        "GND,p1": ("LAS,p2", "RING,vn", "RING,gnd", "VS,p2"),
    }
    net = {"instances": inst, "connections": conn, "ports": {"pout": "JOIN,c"}}
    models = {"ground": lambda: 0, "laser": cx.cw_laser(),
              "f2ri": cx.field_to_ri(), "ring": cx.va("ring_mod"),
              "ri2f": cx.ri_to_field(), "sine": sine_source(),
              "term": field_terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=400)


def settle_periods(design: dict, freq: float) -> int:
    """Whole drive periods (>= N_SETTLE_TAU photon lifetimes) before the FFT."""
    return max(1, int(np.ceil(N_SETTLE_TAU * design["tau"] * freq)))


def run(c, design: dict, v_ac: float, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step BDF2; returns (t_abs, E_thru(t)) over the NPT-sample window."""
    params = {"VS.v_ac": float(v_ac), "VS.freq": float(freq)}
    t0_win = settle_periods(design, freq) / freq
    ts = t0_win + jnp.arange(NPT) * DT
    t_max = t0_win + NPT * DT
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=DT, y0=c.dc(params=params), params=params,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=int(t_max / DT) + 20, throw=False,
        stepsize_controller=diffrax.ConstantStepSize())
    assert sol.result == diffrax.RESULTS.successful, \
        f"transient failed at f={freq/1e9:g} GHz: {sol.result}"
    return np.asarray(sol.ts)[:NPT], np.asarray(c.port(sol.ys, "pout"))[:NPT]


# ===========================================================================
# Golden reference: integrate the same CMT ODE independently (scipy)
# ===========================================================================
def delta_of_v(design: dict, v: np.ndarray | float):
    """delta(V) = 2*pi*c*(1/lambda_light - 1/lambda_res(V)) — the ring detuning."""
    lam_l = design["lambda_light_nm"] * 1e-9
    lam_res = WAVELENGTH_NM * 1e-9 + DL_DV_PM * 1e-12 * v
    return 2 * np.pi * C0 * (1.0 / lam_l - 1.0 / lam_res)


def golden(design: dict, v_ac: float, freq: float,
           t_grid: np.ndarray) -> np.ndarray:
    """Through-port field s_out(t) from an independent RK45 CMT integration."""
    tau, krate = design["tau"], design["krate"]
    s_in = np.sqrt(P_LASER)

    def rhs(t, y):
        a = y[0] + 1j * y[1]
        v = v_ac * np.sin(2 * np.pi * freq * t)
        da = (-1.0 / tau + 1j * delta_of_v(design, v)) * a + 1j * krate * s_in
        return [da.real, da.imag]

    a0 = 1j * krate * s_in / (1.0 / tau - 1j * design["delta0"])   # V=0 steady
    sol = solve_ivp(rhs, [0.0, float(t_grid[-1])], [a0.real, a0.imag],
                    t_eval=t_grid, rtol=1e-11, atol=1e-14, max_step=DT)
    a = sol.y[0] + 1j * sol.y[1]
    return s_in + 1j * a


def quasi_static(design: dict, v_ac: float, freq: float,
                 t: np.ndarray) -> np.ndarray:
    """Adiabatic (instantaneous-steady-state) through field: the swept Lorentzian.

    A_qs(t) = j*kappa^2 s_in / (1/tau - j*delta(t)); the f_RF -> 0 limit of the
    ODE, in which the ring tracks the resonance with no lag."""
    tau, krate = design["tau"], design["krate"]
    s_in = np.sqrt(P_LASER)
    v = v_ac * np.sin(2 * np.pi * freq * t)
    a_qs = 1j * krate * s_in / (1.0 / tau - 1j * delta_of_v(design, v))
    return s_in + 1j * a_qs


# ===========================================================================
# Analysis helpers
# ===========================================================================
def spectrum(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.fft.fftfreq(len(e), d=DT), np.fft.fft(e) / len(e)


def line(f: np.ndarray, a: np.ndarray, f0: float) -> complex:
    i = int(np.argmin(np.abs(f - f0)))
    assert abs(f[i] - f0) < 1e3, f"{f0/1e9} GHz is not on the FFT bin grid"
    return complex(a[i])


def comb_lines(f: np.ndarray, a: np.ndarray, freq: float,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Orders n and line powers P_n at each harmonic n*freq within Nyquist."""
    nyq = 1.0 / (2.0 * DT)
    n_max = int((nyq - freq) / freq)
    ns = np.arange(-n_max, n_max + 1)
    p = np.array([abs(line(f, a, n * freq)) ** 2 for n in ns])
    return ns, p


def rms_order(ns: np.ndarray, p: np.ndarray) -> float:
    """RMS comb order sqrt(sum n^2 P_n / sum P_n) — comb bandwidth / f_RF."""
    return float(np.sqrt(np.sum(ns ** 2 * p) / np.sum(p)))


def dbc(p, ref: float = P_LASER):
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-300) / ref)


# ===========================================================================
# Main
# ===========================================================================
def main(frf: float = F_RF, swing: float = SWING_HWHM,
         kappa2: float | None = None) -> int:
    design = comb_device(kappa2)
    v_ac = vac_for_swing(design, swing)
    f_cav = design["f_bw"]
    print(f"ring EO comb (coupler + EO phase shifter + cavity): R = {RADIUS_UM} um, "
          f"k^2 = {design['kappa2']:.3f}, tau = {design['tau']*1e12:.2f} ps, "
          f"linewidth = {design['fwhm_pm']:.0f} pm, f_cav = {f_cav/1e9:.1f} GHz")
    print(f"bias: laser blue-detuned {design['detune_pm']:.0f} pm to max slope; "
          f"drive f_RF = {frf/1e9:.0f} GHz, V_ac = {v_ac:.2f} V "
          f"(resonance swings +-{swing:.1f} HWHM = +-{DL_DV_PM*v_ac:.0f} pm)")

    c = build(design)

    # ---- part 1: sub-components == ring_mod.va, and both == the CMT golden --
    t, e = run(c, design, v_ac, frf)
    f, a = spectrum(e)
    ns, p_n = comb_lines(f, a, frf)

    _, e_mono = run(build_monolithic(design), design, v_ac, frf)   # VA twin
    err_mono = float(np.max(np.abs(e - e_mono)))
    n_teeth = int(np.sum(p_n > 10 ** (-2.0) * p_n.max()))   # within 20 dB of peak
    check("sub-components == ring_mod.va  (block diagram == VA model)",
          err_mono < 1e-9,
          f"max |coupler+PS+cavity - ring_mod.va| = {err_mono:.1e} "
          f"(~{n_teeth} teeth within 20 dB of the peak)")

    # each tooth also matches an *independent* high-accuracy CMT integration;
    # residual is the fixed-step BDF2 vs scipy RK45 difference on the weak lines
    eg = golden(design, v_ac, frf, t)                              # CMT ODE
    _, ag = spectrum(eg)
    _, p_g = comb_lines(f, ag, frf)
    big = p_g > 10 ** (-6.0) * P_LASER          # lines above -60 dBc
    err_db = float(np.max(np.abs(dbc(p_n[big]) - dbc(p_g[big]))))
    check("comb == independent CMT ODE  (line-by-line)", err_db < 1.0,
          f"{int(big.sum())} lines match the CMT integration to {err_db:.3f} dB")

    # ---- part 2: line spacing = f_RF exactly ------------------------------
    total_power = float(np.mean(np.abs(e) ** 2))
    off_grid = 1.0 - float(p_n.sum()) / total_power
    check("line spacing = f_RF (teeth only on drive harmonics)",
          abs(off_grid) < 1e-6,
          f"{off_grid*100:.2e}% of the power sits off the {frf/1e9:.0f} GHz grid")

    # ---- part 3: cycle-averaged passivity ---------------------------------
    # overcoupled ring dumps stored energy, so |E_thru|^2 can transiently exceed
    # P_laser; but a passive, lossy cavity cannot on average -> mean <= P_laser.
    peak = float(np.max(np.abs(e) ** 2))
    check("passivity  (cycle-averaged power <= laser)",
          total_power <= P_LASER * (1 + 1e-6),
          f"mean P_thru = {total_power/P_LASER:.4f} P_laser "
          f"(instantaneous peak {peak/P_LASER:.2f} P_laser — cavity overshoot)")

    # ---- part 4: cavity photon-lifetime limit on the comb bandwidth -------
    # Sweep f_RF. The adiabatic comb has a FIXED set of harmonic orders (the
    # Fourier series of the swept Lorentzian), so its bandwidth = rms_order*f_RF
    # grows linearly with f_RF. The real ring's photon lifetime rolls off the
    # high orders once f_RF ~ f_cav, so its bandwidth saturates.
    frf_sweep = np.arange(2e9, 80e9 + 1, 6e9)      # all multiples of the bin
    bw_sim, bw_adia = [], []
    # adiabatic rms order is f_RF-independent: measure it once, deep in the
    # adiabatic regime, from the quasi-static waveform.
    t_lo, _ = run(c, design, v_ac, 2e9)
    f_qs, a_qs = spectrum(quasi_static(design, v_ac, 2e9, t_lo))
    ns_qs, p_qs = comb_lines(f_qs, a_qs, 2e9)
    rms_adia = rms_order(ns_qs, p_qs)
    for fr in frf_sweep:
        _, ef = run(c, design, v_ac, float(fr))
        ff, af = spectrum(ef)
        nf, pf = comb_lines(ff, af, float(fr))
        bw_sim.append(rms_order(nf, pf) * fr)
        bw_adia.append(rms_adia * fr)
    bw_sim, bw_adia = np.array(bw_sim), np.array(bw_adia)
    lo_ratio = bw_sim[0] / bw_adia[0]              # near 1: follows adiabatic
    hi_ratio = bw_sim[-1] / bw_adia[-1]            # < 1: cavity-limited
    check("cavity limit  (photon lifetime caps the comb)",
          lo_ratio > 0.9 and hi_ratio < 0.7,
          f"comb BW / adiabatic = {lo_ratio:.2f} at {frf_sweep[0]/1e9:.0f} GHz "
          f"-> {hi_ratio:.2f} at {frf_sweep[-1]/1e9:.0f} GHz (f_cav = "
          f"{f_cav/1e9:.0f} GHz)")

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Ring-modulator electro-optic frequency comb "
                 "(directional coupler + EO phase shifter + cavity mode)",
                 fontsize=13)

    # (A) the driven resonance: laser on the slope, resonance swept +-swing
    axA = ax[0, 0]
    x = np.linspace(-5, 5, 601)                    # detuning in HWHM units
    tk2 = design["tk2"]
    T = ((1 - tk2) ** 2 + x ** 2) / (1 + x ** 2)
    x_bias = design["delta0"] * design["tau"]      # bias in HWHM
    axA.plot(x, T, c="tab:blue")
    axA.axvspan(x_bias - swing, x_bias + swing, color="tab:orange", alpha=0.18,
                label=f"resonance sweep (±{swing:.1f} HWHM)")
    axA.plot([x_bias], [((1 - tk2) ** 2 + x_bias ** 2) / (1 + x_bias ** 2)], "o",
             c="tab:red", label="laser bias (max slope)")
    axA.set(xlabel="laser–resonance detuning  [HWHM]",
            ylabel="through-port |H|²",
            title="the EO phase shifter sweeps the resonance across the laser")
    axA.legend(fontsize=8)
    axA.grid(alpha=0.3)

    # (B) the comb spectrum: sub-component ring vs the golden CMT integration
    axB = ax[0, 1]
    sel = p_n > 1e-9 * P_LASER
    axB.vlines(ns[sel] * frf / 1e9, -90, dbc(p_n[sel]), color="#2c7fb8", lw=2.0,
               label="coupler+PS+cavity ring")
    axB.plot(ns[big] * frf / 1e9, dbc(p_g[big]), "x", color="#d95f02", ms=6,
             label="independent CMT ODE")
    axB.axvline(f_cav / 1e9, color="0.6", ls=":", lw=0.9)
    axB.axvline(-f_cav / 1e9, color="0.6", ls=":", lw=0.9,
                label=f"±f_cav = {f_cav/1e9:.0f} GHz")
    axB.set(ylim=(-70, 3), xlabel="optical frequency offset  [GHz]",
            ylabel="line power  [dBc]",
            title=f"comb at f_RF = {frf/1e9:.0f} GHz, cavity-shaped")
    axB.legend(fontsize=8, loc="lower center")
    axB.grid(alpha=0.3)

    # (C) time-domain through-port power: cavity lag vs the adiabatic sweep
    axC = ax[1, 0]
    n_show = int(round(2.0 / (frf * DT)))          # two RF periods
    tt = (t[:n_show] - t[0]) * 1e12
    p_qs_t = np.abs(quasi_static(design, v_ac, frf, t[:n_show])) ** 2
    axC.plot(tt, np.abs(e[:n_show]) ** 2 / P_LASER, color="#2c7fb8", lw=1.6,
             label="cavity mode (with photon lifetime)")
    axC.plot(tt, p_qs_t / P_LASER, "--", color="tab:green", lw=1.4,
             label="adiabatic (instantaneous Lorentzian)")
    axC.axhline(1.0, color="0.7", lw=0.6, ls=":")
    axC.set(xlabel="time  [ps]", ylabel="through-port power  / P_laser",
            title="cavity lag: the ring can't quite follow the sweep")
    axC.legend(fontsize=8)
    axC.grid(alpha=0.3)

    # (D) comb bandwidth vs f_RF: adiabatic growth then photon-lifetime clamp
    axD = ax[1, 1]
    axD.plot(frf_sweep / 1e9, bw_adia / 1e9, "--", color="tab:green", lw=1.3,
             label=r"adiabatic  ($\propto f_{RF}$)")
    axD.plot(frf_sweep / 1e9, bw_sim / 1e9, "o-", color="#2c7fb8", ms=4,
             label="cavity-limited (sub-component ring)")
    axD.axhline(f_cav / 1e9, color="0.6", ls=":", lw=0.9,
                label=f"f_cav = {f_cav/1e9:.0f} GHz")
    axD.set(xlabel="drive frequency f_RF  [GHz]",
            ylabel="RMS comb bandwidth  [GHz]",
            title="photon lifetime caps the comb bandwidth")
    axD.legend(fontsize=8)
    axD.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    if all(ok for _, ok, _ in CHECKS):
        print("ALL EO-COMB CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED:", [n for n, ok, _ in CHECKS if not ok])
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frf", type=float, default=F_RF,
                    help=f"RF drive tone / comb spacing [Hz] (default {F_RF:g})")
    ap.add_argument("--swing", type=float, default=SWING_HWHM,
                    help="resonance sweep amplitude in half-linewidths "
                         f"(default {SWING_HWHM})")
    ap.add_argument("--kappa2", type=float, default=None,
                    help="override bus power coupling (default 0.10)")
    args = ap.parse_args()
    raise SystemExit(main(frf=args.frf, swing=args.swing, kappa2=args.kappa2))
