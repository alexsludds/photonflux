#!/usr/bin/env python3
"""Electro-optic frequency comb: a CW laser + one phase modulator = a Bessel comb.

Drive a single electro-optic **phase modulator** hard with a pure RF tone and
its output is a *frequency comb* — a fan of equally spaced optical lines. There
is no nonlinearity and no cavity; the comb is the exact Jacobi-Anger expansion
of a sinusoidally modulated phase. A CW field ``E_in = sqrt(P)`` through a phase
modulator driven at ``V(t) = V_ac sin(2*pi*f_RF t)`` comes out as

    E_out(t) = sqrt(il) * sqrt(P) * exp(j*beta*sin(2*pi*f_RF t)),
        modulation index  beta = pi * V_ac / Vpi,

and Jacobi-Anger, ``exp(j*beta*sin x) = sum_n J_n(beta) e^{j n x}``, turns that
into a line at every harmonic ``n*f_RF`` of the drive:

    line at n*f_RF:   E_n = sqrt(il*P) * J_n(beta),   P_n = il * P * J_n(beta)^2.

So the phase modulator is a comb generator whose teeth are Bessel functions of
the drive strength. Everything worth knowing about the comb falls straight out
of Bessel identities and is pinned here line-by-line:

  * **line spacing = f_RF exactly** — the teeth land only on the drive harmonics;
  * **energy conservation** — ``sum_n J_n(beta)^2 = 1``, so a lossless phase
    modulator only *redistributes* the carrier's photons across the comb;
    ``mean|E_out|^2 = il*P`` to machine precision (phase modulation is unitary);
  * **carrier suppression** — the centre tooth is ``J_0(beta)^2``, which *nulls*
    at the Bessel zeros ``beta = 2.4048, 5.5201, ...``. Bias the drive to the
    first zero and the optical carrier vanishes (>40 dB down) while the sidebands
    carry all the power — the textbook carrier-suppressed comb;
  * **comb broadening** — driving harder widens the comb. The RMS comb width is
    exactly ``sum_n n^2 J_n(beta)^2 = beta^2/2`` (a Bessel identity), so the
    comb's RMS bandwidth is ``f_RF * beta/sqrt(2)`` — it grows linearly with the
    drive amplitude, and the number of usable teeth grows like ``2*beta``.

The device under test is an ideal EO phase modulator built inline as a circulax
component (a unidirectional field transfer ``E_out = t*E_in`` with
``t = sqrt(il)*exp(j*pi*(V+vbias)/Vpi)`` — the phase twin of the intensity
``cx.mzm()``; a lossless all-pass cannot go through the scattering ``s_to_y``
path, so it is written as a driven field source like ``cx.cw_laser``). It is
driven by a real voltage electrode from an ideal RF sine source, so the comb is
genuinely *electro-optic*: ``beta`` is set by the electrode voltage over Vpi.
Optional ``--rs`` puts a source resistance in series with the electrode
capacitance, adding the electrode RC pole so the effective drive — and the comb
width — rolls off at high f_RF (the real-modulator bandwidth limit).

The whole thing is one JAX system: CW laser + phase modulator solved by a single
Newton DC and a fixed-step BDF2 transient, then a leak-free rectangular FFT
(2 ns = 40 exact periods of the 20 GHz drive) reads the teeth. Every tooth is
checked against ``il*P*J_n(beta)^2`` and against a numpy "golden" evaluation of
the closed-form map — the VA/OSDI-free coherent-field pipeline adds nothing.

    .venv-circulax/bin/python examples/eo_comb.py
    .venv-circulax/bin/python examples/eo_comb.py --beta 3.8      # broader comb
    .venv-circulax/bin/python examples/eo_comb.py --frf 40e9      # 40 GHz teeth
    .venv-circulax/bin/python examples/eo_comb.py --rs 200        # electrode RC roll-off

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
from scipy.special import jv

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component, source
from circulax.solvers.transient import BDF2VectorizedTransientSolver

from _progress import transient_progress_meter
from photonflux import cx

# --- fixed bench conditions --------------------------------------------------
LAM0_NM = 1310.0     # laser wavelength [nm] (repo default)
P_LASER = 1e-3       # CW input optical power [W]
VPI = 3.0            # phase-modulator half-wave voltage [V]
IL_DB = 0.0          # modulator insertion loss [dB] (0 = ideal; loss only scales)
CEL = 50e-15         # electrode capacitance [F] (forms the RC pole with --rs)
F_RF = 20e9          # default RF drive tone / comb line spacing [Hz]
BETA0 = 2.4048256    # headline drive: first zero of J_0 -> carrier suppressed
BETA_HI = 5.0        # strong drive for the broad-comb panel

# FFT window: 2 ns = 40 exact periods of the 20 GHz drive, 0.5 GHz bins.
# Every harmonic n*F_RF lands on a bin -> a rectangular window is leak-free.
DT = 0.5e-12         # sample / solver step [s]
NPT = 4000           # samples (2 ns)
N_MAX = 45           # highest comb order gathered (n*F_RF <= Nyquist)

OUT = Path(__file__).resolve().parents[1] / "out" / "eo_comb.png"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def beta_to_vac(beta: float) -> float:
    """Electrode amplitude giving modulation index beta = pi*V_ac/Vpi."""
    return beta * VPI / np.pi


def rc_factor(freq: float, r_series: float) -> complex:
    """Electrode low-pass transfer H(f) = 1/(1 + j*2*pi*f*R*Cel)."""
    return 1.0 / (1.0 + 1j * 2.0 * np.pi * freq * r_series * CEL)


def eff_beta(beta: float, freq: float, r_series: float) -> float:
    """Effective modulation index after the electrode RC pole (|H| scales it)."""
    return beta * abs(rc_factor(freq, r_series))


# ===========================================================================
# Inline components: the RF electrode drive and the EO phase modulator
# ===========================================================================
def sine_source():
    """Ideal RF voltage source V = v_ac*sin(2*pi*freq*t) (starts at 0)."""

    @source(ports=("p1", "p2"), states=("i_src",))
    def Sine(signals: Signals, s: States, t: float,
             v_ac: float = beta_to_vac(BETA0), freq: float = F_RF) -> tuple[dict, dict]:
        v = v_ac * jnp.sin(2.0 * jnp.pi * freq * t)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}

    return Sine


def phase_modulator():
    """Ideal electro-optic phase modulator, phase twin of ``cx.mzm()``.

    Optical 2-port (``pin``/``pout``) with a differential drive electrode
    (``vp``/``vn``). The field transmission is pure phase,

        t = sqrt(il) * exp(j*pi*(V + vbias)/vpi),   V = Re(vp - vn),

    so ``E_out = t*E_in`` with ``|t| = sqrt(il)`` (unit magnitude when lossless).
    Written as a unidirectional driven field source — ``pin`` is tapped at high
    impedance (draws nothing, so the laser sets it) and ``pout`` is forced to
    ``t*pin`` by the branch state — because a lossless all-pass ``S = [[0,t],
    [t,0]]`` is singular under circulax's ``s_to_y`` (``t^2 - 1 -> 0`` at every
    zero crossing of the drive). ``cel`` loads the electrode with the junction
    capacitance so a series resistance forms the real electrode RC pole.
    """

    @component(ports=("pin", "pout", "vp", "vn"), states=("i_out",))
    def PhaseModulator(signals: Signals, s: States, vpi: float = VPI,
                       vbias: float = 0.0, il_db: float = IL_DB,
                       cel: float = 50e-15) -> tuple[dict, dict]:
        t_mag = 10.0 ** (-il_db / 20.0)                 # field transmission mag
        vd = (signals.vp - signals.vn).real             # drive read like cx.mzm
        t = t_mag * jnp.exp(1j * jnp.pi * (vd + vbias) / vpi)
        f = {"pin": 0.0, "pout": s.i_out, "vp": 0.0, "vn": 0.0,
             "i_out": signals.pout - t * signals.pin}
        q_el = cel * (signals.vp - signals.vn)          # electrode capacitance
        q = {"vp": q_el, "vn": -q_el}
        return f, q

    return PhaseModulator


def field_terminator():
    """Single complex-node termination that draws nothing (open circuit)."""

    @component(ports=("c",))
    def FieldTerminator(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"c": 0.0}, {}

    return FieldTerminator


# ===========================================================================
# Circuit: CW laser -> phase modulator (RF electrode) -> through field
# ===========================================================================
def build(r_series: float = 0.0):
    inst = {
        "GND": {"component": "ground"},
        "LAS": {"component": "laser",
                "settings": {"wavelength_nm": LAM0_NM, "power": P_LASER}},
        "VS": {"component": "sine"},
        "MOD": {"component": "pm"},
        "TO": {"component": "term"},
    }
    conn = {
        "GND,p1": ("LAS,p2", "VS,p2", "MOD,vn"),
        "LAS,p1": "MOD,pin",
        "MOD,pout": "TO,c",
    }
    if r_series > 0:
        inst["RS"] = {"component": "res", "settings": {"R": r_series}}
        conn["VS,p1"] = "RS,p1"
        conn["RS,p2"] = "MOD,vp"
    else:
        conn["VS,p1"] = "MOD,vp"
    net = {"instances": inst, "connections": conn, "ports": {"pout": "MOD,pout"}}
    from circulax.components.electronic import Resistor
    models = {"ground": lambda: 0, "laser": cx.cw_laser(), "sine": sine_source(),
              "pm": phase_modulator(), "term": field_terminator(), "res": Resistor}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def settle_time(freq: float, r_series: float) -> float:
    """Whole-period settle before the FFT window (>=18 electrode RC taus).

    With ``--rs`` the electrode drive rings up through the RC pole; capturing
    before it settles would leak an aperiodic transient into the rectangular
    FFT. Zero for the ideal drive (r_series = 0)."""
    if r_series <= 0:
        return 0.0
    n_periods = int(np.ceil(18.0 * r_series * CEL * freq))  # e^-18 ~ 2e-8 left
    return n_periods / freq


def run(c, beta: float, freq: float = F_RF, r_series: float = 0.0,
        progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step BDF2; returns (t_abs, E_out(t)) over the NPT-sample window."""
    params = {"VS.v_ac": float(beta_to_vac(beta)), "VS.freq": float(freq)}
    t0_win = settle_time(freq, r_series)
    ts = t0_win + jnp.arange(NPT) * DT
    t_max = t0_win + NPT * DT
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=DT, y0=c.dc(params=params), params=params,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=int(t_max / DT) + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
        progress_meter=transient_progress_meter(progress))
    assert sol.result == diffrax.RESULTS.successful, \
        f"transient failed at beta={beta:g}, f={freq/1e9:g} GHz: {sol.result}"
    return np.asarray(sol.ts)[:NPT], np.asarray(c.port(sol.ys, "pout"))[:NPT]


# ===========================================================================
# Analysis helpers
# ===========================================================================
def spectrum(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-window FFT — exact for lines on the 0.5 GHz bin grid."""
    return np.fft.fftfreq(len(e), d=DT), np.fft.fft(e) / len(e)


def line(f: np.ndarray, a: np.ndarray, f0: float) -> complex:
    i = int(np.argmin(np.abs(f - f0)))
    assert abs(f[i] - f0) < 1e3, f"{f0/1e9} GHz is not on the FFT bin grid"
    return complex(a[i])


def comb_lines(f: np.ndarray, a: np.ndarray, freq: float,
               n_max: int = N_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Orders n and line powers P_n at each harmonic n*freq within Nyquist."""
    nyq = 1.0 / (2.0 * DT)
    ns = np.array([n for n in range(-n_max, n_max + 1)
                   if abs(n * freq) < nyq - freq])
    p = np.array([abs(line(f, a, n * freq)) ** 2 for n in ns])
    return ns, p


def db_rel(p, ref: float = P_LASER):
    """Power relative to the input carrier [dB]."""
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-300) / ref)


def golden(t: np.ndarray, beta: float, freq: float, r_series: float = 0.0,
           il_db: float = IL_DB):
    """The exact closed-form map evaluated directly in numpy.

    The electrode RC scales the drive by |H| and delays it by arg(H), so the
    modulation index becomes ``beta*|H|`` at phase ``arg(H)``."""
    il = 10.0 ** (-il_db / 10.0)
    h = rc_factor(freq, r_series)
    return np.sqrt(il * P_LASER) * np.exp(
        1j * beta * abs(h) * np.sin(2 * np.pi * freq * t + np.angle(h)))


# ===========================================================================
# Main
# ===========================================================================
def main(beta: float = BETA0, frf: float = F_RF, r_series: float = 0.0) -> int:
    il = 10.0 ** (-IL_DB / 10.0)
    print(f"EO frequency comb: CW {P_LASER*1e3:.1f} mW @ {LAM0_NM:.0f} nm -> "
          f"phase modulator (Vpi = {VPI} V, IL = {IL_DB} dB)")
    print(f"drive: f_RF = {frf/1e9:.0f} GHz, headline beta = {beta:.4f} "
          f"(V_ac = {beta_to_vac(beta):.3f} V), R_series = {r_series:g} ohm")

    c = build(r_series=r_series)
    # |H| of the electrode RC at f_RF: 1 for the ideal drive, <1 with --rs
    h_mag = abs(rc_factor(frf, r_series))
    beff = eff_beta(beta, frf, r_series)
    # The ideal drive is an *algebraic* map (E_out = sqrt(P) e^{j beta sin}), so
    # the pipeline reproduces it to machine precision. With --rs the electrode is
    # a real RC ODE that BDF2 integrates, adding a small O(dt^2) transfer error;
    # the comb is still a Bessel comb at beta_eff, pinned to that looser bound.
    ideal = r_series <= 0
    tol_line = 1e-4 if ideal else 3e-2
    tol_gold = 1e-10 if ideal else 5e-4
    tol_rms = 1e-3 if ideal else 5e-3
    if r_series > 0:
        print(f"electrode RC: f_pole = {1/(2*np.pi*r_series*CEL)/1e9:.1f} GHz, "
              f"|H({frf/1e9:.0f} GHz)| = {h_mag:.3f} -> effective beta = {beff:.4f}")

    # ---- part 1: the comb spectrum vs the Bessel teeth --------------------
    t, e = run(c, beta, frf, r_series, progress=True)
    f, a = spectrum(e)
    ns, p_n = comb_lines(f, a, frf)
    p_th = il * P_LASER * jv(ns, beff) ** 2

    big = p_th > 1e-6 * p_th.max()          # teeth well above the solver floor
    err_line = float(np.max(np.abs(p_n[big] - p_th[big]) / p_th[big]))
    check("Bessel teeth  P_n = il*P*J_n(beta)^2", err_line < tol_line,
          f"{int(big.sum())} teeth, max rel err {err_line:.2e} "
          f"(effective beta = {beff:.4f})")

    # golden: the whole laser->PM->BDF2->FFT pipeline == the closed-form map
    err_gold = float(np.max(np.abs(e - golden(t, beta, frf, r_series))))
    check("golden map  E_out == sqrt(il P) e^{j beta sin}", err_gold < tol_gold,
          f"max |E_sim - E_exact| = {err_gold:.2e} "
          + ("(coherent-field pipeline adds nothing)" if ideal
             else "(BDF2 integrates the electrode RC)"))

    # ---- part 2: line spacing = f_RF exactly ------------------------------
    grid_power = float(p_n.sum())
    total_power = float(np.mean(np.abs(e) ** 2))
    off_grid = 1.0 - grid_power / total_power
    check("line spacing = f_RF (teeth only on drive harmonics)",
          off_grid < 1e-9,
          f"{off_grid*100:.2e}% of the power sits off the {frf/1e9:.0f} GHz grid")

    # ---- part 3: energy conservation (phase modulation is unitary) --------
    cons = total_power / (il * P_LASER) - 1.0
    sum_j2 = float(np.sum(jv(ns, beff) ** 2))
    check("energy conservation  sum_n J_n^2 = 1", abs(cons) < 1e-9,
          f"mean|E_out|^2/(il*P) - 1 = {cons:+.2e}, "
          f"sum J_n^2 = {sum_j2:.6f} (only redistributes photons)")

    # ---- part 4: carrier suppression at the J_0 zero ----------------------
    p_carrier = abs(line(f, a, 0.0)) ** 2
    p_side = abs(line(f, a, frf)) ** 2
    supp_db = 10.0 * np.log10(p_side / max(p_carrier, 1e-300))
    # the carrier only nulls when the *effective* index hits the J_0 zero
    check("carrier suppression at beta = 2.4048", supp_db > 40.0 if
          abs(beff - BETA0) < 1e-3 else True,
          f"carrier is {supp_db:.1f} dB below the first sideband "
          f"(J_0({beff:.4f})^2 = {jv(0, beff)**2:.2e})")

    # ---- part 5: broaden the comb by driving harder -----------------------
    # RMS comb width identity: sum_n n^2 J_n(beta)^2 = beta^2/2 exactly.
    betas = np.linspace(0.3, 6.0, 24)
    rms_order = np.empty_like(betas)
    n_teeth = np.empty_like(betas)          # teeth within 10 dB of the peak
    carrier_pw = np.empty_like(betas)
    side1_pw = np.empty_like(betas)
    for k, b in enumerate(betas):
        _, eb = run(c, b, frf, r_series)
        fb, ab = spectrum(eb)
        nb, pb = comb_lines(fb, ab, frf)
        m2 = float(np.sum(nb ** 2 * pb) / np.sum(pb))
        rms_order[k] = np.sqrt(m2)
        n_teeth[k] = int(np.sum(pb >= 0.1 * pb.max()))
        carrier_pw[k] = pb[nb == 0][0]
        side1_pw[k] = pb[nb == 1][0]
    betas_eff = betas * h_mag               # index actually applied to the light
    rms_th = betas_eff / np.sqrt(2.0)       # from sum n^2 J_n^2 = beta^2/2
    err_rms = float(np.max(np.abs(rms_order - rms_th) / rms_th))
    check("comb width  RMS order = beta/sqrt(2)", err_rms < tol_rms,
          f"max rel err {err_rms:.2e} over beta in [0.3, 6] "
          f"(sum n^2 J_n^2 = beta^2/2)")
    slope = float(np.polyfit(betas, rms_order, 1)[0])
    check("comb width grows linearly with drive",
          abs(slope - h_mag / np.sqrt(2)) < 1e-2,
          f"d(RMS order)/d(beta) = {slope:.4f} "
          f"(theory |H|/sqrt(2) = {h_mag/np.sqrt(2):.4f})")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    rc_tag = (f", R_series = {r_series:g} Ω" if r_series > 0 else "")
    fig.suptitle("Electro-optic frequency comb — CW laser + one phase modulator "
                 f"(f_RF = {frf/1e9:.0f} GHz{rc_tag})", fontsize=13)

    # (A) headline comb spectrum vs Bessel teeth
    a0 = ax[0, 0]
    a0.vlines(ns * frf / 1e9, -120, db_rel(p_n), color="#2c7fb8", lw=2.2,
              label="circulax field")
    a0.plot(ns * frf / 1e9, db_rel(p_th), "x", color="#d95f02", ms=7,
            label=r"$il\,P\,J_n(\beta)^2$")
    if abs(beff - BETA0) < 1e-3:
        a0.annotate("carrier suppressed\n(J₀ zero)", (0, -80), (0, -55),
                    ha="center", fontsize=8, color="tab:red",
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    a0.set(xlim=(-(np.ceil(beff) + 4) * frf / 1e9, (np.ceil(beff) + 4) * frf / 1e9),
           ylim=(-90, 5), xlabel="optical frequency offset [GHz]",
           ylabel="line power [dB re carrier]",
           title=f"comb spectrum at beta = {beff:.4f}")
    a0.legend(fontsize=8, loc="lower center")
    a0.grid(alpha=0.3)

    # (B) strong-drive broad comb (fixed high beta) to show broadening
    beff_hi = eff_beta(BETA_HI, frf, r_series)
    _, e_hi = run(c, BETA_HI, frf, r_series)
    f_hi, a_hi = spectrum(e_hi)
    n_hi, p_hi = comb_lines(f_hi, a_hi, frf)
    a1 = ax[0, 1]
    a1.vlines(n_hi * frf / 1e9, -120, db_rel(p_hi), color="#1b9e77", lw=2.2,
              label="circulax field")
    a1.plot(n_hi * frf / 1e9, db_rel(il * P_LASER * jv(n_hi, beff_hi) ** 2),
            "x", color="#d95f02", ms=7, label=r"$il\,P\,J_n(\beta)^2$")
    a1.set(xlim=(-(beff_hi + 5) * frf / 1e9, (beff_hi + 5) * frf / 1e9),
           ylim=(-60, 5), xlabel="optical frequency offset [GHz]",
           ylabel="line power [dB re carrier]",
           title=f"broad comb at beta = {beff_hi:.1f} "
                 f"(~{2*beff_hi:.0f} teeth)")
    a1.legend(fontsize=8, loc="lower center")
    a1.grid(alpha=0.3)

    # (C) carrier + first sideband vs beta — the Bessel nulls.
    # nominal beta on the x-axis; the light sees beta*|H|, so analytic curves
    # and the zero markers are scaled by |H| (identical when --rs is off).
    a2 = ax[1, 0]
    bg = np.linspace(0.1, 6.0, 400)
    a2.plot(betas, db_rel(carrier_pw), "o", color="#2c7fb8", ms=4,
            label="carrier (sim)")
    a2.plot(bg, db_rel(il * P_LASER * jv(0, bg * h_mag) ** 2), "-",
            color="#2c7fb8", lw=1, label=r"$J_0(\beta)^2$")
    a2.plot(betas, db_rel(side1_pw), "s", color="#d95f02", ms=4,
            label="1st sideband (sim)")
    a2.plot(bg, db_rel(il * P_LASER * jv(1, bg * h_mag) ** 2), "-",
            color="#d95f02", lw=1, label=r"$J_1(\beta)^2$")
    for z in (2.4048, 5.5201):
        a2.axvline(z / h_mag, color="0.6", ls=":", lw=0.8)
    a2.set(xlim=(0, 6), ylim=(-50, 3),
           xlabel=r"modulation index $\beta = \pi V_{ac}/V_\pi$",
           ylabel="line power [dB re carrier]",
           title="carrier nulls at the Bessel zeros (2.40, 5.52)")
    a2.legend(fontsize=8, ncol=2)
    a2.grid(alpha=0.3)

    # (D) comb width grows linearly with drive
    a3 = ax[1, 1]
    a3.plot(betas, rms_order, "o", color="#7570b3", ms=5, label="RMS order (sim)")
    a3.plot(bg, bg * h_mag / np.sqrt(2), "-", color="#7570b3", lw=1,
            label=r"$\beta/\sqrt{2}$")
    a3b = a3.twinx()
    a3b.plot(betas, n_teeth, "^", color="#e7298a", ms=5, alpha=0.7,
             label="teeth within 10 dB")
    a3b.plot(bg, 2 * bg * h_mag, ":", color="#e7298a", lw=1,
             label=r"$\sim 2\beta$")
    a3.set(xlabel=r"modulation index $\beta$",
           ylabel="RMS comb order  (bandwidth / f_RF)",
           title="comb bandwidth grows linearly with drive")
    a3b.set_ylabel("# teeth within 10 dB of peak", color="#e7298a")
    a3.legend(fontsize=8, loc="upper left")
    a3b.legend(fontsize=8, loc="lower right")
    a3.grid(alpha=0.3)

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
    ap.add_argument("--beta", type=float, default=BETA0,
                    help=f"modulation index for the headline spectrum "
                         f"(default {BETA0:.4f} = first J_0 zero)")
    ap.add_argument("--frf", type=float, default=F_RF,
                    help=f"RF drive tone / comb line spacing [Hz] "
                         f"(default {F_RF:g})")
    ap.add_argument("--rs", type=float, default=0.0,
                    help="electrode series resistance [ohm]; adds the RC pole "
                         "(default 0 = ideal drive)")
    args = ap.parse_args()
    raise SystemExit(main(beta=args.beta, frf=args.frf, r_series=args.rs))
