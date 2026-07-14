#!/usr/bin/env python3
"""Four-wave mixing in a chi(3) waveguide, pinned against every textbook scaling.

``models/waveguide_nl.va`` with the absorption channels off (beta_tpa =
sigma_fca = dn_dn = 0) is a pure Kerr segment — the real part of chi(3):

    E_out(t) = sqrt(T) * E_in(t) * exp(-j*k*P_in(t)),      T = exp(-alpha*L)
    k = gamma * L_nl [rad/W],   gamma = 2*pi*n2/(lambda*A_eff) [1/(W*m)]
    L_nl = (1 + exp(-alpha*L))/2 * L    (the model's trapezoid of the exact
    integral gamma*int P dz = gamma*P_in*L_eff — exact endpoint intensities
    over the geometric length, second-order accurate; a cascade of N
    segments converges on the distributed value at 1/N^2, checked below)

Nothing about four-wave mixing is coded anywhere: the Kerr phase just acts on
the INSTANTANEOUS envelope power. Launch a pump at envelope frequency f_p and
a signal at f_s and the envelope power beats at Om = f_s - f_p; the phase
modulation at Om scatters the tones into idlers. That is FWM, emergent.

The lumped two-tone response has an exact Jacobi-Anger line spectrum, which
makes the model completely checkable: with x = 2*k*sqrt(P_p*P_s), the line
at f_p + m*Om carries

    P_m = T * ( P_p*J_m(x)^2 + P_s*J_{m-1}(x)^2 )        (any drive level)

whose small-signal limit is the textbook phase-matched FWM result

    idler (m = -1, at 2*f_p - f_s):  P_i = T * (k*P_p)^2 * P_s
      -> P_i ~ P_p^2 (slope 2), ~ P_s (slope 1), ~ L_eff^2 (slope 2)
    XPM/SPM: signal phase slides 2*k*P_p per watt of pump, the pump only
      k*P_p — the chi(3) factor of two, emergent from |E|^2 E.

Self-checks (all asserted, machine-verifiable):
  1. identity     — the strongest generated line sits exactly at 2*f_p - f_s;
                    with n2 = 0 it vanishes into the solver floor
  2. toolchain    — every spectral line matches the exact map evaluated in
                    numpy on the same samples (VA -> JAX -> BDF2 adds nothing)
  3. pump slope   — d log P_i / d log P_p = 2, absolute eta = T*(k*P_p)^2
  4. signal slope — d log P_i / d log P_s = 1
  5. asymmetry    — P(2f_p-f_s)/P(2f_s-f_p) = P_p/P_s (Bessel-exact form)
  6. XPM/SPM      — phase-slope ratio signal/pump = 2
  7. length       — slope 2 in L at negligible loss; N-segment cascade
                    converges (1/N^2) on the distributed gamma*L_eff
  8. strong drive — 11 comb lines vs the Bessel formula at x ~ 2 (pump
                    depletion regime), plus exact energy conservation

Envelope convention: node value E = re + j*im, |E|^2 = power [W]; a tone at
envelope +f is optical nu0 - f, and 2*f_p - f_s maps identically in both
frames (2*(nu0-f_p) - (nu0-f_s) = nu0 - (2*f_p-f_s)). No dispersion is
modelled in a lumped segment, so phase matching is perfect (Dbeta = 0) —
the correct limit for a short/dispersion-engineered waveguide.

    .venv-circulax/bin/python examples/wg_fwm.py

        -> out/wg_fwm.png
"""
from __future__ import annotations

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
from circulax.components.base_component import Signals, States, source
from circulax.solvers.transient import BDF2VectorizedTransientSolver

from cavity import terminator
from lightspice import cx

C0 = 2.99792458e8

LAM0_NM = 1310.0
A_EFF_UM2 = 0.1
N2_KERR = 4.5e-18            # m^2/W (silicon-wire Re chi3; TPA off here)

F_P = 20e9                   # pump envelope frequency [Hz]
F_S = 30e9                   # signal envelope frequency [Hz]
F_BEAT = F_S - F_P           # all mixing products land on this 10 GHz grid

L_UM = 5000.0                # base device: 5 mm
LOSS_DB_M = 100.0
P_P0, P_S0 = 10e-3, 1e-4     # base drive: 10 mW pump, 0.1 mW signal

DT = 0.5e-12                 # sample/solver step; bins are 0.5 GHz
NPT = 4000                   # 2 ns = 20 exact beat periods -> leak-free FFT

OUT = Path(__file__).resolve().parents[1] / "out" / "wg_fwm.png"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ===========================================================================
# analytics
# ===========================================================================
def gamma_w(n2: float = N2_KERR) -> float:
    """Nonlinear parameter gamma = 2*pi*n2/(lambda*A_eff) [1/(W*m)]."""
    return 2 * np.pi * n2 / (LAM0_NM * 1e-9 * A_EFF_UM2 * 1e-12)


def k_model(length_um: float, loss_db_m: float, n2: float = N2_KERR):
    """(k [rad/W], power transmission T) of ONE waveguide_nl segment."""
    length = length_um * 1e-6
    alpha = loss_db_m * np.log(10) / 10
    lnl = 0.5 * (1 + np.exp(-alpha * length)) * length
    return gamma_w(n2) * lnl, np.exp(-alpha * length)


def k_cascade(n_seg: int, length_um: float, loss_db_m: float) -> float:
    """Total k of n_seg cascaded segments (phase-only maps compose exactly:
    |E|^2(t) only picks up T per segment, so k_total = sum k_seg*T^i)."""
    alpha = loss_db_m * np.log(10) / 10
    dz = length_um * 1e-6 / n_seg
    kseg, _ = k_model(length_um / n_seg, loss_db_m)
    return kseg * np.sum(np.exp(-alpha * dz * np.arange(n_seg)))


def k_distributed(length_um: float, loss_db_m: float) -> float:
    """The N -> inf limit: gamma * L_eff (the textbook NLSE value)."""
    length = length_um * 1e-6
    alpha = loss_db_m * np.log(10) / 10
    return gamma_w() * (1 - np.exp(-alpha * length)) / alpha


def bessel_line(m: int, p_p: float, p_s: float, k: float, t_pwr: float):
    """Exact power of the line at f_p + m*(f_s - f_p) (Jacobi-Anger)."""
    x = 2 * k * np.sqrt(p_p * p_s)
    return t_pwr * (p_p * jv(m, x) ** 2 + p_s * jv(m - 1, x) ** 2)


def golden_out(t: np.ndarray, p_p: float, p_s: float, k: float,
               t_pwr: float) -> np.ndarray:
    """The exact lumped map evaluated directly in numpy (the 'golden' twin
    of the whole VA -> bosdi -> JAX -> BDF2 pipeline)."""
    e_in = (np.sqrt(p_p) * np.exp(2j * np.pi * F_P * t)
            + np.sqrt(p_s) * np.exp(2j * np.pi * F_S * t))
    return np.sqrt(t_pwr) * e_in * np.exp(-1j * k * np.abs(e_in) ** 2)


# ===========================================================================
# circuit
# ===========================================================================
def two_tone():
    """Pump + signal on one bus: E = sqrt(p_p)e^{j2pi f_p t} + sqrt(p_s)e^{j2pi f_s t}."""
    @source(ports=("p1", "p2"), states=("i_src",))
    def TwoTone(signals: Signals, s: States, t: float,
                p_p: float = P_P0, p_s: float = P_S0) -> tuple[dict, dict]:
        field = (jnp.sqrt(p_p) * jnp.exp(2j * jnp.pi * F_P * t)
                 + jnp.sqrt(p_s) * jnp.exp(2j * jnp.pi * F_S * t))
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - field}, {}

    return TwoTone


def build(n_seg: int = 1, length_um: float = L_UM,
          loss_db_m: float = LOSS_DB_M):
    seg = dict(lambda_nm=LAM0_NM, length_um=length_um / n_seg,
               loss_db_m=loss_db_m, a_eff_um2=A_EFF_UM2,
               beta_tpa=0.0, sigma_fca=0.0, tau_fc=1e-9,
               n2_kerr=N2_KERR, dn_dn=0.0)
    inst = {
        "GND": {"component": "ground"},
        "SRC": {"component": "tone"},
        "TAP": {"component": "f2ri"},
        "TO": {"component": "term"},
    }
    conn = {"SRC,p1": "TAP,c",
            "GND,p1": tuple(["SRC,p2"] + [f"W{i},gnd" for i in range(n_seg)])}
    prev_re, prev_im = "TAP,re", "TAP,im"
    for i in range(n_seg):
        inst[f"W{i}"] = {"component": "wg", "settings": dict(seg)}
        conn[prev_re] = f"W{i},in_re"
        conn[prev_im] = f"W{i},in_im"
        prev_re, prev_im = f"W{i},out_re", f"W{i},out_im"
    conn[prev_re] = "TO,re"
    conn[prev_im] = "TO,im"
    net = {"instances": inst, "connections": conn,
           "ports": {"po_re": prev_re, "po_im": prev_im}}
    models = {"ground": lambda: 0, "tone": two_tone(),
              "f2ri": cx.field_to_ri(), "wg": cx.va("waveguide_nl"),
              "term": terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def run(c, params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step BDF2 over exactly NPT samples; returns (t, E_out(t))."""
    t_max = NPT * DT
    ts = jnp.arange(0.0, t_max, DT)
    y0 = c.dc(params=params)
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=DT, y0=y0, params=params,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=NPT + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    assert sol.result == diffrax.RESULTS.successful, f"transient failed: {sol.result}"
    e = (np.asarray(c.port(sol.ys, "po_re").real)
         + 1j * np.asarray(c.port(sol.ys, "po_im").real))
    # diffrax appends t1 as an extra save; keep exactly NPT uniform samples
    # (an integer number of beat periods -> leak-free rectangular FFT)
    return np.asarray(sol.ts)[:NPT], e[:NPT]


def spectrum(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-window FFT — exact for lines on the 0.5 GHz bin grid."""
    return np.fft.fftfreq(len(e), d=DT), np.fft.fft(e) / len(e)


def line(f: np.ndarray, a: np.ndarray, f0: float) -> complex:
    i = int(np.argmin(np.abs(f - f0)))
    assert abs(f[i] - f0) < 1e3, f"{f0/1e9} GHz is not on the FFT bin grid"
    return complex(a[i])


def dbm(p: float | np.ndarray):
    return 10 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-30) / 1e-3)


# ===========================================================================
def main() -> int:
    k0, t0 = k_model(L_UM, LOSS_DB_M)
    print(f"chi(3) waveguide: gamma = {gamma_w():.1f} /W/m, L = {L_UM/1e3:.1f} mm "
          f"@ {LOSS_DB_M:.0f} dB/m -> k = {k0:.4f} rad/W, T = {t0:.4f}")
    print(f"pump {P_P0*1e3:.0f} mW @ +{F_P/1e9:.0f} GHz, "
          f"signal {P_S0*1e3:.2f} mW @ +{F_S/1e9:.0f} GHz (envelope frame; "
          f"+f is optical nu0 - f)")

    f_i = 2 * F_P - F_S          # idler, +10 GHz
    f_i2 = 2 * F_S - F_P         # secondary idler, +40 GHz

    # ---- part 1: emergence, identity, toolchain ---------------------------
    c = build()
    t, e = run(c)
    f, a = spectrum(e)
    p_lines = np.abs(a) ** 2

    excl = (np.abs(f - F_P) < 1e6) | (np.abs(f - F_S) < 1e6)
    i_max = int(np.argmax(np.where(excl, 0.0, p_lines)))
    p_i = abs(line(f, a, f_i)) ** 2
    check("idler at 2*f_p - f_s",
          abs(f[i_max] - f_i) < 1e3,
          f"strongest generated line at {f[i_max]/1e9:+.1f} GHz "
          f"(expect {f_i/1e9:+.1f}), P_i = {dbm(p_i):.1f} dBm")

    _, ag = spectrum(golden_out(t, P_P0, P_S0, k0, t0))
    big = np.abs(ag) ** 2 > 1e-21        # every line down to -180 dBm
    err_tc = float(np.max(np.abs(a[big] - ag[big]) / np.abs(ag[big])))
    check("toolchain == exact map", err_tc < 1e-8,
          f"max relative line error {err_tc:.2e} over {int(big.sum())} lines "
          f"spanning 190 dB (VA -> JAX -> BDF2 adds nothing)")

    _, e_off = run(c, params={"W0.n2_kerr": 0.0})
    f2, a_off = spectrum(e_off)
    p_i_off = abs(line(f2, a_off, f_i)) ** 2
    check("n2 = 0 kills the idler", p_i_off < 1e-12 * p_i,
          f"idler {dbm(p_i_off):.0f} dBm with n2 = 0 "
          f"vs {dbm(p_i):.1f} dBm with n2 on")

    # ---- part 2: pump-power scaling + XPM/SPM ------------------------------
    pp_sweep = np.array([2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    p_i_pp, th_p, th_s = [], [], []
    for pp in pp_sweep:
        _, e2 = run(c, params={"SRC.p_p": float(pp)})
        f2, a2 = spectrum(e2)
        p_i_pp.append(abs(line(f2, a2, f_i)) ** 2)
        th_p.append(np.angle(line(f2, a2, F_P)))
        th_s.append(np.angle(line(f2, a2, F_S)))
    p_i_pp = np.array(p_i_pp)

    slope_pp = np.polyfit(np.log(pp_sweep), np.log(p_i_pp), 1)[0]
    check("P_i ~ P_p^2", abs(slope_pp - 2) < 0.01,
          f"fitted slope {slope_pp:.4f} (theory 2)")

    eta_sim = p_i_pp / P_S0
    eta_th = t0 * (k0 * pp_sweep) ** 2
    err_eta = float(np.max(np.abs(eta_sim / eta_th - 1)))
    check("absolute eta = T*(k*P_p)^2", err_eta < 5e-3,
          f"eta = {eta_sim[2]:.3e} at {pp_sweep[2]*1e3:.0f} mW "
          f"(theory {eta_th[2]:.3e}), max err {err_eta*100:.3f}%")

    sl_p = np.polyfit(pp_sweep, np.unwrap(th_p), 1)[0]
    sl_s = np.polyfit(pp_sweep, np.unwrap(th_s), 1)[0]
    check("XPM/SPM = 2", abs(sl_s / sl_p - 2) < 0.01,
          f"signal phase slope {sl_s:+.3f} rad/W, pump {sl_p:+.3f} rad/W "
          f"-> ratio {sl_s/sl_p:.4f} (and -k = {-k0:.3f})")

    # ---- part 3: signal-power scaling + asymmetry --------------------------
    ps_sweep = np.array([2e-5, 5e-5, 1e-4, 2e-4, 5e-4])
    p_i_ps = []
    for ps in ps_sweep:
        _, e2 = run(c, params={"SRC.p_s": float(ps)})
        f2, a2 = spectrum(e2)
        p_i_ps.append(abs(line(f2, a2, f_i)) ** 2)
    p_i_ps = np.array(p_i_ps)
    slope_ps = np.polyfit(np.log(ps_sweep), np.log(p_i_ps), 1)[0]
    check("P_i ~ P_s^1", abs(slope_ps - 1) < 0.01,
          f"fitted slope {slope_ps:.4f} (theory 1)")

    p_i2 = abs(line(f, a, f_i2)) ** 2
    ratio = p_i / p_i2
    ratio_th = bessel_line(-1, P_P0, P_S0, k0, t0) / \
        bessel_line(2, P_P0, P_S0, k0, t0)
    check("idler asymmetry = P_p/P_s", abs(ratio / ratio_th - 1) < 1e-3,
          f"P(2fp-fs)/P(2fs-fp) = {ratio:.1f} "
          f"(Bessel-exact {ratio_th:.1f}, naive P_p/P_s = {P_P0/P_S0:.0f})")

    # ---- part 4: length scaling + N-segment convergence ---------------------
    ll_sweep = np.array([1e3, 2e3, 5e3, 1e4, 2e4])      # um, ~lossless
    p_i_ll = []
    for lum in ll_sweep:
        _, e2 = run(c, params={"W0.length_um": float(lum),
                               "W0.loss_db_m": 0.01})
        f2, a2 = spectrum(e2)
        p_i_ll.append(abs(line(f2, a2, f_i)) ** 2)
    p_i_ll = np.array(p_i_ll)
    slope_ll = np.polyfit(np.log(ll_sweep), np.log(p_i_ll), 1)[0]
    check("P_i ~ L^2", abs(slope_ll - 2) < 0.01,
          f"fitted slope {slope_ll:.4f} at 0.01 dB/m (theory 2)")

    # lossy device: cascade N segments -> distributed gamma*L_eff
    l_c, loss_c = 1e4, 600.0                 # 10 mm at 600 dB/m: alpha*L = 1.38
    n_segs = (1, 2, 4, 8, 16)
    k_dist = k_distributed(l_c, loss_c)
    eta_n, err_n = [], []
    for n in n_segs:
        cn = build(n_seg=n, length_um=l_c, loss_db_m=loss_c)
        _, e2 = run(cn)
        f2, a2 = spectrum(e2)
        eta = abs(line(f2, a2, f_i)) ** 2 / P_S0
        eta_n.append(eta)
        k_n = k_cascade(n, l_c, loss_c)
        err_n.append(abs(eta / (np.exp(-loss_c * np.log(10) / 10 * l_c * 1e-6)
                                * (k_dist * P_P0) ** 2) - 1))
        print(f"    N = {n:2d}: k_N = {k_n:.5f} rad/W "
              f"(distributed {k_dist:.5f}), eta off by {err_n[-1]*100:6.3f}%")
    conv = err_n[0] / err_n[-1]
    check("N-segment -> distributed NLSE", err_n[-1] < 5e-3 and conv > 100,
          f"eta error {err_n[0]*100:.2f}% (N=1) -> {err_n[-1]*100:.4f}% (N=16), "
          f"x{conv:.0f} ~ 1/N^2 = x{n_segs[-1]**2}")

    # ---- part 5: strong drive — Bessel comb + energy conservation ----------
    pp_hi, ps_hi = 1.0, 1.0
    _, e_hi = run(c, params={"SRC.p_p": pp_hi, "SRC.p_s": ps_hi})
    f_hi, a_hi = spectrum(e_hi)
    x_hi = 2 * k0 * np.sqrt(pp_hi * ps_hi)
    ms = np.arange(-4, 7)
    p_sim = np.array([abs(line(f_hi, a_hi, F_P + m * F_BEAT)) ** 2 for m in ms])
    p_th = np.array([bessel_line(int(m), pp_hi, ps_hi, k0, t0) for m in ms])
    err_comb = float(np.max(np.abs(p_sim / p_th - 1)))
    check("strong-drive Bessel comb", err_comb < 1e-3,
          f"11 lines over {dbm(p_th.max()) - dbm(p_th.min()):.0f} dB at "
          f"x = {x_hi:.2f}: max err {err_comb*100:.4f}% "
          f"(pump depleted to J0^2 = {jv(0, x_hi)**2:.2f})")

    p_out_mean = float(np.mean(np.abs(e_hi) ** 2))
    cons = p_out_mean / (t0 * (pp_hi + ps_hi)) - 1
    check("energy conservation", abs(cons) < 1e-8,
          f"mean P_out / (T * P_in) - 1 = {cons:+.2e} "
          f"(Kerr mixing only redistributes photons)")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.5))
    fig.suptitle("Four-wave mixing in a chi(3) waveguide "
                 "(models/waveguide_nl.va, Kerr only)", fontsize=13)

    a0 = ax[0, 0]
    sel = p_lines > 1e-26
    a0.vlines(f[sel] / 1e9, -230, dbm(p_lines[sel]), color="#2c7fb8", lw=1.8)
    p_off_lines = np.abs(a_off) ** 2
    sel0 = p_off_lines > 1e-26
    a0.vlines(f2[sel0] / 1e9, -230, dbm(p_off_lines[sel0]),
              color="0.75", lw=3.5, alpha=0.8, zorder=0)
    for fx, name, dy in ((F_P, "pump", 3), (F_S, "signal", 3),
                         (f_i, "idler\n2fp-fs", 6), (f_i2, "2fs-fp", 6)):
        a0.annotate(name, (fx / 1e9, dbm(abs(line(f, a, fx))**2) + dy),
                    ha="center", fontsize=8)
    a0.set(xlim=(-15, 65), ylim=(-150, 30), xlabel="envelope f [GHz]",
           ylabel="line power [dBm]",
           title="output spectrum (grey: n2 = 0 — no mixing)")

    a1 = ax[0, 1]
    a1.loglog(pp_sweep * 1e3, p_i_pp * 1e3, "o", color="#d95f02",
              label=f"vs pump: slope {slope_pp:.3f}")
    a1.loglog(pp_sweep * 1e3, t0 * (k0 * pp_sweep) ** 2 * P_S0 * 1e3, "-",
              color="#d95f02", lw=1, label=r"$T(kP_p)^2P_s$")
    a1.loglog(ps_sweep * 1e3, p_i_ps * 1e3, "s", color="#1b9e77",
              label=f"vs signal: slope {slope_ps:.3f}")
    a1.loglog(ps_sweep * 1e3, t0 * (k0 * P_P0) ** 2 * ps_sweep * 1e3, "-",
              color="#1b9e77", lw=1, label=r"$T(kP_p)^2P_s$")
    a1.set(xlabel="swept power [mW]", ylabel="idler power [mW]",
           title=r"$P_i \propto P_p^2\,P_s$")
    a1.legend(fontsize=8)

    a2 = ax[0, 2]
    a2.plot(pp_sweep * 1e3, np.unwrap(th_p) - th_p[0], "o-",
            color="#7570b3", label=f"pump (SPM): {sl_p:+.3f} rad/W")
    a2.plot(pp_sweep * 1e3, np.unwrap(th_s) - th_s[0], "s-",
            color="#e7298a", label=f"signal (XPM): {sl_s:+.3f} rad/W")
    a2.set(xlabel="pump power [mW]", ylabel="line phase shift [rad]",
           title=f"XPM/SPM = {sl_s/sl_p:.4f} (theory 2)")
    a2.legend(fontsize=8)

    a3 = ax[1, 0]
    a3.loglog(ll_sweep / 1e3, p_i_ll * 1e3, "o", color="#2c7fb8",
              label=f"sim: slope {slope_ll:.3f}")
    a3.loglog(ll_sweep / 1e3,
              (gamma_w() * ll_sweep * 1e-6 * P_P0) ** 2 * P_S0 * 1e3, "-",
              color="#2c7fb8", lw=1, label=r"$(\gamma P_p L)^2 P_s$")
    a3.set(xlabel="length [mm]", ylabel="idler power [mW]",
           title=r"$P_i \propto L^2$ (0.01 dB/m)")
    a3.legend(fontsize=8)

    a4 = ax[1, 1]
    a4.loglog(n_segs, np.array(err_n) * 100, "o-", color="#d95f02",
              label="|eta error| vs distributed")
    a4.loglog(n_segs, err_n[0] * 100 / np.array(n_segs) ** 2, "--",
              color="0.4", label=r"$1/N^2$")
    a4.set(xlabel="segments N", ylabel="error [%]",
           title=f"lumped -> distributed NLSE ({l_c/1e3:.0f} mm, "
                 f"{loss_c:.0f} dB/m)")
    a4.legend(fontsize=8)

    a5 = ax[1, 2]
    a5.plot(ms, dbm(p_sim), "o", ms=8, color="#2c7fb8", label="simulated")
    a5.plot(ms, dbm(p_th), "x", ms=9, color="#d95f02",
            label=r"$T(P_pJ_m^2{+}P_sJ_{m-1}^2)$")
    a5.set(xlabel=r"line order m (at $f_p + m\,\Omega$)",
           ylabel="line power [dBm]",
           title=f"strong drive x = {x_hi:.2f}: Bessel comb, line-by-line")
    a5.legend(fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    if all(ok for _, ok, _ in CHECKS):
        print("ALL FWM CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED:", [n for n, ok, _ in CHECKS if not ok])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
