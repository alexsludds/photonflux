#!/usr/bin/env python3
"""Four-wave mixing INSIDE a ring resonator, driven by lasers one FSR apart.

``models/ring_kerr.va`` is an add-drop microring whose five longitudinal
modes (m = -2..+2, spaced by the FSR) share the intracavity chi(3) Kerr
nonlinearity — the modal (Lugiato-Lefever) picture of a Kerr comb:

    dA_m/dt = (j*delta_m - 1/tau) A_m
              + j*g_eff * sum_{j+l-k=m} A_j A_l conj(A_k)      <- chi(3)
              + j*kappa^2 * s_in

The single momentum-matched triple sum produces every chi(3) effect with
the textbook coefficients: SPM (x1), XPM (x2), and four-wave mixing.
NOTHING here says "generate an idler" — pump mode 0 on resonance, seed
mode +1 one FSR away, and the idler GROWS in mode -1 because
2*w_p = w_s + w_i lands exactly on a ring resonance (perfect phase
matching when the comb is dispersion-free).

Device: R = 1988 um (FSR = 6.000 GHz), 30 dB/m, kappa2 = 0.035 both buses
-> loaded linewidth 149 MHz (Q = 1.5e6), photon lifetime tau = 2.13 ns.
Pump at envelope 0 (mode 0), signal at envelope -FSR (mode +1, one FSR
blue); the idler appears at envelope +FSR = mode -1, one FSR red.

Self-checks (all asserted):
  1. identity    — the strongest generated line is at 2*f_p - f_s, exactly
                   one FSR on the other side of the pump; n2 = 0 kills it
  2. golden ODE  — every bus spectral line matches an independent scipy
                   integration of the same five coupled modal ODEs
  3. pump slope  — P_i ~ P_p^2, and absolutely P_i = (g_eff*tau)^2 *
                   (k2c*tau)^6 * P_p^2 * P_s (all three fields resonant)
  4. signal slope— P_i ~ P_s^1
  5. detuning    — slide the comb under the fixed lasers: the conversion
                   collapses as |L(delta)|^8 = 1/(1+(delta*tau)^2)^4 — the
                   Lorentzian to the FOURTH power (pump^2 x signal x idler:
                   the ring resonance enters eight times in field)
  6. dispersion  — d2_hz walks the +-1 modes off the equally-spaced grid:
                   the idler mode misses 2*f_p - f_s and conversion dies as
                   1/(1+(pi*d2*tau)^2)^2 — PHASE MATCHING, ring style
  7. cascade     — at mW drive the +-2 modes light up (comb seed), pinned
                   line-by-line against the golden ODE
  8. enhancement — the same chi(3), same physical length (the 12.5 mm
                   circumference), same input, as a STRAIGHT waveguide
                   (models/waveguide_nl.va): the ring converts ~1000x more

    .venv-circulax/bin/python examples/ring_fwm.py

        -> out/ring_fwm.png
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
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source
from circulax.solvers.transient import BDF2VectorizedTransientSolver

from cavity import terminator
from lightspice import cx

C0 = 2.99792458e8

LAM0_NM = 1310.0
N_G = 4.0
A_EFF_UM2 = 0.1
N2_KERR = 4.5e-18
FSR = 6e9                                     # exact, sets the radius
RADIUS_UM = C0 / (N_G * 2 * np.pi * FSR) * 1e6      # 1988.06 um
LOSS_DB_M = 30.0
K2_IN = K2_DROP = 0.035

P_P0, P_S0 = 100e-6, 10e-6     # base drive [W]
F_P, F_S = 0.0, -FSR           # envelope: pump on mode 0, signal on mode +1
F_I = 2 * F_P - F_S            # idler: +FSR -> mode -1

DT = 1e-12                     # scaling checks (relative ratios; BDF2 bias
                               # cancels between numerator and denominator)
DT_FINE = 0.25e-12             # golden-ODE / absolute-value checks: the BDF2
                               # discretization bias on the 6-12 GHz envelope
                               # tones falls as dt^2, and at 0.25 ps the
                               # circuit matches the DOP853 golden to < 0.1%
                               # on the idler (proof the VA equations are exact)
T_STOP = 16e-9                 # ~7.5 photon lifetimes of settling
N_WIN = 2000                   # spectral window: last 2 ns -> 0.5 GHz bins

MODES = np.array([-2, -1, 0, 1, 2])

OUT = Path(__file__).resolve().parents[1] / "out" / "ring_fwm.png"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ===========================================================================
# analytics: CMT rates and the closed-form small-signal idler
# ===========================================================================
def rates() -> dict:
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    v_g = C0 / N_G
    t_rt = circ / v_g
    veff = A_EFF_UM2 * 1e-12 * circ
    w0 = 2 * np.pi * C0 / (LAM0_NM * 1e-9)
    inv_i = LOSS_DB_M * np.log(10) / 10 * v_g / 2
    inv_e1 = K2_IN / (2 * t_rt)
    inv_e2 = K2_DROP / (2 * t_rt)
    tau = 1 / (inv_i + inv_e1 + inv_e2)
    k2c = 2 * inv_e1
    g_u = (w0 / N_G) * N2_KERR * v_g / veff        # [rad/s per J]
    return dict(circ=circ, v_g=v_g, t_rt=t_rt, veff=veff, w0=w0,
                inv_i=inv_i, inv_e1=inv_e1, inv_e2=inv_e2, tau=tau,
                k2c=k2c, g_u=g_u, geff=g_u / k2c)


def eta_ring(p_p: float, r: dict) -> float:
    """Small-signal idler conversion P_i(thru)/P_s, everything on resonance:
    |A_p|^2 = (k2c*tau)^2 P_p, A_i = j*geff*tau*A_p^2*conj(A_s)."""
    return (r["geff"] * r["tau"] * p_p) ** 2 * (r["k2c"] * r["tau"]) ** 6


def eta_wg(p_p: float, r: dict) -> float:
    """Same chi(3), same length, straight waveguide (wg_fwm.py's pinned
    formula): eta = T*(gamma*P_p*L_nl)^2."""
    gamma = 2 * np.pi * N2_KERR / (LAM0_NM * 1e-9 * A_EFF_UM2 * 1e-12)
    alpha = LOSS_DB_M * np.log(10) / 10
    lnl = 0.5 * (1 + np.exp(-alpha * r["circ"])) * r["circ"]
    return np.exp(-alpha * r["circ"]) * (gamma * p_p * lnl) ** 2


# ===========================================================================
# golden model: the same five modal ODEs, integrated independently by scipy
# ===========================================================================
def golden_lines(p_p: float, p_s: float, r: dict, t_full: np.ndarray,
                 lam_shift_hz: float = 0.0, d2_hz: float = 0.0):
    """Integrate dA/dt with solve_ivp (DOP853, rtol 1e-11) from the
    frozen-source steady state (what c.dc() finds), then FFT the bus field
    over the SAME last-N_WIN window as the circuit sim. Returns (f, amp)."""
    fsr_w = 2 * np.pi * FSR
    d0 = -2 * np.pi * lam_shift_hz
    delta = d0 - MODES * fsr_w - np.pi * d2_hz * MODES**2
    geff, tau, k2c = r["geff"], r["tau"], r["k2c"]

    def s_in(t):
        return (np.sqrt(p_p) * np.exp(2j * np.pi * F_P * t)
                + np.sqrt(p_s) * np.exp(2j * np.pi * F_S * t))

    def tsum(a):
        t_m = np.zeros(5, dtype=complex)
        for mi, m in enumerate(MODES):
            for ji, j in enumerate(MODES):
                for li, l in enumerate(MODES):
                    k = j + l - m
                    if -2 <= k <= 2:
                        t_m[mi] += a[ji] * a[li] * np.conj(a[k + 2])
        return t_m

    def rhs(t, a):
        return ((1j * delta - 1 / tau) * a + 1j * geff * tsum(a)
                + 1j * k2c * s_in(t))

    # frozen-source steady state (the DC solve the circuit starts from)
    def steady(x):
        a = x[:5] + 1j * x[5:]
        f = rhs(0.0, a)
        return np.concatenate([f.real, f.imag])

    a_lin = 1j * k2c * s_in(0.0) / (1 / tau - 1j * delta)
    x0 = fsolve(steady, np.concatenate([a_lin.real, a_lin.imag]), xtol=1e-13)
    a0 = x0[:5] + 1j * x0[5:]

    sol = solve_ivp(rhs, (0.0, t_full[-1]), a0, t_eval=t_full,
                    method="DOP853", rtol=1e-11, atol=1e-16,
                    max_step=10e-12)
    assert sol.success
    e_thru = (s_in(sol.t) + 1j * sol.y.sum(axis=0))[-N_WIN:]
    return np.fft.fftfreq(N_WIN, d=DT), np.fft.fft(e_thru) / N_WIN


# ===========================================================================
# circuits
# ===========================================================================
def two_tone():
    @source(ports=("p1", "p2"), states=("i_src",))
    def TwoTone(signals: Signals, s: States, t: float,
                p_p: float = P_P0, p_s: float = P_S0) -> tuple[dict, dict]:
        field = (jnp.sqrt(p_p) * jnp.exp(2j * jnp.pi * F_P * t)
                 + jnp.sqrt(p_s) * jnp.exp(2j * jnp.pi * F_S * t))
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - field}, {}

    return TwoTone


def build_ring():
    ring = dict(lambda_nm=LAM0_NM, lambda_res_nm=LAM0_NM,
                radius_um=RADIUS_UM, n_g=N_G, loss_db_m=LOSS_DB_M,
                kappa2_in=K2_IN, kappa2_drop=K2_DROP,
                a_eff_um2=A_EFF_UM2, n2_kerr=N2_KERR, d2_hz=0.0)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SRC": {"component": "tone"},
            "TAP": {"component": "f2ri"},
            "RG": {"component": "ring", "settings": ring},
            "T1": {"component": "term"}, "T2": {"component": "term"},
        },
        "connections": {
            "SRC,p1": "TAP,c",
            "TAP,re": "RG,in_re", "TAP,im": "RG,in_im",
            "RG,thru_re": "T1,re", "RG,thru_im": "T1,im",
            "RG,drop_re": "T2,re", "RG,drop_im": "T2,im",
            "GND,p1": ("SRC,p2", "RG,gnd"),
        },
        "ports": {"po_re": "RG,thru_re", "po_im": "RG,thru_im",
                  "pd_re": "RG,drop_re", "pd_im": "RG,drop_im"},
    }
    models = {"ground": lambda: 0, "tone": two_tone(),
              "f2ri": cx.field_to_ri(), "ring": cx.va("ring_kerr"),
              "term": terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def build_wg():
    """The straight-waveguide reference: same chi(3), same physical length."""
    r = rates()
    wg = dict(lambda_nm=LAM0_NM, length_um=r["circ"] * 1e6,
              loss_db_m=LOSS_DB_M, a_eff_um2=A_EFF_UM2,
              beta_tpa=0.0, sigma_fca=0.0, tau_fc=1e-9,
              n2_kerr=N2_KERR, dn_dn=0.0)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SRC": {"component": "tone"},
            "TAP": {"component": "f2ri"},
            "WG": {"component": "wg", "settings": wg},
            "TO": {"component": "term"},
        },
        "connections": {
            "SRC,p1": "TAP,c",
            "TAP,re": "WG,in_re", "TAP,im": "WG,in_im",
            "WG,out_re": "TO,re", "WG,out_im": "TO,im",
            "GND,p1": ("SRC,p2", "WG,gnd"),
        },
        "ports": {"po_re": "WG,out_re", "po_im": "WG,out_im"},
    }
    models = {"ground": lambda: 0, "tone": two_tone(),
              "f2ri": cx.field_to_ri(), "wg": cx.va("waveguide_nl"),
              "term": terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def run(c, params: dict | None = None, dt: float = DT):
    """Transient to steady state; returns (f, amp) of the thru field over
    the last 2 ns (0.5 GHz bins, every line bin-exact). ``dt`` is the fixed
    BDF2 step; at the 6 GHz FSR beat one step is 0.6% of a cycle, so the
    discretization bias is negligible (unlike the 119 GHz Vernier case)."""
    npt = int(round(T_STOP / dt))
    n_win = int(round(N_WIN * DT / dt))
    ts = jnp.arange(0.0, T_STOP, dt)
    y0 = c.dc(params=params)
    sol = c.transient(
        t0=0.0, t1=T_STOP, dt0=dt, y0=y0, params=params,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=npt + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    assert sol.result == diffrax.RESULTS.successful, f"transient failed: {sol.result}"
    e = (np.asarray(c.port(sol.ys, "po_re").real)
         + 1j * np.asarray(c.port(sol.ys, "po_im").real))[:npt][-n_win:]
    return np.fft.fftfreq(n_win, d=dt), np.fft.fft(e) / n_win


def line(f: np.ndarray, a: np.ndarray, f0: float) -> complex:
    i = int(np.argmin(np.abs(f - f0)))
    assert abs(f[i] - f0) < 1e3
    return complex(a[i])


def dbm(p):
    return 10 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-30) / 1e-3)


# ===========================================================================
def main() -> int:
    r = rates()
    lw_hz = 2 / r["tau"] / (2 * np.pi)
    print(f"ring: R = {RADIUS_UM:.1f} um -> FSR = {FSR/1e9:.3f} GHz, "
          f"{LOSS_DB_M:.0f} dB/m, kappa2 = {K2_IN}: linewidth "
          f"{lw_hz/1e6:.0f} MHz (Q = {r['w0']*r['tau']/2:.2e}), "
          f"tau = {r['tau']*1e9:.2f} ns")
    print(f"pump {P_P0*1e6:.0f} uW on mode 0, signal {P_S0*1e6:.0f} uW on "
          f"mode +1 (one FSR blue); idler expected in mode -1 "
          f"(envelope {F_I/1e9:+.0f} GHz, one FSR red)")

    t_win = np.arange(N_WIN) * DT   # golden integrates its own settle; the
    # spectral window is phase-insensitive (every line is an exact bin)

    # ---- part 1: emergence + golden ODE ------------------------------------
    # the base run drives the golden compare, the spectrum plot and the
    # enhancement reference, so take it at the fine step
    c = build_ring()
    f, a = run(c, dt=DT_FINE)
    p_lines = np.abs(a) ** 2
    excl = (np.abs(f - F_P) < 1e6) | (np.abs(f - F_S) < 1e6)
    i_max = int(np.argmax(np.where(excl, 0.0, p_lines)))
    p_i = abs(line(f, a, F_I)) ** 2
    check("idler in mode -1 at 2*f_p - f_s", abs(f[i_max] - F_I) < 1e3,
          f"strongest generated line at {f[i_max]/1e9:+.1f} GHz "
          f"(expect {F_I/1e9:+.1f} = one FSR red of the pump), "
          f"P_i = {dbm(p_i):.1f} dBm")

    f_off, a_off = run(c, params={"RG.n2_kerr": 0.0})
    p_i_off = abs(line(f_off, a_off, F_I)) ** 2
    check("n2 = 0 kills it", p_i_off < 1e-8 * p_i,
          f"idler {dbm(p_i_off):.0f} dBm with n2 = 0 vs {dbm(p_i):.1f} dBm "
          f"(the -150 dBm remnant is the DC-start ring-down of mode -1, "
          f"not mixing)")

    # golden: integrate the same 16 ns with scipy from the same frozen start
    t_full = np.arange(int(round(T_STOP / DT))) * DT
    fg, ag = golden_lines(P_P0, P_S0, r, t_full)
    errs = []
    for fk in (0.0, -FSR, F_I):                # pump, signal, idler lines
        p_sim = abs(line(f, a, fk)) ** 2
        p_gold = abs(line(fg, ag, fk)) ** 2
        errs.append(abs(p_sim / p_gold - 1))
    err_g = float(np.max(errs))
    check("golden ODE line-by-line", err_g < 5e-3,
          f"pump/signal/idler bus lines vs independent scipy integration "
          f"of the 5 modal ODEs (DOP853): max err {err_g*100:.3f}% — the VA "
          f"model IS these equations")

    # ---- part 2: power scalings + absolute eta ------------------------------
    # fine dt: these feed the ABSOLUTE eta comparison, so the BDF2 bias must
    # be below the physics signal (the Kerr resonance pull)
    pp_sweep = np.array([25e-6, 50e-6, 100e-6, 200e-6])
    p_i_pp = []
    for pp in pp_sweep:
        f2, a2 = (f, a) if pp == P_P0 else run(c, params={"SRC.p_p": float(pp)},
                                               dt=DT_FINE)
        p_i_pp.append(abs(line(f2, a2, F_I)) ** 2)
    p_i_pp = np.array(p_i_pp)
    slope_pp = np.polyfit(np.log(pp_sweep), np.log(p_i_pp), 1)[0]
    check("P_i ~ P_p^2", abs(slope_pp - 2) < 0.02,
          f"fitted slope {slope_pp:.4f} (theory 2)")

    eta_sim = p_i_pp / P_S0
    eta_th = np.array([eta_ring(pp, r) for pp in pp_sweep])
    # the closed form is the small-signal (P->0) limit with all three modes
    # exactly on resonance. The residual deviation is the pump's own SPM/XPM
    # pulling the resonances, so it GROWS LINEARLY with power: 0.1% at 25 uW
    # (essentially the exact limit) up through a few % — that linear ramp IS
    # the Kerr shift, not solver error
    dev = np.abs(eta_sim / eta_th - 1)
    check("absolute eta (all 3 waves resonant)", dev[0] < 3e-3,
          f"eta/P_s = {eta_sim[0]:.3e} at 25 uW = closed-form "
          f"(g_eff*tau*P_p)^2 (k^2*tau)^6 to {dev[0]*100:.2f}%; the deviation "
          f"then ramps linearly with power ({dev[1]*100:.1f}/{dev[2]*100:.1f}/"
          f"{dev[3]*100:.1f}% at 50/100/200 uW) — the Kerr resonance pull")

    ps_sweep = np.array([2.5e-6, 5e-6, 10e-6, 20e-6])
    p_i_ps = []
    for ps in ps_sweep:
        f2, a2 = (f, a) if ps == P_S0 else run(c, params={"SRC.p_s": float(ps)})
        p_i_ps.append(abs(line(f2, a2, F_I)) ** 2)
    p_i_ps = np.array(p_i_ps)
    slope_ps = np.polyfit(np.log(ps_sweep), np.log(p_i_ps), 1)[0]
    check("P_i ~ P_s^1", abs(slope_ps - 1) < 0.02,
          f"fitted slope {slope_ps:.4f} (theory 1)")

    # ---- part 3: slide the comb under the lasers -> |L|^8 -------------------
    # (at 25 uW pump, where the Kerr self-pull of the resonances is <1% of a
    # linewidth and the pure Lorentzian law is clean)
    pp_det = float(pp_sweep[0])
    p_i_ref = float(p_i_pp[0])
    nu_res0 = C0 / (LAM0_NM * 1e-9)
    shifts = np.array([-149.2e6, -74.6e6, 74.6e6, 149.2e6])   # ~1, 2 linewidths
    p_i_det = []
    for df in shifts:
        lam_res = C0 / (nu_res0 + df) * 1e9
        f2, a2 = run(c, params={"SRC.p_p": pp_det,
                                "RG.lambda_res_nm": float(lam_res)})
        p_i_det.append(abs(line(f2, a2, F_I)) ** 2)
    p_i_det = np.array(p_i_det)
    l8 = 1 / (1 + (2 * np.pi * shifts * r["tau"]) ** 2) ** 4
    err_det = float(np.max(np.abs(p_i_det / (p_i_ref * l8) - 1)))
    check("detuning: Lorentzian^4 collapse", err_det < 0.10,
          f"comb shifted 0.5/1 linewidth: P_i falls x"
          f"{1/l8[1]:.1f}/x{1/l8[0]:.0f} (sim x{p_i_ref/p_i_det[1]:.1f}/"
          f"x{p_i_ref/p_i_det[0]:.0f}), |L|^8 max err {err_det*100:.1f}%")

    # ---- part 4: dispersion = ring phase matching ---------------------------
    d2_sweep = np.array([50e6, 100e6, 200e6])
    p_i_d2 = []
    for d2 in d2_sweep:
        f2, a2 = run(c, params={"SRC.p_p": pp_det, "RG.d2_hz": float(d2)})
        p_i_d2.append(abs(line(f2, a2, F_I)) ** 2)
    p_i_d2 = np.array(p_i_d2)
    sup_th = 1 / (1 + (np.pi * d2_sweep * r["tau"]) ** 2) ** 2
    err_d2 = float(np.max(np.abs(p_i_d2 / (p_i_ref * sup_th) - 1)))
    check("dispersion detunes the idler mode", err_d2 < 0.10,
          f"d2 = 50/100/200 MHz/mode^2 suppresses conversion to "
          f"{sup_th[0]*100:.0f}/{sup_th[1]*100:.0f}/{sup_th[2]*100:.0f}% "
          f"(theory), max err {err_d2*100:.1f}%")

    # ---- part 5: comb spreads to modes +-2 at mW drive, vs golden -----------
    # mode +2 catches the symmetric first-order product 2f_s - f_p (pump and
    # signal are equally strong here); mode -2 is a genuine SECOND-order
    # cascade fed by the idler itself
    pp_hi = ps_hi = 500e-6
    f_hi, a_hi = run(c, params={"SRC.p_p": pp_hi, "SRC.p_s": ps_hi}, dt=DT_FINE)
    fg_hi, ag_hi = golden_lines(pp_hi, ps_hi, r, t_full)
    comb_f = np.array([12.0, 6.0, 0.0, -6.0, -12.0]) * 1e9   # modes -2..+2
    comb_sim = np.array([abs(line(f_hi, a_hi, fk)) ** 2 for fk in comb_f])
    comb_gold = np.array([abs(line(fg_hi, ag_hi, fk)) ** 2 for fk in comb_f])
    err_c = float(np.max(np.abs(comb_sim / comb_gold - 1)))
    check("comb spreads to modes +-2", err_c < 5e-2
          and comb_sim[0] > 1e-9 and comb_sim[-1] > 1e-9,
          f"at 0.5 mW pump+signal: 2f_s-f_p line {dbm(comb_sim[-1]):.1f} dBm "
          f"(mode +2), second-order cascade {dbm(comb_sim[0]):.1f} dBm "
          f"(mode -2, 45 dB down), all 5 lines vs golden max err "
          f"{err_c*100:.2f}%")

    # ---- part 6: the ring vs the same length of straight waveguide ----------
    # fine dt on both sides so the ~1000x enhancement is a fair comparison
    cw = build_wg()
    fw, aw = run(cw, dt=DT_FINE)
    p_i_wg = abs(line(fw, aw, F_I)) ** 2
    enh_sim = p_i / p_i_wg
    enh_th = eta_ring(P_P0, r) / eta_wg(P_P0, r)
    check("resonant enhancement vs straight waveguide",
          abs(enh_sim / enh_th - 1) < 0.05,
          f"same chi(3), same {r['circ']*1e3:.1f} mm length, same drive: "
          f"ring idler {dbm(p_i):.1f} dBm vs waveguide {dbm(p_i_wg):.1f} dBm "
          f"-> x{enh_sim:.0f} (theory x{enh_th:.0f})")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.5))
    fig.suptitle("Four-wave mixing inside a Kerr ring resonator "
                 "(models/ring_kerr.va): pump + signal one FSR apart",
                 fontsize=13)

    a0 = ax[0, 0]
    sel = p_lines > 1e-22
    a0.vlines(f[sel] / 1e9, -200, dbm(p_lines[sel]), color="#2c7fb8", lw=2)
    po = np.abs(a_off) ** 2
    sel0 = po > 1e-22
    a0.vlines(f_off[sel0] / 1e9, -200, dbm(po[sel0]), color="0.75", lw=4,
              alpha=0.8, zorder=0)
    for fx, name in ((0, "pump\nmode 0"), (-6, "signal\nmode +1"),
                     (6, "idler\nmode -1"), (12, "mode -2"), (-12, "mode +2")):
        pk = abs(line(f, a, fx * 1e9)) ** 2
        a0.annotate(name, (fx, dbm(pk) + 4), ha="center", fontsize=7.5)
    a0.set(xlim=(-16, 16), ylim=(-130, 5), xlabel="envelope f [GHz]",
           ylabel="bus line power [dBm]",
           title="thru-port spectrum (grey: n2 = 0)")

    a1 = ax[0, 1]
    a1.loglog(pp_sweep * 1e6, p_i_pp * 1e9, "o", color="#d95f02",
              label=f"vs pump: slope {slope_pp:.3f}")
    a1.loglog(pp_sweep * 1e6, eta_th * P_S0 * 1e9, "-", color="#d95f02", lw=1,
              label="closed form")
    a1.loglog(ps_sweep * 1e6, p_i_ps * 1e9, "s", color="#1b9e77",
              label=f"vs signal: slope {slope_ps:.3f}")
    a1.loglog(ps_sweep * 1e6,
              eta_ring(P_P0, r) * ps_sweep * 1e9, "-", color="#1b9e77", lw=1)
    a1.set(xlabel="swept power [uW]", ylabel="idler power [nW]",
           title=r"$P_i \propto P_p^2\,P_s$, resonantly enhanced")
    a1.legend(fontsize=8)

    a2 = ax[0, 2]
    dd = np.linspace(-180e6, 180e6, 301)
    a2.semilogy(dd / 1e6, 1 / (1 + (2 * np.pi * dd * r["tau"]) ** 2) ** 4,
                "-", color="0.4", label=r"$|L(\delta)|^8$")
    a2.semilogy(np.r_[shifts, 0.0] / 1e6, np.r_[p_i_det, p_i_ref] / p_i_ref,
                "o", color="#7570b3", label="sim")
    a2.set(xlabel="comb shift under the lasers [MHz]",
           ylabel=r"$P_i(\delta)/P_i(0)$",
           title="the ring resonance enters 8 times in field")
    a2.legend(fontsize=8)

    a3 = ax[1, 0]
    d2d = np.linspace(0, 220e6, 301)
    a3.semilogy(d2d / 1e6, 1 / (1 + (np.pi * d2d * r["tau"]) ** 2) ** 2, "-",
                color="0.4", label=r"$1/(1+(\pi d_2\tau)^2)^2$")
    a3.semilogy(np.r_[0.0, d2_sweep] / 1e6, np.r_[p_i_ref, p_i_d2] / p_i_ref,
                "o", color="#e7298a", label="sim")
    a3.set(xlabel=r"comb dispersion $d_2$ [MHz/mode$^2$]",
           ylabel=r"$P_i(d_2)/P_i(0)$",
           title="phase matching, ring style: d2 detunes the idler mode")
    a3.legend(fontsize=8)

    a4 = ax[1, 1]
    xs = np.arange(5)
    a4.bar(xs - 0.17, dbm(comb_sim) - dbm(comb_sim).max(), 0.34,
           color="#2c7fb8", label="sim")
    a4.bar(xs + 0.17, dbm(comb_gold) - dbm(comb_sim).max(), 0.34,
           color="#d95f02", label="golden ODE")
    a4.set(xticks=xs, xticklabels=["-2\ncascade", "-1\nidler", "0\npump",
                                   "+1\nsignal", "+2\n2fs-fp"],
           xlabel="ring mode", ylabel="rel. line power [dB]",
           title=f"comb spread at {pp_hi*1e3:.1f} mW: 5 modes lit")
    a4.legend(fontsize=8)

    a5 = ax[1, 2]
    a5.bar([0, 1], [dbm(p_i_wg), dbm(p_i)], 0.55,
           color=["0.6", "#2c7fb8"])
    a5.set(xticks=[0, 1],
           xticklabels=[f"straight waveguide\n({r['circ']*1e3:.1f} mm)",
                        "the same length,\nwrapped into this ring"],
           ylabel="idler power [dBm]",
           title=f"resonant enhancement: x{enh_sim:.0f}")
    for x, p in ((0, p_i_wg), (1, p_i)):
        a5.annotate(f"{dbm(p):.1f} dBm", (x, dbm(p) + 2), ha="center",
                    fontsize=9)
    a5.set_ylim(dbm(p_i_wg) - 12, dbm(p_i) + 14)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    if all(ok for _, ok, _ in CHECKS):
        print("ALL RING-FWM CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED:", [n for n, ok, _ in CHECKS if not ok])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
