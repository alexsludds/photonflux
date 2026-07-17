#!/usr/bin/env python3
"""Electro-optic frequency response of the traveling-wave MZM, pinned to its poles.

``models/optical_power/mzm_tw.va`` is a power-domain Mach-Zehnder modulator that,
on top of the quasi-static transfer ``T = IL*(0.5 + 0.5*eta*cos(pi*V/vpi))``, adds
the two effects that actually set a traveling-wave electrode's electro-optic (EO)
bandwidth. Both are imposed on the drive as cascaded single-pole roll-offs before
it acts on the optical wave (a reduced-order EO response):

    (1) electrode frequency-dependent loss    -> pole at  f_el   (+ optional f_el2)
    (2) optical/RF velocity walk-off           -> pole at  f_w = 0.443 / T_w,
                                                  T_w = |n_rf - n_opt| * len / c

so the small-signal EO transfer the model *should* realise is the pole cascade

    H(f) = 1/(1 + j f/f_el) . [1/(1 + j f/f_el2)] . 1/(1 + j f/f_w).

This testbench measures that H(f) the way a network analyser measures a real
modulator, with no knowledge of the poles: hold the optical input CW, bias the
electrode at **quadrature** (vbias = vpi/2, the steepest, most linear point of
the cos), drive a small RF sine V(vp,vn) = vbias + A.sin(2.pi.f.t), and read the
fundamental of the output optical power P_out(t) with an FFT. The measured EO
transfer is that fundamental per volt of drive, H_meas(f) = P_out(f) / V_drive(f).
The **EO -3 dB bandwidth** is where |H(f)/H(0)| = 1/sqrt(2) = -3.01 dB -- the same
point a photodiode reports 3 dB down in detected RF power (P_rf ~ |H|^2).

Everything about the bandwidth is emergent from the drive path; nothing reads
f_el or f_w back out of the model. The checks pin the measurement against the
analytic cascade and against the textbook scalings:

  1. pole cascade  -- |H(f)| AND phase match the two-pole (f_el + walk-off)
                      cascade across an 80 GHz sweep spanning 20 dB (VA -> JAX ->
                      BDF2 reproduces the intended reduced-order EO response)
  2. -3 dB BW      -- the measured -3 dB crossing sits on the analytic f_3dB of
                      the cascade (root-found, no fitting)
  3. velocity match-- n_rf = n_opt kills the walk-off pole (T_w -> 0): the response
                      collapses to a single pole at f_el, -3 dB lands exactly on
                      f_el, and the bandwidth jumps vs the mismatched electrode
  4. walk-off law  -- with the electrode made ~lossless (f_el -> inf) the lone
                      walk-off pole sets the BW; sweeping len gives f_3dB ~ 1/len
                      (slope -1) at the absolute value 0.443.c/(|n_rf-n_opt|.len)
  5. pole counting -- the high-frequency roll-off asymptote is -20 dB/decade per
                      active pole: 1 pole (matched) -> -20, 2 (default) -> -40,
                      3 (default + f_el2) -> -60
  6. small-signal  -- at mid-band the fundamental is linear in drive (slope 1) and
                      2nd-harmonic distortion stays >~40 dB down, so H_meas is the
                      genuine small-signal EO response, independent of drive level

    .venv-circulax/bin/python examples/mzm_tw_eo_bw.py
        -> out/mzm_tw_eo_bw.png
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

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source
from circulax.components.electronic import VoltageSource
from circulax.solvers.transient import BDF2VectorizedTransientSolver

from photonflux import cx

C0 = 2.99792458e8

# --- operating point --------------------------------------------------------
P_IN = 1e-3          # CW optical input power [W]
VPI = 1.5            # half-wave voltage [V] (model default)
VBIAS = VPI / 2.0    # quadrature: steepest, most linear point of the cos
AMP = 2e-3           # small-signal RF drive amplitude [V] (deeply linear)

# --- model default electrode ------------------------------------------------
LEN = 4e-3           # electrode length [m]
N_RF = 2.4           # microwave group index
N_OPT = 4.2          # optical group index
F_EL = 35e9          # electrode loss bandwidth [Hz]

# --- transient / FFT extraction (cost is independent of the tone frequency) --
SPP = 64             # samples per RF period
N_SETTLE = 6         # periods discarded so the pole states reach steady state
N_MEAS = 8           # periods FFT'd (tone lands exactly on bin N_MEAS)

OUT = Path(__file__).resolve().parents[1] / "out" / "mzm_tw_eo_bw.png"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ===========================================================================
# analytics -- the reduced-order EO transfer the model is meant to realise
# ===========================================================================
def f_walkoff(len_m: float = LEN, n_rf: float = N_RF, n_opt: float = N_OPT) -> float:
    """Walk-off pole f_w = 0.443/T_w, T_w = |n_rf-n_opt|.len/c (inf when matched)."""
    tw = abs(n_rf - n_opt) * len_m / C0
    return 0.443 / tw if tw > 0 else np.inf


def eo_transfer(f, *, f_el=F_EL, f_el2=0.0, len_m=LEN, n_rf=N_RF, n_opt=N_OPT):
    """H(f): cascade of the electrode-loss pole(s) and the walk-off pole."""
    f = np.asarray(f, dtype=float)
    h = 1.0 / (1.0 + 1j * f / f_el)
    if f_el2 > 0:
        h = h / (1.0 + 1j * f / f_el2)
    f_w = f_walkoff(len_m, n_rf, n_opt)
    if np.isfinite(f_w):
        h = h / (1.0 + 1j * f / f_w)
    return h


def db20(h) -> np.ndarray:
    return 20.0 * np.log10(np.abs(np.asarray(h)))


def f_3db(**kw) -> float:
    """Root-find the -3.01 dB (half-power) point of |H(f)/H(0)| by bisection."""
    target = -10.0 * np.log10(2.0)                 # -3.0103 dB
    lo, hi = 1e6, 1e13
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if db20(eo_transfer(mid, **kw))[()] > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ===========================================================================
# circuit -- CW optical input, quadrature electrode, small RF tone
# ===========================================================================
@source(ports=("p1", "p2"), states=("i_src",))
def RFDrive(signals: Signals, s: States, t: float,
            vbias: float = VBIAS, amp: float = AMP,
            freq: float = 1e9) -> tuple[dict, dict]:
    """Ideal differential electrode source: V(p1,p2) = vbias + amp.sin(2.pi.f.t).

    An ideal voltage source across the electrode, so the electrode pad
    capacitance ``cel`` only loads the (absent) driver and does not touch the
    EO transfer -- the measured H(f) is the modulator's own response.
    """
    v = vbias + amp * jnp.sin(2.0 * jnp.pi * freq * t)
    return {"p1": s.i_src, "p2": -s.i_src,
            "i_src": (signals.p1 - signals.p2) - v}, {}


def build():
    """Power-domain netlist: DC optical power on ``pin``, RF tone on the electrode.

    ``mzm_tw`` is power-domain (an optical node voltage *is* the optical power in
    watts), so the whole system is real: a DC source fixes V(pin) = P_in, the RF
    source drives the electrode, and V(pout) = T(t).P_in is read directly.
    """
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "PIN": {"component": "vdc", "settings": {"V": P_IN}},
            "RF": {"component": "rf"},
            "M": {"component": "mzm_tw"},
        },
        "connections": {
            "GND,p1": ("PIN,p2", "RF,p2", "M,gnd"),
            "PIN,p1": "M,pin",
            "RF,p1": "M,vp",
            "RF,p2": "M,vn",
        },
        "ports": {"pout": "M,pout", "vp": "M,vp", "vn": "M,vn"},
    }
    models = {
        "ground": lambda: 0,
        "vdc": VoltageSource,
        "rf": RFDrive,
        "mzm_tw": cx.va("mzm_tw"),
    }
    return compile_circuit(net, models, backend="dense",
                           is_complex=False, max_steps=200)


def run_tone(c, freq: float, params: dict | None = None,
             amp: float = AMP) -> tuple[np.ndarray, np.ndarray]:
    """One steady-state RF tone. Returns (P_out samples, V_drive samples) over
    exactly N_MEAS periods -- an integer number of periods for a leak-free FFT."""
    p = {"RF.freq": float(freq), "RF.amp": float(amp)}
    if params:
        p.update(params)
    dt = 1.0 / (freq * SPP)
    n = (N_SETTLE + N_MEAS) * SPP
    t_max = n * dt
    ts = jnp.arange(0.0, t_max, dt)
    y0 = c.dc(params=p)
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=dt, y0=y0, params=p,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=n + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    assert sol.result == diffrax.RESULTS.successful, f"transient failed: {sol.result}"
    pout = np.asarray(c.port(sol.ys, "pout").real)
    vd = (np.asarray(c.port(sol.ys, "vp").real)
          - np.asarray(c.port(sol.ys, "vn").real))
    m = N_MEAS * SPP
    return pout[-m:], vd[-m:]


def eo_line(c, freq: float, params: dict | None = None,
            amp: float = AMP) -> complex:
    """Measured EO transfer at ``freq``: output-power fundamental per volt of drive.

    The common DC slope factor (0.5.eta.pi/vpi . IL . P_in) cancels once the
    response is normalised, leaving the pure frequency dependence H(f)."""
    pout, vd = run_tone(c, freq, params, amp)
    m = len(pout)
    po = np.fft.rfft(pout) / m
    vv = np.fft.rfft(vd) / m
    return complex(po[N_MEAS] / vv[N_MEAS])


def harmonics(c, freq: float, amp: float) -> tuple[float, float]:
    """(fundamental, 2nd-harmonic) magnitudes of P_out at drive amplitude ``amp``."""
    pout, _ = run_tone(c, freq, amp=amp)
    m = len(pout)
    sp = np.abs(np.fft.rfft(pout) / m)
    return float(sp[N_MEAS]), float(sp[2 * N_MEAS])


def measure_bw(c, freqs: np.ndarray, params: dict | None = None) -> np.ndarray:
    """|H(f)| in dB, normalised to the lowest sweep frequency (|H| -> 1 there)."""
    lines = np.array([eo_line(c, float(f), params) for f in freqs])
    return db20(lines / lines[0]), lines


def cross_3db(freqs: np.ndarray, mag_db: np.ndarray) -> float:
    """Interpolate the -3.01 dB crossing of a monotonic roll-off (log-freq)."""
    target = -10.0 * np.log10(2.0)
    i = int(np.argmax(mag_db <= target))
    assert 0 < i < len(freqs), "sweep does not bracket the -3 dB point"
    lf = np.log10(freqs)
    frac = (target - mag_db[i - 1]) / (mag_db[i] - mag_db[i - 1])
    return float(10.0 ** (lf[i - 1] + frac * (lf[i] - lf[i - 1])))


# ===========================================================================
def main() -> int:
    c = build()
    f_w = f_walkoff()
    print(f"traveling-wave MZM: vpi = {VPI} V, len = {LEN*1e3:.0f} mm, "
          f"n_rf = {N_RF}, n_opt = {N_OPT}")
    print(f"  electrode-loss pole f_el = {F_EL/1e9:.1f} GHz, "
          f"walk-off pole f_w = {f_w/1e9:.2f} GHz "
          f"(T_w = {abs(N_RF-N_OPT)*LEN/C0*1e12:.1f} ps)")
    print(f"  bias at quadrature vbias = {VBIAS} V, drive amp = {AMP*1e3:.0f} mV\n")

    # ---- part 1: default EO response vs the two-pole cascade ---------------
    freqs = np.geomspace(0.3e9, 80e9, 30)
    mag_db, lines = measure_bw(c, freqs)
    h_an = eo_transfer(freqs)
    mag_an = db20(h_an / h_an[0])
    err_mag = float(np.max(np.abs(mag_db - mag_an)))
    check("|H(f)| matches f_el + walk-off cascade", err_mag < 0.1,
          f"max err {err_mag:.3f} dB over {mag_db[0]-mag_db[-1]:.0f} dB span, "
          f"30 tones 0.3-80 GHz")

    # phase, referenced to the drive line and to low frequency
    ph_meas = np.degrees(np.unwrap(np.angle(lines)))
    ph_meas -= ph_meas[0]
    ph_an = np.degrees(np.unwrap(np.angle(h_an)))
    ph_an -= ph_an[0]
    err_ph = float(np.max(np.abs(ph_meas - ph_an)))
    check("phase(H) matches -atan cascade", err_ph < 0.5,
          f"max err {err_ph:.3f} deg (reaches {ph_meas[-1]:.0f} deg at 80 GHz)")

    # ---- part 2: measured -3 dB bandwidth vs analytic ----------------------
    bw_meas = cross_3db(freqs, mag_db)
    bw_an = f_3db()
    check("-3 dB EO bandwidth", abs(bw_meas / bw_an - 1) < 0.01,
          f"measured {bw_meas/1e9:.2f} GHz vs analytic {bw_an/1e9:.2f} GHz "
          f"(below both poles -- the cascade)")

    # ---- part 3: velocity match kills the walk-off pole --------------------
    matched = {"M.n_rf": N_OPT}                    # n_rf = n_opt -> T_w = 0
    mag_m, _ = measure_bw(c, freqs, matched)
    mag_m_an = db20(eo_transfer(freqs, n_rf=N_OPT)
                    / eo_transfer(freqs[0], n_rf=N_OPT))
    err_m = float(np.max(np.abs(mag_m - mag_m_an)))
    bw_matched = cross_3db(freqs, mag_m)
    check("velocity match -> single pole at f_el", err_m < 0.1
          and abs(bw_matched / F_EL - 1) < 0.02,
          f"collapses to one pole: -3 dB at {bw_matched/1e9:.2f} GHz "
          f"(= f_el {F_EL/1e9:.0f} GHz), BW x{bw_matched/bw_meas:.2f} vs mismatched")

    # ---- part 4: walk-off pole scaling f_w ~ 1/len -------------------------
    # push the electrode loss pole far away so the lone walk-off pole sets the BW
    lossless = {"M.f_el": 5e12}
    lens_mm = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    bw_len, fw_th = [], []
    for lmm in lens_mm:
        p = dict(lossless, **{"M.len": float(lmm * 1e-3)})
        mag_l, _ = measure_bw(c, freqs, p)
        bw_len.append(cross_3db(freqs, mag_l))
        fw_th.append(f_3db(f_el=5e12, len_m=lmm * 1e-3))
    bw_len = np.array(bw_len)
    fw_th = np.array(fw_th)
    slope_len = np.polyfit(np.log(lens_mm), np.log(bw_len), 1)[0]
    err_fw = float(np.max(np.abs(bw_len / fw_th - 1)))
    check("walk-off BW ~ 1/len at 0.443.c/(|dn|.len)",
          abs(slope_len + 1) < 0.02 and err_fw < 0.02,
          f"f_3dB slope {slope_len:.3f} in len (theory -1); "
          f"absolute value within {err_fw*100:.1f}% of 0.443.c/(|dn|.len)")

    # ---- part 5: roll-off steepens by one pole; measured tracks the cascade -
    # Measured deep in the stopband (< -50 dB) sinks into the FFT floor, so the
    # slope is read in a strong-signal band (60-120 GHz) and pinned to the
    # analytic cascade there; the -20 dB/dec-per-pole asymptote is the pure
    # analytic limit (f -> inf), reported alongside.
    two_pole = {"M.f_el2": 60e9}
    f_sl = np.array([60e9, 120e9])
    dec = np.log10(f_sl[1] / f_sl[0])
    def meas_slope(params):
        m = np.array([db20(eo_line(c, float(f), params)) for f in f_sl])
        return float((m[1] - m[0]) / dec)
    def cascade_slope(**kw):
        m = db20(eo_transfer(f_sl, **kw))
        return float((m[1] - m[0]) / dec)
    def asymptote(**kw):                            # f -> inf, pure analytic
        fa = np.array([1e12, 2e12])
        m = db20(eo_transfer(fa, **kw))
        return float((m[1] - m[0]) / np.log10(2))
    s1, c1 = meas_slope(matched), cascade_slope(n_rf=N_OPT)          # 1 pole
    s2, c2 = meas_slope(None), cascade_slope()                       # 2 poles
    s3, c3 = meas_slope(two_pole), cascade_slope(f_el2=60e9)         # 3 poles
    asy = (asymptote(n_rf=N_OPT), asymptote(), asymptote(f_el2=60e9))
    err_sl = max(abs(s1 - c1), abs(s2 - c2), abs(s3 - c3))
    check("roll-off tracks the cascade, steepens per pole",
          err_sl < 0.5 and s1 > s2 > s3,
          f"60-120 GHz slope 1/2/3 poles = {s1:.1f}/{s2:.1f}/{s3:.1f} dB/dec "
          f"(cascade {c1:.1f}/{c2:.1f}/{c3:.1f}); asymptote "
          f"{asy[0]:.0f}/{asy[1]:.0f}/{asy[2]:.0f} = -20/pole")

    # ---- part 6: small-signal linearity + low distortion -------------------
    f_lin = 10e9
    amps = np.array([0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3])
    fund, h2 = [], []
    for a in amps:
        f1, f2 = harmonics(c, f_lin, float(a))
        fund.append(f1)
        h2.append(f2)
    fund = np.array(fund)
    h2 = np.array(h2)
    slope_lin = np.polyfit(np.log(amps), np.log(fund), 1)[0]
    hd2_db = float(20 * np.log10(h2[amps == AMP][0] / fund[amps == AMP][0]))
    check("small-signal: fundamental linear, HD2 low",
          abs(slope_lin - 1) < 0.01 and hd2_db < -40,
          f"fundamental slope {slope_lin:.4f} in drive (theory 1); "
          f"HD2 {hd2_db:.1f} dB at {AMP*1e3:.0f} mV")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.5))
    fig.suptitle("Electro-optic frequency response of the traveling-wave MZM "
                 "(models/optical_power/mzm_tw.va)", fontsize=13)
    tgt = -10 * np.log10(2)

    a0 = ax[0, 0]
    ff = np.geomspace(0.3e9, 80e9, 400)
    a0.semilogx(ff / 1e9, db20(eo_transfer(ff) / eo_transfer(0.3e9)),
                "-", color="#2c7fb8", lw=1.4, label="cascade (analytic)")
    a0.semilogx(freqs / 1e9, mag_db, "o", ms=5, color="#2c7fb8",
                label="measured (VA->JAX->BDF2)")
    a0.axhline(tgt, color="0.6", ls=":", lw=1)
    a0.axvline(bw_meas / 1e9, color="#d95f02", ls="--", lw=1,
               label=f"-3 dB = {bw_meas/1e9:.1f} GHz")
    a0.set(xlabel="RF frequency [GHz]", ylabel="|H(f)| [dB]",
           ylim=(-24, 3), title="EO magnitude response")
    a0.legend(fontsize=8)
    a0.grid(alpha=0.3, which="both")

    a1 = ax[0, 1]
    ph_ff = np.degrees(np.unwrap(np.angle(eo_transfer(ff))))
    ph_ff -= ph_ff[0]                               # reference to low frequency
    a1.semilogx(ff / 1e9, ph_ff, "-", color="#7570b3", lw=1.4,
                label="cascade (analytic)")
    a1.semilogx(freqs / 1e9, ph_meas, "o", ms=5, color="#7570b3",
                label="measured")
    a1.set(xlabel="RF frequency [GHz]", ylabel="phase(H) [deg]",
           title="EO phase response")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3, which="both")

    a2 = ax[0, 2]
    a2.semilogx(freqs / 1e9, mag_db, "o-", ms=4, color="#2c7fb8",
                label=f"default: -3 dB {bw_meas/1e9:.1f} GHz")
    a2.semilogx(freqs / 1e9, mag_m, "s-", ms=4, color="#1b9e77",
                label=f"velocity matched: {bw_matched/1e9:.0f} GHz")
    mag_2p, _ = measure_bw(c, freqs, {"M.f_el2": 60e9})
    a2.semilogx(freqs / 1e9, mag_2p, "^-", ms=4, color="#d95f02",
                label="+ 2nd loss pole (60 GHz)")
    a2.axhline(tgt, color="0.6", ls=":", lw=1)
    a2.set(xlabel="RF frequency [GHz]", ylabel="|H(f)| [dB]", ylim=(-24, 3),
           title="walk-off & extra pole reshape the BW")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, which="both")

    a3 = ax[1, 0]
    a3.loglog(lens_mm, bw_len / 1e9, "o", ms=7, color="#d95f02",
              label=f"measured -3 dB (slope {slope_len:.3f})")
    a3.loglog(lens_mm, fw_th / 1e9, "-", color="#d95f02", lw=1,
              label=r"$0.443\,c/(|n_{rf}-n_{opt}|\,\ell)$")
    a3.set(xlabel="electrode length [mm]", ylabel="walk-off -3 dB BW [GHz]",
           title=r"walk-off pole $f_w \propto 1/\ell$ (loss pole removed)")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3, which="both")

    a4 = ax[1, 1]
    xlbl = ["1 pole\n(matched)", "2 poles\n(default)", "3 poles\n(+f_el2)"]
    xpos = np.arange(3)
    a4.bar(xpos - 0.2, [-s1, -s2, -s3], 0.4, color="#2c7fb8", label="measured")
    a4.bar(xpos + 0.2, [-c1, -c2, -c3], 0.4, color="#bdd7e7", label="cascade")
    for x, ay in zip(xpos, asy):                    # analytic f->inf asymptote
        a4.plot([x - 0.4, x + 0.4], [-ay, -ay], color="#d95f02", lw=1.6)
    a4.plot([], [], color="#d95f02", lw=1.6, label="asymptote (-20/pole)")
    a4.set_xticks(xpos)
    a4.set_xticklabels(xlbl, fontsize=8)
    a4.set(ylabel="roll-off at 60-120 GHz [-dB/decade]",
           title="roll-off steepens by one pole each")
    a4.legend(fontsize=8)

    a5 = ax[1, 2]
    a5.loglog(amps * 1e3, fund * 1e3, "o", ms=7, color="#2c7fb8",
              label=f"fundamental (slope {slope_lin:.3f})")
    a5.loglog(amps * 1e3, fund[0] * 1e3 * (amps / amps[0]), "-",
              color="#2c7fb8", lw=1, label="ideal slope 1")
    a5.loglog(amps * 1e3, h2 * 1e3, "s", ms=6, color="#d95f02",
              label="2nd harmonic (HD2)")
    a5.set(xlabel="RF drive amplitude [mV]",
           ylabel=r"$P_{out}$ line amplitude [mW]",
           title=f"small-signal EO @ {f_lin/1e9:.0f} GHz (HD2 {hd2_db:.0f} dB)")
    a5.legend(fontsize=8)
    a5.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nwrote {OUT}")

    if all(ok for _, ok, _ in CHECKS):
        print("ALL EO-BANDWIDTH CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED:", [n for n, ok, _ in CHECKS if not ok])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
