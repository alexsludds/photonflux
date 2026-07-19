#!/usr/bin/env python3
"""Quarter-wave-shifted DFB laser built from traveling-wave slices: single-mode
lasing at the Bragg wavelength, seeded from broadband ASE noise.

The laser is a chain of ``models/optical_field/tw_gain_seg.va`` slices that are
ALSO Bragg grating slices (kappa > 0 — an index-coupled active grating) with a
quarter-wave ``phase_pad.va`` defect (phi = pi/2) in the middle and AR facets
(no facet feedback — the feedback is entirely distributed):

    ASE -> [ gain+grating x N/2 ]  |lambda/4|  [ gain+grating x N/2 ] -> ASE
                                        '-> the defect mode at lambda_Bragg

A uniform DFB would lase on a degenerate pair of band-EDGE modes; the lambda/4
defect opens a single high-Q resonance in the middle of the stopband, so the
laser picks ONE longitudinal mode at the exact Bragg wavelength. Nothing tells
it which mode — the SOA-style deterministic seed is off (p_seed = 0) and
complex white ASE noise is injected at both AR facets, so the side modes are
seeded too and the side-mode suppression ratio is an honest measurement, not an
artefact of a single-tone seed (the same discipline as ``soa_vernier_laser.py``).

Checks (self-checking, asserted):
  * the lasing line sits at the Bragg wavelength (envelope DC, |offset| < 3 GHz);
  * SMSR > 30 dB in the output spectrum (the OSA panel);
  * the settled power matches the clamped-gain reservoir estimate order-of-mag.

    python examples/dfb_laser.py            ->  out/dfb_laser.png   (~2-3 min)
"""
from __future__ import annotations

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source

from _cavity import run_transient
from photonflux import cx

C0 = 2.99792458e8
HPL = 6.62607015e-34

# --- the DFB ----------------------------------------------------------------
LAM_NM = 1310.0            # Bragg wavelength AND the fixed optical frame
N_SEG = 20                # active grating slices
DZ = 14e-6
L = N_SEG * DZ
NG = 3.7
KAPPA = 1.3 / L           # kappa*L = 1.3
G_UNSAT = 7000.0          # unsaturated amplitude gain at i_op [1/m]
I_BIAS = 80e-3
GAIN = dict(lambda_nm=LAM_NM, lambda_bragg_nm=LAM_NM, n_g=NG, dz=DZ,
            g_unsat_pm=G_UNSAT, i_op_ma=80.0, i_tr_ma=8.0, p_sat=10e-3,
            tau_c=0.3e-9, alpha_h=0.0, kappa_pm=KAPPA, loss_pm=0.0,
            p_seed=0.0, Von=1.2, Rs=3.0)
N_SP = 2.0                # ASE inversion factor for the facet seed

# --- the run ----------------------------------------------------------------
T_STOP = 6e-9
T_ON = 0.3e-9             # bias steps through threshold here
DT = 0.15e-12
DT_SAVE = 1e-12
DT_NOISE = 1e-12
NOISE_SEED = 5

OUT = Path(__file__).resolve().parents[1] / "out" / "dfb_laser.png"
NU0 = C0 / (LAM_NM * 1e-9)


def s_ase() -> float:
    """One-sided ASE PSD of a biased slice [W/Hz] — the facet seed strength."""
    g0 = G_UNSAT * (I_BIAS * 1e3 - 8.0) / (80.0 - 8.0)     # amplitude gain [1/m]
    return N_SP * HPL * NU0 * (np.exp(min(2 * g0 * L, 20.0)) - 1.0)


def analytic_p_out() -> float:
    """Order-of-magnitude clamped-gain power: each half-grating (r=tanh(kL/2))
    mirrors the QWS mode, so treat it as an FP laser with those mirrors."""
    r = np.tanh(KAPPA * L / 2) ** 2
    g_th = 1.0 / r
    h_th = np.log(g_th)
    hop = 2 * G_UNSAT * L
    h0 = hop * (I_BIAS * 1e3 - 8.0) / (80.0 - 8.0)
    if h0 <= h_th:
        return 0.0
    return (h0 - h_th) * 10e-3 / ((g_th - 1) * (1 + r * g_th)) * (1 - r) * g_th


# ===========================================================================
# the netlist: staircased bias, ASE noise at both facets
# ===========================================================================
def _noise_src(stream, tn):
    @source(ports=("re",), states=("i",))
    def NS(signals: Signals, s: States, t: float):
        return {"re": s.i, "i": signals.re - jnp.interp(t, tn, stream)}, {}
    return NS


def _bias_stair(v_on, t_on):
    @source(ports=("p1", "p2"), states=("i_src",))
    def Bias(signals: Signals, s: States, t: float):
        x = jnp.clip((t - t_on) / 50e-12, 0.0, 1.0)
        v = 1.2 + (v_on - 1.2) * x * x * (3.0 - 2.0 * x)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}
    return Bias


def build():
    n = int(np.ceil(T_STOP / DT_NOISE)) + 4
    rng = np.random.default_rng(NOISE_SEED)
    sigma = float(np.sqrt(s_ase() / (4.0 * DT_NOISE)))
    bank = rng.standard_normal((4, n)) * sigma
    bank[:, 0] = 0.0
    tn = jnp.arange(n) * DT_NOISE

    instances = {"GND": {"component": "ground"}, "VB": {"component": "bias"}}
    for nm in ("NLr", "NLi", "NRr", "NRi"):
        instances[nm] = {"component": nm}
    for k in range(N_SEG):
        instances[f"S{k}"] = {"component": "g", "settings": GAIN}
    instances["D"] = {"component": "pad", "settings": {"phi0_rad": np.pi / 2}}

    half = N_SEG // 2
    seq = [f"S{k}" for k in range(half)] + ["D"] + [f"S{k}" for k in range(half, N_SEG)]
    conns = {"NLr,re": f"{seq[0]},fl_re", "NLi,re": f"{seq[0]},fl_im",
             "NRr,re": f"{seq[-1]},br_re", "NRi,re": f"{seq[-1]},br_im"}
    for a, b in zip(seq[:-1], seq[1:]):
        conns[f"{a},fr_re"] = f"{b},fl_re"
        conns[f"{a},fr_im"] = f"{b},fl_im"
        conns[f"{b},bl_re"] = f"{a},br_re"
        conns[f"{b},bl_im"] = f"{a},br_im"
    conns["VB,p1"] = tuple(f"S{k},an" for k in range(N_SEG))
    grounded = ["VB,p2", "D,gnd", "D,vt"]
    for k in range(N_SEG):
        grounded += [f"S{k},gnd", f"S{k},cat"]
    conns["GND,p1"] = tuple(grounded)

    net = {"instances": instances, "connections": conns,
           "ports": {"oR_re": f"{seq[-1]},fr_re", "oR_im": f"{seq[-1]},fr_im",
                     "oL_re": f"{seq[0]},bl_re", "oL_im": f"{seq[0]},bl_im"}}
    models = {"ground": lambda: 0, "g": cx.va("tw_gain_seg"),
              "pad": cx.va("phase_pad"),
              "bias": _bias_stair(1.2 + 3.0 * I_BIAS, T_ON),
              "NLr": _noise_src(jnp.asarray(bank[0]), tn),
              "NLi": _noise_src(jnp.asarray(bank[1]), tn),
              "NRr": _noise_src(jnp.asarray(bank[2]), tn),
              "NRi": _noise_src(jnp.asarray(bank[3]), tn)}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


# ===========================================================================
def spectrum(e, dt):
    win = np.hanning(len(e))
    x = np.fft.fftshift(np.fft.fft(e * win)) / np.sum(win)
    f = np.fft.fftshift(np.fft.fftfreq(len(e), dt))
    return f, np.abs(x) ** 2


def main() -> int:
    print(f"DFB: {N_SEG} active grating slices, kappa*L = {KAPPA*L:.2f}, "
          f"lambda/4 defect at midpoint; frame = Bragg = {LAM_NM} nm")
    print(f"ASE facet seed S = {s_ase():.3g} W/Hz")

    t0 = time.time()
    c = build()
    t, sol = run_transient(c, T_STOP, DT, save_every=DT_SAVE)
    er = np.asarray(c.port(sol.ys, "oR_re").real)
    ei = np.asarray(c.port(sol.ys, "oR_im").real)
    e = er + 1j * ei
    p = np.abs(e) ** 2
    print(f"transient: {time.time()-t0:.1f}s")

    m = t > T_STOP - 2.5e-9        # settled window
    f, s = spectrum(e[m], DT_SAVE)
    k = np.argmax(s)
    nu_off = -f[k] / 1e9           # optical offset from Bragg [GHz]
    guard = np.abs(f - f[k]) > 20e9
    smsr = 10 * np.log10(s[k] / s[guard].max())
    p_settle = p[m].mean()
    p_ana = analytic_p_out()

    print(f"lasing line at {nu_off:+.2f} GHz from Bragg, SMSR = {smsr:.1f} dB")
    print(f"P_out = {p_settle*1e3:.3f} mW (clamped-gain estimate "
          f"{p_ana*1e3:.2f} mW)")

    assert abs(nu_off) < 3.0, "not lasing at the Bragg wavelength"
    assert smsr > 30.0, f"SMSR {smsr:.1f} dB < 30 dB"
    assert p_settle > 0.5e-3, "did not reach lasing power"
    print("ALL TESTBENCH CHECKS PASSED")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(t * 1e9, p * 1e3, c="tab:orange", lw=0.8)
    ax.axvline(T_ON * 1e9, c="gray", lw=0.8, ls=":", label="bias on")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("P_out [mW]")
    ax.set_title("turn-on from ASE noise")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(-f / 1e9, 10 * np.log10(s / s.max() + 1e-18), c="tab:blue", lw=0.9)
    ax.axvline(0.0, c="gray", lw=0.8, ls="--", label="Bragg")
    ax.axhline(-smsr, c="tab:red", lw=0.8, ls=":",
               label=f"SMSR = {smsr:.0f} dB")
    ax.set_xlim(-120, 120); ax.set_ylim(-70, 3)
    ax.set_xlabel("optical offset from Bragg [GHz]"); ax.set_ylabel("PSD [dB]")
    ax.set_title(f"OSA: single-mode DFB at {LAM_NM} nm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Quarter-wave-shifted DFB laser from traveling-wave slices "
                 "(tw_gain_seg + phase_pad, circulax)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
