#!/usr/bin/env python3
"""Nonlinear absorption capping the quality factor of a high-Q silicon ring.

``models/optical_field/ring_nl.va`` is an all-pass microring whose INTRINSIC loss grows
with the stored field: two-photon absorption of the circulating intensity,
plus free-carrier absorption from the TPA-generated carrier population.
Resonant enhancement makes this brutal — at Q ~ 1e6 a milliwatt in the bus
becomes ~100 mW circulating, and the extra loss collapses the very Q that
produced the enhancement:

    1/tau_i(U, N) = 1/tau_i0 + beta*v_g^2*U/(2*V_eff) + sigma*N*v_g/2
    tau_fc dN/dt  = tau_fc * beta*I_circ^2/(2*h*f0) - N

Three self-checking parts (all DC sweeps of the probe wavelength, vectorised
over the lambda parameter; dispersion terms off for parts 1-2 so the line
stays symmetric and the FWHM -> Q map is clean):

1. **Spectra vs input power** — the resonance dip of one device fills in and
   broadens as P_in rises; every curve is pinned against an independent
   numpy root-solve of the same steady-state equations.
2. **Q_loaded vs P_in for three intrinsic-loss devices** (30/100/300 dB/m,
   all 1.4x overcoupled). The punchline: above ~0.1 mW the TPA+FCA ceiling —
   not the fabrication loss — sets the achievable Q, and the three curves
   converge.
3. **Free-carrier dispersion back on** — the same device at rising power
   with Kerr (red) and FCD (blue) shifts enabled: the line pulls blue and
   goes asymmetric, the familiar high-Q silicon resonance shark-fin.

    .venv-circulax/bin/python examples/ring_tpa_q.py

        -> out/ring_tpa_q.png
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from circulax import compile_circuit

from _cavity import port_power, terminator
from photonflux import cx

C0 = 2.99792458e8
HPL = 6.62607015e-34

LAM0_NM = 1310.0
BASE = dict(lambda_res_nm=LAM0_NM, radius_um=10.0, n_g=4.0,
            a_eff_um2=0.1, beta_tpa=8e-12, sigma_fca=1.45e-21, tau_fc=1e-9,
            n2_kerr=0.0, dn_dn=0.0)      # dispersion off for parts 1-2
OVERCOUPLE = 1.4                          # kappa2 = 1.4 * alpha * L, all devices
LOSSES_DB_M = (30.0, 100.0, 300.0)        # three fabrication qualities
POWERS = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)   # bus input power [W]

OUT = Path(__file__).resolve().parents[1] / "out" / "ring_tpa_q.png"


# ===========================================================================
# analytic steady state (independent numpy root-solve of the model equations)
# ===========================================================================
def device(loss_db_m: float) -> dict:
    circ = 2 * np.pi * BASE["radius_um"] * 1e-6
    alpha = loss_db_m * np.log(10) / 10
    d = dict(BASE, loss_db_m=loss_db_m, kappa2=OVERCOUPLE * alpha * circ)
    v_g = C0 / d["n_g"]
    t_rt = circ / v_g
    d["_inv_ti"] = alpha * v_g / 2
    d["_inv_te"] = d["kappa2"] / (2 * t_rt)
    d["_k2c"] = 2 * d["_inv_te"]
    d["_veff"] = d["a_eff_um2"] * 1e-12 * circ
    d["_vg"] = v_g
    d["_eph"] = HPL * C0 / (LAM0_NM * 1e-9)
    d["_q_lin"] = (2 * np.pi * C0 / (LAM0_NM * 1e-9)
                   / (2 * (d["_inv_ti"] + d["_inv_te"])))
    return d


def inv_tau_tot(u: float, d: dict) -> float:
    icirc = u * d["_vg"] / d["_veff"]
    nfc = d["tau_fc"] * d["beta_tpa"] * icirc**2 / (2 * d["_eph"])
    return (d["_inv_ti"] + d["_inv_te"]
            + d["beta_tpa"] * d["_vg"]**2 * u / (2 * d["_veff"])
            + d["sigma_fca"] * nfc * d["_vg"] / 2)


def t_analytic(delta: np.ndarray, p_in: float, d: dict) -> np.ndarray:
    """|T|^2 with the stored energy solved self-consistently per point."""
    t = np.empty_like(delta)
    u_max = d["_k2c"] * p_in / (d["_inv_ti"] + d["_inv_te"])**2
    for i, dl in enumerate(delta):
        # root in x = U/U_max: brentq's absolute xtol would swamp U ~ 1e-15 J
        f = lambda x: x * u_max * (inv_tau_tot(x * u_max, d)**2 + dl**2) \
            - d["_k2c"] * p_in
        u = u_max * brentq(f, 0.0, 1.001, xtol=1e-15, rtol=1e-14)
        it = inv_tau_tot(u, d)
        t[i] = abs(1 - d["_k2c"] * (it + 1j * dl) / (it**2 + dl**2))**2
    return t


def fwhm_of(delta: np.ndarray, t: np.ndarray) -> float:
    half = 0.5 * (1 + t.min())
    below = np.where(t < half)[0]
    return delta[below[-1]] - delta[below[0]]        # [rad/s]


def q_from_sweep(delta: np.ndarray, t: np.ndarray) -> float:
    w0 = 2 * np.pi * C0 / (LAM0_NM * 1e-9)
    return w0 / fwhm_of(delta, t)


# ===========================================================================
# VA sweeps (vectorised DC over the probe wavelength)
# ===========================================================================
def build(d: dict, p_in: float):
    settings = {k: v for k, v in d.items() if not k.startswith("_")}
    settings["lambda_nm"] = LAM0_NM
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LAS": {"component": "laser", "settings": {"power": p_in}},
            "TAP": {"component": "f2ri"},
            "RG": {"component": "ring", "settings": settings},
            "TO": {"component": "term"},
        },
        "connections": {
            "LAS,p1": "TAP,c",
            "TAP,re": "RG,in_re", "TAP,im": "RG,in_im",
            "RG,out_re": "TO,re", "RG,out_im": "TO,im",
            "GND,p1": ("LAS,p2", "RG,gnd"),
        },
        "ports": {"po_re": "RG,out_re", "po_im": "RG,out_im"},
    }
    models = {"ground": lambda: 0, "laser": cx.cw_laser(),
              "f2ri": cx.field_to_ri(), "ring": cx.va("ring_nl"),
              "term": terminator()}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def va_sweep(d: dict, p_in: float, span: float, n: int = 361):
    """DC transmission vs detuning delta = w_l - w_res in [-span, span]."""
    delta = np.linspace(-span, span, n)
    nu0 = C0 / (LAM0_NM * 1e-9)
    lam_nm = C0 / (nu0 + delta / (2 * np.pi)) * 1e9
    c = build(d, p_in)
    y = c.dc(params={"RG.lambda_nm": jnp.asarray(lam_nm)})
    return delta, port_power(c, y, "po_re", "po_im") / p_in


def va_sweep_continuation(d: dict, p_in: float, span: float, n: int = 241):
    """Red -> blue sweep for the dispersion-on (bistable) line.

    Once FCD pulls the line over, the DC problem is multivalued: a
    cold-started Newton lands on either branch (or, at the fold where the
    Jacobian is singular, silently on neither — any true steady state of
    this passive ring has T <= 1, so T > 1 flags a died solve). And each
    scalar-parameter ``c.dc`` re-jits (~0.7 s), so a naive point-by-point
    continuation is also slow. Hybrid:

    1. one vectorised cold solve over the whole sweep (fast, correct
       wherever the solution is unique),
    2. warm-started continuation only across the resonance window,
       entering from the red side — the swept-laser measurement,
    3. at a fold failure, reseed from the blue neighbour's cold solution;
       a point that survives neither is dropped as NaN (it sits exactly on
       the singular fold).
    """
    delta = np.linspace(span, -span, n)          # red -> blue
    nu0 = C0 / (LAM0_NM * 1e-9)
    lam_nm = C0 / (nu0 + delta / (2 * np.pi)) * 1e9
    c = build(d, p_in)
    y_all = c.dc(params={"RG.lambda_nm": jnp.asarray(lam_nm)})
    t = np.asarray(port_power(c, y_all, "po_re", "po_im")) / p_in

    act = np.where((t < 0.998) | (t > 1.0 + 1e-9))[0]
    if len(act):
        i0 = max(int(act.min()) - 8, 0)
        i1 = min(int(act.max()) + 8, n - 1)
        y = y_all[i0]
        for i in range(i0, i1 + 1):
            y = c.dc(y_guess=y, params={"RG.lambda_nm": float(lam_nm[i])},
                     rtol=1e-10, atol=1e-12)
            ti = float(port_power(c, y, "po_re", "po_im")) / p_in
            if ti > 1.0 + 1e-9:
                y = c.dc(y_guess=y_all[min(i + 1, n - 1)],
                         params={"RG.lambda_nm": float(lam_nm[i])},
                         rtol=1e-10, atol=1e-12)
                ti = float(port_power(c, y, "po_re", "po_im")) / p_in
                if ti > 1.0 + 1e-9:
                    ti = np.nan          # the singular fold point itself
                    y = y_all[min(i + 1, n - 1)]
            t[i] = ti
    return delta[::-1], t[::-1]


def span_for(d: dict, p_in: float) -> float:
    """Sweep +-3 broadened half-widths (estimate the width analytically)."""
    coarse = np.linspace(-60, 60, 41) * (d["_inv_ti"] + d["_inv_te"]) / 10
    return 3.0 * fwhm_of(coarse, t_analytic(coarse, p_in, d))


# ===========================================================================
def main() -> int:
    devs = {ldb: device(ldb) for ldb in LOSSES_DB_M}
    for ldb, d in devs.items():
        print(f"device {ldb:5.0f} dB/m: kappa2 = {d['kappa2']:.2e}, "
              f"linear Q_loaded = {d['_q_lin']:.3g}")

    # ---- parts 1+2: spectra and Q(P) for the three devices ------------------
    q_va = {ldb: [] for ldb in LOSSES_DB_M}
    q_ref = {ldb: [] for ldb in LOSSES_DB_M}
    spectra = {}                       # (ldb, p) -> (delta, t_va, t_ref)
    for ldb, d in devs.items():
        for p in POWERS:
            span = span_for(d, p)
            delta, t_va = va_sweep(d, p, span)
            t_ref = t_analytic(delta, p, d)
            err = np.abs(t_va - t_ref).max()
            assert err < 1e-4, f"{ldb} dB/m at {p} W: VA vs analytic {err:.2e}"
            q_va[ldb].append(q_from_sweep(delta, t_va))
            q_ref[ldb].append(q_from_sweep(delta, t_ref))
            if ldb == LOSSES_DB_M[0]:
                spectra[p] = (delta, t_va, t_ref)
        q_va[ldb] = np.asarray(q_va[ldb])
        q_ref[ldb] = np.asarray(q_ref[ldb])
        print(f"  {ldb:5.0f} dB/m: Q {q_va[ldb][0]:.3g} at {POWERS[0]*1e3:g} mW "
              f"-> {q_va[ldb][-1]:.3g} at {POWERS[-1]*1e3:g} mW")

    # checks: linear limit, agreement with analytic, monotone collapse,
    # and convergence of the three devices onto the nonlinear ceiling
    for ldb, d in devs.items():
        assert abs(q_va[ldb][0] - d["_q_lin"]) / d["_q_lin"] < 0.05
        assert np.abs(q_va[ldb] / q_ref[ldb] - 1).max() < 0.05
        # falling with power (2% slack: FWHM is read off a finite sweep grid,
        # and the first decade barely moves the highest-loss device)
        assert np.all(np.diff(q_va[ldb]) < 0.02 * q_va[ldb][:-1])
        assert q_va[ldb][-1] < 0.65 * q_va[ldb][0], "no Q collapse?"
    r_lo = q_va[LOSSES_DB_M[0]][0] / q_va[LOSSES_DB_M[-1]][0]
    r_hi = q_va[LOSSES_DB_M[0]][-1] / q_va[LOSSES_DB_M[-1]][-1]
    print(f"Q(best)/Q(worst): {r_lo:.2f} at low power -> {r_hi:.2f} at "
          f"{POWERS[-1]*1e3:g} mW (TPA+FCA ceiling, not fabrication, wins)")
    assert r_lo > 8.0 and r_hi < 3.0
    print("ALL TESTBENCH CHECKS PASSED")

    # ---- part 3: dispersion back on — the asymmetric pulled line ------------
    d_disp = device(LOSSES_DB_M[0])
    d_disp.update(n2_kerr=4.5e-18, dn_dn=-4e-27)
    disp = {}
    for p in (1e-6, 3e-4, 1e-3):
        span = 2.5 * span_for(devs[LOSSES_DB_M[0]], p)
        disp[p] = va_sweep_continuation(d_disp, p, span)
        t_d = disp[p][1]
        assert np.nanmax(t_d) < 1.0 + 1e-6, "passive ring must not gain"
        assert np.isnan(t_d).sum() <= 2, "more than the fold point failed"

    # ---- plot ---------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2)
    w0 = 2 * np.pi * C0 / (LAM0_NM * 1e-9)

    ax = fig.add_subplot(gs[0, 0])
    cm = plt.cm.plasma(np.linspace(0.05, 0.85, len(POWERS)))
    for p, c_ in zip(POWERS, cm):
        delta, t_va, t_ref = spectra[p]
        pm = -delta / w0 * LAM0_NM * 1e3     # WAVELENGTH detuning in pm
        ax.plot(pm, t_va, c=c_, label=f"{p*1e3:g} mW")
        ax.plot(pm, t_ref, "k:", lw=0.7)
    ax.set_xlabel("wavelength detuning [pm]"); ax.set_ylabel("|T|²")
    ax.set_title(f"{LOSSES_DB_M[0]:.0f} dB/m ring: the dip fills in and "
                 "broadens\n(dots: analytic steady state)", fontsize=9)
    ax.legend(fontsize=7, title="P_in", title_fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for (ldb, d), c_ in zip(devs.items(), ("tab:blue", "tab:green", "tab:red")):
        ax.loglog(np.asarray(POWERS) * 1e3, q_va[ldb], "o-", c=c_,
                  label=f"{ldb:.0f} dB/m (linear Q = {d['_q_lin']:.2g})")
        ax.loglog(np.asarray(POWERS) * 1e3, q_ref[ldb], "k:", lw=0.8)
    ax.set_xlabel("bus input power [mW]"); ax.set_ylabel("loaded Q (measured FWHM)")
    ax.set_title("the TPA+FCA ceiling: fabrication quality stops mattering\n"
                 "(dots: analytic)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, :])
    for (p, (delta, t_va)), c_ in zip(disp.items(),
                                      ("tab:blue", "tab:orange", "tab:red")):
        pm = -delta / w0 * LAM0_NM * 1e3     # wavelength convention: blue < 0
        ax.plot(pm, t_va, c=c_, label=f"{p*1e3:g} mW")
    ax.set_xlabel("wavelength detuning from the COLD resonance [pm]")
    ax.set_ylabel("|T|²")
    ax.set_title("free-carrier dispersion on: the line pulls blue and goes "
                 "asymmetric (Kerr red + FCD blue; swept red -> blue with "
                 "warm-started continuation)", fontsize=9)
    ax.legend(fontsize=8, title="P_in", title_fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("TPA + free-carrier absorption limiting a high-Q silicon "
                 "microring (models/optical_field/ring_nl.va, circulax)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
