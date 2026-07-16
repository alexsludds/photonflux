"""models/optical_field/ring_selfheat.va — thermo-optic self-heating pins.

The self-heating all-pass ring is the CMT ring of ``ring_mod.va`` plus a lumped
one-pole thermal reservoir: the absorbed fraction of the circulating power heats
the ring, silicon's dn/dT>0 red-shifts the resonance, and that shift feeds back
into how much power the ring stores. Steady-state pins here (the transient
forward/backward hysteresis *demo* lives in ``examples/ring_selfheat.py``):

1. the **cold** limit (``heat_frac=0``): the through-port DC sweep collapses onto
   the exact all-pass Lorentzian of ``ring_mod.va`` (circulax, ``cx.va``);
2. the **steady-state self-consistency** having three roots (two stable branches
   + one unstable) over a non-empty wavelength window at high power, widening
   with power — the analytic signature of bistability (pure numpy).
"""
from __future__ import annotations

import numpy as np
import pytest

from circuit_helpers import build
from photonflux import cx

C0 = 2.99792458e8

# device — mirrors models/optical_field/ring_selfheat.va defaults
RADIUS_UM = 8.0
N_G = 4.0
LOSS_DB_M = 300.0
KAPPA2 = 0.004
RTH_K_W = 3.0e4
DL_DT_PM = 80.0
LAM_RES_NM = 1310.0


def cmt() -> tuple[float, float, float, float]:
    """(tau, tau*kappa^2, inv_tau_i, inv_tau_e) from the physical device."""
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    v_g = C0 / N_G
    t_rt = circ / v_g
    alpha = LOSS_DB_M * np.log(10) / 10
    inv_tau_i = alpha * v_g / 2
    inv_tau_e = KAPPA2 / (2 * t_rt)
    tau = 1 / (inv_tau_i + inv_tau_e)
    return tau, tau * 2 * inv_tau_e, inv_tau_i, inv_tau_e


def cold_T(lam_nm: np.ndarray) -> np.ndarray:
    """Cold all-pass Lorentzian at laser wavelength lam_nm (resonance fixed)."""
    tau, tk2, _, _ = cmt()
    delta = 2 * np.pi * C0 * (1 / (lam_nm * 1e-9) - 1 / (LAM_RES_NM * 1e-9))
    return ((1 - tk2) ** 2 + (tau * delta) ** 2) / (1 + (tau * delta) ** 2)


def steady_dt_roots(lam_nm: float, power_w: float, heat_frac: float = 1.0) -> list[float]:
    """All steady temperature-rise roots of the self-heating loop at one wavelength."""
    tau, tk2, inv_tau_i, inv_tau_e = cmt()
    coeff = RTH_K_W * heat_frac * (inv_tau_i / inv_tau_e)
    lam_l = lam_nm * 1e-9

    def g(dt: float) -> float:
        lam_res = LAM_RES_NM * 1e-9 + DL_DT_PM * 1e-12 * dt
        d = 2 * np.pi * C0 * (1 / lam_l - 1 / lam_res)
        A2 = tk2 ** 2 * power_w / (1 + (tau * d) ** 2)
        return dt - coeff * A2

    grid = np.linspace(0.0, coeff * tk2 ** 2 * power_w * 1.05 + 1e-12, 6000)
    gv = g(grid)
    roots = []
    for i in np.where(np.diff(np.sign(gv)) != 0)[0]:
        a, b = grid[i], grid[i + 1]
        for _ in range(60):
            m = 0.5 * (a + b)
            if g(a) * g(m) <= 0:
                b = m
            else:
                a = m
        roots.append(0.5 * (a + b))
    return roots


# ---------------------------------------------------------------------------
# 1. analytic bistability — pure numpy, no simulation engine
# ---------------------------------------------------------------------------
def test_selfheat_bistable_window_analytic():
    """At 50 uW a band of wavelengths has three steady roots (two stable
    branches + one unstable), and the band widens with power."""
    lams = np.linspace(LAM_RES_NM - 0.02, LAM_RES_NM + 0.4, 3000)

    def window_pm(power_w: float) -> float:
        n = np.array([len(steady_dt_roots(l, power_w)) for l in lams])
        assert n.max() == 3, "self-consistency should be cubic (<=3 roots)"
        band = lams[n >= 3]
        return (band.max() - band.min()) * 1e3 if band.size else 0.0

    w50 = window_pm(50e-6)
    w100 = window_pm(100e-6)
    assert w50 > 20.0, f"expected a clear bistable window at 50 uW, got {w50:.1f} pm"
    assert w100 > w50, "bistable window must widen with input power"


def test_selfheat_no_bistability_at_low_power():
    """Below threshold the loop is single-valued everywhere (one root)."""
    lams = np.linspace(LAM_RES_NM - 0.02, LAM_RES_NM + 0.05, 1500)
    n = np.array([len(steady_dt_roots(l, 1e-6)) for l in lams])  # 1 uW
    assert n.max() == 1, "1 uW should be monostable"


def test_selfheat_off_when_heat_frac_zero():
    """heat_frac=0 removes the feedback: exactly one root, DT=0, at any power."""
    for lam in (LAM_RES_NM, LAM_RES_NM + 0.05, LAM_RES_NM + 0.1):
        assert steady_dt_roots(lam, 100e-6, heat_frac=0.0) == pytest.approx([0.0], abs=1e-12)


# ---------------------------------------------------------------------------
# 2. cold limit matches ring_mod's all-pass Lorentzian (circulax / cx.va)
# ---------------------------------------------------------------------------
def test_selfheat_cold_lorentzian():
    """heat_frac=0 -> the through-port DC wavelength sweep is the exact cold
    Lorentzian (the laser wavelength is the `lam_nm` node, swept as SRC_lam_nm)."""
    import jax.numpy as jnp

    p_in = 1e-6
    c = build(
        cx.va("ring_selfheat"),
        {"in_re": np.sqrt(p_in), "in_im": 0.0, "lam_nm": LAM_RES_NM},
        settings=dict(lambda_res_nm=LAM_RES_NM, radius_um=RADIUS_UM, n_g=N_G,
                      loss_db_m=LOSS_DB_M, kappa2=KAPPA2, heat_frac=0.0,
                      rth_k_w=RTH_K_W, dl_dt_pm=DL_DT_PM),
        reads=["out_re", "out_im"],
        terms=[("out_re", "out_im")],
    )
    lam = np.linspace(LAM_RES_NM - 0.01, LAM_RES_NM + 0.01, 41)
    y = c.dc(params={"SRC_lam_nm.V": jnp.asarray(lam)})
    T = np.asarray((c.port(y, "out_re") ** 2 + c.port(y, "out_im") ** 2).real) / p_in
    assert np.abs(T - cold_T(lam)).max() < 1e-6
