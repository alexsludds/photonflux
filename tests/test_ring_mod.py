"""models/optical_field/ring_mod.va — CMT microring modulator physics pins (circulax).

The model is parameterised by the physical device (radius, group index,
junction propagation loss, bus power coupling, pm/V tuning); these tests pin
the derived CMT steady-state Lorentzian

    |H|^2 = ((1 - tau*k^2)^2 + (tau*delta)^2) / (1 + (tau*delta)^2)

with 1/tau_i = alpha*v_g/2 and 1/tau_e = kappa2/(2*T_rt), lowered to JAX by
cx.va and solved by circulax. The full driver testbench (SKY130 inverter,
transient, eye diagram) lives in examples/ring_mod_sky130.py.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from circuit_helpers import build
from photonflux import cx

# the user-specified device
RADIUS_UM = 7.5
N_G = 4.0
LOSS_DB_M = 7000.0
KAPPA2 = 0.10
DL_DV_PM = 45.0
LAM_NM = 1310.0
P_IN = 1e-3

C0 = 2.99792458e8


def cmt(kappa2: float = KAPPA2) -> tuple[float, float]:
    """(tau, tau*kappa^2) from the physical device parameters."""
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    v_g = C0 / N_G
    t_rt = circ / v_g
    alpha = LOSS_DB_M * np.log(10) / 10
    inv_tau_i = alpha * v_g / 2
    inv_tau_e = kappa2 / (2 * t_rt)
    tau = 1 / (inv_tau_i + inv_tau_e)
    return tau, tau * 2 * inv_tau_e


def analytic_T(v: np.ndarray, kappa2: float = KAPPA2) -> np.ndarray:
    tau, tk2 = cmt(kappa2)
    lam = LAM_NM * 1e-9
    delta = 2 * np.pi * C0 * (1 / lam - 1 / (lam + DL_DV_PM * 1e-12 * v))
    return ((1 - tk2) ** 2 + (tau * delta) ** 2) / (1 + (tau * delta) ** 2)


def _sweep(kappa2: float, v: np.ndarray) -> np.ndarray:
    """Through-port |H|^2 over a tuning-voltage sweep, solved by circulax."""
    c = build(cx.va("ring_mod"),
              {"in_re": np.sqrt(P_IN), "in_im": 0.0, "vp": 0.0, "vn": 0.0},
              settings=dict(lambda_nm=LAM_NM, lambda_res_nm=LAM_NM,
                            radius_um=RADIUS_UM, n_g=N_G, loss_db_m=LOSS_DB_M,
                            kappa2=kappa2, dl_dv_pm=DL_DV_PM),
              reads=["out_re", "out_im"])
    y = c.dc(params={"SRC_vp.V": jnp.asarray(v)})
    return np.asarray((c.port(y, "out_re") ** 2 + c.port(y, "out_im") ** 2).real) / P_IN


def test_ring_lorentzian():
    v = np.linspace(-4.0, 4.0, 41)
    assert np.abs(_sweep(KAPPA2, v) - analytic_T(v)).max() < 1e-6


def test_ring_overcoupled_dip_floor():
    # kappa2 = 10% > critical (alpha*L = 7.6%): dip bottoms at (1 - tau*k^2)^2
    T = _sweep(KAPPA2, np.linspace(-4.0, 4.0, 41))
    _, tk2 = cmt(KAPPA2)
    assert T.min() == pytest.approx((1 - tk2) ** 2, abs=1e-6)
    assert 0.01 < T.min() < 0.03


def test_ring_critical_coupling_null():
    # kappa2 = alpha*L exactly -> perfect null on resonance. Round to the %g
    # precision the netlist carries so the analytic reference matches the model.
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    k2_crit = float(f"{LOSS_DB_M * np.log(10) / 10 * circ:g}")
    v = np.linspace(-4.0, 4.0, 41)
    T = _sweep(k2_crit, v)
    assert np.abs(T - analytic_T(v, k2_crit)).max() < 1e-6
    assert T.min() < 1e-9
