"""models/ring_mod.va — CMT microring modulator physics pins.

The model is parameterised by the physical device (radius, group index,
junction propagation loss, bus power coupling, pm/V tuning); these tests pin
the derived CMT behaviour in both engines — ngspice (OSDI) and circulax
(bosdi JAX via cx.va) — against the analytic steady-state Lorentzian

    |H|^2 = ((1 - tau*k^2)^2 + (tau*delta)^2) / (1 + (tau*delta)^2)

with 1/tau_i = alpha*v_g/2 and 1/tau_e = kappa2/(2*T_rt).

The full driver testbench (SKY130 inverter, transient, eye diagram) lives in
examples/ring_mod_sky130.py.
"""
from __future__ import annotations

import numpy as np
import pytest

import lightspice as ls

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


def _ngspice_sweep(kappa2: float) -> tuple[np.ndarray, np.ndarray]:
    ckt = ls.Circuit("ring dc")
    # steep Lorentzian slopes need better than ngspice's reltol=1e-3 default
    ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
    ckt.device(ls.va("ring_mod"), "ring",
               "ein_re", "ein_im", "eout_re", "eout_im", "vp", "0", "0",
               lambda_nm=LAM_NM, lambda_res_nm=LAM_NM,
               radius_um=RADIUS_UM, n_g=N_G, loss_db_m=LOSS_DB_M,
               kappa2=kappa2, dl_dv_pm=DL_DV_PM)
    ckt.raw(f"Vinre ein_re 0 {np.sqrt(P_IN)}", "Vinim ein_im 0 0", "Vt vp 0 0")
    r = ls.Engine().run(ckt, "dc vt -4 4 0.2")
    T = (r["eout_re"] ** 2 + r["eout_im"] ** 2) / P_IN
    return r["v-sweep"], T


def test_ring_lorentzian_ngspice():
    v, T = _ngspice_sweep(KAPPA2)
    assert np.abs(T - analytic_T(v)).max() < 1e-9


def test_ring_overcoupled_dip_floor_ngspice():
    # kappa2 = 10% > critical (alpha*L = 7.6%): dip bottoms at (1 - tau*k^2)^2
    _, T = _ngspice_sweep(KAPPA2)
    _, tk2 = cmt(KAPPA2)
    assert T.min() == pytest.approx((1 - tk2) ** 2, abs=1e-6)
    assert 0.01 < T.min() < 0.03


def test_ring_critical_coupling_null_ngspice():
    # kappa2 = alpha*L exactly -> perfect null on resonance. Round to the %g
    # precision the netlist carries so the analytic reference sees the same
    # value the model does.
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    k2_crit = float(f"{LOSS_DB_M * np.log(10) / 10 * circ:g}")
    v, T = _ngspice_sweep(k2_crit)
    assert np.abs(T - analytic_T(v, k2_crit)).max() < 1e-9
    assert T.min() < 1e-12


def test_ring_circulax_matches_ngspice():
    pytest.importorskip("circulax")
    pytest.importorskip("bosdi")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from circulax import compile_circuit
    from circulax.components.electronic import VoltageSource

    from lightspice import cx

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VRE": {"component": "vsrc", "settings": {"V": float(np.sqrt(P_IN))}},
            "VIM": {"component": "vsrc", "settings": {"V": 0.0}},
            "VT": {"component": "vsrc", "settings": {"V": 0.0}},
            "RING": {"component": "ring",
                     "settings": {"lambda_nm": LAM_NM, "lambda_res_nm": LAM_NM,
                                  "radius_um": RADIUS_UM, "n_g": N_G,
                                  "loss_db_m": LOSS_DB_M, "kappa2": KAPPA2,
                                  "dl_dv_pm": DL_DV_PM}},
        },
        "connections": {
            "GND,p1": ("VRE,p2", "VIM,p2", "VT,p2", "RING,vn", "RING,gnd"),
            "VRE,p1": "RING,in_re",
            "VIM,p1": "RING,in_im",
            "VT,p1": "RING,vp",
        },
        "ports": {"ore": "RING,out_re", "oim": "RING,out_im"},
    }
    c = compile_circuit(
        net, {"ground": lambda: 0, "vsrc": VoltageSource, "ring": cx.va("ring_mod")},
        backend="dense",
    )
    v = np.linspace(-4.0, 4.0, 25)
    y = c.dc(params={"VT.V": jnp.asarray(v)})
    T = np.asarray((c.port(y, "ore") ** 2 + c.port(y, "oim") ** 2).real) / P_IN
    assert np.abs(T - analytic_T(v)).max() < 1e-6
