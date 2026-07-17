"""The microring modulator, decomposed into coherent-field sub-components.

``cx.ring_modulator`` wires three temporal coupled-mode building blocks —
``cx.directional_coupler`` + ``cx.ring_phase_shifter`` + ``cx.cavity_mode``,
sharing one internal cavity node — into a ring that is a drop-in coherent-field
twin of the monolithic ``models/optical_field/ring_mod.va``. These tests pin
the steady state two ways:

1. the decomposed through-port DC sweep collapses onto the exact CMT Lorentzian
   ``|H|^2 = ((1 - tau*k^2)^2 + (tau*delta)^2) / (1 + (tau*delta)^2)``;
2. the decomposed ring reproduces ring_mod.va's through-port *field* to machine
   precision at every bias — block diagram == VA model.

The photon-lifetime *dynamics* the decomposition preserves (which the adiabatic
limit would lose) are exercised in the transient bench ``examples/eo_comb.py``.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component
from circulax.components.electronic import VoltageSource

from photonflux import cx

# the user-specified device, matching tests/test_ring_mod.py
RADIUS_UM = 7.5
N_G = 4.0
LOSS_DB_M = 7000.0
KAPPA2 = 0.10
DL_DV_PM = 45.0
LAM_NM = 1310.0
P_IN = 1e-3

C0 = 2.99792458e8

SETTINGS = dict(lambda_nm=LAM_NM, lambda_res_nm=LAM_NM, radius_um=RADIUS_UM,
                n_g=N_G, n_eff=2.4, loss_db_m=LOSS_DB_M, kappa2=KAPPA2,
                dl_dv_pm=DL_DV_PM)


def _field_terminator():
    """Single complex-node termination that draws nothing (open circuit)."""

    @component(ports=("c",))
    def FieldTerminator(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"c": 0.0}, {}

    return FieldTerminator


def _base(settings: dict) -> tuple[dict, dict, dict]:
    """The shared CW-laser -> ring -> terminator scaffold (minus the ring)."""
    inst = {
        "GND": {"component": "ground"},
        "LAS": {"component": "laser",
                "settings": {"wavelength_nm": settings["lambda_nm"],
                             "power": P_IN}},
        "VT": {"component": "vsrc", "settings": {"V": 0.0}},
        "TO": {"component": "term"},
    }
    models = {"ground": lambda: 0, "laser": cx.cw_laser(),
              "vsrc": VoltageSource, "term": _field_terminator()}
    return inst, {}, models


def _decomposed(settings: dict):
    """CW laser -> [coupler | phase shifter | cavity] ring -> through port."""
    inst, conn, models = _base(settings)
    ring = cx.ring_modulator(settings)
    inst.update(ring.instances)
    models.update(ring.models)
    conn.update(ring.connections)
    conn["LAS,p1"] = ring.sin
    conn[ring.sout] = "TO,c"
    conn["VT,p1"] = ring.vp
    conn["GND,p1"] = ("LAS,p2", ring.vn, "VT,p2")
    net = {"instances": inst, "connections": conn, "ports": {"sout": ring.sout}}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def _monolithic(settings: dict):
    """The same ring as the single ring_mod.va block — the reference twin."""
    inst, conn, models = _base(settings)
    inst.update({
        "TAP": {"component": "f2ri"},
        "RING": {"component": "ring", "settings": settings},
        "JOIN": {"component": "ri2f"},
    })
    models.update({"f2ri": cx.field_to_ri(), "ring": cx.va("ring_mod"),
                   "ri2f": cx.ri_to_field()})
    conn.update({
        "LAS,p1": "TAP,c",
        "TAP,re": "RING,in_re", "TAP,im": "RING,in_im",
        "RING,out_re": "JOIN,re", "RING,out_im": "JOIN,im",
        "JOIN,c": "TO,c",
        "VT,p1": "RING,vp",
        "GND,p1": ("LAS,p2", "RING,vn", "RING,gnd", "VT,p2"),
    })
    net = {"instances": inst, "connections": conn, "ports": {"sout": "JOIN,c"}}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def _through_field(c, v: np.ndarray) -> np.ndarray:
    """Through-port complex field s_out(V) over a tuning-voltage sweep."""
    y = c.dc(params={"VT.V": jnp.asarray(v)})
    return np.asarray(c.port(y, "sout"))


def _analytic_T(v: np.ndarray) -> np.ndarray:
    rates = cx.ring_cmt_rates(radius_um=RADIUS_UM, n_g=N_G, loss_db_m=LOSS_DB_M,
                              kappa2=KAPPA2)
    tau = 1.0 / (rates["inv_tau_e"] + rates["inv_tau_i"])
    tk2 = tau * 2.0 * rates["inv_tau_e"]
    lam = LAM_NM * 1e-9
    delta = 2 * np.pi * C0 * (1 / lam - 1 / (lam + DL_DV_PM * 1e-12 * v))
    return ((1 - tk2) ** 2 + (tau * delta) ** 2) / (1 + (tau * delta) ** 2)


def test_decomposition_matches_lorentzian():
    """Through-port |H|^2 of the composed ring == the analytic CMT Lorentzian."""
    v = np.linspace(-4.0, 4.0, 41)
    e = _through_field(_decomposed(SETTINGS), v)
    T = np.abs(e) ** 2 / P_IN
    assert np.abs(T - _analytic_T(v)).max() < 1e-6


def test_decomposition_matches_ring_mod_va():
    """Block diagram == VA model: identical through-port field at every bias."""
    v = np.linspace(-4.0, 4.0, 41)
    e_dec = _through_field(_decomposed(SETTINGS), v)
    e_mono = _through_field(_monolithic(SETTINGS), v)
    assert np.max(np.abs(e_dec - e_mono)) < 1e-9


def test_ring_cmt_rates_match_ring_mod_va():
    """The rate split reproduces ring_mod.va's derivation (and kappa^2=2/tau_e)."""
    rates = cx.ring_cmt_rates(radius_um=RADIUS_UM, n_g=N_G, loss_db_m=LOSS_DB_M,
                              kappa2=KAPPA2)
    circ = 2 * np.pi * RADIUS_UM * 1e-6
    v_g = C0 / N_G
    t_rt = circ / v_g
    alpha = LOSS_DB_M * np.log(10) / 10
    assert rates["inv_tau_e"] == pytest.approx(KAPPA2 / (2 * t_rt), rel=1e-12)
    assert rates["inv_tau_i"] == pytest.approx(alpha * v_g / 2, rel=1e-12)
    assert rates["cj"] == pytest.approx(0.5 * 1e-15 * circ * 1e6, rel=1e-12)
