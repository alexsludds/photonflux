"""models/optical_field/tw_seg.va, tw_gain_seg.va, phase_pad.va — traveling-wave
multi-section laser slices (circulax).

These pin the *linear, static* physics of the section models that the DFB / DBR
/ FP laser examples are built from — the parts an analytic reference nails
exactly, solved by a plain DC operating point (no lasing bifurcation):

  * a Bragg grating of M cascaded tw_seg slices reflects |r| = tanh(kappa*L) at
    the Bragg frame and traces the coupled-mode stopband |r(delta)|^2;
  * a tw_gain_seg slice is transparent at its transparency current and its
    single-slice power gain is the discrete stencil 1/(1 - gamma*dz)^2;
  * a quarter-wave (phi = pi/2) phase_pad defect between two grating halves
    opens a transmission resonance at the exact Bragg wavelength (the DFB mode).

The emergent lasing testbenches (threshold, slope efficiency, SMSR, mode hop)
live in examples/tw_fp_laser.py, examples/dfb_laser.py, examples/dbr_laser.py.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component
from circulax.components.electronic import VoltageSource

from photonflux import cx

C0 = 2.99792458e8
LAM = 1310.0
NG = 3.7


def _term():
    @component(ports=("re", "im"))
    def T(signals: Signals, s: States):
        return {"re": 0.0, "im": 0.0}, {}
    return T


# ---------------------------------------------------------------------------
# a passive Bragg grating from M cascaded tw_seg slices
# ---------------------------------------------------------------------------
def _grating(m: int, dz: float, kappa: float, lam: float, defect_phi=None):
    """Forward-driven grating; returns compiled circuit with refl/thru ports.

    defect_phi (rad), if given, inserts a phase_pad in the middle (a QWS-DFB).
    """
    seg = dict(lambda_nm=lam, lambda_bragg_nm=LAM, n_g=NG, dz=dz,
               kappa_pm=kappa, gamma_pm=0.0)
    insts = {"GND": {"component": "ground"},
             "FR": {"component": "vsrc", "settings": {"V": 1.0}},
             "FI": {"component": "vsrc", "settings": {"V": 0.0}},
             "TL": {"component": "term"}, "TT": {"component": "term"}}
    names = [f"S{k}" for k in range(m)]
    for n in names:
        insts[n] = {"component": "seg", "settings": seg}
    seq = list(names)
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "seg": cx.va("tw_seg"), "term": _term()}
    if defect_phi is not None:
        insts["D"] = {"component": "pad", "settings": {"phi0_rad": defect_phi}}
        models["pad"] = cx.va("phase_pad")
        seq = names[: m // 2] + ["D"] + names[m // 2:]

    conns = {"FR,p1": f"{seq[0]},fl_re", "FI,p1": f"{seq[0]},fl_im",
             f"{seq[0]},bl_re": "TL,re", f"{seq[0]},bl_im": "TL,im",
             f"{seq[-1]},fr_re": "TT,re", f"{seq[-1]},fr_im": "TT,im"}
    for a, b in zip(seq[:-1], seq[1:]):
        conns[f"{a},fr_re"] = f"{b},fl_re"
        conns[f"{a},fr_im"] = f"{b},fl_im"
        conns[f"{b},bl_re"] = f"{a},br_re"
        conns[f"{b},bl_im"] = f"{a},br_im"
    grounded = ["FR,p2", "FI,p2", f"{seq[-1]},br_re", f"{seq[-1]},br_im"]
    for n in names:
        grounded += [f"{n},gnd", f"{n},vt"]
    if defect_phi is not None:
        grounded += ["D,gnd", "D,vt"]
    conns["GND,p1"] = tuple(grounded)
    net = {"instances": insts, "connections": conns,
           "ports": {"rf_re": f"{seq[0]},bl_re", "rf_im": f"{seq[0]},bl_im",
                     "tr_re": f"{seq[-1]},fr_re", "tr_im": f"{seq[-1]},fr_im"}}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def _refl(c, lam):
    y = c.dc()
    return (complex(c.port(y, "rf_re")).real ** 2
            + complex(c.port(y, "rf_im")).real ** 2)


def _analytic_R(delta, kappa, L):
    S = np.lib.scimath.sqrt(kappa * kappa - delta * delta)
    r = -1j * kappa * np.sinh(S * L) / (S * np.cosh(S * L)
                                        + 1j * delta * np.sinh(S * L))
    return abs(r) ** 2


def test_grating_bragg_reflectivity():
    """At the Bragg frame the M-slice grating reflects tanh(kappa*L)^2."""
    M, dz, kappa = 200, 1.5e-6, 4000.0
    L = M * dz
    R = _refl(_grating(M, dz, kappa, LAM), LAM)
    assert R == pytest.approx(np.tanh(kappa * L) ** 2, rel=0.03)


def test_grating_stopband_shape():
    """The reflection spectrum traces the coupled-mode stopband near the band:
    peak at Bragg, first nulls and side lobes at the analytic detunings."""
    M, dz, kappa = 200, 1.5e-6, 4000.0
    L = M * dz
    for dlam in (0.0, 0.4, 0.8):        # nm from Bragg, inside/near the band
        lam = LAM + dlam
        delta = 2 * np.pi * NG * (1 / (lam * 1e-9) - 1 / (LAM * 1e-9))
        R = _refl(_grating(M, dz, kappa, lam), lam)
        assert abs(R - _analytic_R(delta, kappa, L)) < 0.05


# ---------------------------------------------------------------------------
# a single active gain slice
# ---------------------------------------------------------------------------
def _gain_slice_gain(i_ma, dz=30e-6, g_unsat=8000.0, p_in=1e-6):
    seg = dict(n_g=NG, dz=dz, g_unsat_pm=g_unsat, i_op_ma=80.0, i_tr_ma=8.0,
               p_sat=10e-3, kappa_pm=0.0, p_seed=0.0)
    v = 1.2 + 3.0 * i_ma * 1e-3
    insts = {"GND": {"component": "ground"},
             "FR": {"component": "vsrc", "settings": {"V": float(np.sqrt(p_in))}},
             "FI": {"component": "vsrc", "settings": {"V": 0.0}},
             "VB": {"component": "vsrc", "settings": {"V": v}},
             "TT": {"component": "term"}}
    insts["S"] = {"component": "g", "settings": seg}
    conns = {"FR,p1": "S,fl_re", "FI,p1": "S,fl_im", "VB,p1": "S,an",
             "S,fr_re": "TT,re", "S,fr_im": "TT,im",
             "GND,p1": ("FR,p2", "FI,p2", "VB,p2", "S,gnd", "S,cat",
                        "S,bl_re", "S,bl_im", "S,br_re", "S,br_im")}
    net = {"instances": insts, "connections": conns,
           "ports": {"o_re": "S,fr_re", "o_im": "S,fr_im"}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "g": cx.va("tw_gain_seg"), "term": _term()}
    c = compile_circuit(net, models, backend="dense", is_complex=True,
                        max_steps=300)
    y = c.dc()
    return (complex(c.port(y, "o_re")).real ** 2
            + complex(c.port(y, "o_im")).real ** 2) / p_in


def test_gain_slice_transparent_at_itr():
    assert _gain_slice_gain(8.0) == pytest.approx(1.0, abs=1e-3)


def test_gain_slice_discrete_stencil():
    """Small-signal single-slice power gain is the upwind stencil 1/(1-g*dz)^2."""
    dz, g_unsat, i_ma = 30e-6, 8000.0, 50.0
    g0 = g_unsat * (i_ma - 8.0) / (80.0 - 8.0)          # amplitude gain [1/m]
    assert _gain_slice_gain(i_ma, dz, g_unsat) == pytest.approx(
        1.0 / (1.0 - g0 * dz) ** 2, rel=1e-3)


# ---------------------------------------------------------------------------
# the quarter-wave defect (DFB mode)
# ---------------------------------------------------------------------------
def test_qws_defect_transmission_peak():
    """A pi/2 phase_pad between two grating halves opens a sharp transmission
    resonance at the exact Bragg wavelength (the single DFB lasing mode); the
    uniform grating (no defect) instead reflects there."""
    M, dz, kappa = 40, 7.2e-6, 1.3 / (40 * 7.2e-6)
    t_defect = _grating(M, dz, kappa, LAM, defect_phi=np.pi / 2)
    y = t_defect.dc()
    T = (complex(t_defect.port(y, "tr_re")).real ** 2
         + complex(t_defect.port(y, "tr_im")).real ** 2)
    R = (complex(t_defect.port(y, "rf_re")).real ** 2
         + complex(t_defect.port(y, "rf_im")).real ** 2)
    assert T > 0.9 and R < 0.05           # defect mode: transmits, does not reflect
    # the same grating with NO defect reflects at Bragg (stopband)
    R_uniform = _refl(_grating(M, dz, kappa, LAM), LAM)
    assert R_uniform > 0.6
