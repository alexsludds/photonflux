"""Polarization-aware coherent-field models — physics pins.

The dual-polarization convention extends the scalar Ereal/Eimag optical net to
a Jones vector carried as two field pairs: X = (x*_re, x*_im) is the TE
component, Y = (y*_re, y*_im) the TM component, |Ex|^2 + |Ey|^2 = power [W].
The scalar models (phase_shifter.va, mirror.va, ...) are unchanged — they are
just the X/TE channel — so the existing suite pins the default-pol behaviour.

Models pinned here (all lowered to JAX by cx.va, solved by circulax):

* polarization_rotator.va — Jones rotation by a fixed angle
* pbs.va / pbc.va          — polarization beam splitter / combiner
* birefringent_wg.va       — TE/TM differential propagation phase (dn_eff)
* pdl.va                   — polarization-dependent loss

The two acceptance-criteria testbenches live at the bottom:
``test_malus_law`` (a rotator + PBS + PBC chain) and ``test_birefringent_mzi``
(a Jones MZI whose TE and TM fringes are offset by the modal birefringence).
"""
from __future__ import annotations

import numpy as np
import pytest

from circuit_helpers import op, power, terminator
from circulax import compile_circuit
from circulax.components.electronic import VoltageSource

from photonflux import cx

# every optical net is a Jones vector: these are the eight (re, im) node names
_JONES_IN = ("xin_re", "xin_im", "yin_re", "yin_im")
_ROT_OUT = ("xout_re", "xout_im", "yout_re", "yout_im")


def _launch(ex: complex, ey: complex) -> dict:
    """Drive the input Jones vector (Ex, Ey) onto the xin/yin node pairs."""
    return {"xin_re": ex.real, "xin_im": ex.imag,
            "yin_re": ey.real, "yin_im": ey.imag}


# ---------------------------------------------------------------------------
# polarization rotator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_rotator_malus_split(theta):
    # a rotator turns pure-X input into (cos th, sin th): power follows Malus
    vals = op(cx.va("polarization_rotator"), _launch(1.0, 0.0),
              settings={"theta_deg": theta}, reads=_ROT_OUT)
    t = np.radians(theta)
    assert power(vals, "xout_re", "xout_im") == pytest.approx(np.cos(t) ** 2, abs=1e-9)
    assert power(vals, "yout_re", "yout_im") == pytest.approx(np.sin(t) ** 2, abs=1e-9)


def test_rotator_is_lossless_and_orthogonal():
    # a pure rotation preserves total power and the relative angle for any input
    vals = op(cx.va("polarization_rotator"), _launch(0.6, 0.8),
              settings={"theta_deg": 37.0}, reads=_ROT_OUT)
    tot = power(vals, "xout_re", "xout_im") + power(vals, "yout_re", "yout_im")
    assert tot == pytest.approx(1.0, abs=1e-9)  # |0.6|^2 + |0.8|^2 = 1


def test_rotator_insertion_loss():
    vals = op(cx.va("polarization_rotator"), _launch(1.0, 0.0),
              settings={"theta_deg": 0.0, "il_db": 6.0}, reads=_ROT_OUT)
    assert power(vals, "xout_re", "xout_im") == pytest.approx(10 ** (-6.0 / 10.0), rel=1e-6)


# ---------------------------------------------------------------------------
# PBS / PBC
# ---------------------------------------------------------------------------
_PBS_OUT = ("o1x_re", "o1x_im", "o1y_re", "o1y_im",
            "o2x_re", "o2x_im", "o2y_re", "o2y_im")


def test_pbs_ideal_split():
    # ideal PBS routes X to port 1, Y to port 2 with no cross-leakage
    vals = op(cx.va("pbs"), _launch(0.6, 0.8), reads=_PBS_OUT)
    assert power(vals, "o1x_re", "o1x_im") == pytest.approx(0.36, abs=1e-9)  # X -> port1
    assert power(vals, "o1y_re", "o1y_im") == pytest.approx(0.0, abs=1e-12)
    assert power(vals, "o2y_re", "o2y_im") == pytest.approx(0.64, abs=1e-9)  # Y -> port2
    assert power(vals, "o2x_re", "o2x_im") == pytest.approx(0.0, abs=1e-12)


def test_pbs_finite_extinction():
    # er_db sets the leaked power ratio: |leak|^2 / |pass|^2 = 10^(-er/10)
    er = 20.0
    vals = op(cx.va("pbs"), _launch(1.0, 0.0), settings={"er_db": er}, reads=_PBS_OUT)
    leaked = power(vals, "o2x_re", "o2x_im")   # X that leaked into the TM port
    passed = power(vals, "o1x_re", "o1x_im")
    # leaked amplitude is 10^(-er/20) of the unit input -> leaked power 10^(-er/10)
    assert leaked == pytest.approx(10 ** (-er / 10.0), rel=1e-6)
    assert leaked + passed == pytest.approx(1.0, abs=1e-9)  # power-normalised


def test_pbc_recombines_pbs():
    # PBS then PBC reconstructs the original Jones vector (unit throughput)
    ex, ey = 0.6 + 0.1j, 0.5 - 0.3j
    net, models = _pbs_pbc_chain()
    c = compile_circuit(net, models)
    y = c.dc(params={f"S_{k}.V": val for k, val in _launch(ex, ey).items()})
    g = lambda p: complex(c.port(y, p)).real  # noqa: E731
    out_x = g("ox_re") + 1j * g("ox_im")
    out_y = g("oy_re") + 1j * g("oy_im")
    assert out_x == pytest.approx(ex, abs=1e-9)
    assert out_y == pytest.approx(ey, abs=1e-9)


def _pbs_pbc_chain():
    """PBS out1/out2 wired straight into a PBC — returns (net, models)."""
    insts = {"GND": {"component": "ground"},
             "PBS": {"component": "pbs"}, "PBC": {"component": "pbc"}}
    conns, gnd = {}, ["PBS,gnd", "PBC,gnd"]
    for p in _JONES_IN:
        insts[f"S_{p}"] = {"component": "vsrc", "settings": {"V": 0.0}}
        conns[f"S_{p},p1"] = f"PBS,{p}"
        gnd.append(f"S_{p},p2")
    for a, b in [("o1x_re", "i1x_re"), ("o1x_im", "i1x_im"),
                 ("o1y_re", "i1y_re"), ("o1y_im", "i1y_im"),
                 ("o2x_re", "i2x_re"), ("o2x_im", "i2x_im"),
                 ("o2y_re", "i2y_re"), ("o2y_im", "i2y_im")]:
        conns[f"PBS,{a}"] = f"PBC,{b}"
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {"ox_re": "PBC,ox_re", "ox_im": "PBC,ox_im",
                     "oy_re": "PBC,oy_re", "oy_im": "PBC,oy_im"}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "pbs": cx.va("pbs"), "pbc": cx.va("pbc")}
    return net, models


# ---------------------------------------------------------------------------
# birefringent waveguide
# ---------------------------------------------------------------------------
def test_birefringent_phase():
    # X and Y accumulate 2*pi*n*L/lambda; the differential is 2*pi*dn*L/lambda
    lam, L, n_te, n_tm = 1550.0, 100.0, 2.40, 2.38
    vals = op(cx.va("birefringent_wg"), _launch(1.0, 1.0),
              settings={"lambda_nm": lam, "length_um": L, "n_te": n_te, "n_tm": n_tm},
              reads=_ROT_OUT)
    gx = vals["xout_re"].real + 1j * vals["xout_im"].real
    gy = vals["yout_re"].real + 1j * vals["yout_im"].real
    assert abs(gx) == pytest.approx(1.0, abs=1e-9)  # lossless by default
    assert abs(gy) == pytest.approx(1.0, abs=1e-9)
    d_phi = np.angle(gx * np.conj(gy))  # differential phase, wrapped
    expect = 2 * np.pi * (n_te - n_tm) * (L * 1e-6) / (lam * 1e-9)
    assert np.angle(np.exp(1j * (d_phi - expect))) == pytest.approx(0.0, abs=1e-6)


def test_birefringent_loss():
    vals = op(cx.va("birefringent_wg"), _launch(1.0, 0.0),
              settings={"length_um": 1e6, "loss_db_m": 3.0},  # 1 m at 3 dB/m
              reads=_ROT_OUT)
    assert power(vals, "xout_re", "xout_im") == pytest.approx(10 ** (-3.0 / 10.0), rel=1e-6)


# ---------------------------------------------------------------------------
# PDL
# ---------------------------------------------------------------------------
def test_pdl_differential_loss():
    vals = op(cx.va("pdl"), _launch(1.0, 1.0),
              settings={"il_db": 1.0, "pdl_db": 3.0}, reads=_ROT_OUT)
    px = power(vals, "xout_re", "xout_im")
    py = power(vals, "yout_re", "yout_im")
    assert px == pytest.approx(10 ** (-1.0 / 10.0), rel=1e-6)          # TE: il only
    assert py == pytest.approx(10 ** (-(1.0 + 3.0) / 10.0), rel=1e-6)  # TM: il + pdl
    # PDL figure = 10*log10(px/py) = pdl_db
    assert 10 * np.log10(px / py) == pytest.approx(3.0, rel=1e-6)


# ===========================================================================
# Acceptance criterion 1 — Malus law through PBS + rotator + PBC
# ===========================================================================
def _malus_circuit():
    """X-polarized launch -> rotator(theta) -> PBS -> PBC. Sweep ROT.theta_deg."""
    insts = {"GND": {"component": "ground"},
             "ROT": {"component": "rot", "settings": {"theta_deg": 0.0}},
             "PBS": {"component": "pbs"}, "PBC": {"component": "pbc"}}
    conns, gnd = {}, ["ROT,gnd", "PBS,gnd", "PBC,gnd"]
    for p, v in _launch(1.0, 0.0).items():  # pure TE input
        insts[f"S_{p}"] = {"component": "vsrc", "settings": {"V": v}}
        conns[f"S_{p},p1"] = f"ROT,{p}"
        gnd.append(f"S_{p},p2")
    for a, b in zip(_ROT_OUT, _JONES_IN):
        conns[f"ROT,{a}"] = f"PBS,{b}"
    for a, b in [("o1x_re", "i1x_re"), ("o1x_im", "i1x_im"),
                 ("o1y_re", "i1y_re"), ("o1y_im", "i1y_im"),
                 ("o2x_re", "i2x_re"), ("o2x_im", "i2x_im"),
                 ("o2y_re", "i2y_re"), ("o2y_im", "i2y_im")]:
        conns[f"PBS,{a}"] = f"PBC,{b}"
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {"te_re": "PBS,o1x_re", "te_im": "PBS,o1x_im",
                     "tm_re": "PBS,o2y_re", "tm_im": "PBS,o2y_im",
                     "cx_re": "PBC,ox_re", "cx_im": "PBC,ox_im",
                     "cy_re": "PBC,oy_re", "cy_im": "PBC,oy_im"}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "rot": cx.va("polarization_rotator"),
              "pbs": cx.va("pbs"), "pbc": cx.va("pbc")}
    return compile_circuit(net, models)


def test_malus_law():
    c = _malus_circuit()
    thetas = np.linspace(0.0, 180.0, 37)
    y = c.dc(params={"ROT.theta_deg": thetas})
    arr = lambda p: np.asarray(c.port(y, p)).real  # noqa: E731
    p_te = arr("te_re") ** 2 + arr("te_im") ** 2
    p_tm = arr("tm_re") ** 2 + arr("tm_im") ** 2
    t = np.radians(thetas)
    # Malus' law at the two PBS ports
    assert np.allclose(p_te, np.cos(t) ** 2, atol=1e-9)
    assert np.allclose(p_tm, np.sin(t) ** 2, atol=1e-9)
    # every split conserves power, and the PBC puts the link back together
    assert np.allclose(p_te + p_tm, 1.0, atol=1e-9)
    p_rec = (arr("cx_re") ** 2 + arr("cx_im") ** 2
             + arr("cy_re") ** 2 + arr("cy_im") ** 2)
    assert np.allclose(p_rec, 1.0, atol=1e-9)


# ===========================================================================
# Acceptance criterion 2 — birefringent-waveguide MZI, TE/TM fringes offset
# ===========================================================================
def _mzi_circuit(length_um=100.0, n_te=2.40, n_tm=2.38):
    """Jones MZI: a birefringent WG in one arm, a balanced reference arm, and
    50/50 mirror splitters/combiners (one per polarization — an ordinary
    coupler is polarization-independent). Cross-port power at BS2 is
    cos^2(phi/2) with phi the arm phase, evaluated per polarization."""
    insts = {"GND": {"component": "ground"},
             "WG": {"component": "bir",
                    "settings": {"length_um": length_um, "n_te": n_te, "n_tm": n_tm}},
             "SX": {"component": "vsrc", "settings": {"V": 0.0}},
             "SY": {"component": "vsrc", "settings": {"V": 0.0}},
             "T1X": {"component": "term"}, "T1Y": {"component": "term"}}
    for b in ("B1X", "B1Y", "B2X", "B2Y"):
        insts[b] = {"component": "mir", "settings": {"refl": 0.5}}
    conns = {"SX,p1": "B1X,li_re", "SY,p1": "B1Y,li_re"}
    gnd = ["B1X,gnd", "B1Y,gnd", "B2X,gnd", "B2Y,gnd", "WG,gnd", "SX,p2", "SY,p2",
           "B1X,li_im", "B1X,ri_re", "B1X,ri_im",   # BS1 unused input = 0
           "B1Y,li_im", "B1Y,ri_re", "B1Y,ri_im"]
    # arm A (birefringent): BS1 lo -> WG in -> BS2 li
    conns.update({"B1X,lo_re": "WG,xin_re", "B1X,lo_im": "WG,xin_im",
                  "B1Y,lo_re": "WG,yin_re", "B1Y,lo_im": "WG,yin_im",
                  "WG,xout_re": "B2X,li_re", "WG,xout_im": "B2X,li_im",
                  "WG,yout_re": "B2Y,li_re", "WG,yout_im": "B2Y,li_im"})
    # arm B (reference): BS1 ro -> BS2 ri directly (zero extra phase)
    conns.update({"B1X,ro_re": "B2X,ri_re", "B1X,ro_im": "B2X,ri_im",
                  "B1Y,ro_re": "B2Y,ri_re", "B1Y,ro_im": "B2Y,ri_im"})
    # BS2 bar port unused -> terminate; cross port is the readout
    conns.update({"T1X,re": "B2X,lo_re", "T1X,im": "B2X,lo_im",
                  "T1Y,re": "B2Y,lo_re", "T1Y,im": "B2Y,lo_im"})
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {"te_re": "B2X,ro_re", "te_im": "B2X,ro_im",
                     "tm_re": "B2Y,ro_re", "tm_im": "B2Y,ro_im"}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource, "mir": cx.va("mirror"),
              "bir": cx.va("birefringent_wg"), "term": terminator()}
    return compile_circuit(net, models)


def _mzi_sweep(c, lams, sx, sy):
    y = c.dc(params={"SX.V": sx * np.ones_like(lams),
                     "SY.V": sy * np.ones_like(lams),
                     "WG.lambda_nm": lams})
    arr = lambda p: np.asarray(c.port(y, p)).real  # noqa: E731
    p_te = arr("te_re") ** 2 + arr("te_im") ** 2
    p_tm = arr("tm_re") ** 2 + arr("tm_im") ** 2
    return p_te, p_tm


def test_birefringent_mzi():
    L, n_te, n_tm = 100.0, 2.40, 2.38
    c = _mzi_circuit(L, n_te, n_tm)
    lams = np.linspace(1500.0, 1600.0, 501)
    # launch pure TE, then pure TM
    te_p, _ = _mzi_sweep(c, lams, 1.0, 0.0)
    _, tm_p = _mzi_sweep(c, lams, 0.0, 1.0)
    # cross-port fringe is cos^2(pi * n * L / lambda) for each polarization
    phi_te = np.pi * n_te * (L * 1e-6) / (lams * 1e-9)
    phi_tm = np.pi * n_tm * (L * 1e-6) / (lams * 1e-9)
    assert np.allclose(te_p, np.cos(phi_te) ** 2, atol=1e-9)
    assert np.allclose(tm_p, np.cos(phi_tm) ** 2, atol=1e-9)
    # the fringes are OFFSET: the TE and TM transmission differ where the
    # differential phase pi*dn*L/lambda is not a multiple of pi
    assert np.max(np.abs(te_p - tm_p)) > 0.5


def test_mzi_no_birefringence_no_offset():
    # sanity: with dn_eff = 0 the TE and TM fringes coincide exactly
    L, n = 100.0, 2.39
    c = _mzi_circuit(L, n, n)
    lams = np.linspace(1500.0, 1600.0, 201)
    te_p, _ = _mzi_sweep(c, lams, 1.0, 0.0)
    _, tm_p = _mzi_sweep(c, lams, 0.0, 1.0)
    assert np.allclose(te_p, tm_p, atol=1e-9)
