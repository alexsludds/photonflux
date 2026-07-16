"""photonflux.cx — Verilog-A photonics and SKY130 FETs as circulax components.

These tests need the circulax stack (``pip install circulax[verilog-a]
openvaf-py`` into the venv) and the ChipFlow openvaf fork at
``bin/openvaf-ir``; they self-skip otherwise. Run with:

    .venv-circulax/bin/python -m pytest tests/test_cx.py -v

The SKY130 FET is pinned to its own physics — off-state at gmin, a monotone
transfer curve, an on-current in the physical band, and an inverter that rails
and trips near mid-supply — rather than to a live ngspice run (there is no
ngspice simulation flow; ngspice only extracts the BSIM4 card).
"""
from __future__ import annotations

import numpy as np
import pytest

circulax = pytest.importorskip("circulax")
pytest.importorskip("bosdi")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from circulax import compile_circuit  # noqa: E402
from circulax.components.electronic import Resistor, VoltageSource  # noqa: E402

from photonflux import cx  # noqa: E402

needs_openvaf_ir = pytest.mark.skipif(
    not cx.openvaf_ir_path().exists(), reason="bin/openvaf-ir not built"
)


# ---------------------------------------------------------------------------
# Verilog-A photonics -> pure-JAX components (exact + differentiable)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pd_circuit():
    PD = cx.va("photodiode")
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VP": {"component": "vsrc", "settings": {"V": 9e-3}},
            "PD": {"component": "pd",
                   "settings": {"R": 0.8, "Idk": 1e-9, "Cj": 100e-15}},
            "RL": {"component": "res", "settings": {"R": 200.0}},
        },
        "connections": {
            "GND,p1": ("VP,p2", "PD,gnd", "PD,cat", "RL,p2"),
            "VP,p1": "PD,popt",
            "PD,an": "RL,p1",
        },
        "ports": {"vload": "PD,an"},
    }
    models = {"ground": lambda: 0, "vsrc": VoltageSource, "pd": PD, "res": Resistor}
    return compile_circuit(net, models)


def test_va_photodiode_dc_exact(pd_circuit):
    """V(load) = -(R*P + Idk)*RL through the bosdi-compiled photodiode.va."""
    y = pd_circuit.dc()
    vload = float(pd_circuit.port(y, "vload").real)
    expect = -(0.8 * 9e-3 + 1e-9) * 200.0
    assert vload == pytest.approx(expect, rel=1e-9)


def test_va_photodiode_grad_exact(pd_circuit):
    """jax.grad through the DC solve returns the analytic sensitivity."""

    def vload(r_pd: float) -> float:
        y = pd_circuit.dc(params={"PD.R": r_pd})
        return pd_circuit.port(y, "vload").real

    g = float(jax.grad(vload)(0.8))
    assert g == pytest.approx(-9e-3 * 200.0, rel=1e-9)  # dV/dR = -P*RL


# ---------------------------------------------------------------------------
# CW laser + MZM (house convention: lasers are CW, modulators modulate)
# ---------------------------------------------------------------------------

def test_cw_laser_mzm_transfer():
    """|E_out|^2 through the MZM matches the mzm.va intensity transfer."""
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("po_p", "po_n"))
    def MatchedAbsorber(signals: Signals, s: States, Yopt: float = 1.0):
        e = signals.po_p - signals.po_n
        return {"po_p": Yopt * e, "po_n": -Yopt * e}, {}

    P, VPI, IL_DB, ER_DB = 2e-3, 3.3, 3.0, 20.0
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LAS": {"component": "laser",
                    "settings": {"wavelength_nm": 1310.0, "power": P}},
            "VMOD": {"component": "vsrc", "settings": {"V": 0.0}},
            "MOD": {"component": "mzm",
                    "settings": {"vpi": VPI, "vbias": 0.0,
                                 "il_db": IL_DB, "er_db": ER_DB}},
            "TERM": {"component": "absorber"},
        },
        "connections": {
            "GND,p1": ("LAS,p2", "VMOD,p2", "MOD,vn", "TERM,po_n"),
            "LAS,p1": "MOD,pin",
            "VMOD,p1": "MOD,vp",
            "MOD,pout": "TERM,po_p",
        },
        "ports": {"pout": "MOD,pout"},
    }
    c = compile_circuit(
        net,
        {"ground": lambda: 0, "laser": cx.cw_laser(), "mzm": cx.mzm(),
         "vsrc": VoltageSource, "absorber": MatchedAbsorber},
        is_complex=True,
    )

    il = 10 ** (-IL_DB / 10)
    er = 10 ** (ER_DB / 10)
    eta = (er - 1) / (er + 1)

    for v in (0.0, VPI / 2, VPI):
        y = c.dc(params={"VMOD.V": v})
        p_out = float(jnp.abs(c.port(y, "pout")) ** 2)
        expect = P * il * (0.5 + 0.5 * eta * np.cos(np.pi * v / VPI))
        assert p_out == pytest.approx(expect, rel=1e-9), f"V={v}"


# ---------------------------------------------------------------------------
# SKY130 FETs -> OSDI descriptors (exact BSIM4.8, card resolved by ngspice)
# ---------------------------------------------------------------------------

@needs_openvaf_ir
def test_sky130_card_extraction(sky130_available):
    if not sky130_available:
        pytest.skip("SKY130 PDK not installed")
    card = cx.sky130_card("nfet_01v8", w=1.0, l=0.15)
    assert 700 < len(card) < 800
    assert card["toxe"] == pytest.approx(4.148e-9, rel=1e-3)
    # tnom converted to Celsius (ngspice's showmod reports Kelvin)
    assert card["tnom"] == pytest.approx(30.0, abs=0.1)
    # the W=1u / L=0.15u bin
    assert card["lmin"] <= 0.15e-6 <= card["lmax"]
    assert card["wmin"] <= 1.0e-6 <= card["wmax"]


@needs_openvaf_ir
def test_sky130_nfet_transfer(sky130_available):
    """The OSDI nfet, evaluated natively in circulax, has the right transfer:
    gmin off-state, monotone Id(Vgs), and an on-current in the physical band."""
    if not sky130_available:
        pytest.skip("SKY130 PDK not installed")
    NFET = cx.sky130_fet("nfet_01v8", w=1.0, l=0.15)
    RS = 1e-3
    vg = np.arange(0.0, 1.81, 0.3)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VD": {"component": "vsrc", "settings": {"V": 1.8}},
            "VG": {"component": "vsrc", "settings": {"V": 0.0}},
            "RS": {"component": "res", "settings": {"R": RS}},
            "M1": {"component": "nfet"},
        },
        "connections": {
            "GND,p1": ("VD,p2", "VG,p2", "M1,s", "M1,b"),
            "VD,p1": "RS,p1", "RS,p2": "M1,d", "VG,p1": "M1,g",
        },
        "ports": {"d": "M1,d", "vdd": "VD,p1"},
    }
    c = compile_circuit(
        net,
        {"ground": lambda: 0, "vsrc": VoltageSource, "res": Resistor, "nfet": NFET},
        backend="dense", max_steps=200,
    )
    y = c.dc(params={"VG.V": jnp.asarray(vg)})
    id_cx = np.asarray((c.port(y, "vdd") - c.port(y, "d")).real / RS)

    # off-state at gmin (not the sign-flipped tens of nA the broken jax
    # backend produces), monotone transfer, physical on-current at 1.8 V
    assert abs(id_cx[0]) < 1e-10
    assert np.all(np.diff(id_cx) > 0)
    assert 200e-6 < id_cx[-1] < 800e-6


@needs_openvaf_ir
def test_sky130_inverter_vtc(sky130_available):
    """The CMOS inverter rails to the supplies and trips near mid-supply."""
    if not sky130_available:
        pytest.skip("SKY130 PDK not installed")
    NFET = cx.sky130_fet("nfet_01v8", w=1.0, l=0.15)
    PFET = cx.sky130_fet("pfet_01v8", w=2.0, l=0.15)
    vin = np.arange(0.0, 1.81, 0.1)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VDD": {"component": "vsrc", "settings": {"V": 1.8}},
            "VIN": {"component": "vsrc", "settings": {"V": 0.0}},
            "MP": {"component": "pfet"}, "MN": {"component": "nfet"},
        },
        "connections": {
            "GND,p1": ("VDD,p2", "VIN,p2", "MN,s", "MN,b"),
            "VDD,p1": ("MP,s", "MP,b"),
            "VIN,p1": ("MP,g", "MN,g"),
            "MP,d": "MN,d",
        },
        "ports": {"out": "MN,d"},
    }
    c = compile_circuit(
        net,
        {"ground": lambda: 0, "vsrc": VoltageSource, "nfet": NFET, "pfet": PFET},
        backend="dense", max_steps=300,
    )
    vout = np.asarray(c.port(c.dc(params={"VIN.V": jnp.asarray(vin)}), "out").real)
    assert np.isfinite(vout).all()
    assert abs(vout[0] - 1.8) < 1e-3           # rails to VDD for a low input
    assert abs(vout[-1]) < 1e-3                # rails to GND for a high input
    assert np.all(np.diff(vout) <= 1e-9)       # monotone decreasing
    trip = vin[int(np.argmin(np.abs(vout - 0.9)))]
    assert 0.7 < trip < 1.1                    # switches near mid-supply
