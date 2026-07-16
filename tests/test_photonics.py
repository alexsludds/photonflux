"""Steady-state physics of the power-domain Verilog-A models, lowered to JAX by
``cx.va`` and solved by circulax.

These pin the compiled behaviour of the lasers, photodiode and MZM to their
analytic model equations, catching model regressions and toolchain breakage
alike. Optical-power nodes carry watts as a node voltage (the power-domain
convention); the transient turn-on / relaxation studies are in
``examples/*.py``.
"""
import pytest

from circuit_helpers import op
from photonflux import cx


def _laser_op(model: str, v_drive: float, **params) -> float:
    """Optical power [W] of laser <model> driven at V(an) = v_drive."""
    vals = op(cx.va(model), {"an": v_drive, "cat": 0.0},
              settings=params, reads=["popt"])
    return vals["popt"].real


def test_laser_dml_li_curve():
    # P = slope * ((V-Von)/Rs - Ith), clamped at 0
    assert _laser_op("laser_dml", 1.2) == pytest.approx(0.0, abs=1e-9)
    assert _laser_op("laser_dml", 1.25) == pytest.approx(0.0, abs=1e-9)   # at threshold
    assert _laser_op("laser_dml", 1.5) == pytest.approx(0.015, rel=1e-3)  # 60mA -> 15mW
    # (circulax param names are case-sensitive: Ith, not ith)
    assert _laser_op("laser_dml", 1.5, slope=0.5, Ith=20e-3) == pytest.approx(
        0.020, rel=1e-3)


# laser_rate's above-threshold output is a transient attractor (Newton DC lands
# on the dark branch, like the FP laser); its turn-on delay + relaxation are
# pinned by webapp example 09 "DML vs rate-equation laser".


def test_photodiode_responsivity():
    # anode grounded, cathode -> RL -> ground; photocurrent I = R*P + Idk flows
    # anode->cathode, so V(cat) = +I*RL. The current is load-independent.
    for rl in (1.0, 200.0):
        assert _pd_current(0.7, 2e-3, rl) == pytest.approx(0.7 * 2e-3 + 1e-9, rel=1e-6)


def _pd_current(r: float, p: float, rl: float = 1.0) -> float:
    """Photocurrent I = R*P + Idk, read as V(cat)/RL across a cathode load."""
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSource
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VP": {"component": "vsrc", "settings": {"V": p}},
            "PD": {"component": "pd", "settings": {"R": r}},
            "RL": {"component": "res", "settings": {"R": rl}},
        },
        "connections": {
            "GND,p1": ("VP,p2", "PD,gnd", "PD,an", "RL,p2"),
            "VP,p1": "PD,popt", "PD,cat": "RL,p1",
        },
        "ports": {"cat": "PD,cat"},
    }
    c = compile_circuit(net, {"ground": lambda: 0, "vsrc": VoltageSource,
                              "pd": cx.va("photodiode"), "res": Resistor},
                        backend="dense")
    return complex(c.port(c.dc(), "cat")).real / rl


def test_mzm_transfer():
    def t_of(v: float) -> float:
        vals = op(cx.va("mzm"),
                  {"pin": 1e-3, "vp": v, "vn": 0.0},
                  settings={"vpi": 3.0, "il_db": 3.0, "er_db": 20.0},
                  reads=["pout"])
        return vals["pout"].real / 1e-3

    il = 10 ** (-0.3)
    assert t_of(0.0) == pytest.approx(il * (0.5 + 0.5 * 99 / 101), rel=1e-3)
    assert t_of(1.5) == pytest.approx(il * 0.5, rel=1e-3)            # quadrature
    assert t_of(0.0) / t_of(3.0) == pytest.approx(100, rel=1e-2)     # 20 dB ER
