"""SKY130 process corners: parsing, robust aggregation, and web-app plumbing."""
import sys
from pathlib import Path

import numpy as np
import pytest

from photonflux.corners import CORNERS, aggregate, check_corner, parse_corners

REPO = Path(__file__).resolve().parents[1]


def test_parse_corners():
    assert parse_corners(None) == ("tt",)
    assert parse_corners("") == ("tt",)
    assert parse_corners("all") == CORNERS
    assert parse_corners("ss, ff,ss") == ("ss", "ff")      # order kept, deduped
    assert parse_corners(["fs", "sf"]) == ("fs", "sf")
    with pytest.raises(ValueError, match="unknown SKY130 process corner"):
        parse_corners("tt,xx")
    with pytest.raises(ValueError):
        check_corner("TT")                                  # case matters to .lib


def test_aggregate_worst_follows_the_objective_direction():
    vals = {"tt": -3.0, "ss": -4.5, "ff": -2.0}
    assert aggregate(vals, "worst", maximize=True) == -4.5
    assert aggregate(vals, "worst", maximize=False) == -2.0
    assert aggregate(vals, "mean") == pytest.approx(-3.1666666, rel=1e-6)


def test_aggregate_fails_the_design_when_any_corner_fails():
    """Scoring only the corners that converged would reward fragile designs."""
    assert aggregate({"tt": 1.0, "ss": None}, "worst") is None
    assert aggregate({"tt": 1.0, "ss": float("nan")}, "mean") is None
    assert aggregate({"tt": 1.0, "ss": -np.inf}, "worst") is None
    with pytest.raises(ValueError, match="aggregate"):
        aggregate({"tt": 1.0}, "median")


def test_sky130_card_rejects_an_unknown_corner():
    from photonflux import cx

    with pytest.raises(ValueError, match="unknown SKY130 process corner"):
        cx.sky130_card("nfet_01v8", w=1.0, l=0.15, corner="typ")


# ---------------------------------------------------------------------------
# web app: the corner reaches the FET model key, and "all" is a fan-out
# ---------------------------------------------------------------------------
@pytest.fixture
def simulate():
    sys.path.insert(0, str(REPO / "webapp"))
    import simulate as sim

    return sim


def _fet_schematic(corner=None):
    sch = {
        "instances": {
            "M1": {"type": "sky130_nfet", "settings": {"w_um": 1.0, "l_um": 0.15}},
            "V1": {"type": "vdc", "settings": {"V": 1.8}},
            "G": {"type": "ground"},
        },
        "wires": [["V1,p1", "M1,d"], ["V1,p1", "M1,g"], ["V1,p2", "G,p1"],
                  ["M1,s", "G,p1"], ["M1,b", "G,p1"]],
        "probes": [{"name": "vd", "at": "M1,d"}],
    }
    if corner is not None:
        sch["corner"] = corner
    return sch


def test_netlist_keys_fets_by_corner(simulate):
    net, meta = simulate.schematic_to_netlist(_fet_schematic())
    assert net["instances"]["M1"]["component"] == "sky130_nfet:1x0.15@tt"
    net, meta = simulate.schematic_to_netlist(_fet_schematic("ss"))
    key = net["instances"]["M1"]["component"]
    assert key == "sky130_nfet:1x0.15@ss"
    assert meta["sky130_geoms"][key] == ("nfet_01v8", 1.0, 0.15, "ss")


def test_netlist_refuses_a_multi_corner_schematic(simulate):
    """"all" is resolved into one run per corner before the netlist is built;
    reaching the netlist with several corners is a routing error, not a
    silent tt run."""
    with pytest.raises(simulate.NetlistError, match="single corner"):
        simulate.schematic_to_netlist(_fet_schematic("all"))
    with pytest.raises(simulate.NetlistError, match="unknown SKY130"):
        simulate.schematic_to_netlist(_fet_schematic("nominal"))


def test_gui_globals_corner_is_honored(simulate):
    """The GUI keeps the corner in globals.corner; notebook runs post that
    state as-is, so the backend must read it rather than silently run tt."""
    sch = _fet_schematic()
    sch["globals"] = {"corner": "ff"}
    assert simulate.sch_corners(sch) == ("ff",)
    sch["corner"] = "ss"                       # an explicit key wins
    assert simulate.sch_corners(sch) == ("ss",)


def test_a_schematic_without_fets_ignores_multi_corner(simulate):
    """No SKY130 FET -> corner-independent: one run, not five identical ones."""
    sch = _fet_schematic("all")
    assert len(simulate.sch_corners(sch)) == 5
    del sch["instances"]["M1"]
    assert simulate.sch_corners(sch) == ("tt",)


# ---------------------------------------------------------------------------
# physics: the corners actually move the device the way their names say
# ---------------------------------------------------------------------------
def _ion(device: str, corner: str) -> float:
    """|Id| of a 1.8 V device, fully on, drain at the opposite rail."""
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSource

    from photonflux import cx

    w = 1.0 if device.startswith("n") else 2.0
    fet = cx.sky130_fet(device, w=w, l=0.15, corner=corner)
    rs = 1e-3
    if device.startswith("n"):   # gate + drain high, source grounded
        conn = {"GND,p1": ("VD,p2", "M1,s", "M1,b"),
                "VD,p1": ("RS,p1", "M1,g"), "RS,p2": "M1,d"}
    else:                        # source + body high, gate + drain low
        conn = {"GND,p1": ("VD,p2", "M1,g", "RS,p2"),
                "VD,p1": ("M1,s", "M1,b"), "M1,d": "RS,p1"}
    net = {"instances": {"GND": {"component": "ground"},
                         "VD": {"component": "vsrc", "settings": {"V": 1.8}},
                         "RS": {"component": "res", "settings": {"R": rs}},
                         "M1": {"component": "fet"}},
           "connections": conn,
           "ports": {"a": "RS,p1", "b": "RS,p2"}}
    c = compile_circuit(net, {"ground": lambda: 0, "vsrc": VoltageSource,
                              "res": Resistor, "fet": fet},
                        backend="dense", max_steps=200)
    y = c.dc()
    return abs(float((c.port(y, "a") - c.port(y, "b")).real)) / rs


def test_corner_drive_strength_ordering(sky130_available):
    """ss < tt < ff for both flavors; the skewed corners move NMOS and PMOS
    in opposite directions. SKY130's skewed names read PMOS-first: ``sf`` is a
    *fast* NMOS with a *slow* PMOS, ``fs`` the reverse (see corners.py).

    First run extracts 8 new cards (~1-2 min each); cached afterwards."""
    if not sky130_available:
        pytest.skip("SKY130 PDK / openvaf-ir not installed")
    n = {c: _ion("nfet_01v8", c) for c in CORNERS}
    p = {c: _ion("pfet_01v8", c) for c in CORNERS}
    assert n["ss"] < n["tt"] < n["ff"]
    assert p["ss"] < p["tt"] < p["ff"]
    assert n["fs"] < n["tt"] < n["sf"]
    assert p["sf"] < p["tt"] < p["fs"]
