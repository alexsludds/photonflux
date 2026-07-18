"""webapp/simulate.flatten_schematic — user-subcircuit inlining.

The editor stores subcircuit definitions under schematic["subcircuits"] and
references them with instances typed "subckt:<def>"; boundary ports are
marker instances (subckt_port_e / subckt_port_o) whose instance name is the
port name. Flattening namespaces inlined instances and probes as
"<parent>__<child>" and dissolves the markers into net aliases. These tests
pin that contract without touching the jax/circulax stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

from simulate import NetlistError, flatten_schematic  # noqa: E402


def _rx_frontend_def(r: float = 500.0) -> dict:
    """pin (optical) -> photodiode -> R -> out (electrical), with a probe."""
    return {
        "schematic": {
            "instances": {
                "pin": {"type": "subckt_port_o", "settings": {}},
                "out": {"type": "subckt_port_e", "settings": {}},
                "PD1": {"type": "photodiode", "settings": {}},
                "R1": {"type": "resistor", "settings": {"R": r}},
            },
            "wires": [
                {"from": "pin,p", "to": "PD1,pin"},
                {"from": "PD1,po_p", "to": "R1,a"},
                {"from": "R1,b", "to": "out,p"},
            ],
            "probes": [{"name": "ipd", "at": "PD1,po_p", "color": "#4ade80"}],
        }
    }


def _top(instances, wires, probes=(), subcircuits=None) -> dict:
    sch = {"instances": instances, "wires": wires, "probes": list(probes)}
    if subcircuits is not None:
        sch["subcircuits"] = subcircuits
    return sch


def _wire_set(flat) -> set[frozenset]:
    return {frozenset(w) for w in flat["wires"]}


def test_flat_schematic_passes_through_unchanged():
    sch = _top(
        {"LAS1": {"type": "cw_laser", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["LAS1,p2", "GND1,p1"]],
    )
    assert flatten_schematic(sch) is sch


def test_unused_defs_are_stripped_on_the_fast_path():
    sch = _top(
        {"GND1": {"type": "ground", "settings": {}}},
        [],
        subcircuits={"rx_frontend": _rx_frontend_def()},
    )
    flat = flatten_schematic(sch)
    assert "subcircuits" not in flat
    assert flat["instances"] == sch["instances"]


def test_single_instance_inlines_with_namespacing():
    sch = _top(
        {"LAS1": {"type": "cw_laser", "settings": {}},
         "U1": {"type": "subckt:rx_frontend", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["LAS1,p1", "U1,pin"], ["U1,out", "GND1,p1"]],
        probes=[{"name": "vout", "at": "U1,out"}],
        subcircuits={"rx_frontend": _rx_frontend_def()},
    )
    flat = flatten_schematic(sch)
    assert set(flat["instances"]) == {"LAS1", "GND1", "U1__PD1", "U1__R1"}
    assert flat["instances"]["U1__R1"]["settings"] == {"R": 500.0}
    ws = _wire_set(flat)
    # parent wire to U1,pin resolves through the marker to the photodiode
    assert frozenset({"LAS1,p1", "U1__PD1,pin"}) in ws
    # parent wire to U1,out resolves to the resistor's b terminal
    assert frozenset({"U1__GND1,p1", "U1__R1,b"}) not in ws  # sanity: no such inst
    assert frozenset({"R1,b", "GND1,p1"}) not in ws          # must be namespaced
    assert frozenset({"U1__R1,b", "GND1,p1"}) in ws
    # def-internal wiring survives, markers dissolved
    assert frozenset({"U1__PD1,po_p", "U1__R1,a"}) in ws
    assert not any("pin" in ep.split(",")[0] for w in flat["wires"] for ep in w
                   if ep.startswith("U1__pin"))
    # probes: parent probe rewritten, def probe hoisted + namespaced
    by_name = {p["name"]: p for p in flat["probes"]}
    assert by_name["vout"]["at"] == "U1__R1,b"
    assert by_name["U1__ipd"]["at"] == "U1__PD1,po_p"
    assert by_name["U1__ipd"]["color"] == "#4ade80"


def test_two_instances_of_one_def_are_independent():
    sch = _top(
        {"U1": {"type": "subckt:rx_frontend", "settings": {}},
         "U2": {"type": "subckt:rx_frontend", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["U1,out", "GND1,p1"], ["U2,out", "GND1,p1"]],
        subcircuits={"rx_frontend": _rx_frontend_def()},
    )
    flat = flatten_schematic(sch)
    assert {"U1__PD1", "U1__R1", "U2__PD1", "U2__R1"} <= set(flat["instances"])
    names = {p["name"] for p in flat["probes"]}
    assert {"U1__ipd", "U2__ipd"} <= names
    # sibling copies must not alias: patching one (sweeps, pulse mode)
    # must leave the other untouched
    flat["instances"]["U1__R1"]["settings"]["R"] = 123.0
    assert flat["instances"]["U2__R1"]["settings"]["R"] == 500.0


def test_nested_defs_flatten_recursively():
    inner = _rx_frontend_def()
    outer = {
        "schematic": {
            "instances": {
                "in": {"type": "subckt_port_o", "settings": {}},
                "vout": {"type": "subckt_port_e", "settings": {}},
                "RX": {"type": "subckt:rx_frontend", "settings": {}},
            },
            "wires": [
                {"from": "in,p", "to": "RX,pin"},
                {"from": "RX,out", "to": "vout,p"},
            ],
            "probes": [],
        }
    }
    sch = _top(
        {"LAS1": {"type": "cw_laser", "settings": {}},
         "A1": {"type": "subckt:analog_fe", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["LAS1,p1", "A1,in"], ["A1,vout", "GND1,p1"]],
        subcircuits={"rx_frontend": inner, "analog_fe": outer},
    )
    flat = flatten_schematic(sch)
    assert {"A1__RX__PD1", "A1__RX__R1"} <= set(flat["instances"])
    ws = _wire_set(flat)
    # outer port chains through the inner def's marker to the photodiode
    assert frozenset({"LAS1,p1", "A1__RX__PD1,pin"}) in ws
    assert frozenset({"A1__RX__R1,b", "GND1,p1"}) in ws
    assert {p["name"] for p in flat["probes"]} == {"A1__RX__ipd"}


def test_marker_with_several_wires_keeps_all_nets_joined():
    d = {
        "schematic": {
            "instances": {
                "out": {"type": "subckt_port_e", "settings": {}},
                "R1": {"type": "resistor", "settings": {}},
                "R2": {"type": "resistor", "settings": {}},
            },
            "wires": [
                {"from": "R1,b", "to": "out,p"},
                {"from": "R2,b", "to": "out,p"},
            ],
            "probes": [],
        }
    }
    sch = _top(
        {"U1": {"type": "subckt:d", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["U1,out", "GND1,p1"]],
        subcircuits={"d": d},
    )
    flat = flatten_schematic(sch)
    ws = _wire_set(flat)
    # both internal resistors and the external ground share one net through
    # the marker's anchor endpoint
    anchor_hits = [w for w in ws if "GND1,p1" in w]
    assert anchor_hits
    joined = set().union(*ws)
    assert {"U1__R1,b", "U1__R2,b", "GND1,p1"} <= joined


def test_cycle_is_rejected():
    a = {"schematic": {"instances": {
        "X": {"type": "subckt:b", "settings": {}}}, "wires": [], "probes": []}}
    b = {"schematic": {"instances": {
        "Y": {"type": "subckt:a", "settings": {}}}, "wires": [], "probes": []}}
    sch = _top({"U1": {"type": "subckt:a", "settings": {}}}, [],
               subcircuits={"a": a, "b": b})
    with pytest.raises(NetlistError, match="cycle"):
        flatten_schematic(sch)


def test_unknown_def_is_rejected():
    sch = _top({"U1": {"type": "subckt:nope", "settings": {}}}, [],
               subcircuits={})
    with pytest.raises(NetlistError, match="unknown subcircuit"):
        flatten_schematic(sch)


def test_unwired_port_referenced_from_parent_is_rejected():
    d = {"schematic": {"instances": {
        "out": {"type": "subckt_port_e", "settings": {}},
        "R1": {"type": "resistor", "settings": {}}},
        "wires": [], "probes": []}}
    sch = _top(
        {"U1": {"type": "subckt:d", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["U1,out", "GND1,p1"]],
        subcircuits={"d": d},
    )
    with pytest.raises(NetlistError, match="not connected"):
        flatten_schematic(sch)


def test_missing_port_is_rejected():
    sch = _top(
        {"U1": {"type": "subckt:rx_frontend", "settings": {}},
         "GND1": {"type": "ground", "settings": {}}},
        [["U1,nope", "GND1,p1"]],
        subcircuits={"rx_frontend": _rx_frontend_def()},
    )
    with pytest.raises(NetlistError, match="missing or not connected"):
        flatten_schematic(sch)


def test_top_level_marker_is_rejected():
    sch = _top({"pin": {"type": "subckt_port_e", "settings": {}},
                "U1": {"type": "subckt:rx_frontend", "settings": {}}}, [],
               subcircuits={"rx_frontend": _rx_frontend_def()})
    with pytest.raises(NetlistError, match="top-level"):
        flatten_schematic(sch)


def test_flattened_name_collision_is_rejected():
    d = {"schematic": {"instances": {
        "R1": {"type": "resistor", "settings": {}}},
        "wires": [], "probes": []}}
    sch = _top(
        {"U1": {"type": "subckt:d", "settings": {}},
         "U1__R1": {"type": "resistor", "settings": {}}},
        [],
        subcircuits={"d": d},
    )
    with pytest.raises(NetlistError, match="collision"):
        flatten_schematic(sch)


def test_probe_name_collision_is_rejected():
    d = {"schematic": {"instances": {
        "R1": {"type": "resistor", "settings": {}}},
        "wires": [],
        "probes": [{"name": "x", "at": "R1,a"}]}}
    sch = _top(
        {"U1": {"type": "subckt:d", "settings": {}},
         "R9": {"type": "resistor", "settings": {}}},
        [],
        probes=[{"name": "U1__x", "at": "R9,a"}],
        subcircuits={"d": d},
    )
    with pytest.raises(NetlistError, match="probe name collision"):
        flatten_schematic(sch)
