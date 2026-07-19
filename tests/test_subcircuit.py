"""Hierarchical subcircuit flattening (webapp/subcircuit.py).

Pure netlist-transform tests — no JAX. The simulator is a deterministic
function of the flattened netlist, so proving the flattener reproduces the
hand-built flat netlist *exactly* (namespaced refdes, spliced ports, baked
exported params, hierarchical probes) is what the ALE-78 acceptance criterion
"flattened sim matches the manually flattened equivalent" reduces to.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp"))

from subcircuit import (  # noqa: E402
    SubcircuitError,
    flatten_subcircuits,
    subcircuit_ports,
)


def _wdm_drop_def():
    """A WDM drop-filter subcircuit: a ring + a heater between four ports,
    with the ring resonance exported as a single parameter."""
    return {
        "name": "wdm_drop",
        "label": "WDM Drop",
        "params": [
            {"name": "resonance_nm", "default": 1550.0,
             "bind": [{"instance": "RING", "param": "wavelength_nm"}]},
        ],
        "schematic": {
            "instances": {
                "P_in": {"type": "port",
                         "settings": {"name": "in", "domain": "optical"}},
                "P_drop": {"type": "port",
                           "settings": {"name": "drop", "domain": "optical"}},
                "P_vp": {"type": "port",
                         "settings": {"name": "vp", "domain": "electrical"}},
                "P_vn": {"type": "port",
                         "settings": {"name": "vn", "domain": "electrical"}},
                "RING": {"type": "waveguide",
                         "settings": {"wavelength_nm": 1550.0}},
                "HEAT": {"type": "resistor", "settings": {"R": 1000.0}},
            },
            "wires": [
                ["P_in,p", "RING,a"],
                ["RING,b", "P_drop,p"],
                ["P_vp,p", "HEAT,p1"],
                ["P_vn,p", "HEAT,p2"],
            ],
            "probes": [{"name": "n_out", "at": "RING,b"}],
        },
    }


def _wireset(wires):
    """Order- and direction-agnostic wire comparison."""
    return {frozenset(w) for w in wires}


def _nets(wires, extra=()):
    """The union-find net partition (how simulate.py resolves connectivity),
    as a set of frozensets of endpoints — the invariant the sim depends on."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in wires:
        parent[find(a)] = find(b)
    for e in extra:
        find(e)
    groups = {}
    for ep in parent:
        groups.setdefault(find(ep), set()).add(ep)
    return {frozenset(g) for g in groups.values()}


def _same_net(wires, *eps):
    """True if all endpoints land in one union-find net."""
    for net in _nets(wires, eps):
        if set(eps) <= net:
            return True
    return False


def test_declared_ports_in_order():
    ports = subcircuit_ports(_wdm_drop_def())
    assert ports == [
        {"name": "in", "domain": "optical"},
        {"name": "drop", "domain": "optical"},
        {"name": "vp", "domain": "electrical"},
        {"name": "vn", "domain": "electrical"},
    ]


def test_four_instances_match_manual_flatten():
    subs = {"wdm_drop": _wdm_drop_def()}
    reson = {"X1": 1548.0, "X2": 1549.0, "X3": 1550.0, "X4": 1551.0}

    instances = {"LAS": {"type": "cw_laser", "settings": {}},
                 "G": {"type": "ground", "settings": {}}}
    wires = []
    probes = []
    for ref, nm in reson.items():
        instances[ref] = {"type": "wdm_drop", "settings": {"resonance_nm": nm}}
        wires += [[f"{ref},in", "LAS,p1"], [f"{ref},vp", "G,p1"],
                  [f"{ref},vn", "G,p1"]]
        probes.append({"name": f"{ref}_drop", "at": f"{ref},drop"})

    got_i, got_w, got_p = flatten_subcircuits(instances, wires, probes, subs)

    # --- the manually flattened equivalent -------------------------------
    # Every instance is a namespaced primitive with the resonance baked onto
    # the ring; the four rings share LAS/G but nothing else.
    exp_i = {"LAS": {"type": "cw_laser", "settings": {}},
             "G": {"type": "ground", "settings": {}}}
    for ref, nm in reson.items():
        exp_i[f"{ref}.RING"] = {"type": "waveguide",
                                "settings": {"wavelength_nm": nm}}
        exp_i[f"{ref}.HEAT"] = {"type": "resistor", "settings": {"R": 1000.0}}
    exp_nets = {
        # the shared laser bus feeds every ring input (all 'in' ports splice
        # onto the one LAS,p1 net)
        frozenset({"LAS,p1"} | {f"{r}.RING,a" for r in reson}),
        # heater bias legs all land on ground (through ports 'vp'/'vn')
        frozenset({"G,p1"} | {f"{r}.HEAT,p1" for r in reson}
                  | {f"{r}.HEAT,p2" for r in reson}),
    }

    assert got_i == exp_i
    # net partition (ignoring lone drop-port nets with no second endpoint)
    got_nets = {n for n in _nets(got_w) if len(n) > 1}
    assert got_nets == exp_nets
    # probes resolve onto the ring output (the 'drop'/'n_out' net)
    at = {p["name"]: p["at"] for p in got_p}
    for ref in reson:
        assert at[f"{ref}_drop"] == f"{ref}.RING,b"
        assert at[f"{ref}.n_out"] == f"{ref}.RING,b"
    # no pseudo-ports leak into the flat netlist
    assert all(i["type"] != "port" for i in got_i.values())


def test_default_used_when_param_unset():
    subs = {"wdm_drop": _wdm_drop_def()}
    instances = {"X1": {"type": "wdm_drop", "settings": {}}}  # no override
    got_i, _, _ = flatten_subcircuits(instances, [], [], subs)
    assert got_i["X1.RING"]["settings"]["wavelength_nm"] == 1550.0


def test_two_level_nesting():
    inner = _wdm_drop_def()
    pair = {
        "name": "wdm_pair",
        "label": "WDM Pair",
        "params": [],
        "schematic": {
            "instances": {
                "P_a": {"type": "port",
                        "settings": {"name": "a", "domain": "optical"}},
                "A1": {"type": "wdm_drop",
                       "settings": {"resonance_nm": 1540.0}},
                "A2": {"type": "wdm_drop",
                       "settings": {"resonance_nm": 1560.0}},
            },
            "wires": [["P_a,p", "A1,in"], ["A1,drop", "A2,in"]],
            "probes": [],
        },
    }
    subs = {"wdm_drop": inner, "wdm_pair": pair}
    instances = {"Y1": {"type": "wdm_pair", "settings": {}},
                 "SRC": {"type": "cw_laser", "settings": {}}}
    got_i, got_w, _ = flatten_subcircuits(instances, [["Y1,a", "SRC,p1"]], [], subs)

    # two levels of namespacing compose with '.'
    assert "Y1.A1.RING" in got_i
    assert "Y1.A2.RING" in got_i
    assert got_i["Y1.A1.RING"]["settings"]["wavelength_nm"] == 1540.0
    assert got_i["Y1.A2.RING"]["settings"]["wavelength_nm"] == 1560.0
    assert all(i["type"] != "port" for i in got_i.values())
    # outer port 'a' splices through both nesting levels: external SRC reaches
    # the innermost ring input on one net (the boundary labels contract away)
    assert _same_net(got_w, "SRC,p1", "Y1.A1.RING,a")
    # A1,drop -> A2,in chains across the two child instances into one net
    assert _same_net(got_w, "Y1.A1.RING,b", "Y1.A2.RING,a")


def test_direct_self_instantiation_rejected():
    loop = {
        "name": "loop",
        "params": [],
        "schematic": {"instances": {"SELF": {"type": "loop", "settings": {}}},
                      "wires": [], "probes": []},
    }
    with pytest.raises(SubcircuitError, match="cycle"):
        flatten_subcircuits({"X1": {"type": "loop", "settings": {}}},
                            [], [], {"loop": loop})


def test_indirect_self_instantiation_rejected():
    a = {"name": "A", "params": [],
         "schematic": {"instances": {"B1": {"type": "B", "settings": {}}},
                       "wires": [], "probes": []}}
    b = {"name": "B", "params": [],
         "schematic": {"instances": {"A1": {"type": "A", "settings": {}}},
                       "wires": [], "probes": []}}
    with pytest.raises(SubcircuitError, match="cycle"):
        flatten_subcircuits({"X1": {"type": "A", "settings": {}}},
                            [], [], {"A": a, "B": b})


def test_no_subcircuits_normalizes_shape():
    inst = {"R1": {"type": "resistor", "settings": {"R": 1.0}}}
    got_i, got_w, got_p = flatten_subcircuits(
        inst, [{"from": "R1,p1", "to": "R1,p2"}], [], {})
    assert got_i == inst
    assert got_w == [["R1,p1", "R1,p2"]]   # dict wires normalized to pairs
    assert got_p == []
