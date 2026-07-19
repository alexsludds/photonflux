"""Notebook Builder can define and instantiate subcircuits (ALE-78, AC2).

Pure document construction — no server, no JAX. Verifies the Builder emits the
``schematic.subcircuits`` shape the flattener consumes, and that the whole
document flattens to the expected primitive netlist.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp"))

from photonflux.nb import Builder, Schematic, SubcircuitDef  # noqa: E402
from subcircuit import flatten_subcircuits  # noqa: E402


def _build():
    b = Builder()
    # --- define a WDM drop subcircuit: ring + heater between four ports -----
    sub = b.subcircuit("wdm_drop", "WDM Drop")
    ring = sub.add("waveguide", "RING")
    heat = sub.add("resistor", "HEAT", R=1000.0)
    sub.wire(sub.port("in", "optical"), ring.p1)
    sub.wire(ring.p2, sub.port("out", "optical"))
    sub.wire(sub.port("vp", "electrical"), heat.p1)
    sub.wire(sub.port("vn", "electrical"), heat.p2)
    sub.probe(ring.p2, name="n_out")
    sub.export("resonance_nm", 1550.0, (ring, "wavelength_nm"))

    # --- instantiate it 4x on a shared laser bus ----------------------------
    las = b.add("cw_laser", "LAS")
    gnd = b.add("ground", "G")
    b.wire(las.p2, gnd.p1)
    for i, nm in enumerate([1548.0, 1549.0, 1550.0, 1551.0], start=1):
        x = b.instantiate("wdm_drop", ref=f"X{i}", resonance_nm=nm)
        b.wire(las.p1, x.port("in"))
        b.wire(x.port("vp"), gnd.p1)
        b.wire(x.port("vn"), gnd.p1)
        b.probe(x.port("out"), name=f"drop{i}")
    return b


def test_builder_emits_subcircuit_shape():
    doc = _build().doc(title="wdm bench")
    sch = doc["schematic"]
    assert "wdm_drop" in sch["subcircuits"]
    defn = sch["subcircuits"]["wdm_drop"]
    assert defn["label"] == "WDM Drop"
    assert defn["params"] == [
        {"name": "resonance_nm", "default": 1550.0,
         "bind": [{"instance": "RING", "param": "wavelength_nm"}]},
    ]
    # four instances of the subcircuit type placed at top level
    types = [i["type"] for i in sch["instances"].values()]
    assert types.count("wdm_drop") == 4
    # the definition carries port pseudo-components and inner parts
    inner = defn["schematic"]["instances"]
    ports = {i["settings"]["name"] for i in inner.values() if i["type"] == "port"}
    assert ports == {"in", "out", "vp", "vn"}


def test_builder_doc_flattens_to_primitives():
    sch = _build().doc()["schematic"]
    inst, wires, probes = flatten_subcircuits(
        sch["instances"], sch["wires"], sch["probes"], sch["subcircuits"])
    # each instance expands to a namespaced ring + heater, resonance baked in
    for i, nm in enumerate([1548.0, 1549.0, 1550.0, 1551.0], start=1):
        assert inst[f"X{i}.RING"]["settings"]["wavelength_nm"] == nm
        assert inst[f"X{i}.HEAT"]["settings"]["R"] == 1000.0
    assert all(i["type"] != "port" for i in inst.values())
    # hierarchical + external probe names both survive
    names = {p["name"] for p in probes}
    assert {"X1.n_out", "drop1", "X4.n_out", "drop4"} <= names


def test_export_binding_forms():
    b = Builder()
    sub = b.subcircuit("s")
    a = sub.add("waveguide", "A")
    sub.add("waveguide", "B")
    # string form, tuple-with-part form, and list-of-mixed form all normalize
    sub.export("p1", 1.0, "A.wavelength_nm")
    sub.export("p2", 2.0, (a, "neff"))
    sub.export("p3", 3.0, [("A", "length_um"), "B.wavelength_nm"])
    params = {p["name"]: p["bind"] for p in sub.params}
    assert params["p1"] == [{"instance": "A", "param": "wavelength_nm"}]
    assert params["p2"] == [{"instance": "A", "param": "neff"}]
    assert params["p3"] == [{"instance": "A", "param": "length_um"},
                            {"instance": "B", "param": "wavelength_nm"}]


def test_schematic_subcircuits_property_roundtrips():
    doc = _build().doc()
    sch = Schematic(doc)
    assert "wdm_drop" in sch.subcircuits
    # save/load round-trip preserves definitions (AC2)
    import json
    sch2 = Schematic(json.loads(json.dumps(doc)))
    assert sch2.subcircuits["wdm_drop"]["params"] == sch.subcircuits["wdm_drop"]["params"]


def test_subcircuitdef_is_not_a_document():
    b = Builder()
    sub = b.subcircuit("s")
    with pytest.raises(TypeError):
        sub.doc()
