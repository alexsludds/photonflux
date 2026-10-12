"""Small-signal AC and noise analyses (``webapp/simulate.py``).

* Noise: the output noise of a resistor across a capacitor integrates to
  kT/C whatever R is -- the textbook check that the 4kTR source, the adjoint
  transfer and the frequency integral are all right.
* AC: the SKY130 nfet f_T example. Its collapsed OSDI substrate nodes form
  an island tied only to each other; the DC solve regularises it with the
  solver's 1 nS leak, but the AC matrix lacked the leak and was singular at
  every frequency.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp"))

K_B = 1.380649e-23


@pytest.fixture(scope="module")
def sim():
    import jax

    jax.config.update("jax_enable_x64", True)
    import simulate

    return simulate


@pytest.mark.parametrize("r_ohm", [1e3, 1e5])
def test_rc_noise_integrates_to_kt_over_c(sim, r_ohm):
    c_f = 1e-12
    sch = {"instances": {
        "R1": {"type": "resistor", "settings": {"R": r_ohm}},
        "C1": {"type": "capacitor", "settings": {"C": c_f}},
        "G": {"type": "ground", "settings": {}}},
        "wires": [["R1,p1", "C1,p1"], ["R1,p2", "G,p1"], ["C1,p2", "G,p1"]],
        "probes": [{"name": "v", "at": "R1,p1"}]}
    f_c = 1.0 / (2.0 * np.pi * r_ohm * c_f)
    res = sim.run({"schematic": sch, "analysis": {
        "mode": "noise", "probe": "v", "f_start": f_c * 1e-4,
        "f_stop": f_c * 1e4, "points": 2001}})
    assert res["ok"], res.get("error")
    # flat 4kTR below the corner ...
    assert res["traces"][0]["values"][0] ** 2 == pytest.approx(
        4.0 * K_B * 300.0 * r_ohm, rel=1e-3)
    # ... and kT/C in total
    assert res["rms"] ** 2 == pytest.approx(K_B * 300.0 / c_f, rel=0.01)


def test_sky130_nfet_ft_example_solves(sim, sky130_available):
    if not sky130_available:
        pytest.skip("SKY130 PDK / openvaf-ir not installed")
    d = json.loads((REPO / "webapp/examples/06_sky130_nfet_ft.json")
                   .read_text())
    res = sim.run({"schematic": d["schematic"], "analysis": d["analysis"]})
    assert res["ok"], res.get("error")
    ft = [float(m.group(1)) for line in res["log"]
          if (m := re.search(r"f_T\(.*\) = ([0-9.]+) GHz", line))]
    assert ft, res["log"]
    # a 1 um x 150 nm SKY130 nfet at Vgs = 1.2 V: tens of GHz
    assert all(5.0 < f < 300.0 for f in ft)
