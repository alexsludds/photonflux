"""Directed Verilog-A inputs are matched receivers (``webapp/catalog.py``).

The coherent VA models (ring_mod and its family, soa, edfa, the TW laser
slices, wmirror, ...) read their input field through an adapter on a
circulax optical node. The adapter used to be a zero-current tap, i.e. an
open circuit, so a reciprocal nodal element in front of it (a waveguide, a
splitter, a modulator) saw an open port and reflected; the CW laser, an
ideal source, reflects the wave back again. The input then read a standing
wave, not the transmitted field: 0.89 mW instead of 0.5 mW behind a 50/50
combiner, and behind a 1 mm guide anything from 1.1x to 68x the delivered
power depending on the length to a fraction of a wavelength.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))


@pytest.fixture(scope="module")
def sim():
    import jax

    jax.config.update("jax_enable_x64", True)
    import simulate

    return simulate


def _ring_input_power(sim, path: str, length_m: float = 1e-3) -> float:
    """Power [mW] a far-off-resonance ring_mod reads at its input when a
    1 mW laser feeds it through ``path``."""
    inst = {
        "LAS": {"type": "cw_laser",
                "settings": {"power": 1e-3, "wavelength_nm": 1310.0}},
        "GL": {"type": "ground", "settings": {}},
        "RM": {"type": "ring_mod",
               "settings": {"lambda_nm": 1310.0, "lambda_res_nm": 1300.0}},
        "GR": {"type": "ground", "settings": {}},
        "TO": {"type": "opt_term", "settings": {"return_loss_db": 200.0}},
    }
    wires = [{"from": "LAS,p2", "to": "GL,p1"},
             {"from": "RM,vp", "to": "GR,p1"},
             {"from": "RM,vn", "to": "GR,p1"},
             {"from": "RM,gnd", "to": "GR,p1"},
             {"from": "RM,pout", "to": "TO,p1"}]
    if path == "combiner":
        inst["SP"] = {"type": "splitter", "settings": {"split_ratio": 0.5}}
        inst["TS"] = {"type": "opt_term", "settings": {"return_loss_db": 200.0}}
        wires += [{"from": "LAS,p1", "to": "SP,p2"},
                  {"from": "SP,p3", "to": "TS,p1"},
                  {"from": "SP,p1", "to": "RM,pin"}]
    else:
        inst["WG"] = {"type": "waveguide",
                      "settings": {"length_m": length_m, "loss_dB_cm": 10.0}}
        wires += [{"from": "LAS,p1", "to": "WG,p1"},
                  {"from": "WG,p2", "to": "RM,pin"}]
    sch = {"instances": inst, "wires": wires,
           "probes": [{"name": "in", "at": "RM,pin"}]}
    res = sim.run({"schematic": sch, "analysis": {"mode": "dc"}})
    assert res["ok"], res.get("error")
    return next(r["value"] for r in res["rows"] if r.get("name") == "in")


def test_combiner_delivers_half_the_power(sim):
    assert _ring_input_power(sim, "combiner") == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("length_m", [1e-3, 1.0001e-3, 1.0002e-3])
def test_waveguide_in_front_of_a_ring_is_not_a_cavity(sim, length_m):
    # 10 dB/cm over ~1 mm: 1 dB, whatever the length is to a wavelength
    expect = 10.0 ** (-10.0 * length_m * 100.0 / 10.0)
    assert _ring_input_power(sim, "waveguide", length_m) == pytest.approx(
        expect, rel=1e-6)
