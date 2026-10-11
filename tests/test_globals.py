"""Schematic-level globals reach the backend for every caller.

The editor injects ``ui = 1/globals.baud`` into pattern sources before a run,
but notebook (``photonflux.nb``), CLI flow and API callers send the saved
document, whose sources carry no ``ui``. ``simulate.run`` applies the global
itself so they all run at the rate the canvas shows -- including sources
inside subcircuit definitions, which the editor never injected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

import simulate  # noqa: E402


def _prbs_sch(baud: float, ui: float | None = None) -> dict:
    settings = {"order": 7, "rj_ui": 0.0, "sj_ui": 0.0, "dj_ui": 0.0}
    if ui is not None:
        settings["ui"] = ui
    return {
        "globals": {"baud": baud},
        "instances": {
            "SRC": {"type": "prbs", "settings": settings},
            "R1": {"type": "resistor", "settings": {"R": 50.0}},
            "G1": {"type": "ground", "settings": {}},
        },
        "wires": [["SRC,p1", "R1,p1"], ["SRC,p2", "G1,p1"],
                  ["R1,p2", "G1,p1"]],
        "probes": [{"name": "v", "at": "R1,p1"}],
    }


def test_global_baud_sets_the_pattern_ui():
    sch = _prbs_sch(25e9)
    out = simulate.with_global_ui(sch)
    assert out["instances"]["SRC"]["settings"]["ui"] == pytest.approx(40e-12)
    assert "ui" not in sch["instances"]["SRC"]["settings"]   # caller untouched


def test_explicit_ui_wins_and_nothing_is_copied():
    sch = _prbs_sch(25e9, ui=1e-10)
    assert simulate.with_global_ui(sch) is sch


def test_subcircuit_pattern_sources_get_the_global_rate():
    sch = {"globals": {"baud": 50e9}, "instances": {},
           "subcircuits": {"TX": {"schematic": {"instances": {
               "P": {"type": "prbs", "settings": {}}}}}}}
    out = simulate.with_global_ui(sch)
    inner = out["subcircuits"]["TX"]["schematic"]["instances"]["P"]
    assert inner["settings"]["ui"] == pytest.approx(20e-12)


def test_subcircuit_fet_makes_the_schematic_corner_dependent():
    sch = {"corner": "all", "instances": {},
           "subcircuits": {"INV": {"schematic": {"instances": {
               "M1": {"type": "sky130_nfet", "settings": {}}}}}}}
    assert len(simulate.sch_corners(sch)) == 5


def test_run_uses_the_global_rate_without_an_injected_ui():
    import jax

    jax.config.update("jax_enable_x64", True)
    res = simulate.run({"schematic": _prbs_sch(25e9),
                        "analysis": {"mode": "transient", "t_stop": 8e-9,
                                     "points": 2000, "link": {"probe": "v"}}})
    assert res["ok"], res.get("error")
    assert res["link"]["ui"] == pytest.approx(40e-12)
