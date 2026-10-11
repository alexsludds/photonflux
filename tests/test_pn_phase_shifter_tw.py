"""Traveling-wave PN phase shifter (``webapp/catalog.py:pn_phase_shifter_tw``).

Pinned here:
  * at DC (every section at the same bias) the field equals the lumped
    PN-Junction Phase Shifter's: same depletion-law phase and loss;
  * the electrode is a loaded transmission line: a step from a matched source
    sees Z_L = sqrt(L' / (C' + Cj')) and arrives at e_out after
    length * c * sqrt(L' (C' + Cj')) / c -- the junction capacitance loads
    and slows the line;
  * RF / optical walk-off is real: the electro-optic step response of an MZI
    arm stretches as the optical group index moves away from the loaded RF
    index, by about the walk-off window length * |n_L - n_g| / c;
  * the automatic section count (one per <= 100 um, 4..16).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from circulax import compile_circuit  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "webapp"))
sys.path.insert(0, str(_ROOT))

import catalog  # noqa: E402
import simulate  # noqa: E402

C0 = 299792458.0
Z0, NRF, CJ0 = 60.0, 2.4, 0.3e-15 / 1e-6        # defaults: Ohm, -, F/m
VBI = 0.85


def _sch(kind: str, vb: float, step: float = 0.0, L_um: float = 1000.0,
         n_g: float = 4.0, mzi_arm2: float | None = None) -> dict:
    """Laser -> DUT (-> MZI combiner) ; TW electrode: 40 Ohm source and
    termination into the bias (vb), ref grounded."""
    inst = {
        "LAS": {"type": "cw_laser", "settings": {"power": 1e-3,
                                                 "wavelength_nm": 1310.0}},
        "GND": {"type": "ground", "settings": {}},
        "TT": {"type": "opt_term", "settings": {}},
        "VP": {"type": "vpulse", "settings": {
            "v1": vb, "v2": vb + step, "td": 20e-12, "tr": 2e-12,
            "tf": 2e-12, "pw": 1.0, "per": 2.0}},
    }
    wires = [("LAS,p2", "GND,p1"), ("VP,p2", "GND,p1")]
    common = {"length_m": L_um * 1e-6, "n_group": n_g}
    if kind == "tw":
        inst.update({
            "DUT": {"type": "pn_phase_shifter_tw", "settings": common},
            "RS": {"type": "resistor", "settings": {"R": 40.0}},
            "RT": {"type": "resistor", "settings": {"R": 40.0}},
            "VT": {"type": "vdc", "settings": {"V": vb}}})
        wires += [("VP,p1", "RS,p1"), ("RS,p2", "DUT,e_in"),
                  ("DUT,e_out", "RT,p1"), ("RT,p2", "VT,p1"),
                  ("VT,p2", "GND,p1"), ("DUT,ref", "GND,p1")]
        probes = [{"name": "vin", "at": "DUT,e_in"},
                  {"name": "vout", "at": "DUT,e_out"}]
    else:
        inst["DUT"] = {"type": "pn_phase_shifter",
                       "settings": {**common, "vpil_vcm": 2.0,
                                    "loss_dB_cm": 12.0}}
        wires += [("VP,p1", "DUT,cat"), ("DUT,an", "GND,p1")]
        probes = []
    if mzi_arm2 is None:
        wires += [("LAS,p1", "DUT,p1"), ("DUT,p2", "TT,p1")]
        probes.append({"name": "out", "at": "DUT,p2"})
    else:
        inst.update({
            "SP": {"type": "splitter", "settings": {"split_ratio": 0.5}},
            "CB": {"type": "splitter", "settings": {"split_ratio": 0.5}},
            "WG2": {"type": "waveguide", "settings": {
                "length_m": mzi_arm2, "loss_dB_cm": 0.0, "n_group": n_g}}})
        wires += [("LAS,p1", "SP,p1"), ("SP,p2", "DUT,p1"),
                  ("DUT,p2", "CB,p2"), ("SP,p3", "WG2,p1"),
                  ("WG2,p2", "CB,p3"), ("CB,p1", "TT,p1")]
        probes.append({"name": "out", "at": "CB,p1"})
    return {"instances": inst, "probes": probes,
            "wires": [{"from": a, "to": b} for a, b in wires]}


def _dc_field(sch: dict) -> complex:
    net, meta = simulate.schematic_to_netlist(sch)
    c = compile_circuit(net, simulate.build_models(ltis=meta["ltis"]),
                        is_complex=True)
    return complex(c.port(c.dc(), "out"))


@pytest.mark.parametrize("vb", [0.0, 2.0, 4.0])
def test_dc_equals_lumped(vb):
    tw, lumped = _dc_field(_sch("tw", vb)), _dc_field(_sch("lumped", vb))
    assert abs(tw - lumped) < 1e-6 * abs(lumped)        # Newton tolerance


def _cross(t, y, level):
    i = int(np.nonzero(y >= level)[0][0])
    return t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1])


def test_loaded_line_impedance_and_delay():
    vb, step = 2.0, 0.4
    r = simulate.run({"schematic": _sch("tw", vb, step),
                      "analysis": {"mode": "transient", "t_stop": 100e-12,
                                   "points": 4000, "dtmax": 0.05e-12}})
    assert r["ok"], r.get("error")
    t = np.asarray(r["x"])
    tr = {q["name"]: np.asarray(q["values"]) for q in r["traces"]}
    vin, vout = tr["vin"], tr["vout"]
    # loaded line at the bias: C' + Cj'(V) with Cj' = Cj0 / sqrt(1 + V/vbi)
    lp, cp = Z0 * NRF / C0, NRF / (Z0 * C0)
    ctot = cp + CJ0 / np.sqrt(1 + vb / VBI)
    z_l, n_l = np.sqrt(lp / ctot), C0 * np.sqrt(lp * ctot)
    t_in = _cross(t, vin, vin[0] + 0.5 * (vin[-1] - vin[0]))
    t_out = _cross(t, vout, vout[0] + 0.5 * (vout[-1] - vout[0]))
    # the series resistance partly hides Cj on a fast edge: a few % faster
    assert t_out - t_in == pytest.approx(1e-3 * n_l / C0, rel=0.08)
    plateau = vin[np.searchsorted(t, t_in + 6e-12)] - vin[0]
    assert 40.0 * plateau / (step - plateau) == pytest.approx(z_l, rel=0.08)


def _eo_rise(n_g: float) -> float:
    """10-90 % electro-optic step response of a quadrature-biased MZI arm."""
    base = 1000e-6
    r = simulate.run({"schematic": _sch("tw", 2.0, mzi_arm2=base, n_g=n_g),
                      "analysis": {"mode": "dcsweep", "instance": "WG2",
                                   "param": "length_m", "start": base,
                                   "stop": base + 1.31e-6 / 2.4,
                                   "points": 721}})
    out = lambda res: np.asarray(next(  # noqa: E731
        q["values"] for q in res["traces"] if q["name"] == "out"))
    x, p = np.asarray(r["x"]), out(r)
    mid = 0.5 * (p.max() + p.min())
    quad = float(x[np.argmin(np.abs(p - mid) + 1e9 * (np.gradient(p) < 0))])
    r = simulate.run({"schematic": _sch("tw", 2.0, step=0.2, mzi_arm2=quad,
                                        n_g=n_g),
                      "analysis": {"mode": "transient", "t_stop": 80e-12,
                                   "points": 3200, "dtmax": 0.05e-12}})
    assert r["ok"], r.get("error")
    t, y = np.asarray(r["x"]), out(r)
    d = (y - y[0]) / (y[-1] - y[0])
    return float(t[np.nonzero(d >= 0.9)[0][0]] - t[np.nonzero(d >= 0.1)[0][0]])


def test_walk_off_stretches_the_eo_response():
    n_l = C0 * np.sqrt(Z0 * NRF / C0 * (NRF / (Z0 * C0)
                                        + CJ0 / np.sqrt(1 + 2.0 / VBI)))
    matched, fast_light = _eo_rise(n_l), _eo_rise(2.0)
    window = 1e-3 * abs(n_l - 2.0) / C0                 # ~5 ps
    assert fast_light > matched + 0.1 * window
    # rise times add roughly in quadrature with the walk-off window (10-90 %
    # of a uniform window is 0.8 of its length)
    assert fast_light == pytest.approx(
        np.hypot(matched, 0.8 * window), rel=0.15)


def test_section_count_rule():
    assert [catalog.tw_sections(0, L) for L in
            (100e-6, 500e-6, 1.5e-3, 5e-3)] == [4, 5, 15, 16]
    assert catalog.tw_sections(6, 1e-3) == 6
