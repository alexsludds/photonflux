"""Real optical transit (``transit = delayed``) on Waveguide and the
traveling-wave PN phase shifter (``webapp/catalog.py:_waveguide_transit``,
``_pn_phase_shifter_tw(n_stage > 0)``).

Pinned here:
  * at DC the delayed parts equal the quasi-static ones (same field);
  * a phase step launched into a delayed guide arrives L n_g / c later, with
    its edge intact (the Pade stages keep the group delay to < 1 %);
  * a ring closed by a delayed guide has a photon lifetime: on resonance an
    input power step leaves a through-port transient that decays by the
    round-trip amplitude |t a| once per round trip, where the quasi-static
    ring settles at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "webapp"))
sys.path.insert(0, str(_ROOT / "tests"))

import catalog  # noqa: E402
import simulate  # noqa: E402
from test_pn_phase_shifter_tw import _dc_field, _sch  # noqa: E402

C0 = 299792458.0
NEFF, NG, LAM = 2.4, 4.0, 1310.0


def _delayed(sch: dict) -> dict:
    for inst in sch["instances"].values():
        if inst["type"] in ("waveguide", "pn_phase_shifter_tw"):
            inst["settings"]["transit"] = "delayed"
    return sch


def _run(sch: dict, t_stop: float) -> tuple[np.ndarray, dict]:
    r = simulate.run({"schematic": sch, "analysis": {
        "mode": "transient", "t_stop": t_stop, "points": int(t_stop / 25e-15),
        "dtmax": 0.05e-12, "solver": "bdf2"}})
    assert r["ok"], r.get("error")
    return (np.asarray(r["x"]),
            {q["name"]: np.asarray(q["values"]) for q in r["traces"]})


@pytest.mark.parametrize("vb", [0.0, 2.0])
def test_tw_dc_equals_quasi_static(vb):
    qs, dl = _dc_field(_sch("tw", vb)), _dc_field(_delayed(_sch("tw", vb)))
    assert abs(dl - qs) < 1e-9 * abs(qs)


def test_stage_count():
    assert catalog.transit_stages(10e-6, NG) == 1
    assert catalog.transit_stages(1e-3, NG) == 27            # 13.3 ps / 0.5
    assert catalog.transit_stages(1e-3, NG, n_div=10) == 3   # per section
    assert catalog.transit_stages(1.0, NG) == catalog.TRANSIT_MAX_STAGES


def _launch(transit: str, step: float, ring_k: float | None = None,
            L: float = 1e-3) -> dict:
    """Laser -> lumped PN (phase step at 20 ps) -> 1 mm guide -> probe; or,
    with ring_k, the PN feeds an all-pass ring (coupler ring_k, guide L)."""
    inst = {
        "LAS": {"type": "cw_laser", "settings": {"power": 1e-3,
                                                 "wavelength_nm": LAM}},
        "GND": {"type": "ground", "settings": {}},
        "TT": {"type": "opt_term", "settings": {"return_loss_db": 200.0}},
        "VP": {"type": "vpulse", "settings": {
            "v1": 0.0, "v2": step, "td": 20e-12, "tr": 5e-12, "tf": 5e-12,
            "pw": 1.0, "per": 2.0}},
        "PN": {"type": "pn_phase_shifter", "settings": {
            "length_m": 1e-3, "vpil_vcm": 2.0, "loss_dB_cm": 0.0,
            "dloss_dB_cm": 0.0}},
        "WG": {"type": "waveguide", "settings": {
            "length_m": L, "loss_dB_cm": 1.0, "neff": NEFF, "n_group": NG,
            "center_wavelength_nm": LAM, "wavelength_nm": LAM,
            "transit": transit}},
    }
    w = [("LAS,p2", "GND,p1"), ("VP,p2", "GND,p1"), ("VP,p1", "PN,cat"),
         ("PN,an", "GND,p1"), ("LAS,p1", "PN,p1")]
    if ring_k is None:
        w += [("PN,p2", "WG,p1"), ("WG,p2", "TT,p1")]
        probes = [{"name": "out", "at": "WG,p2"}]
    else:
        inst["DC"] = {"type": "dir_coupler", "settings": {"coupling": ring_k}}
        w += [("PN,p2", "DC,p1"), ("DC,p3", "TT,p1"), ("DC,p4", "WG,p1"),
              ("WG,p2", "DC,p2")]
        probes = [{"name": "out", "at": "DC,p3"}]
    return {"instances": inst, "probes": probes,
            "wires": [{"from": a, "to": b} for a, b in w]}


def test_guide_delays_a_phase_step():
    """Interfere the delayed output with a quasi-static copy of the input
    phase: the beat moves L n_g / c after the drive, not at once."""
    tau = 1e-3 * NG / C0
    out = {}
    for transit in ("quasi-static", "delayed"):
        sch = _launch(transit, 2.0)
        inst = sch["instances"]
        inst.update({
            "SP": {"type": "splitter", "settings": {"split_ratio": 0.5}},
            "CB": {"type": "splitter", "settings": {"split_ratio": 0.5}},
            "REF": {"type": "waveguide", "settings": {
                "length_m": 1e-3, "loss_dB_cm": 1.0}}})
        sch["wires"] = [x for x in sch["wires"]
                        if x["from"] != "LAS,p1" and x["to"] != "TT,p1"]
        sch["wires"] += [{"from": a, "to": b} for a, b in (
            ("LAS,p1", "SP,p1"), ("SP,p2", "PN,p1"), ("SP,p3", "REF,p1"),
            ("WG,p2", "CB,p2"), ("REF,p2", "CB,p3"), ("CB,p1", "TT,p1"))]
        sch["probes"] = [{"name": "out", "at": "CB,p1"}]
        t, tr = _run(sch, 60e-12)
        out[transit] = tr["out"]
    qs, dl = out["quasi-static"], out["delayed"]
    assert dl[0] == pytest.approx(qs[0], rel=1e-9)          # same DC point
    assert dl[-1] == pytest.approx(qs[-1], rel=1e-6)
    # the delayed beat is the quasi-static one shifted by tau
    late = t > 20e-12 + tau - 2e-12
    ref = np.interp(t - tau, t, qs)
    swing = abs(qs[-1] - qs[0])
    assert np.max(np.abs(dl - ref)[late]) < 0.1 * swing
    mid = lambda y: t[np.argmax(np.abs(y - y[0]) > 0.5 * swing)]  # noqa: E731
    assert mid(dl) - mid(qs) == pytest.approx(tau, rel=0.02)


def test_ring_photon_lifetime():
    """All-pass ring on resonance (L = m lambda / neff) under a small input
    power step: every field stays real, so the through amplitude settles as
    A + B (t a)^n -- its round-trip differences fall by |t a| per round
    trip. The quasi-static ring settles at once."""
    k, L = 0.3, 1832 * LAM * 1e-9 / NEFF                    # resonant length
    tau = L * NG / C0
    ta = np.sqrt(1 - k) * 10 ** (-1.0 * L * 100 / 20)
    t0 = 20e-12
    amp = {}
    for transit in ("quasi-static", "delayed"):
        sch = _launch(transit, 0.0, ring_k=k, L=L)
        inst = sch["instances"]
        for name in ("PN", "VP"):
            inst.pop(name)
        inst["PM"] = {"type": "pulse_mod", "settings": {
            "p_on": 1e-3, "p_off": 0.9e-3, "t0": t0, "t1": 1.0,
            "tr": 1e-12}}
        sch["wires"] = [{"from": a, "to": b} for a, b in (
            ("LAS,p2", "GND,p1"), ("LAS,p1", "PM,pin"), ("PM,pout", "DC,p1"),
            ("DC,p3", "TT,p1"), ("DC,p4", "WG,p1"), ("WG,p2", "DC,p2"))]
        t, tr = _run(sch, t0 + 7.5 * tau)
        amp[transit] = np.sqrt(tr["out"])
    at = lambda y, n: y[np.searchsorted(t, t0 + (n + 0.5) * tau)]  # noqa: E731
    qs = amp["quasi-static"]
    assert at(qs, 1) == pytest.approx(qs[-1], rel=1e-6)    # settles at once
    s = np.array([at(amp["delayed"], n) for n in range(1, 7)])
    d = s[:-1] - s[1:]
    assert d[1:] / d[:-1] == pytest.approx(np.full(4, ta), rel=0.03)


def test_long_guide_warns_and_numeric_transit_is_stiff():
    # past the stage cap the netlister says so; a numeric transit flag (as a
    # flow `set` writes it) builds the delayed model and picks BDF2 too
    sch = _launch("delayed", 0.0, L=1e-2)
    _net, meta = simulate.schematic_to_netlist(sch)
    assert any("WG" in m and "Pade stages" in m for m in meta["lti_log"])
    sch = _launch("delayed", 0.0)
    sch["instances"]["WG"]["settings"]["transit"] = 1.0
    _net, meta = simulate.schematic_to_netlist(sch)
    assert meta["stiff"] and any(k.startswith("wg_transit@") for k in meta["ltis"])
    assert not [m for m in meta["lti_log"] if "Pade stages" in m]
