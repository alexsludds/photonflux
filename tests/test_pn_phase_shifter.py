"""Depletion PN-junction phase shifter (``webapp/catalog.py:pn_phase_shifter``)
and the coupling-modulated ring built from it (example 47).

Physics pinned here:
  * optics: the segment is exactly an OpticalWaveguide whose n_eff rises and
    loss falls with reverse bias along the depletion-width law
    g(V_R) = sqrt(1 + V_R/vbi) - 1, normalised so dphi(vref) = pi*vref*L/VpiL;
  * electrode: abrupt-junction charge Q = 2*Cj0*L*vbi*g(V_R), i.e.
    C_j = Cj0*L/sqrt(1 + V_R/vbi), plus leakage V_R/rleak;
  * example 47 (PAM4): the push-pull MZI coupler modulates the ring's
    coupling with the resonance fixed; the design point keeps the 2 Vpp swing
    on the under-coupled flank with evenly spaced levels (RLM), tolerant to
    +-0.5 pm of drift.
"""
from __future__ import annotations

import json
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

MODELS = catalog.build_models()
PN = dict(length_um=400.0, vpil_vcm=1.5, vref=2.0, vbi=0.85,
          loss_dB_cm=15.0, dloss_dB_cm=1.5, cj0_ff_um=0.3, rleak=1e9)


def _g(vr, vbi=0.85):
    return np.sqrt(1.0 + vr / vbi) - 1.0


def _out_field(comp: str, settings: dict, vr: float = 0.0) -> complex:
    """Field at the far end of one segment fed by a 1 mW laser."""
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LAS": {"component": "cw_laser",
                    "settings": {"power": 1e-3, "wavelength_nm": 1310.0}},
            "DUT": {"component": comp, "settings": settings},
            "TT": {"component": "opt_term"},
        },
        "connections": {"GND,p1": ("LAS,p2",), "LAS,p1": "DUT,p1",
                        "DUT,p2": "TT,p1"},
        "ports": {"out": "DUT,p2"},
    }
    if comp == "pn_phase_shifter":
        net["instances"]["VR"] = {"component": "vdc", "settings": {"V": vr}}
        net["connections"]["GND,p1"] += ("VR,p2", "DUT,an")
        net["connections"]["VR,p1"] = "DUT,cat"
    c = compile_circuit(net, MODELS, is_complex=True)
    return complex(c.port(c.dc(), "out"))


@pytest.mark.parametrize("vr", [0.0, 1.0, 2.0, 4.0])
def test_equals_biased_waveguide(vr):
    # same field as a plain waveguide with n_eff + dn(V_R) and loss(V_R)
    gn = _g(vr) / _g(PN["vref"])
    dn = 1310e-7 * PN["vref"] / (2 * PN["vpil_vcm"]) * gn
    ref = {"length_um": PN["length_um"], "neff": 2.4 + dn,
           "loss_dB_cm": PN["loss_dB_cm"] - PN["dloss_dB_cm"] * gn}
    e_pn = _out_field("pn_phase_shifter", PN, vr)
    e_wg = _out_field("waveguide", ref)
    assert abs(e_pn - e_wg) < 1e-9 * abs(e_wg)


def test_vpil_sets_phase_at_vref():
    # reverse bias delays the field (n rises): dphi(vref) = pi*vref*L/VpiL
    e0 = _out_field("pn_phase_shifter", PN, 0.0)
    e2 = _out_field("pn_phase_shifter", PN, PN["vref"])
    dphi = -np.angle(e2 / e0)
    expect = np.pi * PN["vref"] * PN["length_um"] * 1e-4 / PN["vpil_vcm"]
    # the laser/terminator bench adds a ~0.3% offset (a plain waveguide shows
    # the same); the exact law is pinned against Waveguide above
    assert dphi == pytest.approx(expect, rel=5e-3)
    # sublinear: twice the bias gives well under twice the phase
    e4 = _out_field("pn_phase_shifter", PN, 2 * PN["vref"])
    assert -np.angle(e4 / e0) < 1.7 * dphi


def test_junction_charge_and_leakage():
    dev = catalog._pn_phase_shifter()(**PN)
    cj0 = PN["cj0_ff_um"] * 1e-15 * PN["length_um"]
    for vr in (0.0, 1.0, 3.0):
        f, q = dev(cat=vr, an=0.0)
        assert float(q["cat"]) == pytest.approx(
            2 * cj0 * PN["vbi"] * _g(vr), rel=1e-12, abs=1e-30)
        assert float(q["an"]) == pytest.approx(-float(q["cat"]))
        assert float(f["cat"]) == pytest.approx(vr / PN["rleak"])
        # small-signal capacitance: Cj0*L/sqrt(1 + V_R/vbi)
        cj = jax.grad(lambda v: dev(cat=v, an=0.0)[1]["cat"].real)(vr)
        assert float(cj) == pytest.approx(
            cj0 / np.sqrt(1 + vr / PN["vbi"]), rel=1e-9)
    # forward bias past the tangent point stays finite (no NaN for Newton)
    f, q = dev(cat=-1.0, an=0.0)
    assert np.isfinite(float(q["cat"])) and np.isfinite(float(f["cat"]))


def _rlm(p):
    """IEEE 802.3 ratio of level mismatch from four ascending levels."""
    vm = (p[0] + p[3]) / 2
    es1 = (p[1] - vm) / (p[0] - vm)
    es2 = (p[2] - vm) / (p[3] - vm)
    return min(3 * es1, 3 * es2, 2 - 3 * es1, 2 - 3 * es2)


def _ex47(drive_dc: bool = True):
    """Example 47; with drive_dc the PRBS becomes a DC source DRV."""
    ex = json.loads((_ROOT / "webapp/examples/47_coupling_modulated_ring.json")
                    .read_text())
    sch, prbs = ex["schematic"], ex["schematic"]["instances"]["PRBS"]
    if drive_dc:
        sch["instances"]["DRV"] = {**prbs, "type": "vdc", "settings": {}}
        del sch["instances"]["PRBS"]
        for w in sch["wires"]:
            for e in ("from", "to"):
                w[e] = w[e].replace("PRBS,", "DRV,")
    return ex, prbs["settings"]


def _set_lambda(sch, lam):
    for inst in sch["instances"].values():
        if "wavelength_nm" in inst["settings"]:
            inst["settings"]["wavelength_nm"] = lam


def _pam4_levels(sch, ps):
    import simulate

    v = ps["v0"] + (ps["v1"] - ps["v0"]) * np.arange(4) / 3
    r = simulate.run({"schematic": sch, "analysis": {
        "mode": "dcsweep", "instance": "DRV", "param": "V",
        "values": v.tolist()}})
    assert r["ok"], r.get("error")
    return np.asarray(next(t for t in r["traces"] if t["name"] == "thru")
                      ["values"])


def test_example_47_pam4_levels():
    ex, ps = _ex47()
    sch = ex["schematic"]
    lam0 = sch["instances"]["LAS"]["settings"]["wavelength_nm"]
    p = _pam4_levels(sch, ps)
    assert np.all(np.diff(p) > 0)                # monotonic: one flank only
    assert _rlm(p) > 0.97
    assert 10 * np.log10(p[3] / p[0]) == pytest.approx(4.0, abs=0.3)
    for d in (-5e-4, 5e-4):                      # +-0.5 pm drift tolerance
        _set_lambda(sch, lam0 + d)
        pd = _pam4_levels(sch, ps)
        assert np.all(np.diff(pd) > 0) and _rlm(pd) > 0.94


def test_example_47_coupling_flank():
    # spectral view at the two drive extremes: the notch gets shallower AND
    # narrower (under-coupled flank) while the resonance stays put
    import simulate

    ex, ps = _ex47()
    lam0 = ex["schematic"]["instances"]["LAS"]["settings"]["wavelength_nm"]
    r = simulate.run({"schematic": ex["schematic"], "analysis": {
        "mode": "dcsweep", "instance": "*", "param": "wavelength_nm",
        "start": lam0 - 0.1, "stop": lam0 + 0.1, "points": 4001,
        "step_instance": "DRV", "step_param": "V",
        "step_values": [ps["v0"], ps["v1"]]}})
    assert r["ok"], r.get("error")
    x = np.asarray(r["x"])
    depth, lres, fwhm = [], [], []
    for tr in r["traces"]:
        if tr["probe"] != "thru":
            continue
        y = np.asarray(tr["values"])
        j = int(np.argmin(y))
        depth.append(y[j] / y.max())
        lres.append(x[j])
        fwhm.append(np.ptp(x[y < (y.max() + y[j]) / 2]))
    assert depth[1] > depth[0] and fwhm[1] < fwhm[0]
    assert abs(lres[1] - lres[0]) < 2e-3


def test_example_47_transient_eye_levels():
    # the shipped PRBS transient + link report at 53.125 GBd: error-free, with
    # the same evenly spaced levels (quasi-static optics; jitter only moves
    # edges)
    import simulate

    ex, ps = _ex47(drive_dc=False)
    ps["ui"] = 1.0 / ex["schematic"]["globals"]["baud"]   # the editor injects this
    r = simulate.run({"schematic": ex["schematic"], "analysis": ex["analysis"]})
    assert r["ok"], r.get("error")
    assert r["link"]["counted"]["sym_errors"] == 0
    assert _rlm(np.sort(r["link"]["qfit"]["levels"])) > 0.97
