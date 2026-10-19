"""Web-app blocks behind example 54: the Host PCB Link and the Modulator
Driver."""
import sys
from pathlib import Path

import numpy as np
import pytest

from photonflux import rfnet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))


@pytest.fixture(scope="module")
def sim():
    import jax

    jax.config.update("jax_enable_x64", True)
    import simulate

    return simulate


def _inst(typ, **s):
    return {"type": typ, "x": 0, "y": 0, "rot": 0, "settings": s}


def test_pcb_link_fit_matches_the_exact_terminated_transfer():
    import lti

    log = []
    _, payload = lti.build("rflink", {"loss_db": 10.0, "f_nyq": 26.5625e9}, log)
    A, B, C, D, zin = payload["real"]
    assert zin == 0.0 and "rel err" in log[0]
    assert float(log[0].split("rel err ")[1]) < 1e-2
    assert np.all(np.linalg.eigvals(A).real < 0)            # stable
    f = np.array([1e6, 1e9, 26.5625e9])
    fit = np.array([(C @ np.linalg.solve(2j * np.pi * fk * np.eye(len(A)) - A, B)).item()
                    + D for fk in f])
    ts = 1 / (64 * 26.5625e9)
    F = rfnet.grid(ts, 1 << 15)
    exact = rfnet.host_module_link(F)["net"].transfer(50.0, 50.0)
    want = np.interp(f, F, np.abs(exact))
    # magnitudes agree (the fit drops only the bulk flight delay)
    assert np.allclose(np.abs(fit), want, rtol=0.01)
    assert abs(fit[0]) == pytest.approx(0.5, rel=0.02)       # ~matched divider


def test_mod_driver_dc_transfer_and_limit(sim):
    def dc(vin, gain, peaking_db=6.0, vsat_d=3.0, vcm=2.0):
        sch = {"instances": {
            "VIN": _inst("vdc", V=vin), "G1": _inst("ground"),
            "DRV": _inst("mod_driver", peaking_db=peaking_db, gain=gain,
                         vsat_d=vsat_d, vcm=vcm),
            "RP": _inst("resistor", R=1e6), "RN": _inst("resistor", R=1e6),
            "G2": _inst("ground"), "G3": _inst("ground")},
            "wires": [{"from": "VIN,p1", "to": "DRV,inp"}, {"from": "VIN,p2", "to": "G1,p1"},
                      {"from": "DRV,outp", "to": "RP,p1"}, {"from": "RP,p2", "to": "G2,p1"},
                      {"from": "DRV,outn", "to": "RN,p1"}, {"from": "RN,p2", "to": "G3,p1"}],
            "probes": [{"name": "vp", "at": "DRV,outp"}, {"name": "vn", "at": "DRV,outn"}]}
        res = sim.run({"schematic": sch, "analysis": {"mode": "dc"}})
        assert res["ok"], res.get("log")
        rows = {r["name"]: float(r["value"]) for r in res["rows"]}
        return rows["vp"], rows["vn"]

    # small signal: DC gain = gain / peaking, around vcm
    vp, vn = dc(0.01, gain=10.0)
    k = 10 ** (6.0 / 20)
    assert vp - vn == pytest.approx(3.0 * np.tanh(10.0 * 0.01 / k / 3.0), rel=1e-6)
    assert (vp + vn) / 2 == pytest.approx(2.0, abs=1e-9)
    # large signal: the output stage limits at vsat_d
    vp, vn = dc(1.0, gain=50.0)
    assert vp - vn == pytest.approx(3.0, rel=1e-3)
