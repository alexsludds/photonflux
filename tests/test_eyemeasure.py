"""Eye-tab stateye scoring (webapp/eyemeasure.py, POST /api/eyemeasure)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

stateye = pytest.importorskip("stateye", reason="optional 'eye' extra")
if not hasattr(stateye, "ffe_lms"):
    pytest.skip("needs the patched stateye (equalizer)", allow_module_level=True)

import eyemeasure  # noqa: E402
import wavesrc  # noqa: E402

BAUD, SPS = 53.125e9, 24.93          # a fractional grid, as the canvas makes
UI = 1.0 / BAUD
FAST = {"nx": 128, "ny": 512}


def _trace(mode, nsym=600, post=0.25, noise=0.002):
    pat = {"mode": mode, "order": 13, "seed": 1, "ui": UI}
    fr = wavesrc._symbols(pat, nsym)
    lv = 0.1 + 0.3 * fr
    isi = lv + post * (np.concatenate([[lv[0]], lv[:-1]]) - lv.mean())
    t = np.arange(int(nsym * SPS)) * UI / SPS
    v = isi[np.minimum((t / UI).astype(int), nsym - 1)]
    v = v + noise * np.random.default_rng(0).standard_normal(v.size)
    return t, v, pat


def _payload(mode, cfg):
    t, v, pat = _trace(mode)
    return {"t": t.tolist(), "values": v.tolist(), "unit": "mW", "ui": UI,
            "levels": 4 if mode == "pam4" else 2, "skip_s": 10 * UI,
            "pattern": pat, "cfg": {**FAST, "s_noise": 0.005, **cfg}}


@pytest.mark.parametrize("method,extra", [
    ("mmse", {}), ("lms", {"mu": 0.1, "passes": 4}),
    ("manual", {"manual": "0, 1.25, -0.25"})])
def test_pam4_tdecq_methods(method, extra):
    r = eyemeasure.measure(_payload("pam4", {"method": method, **extra}))
    assert r["ok"], r.get("error")
    assert r["format"] == "PAM4" and r["family"] in ("outer", "xp")
    assert 0.0 < r["tdecq_db"] < r["tdecq_raw_db"]      # the FFE helps here
    assert len(r["taps"]) == (3 if method == "manual" else 5)
    assert sum(r["taps"]) == pytest.approx(1.0)
    assert len(r["scored"]["t"]) == len(r["scored"]["values"]) > 1000


def test_no_equalizer_scores_the_raw_eye():
    r = eyemeasure.measure(_payload("pam4", {"taps": 0}))
    assert r["ok"] and r["tdecq_db"] == r["tdecq_raw_db"] and r["taps"] == []


def test_nrz_reports_tdec():
    r = eyemeasure.measure(_payload("nrz", {"ber": 1e-12}))
    assert r["ok"], r.get("error")
    assert r["format"] == "NRZ" and r["tdec_db"] is not None
    assert r["oma_tdec_dbm"] is not None


def test_bad_input_is_a_clean_error():
    assert not eyemeasure.measure({"t": [0, 1], "values": [0, 1], "ui": UI})["ok"]
    bad = _payload("pam4", {"method": "manual", "manual": ""})
    assert not eyemeasure.measure(bad)["ok"]
