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


def _trace(mode, nsym=600, post=0.25, noise=0.002, ui=UI):
    pat = {"mode": mode, "order": 13, "seed": 1, "ui": ui}
    fr = wavesrc._symbols(pat, nsym)
    lv = 0.1 + 0.3 * fr
    isi = lv + post * (np.concatenate([[lv[0]], lv[:-1]]) - lv.mean())
    t = np.arange(int(nsym * SPS)) * ui / SPS
    v = isi[np.minimum((t / ui).astype(int), nsym - 1)]
    v = v + noise * np.random.default_rng(0).standard_normal(v.size)
    return t, v, pat


def _payload(mode, cfg, ui=UI):
    t, v, pat = _trace(mode, ui=ui)
    return {"t": t.tolist(), "values": v.tolist(), "unit": "mW", "ui": ui,
            "levels": 4 if mode == "pam4" else 2, "skip_s": 10 * ui,
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


def test_auto_settings_at_200g_lane():
    """At 106.25 GBd "auto" is the 802.3dj reference -- 15-tap FFE (3 pre)
    + 1 DFE tap inside the D3.2 limits, SER 4.56e-4 -- and FFE taps 0 is
    unequalized: no DFE-only equalizer sneaks in."""
    ui = 1.0 / 106.25e9
    r = eyemeasure.measure(_payload("pam4", {}, ui=ui))
    assert r["ok"], r.get("error")
    assert r["standard"] == "802.3dj" and r["ser_target"] == 4.56e-4
    assert len(r["taps"]) == 15 and len(r["dfe_b"]) == 1
    assert r["violations"] == []
    from stateye.equalization import tap_limit_violations
    assert tap_limit_violations(r["taps"], 3, r["dfe_b"]) == []
    off = eyemeasure.measure(_payload("pam4", {"taps": 0}, ui=ui))
    assert off["ok"] and off["dfe_b"] == [] and off["taps"] == []
    assert off["tdecq_db"] == off["tdecq_raw_db"]


def test_pam4_oma_outer_on_the_pattern_runs():
    """OMA and ER come from the pattern's run of 7 threes and of 6 zeros
    (symbols 5000 and 4096 of this PRBS-13Q), settled at 0.40 + 0.3 * 0.15
    and 0.10 - 0.3 * 0.15 through the post-cursor; stateye's own slicing
    counts stray runs on this eye and reads OMA_outer ~20 % low."""
    t, v, pat = _trace("pam4", nsym=5200, post=0.3)
    r = eyemeasure.measure({"t": t.tolist(), "values": v.tolist(),
                            "unit": "mW", "ui": UI, "levels": 4,
                            "skip_s": 10 * UI, "pattern": pat,
                            "cfg": {**FAST, "s_noise": 0.005}})
    assert r["ok"], r.get("error")
    assert r["family"] == "outer"
    assert r["oma"] == pytest.approx(0.39, abs=5e-3)
    assert r["er_db"] == pytest.approx(10 * np.log10(0.445 / 0.055), abs=0.15)


def test_nrz_reports_tdec():
    r = eyemeasure.measure(_payload("nrz", {"ber": 1e-12}))
    assert r["ok"], r.get("error")
    assert r["format"] == "NRZ" and r["tdec_db"] is not None
    assert r["oma_tdec_dbm"] is not None


def test_bad_input_is_a_clean_error():
    assert not eyemeasure.measure({"t": [0, 1], "values": [0, 1], "ui": UI})["ok"]
    bad = _payload("pam4", {"method": "manual", "manual": ""})
    assert not eyemeasure.measure(bad)["ok"]


def test_dfe_and_802_3dj_limits():
    """A DFE tap comes back in the standard's normalization, inside
    0..0.33; the reported equalizer is a legal 802.3dj one."""
    r = eyemeasure.measure(_payload("pam4", {"method": "mmse", "dfe": 1}))
    assert r["ok"], r.get("error")
    assert len(r["dfe_b"]) == 1 and 0.0 <= r["dfe_b"][0] <= 0.33
    assert r["violations"] == []
    from stateye.equalization import tap_limit_violations
    assert tap_limit_violations(r["taps"], 1, r["dfe_b"]) == []


def test_optimal_search_reports_progress():
    eyemeasure.STATUS.update(active=False, text="")
    seen = []
    orig = eyemeasure._status

    def spy(text, i=0, n=0, active=True):
        seen.append(text)
        orig(text, i, n, active)

    eyemeasure._status = spy
    try:
        r = eyemeasure.measure(_payload("pam4", {"method": "optimal",
                                                 "max_evals": 6, "dfe": 1}))
    finally:
        eyemeasure._status = orig
    assert r["ok"], r.get("error")
    assert any(t.startswith("TDECQ search: evaluation 6/6") for t in seen)
    assert r["violations"] == [] and not eyemeasure.STATUS["active"]
