"""Testbench flows (``webapp/flow.py``): multi-step runs sharing variables.

Pinned here:
  * the expression rules (whitelisted AST, comparisons, an output may share
    its function's name) and the check / constraint-penalty arithmetic;
  * a DC flow on example 15's MZI: a wavelength sweep finds the null, the
    next step parks every instance there (``*.wavelength_nm``) and reads the
    operating point; the design for Apply resolves ``*`` to instances;
  * Optimize: Nelder-Mead over a flow variable re-running an inner step, the
    optimum carrying into later steps; ``only`` re-runs one step from prior
    variables;
  * stateye figures of merit on a synthetic PAM4 eye, and example 48's
    106.25 GBd design point (the flow's optimum) meeting its spec.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "webapp"))
sys.path.insert(0, str(_ROOT))

import flow  # noqa: E402
import simulate  # noqa: E402


def _example(name):
    return json.loads((_ROOT / "webapp/examples" / f"{name}.json").read_text())


def _run(sch, fl, **extra):
    res = simulate.run({"schematic": sch,
                        "analysis": {"mode": "flow", "flow": fl, **extra}})
    assert res["ok"], res.get("error")
    return res["flow"]


# --- expressions -------------------------------------------------------------

def test_expression_rules():
    assert flow._eval("a*2 + max(b, 1)", {"a": 1.0, "b": 3.0}) == 5.0
    assert flow._eval(2.5, {}) == 2.5
    # a call is the function, a bare name the variable (rlm = rlm(thru))
    funcs = {"rlm": lambda v: 10 * v}
    assert flow._eval("rlm(x) + rlm", {"rlm": 1.0, "x": 2.0}, funcs) == 21.0
    for bad in ("a.__class__", "a[0]", "(lambda: 1)()", "nope + 1", "open(1)"):
        with pytest.raises(flow.FlowError):
            flow._eval(bad, {"a": 1.0})
    # an undefined output (NaN) poisons a margin instead of vanishing in min()
    assert math.isnan(flow._eval("min(a, b)", {"a": 1.0, "b": float("nan")}))
    # arithmetic failures are flow errors (a step fails, not the whole run),
    # and integer powers overflow instead of hanging the server
    for bad in ("1/b", "sqrt(b - 1)", "9**9**9**9"):
        with pytest.raises(flow.FlowError):
            flow._eval(bad, {"b": 0.0})


def test_nelder_mead_memo_keeps_small_values_distinct():
    # a capacitance-sized parameter: rounding the memo key would merge every
    # point into one evaluation and return the start
    from optimize import nelder_mead

    seen = []

    def f(x):
        seen.append(float(x[0]))
        return ((x[0] - 3e-14) / 1e-14) ** 2

    simplex, fs = nelder_mead(f, np.array([8e-14]), np.array([1e-14]),
                              np.array([1e-13]), 30)
    best = simplex[int(np.argmin(fs))][0]
    assert len(set(seen)) == len(seen) > 10      # no repeats, many points
    assert len(seen) <= 30                        # never past the budget
    assert best == pytest.approx(3e-14, rel=0.05)


def test_check_and_penalty():
    c = flow.check("a >= 1", {"a": 2.0})
    assert c["ok"] and c["value"] == 2.0 and c["op"] == ">="
    assert not flow.check("a <= 1", {"a": None})["ok"]      # undefined fails
    assert flow.check("a*2 < b", {"a": 1.0, "b": 3.0})["ok"]
    assert flow.penalty(["a >= 1", "b <= 2"], {"a": 1.5, "b": 2.0}) == 0.0
    # relative violations: 0.5/1 + 1/2, times the penalty weight
    assert flow.penalty(["a >= 1", "b <= 2"], {"a": 0.5, "b": 3.0}) \
        == pytest.approx(100.0)
    assert math.isinf(flow.penalty(["a >= 1"], {"a": None}))
    with pytest.raises(flow.FlowError):
        flow.penalty(["a == 1"], {"a": 1.0})


# --- a DC flow on example 15's MZI --------------------------------------------

def _mzi_flow():
    return {
        "name": "MZI null", "vars": {"dl_um": 214.6},
        "set": {"WG2.length_m": "100e-6 + dl_um*1e-6"},
        "steps": [
            {"id": "null", "kind": "analysis", "title": "Find the null",
             "analysis": {"mode": "dcsweep", "instance": "*",
                          "param": "wavelength_nm", "start": 1309.0,
                          "stop": 1311.0, "points": 801},
             "outputs": {"lam_null": "xat_min(pout)",
                         "contrast": "dbp(min(pout)/max(pout))"},
             "checks": ["contrast < -20"]},
            {"id": "op", "kind": "analysis", "title": "Park on it",
             "set": {"*.wavelength_nm": "lam_null"},
             "analysis": {"mode": "dc"},
             "outputs": {"p_null": "pout"}, "checks": ["p_null < 1e-3"]},
        ]}


def test_flow_dc_chain_mzi():
    sch = _example("15_imbalanced_mzi_sweep")["schematic"]
    fr = _run(sch, _mzi_flow())
    assert fr["passed"], fr
    v = fr["vars"]
    # a null within the window, sub-sample accurate: parking on it leaves
    # far less than the 2.5 pm sweep grid would
    assert 1309.0 < v["lam_null"] < 1311.0
    assert v["contrast"] < -20 and v["p_null"] < 1e-3
    assert fr["origin"]["lam_null"] == "null" and fr["origin"]["dl_um"] == "vars"
    # Apply: the flow set + the last step's set, '*' resolved to instances
    design = {(d["inst"], d["param"]): d["value"] for d in fr["design"]}
    assert design[("WG2", "length_m")] == pytest.approx(314.6e-6)
    for inst in ("LAS1", "WG1", "WG2"):
        assert design[(inst, "wavelength_nm")] == pytest.approx(v["lam_null"])
    # the schematic itself is untouched
    assert sch["instances"]["WG2"]["settings"]["length_m"] == pytest.approx(
        314.6e-6)
    assert sch["instances"]["LAS1"]["settings"]["wavelength_nm"] == 1310.0


def test_flow_optimize_carries_forward_and_only():
    sch = _example("15_imbalanced_mzi_sweep")["schematic"]
    fl = _mzi_flow()
    # at 1310 nm the output falls monotonically from 0.89 mW (dl 214.3 um)
    # to ~0 (214.5 um): one crossing of 0.3 mW inside the box
    fl["vars"]["dl_um"] = 214.45
    fl["steps"] = [
        {"id": "op", "kind": "analysis", "title": "Output at 1310 nm",
         "analysis": {"mode": "dc"}, "outputs": {"p": "pout"}},
        {"id": "opt", "kind": "optimize", "title": "Hit 0.3 mW",
         "inner": ["op"], "params": [{"var": "dl_um", "min": 214.3,
                                      "max": 214.55}],
         "objective": "(p - 0.3)**2", "maximize": False,
         "constraints": ["p >= 0.25"], "iters": 24},
        {"id": "after", "kind": "analysis", "title": "Re-read",
         "analysis": {"mode": "dc"}, "outputs": {"p_after": "pout"},
         "checks": ["p_after > 0.28", "p_after < 0.32"]},
    ]
    fr = _run(sch, fl)
    assert fr["passed"], [s.get("checks") for s in fr["steps"]]
    opt = fr["steps"][1]
    assert opt["status"] == "ok" and len(opt["history"]) <= 24
    assert 214.3 <= fr["vars"]["dl_um"] <= 214.55
    assert fr["origin"]["dl_um"] == "opt"
    # the optimum carried into the next step
    assert fr["vars"]["p_after"] == pytest.approx(fr["vars"]["p"], rel=1e-9)
    # re-run one step from the prior variables: the earlier ones are skipped
    again = _run(sch, fl, only="after",
                 prior={"vars": fr["vars"], "overrides": fr["overrides"]})
    assert [s["status"] for s in again["steps"]] == ["skipped", "skipped", "ok"]
    assert again["vars"]["p_after"] == pytest.approx(fr["vars"]["p_after"])


def test_flow_errors():
    sch = _example("15_imbalanced_mzi_sweep")["schematic"]
    fl = _mzi_flow()
    dup = copy.deepcopy(fl)
    dup["steps"][1]["id"] = "null"
    res = simulate.run({"schematic": sch,
                        "analysis": {"mode": "flow", "flow": dup}})
    assert not res["ok"] and "unique" in res["error"]
    # a step error stops the flow; the run itself still reports per step
    bad = copy.deepcopy(fl)
    bad["steps"][0]["outputs"]["x0"] = "no_such_probe + 1"
    fr = _run(sch, bad)
    assert [s["status"] for s in fr["steps"]] == ["failed", "not run"]
    assert "no_such_probe" in fr["steps"][0]["error"] and not fr["passed"]
    # a spec without a comparison would always "pass": rejected up front
    spec = copy.deepcopy(fl)
    spec["steps"].append({"id": "ev", "kind": "evaluate", "inner": ["op"],
                          "probes": {"out": "pout"}, "fom": {"er_db": "-2"}})
    fr = _run(sch, spec)
    assert fr["steps"][2]["status"] == "failed"
    assert "spec" in fr["steps"][2]["error"]


# --- figures of merit ------------------------------------------------------------

def test_score_eye_figures_of_merit():
    pytest.importorskip("stateye", reason="optional 'eye' extra")
    import wavesrc

    ui = 1 / 53.125e9
    pat = {"mode": "pam4", "order": 13, "seed": 1, "ui": ui}
    nsym, sps = 800, 16
    lv = 0.1 + 0.3 * wavesrc._symbols(pat, nsym)          # 0.1 .. 0.4 mW
    t = np.arange(nsym * sps) * ui / sps
    v = lv[np.minimum((t / ui).astype(int), nsym - 1)]
    v = v + 1e-3 * np.random.default_rng(0).standard_normal(v.size)
    sc = flow.score_eye(t, v, pat, {"nx": 128, "ny": 512, "s_noise": 0.005})
    assert sc["format"] == "PAM4"
    assert sc["rlm"] == pytest.approx(1.0, abs=0.03)        # evenly spaced
    assert sc["er_db"] == pytest.approx(10 * np.log10(4), abs=0.3)
    fom = flow.fom_values(sc, p_in_mw=0.5)
    assert fom["rel_oma"] == pytest.approx(0.3 / 0.5, rel=0.05)
    assert fom["il_db"] == pytest.approx(10 * np.log10(0.5 / 0.4), abs=0.2)
    assert fom["oma_dbm"] == pytest.approx(10 * np.log10(0.3), abs=0.2)


def test_example_48_junction_rc_costs_tdecq():
    # example 48 drives the PN junctions through the driver's output
    # resistance plus their series resistance: at 106.25 GBd the junction RC
    # closes the eye. The same 300 um design with an ideal (0 Ohm) drive must
    # score clearly better -- the bandwidth limit example 47 leaves out.
    pytest.importorskip("stateye", reason="optional 'eye' extra")
    ex = _example("48_cmr_flow_106g")
    fl = ex["flow"]
    assert [s["id"] for s in fl["steps"]] == ["res", "eye", "opt", "eval"]
    out = {}
    for tag, extra in (("rc", {}), ("ideal", {"r_drv_ohm": 0.0,
                                              "rs_ohm_mm": 1e-6})):
        knobs = {**fl["vars"], **extra}
        fr = _run(ex["schematic"], fl, only="res",
                  prior={"vars": knobs, "overrides": {}})
        assert fr["steps"][0]["status"] == "ok", fr["steps"][0]
        knobs["lam_res"] = fr["vars"]["lam_res"]
        fr = _run(ex["schematic"], fl, only="eye",
                  prior={"vars": knobs, "overrides": {}})
        assert fr["steps"][1]["status"] == "ok", fr["steps"][1]
        out[tag] = fr["vars"]
    assert out["rc"]["lam_res"] == pytest.approx(out["ideal"]["lam_res"])
    assert out["rc"]["tdecq_db"] > out["ideal"]["tdecq_db"] + 1.0
    assert out["rc"]["rlm"] < out["ideal"]["rlm"]


def test_example_49_real_transit_and_optimizer_knobs():
    # example 49 closes its ring with delayed (real-transit) parts through a
    # 10 um, 0.3 dB/cm delay line, drives 30 Ohm electrodes through 30 Ohm,
    # and the optimizer owns DC1 and the TX FFE taps: the k1 variable must
    # reach DC1 (the resonance notch moves with it)
    ex = _example("49_cmr_flow_106g_tw")
    inst, fl = ex["schematic"]["instances"], ex["flow"]
    for name in ("PN_A", "PN_B", "WG_A", "WG_B", "WG_R1", "WG_R2"):
        assert inst[name]["settings"]["transit"] == "delayed", name
    for name in ("WG_R1", "WG_R2"):
        assert inst[name]["settings"]["loss_dB_cm"] == 0.3
    assert fl["vars"]["ring_um"] == 10.0
    assert fl["vars"]["r_drv_ohm"] == fl["vars"]["z0_ohm"] == 30.0
    assert fl["set"]["PRBS.ffe_pre"] == "ffe_pre"
    assert fl["set"]["PRBS.ffe_post"] == "ffe_post"
    opt = next(s for s in fl["steps"] if s["id"] == "opt")
    knobs = {p["var"] for p in opt["params"]}
    assert {"k1", "ffe_pre", "ffe_post"} <= knobs
    assert not {"k2", "z0_ohm", "ring_um"} & knobs
    notch = {}
    for k1 in (0.5, 0.35):
        fr = _run(ex["schematic"], fl, only="res",
                  prior={"vars": {**fl["vars"], "k1": k1}, "overrides": {}})
        assert fr["steps"][0]["status"] == "ok", fr["steps"][0]
        notch[k1] = fr["vars"]["notch_db"]
    assert abs(notch[0.5] - notch[0.35]) > 0.3
