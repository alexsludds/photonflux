"""Flows driving IBIS-AMI blocks: ``INST.ami:name`` set targets, Apply, and
the AMI Rx BER outputs (no simulation)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))


@pytest.fixture(scope="module")
def flow():
    import jax

    jax.config.update("jax_enable_x64", True)
    import flow as F

    return F


def _sch():
    return {"instances": {
        "TX1": {"type": "ami_tx", "x": 0, "y": 0, "rot": 0,
                "settings": {"model": "pf_tx", "ami_params": "tx_tap_p1=-0.1 tx_bw=6e10"}},
        "R1": {"type": "resistor", "x": 0, "y": 0, "rot": 0, "settings": {"R": 50.0}}},
        "wires": [], "probes": []}


def test_split_key_keeps_dotted_ami_names(flow):
    assert flow._split_key("TX1.ami:tx_tap_m1") == ("TX1", "ami:tx_tap_m1")
    assert flow._split_key("RX1.ami:ffe.ffe_ntaps") == ("RX1", "ami:ffe.ffe_ntaps")
    assert flow._split_key("R1.R") == ("R1", "R")


def test_ami_set_targets_merge_into_ami_params(flow):
    import amilink

    fl = {"set": {"TX1.ami:tx_tap_m1": "c", "TX1.ami:tx_tap_p1": "-0.2",
                  "R1.R": "40"}}
    sch = flow._prepare(_sch(), fl, {}, None, {"c": -0.15})
    over = amilink.parse_overrides(sch["instances"]["TX1"]["settings"]["ami_params"])
    assert over == {"tx_tap_p1": -0.2, "tx_bw": 6e10, "tx_tap_m1": -0.15}
    assert sch["instances"]["R1"]["settings"]["R"] == 40.0
    with pytest.raises(flow.FlowError):
        flow._prepare(_sch(), {"set": {"R1.ami:x": "1"}}, {}, None, {})


def test_design_folds_ami_parameters_into_one_setting(flow):
    fl = {"set": {"TX1.ami:tx_tap_m1": "c", "R1.R": "40"}}
    run = flow._Runner(_sch(), fl, [])
    design = {(d["inst"], d["param"]): d["value"] for d in run.design({"c": -0.1}, {})}
    assert ("TX1", "ami:tx_tap_m1") not in design
    assert "tx_tap_m1=-0.1" in design[("TX1", "ami_params")]
    assert "tx_tap_p1=-0.1" in design[("TX1", "ami_params")]
    assert design[("R1", "R")] == 40.0


def test_ami_ber_is_undefined_without_a_report(flow):
    res = {"kind": "transient", "x": [0.0, 1.0], "traces": []}
    ctx, _ = flow._result_ctx(res, _sch(), {}, {})
    assert math.isnan(ctx["ami_ber"]) and math.isnan(ctx["ami_ber_counted"])
    res["ami"] = {"stat": {"ber": 1e-9}, "td": {"counted": {"ber": 0.0}}}
    ctx, _ = flow._result_ctx(res, _sch(), {}, {})
    assert ctx["ami_ber"] == 1e-9 and ctx["ami_ber_counted"] == 0.0
