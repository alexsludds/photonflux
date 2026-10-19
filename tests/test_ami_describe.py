"""AmiParams.describe and the web app's per-parameter AMI form backend."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyibisami")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))


@pytest.fixture(scope="module")
def rx():
    from photonflux import ibisami

    m = ibisami.load("pf_rx")
    yield m
    m.close()


def test_describe_lists_every_settable_parameter(rx):
    d = rx.params.describe()
    keys = [p["key"] for p in d]
    assert len(keys) == len(rx.params.values)
    ctle = next(p for p in d if p["key"] == "ctle_mode")
    assert ctle["group"] == "ctle" and ctle["choices"] == [0, 1, 2]
    assert ctle["tips"][2].startswith("adaptive") and ctle["default"] == 2
    fp1 = next(p for p in d if p["key"] == "ctle_fp1")
    assert fp1["format"] == "Range" and fp1["min"] < fp1["default"] < fp1["max"]
    # every key is one update() accepts, with its default as the value
    for p in d:
        rx.params.update({p["key"]: p["default"]})


def test_form_overrides_round_trip_through_parse_overrides(rx):
    import amilink

    desc = amilink.describe("pf_rx")
    assert "Modulation" not in [p["key"] for p in desc["params"]]
    # what the form writes: integers as integers, floats as typed, booleans
    text = "ctle_fp1=26500000000 ffe_ntaps=11 cdr_calibrate=False ffe_mu=0.002"
    over = amilink.parse_overrides(text)
    rx.params.update(over)
    assert rx.params["ffe_ntaps"] == 11 and rx.params["ctle_fp1"] == 26.5e9
    assert rx.params["cdr_calibrate"] is False
