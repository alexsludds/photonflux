"""Pattern-source levels are runtime parameters (webapp/wavesrc.level_map).

The netlist bakes a PRBS / QAM drive's unit-level shape (v0 = 0, v1 = 1) and
passes v0 and v1 - v0 at run time, so a level change (a bias sweep, an
optimizer step on the drive) reuses the compiled circuit instead of
recompiling it. That is exact because the drive is affine in its levels;
the RLM-warped ladder is not, so it keeps its levels baked.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import simulate  # noqa: E402
import wavesrc  # noqa: E402

UI = 100e-12
SPAN = 200 * UI


def _drive(build, settings, t):
    wt, wv = build(settings, SPAN)
    return np.interp(t, wt, wv)


@pytest.mark.parametrize("extra", [
    {"mode": "nrz"},
    {"mode": "pam4", "order": 9, "ffe_pre_db": 3.0, "ffe_post_db": 6.0},
    {"mode": "pulse"},
    {"mode": "pam4", "rj_ui": 0.05, "dj_ui": 0.05},
])
@pytest.mark.parametrize("v0, v1", [(-0.4, 1.3), (2.0, 2.0), (1.5, -0.5)])
def test_prbs_drive_is_affine_in_its_levels(extra, v0, v1):
    s = {"ui": UI, "tr": 20e-12, **extra, "v0": v0, "v1": v1}
    t = np.linspace(0.0, SPAN, 20001)
    off, gain = wavesrc.level_map(s)
    unit = _drive(wavesrc.prbs_waveform, dict(s, v0=0.0, v1=1.0), t)
    np.testing.assert_allclose(off + gain * unit,
                               _drive(wavesrc.prbs_waveform, s, t),
                               rtol=0, atol=1e-12)


def test_qam_drive_is_affine_in_its_levels():
    s = {"ui": UI, "mode": "qam", "qam": "qam16", "qam_drive": "q",
         "v0": -0.7, "v1": 0.9}
    t = np.linspace(0.0, SPAN, 20001)
    off, gain = wavesrc.level_map(s)
    unit = _drive(wavesrc.qam_drive_waveform, dict(s, v0=0.0, v1=1.0), t)
    np.testing.assert_allclose(off + gain * unit,
                               _drive(wavesrc.qam_drive_waveform, s, t),
                               rtol=0, atol=1e-12)


def test_rlm_ladder_keeps_its_levels_baked():
    assert wavesrc.level_map({"v0": -1.0, "v1": 1.0, "rlm_vpi": 3.0}) is None


def _prbs_schematic(v0, v1):
    return {
        "instances": {
            "SRC": {"type": "prbs", "settings": {
                "mode": "pam4", "order": 7, "v0": v0, "v1": v1, "ui": UI}},
            "R": {"type": "resistor", "settings": {"R": 50.0}},
            "G": {"type": "ground"},
        },
        "wires": [["SRC,p1", "R,p1"], ["SRC,p2", "G,p1"], ["R,p2", "G,p1"]],
        "probes": [{"name": "v", "at": "R,p1"}],
    }


def test_level_change_keeps_the_compiled_source():
    """Same component key (so the same compiled program) for any levels;
    the levels ride along as the instance's offset / gain."""
    a, _ = simulate.schematic_to_netlist(_prbs_schematic(0.5, 3.5), SPAN)
    b, _ = simulate.schematic_to_netlist(_prbs_schematic(1.2, 1.2), SPAN)
    sa, sb = a["instances"]["SRC"], b["instances"]["SRC"]
    assert sa["component"] == sb["component"]
    assert sa["settings"] == {"offset": 0.5, "gain": 3.0}
    assert sb["settings"] == {"offset": 1.2, "gain": 0.0}
