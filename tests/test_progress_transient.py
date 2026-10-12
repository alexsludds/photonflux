"""The web UI's transient progress bar (simulate._progress_save_fn).

Each report is a host round-trip from inside the compiled solver loop, so the
bar rides a coarse save schedule of at most _PROGRESS_POINTS points instead of
one report per saved point (10k reports cost ~5 s on example 49's eye run);
the saved series itself is the same with the bar on or off.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp"))

import simulate  # noqa: E402
from progress import PROGRESS  # noqa: E402


def _run():
    doc = json.loads((REPO / "webapp/examples/20_rlc_step_ringing.json")
                     .read_text())
    res = simulate.run({"schematic": doc["schematic"],
                        "analysis": doc["analysis"]})
    assert res["ok"], res.get("error")
    return np.asarray(res["x"]), {t["name"]: np.asarray(t["values"])
                                  for t in res["traces"]}


def test_progress_is_coarse_and_leaves_the_result_alone(monkeypatch):
    x_off, off = _run()
    reports: list = []
    monkeypatch.setattr(PROGRESS, "_enabled", True)
    monkeypatch.setattr(PROGRESS, "report_time",
                        lambda t: reports.append(float(t)))
    x_on, on = _run()
    assert len(x_on) > simulate._PROGRESS_POINTS
    assert len(reports) == simulate._PROGRESS_POINTS
    assert reports[-1] == x_on[-1]                 # the bar reaches the end
    np.testing.assert_array_equal(x_on, x_off)
    for name, values in off.items():
        np.testing.assert_array_equal(on[name], values)
