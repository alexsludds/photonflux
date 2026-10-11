"""Derived-trace expressions (``webapp/exprs.py``) stay cheap to evaluate.

They arrive from any visitor of a public server and run under the single
simulation lock, so a huge integer power or a string repetition must fail
fast instead of pinning the process.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

import exprs  # noqa: E402


def _apply(text: str) -> tuple[dict, list]:
    res = {"x": [0.0, 1.0, 2.0],
           "traces": [{"name": "v", "values": [1.0, 2.0, 3.0]}]}
    log: list = []
    exprs.apply(res, text, log)
    return res, log


@pytest.mark.parametrize("text", ["big = 9**9**9**9", "s = 'a' * 10**10"])
def test_hostile_expressions_fail_fast(text):
    t0 = time.perf_counter()
    res, log = _apply(text)
    assert time.perf_counter() - t0 < 1.0
    assert "scalars" not in res and len(res["traces"]) == 1
    assert log and log[0].startswith("expressions:")


def test_ordinary_expressions_still_work():
    res, _ = _apply("pk = pk2pk(v)\ndbl = 2 * v\np = 2**10")
    assert {s["name"]: s["value"] for s in res["scalars"]} == {
        "pk": 2.0, "p": 1024.0}
    assert res["traces"][-1]["values"] == [2.0, 4.0, 6.0]
