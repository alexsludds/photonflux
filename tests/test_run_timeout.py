"""The server's per-run time limit (PHOTONFLUX_RUN_TIMEOUT_S).

A run that times out is asked to stop the way the Stop button does, so it
releases the run lock instead of solving on for nobody; a request that times
out while queued never runs, and never cancels the run it was waiting on.
The engine is stubbed (no JAX): the fake run polls the cancel flag the way the
transient solver's save-point callback and the flow runner do.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp"))

import progress  # noqa: E402
import server  # noqa: E402  (webapp/server.py — engine stays unloaded)


def _stub_engine(monkeypatch, started: list) -> None:
    def run(payload):
        progress.PROGRESS.enter()
        try:
            started.append(payload)
            deadline = time.monotonic() + 10
            while not progress.PROGRESS.cancelled:
                if time.monotonic() > deadline:
                    return {"ok": True, "uncancelled": True}
                time.sleep(0.005)
            return {"ok": False, "cancelled": True}
        finally:
            progress.PROGRESS.leave()

    monkeypatch.setattr(server, "simulate", SimpleNamespace(run=run))
    monkeypatch.setattr(server, "_RUN_TIMEOUT_S", 0.2)


def test_timed_out_run_is_cancelled_and_frees_the_lock(monkeypatch):
    started: list = []
    _stub_engine(monkeypatch, started)
    res = server.Handler._run({"n": 1})
    assert res.get("_timeout") and started == [{"n": 1}]
    # the worker stopped at its next poll and let go of the run lock
    assert server._RUN_LOCK.acquire(timeout=2)
    server._RUN_LOCK.release()


def test_queued_request_that_times_out_never_runs(monkeypatch):
    started: list = []
    _stub_engine(monkeypatch, started)
    with server._RUN_LOCK:               # another visitor's run is in flight
        progress.PROGRESS.enter()
        try:
            res = server.Handler._run({"n": 2})
            assert res.get("_timeout")
            assert not progress.PROGRESS.cancelled   # theirs keeps running
        finally:
            progress.PROGRESS.leave()
    time.sleep(0.3)                      # the queued worker gets the lock ...
    assert started == []                 # ... and drops the abandoned run
