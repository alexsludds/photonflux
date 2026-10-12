"""The cap on live compiled circuit structures (simulate._note_structure).

JAX keeps every compiled program in memory; an optimizer that keeps changing
circuit structure would pile them up until the process runs out of memory
mappings. Past PHOTONFLUX_LIVE_STRUCTURES structures the caches are released,
but never while parallel_map solves are in flight.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import simulate  # noqa: E402


@pytest.fixture
def cap(monkeypatch):
    released = []
    monkeypatch.setattr(simulate, "_MAX_LIVE_STRUCTURES", 2)
    monkeypatch.setattr(simulate, "_LIVE_STRUCTURES", set())
    monkeypatch.setattr(simulate, "_PARALLEL_ACTIVE", 0)
    import jax
    monkeypatch.setattr(jax, "clear_caches", lambda: released.append(1))
    return released


def test_release_only_when_a_new_structure_exceeds_the_cap(cap):
    log: list = []
    for s in ("a", "b", "a", "b"):
        simulate._note_structure(s, log)
    assert not cap and not log                   # revisits are free
    simulate._note_structure("c", log)
    assert len(cap) == 1 and "released" in log[0]
    assert simulate._LIVE_STRUCTURES == {"c"}


def test_no_release_while_parallel_solves_run(cap, monkeypatch):
    monkeypatch.setattr(simulate, "_PARALLEL_ACTIVE", 1)
    for s in ("a", "b", "c", "d"):
        simulate._note_structure(s, [])
    assert not cap                               # over the cap, held back
    monkeypatch.setattr(simulate, "_PARALLEL_ACTIVE", 0)
    simulate._note_structure("e", [])
    assert len(cap) == 1 and simulate._LIVE_STRUCTURES == {"e"}
