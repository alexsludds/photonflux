"""PRBS-source sequences (webapp/wavesrc.py): prbs, prbs+oma, square,
staircase."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import wavesrc  # noqa: E402


def _sym(mode, sequence, n=2000, **kw):
    f = wavesrc._symbols({"mode": mode, "order": 13, "sequence": sequence,
                          **kw}, n)
    return np.rint(f * (3 if mode == "pam4" else 1)).astype(int)


def _longest_run(s, level):
    best = cur = 0
    for v in s:
        cur = cur + 1 if v == level else 0
        best = max(best, cur)
    return best


def test_prbs_is_the_default():
    assert np.array_equal(_sym("pam4", "prbs"),
                          np.rint(wavesrc._symbols(
                              {"mode": "pam4", "order": 13}, 2000) * 3))


@pytest.mark.parametrize("mode,top,n_top,n_bot", [("pam4", 3, 7, 6),
                                                  ("nrz", 1, 8, 8)])
def test_oma_runs_every_512_symbols(mode, top, n_top, n_bot):
    s = _sym(mode, "prbs+oma", run_ui=1)
    for s0 in range(wavesrc.OMA_RUNS_START, s.size - 64,
                    wavesrc.OMA_RUNS_EVERY):
        assert (s[s0:s0 + n_top] == top).all()
        assert (s[s0 + n_top:s0 + n_top + n_bot] == 0).all()
    plain = _sym(mode, "prbs")
    keep = np.ones(s.size, bool)
    for s0 in range(wavesrc.OMA_RUNS_START, s.size, wavesrc.OMA_RUNS_EVERY):
        keep[s0:s0 + n_top + n_bot] = False
    assert np.array_equal(s[keep], plain[keep])      # the PRBS elsewhere
    if mode == "pam4":                               # why the runs exist
        assert _longest_run(plain[:600], 0) < 6


def test_square_and_staircase():
    assert list(_sym("pam4", "square", 32, run_ui=4)[:12]) == \
        [3] * 4 + [0] * 4 + [3] * 4
    assert list(_sym("pam4", "staircase", 12, run_ui=2)) == \
        [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1]
    assert list(_sym("nrz", "staircase", 8, run_ui=2)) == [1, 1, 0, 0] * 2


def test_unknown_sequence_is_rejected():
    with pytest.raises(ValueError, match="sequence"):
        _sym("nrz", "ssprq")
