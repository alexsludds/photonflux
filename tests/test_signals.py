"""PRBS pattern generation (engine-agnostic stimulus)."""
import numpy as np

from photonflux import prbs


def test_prbs7_period_and_balance():
    seq = prbs(7)
    assert len(seq) == 127
    assert seq.sum() in (63, 64)
    # maximal-length: every nonzero 7-bit window appears exactly once
    windows = {tuple(np.roll(seq, -i)[:7]) for i in range(127)}
    assert len(windows) == 127


def test_prbs_orders():
    for order in (5, 7, 9, 11, 15):
        seq = prbs(order)
        assert len(seq) == 2**order - 1
        assert set(np.unique(seq)) == {0, 1}
