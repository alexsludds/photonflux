"""PRBS pattern generation (engine-agnostic stimulus)."""
import numpy as np
import pytest

from photonflux import prbs


def test_prbs7_period_and_balance():
    seq = prbs(7)
    assert len(seq) == 127
    assert seq.sum() in (63, 64)
    # maximal-length: every nonzero 7-bit window appears exactly once
    windows = {tuple(np.roll(seq, -i)[:7]) for i in range(127)}
    assert len(windows) == 127


def test_prbs_orders():
    for order in (5, 7, 9, 11, 13, 15):
        seq = prbs(order)
        assert len(seq) == 2**order - 1
        assert set(np.unique(seq)) == {0, 1}


def test_prbs13_is_maximal_length():
    """PRBS-13 needs the four-term IEEE 802.3 polynomial (degree 13 has no
    primitive trinomial), so it exercises the multi-tap feedback path."""
    seq = prbs(13)
    assert len(seq) == 8191
    assert seq.sum() == 4096  # 2^(n-1) ones over one period
    # maximal-length <=> every nonzero 13-bit window appears exactly once
    windows = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([seq, seq[:12]]), 13
    )
    codes = {int("".join(map(str, w)), 2) for w in windows}
    assert len(codes) == 8191
    assert 0 not in codes


def test_prbs13_run_lengths_for_tdec():
    """stateye's _8180 OMA/TDEC filters need runs of >=8 ones and >=8 zeros.

    They are found independently, not adjacently: a full PRBS-13 period has 16
    of each, but zero occurrences of the contiguous 0^8->1^8 window that the
    _8180 *edge-time* filters demand (those fall back to _4140).
    """
    seq = prbs(13)
    doubled = "".join(map(str, np.concatenate([seq, seq])))
    assert max(len(r) for r in doubled.split("0") if r) == 13
    assert max(len(r) for r in doubled.split("1") if r) == 12
    assert "0" * 8 + "1" * 8 not in doubled

    # a truncated PRBS-13 is not a substitute: the first 511 bits have no
    # run of 8 ones at all, so oma_8180 would come back NaN
    head = "".join(map(str, seq[:511]))
    assert max(len(r) for r in head.split("0") if r) < 8


def test_pam4_gray_mapping():
    from photonflux.signals import pam4_gray
    # 00->0, 01->1, 11->2, 10->3: adjacent levels differ by one bit
    assert pam4_gray([0, 0, 0, 1, 1, 1, 1, 0]).tolist() == [0, 1, 2, 3]
    sym = pam4_gray(np.tile(prbs(13), 2))    # PRBS-13Q
    assert sym.size == 8191
    assert np.bincount(sym).tolist() == [2047, 2048, 2048, 2048]
    with pytest.raises(ValueError, match="even"):
        pam4_gray([1, 0, 1])
