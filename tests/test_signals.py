"""PRBS, PWL emission, sampling, Q/BER, phase recovery."""
import numpy as np
import pytest

import photonflux as ls


def test_prbs7_period_and_balance():
    seq = ls.prbs(7)
    assert len(seq) == 127
    assert seq.sum() in (63, 64)
    # maximal-length: every nonzero 7-bit window appears exactly once
    windows = {tuple(np.roll(seq, -i)[:7]) for i in range(127)}
    assert len(windows) == 127


def test_prbs_orders():
    for order in (5, 7, 9, 11, 15):
        seq = ls.prbs(order)
        assert len(seq) == 2**order - 1
        assert set(np.unique(seq)) == {0, 1}


def test_nrz_pwl_roundtrip():
    bits = np.array([0, 1, 1, 0])
    s = ls.nrz_pwl(bits, 1e-9, 0.1e-9, v0=0.0, v1=1.0)
    assert s.startswith("PWL(") and s.endswith(")")
    pairs = np.fromstring(s[4:-1], sep=" ")  # noqa: NPY201 - flat t,v pairs
    t, v = pairs[0::2], pairs[1::2]
    # sampled at bit centres the PWL reproduces the bits
    centers = (np.arange(4) + 0.5) * 1e-9
    assert np.allclose(np.interp(centers, t, v), bits)


def test_q_ber_clean_signal():
    bits = ls.prbs(7, 64)
    samples = bits.astype(float) + np.random.default_rng(0).normal(0, 1e-3, 64)
    stats = ls.q_ber(samples, bits)
    assert stats.q > 100
    assert stats.ber < 1e-300


def test_best_sampling_recovers_delay():
    rng = np.random.default_rng(1)
    t_bit = 20e-12
    bits = ls.prbs(7, 63)
    t = np.arange(0, 64 * t_bit, 0.5e-12)
    delay = 9e-12
    ideal = np.repeat(bits.astype(float), int(t_bit / 0.5e-12))
    v = np.interp(t - delay, np.arange(len(ideal)) * 0.5e-12, ideal, left=0.0)
    v += rng.normal(0, 0.01, len(v))
    offset, _, stats = ls.best_sampling(t, v, bits, t_bit, skip=2)
    assert offset == pytest.approx(delay, abs=t_bit / 8)
    assert stats.ber < 1e-6


def test_sensitivity_interpolation():
    p = np.array([-10.0, -13.0, -16.0, -19.0])
    ber = np.array([1e-12, 1e-9, 1e-5, 1e-2])
    thr = ls.sensitivity(p, ber, 1e-9)
    assert thr == pytest.approx(-13.0, abs=0.01)
    assert ls.sensitivity(p, ber, 1e-15) is None
