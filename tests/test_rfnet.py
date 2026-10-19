"""photonflux.rfnet: ABCD cascades for terminated RF channels."""
import numpy as np
import pytest

from photonflux import rfnet

TS = 1 / 53.125e9 / 32
N = 1 << 14
F = rfnet.grid(TS, N)
K = int(np.argmin(np.abs(F - 26.5625e9)))


def test_lossless_matched_line_is_a_pure_delay():
    ln = rfnet.line_rlgc(F, z0=50.0, er=4.0)
    s = rfnet.Net.line(ln, 0.1).s(50.0)
    assert np.allclose(np.abs(s[1, 0]), 1.0, atol=1e-9)
    assert np.abs(s[0, 0]).max() < 1e-9
    assert ln.delay_s_m(10e9) == pytest.approx(2.0 / rfnet.C0, rel=1e-6)


def test_lossy_line_is_reciprocal_passive_and_calibrates():
    ln = rfnet.line_rlgc(F, z0=50.0, er=3.6, tand=0.004, cond_db_m_rtghz=7.0)
    length = rfnet.length_for_loss(ln, 10.0, F[K])
    s = rfnet.Net.line(ln, length).s(50.0)
    assert -20 * np.log10(abs(s[1, 0, K])) == pytest.approx(10.0, abs=1e-6)
    assert np.allclose(s[1, 0], s[0, 1])
    power = np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2
    assert power.max() <= 1.0 + 1e-9
    # the skin effect lifts Zc at low frequency; it tends to z0 at high
    assert abs(ln.zc[K]) == pytest.approx(50.0, rel=0.01)


def test_transfer_is_half_s21_when_matched_and_counts_reflections():
    ln = rfnet.line_rlgc(F, z0=50.0, er=3.6, tand=0.004, cond_db_m_rtghz=7.0)
    net = rfnet.Net.shunt_c(F, 100e-15) @ rfnet.Net.line(ln, 0.05)
    assert np.allclose(net.transfer(50.0, 50.0), net.s(50.0)[1, 0] / 2)
    # a mismatched source: the load voltage is the forward wave times
    # (1 + reflections); compare with an explicit two-port solve
    zs, zl = 30.0, 75.0
    (a, b), (c, d) = net.abcd
    v2 = zl / (a * zl + b + zs * (c * zl + d))
    assert np.allclose(net.transfer(zs, zl), v2)


def test_flipped_and_zin():
    ln = rfnet.line_rlgc(F, z0=40.0, er=3.0, cond_db_m_rtghz=5.0)
    net = rfnet.Net.series_l(F, 50e-12) @ rfnet.Net.line(ln, 0.01) @ rfnet.Net.shunt_c(F, 80e-15)
    s, sf = net.s(50.0), net.flipped().s(50.0)
    assert np.allclose(s[0, 0], sf[1, 1]) and np.allclose(s[1, 0], sf[0, 1])
    # a line terminated in its own Zc presents Zc
    line = rfnet.Net.line(ln, 0.02)
    assert np.allclose(line.zin(ln.zc), ln.zc)


def test_impulse_is_causal_and_keeps_dc():
    ln = rfnet.line_rlgc(F, z0=50.0, er=3.6, tand=0.004, cond_db_m_rtghz=7.0)
    net = rfnet.Net.line(ln, 0.1)
    h = rfnet.impulse(F, net.transfer(50.0, 50.0), TS, N)
    delay = int(0.1 * np.sqrt(3.6) / rfnet.C0 / TS)
    assert h.sum() * TS == pytest.approx(0.5, rel=1e-3)
    assert np.abs(h[: delay - 20]).max() < 1e-3 * np.abs(h).max()
