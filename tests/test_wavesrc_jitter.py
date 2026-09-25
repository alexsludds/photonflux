"""PRBS-source edge jitter (webapp/wavesrc.py): SJ, DJ and the pj_* aliases."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import wavesrc  # noqa: E402

UI = 100e-12


def _off(**settings):
    return wavesrc._edge_offsets(settings, 4000, UI) / UI   # in UI


def test_no_jitter_is_zero():
    assert not _off().any()


def test_dj_is_dual_dirac():
    """Every edge sits at exactly +-DJ/2, both halves populated."""
    off = _off(dj_ui=0.1)
    assert set(np.round(off, 12)) == {-0.05, 0.05}
    assert 0.4 < np.mean(off > 0) < 0.6
    assert np.ptp(off) == pytest.approx(0.1)          # dj_ui is peak-to-peak


def test_sj_is_sinusoidal_with_peak_amplitude():
    off = _off(sj_ui=0.08, sj_freq=100e6)             # 100 UI per period
    assert np.max(off) == pytest.approx(0.08, rel=1e-3)
    assert np.min(off) == pytest.approx(-0.08, rel=1e-3)


def test_legacy_pj_names_still_drive_sj():
    np.testing.assert_array_equal(_off(pj_ui=0.05, pj_freq=50e6),
                                  _off(sj_ui=0.05, sj_freq=50e6))


def test_dj_and_rj_add():
    both = _off(dj_ui=0.1, rj_ui=0.01)
    assert np.std(both) == pytest.approx(np.hypot(0.05, 0.01), rel=0.05)


def test_pulse_mode_is_jitter_free():
    """Pulse/COM extracts the single-UI response: the (now default) jitter
    must not move its edges."""
    base = {"mode": "pulse", "ui": UI, "tr": 10e-12, "v0": 0.0, "v1": 1.0}
    t0, v0 = wavesrc.prbs_waveform(base, 20 * UI)
    t1, v1 = wavesrc.prbs_waveform({**base, "rj_ui": 0.05, "sj_ui": 0.05,
                                    "dj_ui": 0.05}, 20 * UI)
    np.testing.assert_allclose(t0, t1, atol=1e-15)
    np.testing.assert_array_equal(v0, v1)
