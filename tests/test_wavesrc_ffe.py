"""PRBS-source TX FFE (webapp/wavesrc.py): linear taps relative to the main."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import wavesrc  # noqa: E402


def _ffe(levels, **settings):
    return wavesrc._tx_ffe(settings, np.asarray(levels, float))


def test_taps_are_linear_ratios_to_the_main_cursor():
    """A lone high symbol in a run of lows comes out as the tap vector
    [pre, main, post] in time order, peak-normalized by sum |taps|."""
    lv = np.array([-1.0, -1, -1, 1, -1, -1, -1])
    pre, post = -0.1, -0.25
    y = _ffe(lv, ffe_pre=pre, ffe_post=post)
    taps = np.array([pre, 1.0, post]) / (1 + abs(pre) + abs(post))
    # y = conv(x, taps) with x = lv; the isolated +2 step above the -1 run
    # shows each tap as an excursion of 2*tap from the run's level
    run = -taps.sum()
    np.testing.assert_allclose(y[[2, 3, 4]] - run, 2 * taps, atol=1e-12)


def test_zero_taps_pass_through():
    lv = np.array([0.0, 1, 1, 0, 1])
    np.testing.assert_array_equal(_ffe(lv, ffe_pre=0.0, ffe_post=0.0), lv)


def test_peak_swing_is_kept():
    rng = np.random.default_rng(0)
    lv = rng.choice([0.0, 1 / 3, 2 / 3, 1.0], 2000)
    y = _ffe(lv, ffe_pre=-0.15, ffe_post=-0.3)
    assert y.min() >= -1e-12 and y.max() <= 1 + 1e-12


@pytest.mark.parametrize("db", [3.0, 5.0, 12.0])
def test_legacy_db_knob_maps_onto_the_linear_tap(db):
    lv = np.random.default_rng(1).choice([0.0, 1.0], 500)
    tap = -(10.0 ** (-db / 20.0))
    np.testing.assert_allclose(_ffe(lv, ffe_post_db=db),
                               _ffe(lv, ffe_post=tap), atol=1e-12)
    # an explicit linear tap wins over a stale dB value
    np.testing.assert_allclose(_ffe(lv, ffe_post=-0.1, ffe_post_db=db),
                               _ffe(lv, ffe_post=-0.1), atol=1e-12)
