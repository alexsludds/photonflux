"""Bus gap -> power coupling map (photonflux.coupler)."""
import math

import numpy as np
import pytest

from photonflux.coupler import DEFAULT_O_BAND, GapCoupling


def test_reference_gap_reproduces_the_example_device():
    """The default calibration is anchored to examples/ring_mod_sky130.py."""
    assert DEFAULT_O_BAND.kappa2(200.0) == pytest.approx(0.10, rel=1e-12)


def test_kappa2_falls_monotonically_with_gap():
    gaps = np.linspace(120.0, 400.0, 40)
    k2 = DEFAULT_O_BAND.kappa2(gaps)
    assert np.all(np.diff(k2) < 0)
    assert np.all((k2 > 0) & (k2 < 1))


def test_one_decay_length_halves_the_field():
    c = DEFAULT_O_BAND
    ratio = c.kappa(c.g0_nm + c.gd_nm) / c.kappa(c.g0_nm)
    assert ratio == pytest.approx(math.exp(-1.0), rel=1e-12)
    # power coupling therefore falls by exp(-2)
    assert c.kappa2(c.g0_nm + c.gd_nm) / c.kappa2(c.g0_nm) == pytest.approx(
        math.exp(-2.0), rel=1e-12)


def test_gap_for_kappa2_inverts_kappa2():
    for k2 in (0.02, 0.076, 0.10, 0.25, 0.5):
        g = DEFAULT_O_BAND.gap_for_kappa2(k2)
        assert DEFAULT_O_BAND.kappa2(g) == pytest.approx(k2, rel=1e-9)


def test_kappa2_is_clipped_inside_the_va_parameter_range():
    """ring_mod.va declares kappa2 on the open range (0:1)."""
    assert 0.0 < DEFAULT_O_BAND.kappa2(1.0) < 1.0        # absurdly tight gap
    assert 0.0 < DEFAULT_O_BAND.kappa2(5000.0) < 1.0     # absurdly wide gap


def test_from_points_recovers_a_known_calibration():
    truth = GapCoupling(kappa0=0.32, g0_nm=180.0, gd_nm=95.0)
    gaps = np.array([150.0, 200.0, 250.0, 300.0, 350.0])
    fit = GapCoupling.from_points(gaps, truth.kappa2(gaps), g0_nm=180.0)
    assert fit.gd_nm == pytest.approx(truth.gd_nm, rel=1e-6)
    assert fit.kappa0 == pytest.approx(truth.kappa0, rel=1e-6)


def test_rejects_bad_calibrations():
    with pytest.raises(ValueError):
        GapCoupling(kappa0=1.5, g0_nm=200.0, gd_nm=110.0)
    with pytest.raises(ValueError):
        GapCoupling(kappa0=0.3, g0_nm=200.0, gd_nm=-1.0)
    with pytest.raises(ValueError):
        GapCoupling.from_points([200.0], [0.1])
    with pytest.raises(ValueError):  # coupling rising with gap is backwards
        GapCoupling.from_points([200.0, 300.0], [0.05, 0.20])
