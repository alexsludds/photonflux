"""models/optical_field/phase_shifter.va — ideal EO phase shifter physics pins.

The element rotates the coherent field by phi = pi*(V + vbias)/vpi and (by
default) leaves its magnitude untouched:

    E_out = s * e^{j*phi} * E_in ,   s = 10^(-il_db/20)

A unit real field driven into (in_re, in_im) = (1, 0) therefore reads back as
(out_re, out_im) = s*(cos phi, sin phi). These tests pin that transfer,
lowered to JAX by cx.va and solved by circulax.
"""
from __future__ import annotations

import numpy as np
import pytest

from circuit_helpers import op
from photonflux import cx


def _field(v: float, **params) -> complex:
    """DC-solve a unit real field through the phase shifter -> E_out."""
    vals = op(cx.va("phase_shifter"),
              {"in_re": 1.0, "in_im": 0.0, "vp": v, "vn": 0.0},
              settings=params,
              reads=["out_re", "out_im"])
    return complex(vals["out_re"].real, vals["out_im"].real)


def test_phase_vs_voltage():
    # phi = pi*V/vpi: half-wave (field flips sign) at V = vpi, full 2*pi at 2*vpi
    for v, phi in [(0.0, 0.0), (1.5, np.pi / 2), (3.0, np.pi), (6.0, 2 * np.pi)]:
        e = _field(v, vpi=3.0)
        assert e.real == pytest.approx(np.cos(phi), abs=1e-9)
        assert e.imag == pytest.approx(np.sin(phi), abs=1e-9)


def test_ideal_is_lossless():
    # il_db = 0: magnitude is preserved at every bias (pure rotation)
    for v in np.linspace(-6.0, 6.0, 13):
        assert abs(_field(float(v), vpi=3.0)) == pytest.approx(1.0, abs=1e-9)


def test_bias_offset():
    # vbias shifts the operating point: V = 0, vbias = vpi -> phi = pi
    e = _field(0.0, vpi=3.0, vbias=3.0)
    assert e.real == pytest.approx(-1.0, abs=1e-9)
    assert e.imag == pytest.approx(0.0, abs=1e-9)


def test_insertion_loss():
    # il_db attenuates the field by 10^(-il_db/20) with phase unchanged at V=0
    e = _field(0.0, vpi=3.0, il_db=6.0)
    assert abs(e) == pytest.approx(10 ** (-6.0 / 20.0), rel=1e-6)
    assert np.angle(e) == pytest.approx(0.0, abs=1e-9)
