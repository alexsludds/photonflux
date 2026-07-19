"""Steady-state physics of the in-transient ``fiber_nl`` split-step component
(``webapp/catalog.py::_fiber_nl_field``, built by ``webapp/lti.py``).

CW steady state exercises the Kerr path and the linear (gamma = 0) reduction to
``fiber_cd`` without a full transient — dispersion is trivial for a constant
envelope, so these stay fast. The transient four-wave-mixing / soliton behaviour
is validated in ``examples/fiber_nl_ssfm.py`` (per the repo's DC-tests /
transient-examples split) against the ``webapp/ssfm.py`` reference engine, whose
own analytics are pinned in ``tests/test_ssfm.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from circulax import compile_circuit  # noqa: E402
from circulax.components.base_component import (  # noqa: E402
    Signals, States, component, source)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import catalog  # noqa: E402
import lti  # noqa: E402

BASE = dict(length_km=5.0, D_ps=0.0, lambda_nm=1550.0, atten_db_km=0.0,
            gamma_per_W_km=1.3, n_seg=20, fit_bw=60e9, n_poles=12)


def _cw(power: float):
    @source(ports=("p1", "p2"), states=("i_src",))
    def CW(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
        field = jnp.sqrt(power)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - field}, {}
    return CW


def _term():
    @component(ports=("p1",))
    def Term(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"p1": 0.0}, {}      # zero-current open (output is driven)
    return Term


def _cw_out(settings: dict, power: float) -> complex:
    """Compiled CW steady-state output field of one fiber_nl instance."""
    _, payload = lti.build("fiber_nl", settings, [])
    model = catalog._fiber_nl_field(*payload["fiber_nl"])
    net = {
        "instances": {"GND": {"component": "ground"},
                      "SRC": {"component": "src"},
                      "F": {"component": "fib"},
                      "T": {"component": "term"}},
        "connections": {"SRC,p1": "F,p1", "GND,p1": "SRC,p2", "F,p2": "T,p1"},
        "ports": {"out": "F,p2"},
    }
    models = {"ground": lambda: 0, "src": _cw(power), "fib": model,
              "term": _term()}
    c = compile_circuit(net, models, backend="dense", is_complex=True,
                        max_steps=400)
    return complex(c.port(c.dc(), "out"))


def test_fiber_nl_linear_limit_is_lossless_passthrough():
    # gamma = 0, lossless, CW: dispersion all-pass has unit DC gain -> out == in
    out = _cw_out({**BASE, "gamma_per_W_km": 0.0}, 0.1)
    assert abs(out) ** 2 == pytest.approx(0.1, rel=1e-6)
    assert np.angle(out) == pytest.approx(0.0, abs=1e-6)


def test_fiber_nl_linear_attenuation_matches_fiber_cd():
    # gamma = 0 with loss: CW power is the fiber_cd attenuation amp^2 * P
    out = _cw_out({**BASE, "gamma_per_W_km": 0.0, "atten_db_km": 0.2}, 0.1)
    amp2 = 10.0 ** (-0.2 * 5.0 / 10.0)
    assert abs(out) ** 2 == pytest.approx(amp2 * 0.1, rel=1e-4)


def test_fiber_nl_cw_kerr_phase():
    # lossless CW Kerr phase is exactly -gamma * P * L
    p, gamma, L = 0.1, 1.3 / 1e3, 5000.0
    out = _cw_out(BASE, p)
    assert abs(out) ** 2 == pytest.approx(p, rel=1e-6)            # phase only
    assert np.angle(out) == pytest.approx(-gamma * p * L, rel=1e-4)


def test_fiber_nl_cw_kerr_scales_with_power():
    gamma, L = 1.3 / 1e3, 5000.0
    for p in (0.05, 0.1, 0.2):
        assert np.angle(_cw_out(BASE, p)) == pytest.approx(
            -gamma * p * L, rel=1e-4)


def test_fiber_nl_lossy_kerr_uses_effective_length():
    # with loss the accumulated Kerr phase -> -gamma * P * Leff as n_seg grows
    p, gamma, L = 0.1, 1.3 / 1e3, 5000.0
    alpha = 0.2 * np.log(10) / 10 / 1e3
    leff = (1.0 - np.exp(-alpha * L)) / alpha
    phi = np.angle(_cw_out({**BASE, "atten_db_km": 0.2, "n_seg": 60}, p))
    assert phi == pytest.approx(-gamma * p * leff, rel=0.02)
