"""Steady-state physics of the coherent transceiver circuit components
(``webapp/catalog.py``: ``iq_modulator`` and ``coherent_rx``), solved by
circulax — the ALE-77 component acceptance path.

These mirror ``test_photonics`` (DC operating points against the analytic model
equations). The end-to-end DSP that consumes these front-ends is pinned
separately, in pure numpy, by ``test_coherent.py``.

Physics pinned here:
  * coherent_rx: balanced I/Q photocurrents track ``R*Re/Im(E_sig E_lo^*)`` —
    the 90-degree hybrid beat between signal and local oscillator;
  * iq_modulator: the null-biased nested MZM maps the I/Q drives onto the
    complex field as ``sin(pi*V/(2*vpi))`` per axis, in quadrature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from circulax import compile_circuit  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "webapp"))
sys.path.insert(0, str(_ROOT))

import catalog  # noqa: E402

MODELS = catalog.build_models()


def _rx(Ps, Pl, ph_s, ph_l, R=0.8):
    """Detected (i_I, i_Q) for two CW tones into the coherent receiver."""
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LS": {"component": "cw_laser",
                   "settings": {"power": Ps, "phase": ph_s,
                                "wavelength_nm": 1550.0}},
            "LL": {"component": "cw_laser",
                   "settings": {"power": Pl, "phase": ph_l,
                                "wavelength_nm": 1550.0}},
            "RX": {"component": "coherent_rx", "settings": {"R": R}},
            "RI": {"component": "resistor", "settings": {"R": 1.0}},
            "RQ": {"component": "resistor", "settings": {"R": 1.0}},
        },
        "connections": {
            "GND,p1": ("LS,p2", "LL,p2", "RX,i_n", "RX,q_n", "RI,p2", "RQ,p2"),
            "LS,p1": "RX,sig", "LL,p1": "RX,lo",
            "RX,i_p": "RI,p1", "RX,q_p": "RQ,p1",
        },
        "ports": {"ip": "RX,i_p", "qp": "RX,q_p"},
    }
    c = compile_circuit(net, MODELS, is_complex=True)
    y = c.dc()
    return complex(c.port(y, "ip")).real, complex(c.port(y, "qp")).real


def _iq_chain(Vi, Vq, vpi=3.0, P=1e-3, Plo=1e-3, R=0.8):
    """Drive the IQ modulator at (Vi, Vq); read the coherent-RX I/Q currents."""
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LS": {"component": "cw_laser",
                   "settings": {"power": P, "phase": 0.0,
                                "wavelength_nm": 1550.0}},
            "LL": {"component": "cw_laser",
                   "settings": {"power": Plo, "phase": 0.0,
                                "wavelength_nm": 1550.0}},
            "MOD": {"component": "iq_modulator",
                    "settings": {"vpi": vpi, "il_db": 0.0}},
            "VI": {"component": "vdc", "settings": {"V": Vi}},
            "VQ": {"component": "vdc", "settings": {"V": Vq}},
            "RX": {"component": "coherent_rx", "settings": {"R": R}},
            "RI": {"component": "resistor", "settings": {"R": 1.0}},
            "RQ": {"component": "resistor", "settings": {"R": 1.0}},
        },
        "connections": {
            "GND,p1": ("LS,p2", "LL,p2", "MOD,vin", "MOD,vqn", "VI,p2",
                       "VQ,p2", "RX,i_n", "RX,q_n", "RI,p2", "RQ,p2"),
            "LS,p1": "MOD,pin", "MOD,pout": "RX,sig", "LL,p1": "RX,lo",
            "VI,p1": "MOD,vip", "VQ,p1": "MOD,vqp",
            "RX,i_p": "RI,p1", "RX,q_p": "RQ,p1",
        },
        "ports": {"ip": "RX,i_p", "qp": "RX,q_p"},
    }
    c = compile_circuit(net, MODELS, is_complex=True)
    y = c.dc()
    return complex(c.port(y, "ip")).real, complex(c.port(y, "qp")).real


# ---------------------------------------------------------------------------
# coherent receiver: 90-degree hybrid beat + balanced detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.0, np.pi / 2, np.pi / 4, -np.pi / 3])
def test_coherent_rx_beat_phase(phi):
    Ps, Pl, R = 2e-3, 8e-3, 0.8
    i_i, i_q = _rx(Ps, Pl, phi, 0.0)
    amp = R * np.sqrt(Ps * Pl)                       # R*|E_sig||E_lo|
    assert i_i == pytest.approx(amp * np.cos(phi), rel=1e-4, abs=1e-9)
    assert i_q == pytest.approx(amp * np.sin(phi), rel=1e-4, abs=1e-9)


def test_coherent_rx_scales_with_lo_power():
    # coherent gain ~ sqrt(P_lo): a 4x LO power doubles the beat amplitude
    i1, _ = _rx(1e-3, 1e-3, 0.0, 0.0)
    i4, _ = _rx(1e-3, 4e-3, 0.0, 0.0)
    assert i4 == pytest.approx(2.0 * i1, rel=1e-4)


# ---------------------------------------------------------------------------
# IQ modulator: nested-MZM complex-plane mapping
# ---------------------------------------------------------------------------

def test_iq_modulator_quadrature_axes():
    sc = 0.8 * np.sqrt(1e-3 * 1e-3) * 0.5            # R*sqrt(P*Plo)*0.5*il
    # full-swing I drive lands purely on the I axis
    i_i, i_q = _iq_chain(3.0, 0.0)
    assert i_i == pytest.approx(sc, rel=1e-3)
    assert abs(i_q) < 1e-6 * sc + 1e-9
    # full-swing Q drive lands purely on the Q axis
    i_i, i_q = _iq_chain(0.0, 3.0)
    assert abs(i_i) < 1e-6 * sc + 1e-9
    assert i_q == pytest.approx(sc, rel=1e-3)


def test_iq_modulator_is_bipolar_and_sinusoidal():
    sc = 0.8 * np.sqrt(1e-3 * 1e-3) * 0.5
    # null bias -> odd (bipolar) drive->field map: V and -V mirror
    i_pos, _ = _iq_chain(1.5, 0.0)
    i_neg, _ = _iq_chain(-1.5, 0.0)
    assert i_pos == pytest.approx(-i_neg, rel=1e-4)
    # sin law: half-swing gives sin(pi/4) of the full-swing amplitude
    assert i_pos == pytest.approx(sc * np.sin(np.pi / 4), rel=1e-3)


def test_iq_modulator_qpsk_corners():
    # the four QPSK drives map to the four quadrants with equal magnitude
    corners = [(3.0, 3.0), (3.0, -3.0), (-3.0, 3.0), (-3.0, -3.0)]
    mags = []
    for vi, vq in corners:
        i_i, i_q = _iq_chain(vi, vq)
        mags.append(np.hypot(i_i, i_q))
        assert np.sign(i_i) == np.sign(vi)
        assert np.sign(i_q) == np.sign(vq)
    assert np.allclose(mags, mags[0], rtol=1e-4)
