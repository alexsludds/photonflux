"""Steady-state physics of the EDFA (``models/optical_field/edfa.va``) — the
pumped, ms-timescale, wavelength-shaped optical amplifier — lowered to JAX by
``cx.va`` and solved by circulax.

Each test compares the compiled model against an independent numpy solution of
the same equations: the pump-driven Agrawal-Olsson gain reservoir (closed form
for the unsaturated gain, a brentq root-find for the saturated gain) and the
detuned one-pole spectral filter (closed-form Lorentzian). The gain-transient
study — dropping 7 of 8 WDM channels and watching the survivor surge over
``tau_c`` — is transient and lives in ``examples/edfa_wdm.py``, which carries
its own analytic asserts.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from circuit_helpers import op, power
from photonflux import cx

C0 = 2.99792458e8

# device defaults (mirror the catalog entry). lambda_ref == lambda_peak by
# default so the spectral filter is at its peak (unity shape) unless a test
# detunes it, isolating the gain-reservoir physics from the spectral shape.
EDFA = dict(g0_db=30.0, p_pump_mw=100.0, p_pump_op_mw=100.0, p_pump_tr_mw=8.0,
            p_sat=5e-3, tau_c=10e-3, lambda_ref_nm=1550.0,
            lambda_peak_nm=1550.0, gain_bw_nm=30.0, alpha_h=0.0, p_ase=0.0)


# ---------------------------------------------------------------------------
# analytic references (the same equations the VA model integrates)
# ---------------------------------------------------------------------------
def edfa_h0(p_pump_mw: float) -> float:
    """Unsaturated log gain: linear in pump power, transparent at p_pump_tr."""
    hop = EDFA["g0_db"] * np.log(10) / 10
    return hop * (p_pump_mw - EDFA["p_pump_tr_mw"]) / (
        EDFA["p_pump_op_mw"] - EDFA["p_pump_tr_mw"])


def edfa_gain(p_in: float, p_pump_mw: float, *, p_sat: float | None = None) -> float:
    """Saturated peak gain G = e^h from h0 - h - (e^h - 1)*P_in/p_sat = 0."""
    p_sat = EDFA["p_sat"] if p_sat is None else p_sat
    h0 = edfa_h0(p_pump_mw)
    if h0 <= 0.0:                      # below transparency: root bracket flips
        f = lambda h: h0 - h - (np.exp(h) - 1) * p_in / p_sat
        return np.exp(brentq(f, h0 - 5.0, 0.0))
    f = lambda h: h0 - h - (np.exp(h) - 1) * p_in / p_sat
    return np.exp(brentq(f, -30.0, h0 + 1e-9))


def edfa_tau_bw(peak_nm: float, gain_bw_nm: float) -> float:
    df_fwhm = C0 / (peak_nm * 1e-9) ** 2 * (gain_bw_nm * 1e-9)
    return 1.0 / (np.pi * df_fwhm)


def edfa_spectral_gain(lam_nm: float, peak_nm: float, gain_bw_nm: float,
                       g_peak: float) -> float:
    """Lorentzian gain seen by a carrier at lam_nm (peak gain g_peak).

    A carrier at lam and the gain peak at peak_nm are separated in the shared
    baseband frame by dw = 2*pi*c*(1/peak - 1/lam) regardless of the reference
    frame, so gain = g_peak / (1 + (tau_bw*dw)^2)."""
    tau_bw = edfa_tau_bw(peak_nm, gain_bw_nm)
    dw = 2 * np.pi * C0 * (1.0 / (peak_nm * 1e-9) - 1.0 / (lam_nm * 1e-9))
    return g_peak / (1.0 + (tau_bw * dw) ** 2)


# ---------------------------------------------------------------------------
# one-device netlist: forward power gain through the EDFA
# ---------------------------------------------------------------------------
def _edfa_fwd(p_in: float, **params) -> float:
    vals = op(cx.va("edfa"),
              {"fi_re": np.sqrt(p_in), "fi_im": 0.0},
              settings={**EDFA, **params},
              reads=["fo_re", "fo_im"], is_complex=True)
    return power(vals, "fo_re", "fo_im")


# ---------------------------------------------------------------------------
# gain reservoir (with lambda_ref == lambda_peak so the spectral shape is unity)
# ---------------------------------------------------------------------------
def test_edfa_small_signal_peak_gain():
    # 0.1 nW probe at the operating pump: negligible saturation, full 30 dB
    assert 10 * np.log10(_edfa_fwd(1e-10) / 1e-10) == pytest.approx(30.0, abs=1e-2)


def test_edfa_transparency_and_absorption():
    # pump == p_pump_tr: transparent (unity gain); below: the fibre absorbs
    assert _edfa_fwd(1e-9, p_pump_mw=EDFA["p_pump_tr_mw"]) / 1e-9 == \
        pytest.approx(1.0, rel=1e-3)
    # under-pumped (2 mW < 8 mW transparency): net absorption, matching the
    # unsaturated log gain e^{h0} < 1
    g_absorb = _edfa_fwd(1e-9, p_pump_mw=2.0) / 1e-9
    assert g_absorb < 1.0
    assert g_absorb == pytest.approx(np.exp(edfa_h0(2.0)), rel=2e-3)


def test_edfa_pump_dependence():
    # unsaturated log gain is linear in pump power (h0 curve)
    for p_pump in (20.0, 50.0, 100.0, 150.0):
        g = _edfa_fwd(1e-10, p_pump_mw=p_pump) / 1e-10
        assert np.log(g) == pytest.approx(edfa_h0(p_pump), rel=2e-3, abs=2e-3)
    # more pump -> more gain
    lo = _edfa_fwd(1e-6, p_pump_mw=50.0)
    hi = _edfa_fwd(1e-6, p_pump_mw=150.0)
    assert hi > lo


def test_edfa_gain_saturation():
    # gain compresses along the Agrawal-Olsson solution as the input grows
    for p_in in (1e-4, 1e-3, 5e-3, 1e-2):
        assert _edfa_fwd(p_in) / p_in == pytest.approx(
            edfa_gain(p_in, EDFA["p_pump_mw"]), rel=1e-3)
    # a smaller p_sat compresses harder at the same input
    g_small = _edfa_fwd(1e-3, p_sat=1e-3) / 1e-3
    g_large = _edfa_fwd(1e-3, p_sat=20e-3) / 1e-3
    assert g_small < g_large


def test_edfa_output_saturation_clamp():
    # deep saturation: the OUTPUT power tends to a soft ceiling set by p_sat
    # (P_out = G*P_in with G -> 1 + h0*p_sat/P_in for P_in >> p_sat), so
    # doubling a large input barely changes the output.
    p_a, p_b = 50e-3, 100e-3
    out_a, out_b = _edfa_fwd(p_a), _edfa_fwd(p_b)
    # 2x (3.01 dB) more input, but the amplifier is clamped: clearly less than
    # 3 dB more output (the gain has compressed)
    assert 10 * np.log10(out_b / out_a) < 2.7
    # and both match the analytic saturated gain
    assert out_a / p_a == pytest.approx(edfa_gain(p_a, EDFA["p_pump_mw"]), rel=1e-3)
    assert out_b / p_b == pytest.approx(edfa_gain(p_b, EDFA["p_pump_mw"]), rel=1e-3)


# ---------------------------------------------------------------------------
# wavelength-dependent gain across a WDM grid
# ---------------------------------------------------------------------------
def test_edfa_spectral_gain_peak():
    # gain is maximal at lambda_peak and rolls off symmetrically in frequency
    g_peak = _edfa_fwd(1e-10, lambda_ref_nm=1532.0, lambda_peak_nm=1532.0) / 1e-10
    g_lo = _edfa_fwd(1e-10, lambda_ref_nm=1520.0, lambda_peak_nm=1532.0) / 1e-10
    g_hi = _edfa_fwd(1e-10, lambda_ref_nm=1544.0, lambda_peak_nm=1532.0) / 1e-10
    assert g_peak > g_lo and g_peak > g_hi


def test_edfa_spectral_gain_wdm_grid():
    # An 8-channel C-band grid (1530..1565 nm) into an EDFA peaked at 1532 nm.
    # Each channel's gain is a per-point DC solve with the carrier at baseband
    # zero (lambda_ref == channel); the reservoir sees only that channel's
    # (tiny) power, so this is the small-signal shared-frame per-channel gain.
    peak_nm, gain_bw_nm = 1532.0, 25.0
    channels = np.linspace(1530.0, 1565.0, 8)
    p_probe = 1e-10
    # peak gain reference (measure at the peak so tiny compression cancels)
    g_peak = _edfa_fwd(p_probe, lambda_ref_nm=peak_nm, lambda_peak_nm=peak_nm,
                       gain_bw_nm=gain_bw_nm) / p_probe
    for lam in channels:
        g = _edfa_fwd(p_probe, lambda_ref_nm=lam, lambda_peak_nm=peak_nm,
                      gain_bw_nm=gain_bw_nm) / p_probe
        g_ref = edfa_spectral_gain(lam, peak_nm, gain_bw_nm, g_peak)
        assert 10 * np.log10(g) == pytest.approx(10 * np.log10(g_ref), abs=0.05)
    # meaningful tilt across the band (outer channel rolled off vs the peak)
    g_edge = _edfa_fwd(p_probe, lambda_ref_nm=1565.0, lambda_peak_nm=peak_nm,
                       gain_bw_nm=gain_bw_nm) / p_probe
    assert 10 * np.log10(g_peak / g_edge) > 2.0


def test_edfa_chirp():
    # linewidth-enhancement factor: gain saturation chirps the field by
    # phi = -0.5*alpha_h*h (h = ln G), without changing the gain magnitude.
    p_in, alpha = 1e-3, 1.0     # keep |phi| < pi so atan2 doesn't wrap
    s = {**EDFA, "alpha_h": alpha, "lambda_ref_nm": EDFA["lambda_peak_nm"]}
    vals = op(cx.va("edfa"), {"fi_re": np.sqrt(p_in), "fi_im": 0.0},
              settings=s, reads=["fo_re", "fo_im"], is_complex=True)
    g = power(vals, "fo_re", "fo_im") / p_in
    # magnitude unchanged by alpha_h (matches the alpha_h = 0 saturated gain)
    assert g == pytest.approx(edfa_gain(p_in, EDFA["p_pump_mw"]), rel=1e-3)
    phase = np.arctan2(vals["fo_im"].real, vals["fo_re"].real)
    assert phase == pytest.approx(-0.5 * alpha * np.log(g), rel=1e-3)


def test_edfa_gain_bandwidth_widens_the_top():
    # a wider gain_bw_nm flattens the gain across the band (less tilt)
    peak, lam = 1532.0, 1560.0
    p = 1e-10
    g_pk_narrow = _edfa_fwd(p, lambda_ref_nm=peak, lambda_peak_nm=peak,
                            gain_bw_nm=15.0) / p
    g_ed_narrow = _edfa_fwd(p, lambda_ref_nm=lam, lambda_peak_nm=peak,
                            gain_bw_nm=15.0) / p
    g_pk_wide = _edfa_fwd(p, lambda_ref_nm=peak, lambda_peak_nm=peak,
                          gain_bw_nm=60.0) / p
    g_ed_wide = _edfa_fwd(p, lambda_ref_nm=lam, lambda_peak_nm=peak,
                          gain_bw_nm=60.0) / p
    tilt_narrow = 10 * np.log10(g_pk_narrow / g_ed_narrow)
    tilt_wide = 10 * np.log10(g_pk_wide / g_ed_wide)
    assert tilt_wide < tilt_narrow
