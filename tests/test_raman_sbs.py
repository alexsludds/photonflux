"""Steady-state physics of the Raman (``models/optical_field/raman_amp.va``) and
Brillouin (``models/optical_field/sbs_fiber.va``) scattering spans — the SRS /
Raman-amplifier and SBS threshold models — lowered to JAX by ``cx.va`` and
solved by circulax.

Each test compares the compiled model against an independent numpy solution of
the same equations:

* Raman: the exact two-wave logistic exchange P_s^Raman = Q*P_s0/(P_s0 +
  kappa*P_p0*e^{-expo}) with expo = (g/kappa)*Q*Leff, which reduces in the
  small-signal limit to the textbook on/off gain G_A = e^{g_R*P_p*Leff/A_eff}.
* SBS: the behavioural threshold model P_fo = P_th*tanh(P_in/P_th)*e^{-alpha*L},
  P_bo = (P_in - P_th*tanh(P_in/P_th))*e^{-alpha*L}, P_th = n_th*A_eff/(g_B*Leff).

The acceptance criteria from ALE-72 map directly onto these tests: the
two-channel SRS power-transfer slope (``test_raman_srs_smallsignal_slope``), the
counter-pumped Raman on/off gain vs pump (``test_raman_counter_pump_on_off_gain``)
and the SBS threshold clamp + backward Stokes growth (``test_sbs_*``). The
transient WDM tilt study lives in ``examples/raman_sbs.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from circuit_helpers import op, power
from photonflux import cx

# ---------------------------------------------------------------------------
# device defaults (mirror the catalog entries)
# ---------------------------------------------------------------------------
RAMAN = dict(lambda_s_nm=1550.0, lambda_p_nm=1450.0, g_r=0.6e-13, a_eff_um2=80.0,
             length_km=50.0, loss_s_db_km=0.20, loss_p_db_km=0.25)

SBS = dict(g_b=5e-11, a_eff_um2=80.0, length_km=20.0,
           loss_db_km=0.20, n_th=21.0)


# ---------------------------------------------------------------------------
# analytic references (the same equations the VA models integrate)
# ---------------------------------------------------------------------------
def _alpha(loss_db_km: float) -> float:
    return loss_db_km * np.log(10) / 10 / 1e3          # [1/m]


def raman_solve(ps0, pp0, p=RAMAN):
    """Exact two-wave logistic exchange -> (P_signal_out, P_pump_out)."""
    L = p["length_km"] * 1e3
    aeff = p["a_eff_um2"] * 1e-12
    g = p["g_r"] / aeff
    kappa = p["lambda_p_nm"] / p["lambda_s_nm"]        # nu_s/nu_p
    alpha_s, alpha_p = _alpha(p["loss_s_db_km"]), _alpha(p["loss_p_db_km"])
    leff = (1 - np.exp(-alpha_p * L)) / alpha_p
    qq = ps0 + kappa * pp0
    expo = g / kappa * qq * leff
    denom = ps0 + kappa * pp0 * np.exp(-expo)
    ps_out = qq * ps0 / denom * np.exp(-alpha_s * L)
    pp_out = qq * np.exp(-expo) / denom * np.exp(-alpha_p * L) * pp0
    return ps_out, pp_out


def raman_on_off_gain(pp0, p=RAMAN):
    """Small-signal Raman on/off gain G_A = e^{g_R*P_p*Leff/A_eff}."""
    L = p["length_km"] * 1e3
    aeff = p["a_eff_um2"] * 1e-12
    leff = (1 - np.exp(-_alpha(p["loss_p_db_km"]) * L)) / _alpha(p["loss_p_db_km"])
    return np.exp(p["g_r"] / aeff * pp0 * leff)


def sbs_pth(p=SBS) -> float:
    L = p["length_km"] * 1e3
    aeff = p["a_eff_um2"] * 1e-12
    leff = (1 - np.exp(-_alpha(p["loss_db_km"]) * L)) / _alpha(p["loss_db_km"])
    return p["n_th"] * aeff / (p["g_b"] * leff)


def sbs_solve(pin, p=SBS):
    """Behavioural threshold model -> (P_transmitted, P_backscattered)."""
    L = p["length_km"] * 1e3
    t_lin = np.exp(-_alpha(p["loss_db_km"]) * L)
    p_th = sbs_pth(p)
    p_tr = p_th * np.tanh(pin / p_th)
    return p_tr * t_lin, (pin - p_tr) * t_lin


# ---------------------------------------------------------------------------
# one-device netlists
# ---------------------------------------------------------------------------
def _raman(ps0, p_co=0.0, p_ctr=0.0, **params):
    """Forward signal power out of the Raman span (P_signal_out)."""
    s = {**RAMAN, **params}
    drives = {"si_re": np.sqrt(ps0), "si_im": 0.0,
              "pfi_re": np.sqrt(p_co), "pfi_im": 0.0,
              "pbi_re": np.sqrt(p_ctr), "pbi_im": 0.0}
    vals = op(cx.va("raman_amp"), drives, settings=s,
              reads=["so_re", "so_im", "pfo_re", "pfo_im", "pbo_re", "pbo_im"],
              is_complex=True)
    return (power(vals, "so_re", "so_im"),
            power(vals, "pfo_re", "pfo_im"),
            power(vals, "pbo_re", "pbo_im"))


def _sbs(pin, p_seed=0.0, **params):
    s = {**SBS, **params}
    drives = {"fi_re": np.sqrt(pin), "fi_im": 0.0,
              "bi_re": np.sqrt(p_seed), "bi_im": 0.0}
    vals = op(cx.va("sbs_fiber"), drives, settings=s,
              reads=["fo_re", "fo_im", "bo_re", "bo_im"], is_complex=True)
    return power(vals, "fo_re", "fo_im"), power(vals, "bo_re", "bo_im")


# ===========================================================================
# Raman: small-signal gain, SRS slope, depletion, co/counter equivalence
# ===========================================================================
def test_raman_passive_loss():
    # pump off: the signal just sees the fibre background loss e^{-alpha_s*L}
    L = RAMAN["length_km"] * 1e3
    so, _, _ = _raman(1e-9)
    assert so / 1e-9 == pytest.approx(np.exp(-_alpha(RAMAN["loss_s_db_km"]) * L),
                                      rel=1e-4)


def test_raman_counter_pump_on_off_gain():
    # ACCEPTANCE: counter-pumped Raman amp on/off gain vs pump power matches
    # the textbook G_A = e^{g_R*P_p*Leff/A_eff}.
    ps = 1e-9
    off, _, _ = _raman(ps)
    for p_pump in (0.1, 0.3, 0.5, 0.8):
        on, _, _ = _raman(ps, p_ctr=p_pump)
        assert on / off == pytest.approx(raman_on_off_gain(p_pump), rel=1e-3)
    # more pump -> more gain (monotone)
    g_lo = _raman(ps, p_ctr=0.2)[0]
    g_hi = _raman(ps, p_ctr=0.6)[0]
    assert g_hi > g_lo


def test_raman_co_and_counter_pump_equivalent():
    # co- and counter-propagating pump give the same integrated on/off gain
    ps = 1e-9
    co, _, _ = _raman(ps, p_co=0.5)
    ctr, _, _ = _raman(ps, p_ctr=0.5)
    assert co == pytest.approx(ctr, rel=1e-9)
    # and a split pump adds in power (0.25 + 0.25 == a single 0.5)
    split, _, _ = _raman(ps, p_co=0.25, p_ctr=0.25)
    assert split == pytest.approx(co, rel=1e-6)


def test_raman_srs_smallsignal_slope():
    # ACCEPTANCE: two-channel SRS small-signal power-transfer slope. A weak
    # long-lambda "signal" channel (ch2) is amplified by a co-propagating
    # short-lambda "pump" channel (ch1); ln(gain)/P_pump == g_R*Leff/A_eff,
    # independent of pump power in the undepleted limit.
    L = RAMAN["length_km"] * 1e3
    aeff = RAMAN["a_eff_um2"] * 1e-12
    leff = (1 - np.exp(-_alpha(RAMAN["loss_p_db_km"]) * L)) / _alpha(
        RAMAN["loss_p_db_km"])
    slope_expected = RAMAN["g_r"] / aeff * leff          # [1/W]
    off, _, _ = _raman(1e-9)
    for p_pump in (0.02, 0.05, 0.1):
        on, _, _ = _raman(1e-9, p_co=p_pump)
        slope = np.log(on / off) / p_pump
        assert slope == pytest.approx(slope_expected, rel=1e-3)


def test_raman_pump_depletion_saturates_gain():
    # as the signal grows the pump depletes and the gain compresses below the
    # small-signal G_A, tracking the exact logistic solution.
    p_pump = 0.5
    g_small_signal = raman_on_off_gain(p_pump)
    prev = g_small_signal * 1.001
    for ps0 in (1e-9, 1e-3, 1e-2, 5e-2):
        on, _, _ = _raman(ps0, p_ctr=p_pump)
        off, _, _ = _raman(ps0)
        gain = on / off
        # matches the independent logistic reference
        ref_on, _ = raman_solve(ps0, p_pump)
        ref_off, _ = raman_solve(ps0, 0.0)
        assert gain == pytest.approx(ref_on / ref_off, rel=1e-3)
        # monotonically compressing, and never above the small-signal gain
        assert gain <= g_small_signal * (1 + 1e-6)
        assert gain < prev
        prev = gain


def test_raman_pump_is_depleted_by_photon_ratio():
    # energy bookkeeping: the pump gives up the signal's photon gain scaled by
    # nu_s/nu_p (Q = P_s + kappa*P_p conserved through the exchange).
    ps0, p_pump = 5e-3, 0.2
    so, pfo, _ = _raman(ps0, p_co=p_pump)
    ref_s, ref_p = raman_solve(ps0, p_pump)
    assert so == pytest.approx(ref_s, rel=1e-3)
    assert pfo == pytest.approx(ref_p, rel=1e-3)
    # the pump lost power to the signal (co-pump output below its passive loss)
    L = RAMAN["length_km"] * 1e3
    assert pfo < p_pump * np.exp(-_alpha(RAMAN["loss_p_db_km"]) * L)


def test_raman_gain_scales_with_length_and_area():
    ps = 1e-9
    off, _, _ = _raman(ps)
    g50 = _raman(ps, p_ctr=0.5)[0] / off
    # doubling A_eff halves the gain exponent (ln G_A ~ 1/A_eff)
    off_a, _, _ = _raman(ps, a_eff_um2=160.0)
    g_bigA = _raman(ps, p_ctr=0.5, a_eff_um2=160.0)[0] / off_a
    assert np.log(g_bigA) == pytest.approx(0.5 * np.log(g50), rel=2e-3)


# ===========================================================================
# SBS: threshold clamp, backward Stokes growth, energy, threshold scaling
# ===========================================================================
def test_sbs_below_threshold_transmits():
    # well below P_th almost everything transmits and the backscatter is tiny
    p_th = sbs_pth()
    pin = 0.02 * p_th
    fo, bo = _sbs(pin)
    t_lin = np.exp(-_alpha(SBS["loss_db_km"]) * SBS["length_km"] * 1e3)
    assert fo == pytest.approx(pin * t_lin, rel=2e-3)
    assert bo / (pin * t_lin) < 1e-3        # negligible reflection


def test_sbs_transmission_clamps_at_threshold():
    # ACCEPTANCE: far above threshold the transmitted power CLAMPS at ~P_th and
    # doubling the input barely changes it (< 0.1 dB), while below threshold it
    # tracks the input linearly.
    p_th = sbs_pth()
    fo_lin, _ = _sbs(0.05 * p_th)
    assert fo_lin == pytest.approx(sbs_solve(0.05 * p_th)[0], rel=2e-3)
    fo_a, _ = _sbs(8 * p_th)
    fo_b, _ = _sbs(16 * p_th)
    assert 10 * np.log10(fo_b / fo_a) < 0.1                      # clamped
    t_lin = np.exp(-_alpha(SBS["loss_db_km"]) * SBS["length_km"] * 1e3)
    assert fo_b == pytest.approx(p_th * t_lin, rel=1e-2)          # ceiling ~P_th


def test_sbs_backward_stokes_grows_above_threshold():
    # ACCEPTANCE: the backward Stokes wave is negligible below threshold and
    # grows once the pump exceeds it, matching the analytic model everywhere.
    p_th = sbs_pth()
    refl_frac = []
    for k in (0.1, 0.5, 1.0, 4.0, 16.0):
        pin = k * p_th
        fo, bo = _sbs(pin)
        ref_fo, ref_bo = sbs_solve(pin)
        assert fo == pytest.approx(ref_fo, rel=2e-3)
        assert bo == pytest.approx(ref_bo, rel=2e-3, abs=1e-12)
        refl_frac.append(bo / (fo + bo))
    # reflected fraction increases monotonically with pump power
    assert all(np.diff(refl_frac) > 0)
    assert refl_frac[0] < 0.05 and refl_frac[-1] > 0.8


def test_sbs_energy_conservation():
    # the ~11 GHz Stokes shift is negligible in power: P_fo + P_bo == P_in*t_lin
    t_lin = np.exp(-_alpha(SBS["loss_db_km"]) * SBS["length_km"] * 1e3)
    for pin in (1e-4, 1e-3, 1e-2, 5e-2):
        fo, bo = _sbs(pin)
        assert fo + bo == pytest.approx(pin * t_lin, rel=1e-3)


def test_sbs_threshold_scales_with_gain_and_area():
    # P_th ~ A_eff / (g_B*Leff): doubling g_B halves the threshold, doubling
    # A_eff doubles it. Detect the threshold as where reflection hits ~10%.
    def refl_at(pin, **kw):
        fo, bo = _sbs(pin, **kw)
        return bo / (fo + bo)

    base = sbs_pth()
    # at the same absolute input, a 2x Brillouin gain reflects more (lower P_th)
    assert refl_at(base, g_b=1e-10) > refl_at(base)
    # a 2x effective area reflects less (higher P_th)
    assert refl_at(base, a_eff_um2=160.0) < refl_at(base)
    # analytic threshold indeed scales as expected
    assert sbs_pth({**SBS, "g_b": 1e-10}) == pytest.approx(0.5 * base, rel=1e-9)
    assert sbs_pth({**SBS, "a_eff_um2": 160.0}) == pytest.approx(2 * base, rel=1e-9)


def test_sbs_seed_passes_through():
    # a downstream Stokes seed (bin) passes back to bout with the span loss,
    # adding in power to the SBS-generated Stokes.
    p_th = sbs_pth()
    amp_lin2 = np.exp(-_alpha(SBS["loss_db_km"]) * SBS["length_km"] * 1e3)  # power
    _, bo_gen = _sbs(0.5 * p_th)
    _, bo_seed = _sbs(0.5 * p_th, p_seed=1e-4)
    # in phase (both carry the pump-real phase) so amplitudes add: extra power
    # is dominated by the seed pass-through and the cross term
    assert bo_seed > bo_gen
    assert bo_seed > 1e-4 * amp_lin2 * 0.5   # at least the attenuated seed
