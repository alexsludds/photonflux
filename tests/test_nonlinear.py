"""Steady-state physics of the nonlinear / coherent-field Verilog-A models —
the TPA+FCA waveguide, the SOA gain reservoir, the mirror, the ring-comb
filter, and the nonlinear ring — lowered to JAX by ``cx.va`` and solved by
circulax.

Each test compares the compiled model against an independent numpy solution of
the same equations (closed forms where they exist, root-finds where the steady
state is transcendental). The four-wave-mixing and Fabry-Perot-laser studies
are transient and live in ``examples/wg_fwm.py``, ``ring_fwm.py`` and
``soa_fp_laser.py``, which carry their own analytic asserts.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.optimize import brentq

from circuit_helpers import build, op, power
from photonflux import cx

C0 = 2.99792458e8
HPL = 6.62607015e-34


# ===========================================================================
# waveguide_nl — TPA + FCA transmission
# ===========================================================================
WG = dict(lambda_nm=1310.0, length_um=5000.0, loss_db_m=200.0,
          a_eff_um2=0.1, beta_tpa=8e-12, sigma_fca=1.45e-21, tau_fc=1e-9)


def wg_analytic(p_in: float, **over) -> float:
    """Steady-state P_out/P_in of waveguide_nl's lumped equations."""
    p = {**WG, **over}
    alpha = p["loss_db_m"] * np.log(10) / 10
    length = p["length_um"] * 1e-6
    aeff = p["a_eff_um2"] * 1e-12
    leff = (1 - np.exp(-alpha * length)) / alpha
    eph = HPL * C0 / (p["lambda_nm"] * 1e-9)
    iin = p_in / aeff
    t_tpa = np.exp(-alpha * length) / (1 + p["beta_tpa"] * iin * leff)

    def fca_residual(nfc: float) -> float:
        t_fca = np.exp(-p["sigma_fca"] * nfc * leff)
        iavg = 0.5 * iin * (1 + t_tpa * t_fca)
        return p["tau_fc"] * p["beta_tpa"] * iavg**2 / (2 * eph) - nfc

    nfc = brentq(fca_residual, 0.0, 1e27)
    return t_tpa * np.exp(-p["sigma_fca"] * nfc * leff)


def _wg_T(p_in: float, **params) -> float:
    vals = op(cx.va("waveguide_nl"),
              {"in_re": np.sqrt(p_in), "in_im": 0.0},
              settings={**WG, **params}, reads=["out_re", "out_im"])
    return power(vals, "out_re", "out_im") / p_in


def test_waveguide_linear_limit():
    # beta = 0: plain exp(-alpha*L) whatever the power
    t = _wg_T(10e-3, beta_tpa=0.0)
    assert t == pytest.approx(10 ** (-200.0 * 5000e-6 / 10), rel=1e-6)


def test_waveguide_tpa_fca_transmission():
    # transmission droops with power, exactly as the lumped equations say
    for p_in in (1e-4, 1e-3, 10e-3, 100e-3):
        assert _wg_T(p_in) == pytest.approx(wg_analytic(p_in), rel=1e-6)
    assert _wg_T(100e-3) < 0.9 * _wg_T(1e-4)


def test_waveguide_nl_phase():
    # Kerr + FCD rotate the field: |phase| grows with power, |E| untouched
    p_in = 50e-3
    vals = op(cx.va("waveguide_nl"),
              {"in_re": np.sqrt(p_in), "in_im": 0.0},
              settings={**WG, "n2_kerr": 4.5e-18, "dn_dn": -4e-27},
              reads=["out_re", "out_im"])
    phi = np.arctan2(vals["out_im"].real, vals["out_re"].real)
    assert abs(phi) > 0.05                       # a measurable nonlinear phase
    assert power(vals, "out_re", "out_im") / p_in == pytest.approx(
        wg_analytic(p_in), rel=1e-6)


def wg_kerr_k(**over) -> tuple[float, float]:
    """(k [rad/W], power transmission T) of the pure-Kerr segment: the phase is
    trapezoidal in z (endpoint-average intensity x geometric length), so
    k = gamma * (1 + e^{-alpha*L})/2 * L with gamma = 2*pi*n2/(lambda*A_eff)."""
    p = {**WG, **over}
    alpha = p["loss_db_m"] * np.log(10) / 10
    length = p["length_um"] * 1e-6
    gamma = 2 * np.pi * p["n2_kerr"] / (p["lambda_nm"] * 1e-9
                                        * p["a_eff_um2"] * 1e-12)
    return (gamma * 0.5 * (1 + np.exp(-alpha * length)) * length,
            np.exp(-alpha * length))


def test_waveguide_kerr_phase_lumping():
    # beta = sigma = 0: phi = -k*P_in exactly (pins the trapezoidal lumping that
    # examples/wg_fwm.py shows converging on the distributed NLSE)
    kerr = dict(beta_tpa=0.0, sigma_fca=0.0, n2_kerr=4.5e-18, dn_dn=0.0)
    k, _ = wg_kerr_k(**kerr)
    for p_in in (10e-3, 50e-3):
        vals = op(cx.va("waveguide_nl"),
                  {"in_re": np.sqrt(p_in), "in_im": 0.0},
                  settings={**WG, **kerr}, reads=["out_re", "out_im"])
        phi = np.arctan2(vals["out_im"].real, vals["out_re"].real)
        assert phi == pytest.approx(-k * p_in, rel=1e-6)


# ===========================================================================
# soa — gain, transparency, saturation (Agrawal-Olsson reservoir)
# ===========================================================================
SOA = dict(g0_db=20.0, i_op_ma=80.0, i_tr_ma=8.0, p_sat=10e-3,
           tau_c=0.3e-9, tau_bw=1e-12, p_seed=0.0, Von=1.2, Rs=3.0)


def soa_h0(i_ma: float) -> float:
    hop = SOA["g0_db"] * np.log(10) / 10
    return hop * (i_ma - SOA["i_tr_ma"]) / (SOA["i_op_ma"] - SOA["i_tr_ma"])


def soa_gain(p_in: float, i_ma: float) -> float:
    """Saturated gain: root of h0 - h - (e^h - 1)*P_in/p_sat = 0."""
    h0 = soa_h0(i_ma)
    f = lambda h: h0 - h - (np.exp(h) - 1) * p_in / SOA["p_sat"]
    return np.exp(brentq(f, -30.0, h0 + 1e-9)) if h0 > 0 else np.exp(
        brentq(f, h0 - 1e-9, 30.0))


def _soa_fwd(p_in: float, i_ma: float, *, p_back: float = 0.0, **params) -> float:
    """Forward output power of the SOA at bias current i_ma, optionally with a
    saturating backward drive p_back."""
    v = SOA["Von"] + SOA["Rs"] * i_ma * 1e-3
    vals = op(cx.va("soa"),
              {"fi_re": np.sqrt(p_in), "fi_im": 0.0,
               "bi_re": np.sqrt(p_back), "bi_im": 0.0,
               "an": v, "cat": 0.0},
              settings={**SOA, **params},
              reads=["fo_re", "fo_im"], terms=[("bo_re", "bo_im")])
    return power(vals, "fo_re", "fo_im")


def test_soa_small_signal_gain():
    # 10 nW probe at i_op: the full g0 = 20 dB (at 1 uW the reservoir already
    # compresses by ~1%, pinned by test_soa_gain_saturation)
    assert _soa_fwd(1e-8, 80.0) / 1e-8 == pytest.approx(100.0, rel=1e-3)


def test_soa_transparency_and_absorption():
    assert _soa_fwd(1e-6, 8.0) / 1e-6 == pytest.approx(1.0, rel=1e-3)
    assert _soa_fwd(1e-6, 0.0) / 1e-6 < 0.7           # unbiased: an absorber


def test_soa_gain_saturation():
    # gain compresses along the analytic reservoir solution
    for p_in in (0.1e-3, 1e-3, 5e-3):
        assert _soa_fwd(p_in, 80.0) / p_in == pytest.approx(
            soa_gain(p_in, 80.0), rel=1e-3)
    assert soa_gain(5e-3, 80.0) < 30                  # sanity: deep saturation


def test_soa_bidirectional_shared_gain():
    # backward power saturates the forward gain through the shared reservoir
    p_probe, p_drive = 1e-6, 2e-3
    g_fwd = _soa_fwd(p_probe, 80.0, p_back=p_drive) / p_probe
    assert g_fwd == pytest.approx(soa_gain(p_probe + p_drive, 80.0), rel=1e-3)


def test_soa_chirp_phase():
    # alpha_h rotates the output by -alpha_h*h/2
    v = SOA["Von"] + SOA["Rs"] * 80e-3
    vals = op(cx.va("soa"),
              {"fi_re": 1e-6, "fi_im": 0.0, "bi_re": 0.0, "bi_im": 0.0,
               "an": v, "cat": 0.0},
              settings={**SOA, "alpha_h": 4.0},
              reads=["fo_re", "fo_im"], terms=[("bo_re", "bo_im")])
    phi = np.arctan2(vals["fo_im"].real, vals["fo_re"].real)
    h = np.log(soa_gain(0.0, 80.0))
    assert np.angle(np.exp(1j * (phi + 0.5 * 4.0 * h))) == pytest.approx(0.0, abs=1e-3)


# ===========================================================================
# mirror — unitarity and phase trim
# ===========================================================================
def test_mirror_split_and_unitarity():
    vals = op(cx.va("mirror"),
              {"li_re": 1.0, "li_im": 0.0, "ri_re": 0.0, "ri_im": 0.0},
              settings={"refl": 0.3},
              reads=["lo_re", "lo_im", "ro_re", "ro_im"])
    assert power(vals, "lo_re", "lo_im") == pytest.approx(0.3, rel=1e-9)
    assert power(vals, "ro_re", "ro_im") == pytest.approx(0.7, rel=1e-9)
    # transmission carries the unitary j: purely imaginary output
    assert abs(vals["ro_re"].real) < 1e-12
    assert vals["ro_im"].real == pytest.approx(np.sqrt(0.7), rel=1e-9)


def test_mirror_reflection_phase():
    vals = op(cx.va("mirror"),
              {"li_re": 1.0, "li_im": 0.0, "ri_re": 0.0, "ri_im": 0.0},
              settings={"refl": 1.0, "phi_r_deg": 90.0},
              reads=["lo_re", "lo_im", "ro_re", "ro_im"])
    assert abs(vals["lo_re"].real) < 1e-12
    assert vals["lo_im"].real == pytest.approx(1.0, rel=1e-9)


# ===========================================================================
# ring_filter — comb of five modes, heater tuning
# ===========================================================================
RF = dict(lambda_res_nm=1310.0, radius_um=100.0, n_g=4.0, loss_db_m=100.0,
          kappa2_in=0.05, kappa2_drop=0.05, r_heater=500.0, dl_dmw_pm=20.0)


def rf_rates() -> tuple[float, float, float, float]:
    circ = 2 * np.pi * RF["radius_um"] * 1e-6
    v_g = C0 / RF["n_g"]
    t_rt = circ / v_g
    inv_ti = RF["loss_db_m"] * np.log(10) / 10 * v_g / 2
    inv_te1 = RF["kappa2_in"] / (2 * t_rt)
    inv_te2 = RF["kappa2_drop"] / (2 * t_rt)
    tau = 1 / (inv_ti + inv_te1 + inv_te2)
    return tau, inv_te1, inv_te2, 1 / t_rt   # (tau, 1/te1, 1/te2, FSR[Hz])


def rf_drop_analytic(dnu: np.ndarray) -> np.ndarray:
    """|T_drop|^2 of the five-mode comb at detuning dnu [Hz] from m=0."""
    tau, ite1, ite2, fsr = rf_rates()
    amp = np.zeros_like(dnu, dtype=complex)
    for m in (-2, -1, 0, 1, 2):
        delta = 2 * np.pi * (dnu - m * fsr)
        amp += 1j * np.sqrt(2 * ite1) * np.sqrt(2 * ite2) / (1 / tau - 1j * delta)
    return np.abs(amp) ** 2


def _rf_drop(lambda_nm: float, v_heat: float = 0.0) -> float:
    vals = op(cx.va("ring_filter"),
              {"in_re": 1e-3, "in_im": 0.0, "hp": v_heat, "hn": 0.0},
              settings={"lambda_nm": lambda_nm, **RF},
              reads=["drop_re", "drop_im"], terms=[("thru_re", "thru_im")])
    return power(vals, "drop_re", "drop_im") / 1e-6


def test_ring_filter_drop_peak_and_comb():
    lam0 = RF["lambda_res_nm"]
    tau, ite1, ite2, fsr = rf_rates()
    t_peak = rf_drop_analytic(np.array([0.0]))[0]
    assert _rf_drop(lam0) == pytest.approx(t_peak, rel=1e-6)
    assert 0.5 < t_peak < 1.0    # strongly coupled add-drop
    # the comb repeats one FSR away (m = +1 mode): same peak height
    lam_p1 = 1 / (1 / (lam0 * 1e-9) + fsr / C0) * 1e9
    assert _rf_drop(lam_p1) == pytest.approx(
        rf_drop_analytic(np.array([fsr]))[0], rel=1e-4)
    # and is dark between modes
    lam_half = 1 / (1 / (lam0 * 1e-9) + 0.5 * fsr / C0) * 1e9
    assert _rf_drop(lam_half) < 0.01 * t_peak


def test_ring_filter_heater_shift():
    # heater power red-shifts the comb: peak moves to lambda_res + dl*P
    v = 2.0
    p_mw = v**2 / RF["r_heater"] * 1e3
    lam_shift = RF["lambda_res_nm"] + RF["dl_dmw_pm"] * p_mw * 1e-3   # pm -> nm
    t_on_peak = _rf_drop(lam_shift, v_heat=v)
    assert t_on_peak == pytest.approx(rf_drop_analytic(np.array([0.0]))[0], rel=1e-4)
    # the un-shifted wavelength has fallen off the peak
    assert _rf_drop(RF["lambda_res_nm"], v_heat=v) < 0.9 * t_on_peak


# ===========================================================================
# ring_nl — TPA/FCA-limited quality factor
# ===========================================================================
RNL = dict(lambda_res_nm=1310.0, radius_um=10.0, n_g=4.0, loss_db_m=30.0,
           kappa2=6e-4, a_eff_um2=0.1, beta_tpa=8e-12, sigma_fca=1.45e-21,
           tau_fc=1e-9, n2_kerr=0.0, dn_dn=0.0)


def rnl_rates() -> tuple[float, float, float]:
    circ = 2 * np.pi * RNL["radius_um"] * 1e-6
    v_g = C0 / RNL["n_g"]
    inv_ti = RNL["loss_db_m"] * np.log(10) / 10 * v_g / 2
    inv_te = RNL["kappa2"] / (2 * circ / v_g)
    return inv_ti, inv_te, v_g


def _rnl_sweep(p_in: float, span_pm: float = 8.0, n: int = 241, **over):
    lam0 = RNL["lambda_res_nm"]
    lams = lam0 + np.linspace(-span_pm, span_pm, n) * 1e-3   # pm -> nm
    c = build(cx.va("ring_nl"), {"in_re": np.sqrt(p_in), "in_im": 0.0},
              settings={"lambda_nm": lam0, **RNL, **over},
              reads=["out_re", "out_im"])
    y = c.dc(params={"DUT.lambda_nm": jnp.asarray(lams)})
    t = np.asarray((c.port(y, "out_re") ** 2 + c.port(y, "out_im") ** 2).real) / p_in
    return lams, t


def fwhm_pm(lams: np.ndarray, t: np.ndarray) -> float:
    half = 0.5 * (1 + t.min())
    below = np.where(t < half)[0]
    return (lams[below[-1]] - lams[below[0]]) * 1e3   # nm -> pm


def test_ring_nl_linear_lorentzian():
    # beta_tpa = 0 recovers the exact linear CMT Lorentzian (at Q ~ 1.2e6 even
    # 1 uW builds enough circulating power for ~0.5% of FCA, so the "linear" pin
    # must actually switch the nonlinearity off)
    inv_ti, inv_te, _ = rnl_rates()
    tau = 1 / (inv_ti + inv_te)
    lams, t = _rnl_sweep(1e-6, beta_tpa=0.0)
    lam = lams * 1e-9
    delta = 2 * np.pi * C0 * (1 / lam - 1 / (RNL["lambda_res_nm"] * 1e-9))
    t_ref = ((1 - tau * 2 * inv_te) ** 2 + (tau * delta) ** 2) / (1 + (tau * delta) ** 2)
    assert np.abs(t - t_ref).max() < 1e-6
    q_loaded = 2 * np.pi * C0 / (RNL["lambda_res_nm"] * 1e-9) * tau / 2
    # rel = 0.1: the FWHM is ~1.1 pm read off a 0.067 pm sweep grid
    assert RNL["lambda_res_nm"] / (fwhm_pm(lams, t) * 1e-3) == pytest.approx(
        q_loaded, rel=0.1)


def test_ring_nl_q_droop_with_power():
    # nonlinear absorption broadens the line and lifts the dip floor
    lams_lo, t_lo = _rnl_sweep(1e-6)
    lams_hi, t_hi = _rnl_sweep(0.5e-3, span_pm=60.0)
    q_lo = RNL["lambda_res_nm"] / (fwhm_pm(lams_lo, t_lo) * 1e-6)
    q_hi = RNL["lambda_res_nm"] / (fwhm_pm(lams_hi, t_hi) * 1e-6)
    assert q_hi < 0.6 * q_lo          # the Q collapse
    assert t_hi.min() > t_lo.min()    # coupling condition walks away from critical
