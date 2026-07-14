"""Physics pins for the nonlinear-absorption waveguide, the SOA, and the
cavity building blocks (mirror, ring_filter, ring_nl), run through ngspice.

Each test compares the compiled OSDI model against an independent numpy
solution of the same model equations (closed forms where they exist,
root-finds where the steady state is transcendental).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

import lightspice as ls

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


def _wg_op(eng, p_in: float, **params) -> float:
    ckt = ls.Circuit("wg nl")
    ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
    ckt.raw(f"Vre in_re 0 {np.sqrt(p_in)}", "Vim in_im 0 0")
    ckt.device(ls.va("waveguide_nl"), "wg",
               "in_re", "in_im", "out_re", "out_im", "0", **{**WG, **params})
    r = eng.op(ckt)
    return float(r["out_re"][0] ** 2 + r["out_im"][0] ** 2) / p_in


def test_waveguide_linear_limit(eng):
    # beta = 0: plain exp(-alpha*L) whatever the power
    t = _wg_op(eng, 10e-3, beta_tpa=0.0)
    assert t == pytest.approx(10 ** (-200.0 * 5000e-6 / 10), rel=1e-6)


def test_waveguide_tpa_fca_transmission(eng):
    # transmission droops with power, exactly as the lumped equations say
    for p_in in (1e-4, 1e-3, 10e-3, 100e-3):
        assert _wg_op(eng, p_in) == pytest.approx(wg_analytic(p_in), rel=1e-6)
    assert _wg_op(eng, 100e-3) < 0.9 * _wg_op(eng, 1e-4)


def test_waveguide_nl_phase(eng):
    # Kerr + FCD rotate the field: check |phase| grows with power and that
    # the rotation leaves |E| untouched (phase-only nonlinearity)
    p_in = 50e-3
    ckt = ls.Circuit("wg phase")
    ckt.raw(f"Vre in_re 0 {np.sqrt(p_in)}", "Vim in_im 0 0")
    ckt.device(ls.va("waveguide_nl"), "wg",
               "in_re", "in_im", "out_re", "out_im", "0",
               **{**WG, "n2_kerr": 4.5e-18, "dn_dn": -4e-27})
    r = eng.op(ckt)
    phi = np.arctan2(float(r["out_im"][0]), float(r["out_re"][0]))
    assert abs(phi) > 0.05          # a measurable nonlinear phase
    t_power = (r["out_re"][0] ** 2 + r["out_im"][0] ** 2) / p_in
    assert t_power == pytest.approx(wg_analytic(p_in), rel=1e-6)


def wg_kerr_k(**over) -> tuple[float, float]:
    """(k [rad/W], power transmission T) of the pure-Kerr segment: the phase
    is trapezoidal in z (endpoint-average intensity x geometric length), so
    k = gamma * (1 + e^{-alpha*L})/2 * L with gamma = 2*pi*n2/(lambda*A_eff)."""
    p = {**WG, **over}
    alpha = p["loss_db_m"] * np.log(10) / 10
    length = p["length_um"] * 1e-6
    gamma = 2 * np.pi * p["n2_kerr"] / (p["lambda_nm"] * 1e-9
                                        * p["a_eff_um2"] * 1e-12)
    return (gamma * 0.5 * (1 + np.exp(-alpha * length)) * length,
            np.exp(-alpha * length))


def test_waveguide_kerr_phase_lumping(eng):
    # beta = sigma = 0: phi = -k*P_in exactly (pins the trapezoidal lumping
    # that examples/wg_fwm.py shows converging on the distributed NLSE)
    kerr = dict(beta_tpa=0.0, sigma_fca=0.0, n2_kerr=4.5e-18, dn_dn=0.0)
    k, _ = wg_kerr_k(**kerr)
    for p_in in (10e-3, 50e-3):
        ckt = ls.Circuit("wg kerr phase")
        ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
        ckt.raw(f"Vre in_re 0 {np.sqrt(p_in)}", "Vim in_im 0 0")
        ckt.device(ls.va("waveguide_nl"), "wg",
                   "in_re", "in_im", "out_re", "out_im", "0",
                   **{**WG, **kerr})
        r = eng.op(ckt)
        phi = np.arctan2(float(r["out_im"][0]), float(r["out_re"][0]))
        assert phi == pytest.approx(-k * p_in, rel=1e-6)


def test_waveguide_fwm_idler(eng):
    """chi(3) four-wave mixing through the OSDI/ngspice toolchain: pump and
    signal tones beat, the instantaneous Kerr phase scatters them, and the
    idler at 2*f_p - f_s carries the textbook eta = T*(k*P_p)^2*P_s — with
    slope 2 in pump power. (examples/wg_fwm.py is the full circulax study.)
    """
    from scipy.special import jv

    kerr = dict(loss_db_m=100.0, beta_tpa=0.0, sigma_fca=0.0,
                n2_kerr=4.5e-18, dn_dn=0.0)
    k, t_pwr = wg_kerr_k(**kerr)
    f_p, f_s, p_s = 20e9, 30e9, 1e-4
    dt, t1 = 0.5e-12, 2e-9

    def idler(p_p: float) -> float:
        ar, ai = np.sqrt(p_p), np.sqrt(p_s)
        ckt = ls.Circuit("wg fwm")
        ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
        # series-stacked tones: E = sqrt(Pp)e^{j2pi fp t} + sqrt(Ps)e^{j2pi fs t}
        ckt.raw(f"Vre1 in_re mre SIN(0 {ar} {f_p} 0 0 90)",
                f"Vre2 mre 0     SIN(0 {ai} {f_s} 0 0 90)",
                f"Vim1 in_im mim SIN(0 {ar} {f_p} 0 0 0)",
                f"Vim2 mim 0     SIN(0 {ai} {f_s} 0 0 0)")
        ckt.device(ls.va("waveguide_nl"), "wg",
                   "in_re", "in_im", "out_re", "out_im", "0",
                   **{**WG, **kerr})
        r = eng.tran(ckt, f"{dt*1e12}p", f"{t1*1e9}n")
        # resample the last 10 beat periods onto an exact uniform grid
        tt = 1e-9 + dt * np.arange(2000)
        e = (np.interp(tt, r.t, r["out_re"])
             + 1j * np.interp(tt, r.t, r["out_im"]))
        a = np.fft.fft(e) / len(e)
        f = np.fft.fftfreq(len(e), d=dt)
        return float(np.abs(a[np.argmin(np.abs(f - (2 * f_p - f_s)))]) ** 2)

    for p_p in (10e-3, 20e-3):
        x = 2 * k * np.sqrt(p_p * p_s)
        p_i_th = t_pwr * (p_p * jv(1, x) ** 2 + p_s * jv(2, x) ** 2)
        assert idler(p_p) == pytest.approx(p_i_th, rel=2e-2)
    # and the slope-2 signature, self-normalised
    assert idler(20e-3) / idler(10e-3) == pytest.approx(4.0, rel=2e-2)


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


def _soa_out(eng, p_in: float, i_ma: float, **params) -> float:
    v = SOA["Von"] + SOA["Rs"] * i_ma * 1e-3
    ckt = ls.Circuit("soa op")
    ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
    ckt.raw(f"Vre fi_re 0 {np.sqrt(p_in)}", "Vim fi_im 0 0",
            f"Vb an 0 {v}")
    ckt.device(ls.va("soa"), "amp",
               "fi_re", "fi_im", "fo_re", "fo_im",
               "0", "0", "bo_re", "bo_im",       # backward inputs dark
               "an", "0", "0", **{**SOA, **params})
    r = eng.op(ckt)
    return float(r["fo_re"][0] ** 2 + r["fo_im"][0] ** 2)


def test_soa_small_signal_gain(eng):
    # 10 nW probe at i_op: the full g0 = 20 dB (at 1 uW the reservoir already
    # compresses by ~1% — pinned exactly by test_soa_gain_saturation)
    p = _soa_out(eng, 1e-8, 80.0)
    assert p / 1e-8 == pytest.approx(100.0, rel=1e-3)


def test_soa_transparency_and_absorption(eng):
    assert _soa_out(eng, 1e-6, 8.0) / 1e-6 == pytest.approx(1.0, rel=1e-3)
    assert _soa_out(eng, 1e-6, 0.0) / 1e-6 < 0.7      # unbiased: an absorber


def test_soa_gain_saturation(eng):
    # gain compresses along the analytic reservoir solution
    for p_in in (0.1e-3, 1e-3, 5e-3):
        g_ref = soa_gain(p_in, 80.0)
        assert _soa_out(eng, p_in, 80.0) / p_in == pytest.approx(g_ref, rel=1e-3)
    assert soa_gain(5e-3, 80.0) < 30  # sanity: deep saturation at 5 mW in


def test_soa_bidirectional_shared_gain(eng):
    # backward power saturates the forward gain through the shared reservoir
    p_probe, p_sat_drive = 1e-6, 2e-3
    v = SOA["Von"] + SOA["Rs"] * 80e-3
    ckt = ls.Circuit("soa bidi")
    ckt.raw(f"Vre fi_re 0 {np.sqrt(p_probe)}", "Vim fi_im 0 0",
            f"Vbr bi_re 0 {np.sqrt(p_sat_drive)}", "Vbi bi_im 0 0",
            f"Vb an 0 {v}")
    ckt.device(ls.va("soa"), "amp",
               "fi_re", "fi_im", "fo_re", "fo_im",
               "bi_re", "bi_im", "bo_re", "bo_im",
               "an", "0", "0", **SOA)
    r = eng.op(ckt)
    g_fwd = float(r["fo_re"][0] ** 2 + r["fo_im"][0] ** 2) / p_probe
    # the reservoir sees p_probe + p_sat_drive
    g_ref = soa_gain(p_probe + p_sat_drive, 80.0)
    assert g_fwd == pytest.approx(g_ref, rel=1e-3)


def test_soa_chirp_phase(eng):
    # alpha_h rotates the output by -alpha_h*h/2
    v = SOA["Von"] + SOA["Rs"] * 80e-3
    ckt = ls.Circuit("soa chirp")
    ckt.raw("Vre fi_re 0 1u", "Vim fi_im 0 0", f"Vb an 0 {v}")
    ckt.device(ls.va("soa"), "amp",
               "fi_re", "fi_im", "fo_re", "fo_im", "0", "0", "bo_re", "bo_im",
               "an", "0", "0", **{**SOA, "alpha_h": 4.0})
    r = eng.op(ckt)
    phi = np.arctan2(float(r["fo_im"][0]), float(r["fo_re"][0]))
    h = np.log(soa_gain(0.0, 80.0))
    assert np.angle(np.exp(1j * (phi + 0.5 * 4.0 * h))) == pytest.approx(0.0, abs=1e-3)


# ===========================================================================
# mirror — unitarity and phase trim
# ===========================================================================
def test_mirror_split_and_unitarity(eng):
    ckt = ls.Circuit("mirror")
    ckt.raw("Vre li_re 0 1", "Vim li_im 0 0")
    ckt.device(ls.va("mirror"), "m1",
               "li_re", "li_im", "lo_re", "lo_im",
               "0", "0", "ro_re", "ro_im", "0", refl=0.3)
    r = eng.op(ckt)
    p_r = float(r["lo_re"][0] ** 2 + r["lo_im"][0] ** 2)
    p_t = float(r["ro_re"][0] ** 2 + r["ro_im"][0] ** 2)
    assert p_r == pytest.approx(0.3, rel=1e-9)
    assert p_t == pytest.approx(0.7, rel=1e-9)
    # transmission carries the unitary j: purely imaginary output
    assert abs(float(r["ro_re"][0])) < 1e-12
    assert float(r["ro_im"][0]) == pytest.approx(np.sqrt(0.7), rel=1e-9)


def test_mirror_reflection_phase(eng):
    ckt = ls.Circuit("mirror phase")
    ckt.raw("Vre li_re 0 1", "Vim li_im 0 0")
    ckt.device(ls.va("mirror"), "m1",
               "li_re", "li_im", "lo_re", "lo_im",
               "0", "0", "ro_re", "ro_im", "0", refl=1.0, phi_r_deg=90.0)
    r = eng.op(ckt)
    assert abs(float(r["lo_re"][0])) < 1e-12
    assert float(r["lo_im"][0]) == pytest.approx(1.0, rel=1e-9)


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


def _rf_drop(eng, lambda_nm: float, v_heat: float = 0.0) -> float:
    ckt = ls.Circuit("ring filter")
    ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
    ckt.raw("Vre in_re 0 1m", "Vim in_im 0 0", f"Vh hp 0 {v_heat}")
    ckt.device(ls.va("ring_filter"), "rf",
               "in_re", "in_im", "thru_re", "thru_im", "drop_re", "drop_im",
               "hp", "0", "0", lambda_nm=lambda_nm, **RF)
    r = eng.op(ckt)
    return float(r["drop_re"][0] ** 2 + r["drop_im"][0] ** 2) / 1e-6


def test_ring_filter_drop_peak_and_comb(eng):
    lam0 = RF["lambda_res_nm"]
    tau, ite1, ite2, fsr = rf_rates()
    t_peak = rf_drop_analytic(np.array([0.0]))[0]
    assert _rf_drop(eng, lam0) == pytest.approx(t_peak, rel=1e-6)
    assert 0.5 < t_peak < 1.0    # strongly coupled add-drop
    # the comb repeats one FSR away (m = +1 mode): same peak height
    lam_p1 = 1 / (1 / (lam0 * 1e-9) + fsr / C0) * 1e9
    assert _rf_drop(eng, lam_p1) == pytest.approx(
        rf_drop_analytic(np.array([fsr]))[0], rel=1e-4)
    # and is dark between modes
    lam_half = 1 / (1 / (lam0 * 1e-9) + 0.5 * fsr / C0) * 1e9
    assert _rf_drop(eng, lam_half) < 0.01 * t_peak


def test_ring_filter_heater_shift(eng):
    # heater power red-shifts the comb: peak moves to lambda_res + dl*P
    v = 2.0
    p_mw = v**2 / RF["r_heater"] * 1e3
    lam_shift = RF["lambda_res_nm"] + RF["dl_dmw_pm"] * p_mw * 1e-3   # pm -> nm
    t_on_peak = _rf_drop(eng, lam_shift, v_heat=v)
    assert t_on_peak == pytest.approx(rf_drop_analytic(np.array([0.0]))[0], rel=1e-4)
    # the un-shifted wavelength has fallen off the peak
    assert _rf_drop(eng, RF["lambda_res_nm"], v_heat=v) < 0.9 * t_on_peak


# ===========================================================================
# soa + mirror — Fabry-Perot laser (the integration pin)
# ===========================================================================
def test_fp_laser_threshold(eng):
    """SOA between two partial reflectors lases at G_th = 1/(r1*r2).

    DC Newton lands on the dark stationary branch above threshold (it exists
    and Newton likes it; stability is a transient property), so the operating
    point comes from a transient settle — physically, a turn-on.
    """
    r1, r2 = 0.9, 0.3
    h_th = -0.5 * np.log(r1 * r2)
    g_th = np.exp(h_th)
    hop = SOA["g0_db"] * np.log(10) / 10
    i_th = SOA["i_tr_ma"] + h_th / hop * (SOA["i_op_ma"] - SOA["i_tr_ma"])
    assert 18.0 < i_th < 19.0    # the device this test pins

    def p_settled(i_ma: float) -> float:
        v_lo = SOA["Von"] + SOA["Rs"] * 5e-3          # start below threshold
        v_hi = SOA["Von"] + SOA["Rs"] * i_ma * 1e-3
        ckt = ls.Circuit("fp laser")
        ckt.raw(".options reltol=1e-9 abstol=1e-14 vntol=1e-10")
        ckt.raw(f"Vb an 0 PULSE({v_lo} {v_hi} 0.2n 10p 10p 10n 20n)")
        ckt.device(ls.va("soa"), "amp",
                   "fi_re", "fi_im", "fo_re", "fo_im",
                   "bi_re", "bi_im", "bo_re", "bo_im",
                   "an", "0", "0", **{**SOA, "p_seed": 1e-9})
        ckt.device(ls.va("mirror"), "m1",
                   "0", "0", "m1lo_re", "m1lo_im",
                   "bo_re", "bo_im", "fi_re", "fi_im", "0", refl=r1)
        ckt.device(ls.va("mirror"), "m2",
                   "fo_re", "fo_im", "bi_re", "bi_im",
                   "0", "0", "out_re", "out_im", "0", refl=r2)
        r = eng.tran(ckt, "2p", "6n")
        p = r["out_re"] ** 2 + r["out_im"] ** 2
        return float(p[r.t > 5e-9].mean())

    def p_analytic(i_ma: float) -> float:
        # reservoir balance with the gain clamped at G_th (r1/r2 are POWER
        # reflectivities, matching the mirror's refl parameter):
        #   h0 - h_th = (G_th - 1)*(P_fi + P_bi)/p_sat,
        #   P_fi + P_bi = P_fi*(1 + r2*G_th),  P_out = (1-r2)*G_th*P_fi
        h0 = hop * (i_ma - SOA["i_tr_ma"]) / (SOA["i_op_ma"] - SOA["i_tr_ma"])
        p_fi = (h0 - h_th) * SOA["p_sat"] / ((g_th - 1) * (1 + r2 * g_th))
        return (1 - r2) * g_th * p_fi

    assert p_settled(15.0) < 1e-5                     # below threshold: dark
    for i_ma in (30.0, 50.0):                         # clamped-gain L-I line
        assert p_settled(i_ma) == pytest.approx(p_analytic(i_ma), rel=0.01)


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


def _rnl_sweep(eng, p_in: float, span_pm: float = 8.0, n: int = 241, **over):
    lam0 = RNL["lambda_res_nm"]
    lams = lam0 + np.linspace(-span_pm, span_pm, n) * 1e-3   # pm -> nm
    t = np.empty(n)
    for i, lam in enumerate(lams):
        ckt = ls.Circuit("ring nl")
        ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
        ckt.raw(f"Vre in_re 0 {np.sqrt(p_in)}", "Vim in_im 0 0")
        ckt.device(ls.va("ring_nl"), "rg",
                   "in_re", "in_im", "out_re", "out_im", "0",
                   lambda_nm=lam, **{**RNL, **over})
        r = eng.op(ckt)
        t[i] = (r["out_re"][0] ** 2 + r["out_im"][0] ** 2) / p_in
    return lams, t


def fwhm_pm(lams: np.ndarray, t: np.ndarray) -> float:
    half = 0.5 * (1 + t.min())
    below = np.where(t < half)[0]
    return (lams[below[-1]] - lams[below[0]]) * 1e3   # nm -> pm


def test_ring_nl_linear_lorentzian(eng):
    # beta_tpa = 0 recovers the exact linear CMT Lorentzian (at Q ~ 1.2e6
    # even 1 uW of input builds enough circulating power for ~0.5% of FCA,
    # so the "linear" pin must actually switch the nonlinearity off)
    inv_ti, inv_te, _ = rnl_rates()
    tau = 1 / (inv_ti + inv_te)
    lams, t = _rnl_sweep(eng, 1e-6, beta_tpa=0.0)
    lam = lams * 1e-9
    delta = 2 * np.pi * C0 * (1 / lam - 1 / (RNL["lambda_res_nm"] * 1e-9))
    t_ref = ((1 - tau * 2 * inv_te) ** 2 + (tau * delta) ** 2) / (1 + (tau * delta) ** 2)
    assert np.abs(t - t_ref).max() < 1e-6
    q_loaded = 2 * np.pi * C0 / (RNL["lambda_res_nm"] * 1e-9) * tau / 2
    # rel = 0.1: the FWHM is ~1.1 pm read off a 0.067 pm sweep grid
    assert RNL["lambda_res_nm"] / (fwhm_pm(lams, t) * 1e-3) == pytest.approx(
        q_loaded, rel=0.1)


def test_ring_nl_q_droop_with_power(eng):
    # nonlinear absorption broadens the line and lifts the dip floor
    lams_lo, t_lo = _rnl_sweep(eng, 1e-6)
    lams_hi, t_hi = _rnl_sweep(eng, 0.5e-3, span_pm=60.0)
    q_lo = RNL["lambda_res_nm"] / (fwhm_pm(lams_lo, t_lo) * 1e-6)
    q_hi = RNL["lambda_res_nm"] / (fwhm_pm(lams_hi, t_hi) * 1e-6)
    assert q_hi < 0.6 * q_lo          # the Q collapse
    assert t_hi.min() > t_lo.min()    # coupling condition walks away from critical


# ===========================================================================
# ring_kerr — intracavity four-wave mixing (the modal chi(3) ring)
# ===========================================================================
RK = dict(lambda_nm=1310.0, lambda_res_nm=1310.0, radius_um=2000.0, n_g=4.0,
          loss_db_m=30.0, kappa2_in=0.035, kappa2_drop=0.035,
          a_eff_um2=0.1, n2_kerr=4.5e-18, d2_hz=0.0)


def rk_rates() -> dict:
    circ = 2 * np.pi * RK["radius_um"] * 1e-6
    v_g = C0 / RK["n_g"]
    t_rt = circ / v_g
    w0 = 2 * np.pi * C0 / (RK["lambda_nm"] * 1e-9)
    inv_i = RK["loss_db_m"] * np.log(10) / 10 * v_g / 2
    inv_e1 = RK["kappa2_in"] / (2 * t_rt)
    inv_e2 = RK["kappa2_drop"] / (2 * t_rt)
    tau = 1 / (inv_i + inv_e1 + inv_e2)
    k2c = 2 * inv_e1
    g_u = (w0 / RK["n_g"]) * RK["n2_kerr"] * v_g / (RK["a_eff_um2"] * 1e-12
                                                    * circ)
    return dict(fsr=1 / t_rt, tau=tau, k2c=k2c, geff=g_u / k2c)


def test_ring_kerr_fwm_idler(eng):
    """Intracavity FWM through the OSDI/ngspice toolchain: pump on mode 0,
    signal one FSR away on mode +1, and the idler emerges in mode -1 with
    the resonant CMT efficiency P_i = (geff*tau*P_p)^2 (k2c*tau)^6 P_s —
    slope 2 in pump power. (examples/ring_fwm.py is the full study.)"""
    r = rk_rates()
    p_s = 10e-6
    dt_out = 0.5e-12
    t_settle, n_win = 12e-9, 4096
    t_win = 12 / r["fsr"]                     # 12 beat periods
    t1 = t_settle + t_win

    def idler(p_p: float) -> float:
        a_s = np.sqrt(p_s)
        ckt = ls.Circuit("ring fwm")
        ckt.raw(".options reltol=1e-10 abstol=1e-15 vntol=1e-12")
        # pump at envelope 0 (mode 0) in series with the signal tone at
        # envelope -FSR (mode +1): E = sqrt(Pp) + sqrt(Ps)e^{-j*2pi*FSR*t}
        ckt.raw(f"Vre1 in_re mre {np.sqrt(p_p)}",
                f"Vre2 mre 0     SIN(0 {a_s} {r['fsr']} 0 0 90)",
                "Vim1 in_im mim 0",
                f"Vim2 mim 0     SIN(0 {-a_s} {r['fsr']} 0 0 0)")
        ckt.device(ls.va("ring_kerr"), "rk",
                   "in_re", "in_im", "thru_re", "thru_im",
                   "drop_re", "drop_im", "0", **RK)
        res = eng.tran(ckt, f"{dt_out*1e12}p", f"{t1*1e9}n")
        tt = t_settle + (t_win / n_win) * np.arange(n_win)
        e = (np.interp(tt, res.t, res["thru_re"])
             + 1j * np.interp(tt, res.t, res["thru_im"]))
        a = np.fft.fft(e) / n_win
        f = np.fft.fftfreq(n_win, d=t_win / n_win)
        return float(np.abs(a[np.argmin(np.abs(f - r["fsr"]))]) ** 2)

    for p_p in (50e-6, 100e-6):
        p_i_th = (r["geff"] * r["tau"] * p_p) ** 2 \
            * (r["k2c"] * r["tau"]) ** 6 * p_s
        assert idler(p_p) == pytest.approx(p_i_th, rel=5e-2)
    assert idler(100e-6) / idler(50e-6) == pytest.approx(4.0, rel=5e-2)
