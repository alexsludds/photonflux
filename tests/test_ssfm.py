"""Split-step Fourier nonlinear-fibre engine (``webapp/ssfm.py``), pinned against
textbook analytics — the ALE-71 acceptance criteria.

These are pure-numpy checks of the propagator itself (no circulax). The
in-transient ``fiber_nl`` webapp component that shares this physics is exercised
by ``tests/test_fiber_nl.py`` (steady state) and ``examples/fiber_nl_ssfm.py``
(transient FWM), matching the repo split of DC tests vs transient examples.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import ssfm  # noqa: E402
import vf  # noqa: E402
import lti  # noqa: E402


# ===========================================================================
# N = 1 soliton: shape-preserving over >= 3 soliton periods
# ===========================================================================
def test_fundamental_soliton_shape_preserving():
    # anomalous dispersion + self-focusing Kerr, P0 = |beta2|/(gamma T0^2)
    p = ssfm.fiber_params(length_km=1.0, D_ps=17.0, atten_db_km=0.0,
                          gamma_per_W_km=1.3)
    assert p.beta2 < 0.0                       # anomalous — solitons need it
    t0 = 20e-12
    n = 4096
    dt = 40 * t0 / n
    t = (np.arange(n) - n / 2) * dt
    a0 = ssfm.soliton_field(t, t0, p)

    p3 = replace(p, length=3.0 * ssfm.soliton_period(t0, p))
    a_out = ssfm.propagate(a0, dt, p3, max_phase=2e-3)
    shape_err = np.max(np.abs(np.abs(a_out) - np.abs(a0))) / np.abs(a0).max()
    assert shape_err < 1e-3                    # fundamental soliton is invariant
    # energy is conserved (lossless)
    assert np.sum(np.abs(a_out) ** 2) == pytest.approx(
        np.sum(np.abs(a0) ** 2), rel=1e-6)


def test_soliton_disperses_without_kerr():
    # sanity: the SAME launch pulse with gamma -> 0 broadens (no balance)
    p = ssfm.fiber_params(length_km=1.0, D_ps=17.0, atten_db_km=0.0,
                          gamma_per_W_km=1.3)
    t0 = 20e-12
    n = 4096
    dt = 40 * t0 / n
    t = (np.arange(n) - n / 2) * dt
    a0 = ssfm.soliton_field(t, t0, p)
    p_lin = replace(p, length=3.0 * ssfm.soliton_period(t0, p), gamma=0.0)
    a_out = ssfm.propagate(a0, dt, p_lin)
    assert np.abs(a_out).max() < 0.7 * np.abs(a0).max()   # dispersed, lower peak


# ===========================================================================
# SPM: number of spectral peaks vs peak nonlinear phase (Agrawal, Fig. 4.2)
# ===========================================================================
def _spm_peak_count(phi_max: float) -> int:
    p = ssfm.fiber_params(length_km=1.0, D_ps=0.0, atten_db_km=0.0,
                          gamma_per_W_km=1.3)
    p = replace(p, beta2=0.0, beta3=0.0)       # pure SPM, no dispersion
    n = 8192
    t0 = 40e-12
    dt = 40 * t0 / n
    t = (np.arange(n) - n / 2) * dt
    p0 = phi_max / (p.gamma * p.length)
    a0 = np.sqrt(p0) * np.exp(-0.5 * (t / t0) ** 2)   # Gaussian
    a_out = ssfm.propagate(a0, dt, p, max_phase=1e-3)
    s = np.abs(np.fft.fftshift(np.fft.fft(a_out))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, dt))
    band = (f > -3e10) & (f < 3e10)
    sb = s[band]
    return int(((sb[1:-1] > sb[:-2]) & (sb[1:-1] > sb[2:])
                & (sb[1:-1] > 1e-4 * sb.max())).sum())


def test_spm_spectral_peak_count():
    # M peaks when phi_max ~ (M - 1/2) pi
    for m in (1, 2, 3, 4):
        phi = (m - 0.5) * np.pi
        assert _spm_peak_count(phi) == m


# ===========================================================================
# linear limit (gamma = 0) reproduces the fibre dispersion all-pass / fiber_cd
# ===========================================================================
def test_linear_limit_matches_exact_dispersion():
    # SSFM with gamma = 0 is a single exact spectral multiply -> the all-pass
    # amp * exp(-j theta(w)) that fiber_cd fits, to machine precision.
    p = ssfm.fiber_params(length_km=10.0, D_ps=17.0, lambda_nm=1550.0,
                          atten_db_km=0.2, gamma_per_W_km=0.0)
    n = 8192
    dt = 1.0 / 240e9
    rng = np.random.default_rng(0)
    a0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a_out = ssfm.propagate(a0, dt, p)
    f = np.fft.fftfreq(n, dt)
    w = 2.0 * np.pi * f
    h = np.fft.fft(a_out) / np.fft.fft(a0)
    amp = 10.0 ** (-0.2 * 10.0 / 20.0)
    theta = 0.5 * p.beta2 * p.length * w ** 2 + p.beta3 * p.length * w ** 3 / 6.0
    h_exact = amp * np.exp(-1j * theta)
    band = np.abs(f) < 60e9
    assert np.max(np.abs(h[band] - h_exact[band])) < 1e-9


def test_linear_limit_matches_fiber_cd_vectorfit():
    # ... and matches the actual fiber_cd vector-fit model to < 1% in-band
    # (fiber_cd adds a bulk transit latency Td; restore it before comparing).
    settings = dict(length_km=10.0, D_ps=17.0, S_ps=0.0, lambda_nm=1550.0,
                    atten_db_km=0.2, fit_bw=60e9, n_poles=28)
    _, payload = lti.build("fiber", settings, [])
    poles, res, d = payload["cplx"]

    p = ssfm.fiber_params(length_km=10.0, D_ps=17.0, lambda_nm=1550.0,
                          atten_db_km=0.2, gamma_per_W_km=0.0)
    n = 8192
    dt = 1.0 / 240e9
    rng = np.random.default_rng(1)
    a0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a_out = ssfm.propagate(a0, dt, p)
    f = np.fft.fftfreq(n, dt)
    w = 2.0 * np.pi * f
    h_ssfm = np.fft.fft(a_out) / np.fft.fft(a0)

    b2l, b3l = p.beta2 * p.length, p.beta3 * p.length
    w_max = 2.0 * np.pi * 60e9
    td = 1.5 * (abs(b2l) * w_max + 0.5 * abs(b3l) * w_max ** 2)   # fiber_cd Td
    h_ssfm = h_ssfm * np.exp(-1j * w * td)
    h_vf = vf.eval_fit(poles, res, d, f)
    band = np.abs(f) < 55e9
    rel = np.max(np.abs(h_ssfm[band] - h_vf[band])) / np.abs(h_vf[band]).mean()
    assert rel < 0.01


# ===========================================================================
# four-wave mixing: idler efficiency vs phase mismatch follows sinc^2
# ===========================================================================
def test_fwm_efficiency_vs_phase_mismatch():
    # degenerate FWM: strong pump at envelope 0, weak signal at +Omega, idler
    # at -Omega. Small-signal (gamma Pp L << 1): eta ~ eta0 sinc^2(dbeta L / 2),
    # dbeta = beta2 Omega^2. Sweep beta2 and check the sinc.
    p0 = ssfm.fiber_params(length_km=2.0, D_ps=0.0, atten_db_km=0.0,
                           gamma_per_W_km=1.3)
    pp, ps = 5e-3, 1e-7
    assert p0.gamma * pp * p0.length < 0.02          # small-signal regime
    omega = 2.0 * np.pi * 50e9
    n = 2048
    dt = 60 * (2.0 * np.pi / omega) / n              # 60 exact beat periods
    t = np.arange(n) * dt
    a0 = np.sqrt(pp) + np.sqrt(ps) * np.exp(1j * omega * t)
    f = np.fft.fftfreq(n, dt)
    i_idler = int(np.argmin(np.abs(f + 50e9)))

    def eff(beta2: float) -> float:
        p = replace(p0, beta2=beta2, beta3=0.0)
        a_out = ssfm.propagate(a0, dt, p, max_phase=1e-3)
        return np.abs(np.fft.fft(a_out)[i_idler] / n) ** 2 / ps

    lam, c0 = 1550e-9, 299792458.0
    eta0 = eff(0.0)
    for d_ps in (2.0, 4.0, 8.0, 12.0):
        beta2 = -(d_ps * 1e-6) * lam ** 2 / (2.0 * np.pi * c0)
        dbeta = beta2 * omega ** 2
        sinc2 = np.sinc(dbeta * p0.length / 2.0 / np.pi) ** 2
        assert eff(beta2) / eta0 == pytest.approx(sinc2, abs=0.02)


def test_fwm_needs_kerr():
    # no idler without the Kerr term
    p0 = ssfm.fiber_params(length_km=2.0, D_ps=0.0, atten_db_km=0.0,
                           gamma_per_W_km=1.3)
    pp, ps = 5e-3, 1e-7
    omega = 2.0 * np.pi * 50e9
    n = 2048
    dt = 60 * (2.0 * np.pi / omega) / n
    t = np.arange(n) * dt
    a0 = np.sqrt(pp) + np.sqrt(ps) * np.exp(1j * omega * t)
    f = np.fft.fftfreq(n, dt)
    i_idler = int(np.argmin(np.abs(f + 50e9)))
    a_lin = ssfm.propagate(a0, dt, replace(p0, gamma=0.0))
    assert np.abs(np.fft.fft(a_lin)[i_idler] / n) ** 2 / ps < 1e-15


# ===========================================================================
# parameter conversions match lti.py's fiber path (so the linear limit lines up)
# ===========================================================================
def test_fiber_params_match_lti_beta():
    p = ssfm.fiber_params(length_km=10.0, D_ps=17.0, S_ps=0.06,
                          lambda_nm=1550.0, atten_db_km=0.2)
    d_ps, s_ps = lti._resolve_dispersion(1550e-9, 17.0, 0.06, 0.0)
    b2, b3 = lti._beta23(1550e-9, d_ps, s_ps)
    assert p.beta2 == pytest.approx(b2, rel=1e-12)
    assert p.beta3 == pytest.approx(b3, rel=1e-12)
    assert p.alpha == pytest.approx(0.2 * np.log(10) / 10 / 1e3, rel=1e-12)


def test_fiber_params_lambda0_profile():
    # near the zero-dispersion wavelength D ~ 0 and beta3 dominates
    p = ssfm.fiber_params(length_km=10.0, lambda_nm=1310.0, lambda0_nm=1310.0,
                          S_ps=0.092, atten_db_km=0.3)
    assert abs(p.beta2) < 1e-27         # essentially zero-dispersion
    assert abs(p.beta3) > 0.0
