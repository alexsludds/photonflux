"""Split-step Fourier propagation of the scalar nonlinear Schrodinger equation.

The webapp's ``fiber_cd`` component (``lti.py::build``) is a *linear* chromatic
dispersion all-pass. Real km-scale fibre also self-phase-modulates: the Kerr
effect ``n2`` makes the refractive index intensity dependent, so the pulse
shapes its own spectrum (SPM), tones cross-modulate and mix (XPM/FWM), and — in
the anomalous-dispersion regime — dispersion and nonlinearity can balance into a
soliton. The standard tool for this in VPI / Lumerical INTERCONNECT is the
split-step Fourier method (SSFM), implemented here.

Physics (retarded frame, coherent baseband envelope ``A(z, T)``, |A|^2 = power
[W]) — Agrawal, *Nonlinear Fiber Optics*, eq. 2.3.44::

    dA/dz = -(alpha/2) A                          (attenuation)
            + i (beta2/2) d^2A/dT^2               (GVD)
            - i (beta3/6) d^3A/dT^3               (third-order dispersion)
            - i gamma |A|^2 A                     (Kerr / SPM)

Sign convention matches the rest of the repo. Envelopes use the physics
``e^{+j w t}`` frame (a tone at envelope frequency ``+f`` is ``e^{+j2pi f t}``,
as in ``examples/wg_fwm.py``), so in the Fourier domain ``d/dT -> j w`` and the
linear operator is

    L(w) = -alpha/2 - j (beta2/2 w^2 + beta3/6 w^3).

Over the full length this is exactly ``fiber_cd``'s target all-pass
``amp * exp(-j theta(w))`` with ``theta = beta2 L/2 w^2 + beta3 L/6 w^3`` and
``amp = exp(-alpha L/2)`` — so the ``gamma = 0`` limit reproduces the linear
fibre (pinned in ``tests/test_ssfm.py``). The Kerr term carries the same
``exp(-j gamma |A|^2 dz)`` phase-lag sign as ``models/optical_field/waveguide_nl.va``.

The symmetric (Strang) split advances one step ``dz`` as a half linear step, a
full nonlinear phase, and a half linear step; the local error is O(dz^3) and the
global error O(dz^2). ``propagate`` defaults to an adaptive step that caps the
per-step nonlinear phase (Sinkin et al., JLT 21, 61 (2003)); ``gamma = 0`` short
-circuits to a single exact FFT multiply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_C0 = 299792458.0


# ---------------------------------------------------------------------------
# physical parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FiberParams:
    """SI nonlinear-fibre coefficients (all per metre / watt / second)."""

    length: float      # m
    alpha: float       # power attenuation [1/m]
    beta2: float       # GVD [s^2/m]
    beta3: float       # TOD [s^3/m]
    gamma: float       # nonlinear parameter [1/(W*m)]


def fiber_params(
    *,
    length_km: float = 10.0,
    lambda_nm: float = 1550.0,
    atten_db_km: float = 0.2,
    gamma_per_W_km: float = 1.3,
    D_ps: float = 17.0,
    S_ps: float = 0.0,
    lambda0_nm: float = 0.0,
) -> FiberParams:
    """Webapp-facing fibre parameters -> SI :class:`FiberParams`.

    ``D_ps`` [ps/nm/km], ``S_ps`` [ps/nm^2/km] and the ``lambda0_nm`` G.652
    zero-dispersion profile are converted to ``beta2``/``beta3`` with the *same*
    formulas as ``lti.py::build``'s ``fiber`` path, so the ``gamma = 0`` limit
    of :func:`propagate` matches ``fiber_cd`` to the vector-fit error. ``alpha``
    is the power attenuation [1/m]; ``gamma_per_W_km`` [1/(W*km)] is the Kerr
    parameter (silicon-fibre C-band ~1.3 /W/km).
    """
    L = float(length_km) * 1e3
    lam = float(lambda_nm) * 1e-9
    if lambda0_nm > 0.0:
        # G.652 profile around the zero-dispersion wavelength: S_ps is read as
        # S0 (the slope AT lambda0); D follows, exactly as fiber_cd does it.
        S0 = S_ps if S_ps > 0.0 else 0.092
        lnm, l0 = lam * 1e9, float(lambda0_nm)
        D_ps = S0 / 4.0 * (lnm - l0 ** 4 / lnm ** 3)
        S_ps = S0 / 4.0 * (1.0 + 3.0 * l0 ** 4 / lnm ** 4)
    D_si = D_ps * 1e-6         # ps/nm/km  -> s/m^2
    S_si = S_ps * 1e3          # ps/nm^2/km -> s/m^3
    beta2 = -D_si * lam ** 2 / (2.0 * np.pi * _C0)
    beta3 = lam ** 3 * (S_si * lam + 2.0 * D_si) / (2.0 * np.pi * _C0) ** 2
    alpha = float(atten_db_km) * np.log(10.0) / 10.0 / 1e3       # 1/m
    gamma = float(gamma_per_W_km) / 1e3                          # 1/(W*m)
    return FiberParams(length=L, alpha=alpha, beta2=beta2, beta3=beta3,
                       gamma=gamma)


# ---------------------------------------------------------------------------
# split-step Fourier propagation
# ---------------------------------------------------------------------------
def linear_operator(omega: np.ndarray, p: FiberParams) -> np.ndarray:
    """Per-metre linear operator ``L(w)`` (attenuation + dispersion)."""
    return (-0.5 * p.alpha
            - 1j * (0.5 * p.beta2 * omega ** 2 + p.beta3 * omega ** 3 / 6.0))


def propagate(
    A0: np.ndarray,
    dt: float,
    p: FiberParams,
    *,
    n_steps: int | None = None,
    max_phase: float = 5e-3,
    max_step: float | None = None,
    return_trace: bool = False,
):
    """Propagate the envelope ``A0`` (sampled at spacing ``dt``) over the fibre.

    Symmetric split-step Fourier. With ``n_steps`` the grid is uniform in ``z``;
    otherwise the step adapts so the peak nonlinear phase ``gamma*max|A|^2*dz``
    stays under ``max_phase`` (and, if given, ``dz <= max_step``). ``gamma = 0``
    is exact in one FFT multiply. Returns the output field ``A(L)``; with
    ``return_trace=True`` returns ``(A_L, z_grid, |A(z)|^2 peak-power trace)``.
    """
    A = np.asarray(A0, dtype=complex).copy()
    n = A.size
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    L = p.length
    Lop = linear_operator(omega, p)

    # pure-linear fibre: one exact spectral multiply, no splitting error
    if p.gamma == 0.0 or not np.any(A):
        out = np.fft.ifft(np.fft.fft(A) * np.exp(Lop * L))
        if return_trace:
            return out, np.array([0.0, L]), np.array([np.abs(A0).max() ** 2,
                                                      np.abs(out).max() ** 2])
        return out

    z = 0.0
    zs = [0.0]
    peaks = [float(np.abs(A).max() ** 2)]
    # cache the half-step propagator for the last dz — a hit every step in the
    # uniform (n_steps) grid, a cheap recompute when the adaptive step changes.
    dz_last = -1.0
    expL_half = None

    while z < L - 1e-15 * L:
        pmax = float(np.abs(A).max() ** 2)
        if n_steps is not None:
            dz = L / n_steps
        else:
            dz = max_phase / (p.gamma * pmax) if pmax > 0 else L
            if max_step is not None:
                dz = min(dz, max_step)
        dz = min(dz, L - z)
        if dz != dz_last:
            expL_half = np.exp(Lop * (dz / 2.0))
            dz_last = dz

        # half linear -> full nonlinear -> half linear (Strang)
        A = np.fft.ifft(np.fft.fft(A) * expL_half)
        A *= np.exp(-1j * p.gamma * np.abs(A) ** 2 * dz)
        A = np.fft.ifft(np.fft.fft(A) * expL_half)

        z += dz
        zs.append(z)
        peaks.append(float(np.abs(A).max() ** 2))

    if return_trace:
        return A, np.asarray(zs), np.asarray(peaks)
    return A


# ---------------------------------------------------------------------------
# analytic references (used by the tests / example self-checks)
# ---------------------------------------------------------------------------
def soliton_field(t: np.ndarray, t0: float, p: FiberParams, order: int = 1) -> np.ndarray:
    """Fundamental (N=1) or higher-order soliton launch field.

    ``A(0,T) = N * sqrt(|beta2| / (gamma T0^2)) * sech(T/T0)`` — the peak power
    ``P0 = |beta2|/(gamma T0^2)`` makes the nonlinear length equal the
    dispersion length ``LD = T0^2/|beta2|``, the fundamental-soliton balance.
    Requires anomalous dispersion (``beta2 < 0``) and ``gamma > 0``.
    """
    p0 = abs(p.beta2) / (p.gamma * t0 ** 2)
    return order * np.sqrt(p0) / np.cosh(t / t0)


def soliton_period(t0: float, p: FiberParams) -> float:
    """Soliton period ``z0 = (pi/2) * T0^2 / |beta2|`` [m]."""
    return 0.5 * np.pi * t0 ** 2 / abs(p.beta2)
