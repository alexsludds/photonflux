"""Compact vector fitting + state-space realisation for LTI components.

``vector_fit`` implements the Gustavsen/Semlyen relocation iteration: fit
H(s) ~ d + sum r_k/(s - p_k) by iteratively solving the linear
sigma-approximation problem and taking the zeros of sigma as the next pole
set. The solver always works with free complex poles; for real (electrical)
systems the sample set is extended over negative frequencies with the
conjugated response, which forces conjugate-symmetric results through the
data itself, and the pole set is exactly symmetrised afterwards
(``realify`` then gives a real A, B, C, D). For optical baseband envelopes
(``complex_pairs=False``) H(-f) != conj(H(f)) is perfectly legal and the
poles stay free.

Plain numpy least squares throughout; band-limited fits of smooth channel
and dispersion responses converge in a handful of iterations.
"""
from __future__ import annotations

import numpy as np


def _initial_poles(w: np.ndarray, n: int) -> np.ndarray:
    wmax = max(np.abs(w).max(), 1.0)
    lo = max(np.abs(w[np.abs(w) > 0]).min() if (np.abs(w) > 0).any() else 1.0,
             wmax * 1e-4)
    if w.min() < 0:                       # signed band: spread linearly
        ws = np.linspace(w.min(), w.max(), n)
        return np.asarray([-0.05 * wmax + 1j * x for x in ws], complex)
    beta = np.logspace(np.log10(lo), np.log10(wmax), max(n // 2, 1))
    poles = []
    for b in beta:
        poles += [(-0.01 + 1j) * b, (-0.01 - 1j) * b]
    return np.asarray(poles[:n] if len(poles) >= n
                      else poles + [-wmax] * (n - len(poles)), complex)


def _fit_fixed_poles(s, H, poles, wt):
    cols = [1.0 / (s[:, None] - poles[None, :]), np.ones((len(s), 1))]
    M = np.hstack(cols) * wt[:, None]
    x, *_ = np.linalg.lstsq(M, H * wt, rcond=None)
    Hfit = (M @ x) / wt
    err = float(np.sqrt(np.mean(np.abs(Hfit - H) ** 2))
                / max(np.abs(H).max(), 1e-30))
    return x[:-1], x[-1], err


def _symmetrise(poles: np.ndarray) -> np.ndarray:
    """Snap a nearly-conjugate-symmetric pole set to exact symmetry."""
    out = []
    used = np.zeros(len(poles), bool)
    for i, p in enumerate(poles):
        if used[i]:
            continue
        used[i] = True
        if abs(p.imag) < 1e-6 * max(abs(p.real), 1.0):
            out.append(complex(p.real))
            continue
        d = np.abs(poles - np.conj(p)) + used * 1e30
        j = int(np.argmin(d))
        if d[j] < 0.2 * abs(p):
            used[j] = True
            pp = 0.5 * (p + np.conj(poles[j]))
            out += [pp, np.conj(pp)]
        else:                              # unpaired: make it real
            out.append(complex(p.real))
    return np.asarray(out, complex)


def vector_fit(f: np.ndarray, H: np.ndarray, n_poles: int = 8,
               n_iter: int = 14, complex_pairs: bool = True,
               weight: np.ndarray | None = None):
    """Fit H(j*2*pi*f) -> (poles, residues, d, rel_rms_err).

    ``complex_pairs=True``: H is a real system's response sampled on f >= 0;
    the result has conjugate-symmetric poles/residues and a real d.
    ``complex_pairs=False``: f may span negative values (baseband optical);
    poles/residues/d are free complex.
    """
    f = np.asarray(f, float)
    H = np.asarray(H, complex)
    wt0 = np.ones(len(f)) if weight is None else np.asarray(weight, float)
    if complex_pairs:
        f_all = np.concatenate([-f[::-1], f])
        H_all = np.concatenate([np.conj(H)[::-1], H])
        wt = np.concatenate([wt0[::-1], wt0])
    else:
        f_all, H_all, wt = f, H, wt0
    # normalise frequency so basis columns are O(1) — raw 1/(s - p) columns
    # sit ~1e-11 next to the constant column and wreck the LS conditioning
    w0 = max(2.0 * np.pi * np.abs(f_all).max(), 1.0)
    s = 2j * np.pi * f_all / w0
    poles = _initial_poles(2.0 * np.pi * f_all / w0, n_poles)

    for _ in range(n_iter):
        n = len(poles)
        M = np.hstack([
            1.0 / (s[:, None] - poles[None, :]),
            np.ones((len(s), 1)),
            -(H_all[:, None] / (s[:, None] - poles[None, :])),
        ]) * wt[:, None]
        x, *_ = np.linalg.lstsq(M, H_all * wt, rcond=None)
        sig = x[n + 1:]
        A = np.diag(poles) - np.outer(np.ones(n), sig)
        new_poles = np.linalg.eigvals(A)
        new_poles = np.where(new_poles.real > 0,
                             -new_poles.real + 1j * new_poles.imag,
                             new_poles)
        poles = _symmetrise(new_poles) if complex_pairs else new_poles

    res, d, err = _fit_fixed_poles(s, H_all, poles, wt)
    poles = poles * w0          # undo the frequency normalisation
    res = res * w0
    if complex_pairs:
        # exact conjugate residues + real d
        res = res.copy()
        used = np.zeros(len(poles), bool)
        for i, p in enumerate(poles):
            if used[i]:
                continue
            used[i] = True
            if abs(p.imag) < 1e-6 * max(abs(p.real), 1.0):
                res[i] = res[i].real
                continue
            j = int(np.argmin(np.abs(poles - np.conj(p)) + used * 1e30))
            used[j] = True
            r = 0.5 * (res[i] + np.conj(res[j]))
            res[i], res[j] = r, np.conj(r)
        d = complex(d.real)
    return np.asarray(poles), np.asarray(res, complex), complex(d), err


def realify(poles: np.ndarray, res: np.ndarray, d: complex):
    """Conjugate-symmetric pole/residue set -> real (A, B, C, D)."""
    A_blocks, B_rows, C_cols = [], [], []
    used = np.zeros(len(poles), bool)
    for i, p in enumerate(poles):
        if used[i]:
            continue
        used[i] = True
        if abs(p.imag) < 1e-6 * max(abs(p.real), 1.0):
            A_blocks.append(np.array([[p.real]]))
            B_rows.append([1.0])
            C_cols.append([res[i].real])
            continue
        j = int(np.argmin(np.abs(poles - np.conj(p)) + used * 1e30))
        used[j] = True
        a, b = p.real, abs(p.imag)
        rr = res[i] if p.imag > 0 else res[j]
        c, dd = rr.real, rr.imag
        # states [x1, x2]: dx = [[a, -b],[b, a]] x + [1, 0] u
        # y contribution = 2*(c*x1 - dd*x2)  (sum of the conjugate pair)
        A_blocks.append(np.array([[a, -b], [b, a]]))
        B_rows.append([1.0, 0.0])
        C_cols.append([2.0 * c, -2.0 * dd])
    n = sum(blk.shape[0] for blk in A_blocks)
    A = np.zeros((n, n))
    B = np.zeros(n)
    C = np.zeros(n)
    at = 0
    for blk, brow, ccol in zip(A_blocks, B_rows, C_cols):
        m = blk.shape[0]
        A[at:at + m, at:at + m] = blk
        B[at:at + m] = brow
        C[at:at + m] = ccol
        at += m
    return A, B, C, float(np.real(d))


def eval_fit(poles, res, d, f):
    s = 2j * np.pi * np.asarray(f, float)
    H = np.full(len(s), complex(d))
    for p, r in zip(poles, res):
        H += r / (s - p)
    return H


def min_phase(f: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Minimum-phase complex response from a magnitude sampled on a uniform
    positive-frequency grid (discrete Hilbert transform of log|H|)."""
    lm = np.log(np.maximum(mag, 1e-12))
    ext = np.concatenate([lm, lm[::-1]])          # even extension
    n = len(f)
    X = np.fft.fft(ext)
    h = np.zeros(2 * n)
    h[0] = 1.0
    h[1:n] = 2.0
    h[n] = 1.0
    ph = -np.imag(np.fft.ifft(X * h))[:n]
    return mag * np.exp(1j * ph)
