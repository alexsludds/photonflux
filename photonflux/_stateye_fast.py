"""Vectorized drop-ins for stateye's vertical-bathtub builders.

``stateye.measurements.bathtub.generate_vertical_bathtub_{nrz,pam4}`` fit a
dual-Dirac Q-scale line to every histogram column separately, in Python:
``nx`` columns x data patterns x thresholds calls of ``q_scale_fit``, each a
cumsum, an ``erfinv`` and an ``np.polyfit``. A 1024-column PAM-4 eye makes
~300k such calls, 30 s of a TDECQ measurement.

The functions here do the same arithmetic for all columns of a pattern at
once. Same signatures, same in-place outputs; the per-column line fit is
the closed-form least-squares solution (what ``np.polyfit(deg=1)`` solves),
so results agree with stateye's to rounding. :func:`install` swaps them into
``stateye.eye`` (which imported the originals by name).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.special import erfinv


def _bathtub_rows(input_scale: np.ndarray, H: np.ndarray, rho_t: float,
                  mean_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """``q_scale_fit`` and the raw normalized CDF for every row of ``H``
    (columns x y-bins, C-contiguous) at once -> ``(fitted BER, raw CDF)``.

    Histograms are sparse (~1 % of bins populated), so the Q-scale transform
    and the line fit run on the populated bins only."""
    S = H.shape[0]
    cdf = np.cumsum(H, axis=1)
    # the original's max(cumsum over populated bins) is the row total (H >= 0)
    tot = cdf[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = cdf * (rho_t / np.max(cdf, axis=1, keepdims=True))
    rows, cols = np.nonzero(H)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = cdf[rows, cols] * (rho_t / tot[rows])
        qv = np.sqrt(2) * erfinv(1 - c / (0.5 * rho_t))
    ok = np.isfinite(qv)
    r, q, sc = rows[ok], qv[ok], input_scale[cols[ok]]
    n = np.bincount(r, minlength=S)

    out = np.empty(H.shape, dtype=float)
    fit = n > 1
    if fit.any():
        # least-squares line sc = sigma * q + mean per row (np.polyfit deg 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            qm = np.bincount(r, q, S) / n
            sm = np.bincount(r, sc, S) / n
            dq = q - qm[r]
            sigma = (np.bincount(r, dq * (sc - sm[r]), S)
                     / np.bincount(r, dq * dq, S))
        mean = sm - sigma * qm
        sig, mu = sigma[fit, None], mean[fit, None]
        dx = abs(input_scale[1] - input_scale[0])
        gaussian = np.exp(-((input_scale[None, :] - (mu + mean_shift)) ** 2)
                          / (2 * sig ** 2))
        gaussian /= np.abs(sig) * np.sqrt(2 * np.pi)
        out[fit] = rho_t * np.cumsum(gaussian, axis=1) * dx
    for row in np.nonzero(~fit)[0]:
        # No usable Q-scale points: no data -> no errors; else a sharp cutoff
        # at the first populated bin (stateye's fallback, per column).
        out[row] = rho_t
        first = np.nonzero(H[row])[0]
        if first.size:
            out[row, :first[0]] = 0.0
    return out, raw


def generate_vertical_bathtub_nrz(hist, bathtub, raw_bathtub, pattern_indices,
                                  pattern_counts, data_values, sensitivity,
                                  y_scale):
    bathtub[:] = 0.0
    raw_bathtub[:] = 0.0
    roll_idx = int(round((sensitivity / 2) / (y_scale[1] - y_scale[0])))
    total = np.sum(pattern_counts)
    for i, idx in enumerate(pattern_indices):
        H = np.ascontiguousarray(hist[:, :, idx])
        bit_value = data_values[idx]
        rho = pattern_counts[i] / total
        if bit_value == 0:
            fit, raw = _bathtub_rows(y_scale[::-1],
                                     np.ascontiguousarray(H[:, ::-1]), rho,
                                     sensitivity / 2)
            bathtub[:, :, 0] += fit[:, ::-1]
            raw_bathtub[:, :, 0] += np.roll(raw[:, ::-1], roll_idx, axis=1)
        elif bit_value == 1:
            fit, raw = _bathtub_rows(y_scale, H, rho, -sensitivity / 2)
            bathtub[:, :, 0] += fit
            raw_bathtub[:, :, 0] += np.roll(raw, -roll_idx, axis=1)
        else:
            raise NotImplementedError("ERROR: `bit_value` must be either 0 or 1.")


def _pam4_level_fits(hist, idx, bit_value, rho, y_scale, n_eyes):
    """One pattern's {direction: (fit, raw)}: a level integrates up through
    the eyes below it and down through those above, so it needs at most two
    fits whichever eye they serve."""
    if bit_value not in (0, 1, 2, 3):
        raise NotImplementedError(
            "ERROR: `bit_value` must be either 0, 1, 2, or 3.")
    # one contiguous copy: hist[:, :, idx] is a strided view of the whole
    # (columns x bins x patterns) cube
    H = np.ascontiguousarray(hist[:, :, idx])
    fits = {}
    for up in {bit_value > e for e in range(n_eyes)}:
        if up:
            fits[up] = _bathtub_rows(y_scale, H, rho, 0)
        else:
            fit, raw = _bathtub_rows(y_scale[::-1],
                                     np.ascontiguousarray(H[:, ::-1]), rho, 0)
            fits[up] = (fit[:, ::-1], raw[:, ::-1])
    return fits


def generate_vertical_bathtub_pam4(hist, bathtub, raw_bathtub,
                                   pattern_indices, pattern_counts,
                                   data_values, y_scale):
    from concurrent.futures import ThreadPoolExecutor

    n_eyes = bathtub.shape[2]
    # accumulate per eye in contiguous buffers; bathtub[:, :, e] is strided
    acc = np.zeros((n_eyes,) + bathtub.shape[:2])
    acc_raw = np.zeros_like(acc)
    total = np.sum(pattern_counts)
    jobs = [(hist, idx, data_values[idx], pattern_counts[i] / total, y_scale,
             n_eyes) for i, idx in enumerate(pattern_indices)]
    # Patterns are independent and numpy releases the GIL on these array
    # ops, so fit a few at a time in threads; each eye still sums the
    # patterns in stateye's order, so the result is unchanged.
    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(jobs), workers):
            chunk = list(pool.map(lambda j: _pam4_level_fits(*j),
                                  jobs[start:start + workers]))
            for (_h, _i, bit_value, *_r), fits in zip(
                    jobs[start:start + workers], chunk):
                for eye_idx in range(n_eyes):
                    fit, raw = fits[bit_value > eye_idx]
                    acc[eye_idx] += fit
                    acc_raw[eye_idx] += raw
    bathtub[:] = np.moveaxis(acc, 0, 2)
    raw_bathtub[:] = np.moveaxis(acc_raw, 0, 2)


def install(stateye) -> None:
    """Point ``stateye.eye`` at the vectorized builders (idempotent).

    Only replaces stateye's own originals, so a stateye that already ships
    something else is left alone."""
    eye = stateye.eye
    for name, fast in (("generate_vertical_bathtub_nrz",
                        generate_vertical_bathtub_nrz),
                       ("generate_vertical_bathtub_pam4",
                        generate_vertical_bathtub_pam4)):
        cur = getattr(eye, name, None)
        if getattr(cur, "__module__", "") == "stateye.measurements.bathtub":
            setattr(eye, name, fast)
