"""Link-level analysis: Q factor, BER, eye folding, sensitivity extraction."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["QStats", "q_ber", "best_sampling", "eye_fold", "sensitivity"]


@dataclass
class QStats:
    mu0: float
    mu1: float
    sigma0: float
    sigma1: float
    q: float
    ber: float
    n0: int
    n1: int

    @property
    def eye(self) -> float:
        return abs(self.mu1 - self.mu0)


def q_ber(samples: np.ndarray, bits: np.ndarray) -> QStats:
    """Gaussian Q-factor BER estimate from bit-centre samples.

    `bits` is the transmitted pattern; the trailing len(samples) bits are
    used (so settling bits can be skipped from `samples` independently).
    Sign-agnostic: works for inverting receivers.
    """
    used = np.asarray(bits)[-len(samples):]
    v1 = samples[used == 1]
    v0 = samples[used == 0]
    if len(v0) == 0 or len(v1) == 0:
        raise ValueError("need both 0 and 1 bits in the sampled window")
    mu1, s1 = float(v1.mean()), float(v1.std() + 1e-15)
    mu0, s0 = float(v0.mean()), float(v0.std() + 1e-15)
    q = abs(mu1 - mu0) / (s0 + s1)
    ber = 0.5 * math.erfc(q / math.sqrt(2.0))
    return QStats(mu0, mu1, s0, s1, q, ber, len(v0), len(v1))


def best_sampling(
    t: np.ndarray,
    v: np.ndarray,
    bits: np.ndarray,
    t_bit: float,
    skip: int = 0,
    nphases: int = 32,
) -> tuple[float, np.ndarray, QStats]:
    """CDR-style sampling-phase recovery: scan the sampling comb across one
    unit interval and keep the phase that maximises Q.

    A real link has group delay (laser response, driver RC, PD/load pole),
    so sampling blindly at transmit-clock bit centres lands on edges and
    reads as catastrophic "noise". Returns (offset, samples, stats) at the
    best phase. Make sure the waveform extends ~one bit past the last bit
    centre so late phases stay in-range.
    """
    from .signals import sample_centers

    nbits = len(bits)
    best: tuple[float, np.ndarray, QStats] | None = None
    for off in np.linspace(0.0, t_bit, nphases, endpoint=False):
        samples = sample_centers(t, v, t_bit, nbits, skip=skip, t0=off)
        stats = q_ber(samples, bits)
        if best is None or stats.q > best[2].q:
            best = (float(off), samples, stats)
    return best


def eye_fold(
    t: np.ndarray,
    v: np.ndarray,
    ui: float,
    folds: int = 2,
    t_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a waveform modulo `folds` unit intervals for an eye diagram.

    Returns (phase, value) where phase is in seconds within [0, folds*ui).
    """
    mask = t >= t_start
    tt, vv = t[mask], v[mask]
    if len(tt) == 0:
        raise ValueError("t_start is beyond the end of the waveform")
    phase = (tt - tt[0]) % (folds * ui)
    return phase, vv


def sensitivity(
    p_dbm: np.ndarray, ber: np.ndarray, target: float
) -> float | None:
    """Interpolate received power (dBm) at a target BER, or None if the
    sweep never crosses it."""
    ber = np.clip(np.asarray(ber, dtype=float), 1e-300, 1.0)
    order = np.argsort(ber)
    x = np.log10(ber[order])
    y = np.asarray(p_dbm)[order]
    lt = math.log10(target)
    if not (x.min() <= lt <= x.max()):
        return None
    return float(np.interp(lt, x, y))
