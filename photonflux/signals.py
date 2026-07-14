"""Bit patterns and stimulus generation for link simulations.

Conventions: NRZ levels, bit period `t_bit`, edges of width `t_rise`
centred on bit boundaries (matching how the PWL sources in this repo have
always been built).
"""
from __future__ import annotations

import numpy as np

__all__ = ["prbs", "nrz_pwl", "sample_centers"]

# Fibonacci LFSR second tap for x^n + x^k + 1 generators.
_PRBS_TAPS = {5: 3, 6: 5, 7: 6, 9: 5, 11: 9, 15: 14, 23: 18, 31: 28}


def prbs(order: int = 7, nbits: int | None = None, seed: int = 1) -> np.ndarray:
    """One (or part of one) period of a PRBS-`order` sequence as 0/1 ints."""
    if order not in _PRBS_TAPS:
        raise ValueError(f"unsupported PRBS order {order}; choose {sorted(_PRBS_TAPS)}")
    if seed <= 0 or seed >= (1 << order):
        raise ValueError("seed must be a nonzero state narrower than the register")
    nbits = nbits or (1 << order) - 1
    k = _PRBS_TAPS[order]
    reg = seed
    out = np.empty(nbits, dtype=np.int8)
    for i in range(nbits):
        out[i] = reg & 1
        fb = ((reg >> (order - 1)) ^ (reg >> (k - 1))) & 1
        reg = ((reg << 1) | fb) & ((1 << order) - 1)
    return out


def nrz_pwl(
    bits: np.ndarray,
    t_bit: float,
    t_rise: float,
    v0: float,
    v1: float,
    t0: float = 0.0,
) -> str:
    """Compact PWL() string: one level per bit, `t_rise`-wide edges at
    bit boundaries. Suitable for `Vin in 0 <returned string>`."""
    levels = [v1 if b else v0 for b in bits]
    pts = [(t0, levels[0])]
    for i in range(1, len(levels)):
        if levels[i] != levels[i - 1]:
            tb = t0 + i * t_bit
            pts.append((tb - t_rise / 2, levels[i - 1]))
            pts.append((tb + t_rise / 2, levels[i]))
    pts.append((t0 + len(levels) * t_bit, levels[-1]))
    return "PWL(" + " ".join(f"{t:.6e} {v:.6g}" for t, v in pts) + ")"


def sample_centers(
    t: np.ndarray,
    v: np.ndarray,
    t_bit: float,
    nbits: int,
    skip: int = 0,
    t0: float = 0.0,
) -> np.ndarray:
    """Interpolate `v(t)` at the centre of each bit; drop the first `skip`
    bits (receiver settling)."""
    centers = t0 + (np.arange(nbits) + 0.5) * t_bit
    return np.interp(centers, t, v)[skip:]
