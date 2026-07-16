"""Bit patterns for link simulations (engine-agnostic).

``prbs`` returns a maximal-length pseudo-random bit sequence; drive it into a
circulax stimulus source (see ``webapp/wavesrc.py`` or
``examples/ring_mod_sky130.py``).
"""
from __future__ import annotations

import numpy as np

__all__ = ["prbs", "sample_centers"]

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
