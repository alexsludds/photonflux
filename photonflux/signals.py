"""Bit patterns for link simulations (engine-agnostic).

``prbs`` returns a maximal-length pseudo-random bit sequence; drive it into a
circulax stimulus source (see ``webapp/wavesrc.py`` or
``examples/ring_mod_sky130.py``).
"""
from __future__ import annotations

import numpy as np

__all__ = ["prbs", "sample_centers"]

# Fibonacci LFSR feedback taps: the exponents of G(x) excluding the x^0 term.
# Most orders take a primitive trinomial x^n + x^k + 1, but degree 13 has none,
# so PRBS-13 uses the four-term IEEE 802.3 polynomial (the pattern 802.3bs/cd
# specify for TDECQ) -- hence a tap *list* rather than a single second tap.
_PRBS_TAPS = {
    5: (5, 3),
    6: (6, 5),
    7: (7, 6),
    9: (9, 5),
    11: (11, 9),
    13: (13, 12, 2, 1),  # x^13 + x^12 + x^2 + x + 1  (IEEE 802.3 PRBS13)
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}


def prbs(order: int = 7, nbits: int | None = None, seed: int = 1) -> np.ndarray:
    """One (or part of one) period of a PRBS-`order` sequence as 0/1 ints."""
    if order not in _PRBS_TAPS:
        raise ValueError(f"unsupported PRBS order {order}; choose {sorted(_PRBS_TAPS)}")
    if seed <= 0 or seed >= (1 << order):
        raise ValueError("seed must be a nonzero state narrower than the register")
    nbits = nbits or (1 << order) - 1
    taps = _PRBS_TAPS[order]
    mask = (1 << order) - 1
    reg = seed
    out = np.empty(nbits, dtype=np.int8)
    for i in range(nbits):
        out[i] = reg & 1
        fb = 0
        for t in taps:
            fb ^= reg >> (t - 1)
        reg = ((reg << 1) | (fb & 1)) & mask
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
