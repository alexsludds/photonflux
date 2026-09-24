"""Bus-to-ring gap -> power coupling, so a ring can be swept by its layout.

``models/optical_field/ring_mod.va`` is parameterised by ``kappa2`` (the bus
power coupling), which is the right physical knob for the solver but the wrong
one for a designer: what goes on a mask is a **gap**. This module is the map
between them.

The evanescent field in the gap decays exponentially, so the field cross-
coupling of a point coupler goes as

    kappa(g)  = kappa0 * exp(-(g - g0) / g_d)
    kappa2(g) = kappa(g)^2

with ``g_d`` the gap decay length (~100-130 nm for a 220 nm SOI strip at
1310 nm; shorter at 1310 than at 1550 because the mode is better confined).

**This fit is the dominant source of absolute error in any gap study.**
Relative comparisons across gaps are trustworthy; turning an optimal
``kappa2`` back into a mask gap in nm is only as good as the calibration. The
defaults below are anchored to the 7.5 um / kappa2 = 0.10 ring that
``examples/ring_mod_sky130.py`` and ``tests/test_ring_mod.py`` are built on,
*not* to measured silicon. Replace them with :meth:`GapCoupling.from_points`
fitted to FDTD or measured data before quoting a gap.

Second-order effects this deliberately does not model: gap-dependent coupler
excess loss, the resonance pull from the coupler's own phase, and the
breakdown of the point-coupler idealisation as the gap closes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["GapCoupling", "DEFAULT_O_BAND"]


@dataclass(frozen=True)
class GapCoupling:
    """Exponential gap -> field-coupling calibration.

    ``kappa0`` is the field coupling at the reference gap ``g0_nm``; ``gd_nm``
    is the decay length. All gaps in nm.
    """

    kappa0: float
    g0_nm: float
    gd_nm: float

    def __post_init__(self) -> None:
        if not 0.0 < self.kappa0 <= 1.0:
            raise ValueError(f"kappa0 must be in (0, 1], got {self.kappa0}")
        if self.gd_nm <= 0.0:
            raise ValueError(f"gd_nm must be positive, got {self.gd_nm}")

    def kappa(self, gap_nm):
        """Field cross-coupling at `gap_nm` (scalar or array)."""
        g = np.asarray(gap_nm, dtype=float)
        return self.kappa0 * np.exp(-(g - self.g0_nm) / self.gd_nm)

    def kappa2(self, gap_nm):
        """Power cross-coupling |kappa|^2 -- the ``kappa2`` of ring_mod.va.

        Clipped to just under 1: the exponential is unbounded as the gap
        closes, but a point coupler cannot transfer more than all the power,
        and ``ring_mod.va`` declares ``kappa2`` on the open range (0:1).
        """
        return np.clip(self.kappa(gap_nm) ** 2, 1e-12, 1.0 - 1e-12)

    def gap_for_kappa2(self, kappa2):
        """Invert :meth:`kappa2` -- the gap [nm] that yields this coupling."""
        k2 = np.asarray(kappa2, dtype=float)
        if np.any(k2 <= 0.0) or np.any(k2 >= 1.0):
            raise ValueError("kappa2 must lie strictly inside (0, 1)")
        return self.g0_nm - self.gd_nm * np.log(np.sqrt(k2) / self.kappa0)

    @classmethod
    def from_points(cls, gaps_nm, kappa2, *, g0_nm: float | None = None) -> "GapCoupling":
        """Least-squares fit of the exponential to measured/FDTD points.

        `kappa2` is power coupling, so the fit is linear in log(sqrt(kappa2)).
        Needs at least two gaps.
        """
        g = np.asarray(gaps_nm, dtype=float).ravel()
        k2 = np.asarray(kappa2, dtype=float).ravel()
        if g.size != k2.size:
            raise ValueError("gaps_nm and kappa2 must be the same length")
        if g.size < 2:
            raise ValueError("need at least two points to fit a decay length")
        if np.any(k2 <= 0.0) or np.any(k2 >= 1.0):
            raise ValueError("kappa2 values must lie strictly inside (0, 1)")
        # log(kappa) = log(kappa0) - (g - g0)/g_d  -> straight line in g
        slope, intercept = np.polyfit(g, np.log(np.sqrt(k2)), 1)
        if slope >= 0.0:
            raise ValueError(
                "fitted coupling grows with gap; check the data orientation")
        gd = -1.0 / slope
        ref = float(g0_nm) if g0_nm is not None else float(g.min())
        return cls(kappa0=float(math.exp(intercept + slope * ref)),
                   g0_nm=ref, gd_nm=float(gd))


# Anchored so that the 200 nm reference gap reproduces kappa2 = 0.10 -- the
# device in examples/ring_mod_sky130.py (critical coupling for that ring is
# 0.076, so the reference sits mildly overcoupled and a gap sweep crosses
# critical from both sides). PLACEHOLDER CALIBRATION: see module docstring.
DEFAULT_O_BAND = GapCoupling(kappa0=math.sqrt(0.10), g0_nm=200.0, gd_nm=110.0)
