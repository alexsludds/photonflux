"""SKY130 process corners and corner-robust objective aggregation.

The SKY130 PDK ships five FET process corners as ``.lib`` sections of
``sky130.lib.spice``. All five keep the typical resistor/capacitor models, so
these are FET-only corners. **Mind the skewed-corner letters**: measured on
``nfet_01v8`` / ``pfet_01v8`` (on-current at |Vgs| = |Vds| = 1.8 V, pinned by
``tests/test_corners.py``), SKY130's names read *PMOS first*::

    tt  typical NMOS / typical PMOS
    ss  slow NMOS / slow PMOS     (slowest edges; Ion -15 % N, -30 % P)
    ff  fast NMOS / fast PMOS     (fastest edges; Ion +15 % N, +30 % P)
    sf  FAST NMOS / SLOW PMOS     (skewed: inverter trip point drops)
    fs  SLOW NMOS / FAST PMOS     (skewed: inverter trip point rises)

``cx.sky130_fet(..., corner=...)`` extracts the card for any of them; a design
is corner-robust when its *worst* corner still meets the target.
:func:`aggregate` folds a per-corner objective into that single number.

Engine-agnostic (numpy only), so the web app can import it without JAX.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

__all__ = ["CORNERS", "check_corner", "parse_corners", "aggregate",
           "AGGREGATES"]

CORNERS: tuple[str, ...] = ("tt", "ss", "ff", "sf", "fs")

# worst: the design is only as good as its worst corner (a guarantee).
# mean:  average across corners (a typical-die figure, not a guarantee).
AGGREGATES: tuple[str, ...] = ("worst", "mean")


def check_corner(corner: str) -> str:
    """Return ``corner`` if it is a SKY130 FET corner, else raise."""
    if corner not in CORNERS:
        raise ValueError(
            f"unknown SKY130 process corner {corner!r}; choose one of "
            f"{', '.join(CORNERS)}")
    return corner


def parse_corners(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """``"tt,ss"`` / ``["tt", "ss"]`` / ``"all"`` -> a validated corner tuple.

    Order is preserved and duplicates dropped; ``None`` or empty means
    ``("tt",)``.
    """
    if spec is None:
        return ("tt",)
    if isinstance(spec, str):
        if spec.strip().lower() == "all":
            return CORNERS
        spec = [s for s in spec.replace(" ", "").split(",") if s]
    out = tuple(dict.fromkeys(check_corner(str(s)) for s in spec))
    return out or ("tt",)


def aggregate(values: Mapping[str, float | None], mode: str = "worst", *,
              maximize: bool = True) -> float | None:
    """Fold per-corner objective values into one robust score.

    ``values`` maps corner -> objective (``None`` or non-finite = the corner
    could not be evaluated). A corner that fails makes the design fail --
    returning the aggregate of the survivors would reward designs that only
    work at the corners that happened to converge -- so any missing value
    returns ``None``.

    ``mode="worst"`` is the minimum when maximizing and the maximum when
    minimizing; ``mode="mean"`` is the arithmetic mean.
    """
    if mode not in AGGREGATES:
        raise ValueError(f"unknown corner aggregate {mode!r}; choose one of "
                         f"{', '.join(AGGREGATES)}")
    if not values:
        raise ValueError("no corner values to aggregate")
    vals = []
    for v in values.values():
        if v is None or not np.isfinite(v):
            return None
        vals.append(float(v))
    if mode == "mean":
        return float(np.mean(vals))
    return float(min(vals) if maximize else max(vals))
