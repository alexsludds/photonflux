"""Progress bar for JAX/diffrax transient solves.

Thin wrapper over diffrax's built-in progress-meter mechanism. diffrax's
``diffeqsolve`` (which ``circulax.Circuit.transient`` forwards ``**kwargs``
to) accepts a ``progress_meter=`` argument; pass the return value of
:func:`transient_progress_meter` straight into ``circuit.transient(...)`` and
a bar tracks the solve as it steps through simulated time::

    from _progress import transient_progress_meter

    sol = circuit.transient(..., progress_meter=transient_progress_meter())

The meter measures wall-clock *coverage of the time interval* ``[t0, t1]``,
not step count, so it advances smoothly regardless of adaptive stepping.
"""
from __future__ import annotations

import importlib.util

import diffrax


def transient_progress_meter(enabled: bool = True):
    """Return a diffrax progress meter for a transient solve.

    Args:
        enabled: When ``False`` (e.g. inside a sweep or optimiser loop, where
            one bar per solve is just noise) returns diffrax's no-op meter so
            nothing is printed. When ``True`` prefers tqdm's bar and falls back
            to diffrax's plain-text meter if tqdm is not installed.

    Returns:
        A ``diffrax.AbstractProgressMeter`` to pass as
        ``circuit.transient(..., progress_meter=<this>)``.
    """
    if not enabled:
        return diffrax.NoProgressMeter()
    if importlib.util.find_spec("tqdm") is not None:
        return diffrax.TqdmProgressMeter()
    return diffrax.TextProgressMeter()
