"""Compile-once DC and transient solves for circulax circuits.

``Circuit.dc()`` and ``Circuit.transient()`` hand ``eqx.filter_jit`` a freshly
built callable on every call (the transient's ODE term is a new ``lambda``,
the DC Newton step a new closure). A jit cache keys on those callables, so
every call re-traces and re-compiles the whole solver: 1-3 s of XLA work
around a solve that often takes milliseconds, paid again on each re-run,
sweep point, optimizer evaluation and noise seed.

These wrappers pass only module-level callables and keep every number
dynamic, so a repeat solve of the same circuit structure reuses the compiled
program. The solver, its arithmetic and its results are unchanged; only the
compile is skipped.

    from photonflux import solvers

    y0 = solvers.dc(circuit)                       # cached Newton DC
    sol = solvers.transient(circuit, t0=0.0, t1=t_stop, dt0=dt, y0=y0,
                            saveat=ts, max_steps=n_steps)
"""
from __future__ import annotations

from typing import Any

__all__ = ["dc", "transient", "step_budget"]


def _zero_vf(t, y, args):
    import jax.numpy as jnp

    return jnp.zeros_like(y)


_ZERO_TERM = None


def _zero_term():
    """circulax's placeholder ODE term (its solvers never call it), built
    once so the transient jit cache sees the same static callable."""
    global _ZERO_TERM
    if _ZERO_TERM is None:
        import diffrax

        _ZERO_TERM = diffrax.ODETerm(_zero_vf)
    return _ZERO_TERM


def step_budget(n: int) -> int:
    """``max_steps`` rounded up to 1-2-5 x 10^k.

    ``max_steps`` is a compile-time constant of the transient loop, so an
    exact ``int(t_stop / dt) + 10`` recompiles on every t_stop edit. The loop
    stops at t1 either way, so a rounded-up budget gives identical results
    and lets nearby run lengths share one compiled program.
    """
    n = max(int(n), 16)
    scale = 1
    while True:
        for m in (1, 2, 5):
            if m * scale >= n:
                return m * scale
        scale *= 10


_DC_JIT = None


def _dc_jit():
    global _DC_JIT
    if _DC_JIT is None:
        import equinox as eqx
        from circulax.circuit import Circuit

        @eqx.filter_jit
        def run(solver, groups, sys_size, y_guess, params, rtol, atol,
                max_steps):
            # A throwaway Circuit around the traced groups: its .dc() applies
            # the parameter updates (vmapping over array-valued ones) and
            # runs the Newton solve inside this one compiled program.
            view = Circuit(solver, groups, sys_size, {}, rtol=rtol,
                           atol=atol, max_steps=max_steps)
            return view.dc(y_guess, params=params)

        _DC_JIT = run
    return _DC_JIT


def dc(circuit: Any, y_guess: Any = None, *,
       params: dict[str, Any] | None = None) -> Any:
    """``circuit.dc(y_guess, params=params)``, compiled once per structure.

    Same semantics as :meth:`circulax.Circuit.dc`: array-valued ``params``
    vmap over their shared leading axis and return ``(batch, n)``.
    """
    import jax.numpy as jnp

    y = circuit._zero_guess() if y_guess is None else jnp.asarray(y_guess)
    arrays = {k: jnp.asarray(v) for k, v in (params or {}).items()}
    return _dc_jit()(circuit.solver, circuit.groups, circuit.sys_size, y,
                     arrays, float(circuit.rtol), float(circuit.atol),
                     int(circuit.max_steps))


def transient(circuit: Any, *, y0: Any, max_steps: int, **kwargs: Any) -> Any:
    """``circuit.transient(...)`` that reuses its compiled solver.

    ``y0`` is required (circulax's implicit DC fallback would compile its
    own Newton loop; use :func:`dc`). ``max_steps`` is rounded up with
    :func:`step_budget`; everything else passes through unchanged.
    """
    if y0 is None:
        raise ValueError("solvers.transient needs y0 (use solvers.dc first)")
    kwargs.setdefault("term", _zero_term())
    return circuit.transient(y0=y0, max_steps=step_budget(max_steps),
                             **kwargs)
