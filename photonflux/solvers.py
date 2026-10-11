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


# ---------------------------------------------------------------------------
# BDF2 with a reused Newton matrix
# ---------------------------------------------------------------------------

_BDF2_REUSE = None


def bdf2_reuse_solver(linear_solver: Any, **kwargs: Any) -> Any:
    """A ``BDF2ReuseSolver`` for a dense circulax linear solver (the class
    subclasses a circulax solver, so it is built on first use)."""
    global _BDF2_REUSE
    if _BDF2_REUSE is None:
        _BDF2_REUSE = _make_bdf2_reuse()
    return _BDF2_REUSE(linear_solver=linear_solver, **kwargs)


def _make_bdf2_reuse():
    import diffrax
    import jax
    import jax.numpy as jnp

    from circulax.solvers.assembly import (
        assemble_residual_only_complex,
        assemble_residual_only_real,
        assemble_system_complex,
        assemble_system_real,
    )
    from circulax.solvers.linear import (
        DAMPING_EPS,
        DAMPING_FACTOR,
        GROUND_STIFFNESS,
    )
    from circulax.solvers.transient import (
        BDF2VectorizedTransientSolver,
        _bdf2_preamble,
        _unpack_args,
    )

    class BDF2ReuseSolver(BDF2VectorizedTransientSolver):
        """circulax's BDF2 that stops rebuilding the Newton matrix every
        iteration.

        ``BDF2VectorizedTransientSolver`` assembles the Jacobian and solves a
        fresh dense LU on every Newton iteration of every step. At a fixed
        step that matrix, ``dF/dy + (alpha/h) dQ/dy``, barely moves between
        steps (and is constant for the linear part of a circuit), so this
        solver keeps its inverse in the solver state and iterates with it
        (a chord, or modified-Newton, iteration): each iteration is one
        residual evaluation and one matrix-vector product.

        The inverse is rebuilt at the predictor when ``alpha/h`` has moved by
        more than ``coef_rtol`` (the first two steps, any step-size change).
        If the chord iteration contracts by less than ``max_rate`` per
        iteration (a strongly nonlinear step: a transistor switching, a laser
        turning on) or has not converged in ``chord_max_steps``, the step
        finishes with circulax's full Newton from where the chord left off,
        and the next step rebuilds the inverse.

        Accuracy: both iterations converge to the same root. A chord
        iterate's remaining error is about ``rate * |update|`` rather than
        Newton's ``|update|^2``, so convergence is declared on the tighter
        ``err < tight`` (circulax's scaled max-norm, in which full Newton
        stops at ``err < 1``; at the default 1e-5 tolerances, ``tight =
        1e-3`` is a 1e-8 update), after at least two iterations.
        """

        max_rate: float = 0.25
        coef_rtol: float = 1e-3
        tight: float = 1e-3
        chord_max_steps: int = 12

        def init(self, terms, t0, t1, y0, args):
            y_nm1, h_nm1, q0 = super().init(terms, t0, t1, y0, args)
            n = y0.shape[0]
            # NaN coefficient: no matrix yet, so the first step builds one
            return (y_nm1, h_nm1, q0, jnp.zeros((n, n), y0.dtype),
                    jnp.asarray(jnp.nan, y0.dtype))

        def step(self, terms, t0, t1, y0, args, solver_state, options):
            groups, num_vars, hist_t, hist_y = _unpack_args(args)
            h_n = t1 - t0
            ls = self.linear_solver
            cplx = getattr(ls, "is_complex", False)
            y_nm1, h_nm1, q_nm1, inv0, coef0 = solver_state
            y_pred, alpha, make_residual, hist_state = _bdf2_preamble(
                y0, t0, h_n, (y_nm1, h_nm1, q_nm1), groups, num_vars, cplx)
            coef = alpha / h_n
            ground = [0, num_vars] if cplx else [0]
            n = y0.shape[0]
            diag = jnp.arange(n)
            atol, rtol = self.newton_atol, self.newton_rtol

            def system(y):
                # the matrix DenseSolver._solve_impl builds and solves
                asm = assemble_system_complex if cplx else assemble_system_real
                f, q, vals = asm(y, groups, t1, h_n, alpha=alpha,
                                 hist_t=hist_t, hist_y=hist_y)
                J = jnp.zeros((n, n), dtype=y.dtype)
                J = J.at[ls.static_rows, ls.static_cols].add(vals)
                J = J.at[diag, diag].add(ls.g_leak)
                for idx in ls.ground_indices:
                    J = J.at[idx, idx].add(GROUND_STIFFNESS)
                return residual_of(f, q, y), J

            def residual_of(f, q, y):
                r = make_residual(f, q)
                for idx in ground:
                    r = r.at[idx].add(GROUND_STIFFNESS * y[idx])
                return r

            def residual(y):
                asm = (assemble_residual_only_complex if cplx
                       else assemble_residual_only_real)
                f, q = asm(y, groups, t1, h_n, hist_t=hist_t, hist_y=hist_y)
                return residual_of(f, q, y)

            def update(y, delta):
                # circulax's damped Newton update and scaled convergence norm
                damping = jnp.minimum(
                    1.0, DAMPING_FACTOR / (jnp.max(jnp.abs(delta))
                                           + DAMPING_EPS))
                y_new = y + delta * damping
                return y_new, jnp.max(jnp.abs(y - y_new)
                                      / (atol + rtol * jnp.abs(y_new)))

            inf = jnp.asarray(jnp.inf, y0.dtype)

            def chord(inv, y_start):
                def body(c):
                    k, y, err_prev, _err, _done, _slow = c
                    y_new, err = update(y, -(inv @ residual(y)))
                    done = (k > 0) & (err < self.tight)
                    slow = (k > 0) & (err > self.max_rate * err_prev) & ~done
                    return k + 1, y_new, err, err, done, slow

                def cond(c):
                    k, _y, _ep, _err, done, slow = c
                    return ~done & ~slow & (k < self.chord_max_steps)

                _k, y, _ep, err, done, _slow = jax.lax.while_loop(
                    cond, body, (jnp.asarray(0), y_start, inf, inf,
                                 jnp.asarray(False), jnp.asarray(False)))
                return y, err, done

            def newton(y_start):
                def body(c):
                    k, y, _err, _done = c
                    r, J = system(y)
                    y_new, err = update(y, jnp.linalg.solve(J, -r))
                    return k + 1, y_new, err, err < self.tight

                def cond(c):
                    k, _y, _err, done = c
                    return ~done & (k < self.newton_max_steps)

                _k, y, err, done = jax.lax.while_loop(
                    cond, body, (jnp.asarray(0), y_start, inf,
                                 jnp.asarray(False)))
                return y, err, done

            # t1 - t0 wobbles in its last bits even at a constant step, and a
            # slightly-off matrix only slows the chord (the residual always
            # uses the exact coefficient): rebuild on a real change only
            stale = ~(jnp.abs(coef - coef0) <= self.coef_rtol * jnp.abs(coef))
            inv = jax.lax.cond(stale, lambda: jnp.linalg.inv(system(y_pred)[1]),
                               lambda: inv0)
            y1, err1, done1 = chord(inv, y_pred)
            y_next, err, done = jax.lax.cond(
                done1, lambda: (y1, err1, done1), lambda: newton(y1))
            # Out of iterations: accept on circulax's own criterion (err < 1)
            # rather than fail a step plain Newton would have taken.
            result = jax.lax.cond(
                done | (err < 1.0), lambda: diffrax.RESULTS.successful,
                lambda: diffrax.RESULTS.nonlinear_divergence)
            # a step that needed full Newton leaves the matrix behind it stale
            state = (*hist_state, inv, jnp.where(done1, coef, jnp.nan))
            return (y_next, y_next - y_pred, {"y0": y0, "y1": y_next},
                    state, result)

    return BDF2ReuseSolver
