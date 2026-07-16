"""Shared pieces for the SOA-cavity examples (soa_fp_laser, soa_vernier_laser,
ring_tpa_q): a smooth staircase bias source, an open-circuit terminator for
driven-but-unused optical outputs, and the fixed-step BDF2 transient runner.
"""
from __future__ import annotations

import diffrax
import jax.numpy as jnp
import numpy as np

from _progress import transient_progress_meter
from circulax.components.base_component import Signals, States, component, source


def staircase_source(levels: np.ndarray, t_step: float, t_edge: float = 50e-12):
    """Voltage staircase: levels[k] during [k*t_step, (k+1)*t_step), smoothstep
    edges of duration t_edge (C1-continuous, keeps BDF2 happy)."""
    lv = jnp.asarray(levels, dtype=jnp.float64)
    padded = jnp.concatenate([lv[:1], lv])
    n = len(levels)

    @source(ports=("p1", "p2"), states=("i_src",))
    def Staircase(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
        i = jnp.clip(jnp.floor(t / t_step).astype(jnp.int32), 0, n - 1)
        prev, cur = padded[i], padded[i + 1]
        x = jnp.clip((t - i * t_step) / t_edge, 0.0, 1.0)
        v = prev + (cur - prev) * x * x * (3.0 - 2.0 * x)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}

    return Staircase


def terminator():
    """Infinite-impedance 2-node termination.

    circulax requires every component port to be connected; VA models drive
    their unused optical outputs (a mirror's back emission, a ring's thru
    port), so those nodes need a counterparty that draws nothing. This is it.
    """
    @component(ports=("re", "im"))
    def Terminator(signals: Signals, s: States) -> tuple[dict, dict]:
        return {"re": 0.0, "im": 0.0}, {}

    return Terminator


def run_transient(c, t_max: float, dt: float, save_every: float, y0=None,
                  progress: bool = True):
    """DC then fixed-step BDF2 (circulax 0.2.1's adaptive retry path
    misreports divergence with VA/OSDI devices in a complex system — the
    ring_mod_sky130.py solver note)."""
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    if y0 is None:
        y0 = c.dc()
    ts = jnp.arange(0.0, t_max, save_every)
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=dt, y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=int(t_max / dt) + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
        progress_meter=transient_progress_meter(progress),
    )
    assert sol.result == diffrax.RESULTS.successful, f"transient failed: {sol.result}"
    return np.asarray(sol.ts), sol


def port_power(c, sol_or_y, re: str, im: str) -> np.ndarray:
    """|E|^2 from a (re, im) node pair probed as circuit ports."""
    er = np.asarray(c.port(sol_or_y, re).real)
    ei = np.asarray(c.port(sol_or_y, im).real)
    return er**2 + ei**2
