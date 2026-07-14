"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _LaserDml_setup(Rs: float=5.0, tau: float=0.0, Von: float=1.2, slope: float=0.3, Ith: float=0.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v21 = _mfactor * -1.0
    i_v19 = -tau
    return jnp.array([i_v21, i_v19])

def _LaserDml_combined(signals: Signals, s: States, init, Rs: float=5.0, tau: float=0.0, Von: float=1.2, slope: float=0.3, Ith: float=0.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    v20 = jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(signals.an - signals.cat - Von, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs)))
    v22 = v20 < 0.0
    v24 = jnp.where(v22, 0.0, v20)
    v32 = slope * (v24 - Ith)
    v34 = v32 < 0.0
    v74 = jnp.where(v22, 0.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(1.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs))))
    v104 = _mfactor * v74
    v107 = _mfactor * -v74
    v77 = jnp.where(v34, 0.0, v74 * slope)
    j_resist = jnp.array([[v104, v107, 0.0, 0.0, 0.0], [v107, v104, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, init[0]], [v77, -v77, -1.0, 1.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, init[1], tau, 0.0]])
    return ({'an': _mfactor * v24, 'cat': _mfactor * -v24, 'popt': _mfactor * s.i_br2, 'gnd': _mfactor * -s.i_br2, 'i_br2': jnp.where(v34, 0.0, v32) - (signals.popt - signals.gnd)}, {'i_br2': -(tau * (signals.popt - signals.gnd))}, j_resist, j_react)

def _LaserDml_jacobian(signals: Signals, s: States, init, Rs: float=5.0, tau: float=0.0, Von: float=1.2, slope: float=0.3, Ith: float=0.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _LaserDml_combined(signals, s, init, Rs=Rs, tau=tau, Von=Von, slope=slope, Ith=Ith, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('an', 'cat', 'popt', 'gnd'), states=('i_br2',), jacobian_fn=_LaserDml_jacobian, combined_fn=_LaserDml_combined, differentiable_params=None)
def LaserDml(signals: Signals, s: States, init, Rs: float=5.0, tau: float=0.0, Von: float=1.2, slope: float=0.3, Ith: float=0.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _LaserDml_combined(signals, s, init, Rs=Rs, tau=tau, Von=Von, slope=slope, Ith=Ith, _mfactor=_mfactor)
    return (f, q)

@LaserDml.setup
def _LaserDml_register_setup(*_a, **_kw):
    return _LaserDml_setup(*_a, **_kw)