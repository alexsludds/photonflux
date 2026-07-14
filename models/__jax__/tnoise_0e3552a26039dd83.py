"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _Tnoise_setup(R: float=1000.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v18 = jnp.where((R == 0.0) | ~jnp.isfinite(R), 0.0, jnp.divide(1.0, jnp.where((R == 0.0) | ~jnp.isfinite(R), 1.0, R)))
    i_v25 = _mfactor * i_v18
    i_v27 = _mfactor * -i_v18
    return jnp.array([i_v25, i_v27])

def _Tnoise_combined(signals: Signals, s: States, init, R: float=1000.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v25 = init[0]
    i_v27 = init[1]
    v19 = jnp.where((R == 0.0) | ~jnp.isfinite(R), 0.0, jnp.divide(signals.a - signals.b, jnp.where((R == 0.0) | ~jnp.isfinite(R), 1.0, R)))
    j_resist = jnp.array([[i_v25, i_v27, 0.0], [i_v27, i_v25, 0.0], [0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return ({'a': _mfactor * v19, 'b': _mfactor * -v19}, {}, j_resist, j_react)

def _Tnoise_jacobian(signals: Signals, s: States, init, R: float=1000.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _Tnoise_combined(signals, s, init, R=R, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('a', 'b', 'gnd'), jacobian_fn=_Tnoise_jacobian, combined_fn=_Tnoise_combined, differentiable_params=None)
def Tnoise(signals: Signals, s: States, init, R: float=1000.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _Tnoise_combined(signals, s, init, R=R, _mfactor=_mfactor)
    return (f, q)

@Tnoise.setup
def _Tnoise_register_setup(*_a, **_kw):
    return _Tnoise_setup(*_a, **_kw)