"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _Softamp_setup(vsat: float=0.8, gain: float=4.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v20 = _mfactor * -1.0
    return jnp.array([i_v20])

def _Softamp_combined(signals: Signals, s: States, init, vsat: float=0.8, gain: float=4.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    v22 = jnp.tanh(jnp.where((vsat == 0.0) | ~jnp.isfinite(vsat), 0.0, jnp.divide(gain * (signals.inp - signals.gnd), jnp.where((vsat == 0.0) | ~jnp.isfinite(vsat), 1.0, vsat))))
    v36 = jnp.where((vsat == 0.0) | ~jnp.isfinite(vsat), 0.0, jnp.divide(gain, jnp.where((vsat == 0.0) | ~jnp.isfinite(vsat), 1.0, vsat))) * (1.0 - v22 * v22) * vsat
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, init[0]], [v36, -1.0, 1.0 - v36, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    return ({'outp': _mfactor * s.i_br1, 'gnd': _mfactor * -s.i_br1, 'i_br1': vsat * v22 - (signals.outp - signals.gnd)}, {}, j_resist, j_react)

def _Softamp_jacobian(signals: Signals, s: States, init, vsat: float=0.8, gain: float=4.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _Softamp_combined(signals, s, init, vsat=vsat, gain=gain, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('inp', 'outp', 'gnd'), states=('i_br1',), jacobian_fn=_Softamp_jacobian, combined_fn=_Softamp_combined, differentiable_params=None)
def Softamp(signals: Signals, s: States, init, vsat: float=0.8, gain: float=4.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _Softamp_combined(signals, s, init, vsat=vsat, gain=gain, _mfactor=_mfactor)
    return (f, q)

@Softamp.setup
def _Softamp_register_setup(*_a, **_kw):
    return _Softamp_setup(*_a, **_kw)