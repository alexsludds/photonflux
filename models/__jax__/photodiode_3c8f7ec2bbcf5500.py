"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _Photodiode_setup(R: float=0.8, Cj: float=1e-13, Idk: float=1e-09, _mfactor: float=1.0) -> jnp.ndarray:
    i_v20 = _mfactor * R
    i_v23 = _mfactor * Cj
    i_v25 = _mfactor * -R
    i_v27 = _mfactor * -Cj
    return jnp.array([i_v20, i_v23, i_v25, i_v27])

def _Photodiode_combined(signals: Signals, s: States, init, R: float=0.8, Cj: float=1e-13, Idk: float=1e-09, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v20 = init[0]
    i_v23 = init[1]
    i_v25 = init[2]
    i_v27 = init[3]
    v20 = R * (signals.popt - signals.gnd) + Idk
    v29 = Cj * (signals.an - signals.cat)
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [i_v20, i_v25, 0.0, 0.0], [i_v25, i_v20, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, i_v23, i_v27], [0.0, 0.0, i_v27, i_v23]])
    return ({'an': _mfactor * v20, 'cat': _mfactor * -v20}, {'an': _mfactor * v29, 'cat': _mfactor * -v29}, j_resist, j_react)

def _Photodiode_jacobian(signals: Signals, s: States, init, R: float=0.8, Cj: float=1e-13, Idk: float=1e-09, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _Photodiode_combined(signals, s, init, R=R, Cj=Cj, Idk=Idk, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('popt', 'gnd', 'an', 'cat'), jacobian_fn=_Photodiode_jacobian, combined_fn=_Photodiode_combined, differentiable_params=None)
def Photodiode(signals: Signals, s: States, init, R: float=0.8, Cj: float=1e-13, Idk: float=1e-09, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _Photodiode_combined(signals, s, init, R=R, Cj=Cj, Idk=Idk, _mfactor=_mfactor)
    return (f, q)

@Photodiode.setup
def _Photodiode_register_setup(*_a, **_kw):
    return _Photodiode_setup(*_a, **_kw)