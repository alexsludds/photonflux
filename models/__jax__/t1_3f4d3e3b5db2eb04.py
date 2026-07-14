"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _T1_setup(Rs: float=50.0, tau_c: float=1e-09, Von: float=0.9, _mfactor: float=1.0) -> jnp.ndarray:
    i_v22 = _mfactor * -1.0
    i_v25 = _mfactor * -tau_c
    i_v27 = _mfactor * tau_c
    return jnp.array([i_v22, i_v25, i_v27])

def _T1_combined(signals: Signals, s: States, init, Rs: float=50.0, tau_c: float=1e-09, Von: float=0.9, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v22 = init[0]
    i_v25 = init[1]
    i_v27 = init[2]
    v20 = jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(signals.vp - signals.vn - Von, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs)))
    v22 = v20 < 0.0
    v24 = jnp.where(v22, 0.0, v20)
    v38 = s.v_ni - signals.gnd - v24 * 1000.0
    v75 = tau_c * (s.v_ni - signals.gnd)
    v87 = jnp.where(v22, 0.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(1.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs))))
    v122 = _mfactor * v87
    v125 = _mfactor * -v87
    v89 = v87 * 1000.0
    v128 = _mfactor * v89
    j_resist = jnp.array([[v122, v125, 0.0, 0.0], [v125, v122, 0.0, 0.0], [v128, _mfactor * -v89, _mfactor, i_v22], [_mfactor * (0.0 - v89), v128, i_v22, _mfactor]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, i_v27, i_v25], [0.0, 0.0, i_v25, i_v27]])
    return ({'vp': _mfactor * v24, 'vn': _mfactor * -v24, 'gnd': _mfactor * -v38, 'v_ni': _mfactor * v38}, {'gnd': _mfactor * -v75, 'v_ni': _mfactor * v75}, j_resist, j_react)

def _T1_jacobian(signals: Signals, s: States, init, Rs: float=50.0, tau_c: float=1e-09, Von: float=0.9, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _T1_combined(signals, s, init, Rs=Rs, tau_c=tau_c, Von=Von, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('vp', 'vn', 'gnd'), states=('v_ni',), jacobian_fn=_T1_jacobian, combined_fn=_T1_combined, differentiable_params=None)
def T1(signals: Signals, s: States, init, Rs: float=50.0, tau_c: float=1e-09, Von: float=0.9, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _T1_combined(signals, s, init, Rs=Rs, tau_c=tau_c, Von=Von, _mfactor=_mfactor)
    return (f, q)

@T1.setup
def _T1_register_setup(*_a, **_kw):
    return _T1_setup(*_a, **_kw)