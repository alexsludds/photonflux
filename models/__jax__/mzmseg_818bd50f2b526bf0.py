"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _MzmSeg_setup(il_db: float=3.0, er_db: float=20.0, vpi: float=3.0, cel: float=6.000000000000001e-14, vbias: float=1.5, _mfactor: float=1.0) -> jnp.ndarray:
    i_v19 = jnp.power(jnp.maximum(10.0, 0.0), jnp.divide(-il_db, 10.0))
    i_v22 = jnp.power(jnp.maximum(10.0, 0.0), jnp.divide(er_db, 10.0))
    i_v24 = i_v22 + 1.0
    i_v25 = jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 0.0, jnp.divide(i_v22 - 1.0, jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 1.0, i_v24)))
    i_v31 = 0.5714285714285714 * cel
    i_v44 = _mfactor * i_v31
    i_v47 = _mfactor * -i_v31
    i_v34 = 0.2857142857142857 * cel
    i_v49 = _mfactor * i_v34
    i_v51 = _mfactor * -i_v34
    i_v36 = 0.14285714285714285 * cel
    i_v53 = _mfactor * i_v36
    i_v55 = _mfactor * -i_v36
    i_v57 = _mfactor * -1.0
    return jnp.array([i_v19, i_v25, i_v44, i_v47, i_v49, i_v51, i_v53, i_v55, i_v57])

def _MzmSeg_combined(signals: Signals, s: States, init, il_db: float=3.0, er_db: float=20.0, vpi: float=3.0, cel: float=6.000000000000001e-14, vbias: float=1.5, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v19 = init[0]
    i_v44 = init[2]
    i_v47 = init[3]
    i_v49 = init[4]
    i_v51 = init[5]
    i_v53 = init[6]
    i_v55 = init[7]
    v83 = 0.5714285714285714 * cel * (signals.vp1 - signals.vn1)
    v84 = 0.2857142857142857 * cel * (signals.vp2 - signals.vn2)
    v85 = 0.14285714285714285 * cel * (signals.vp3 - signals.vn3)
    v50 = 0.5 * init[1]
    v31 = jnp.where((vpi == 0.0) | ~jnp.isfinite(vpi), 0.0, jnp.divide(3.141592653589793, jnp.where((vpi == 0.0) | ~jnp.isfinite(vpi), 1.0, vpi)))
    v47 = v31 * (0.5714285714285714 * (signals.vp1 - signals.vn1) + 0.2857142857142857 * (signals.vp2 - signals.vn2) + 0.14285714285714285 * (signals.vp3 - signals.vn3) + vbias)
    v54 = i_v19 * (0.5 + v50 * jnp.cos(v47))
    v118 = v31 * -jnp.sin(v47) * v50 * i_v19 * (signals.pin - signals.gnd)
    v121 = 0.5714285714285714 * v118
    v122 = 0.2857142857142857 * v118
    v119 = 0.14285714285714285 * v118
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[8]], [v54, v121, -v121, v122, -v122, v119, -v119, -1.0, 1.0 - v54, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v44, i_v47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v47, i_v44, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, i_v49, i_v51, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, i_v51, i_v49, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, i_v53, i_v55, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, i_v55, i_v53, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'pout': _mfactor * s.i_br4, 'gnd': _mfactor * -s.i_br4, 'i_br4': v54 * (signals.pin - signals.gnd) - (signals.pout - signals.gnd)}, {'vp1': _mfactor * v83, 'vn1': _mfactor * -v83, 'vp2': _mfactor * v84, 'vn2': _mfactor * -v84, 'vp3': _mfactor * v85, 'vn3': _mfactor * -v85}, j_resist, j_react)

def _MzmSeg_jacobian(signals: Signals, s: States, init, il_db: float=3.0, er_db: float=20.0, vpi: float=3.0, cel: float=6.000000000000001e-14, vbias: float=1.5, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _MzmSeg_combined(signals, s, init, il_db=il_db, er_db=er_db, vpi=vpi, cel=cel, vbias=vbias, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('pin', 'vp1', 'vn1', 'vp2', 'vn2', 'vp3', 'vn3', 'pout', 'gnd'), states=('i_br4',), jacobian_fn=_MzmSeg_jacobian, combined_fn=_MzmSeg_combined, differentiable_params=None)
def MzmSeg(signals: Signals, s: States, init, il_db: float=3.0, er_db: float=20.0, vpi: float=3.0, cel: float=6.000000000000001e-14, vbias: float=1.5, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _MzmSeg_combined(signals, s, init, il_db=il_db, er_db=er_db, vpi=vpi, cel=cel, vbias=vbias, _mfactor=_mfactor)
    return (f, q)

@MzmSeg.setup
def _MzmSeg_register_setup(*_a, **_kw):
    return _MzmSeg_setup(*_a, **_kw)