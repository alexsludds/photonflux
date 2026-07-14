"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _MzmTw_setup(il_db: float=3.0, er_db: float=20.0, f_el: float=35000000000.0, n_rf: float=2.4, n_opt: float=4.2, len: float=0.004, vpi: float=1.5, cel: float=5.000000000000001e-15, vbias: float=0.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v19 = jnp.power(jnp.maximum(10.0, 0.0), jnp.divide(-il_db, 10.0))
    i_v22 = jnp.power(jnp.maximum(10.0, 0.0), jnp.divide(er_db, 10.0))
    i_v24 = i_v22 + 1.0
    i_v25 = jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 0.0, jnp.divide(i_v22 - 1.0, jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 1.0, i_v24)))
    i_v26 = 6.283185307179586 * f_el
    i_v29 = jnp.where((i_v26 == 0.0) | ~jnp.isfinite(i_v26), 0.0, jnp.divide(1.0, jnp.where((i_v26 == 0.0) | ~jnp.isfinite(i_v26), 1.0, i_v26)))
    i_v31 = n_rf - n_opt
    i_v41 = jnp.divide(jnp.divide(jnp.where(i_v31 < 0.0, -i_v31, i_v31) * len, 299792458.0), 2.7834510910805568)
    i_v53 = _mfactor * cel
    i_v56 = _mfactor * -cel
    i_v58 = _mfactor * -1.0
    i_v30 = -i_v29
    i_v43 = -i_v41
    return jnp.array([i_v19, i_v25, i_v29, i_v41, i_v53, i_v56, i_v58, i_v30, i_v43])

def _MzmTw_combined(signals: Signals, s: States, init, il_db: float=3.0, er_db: float=20.0, f_el: float=35000000000.0, n_rf: float=2.4, n_opt: float=4.2, len: float=0.004, vpi: float=1.5, cel: float=5.000000000000001e-15, vbias: float=0.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v19 = init[0]
    i_v53 = init[4]
    i_v56 = init[5]
    v118 = cel * (signals.vp - signals.vn)
    v77 = 0.5 * init[1]
    v68 = jnp.where((vpi == 0.0) | ~jnp.isfinite(vpi), 0.0, jnp.divide(3.141592653589793 * s.v_ewlk, jnp.where((vpi == 0.0) | ~jnp.isfinite(vpi), 1.0, vpi)))
    v81 = i_v19 * (0.5 + v77 * jnp.cos(v68))
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[6]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], [v81, 0.0, 0.0, -1.0, 1.0 - v81, 0.0, jnp.where((vpi == 0.0) | ~jnp.isfinite(vpi), 0.0, jnp.divide(3.141592653589793, jnp.where((vpi == 0.0) | ~jnp.isfinite(vpi), 1.0, vpi))) * -jnp.sin(v68) * v77 * i_v19 * (signals.pin - signals.gnd), 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v53, i_v56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v56, i_v53, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, init[7], 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[8], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'pout': _mfactor * s.i_br4, 'gnd': _mfactor * -s.i_br4, 'v_eflt': _mfactor * s.i_eflt, 'v_ewlk': _mfactor * s.i_ewlk, 'i_eflt': signals.vp - signals.vn + vbias - s.v_eflt, 'i_ewlk': s.v_eflt - s.v_ewlk, 'i_br4': v81 * (signals.pin - signals.gnd) - (signals.pout - signals.gnd)}, {'vp': _mfactor * v118, 'vn': _mfactor * -v118, 'i_eflt': -(init[2] * s.v_eflt), 'i_ewlk': -(init[3] * s.v_ewlk)}, j_resist, j_react)

def _MzmTw_jacobian(signals: Signals, s: States, init, il_db: float=3.0, er_db: float=20.0, f_el: float=35000000000.0, n_rf: float=2.4, n_opt: float=4.2, len: float=0.004, vpi: float=1.5, cel: float=5.000000000000001e-15, vbias: float=0.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _MzmTw_combined(signals, s, init, il_db=il_db, er_db=er_db, f_el=f_el, n_rf=n_rf, n_opt=n_opt, len=len, vpi=vpi, cel=cel, vbias=vbias, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('pin', 'vp', 'vn', 'pout', 'gnd'), states=('v_eflt', 'v_ewlk', 'i_eflt', 'i_ewlk', 'i_br4'), jacobian_fn=_MzmTw_jacobian, combined_fn=_MzmTw_combined, differentiable_params=None)
def MzmTw(signals: Signals, s: States, init, il_db: float=3.0, er_db: float=20.0, f_el: float=35000000000.0, n_rf: float=2.4, n_opt: float=4.2, len: float=0.004, vpi: float=1.5, cel: float=5.000000000000001e-15, vbias: float=0.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _MzmTw_combined(signals, s, init, il_db=il_db, er_db=er_db, f_el=f_el, n_rf=n_rf, n_opt=n_opt, len=len, vpi=vpi, cel=cel, vbias=vbias, _mfactor=_mfactor)
    return (f, q)

@MzmTw.setup
def _MzmTw_register_setup(*_a, **_kw):
    return _MzmTw_setup(*_a, **_kw)