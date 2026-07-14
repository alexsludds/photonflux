"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _RingMod_setup(lambda_res_nm: float=1310.0, q_i: float=60000.0, q_e: float=60000.0, dl_dv_pm: float=30.0, lambda_nm: float=1310.0, cj: float=0.0, rleak: float=100000000.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v16 = lambda_res_nm * 1e-09
    i_v19 = jnp.where((i_v16 == 0.0) | ~jnp.isfinite(i_v16), 0.0, jnp.divide(1883651567.3088531, jnp.where((i_v16 == 0.0) | ~jnp.isfinite(i_v16), 1.0, i_v16)))
    i_v24 = 2.0 * q_e
    i_v26 = jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 0.0, jnp.divide(i_v19, jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 1.0, i_v24)))
    i_v27 = jnp.where((2.0 * q_i == 0.0) | ~jnp.isfinite(2.0 * q_i), 0.0, jnp.divide(i_v19, jnp.where((2.0 * q_i == 0.0) | ~jnp.isfinite(2.0 * q_i), 1.0, 2.0 * q_i))) + i_v26
    i_v28 = jnp.where((i_v27 == 0.0) | ~jnp.isfinite(i_v27), 0.0, jnp.divide(1.0, jnp.where((i_v27 == 0.0) | ~jnp.isfinite(i_v27), 1.0, i_v27)))
    i_v30 = i_v28 * 2.0 * i_v26
    i_v40 = jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 0.0, jnp.divide(1.0, jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 1.0, rleak)))
    i_v47 = _mfactor * i_v40
    i_v50 = _mfactor * cj
    i_v52 = _mfactor * -i_v40
    i_v54 = _mfactor * -cj
    i_v56 = _mfactor * -i_v28
    i_v43 = 0.0 - i_v28
    i_v59 = _mfactor * i_v43
    i_v61 = _mfactor * -i_v30
    i_v63 = _mfactor * i_v30
    i_v65 = _mfactor * -1.0
    i_v68 = _mfactor * i_v28
    i_v72 = _mfactor * (0.0 - i_v30)
    i_v79 = _mfactor * (i_v28 - i_v43)
    return jnp.array([i_v28, i_v30, i_v47, i_v50, i_v52, i_v54, i_v56, i_v59, i_v61, i_v63, i_v65, i_v68, i_v72, i_v79])

def _RingMod_combined(signals: Signals, s: States, init, lambda_res_nm: float=1310.0, q_i: float=60000.0, q_e: float=60000.0, dl_dv_pm: float=30.0, lambda_nm: float=1310.0, cj: float=0.0, rleak: float=100000000.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v28 = init[0]
    i_v30 = init[1]
    i_v47 = init[2]
    i_v50 = init[3]
    i_v52 = init[4]
    i_v54 = init[5]
    i_v56 = init[6]
    i_v63 = init[9]
    i_v65 = init[10]
    i_v68 = init[11]
    v88 = jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 0.0, jnp.divide(signals.vp - signals.vn, jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 1.0, rleak)))
    v101 = cj * (signals.vp - signals.vn)
    v43 = dl_dv_pm * 1e-12
    v46 = lambda_res_nm * 1e-09 + v43 * (signals.vp - signals.vn)
    v62 = i_v28 * (1883651567.3088531 * (jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 0.0, jnp.divide(1.0, jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 1.0, lambda_nm * 1e-09))) - jnp.where((v46 == 0.0) | ~jnp.isfinite(v46), 0.0, jnp.divide(1.0, jnp.where((v46 == 0.0) | ~jnp.isfinite(v46), 1.0, v46)))))
    v68 = s.v_are - signals.gnd + v62 * (s.v_aim - signals.gnd) + i_v30 * (signals.in_im - signals.gnd)
    v78 = s.v_aim - signals.gnd - v62 * (s.v_are - signals.gnd) - i_v30 * (signals.in_re - signals.gnd)
    v99 = i_v28 * (s.v_are - signals.gnd)
    v100 = i_v28 * (s.v_aim - signals.gnd)
    v134 = jnp.where((v46 * v46 == 0.0) | ~jnp.isfinite(v46 * v46), 0.0, jnp.divide(v43, jnp.where((v46 * v46 == 0.0) | ~jnp.isfinite(v46 * v46), 1.0, v46 * v46))) * 1883651567.3088531 * i_v28
    v135 = v134 * (s.v_aim - signals.gnd)
    v159 = -v135
    v144 = v134 * (s.v_are - signals.gnd)
    v145 = 0.0 - v144
    v164 = v159 - v145
    v146 = 0.0 - v62
    v165 = -1.0 - v146
    v166 = -v62 - 1.0
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, i_v47, i_v52, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v52, i_v47, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v63, init[8], 0.0, 0.0, _mfactor * v164, _mfactor * -v164, _mfactor * (-v165 - v166 - -i_v30 - i_v30), _mfactor * v165, _mfactor * v166, i_v65, i_v65], [0.0, i_v63, 0.0, 0.0, _mfactor * v135, _mfactor * v159, _mfactor * (-1.0 - v62 - i_v30), _mfactor, _mfactor * v62, 0.0, 0.0], [init[12], 0.0, 0.0, 0.0, _mfactor * v145, _mfactor * v144, _mfactor * (v62 - 1.0 - (0.0 - i_v30)), _mfactor * v146, _mfactor, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v50, i_v54, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v54, i_v50, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[13], i_v56, init[7], 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v56, i_v68, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v56, 0.0, i_v68, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'out_re': _mfactor * s.i_br5, 'out_im': _mfactor * s.i_br6, 'vp': _mfactor * v88, 'vn': _mfactor * -v88, 'gnd': _mfactor * (-v68 - v78 - s.i_br5 - s.i_br6), 'v_are': _mfactor * v68, 'v_aim': _mfactor * v78, 'i_br5': signals.in_re - signals.gnd - (s.v_aim - signals.gnd) - (signals.out_re - signals.gnd), 'i_br6': signals.in_im - signals.gnd + s.v_are - signals.gnd - (signals.out_im - signals.gnd)}, {'vp': _mfactor * v101, 'vn': _mfactor * -v101, 'gnd': _mfactor * (-v99 - v100), 'v_are': _mfactor * v99, 'v_aim': _mfactor * v100}, j_resist, j_react)

def _RingMod_jacobian(signals: Signals, s: States, init, lambda_res_nm: float=1310.0, q_i: float=60000.0, q_e: float=60000.0, dl_dv_pm: float=30.0, lambda_nm: float=1310.0, cj: float=0.0, rleak: float=100000000.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _RingMod_combined(signals, s, init, lambda_res_nm=lambda_res_nm, q_i=q_i, q_e=q_e, dl_dv_pm=dl_dv_pm, lambda_nm=lambda_nm, cj=cj, rleak=rleak, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('in_re', 'in_im', 'out_re', 'out_im', 'vp', 'vn', 'gnd'), states=('v_are', 'v_aim', 'i_br5', 'i_br6'), jacobian_fn=_RingMod_jacobian, combined_fn=_RingMod_combined, differentiable_params=None)
def RingMod(signals: Signals, s: States, init, lambda_res_nm: float=1310.0, q_i: float=60000.0, q_e: float=60000.0, dl_dv_pm: float=30.0, lambda_nm: float=1310.0, cj: float=0.0, rleak: float=100000000.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _RingMod_combined(signals, s, init, lambda_res_nm=lambda_res_nm, q_i=q_i, q_e=q_e, dl_dv_pm=dl_dv_pm, lambda_nm=lambda_nm, cj=cj, rleak=rleak, _mfactor=_mfactor)
    return (f, q)

@RingMod.setup
def _RingMod_register_setup(*_a, **_kw):
    return _RingMod_setup(*_a, **_kw)