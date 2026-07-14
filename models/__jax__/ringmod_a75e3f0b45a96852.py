"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _RingMod_setup(radius_um: float=7.5, n_g: float=4.0, loss_db_m: float=7000.0, kappa2: float=0.1, cj_ff_um: float=0.5, lambda_res_nm: float=1310.0, dl_dv_pm: float=45.0, lambda_nm: float=1310.0, rleak: float=100000000.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v21 = jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 0.0, jnp.divide(299792458.0, jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 1.0, n_g)))
    i_v31 = 2.0 * jnp.divide(6.283185307179586 * radius_um * 1e-06, i_v21)
    i_v32 = jnp.where((i_v31 == 0.0) | ~jnp.isfinite(i_v31), 0.0, jnp.divide(kappa2, jnp.where((i_v31 == 0.0) | ~jnp.isfinite(i_v31), 1.0, i_v31)))
    i_v34 = jnp.divide(jnp.divide(loss_db_m * 2.302585092994046, 10.0) * i_v21, 2.0) + i_v32
    i_v35 = jnp.where((i_v34 == 0.0) | ~jnp.isfinite(i_v34), 0.0, jnp.divide(1.0, jnp.where((i_v34 == 0.0) | ~jnp.isfinite(i_v34), 1.0, i_v34)))
    i_v37 = i_v35 * 2.0 * i_v32
    i_v43 = cj_ff_um * 1e-15 * radius_um * 2.0 * 3.141592653589793
    i_v57 = jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 0.0, jnp.divide(1.0, jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 1.0, rleak)))
    i_v63 = _mfactor * i_v57
    i_v66 = _mfactor * i_v43
    i_v68 = _mfactor * -i_v57
    i_v70 = _mfactor * -i_v43
    i_v72 = _mfactor * -i_v35
    i_v60 = 0.0 - i_v35
    i_v75 = _mfactor * i_v60
    i_v77 = _mfactor * -i_v37
    i_v79 = _mfactor * i_v37
    i_v81 = _mfactor * -1.0
    i_v84 = _mfactor * i_v35
    i_v88 = _mfactor * (0.0 - i_v37)
    i_v95 = _mfactor * (i_v35 - i_v60)
    return jnp.array([i_v35, i_v37, i_v43, i_v63, i_v66, i_v68, i_v70, i_v72, i_v75, i_v77, i_v79, i_v81, i_v84, i_v88, i_v95])

def _RingMod_combined(signals: Signals, s: States, init, radius_um: float=7.5, n_g: float=4.0, loss_db_m: float=7000.0, kappa2: float=0.1, cj_ff_um: float=0.5, lambda_res_nm: float=1310.0, dl_dv_pm: float=45.0, lambda_nm: float=1310.0, rleak: float=100000000.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v35 = init[0]
    i_v37 = init[1]
    i_v63 = init[3]
    i_v66 = init[4]
    i_v68 = init[5]
    i_v70 = init[6]
    i_v72 = init[7]
    i_v79 = init[10]
    i_v81 = init[11]
    i_v84 = init[12]
    v104 = jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 0.0, jnp.divide(signals.vp - signals.vn, jnp.where((rleak == 0.0) | ~jnp.isfinite(rleak), 1.0, rleak)))
    v122 = init[2] * (signals.vp - signals.vn)
    v60 = dl_dv_pm * 1e-12
    v63 = lambda_res_nm * 1e-09 + v60 * (signals.vp - signals.vn)
    v79 = i_v35 * (1883651567.3088531 * (jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 0.0, jnp.divide(1.0, jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 1.0, lambda_nm * 1e-09))) - jnp.where((v63 == 0.0) | ~jnp.isfinite(v63), 0.0, jnp.divide(1.0, jnp.where((v63 == 0.0) | ~jnp.isfinite(v63), 1.0, v63)))))
    v85 = s.v_are - signals.gnd + v79 * (s.v_aim - signals.gnd) + i_v37 * (signals.in_im - signals.gnd)
    v95 = s.v_aim - signals.gnd - v79 * (s.v_are - signals.gnd) - i_v37 * (signals.in_re - signals.gnd)
    v120 = i_v35 * (s.v_are - signals.gnd)
    v121 = i_v35 * (s.v_aim - signals.gnd)
    v155 = jnp.where((v63 * v63 == 0.0) | ~jnp.isfinite(v63 * v63), 0.0, jnp.divide(v60, jnp.where((v63 * v63 == 0.0) | ~jnp.isfinite(v63 * v63), 1.0, v63 * v63))) * 1883651567.3088531 * i_v35
    v156 = v155 * (s.v_aim - signals.gnd)
    v180 = -v156
    v165 = v155 * (s.v_are - signals.gnd)
    v166 = 0.0 - v165
    v185 = v180 - v166
    v167 = 0.0 - v79
    v186 = -1.0 - v167
    v187 = -v79 - 1.0
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, i_v63, i_v68, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v68, i_v63, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v79, init[9], 0.0, 0.0, _mfactor * v185, _mfactor * -v185, _mfactor * (-v186 - v187 - -i_v37 - i_v37), _mfactor * v186, _mfactor * v187, i_v81, i_v81], [0.0, i_v79, 0.0, 0.0, _mfactor * v156, _mfactor * v180, _mfactor * (-1.0 - v79 - i_v37), _mfactor, _mfactor * v79, 0.0, 0.0], [init[13], 0.0, 0.0, 0.0, _mfactor * v166, _mfactor * v165, _mfactor * (v79 - 1.0 - (0.0 - i_v37)), _mfactor * v167, _mfactor, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v66, i_v70, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v70, i_v66, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[14], i_v72, init[8], 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v72, i_v84, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v72, 0.0, i_v84, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'out_re': _mfactor * s.i_br5, 'out_im': _mfactor * s.i_br6, 'vp': _mfactor * v104, 'vn': _mfactor * -v104, 'gnd': _mfactor * (-v85 - v95 - s.i_br5 - s.i_br6), 'v_are': _mfactor * v85, 'v_aim': _mfactor * v95, 'i_br5': signals.in_re - signals.gnd - (s.v_aim - signals.gnd) - (signals.out_re - signals.gnd), 'i_br6': signals.in_im - signals.gnd + s.v_are - signals.gnd - (signals.out_im - signals.gnd)}, {'vp': _mfactor * v122, 'vn': _mfactor * -v122, 'gnd': _mfactor * (-v120 - v121), 'v_are': _mfactor * v120, 'v_aim': _mfactor * v121}, j_resist, j_react)

def _RingMod_jacobian(signals: Signals, s: States, init, radius_um: float=7.5, n_g: float=4.0, loss_db_m: float=7000.0, kappa2: float=0.1, cj_ff_um: float=0.5, lambda_res_nm: float=1310.0, dl_dv_pm: float=45.0, lambda_nm: float=1310.0, rleak: float=100000000.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _RingMod_combined(signals, s, init, radius_um=radius_um, n_g=n_g, loss_db_m=loss_db_m, kappa2=kappa2, cj_ff_um=cj_ff_um, lambda_res_nm=lambda_res_nm, dl_dv_pm=dl_dv_pm, lambda_nm=lambda_nm, rleak=rleak, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('in_re', 'in_im', 'out_re', 'out_im', 'vp', 'vn', 'gnd'), states=('v_are', 'v_aim', 'i_br5', 'i_br6'), jacobian_fn=_RingMod_jacobian, combined_fn=_RingMod_combined, differentiable_params=None)
def RingMod(signals: Signals, s: States, init, radius_um: float=7.5, n_g: float=4.0, loss_db_m: float=7000.0, kappa2: float=0.1, cj_ff_um: float=0.5, lambda_res_nm: float=1310.0, dl_dv_pm: float=45.0, lambda_nm: float=1310.0, rleak: float=100000000.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _RingMod_combined(signals, s, init, radius_um=radius_um, n_g=n_g, loss_db_m=loss_db_m, kappa2=kappa2, cj_ff_um=cj_ff_um, lambda_res_nm=lambda_res_nm, dl_dv_pm=dl_dv_pm, lambda_nm=lambda_nm, rleak=rleak, _mfactor=_mfactor)
    return (f, q)

@RingMod.setup
def _RingMod_register_setup(*_a, **_kw):
    return _RingMod_setup(*_a, **_kw)