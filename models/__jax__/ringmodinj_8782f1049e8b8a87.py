"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _RingModInj_setup(Rs: float=50.0, cj_ff_um: float=1.0, radius_um: float=7.5, tau_c: float=1e-09, n_g: float=4.0, loss_db_m: float=3000.0, kappa2: float=0.05, lambda_res_nm: float=1310.0, dl_di_pm_ma: float=50.0, lambda_nm: float=1310.0, Von: float=0.9, fca_db_m_ma: float=400.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v25 = cj_ff_um * 1e-15 * radius_um * 2.0 * 3.141592653589793
    i_v31 = jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 0.0, jnp.divide(299792458.0, jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 1.0, n_g)))
    i_v41 = 2.0 * jnp.divide(6.283185307179586 * radius_um * 1e-06, i_v31)
    i_v42 = jnp.where((i_v41 == 0.0) | ~jnp.isfinite(i_v41), 0.0, jnp.divide(kappa2, jnp.where((i_v41 == 0.0) | ~jnp.isfinite(i_v41), 1.0, i_v41)))
    i_v44 = jnp.divide(jnp.divide(loss_db_m * 2.302585092994046, 10.0) * i_v31, 2.0) + i_v42
    i_v45 = jnp.where((i_v44 == 0.0) | ~jnp.isfinite(i_v44), 0.0, jnp.divide(1.0, jnp.where((i_v44 == 0.0) | ~jnp.isfinite(i_v44), 1.0, i_v44)))
    i_v47 = i_v45 * 2.0 * i_v42
    i_v62 = _mfactor * i_v25
    i_v65 = _mfactor * -i_v25
    i_v67 = _mfactor * -tau_c
    i_v61 = 0.0 - i_v45
    i_v70 = _mfactor * i_v61
    i_v74 = _mfactor * (0.0 - i_v47)
    i_v76 = _mfactor * i_v47
    i_v78 = _mfactor * -1.0
    i_v82 = _mfactor * i_v45
    i_v87 = _mfactor * tau_c
    i_v92 = _mfactor * (tau_c - i_v61 - i_v61)
    i_v94 = _mfactor * -i_v45
    return jnp.array([i_v25, i_v31, i_v42, i_v45, i_v47, i_v62, i_v65, i_v67, i_v70, i_v74, i_v76, i_v78, i_v82, i_v87, i_v92, i_v94])

def _RingModInj_combined(signals: Signals, s: States, init, Rs: float=50.0, cj_ff_um: float=1.0, radius_um: float=7.5, tau_c: float=1e-09, n_g: float=4.0, loss_db_m: float=3000.0, kappa2: float=0.05, lambda_res_nm: float=1310.0, dl_di_pm_ma: float=50.0, lambda_nm: float=1310.0, Von: float=0.9, fca_db_m_ma: float=400.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v31 = init[1]
    i_v45 = init[3]
    i_v47 = init[4]
    i_v62 = init[5]
    i_v65 = init[6]
    i_v67 = init[7]
    i_v70 = init[8]
    i_v74 = init[9]
    i_v76 = init[10]
    i_v78 = init[11]
    i_v82 = init[12]
    i_v94 = init[15]
    v20 = jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(signals.vp - signals.vn - Von, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs)))
    v22 = v20 < 0.0
    v33 = jnp.where(v22, 0.0, v20)
    v203 = init[0] * (signals.vp - signals.vn)
    v50 = s.v_ni - signals.gnd - v33 * 1000.0
    v54 = s.v_ni - signals.gnd > 0.0
    v56 = jnp.where(v54, s.v_ni - signals.gnd, 0.0)
    v120 = i_v45 * (jnp.divide(jnp.divide((loss_db_m + fca_db_m_ma * v56) * 2.302585092994046, 10.0) * i_v31, 2.0) + init[2])
    v103 = dl_di_pm_ma * 1e-12
    v105 = lambda_res_nm * 1e-09 - v103 * v56
    v123 = i_v45 * (1883651567.3088531 * (jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 0.0, jnp.divide(1.0, jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 1.0, lambda_nm * 1e-09))) - jnp.where((v105 == 0.0) | ~jnp.isfinite(v105), 0.0, jnp.divide(1.0, jnp.where((v105 == 0.0) | ~jnp.isfinite(v105), 1.0, v105)))))
    v129 = v120 * (s.v_are - signals.gnd) + v123 * (s.v_aim - signals.gnd) + i_v47 * (signals.in_im - signals.gnd)
    v147 = v120 * (s.v_aim - signals.gnd) - v123 * (s.v_are - signals.gnd) - i_v47 * (signals.in_re - signals.gnd)
    v204 = tau_c * (s.v_ni - signals.gnd)
    v205 = i_v45 * (s.v_are - signals.gnd)
    v206 = i_v45 * (s.v_aim - signals.gnd)
    v239 = jnp.where(v22, 0.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(1.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs))))
    v424 = _mfactor * v239
    v430 = _mfactor * -v239
    v242 = v239 * 1000.0
    v438 = _mfactor * v242
    v245 = jnp.where(v54, 1.0, 0.0)
    v263 = jnp.divide(jnp.divide(v245 * fca_db_m_ma * 2.302585092994046, 10.0) * i_v31, 2.0) * i_v45
    v267 = jnp.where((v105 * v105 == 0.0) | ~jnp.isfinite(v105 * v105), 0.0, jnp.divide(0.0 - v245 * v103, jnp.where((v105 * v105 == 0.0) | ~jnp.isfinite(v105 * v105), 1.0, v105 * v105))) * 1883651567.3088531 * i_v45
    v269 = v263 * (s.v_are - signals.gnd) + v267 * (s.v_aim - signals.gnd)
    v280 = v263 * (s.v_aim - signals.gnd) - v267 * (s.v_are - signals.gnd)
    v304 = -1.0 - v269 - v280
    v281 = 0.0 - v123
    v305 = 0.0 - v120 - v281
    v306 = v281 - v120
    v286 = 0.0 - i_v47
    v458 = _mfactor * v120
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, v424, v430, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, v430, v424, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v76, i_v74, 0.0, 0.0, v438, _mfactor * -v242, _mfactor * (-v304 - v305 - v306 - v286 - i_v47), _mfactor * v305, _mfactor * v306, _mfactor * v304, i_v78, i_v78], [0.0, i_v76, 0.0, 0.0, 0.0, 0.0, _mfactor * (-v269 - v120 - v123 - i_v47), v458, _mfactor * v123, _mfactor * v269, 0.0, 0.0], [i_v74, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * (-v280 - v281 - v120 - v286), _mfactor * v281, v458, _mfactor * v280, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, _mfactor * (0.0 - v242), v438, i_v78, 0.0, 0.0, _mfactor, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v62, i_v65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v65, i_v62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[14], i_v70, i_v70, i_v67, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v94, i_v82, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v94, 0.0, i_v82, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v67, 0.0, 0.0, init[13], 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'out_re': _mfactor * s.i_br6, 'out_im': _mfactor * s.i_br7, 'vp': _mfactor * v33, 'vn': _mfactor * -v33, 'gnd': _mfactor * (-v50 - v129 - v147 - s.i_br6 - s.i_br7), 'v_are': _mfactor * v129, 'v_aim': _mfactor * v147, 'v_ni': _mfactor * v50, 'i_br6': signals.in_re - signals.gnd - (s.v_aim - signals.gnd) - (signals.out_re - signals.gnd), 'i_br7': signals.in_im - signals.gnd + s.v_are - signals.gnd - (signals.out_im - signals.gnd)}, {'vp': _mfactor * v203, 'vn': _mfactor * -v203, 'gnd': _mfactor * (-v204 - v205 - v206), 'v_are': _mfactor * v205, 'v_aim': _mfactor * v206, 'v_ni': _mfactor * v204}, j_resist, j_react)

def _RingModInj_jacobian(signals: Signals, s: States, init, Rs: float=50.0, cj_ff_um: float=1.0, radius_um: float=7.5, tau_c: float=1e-09, n_g: float=4.0, loss_db_m: float=3000.0, kappa2: float=0.05, lambda_res_nm: float=1310.0, dl_di_pm_ma: float=50.0, lambda_nm: float=1310.0, Von: float=0.9, fca_db_m_ma: float=400.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _RingModInj_combined(signals, s, init, Rs=Rs, cj_ff_um=cj_ff_um, radius_um=radius_um, tau_c=tau_c, n_g=n_g, loss_db_m=loss_db_m, kappa2=kappa2, lambda_res_nm=lambda_res_nm, dl_di_pm_ma=dl_di_pm_ma, lambda_nm=lambda_nm, Von=Von, fca_db_m_ma=fca_db_m_ma, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('in_re', 'in_im', 'out_re', 'out_im', 'vp', 'vn', 'gnd'), states=('v_are', 'v_aim', 'v_ni', 'i_br6', 'i_br7'), jacobian_fn=_RingModInj_jacobian, combined_fn=_RingModInj_combined, differentiable_params=None)
def RingModInj(signals: Signals, s: States, init, Rs: float=50.0, cj_ff_um: float=1.0, radius_um: float=7.5, tau_c: float=1e-09, n_g: float=4.0, loss_db_m: float=3000.0, kappa2: float=0.05, lambda_res_nm: float=1310.0, dl_di_pm_ma: float=50.0, lambda_nm: float=1310.0, Von: float=0.9, fca_db_m_ma: float=400.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _RingModInj_combined(signals, s, init, Rs=Rs, cj_ff_um=cj_ff_um, radius_um=radius_um, tau_c=tau_c, n_g=n_g, loss_db_m=loss_db_m, kappa2=kappa2, lambda_res_nm=lambda_res_nm, dl_di_pm_ma=dl_di_pm_ma, lambda_nm=lambda_nm, Von=Von, fca_db_m_ma=fca_db_m_ma, _mfactor=_mfactor)
    return (f, q)

@RingModInj.setup
def _RingModInj_register_setup(*_a, **_kw):
    return _RingModInj_setup(*_a, **_kw)