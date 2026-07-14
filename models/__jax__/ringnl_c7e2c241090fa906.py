"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _RingNl_setup(radius_um: float=10.0, n_g: float=4.0, a_eff_um2: float=0.1, lambda_nm: float=1310.0, loss_db_m: float=30.0, kappa2: float=0.0006, Nscl: float=1e+24, beta_tpa: float=8e-12, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, lambda_res_nm: float=1310.0, n2_kerr: float=4.5e-18, dn_dn: float=-4e-27, _mfactor: float=1.0) -> jnp.ndarray:
    i_v21 = jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 0.0, jnp.divide(299792458.0, jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 1.0, n_g)))
    i_v19 = 6.283185307179586 * radius_um * 1e-06
    i_v28 = a_eff_um2 * 1e-12 * i_v19
    i_v29 = lambda_nm * 1e-09
    i_v32 = jnp.where((i_v29 == 0.0) | ~jnp.isfinite(i_v29), 0.0, jnp.divide(1.9864458571489286e-25, jnp.where((i_v29 == 0.0) | ~jnp.isfinite(i_v29), 1.0, i_v29)))
    i_v34 = jnp.where((i_v29 == 0.0) | ~jnp.isfinite(i_v29), 0.0, jnp.divide(1883651567.3088531, jnp.where((i_v29 == 0.0) | ~jnp.isfinite(i_v29), 1.0, i_v29)))
    i_v41 = jnp.divide(jnp.divide(loss_db_m * 2.302585092994046, 10.0) * i_v21, 2.0)
    i_v42 = 2.0 * jnp.divide(i_v19, i_v21)
    i_v43 = jnp.where((i_v42 == 0.0) | ~jnp.isfinite(i_v42), 0.0, jnp.divide(kappa2, jnp.where((i_v42 == 0.0) | ~jnp.isfinite(i_v42), 1.0, i_v42)))
    i_v45 = i_v41 + i_v43
    i_v46 = jnp.where((i_v45 == 0.0) | ~jnp.isfinite(i_v45), 0.0, jnp.divide(1.0, jnp.where((i_v45 == 0.0) | ~jnp.isfinite(i_v45), 1.0, i_v45)))
    i_v47 = 2.0 * i_v43
    i_v48 = i_v46 * i_v47
    i_v75 = 1883651567.3088531 * (jnp.where((i_v29 == 0.0) | ~jnp.isfinite(i_v29), 0.0, jnp.divide(1.0, jnp.where((i_v29 == 0.0) | ~jnp.isfinite(i_v29), 1.0, i_v29))) - jnp.where((lambda_res_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_res_nm * 1e-09), 0.0, jnp.divide(1.0, jnp.where((lambda_res_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_res_nm * 1e-09), 1.0, lambda_res_nm * 1e-09))))
    i_v90 = 0.0 - i_v46
    i_v91 = _mfactor * i_v90
    i_v88 = -tau_fc
    i_v97 = _mfactor * i_v88
    i_v99 = _mfactor * (0.0 - i_v48)
    i_v101 = _mfactor * i_v48
    i_v103 = _mfactor * -1.0
    i_v107 = _mfactor * i_v46
    i_v112 = _mfactor * tau_fc
    i_v115 = _mfactor * (i_v46 - i_v90 - i_v88)
    i_v117 = _mfactor * -i_v46
    return jnp.array([i_v21, i_v28, i_v32, i_v34, i_v41, i_v43, i_v46, i_v47, i_v48, i_v75, i_v91, i_v97, i_v99, i_v101, i_v103, i_v107, i_v112, i_v115, i_v117])

def _RingNl_combined(signals: Signals, s: States, init, radius_um: float=10.0, n_g: float=4.0, a_eff_um2: float=0.1, lambda_nm: float=1310.0, loss_db_m: float=30.0, kappa2: float=0.0006, Nscl: float=1e+24, beta_tpa: float=8e-12, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, lambda_res_nm: float=1310.0, n2_kerr: float=4.5e-18, dn_dn: float=-4e-27, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v21 = init[0]
    i_v28 = init[1]
    i_v46 = init[6]
    i_v47 = init[7]
    i_v48 = init[8]
    i_v91 = init[10]
    i_v97 = init[11]
    i_v99 = init[12]
    i_v101 = init[13]
    i_v103 = init[14]
    i_v107 = init[15]
    i_v117 = init[18]
    v70 = jnp.where((i_v47 == 0.0) | ~jnp.isfinite(i_v47), 0.0, jnp.divide((s.v_are - signals.gnd) * (s.v_are - signals.gnd) + (s.v_aim - signals.gnd) * (s.v_aim - signals.gnd), jnp.where((i_v47 == 0.0) | ~jnp.isfinite(i_v47), 1.0, i_v47)))
    v73 = jnp.where((i_v28 == 0.0) | ~jnp.isfinite(i_v28), 0.0, jnp.divide(v70 * i_v21, jnp.where((i_v28 == 0.0) | ~jnp.isfinite(i_v28), 1.0, i_v28)))
    v95 = beta_tpa * v73
    v97 = 2.0 * init[2]
    v107 = s.v_nc - signals.gnd - jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(tau_fc * jnp.divide(v95 * v73, v97), jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v81 = beta_tpa * i_v21 * i_v21
    v83 = 2.0 * i_v28
    v77 = (s.v_nc - signals.gnd) * Nscl
    v130 = i_v46 * (init[4] + init[5] + jnp.where((v83 == 0.0) | ~jnp.isfinite(v83), 0.0, jnp.divide(v81 * v70, jnp.where((v83 == 0.0) | ~jnp.isfinite(v83), 1.0, v83))) + jnp.divide(sigma_fca * v77 * i_v21, 2.0))
    v118 = jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 0.0, jnp.divide(init[3], jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 1.0, n_g)))
    v133 = i_v46 * (init[9] + v118 * (n2_kerr * v73 + dn_dn * v77))
    v138 = v130 * (s.v_are - signals.gnd) + v133 * (s.v_aim - signals.gnd) + i_v48 * (signals.in_im - signals.gnd)
    v150 = v130 * (s.v_aim - signals.gnd) - v133 * (s.v_are - signals.gnd) - i_v48 * (signals.in_re - signals.gnd)
    v174 = tau_fc * (s.v_nc - signals.gnd)
    v175 = i_v46 * (s.v_are - signals.gnd)
    v176 = i_v46 * (s.v_aim - signals.gnd)
    v201 = s.v_are - signals.gnd + s.v_are - signals.gnd
    v206 = jnp.where((i_v47 == 0.0) | ~jnp.isfinite(i_v47), 0.0, jnp.divide(1.0, jnp.where((i_v47 == 0.0) | ~jnp.isfinite(i_v47), 1.0, i_v47)))
    v209 = jnp.where((i_v28 == 0.0) | ~jnp.isfinite(i_v28), 0.0, jnp.divide(v206 * i_v21, jnp.where((i_v28 == 0.0) | ~jnp.isfinite(i_v28), 1.0, i_v28)))
    v229 = jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(jnp.divide(v209 * beta_tpa * v73 + v209 * v95, v97) * tau_fc, jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v243 = jnp.where((v83 == 0.0) | ~jnp.isfinite(v83), 0.0, jnp.divide(v206 * v81, jnp.where((v83 == 0.0) | ~jnp.isfinite(v83), 1.0, v83))) * i_v46
    v244 = v201 * v243
    v254 = v209 * n2_kerr * v118 * i_v46
    v255 = v201 * v254
    v261 = v244 * (s.v_are - signals.gnd) + v130 + v255 * (s.v_aim - signals.gnd)
    v279 = v244 * (s.v_aim - signals.gnd) - (v255 * (s.v_are - signals.gnd) + v133)
    v303 = v201 * v229 - v261 - v279
    v202 = s.v_aim - signals.gnd + s.v_aim - signals.gnd
    v245 = v202 * v243
    v256 = v202 * v254
    v262 = v245 * (s.v_are - signals.gnd) + (v256 * (s.v_aim - signals.gnd) + v133)
    v280 = v245 * (s.v_aim - signals.gnd) + v130 - v256 * (s.v_are - signals.gnd)
    v304 = v202 * v229 - v262 - v280
    v242 = jnp.divide(Nscl * sigma_fca * i_v21, 2.0) * i_v46
    v253 = Nscl * dn_dn * v118 * i_v46
    v263 = v242 * (s.v_are - signals.gnd) + v253 * (s.v_aim - signals.gnd)
    v281 = v242 * (s.v_aim - signals.gnd) - v253 * (s.v_are - signals.gnd)
    v305 = -1.0 - v263 - v281
    v285 = 0.0 - i_v48
    v231 = 0.0 - v229
    v232 = v201 * v231
    v233 = v202 * v231
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [i_v101, i_v99, 0.0, 0.0, _mfactor * (-v303 - v304 - v305 - v285 - i_v48), _mfactor * v303, _mfactor * v304, _mfactor * v305, i_v103, i_v103], [0.0, i_v101, 0.0, 0.0, _mfactor * (-v261 - v262 - v263 - i_v48), _mfactor * v261, _mfactor * v262, _mfactor * v263, 0.0, 0.0], [i_v99, 0.0, 0.0, 0.0, _mfactor * (-v279 - v280 - v281 - v285), _mfactor * v279, _mfactor * v280, _mfactor * v281, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, _mfactor * (-v232 - v233 - 1.0), _mfactor * v232, _mfactor * v233, _mfactor, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, init[17], i_v91, i_v91, i_v97, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v117, i_v107, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v117, 0.0, i_v107, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v97, 0.0, 0.0, init[16], 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'out_re': _mfactor * s.i_br5, 'out_im': _mfactor * s.i_br6, 'gnd': _mfactor * (-v107 - v138 - v150 - s.i_br5 - s.i_br6), 'v_are': _mfactor * v138, 'v_aim': _mfactor * v150, 'v_nc': _mfactor * v107, 'i_br5': signals.in_re - signals.gnd - (s.v_aim - signals.gnd) - (signals.out_re - signals.gnd), 'i_br6': signals.in_im - signals.gnd + s.v_are - signals.gnd - (signals.out_im - signals.gnd)}, {'gnd': _mfactor * (-v174 - v175 - v176), 'v_are': _mfactor * v175, 'v_aim': _mfactor * v176, 'v_nc': _mfactor * v174}, j_resist, j_react)

def _RingNl_jacobian(signals: Signals, s: States, init, radius_um: float=10.0, n_g: float=4.0, a_eff_um2: float=0.1, lambda_nm: float=1310.0, loss_db_m: float=30.0, kappa2: float=0.0006, Nscl: float=1e+24, beta_tpa: float=8e-12, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, lambda_res_nm: float=1310.0, n2_kerr: float=4.5e-18, dn_dn: float=-4e-27, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _RingNl_combined(signals, s, init, radius_um=radius_um, n_g=n_g, a_eff_um2=a_eff_um2, lambda_nm=lambda_nm, loss_db_m=loss_db_m, kappa2=kappa2, Nscl=Nscl, beta_tpa=beta_tpa, sigma_fca=sigma_fca, tau_fc=tau_fc, lambda_res_nm=lambda_res_nm, n2_kerr=n2_kerr, dn_dn=dn_dn, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('in_re', 'in_im', 'out_re', 'out_im', 'gnd'), states=('v_are', 'v_aim', 'v_nc', 'i_br5', 'i_br6'), jacobian_fn=_RingNl_jacobian, combined_fn=_RingNl_combined, differentiable_params=None)
def RingNl(signals: Signals, s: States, init, radius_um: float=10.0, n_g: float=4.0, a_eff_um2: float=0.1, lambda_nm: float=1310.0, loss_db_m: float=30.0, kappa2: float=0.0006, Nscl: float=1e+24, beta_tpa: float=8e-12, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, lambda_res_nm: float=1310.0, n2_kerr: float=4.5e-18, dn_dn: float=-4e-27, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _RingNl_combined(signals, s, init, radius_um=radius_um, n_g=n_g, a_eff_um2=a_eff_um2, lambda_nm=lambda_nm, loss_db_m=loss_db_m, kappa2=kappa2, Nscl=Nscl, beta_tpa=beta_tpa, sigma_fca=sigma_fca, tau_fc=tau_fc, lambda_res_nm=lambda_res_nm, n2_kerr=n2_kerr, dn_dn=dn_dn, _mfactor=_mfactor)
    return (f, q)

@RingNl.setup
def _RingNl_register_setup(*_a, **_kw):
    return _RingNl_setup(*_a, **_kw)