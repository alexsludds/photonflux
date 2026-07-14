"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _WaveguideNl_setup(length_um: float=1000.0, a_eff_um2: float=0.1, loss_db_m: float=200.0, lambda_nm: float=1310.0, beta_tpa: float=8e-12, Nscl: float=1e+24, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, dn_dn: float=-4e-27, n2_kerr: float=4.5e-18, _mfactor: float=1.0) -> jnp.ndarray:
    i_v16 = length_um * 1e-06
    i_v19 = a_eff_um2 * 1e-12
    i_v25 = jnp.divide(loss_db_m * 2.302585092994046, 10.0)
    i_v30 = jnp.where((i_v25 == 0.0) | ~jnp.isfinite(i_v25), 0.0, jnp.divide(1.0 - jnp.exp(-i_v25 * i_v16 - (-i_v25 * i_v16).real + jnp.clip((-i_v25 * i_v16).real, -709.0, 709.0)), jnp.where((i_v25 == 0.0) | ~jnp.isfinite(i_v25), 1.0, i_v25)))
    i_v31 = lambda_nm * 1e-09
    i_v34 = jnp.where((i_v31 == 0.0) | ~jnp.isfinite(i_v31), 0.0, jnp.divide(1.9864458571489286e-25, jnp.where((i_v31 == 0.0) | ~jnp.isfinite(i_v31), 1.0, i_v31)))
    i_v56 = _mfactor * -tau_fc
    i_v59 = _mfactor * -1.0
    i_v62 = _mfactor * tau_fc
    return jnp.array([i_v16, i_v19, i_v25, i_v30, i_v34, i_v56, i_v59, i_v62])

def _WaveguideNl_combined(signals: Signals, s: States, init, length_um: float=1000.0, a_eff_um2: float=0.1, loss_db_m: float=200.0, lambda_nm: float=1310.0, beta_tpa: float=8e-12, Nscl: float=1e+24, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, dn_dn: float=-4e-27, n2_kerr: float=4.5e-18, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v16 = init[0]
    i_v19 = init[1]
    i_v30 = init[3]
    i_v56 = init[5]
    i_v59 = init[6]
    i_v62 = init[7]
    v49 = jnp.where((i_v19 == 0.0) | ~jnp.isfinite(i_v19), 0.0, jnp.divide((signals.in_re - signals.gnd) * (signals.in_re - signals.gnd) + (signals.in_im - signals.gnd) * (signals.in_im - signals.gnd), jnp.where((i_v19 == 0.0) | ~jnp.isfinite(i_v19), 1.0, i_v19)))
    v31 = jnp.exp(-init[2] * i_v16 - (-init[2] * i_v16).real + jnp.clip((-init[2] * i_v16).real, -709.0, 709.0))
    v57 = 1.0 + beta_tpa * v49 * i_v30
    v58 = jnp.where((v57 == 0.0) | ~jnp.isfinite(v57), 0.0, jnp.divide(v31, jnp.where((v57 == 0.0) | ~jnp.isfinite(v57), 1.0, v57)))
    v73 = v49 * v58
    v65 = -sigma_fca
    v62 = (s.v_nc - signals.gnd) * Nscl
    v68 = jnp.exp(v65 * v62 * i_v30 - (v65 * v62 * i_v30).real + jnp.clip((v65 * v62 * i_v30).real, -709.0, 709.0))
    v78 = 0.5 * (v49 + v73 * v68)
    v80 = beta_tpa * v78
    v82 = 2.0 * init[4]
    v92 = s.v_nc - signals.gnd - jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(tau_fc * jnp.divide(v80 * v78, v82), jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v130 = tau_fc * (s.v_nc - signals.gnd)
    v71 = jnp.sqrt(jnp.maximum(v58 * v68, 1e-300))
    v40 = lambda_nm * 1e-09
    v97 = jnp.where((v40 == 0.0) | ~jnp.isfinite(v40), 0.0, jnp.divide(-6.283185307179586, jnp.where((v40 == 0.0) | ~jnp.isfinite(v40), 1.0, v40)))
    v104 = v97 * (n2_kerr * v78 + dn_dn * v62) * i_v16
    v107 = jnp.cos(v104)
    v208 = jnp.sin(v104)
    v111 = v107 * (signals.in_re - signals.gnd) - v208 * (signals.in_im - signals.gnd)
    v118 = v208 * (signals.in_re - signals.gnd) + v107 * (signals.in_im - signals.gnd)
    v145 = signals.in_re - signals.gnd + signals.in_re - signals.gnd
    v150 = jnp.where((i_v19 == 0.0) | ~jnp.isfinite(i_v19), 0.0, jnp.divide(1.0, jnp.where((i_v19 == 0.0) | ~jnp.isfinite(i_v19), 1.0, i_v19)))
    v157 = 0.0 - jnp.where((v57 * v57 == 0.0) | ~jnp.isfinite(v57 * v57), 0.0, jnp.divide(v150 * beta_tpa * i_v30 * v31, jnp.where((v57 * v57 == 0.0) | ~jnp.isfinite(v57 * v57), 1.0, v57 * v57)))
    v177 = (v150 + (v150 * v58 + v157 * v49) * v68) * 0.5
    v194 = jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(jnp.divide(v177 * beta_tpa * v78 + v177 * v80, v82) * tau_fc, jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v279 = v145 * v194
    v146 = signals.in_im - signals.gnd + signals.in_im - signals.gnd
    v280 = v146 * v194
    v161 = Nscl * v65 * i_v30 * v68
    v176 = v161 * v73 * 0.5
    v195 = 1.0 - jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(jnp.divide(v176 * beta_tpa * v78 + v176 * v80, v82) * tau_fc, jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v261 = -v195
    v196 = 0.0 - v194
    v197 = v145 * v196
    v198 = v146 * v196
    v164 = 2.0 * v71
    v166 = jnp.divide(v157 * v68, v164)
    v167 = v145 * v166
    v207 = v177 * n2_kerr * v97 * i_v16
    v209 = -v208
    v211 = v207 * v209
    v212 = v145 * v211
    v220 = v207 * v107
    v221 = v145 * v220
    v232 = v167 * v111 + (v212 * (signals.in_re - signals.gnd) + v107 - v221 * (signals.in_im - signals.gnd)) * v71
    v168 = v146 * v166
    v213 = v146 * v211
    v222 = v146 * v220
    v235 = v168 * v111 + (v213 * (signals.in_re - signals.gnd) - (v222 * (signals.in_im - signals.gnd) + v208)) * v71
    v165 = jnp.divide(v161 * v58, v164)
    v206 = (v176 * n2_kerr + Nscl * dn_dn) * v97 * i_v16
    v210 = v206 * v209
    v219 = v206 * v107
    v238 = v165 * v111 + (v210 * (signals.in_re - signals.gnd) - v219 * (signals.in_im - signals.gnd)) * v71
    v252 = v167 * v118 + (v221 * (signals.in_re - signals.gnd) + v208 + v212 * (signals.in_im - signals.gnd)) * v71
    v255 = v168 * v118 + (v222 * (signals.in_re - signals.gnd) + (v213 * (signals.in_im - signals.gnd) + v107)) * v71
    v258 = v165 * v118 + (v219 * (signals.in_re - signals.gnd) + v210 * (signals.in_im - signals.gnd)) * v71
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [_mfactor * v279, _mfactor * v280, 0.0, 0.0, _mfactor * (-v279 - v280 - v261), _mfactor * v261, i_v59, i_v59], [_mfactor * v197, _mfactor * v198, 0.0, 0.0, _mfactor * (-v197 - v198 - v195), _mfactor * v195, 0.0, 0.0], [v232, v235, -1.0, 0.0, -v232 - v235 - v238 - -1.0, v238, 0.0, 0.0], [v252, v255, 0.0, -1.0, -v252 - v255 - v258 - -1.0, v258, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v62, i_v56, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, i_v56, i_v62, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'out_re': _mfactor * s.i_br3, 'out_im': _mfactor * s.i_br4, 'gnd': _mfactor * (-v92 - s.i_br3 - s.i_br4), 'v_nc': _mfactor * v92, 'i_br3': v71 * v111 - (signals.out_re - signals.gnd), 'i_br4': v71 * v118 - (signals.out_im - signals.gnd)}, {'gnd': _mfactor * -v130, 'v_nc': _mfactor * v130}, j_resist, j_react)

def _WaveguideNl_jacobian(signals: Signals, s: States, init, length_um: float=1000.0, a_eff_um2: float=0.1, loss_db_m: float=200.0, lambda_nm: float=1310.0, beta_tpa: float=8e-12, Nscl: float=1e+24, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, dn_dn: float=-4e-27, n2_kerr: float=4.5e-18, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _WaveguideNl_combined(signals, s, init, length_um=length_um, a_eff_um2=a_eff_um2, loss_db_m=loss_db_m, lambda_nm=lambda_nm, beta_tpa=beta_tpa, Nscl=Nscl, sigma_fca=sigma_fca, tau_fc=tau_fc, dn_dn=dn_dn, n2_kerr=n2_kerr, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('in_re', 'in_im', 'out_re', 'out_im', 'gnd'), states=('v_nc', 'i_br3', 'i_br4'), jacobian_fn=_WaveguideNl_jacobian, combined_fn=_WaveguideNl_combined, differentiable_params=None)
def WaveguideNl(signals: Signals, s: States, init, length_um: float=1000.0, a_eff_um2: float=0.1, loss_db_m: float=200.0, lambda_nm: float=1310.0, beta_tpa: float=8e-12, Nscl: float=1e+24, sigma_fca: float=1.45e-21, tau_fc: float=1e-09, dn_dn: float=-4e-27, n2_kerr: float=4.5e-18, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _WaveguideNl_combined(signals, s, init, length_um=length_um, a_eff_um2=a_eff_um2, loss_db_m=loss_db_m, lambda_nm=lambda_nm, beta_tpa=beta_tpa, Nscl=Nscl, sigma_fca=sigma_fca, tau_fc=tau_fc, dn_dn=dn_dn, n2_kerr=n2_kerr, _mfactor=_mfactor)
    return (f, q)

@WaveguideNl.setup
def _WaveguideNl_register_setup(*_a, **_kw):
    return _WaveguideNl_setup(*_a, **_kw)