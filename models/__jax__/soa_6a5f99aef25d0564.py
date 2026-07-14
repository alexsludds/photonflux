"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _Soa_setup(Rs: float=3.0, g0_db: float=20.0, i_tr_ma: float=8.0, i_op_ma: float=80.0, tau_c: float=3e-10, p_sat: float=0.01, alpha_h: float=0.0, p_seed: float=1e-09, tau_bw: float=1e-12, Von: float=1.2, h_clamp: float=12.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v22 = jnp.divide(g0_db * 2.302585092994046, 10.0)
    i_v32 = jnp.sqrt(jnp.maximum(p_seed, 1e-300))
    i_v39 = _mfactor * tau_bw
    i_v45 = _mfactor * -tau_c
    i_v48 = _mfactor * -1.0
    i_v36 = 0.0 - tau_bw
    i_v50 = _mfactor * i_v36
    i_v61 = _mfactor * tau_c
    i_v64 = _mfactor * -tau_bw
    i_v69 = _mfactor * (tau_c - i_v36 - i_v36 - i_v36 - i_v36)
    return jnp.array([i_v22, i_v32, i_v39, i_v45, i_v48, i_v50, i_v61, i_v64, i_v69])

def _Soa_combined(signals: Signals, s: States, init, Rs: float=3.0, g0_db: float=20.0, i_tr_ma: float=8.0, i_op_ma: float=80.0, tau_c: float=3e-10, p_sat: float=0.01, alpha_h: float=0.0, p_seed: float=1e-09, tau_bw: float=1e-12, Von: float=1.2, h_clamp: float=12.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v22 = init[0]
    i_v32 = init[1]
    i_v39 = init[2]
    i_v45 = init[3]
    i_v48 = init[4]
    i_v50 = init[5]
    i_v64 = init[7]
    v46 = s.v_nh - signals.gnd > h_clamp
    v48 = jnp.where(v46, h_clamp, s.v_nh - signals.gnd)
    v50 = jnp.exp(v48 - v48.real + jnp.clip(v48.real, -709.0, 709.0))
    v84 = jnp.sqrt(jnp.maximum(v50, 1e-300))
    v89 = -0.5 * alpha_h
    v90 = v89 * v48
    v92 = jnp.cos(v90)
    v93 = v84 * v92
    v263 = jnp.sin(v90)
    v96 = v84 * v263
    v111 = signals.fo_re - signals.gnd - (v93 * (signals.fi_re - signals.gnd) - v96 * (signals.fi_im - signals.gnd)) - i_v32
    v182 = tau_bw * (signals.fo_re - signals.gnd)
    v126 = signals.fo_im - signals.gnd - (v96 * (signals.fi_re - signals.gnd) + v93 * (signals.fi_im - signals.gnd))
    v183 = tau_bw * (signals.fo_im - signals.gnd)
    v142 = signals.bo_re - signals.gnd - (v93 * (signals.bi_re - signals.gnd) - v96 * (signals.bi_im - signals.gnd)) - i_v32
    v184 = tau_bw * (signals.bo_re - signals.gnd)
    v157 = signals.bo_im - signals.gnd - (v96 * (signals.bi_re - signals.gnd) + v93 * (signals.bi_im - signals.gnd))
    v185 = tau_bw * (signals.bo_im - signals.gnd)
    v20 = jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(signals.an - signals.cat - Von, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs)))
    v22 = v20 < 0.0
    v24 = jnp.where(v22, 0.0, v20)
    v40 = i_op_ma - i_tr_ma
    v73 = v50 - 1.0
    v62 = (signals.fi_re - signals.gnd) * (signals.fi_re - signals.gnd) + (signals.fi_im - signals.gnd) * (signals.fi_im - signals.gnd) + (signals.bi_re - signals.gnd) * (signals.bi_re - signals.gnd) + (signals.bi_im - signals.gnd) * (signals.bi_im - signals.gnd)
    v77 = s.v_nh - signals.gnd - jnp.where((v40 == 0.0) | ~jnp.isfinite(v40), 0.0, jnp.divide(i_v22 * (v24 * 1000.0 - i_tr_ma), jnp.where((v40 == 0.0) | ~jnp.isfinite(v40), 1.0, v40))) + jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 0.0, jnp.divide(v73 * v62, jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 1.0, p_sat)))
    v181 = tau_c * (s.v_nh - signals.gnd)
    v281 = 0.0 - v93
    v484 = _mfactor * v281
    v485 = _mfactor * v96
    v223 = jnp.where(v46, 0.0, 1.0)
    v225 = v223 * v50
    v261 = jnp.divide(v225, 2.0 * v84)
    v262 = v223 * v89
    v268 = v261 * v92 + v262 * -v263 * v84
    v273 = v261 * v263 + v262 * v92 * v84
    v277 = v268 * (signals.fi_re - signals.gnd) - v273 * (signals.fi_im - signals.gnd)
    v280 = 0.0 - v277
    v279 = 0.0 - v96
    v493 = _mfactor * v279
    v291 = v273 * (signals.fi_re - signals.gnd) + v268 * (signals.fi_im - signals.gnd)
    v294 = 0.0 - v291
    v301 = v268 * (signals.bi_re - signals.gnd) - v273 * (signals.bi_im - signals.gnd)
    v304 = 0.0 - v301
    v315 = v273 * (signals.bi_re - signals.gnd) + v268 * (signals.bi_im - signals.gnd)
    v318 = 0.0 - v315
    v217 = jnp.where(v22, 0.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(1.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs))))
    v520 = _mfactor * v217
    v523 = _mfactor * -v217
    v250 = jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 0.0, jnp.divide((signals.fi_re - signals.gnd + signals.fi_re - signals.gnd) * v73, jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 1.0, p_sat)))
    v341 = -v250 - v281 - v279
    v251 = jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 0.0, jnp.divide((signals.fi_im - signals.gnd + signals.fi_im - signals.gnd) * v73, jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 1.0, p_sat)))
    v342 = -v251 - v96 - v281
    v252 = jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 0.0, jnp.divide((signals.bi_re - signals.gnd + signals.bi_re - signals.gnd) * v73, jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 1.0, p_sat)))
    v367 = -v252 - v281 - v279
    v253 = jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 0.0, jnp.divide((signals.bi_im - signals.gnd + signals.bi_im - signals.gnd) * v73, jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 1.0, p_sat)))
    v368 = -v253 - v96 - v281
    v222 = jnp.where((v40 == 0.0) | ~jnp.isfinite(v40), 0.0, jnp.divide(v217 * 1000.0 * i_v22, jnp.where((v40 == 0.0) | ~jnp.isfinite(v40), 1.0, v40)))
    v538 = _mfactor * v222
    v255 = 1.0 + jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 0.0, jnp.divide(v225 * v62, jnp.where((p_sat == 0.0) | ~jnp.isfinite(p_sat), 1.0, p_sat)))
    v324 = -v255
    v364 = v324 - v280 - v294 - v304 - v318
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [v484, v485, _mfactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * (v277 - v281 - v96 - 1.0), _mfactor * v280], [v493, v484, 0.0, _mfactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * (v291 - v279 - v281 - 1.0), _mfactor * v294], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, v484, v485, _mfactor, 0.0, 0.0, 0.0, _mfactor * (v301 - v281 - v96 - 1.0), _mfactor * v304], [0.0, 0.0, 0.0, 0.0, v493, v484, 0.0, _mfactor, 0.0, 0.0, _mfactor * (v315 - v279 - v281 - 1.0), _mfactor * v318], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v520, v523, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v523, v520, 0.0, 0.0], [_mfactor * v341, _mfactor * v342, i_v48, i_v48, _mfactor * v367, _mfactor * v368, i_v48, i_v48, v538, _mfactor * -v222, _mfactor * (-v364 - v341 - v342 - v367 - v368 - -1.0 - -1.0 - -1.0 - -1.0), _mfactor * v364], [_mfactor * v250, _mfactor * v251, 0.0, 0.0, _mfactor * v252, _mfactor * v253, 0.0, 0.0, _mfactor * (0.0 - v222), v538, _mfactor * (v324 - v250 - v251 - v252 - v253), _mfactor * v255]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, i_v39, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v64, 0.0], [0.0, 0.0, 0.0, i_v39, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v64, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v39, 0.0, 0.0, 0.0, i_v64, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v39, 0.0, 0.0, i_v64, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, i_v50, i_v50, 0.0, 0.0, i_v50, i_v50, 0.0, 0.0, init[8], i_v45], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v45, init[6]]])
    return ({'fo_re': _mfactor * v111, 'fo_im': _mfactor * v126, 'bo_re': _mfactor * v142, 'bo_im': _mfactor * v157, 'an': _mfactor * v24, 'cat': _mfactor * -v24, 'gnd': _mfactor * (-v77 - v111 - v126 - v142 - v157), 'v_nh': _mfactor * v77}, {'fo_re': _mfactor * v182, 'fo_im': _mfactor * v183, 'bo_re': _mfactor * v184, 'bo_im': _mfactor * v185, 'gnd': _mfactor * (-v181 - v182 - v183 - v184 - v185), 'v_nh': _mfactor * v181}, j_resist, j_react)

def _Soa_jacobian(signals: Signals, s: States, init, Rs: float=3.0, g0_db: float=20.0, i_tr_ma: float=8.0, i_op_ma: float=80.0, tau_c: float=3e-10, p_sat: float=0.01, alpha_h: float=0.0, p_seed: float=1e-09, tau_bw: float=1e-12, Von: float=1.2, h_clamp: float=12.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _Soa_combined(signals, s, init, Rs=Rs, g0_db=g0_db, i_tr_ma=i_tr_ma, i_op_ma=i_op_ma, tau_c=tau_c, p_sat=p_sat, alpha_h=alpha_h, p_seed=p_seed, tau_bw=tau_bw, Von=Von, h_clamp=h_clamp, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('fi_re', 'fi_im', 'fo_re', 'fo_im', 'bi_re', 'bi_im', 'bo_re', 'bo_im', 'an', 'cat', 'gnd'), states=('v_nh',), jacobian_fn=_Soa_jacobian, combined_fn=_Soa_combined, differentiable_params=None)
def Soa(signals: Signals, s: States, init, Rs: float=3.0, g0_db: float=20.0, i_tr_ma: float=8.0, i_op_ma: float=80.0, tau_c: float=3e-10, p_sat: float=0.01, alpha_h: float=0.0, p_seed: float=1e-09, tau_bw: float=1e-12, Von: float=1.2, h_clamp: float=12.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _Soa_combined(signals, s, init, Rs=Rs, g0_db=g0_db, i_tr_ma=i_tr_ma, i_op_ma=i_op_ma, tau_c=tau_c, p_sat=p_sat, alpha_h=alpha_h, p_seed=p_seed, tau_bw=tau_bw, Von=Von, h_clamp=h_clamp, _mfactor=_mfactor)
    return (f, q)

@Soa.setup
def _Soa_register_setup(*_a, **_kw):
    return _Soa_setup(*_a, **_kw)