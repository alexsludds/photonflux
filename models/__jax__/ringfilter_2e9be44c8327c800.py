"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _RingFilter_setup(radius_um: float=100.0, n_g: float=4.0, loss_db_m: float=100.0, kappa2_in: float=0.05, kappa2_drop: float=0.05, r_heater: float=500.0, lambda_res_nm: float=1310.0, dl_dmw_pm: float=20.0, lambda_nm: float=1310.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v21 = jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 0.0, jnp.divide(299792458.0, jnp.where((n_g == 0.0) | ~jnp.isfinite(n_g), 1.0, n_g)))
    i_v24 = jnp.divide(6.283185307179586 * radius_um * 1e-06, i_v21)
    i_v25 = jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 0.0, jnp.divide(6.283185307179586, jnp.where((i_v24 == 0.0) | ~jnp.isfinite(i_v24), 1.0, i_v24)))
    i_v32 = 2.0 * i_v24
    i_v33 = jnp.where((i_v32 == 0.0) | ~jnp.isfinite(i_v32), 0.0, jnp.divide(kappa2_in, jnp.where((i_v32 == 0.0) | ~jnp.isfinite(i_v32), 1.0, i_v32)))
    i_v38 = jnp.divide(jnp.divide(loss_db_m * 2.302585092994046, 10.0) * i_v21, 2.0) + i_v33 + jnp.where((i_v32 == 0.0) | ~jnp.isfinite(i_v32), 0.0, jnp.divide(kappa2_drop, jnp.where((i_v32 == 0.0) | ~jnp.isfinite(i_v32), 1.0, i_v32)))
    i_v39 = jnp.where((i_v38 == 0.0) | ~jnp.isfinite(i_v38), 0.0, jnp.divide(1.0, jnp.where((i_v38 == 0.0) | ~jnp.isfinite(i_v38), 1.0, i_v38)))
    i_v41 = i_v39 * 2.0 * i_v33
    i_v43 = jnp.sqrt(jnp.maximum(jnp.where((kappa2_in == 0.0) | ~jnp.isfinite(kappa2_in), 0.0, jnp.divide(kappa2_drop, jnp.where((kappa2_in == 0.0) | ~jnp.isfinite(kappa2_in), 1.0, kappa2_in))), 1e-300))
    i_v46 = jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 0.0, jnp.divide(1.0, jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 1.0, r_heater)))
    i_v71 = _mfactor * i_v46
    i_v74 = _mfactor * -i_v46
    i_v76 = _mfactor * -i_v39
    i_v62 = 0.0 - i_v39
    i_v79 = _mfactor * i_v62
    i_v81 = _mfactor * (-i_v41 - i_v41 - i_v41 - i_v41 - i_v41)
    i_v57 = 0.0 - i_v41
    i_v83 = _mfactor * (i_v41 - i_v57 - i_v57 - i_v57 - i_v57)
    i_v101 = _mfactor * -1.0
    i_v106 = _mfactor * i_v39
    i_v108 = _mfactor * i_v41
    i_v111 = _mfactor * i_v57
    i_v58 = -i_v43
    i_v153 = _mfactor * (i_v39 - i_v62 - i_v62 - i_v62 - i_v62 - i_v62 - i_v62 - i_v62 - i_v62 - i_v62)
    i_v139 = i_v43 - i_v58 - i_v58 - i_v58 - i_v58 - -1.0
    i_v149 = i_v58 - i_v43 - i_v43 - i_v43 - i_v43 - -1.0
    return jnp.array([i_v25, i_v39, i_v41, i_v43, i_v71, i_v74, i_v76, i_v79, i_v81, i_v83, i_v101, i_v106, i_v108, i_v111, i_v58, i_v153, i_v139, i_v149])

def _RingFilter_combined(signals: Signals, s: States, init, radius_um: float=100.0, n_g: float=4.0, loss_db_m: float=100.0, kappa2_in: float=0.05, kappa2_drop: float=0.05, r_heater: float=500.0, lambda_res_nm: float=1310.0, dl_dmw_pm: float=20.0, lambda_nm: float=1310.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v25 = init[0]
    i_v39 = init[1]
    i_v41 = init[2]
    i_v43 = init[3]
    i_v71 = init[4]
    i_v74 = init[5]
    i_v76 = init[6]
    i_v79 = init[7]
    i_v101 = init[10]
    i_v106 = init[11]
    i_v108 = init[12]
    i_v111 = init[13]
    i_v58 = init[14]
    v65 = jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 0.0, jnp.divide(signals.hp - signals.hn, jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 1.0, r_heater)))
    v71 = dl_dmw_pm * 1e-12
    v73 = lambda_res_nm * 1e-09 + v71 * (jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 0.0, jnp.divide((signals.hp - signals.hn) * (signals.hp - signals.hn), jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 1.0, r_heater))) * 1000.0)
    v82 = 1883651567.3088531 * (jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 0.0, jnp.divide(1.0, jnp.where((lambda_nm * 1e-09 == 0.0) | ~jnp.isfinite(lambda_nm * 1e-09), 1.0, lambda_nm * 1e-09))) - jnp.where((v73 == 0.0) | ~jnp.isfinite(v73), 0.0, jnp.divide(1.0, jnp.where((v73 == 0.0) | ~jnp.isfinite(v73), 1.0, v73))))
    v84 = 2.0 * i_v25
    v100 = i_v39 * (v82 + v84)
    v105 = i_v41 * (signals.in_im - signals.gnd)
    v106 = s.v_ar_m2 - signals.gnd + v100 * (s.v_ai_m2 - signals.gnd) + v105
    v115 = i_v41 * (signals.in_re - signals.gnd)
    v116 = s.v_ai_m2 - signals.gnd - v100 * (s.v_ar_m2 - signals.gnd) - v115
    v122 = i_v39 * (v82 + i_v25)
    v127 = s.v_ar_m1 - signals.gnd + v122 * (s.v_ai_m1 - signals.gnd) + v105
    v136 = s.v_ai_m1 - signals.gnd - v122 * (s.v_ar_m1 - signals.gnd) - v115
    v142 = i_v39 * v82
    v147 = s.v_ar_0 - signals.gnd + v142 * (s.v_ai_0 - signals.gnd) + v105
    v156 = s.v_ai_0 - signals.gnd - v142 * (s.v_ar_0 - signals.gnd) - v115
    v162 = i_v39 * (v82 - i_v25)
    v167 = s.v_ar_p1 - signals.gnd + v162 * (s.v_ai_p1 - signals.gnd) + v105
    v176 = s.v_ai_p1 - signals.gnd - v162 * (s.v_ar_p1 - signals.gnd) - v115
    v182 = i_v39 * (v82 - v84)
    v187 = s.v_ar_p2 - signals.gnd + v182 * (s.v_ai_p2 - signals.gnd) + v105
    v196 = s.v_ai_p2 - signals.gnd - v182 * (s.v_ar_p2 - signals.gnd) - v115
    v239 = i_v39 * (s.v_ar_m2 - signals.gnd)
    v240 = i_v39 * (s.v_ai_m2 - signals.gnd)
    v241 = i_v39 * (s.v_ar_m1 - signals.gnd)
    v242 = i_v39 * (s.v_ai_m1 - signals.gnd)
    v243 = i_v39 * (s.v_ar_0 - signals.gnd)
    v244 = i_v39 * (s.v_ai_0 - signals.gnd)
    v245 = i_v39 * (s.v_ar_p1 - signals.gnd)
    v246 = i_v39 * (s.v_ai_p1 - signals.gnd)
    v247 = i_v39 * (s.v_ar_p2 - signals.gnd)
    v248 = i_v39 * (s.v_ai_p2 - signals.gnd)
    v205 = s.v_ai_m2 - signals.gnd + s.v_ai_m1 - signals.gnd + s.v_ai_0 - signals.gnd + s.v_ai_p1 - signals.gnd + s.v_ai_p2 - signals.gnd
    v200 = s.v_ar_m2 - signals.gnd + s.v_ar_m1 - signals.gnd + s.v_ar_0 - signals.gnd + s.v_ar_p1 - signals.gnd + s.v_ar_p2 - signals.gnd
    v337 = jnp.where((v73 * v73 == 0.0) | ~jnp.isfinite(v73 * v73), 0.0, jnp.divide(jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 0.0, jnp.divide(signals.hp - signals.hn + signals.hp - signals.hn, jnp.where((r_heater == 0.0) | ~jnp.isfinite(r_heater), 1.0, r_heater))) * 1000.0 * v71, jnp.where((v73 * v73 == 0.0) | ~jnp.isfinite(v73 * v73), 1.0, v73 * v73))) * 1883651567.3088531 * i_v39
    v338 = v337 * (s.v_ai_m2 - signals.gnd)
    v480 = -v338
    v347 = v337 * (s.v_ar_m2 - signals.gnd)
    v348 = 0.0 - v347
    v357 = v337 * (s.v_ai_m1 - signals.gnd)
    v366 = v337 * (s.v_ar_m1 - signals.gnd)
    v367 = 0.0 - v366
    v376 = v337 * (s.v_ai_0 - signals.gnd)
    v385 = v337 * (s.v_ar_0 - signals.gnd)
    v386 = 0.0 - v385
    v395 = v337 * (s.v_ai_p1 - signals.gnd)
    v404 = v337 * (s.v_ar_p1 - signals.gnd)
    v405 = 0.0 - v404
    v414 = v337 * (s.v_ai_p2 - signals.gnd)
    v423 = v337 * (s.v_ar_p2 - signals.gnd)
    v424 = 0.0 - v423
    v580 = v480 - v348 - v357 - v367 - v376 - v386 - v395 - v405 - v414 - v424
    v349 = 0.0 - v100
    v486 = -1.0 - v349
    v487 = -v100 - 1.0
    v483 = -i_v41
    v354 = 0.0 - i_v41
    v368 = 0.0 - v122
    v506 = -1.0 - v368
    v507 = v368 - 1.0
    v387 = 0.0 - v142
    v529 = -1.0 - v387
    v530 = v387 - 1.0
    v406 = 0.0 - v162
    v556 = -1.0 - v406
    v557 = v406 - 1.0
    v425 = 0.0 - v182
    v591 = -1.0 - v425
    v592 = v425 - 1.0
    v780 = v483 - 1.0
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v71, i_v74, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v74, i_v71, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [init[9], init[8], 0.0, 0.0, 0.0, 0.0, _mfactor * v580, _mfactor * -v580, _mfactor * (-v486 - v487 - (v483 - i_v41 - i_v41 - i_v41 - i_v41) - (i_v41 - v354 - v354 - v354 - v354) - v506 - v507 - v529 - v530 - v556 - v557 - v591 - v592), _mfactor * v486, _mfactor * v487, _mfactor * v506, _mfactor * v507, _mfactor * v529, _mfactor * v530, _mfactor * v556, _mfactor * v557, _mfactor * v591, _mfactor * v592, i_v101, i_v101, i_v101, i_v101], [0.0, i_v108, 0.0, 0.0, 0.0, 0.0, _mfactor * v338, _mfactor * v480, _mfactor * (-1.0 - v100 - i_v41), _mfactor, _mfactor * v100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v111, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v348, _mfactor * v347, _mfactor * (v100 - 1.0 - v354), _mfactor * v349, _mfactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v108, 0.0, 0.0, 0.0, 0.0, _mfactor * v357, _mfactor * -v357, _mfactor * (v780 - v122), 0.0, 0.0, _mfactor, _mfactor * v122, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v111, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v367, _mfactor * v366, _mfactor * (i_v41 - v368 - 1.0), 0.0, 0.0, _mfactor * v368, _mfactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v108, 0.0, 0.0, 0.0, 0.0, _mfactor * v376, _mfactor * -v376, _mfactor * (v780 - v142), 0.0, 0.0, 0.0, 0.0, _mfactor, _mfactor * v142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v111, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v386, _mfactor * v385, _mfactor * (i_v41 - v387 - 1.0), 0.0, 0.0, 0.0, 0.0, _mfactor * v387, _mfactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v108, 0.0, 0.0, 0.0, 0.0, _mfactor * v395, _mfactor * -v395, _mfactor * (v780 - v162), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, _mfactor * v162, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [i_v111, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v405, _mfactor * v404, _mfactor * (i_v41 - v406 - 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v406, _mfactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v108, 0.0, 0.0, 0.0, 0.0, _mfactor * v414, _mfactor * -v414, _mfactor * (v780 - v182), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, _mfactor * v182, 0.0, 0.0, 0.0, 0.0], [i_v111, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v424, _mfactor * v423, _mfactor * (i_v41 - v425 - 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor * v425, _mfactor, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -5.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, init[16], 0.0, i_v58, 0.0, i_v58, 0.0, i_v58, 0.0, i_v58, 0.0, i_v58, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, init[17], i_v43, 0.0, i_v43, 0.0, i_v43, 0.0, i_v43, 0.0, i_v43, 0.0, 0.0, 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, init[15], i_v76, i_v79, i_v79, i_v79, i_v79, i_v79, i_v79, i_v79, i_v79, i_v79, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v106, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'thru_re': _mfactor * s.i_br15, 'thru_im': _mfactor * s.i_br16, 'drop_re': _mfactor * s.i_br17, 'drop_im': _mfactor * s.i_br18, 'hp': _mfactor * v65, 'hn': _mfactor * -v65, 'gnd': _mfactor * (-v106 - v116 - v127 - v136 - v147 - v156 - v167 - v176 - v187 - v196 - s.i_br15 - s.i_br16 - s.i_br17 - s.i_br18), 'v_ar_m2': _mfactor * v106, 'v_ai_m2': _mfactor * v116, 'v_ar_m1': _mfactor * v127, 'v_ai_m1': _mfactor * v136, 'v_ar_0': _mfactor * v147, 'v_ai_0': _mfactor * v156, 'v_ar_p1': _mfactor * v167, 'v_ai_p1': _mfactor * v176, 'v_ar_p2': _mfactor * v187, 'v_ai_p2': _mfactor * v196, 'i_br15': signals.in_re - signals.gnd - v205 - (signals.thru_re - signals.gnd), 'i_br16': signals.in_im - signals.gnd + v200 - (signals.thru_im - signals.gnd), 'i_br17': i_v58 * v205 - (signals.drop_re - signals.gnd), 'i_br18': i_v43 * v200 - (signals.drop_im - signals.gnd)}, {'gnd': _mfactor * (-v239 - v240 - v241 - v242 - v243 - v244 - v245 - v246 - v247 - v248), 'v_ar_m2': _mfactor * v239, 'v_ai_m2': _mfactor * v240, 'v_ar_m1': _mfactor * v241, 'v_ai_m1': _mfactor * v242, 'v_ar_0': _mfactor * v243, 'v_ai_0': _mfactor * v244, 'v_ar_p1': _mfactor * v245, 'v_ai_p1': _mfactor * v246, 'v_ar_p2': _mfactor * v247, 'v_ai_p2': _mfactor * v248}, j_resist, j_react)

def _RingFilter_jacobian(signals: Signals, s: States, init, radius_um: float=100.0, n_g: float=4.0, loss_db_m: float=100.0, kappa2_in: float=0.05, kappa2_drop: float=0.05, r_heater: float=500.0, lambda_res_nm: float=1310.0, dl_dmw_pm: float=20.0, lambda_nm: float=1310.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _RingFilter_combined(signals, s, init, radius_um=radius_um, n_g=n_g, loss_db_m=loss_db_m, kappa2_in=kappa2_in, kappa2_drop=kappa2_drop, r_heater=r_heater, lambda_res_nm=lambda_res_nm, dl_dmw_pm=dl_dmw_pm, lambda_nm=lambda_nm, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('in_re', 'in_im', 'thru_re', 'thru_im', 'drop_re', 'drop_im', 'hp', 'hn', 'gnd'), states=('v_ar_m2', 'v_ai_m2', 'v_ar_m1', 'v_ai_m1', 'v_ar_0', 'v_ai_0', 'v_ar_p1', 'v_ai_p1', 'v_ar_p2', 'v_ai_p2', 'i_br15', 'i_br16', 'i_br17', 'i_br18'), jacobian_fn=_RingFilter_jacobian, combined_fn=_RingFilter_combined, differentiable_params=None)
def RingFilter(signals: Signals, s: States, init, radius_um: float=100.0, n_g: float=4.0, loss_db_m: float=100.0, kappa2_in: float=0.05, kappa2_drop: float=0.05, r_heater: float=500.0, lambda_res_nm: float=1310.0, dl_dmw_pm: float=20.0, lambda_nm: float=1310.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _RingFilter_combined(signals, s, init, radius_um=radius_um, n_g=n_g, loss_db_m=loss_db_m, kappa2_in=kappa2_in, kappa2_drop=kappa2_drop, r_heater=r_heater, lambda_res_nm=lambda_res_nm, dl_dmw_pm=dl_dmw_pm, lambda_nm=lambda_nm, _mfactor=_mfactor)
    return (f, q)

@RingFilter.setup
def _RingFilter_register_setup(*_a, **_kw):
    return _RingFilter_setup(*_a, **_kw)