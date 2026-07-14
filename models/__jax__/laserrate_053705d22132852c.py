"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _LaserRate_setup(Rs: float=5.0, Nscl: float=1e+24, Sscl: float=1e+21, vg: float=75000000.0, a0: float=2.5e-20, Va: float=1e-16, taun: float=2e-09, Gam: float=0.3, taup: float=2e-12, beta: float=0.0001, Csc: float=1e-12, eta0: float=0.4, Eph: float=1.516e-19, Von: float=1.2, Ntr: float=1e+24, eps: float=1.5e-23, etai: float=0.8, _mfactor: float=1.0) -> jnp.ndarray:
    i_v55 = _mfactor * -Csc
    i_v54 = 0.0 - Csc
    i_v59 = _mfactor * i_v54
    i_v61 = _mfactor * -1.0
    i_v63 = _mfactor * Csc
    i_v51 = Sscl * (0.5 * eta0 * jnp.where((Gam * taup == 0.0) | ~jnp.isfinite(Gam * taup), 0.0, jnp.divide(Va * Eph, jnp.where((Gam * taup == 0.0) | ~jnp.isfinite(Gam * taup), 1.0, Gam * taup))))
    i_v70 = _mfactor * (Csc - i_v54)
    i_v68 = -i_v51 - -1.0
    return jnp.array([i_v55, i_v59, i_v61, i_v63, i_v51, i_v70, i_v68])

def _LaserRate_combined(signals: Signals, s: States, init, Rs: float=5.0, Nscl: float=1e+24, Sscl: float=1e+21, vg: float=75000000.0, a0: float=2.5e-20, Va: float=1e-16, taun: float=2e-09, Gam: float=0.3, taup: float=2e-12, beta: float=0.0001, Csc: float=1e-12, eta0: float=0.4, Eph: float=1.516e-19, Von: float=1.2, Ntr: float=1e+24, eps: float=1.5e-23, etai: float=0.8, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v55 = init[0]
    i_v63 = init[3]
    v20 = jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(signals.an - signals.cat - Von, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs)))
    v22 = v20 < 0.0
    v24 = jnp.where(v22, 0.0, v20)
    v62 = 1.602176462e-19 * Va
    v31 = (s.v_nn - signals.gnd) * Nscl
    v43 = vg * a0
    v49 = v43 * (v31 - Ntr)
    v35 = (s.v_ns - signals.gnd) * Sscl
    v37 = v35 > 0.0
    v39 = jnp.where(v37, v35, 0.0)
    v52 = 1.0 + eps * v39
    v53 = jnp.where((v52 == 0.0) | ~jnp.isfinite(v52), 0.0, jnp.divide(v49, jnp.where((v52 == 0.0) | ~jnp.isfinite(v52), 1.0, v52)))
    v89 = jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(Csc * (jnp.where((v62 == 0.0) | ~jnp.isfinite(v62), 0.0, jnp.divide(etai * v24, jnp.where((v62 == 0.0) | ~jnp.isfinite(v62), 1.0, v62))) - jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 0.0, jnp.divide(v31, jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 1.0, taun))) - v53 * v39), jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v71 = Gam * v53
    v79 = Gam * beta
    v102 = 0.0 - jnp.where((Sscl == 0.0) | ~jnp.isfinite(Sscl), 0.0, jnp.divide(Csc * (v71 * v39 - jnp.where((taup == 0.0) | ~jnp.isfinite(taup), 0.0, jnp.divide(v35, jnp.where((taup == 0.0) | ~jnp.isfinite(taup), 1.0, taup))) + jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 0.0, jnp.divide(v79 * v31, jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 1.0, taun)))), jnp.where((Sscl == 0.0) | ~jnp.isfinite(Sscl), 1.0, Sscl)))
    v139 = Csc * (s.v_nn - signals.gnd)
    v140 = Csc * (s.v_ns - signals.gnd)
    v161 = jnp.where(v22, 0.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 0.0, jnp.divide(1.0, jnp.where((Rs == 0.0) | ~jnp.isfinite(Rs), 1.0, Rs))))
    v283 = _mfactor * v161
    v286 = _mfactor * -v161
    v205 = jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide(jnp.where((v62 == 0.0) | ~jnp.isfinite(v62), 0.0, jnp.divide(v161 * etai, jnp.where((v62 == 0.0) | ~jnp.isfinite(v62), 1.0, v62))) * Csc, jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v290 = _mfactor * v205
    v168 = jnp.where((v52 == 0.0) | ~jnp.isfinite(v52), 0.0, jnp.divide(Nscl * v43, jnp.where((v52 == 0.0) | ~jnp.isfinite(v52), 1.0, v52)))
    v206 = jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide((0.0 - jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 0.0, jnp.divide(Nscl, jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 1.0, taun))) - v168 * v39) * Csc, jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v214 = jnp.where((Sscl == 0.0) | ~jnp.isfinite(Sscl), 0.0, jnp.divide((v168 * Gam * v39 + jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 0.0, jnp.divide(Nscl * v79, jnp.where((taun == 0.0) | ~jnp.isfinite(taun), 1.0, taun)))) * Csc, jnp.where((Sscl == 0.0) | ~jnp.isfinite(Sscl), 1.0, Sscl)))
    v216 = 0.0 - v214
    v225 = v206 - v216
    v162 = jnp.where(v37, Sscl, 0.0)
    v171 = 0.0 - jnp.where((v52 * v52 == 0.0) | ~jnp.isfinite(v52 * v52), 0.0, jnp.divide(v162 * eps * v49, jnp.where((v52 * v52 == 0.0) | ~jnp.isfinite(v52 * v52), 1.0, v52 * v52)))
    v207 = jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 0.0, jnp.divide((0.0 - (v171 * v39 + v162 * v53)) * Csc, jnp.where((Nscl == 0.0) | ~jnp.isfinite(Nscl), 1.0, Nscl)))
    v217 = 0.0 - jnp.where((Sscl == 0.0) | ~jnp.isfinite(Sscl), 0.0, jnp.divide((v171 * Gam * v39 + v162 * v71 - jnp.where((taup == 0.0) | ~jnp.isfinite(taup), 0.0, jnp.divide(Sscl, jnp.where((taup == 0.0) | ~jnp.isfinite(taup), 1.0, taup)))) * Csc, jnp.where((Sscl == 0.0) | ~jnp.isfinite(Sscl), 1.0, Sscl)))
    v226 = v207 - v217
    v210 = 0.0 - v207
    j_resist = jnp.array([[v283, v286, 0.0, 0.0, 0.0, 0.0, 0.0], [v286, v283, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [v290, _mfactor * -v205, 0.0, _mfactor * (-v225 - v226), _mfactor * v225, _mfactor * v226, init[2]], [_mfactor * (0.0 - v205), v290, 0.0, _mfactor * (v206 - v210), _mfactor * (0.0 - v206), _mfactor * v210, 0.0], [0.0, 0.0, 0.0, _mfactor * (v214 - v217), _mfactor * v216, _mfactor * v217, 0.0], [0.0, 0.0, -1.0, init[6], 0.0, init[4], 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, init[5], i_v55, init[1], 0.0], [0.0, 0.0, 0.0, i_v55, i_v63, 0.0, 0.0], [0.0, 0.0, 0.0, i_v55, 0.0, i_v63, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'an': _mfactor * v24, 'cat': _mfactor * -v24, 'popt': _mfactor * s.i_br4, 'gnd': _mfactor * (v89 - v102 - s.i_br4), 'v_nn': _mfactor * (0.0 - v89), 'v_ns': _mfactor * v102, 'i_br4': 0.5 * eta0 * jnp.where((Gam * taup == 0.0) | ~jnp.isfinite(Gam * taup), 0.0, jnp.divide(Va * Eph, jnp.where((Gam * taup == 0.0) | ~jnp.isfinite(Gam * taup), 1.0, Gam * taup))) * v35 - (signals.popt - signals.gnd)}, {'gnd': _mfactor * (-v139 - v140), 'v_nn': _mfactor * v139, 'v_ns': _mfactor * v140}, j_resist, j_react)

def _LaserRate_jacobian(signals: Signals, s: States, init, Rs: float=5.0, Nscl: float=1e+24, Sscl: float=1e+21, vg: float=75000000.0, a0: float=2.5e-20, Va: float=1e-16, taun: float=2e-09, Gam: float=0.3, taup: float=2e-12, beta: float=0.0001, Csc: float=1e-12, eta0: float=0.4, Eph: float=1.516e-19, Von: float=1.2, Ntr: float=1e+24, eps: float=1.5e-23, etai: float=0.8, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _LaserRate_combined(signals, s, init, Rs=Rs, Nscl=Nscl, Sscl=Sscl, vg=vg, a0=a0, Va=Va, taun=taun, Gam=Gam, taup=taup, beta=beta, Csc=Csc, eta0=eta0, Eph=Eph, Von=Von, Ntr=Ntr, eps=eps, etai=etai, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('an', 'cat', 'popt', 'gnd'), states=('v_nn', 'v_ns', 'i_br4'), jacobian_fn=_LaserRate_jacobian, combined_fn=_LaserRate_combined, differentiable_params=None)
def LaserRate(signals: Signals, s: States, init, Rs: float=5.0, Nscl: float=1e+24, Sscl: float=1e+21, vg: float=75000000.0, a0: float=2.5e-20, Va: float=1e-16, taun: float=2e-09, Gam: float=0.3, taup: float=2e-12, beta: float=0.0001, Csc: float=1e-12, eta0: float=0.4, Eph: float=1.516e-19, Von: float=1.2, Ntr: float=1e+24, eps: float=1.5e-23, etai: float=0.8, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _LaserRate_combined(signals, s, init, Rs=Rs, Nscl=Nscl, Sscl=Sscl, vg=vg, a0=a0, Va=Va, taun=taun, Gam=Gam, taup=taup, beta=beta, Csc=Csc, eta0=eta0, Eph=Eph, Von=Von, Ntr=Ntr, eps=eps, etai=etai, _mfactor=_mfactor)
    return (f, q)

@LaserRate.setup
def _LaserRate_register_setup(*_a, **_kw):
    return _LaserRate_setup(*_a, **_kw)