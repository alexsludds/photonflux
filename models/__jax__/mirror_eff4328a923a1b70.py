"""Auto-generated from Verilog-A MIR — do not edit by hand.

Regenerate with: ``python -m circulax.va <path/to/device.va>``
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States
from bosdi.circulax.va_component import va_component

def _Mirror_setup(refl: float=0.3, loss_db: float=0.0, phi_r_deg: float=0.0, _mfactor: float=1.0) -> jnp.ndarray:
    i_v24 = jnp.power(jnp.maximum(10.0, 0.0), jnp.divide(-loss_db, 20.0))
    i_v30 = i_v24 * jnp.sqrt(jnp.maximum(refl, 1e-300))
    i_v28 = jnp.divide(phi_r_deg * 3.141592653589793, 180.0)
    i_v32 = i_v30 * jnp.cos(i_v28)
    i_v34 = i_v30 * jnp.sin(i_v28)
    i_v40 = _mfactor * -1.0
    i_v43 = _mfactor * -1e-12
    i_v50 = _mfactor * 1e-12
    i_v35 = 0.0 - i_v34
    i_v19 = jnp.sqrt(jnp.maximum(1.0 - refl, 1e-300))
    i_v36 = i_v24 * i_v19
    i_v37 = 0.0 - i_v36
    i_v39 = -i_v24 * i_v19
    i_v57 = 1.0 - i_v32 - i_v35 - i_v37
    i_v64 = -i_v34 - i_v32 - -1.0 - i_v36
    i_v72 = -i_v39 - i_v35 - i_v32 - -1.0
    i_v79 = -i_v36 - i_v32 - i_v34 - -1.0
    return jnp.array([i_v32, i_v34, i_v40, i_v43, i_v50, i_v35, i_v37, i_v36, i_v39, i_v57, i_v64, i_v72, i_v79])

def _Mirror_combined(signals: Signals, s: States, init, refl: float=0.3, loss_db: float=0.0, phi_r_deg: float=0.0, _mfactor: float=1.0) -> tuple:
    """Combined physics + Jacobian — single hoist block, auto-generated from VA MIR."""
    i_v32 = init[0]
    i_v34 = init[1]
    i_v40 = init[2]
    i_v43 = init[3]
    i_v50 = init[4]
    i_v35 = init[5]
    i_v36 = init[7]
    i_v39 = init[8]
    v90 = 1e-12 * (s.v_dmy - signals.gnd)
    j_resist = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _mfactor, i_v40, i_v40, i_v40, i_v40, i_v40], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v40, _mfactor, 0.0, 0.0, 0.0, 0.0], [i_v32, i_v35, -1.0, 0.0, 0.0, init[6], 0.0, 0.0, init[9], 0.0, 0.0, 0.0, 0.0, 0.0], [i_v34, i_v32, 0.0, -1.0, i_v36, 0.0, 0.0, 0.0, init[10], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, i_v39, 0.0, 0.0, i_v32, i_v35, -1.0, 0.0, init[11], 0.0, 0.0, 0.0, 0.0, 0.0], [i_v36, 0.0, 0.0, 0.0, i_v34, i_v32, 0.0, -1.0, init[12], 0.0, 0.0, 0.0, 0.0, 0.0]])
    j_react = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v50, i_v43, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_v43, i_v50, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return ({'lo_re': _mfactor * s.i_br5, 'lo_im': _mfactor * s.i_br6, 'ro_re': _mfactor * s.i_br7, 'ro_im': _mfactor * s.i_br8, 'gnd': _mfactor * (-(s.v_dmy - signals.gnd) - s.i_br5 - s.i_br6 - s.i_br7 - s.i_br8), 'v_dmy': _mfactor * (s.v_dmy - signals.gnd), 'i_br5': i_v32 * (signals.li_re - signals.gnd) - i_v34 * (signals.li_im - signals.gnd) - i_v36 * (signals.ri_im - signals.gnd) - (signals.lo_re - signals.gnd), 'i_br6': i_v34 * (signals.li_re - signals.gnd) + i_v32 * (signals.li_im - signals.gnd) + i_v36 * (signals.ri_re - signals.gnd) - (signals.lo_im - signals.gnd), 'i_br7': i_v39 * (signals.li_im - signals.gnd) + i_v32 * (signals.ri_re - signals.gnd) - i_v34 * (signals.ri_im - signals.gnd) - (signals.ro_re - signals.gnd), 'i_br8': i_v36 * (signals.li_re - signals.gnd) + i_v34 * (signals.ri_re - signals.gnd) + i_v32 * (signals.ri_im - signals.gnd) - (signals.ro_im - signals.gnd)}, {'gnd': _mfactor * -v90, 'v_dmy': _mfactor * v90}, j_resist, j_react)

def _Mirror_jacobian(signals: Signals, s: States, init, refl: float=0.3, loss_db: float=0.0, phi_r_deg: float=0.0, _mfactor: float=1.0) -> tuple:
    """Jacobian wrapper — slices the combined function's output."""
    _f, _q, j_resist, j_react = _Mirror_combined(signals, s, init, refl=refl, loss_db=loss_db, phi_r_deg=phi_r_deg, _mfactor=_mfactor)
    return (j_resist, j_react)

@va_component(ports=('li_re', 'li_im', 'lo_re', 'lo_im', 'ri_re', 'ri_im', 'ro_re', 'ro_im', 'gnd'), states=('v_dmy', 'i_br5', 'i_br6', 'i_br7', 'i_br8'), jacobian_fn=_Mirror_jacobian, combined_fn=_Mirror_combined, differentiable_params=None)
def Mirror(signals: Signals, s: States, init, refl: float=0.3, loss_db: float=0.0, phi_r_deg: float=0.0, _mfactor: float=1.0) -> PhysicsReturn:
    """Auto-generated from Verilog-A — thin wrapper over ``_combined``."""
    f, q, _j_f, _j_q = _Mirror_combined(signals, s, init, refl=refl, loss_db=loss_db, phi_r_deg=phi_r_deg, _mfactor=_mfactor)
    return (f, q)

@Mirror.setup
def _Mirror_register_setup(*_a, **_kw):
    return _Mirror_setup(*_a, **_kw)