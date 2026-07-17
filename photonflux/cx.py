"""circulax bridge: Verilog-A models and SKY130 PDK transistors as JAX components.

Two entry points, both returning circulax ``CircuitComponent`` classes built by
`bosdi <https://pypi.org/project/bosdi/>`_ (``pip install circulax[verilog-a]
openvaf-py``) from real Verilog-A:

  ``cx.va("photodiode")``
      any ``models/*.va`` compiled straight to a differentiable JAX component.
      Same physics as the ngspice/OSDI path, but ``jax.grad`` works through it.

  ``cx.sky130_fet("nfet_01v8", w=1.0, l=0.15)``
      a real SKY130 PDK transistor. The volare model card is resolved by
      ngspice itself (``.lib`` + bin selection + `{expression}` evaluation via
      ``showmod``). Two backends:

      * ``backend="osdi"`` (default): the BSIM4.8 Verilog-A compiled to an
        OSDI binary by the ChipFlow openvaf fork (``bin/openvaf-ir``) and
        evaluated natively inside circulax's Newton loop — the same physics
        ngspice would run, exact to the card. Not differentiable through the
        FET's own parameters (the photonics still are).
      * ``backend="jax"`` (experimental): pure-JAX lowering via bosdi. Fully
        differentiable, but bosdi 0.1.5's optimized-MIR ingestion miscompiles
        BSIM4's nested conditionals (its docs say as much): the nfet is a few
        % off in strong inversion with a broken subthreshold region, and the
        pfet is unusable. Kept for when the fixed ingestion path ships.

Everything is content-hash cached under ``models/__jax__/``.

Physics note: ngspice solves sky130 with its builtin BSIM4v5 (the cards say
``version=4.5``); this bridge feeds the same card to a BSIM4.8 implementation,
so small version-skew differences vs ngspice-with-sky130 are expected even on
the osdi backend. The unit tests pin the agreement envelope.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import toolchain

__all__ = ["va", "sky130_fet", "sky130_card", "cw_laser", "mzm",
           "field_to_ri", "ri_to_field",
           "directional_coupler", "ring_phase_shifter", "cavity_mode",
           "ring_cmt_rates", "ring_modulator", "RingModulatorParts"]

CACHE_DIR = toolchain.MODELS_DIR / "__jax__"
BSIM4_VA = toolchain.REPO / "vendor" / "BSIM4" / "bsim4.va"

# The cogenda BSIM4.8 VA marks "parameter not given" with the in-band sentinel
# -12345789 and resolves defaults at runtime (model-setup mutates the param).
# bosdi lowers openvaf's *split* functions, and later functions re-read the raw
# parameter input, so a baked sentinel leaks into live arithmetic (0**-1.3 ->
# inf). Never bake sentinels: parameters the card doesn't give are baked with
# ngspice's own resolved BSIM4 defaults (a bare level=54 model, read back with
# showmod), which is the same default resolution the C code applies.
# Instance (geometry) parameters that stay runtime-settable on the component.
# "as" is missing on purpose: it is a Python keyword, so it must be baked
# static (bosdi's emitter would otherwise produce invalid Python). "nrd"/"nrs"
# are baked too: with rdsmod=0 the d/di and s/si nodes are collapsed (exactly
# like ngspice), which shorts the rsh*nrd sheet resistors those params feed.
_SKY130_RUNTIME = ("w", "l", "nf", "ad", "pd", "ps", "sa", "sb", "sd", "delvto")


# ---------------------------------------------------------------------------
# generic .va -> circulax component
# ---------------------------------------------------------------------------

def va(
    name: str | Path,
    *,
    models_dir: str | Path | None = None,
    differentiable_params: tuple[str, ...] | None = None,
    static_params: dict[str, float] | None = None,
    class_name: str | None = None,
):
    """Compile ``models/<name>.va`` into a circulax component class (cached).

    ``differentiable_params=None`` (default) keeps *every* parameter a JAX
    leaf so ``jax.grad`` works through it; pass a tuple to restrict (faster),
    or ``()`` to fold all parameters as compile-time constants.
    """
    src = Path(name)
    if src.suffix != ".va":
        base = Path(models_dir) if models_dir else toolchain.MODELS_DIR
        stem = Path(name).name
        src = base / f"{stem}.va"
        if not src.exists():
            # models/ groups sources into convention subfolders
            # (optical_power/, optical_field/, util/); resolve by bare name.
            src = next(iter(sorted(base.glob(f"**/{stem}.va"))), src)
    if not src.exists():
        raise FileNotFoundError(src)
    _check_va_support(src)
    cls = class_name or _camel(src.stem)
    module = _lower_cached(
        va_path=src,
        class_name=cls,
        static_params=static_params or {},
        differentiable_params=differentiable_params,
        allow_analog_in_cond=False,
    )
    return module[cls]


# ---------------------------------------------------------------------------
# photonic sources & modulators (coherent-field convention)
# ---------------------------------------------------------------------------
#
# Policy: the only lasers in this environment are CW lasers — parameterised by
# wavelength and power, emitting a constant field E = sqrt(P)·e^{j·phase}.
# Modulation is done by modulators (MZM below), never inside the laser.
# Fields are baseband envelopes at the laser wavelength; ``wavelength_nm``
# does not change E itself but must match the ``wavelength_nm`` fed to the
# dispersive passives (circulax's OpticalWaveguide, Grating, ...).

_COMPONENT_CACHE: dict[str, Any] = {}


def cw_laser():
    """The CW laser: ``settings = {"wavelength_nm", "power", "phase",
    "ref_wavelength_nm"}``.

    Two-terminal field source (ports ``p1``/``p2``, ground ``p2``): enforces
    ``E = sqrt(power) * exp(j*phase)`` across its ports, like an ideal voltage
    source. ``power`` in watts, ``wavelength_nm`` in nanometres.

    Multi-carrier (WDM) support: fields are baseband envelopes. With a single
    carrier the envelope is at the laser wavelength and is constant. To carry
    several wavelengths on one shared node (a DWDM bus) they must live in a
    *common* baseband frame rotating at a reference optical frequency. Set
    ``ref_wavelength_nm`` to that shared reference and the laser emits its true
    tone in that frame,

        E(t) = sqrt(power) * exp(j*(2*pi*f_off*t + phase)),
        f_off = c*(1/ref_wavelength - 1/wavelength),

    so lasers at different wavelengths appear as distinct tones that coexist
    and beat on the bus. ``ref_wavelength_nm = 0`` (default) keeps the legacy
    single-carrier behaviour (constant field, no offset). The sign matches the
    ring/waveguide detuning convention ``delta = 2*pi*c*(1/lambda_nm -
    1/lambda_res)``: a wavelength-selective device set to ``lambda_nm =
    ref_wavelength_nm`` with its resonance at ``wavelength`` sees this tone on
    resonance, so each ring picks out its own carrier in a single solve.
    """
    if "cw_laser" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["cw_laser"]
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    c0 = 299792458.0

    @source(ports=("p1", "p2"), states=("i_src",))
    def CWLaser(
        signals: Signals,
        s: States,
        t: float,
        wavelength_nm: float = 1310.0,
        power: float = 1e-3,
        phase: float = 0.0,
        ref_wavelength_nm: float = 0.0,
    ) -> tuple[dict, dict]:
        # baseband tone offset; when ref<=0 the reference collapses to the
        # laser's own wavelength -> w_off = 0 (legacy single-carrier). Select
        # the (never-zero) reference before dividing so the dry-run's plain
        # Python floats don't hit 1/0.
        ref_safe = jnp.where(ref_wavelength_nm > 0.0,
                             ref_wavelength_nm, wavelength_nm)
        w_off = 2.0 * jnp.pi * c0 * (1.0 / (ref_safe * 1e-9)
                                     - 1.0 / (wavelength_nm * 1e-9))
        field = jnp.sqrt(power) * jnp.exp(1j * (w_off * t + phase))
        constraint = (signals.p1 - signals.p2) - field
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}

    _COMPONENT_CACHE["cw_laser"] = CWLaser
    return CWLaser


def mzm():
    """Mach-Zehnder modulator, coherent-field twin of ``models/optical_power/mzm.va``.

    Optical 2-port (``pin``/``pout``, S-matrix element like circulax's
    OpticalWaveguide) with differential drive electrodes (``vp``/``vn``).
    Intensity transfer matches the VA model:

        T(V) = IL * (0.5 + 0.5*eta*cos(pi*(V + vbias)/vpi)),  eta = (ER-1)/(ER+1)

    and the field transmission is ``t = sqrt(T)``. ``cel`` puts the electrode
    capacitance on the drive so a transistor driver sees a real load. The
    drive voltage is read as the real part of the (complex-assembled)
    electrical nodes — non-holomorphic like the |E|^2 photodiode, which the
    real-2N solver differentiates correctly.
    """
    if "mzm" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["mzm"]
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component
    from circulax.s_transforms import s_to_y

    @component(ports=("pin", "pout", "vp", "vn"))
    def MZModulator(
        signals: Signals,
        s: States,
        vpi: float = 3.0,        # half-wave voltage [V]
        vbias: float = 0.0,      # built-in bias offset [V]
        il_db: float = 3.0,      # excess insertion loss [dB]
        er_db: float = 20.0,     # extinction ratio [dB]
        cel: float = 50e-15,     # electrode capacitance [F]
    ) -> tuple[dict, dict]:
        il = 10.0 ** (-il_db / 10.0)
        er = 10.0 ** (er_db / 10.0)
        eta = (er - 1.0) / (er + 1.0)
        vd = (signals.vp - signals.vn).real
        trans = il * (0.5 + 0.5 * eta * jnp.cos(jnp.pi * (vd + vbias) / vpi))
        t = jnp.sqrt(trans)

        S = jnp.array([[0.0 * t, t], [t, 0.0 * t]], dtype=jnp.complex128)
        Y = s_to_y(S)
        v_vec = jnp.array([signals.pin, signals.pout], dtype=jnp.complex128)
        i_vec = Y @ v_vec

        f = {"pin": i_vec[0], "pout": i_vec[1], "vp": 0.0, "vn": 0.0}
        q_el = cel * (signals.vp - signals.vn)
        q = {"vp": q_el, "vn": -q_el}
        return f, q

    _COMPONENT_CACHE["mzm"] = MZModulator
    return MZModulator


def field_to_ri():
    """Adapter: complex field node -> (re, im) real node pair.

    The repo's coherent Verilog-A models (``laser_cw.va``, ``ring_mod.va``)
    carry complex fields as Ereal/Eimag node *pairs*, while circulax's optical
    nodes are single complex unknowns. This tap enforces ``V(re) = Re(V(c))``
    and ``V(im) = Im(V(c))`` (drawing nothing from ``c``), so a VA model's
    re/im inputs can read a circulax field. Pair with :func:`ri_to_field` on
    the model's output side.
    """
    if "field_to_ri" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["field_to_ri"]
    from circulax.components.base_component import Signals, States, component

    @component(ports=("c", "re", "im"), states=("i_re", "i_im"))
    def FieldToRI(signals: Signals, s: States) -> tuple[dict, dict]:
        return {
            "c": 0.0,
            "re": s.i_re,
            "im": s.i_im,
            "i_re": signals.re - signals.c.real,
            "i_im": signals.im - signals.c.imag,
        }, {}

    _COMPONENT_CACHE["field_to_ri"] = FieldToRI
    return FieldToRI


def ri_to_field():
    """Adapter: (re, im) real node pair -> complex field node.

    Enforces ``V(c) = V(re) + j*V(im)`` (reading re/im at high impedance);
    the branch state supplies whatever current the complex-side load draws.
    Inverse of :func:`field_to_ri`.
    """
    if "ri_to_field" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["ri_to_field"]
    from circulax.components.base_component import Signals, States, component

    @component(ports=("re", "im", "c"), states=("i_c",))
    def RIToField(signals: Signals, s: States) -> tuple[dict, dict]:
        return {
            "re": 0.0,
            "im": 0.0,
            "c": s.i_c,
            "i_c": signals.c - (signals.re.real + 1j * signals.im.real),
        }, {}

    _COMPONENT_CACHE["ri_to_field"] = RIToField
    return RIToField


# ---------------------------------------------------------------------------
# microring modulator, as temporal coupled-mode sub-components
# ---------------------------------------------------------------------------
#
# ``models/optical_field/ring_mod.va`` is a monolithic coupled-mode-theory (CMT)
# ring: one block that solves
#
#     dA/dt = (-1/tau + j*delta(V)) A + j*kappa^2 s_in ,   s_out = s_in + j*A
#
# with ``1/tau = 1/tau_i + 1/tau_e``. The same physics factors into three
# coherent-field building blocks that share one internal complex node — the
# circulating cavity field ``A`` — whose Kirchhoff sum on that node reproduces
# the ODE exactly (verified to machine precision against ring_mod.va in
# ``tests/test_ring_decomposition.py`` and ``examples/eo_comb.py``):
#
#   * :func:`directional_coupler` — the bus<->ring point coupler: couples the
#     input into the cavity mode (external rate 1/tau_e) and taps the through
#     port ``s_out = s_in + j*A``;
#   * :func:`ring_phase_shifter` — the intracavity depletion phase shifter: the
#     electrode voltage sets the cavity detuning ``delta(V)`` (the electro-optic
#     drive). NB this is the *cavity-detuning* element, distinct from the
#     memoryless field-rotating ``models/optical_field/phase_shifter.va``;
#   * :func:`cavity_mode` — the ring loop itself: the circulating field's energy
#     storage (``d/dt``, i.e. the photon lifetime) plus its round-trip loss
#     (rate 1/tau_i).
#
# Why sub-components and not a literal coupler + waveguide feedback loop: the
# coherent-field primitives are memoryless (an optical S-matrix has no delay),
# so a physical loop would be algebraic and would model only the adiabatic,
# lifetime-free ring; localising the round trip into the cavity-mode storage
# element is what keeps the photon-lifetime dynamics. :func:`ring_modulator`
# wires the three into a drop-in coherent-field twin of ring_mod.va.

_C0 = 299792458.0  # speed of light [m/s]


def directional_coupler():
    """Bus<->ring point coupler (coherent-field CMT sub-component).

    ``settings = {"inv_tau_e"}`` — the external (amplitude) decay rate
    ``1/tau_e = kappa2/(2*T_rt)`` set by the bus power coupling. Couples the
    bus input into the shared cavity node ``a`` and taps the through port:

        contributes  a/tau_e - j*kappa^2 s_in  to the cavity node ``a``,
        drives       s_out = s_in + j*a        on the through port.

    For a lossless point coupler energy conservation fixes ``kappa^2 =
    2/tau_e``, so the single rate ``inv_tau_e`` sets both the coupling loss and
    the drive/output strength. Ports: ``sin`` (bus input — an ideal tap that
    draws nothing), ``sout`` (through output, driven), ``a`` (shared cavity
    field, joined to the phase shifter and cavity mode).
    """
    if "directional_coupler" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["directional_coupler"]
    from circulax.components.base_component import Signals, States, component

    @component(ports=("sin", "sout", "a"), states=("i_out",))
    def DirectionalCoupler(
        signals: Signals, s: States, inv_tau_e: float = 1.0,
    ) -> tuple[dict, dict]:
        krate = 2.0 * inv_tau_e                      # kappa^2 (lossless coupler)
        return {
            "sin": 0.0,                              # ideal input tap
            "a": inv_tau_e * signals.a - 1j * krate * signals.sin,
            "sout": s.i_out,
            "i_out": signals.sout - (signals.sin + 1j * signals.a),
        }, {}

    _COMPONENT_CACHE["directional_coupler"] = DirectionalCoupler
    return DirectionalCoupler


def ring_phase_shifter():
    """Intracavity electro-optic phase shifter (cavity-detuning sub-component).

    ``settings = {"lambda_nm", "lambda_res_nm", "dl_dv_pm", "cj", "rleak"}``.
    The depletion electrode moves the resonance linearly,
    ``lambda_res(V) = lambda_res_nm + dl_dv_pm*V``, i.e. it sets the CMT
    detuning

        delta(V) = 2*pi*c*(1/lambda_nm - 1/lambda_res(V))

    and contributes ``-j*delta(V)*a`` to the shared cavity node ``a`` — the
    same electro-optic map as ring_mod.va. Ports: ``a`` (shared cavity field),
    ``vp``/``vn`` (differential electrode). The electrode presents junction
    capacitance ``cj`` plus leakage ``rleak`` — the load a real driver sees.

    This is the ring's *cavity-detuning* phase shifter (it modulates the
    resonance of a stored mode); the standalone
    ``models/optical_field/phase_shifter.va`` is the memoryless field rotator
    ``E_out = e^{j*phi} E_in``. Both are "EO phase shifters"; only this one
    lives inside a cavity.
    """
    if "ring_phase_shifter" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["ring_phase_shifter"]
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("a", "vp", "vn"))
    def RingPhaseShifter(
        signals: Signals, s: States,
        lambda_nm: float = 1310.0,
        lambda_res_nm: float = 1310.0,
        dl_dv_pm: float = 45.0,
        cj: float = 0.0,
        rleak: float = 1e8,
    ) -> tuple[dict, dict]:
        v = (signals.vp - signals.vn).real
        lam_res = lambda_res_nm * 1e-9 + dl_dv_pm * 1e-12 * v
        delta = 2.0 * jnp.pi * _C0 * (1.0 / (lambda_nm * 1e-9) - 1.0 / lam_res)
        # electrode load: junction capacitance (reactive q) + leakage (resistive
        # f) — matches ring_mod.va's electrical port. Harmless under an ideal
        # drive (the source fixes V), so it never perturbs the optical output.
        gleak = (signals.vp - signals.vn) / rleak
        f = {"a": -1j * delta * signals.a, "vp": gleak, "vn": -gleak}
        q_el = cj * (signals.vp - signals.vn)
        return f, {"vp": q_el, "vn": -q_el}

    _COMPONENT_CACHE["ring_phase_shifter"] = RingPhaseShifter
    return RingPhaseShifter


def cavity_mode():
    """Ring-loop cavity mode (energy storage + round-trip loss sub-component).

    ``settings = {"inv_tau_i"}`` — the intrinsic (amplitude) decay rate
    ``1/tau_i = alpha*v_g/2`` from the round-trip propagation loss. Contributes
    ``dA/dt + a/tau_i`` to the shared cavity node ``a``: the ``d/dt`` is the
    photon storage (the photon lifetime), ``a/tau_i`` the round-trip loss. This
    is the element that localises the round trip, so the ring keeps its
    photon-lifetime dynamics instead of collapsing to the adiabatic limit.
    Single port ``a`` (the shared cavity field).
    """
    if "cavity_mode" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["cavity_mode"]
    from circulax.components.base_component import Signals, States, component

    @component(ports=("a",))
    def CavityMode(
        signals: Signals, s: States, inv_tau_i: float = 1.0,
    ) -> tuple[dict, dict]:
        return {"a": inv_tau_i * signals.a}, {"a": signals.a}

    _COMPONENT_CACHE["cavity_mode"] = CavityMode
    return CavityMode


@dataclass
class RingModulatorParts:
    """A decomposed-microring netlist fragment, ready to stitch into a circuit.

    ``instances``/``connections``/``models`` are the ring-internal pieces (the
    coupler, intracavity phase shifter and cavity mode, joined on an internal
    cavity node). The terminal names ``sin``/``sout``/``vp``/``vn`` are the
    *external* nets the caller wires: bus input field, through-port output
    field, and the two electrode nodes. Merge the three dicts into the netlist
    and connect the four terminals, e.g.::

        ring = cx.ring_modulator(ring_settings)
        inst.update(ring.instances); mdl.update(ring.models)
        conn.update(ring.connections)
        conn["LAS,p1"] = ring.sin
        conn[ring.sout] = "TERM,c"
        conn["VDRV,p1"] = ring.vp
        conn["GND,p1"] = (ring.vn, ...)
    """

    instances: dict
    connections: dict
    models: dict
    sin: str
    sout: str
    vp: str
    vn: str


def ring_cmt_rates(
    *,
    radius_um: float = 7.5,
    n_g: float = 4.0,
    loss_db_m: float = 7000.0,
    kappa2: float = 0.10,
    cj_ff_um: float = 0.5,
) -> dict[str, float]:
    """Device geometry -> CMT decay rates, exactly as ring_mod.va derives them.

    Returns ``{"inv_tau_e", "inv_tau_i", "cj"}``::

        circ = 2*pi*R,  v_g = c/n_g,  T_rt = circ/v_g
        1/tau_i = loss*ln(10)/10 * v_g/2      (round-trip loss)
        1/tau_e = kappa2/(2*T_rt)             (bus coupling)
        cj      = cj_ff_um * (2*pi*R[um]) fF  (junction capacitance)

    so a ring assembled by :func:`ring_modulator` reproduces the monolithic
    model driven by the same physical parameters.
    """
    import math

    circ = 2.0 * math.pi * radius_um * 1e-6
    v_g = _C0 / n_g
    t_rt = circ / v_g
    alpha = loss_db_m * math.log(10.0) / 10.0
    return {
        "inv_tau_e": kappa2 / (2.0 * t_rt),
        "inv_tau_i": alpha * v_g / 2.0,
        "cj": cj_ff_um * 1e-15 * 2.0 * math.pi * radius_um,
    }


def ring_modulator(settings: dict | None = None, *,
                   prefix: str = "RING") -> RingModulatorParts:
    """Microring modulator built from its coherent-field sub-components.

    A drop-in coherent-field twin of ``models/optical_field/ring_mod.va``: it
    accepts the *same* physical parameters and wires :func:`directional_coupler`
    + :func:`ring_phase_shifter` + :func:`cavity_mode` into a ring whose
    through-port field reproduces the monolithic model to machine precision
    (see ``tests/test_ring_decomposition.py``). ``settings`` mirrors
    ring_mod.va::

        lambda_nm, lambda_res_nm, radius_um, n_g, loss_db_m, kappa2,
        dl_dv_pm, cj_ff_um, rleak

    (``n_eff`` is accepted and ignored, exactly as in the aligned single-mode
    ring_mod.va — the thermal tuner fixes which cold mode is on the grid.)
    Returns a :class:`RingModulatorParts` fragment; connect its ``sin`` to the
    bus field, ``sout`` to the next stage, and ``vp``/``vn`` to the drive.
    Instances are namespaced by ``prefix`` so several rings can coexist.
    """
    p = dict(settings or {})
    rates = ring_cmt_rates(
        radius_um=p.get("radius_um", 7.5),
        n_g=p.get("n_g", 4.0),
        loss_db_m=p.get("loss_db_m", 7000.0),
        kappa2=p.get("kappa2", 0.10),
        cj_ff_um=p.get("cj_ff_um", 0.5),
    )
    cpl, ps, cav = f"{prefix}_CPL", f"{prefix}_PS", f"{prefix}_CAV"
    return RingModulatorParts(
        instances={
            cpl: {"component": "ring_coupler",
                  "settings": {"inv_tau_e": rates["inv_tau_e"]}},
            ps: {"component": "ring_phaseshifter",
                 "settings": {"lambda_nm": p.get("lambda_nm", 1310.0),
                              "lambda_res_nm": p.get("lambda_res_nm", 1310.0),
                              "dl_dv_pm": p.get("dl_dv_pm", 45.0),
                              "cj": rates["cj"],
                              "rleak": p.get("rleak", 1e8)}},
            cav: {"component": "ring_cavity",
                  "settings": {"inv_tau_i": rates["inv_tau_i"]}},
        },
        # the coupler, phase shifter and cavity share one internal cavity node
        connections={f"{cpl},a": (f"{ps},a", f"{cav},a")},
        models={"ring_coupler": directional_coupler(),
                "ring_phaseshifter": ring_phase_shifter(),
                "ring_cavity": cavity_mode()},
        sin=f"{cpl},sin", sout=f"{cpl},sout", vp=f"{ps},vp", vn=f"{ps},vn",
    )


# ---------------------------------------------------------------------------
# SKY130 FETs
# ---------------------------------------------------------------------------

def _showmod_card(netlist: str, device_ref: str, cache_name: str) -> dict[str, float]:
    """Load ``netlist``, ``showmod device_ref : all``, parse to a dict (cached).

    ngspice reports ``tnom`` in Kelvin; the Verilog-A parameter is Celsius, so
    it is converted here. Values carry showmod's 6 significant digits.
    """
    cache = CACHE_DIR / cache_name
    if cache.exists():
        return json.loads(cache.read_text())

    from ._ngspice import NgSpice

    ng = NgSpice.get()
    ng.load_netlist(netlist)
    ng.cmd("op")
    lines = ng.cmd(f"showmod {device_ref} : all", check=False)
    card: dict[str, float] = {}
    for line in lines:
        m = re.match(r"^\s*([a-z0-9_]+)\s+(-?[0-9][0-9.eE+-]*)\s*$", line)
        if not m:
            continue
        try:
            card[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    if "vth0" not in card or "toxe" not in card:
        raise RuntimeError(
            f"showmod extraction for {device_ref} came back without BSIM4 "
            f"params ({len(card)} values); transcript tail: {lines[-5:]}"
        )
    if "tnom" in card:
        card["tnom"] -= 273.15
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(card, indent=0))
    return card


def sky130_card(device: str, w: float, l: float, corner: str = "tt") -> dict[str, float]:  # noqa: E741
    """The fully-resolved BSIM4 model card for a sky130 FET, as ngspice sees it.

    ``device`` is the short cell name (``nfet_01v8``, ``pfet_01v8``, ...);
    ``w``/``l`` are the sky130 subckt values in **um** and only matter for
    model-bin selection. ngspice does all the heavy lifting: ``.lib``
    stitching, ``{...}`` card expressions, and W/L bin selection; we read the
    result back with ``showmod``. Cached as JSON.
    """
    key = _hash(str(toolchain.sky130_lib()), device, corner, f"{w:.6g}", f"{l:.6g}")
    return _showmod_card(
        f""".title sky130 card extraction
.lib {toolchain.sky130_lib()} {corner}
Xm1 d g 0 0 sky130_fd_pr__{device} w={w} l={l}
Vd d 0 0
Vg g 0 0
.end
""",
        f"m.xm1.msky130_fd_pr__{device}",
        f"card_{device}_{corner}_{key}.json",
    )


def _bsim4_defaults() -> dict[str, float]:
    """ngspice's resolved defaults for a bare BSIM4 (level=54) model."""
    return _showmod_card(
        """.title bare bsim4 defaults
.model mdef nmos level=54 version=4.8
M1 d g 0 0 mdef w=1u l=0.15u
Vd d 0 0
Vg g 0 0
.end
""",
        "m1",
        "bsim4_defaults.json",
    )


def sky130_fet(
    device: str,
    *,
    w: float,
    l: float,  # noqa: E741
    corner: str = "tt",
    backend: str = "osdi",
    differentiable_params: tuple[str, ...] | None = (),
):
    """A real SKY130 transistor for circulax (cached).

    ``w``/``l`` in **um** (sky130 subckt convention). They select the model
    bin and become the instance geometry; per-instance overrides go through
    netlist ``settings`` in **meters** (``{"w": 2e-6}``) and must stay inside
    the same bin. Ports are ``("d", "g", "s", "b")``.

    ``backend="osdi"`` (default) returns an ``OsdiModelDescriptor`` — exact
    BSIM4.8 physics evaluated natively; pass it in ``models_map`` like any
    component. ``backend="jax"`` returns an experimental pure-JAX component
    class (see module docstring for its accuracy caveats);
    ``differentiable_params`` applies only there.
    """
    if not BSIM4_VA.exists():
        raise FileNotFoundError(
            f"{BSIM4_VA} not found — the BSIM4.8 Verilog-A source (cogenda "
            "VA-BSIM48 port) is vendored under vendor/BSIM4/"
        )
    card = sky130_card(device, w=w, l=l, corner=corner)
    if backend == "osdi":
        return _sky130_fet_osdi(device, card, w=w, l=l)
    if backend != "jax":
        raise ValueError(f"backend must be 'osdi' or 'jax', got {backend!r}")
    defaults = _bsim4_defaults()

    static: dict[str, float] = {}
    missing: list[str] = []
    for name in _bsim4_param_names():
        # harness params (_temperature, _ckt_gmin, _mfactor, _min) and `off`
        # are solver-owned runtime inputs, never model-card values
        if name in _SKY130_RUNTIME or name.startswith("_") or name == "off":
            continue
        if name in card:
            static[name] = card[name]
        elif name in ("as", "nrd", "nrs"):
            static[name] = 0.0  # sky130 subckt defaults (see _SKY130_RUNTIME)
        elif name == "type":
            static[name] = -1.0 if device.startswith("p") else 1.0
        elif name in defaults:
            static[name] = defaults[name]
        else:
            missing.append(name)
    if missing:
        import warnings

        warnings.warn(
            f"sky130_fet({device}): {len(missing)} BSIM4 VA params have no "
            f"card value and no ngspice default, baked as 0.0: {missing}",
            stacklevel=2,
        )
        static.update(dict.fromkeys(missing, 0.0))

    # runtime defaults: requested geometry in meters, sky130 subckt zeros
    runtime_defaults = {p: 0.0 for p in _SKY130_RUNTIME}
    runtime_defaults.update({"w": w * 1e-6, "l": l * 1e-6, "nf": 1.0})

    cls = f"SKY130_{device}_{corner}".upper()
    module = _lower_cached(
        va_path=BSIM4_VA,
        class_name=cls,
        static_params=static,
        differentiable_params=differentiable_params,
        allow_analog_in_cond=True,
        runtime_defaults=runtime_defaults,
        collapse_pairs=_bsim4_collapse_pairs(static),
    )
    return module[cls]


def openvaf_ir_path() -> Path:
    """The ChipFlow-fork openvaf binary used for OSDI FET compilation.

    Override with ``PHOTONFLUX_OPENVAF_IR``. See ``toolchain.openvaf_ir_path``.
    """
    return toolchain.openvaf_ir_path()


def _bsim4_osdi() -> Path:
    """Compile bsim4.va to OSDI with the fork binary (content-hash cached)."""
    import subprocess

    binary = openvaf_ir_path()
    if not binary.exists():
        raise FileNotFoundError(
            f"{binary} not found — the SKY130 FET path needs the ChipFlow "
            "openvaf fork (github.com/robtaylor/OpenVAF, branch vajax). Build "
            "and install it in one step:\n"
            "    scripts/build-openvaf.sh\n"
            "then check the toolchain with `python -m photonflux doctor`. To "
            "use a binary elsewhere, set PHOTONFLUX_OPENVAF_IR to its path."
        )
    key = _hash(BSIM4_VA.read_text(), str(binary.stat().st_mtime_ns))
    out = CACHE_DIR / f"bsim4_{key}.osdi"
    if not out.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(binary), str(BSIM4_VA), "-o", str(out)],
            check=True, capture_output=True, text=True,
        )
    return out


_OSDI_DESCRIPTORS: dict[str, Any] = {}


def _sky130_fet_osdi(device: str, card: dict[str, float], *, w: float, l: float):  # noqa: E741
    """OSDI-backed SKY130 FET: exact BSIM4.8, evaluated natively in circulax."""
    from bosdi.circulax.osdi_component import osdi_component

    osdi_path = _bsim4_osdi()
    cache_key = f"{device}|{osdi_path}|{w:.6g}|{l:.6g}|{_hash(json.dumps(card, sort_keys=True))}"
    if cache_key in _OSDI_DESCRIPTORS:
        return _OSDI_DESCRIPTORS[cache_key]

    params: dict[str, float] = {
        "$mfactor": 1.0,  # make_instance defaults it to 0.0 -> all currents scale to 0
        "type": -1.0 if device.startswith("p") else 1.0,
        "w": w * 1e-6,
        "l": l * 1e-6,
        "nf": 1.0,
        "_ckt_gmin": 1e-12,
    }
    params.update(dict.fromkeys(
        ("ad", "as", "pd", "ps", "nrd", "nrs", "sa", "sb", "sd", "delvto"), 0.0
    ))
    # card values win; anything not given stays NaN-defaulted inside OSDI,
    # which runs the VA's own $param_given default resolution (exactly like
    # ngspice's model setup)
    params.update({k: v for k, v in card.items() if not k.endswith("_max")})

    desc = osdi_component(
        osdi_path=str(osdi_path),
        ports=("d", "g", "s", "b"),
        default_params={},
    )
    # osdi_component zero-fills parameters we don't name, but 0.0 means
    # "given as zero". NaN is the OSDI runtime's "not given" marker — it runs
    # the VA's own $param_given default resolution, exactly like ngspice —
    # so build the full vector ourselves: NaN everywhere, card values on top.
    full = dict.fromkeys(desc.param_names, float("nan"))
    for k, v in params.items():
        idx = desc._name_to_idx.get(k.lower())
        if idx is not None:
            full[desc.param_names[idx]] = v
    desc.default_params = full
    _OSDI_DESCRIPTORS[cache_key] = desc
    return desc


def _bsim4_collapse_pairs(static: dict[str, float]) -> frozenset[frozenset[str]]:
    """Which internal-node collapses are live under the card's mode flags.

    OpenVAF emits a CollapseHint for every conditional ``V(a,b) <+ 0`` in the
    VA; whether it fires depends on the resistance-mode flags, which ngspice
    evaluates at setup (rdsmod=0 merges d/di and s/si, rgatemod=0 merges
    g/gm/gi, rbodymod=0 merges the substrate network into b). bosdi 0.1.5
    applies *all* hints unconditionally, which with rbodymod=1 would short the
    live substrate network — so we filter the hints to the ones ngspice would
    actually apply for this card.
    """
    pairs: set[frozenset[str]] = set()
    if static.get("rdsmod", 0.0) == 0.0:
        pairs |= {frozenset(("d", "di")), frozenset(("s", "si"))}
    if static.get("rgatemod", 0.0) == 0.0:
        pairs |= {frozenset(("g", "gm")), frozenset(("gm", "gi"))}
    if static.get("rbodymod", 0.0) == 0.0:
        pairs |= {frozenset(("sbulk", "b")), frozenset(("b", "bi")),
                  frozenset(("b", "dbulk"))}
    return frozenset(pairs)


# ---------------------------------------------------------------------------
# lowering + cache
# ---------------------------------------------------------------------------

def _camel(stem: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^0-9a-zA-Z]+", stem) if part)


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _bsim4_param_names() -> frozenset[str]:
    """All BSIM4 VA parameter names, from a cached bare lowering."""
    cache = CACHE_DIR / f"bsim4_params_{_hash(BSIM4_VA.read_text())}.json"
    if cache.exists():
        return frozenset(json.loads(cache.read_text()))
    from bosdi.va import compile_va, lower

    dump = compile_va(str(BSIM4_VA), allow_analog_in_cond=True)
    dev = lower(dump.modules[0], class_name="Probe")
    names = sorted({p[0] for p in dev.params})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(names))
    return frozenset(names)


class _FilteredCollapseRe:
    """Drop-in for bosdi's collapse-decl regex that hides filtered-out pairs.

    ``_collapse_trivial_nodes`` only calls ``.search`` on it; returning None
    for a disallowed pair makes bosdi treat that CollapseHint as absent.
    """

    def __init__(self, orig: Any, allowed: frozenset[frozenset[str]]):
        self._orig, self._allowed = orig, allowed

    def search(self, raw: str):
        m = self._orig.search(raw)
        if m is not None and frozenset(m.groups()) not in self._allowed:
            return None
        return m


_CODE_LINE_RE = re.compile(r"^\s*//.*$", re.M)


def _check_va_support(src: Path) -> None:
    """Fail fast (and clearly) on VA constructs the JAX lowering can't do.

    * ``absdelay()`` needs solver-side signal history that neither the bosdi
      lowering nor circulax's integrators provide — without this check it
      dies much later inside the emitted-source repair with a cryptic
      branch-alias error. Model transport delay as poles (mzm_tw's walk-off
      pole) or use the webapp's vector-fitted LTI blocks (channel/fiber_cd),
      which realise delay-like responses with state-space poles.
    * ``white_noise()``/``flicker_noise()`` parse, but openvaf's MIR keeps
      noise separate from the physics and bosdi drops it — the contribution
      silently vanishes. Warn so model authors know to use the webapp's
      transient-noise seeds / .noise analysis sources instead.
    """
    text = _CODE_LINE_RE.sub("", src.read_text())
    if re.search(r"\babsdelay\s*\(", text):
        raise NotImplementedError(
            f"{src.name}: absdelay() is not supported by the VA->JAX "
            "lowering (no signal history in the solvers). Approximate the "
            "delay with poles (see models/optical_power/mzm_tw.va) or use a vector-fitted "
            "LTI component (webapp channel / fiber_cd).")
    if re.search(r"\b(white_noise|flicker_noise)\s*\(", text):
        import warnings

        warnings.warn(
            f"{src.name}: white_noise()/flicker_noise() contributions are "
            "dropped by the VA->JAX lowering — declared noise will NOT "
            "appear in simulations. Use the webapp's transient noise seeds "
            "or the small-signal noise analysis sources instead.",
            stacklevel=3)


_VA_SCALE = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3,
             "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
             "a": 1e-18}
_VA_PARAM_RE = re.compile(
    r"^\s*(?:local)?param(?:eter)?\s+real\s+(\w+)\s*=\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(T|G|M|K|k|m|u|n|p|f|a)?\s*(?:;|from)", re.M)


def _va_literal_defaults(va_text: str) -> dict[str, float]:
    """``(local)parameter real NAME = <literal>`` defaults, suffixes resolved.

    bosdi 0.1.5's ``parse_va_defaults`` misses two legal VA spellings — it
    drops ``localparam`` declarations entirely and reads scale-suffixed
    literals (``2n``, ``50p``, ``4m``) as 0.0. Either way the emitted
    component bakes a silent zero: a localparam can never be overridden at
    instantiation, so e.g. laser_rate's ``Nscl = 1e24`` normalisation became
    0.0 and froze the rate equations at N = S = 0. Only simple numeric
    literals are parsed here (expressions stay bosdi's problem); the result
    is merged over bosdi's defaults in :func:`_lower_cached`.
    """
    out: dict[str, float] = {}
    for m in _VA_PARAM_RE.finditer(va_text):
        name, num, suf = m.groups()
        out[name] = float(num) * (_VA_SCALE[suf] if suf else 1.0)
    return out


def _lower_cached(
    *,
    va_path: Path,
    class_name: str,
    static_params: dict[str, float],
    differentiable_params: tuple[str, ...] | None,
    allow_analog_in_cond: bool,
    runtime_defaults: dict[str, float] | None = None,
    collapse_pairs: frozenset[frozenset[str]] | None = None,
) -> dict[str, Any]:
    """bosdi compile+lower+emit with an on-disk source cache; returns the
    exec'd module namespace. ``collapse_pairs`` enables node collapse
    restricted to the given node pairs (None disables collapse entirely).

    NB the cache key covers the VA text and lowering inputs but NOT this
    module's emitted-source repairs (_fix_branch_aliases/_harden_emitted):
    after changing repair logic, delete the affected models/__jax__/*.py."""
    import bosdi

    va_text = va_path.read_text()
    literal_defaults = _va_literal_defaults(va_text)
    key = _hash(
        va_text,
        class_name,
        json.dumps(static_params, sort_keys=True),
        repr(differentiable_params),
        json.dumps(runtime_defaults or {}, sort_keys=True),
        json.dumps(literal_defaults, sort_keys=True),
        repr(sorted(sorted(p) for p in collapse_pairs) if collapse_pairs else None),
        getattr(bosdi, "__version__", "?"),
    )
    cache = CACHE_DIR / f"{class_name.lower()}_{key}.py"
    if not cache.exists():
        import bosdi.va.lowering as _lowering
        from bosdi.va import compile_va, emit_source, lower
        from bosdi.va.va_defaults import ParamSpec, parse_va_defaults

        dump = compile_va(str(va_path), allow_analog_in_cond=allow_analog_in_cond)
        cm = dump.modules[0]
        defaults = parse_va_defaults(va_text)
        # repair bosdi's missed localparams / scale-suffixed literals first;
        # the caller's explicit runtime_defaults still win below
        for pname, pval in literal_defaults.items():
            spec = defaults.get(pname)
            if spec is None or getattr(spec, "default", None) != pval:
                defaults[pname] = ParamSpec(type_="float", default=pval)
        for pname, pval in (runtime_defaults or {}).items():
            defaults[pname] = ParamSpec(type_="float", default=pval)

        orig_re = _lowering._COLLAPSE_DECL_RE
        try:
            if collapse_pairs:
                _lowering._COLLAPSE_DECL_RE = _FilteredCollapseRe(orig_re, collapse_pairs)
            dev = lower(
                cm,
                va_defaults=defaults,
                static_params=static_params or None,
                differentiable_params=differentiable_params,
                class_name=class_name,
                collapse_nodes=bool(collapse_pairs),
            )
        finally:
            _lowering._COLLAPSE_DECL_RE = orig_re
        src = emit_source([dev])
        src = _fix_branch_aliases(src, class_name)
        src = _harden_emitted(src)
        ast.parse(src)  # fail here, not at import time
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(src)

    ns: dict[str, Any] = {}
    code = compile(cache.read_text(), str(cache), "exec")
    exec(code, ns)  # noqa: S102 — our own generated, cached source
    if class_name not in ns:
        raise RuntimeError(f"emitted module {cache} does not define {class_name}")
    return ns


# ---------------------------------------------------------------------------
# emitted-source repair: numerical hardening
# ---------------------------------------------------------------------------
#
# Three artifacts of bosdi 0.1.5's emitter produce inf/NaN in the *Jacobian*
# even where the residual is finite (JAX evaluates both sides of jnp.where):
#
#   1. safe-divide guards are emitted as ``x / where(c, 1e-300, y)`` — when the
#      guard fires (y == 0, a branch the VA never takes), the division blows up
#      to ~1e300/inf instead of returning a don't-care. We rewrite to
#      ``where(c, 0.0, x / where(c, 1.0, y))``: identical when y is healthy,
#      harmless when it isn't.
#   2. node collapse leaves ``(signals.d - signals.d)`` self-differences.
#   3. SCCP occasionally folds a dead conductance to a literal ``float("inf")``
#      which then multiplies one of those zeros: inf * 0 = NaN.
#
# Folding self-differences to 0.0 and then pruning multiplications by that
# literal zero eliminates 2 and 3; the where-rewrite eliminates 1.


class _Hardener(ast.NodeTransformer):
    @staticmethod
    def _is_zero(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value == 0.0

    @staticmethod
    def _same_ref(a: ast.AST, b: ast.AST) -> bool:
        return (
            isinstance(a, ast.Attribute) and isinstance(b, ast.Attribute)
            and a.attr == b.attr
            and isinstance(a.value, ast.Name) and isinstance(b.value, ast.Name)
            and a.value.id == b.value.id
        )

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Sub) and self._same_ref(node.left, node.right):
            return ast.copy_location(ast.Constant(0.0), node)
        if isinstance(node.op, ast.Mult) and (
            self._is_zero(node.left) or self._is_zero(node.right)
        ):
            return ast.copy_location(ast.Constant(0.0), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # jnp.clip(x, lo, hi) — bosdi's exp()-overflow guard. jnp.clip
        # rejects complex input, but circulax's complex assembly evaluates
        # the physics at complex y (real-2N partials), so rewrite to
        #     (x - x.real) + jnp.clip(x.real, lo, hi)
        # identical for real x; clips only the (overflow-relevant) real part
        # of a complex x and passes the imaginary part through.
        if (
            isinstance(node.func, ast.Attribute) and node.func.attr == "clip"
            and len(node.args) == 3
        ):
            x, lo, hi = node.args
            x_real = ast.Attribute(value=x, attr="real", ctx=ast.Load())
            clipped = ast.Call(func=node.func, args=[x_real, lo, hi], keywords=[])
            out = ast.BinOp(
                left=ast.BinOp(left=x, op=ast.Sub(),
                               right=ast.Attribute(value=x, attr="real",
                                                   ctx=ast.Load())),
                op=ast.Add(), right=clipped,
            )
            return ast.copy_location(ast.fix_missing_locations(out), node)
        # jnp.divide(num, jnp.where(cond, 1e-300, den))
        if not (
            isinstance(node.func, ast.Attribute) and node.func.attr == "divide"
            and len(node.args) == 2 and isinstance(node.args[1], ast.Call)
        ):
            return node
        guard = node.args[1]
        if not (
            isinstance(guard.func, ast.Attribute) and guard.func.attr == "where"
            and len(guard.args) == 3
            and isinstance(guard.args[1], ast.Constant)
            and guard.args[1].value == 1e-300
        ):
            return node
        cond, den = guard.args[0], guard.args[2]
        jnp_where = ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()), attr="where", ctx=ast.Load()
        )
        safe_den = ast.Call(
            func=jnp_where, args=[cond, ast.Constant(1.0), den], keywords=[]
        )
        div = ast.Call(func=node.func, args=[node.args[0], safe_den], keywords=[])
        out = ast.Call(func=jnp_where, args=[cond, ast.Constant(0.0), div], keywords=[])
        return ast.copy_location(ast.fix_missing_locations(out), node)


def _harden_emitted(src: str) -> str:
    tree = _Hardener().visit(ast.parse(src))
    return ast.unparse(ast.fix_missing_locations(tree))


# ---------------------------------------------------------------------------
# emitted-source repair: branch-current alias collisions
# ---------------------------------------------------------------------------
#
# bosdi 0.1.5 names implicit branch-current states two different ways: the
# DAE/state side falls back to "i_br<id>" while probe reads inside the same
# function resolve to "i_<hi-node>" — and several branches can share a hi node
# (BSIM4's substrate network hangs RBSB and RBPS off the same node), so the
# emitted body references states that don't exist (`s.i_sbulk`). The repair is
# purely structural: every branch-definition row "i_brNN": (hi - lo)*g - s.X
# tells us NN's true (hi, lo); in any other row (a KCL sum for node K), an
# aliased current can only be one of the alias's branches incident to K, and
# same-alias branches share their hi node so the assignment order inside a sum
# is immaterial.

def _fix_branch_aliases(src: str, class_name: str) -> str:
    tree = ast.parse(src)

    # declared states from the @va_component(states=(...)) decorator
    states: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "va_component":
            for kw in node.keywords:
                if kw.arg == "states" and isinstance(kw.value, ast.Tuple):
                    states = {ast.literal_eval(e) for e in kw.value.elts}
    if not states:
        return src

    combined = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == f"_{class_name}_combined"),
        None,
    )
    if combined is None:
        return src

    ret = next(n for n in ast.walk(combined) if isinstance(n, ast.Return))
    if not isinstance(ret.value, ast.Tuple) or not isinstance(ret.value.elts[0], ast.Dict):
        return src
    f_dict = ret.value.elts[0]

    def s_attrs(node: ast.AST) -> list[ast.Attribute]:
        return [
            n for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "s"
        ]

    aliased = [a for a in s_attrs(f_dict) if a.attr not in states]
    if not aliased:
        return src

    def node_key(attr: ast.Attribute) -> str:
        # s.v_bi -> "v_bi" row; signals.b -> "b" row
        return attr.attr

    # branch rows: "i_brNN": ((hi - lo) * g) - s.alias   (resistive branch), or
    #              "i_brNN": ((hi - lo)) - expr           (voltage constraint —
    #              no self-current term; its unknown is aliased "i_<hi>")
    branches: dict[str, tuple[str, str, str]] = {}  # br -> (alias, hi, lo)
    for k, v in zip(f_dict.keys, f_dict.values):
        key = ast.literal_eval(k) if isinstance(k, ast.Constant) else None
        if not (isinstance(key, str) and key in states and key.startswith("i_br")):
            continue
        bad = [a for a in s_attrs(v) if a.attr not in states]
        subs = [
            n for n in ast.walk(v)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
            and isinstance(n.left, ast.Attribute) and isinstance(n.right, ast.Attribute)
        ]
        if len(bad) > 1 or not subs:
            continue
        hi, lo = node_key(subs[0].left), node_key(subs[0].right)
        alias = bad[0].attr if bad else f"i_{hi.removeprefix('v_')}"
        branches[key] = (alias, hi, lo)
        if bad:
            bad[0].attr = key  # the row's own current: s.alias -> s.i_brNN

    # KCL rows: assign each alias occurrence a distinct incident branch
    for k, v in zip(f_dict.keys, f_dict.values):
        key = ast.literal_eval(k) if isinstance(k, ast.Constant) else None
        if not isinstance(key, str) or key in branches:
            continue
        # row keys line up with the attr spelling on both sides: internal
        # nodes are "v_x" (s.v_x), ports are bare names (signals.x)
        row_node = key
        remaining = {a.attr for a in s_attrs(v) if a.attr not in states}
        for alias in remaining:
            occurrences = [a for a in s_attrs(v) if a.attr == alias]
            candidates = [
                br for br, (al, hi, lo) in branches.items()
                if al == alias and row_node in (hi, lo)
            ]
            if len(candidates) != len(occurrences):
                raise RuntimeError(
                    f"branch-alias repair failed for {class_name}: row {key!r} has "
                    f"{len(occurrences)}x s.{alias} but {len(candidates)} candidate "
                    f"branches {candidates}"
                )
            for attr, br in zip(occurrences, candidates):
                attr.attr = br

    leftovers = {
        a.attr
        for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
        for a in s_attrs(fn) if a.attr not in states
    }
    if leftovers:
        raise RuntimeError(
            f"branch-alias repair incomplete for {class_name}: unresolved {sorted(leftovers)}"
        )
    return ast.unparse(tree)
