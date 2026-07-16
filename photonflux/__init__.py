"""photonflux — Verilog-A photonics + SKY130 electronics, solved in circulax.

Optics and electronics live in one differentiable JAX system: Verilog-A
compact models and real SKY130 BSIM4 transistors are lowered to circulax
components (gdsfactory's JAX/Diffrax circuit simulator), so Newton DC, Diffrax
transient, AC, and ``jax.grad`` run end-to-end through the photonics.

The whole API is ``photonflux.cx``:

    from photonflux import cx

    LASER = cx.cw_laser()                              # CW field source, E = sqrt(P)
    MOD   = cx.mzm()                                   # field-convention MZM
    RING  = cx.va("ring_mod")                          # any models/*.va -> JAX component
    NFET  = cx.sky130_fet("nfet_01v8", w=1.0, l=0.15)  # real SKY130 BSIM4 device

All four drop into a circulax netlist as ordinary components; see the
``examples/`` scripts and ``webapp/`` for full circuits.

House convention: optical nodes carry the coherent field (complex ``E``,
power = ``|E|^2``); lasers are CW only and all modulation happens in modulators.
"""
from . import cx
from .signals import prbs, sample_centers
from .toolchain import doctor, doctor_report, sky130_lib

__version__ = "0.2.0"

__all__ = [
    "cx",
    "prbs",
    "sample_centers",
    "doctor",
    "doctor_report",
    "sky130_lib",
]
