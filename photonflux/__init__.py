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

__version__ = "0.2.0"

__all__ = [
    "cx",
    "prbs",
    "sample_centers",
    "doctor",
    "doctor_report",
    "sky130_lib",
]

# Lazy attribute loading (PEP 562): `photonflux.cx` drags in the whole JAX
# stack, which the lightweight HTTP notebook client (`photonflux.nb`) must not
# pay for — `from photonflux.nb import Session` should work in any kernel that
# has numpy. Every public name resolves exactly as before, just on first use.
_LAZY = {
    "cx": ("photonflux.cx", None),
    "nb": ("photonflux.nb", None),
    "signals": ("photonflux.signals", None),
    "toolchain": ("photonflux.toolchain", None),
    "prbs": ("photonflux.signals", "prbs"),
    "sample_centers": ("photonflux.signals", "sample_centers"),
    "doctor": ("photonflux.toolchain", "doctor"),
    "doctor_report": ("photonflux.toolchain", "doctor_report"),
    "sky130_lib": ("photonflux.toolchain", "sky130_lib"),
}


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


def __getattr__(name: str):
    try:
        modname, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    mod = importlib.import_module(modname)
    value = mod if attr is None else getattr(mod, attr)
    globals()[name] = value      # cache: __getattr__ runs once per name
    return value
