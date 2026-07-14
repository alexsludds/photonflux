"""Photonic device access and link-building helpers.

Optical convention (power domain): a node voltage represents optical power
in watts. Lasers source it, modulators scale it, photodiodes convert it to
current. The Verilog-A sources in models/ are the single source of truth
for device physics — compiled (with caching) on first use.

A coherent-field convention (paired Ereal/Eimag nodes, see laser_cw.va and
cart2pol.va / pol2cart.va) is available for phase-sensitive experiments.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import toolchain
from .circuit import Circuit
from .compiler import VaModule, compile_va

__all__ = ["va", "library", "add_fiber", "attenuation_lin"]

_cache: dict[Path, VaModule] = {}


def va(name: str | Path, models_dir: str | Path | None = None) -> VaModule:
    """Get a compiled Verilog-A module by model name or .va path.

    `va("laser_dml")` resolves to models/laser_dml.va; an explicit path
    works for models outside the repo. Compilation is content-hash cached.
    """
    p = Path(name)
    if p.suffix != ".va":
        base = Path(models_dir) if models_dir else toolchain.MODELS_DIR
        p = base / f"{name}.va"
    p = p.resolve()
    if p not in _cache:
        _cache[p] = compile_va(p)
    return _cache[p]


def library() -> dict[str, VaModule]:
    """Compile and return every model in models/, keyed by module name."""
    from .compiler import compile_all

    return {m.name: m for m in compile_all()}


def attenuation_lin(loss_db: float) -> float:
    """Power-domain attenuation factor for a loss in dB (optical power is a
    node *voltage*, so it scales linearly with the power loss)."""
    return 10.0 ** (-loss_db / 10.0)


def add_fiber(
    ckt: Circuit,
    name: str,
    nin: str,
    nout: str,
    loss_db: float = 3.0,
    delay: float = 0.0,
    length_m: float | None = None,
    n_group: float = 1.468,
) -> None:
    """Optical fiber in the power domain: attenuation plus (optionally)
    true group delay.

    Delay uses a source-driven, far-end-matched ideal transmission line, so
    it is a pure delay with no reflections. Specify either `delay` seconds
    or a physical `length_m` (delay = n_g * L / c).
    """
    if length_m is not None:
        delay = n_group * length_m / 299_792_458.0
    k = attenuation_lin(loss_db)
    if delay <= 0:
        ckt.raw(f"E{name} {nout} 0 {nin} 0 {k:.6g}")
        return
    a, b = f"{name}_a", f"{name}_b"
    ckt.raw(
        f"""
        * fiber {name}: {loss_db} dB, {delay*1e12:.3f} ps group delay
        E{name}_in {a} 0 {nin} 0 {k:.6g}
        T{name} {a} 0 {b} 0 Z0=50 TD={delay:.6e}
        R{name}_term {b} 0 50
        E{name}_out {nout} 0 {b} 0 1
        """
    )


def dbm(p_watts: float) -> float:
    return 10.0 * math.log10(max(p_watts, 1e-300) / 1e-3)
