"""SKY130 PDK passive extraction: real R / C values out of ngspice.

The PDK's resistors and MiM caps are geometry-parameterized subckts/models
(sheet resistance, contact resistance, fringe capacitance, corner params —
all resolved by ngspice from the volare library). Rather than re-deriving
those formulas, each geometry is measured once inside ngspice:

  * resistor:  V=1 across the device at DC  ->  R = 1 / I(op)
  * capacitor: 1 V AC across the device      ->  C = |Im I| / (2*pi*f)

and cached as JSON under ``models/__jax__/sky130_passives.json`` (the sky130
library parse costs minutes, the cached lookup nothing). All cache misses in
a run are batched into a single netlist -> one library parse total.

In circulax the extracted value backs an ideal Resistor/Capacitor — exact at
the operating point the PDK models them for (DC/small-signal; the body pin of
the precision resistors is tied to ground for extraction).
"""
from __future__ import annotations

import json
import math

from photonflux import toolchain

CACHE_FILE = toolchain.MODELS_DIR / "__jax__" / "sky130_passives.json"
_AC_FREQ = 1e6  # far below any parasitic pole, far above numeric noise

# cell name -> (kind, element template). {n}: measurement index, {w}/{l}: um.
CELLS = {
    "res_generic_po": ("res", "R{n} n{n} 0 sky130_fd_pr__res_generic_po w={w} l={l}"),
    "res_generic_nd": ("res", "R{n} n{n} 0 sky130_fd_pr__res_generic_nd w={w} l={l}"),
    "res_high_po_0p69": ("res", "X{n} n{n} 0 0 sky130_fd_pr__res_high_po_0p69 l={l}"),
    "res_xhigh_po_0p69": ("res", "X{n} n{n} 0 0 sky130_fd_pr__res_xhigh_po_0p69 l={l}"),
    "cap_mim_m3_1": ("cap", "X{n} n{n} 0 sky130_fd_pr__cap_mim_m3_1 w={w} l={l}"),
    "cap_mim_m3_2": ("cap", "X{n} n{n} 0 sky130_fd_pr__cap_mim_m3_2 w={w} l={l}"),
}


def _key(cell: str, w: float, l: float) -> str:  # noqa: E741
    return f"{cell}|w={w:g}|l={l:g}"


def _load_cache() -> dict[str, float]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def resolve(requests: list[tuple[str, float, float]]) -> dict[str, float]:
    """[(cell, w_um, l_um), ...] -> {key: value} (ohms or farads).

    Cached geometries are returned instantly; all misses are measured in one
    ngspice session (a single sky130 library parse).
    """
    cache = _load_cache()
    out: dict[str, float] = {}
    misses: list[tuple[str, str, float, float]] = []
    for cell, w, l in requests:  # noqa: E741
        if cell not in CELLS:
            raise ValueError(f"unknown sky130 passive cell {cell!r}")
        k = _key(cell, w, l)
        if k in cache:
            out[k] = cache[k]
        elif all(k != m[0] for m in misses):
            misses.append((k, cell, w, l))
    if not misses:
        return out

    from photonflux._ngspice import NgSpice

    lines = [".title sky130 passive extraction",
             f".lib {toolchain.sky130_lib()} tt"]
    for n, (_k, cell, w, l) in enumerate(misses):  # noqa: E741
        kind, tmpl = CELLS[cell]
        src = "DC 1" if kind == "res" else "DC 0 AC 1"
        lines.append(f"V{n} n{n} 0 {src}")
        lines.append(tmpl.format(n=n, w=f"{w:g}", l=f"{l:g}"))
    lines.append(".end")

    ng = NgSpice.get()
    ng.load_netlist("\n".join(lines))
    kinds = [CELLS[cell][0] for _k, cell, _w, _l in misses]
    if "res" in kinds:
        ng.cmd("op")
        for n, (k, cell, _w, _l) in enumerate(misses):
            if CELLS[cell][0] != "res":
                continue
            i = float(ng.vector(f"v{n}#branch")[0])
            if not math.isfinite(i) or abs(i) < 1e-30:
                raise RuntimeError(f"sky130 extraction: no current through {k}")
            cache[k] = out[k] = abs(1.0 / i)
    if "cap" in kinds:
        ng.cmd(f"ac lin 1 {_AC_FREQ:g} {_AC_FREQ:g}")
        for n, (k, cell, _w, _l) in enumerate(misses):
            if CELLS[cell][0] != "cap":
                continue
            i = complex(ng.vector(f"v{n}#branch")[0])
            cache[k] = out[k] = abs(i.imag) / (2 * math.pi * _AC_FREQ)
    ng.free_plots()

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return out
