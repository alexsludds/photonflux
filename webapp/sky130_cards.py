"""Batch pre-extraction of SKY130 FET model cards (one ngspice library parse).

``photonflux.cx.sky130_card`` extracts one BSIM4 card per call, and each call
re-loads the volare corner library — the ~30-80 s cost. An AC/DC *sweep* over
``w_um``/``l_um`` needs a distinct card per value, so doing them one at a time
is minutes of the same library parse repeated.

This batches every not-yet-cached geometry into a single netlist: one ``.lib``
parse, one ``op``, then a ``showmod`` per instance. The parsed cards are
written to the exact on-disk cache files ``cx.sky130_card`` reads next (same
hash + path), so the subsequent ``cx.sky130_fet`` calls are all cache hits.

Best-effort: any failure raises, and the caller falls back to the per-value
path (which still works, just slower).
"""
from __future__ import annotations

import json
import re

from photonflux import cx, toolchain

_CACHE = toolchain.MODELS_DIR / "__jax__"
_CARD_RE = re.compile(r"^\s*([a-z0-9_]+)\s+(-?[0-9][0-9.eE+-]*)\s*$")


def _cache_path(device: str, w: float, l: float, corner: str):  # noqa: E741
    key = cx._hash(str(toolchain.sky130_lib()), device, corner,
                   f"{w:.6g}", f"{l:.6g}")
    return _CACHE / f"card_{device}_{corner}_{key}.json"


def prewarm(device: str, geoms: list[tuple[float, float]], corner: str = "tt") -> int:
    """Extract every uncached (device, w, l) card in one ngspice session.

    ``geoms`` is a list of ``(w_um, l_um)``. Returns the number of cards
    actually extracted (0 if all were already cached). Raises on ngspice error.
    """
    missing: list[tuple[float, float]] = []
    seen: set[tuple[str, str]] = set()
    for w, l in geoms:  # noqa: E741
        path = _cache_path(device, w, l, corner)
        gk = (f"{w:.6g}", f"{l:.6g}")
        if not path.exists() and gk not in seen:
            missing.append((w, l))
            seen.add(gk)
    if not missing:
        return 0

    lib = toolchain.sky130_lib()
    lines = [".title sky130 batch card extraction", f".lib {lib} {corner}"]
    for i, (w, l) in enumerate(missing):  # noqa: E741
        lines.append(
            f"Xm{i} d{i} g 0 0 sky130_fd_pr__{device} w={w:g} l={l:g}")
        lines.append(f"Vd{i} d{i} 0 0")
    lines.append("Vg g 0 0")
    lines.append(".end")

    from photonflux._ngspice import NgSpice

    ng = NgSpice.get()
    ng.load_netlist("\n".join(lines))
    ng.cmd("op")
    _CACHE.mkdir(parents=True, exist_ok=True)
    for i, (w, l) in enumerate(missing):  # noqa: E741
        ref = f"m.xm{i}.msky130_fd_pr__{device}"
        out = ng.cmd(f"showmod {ref} : all", check=False)
        card: dict[str, float] = {}
        for line in out:
            m = _CARD_RE.match(line)
            if not m:
                continue
            try:
                card[m.group(1)] = float(m.group(2))
            except ValueError:
                continue
        if "vth0" not in card or "toxe" not in card:
            raise RuntimeError(
                f"batch showmod for {device} w={w} l={l} came back without "
                f"BSIM4 params ({len(card)} values); tail: {out[-4:]}")
        if "tnom" in card:
            card["tnom"] -= 273.15  # ngspice reports Kelvin; VA wants Celsius
        _cache_path(device, w, l, corner).write_text(json.dumps(card, indent=0))
    return len(missing)
