"""Batch pre-extraction of SKY130 FET model cards (one ngspice library parse).

``photonflux.cx.sky130_card`` extracts one BSIM4 card per call, and each call
re-loads the volare corner library — the ~30-80 s cost. An AC/DC *sweep* over
``w_um``/``l_um`` needs a distinct card per value, so doing them one at a time
is minutes of the same library parse repeated.

This batches every not-yet-cached geometry into a single netlist: one ``.lib``
parse, one ``op``, then a ``showmod`` per instance. The parsed cards are
written to the exact on-disk cache files ``cx.sky130_card`` reads next
(``cx.sky130_card_cache``), so the subsequent ``cx.sky130_fet`` calls are all
cache hits.

Best-effort: any failure raises, and the caller falls back to the per-value
path (which still works, just slower).
"""
from __future__ import annotations

import json

from photonflux import cx, toolchain


def prewarm(device: str, geoms: list[tuple[float, float]], corner: str = "tt") -> int:
    """Extract every uncached (device, w, l) card in one ngspice session.

    ``geoms`` is a list of ``(w_um, l_um)``. Returns the number of cards
    actually extracted (0 if all were already cached). Raises on ngspice error.
    """
    missing: list[tuple[float, float]] = []
    seen: set[tuple[str, str]] = set()
    for w, l in geoms:  # noqa: E741
        gk = (f"{w:.6g}", f"{l:.6g}")
        if not cx.sky130_card_cache(device, w, l, corner).exists() \
                and gk not in seen:
            missing.append((w, l))
            seen.add(gk)
    if not missing:
        return 0

    lines = [".title sky130 batch card extraction",
             f".lib {toolchain.sky130_lib()} {corner}"]
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
    cx.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i, (w, l) in enumerate(missing):  # noqa: E741
        ref = f"m.xm{i}.msky130_fd_pr__{device}"
        card = cx.parse_showmod(ng.cmd(f"showmod {ref} : all", check=False),
                                f"{device} w={w} l={l}")
        cx.sky130_card_cache(device, w, l, corner).write_text(
            json.dumps(card, indent=0))
    return len(missing)
