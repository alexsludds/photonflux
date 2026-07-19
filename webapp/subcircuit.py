"""Flatten user-defined subcircuit instances into a primitive netlist.

A *subcircuit definition* is a mini-schematic (its own instances/wires/probes)
plus a boundary and an exported-parameter list:

  * boundary — ``port`` pseudo-component instances mark where the definition
    connects to the outside. Each carries ``settings.name`` (the external port
    name) and ``settings.domain`` ("optical"/"electrical", for wiring checks).
    A ``port`` has a single pin ``p`` wired to the internal net it exposes.
  * params   — ``[{"name", "default", "bind": [{"instance", "param"}, ...]}]``.
    An instance of the subcircuit sets these by name; each value is written
    onto every bound inner-instance setting.

At netlist-build time :func:`flatten_subcircuits` splices every instance whose
type names a definition into its parent schematic:

  * inner refdes are namespaced with the instance ref — ``X1`` of a def whose
    body has ``WG1`` yields ``X1.WG1`` (the hierarchical name the issue asks
    for); nesting composes — ``X1.Y1.WG1``.
  * each declared ``port`` pin endpoint is merged onto the outer net wired to
    ``X1,<portname>`` (union-find downstream sees identical endpoint strings),
    and the ``port`` marker itself is dropped — it is not a real component.
  * exported-parameter values are baked into the bound inner settings.
  * probes are renamed hierarchically — an inner probe ``n_out`` becomes
    ``X1.n_out``.

Recursion is bounded by cycle detection over the definition-name stack, so a
subcircuit that instantiates itself (directly or indirectly) raises a clear
:class:`SubcircuitError` instead of recursing forever.

This module is deliberately free of catalog/JAX imports so it stays a pure,
unit-testable transform on plain schematic dicts.
"""
from __future__ import annotations

from copy import deepcopy

SEP = "."
PORT_TYPE = "port"
PORT_PIN = "p"


class SubcircuitError(ValueError):
    """A subcircuit definition or instantiation is malformed (e.g. a cycle)."""


def _wire_ends(w) -> tuple[str, str]:
    """Accept either ``[from, to]`` pairs or ``{"from", "to"}`` dicts."""
    if isinstance(w, (list, tuple)):
        return w[0], w[1]
    return w["from"], w["to"]


def _split_ep(ep: str) -> tuple[str, str]:
    ref, _, port = ep.partition(",")
    return ref, port


def subcircuit_ports(defn: dict) -> list[dict]:
    """Ordered ``[{"name", "domain"}]`` the definition exposes, read from its
    ``port`` pseudo-instances (insertion order). Nameless ports are skipped."""
    out: list[dict] = []
    insts = (defn.get("schematic") or {}).get("instances") or {}
    for inst in insts.values():
        if inst.get("type") != PORT_TYPE:
            continue
        s = inst.get("settings") or {}
        name = str(s.get("name") or "").strip()
        if not name:
            continue
        out.append({"name": name, "domain": s.get("domain") or "optical"})
    return out


def _apply_overrides(defn: dict, inst_settings: dict, inner: dict) -> None:
    """Bake this instance's exported-param values onto the bound inner settings.

    A value comes from the instance's settings when present, else the exported
    param's declared default. It is written onto every ``{instance, param}``
    binding (missing inner instances are ignored — a stale binding is not fatal).
    """
    for p in defn.get("params") or []:
        name = p.get("name")
        if not name:
            continue
        val = inst_settings.get(name, p.get("default"))
        if val is None:
            continue
        for b in p.get("bind") or []:
            tgt = inner.get(b.get("instance"))
            if tgt is None:
                continue
            tgt.setdefault("settings", {})[b.get("param")] = val


def flatten_subcircuits(instances: dict, wires: list, probes: list,
                        subcircuits: dict) -> tuple[dict, list, list]:
    """Return ``(instances, wires, probes)`` with every subcircuit instance
    spliced in and only primitive instances remaining.

    ``wires`` come back as ``[from, to]`` pairs, ``probes`` as dicts. When
    there are no definitions the inputs are returned normalized (a shallow copy
    of ``instances``, wires as pairs) so callers get one consistent shape.
    """
    out_i: dict[str, dict] = {}
    out_w: list[list] = []
    out_p: list[dict] = []
    _expand(instances or {}, wires or [], probes or [], subcircuits or {},
            "", (), out_i, out_w, out_p)
    # A ``port`` marker only means something one level up; any that reach the
    # top level (a stray port placed on a normal sheet) contribute nothing —
    # drop them and any wires/probes referencing them so the netlist stays clean.
    stray = {r for r, i in out_i.items() if i.get("type") == PORT_TYPE}
    if stray:
        for r in stray:
            del out_i[r]
        out_w = [w for w in out_w
                 if _split_ep(w[0])[0] not in stray
                 and _split_ep(w[1])[0] not in stray]
        out_p = [p for p in out_p if _split_ep(p["at"])[0] not in stray]

    return _contract_boundaries(out_i, out_w, out_p)


def _contract_boundaries(instances: dict, wires: list,
                         probes: list) -> tuple[dict, list, list]:
    """Dissolve boundary net-labels (``X1,in``, whose instance was flattened
    away) so only endpoints on real primitive instances remain.

    Splicing joins an outer net to an inner net through a boundary endpoint
    whose instance no longer exists; left in place it would reach circulax's
    ``connections`` and reference a phantom instance. We union every wire, then
    re-emit each net as a star between its *real* endpoints and rewrite probes
    onto a real endpoint of their net — simulate.py's own union-find recovers
    the identical partition.
    """
    real = set(instances)

    def is_real(ep: str) -> bool:
        return _split_ep(ep)[0] in real

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in wires:
        parent[find(a)] = find(b)
    for p in probes:            # a probe-only boundary endpoint still resolves
        find(p["at"])

    groups: dict[str, list[str]] = {}
    for ep in parent:
        groups.setdefault(find(ep), []).append(ep)

    canon: dict[str, str] = {}   # pseudo endpoint -> real representative
    new_w: list[list] = []
    for members in groups.values():
        reals = sorted({e for e in members if is_real(e)})
        rep = reals[0] if reals else None
        if rep is not None:
            for e in members:
                if not is_real(e):
                    canon[e] = rep
            for e in reals[1:]:
                new_w.append([rep, e])

    new_p: list[dict] = []
    for p in probes:
        at = p["at"] if is_real(p["at"]) else canon.get(p["at"])
        if at is None:           # a probe on a net with no real port — drop it
            continue
        p["at"] = at
        new_p.append(p)

    return instances, new_w, new_p


def _expand(instances: dict, wires: list, probes: list, subcircuits: dict,
            prefix: str, stack: tuple, out_i: dict, out_w: list,
            out_p: list) -> None:
    def ns(ref: str) -> str:
        return prefix + ref if prefix else ref

    def ns_ep(ep: str) -> str:
        ref, port = _split_ep(ep)
        return f"{ns(ref)},{port}"

    # 1. copy this level's own primitives + port markers (namespaced)
    for ref, inst in instances.items():
        if inst.get("type") not in subcircuits:
            out_i[ns(ref)] = deepcopy(inst)

    # 2. this level's own wires/probes (namespaced; port splicing is applied by
    #    whichever level instantiates *this* schematic, so only pass-through here)
    for w in wires:
        a, b = _wire_ends(w)
        out_w.append([ns_ep(a), ns_ep(b)])
    for pr in probes:
        np_ = deepcopy(pr)
        np_["name"] = ns(pr.get("name", ""))
        np_["at"] = ns_ep(pr["at"])
        out_p.append(np_)

    # 3. expand each subcircuit instance into a child namespace, then splice its
    #    boundary ports onto the outer nets
    for ref, inst in instances.items():
        typ = inst.get("type")
        if typ not in subcircuits:
            continue
        if typ in stack:
            chain = " -> ".join(stack + (typ,))
            raise SubcircuitError(
                f"subcircuit '{typ}' instantiates itself (cycle: {chain})")
        defn = subcircuits[typ]
        dsch = defn.get("schematic") or {}
        inner = deepcopy(dsch.get("instances") or {})
        _apply_overrides(defn, inst.get("settings") or {}, inner)

        child_i: dict[str, dict] = {}
        child_w: list[list] = []
        child_p: list[dict] = []
        _expand(inner, dsch.get("wires") or [], dsch.get("probes") or [],
                subcircuits, ns(ref) + SEP, stack + (typ,),
                child_i, child_w, child_p)

        # boundary: each port pin endpoint -> the outer "X1,<portname>" endpoint
        alias: dict[str, str] = {}
        for iref in list(child_i):
            if child_i[iref].get("type") != PORT_TYPE:
                continue
            pname = str((child_i[iref].get("settings") or {})
                        .get("name") or "").strip()
            if pname:
                alias[f"{iref},{PORT_PIN}"] = f"{ns(ref)},{pname}"
            del child_i[iref]      # markers are not emitted as real instances

        def fix(ep: str) -> str:
            return alias.get(ep, ep)

        out_i.update(child_i)
        for a, b in child_w:
            out_w.append([fix(a), fix(b)])
        for pr in child_p:
            pr["at"] = fix(pr["at"])
            out_p.append(pr)
