"""Schematic-JSON -> circulax netlist -> run -> plottable JSON.

The frontend sends::

    {
      "schematic": {
        "instances": {"LAS1": {"type": "cw_laser", "settings": {...}}, ...},
        "wires":     [["LAS1,p1", "MOD1,pin"], ...],
        "probes":    [{"name": "vout", "at": "OA1,out_p"}, ...]
      },
      "analysis": {"mode": "transient", "t_stop": 4e-9, "points": 800, ...}
                | {"mode": "dc"}
                | {"mode": "dcsweep", "instance": "V1", "param": "V",
                   "start": -3, "stop": 3, "points": 121}
    }

and gets back ``{"ok": true, "kind": ..., "traces": [...], "log": [...]}``.

Composites (ring_mod) are expanded here; probes are classified optical vs
electrical from the catalog port domains so the frontend can plot mW vs V.
Compiled circuits are cached on the full schematic JSON (settings included),
so a re-run with only analysis changes skips the JAX compile.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from typing import Any

from catalog import CATALOG, build_models
from progress import RunCancelled

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Flattened subcircuits produce dotted hierarchical refdes/probe names
# (``X1.WG1``, ``X1.n_out``); accept those in addition to plain names.
_HIER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

_CIRCUIT_CACHE: dict[str, Any] = {}   # schematic-hash -> (circuit, meta)
_MODELS_CACHE: dict[str, Any] = {}    # sky130-geometry-key set -> models_map


# ---------------------------------------------------------------------------
# schematic -> circulax netlist
# ---------------------------------------------------------------------------

class NetlistError(ValueError):
    pass


def _endpoint(ep: str, instances: dict) -> tuple[str, str]:
    """Validate 'INST,port' against the catalog; returns (inst, port)."""
    try:
        inst, port = ep.split(",")
    except ValueError:
        raise NetlistError(f"bad endpoint {ep!r} (want 'INSTANCE,port')")
    if inst not in instances:
        raise NetlistError(f"endpoint {ep!r}: no such instance")
    ctype = instances[inst]["type"]
    entry = CATALOG.get(ctype)
    if entry is None:
        raise NetlistError(f"instance {inst!r}: unknown component type {ctype!r}")
    if port not in {p["name"] for p in entry["ports"]}:
        raise NetlistError(f"endpoint {ep!r}: {ctype} has no port {port!r}")
    return inst, port


def _port_domain(ep: str, instances: dict) -> str:
    inst, port = ep.split(",")
    entry = CATALOG[instances[inst]["type"]]
    return next(p["domain"] for p in entry["ports"] if p["name"] == port)


DEFAULT_WAVE_SPAN = 2.56e-8   # pattern-source coverage when no t_stop known
_C_LIGHT = 299792458.0        # optical-frequency <-> wavelength conversions


def schematic_to_netlist(sch: dict, wave_span: float = DEFAULT_WAVE_SPAN,
                         noise_cfg: dict | None = None) -> tuple[dict, dict]:
    """Returns (circulax net dict, meta).

    meta: {"probe_domains": {name: domain}, "has_osdi": bool,
           "param_root": {ui_inst: circulax_inst}}

    ``wave_span`` is how much waveform the pattern sources must cover
    (bucketed from the transient t_stop so the baked arrays outlive small
    t_stop edits).
    """
    instances = sch.get("instances") or {}
    wires = sch.get("wires") or []
    probes = sch.get("probes") or []

    # Flatten user-defined subcircuits first, so everything below operates on a
    # primitive netlist with hierarchical refdes (``X1.WG1``). Definition edits
    # ride in ``sch`` and so already invalidate the compile cache.
    subcircuits = sch.get("subcircuits") or {}
    if subcircuits:
        from subcircuit import flatten_subcircuits, SubcircuitError
        try:
            instances, wires, probes = flatten_subcircuits(
                instances, wires, probes, subcircuits)
        except SubcircuitError as e:
            raise NetlistError(str(e)) from e

    if not instances:
        raise NetlistError("empty schematic — place some components first")
    for name, inst in instances.items():
        if not _HIER_NAME_RE.match(name):
            raise NetlistError(f"bad instance name {name!r}")
        if inst.get("type") not in CATALOG:
            raise NetlistError(f"{name}: unknown component type {inst.get('type')!r}")
    if not any(i["type"] == "ground" for i in instances.values()):
        raise NetlistError("no ground symbol — every circuit needs a reference")

    # --- expand composites, build endpoint remapping ------------------------
    cx_instances: dict[str, dict] = {}
    extra_connections: list[tuple[str, str]] = []
    remap: dict[str, str] = {}          # "UIINST,port" -> "CXINST,port"
    param_root: dict[str, str] = {}     # UI instance -> instance owning its settings
    passive_reqs: list[tuple[str, str, str, float, float]] = []  # PDK R/C extractions
    patterns: dict[str, dict] = {}      # PRBS instances (for eye/BER post-proc)
    rx_eq: dict[str, dict] = {}         # Rx FFE/DFE blocks (for the link report)
    ltis: dict[str, dict] = {}          # model key -> state-space payload
    lti_log: list[str] = []
    noisy: dict[str, tuple] = {}        # model key -> (kind, bank, dt_n)
    noise_insts: list[str] = []         # instances taking a seed_idx param
    has_osdi = False

    def _make_noisy(name: str, kind: str) -> str:
        import wavesrc

        seeds = int(noise_cfg.get("seeds", 1))
        bank, dt_n = wavesrc.noise_bank(
            # complex ASE needs two independent streams (re/im quadratures):
            # two bank rows per seed, split again inside the component
            name, seeds * 2 if kind == "ase" else seeds, wave_span,
            float(noise_cfg.get("bw", 50e9)),
            int(noise_cfg.get("seed", 1)))
        if kind == "ase":
            # dark start: the initial DC solve freezes n(t=0) as a constant,
            # which would coherently pre-charge every resonator at envelope
            # DC and rig laser mode competition (example 37)
            bank[:, 0] = 0.0
        key = f"_{kind}:{name}"
        noisy[key] = (kind, bank, dt_n)
        noise_insts.append(name)
        return key

    for name, inst in instances.items():
        ctype = inst["type"]
        entry = CATALOG[ctype]
        # JSON loses the int/float distinction ("V": 0 arrives as int); an int
        # setting would be traced as an int64 JAX leaf and silently truncate
        # any float sweep/override later, so coerce every number to float.
        # Catalog defaults are baked in underneath so what the UI displays is
        # exactly what simulates (they can differ from circulax's internal
        # defaults, e.g. vdc V: catalog 1.0 vs circulax 0.0). Non-numeric
        # params (enum/text, e.g. the PRBS mode) pass through untouched.
        settings = {
            p["name"]: (float(p["default"])
                        if isinstance(p["default"], (int, float))
                        and not isinstance(p["default"], bool)
                        else p["default"])
            for p in entry["params"]
        }
        settings.update({
            k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in (inst.get("settings") or {}).items()
        })
        # UI-unit -> component-unit mappings (e.g. fiber length_m -> length_um).
        # A legacy schematic that already sets the target key directly wins.
        for p in entry["params"]:
            m = p.get("map")
            if m and p["name"] in settings:
                v = float(settings.pop(p["name"])) * m.get("scale", 1.0)
                if m["to"] not in (inst.get("settings") or {}):
                    settings[m["to"]] = v
        # Rx FFE/DFE blocks are inline unity buffers in the transient (see
        # catalog._rx_eq_buffer); record their tap count + adaptation rate so
        # the link report can equalize the received probe in post-processing.
        if entry.get("eq"):
            rx_eq[name] = {"kind": entry["eq"],
                           "n_taps": int(float(settings.get("n_taps", 0))),
                           "adapt_rate": float(settings.get("adapt_rate", 0.0))}
        expand = entry.get("expand")
        if expand:
            for sub, spec in expand["instances"].items():
                sub_name = f"{name}__{sub}"
                sub_inst: dict[str, Any] = {"component": spec["component"]}
                if spec.get("settings") == "ALL":
                    sub_inst["settings"] = settings
                    param_root[name] = sub_name
                cx_instances[sub_name] = sub_inst
            for a, b in expand["connections"]:
                extra_connections.append((f"{name}__{a}", f"{name}__{b}"))
            for ui_port, target in expand["port_map"].items():
                remap[f"{name},{ui_port}"] = f"{name}__{target}"
        else:
            param_root[name] = name
            wave = entry.get("wave")
            sky = entry.get("sky130")
            if wave:
                import wavesrc

                try:
                    if wave == "prbs":
                        if str(settings.get("mode")) == "qam":
                            wt, wv = wavesrc.qam_drive_waveform(settings,
                                                               wave_span)
                        else:
                            wt, wv = wavesrc.prbs_waveform(settings, wave_span)
                        patterns[name] = dict(settings)
                    else:
                        wt, wv = wavesrc.pwl_waveform(settings)
                except ValueError as e:
                    raise NetlistError(f"{name}: {e}") from e
                wkey = wavesrc.wave_key(wave, wt, wv)
                cx_instances[name] = {"component": wkey, "_wave": (wt, wv)}
                settings = {}
            elif entry.get("lti"):
                import lti as lti_mod

                try:
                    lkey, payload = lti_mod.build(entry["lti"], settings,
                                                  lti_log)
                except ValueError as e:
                    raise NetlistError(f"{name}: {e}") from e
                ltis[lkey] = payload
                cx_instances[name] = {"component": lkey}
                settings = {}
            elif (noise_cfg and ctype == "cw_laser"
                  and float(settings.get("rin_db", 0.0)) < 0):
                cx_instances[name] = {"component": _make_noisy(name, "cwn")}
            elif noise_cfg and ctype == "photodiode":
                cx_instances[name] = {"component": _make_noisy(name, "pdn")}
            elif noise_cfg and ctype == "ase_src":
                cx_instances[name] = {"component": _make_noisy(name, "ase")}
            elif (noise_cfg and ctype == "tia"
                  and float(settings.get("in_noise", 0.0)) > 0):
                cx_instances[name] = {"component": _make_noisy(name, "tian")}
            elif sky and sky["kind"] == "fet":
                has_osdi = True
                w = float(settings.pop("w_um"))
                length = float(settings.pop("l_um"))
                model_key = f"{ctype}:{w:g}x{length:g}"
                cx_instances[name] = {"component": model_key,
                                      "_geom": (sky["device"], w, length)}
                settings = {}
            elif sky:  # res/cap: value measured from the PDK, ideal element
                w = float(settings.pop("w_um", 0.0))
                length = float(settings.pop("l_um"))
                passive_reqs.append((name, sky["kind"], sky["cell"], w, length))
                cx_instances[name] = {
                    "component": "resistor" if sky["kind"] == "res" else "capacitor"}
                settings = {}
            else:
                if ctype == "cw_laser":
                    # rin_db is a webapp-side knob; the clean cx.cw_laser
                    # doesn't take it (RIN needs the noisy variant)
                    settings.pop("rin_db", None)
                cx_instances[name] = {"component": ctype}
            if settings:
                cx_instances[name].setdefault("settings", {}).update(settings)

    # measure any PDK passives (batched: one sky130 library parse per miss set)
    extracted: dict[str, float] = {}
    if passive_reqs:
        import sky130_passives

        values = sky130_passives.resolve(
            [(cell, w, l) for _n, _k, cell, w, l in passive_reqs])
        for pname, kind, cell, w, length in passive_reqs:
            val = values[sky130_passives._key(cell, w, length)]
            cx_instances[pname]["settings"] = (
                {"R": val} if kind == "res" else {"C": val})
            extracted[pname] = (kind, val)

    def resolve(ep: str) -> str:
        return remap.get(ep, ep)

    # --- union-find nets over wire endpoints --------------------------------
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for w in wires:
        a, b = (w[0], w[1]) if isinstance(w, (list, tuple)) else (w["from"], w["to"])
        _endpoint(a, instances)
        _endpoint(b, instances)
        union(resolve(a), resolve(b))
    for a, b in extra_connections:
        union(a, b)

    nets: dict[str, list[str]] = {}
    for ep in parent:
        nets.setdefault(find(ep), []).append(ep)

    connections: dict[str, Any] = {}
    for members in nets.values():
        members = sorted(members)
        if len(members) < 2:
            continue
        head, rest = members[0], members[1:]
        connections[head] = rest[0] if len(rest) == 1 else tuple(rest)

    # --- probes -> circulax ports -------------------------------------------
    ports: dict[str, str] = {}
    probe_domains: dict[str, str] = {}
    spectrum_probes: list[str] = []       # optical probes wanting an OSA plot
    spectrum_windows: dict[str, tuple] = {}   # pname -> (start, stop) in seconds
    for probe in probes:
        pname = probe["name"]
        if not _HIER_NAME_RE.match(pname):
            raise NetlistError(f"bad probe name {pname!r}")
        _endpoint(probe["at"], instances)
        ports[pname] = resolve(probe["at"])
        probe_domains[pname] = _port_domain(probe["at"], instances)
        if probe.get("spectrum") and probe_domains[pname] == "optical":
            spectrum_probes.append(pname)
            start, stop = probe.get("spec_start"), probe.get("spec_stop")
            spectrum_windows[pname] = (
                float(start) if start is not None else None,
                float(stop) if stop is not None else None,
            )
    if not ports:
        raise NetlistError("no probes — attach at least one probe to a net "
                           "so there is something to record")

    # strip helper keys
    geoms: dict[str, tuple[float, float]] = {}
    waveforms: dict[str, tuple] = {}
    for inst in cx_instances.values():
        g = inst.pop("_geom", None)
        if g:
            geoms[inst["component"]] = g
        w = inst.pop("_wave", None)
        if w:
            waveforms[inst["component"]] = w

    # electrical-only circuits compile as real systems (half the unknowns,
    # and Circuit.ac() only supports real-valued circuits)
    any_optical = any(
        p["domain"] == "optical"
        for inst in instances.values()
        for p in CATALOG[inst["type"]]["ports"]
    )

    # hard-DC devices (laser_rate: lasing threshold bifurcation) get a
    # pseudo-transient settle before the DC point is trusted; record the
    # user's source delays so the settle can stage a clean bias ramp-up
    hard_dc = any(CATALOG[i["type"]].get("hard_dc") for i in instances.values())
    stiff = any(CATALOG[i["type"]].get("stiff") for i in instances.values())
    src_delay: dict[str, float] = {}
    if hard_dc:
        for name, inst in instances.items():
            if inst["type"] in ("vdc", "vsin"):
                src_delay[name] = float((inst.get("settings") or {}).get("delay", 0.0))
            elif inst["type"] == "vpulse":
                src_delay[name] = -1.0   # marker: pulse source

    # reference wavelength for optical-spectrum probes = the frame the field
    # envelopes live in. In a DWDM bus every laser shares one baseband
    # reference (ref_wavelength_nm); use it so each carrier lands at its true
    # wavelength. Otherwise (single carrier) the lone laser's own wavelength.
    laser_set = [(i.get("settings") or {}) for i in instances.values()
                 if i["type"] == "cw_laser"]
    ref_wl = [float(s["ref_wavelength_nm"]) for s in laser_set
              if float(s.get("ref_wavelength_nm", 0.0)) > 0.0]
    laser_wl = [float(s.get("wavelength_nm", 1310.0)) for s in laser_set]
    carrier_nm = ref_wl[0] if ref_wl else (laser_wl[0] if laser_wl else 1310.0)
    # largest WDM carrier offset from the reference frame: the transient must
    # sample fast enough (Nyquist > this) or the outer carriers alias inward.
    max_off = 0.0
    for s in laser_set:
        rw = float(s.get("ref_wavelength_nm", 0.0))
        if rw > 0.0:
            lw = float(s.get("wavelength_nm", 1310.0))
            max_off = max(max_off, abs(_C_LIGHT * (1.0 / (rw * 1e-9)
                                                   - 1.0 / (lw * 1e-9))))

    net = {"instances": cx_instances, "connections": connections, "ports": ports}
    meta = {"probe_domains": probe_domains, "has_osdi": has_osdi,
            "spectrum_probes": spectrum_probes, "carrier_nm": carrier_nm,
            "spectrum_windows": spectrum_windows,
            "wdm_max_offset_hz": max_off,
            "param_root": param_root, "sky130_geoms": geoms,
            "types": {n: i["type"] for n, i in instances.items()},
            "probe_at": {p["name"]: p["at"] for p in probes},
            "extracted": extracted, "is_complex": any_optical,
            "hard_dc": hard_dc, "stiff": stiff, "src_delay": src_delay,
            "waveforms": waveforms, "patterns": patterns, "rx_eq": rx_eq,
            "noisy": noisy, "noise_insts": noise_insts,
            "noise_cfg": noise_cfg, "ltis": ltis, "lti_log": lti_log}
    return net, meta


# ---------------------------------------------------------------------------
# compile (cached)
# ---------------------------------------------------------------------------

def _get_circuit(sch: dict, wave_span: float = DEFAULT_WAVE_SPAN,
                 noise_cfg: dict | None = None
                 ) -> tuple[Any, dict, list[str]]:
    log: list[str] = []
    key = hashlib.sha256(
        (json.dumps(sch, sort_keys=True, default=str) + f"|{wave_span:g}"
         + f"|{json.dumps(noise_cfg, sort_keys=True)}")
        .encode()
    ).hexdigest()
    if key in _CIRCUIT_CACHE:
        circuit, meta = _CIRCUIT_CACHE[key]
        log.append("compile: cache hit")
        return circuit, meta, log

    t0 = time.perf_counter()
    net, meta = schematic_to_netlist(sch, wave_span, noise_cfg)
    for pname, (kind, val) in meta["extracted"].items():
        unit = "Ohm" if kind == "res" else "F"
        log.append(f"sky130 {pname}: PDK-measured value = {val:.6g} {unit}")
    if meta["extracted"]:
        log.append(f"netlist + PDK extraction in {time.perf_counter() - t0:.2f}s")

    for line in meta["lti_log"]:
        log.append(line)
    t0 = time.perf_counter()
    mkey = ",".join(sorted(meta["sky130_geoms"]) + sorted(meta["waveforms"])
                    + sorted(meta["noisy"]) + sorted(meta["ltis"]))
    if meta["noisy"]:
        mkey += "|" + json.dumps(noise_cfg, sort_keys=True)
    if mkey not in _MODELS_CACHE:
        _MODELS_CACHE[mkey] = build_models(meta["sky130_geoms"],
                                           meta["waveforms"], meta["noisy"],
                                           meta["ltis"])
        if len(_MODELS_CACHE) > 8:   # waveform/noise closures hold arrays
            _MODELS_CACHE.pop(next(iter(_MODELS_CACHE)))
    models = _MODELS_CACHE[mkey]
    log.append(f"models ready in {time.perf_counter() - t0:.2f}s")

    from circulax import compile_circuit

    t0 = time.perf_counter()
    circuit = compile_circuit(net, models, backend="dense",
                              is_complex=meta["is_complex"], max_steps=300)
    kind = "complex" if meta["is_complex"] else "real"
    log.append(
        f"compiled in {time.perf_counter() - t0:.2f}s: "
        f"{circuit.sys_size} {kind} unknowns, {len(circuit.groups)} groups"
    )
    _CIRCUIT_CACHE[key] = (circuit, meta)
    if len(_CIRCUIT_CACHE) > 16:
        _CIRCUIT_CACHE.pop(next(iter(_CIRCUIT_CACHE)))
    return circuit, meta, log


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def _trace(name: str, domain: str, values) -> dict:
    import numpy as np

    v = np.asarray(values)
    if domain == "optical":
        return {"name": name, "domain": domain, "unit": "mW",
                "values": (np.abs(v) ** 2 * 1e3).real.tolist()}
    return {"name": name, "domain": domain, "unit": "V",
            "values": v.real.tolist()}


def _param_path(instance: str, param: str, meta: dict) -> tuple[str, float]:
    """-> (circulax param path, scale to apply to swept values)."""
    ctype = meta["types"].get(instance)
    entry = CATALOG.get(ctype, {})
    if entry.get("sky130") and param in ("w_um", "l_um"):
        raise NetlistError(
            f"{instance}.{param} is baked into the extracted PDK model and "
            "cannot be swept — edit the value and re-run instead.")
    scale = 1.0
    for p in entry.get("params", []):
        if p["name"] == param and p.get("map"):
            param = p["map"]["to"]
            scale = p["map"].get("scale", 1.0)
            break
    root = meta["param_root"].get(instance, instance)
    return f"{root}.{param}", scale


def _dc_solve(circuit, meta, log, params=None):
    """DC operating point; pseudo-transient settling for hard-DC devices.

    A laser above threshold is a bifurcation for plain Newton: the photon
    equation stalls on the dark (spontaneous-emission) branch, which is not
    an equilibrium but a Jacobian sign-flip point. The dark point is
    dynamically *unstable* there, so integrating the real dynamics for a few
    carrier lifetimes (sources frozen at their t = 0 values) flows onto the
    lasing attractor; one Newton polish from that state then converges.
    """
    if not meta.get("hard_dc"):
        return circuit.dc(params=params) if params else circuit.dc()
    import diffrax
    import numpy as np
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    # Fixed-step BDF2 from the cold (all-zero) state with the DC sources
    # applied: the first implicit step projects onto the constraint
    # manifold, then the integration follows the physical turn-on to the
    # lasing attractor (adaptive controllers reject their way to death on
    # this trajectory). Time-shaped sources are frozen at their t = 0
    # levels so the settle target matches DC-at-t=0 semantics.
    t_settle, dt = 5e-8, 1e-11    # ~25 carrier lifetimes of the default DFB
    stage = dict(params or {})
    for inst, d in meta.get("src_delay", {}).items():
        ctype = meta["types"][inst]
        if ctype == "vpulse":
            stage[f"{inst}.td"] = 1.0        # hold at v1
        elif d > 0:
            stage[f"{inst}.delay"] = 1.0     # user-delayed: stays off
        elif ctype == "vsin":
            stage[f"{inst}.freq"] = 0.0      # hold at V*sin(phase)
    sol = circuit.transient(
        t0=0.0, t1=t_settle, dt0=dt, y0=circuit._zero_guess(),
        saveat=diffrax.SaveAt(t1=True), params=stage or None,
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=circuit.solver, newton_max_steps=40),
        max_steps=int(t_settle / dt) + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    y = circuit.dc(np.asarray(sol.ys)[-1], params=params)
    if log is not None:
        log.append(f"hard-DC device: BDF2 settle ({t_settle:g}s) "
                   "+ Newton polish")
    return y


def _run_dc(circuit, meta, log) -> dict:
    t0 = time.perf_counter()
    y = _dc_solve(circuit, meta, log)
    log.append(f"DC solve in {time.perf_counter() - t0:.2f}s")
    rows = []
    for name, domain in meta["probe_domains"].items():
        val = complex(circuit.port(y, name))
        if domain == "optical":
            rows.append({"name": name, "domain": domain,
                         "value": abs(val) ** 2 * 1e3, "unit": "mW",
                         "extra": f"|E| = {abs(val):.4g}, arg = {__import__('cmath').phase(val):.3f} rad"})
        else:
            rows.append({"name": name, "domain": domain,
                         "value": val.real, "unit": "V", "extra": ""})
    return {"kind": "op", "rows": rows}


def _run_dcsweep(circuit, meta, analysis, log) -> dict:
    import jax.numpy as jnp
    import numpy as np

    inst = analysis["instance"]
    param = analysis["param"]
    # explicit value list (from the run-config pane) or start/stop/points
    if analysis.get("values"):
        x = np.asarray([float(v) for v in analysis["values"]], float)
        points = len(x)
    else:
        start, stop = float(analysis["start"]), float(analysis["stop"])
        points = max(2, min(int(analysis.get("points", 101)), 20001))
        x = np.linspace(start, stop, points)
    color_mode = (analysis.get("overlay") or {}).get("color_mode", "shaded")

    # instance "*": sweep the named parameter on every instance that has it,
    # in lockstep — e.g. wavelength_nm across all waveguides/gratings of an
    # interferometer traces its spectral response. Compile-baked parameters
    # (pattern sources, vector-fitted channels, PDK geometry) are skipped.
    if inst == "*":
        paths = []
        for iname, ctype in meta["types"].items():
            entry = CATALOG.get(ctype, {})
            if entry.get("wave") or entry.get("lti") or entry.get("sky130"):
                continue
            if any(p["name"] == param for p in entry.get("params", [])):
                paths.append(_param_path(iname, param, meta))
        if not paths:
            raise NetlistError(
                f"no swept instance: nothing on the canvas has a live "
                f"parameter {param!r}")
    else:
        paths = [_param_path(inst, param, meta)]
    path, scale = paths[0]

    # optional stepped second parameter (SPICE .step): one solve for the
    # whole family — the sweep is tiled per step value and solved vectorized
    step_inst = analysis.get("step_instance")
    svals = [float(v) for v in (analysis.get("step_values") or [])][:10]
    if step_inst and svals:
        step_param = analysis["step_param"]
        spath, sscale = _param_path(step_inst, step_param, meta)
        xx = np.tile(x, len(svals))
        ss = np.repeat(np.asarray(svals, float), points)
        t0 = time.perf_counter()
        y0 = _dc_solve(circuit, meta, log) if meta.get("hard_dc") else None
        pd = {p: jnp.asarray(xx * sc) for p, sc in paths}
        pd[spath] = jnp.asarray(ss * sscale)
        y = circuit.dc(y0, params=pd)
        log.append(
            f"DC sweep ({points} points of {len(paths)} x {param} x "
            f"{len(svals)} steps of {spath}) in "
            f"{time.perf_counter() - t0:.2f}s")
        traces = []
        for name, domain in meta["probe_domains"].items():
            vals = circuit.port(y, name)
            for si, sv in enumerate(svals):
                suffix = f"{sv:g}V" if step_param == "V" else f"{sv:g}"
                tr = _trace(f"{name} @ {suffix}", domain,
                            vals[si * points:(si + 1) * points])
                tr["probe"] = name
                if color_mode == "distinct":
                    tr["color"] = _OVERLAY_PALETTE[si % len(_OVERLAY_PALETTE)]
                else:
                    tr["step_frac"] = si / max(1, len(svals) - 1)
                traces.append(tr)
        return {"kind": "sweep", "x": x.tolist(),
                "xlabel": (f"{param} (all)" if inst == "*"
                           else f"{inst}.{param}"),
                "step_label": f"{step_inst}.{step_param}", "traces": traces}

    t0 = time.perf_counter()
    y0 = _dc_solve(circuit, meta, log) if meta.get("hard_dc") else None
    y = circuit.dc(y0, params={p: jnp.asarray(x * sc) for p, sc in paths})
    swept = path if len(paths) == 1 else f"{len(paths)} instances' {param}"
    log.append(f"DC sweep ({points} points of {swept}) in "
               f"{time.perf_counter() - t0:.2f}s")
    traces = [_trace(name, domain, circuit.port(y, name))
              for name, domain in meta["probe_domains"].items()]
    return {"kind": "sweep", "x": x.tolist(),
            "xlabel": (f"{param} (all)" if inst == "*"
                       else f"{inst}.{param}"),
            "traces": traces}


def _run_transient(circuit, meta, analysis, log) -> dict:
    import diffrax
    import jax.numpy as jnp
    import numpy as np

    t_stop = float(analysis.get("t_stop", 4e-9))
    if not (0 < t_stop <= 1.0):
        raise NetlistError(f"t_stop out of range: {t_stop}")
    points = max(16, min(int(analysis.get("points", 800)), 20000))
    # BDF2 by default for OSDI devices (required), stiff devices (lasers,
    # injection rings), and pattern-driven circuits (SERDES runs want
    # uniform steps; the PID controller reject-storms on some optical link
    # topologies).
    solver = analysis.get("solver") or (
        "bdf2" if meta["has_osdi"] or meta.get("hard_dc")
        or meta.get("stiff") or meta.get("patterns") else "adaptive")
    rtol = float(analysis.get("rtol", 1e-4))
    atol = float(analysis.get("atol", 1e-7))
    dtmax_user = analysis.get("dtmax")
    dtmax_user = float(dtmax_user) if dtmax_user else None
    dtmax = dtmax_user
    if dtmax is None:
        # Pattern/PWL sources are exogenous waveforms: in a lightly-reactive
        # circuit the error controller sees nothing and leaps whole symbols,
        # and the dense output then interpolates across them. Cap the step
        # so every UI (and PWL segment) is actually visited.
        cands = [float(p.get("ui", 100e-12)) / 8.0
                 for p in meta.get("patterns", {}).values()]
        cands += [2.0 * float(np.median(np.diff(wt)))
                  for key, (wt, _wv) in meta.get("waveforms", {}).items()
                  if key.startswith("pwl") and len(wt) > 1]
        if cands:
            dtmax = min(cands)

    # transient noise: N seeds re-run the same compiled circuit, selecting
    # each component's noise-bank row through its seed_idx runtime param
    noise_cfg = meta.get("noise_cfg")
    seeds = int(noise_cfg.get("seeds", 1)) if noise_cfg else 1
    if noise_cfg and meta.get("noisy"):
        dt_n = min(v[2] for v in meta["noisy"].values())
        dtmax = min(dtmax or 1e9, dt_n)

    ts = jnp.linspace(0.0, t_stop, points)
    # Live progress for the web UI: the solver saves at each of these `points`
    # timestamps inside the compiled loop, so a jax.debug.callback hung off the
    # SaveAt fn fires on the host as the solve steps through simulated time,
    # reporting t/t_stop in [0, 1]. The fn otherwise returns y unchanged, so the
    # saved series is identical to the default SaveAt(ts=ts). (diffrax's own
    # progress_meter= is a no-op here — circulax's circuit_diffeqsolve drops it.)
    #
    # Only wire the callback when the web server has opted in (PROGRESS.enabled):
    # a per-save-point host round-trip is pure overhead for batch callers that
    # loop over simulate.run (warmup, the optimiser, tests), so they keep the
    # plain SaveAt and run exactly as before.
    from progress import PROGRESS

    if PROGRESS.enabled:
        import jax

        def _save_fn(t, y, args):
            jax.debug.callback(PROGRESS.report, t / t_stop)
            return y

        saveat = diffrax.SaveAt(ts=ts, fn=_save_fn)
        PROGRESS.set_seeds(seeds)
    else:
        saveat = diffrax.SaveAt(ts=ts)

    def solve_one(seed_params):
        y0 = _dc_solve(circuit, meta, None, params=seed_params)
        if solver == "bdf2":
            from circulax.solvers.transient import (
                BDF2VectorizedTransientSolver,
            )

            dt = dtmax_user or min(dtmax or 1e9,
                                   t_stop / max(points * 4, 2000))
            return dt, circuit.transient(
                t0=0.0, t1=t_stop, dt0=dt, y0=y0, saveat=saveat,
                params=seed_params,
                transient_solver=BDF2VectorizedTransientSolver(
                    linear_solver=circuit.solver, newton_max_steps=40),
                max_steps=int(t_stop / dt) + 10, throw=False,
                stepsize_controller=diffrax.ConstantStepSize(),
            )
        controller = diffrax.PIDController(
            rtol=rtol, atol=atol, **({"dtmax": dtmax} if dtmax else {}))
        return None, circuit.transient(
            t0=0.0, t1=t_stop, dt0=min(1e-12, t_stop / 1e3), y0=y0,
            saveat=saveat, params=seed_params, max_steps=400_000,
            throw=False, stepsize_controller=controller,
        )

    t0c = time.perf_counter()
    sols = []
    for k in range(seeds):
        PROGRESS.set_seed(k)
        seed_params = (
            {f"{i}.seed_idx": float(k) for i in meta.get("noise_insts", [])}
            or None)
        dt_used, sol = solve_one(seed_params)
        if sol.result != diffrax.RESULTS.successful:
            raise NetlistError(
                f"transient solver failed (seed {k}): {sol.result}. Try the "
                "fixed-step BDF2 solver, a smaller max step, or check for "
                "floating nodes.")
        sols.append(sol)
    tag = (f"BDF2, dt = {dt_used:.3g}s" if solver == "bdf2"
           else "adaptive PID")
    seed_tag = f" x {seeds} noise seeds" if seeds > 1 else ""
    log.append(f"transient ({tag}){seed_tag} in "
               f"{time.perf_counter() - t0c:.2f}s")
    # WDM rotating-frame accuracy: BDF2's per-step amplitude error on a tone
    # at offset f grows ~(2*pi*f*dt)^2, so an under-resolved outer carrier is
    # silently damped (channels tilt) long before it aliases. Flag it.
    max_off = float(meta.get("wdm_max_offset_hz", 0.0) or 0.0)
    if max_off > 0.0 and solver == "bdf2" and dt_used:
        import math
        wdt = 2.0 * math.pi * max_off * dt_used
        if wdt > 0.2:
            need = 0.2 / (2.0 * math.pi * max_off)
            log.append(
                f"WARNING WDM accuracy: BDF2 dt = {dt_used:.3g}s gives "
                f"w*dt = {wdt:.2f} at the outermost carrier "
                f"(±{max_off / 1e9:.0f} GHz) — its amplitude is numerically "
                f"damped and the channels will tilt. Set max dt <= "
                f"{need:.2g}s (analysis dtmax) for <2% error.")

    t = np.asarray(sols[0].ts)
    traces = []
    for name, domain in meta["probe_domains"].items():
        for k, sol in enumerate(sols):
            tr = _trace(name if seeds == 1 else f"{name}#{k}", domain,
                        circuit.port(sol.ys, name))
            tr["probe"] = name
            if seeds > 1:
                tr["step_frac"] = k / (seeds - 1)
            traces.append(tr)
    out = {"kind": "transient", "x": t.tolist(), "xlabel": "time [s]",
           "traces": traces}

    # optical-spectrum probes: FFT the complex field envelope of each flagged
    # optical node and add an OSA-style plot (dB power vs wavelength) beside
    # the time-domain traces
    spec_names = meta.get("spectrum_probes") or []
    if spec_names:
        # aliasing guard: the saved series must resolve the WDM carriers
        max_off = float(meta.get("wdm_max_offset_hz", 0.0))
        if max_off > 0 and len(t) > 1:
            fs = 1.0 / float(t[1] - t[0])
            if max_off > 0.45 * fs:
                need = int(np.ceil(2.5 * max_off * float(t[-1] - t[0])))
                log.append(
                    f"WARNING optical spectrum: WDM carriers reach "
                    f"±{max_off / 1e9:.0f} GHz but the transient samples only "
                    f"{fs / 2e9:.0f} GHz (Nyquist) — the outer carriers ALIAS "
                    f"inward. Raise 'points' to >= {need} (or shorten t_stop) "
                    f"for an unaliased spectrum.")
        eps = _optical_spectra(circuit, sols[0], t, spec_names,
                               meta.get("carrier_nm", 1310.0), log,
                               meta.get("spectrum_windows") or {})
        if eps:
            out.setdefault("extra_plots", []).extend(eps)
    return out


def _optical_spectra(circuit, sol, t, names, carrier_nm, log, windows=None):
    """One OSA plot per flagged optical probe: |FFT(E)|^2 (dB) vs wavelength,
    centred on the CW-laser carrier.

    A single FFT (resolution fs/N, ~250 MHz over the usual full record) with a
    4-term Blackman-Harris window: the window's ~-92 dB sidelobes keep the
    floor between the lines clean (true spectrum, not leakage), and the fine
    resolution renders each tone as a sharp peak. The trace is trimmed to the
    contiguous wavelength window that holds the signal but keeps EVERY bin
    inside, so it reads as a real OSA sweep (dense floor + lines) rather than a
    handful of decimated peak samples.

    Each probe may carry a (start, stop) time window; when set, the FFT runs
    over only that slice of the transient (either bound may be None for open).
    A shorter window trades resolution for the ability to isolate a settled
    span (e.g. skipping the turn-on transient).
    """
    import numpy as np

    t = np.asarray(t)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    f0 = _C_LIGHT / (carrier_nm * 1e-9)          # carrier optical frequency
    floor_db = -90.0                             # display floor (dB below peak)
    windows = windows or {}
    plots = []
    for name in names:
        E = np.asarray(circuit.port(sol.ys, name)).astype(complex)
        win_t = windows.get(name)
        if win_t and (win_t[0] is not None or win_t[1] is not None):
            lo_t = win_t[0] if win_t[0] is not None else -np.inf
            hi_t = win_t[1] if win_t[1] is not None else np.inf
            if lo_t >= hi_t:
                log.append(f"WARNING optical spectrum {name}: window start "
                           f"{win_t[0]} >= stop {win_t[1]} — using the full "
                           f"record instead")
            elif int(((t >= lo_t) & (t <= hi_t)).sum()) >= 16:
                E = E[(t >= lo_t) & (t <= hi_t)]
            else:
                log.append(f"WARNING optical spectrum {name}: time window "
                           f"[{win_t[0]}, {win_t[1]}] s holds < 16 samples — "
                           f"using the full record instead")
        n = len(E)
        if n < 16:
            continue
        k = np.arange(n)
        win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / (n - 1))
               + 0.14128 * np.cos(4 * np.pi * k / (n - 1))
               - 0.01168 * np.cos(6 * np.pi * k / (n - 1)))
        X = np.fft.fftshift(np.fft.fft(E * win))
        S = np.abs(X) ** 2
        fb = np.fft.fftshift(np.fft.fftfreq(n, dt))      # baseband offset [Hz]
        # coherent envelopes use the physics e^{-i w t} convention, so a
        # baseband component at +fb is an optical tone at f0 - fb: a WDM laser
        # blue-shifted from the reference (shorter lambda) sits at +fb.
        wl = _C_LIGHT / (f0 - fb) * 1e9                  # -> wavelength [nm]
        sdb = 10.0 * np.log10(S / S.max() + 1e-15)
        order = np.argsort(wl)
        wl, sdb = wl[order], np.maximum(sdb[order], floor_db)
        # contiguous window: the wavelength span holding every bin within 80 dB
        # of the peak, padded, then keep ALL bins inside (a real trace)
        sig = np.where(sdb > floor_db + 10.0)[0]
        if sig.size < 2:
            continue
        lo, hi = wl[sig[0]], wl[sig[-1]]
        pad = 0.15 * (hi - lo) + 1e-4
        band = (wl >= lo - pad) & (wl <= hi + pad)
        wlk, sdk = wl[band], sdb[band]
        if wlk.size < 8:
            continue
        plots.append({
            "x": wlk.tolist(), "xlabel": "wavelength [nm]", "xunit": "nm",
            "ydb": True, "yunit": "dB",
            "traces": [{"name": f"{name} spectrum", "domain": "optical",
                        "unit": "dB", "values": sdk.tolist(),
                        "color": "#e6862c"}],
        })
    if plots:
        log.append(f"optical spectrum: {len(plots)} probe(s), "
                   f"carrier {carrier_nm:g} nm")
    return plots


def _ac_freqs(analysis):
    import numpy as np

    f_start = float(analysis.get("f_start", 1e6))
    f_stop = float(analysis.get("f_stop", 1e11))
    if not (0 < f_start < f_stop):
        raise NetlistError("AC sweep needs 0 < f_start < f_stop")
    points = max(8, min(int(analysis.get("points", 121)), 2001))
    return np.logspace(np.log10(f_start), np.log10(f_stop), points)


def _ac_pairs(port_names: list[str]) -> list[tuple[str, str, str]]:
    """Probe names -> [(suffix, in_probe, out_probe)] for each in_<x>/out_<x>."""
    pairs = []
    for name in port_names:
        if name.startswith("in_") and ("out_" + name[3:]) in port_names:
            pairs.append((name[3:], name, "out_" + name[3:]))
    if not pairs:
        raise NetlistError(
            "AC analysis pairs probes by name: add probes named in_<x> "
            "(drive) and out_<x> (response), e.g. in_01v8 / out_01v8.")
    return pairs


def _ac_system(circuit):
    """Linearize at the DC op point -> dict(G, C, gnd, n_dead).

    Assembled here rather than via circuit.ac() because collapsed OSDI
    internal nodes come back as all-zero rows — genuinely decoupled unknowns
    the DC path regularizes but the stock AC path does not — so they are
    detected and pinned before factorization.
    """
    import numpy as np

    from circulax.solvers.ac_sweep import _build_index_arrays
    from circulax.solvers.assembly import assemble_gc_real

    if any(getattr(g, "is_fdomain", False) for g in circuit.groups.values()):
        raise NetlistError("AC analysis does not support frequency-domain "
                           "(dispersive optical) components")
    y_dc = circuit.dc()
    G_vals, C_vals = assemble_gc_real(y_dc, circuit.groups)
    rows, cols, gnd, _ = _build_index_arrays(
        circuit.groups, circuit.sys_size, is_complex=False)
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    n = circuit.sys_size
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    np.add.at(G, (rows, cols), np.asarray(G_vals))
    np.add.at(C, (rows, cols), np.asarray(C_vals))
    dead = ((~np.abs(G).any(axis=1)) & (~np.abs(C).any(axis=1))
            & (~np.abs(G).any(axis=0)) & (~np.abs(C).any(axis=0)))
    G[dead, dead] = 1.0
    return {"G": G, "C": C, "gnd": np.asarray(gnd).reshape(-1),
            "n": n, "n_dead": int(dead.sum())}


def _ac_system_complex(circuit):
    """Linearize an optical (complex field-envelope) circuit into the full
    2N real-DOF (re/im) small-signal system -> dict(G, C, gnd, n, n_dead).

    circulax's real AC path (assemble_gc_real / setup_ac_sweep) stamps only the
    real block of the Jacobian, which silently drops the electro-optic coupling:
    for the power-domain laser/MZM/PD models the field envelope is real at the
    operating point, but the sensitivity of detected power to electrode voltage
    routes through the field (imaginary) DOFs that the real path discards. Here
    we recover G and C over all 2N real unknowns from the transient Newton
    matrix, which is exactly affine in 1/dt (J(dt) = G + C/dt): read it at two
    step sizes and solve the two-point system for G and C. This is the same
    (G, C) the transient integrator uses, so the AC response matches it.
    """
    import numpy as np

    import jax.numpy as jnp
    from circulax.solvers.ac_sweep import _build_index_arrays
    from circulax.solvers.assembly import assemble_system_complex

    if any(getattr(g, "is_fdomain", False) for g in circuit.groups.values()):
        raise NetlistError("AC analysis does not support frequency-domain "
                           "(dispersive optical) components")
    y_dc = jnp.asarray(np.asarray(circuit.dc()))
    # two well-separated step sizes; J is affine in 1/dt so any pair recovers
    # (G, C) exactly (float64 keeps the 1e6 spread well-conditioned).
    dt1, dt2 = 1e-6, 1e-12
    j1 = np.asarray(assemble_system_complex(y_dc, circuit.groups, 0.0, dt1)[2])
    j2 = np.asarray(assemble_system_complex(y_dc, circuit.groups, 0.0, dt2)[2])
    c_vals = (j1 - j2) / ((1.0 / dt1) - (1.0 / dt2))
    g_vals = j1 - c_vals * (1.0 / dt1)

    rows, cols, gnd, n = _build_index_arrays(
        circuit.groups, circuit.sys_size, is_complex=True)
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    np.add.at(G, (rows, cols), g_vals)
    np.add.at(C, (rows, cols), c_vals)
    dead = ((~np.abs(G).any(axis=1)) & (~np.abs(C).any(axis=1))
            & (~np.abs(G).any(axis=0)) & (~np.abs(C).any(axis=0)))
    G[dead, dead] = 1.0
    return {"G": G, "C": C, "gnd": np.asarray(gnd).reshape(-1),
            "n": n, "n_dead": int(dead.sum())}


def _ac_Y(sysm, f):
    import numpy as np

    from circulax.solvers.ac_sweep import GROUND_STIFFNESS

    Y = (sysm["G"] + 1j * 2.0 * np.pi * f * sysm["C"]).astype(complex)
    Y[sysm["gnd"], sysm["gnd"]] += GROUND_STIFFNESS
    return Y


def _run_noise(circuit, meta, analysis, log) -> dict:
    """Small-signal output noise: adjoint transfer x per-source PSDs.

    For each frequency one transposed solve z = Y^T \\ e_out gives the
    transfer from a current injection at any node pair to V(out); the output
    PSD is the sum of |z_p - z_n|^2 * S_i over sources. Included: resistor
    thermal 4kT/R (ideal + PDK-measured), TIA input-referred current noise,
    diode shot 2qId at the operating point. FET channel noise and optical
    (photodiode) noise are not included here — use transient noise seeds
    for the optical chain.
    """
    import numpy as np

    if meta["is_complex"]:
        raise NetlistError(
            "noise analysis handles electrical-only circuits — for optical "
            "links use transient noise (seeds) instead")
    probe = analysis.get("probe") or next(iter(meta["probe_domains"]))
    if probe not in meta["probe_domains"]:
        raise NetlistError(f"noise analysis: no probe named {probe!r}")
    out_node = circuit._resolve_port_node(probe)
    if out_node == 0:
        raise NetlistError("the noise output probe sits on ground")
    freqs = _ac_freqs(analysis)
    sysm = _ac_system(circuit)
    if sysm["n_dead"]:
        log.append(f"noise: pinned {sysm['n_dead']} collapsed internal nodes")

    kT4 = 4.0 * 1.380649e-23 * 300.0
    q2 = 2.0 * 1.602176634e-19
    sources: list[tuple[str, int, int, float]] = []  # label, p, n, PSD [A^2/Hz]

    grp = circuit.groups.get("resistor")
    if grp is not None:
        vi = np.asarray(grp.var_indices)
        for i, r in enumerate(np.asarray(grp.params.R)):
            sources.append(("resistors", int(vi[i, 0]), int(vi[i, 1]),
                            kT4 / float(r)))
    grp = circuit.groups.get("tia")
    if grp is not None:
        vi = np.asarray(grp.var_indices)      # [inp, out, i_vg, x1, x2, i_out]
        for i, inz in enumerate(np.asarray(grp.params.in_noise)):
            if inz > 0:
                sources.append(("TIA input", int(vi[i, 0]), 0,
                                float(inz) ** 2))
    grp = circuit.groups.get("diode")
    if grp is not None:
        y_dc = np.asarray(circuit.dc()).real
        vi = np.asarray(grp.var_indices)
        Is = np.asarray(grp.params.Is)
        nid = np.asarray(grp.params.n)
        Vt = np.asarray(grp.params.Vt)
        for i in range(vi.shape[0]):
            vd = y_dc[int(vi[i, 0])] - y_dc[int(vi[i, 1])]
            i_d = float(Is[i] * (np.exp(vd / (nid[i] * Vt[i])) - 1.0))
            sources.append(("diode shot", int(vi[i, 0]), int(vi[i, 1]),
                            q2 * abs(i_d)))
    if not sources:
        raise NetlistError(
            "no noise sources found — noise comes from resistors "
            "(incl. PDK), TIA in_noise, and diodes")
    if meta["has_osdi"]:
        log.append("noise: note — SKY130 FET channel/flicker noise is not "
                   "modelled yet; resistors/TIA/diodes only")

    e = np.zeros(sysm["n"])
    e[out_node] = 1.0
    labels = sorted({s[0] for s in sources})
    by_label = {lb: np.zeros(len(freqs)) for lb in labels}
    for k, f in enumerate(freqs):
        z = np.linalg.solve(_ac_Y(sysm, f).T, e)
        for lb, p, n_, S in sources:
            tr2 = abs(z[p] - z[n_]) ** 2
            by_label[lb][k] += tr2 * S
    total = np.sum(list(by_label.values()), axis=0)
    vrms = float(np.sqrt(np.trapezoid(total, freqs)))
    log.append(f"noise: {len(sources)} sources; integrated output noise "
               f"over [{freqs[0]:.3g}, {freqs[-1]:.3g}] Hz = {vrms:.4g} Vrms")

    palette = ["#f08fb0", "#8fd18f", "#9fa8ff", "#e6c86e"]
    traces = [{"name": f"onoise({probe})", "domain": "derived",
               "unit": "V/rtHz", "values": np.sqrt(total).tolist()}]
    if len(labels) > 1:
        for i, lb in enumerate(labels):
            traces.append({"name": lb, "domain": "derived", "unit": "V/rtHz",
                           "values": np.sqrt(by_label[lb]).tolist(),
                           "color": palette[i % len(palette)]})
    return {"kind": "noise", "x": freqs.tolist(),
            "xlabel": "frequency [Hz]", "xlog": True, "traces": traces,
            "rms": vrms}


def _src_isrc_index(circuit, meta, probe_name):
    """i_src (branch-current) index of the vdc instance a probe sits on."""
    import numpy as np

    inst = meta["probe_at"][probe_name].split(",")[0]
    node = circuit._resolve_port_node(probe_name)
    group = circuit.groups.get("vdc")
    if group is None:
        raise NetlistError("direct AC drive needs DC-voltage (vdc) sources")
    vi = np.asarray(group.var_indices)          # rows: [p1, p2, i_src]
    hits = np.where((vi[:, 0] == node) | (vi[:, 1] == node))[0]
    if len(hits) != 1:
        raise NetlistError(
            f"probe {probe_name!r}: cannot uniquely identify source {inst!r} "
            f"by its probed terminal ({len(hits)} candidates) — probe the p1 "
            "terminal, and give each measured device its own source.")
    return int(vi[hits[0], 2])


def _ac_direct(circuit, sysm, meta, pairs, freqs):
    """SPICE-style fixture-free h21: drive the gate's bias source with 1 V AC
    (all other sources are AC shorts, so the drain source is the short-circuit
    ammeter) and take the ratio of the two sources' branch currents. Nothing
    but the device itself carries the measured currents — there is no bias
    resistor or choke to influence the response.

    Returns [(suffix, out_probe, h21_db, None), ...].
    """
    import numpy as np

    drives = []   # (suffix, p_out, i_src_drive, i_src_sense)
    for suffix, p_in, p_out in pairs:
        drives.append((suffix, p_out,
                       _src_isrc_index(circuit, meta, p_in),
                       _src_isrc_index(circuit, meta, p_out)))
    # one RHS column per distinct drive source; one factorization per freq
    drive_idx = sorted({d[2] for d in drives})
    col = {ix: k for k, ix in enumerate(drive_idx)}
    B = np.zeros((sysm["n"], len(drive_idx)), complex)
    for ix in drive_idx:
        B[ix, col[ix]] = 1.0    # source constraint row: (v_p1 - v_p2) = 1 AC

    h21 = {d[0]: np.empty(len(freqs), complex) for d in drives}
    for k, f in enumerate(freqs):
        X = np.linalg.solve(_ac_Y(sysm, f), B)
        for suffix, _p_out, i_drv, i_sen in drives:
            x = X[:, col[i_drv]]
            h21[suffix][k] = x[i_sen] / x[i_drv]
    floor = 1e-12
    return [(suffix, p_out,
             20.0 * np.log10(np.maximum(np.abs(h21[suffix]), floor)), None)
            for suffix, p_out, _i, _o in drives]


def _ac_sparam(circuit, sysm, pairs, freqs, z0):
    """z0-terminated S-parameter fixture method for node-attached probe pairs.

    Returns [(suffix, out_probe, h21_db, s21_db), ...]. Note: the measured
    two-port includes whatever bias network hangs on the probed nodes; prefer
    probing the bias sources directly (the fixture-free path) for f_T.
    """
    import numpy as np

    port_names = sorted({p for _s, p_in, p_out in pairs for p in (p_in, p_out)})
    port_nodes = np.asarray([circuit._resolve_port_node(p) for p in port_names])
    if (port_nodes == 0).any():
        raise NetlistError("an AC probe sits on the ground net — move it to "
                           "a signal node")
    n_ports = len(port_names)
    eye_p = np.eye(n_ports)
    rhs = np.zeros((sysm["n"], n_ports), complex)
    rhs[port_nodes, np.arange(n_ports)] = 2.0 / z0
    S = np.empty((len(freqs), n_ports, n_ports), complex)
    for k, f in enumerate(freqs):
        Y = _ac_Y(sysm, f)
        Y[port_nodes, port_nodes] += 1.0 / z0
        S[k] = np.linalg.solve(Y, rhs)[port_nodes, :] - eye_p

    idx = {nm: i for i, nm in enumerate(port_names)}
    floor = 1e-12
    out = []
    for suffix, p_in, p_out in pairs:
        i, o = idx[p_in], idx[p_out]
        s11, s21 = S[:, i, i], S[:, o, i]
        s12, s22 = S[:, i, o], S[:, o, o]
        h21 = -2.0 * s21 / ((1.0 - s11) * (1.0 + s22) + s12 * s21)
        out.append((suffix, p_out,
                    20.0 * np.log10(np.maximum(np.abs(h21), floor)),
                    20.0 * np.log10(np.maximum(np.abs(s21), floor))))
    return out


def _measure_ac(circuit, meta, freqs, z0, log):
    """Classify probe pairs and measure each with the right method.

    A pair whose in_/out_ probes both sit on ``vdc`` source terminals is
    measured fixture-free (direct source drive / branch-current sense); pairs
    on plain circuit nodes get the z0-terminated S-parameter fixture.
    """
    pairs = _ac_pairs(list(meta["probe_domains"]))
    direct, sparam = [], []
    for pr in pairs:
        _suffix, p_in, p_out = pr
        kinds = [meta["types"].get(meta["probe_at"][p].split(",")[0]) == "vdc"
                 for p in (p_in, p_out)]
        if all(kinds):
            direct.append(pr)
        elif not any(kinds):
            sparam.append(pr)
        else:
            raise NetlistError(
                f"AC pair {_suffix!r}: put both probes on DC-source terminals "
                "(fixture-free h21) or both on circuit nodes (S-parameters) — "
                "not one of each.")
    sysm = _ac_system(circuit)
    if sysm["n_dead"]:
        log.append(f"AC: pinned {sysm['n_dead']} collapsed internal nodes")
    results = []
    if direct:
        results += _ac_direct(circuit, sysm, meta, direct, freqs)
        log.append(f"AC: {len(direct)} pair(s) measured fixture-free "
                   "(source-drive h21)")
    if sparam:
        results += _ac_sparam(circuit, sysm, sparam, freqs, z0)
        log.append(f"AC: {len(sparam)} pair(s) measured with z0 = {z0:g} Ohm "
                   "S-parameter ports")
    return results


def _ft_from_h21(h21_db, freqs):
    """First downward 0-dB crossing of |h21| (log-log interpolated), or None."""
    import numpy as np

    above = h21_db > 0
    for k in range(1, len(freqs)):
        if above[k - 1] and not above[k]:
            f0, f1 = np.log10(freqs[k - 1]), np.log10(freqs[k])
            d0, d1 = h21_db[k - 1], h21_db[k]
            return 10 ** (f0 + (0 - d0) * (f1 - f0) / (d1 - d0))
    return None


def _bw_3db(freqs, mag_db):
    """First -3.01 dB crossing of a roll-off, referenced to the low-freq band.

    ``mag_db`` is the absolute magnitude in dB; the reference is its value at
    the lowest sweep frequency (assumed to sit in the passband). Log-freq
    interpolated crossing, or None if the sweep never drops 3 dB."""
    import numpy as np

    rel = np.asarray(mag_db) - mag_db[0]
    target = -10.0 * np.log10(2.0)                     # -3.0103 dB
    below = rel <= target
    for k in range(1, len(freqs)):
        if below[k] and not below[k - 1]:
            lf = np.log10(freqs)
            frac = (target - rel[k - 1]) / (rel[k] - rel[k - 1])
            return float(10.0 ** (lf[k - 1] + frac * (lf[k] - lf[k - 1])))
    return None


def _measure_ac_optical(circuit, meta, freqs, log):
    """Small-signal EO/OE transfer H(f) = V(out_<x>) per volt on the in_<x>
    drive source, for optical (complex) circuits.

    ``in_<x>`` must sit on a DC-voltage (vdc) source terminal — the electrode
    bias source, driven with 1 V AC — and ``out_<x>`` on any signal node (the
    detected output). The full 2N field-DOF system carries the electro-optic
    coupling, so |H(f)| rolls off at the modulator's EO bandwidth exactly as a
    network analyser measures it. Returns [(suffix, out_probe, H_complex), ...].
    """
    import numpy as np

    pairs = _ac_pairs(list(meta["probe_domains"]))
    sysm = _ac_system_complex(circuit)
    if sysm["n_dead"]:
        log.append(f"AC: pinned {sysm['n_dead']} collapsed field DOFs")

    drives = []   # (suffix, p_out, i_src_drive, out_node)
    for suffix, p_in, p_out in pairs:
        out_node = circuit._resolve_port_node(p_out)
        if out_node == 0:
            raise NetlistError(f"AC pair {suffix!r}: out_{suffix} sits on the "
                               "ground net — move it to a signal node")
        drives.append((suffix, p_out,
                       _src_isrc_index(circuit, meta, p_in), out_node))

    drive_idx = sorted({d[2] for d in drives})
    col = {ix: k for k, ix in enumerate(drive_idx)}
    B = np.zeros((sysm["n"], len(drive_idx)), complex)
    for ix in drive_idx:
        B[ix, col[ix]] = 1.0          # 1 V AC on that source's constraint row
    H = {d[0]: np.empty(len(freqs), complex) for d in drives}
    for k, f in enumerate(freqs):
        X = np.linalg.solve(_ac_Y(sysm, f), B)
        for suffix, _p_out, i_drv, out_node in drives:
            H[suffix][k] = X[out_node, col[i_drv]]
    return [(suffix, p_out, H[suffix]) for suffix, p_out, _i, _o in drives]


def _run_ac_optical(circuit, meta, freqs, log) -> dict:
    """EO-bandwidth AC for optical circuits: |H(f)| of each in_/out_ pair,
    normalised to its passband (0 dB at the lowest sweep frequency)."""
    import numpy as np

    t0 = time.perf_counter()
    results = _measure_ac_optical(circuit, meta, freqs, log)
    log.append(f"EO AC sweep ({len(freqs)} freqs) in "
               f"{time.perf_counter() - t0:.2f}s")
    traces = []
    for suffix, p_out, H in results:
        floor = 1e-18
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), floor))
        mag_db -= mag_db[0]                            # 0 dB in the passband
        bw = _bw_3db(freqs, mag_db)
        if bw:
            log.append(f"EO -3 dB bandwidth({suffix}) = {bw / 1e9:.2f} GHz")
        bw_tag = f"  (-3dB {bw / 1e9:.1f}G)" if bw else ""
        traces.append({"name": f"EO |H| {suffix}{bw_tag}", "domain": "electrical",
                       "unit": "dB", "values": mag_db.tolist(),
                       "probe": p_out, "step_frac": 1.0})
    return {"kind": "ac", "x": freqs.tolist(), "xlabel": "frequency [Hz]",
            "xlog": True, "traces": traces}


def _run_ac(circuit, meta, analysis, log) -> dict:
    """AC sweep; probe pairs named ``in_<x>`` / ``out_<x>``.

    Optical (complex) circuits measure the EO/OE transfer |H(f)| = V(out) per
    volt of electrode drive (the network-analyser bandwidth of a modulator).
    Electrical circuits: pairs probed on ``vdc`` source terminals give the
    fixture-free |h21| (small-signal current gain; its 0 dB crossing is f_T);
    pairs on plain nodes give the z0 S-parameter fixture (|S21| + |h21|).
    """
    freqs = _ac_freqs(analysis)
    if meta["is_complex"]:
        return _run_ac_optical(circuit, meta, freqs, log)
    z0 = float(analysis.get("z0", 50.0))

    t0 = time.perf_counter()
    results = _measure_ac(circuit, meta, freqs, z0, log)
    log.append(f"AC sweep ({len(freqs)} freqs) in {time.perf_counter() - t0:.2f}s")

    traces = []
    for suffix, p_out, h21_db, s21_db in results:
        ft = _ft_from_h21(h21_db, freqs)
        if ft:
            log.append(f"f_T({suffix}) = {ft / 1e9:.2f} GHz  (|h21| = 0 dB)")
        # f_T in the legend label: |h21| has no flat band (gate current rises
        # with f), so the readable "bandwidth" is the 0 dB crossing, not a
        # -3 dB corner. color: both take the out_<x> probe's hue.
        ft_tag = f"  (fT {ft / 1e9:.1f}G)" if ft else ""
        traces.append({"name": f"h21 {suffix}{ft_tag}", "domain": "electrical",
                       "unit": "dB", "values": h21_db.tolist(),
                       "probe": p_out, "step_frac": 1.0})
        if s21_db is not None:
            traces.append({"name": f"S21 {suffix}", "domain": "electrical",
                           "unit": "dB", "values": s21_db.tolist(),
                           "probe": p_out, "step_frac": 0.35})
    return {"kind": "ac", "x": freqs.tolist(), "xlabel": "frequency [Hz]",
            "xlog": True, "traces": traces}


def _fmt_si(v: float) -> str:
    """Compact value label, no SI-prefixing (the param's own unit sets scale)."""
    return f"{v:g}"


def _run_ac_sweep(sch: dict, analysis: dict) -> dict:
    """AC sweep of one instance parameter — recompiles the circuit per value.

    Handles both runtime params (R, ideal-FET W, ...) and rebuild params
    (SKY130 ``w_um``/``l_um``, baked into the OSDI model): each swept value is
    applied to a fresh copy of the schematic and compiled through the normal
    cached path, so the same sweep re-runs instantly. Plots the |h21| family
    (shaded by value) and logs f_T for each value.
    """
    import copy

    inst = analysis.get("sweep_instance")
    param = analysis.get("sweep_param")
    values = [float(v) for v in (analysis.get("sweep_values") or [])][:8]
    if not (inst and param and values):
        raise NetlistError("AC parameter sweep needs sweep_instance, "
                           "sweep_param and sweep_values")

    instances = sch.get("instances") or {}
    if inst not in instances:
        raise NetlistError(f"AC sweep: no such instance {inst!r}")
    entry = CATALOG.get(instances[inst].get("type"), {})
    spec = next((p for p in entry.get("params", []) if p["name"] == param), None)
    if spec is None:
        raise NetlistError(
            f"AC sweep: {instances[inst].get('type')} has no parameter {param!r}")
    unit = spec.get("unit", "")

    log: list[str] = []
    sky = entry.get("sky130")
    if spec.get("rebuild") and sky and sky.get("kind") == "fet" \
            and param in ("w_um", "l_um"):
        # Each SKY130 geometry needs its own BSIM4 card, and extracting them one
        # at a time re-parses the volare library every time (~30-80 s each). Do
        # them all in a single library parse up front so the per-value compiles
        # below are cache hits. Best-effort: fall back to the per-value path.
        base = {p["name"]: float(p["default"]) for p in entry["params"]}
        base.update({k: float(v) for k, v in
                     (instances[inst].get("settings") or {}).items()
                     if k in ("w_um", "l_um")})
        geoms = [((v, base["l_um"]) if param == "w_um" else (base["w_um"], v))
                 for v in values]
        try:
            import sky130_cards
            t0 = time.perf_counter()
            n_new = sky130_cards.prewarm(sky["device"], geoms)
            if n_new:
                log.append(
                    f"extracted {n_new} new SKY130 {sky['device']} card(s) in "
                    f"one library parse ({time.perf_counter() - t0:.1f}s); "
                    f"{len(values) - n_new} already cached.")
        except Exception as exc:  # noqa: BLE001 — never block the sweep on this
            log.append(f"batch card prewarm skipped ({type(exc).__name__}: "
                       f"{exc}); extracting per value instead.")
    if spec.get("rebuild"):
        log.append(
            f"AC sweep of {inst}.{param}: compiling {len(values)} circuits "
            "(first use of a new geometry is slow, then cached).")

    freqs = _ac_freqs(analysis)
    z0 = float(analysis.get("z0", 50.0))
    traces = []
    ft_log: list[str] = []
    t0 = time.perf_counter()
    for vi, val in enumerate(values):
        sch2 = copy.deepcopy(sch)
        sch2["instances"][inst].setdefault("settings", {})[param] = val
        circuit, meta, _clog = _get_circuit(sch2)
        if meta["is_complex"]:
            raise NetlistError("AC analysis supports electrical-only circuits.")
        results = _measure_ac(circuit, meta, freqs, z0,
                              log if vi == 0 else [])
        frac = vi / max(1, len(values) - 1)
        vlabel = f"{_fmt_si(val)}{unit}"
        many = len(results) > 1
        for suffix, p_out, h21_db, _s21_db in results:
            ft = _ft_from_h21(h21_db, freqs)
            ft_tag = f"  (fT {ft / 1e9:.1f}G)" if ft else ""
            nm = (f"h21 {suffix} @ {vlabel}{ft_tag}" if many
                  else f"h21 @ {param}={vlabel}{ft_tag}")
            traces.append({"name": nm, "domain": "electrical", "unit": "dB",
                           "values": h21_db.tolist(),
                           "probe": p_out, "step_frac": frac})
            tag = f"{suffix}, " if many else ""
            ft_log.append(
                f"f_T({tag}{param}={vlabel}) = "
                + (f"{ft / 1e9:.2f} GHz" if ft else "n/a (no 0 dB crossing)"))
    log.append(f"AC parameter sweep: {len(values)} values of {inst}.{param} x "
               f"{len(freqs)} freqs in {time.perf_counter() - t0:.2f}s")
    log.extend(ft_log)
    return {"kind": "ac", "x": freqs.tolist(), "xlabel": "frequency [Hz]",
            "xlog": True, "traces": traces, "log": log}


# ---------------------------------------------------------------------------
# generic parameter-sweep overlay (transient / noise / pulse / 2-axis AC)
# ---------------------------------------------------------------------------
# DC and single-axis AC sweeps keep their own fast engines (_run_dcsweep,
# _run_ac_sweep); this fans ANY plot-producing analysis out over one or two
# swept parameters by re-running it per grid point (recursive run(), reusing
# every per-mode runner) and overlaying the results. Traces share the fixed
# time/freq grid, so overlay is a concat; optical spectra keep a per-run
# wavelength mask, so they are merged onto a common (union) grid instead.

_SWEEP_MAX_RUNS = 24
_OVERLAY_PALETTE = ["#6ecbf5", "#fbbf24", "#f472d0", "#4ade80", "#f87171",
                    "#9d8cff", "#e6862c", "#38bdf8", "#facc15", "#fb7185",
                    "#34d399", "#c084fc"]


def _expand_sweep_grid(axes: list) -> list:
    """axes -> runs; each run is [(inst, param, val), ...] (one tuple per axis),
    the full Cartesian product of the axes' value lists."""
    import itertools

    per_axis = []
    for ax in axes:
        inst, param = ax.get("instance"), ax.get("param")
        vals = [float(v) for v in (ax.get("values") or [])]
        if not (inst and param and vals):
            raise NetlistError("run configuration: each sweep axis needs an "
                               "instance, a parameter and at least one value")
        per_axis.append([(inst, param, v) for v in vals])
    return [list(combo) for combo in itertools.product(*per_axis)]


def _patch_point(sch: dict, inst: str, param: str, val: float) -> None:
    """Write a swept value into per-instance ``settings`` by its UI param name
    (``_get_circuit`` applies catalog map/scale downstream — same as
    optimize._patch / _run_ac_sweep). ``inst == "*"`` broadcasts to every
    non-baked instance that has the parameter (the _run_dcsweep skip-set)."""
    insts = sch.get("instances") or {}
    if inst == "*":
        hit = False
        for node in insts.values():
            entry = CATALOG.get(node.get("type"), {})
            if entry.get("wave") or entry.get("lti") or entry.get("sky130"):
                continue
            if any(p["name"] == param for p in entry.get("params", [])):
                node.setdefault("settings", {})[param] = val
                hit = True
        if not hit:
            raise NetlistError(
                f"sweep '*': nothing on the canvas has a live "
                f"parameter {param!r}")
    elif inst not in insts:
        raise NetlistError(f"sweep: no such instance {inst!r}")
    else:
        insts[inst].setdefault("settings", {})[param] = val


def _prewarm_sky130_for_sweep(sch: dict, axes: list, log: list) -> None:
    """If an axis targets a SKY130 FET geometry (w_um/l_um), extract every card
    up front in one library parse so the per-point compiles are cache hits
    (mirrors the _run_ac_sweep prewarm)."""
    insts = sch.get("instances") or {}
    for ax in axes:
        inst, param = ax.get("instance"), ax.get("param")
        if param not in ("w_um", "l_um") or inst == "*" or inst not in insts:
            continue
        entry = CATALOG.get(insts[inst].get("type"), {})
        sky = entry.get("sky130")
        spec = next((p for p in entry.get("params", [])
                     if p["name"] == param), None)
        if not (sky and sky.get("kind") == "fet"
                and spec and spec.get("rebuild")):
            continue
        base = {p["name"]: float(p["default"]) for p in entry["params"]}
        base.update({k: float(v) for k, v in
                     (insts[inst].get("settings") or {}).items()
                     if k in ("w_um", "l_um")})
        vals = [float(v) for v in (ax.get("values") or [])]
        geoms = [((v, base["l_um"]) if param == "w_um" else (base["w_um"], v))
                 for v in vals]
        try:
            import sky130_cards
            t0 = time.perf_counter()
            n_new = sky130_cards.prewarm(sky["device"], geoms)
            if n_new:
                log.append(
                    f"extracted {n_new} new SKY130 {sky['device']} card(s) in "
                    f"one library parse ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001 — never block the sweep on this
            log.append(f"batch card prewarm skipped "
                       f"({type(exc).__name__}: {exc})")


def _merge_run(merged, res, run_i, frac, label, color_mode):
    """Fold one run's result into the overlay. Main traces (fixed grid) concat;
    optical spectra are stashed for a union-grid merge at the end."""
    def tag(tr):
        tr = dict(tr)
        tr["name"] = f"{tr.get('name', '')} @ {label}"
        if color_mode == "distinct":
            tr["color"] = _OVERLAY_PALETTE[run_i % len(_OVERLAY_PALETTE)]
            tr.pop("step_frac", None)
        else:                                    # shaded family (default)
            tr["step_frac"] = frac               # traceStroke shades probe hue
            tr.pop("color", None)
        return tr

    if merged is None:
        merged = {"kind": res.get("kind"), "x": res.get("x"),
                  "xlabel": res.get("xlabel"), "xlog": res.get("xlog"),
                  "traces": [], "extra_plots": [], "_x0": res.get("x"),
                  "_ep": {}}
        if res.get("step_label"):
            merged["step_label"] = res["step_label"]

    x0, rx = merged["_x0"], res.get("x")
    if x0 is not None and rx is not None and len(rx) != len(x0):
        merged.setdefault("_warn", []).append(
            f"skipped '{label}': x length {len(rx)} != {len(x0)}")
    else:
        merged["traces"].extend(tag(t) for t in (res.get("traces") or []))

    for ep in (res.get("extra_plots") or []):
        key = f"{ep.get('xlabel', '')}|{ep.get('yunit', '')}"
        slot = merged["_ep"].setdefault(key, {"tmpl": ep, "runs": []})
        slot["runs"].append((run_i, frac, label, ep))
    return merged


def _finalize_extra_plots(merged, color_mode):
    """Merge each run's optical-spectrum plots onto a common (union) x grid so
    overlaid spectra share one axis; a bin a run dropped below its -60 dB floor
    becomes null and uPlot leaves it blank (the idler line grows in with power).
    """
    out = []
    for slot in merged["_ep"].values():
        tmpl = slot["tmpl"]
        xset = set()
        for _ri, _fr, _lab, ep in slot["runs"]:
            xset.update(round(float(v), 6) for v in ep["x"])
        union = sorted(xset)
        traces = []
        for ri, fr, lab, ep in slot["runs"]:
            base = ep["traces"][0]               # OSA plots are single-trace
            lut = {round(float(ep["x"][j]), 6): base["values"][j]
                   for j in range(len(ep["x"]))}
            tr = {"name": f"{base.get('name', '')} @ {lab}",
                  "domain": base.get("domain"), "unit": base.get("unit"),
                  "values": [lut.get(w, None) for w in union]}
            if color_mode == "distinct":
                tr["color"] = _OVERLAY_PALETTE[ri % len(_OVERLAY_PALETTE)]
            else:                                # shade the source probe's hue
                tr["step_frac"] = fr
                tr["probe"] = str(base.get("name", "")).replace(" spectrum", "")
            traces.append(tr)
        out.append({"x": union, "xlabel": tmpl.get("xlabel"),
                    "xunit": tmpl.get("xunit"), "xlog": tmpl.get("xlog"),
                    "ydb": tmpl.get("ydb"), "yunit": tmpl.get("yunit"),
                    "traces": traces})
    merged["extra_plots"] = out


def _run_sweep_overlay(payload: dict) -> dict:
    """Fan a plot-producing analysis out over 1-2 swept parameters and overlay
    the results (the generic engine behind the run-configuration pane for
    transient / noise / pulse / 2-axis AC)."""
    import copy

    sch = payload.get("schematic") or {}
    analysis = payload.get("analysis") or {}
    rc = analysis.get("run_config") or {}
    axes = rc.get("sweep") or []
    color_mode = (rc.get("overlay") or {}).get("color_mode", "shaded")

    grid = _expand_sweep_grid(axes)
    if not grid:
        raise NetlistError("run configuration: the sweep has no values")
    if len(grid) > _SWEEP_MAX_RUNS:
        raise NetlistError(
            f"parameter sweep expands to {len(grid)} runs (limit "
            f"{_SWEEP_MAX_RUNS}) — reduce the number of values.")

    inner = {k: v for k, v in analysis.items() if k != "run_config"}
    n = len(grid)
    log = [f"parameter sweep: {n} runs over "
           + " x ".join(f"{ax['instance']}.{ax['param']}" for ax in axes)]
    _prewarm_sky130_for_sweep(copy.deepcopy(sch), axes, log)

    from progress import PROGRESS
    PROGRESS.set_runs(n)  # spread the web-UI bar across all sweep points

    merged = None
    t0 = time.perf_counter()
    for i, point in enumerate(grid):
        PROGRESS.set_run(i)
        frac = i / max(1, n - 1)
        label = ", ".join(f"{inst}.{param}={_fmt_si(val)}"
                          for inst, param, val in point)
        sch_i = copy.deepcopy(sch)
        for inst, param, val in point:
            _patch_point(sch_i, inst, param, float(val))
        res = run({"schematic": sch_i, "analysis": inner})
        if res.get("cancelled"):
            # User hit Stop mid-sweep: abort the whole fan-out instead of
            # churning through the remaining points (the cancel flag stays set,
            # so each would just re-raise). _run_inner turns this into a clean
            # cancelled result.
            raise RunCancelled("run stopped by user")
        if not res.get("ok"):
            log.append(f"  [{i + 1}/{n}] {label}: FAILED — {res.get('error')}")
            continue
        if res.get("kind") == "op" or not res.get("traces"):
            raise NetlistError(
                "parameter overlay needs a plot-producing analysis; the DC "
                "operating point has no curve to overlay — use DC sweep, "
                "transient, AC or noise.")
        merged = _merge_run(merged, res, i, frac, label, color_mode)
        log.append(f"  [{i + 1}/{n}] {label}: "
                   f"{len(res.get('traces') or [])} traces")

    if merged is None:
        return {"ok": False, "log": log,
                "error": "every point in the parameter sweep failed — see log"}
    _finalize_extra_plots(merged, color_mode)
    warns = merged.pop("_warn", [])
    for k in ("_x0", "_ep"):
        merged.pop(k, None)
    log.append(f"parameter sweep total {time.perf_counter() - t0:.2f}s")
    merged["log"] = warns + log
    merged["sweep_overlay"] = True
    merged["ok"] = True
    return merged


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _link_cfg_with_eq(link: dict, meta: dict, log: list) -> dict:
    """Fold placed Rx FFE/DFE blocks into the link-report config.

    Receiver equalization is configured by dropping Rx FFE / Rx DFE blocks on
    the schematic (each carries a tap count + adaptation rate); the link
    report reads them here instead of from a global toolbar setting. With
    several blocks of one kind the first placed wins.
    """
    cfg = dict(link or {})
    rx_eq = meta.get("rx_eq") or {}
    for kind, taps_key, mu_key in (("ffe", "ffe_taps", "ffe_mu"),
                                   ("dfe", "dfe_taps", "dfe_mu")):
        blocks = [(n, e) for n, e in rx_eq.items() if e["kind"] == kind]
        if not blocks:
            continue
        name, eq = blocks[0]
        cfg[taps_key] = eq["n_taps"]
        cfg[mu_key] = eq["adapt_rate"]
        if len(blocks) > 1:
            log.append(f"link report: {len(blocks)} Rx {kind.upper()} blocks "
                       f"placed — using {name} ({eq['n_taps']} taps)")
    return cfg


def run(payload: dict) -> dict:
    # Bracket the whole run so the web UI's /api/progress poll sees a live
    # bar appear on Run and clear when the result lands. Re-entrancy safe: the
    # recursive run() calls a parameter sweep makes nest without resetting.
    from progress import PROGRESS
    PROGRESS.enter()
    try:
        return _run_inner(payload)
    finally:
        PROGRESS.leave()


def _run_inner(payload: dict) -> dict:
    try:
        sch = payload.get("schematic") or {}
        analysis = payload.get("analysis") or {"mode": "dc"}
        mode = analysis.get("mode", "dc")
        # Generic parameter-sweep overlay (transient/noise/pulse, and 2-axis
        # AC): fan the analysis out per grid point and overlay. DC and
        # single-axis AC keep their own fast engines below, so they are
        # excluded here (DC sweeps arrive as mode "dcsweep"; single-axis AC
        # carries sweep_values).
        rc = analysis.get("run_config") or {}
        if rc.get("sweep") and mode != "optimize" \
                and not (mode == "ac" and analysis.get("sweep_values")):
            return _run_sweep_overlay(payload)
        # AC parameter sweep compiles a fresh circuit per value, so it owns
        # its own circuit lifecycle (and log) instead of a single _get_circuit.
        if mode == "ac" and analysis.get("sweep_values"):
            result = _run_ac_sweep(sch, analysis)
            result["ok"] = True
            return result
        if mode == "optimize":
            import optimize

            return optimize.run_optimize(payload)
        wave_span = DEFAULT_WAVE_SPAN
        noise_cfg = None
        if mode in ("transient", "pulse"):
            import wavesrc

            wave_span = wavesrc.span_bucket(
                float(analysis.get("t_stop", 4e-9)))
            nz = analysis.get("noise") or {}
            if int(nz.get("seeds", 0) or 0) >= 1:
                noise_cfg = {"seeds": min(int(nz["seeds"]), 16),
                             "bw": float(nz.get("bw", 50e9)),
                             "seed": int(nz.get("seed", 1))}
        if mode == "pulse":
            # rerun the schematic with the PRBS source in single-pulse mode
            import copy

            sch = copy.deepcopy(sch)
            pat_inst = analysis.get("pattern") or next(
                (n for n, i in (sch.get("instances") or {}).items()
                 if i.get("type") == "prbs"), None)
            if not pat_inst:
                raise NetlistError(
                    "pulse/COM analysis needs a PRBS source on the canvas")
            sch["instances"][pat_inst].setdefault("settings", {})["mode"] = \
                "pulse"
        circuit, meta, log = _get_circuit(sch, wave_span, noise_cfg)
        if mode == "dc":
            result = _run_dc(circuit, meta, log)
        elif mode == "dcsweep":
            result = _run_dcsweep(circuit, meta, analysis, log)
        elif mode == "transient":
            result = _run_transient(circuit, meta, analysis, log)
            if analysis.get("link"):
                import linkpost

                cfg = _link_cfg_with_eq(analysis["link"], meta, log)
                result["link"] = linkpost.link_report(result, meta, cfg, log)
            if analysis.get("coherent"):
                import coherent

                result["coherent"] = coherent.coherent_report(
                    result, meta, dict(analysis["coherent"]), log)
        elif mode == "pulse":
            result = _run_transient(circuit, meta, analysis, log)
            import linkpost

            result["pulse"] = linkpost.pulse_report(
                result, meta, {**analysis, "pattern": pat_inst}, log)
        elif mode == "ac":
            result = _run_ac(circuit, meta, analysis, log)
        elif mode == "noise":
            result = _run_noise(circuit, meta, analysis, log)
        else:
            raise NetlistError(f"unknown analysis mode {mode!r}")
        if analysis.get("expressions") and result.get("traces"):
            import exprs

            exprs.apply(result, str(analysis["expressions"]), log)
        result.update({"ok": True, "log": log})
        return result
    except RunCancelled:
        # User hit Stop: not a failure, just an aborted solve. The header shows
        # "stopped" rather than a red error.
        return {"ok": False, "cancelled": True, "error": "run stopped by user"}
    except NetlistError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # solver/compile blow-ups -> readable message
        tb = traceback.format_exc().strip().splitlines()
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb[-12:]}
