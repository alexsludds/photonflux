"""Parameter optimization: derivative-free Nelder-Mead over instance
parameters, with finite-difference sensitivities at the optimum.

The optimizer wraps `simulate.run`: every evaluation patches the parameter
values into a copy of the schematic and runs the inner analysis, so runtime
parameters (R, gain, vpi, ...) and compile-time ones (PRBS settings, SKY130
w/l, channel fits) are handled uniformly through the caches. Objectives:

  expr:<name>     scalar defined in the expressions panel (e.g. obj = pk2pk(vout))
  eye:<probe>     worst inner eye opening at that probe (needs a PRBS source)
  ber:<probe>     log10 of the Q-fit BER from the link report (minimize)
  fom             COM-style FOM from the pulse analysis [dB]
  tdec:<probe>    IEEE 802.3 TDEC at an optical probe [dB] (minimize)
  omatdec:<probe> OMA - TDEC at an optical probe [dBm] (maximize)

Nelder-Mead is bounded by clipping to the user's min/max box. Derivative-
free beats FD-gradient descent here because eye/BER objectives are noisy
and several parameters recompile the circuit (no JAX gradient exists
through a recompile) -- and the TDEC objectives run through stateye's Cython
histogram, which has no gradient at all.
"""
from __future__ import annotations

import copy

import numpy as np

# TDEC settings the browser has no UI for yet. They matter: s_noise sets the
# saturation floor of OMA - TDEC (see photonflux.tdec.oma_tdec_floor_dbm), and
# ref_bw is the reference-receiver bandwidth TDEC is defined through.
TDEC_DEFAULTS = {"s_noise_mW": 0.01, "ber": 1e-12,
                 "ref_bw_factor": 0.75, "ref_bw_order": 4}


def _tdec_objective(res: dict, probe: str, ui_hint: float | None,
                    kind: str, cfg: dict | None = None):
    """TDEC / OMA-TDEC at an optical probe, via photonflux.tdec + stateye.

    Uses the ``_4140`` level-estimation family, not ``_8180``: the ``_8180``
    metrics need runs of >=8 identical bits, which only a full-period PRBS-13
    supplies (8191 bits ~ 154 ns at 53 GBd) -- far longer than a canvas
    transient. ``tdec_4140`` tracks ``tdec_8180`` to a near-constant +0.01 dB
    bias, so it ranks designs correctly; for the reportable _8180 number run
    examples/mrm_tdec_sky130.py.
    """
    from photonflux import tdec as _tdec

    if ui_hint is None or not (ui_hint > 0):
        raise ValueError(
            f"{kind}:{probe} needs the unit interval — add a PRBS source and "
            "set the top-bar baud rate")

    tr = next((t for t in res.get("traces", [])
               if (t.get("probe") or t["name"]) == probe), None)
    if tr is None:
        raise ValueError(f"{kind}: no probe {probe!r} in the result")
    if tr.get("unit") != "mW":
        raise ValueError(
            f"{kind}:{probe} needs an *optical* probe (power in mW); "
            f"{probe!r} is in {tr.get('unit') or '?'}. TDEC is a transmitter "
            "metric measured on optical power.")

    from photonflux.tdec import MIN_RECORD_UI

    t = np.asarray(res["x"], float)
    v = np.asarray(tr["values"], float)
    if t.size < 64:
        raise ValueError(f"{kind}: only {t.size} samples — raise `points`")

    # Record length is a property of the analysis, not of the design point, so
    # a short transient fails identically everywhere: raise it as the config
    # error it is instead of letting every point look like a bad design.
    n_ui = float(t[-1] - t[0]) / float(ui_hint)
    if n_ui < MIN_RECORD_UI + 8:      # +8 for the settling measure() discards
        raise ValueError(
            f"{kind}:{probe} has only {n_ui:.0f} UI of record; stateye needs "
            f">={MIN_RECORD_UI + 8}. Raise t_stop to at least "
            f"{(MIN_RECORD_UI + 8) * float(ui_hint) * 1e9:.1f} ns.")
    # the solver's dense output is not guaranteed uniform; stateye needs a
    # fixed dt, so resample onto the finest uniform grid the record supports
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    v = np.interp(tu, t, v)

    # A design whose eye is completely shut gives stateye no edges to lock a
    # CDR to, and its histogram reductions then run on empty arrays. That is a
    # legitimately bad point, not a broken objective, so reject it rather than
    # letting it abort the search. Everything above this line raises instead:
    # those are configuration errors that would fail identically at every point.
    span = float(np.max(v) - np.min(v))
    if not np.isfinite(span) or span < 1e-9:
        return None

    c = {**TDEC_DEFAULTS, **(cfg or {})}
    try:
        m = _tdec.measure(
            v, dt, 1.0 / float(ui_hint),
            s_noise_mW=float(c["s_noise_mW"]), ber=float(c["ber"]),
            ref_rx_bw_factor=float(c["ref_bw_factor"]),
            ref_rx_order=int(c["ref_bw_order"]),
            oma_type="4140", strict=False,
        )
    except (ValueError, IndexError, ZeroDivisionError):
        return None

    val = m["tdec_4140"] if kind == "tdec" else m["oma_tdec_dbm"]
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def _patch(sch: dict, params: list[dict], x: np.ndarray) -> dict:
    out = copy.deepcopy(sch)
    for spec, val in zip(params, x):
        inst = out["instances"].get(spec["inst"])
        if inst is None:
            raise ValueError(f"optimize: no instance {spec['inst']!r}")
        inst.setdefault("settings", {})[spec["param"]] = float(val)
    return out


def _objective(res: dict, spec: str, ui_hint: float | None,
               tdec_cfg: dict | None = None):
    import linkpost

    kind, _, arg = spec.partition(":")
    if not res.get("ok"):
        return None
    if kind in ("tdec", "omatdec"):
        return _tdec_objective(res, arg, ui_hint, kind, tdec_cfg)
    if kind == "expr":
        for s in res.get("scalars", []):
            if s["name"] == arg:
                return float(s["value"])
        return None
    if kind == "eye":
        tr = next((t for t in res.get("traces", [])
                   if (t.get("probe") or t["name"]) == arg), None)
        if tr is None or ui_hint is None:
            return None
        nlv = 2
        return linkpost.eye_height(res["x"], tr["values"], ui_hint, nlv)
    if kind == "ber":
        rep = res.get("link")
        if rep and rep.get("qfit", {}).get("ok"):
            return float(np.log10(max(rep["qfit"]["ber_est"], 1e-30)))
        return None
    if kind == "fom":
        rep = res.get("pulse")
        return float(rep["fom_db"]) if rep else None
    raise ValueError(f"unknown objective {spec!r} — use expr:<name>, "
                     "eye:<probe>, ber:<probe>, fom, tdec:<probe> or "
                     "omatdec:<probe>")


def run_optimize(payload: dict) -> dict:
    import simulate

    sch = payload.get("schematic") or {}
    analysis = dict(payload.get("analysis") or {})
    cfg = analysis.pop("optimize", None) or {}
    inner = dict(analysis)
    inner["mode"] = cfg.get("inner_mode", "transient")
    params = cfg.get("params") or []
    if not params:
        return {"ok": False, "error": "optimize: no parameters given "
                "(syntax: INST.param=min:max, ...)"}
    if len(params) > 4:
        return {"ok": False, "error": "optimize: at most 4 parameters"}
    spec = str(cfg.get("objective", "")).strip()
    if not spec:
        return {"ok": False, "error": "optimize: no objective given"}
    maximize = bool(cfg.get("maximize", True))
    iters = min(int(cfg.get("iters", 30)), 120)
    if spec.startswith("ber:") and "link" not in inner:
        inner["link"] = {"probe": spec.split(":", 1)[1],
                         "ffe_taps": 0, "dfe_taps": 0}
    if spec.startswith("fom"):
        inner["mode"] = "pulse"

    tdec_cfg = cfg.get("tdec") or {}
    if spec.startswith(("tdec:", "omatdec:")):
        try:
            import photonflux.tdec  # noqa: F401
        except ImportError as exc:
            return {"ok": False, "error":
                    f"optimize: the {spec.split(':')[0]} objective needs "
                    f"stateye, which is not installed here ({exc}). Install "
                    "the optional extra (pip install -e '.[eye]'; see "
                    "docs/patches/ if the build fails) or optimize "
                    "eye:<probe> instead."}

    lo = np.array([float(p["min"]) for p in params])
    hi = np.array([float(p["max"]) for p in params])
    # UI hint for the eye objective: the first PRBS source's UI
    ui_hint = None
    for inst in (sch.get("instances") or {}).values():
        if inst.get("type") == "prbs":
            ui_hint = float((inst.get("settings") or {}).get("ui", 100e-12))
            break

    evals = {"n": 0}
    history: list[dict] = []

    def f(x):
        x = np.clip(x, lo, hi)
        res = simulate.run({"schematic": _patch(sch, params, x),
                            "analysis": inner})
        try:
            val = _objective(res, spec, ui_hint, tdec_cfg)
        except ValueError as exc:
            # a misconfigured objective (wrong probe domain, no UI) fails the
            # same way at every point, so surface it instead of returning inf
            # forever and reporting a meaningless "optimum"
            raise RuntimeError(f"optimize: {exc}") from None
        evals["n"] += 1
        if val is None:
            return np.inf     # failed run / missing objective: reject
        history.append({"x": [float(v) for v in x], "obj": float(val)})
        return -val if maximize else val

    # start simplex: current values (clipped) + per-axis 15% span steps
    cur = []
    for p in params:
        s = (sch["instances"].get(p["inst"], {}).get("settings") or {})
        cur.append(float(s.get(p["param"], 0.5 * (float(p["min"])
                                                  + float(p["max"])))))
    x0 = np.clip(np.asarray(cur, float), lo, hi)
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        step = 0.15 * (hi[i] - lo[i])
        xi = x0.copy()
        xi[i] = xi[i] + step if xi[i] + step <= hi[i] else xi[i] - step
        simplex.append(xi)
    fs = [f(x) for x in simplex]

    # Nelder-Mead (standard reflect/expand/contract/shrink)
    while evals["n"] < iters:
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]
        cen = np.mean(simplex[:-1], axis=0)
        xr = np.clip(cen + (cen - simplex[-1]), lo, hi)
        fr = f(xr)
        if fr < fs[0]:
            xe = np.clip(cen + 2.0 * (cen - simplex[-1]), lo, hi)
            fe = f(xe)
            simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
        else:
            xc = np.clip(cen + 0.5 * (simplex[-1] - cen), lo, hi)
            fc = f(xc)
            if fc < fs[-1]:
                simplex[-1], fs[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fs[i] = f(simplex[i])

    if not history:
        return {"ok": False, "error":
                f"optimize: {spec} could not be evaluated at any of the "
                f"{evals['n']} points tried. Every run either failed or the "
                "objective was undefined there — widen the parameter box, or "
                "check that the objective's probe exists and carries the "
                "right quantity."}

    # rank only the vertices that actually evaluated: a simplex can finish with
    # its best corner in a region the objective could not score, and reporting
    # that as the optimum would be worse than useless
    finite = [i for i, v in enumerate(fs) if np.isfinite(v)]
    if not finite:
        return {"ok": False, "error":
                f"optimize: {spec} scored {len(history)} point(s) during the "
                "search but none of the final simplex vertices — the optimum "
                "sits against a region where the objective is undefined. "
                "Narrow the parameter box around the feasible region."}
    best_i = min(finite, key=lambda i: fs[i])
    best_x = np.asarray(simplex[best_i])
    best_f = fs[best_i]
    best_obj = -best_f if maximize else best_f

    # FD sensitivities at the optimum (central, 2% of span). None, not NaN:
    # json.dumps emits a bare NaN token, which is invalid JSON and makes the
    # browser's JSON.parse reject the whole response.
    sens = []
    for i, p in enumerate(params):
        h = 0.02 * (hi[i] - lo[i])
        xp, xm = best_x.copy(), best_x.copy()
        xp[i] = min(xp[i] + h, hi[i])
        xm[i] = max(xm[i] - h, lo[i])
        fp, fm = f(xp), f(xm)
        if np.isfinite(fp) and np.isfinite(fm) and xp[i] > xm[i]:
            g = (fp - fm) / (xp[i] - xm[i])
            sens.append(float(-g if maximize else g))
        else:
            sens.append(None)

    # final run at the optimum: full result for the plots
    final = simulate.run({"schematic": _patch(sch, params, best_x),
                          "analysis": inner})
    final["optim"] = {
        "objective": spec, "maximize": maximize, "best_obj": float(best_obj),
        "best": [{"inst": p["inst"], "param": p["param"],
                  "value": float(v)} for p, v in zip(params, best_x)],
        "sens": sens, "evals": evals["n"] + 2 * len(params) + 1,
        "history": [h["obj"] for h in history],
    }
    final.setdefault("log", []).append(
        f"optimize: {spec} -> {best_obj:.6g} after "
        f"{final['optim']['evals']} evaluations")
    return final
