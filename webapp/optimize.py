"""Parameter optimization: derivative-free Nelder-Mead over instance
parameters, with finite-difference sensitivities at the optimum.

The optimizer wraps `simulate.run`: every evaluation patches the parameter
values into a copy of the schematic and runs the inner analysis, so runtime
parameters (R, gain, vpi, ...) and compile-time ones (PRBS settings, SKY130
w/l, channel fits) are handled uniformly through the caches. Objectives:

  expr:<name>   scalar defined in the expressions panel (e.g. obj = pk2pk(vout))
  eye:<probe>   worst inner eye opening at that probe (needs a PRBS source)
  ber:<probe>   log10 of the Q-fit BER from the link report (minimize)
  fom           COM-style FOM from the pulse analysis [dB]

Nelder-Mead is bounded by clipping to the user's min/max box. Derivative-
free beats FD-gradient descent here because eye/BER objectives are noisy
and several parameters recompile the circuit (no JAX gradient exists
through a recompile).
"""
from __future__ import annotations

import copy

import numpy as np


def _patch(sch: dict, params: list[dict], x: np.ndarray) -> dict:
    out = copy.deepcopy(sch)
    for spec, val in zip(params, x):
        inst = out["instances"].get(spec["inst"])
        if inst is None:
            raise ValueError(f"optimize: no instance {spec['inst']!r}")
        inst.setdefault("settings", {})[spec["param"]] = float(val)
    return out


def _objective(res: dict, spec: str, ui_hint: float | None):
    import linkpost

    kind, _, arg = spec.partition(":")
    if not res.get("ok"):
        return None
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
                     "eye:<probe>, ber:<probe> or fom")


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
        val = _objective(res, spec, ui_hint)
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

    order = np.argsort(fs)
    best_x = np.asarray(simplex[order[0]])
    best_f = fs[order[0]]
    best_obj = -best_f if maximize else best_f

    # FD sensitivities at the optimum (central, 2% of span)
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
            sens.append(float("nan"))

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
