"""Testbench flows: a pre-configured chain of analyses run as one job.

A flow lives in a testbench next to its single ``analysis`` (``doc["flow"]``)
and reaches ``/api/run`` as ``analysis = {"mode": "flow", "flow": {...}}``.
Its steps share *flow variables*::

    "flow": {
      "name": "Resonance -> eye -> optimize -> evaluate",
      "vars": {"detune_pm": 2.5},                  # knobs (optimizable)
      "set":  {"PRBS.v0": "vbias - vswing/2"},     # design mapping, every step
      "stateye": {...},                            # the Eye tab's settings
      "steps": [
        {"id": "res", "kind": "analysis",          # any single-run analysis
         "analysis": {"mode": "dcsweep", ...},
         "set": {"*.wavelength_nm": "..."},       # this step only
         "outputs": {"lam_res": "xat_min(thru)"},  # scalars -> variables
         "checks": ["notch_db < -3"]},
        {"id": "opt", "kind": "optimize", "inner": ["res", "eye"],
         "params": [{"var": "detune_pm", "min": 1, "max": 6},
                    {"inst": "WG_A", "param": "length_m", "min": .., "max": ..}],
         "objective": "rel_oma_db", "maximize": true,
         "constraints": ["rlm >= 0.96"], "iters": 30},
        {"id": "eval", "kind": "evaluate", "inner": ["res", "eye"],
         "record": "full_period", "probes": {"in": "mod_in", "out": "thru"},
         "fom": {"rel_oma_db": ">= -2", "tdecq_db": "<= 1.5", "il_db": null}}]}

``set`` targets are ``INST.param`` (``*.param`` broadcasts like a sweep) and
values are numbers or expressions over the variables. Optimize re-runs its
``inner`` step chain per Nelder-Mead evaluation, so a resonance found by the
first step is re-found for every candidate design. Evaluate re-runs the chain
at the current design (the optimum, after an Optimize step) with the
transient stretched over a full PRBS period, then scores the built-in figures
of merit with stateye using the Eye tab's settings.

The schematic is never edited: optimized instance parameters stay in the
flow's ``overrides`` and come back as ``design`` for an explicit Apply.
"""
from __future__ import annotations

import ast
import copy
import math
import re
import time

import numpy as np

import exprs

MAX_STEPS = 12
MAX_ITERS = 120
SAMPLES_PER_UI = 16    # evaluate: save points per UI of the full-period record
SETTLE_UI = 16         # figures of merit: start-up dropped before scoring
_PENALTY = 100.0       # optimize: objective units per 100 % constraint violation

# the Eye tab's stateye defaults (static/index.html tq-* fields)
STATEYE_DEFAULTS = {"rx_bw": "auto", "rx_order": 4, "taps": 5, "pre": 1,
                    "dfe": 0, "method": "mmse", "limits": True,
                    "use_pattern": True, "s_noise": 0.005, "nx": 256,
                    "ny": 1024}

# evaluate: figures of merit it can score (NRZ swaps tdecq_db for tdec_db)
FOM_KEYS = ("rel_oma", "rel_oma_db", "oma_dbm", "il_db", "er_db", "tdecq_db",
            "tdec_db", "rlm")

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NODES = exprs._ALLOWED_NODES + (ast.Compare, ast.Lt, ast.LtE, ast.Gt,
                                 ast.GtE, ast.Eq, ast.NotEq)
def _nan_safe(fn):
    """min/max that return NaN when any argument is NaN (an undefined output
    must not look like a good margin: Python's min(1, nan) is 1)."""
    def f(*a):
        vals = [float(v) for v in a]
        return float("nan") if any(math.isnan(v) for v in vals) else fn(vals)
    return f


_SCALAR_FUNCS = {
    "abs": abs, "min": _nan_safe(min), "max": _nan_safe(max), "round": round,
    "sqrt": math.sqrt, "log10": math.log10, "log": math.log, "exp": math.exp,
    "db": lambda v: 20.0 * math.log10(max(abs(v), 1e-300)),
    "dbp": lambda v: 10.0 * math.log10(max(abs(v), 1e-300)),
}


class FlowError(ValueError):
    """A flow that cannot run as configured (shown to the user verbatim)."""


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------

def _eval(expr, ctx: dict, funcs: dict | None = None):
    """Evaluate a number or a whitelisted expression (exprs' AST rules plus
    comparisons) over ``ctx``. Undefined variables are NaN in ``ctx``."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    funcs = _SCALAR_FUNCS if funcs is None else funcs
    text = str(expr).strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise FlowError(f"cannot parse {text!r}") from None
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise FlowError(f"{text!r}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in funcs):
                raise FlowError(f"{text!r}: only whitelisted functions may be "
                                f"called ({', '.join(sorted(funcs))})")
            if node.keywords:
                raise FlowError(f"{text!r}: keyword arguments are not allowed")
        if (isinstance(node, ast.Name) and id(node) not in called
                and node.id not in ctx):
            raise FlowError(f"{text!r}: unknown name {node.id!r} (variables: "
                            + ", ".join(sorted(k for k in ctx
                                               if not k.startswith("_")))
                            + ")")
    # a called name is always the function, a bare name always the variable,
    # so an output may share its function's name (rlm = rlm(thru)); integer
    # literals become floats so 9**9**9 overflows instead of hanging
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            node.func.id = "__f_" + node.func.id
        elif (isinstance(node, ast.Constant) and isinstance(node.value, int)
              and not isinstance(node.value, bool)):
            node.value = float(node.value)
    ns = {**ctx, **{"__f_" + k: v for k, v in funcs.items()}}
    try:
        return eval(compile(tree, "<flow>", "eval"),  # noqa: S307 — whitelisted
                    {"__builtins__": {}}, ns)
    except FlowError:
        raise
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise FlowError(f"{text!r}: {type(exc).__name__}: {exc}") from None


def _num(expr, ctx: dict) -> float:
    """An expression that must come out as one real number (NaN allowed)."""
    v = _eval(expr, ctx)
    try:
        return float(v)
    except (TypeError, ValueError):
        raise FlowError(f"{expr!r} is not a real number ({v!r})") from None


def _var_ctx(vars_: dict) -> dict:
    return {k: (float("nan") if v is None else float(v))
            for k, v in vars_.items()}


_CMP = {ast.Lt: ("<", lambda a, b: a < b), ast.LtE: ("<=", lambda a, b: a <= b),
        ast.Gt: (">", lambda a, b: a > b), ast.GtE: (">=", lambda a, b: a >= b),
        ast.Eq: ("==", lambda a, b: a == b),
        ast.NotEq: ("!=", lambda a, b: a != b)}


def _comparison(text: str):
    """'lhs op rhs' -> (lhs, op_node_type, rhs) source strings, else None."""
    try:
        tree = ast.parse(str(text).strip(), mode="eval").body
    except SyntaxError:
        raise FlowError(f"cannot parse check {text!r}") from None
    if (isinstance(tree, ast.Compare) and len(tree.ops) == 1
            and type(tree.ops[0]) in _CMP):
        return (ast.unparse(tree.left), type(tree.ops[0]),
                ast.unparse(tree.comparators[0]))
    return None


def check(text: str, vars_: dict) -> dict:
    """Evaluate one pass/fail check; NaN (an undefined output) fails."""
    ctx = _var_ctx(vars_)
    cmp_ = _comparison(text)
    if cmp_ is None:
        ok = bool(_eval(text, ctx))
        return {"expr": text, "ok": ok, "value": None}
    lhs_s, op, rhs_s = cmp_
    lhs, rhs = _num(lhs_s, ctx), _num(rhs_s, ctx)
    ok = bool(math.isfinite(lhs) and _CMP[op][1](lhs, rhs))
    return {"expr": text, "ok": ok,
            "value": lhs if math.isfinite(lhs) else None,
            "op": _CMP[op][0], "bound": rhs}


def penalty(constraints: list, vars_: dict) -> float:
    """Sum of relative violations (0 when feasible; inf when undefined)."""
    total = 0.0
    for text in constraints or []:
        cmp_ = _comparison(text)
        if cmp_ is None or cmp_[1] in (ast.Eq, ast.NotEq):
            raise FlowError(f"constraint {text!r}: write it as 'name >= value' "
                            "or 'name <= value'")
        lhs_s, op, rhs_s = cmp_
        ctx = _var_ctx(vars_)
        lhs, rhs = _num(lhs_s, ctx), _num(rhs_s, ctx)
        if not math.isfinite(lhs):
            return math.inf
        viol = rhs - lhs if op in (ast.Gt, ast.GtE) else lhs - rhs
        total += max(0.0, viol) / max(abs(rhs), 0.1)
    return _PENALTY * total


# ---------------------------------------------------------------------------
# schematic preparation
# ---------------------------------------------------------------------------

def _split_key(key: str) -> tuple[str, str]:
    inst, dot, param = str(key).rpartition(".")
    if not dot or not inst or not param:
        raise FlowError(f"set target {key!r}: write it as INSTANCE.param "
                        "(or *.param for every instance that has it)")
    return inst, param


def _inject_ui(sch: dict) -> None:
    """Pattern sources take their UI from the schematic's global baud rate
    (the editor injects it before a run; saved testbenches carry
    ``globals.baud`` instead)."""
    baud = float((sch.get("globals") or {}).get("baud") or 0.0)
    for inst in (sch.get("instances") or {}).values():
        if inst.get("type") in ("prbs", "qam_source") and baud > 0:
            inst.setdefault("settings", {}).setdefault("ui", 1.0 / baud)


def _assignments(flow: dict, overrides: dict, step_set: dict | None,
                 vars_: dict) -> list[tuple[str, str, float]]:
    """Resolved (inst, param, value) in apply order: the flow's design
    mapping, then optimized overrides, then the step's own set."""
    ctx = _var_ctx(vars_)
    out = []
    for source in ((flow.get("set") or {}), overrides, (step_set or {})):
        for key, expr in source.items():
            inst, param = _split_key(key)
            val = _num(expr, ctx)
            if not math.isfinite(val):
                raise FlowError(f"set {key} = {expr}: not a finite number "
                                "(an output it uses was undefined)")
            out.append((inst, param, val))
    return out


def _prepare(base: dict, flow: dict, overrides: dict, step_set: dict | None,
             vars_: dict) -> dict:
    import simulate

    sch = copy.deepcopy(base)
    _inject_ui(sch)
    for inst, param, val in _assignments(flow, overrides, step_set, vars_):
        try:
            simulate._patch_point(sch, inst, param, val)
        except simulate.NetlistError as exc:
            raise FlowError(f"set {inst}.{param}: {exc}") from None
    return sch


def _expand(sch: dict, inst: str, param: str) -> list[str]:
    """Concrete instance names a set target writes (mirrors _patch_point)."""
    import simulate

    if inst != "*":
        return [inst]
    names = []
    for name, node in (sch.get("instances") or {}).items():
        entry = simulate.CATALOG.get(node.get("type"), {})
        if entry.get("wave") or entry.get("lti") or entry.get("sky130"):
            continue
        if any(p["name"] == param for p in entry.get("params", [])):
            names.append(name)
    return names


def _pattern(sch: dict) -> dict | None:
    """The first PRBS source's settings with catalog defaults filled in."""
    import simulate

    for inst in (sch.get("instances") or {}).values():
        if inst.get("type") == "prbs":
            defs = {p["name"]: p["default"]
                    for p in simulate.CATALOG["prbs"]["params"]}
            return {**defs, **(inst.get("settings") or {})}
    return None


# ---------------------------------------------------------------------------
# figures of merit (stateye through the Eye tab's scoring path)
# ---------------------------------------------------------------------------

def _rlm(levels) -> float:
    """IEEE 802.3 ratio of level mismatch from four levels."""
    p = sorted(levels)
    vm = (p[0] + p[3]) / 2
    if p[3] - p[0] <= 0:                 # collapsed eye: no level ladder
        return float("nan")
    es1 = (p[1] - vm) / (p[0] - vm)
    es2 = (p[2] - vm) / (p[3] - vm)
    return float(min(3 * es1, 3 * es2, 2 - 3 * es1, 2 - 3 * es2))


def score_eye(t, values, pattern: dict, cfg: dict | None) -> dict:
    """stateye on one optical trace [mW]: OMA, ER, TDECQ (PAM4) / TDEC (NRZ),
    RLM and the top level, measured the way the Eye tab's Measure button
    does (same reference receiver, equalizer and noise settings)."""
    import eyemeasure

    if not pattern or not pattern.get("ui"):
        raise FlowError("figures of merit need a PRBS source (for the UI and "
                        "the transmitted pattern)")
    ui = float(pattern["ui"])
    pam4 = str(pattern.get("mode")) == "pam4"
    c = {**STATEYE_DEFAULTS, **(cfg or {})}
    c.setdefault("ser" if pam4 else "ber", 4.8e-4 if pam4 else 1e-12)
    m = eyemeasure.measure({"t": list(t), "values": list(values),
                            "unit": "mW", "ui": ui, "levels": 4 if pam4 else 2,
                            "skip_s": SETTLE_UI * ui, "pattern": pattern,
                            "cfg": c})
    if not m.get("ok"):
        raise FlowError(f"stateye: {m.get('error')}")
    nan = float("nan")
    oma = m.get("oma")
    er = m.get("er_db")
    out = {"format": m["format"], "family": m.get("family"),
           "oma_mw": nan if oma is None else float(oma),
           "er_db": nan if er is None else float(er),
           "log": m.get("log") or []}
    if pam4:
        out["tdecq_db"] = nan if m.get("tdecq_db") is None \
            else float(m["tdecq_db"])
        lv = m.get("levels_raw") or []
        out["rlm"] = (_rlm(lv) if len(lv) == 4 and None not in lv else nan)
    else:
        out["tdec_db"] = nan if m.get("tdec_db") is None \
            else float(m["tdec_db"])
    # top level from the same OMA/ER pair: P1 = OMA * r / (r - 1)
    r = 10 ** (out["er_db"] / 10) if math.isfinite(out["er_db"]) else nan
    out["p_top_mw"] = out["oma_mw"] * r / (r - 1) if r > 1 else nan
    return out


def fom_values(score: dict, p_in_mw: float) -> dict:
    """Figures of merit from a stateye score and the modulator's input power:
    relOMA = OMA / P_in, IL = P_in / P_top (the highest level)."""
    nan = float("nan")
    oma, top = score["oma_mw"], score["p_top_mw"]
    p_in = float(p_in_mw) if p_in_mw and p_in_mw > 0 else nan
    rel = oma / p_in if math.isfinite(oma) and oma > 0 else nan
    out = {"rel_oma": rel,
           "rel_oma_db": 10 * math.log10(rel) if rel > 0 else nan,
           "oma_dbm": 10 * math.log10(oma) if oma > 0 else nan,
           "il_db": 10 * math.log10(p_in / top) if top > 0 else nan,
           "er_db": score["er_db"]}
    for k in ("tdecq_db", "tdec_db", "rlm"):
        if k in score:
            out[k] = score[k]
    return {k: (float(v) if math.isfinite(v) else None) for k, v in out.items()}


# ---------------------------------------------------------------------------
# step outputs
# ---------------------------------------------------------------------------

def _result_ctx(res: dict, sch: dict, vars_: dict, flow: dict):
    """Names an output expression sees: variables, x/t, probe traces, fx
    scalars; functions: exprs' vocabulary + xat_min/xat_max/cross and the
    stateye figures of merit (rlm, er_db, tdecq, rel_oma_db, il_db, ...)."""
    ctx = _var_ctx(vars_)
    x = np.asarray(res.get("x") or [0.0], float)
    ctx["x"] = ctx["t"] = x
    for tr in res.get("traces") or []:
        name = tr.get("probe") or tr["name"]
        if _IDENT.fullmatch(name) and name not in ctx:
            ctx[name] = np.asarray(tr["values"], float)
    for row in res.get("rows") or []:          # DC operating point: scalars
        if _IDENT.fullmatch(row["name"]):
            ctx.setdefault(row["name"], float(row["value"]))
    for s in res.get("scalars") or []:
        ctx.setdefault(s["name"], float(s["value"]))

    funcs = exprs._make_funcs(x)

    def _peak(y, sign):
        y = sign * np.asarray(y, float)
        i = int(np.nanargmin(y))
        if 0 < i < len(y) - 1:      # parabolic refinement between samples
            y0, y1, y2 = y[i - 1], y[i], y[i + 1]
            den = y0 - 2 * y1 + y2
            if den > 0:
                d = 0.5 * (y0 - y2) / den
                return float(x[i] + d * (x[i + 1] - x[i - 1]) / 2)
        return float(x[i])

    def cross(y, level):
        y = np.asarray(y, float) - float(level)
        idx = np.nonzero(np.diff(np.signbit(y)))[0]
        if not idx.size:
            return float("nan")
        i = int(idx[0])
        return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))

    pattern = _pattern(sch)
    cache: dict[int, tuple] = {}   # id -> (array kept alive, score)

    def score(y):
        key = id(y)
        if key not in cache:
            if pattern is None or res.get("kind") != "transient":
                raise FlowError("figures of merit need a transient with a "
                                "PRBS source")
            cache[key] = (y, score_eye(x, np.asarray(y, float), pattern,
                                       flow.get("stateye")))
        return cache[key][1]

    def mean(v):
        """Input power, averaged after the same settling the scorer drops."""
        v = np.asarray(v, float)
        if v.ndim == 0:
            return float(v)
        keep = x >= SETTLE_UI * float(pattern["ui"]) if pattern else slice(None)
        return float(np.mean(v[keep]))

    def fom(name):
        def f(y, pin=None):
            vals = fom_values(score(y), mean(pin) if pin is not None else 0.0)
            v = vals.get(name)
            return float("nan") if v is None else v
        return f

    funcs.update({
        "xat_min": lambda y: _peak(y, 1.0), "xat_max": lambda y: _peak(y, -1.0),
        "cross": cross, "ptp": lambda y: float(np.ptp(y)),
        "rlm": fom("rlm"), "er_db": fom("er_db"), "tdecq": fom("tdecq_db"),
        "tdec": fom("tdec_db"), "oma_dbm": fom("oma_dbm"),
        "oma_mw": lambda y: score(y)["oma_mw"],
        "rel_oma": fom("rel_oma"), "rel_oma_db": fom("rel_oma_db"),
        "il_db": fom("il_db"),
    })
    return ctx, funcs


def _outputs(step: dict, res: dict, sch: dict, vars_: dict,
             flow: dict) -> dict:
    outs = step.get("outputs") or {}
    if not outs:
        return {}
    ctx, funcs = _result_ctx(res, sch, vars_, flow)
    vals: dict = {}
    for name, expr in outs.items():
        if not _IDENT.fullmatch(str(name)):
            raise FlowError(f"output name {name!r} is not an identifier")
        try:
            v = _eval(expr, ctx, funcs)
        except FlowError:
            raise
        except Exception as exc:
            raise FlowError(f"output {name} = {expr}: "
                            f"{type(exc).__name__}: {exc}") from None
        arr = np.asarray(v, float)
        if arr.ndim != 0:
            raise FlowError(f"output {name} = {expr} is an array; reduce it "
                            "to one number (min, max, mean, xat_min, ...)")
        f = float(arr)
        vals[name] = f if math.isfinite(f) else None
        ctx[name] = f
    return vals


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class _Runner:
    def __init__(self, base: dict, flow: dict, log: list):
        self.base, self.flow, self.log = base, flow, log
        self.steps = list(flow.get("steps") or [])
        self.by_id = {s["id"]: s for s in self.steps}
        self.label = ""
        self.design_set: dict | None = None   # set of the step the design is read at
        self.design_tr = False

    # --- one simulate.run --------------------------------------------------
    def simulate(self, sch: dict, analysis: dict) -> dict:
        import simulate
        from progress import PROGRESS, RunCancelled

        if PROGRESS.cancelled:
            raise RunCancelled("run stopped by user")
        PROGRESS.restart()
        res = simulate.run({"schematic": sch, "analysis": analysis})
        if res.get("cancelled"):
            raise RunCancelled("run stopped by user")
        if not res.get("ok"):
            raise FlowError(res.get("error") or "run failed")
        return res

    # --- analysis steps (alone or as an inner chain) -------------------------
    def chain(self, ids: list, vars_: dict, overrides: dict,
              record: str | None = None) -> tuple[dict, dict, dict]:
        """Run analysis steps in order, threading their outputs through the
        variables. Returns (vars, last result, last prepared schematic)."""
        vars_ = dict(vars_)
        res = sch = None
        last_tr = max((i for i, sid in enumerate(ids)
                       if (self.by_id[sid].get("analysis") or {}).get("mode")
                       == "transient"), default=None)
        for i, sid in enumerate(ids):
            step = self.by_id[sid]
            sch = _prepare(self.base, self.flow, overrides, step.get("set"),
                           vars_)
            analysis = copy.deepcopy(step.get("analysis") or {})
            if record == "full_period" and i == last_tr:
                analysis = self.full_period(analysis, sch)
            try:
                res = self.simulate(sch, analysis)
                vars_.update(_outputs(step, res, sch, vars_, self.flow))
            except FlowError as exc:
                raise FlowError(f"{step.get('title') or sid}: {exc}") from None
            # the design is what the last transient (else last step) ran at
            is_tr = analysis.get("mode") == "transient"
            if is_tr or not self.design_tr:
                self.design_set, self.design_tr = step.get("set") or {}, is_tr
        return vars_, res, sch

    def full_period(self, analysis: dict, sch: dict) -> dict:
        """Stretch a transient over one whole PRBS period (plus settling), so
        OMA_outer / TDECQ see the full pattern, at SAMPLES_PER_UI."""
        import simulate

        pat = _pattern(sch)
        if not pat or not pat.get("ui"):
            raise FlowError("record = full PRBS period needs a PRBS source")
        ui = float(pat["ui"])
        period = (1 << int(pat.get("order", 7))) - 1
        nsym = period + 2 * SETTLE_UI + 16
        pts = nsym * SAMPLES_PER_UI
        if pts > simulate.MAX_FLOW_POINTS:
            raise FlowError(
                f"a full PRBS-{pat.get('order')} period at {SAMPLES_PER_UI} "
                f"samples/UI needs {pts} points (limit "
                f"{simulate.MAX_FLOW_POINTS}); use PRBS-13 or shorter, or "
                "record = as the step is set up")
        return {**analysis, "t_stop": nsym * ui, "points": pts,
                "max_points": simulate.MAX_FLOW_POINTS}

    # --- step kinds ----------------------------------------------------------
    def run_analysis(self, step, vars_, overrides):
        vars2, res, _ = self.chain([step["id"]], vars_, overrides)
        outs = {k: vars2.get(k) for k in (step.get("outputs") or {})}
        return vars2, overrides, {"result": res, "outputs": outs}

    def inner_ids(self, step) -> list:
        inner = step.get("inner")
        ids = [inner] if isinstance(inner, str) else list(inner or [])
        if not ids:
            raise FlowError(f"{step.get('title') or step['id']}: pick the "
                            "step(s) it re-runs (inner)")
        for sid in ids:
            if sid not in self.by_id:
                raise FlowError(f"{step['id']}: inner step {sid!r} not found")
            if self.by_id[sid].get("kind", "analysis") != "analysis":
                raise FlowError(f"{step['id']}: inner step {sid!r} must be an "
                                "analysis step")
        return ids

    def run_optimize(self, step, vars_, overrides):
        from optimize import nelder_mead
        from progress import PROGRESS

        ids = self.inner_ids(step)
        params = step.get("params") or []
        if not params:
            raise FlowError("optimize: add at least one parameter to vary")
        if len(params) > 8:
            raise FlowError("optimize: at most 8 parameters")
        objective = str(step.get("objective") or "").strip()
        if not objective:
            raise FlowError("optimize: no objective (an output name or an "
                            "expression over the variables)")
        maximize = bool(step.get("maximize", True))
        constraints = list(step.get("constraints") or [])
        iters = max(4, min(int(step.get("iters", 30)), MAX_ITERS))
        keys, lo, hi, x0 = [], [], [], []
        start_sch = _prepare(self.base, self.flow, overrides, None, vars_)
        for p in params:
            a, b = float(p["min"]), float(p["max"])
            if not b > a:
                raise FlowError(f"optimize: {p} needs min < max")
            if p.get("var"):
                key = ("var", p["var"])
                cur = vars_.get(p["var"])
            else:
                key = ("inst", f"{p['inst']}.{p['param']}")
                node = (start_sch.get("instances") or {}).get(p["inst"])
                if node is None:
                    raise FlowError(f"optimize: no instance {p['inst']!r}")
                cur = (node.get("settings") or {}).get(p["param"])
            keys.append(key)
            lo.append(a)
            hi.append(b)
            x0.append(0.5 * (a + b) if cur is None else float(cur))
        lo, hi = np.asarray(lo), np.asarray(hi)

        history: list[dict] = []
        best: dict = {"J": math.inf}
        first_err: list[str] = []

        def f(x):
            x = np.clip(x, lo, hi)
            n = len(history) + 1
            PROGRESS.set_label(f"{self.label} · eval {n}/{iters}")
            v2, o2 = dict(vars_), dict(overrides)
            for (kind, name), xi in zip(keys, x):
                if kind == "var":
                    v2[name] = float(xi)
                else:
                    o2[name] = float(xi)
            entry = {"x": [float(v) for v in x]}
            try:
                v2, res, _ = self.chain(ids, v2, o2)
                obj = _num(objective, _var_ctx(v2))
                pen = penalty(constraints, v2)
            except FlowError as exc:
                if not first_err:
                    first_err.append(str(exc))
                entry.update(obj=None, J=None, feasible=False, error=str(exc))
                history.append(entry)
                self.log.append(f"  eval {n}: failed — {exc}")
                return math.inf
            J = ((-obj if maximize else obj) + pen
                 if math.isfinite(obj) else math.inf)
            entry.update(obj=obj if math.isfinite(obj) else None,
                         J=J if math.isfinite(J) else None,
                         feasible=bool(pen == 0 and math.isfinite(obj)))
            history.append(entry)
            self.log.append(
                f"  eval {n}: " + ", ".join(
                    f"{nm}={xi:.6g}" for (_, nm), xi in zip(keys, x))
                + f" -> {objective} {obj:.4g}"
                + ("" if pen == 0 else f", penalty {pen:.3g}"))
            if J < best["J"]:
                best.update(J=J, x=np.array(x), vars=v2, overrides=o2, res=res)
            return J

        nelder_mead(f, np.asarray(x0, float), lo, hi, iters)
        if not math.isfinite(best["J"]):
            raise FlowError("optimize: no evaluation succeeded"
                            + (f" — first error: {first_err[0]}"
                               if first_err else ""))
        v2, o2 = best["vars"], best["overrides"]
        bx = {name: float(xi) for (_, name), xi in zip(keys, best["x"])}
        # informational: the optimizer's own (margined) constraints show in
        # the step's checks but the flow's verdict is the specs' business
        checks = [{**check(c, v2), "info": True} for c in constraints]
        feasible = all(c["ok"] for c in checks)
        self.log.append(
            f"optimize: best {objective} = {v2.get(objective, 'n/a')} after "
            f"{len(history)} evaluations"
            + ("" if feasible else " — constraints NOT met"))
        return v2, o2, {"result": best["res"], "outputs": bx,
                        "history": history, "objective": objective,
                        "maximize": maximize, "params": [
                            {"name": name, "kind": kind, "min": float(a),
                             "max": float(b), "start": float(s),
                             "value": bx[name]}
                            for (kind, name), a, b, s in zip(keys, lo, hi, x0)],
                        "checks": checks}

    def run_evaluate(self, step, vars_, overrides):
        ids = self.inner_ids(step)
        probes = step.get("probes") or {}
        out_p, in_p = probes.get("out"), probes.get("in")
        if not out_p:
            raise FlowError("evaluate: pick the modulator output probe")
        spec = dict(step.get("fom") or {})
        unknown = [k for k in spec if k not in FOM_KEYS]
        if unknown:
            raise FlowError(f"evaluate: unknown figure(s) of merit {unknown}; "
                            f"choose from {', '.join(FOM_KEYS)}")
        for k, s_ in spec.items():
            if s_ and (_comparison(f"{k} {s_}") is None
                       or _comparison(f"{k} {s_}")[1] in (ast.Eq, ast.NotEq)):
                raise FlowError(f"evaluate: spec {k} {s_!r} — write it as "
                                "'>= value' or '<= value'")
        before = {k: vars_.get(k) for k in spec}
        record = step.get("record", "full_period")
        v2, res, sch = self.chain(ids, vars_, overrides,
                                  record if record == "full_period" else None)
        pattern = _pattern(sch)
        if not pattern or not pattern.get("ui"):
            raise FlowError("evaluate needs a PRBS source")
        traces = {(tr.get("probe") or tr["name"]): tr
                  for tr in reversed(res.get("traces") or [])}
        if out_p not in traces:
            raise FlowError(f"evaluate: no probe {out_p!r} in the result")
        if traces[out_p].get("unit") != "mW":
            raise FlowError(f"evaluate: {out_p!r} is not an optical probe")
        p_in = None
        if in_p:
            if in_p not in traces:
                raise FlowError(f"evaluate: no probe {in_p!r} in the result")
            t = np.asarray(res["x"], float)
            keep = t >= SETTLE_UI * float(pattern["ui"])
            p_in = float(np.mean(np.asarray(traces[in_p]["values"])[keep]))
        sc = score_eye(res["x"], traces[out_p]["values"], pattern,
                       self.flow.get("stateye"))
        for line in sc["log"]:
            self.log.append(f"  stateye: {line}")
        fom = fom_values(sc, p_in or 0.0)
        if p_in is None:
            for k in ("rel_oma", "rel_oma_db", "il_db"):
                fom[k] = None
        v2.update(fom)
        checks = [check(f"{k} {s}", v2) for k, s in spec.items() if s]
        return v2, overrides, {
            "result": res, "outputs": {k: fom.get(k) for k in spec},
            "fom": fom, "spec": spec, "before": before,
            "family": sc.get("family"), "format": sc.get("format"),
            "p_in_mw": p_in, "checks": checks}

    # --- the whole flow --------------------------------------------------------
    def run(self, start: str | None, only: str | None,
            prior: dict) -> dict:
        from progress import PROGRESS, RunCancelled

        ids = [s.get("id") for s in self.steps]
        if not self.steps:
            raise FlowError("the flow has no steps")
        if len(self.steps) > MAX_STEPS:
            raise FlowError(f"a flow has at most {MAX_STEPS} steps")
        if len(set(ids)) != len(ids) or not all(
                isinstance(i, str) and _IDENT.fullmatch(i) for i in ids):
            raise FlowError("step ids must be unique identifiers")
        for sid in (start, only):
            if sid and sid not in self.by_id:
                raise FlowError(f"no step {sid!r}")
        vars_ = {k: float(v) for k, v in (self.flow.get("vars") or {}).items()}
        overrides: dict = {}
        origin = {k: "vars" for k in vars_}
        if start or only:
            vars_.update(prior.get("vars") or {})
            overrides.update(prior.get("overrides") or {})
            origin.update(prior.get("origin") or {})
        first = ids.index(only or start) if (start or only) else 0
        out, failed = [], None
        for k, step in enumerate(self.steps):
            kind = step.get("kind", "analysis")
            rec = {"id": step["id"], "title": step.get("title") or step["id"],
                   "kind": kind}
            if k < first or (only and step["id"] != only):
                rec["status"] = "skipped"
                out.append(rec)
                continue
            if failed:
                rec["status"] = "not run"
                out.append(rec)
                continue
            self.label = (f"step {k + 1}/{len(self.steps)} · "
                          f"{rec['title']}")
            PROGRESS.set_label(self.label)
            t0 = time.perf_counter()
            self.log.append(f"[{k + 1}] {rec['title']} ({kind})")
            try:
                if kind == "analysis":
                    vars2, ovr2, info = self.run_analysis(step, vars_,
                                                          overrides)
                elif kind == "optimize":
                    vars2, ovr2, info = self.run_optimize(step, vars_,
                                                          overrides)
                elif kind == "evaluate":
                    vars2, ovr2, info = self.run_evaluate(step, vars_,
                                                          overrides)
                else:
                    raise FlowError(f"unknown step kind {kind!r}")
                checks = info.pop("checks", None)
                if checks is None:
                    checks = [check(c, vars2) for c in step.get("checks") or []]
            except RunCancelled:
                raise
            except Exception as exc:   # FlowError, or a bug: fail this step
                msg = (str(exc) if isinstance(exc, FlowError)
                       else f"{type(exc).__name__}: {exc}")
                rec.update(status="failed", error=msg,
                           time_s=time.perf_counter() - t0)
                self.log.append(f"  FAILED: {exc}")
                out.append(rec)
                failed = step["id"]
                continue
            for name in vars2:
                if name not in vars_ or vars2[name] != vars_.get(name):
                    origin[name] = step["id"]
            for name in ovr2:
                if ovr2[name] != overrides.get(name):
                    origin[name] = step["id"]
            vars_, overrides = vars2, ovr2
            rec.update(status="ok", checks=checks,
                       time_s=time.perf_counter() - t0, **info)
            self.log.append(
                f"  done in {rec['time_s']:.1f}s"
                + "".join(f"; {c['expr']}: {'pass' if c['ok'] else 'FAIL'}"
                          + (" (optimizer constraint)" if c.get("info") else "")
                          for c in checks))
            out.append(rec)
        ran = [r for r in out if r.get("status") == "ok"]
        passed = (failed is None and bool(ran)
                  and all(c["ok"] for r in ran for c in r.get("checks") or []
                          if not c.get("info")))
        return {"steps": out, "vars": vars_, "overrides": overrides,
                "origin": origin, "passed": passed, "failed": failed,
                "design": self.design(vars_, overrides)}

    def design(self, vars_: dict, overrides: dict) -> list[dict]:
        """Concrete instance settings of the final design (flow set +
        optimized overrides + the last analysis step's set), for Apply."""
        try:
            assigns = _assignments(self.flow, overrides, self.design_set,
                                   vars_)
        except FlowError:
            return []
        resolved: dict = {}
        for inst, param, val in assigns:
            for name in _expand(self.base, inst, param):
                resolved[(name, param)] = val
        return [{"inst": i, "param": p, "value": v}
                for (i, p), v in resolved.items()]


def run_flow(payload: dict) -> dict:
    """Entry point from simulate.run (analysis.mode == "flow")."""
    import simulate

    sch = payload.get("schematic") or {}
    analysis = payload.get("analysis") or {}
    flow = analysis.get("flow") or {}
    log: list[str] = [f"flow: {flow.get('name') or 'untitled'}"]
    if len(simulate.sch_corners(sch)) > 1:
        return {"ok": False, "error": "flows run at one process corner — pick "
                "a single corner in the Corner menu"}
    try:
        body = _Runner(sch, flow, log).run(analysis.get("start"),
                                           analysis.get("only"),
                                           analysis.get("prior") or {})
    except FlowError as exc:
        return {"ok": False, "error": str(exc), "log": log}
    log.append("flow: " + ("all checks pass" if body["passed"] else
                           "a step failed" if body["failed"] else
                           "some checks FAIL"))
    return {"ok": True, "kind": "flow", "flow": body, "log": log}
