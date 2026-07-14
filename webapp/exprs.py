"""Derived-trace expressions: post-process analysis results server-side.

The UI sends ``analysis.expressions`` as a text blob, one definition per
line::

    icalc = (vout - 1.8) / 500
    gain_db [dB] = db(vout / vin)
    spectrum = spec(vout)

Each line is ``name = expr`` with an optional ``[unit]`` tag. Expressions
see every visible probe trace by name plus ``t``/``x`` (the sweep axis) and
a whitelisted numpy vocabulary. Results the same length as the x axis are
appended as plot traces; ``spec()``/``psd()`` results become an extra
log-frequency plot; scalars go to the run log.

Evaluation is an ast-whitelist interpreter — no attribute access, no
subscripts, no names outside the context, ``__builtins__`` emptied.
"""
from __future__ import annotations

import ast
import re as _re

import numpy as np

_LINE = _re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[([^\]]*)\])?\s*=\s*(.+?)\s*$")

_PALETTE = ["#e6c86e", "#8fd18f", "#f08fb0", "#9fa8ff", "#72d5c8",
            "#e69e6e", "#c98fd1"]

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.FloorDiv, ast.Load,
)


class _Spectrum:
    def __init__(self, f, v, unit=""):
        self.f, self.v, self.unit = f, v, unit


def _make_funcs(x: np.ndarray) -> dict:
    """Whitelisted functions; the closures capture the sweep/time axis."""
    def _uniform(y):
        xu = np.linspace(x[0], x[-1], len(x))
        return xu, np.interp(xu, x, y)

    def spec(y):
        """One-sided amplitude spectrum |Y(f)| (same unit as y)."""
        xu, yu = _uniform(np.asarray(y, float))
        n = len(yu)
        f = np.fft.rfftfreq(n, (xu[-1] - xu[0]) / (n - 1))
        m = np.abs(np.fft.rfft(yu - yu.mean())) * 2.0 / n
        return _Spectrum(f[1:], m[1:])

    def psd(y):
        """One-sided periodogram [unit^2/Hz]."""
        xu, yu = _uniform(np.asarray(y, float))
        n = len(yu)
        dt = (xu[-1] - xu[0]) / (n - 1)
        f = np.fft.rfftfreq(n, dt)
        p = (np.abs(np.fft.rfft(yu - yu.mean())) ** 2) * 2.0 * dt / n
        return _Spectrum(f[1:], p[1:])

    return {
        "db": lambda y: 20.0 * np.log10(np.maximum(np.abs(y), 1e-300)),
        "dbp": lambda y: 10.0 * np.log10(np.maximum(np.abs(y), 1e-300)),
        "abs": np.abs, "mag": np.abs, "abs2": lambda y: np.abs(y) ** 2,
        "sqrt": np.sqrt, "log10": np.log10, "log": np.log, "exp": np.exp,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "min": np.min, "max": np.max, "mean": np.mean, "std": np.std,
        "rms": lambda y: float(np.sqrt(np.mean(np.square(y)))),
        "pk2pk": lambda y: float(np.max(y) - np.min(y)),
        "deriv": lambda y: np.gradient(np.asarray(y, float), x),
        "integ": lambda y: np.concatenate(
            [[0.0], np.cumsum(0.5 * (np.asarray(y)[1:] + np.asarray(y)[:-1])
                              * np.diff(x))]),
        "clip": np.clip,
        "spec": spec, "psd": psd,
    }


def _eval(expr: str, ctx: dict, funcs: dict):
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"'{type(node).__name__}' is not allowed")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in funcs):
                raise ValueError("only whitelisted functions may be called")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
        if isinstance(node, ast.Name) and node.id not in ctx and node.id not in funcs:
            raise ValueError(
                f"unknown name '{node.id}' — traces here: "
                + ", ".join(sorted(k for k in ctx if not k.startswith("_"))))
    return eval(compile(tree, "<expr>", "eval"),  # noqa: S307 — whitelisted
                {"__builtins__": {}}, {**funcs, **ctx})


def apply(result: dict, expr_text: str, log: list) -> None:
    """Evaluate expressions against `result` in place."""
    if not expr_text or not expr_text.strip():
        return
    if "x" not in result or "traces" not in result:
        log.append("expressions: skipped (no plottable axis in this analysis)")
        return
    x = np.asarray(result["x"], float)
    ctx: dict = {"x": x, "t": x}
    for tr in result["traces"]:
        name = tr["name"]
        if _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            ctx[name] = np.asarray(tr["values"], float)
    funcs = _make_funcs(x)
    spectra: list[tuple[str, str, _Spectrum]] = []
    idx = 0
    for line in expr_text.splitlines():
        line = line.split("#", 1)[0]
        if not line.strip():
            continue
        m = _LINE.match(line)
        if not m:
            log.append(f"expressions: cannot parse line {line.strip()!r} "
                       "(want: name [unit] = expr)")
            continue
        name, unit, expr = m.group(1), m.group(2) or "", m.group(3)
        try:
            val = _eval(expr, ctx, funcs)
        except Exception as e:
            log.append(f"expressions: {name}: {e}")
            continue
        if isinstance(val, _Spectrum):
            spectra.append((name, unit, val))
            continue
        arr = np.asarray(val)
        if arr.ndim == 0:
            log.append(f"expr {name} = {float(arr):.6g} {unit}")
            result.setdefault("scalars", []).append(
                {"name": name, "value": float(arr), "unit": unit})
            continue
        if arr.shape != x.shape:
            log.append(f"expressions: {name}: length {arr.shape} does not "
                       f"match the x axis {x.shape}")
            continue
        ctx[name] = arr    # later lines can reference earlier results
        result["traces"].append({
            "name": name, "domain": "derived", "unit": unit,
            "values": arr.tolist(), "color": _PALETTE[idx % len(_PALETTE)],
        })
        idx += 1
    if spectra:
        traces = []
        for name, unit, sp in spectra:
            traces.append({"name": name, "domain": "derived",
                           "unit": unit or sp.unit, "values": sp.v.tolist(),
                           "color": _PALETTE[idx % len(_PALETTE)]})
            idx += 1
        f = spectra[0][2].f
        result.setdefault("extra_plots", []).append({
            "x": f.tolist(), "xlabel": "frequency [Hz]", "xlog": True,
            "traces": traces})
