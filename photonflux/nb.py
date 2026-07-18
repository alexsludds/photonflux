"""Notebook client for the photonflux web UI — the live schematic bridge.

The browser owns the schematic; this module connects to the web server's
live mirror of it (``/api/schematic``, see ``webapp/session.py``) so a
Jupyter kernel can read what is on the canvas *right now*, edit it, run any
analysis on it, and react to canvas edits as they happen::

    from photonflux.nb import Session

    s = Session()                      # http://127.0.0.1:8642
    sch = s.pull()                     # live mirror of the active tab
    res = s.dcsweep("*", "wavelength_nm", 1304, 1316, points=2001)
    res.plot()                         # matplotlib, all probes

    s["CPL.coupling"] = 0.088          # canvas updates live (and undoably)

    for sch in s.watch():              # yields on every canvas edit
        res = s.dcsweep("*", "wavelength_nm", 1304, 1316)
        ...                            # e.g. redraw a live plot

Everything speaks plain HTTP to the stdlib server — importing this module
needs only numpy, not the JAX stack, so it works from any kernel. Analyses
run server-side through the same ``/api/run`` the Run button uses (and the
same compile caches), serialized behind the browser's runs; progress is
polled from ``/api/progress`` and Ctrl-C sends ``/api/cancel``.

``Builder`` constructs schematics programmatically (an N-channel WDM link is
a loop, not forty clicks) and ``Session.push`` drops the result onto the
canvas as an ordinary undoable edit for hand-refinement.

Time/frequency/value arguments accept SPICE-style SI suffix strings
anywhere a number is expected: ``"10n"``, ``"1.5u"``, ``"50G"``, ``"3meg"``.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_URL = os.environ.get("PHOTONFLUX_URL", "http://127.0.0.1:8642")

__all__ = ["Session", "Schematic", "Result", "Builder",
           "SessionError", "RunError", "si"]


class SessionError(RuntimeError):
    """The server could not be reached or refused a session request."""


class RunError(RuntimeError):
    """A simulation failed server-side; ``.log`` carries the server log."""

    def __init__(self, message: str, log: list | None = None):
        super().__init__(message)
        self.log = log or []


_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
       "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def si(v) -> float:
    """Parse a number or SPICE-style SI-suffixed string ("4n", "3meg", "50G").

    Suffixes are case-insensitive except the classic SPICE collision:
    ``m`` is milli and ``meg`` is mega (so ``"1M"`` is 1e-3, like SPICE)."""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"\s*([-+]?[\d._]*\.?[\d_]+(?:[eE][-+]?\d+)?)\s*"
                     r"([a-zA-Z]*)\s*", str(v))
    if not m:
        raise ValueError(f"cannot parse number: {v!r}")
    base = float(m.group(1))
    suf = m.group(2).lower()
    if not suf:
        return base
    if suf not in _SI:
        raise ValueError(f"unknown SI suffix {m.group(2)!r} in {v!r}")
    return base * _SI[suf]


def _maybe_si(v):
    """`si()` for SI-suffixed strings ("2m", "50G"), pass-through for
    everything else — genuine string settings (a PRBS source's ``mode``,
    a PWL source's ``data``) must survive untouched."""
    if isinstance(v, str):
        try:
            return si(v)
        except ValueError:
            return v
    return v


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
class Schematic:
    """A schematic document — the mirror's ``{title, schematic, analysis,
    selection}`` shape, which is also the web UI's Save format.

    Mutating it edits the local copy only; ``Session.push`` sends it back to
    the canvas. ``sch["INST"]`` is an instance dict, ``sch["INST.param"]``
    reads/writes one setting."""

    def __init__(self, doc: dict, rev: int = 0, source: str = ""):
        self.doc = doc
        self.rev = rev
        self.source = source

    @classmethod
    def load(cls, path) -> "Schematic":
        """Load a web-UI Save file (or bare schematic JSON) from disk."""
        doc = json.loads(Path(path).read_text())
        if "schematic" not in doc:            # bare {instances, wires, ...}
            doc = {"title": Path(path).stem, "schematic": doc}
        return cls(doc)

    # --- the parts ---------------------------------------------------------
    @property
    def title(self) -> str:
        return self.doc.get("title") or ""

    @property
    def schematic(self) -> dict:
        return self.doc.get("schematic") or {}

    @property
    def analysis(self) -> dict | None:
        return self.doc.get("analysis")

    @property
    def selection(self) -> dict | None:
        return self.doc.get("selection")

    @property
    def instances(self) -> dict:
        return self.schematic.get("instances") or {}

    @property
    def wires(self) -> list:
        return self.schematic.get("wires") or []

    @property
    def probes(self) -> list:
        return self.schematic.get("probes") or []

    # --- item access: "INST" -> instance dict, "INST.param" -> setting -----
    def _split(self, key: str):
        inst, _, param = str(key).partition(".")
        if inst not in self.instances:
            raise KeyError(f"no instance {inst!r} on the schematic "
                           f"(have: {', '.join(sorted(self.instances)) or '—'})")
        return inst, param or None

    def __getitem__(self, key: str):
        inst, param = self._split(key)
        if param is None:
            return self.instances[inst]
        try:
            return self.instances[inst].get("settings", {})[param]
        except KeyError:
            raise KeyError(f"{inst} has no setting {param!r} (set: "
                           f"{', '.join(self.instances[inst].get('settings', {})) or '—'};"
                           " catalog defaults are applied server-side)") from None

    def __setitem__(self, key: str, value) -> None:
        inst, param = self._split(key)
        if param is None:
            raise KeyError("assign to 'INST.param', not a whole instance")
        self.instances[inst].setdefault("settings", {})[param] = _maybe_si(value)

    def __contains__(self, key: str) -> bool:
        try:
            self.__getitem__(key)
            return True
        except KeyError:
            return False

    def save(self, path) -> Path:
        """Write the document to disk in the web UI's Save format (loadable
        via the Load button, ``Schematic.load``, or ``Session.push``)."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.write_text(json.dumps(self.doc, indent=2))
        return p

    def __repr__(self) -> str:
        s = self.schematic
        mode = (self.analysis or {}).get("mode")
        return (f"<Schematic {self.title!r}: {len(s.get('instances') or {})} "
                f"instances, {len(s.get('wires') or [])} wires, "
                f"{len(s.get('probes') or [])} probes"
                + (f", analysis={mode}" if mode else "")
                + (f", rev={self.rev}" if self.rev else "") + ">")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
class Result:
    """One ``/api/run`` result. Traces are numpy arrays keyed by name::

        res.names                  # every trace name
        res["vout"]                # y-values (V or mW per the probe domain)
        res.x, res.xlabel          # shared x-axis
        res.plot()                 # quick matplotlib look
        res.log                    # the server-side run log

    Sweep families (a stepped second parameter) appear as one trace per step
    value; ``res.family("thru")`` gathers them. DC operating points expose
    ``.table`` instead of traces."""

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def ok(self) -> bool:
        return bool(self.raw.get("ok"))

    @property
    def kind(self) -> str:
        return self.raw.get("kind") or ""

    @property
    def log(self) -> list:
        return self.raw.get("log") or []

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self.raw.get("x") or [], float)

    @property
    def xlabel(self) -> str:
        return self.raw.get("xlabel") or ""

    @property
    def traces(self) -> list[dict]:
        out = list(self.raw.get("traces") or [])
        for ep in self.raw.get("extra_plots") or []:
            out.extend(ep.get("traces") or [])
        return out

    @property
    def names(self) -> list[str]:
        return [t.get("name", "") for t in self.traces]

    @property
    def table(self) -> list[dict]:
        """DC operating point rows: ``[{name, value, unit, domain}, ...]``."""
        return self.raw.get("rows") or []

    def trace(self, name: str) -> dict:
        """The full trace dict (name/domain/unit/values) for one name."""
        for t in self.traces:
            if t.get("name") == name:
                return t
        starts = [t for t in self.traces if str(t.get("name", "")).startswith(name)]
        if len(starts) == 1:
            return starts[0]
        if starts:
            raise KeyError(f"{name!r} is ambiguous: "
                           f"{[t['name'] for t in starts]} — or use .family()")
        raise KeyError(f"no trace {name!r} (have: {self.names or '—'})")

    def __getitem__(self, name: str) -> np.ndarray:
        if self.kind == "op":
            for row in self.table:
                if row.get("name") == name:
                    return row.get("value")
            raise KeyError(f"no operating-point row {name!r} "
                           f"(have: {[r.get('name') for r in self.table]})")
        return np.asarray(self.trace(name)["values"], float)

    def family(self, prefix: str) -> dict[str, np.ndarray]:
        """All traces whose name starts with ``prefix`` (a stepped sweep's
        curve family), keyed by full trace name."""
        out = {t["name"]: np.asarray(t["values"], float)
               for t in self.traces if str(t.get("name", "")).startswith(prefix)}
        if not out:
            raise KeyError(f"no traces starting with {prefix!r} "
                           f"(have: {self.names or '—'})")
        return out

    def _spectrum_plots(self) -> list[dict]:
        return [ep for ep in (self.raw.get("extra_plots") or [])
                if ep.get("xunit") == "nm"]

    @property
    def spectra(self) -> list[str]:
        """Probe names that produced an optical-spectrum plot (probes with
        the "spectrum" flag set in the browser; transient runs only)."""
        out = []
        for ep in self._spectrum_plots():
            for t in ep.get("traces") or []:
                n = str(t.get("name", ""))
                out.append(n[: -len(" spectrum")]
                           if n.endswith(" spectrum") else n)
        return out

    def spectrum(self, name: str | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        """One optical spectrum as ``(wavelength_nm, dB)`` arrays.

        The server FFTs the complex field envelope of each spectrum-flagged
        probe and normalises to that probe's peak; the plot has its own
        wavelength axis (NOT ``Result.x``, which stays the time axis).
        ``name`` is the probe name; omit it when the run has exactly one."""
        plots = self._spectrum_plots()
        if name is not None:
            plots = [ep for ep in plots
                     if any(str(t.get("name", "")).startswith(name)
                            for t in ep.get("traces") or [])]
        if not plots:
            raise KeyError(f"no optical spectrum {name!r} "
                           f"(have: {self.spectra or '—'} — the probe needs "
                           "its spectrum flag set, and only transient runs "
                           "produce spectra)" if name else
                           "no optical spectra in this result (flag a probe "
                           "as spectrum in the browser, or set "
                           "\"spectrum\": true on it — transient runs only)")
        if len(plots) > 1:
            raise KeyError("several spectrum probes "
                           f"({self.spectra}) — pass the probe name")
        ep = plots[0]
        return (np.asarray(ep["x"], float),
                np.asarray(ep["traces"][0]["values"], float))

    def plot(self, *names, ax=None, **kw):
        """Plot traces (all of them by default) against the shared x-axis.
        Returns the matplotlib axes; ``**kw`` passes through to ``ax.plot``."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        picked = ([self.trace(n) for n in names] if names else
                  [t for t in self.traces if t.get("values")])
        for t in picked:
            unit = f" [{t['unit']}]" if t.get("unit") else ""
            ax.plot(self.x, np.asarray(t["values"], float),
                    label=f"{t.get('name', '')}{unit}", **kw)
        if self.raw.get("xlog"):
            ax.set_xscale("log")
        ax.set_xlabel(self.xlabel)
        if picked:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def __repr__(self) -> str:
        if not self.ok:
            return f"<Result FAILED: {self.raw.get('error')}>"
        if self.kind == "op":
            return f"<Result op: {len(self.table)} rows>"
        return (f"<Result {self.kind}: {len(self.x)} points, "
                f"{len(self.traces)} traces: "
                + ", ".join(self.names[:8])
                + (", ..." if len(self.names) > 8 else "") + ">")


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------
class Session:
    """Connection to a running photonflux web server (webapp/server.py)."""

    def __init__(self, url: str = DEFAULT_URL, progress: bool = True):
        self.url = url.rstrip("/")
        self.progress = progress
        self._catalog: dict | None = None

    # --- plumbing ----------------------------------------------------------
    def _req(self, path: str, obj: dict | None = None,
             timeout: float | None = 30.0) -> dict:
        if obj is None:
            req = urllib.request.Request(self.url + path)
        else:
            req = urllib.request.Request(
                self.url + path, data=json.dumps(obj).encode(),
                headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())    # error responses are JSON too
            except Exception:
                raise SessionError(f"HTTP {e.code} from {path}") from None
        except OSError as e:
            raise SessionError(
                f"cannot reach the photonflux server at {self.url} — "
                f"is `python webapp/server.py` running? ({e})") from None

    def catalog(self) -> dict:
        """The component catalog (type -> ports/params), cached."""
        if self._catalog is None:
            self._catalog = self._req("/api/components")
        return self._catalog

    # --- the live mirror ---------------------------------------------------
    def pull(self, required: bool = True) -> Schematic | None:
        """The live schematic from the browser's active tab."""
        snap = self._req("/api/schematic")
        if snap.get("doc") is None:
            if required:
                raise SessionError(
                    "the schematic mirror is empty — open (or reload) "
                    f"{self.url} in a browser so the editor connects, or "
                    "push a schematic with Session.push()")
            return None
        return Schematic(snap["doc"], rev=snap.get("rev", 0),
                         source=snap.get("source", ""))

    schematic = pull   # reads nicely: s.schematic()

    def push(self, sch, title: str | None = None,
             analysis: dict | None = None, base_rev: int | None = None) -> int:
        """Send a schematic to the canvas (an undoable edit in the browser).

        Accepts a ``Schematic``, a ``Builder``, a Save-format document, or a
        bare ``{instances, wires, probes}`` dict. Returns the new mirror rev.
        Without ``base_rev`` this replaces whatever is on the canvas."""
        if isinstance(sch, Builder):
            doc = sch.doc(title=title or "", analysis=analysis)
        elif isinstance(sch, Schematic):
            doc = sch.doc
        elif isinstance(sch, dict) and "schematic" in sch:
            doc = sch
        elif isinstance(sch, dict):
            doc = {"title": title or "", "schematic": sch}
        else:
            raise TypeError(f"cannot push {type(sch).__name__}")
        if title is not None:
            doc["title"] = title
        if analysis is not None:
            doc["analysis"] = analysis
        res = self._req("/api/schematic",
                        {"source": "notebook", "doc": doc,
                         **({"base_rev": base_rev} if base_rev is not None
                            else {})})
        if not res.get("ok"):
            raise SessionError(res.get("error") or "push rejected")
        return res["rev"]

    def __getitem__(self, key: str):
        return self.pull()[key]

    def __setitem__(self, key: str, value) -> None:
        self.set(**{key: value})

    def set(self, **params) -> int:
        """Set instance parameters on the live canvas, e.g.
        ``s.set(**{"MZM1.n_rf": 4.2, "WG1.length_m": "200u"})``.

        Read-modify-write with optimistic concurrency: if a canvas edit races
        us the write is retried on the fresh document, so it edits only the
        named settings and never clobbers other work."""
        for attempt in range(5):
            sch = self.pull()
            for key, value in params.items():
                self._check_param(sch, key)
                sch[key] = value
            res = self._req("/api/schematic",
                            {"source": "notebook", "doc": sch.doc,
                             "base_rev": sch.rev})
            if res.get("ok"):
                return res["rev"]
            if not res.get("conflict"):
                raise SessionError(res.get("error") or "push rejected")
            time.sleep(0.1 * (attempt + 1))
        raise SessionError("the canvas kept changing under us — retry set()")

    def _check_param(self, sch: Schematic, key: str) -> None:
        """Catch typo'd parameter names against the catalog when possible
        (settings hold only non-default values, so absence proves nothing)."""
        inst, param = sch._split(key)
        if param is None:
            raise KeyError("set 'INST.param', not a whole instance")
        if param in sch.instances[inst].get("settings", {}):
            return
        try:
            entry = self.catalog().get(sch.instances[inst].get("type")) or {}
        except SessionError:
            return
        names = [p.get("name") for p in entry.get("params") or []]
        if names and param not in names:
            raise KeyError(f"{sch.instances[inst].get('type')} has no "
                           f"parameter {param!r} (has: {', '.join(names)})")

    def selected(self) -> tuple[str, dict] | None:
        """The instance currently selected in the browser, as ``(ref, inst)``
        — or None. The 'analyze what I just clicked' gesture."""
        sch = self.pull(required=False)
        sel = sch.selection if sch else None
        if sel and sel.get("kind") == "inst":
            ref = sel.get("id")
            inst = sch.instances.get(ref)
            if inst is not None:
                return ref, inst
        return None

    def watch(self, sources: tuple = ("browser",), debounce: float = 0.0,
              initial: bool = False):
        """Yield the live ``Schematic`` every time it changes.

        ``sources`` filters whose edits you care about (by default only the
        browser's, so your own pushes don't re-trigger you); ``debounce``
        waits that many seconds of quiet before yielding, coalescing bursts;
        ``initial=True`` also yields the current state immediately (the
        live-bench loop usually wants to draw once before the first edit).
        Reconnects across dev-server restarts; stop with Ctrl-C / break."""
        last = -1
        prime_from_stream = False
        if not initial:
            # prime with the current rev *now* (not on first next()) so the
            # subscription's connect snapshot is never mistaken for an edit
            try:
                last = self._req("/api/schematic").get("rev", 0)
            except SessionError:
                # server not up yet: let the connect snapshot prime instead
                prime_from_stream = True
        return self._watch(sources, debounce, last,
                           bypass_first=initial,
                           prime_from_stream=prime_from_stream)

    def _watch(self, sources: tuple, debounce: float, last: int,
               bypass_first: bool, prime_from_stream: bool):
        while True:
            try:
                for ev in self._sse("/api/schematic/events"):
                    rev = ev.get("rev", 0)
                    if prime_from_stream:       # late prime (see watch())
                        prime_from_stream = False
                        last = rev
                        continue
                    if rev == last or not rev:
                        continue
                    last = rev
                    # initial=True: the connect snapshot is the state the
                    # caller asked to see first — its last *writer* may be
                    # anyone (often ourselves, right after a push), so the
                    # source filter only applies from the second event on
                    if not bypass_first and sources and \
                            ev.get("source") not in sources:
                        continue
                    if debounce:
                        time.sleep(debounce)
                    sch = self.pull(required=False)
                    if sch is None:
                        continue
                    if not bypass_first and sources and \
                            sch.source not in sources:
                        continue
                    bypass_first = False
                    last = max(last, sch.rev)
                    yield sch
            except (SessionError, OSError, ValueError):
                time.sleep(1.0)     # server restarting (dev reload) — rejoin

    def _sse(self, path: str):
        """Minimal server-sent-events reader (change events as dicts)."""
        req = urllib.request.Request(self.url + path,
                                     headers={"Accept": "text/event-stream"})
        # heartbeats arrive every 15 s; a 60 s socket timeout detects a peer
        # that died without closing and triggers watch()'s reconnect
        resp = urllib.request.urlopen(req, timeout=60.0)
        try:
            event, data = None, []
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line:
                    if event == "change" and data:
                        try:
                            yield json.loads("".join(data))
                        except json.JSONDecodeError:
                            pass
                    event, data = None, []
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data.append(line[len("data:"):].strip())
        finally:
            resp.close()

    # --- running analyses --------------------------------------------------
    def run(self, analysis: dict | None = None, schematic=None,
            progress: bool | None = None) -> Result:
        """Run one analysis and return its :class:`Result`.

        With no arguments this is the notebook twin of the Run button: the
        live canvas schematic with the analysis configured in its toolbar.
        Pass ``analysis`` (see the ``transient``/``dc``/``dcsweep``/``ac``/
        ``noise``/``pulse`` helpers) and/or an explicit ``schematic``
        (a Schematic, Builder, or dict) to override either half. Progress is
        drawn from ``/api/progress``; Ctrl-C cancels the run server-side."""
        if schematic is None:
            live = self.pull()
            sch = live.schematic
            if analysis is None:
                analysis = live.analysis
        elif isinstance(schematic, Builder):
            sch = schematic.doc()["schematic"]
        elif isinstance(schematic, Schematic):
            sch = schematic.schematic
            if analysis is None:
                analysis = schematic.analysis
        elif isinstance(schematic, dict):
            sch = schematic.get("schematic", schematic)
            if analysis is None:
                analysis = schematic.get("analysis")
        else:
            raise TypeError(f"cannot run a {type(schematic).__name__}")
        if not analysis:
            raise SessionError("no analysis: configure one in the browser "
                               "toolbar or pass one (e.g. s.transient('20n'))")
        show = self.progress if progress is None else progress
        bar = _ProgressBar(self) if show else None
        try:
            raw = self._req("/api/run",
                            {"schematic": sch, "analysis": analysis},
                            timeout=None)
        except KeyboardInterrupt:
            self.cancel()
            raise
        finally:
            if bar:
                bar.stop()
        if not raw.get("ok"):
            raise RunError(raw.get("error") or "run failed",
                           log=raw.get("log"))
        return Result(raw)

    def transient(self, t_stop="4n", points: int = 800, solver: str = "",
                  dtmax=None, seeds: int = 0, noise_bw="50G",
                  link: str = "", expressions: str = "",
                  schematic=None) -> Result:
        """Time-domain solve. ``seeds`` > 0 enables transient noise; ``link``
        names the probe for the BER/link report."""
        a: dict = {"mode": "transient", "t_stop": si(t_stop),
                   "points": int(points)}
        if solver:
            a["solver"] = solver
        if dtmax is not None:
            a["dtmax"] = si(dtmax)
        if seeds:
            a["noise"] = {"seeds": int(seeds), "bw": si(noise_bw)}
        if link:
            a["link"] = {"probe": link}
        if expressions:
            a["expressions"] = expressions
        return self.run(a, schematic)

    def dc(self, schematic=None) -> Result:
        """DC operating point (rows in ``Result.table``)."""
        return self.run({"mode": "dc"}, schematic)

    def dcsweep(self, instance: str, param: str, start=None, stop=None,
                points: int = 101, values=None, step_instance: str = "",
                step_param: str = "", step_values=None,
                schematic=None) -> Result:
        """Vectorized DC sweep of one instance parameter (``instance="*"``
        sweeps every instance that has it — the spectrum trick), with an
        optional stepped second parameter for curve families."""
        a: dict = {"mode": "dcsweep", "instance": instance, "param": param}
        if values is not None:
            a["values"] = [si(v) for v in values]
        else:
            a.update(start=si(start), stop=si(stop), points=int(points))
        if step_instance:
            a.update(step_instance=step_instance, step_param=step_param,
                     step_values=[si(v) for v in (step_values or [])])
        return self.run(a, schematic)

    def ac(self, f_start="1meg", f_stop="100G", points: int = 121,
           z0: float = 50, schematic=None) -> Result:
        """AC S-parameter / h21 sweep (probes pair by ``in_<x>``/``out_<x>``)."""
        return self.run({"mode": "ac", "f_start": si(f_start),
                         "f_stop": si(f_stop), "points": int(points),
                         "z0": float(z0)}, schematic)

    def noise(self, probe: str, f_start="1k", f_stop="10G",
              points: int = 121, schematic=None) -> Result:
        """Small-signal output-referred noise at ``probe``."""
        return self.run({"mode": "noise", "probe": probe,
                         "f_start": si(f_start), "f_stop": si(f_stop),
                         "points": int(points)}, schematic)

    def pulse(self, probe: str, t_stop="4n", points: int = 2000,
              ffe_taps: int = 0, dfe_taps: int = 0, schematic=None) -> Result:
        """Pulse/COM analysis at ``probe`` (single-pulse rerun + Wiener EQ)."""
        return self.run({"mode": "pulse", "probe": probe, "t_stop": si(t_stop),
                         "points": int(points), "ffe_taps": int(ffe_taps),
                         "dfe_taps": int(dfe_taps)}, schematic)

    def cancel(self) -> None:
        """Stop the in-flight run (the web UI's Stop button)."""
        try:
            self._req("/api/cancel", {})
        except SessionError:
            pass

    # --- convenience -------------------------------------------------------
    def examples(self) -> list[dict]:
        """The built-in example circuits (id/title/description/group)."""
        return self._req("/api/examples")

    def load_example(self, example_id: str) -> Schematic:
        """Fetch one built-in example as a :class:`Schematic` (push it to the
        canvas with ``s.push(...)``, or run it directly)."""
        doc = self._req(f"/api/examples/{example_id}")
        if "error" in doc and "schematic" not in doc:
            raise SessionError(doc["error"])
        return Schematic(doc)

    def snapshot(self, name: str) -> Path:
        """Freeze the live document to ``<name>.json`` next to the notebook —
        provenance for an analysis that must outlive later canvas edits."""
        return self.pull().save(name)

    def __repr__(self) -> str:
        try:
            snap = self._req("/api/schematic", timeout=3.0)
            state = (f"rev {snap.get('rev')} from {snap.get('source') or '—'}"
                     if snap.get("doc") else "empty mirror")
            return f"<Session {self.url} — {state}>"
        except SessionError:
            return f"<Session {self.url} — UNREACHABLE>"


class _ProgressBar:
    """Poll ``/api/progress`` while a run is in flight and draw one stderr
    line. Quiet runs (compile-cached DC sweeps) never draw: the first paint
    waits until the run has been active for half a second."""

    def __init__(self, session: Session, interval: float = 0.3):
        self._s = session
        self._stop = threading.Event()
        self._drew = False
        self._t = threading.Thread(target=self._loop, args=(interval,),
                                   daemon=True)
        self._t.start()

    def _loop(self, interval: float) -> None:
        started = time.monotonic()
        while not self._stop.wait(interval):
            try:
                p = self._s._req("/api/progress", timeout=3.0)
            except SessionError:
                continue
            if not p.get("active") or time.monotonic() - started < 0.5:
                continue
            frac = float(p.get("frac") or 0.0)
            phase = p.get("phase") or "running"
            runs = int(p.get("runs") or 1)
            tag = f" (run {int(p.get('run', 0)) + 1}/{runs})" if runs > 1 else ""
            n = int(frac * 30)
            sys.stderr.write(f"\r[{'#' * n}{'-' * (30 - n)}] "
                             f"{frac * 100:3.0f}% {phase}{tag}   ")
            sys.stderr.flush()
            self._drew = True

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)
        if self._drew:
            sys.stderr.write("\r" + " " * 60 + "\r")
            sys.stderr.flush()


# ---------------------------------------------------------------------------
# programmatic schematic construction
# ---------------------------------------------------------------------------
class _Part:
    """Handle for one placed instance; attribute access names a port
    (``las.p1`` -> ``"LAS1,p1"``) for ``Builder.wire``/``probe``."""

    def __init__(self, ref: str, type_: str):
        self.ref = ref
        self.type = type_

    def port(self, name: str) -> str:
        return f"{self.ref},{name}"

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.port(name)

    def __repr__(self) -> str:
        return f"<{self.type} {self.ref}>"


_PROBE_COLORS = ["#6ecbf5", "#f5b96e", "#8ef58e", "#f58ef0",
                 "#f5f06e", "#8e9ef5", "#f56e6e", "#6ef5d2"]


class Builder:
    """Construct a schematic programmatically — the parametric-generation
    side of the bridge (an N-channel link is a loop, not N×10 clicks)::

        b = Builder()
        las = b.add("cw_laser", power=1e-3, wavelength_nm=1310)
        gnd = b.add("ground")
        cpl = b.add("dir_coupler", coupling=0.088)
        b.wire(las.p1, cpl.p1); b.wire(las.p2, gnd.p1)
        b.probe(cpl.p3, name="thru")
        s.push(b, title="ring bench")       # onto the canvas, undoably

    Parts placed without explicit coordinates get a plain left-to-right grid
    — push it and tidy the wires in the browser, which is the point."""

    def __init__(self, baud: float = 10e9):
        self.baud = si(baud)
        self.instances: dict[str, dict] = {}
        self.wires: list[dict] = []
        self.probes: list[dict] = []
        self._auto: list[str] = []      # refs awaiting grid placement

    def add(self, type_: str, ref: str | None = None, x: float | None = None,
            y: float | None = None, rot: int = 0, **settings) -> _Part:
        """Place a component; ``**settings`` are its non-default parameters
        (SI-suffix strings accepted). Returns a :class:`_Part` port handle."""
        if ref is None:
            stem = re.sub(r"[^A-Za-z]", "", type_)[:3].upper() or "X"
            n = 1
            while f"{stem}{n}" in self.instances:
                n += 1
            ref = f"{stem}{n}"
        elif ref in self.instances:
            raise ValueError(f"duplicate ref {ref!r}")
        inst = {"type": type_, "x": 0, "y": 0, "rot": int(rot),
                "settings": {k: _maybe_si(v) for k, v in settings.items()}}
        if x is None and y is None:
            self._auto.append(ref)
        else:
            inst["x"], inst["y"] = float(x or 0), float(y or 0)
        self.instances[ref] = inst
        return _Part(ref, type_)

    def wire(self, a: str, b: str) -> None:
        """Connect two ports (``part.p1`` handles or ``"REF,p1"`` strings)."""
        self.wires.append({"from": str(a), "to": str(b)})

    def probe(self, at: str, name: str | None = None) -> str:
        """Attach a probe (what gets recorded and plotted) to a port."""
        name = name or f"probe{len(self.probes) + 1}"
        color = _PROBE_COLORS[len(self.probes) % len(_PROBE_COLORS)]
        self.probes.append({"name": name, "at": str(at), "color": color})
        return name

    def doc(self, title: str = "", analysis: dict | None = None) -> dict:
        """The Save-format document (grid-placing any auto-positioned parts)."""
        for i, ref in enumerate(self._auto):
            self.instances[ref]["x"] = 40 + 180 * (i % 6)
            self.instances[ref]["y"] = 60 + 140 * (i // 6)
        return {"title": title,
                "schematic": {"instances": self.instances, "wires": self.wires,
                              "probes": self.probes, "notes": [],
                              "globals": {"baud": self.baud}},
                "analysis": analysis}

    def schematic(self) -> Schematic:
        return Schematic(self.doc())

    def __repr__(self) -> str:
        return (f"<Builder: {len(self.instances)} instances, "
                f"{len(self.wires)} wires, {len(self.probes)} probes>")
