"""Live progress state for the running simulation, polled by the web UI.

A time-domain solve is a single compiled diffrax/circulax call, so the browser
can't watch it advance the way a shell tqdm bar does. Instead the transient
solver reports where it is in the ``[0, t_stop]`` time interval through a
``jax.debug.callback`` embedded in the diffrax ``SaveAt`` fn (see
``simulate._run_transient``). That callback runs on the host *during* the solve,
updating the singleton here; the frontend polls ``GET /api/progress`` on a timer
while its ``/api/run`` request is in flight and fills a bar from these numbers.

The compile phase (which precedes stepping and can dominate the first run) emits
no callbacks, so ``phase`` starts as ``"compiling"`` and flips to ``"solving"``
once the first step reports — the UI shows an indeterminate bar until then.

Progress is a single monotonically-increasing fraction over the *whole* run,
including any parameter sweep (``runs``) and transient-noise seeds (``seeds``):
each seed of each sweep point owns an equal ``1 / (runs * seeds)`` slice.

A flow (``webapp/flow.py``) additionally publishes its live state through the
``flow_*`` methods: each step's status, its outputs and checks once done, every
Optimize evaluation as it lands, and the flow log. The snapshot carries it as
``flow`` so the Flow tab, the notebook bridge and the CLI can show a long flow
advancing; ``flow_echo`` (a callable) also prints each log line as it is added.
"""
from __future__ import annotations

import threading
import time

FLOW_LOG_TAIL = 400   # log lines a snapshot carries (``log_n`` counts them all)


class RunCancelled(Exception):
    """Raised from inside the transient solve when the user hits Stop.

    The per-save-point host callback checks the cancel flag and raises this to
    abort the compiled diffrax loop cooperatively; ``simulate.run`` turns it
    into a clean ``{"ok": False, "cancelled": True}`` result."""


class _Progress:
    """Thread-safe progress for one in-flight simulation.

    The solve worker mutates it via :meth:`report` (and the structural setters);
    the HTTP handler thread reads it via :meth:`snapshot`. A run is bracketed by
    :meth:`enter`/:meth:`leave`, which are re-entrancy safe so the recursive
    ``run()`` calls a parameter sweep makes don't reset each other.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False  # only the web server turns this on (see enable())
        self._active = False
        self._frac = 0.0
        self._phase = ""      # "" | "compiling" | "solving"
        self._runs = 1        # sweep points (outer)
        self._run_i = 0
        self._seeds = 1       # transient-noise seeds (inner)
        self._seed = 0
        self._depth = 0       # nested run() re-entrancy guard
        self._cancel = False  # set by request_cancel(); the solve callback raises
        self._label = ""      # flow stage text (set_label)
        self._span = 0.0      # simulated t_stop of the current solve
        self._flow: dict | None = None   # live flow state (flow_begin)
        self.flow_echo = None  # optional callable(line, elapsed_s) per log line

    def enable(self) -> None:
        """Opt this process in to per-step progress reporting. The web server
        calls this at startup; library/CLI callers (tests, warmup, a bare
        ``import simulate``) leave it off, so the transient solver skips the
        per-save-point host callback entirely and runs exactly as before."""
        with self._lock:
            self._enabled = True

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    # --- lifecycle: bracket a whole run() ---------------------------------
    def enter(self) -> None:
        with self._lock:
            self._depth += 1
            if self._depth == 1:  # outermost run(): start fresh
                self._active = True
                self._cancel = False  # a stale Stop must not kill the next run
                self._frac = 0.0
                self._phase = "compiling"
                self._runs = 1
                self._run_i = 0
                self._seeds = 1
                self._seed = 0
                self._label = ""
                self._flow = None

    def leave(self) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:  # outermost run() done
                self._active = False
                self._frac = 1.0
                self._phase = ""
                self._label = ""
                self._flow = None

    # --- cooperative cancellation (the web UI's Stop button) --------------
    def request_cancel(self) -> None:
        """Ask the in-flight run to stop. Honoured only while a run is active
        (the flag is cleared when the next run starts), and only at a following
        save point of a time-domain solve — the callback is async, so a few more
        steps may run before it lands. DC/AC/op emit no per-step callback, so
        Stop can't interrupt them mid-solve; the frontend still frees the UI by
        aborting the request."""
        with self._lock:
            if self._active:
                self._cancel = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    # --- structure: sweeps (outer) and noise seeds (inner) ----------------
    def set_runs(self, n: int) -> None:
        with self._lock:
            self._runs = max(1, int(n))

    def set_run(self, i: int) -> None:
        with self._lock:
            self._run_i = max(0, int(i))
            self._seed = 0
            self._phase = "compiling"

    def restart(self) -> None:
        """Start the bar over for the next sub-run of a multi-run job (a flow
        runs many analyses inside one outermost run(), and the fraction is
        otherwise monotonic across all of them)."""
        with self._lock:
            self._frac = 0.0
            self._phase = "compiling"
            self._runs, self._run_i, self._seeds, self._seed = 1, 0, 1, 0

    def set_label(self, text: str) -> None:
        """Free-text stage shown beside the bar (a flow's "step 3/4 · eval
        12/30"); cleared when the outermost run ends."""
        with self._lock:
            self._label = str(text)

    def set_seeds(self, n: int) -> None:
        with self._lock:
            self._seeds = max(1, int(n))

    def set_seed(self, k: int) -> None:
        with self._lock:
            self._seed = max(0, int(k))

    # --- per-step report (the jax.debug.callback target) ------------------
    def set_span(self, t_stop: float) -> None:
        """Simulated span of the current solve, for :meth:`report_time`."""
        with self._lock:
            self._span = float(t_stop)

    def report_time(self, t) -> None:
        """:meth:`report` from a raw solver time. The compiled solve passes
        ``t`` rather than ``t / t_stop`` so its callback closes over nothing
        run-specific and the compiled program is reused across runs."""
        with self._lock:
            span = self._span
        self.report(float(t) / span if span > 0 else 0.0)

    def report(self, local) -> None:
        """Record ``local`` in ``[0, 1]`` — position within the current seed's
        solve — folding it into the run-wide fraction. Called from inside the
        compiled solve. It raises :class:`RunCancelled` only when the user has
        hit Stop (that is how the solve is aborted); otherwise it never raises."""
        with self._lock:
            if self._active and self._cancel:
                raise RunCancelled("run stopped by user")
        try:
            lf = float(local)
        except (TypeError, ValueError):
            return
        if lf < 0.0:
            lf = 0.0
        elif lf > 1.0:
            lf = 1.0
        with self._lock:
            if not self._active:
                return
            denom = self._runs * self._seeds
            g = ((self._run_i * self._seeds + self._seed + lf) / denom
                 if denom else lf)
            if g > self._frac:  # monotonic: never let a late/out-of-order call regress
                self._frac = g
            self._phase = "solving"

    # --- live flow state (flow.py) -----------------------------------------
    # Recorded whether or not enable() was called: these are host-side calls
    # once per step / evaluation, so the in-process CLI can echo them too.
    def flow_begin(self, name: str, steps: list[dict]) -> None:
        """Start a flow's live record; ``steps`` are {id, title, kind,
        status} dicts (status "pending" or "skipped")."""
        with self._lock:
            self._flow = {"name": str(name), "t0": time.monotonic(),
                          "steps": [dict(s) for s in steps], "log": [],
                          "log_n": 0}

    def flow_step(self, sid: str, **fields) -> None:
        """Update one step's live entry (status, time_s, outputs, checks,
        error, eval/iters, ...)."""
        with self._lock:
            for s in (self._flow or {}).get("steps", []):
                if s["id"] == sid:
                    s.update(fields)

    def flow_eval(self, sid: str, entry: dict) -> None:
        """Append one Optimize evaluation to its step's live history."""
        with self._lock:
            for s in (self._flow or {}).get("steps", []):
                if s["id"] == sid:
                    s.setdefault("history", []).append(dict(entry))
                    s["eval"] = len(s["history"])

    def flow_log(self, line: str) -> None:
        with self._lock:
            fl = self._flow
            if fl is not None:
                fl["log"].append(str(line))
                fl["log_n"] += 1
                del fl["log"][:-FLOW_LOG_TAIL]
            elapsed = time.monotonic() - fl["t0"] if fl else 0.0
            echo = self.flow_echo
        if echo is not None:
            try:            # a broken printer must never fail the run
                echo(str(line), elapsed)
            except Exception:  # noqa: BLE001
                pass

    def snapshot(self) -> dict:
        with self._lock:
            out = {"active": self._active, "frac": round(self._frac, 4),
                   "phase": self._phase,
                   "runs": self._runs, "run": self._run_i,
                   "label": self._label}
            fl = self._flow
            if fl is not None:
                out["flow"] = {
                    "name": fl["name"],
                    "elapsed_s": round(time.monotonic() - fl["t0"], 2),
                    "steps": [{**s, "history": list(s["history"])}
                              if "history" in s else dict(s)
                              for s in fl["steps"]],
                    "log": list(fl["log"]), "log_n": fl["log_n"]}
            return out


def printer(stream):
    """A ``flow_echo`` writing ``[flow 1m02s] <line>`` to ``stream``."""
    def echo(line: str, elapsed: float) -> None:
        m, sec = divmod(int(elapsed), 60)
        stream.write(f"[flow {m}m{sec:02d}s] {line}\n")
        stream.flush()
    return echo


PROGRESS = _Progress()
