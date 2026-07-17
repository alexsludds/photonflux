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
"""
from __future__ import annotations

import threading


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
                self._frac = 0.0
                self._phase = "compiling"
                self._runs = 1
                self._run_i = 0
                self._seeds = 1
                self._seed = 0

    def leave(self) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:  # outermost run() done
                self._active = False
                self._frac = 1.0
                self._phase = ""

    # --- structure: sweeps (outer) and noise seeds (inner) ----------------
    def set_runs(self, n: int) -> None:
        with self._lock:
            self._runs = max(1, int(n))

    def set_run(self, i: int) -> None:
        with self._lock:
            self._run_i = max(0, int(i))
            self._seed = 0
            self._phase = "compiling"

    def set_seeds(self, n: int) -> None:
        with self._lock:
            self._seeds = max(1, int(n))

    def set_seed(self, k: int) -> None:
        with self._lock:
            self._seed = max(0, int(k))

    # --- per-step report (the jax.debug.callback target) ------------------
    def report(self, local) -> None:
        """Record ``local`` in ``[0, 1]`` — position within the current seed's
        solve — folding it into the run-wide fraction. Called from inside the
        compiled solve, so it must never raise."""
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

    def snapshot(self) -> dict:
        with self._lock:
            return {"active": self._active, "frac": round(self._frac, 4),
                    "phase": self._phase,
                    "runs": self._runs, "run": self._run_i}


PROGRESS = _Progress()
