"""Run circuits and hand back results as numpy.

The Engine wraps the process-wide libngspice instance: it loads whatever
OSDI modules a Circuit needs, loads the deck, runs analyses, and snapshots
every result vector into a `Result`. Engines are cheap façades — all of
them share the singleton simulator.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ._ngspice import NgSpice
from .circuit import Circuit

__all__ = ["Engine", "Result"]


class Result:
    """A snapshot of one ngspice plot: dict-like, case-insensitive."""

    def __init__(self, plot: str, vectors: dict[str, np.ndarray], log: list[str]):
        self.plot = plot
        self._vectors = vectors
        self._lower = {k.lower(): k for k in vectors}
        self.log = log

    @property
    def names(self) -> list[str]:
        return sorted(self._vectors)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._lower

    def __getitem__(self, name: str) -> np.ndarray:
        key = self._lower.get(name.lower())
        if key is None:
            raise KeyError(f"no vector {name!r}; available: {self.names}")
        return self._vectors[key]

    def v(self, node: str) -> np.ndarray:
        """Node voltage. For photonic nodes this is optical power in W."""
        return self[node]

    def i(self, vsource: str) -> np.ndarray:
        """Current through a voltage source (ngspice `<name>#branch`)."""
        return self[f"{vsource}#branch"]

    @property
    def t(self) -> np.ndarray:
        return self["time"]

    @property
    def f(self) -> np.ndarray:
        return np.real(self["frequency"])

    def __repr__(self) -> str:
        return f"Result(plot={self.plot!r}, vectors={self.names})"


class Engine:
    def __init__(self, library_path: str | None = None):
        self._ng = NgSpice.get(library_path)

    # ------------------------------------------------------------ core
    def run(
        self,
        circuit: Circuit | str,
        analyses: str | Iterable[str],
        osdi: Iterable[str] | None = None,
    ) -> Result:
        """Load `circuit`, run one or more analysis commands, snapshot vectors."""
        if isinstance(circuit, Circuit):
            self._ng.load_osdi(*circuit.osdi_files)
            text = circuit.build()
        else:
            text = circuit
        if osdi:
            self._ng.load_osdi(*osdi)

        self._ng.load_netlist(text)
        log: list[str] = []
        for cmd in [analyses] if isinstance(analyses, str) else list(analyses):
            log += self._ng.cmd(cmd)
        plot = self._ng.current_plot()
        return Result(plot, self._ng.vectors(plot), log)

    # ------------------------------------------------------------ sugar
    def op(self, circuit: Circuit | str, **kw) -> Result:
        return self.run(circuit, "op", **kw)

    def tran(
        self,
        circuit: Circuit | str,
        step: float | str,
        stop: float | str,
        start: float | str | None = None,
        uic: bool = False,
        **kw,
    ) -> Result:
        cmd = f"tran {_t(step)} {_t(stop)}"
        if start is not None:
            cmd += f" {_t(start)}"
        if uic:
            cmd += " uic"
        return self.run(circuit, cmd, **kw)

    def ac(
        self,
        circuit: Circuit | str,
        fstart: float | str,
        fstop: float | str,
        points: int = 20,
        variation: str = "dec",
        **kw,
    ) -> Result:
        return self.run(circuit, f"ac {variation} {points} {_t(fstart)} {_t(fstop)}", **kw)

    def free_plots(self) -> None:
        """Release plot memory accumulated across runs (long sweeps)."""
        self._ng.free_plots()

    @property
    def transcript(self) -> str:
        return self._ng.transcript


def _t(x: float | str) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)
