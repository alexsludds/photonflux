"""Direct ctypes binding to libngspice (the shared-library ngspice API).

There is no ngspice *simulation* flow — every circuit is solved by circulax
(JAX). This binding exists only so that ``cx.sky130_card`` can have ngspice
resolve the SKY130 PDK (``.lib`` corner stitching, ``{...}`` card expressions,
W/L bin selection) and hand back the fully-expanded BSIM4 model card via
``showmod``. It is a ~250-line wrapper that does exactly that and nothing else:

  * synchronous command execution with captured stdout/stderr,
  * netlist loading via ngSpice_Circ,
  * OSDI module loading (deduplicated per process),
  * vector extraction into numpy arrays (real and complex),
  * honest error surfacing (ngspice "Error:" lines raise NgSpiceError).

libngspice keeps *global* state: there is exactly one simulator instance
per process. `NgSpice.get()` returns the process-wide singleton.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_short,
    c_void_p,
)
from ctypes.util import find_library
from pathlib import Path

import numpy as np

__all__ = ["NgSpice", "NgSpiceError", "NgSpiceFatalError"]


class NgSpiceError(RuntimeError):
    """An ngspice command or netlist produced an error."""


class NgSpiceFatalError(NgSpiceError):
    """ngspice requested process exit; the simulator is unusable now."""


class _ngcomplex(Structure):
    _fields_ = [("cx_real", c_double), ("cx_imag", c_double)]


class _vector_info(Structure):
    _fields_ = [
        ("v_name", c_char_p),
        ("v_type", c_int),
        ("v_flags", c_short),
        ("v_realdata", POINTER(c_double)),
        ("v_compdata", POINTER(_ngcomplex)),
        ("v_length", c_int),
    ]


_VF_REAL = 1 << 0
_VF_COMPLEX = 1 << 1

_SendChar = CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
_SendStat = CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
_ControlledExit = CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)


def _default_library_candidates() -> list[str]:
    cands = []
    env = os.environ.get("NGSPICE_LIBRARY_PATH")
    if env:
        cands.append(env)
    cands += [
        "/opt/homebrew/lib/libngspice.dylib",   # Apple Silicon Homebrew
        "/usr/local/lib/libngspice.dylib",      # Intel-mac Homebrew
        "/usr/lib/libngspice.so",
        "/usr/local/lib/libngspice.so",
    ]
    found = find_library("ngspice")
    if found:
        cands.append(found)
    return cands


def find_libngspice() -> str:
    for cand in _default_library_candidates():
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "libngspice not found. Install it (macOS: `brew install libngspice`) "
        "or point NGSPICE_LIBRARY_PATH at the shared library."
    )


class NgSpice:
    """Process-wide singleton wrapper around libngspice."""

    _instance: "NgSpice | None" = None

    @classmethod
    def get(cls, library_path: str | None = None) -> "NgSpice":
        if cls._instance is None:
            cls._instance = cls(library_path)
        return cls._instance

    def __init__(self, library_path: str | None = None):
        if NgSpice._instance is not None:
            raise RuntimeError("libngspice is a singleton; use NgSpice.get()")
        self._lib = ctypes.CDLL(library_path or find_libngspice())
        self._log: list[str] = []          # full session transcript
        self._capture: list[str] | None = None
        self._dead = False
        self._has_circuit = False
        self._osdi_loaded: set[str] = set()

        # Callbacks must stay referenced for the lifetime of the process.
        self._cb_print = _SendChar(self._on_print)
        self._cb_stat = _SendStat(lambda s, _id, _ud: 0)
        self._cb_exit = _ControlledExit(self._on_exit)

        self._lib.ngSpice_Init.restype = c_int
        init = self._lib.ngSpice_Init(
            self._cb_print, self._cb_stat, self._cb_exit, None, None, None, None
        )
        if init != 0:
            raise NgSpiceError(f"ngSpice_Init failed (rc={init})")

        self._lib.ngSpice_Command.restype = c_int
        self._lib.ngSpice_Command.argtypes = [c_char_p]
        self._lib.ngSpice_Circ.restype = c_int
        self._lib.ngSpice_Circ.argtypes = [POINTER(c_char_p)]
        self._lib.ngSpice_CurPlot.restype = c_char_p
        self._lib.ngSpice_AllVecs.restype = POINTER(c_char_p)
        self._lib.ngSpice_AllVecs.argtypes = [c_char_p]
        self._lib.ngSpice_AllPlots.restype = POINTER(c_char_p)
        self._lib.ngGet_Vec_Info.restype = POINTER(_vector_info)
        self._lib.ngGet_Vec_Info.argtypes = [c_char_p]

    # ---------------------------------------------------------------- callbacks
    def _on_print(self, line: bytes, _id: int, _ud) -> int:
        text = (line or b"").decode("utf-8", errors="replace")
        for prefix in ("stdout ", "stderr "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        self._log.append(text)
        if self._capture is not None:
            self._capture.append(text)
        return 0

    def _on_exit(self, status: int, _immediate, _quit, _id, _ud) -> int:
        # ngspice wants the process to die; refuse, but mark ourselves unusable.
        self._dead = True
        return 0

    # ---------------------------------------------------------------- commands
    def _check_alive(self) -> None:
        if self._dead:
            raise NgSpiceFatalError(
                "ngspice hit a fatal internal error earlier in this process; "
                "restart the process (or run this point in a sweep worker)."
            )

    def cmd(self, command: str, check: bool = True) -> list[str]:
        """Run one interactive-mode command; return its captured output lines."""
        self._check_alive()
        self._capture = []
        try:
            rc = self._lib.ngSpice_Command(command.encode("utf-8"))
            out = self._capture
        finally:
            self._capture = None
        errors = [ln for ln in out if "error" in ln.lower()]
        if check and (rc != 0 or self._dead or errors):
            raise NgSpiceError(
                f"ngspice command failed: {command!r}\n" + "\n".join(out[-20:])
            )
        return out

    def load_osdi(self, *paths: str | os.PathLike) -> None:
        """Load OSDI modules (compiled Verilog-A); deduplicated per process."""
        for p in paths:
            key = str(Path(p).resolve())
            if key in self._osdi_loaded:
                continue
            if not Path(key).exists():
                raise FileNotFoundError(f"OSDI module not found: {key}")
            self.cmd(f"osdi {key}")
            self._osdi_loaded.add(key)

    def load_netlist(self, text: str) -> None:
        """Load a complete SPICE deck (first line = title, must contain .end)."""
        self._check_alive()
        if self._has_circuit:
            # drop the previous circuit; ignore the complaint if none is active
            self.cmd("remcirc", check=False)
            self._has_circuit = False
        lines = text.splitlines()
        if not any(ln.strip().lower() == ".end" for ln in lines):
            lines.append(".end")
        arr = (c_char_p * (len(lines) + 1))()
        for i, ln in enumerate(lines):
            arr[i] = ln.encode("utf-8")
        arr[len(lines)] = None
        self._capture = []
        try:
            rc = self._lib.ngSpice_Circ(arr)
            out = self._capture
        finally:
            self._capture = None
        errors = [ln for ln in out if "error" in ln.lower()]
        if rc != 0 or self._dead or errors:
            raise NgSpiceError("netlist load failed:\n" + "\n".join(out[-30:]))
        self._has_circuit = True

    # ---------------------------------------------------------------- results
    def current_plot(self) -> str:
        self._check_alive()
        return (self._lib.ngSpice_CurPlot() or b"").decode()

    def vector_names(self, plot: str | None = None) -> list[str]:
        plot = plot or self.current_plot()
        raw = self._lib.ngSpice_AllVecs(plot.encode())
        names = []
        i = 0
        while raw and raw[i]:
            names.append(raw[i].decode())
            i += 1
        return names

    def vector(self, name: str, plot: str | None = None) -> np.ndarray:
        """Copy one result vector out of ngspice as a numpy array."""
        plot = plot or self.current_plot()
        vi = self._lib.ngGet_Vec_Info(f"{plot}.{name}".encode())
        if not vi:
            raise KeyError(f"vector {name!r} not found in plot {plot!r}")
        v = vi.contents
        n = v.v_length
        if v.v_flags & _VF_COMPLEX and v.v_compdata:
            flat = np.ctypeslib.as_array(
                ctypes.cast(v.v_compdata, POINTER(c_double)), shape=(2 * n,)
            )
            return (flat[0::2] + 1j * flat[1::2]).copy()
        if v.v_realdata:
            return np.ctypeslib.as_array(v.v_realdata, shape=(n,)).copy()
        return np.empty(0)

    def vectors(self, plot: str | None = None) -> dict[str, np.ndarray]:
        plot = plot or self.current_plot()
        return {name: self.vector(name, plot) for name in self.vector_names(plot)}

    def free_plots(self) -> None:
        """Release accumulated plot memory (useful inside long sweeps)."""
        self.cmd("destroy all", check=False)

    @property
    def transcript(self) -> str:
        return "\n".join(self._log)
