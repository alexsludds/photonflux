"""IBIS-AMI models in photonflux, built on PyAMI (``pyibisami``).

An IBIS-AMI model is a SerDes transmitter or receiver delivered as a shared
library with three C entry points (``AMI_Init``, ``AMI_GetWave``,
``AMI_Close``), a ``.ami`` file declaring its parameters, and an ``.ibs`` file
naming the executables per platform. This module loads such models and runs
them:

  * **Parsing** is PyAMI's: ``.ibs`` files through
    :class:`pyibisami.ibis.file.IBISModel`, ``.ami`` parameter trees through
    :class:`pyibisami.ami.parser.AMIParamConfigurator`, and the
    ``AMI_parameters_out`` strings a model returns through
    :func:`pyibisami.ami.parser.ami_parse`.
  * **Parameters** (:class:`AmiParams`) start from the ``.ami`` defaults and
    are validated against each parameter's ``Range``/``List``; the string
    handed to ``AMI_Init`` carries reserved ``In`` parameters (e.g.
    ``Modulation``) as well as the model-specific tree.
  * **Execution** is in-process through ctypes when the library's platform
    matches this interpreter's, or out-of-process through ``ami_host`` (a
    tiny C server speaking a pipe protocol, ``models/ibis_ami/host``): as a
    local subprocess (crash isolation) or inside a Linux container, which is
    how 32-bit and x86-64 Linux vendor models run on an Apple Silicon Mac.
    Windows-only DLLs need a Windows Python.

Why not ``pyibisami.ami.model.AMIModel.getWave`` for the time-domain flow:
it concatenates each call's whole (padded) clock buffer, so recovered clock
times misalign after the first call. :meth:`AmiModel.getwave` reads each
call's ticks up to the ``-1`` terminator and records the model's
``AMI_parameters_out`` after every call, which is what adaptation traces are
made of. :meth:`AmiModel.pyami_model` still hands back a PyAMI ``AMIModel``
for PyAMI's own tooling (e.g. ``get_responses``).

Model discovery (:func:`library`) covers the photonflux models
(``models/ibis_ami``, built by ``python -m photonflux ami build``), the public
third-party kits under ``examples/ibis_ami/public`` and anything dropped into
``examples/ibis_ami/vendor``. See ``examples/ibis_ami/README.md`` and
``docs/ibis-ami.md``.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import select
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PF_MODELS = REPO / "models" / "ibis_ami"
BUILD_ROOT = PF_MODELS / "build"
EXAMPLES = REPO / "examples" / "ibis_ami"
PUBLIC = EXAMPLES / "public"
VENDOR = EXAMPLES / "vendor"

__all__ = [
    "Platform", "binary_platform", "host_platform", "AmiParams", "AmiModel",
    "InitResult", "GetWaveResult", "LibraryEntry", "library", "load",
    "parse_params", "AmiError",
]


class AmiError(RuntimeError):
    """A model could not be loaded or returned failure."""


def _pyami():
    try:
        import pyibisami  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "IBIS-AMI support needs PyAMI: pip install "
            "'pyibis-ami @ git+https://github.com/capn-freako/PyAMI'"
        ) from exc


# ---------------------------------------------------------------------------
# platforms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Platform:
    """OS + CPU of a shared library or of this interpreter."""

    os: str     # "linux" | "darwin" | "windows"
    arch: str   # "x86_64" | "x86" | "arm64" | "arm" | "?"

    @property
    def bits(self) -> int:
        return 32 if self.arch in ("x86", "arm") else 64

    @property
    def tag(self) -> str:
        return f"{self.os}-{self.arch}"

    @property
    def container(self) -> str | None:
        """Docker ``--platform`` that can host this library, if any."""
        if self.os != "linux":
            return None
        return {"x86_64": "linux/amd64", "x86": "linux/386",
                "arm64": "linux/arm64"}.get(self.arch)

    def __str__(self) -> str:
        return f"{self.os} {self.arch} ({self.bits}-bit)"


def host_platform() -> Platform:
    """The platform this Python process can ``dlopen`` libraries for."""
    os_ = platform.system().lower()
    m = platform.machine().lower()
    if struct.calcsize("P") == 4:
        arch = "x86" if m in ("x86_64", "amd64", "i386", "i686", "x86") else "arm"
    else:
        arch = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "arm64",
                "arm64": "arm64"}.get(m, m)
    return Platform(os_, arch)


_ELF_MACHINE = {3: "x86", 62: "x86_64", 183: "arm64", 40: "arm"}
_MACHO_CPU = {7: "x86", 0x01000007: "x86_64", 0x0100000C: "arm64", 12: "arm"}
_PE_MACHINE = {0x14C: "x86", 0x8664: "x86_64", 0xAA64: "arm64"}


def binary_platform(path: str | Path) -> list[Platform]:
    """Platforms a shared library was built for, read from its header (ELF,
    Mach-O incl. universal, PE). Empty if the file is not a library."""
    with open(path, "rb") as f:
        b = f.read(4096)
    if b[:4] == b"\x7fELF":
        endian = "<" if b[5] == 1 else ">"
        mach = struct.unpack_from(endian + "H", b, 18)[0]
        return [Platform("linux", _ELF_MACHINE.get(mach, "?"))]
    magic = b[:4]
    if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        cpu = struct.unpack_from("<i", b, 4)[0]
        return [Platform("darwin", _MACHO_CPU.get(cpu, "?"))]
    if magic == b"\xca\xfe\xba\xbe":   # universal (fat) Mach-O, big-endian
        n = struct.unpack_from(">I", b, 4)[0]
        return [Platform("darwin", _MACHO_CPU.get(
            struct.unpack_from(">i", b, 8 + 20 * i)[0], "?"))
            for i in range(min(n, 8))]
    if b[:2] == b"MZ" and len(b) >= 0x40:
        off = struct.unpack_from("<I", b, 0x3C)[0]
        if off + 6 <= len(b) and b[off:off + 4] == b"PE\0\0":
            mach = struct.unpack_from("<H", b, off + 4)[0]
            return [Platform("windows", _PE_MACHINE.get(mach, "?"))]
    return []


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def _coerce(value: Any, ptype: str) -> Any:
    if isinstance(value, str):
        value = value.strip().strip('"')
    if ptype == "Boolean":
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)
    if ptype == "Integer":
        return int(float(value))
    if ptype in ("Float", "UI", "Tap"):
        return float(value)
    return str(value)


def _default_value(param: Any) -> Any:
    """A parameter's starting value. PyAMI's own trait default skips a
    ``List`` parameter's ``Default`` when ``List_Tip`` is present (it compares
    the raw default string to the parsed list), so resolve it here."""
    fmt, ptype = param.pformat, param.ptype
    if fmt == "List":
        vals = [_coerce(v, ptype) for v in param.pvalue]
        if param.pdefault is not None:
            d = _coerce(param.pdefault, ptype)
            if d in vals:
                return d
        return vals[0]
    if fmt == "Corner":
        return _coerce(param.pvalue[0], ptype)
    if param.pdefault is not None and fmt == "Range":
        return _coerce(param.pdefault, ptype)
    return _coerce(param.pvalue, ptype) if not isinstance(param.pvalue, list) \
        else param.pvalue


class AmiParams:
    """The parameter tree of one ``.ami`` file (parsed by PyAMI) plus the
    values to pass to ``AMI_Init``.

    Values are addressed by leaf name (``params["dfe_ntaps"] = 8``) or by a
    dotted path (``"dfe.dfe_ntaps"``) when a leaf name is ambiguous.
    """

    def __init__(self, ami_text: str):
        _pyami()
        from pyibisami.ami.parser import AMIParamConfigurator

        self.cfg = AMIParamConfigurator(ami_text)
        self.root = self.cfg.root_name
        # PyAMI renames integer node names (tap positions): "-1" -> "pre1",
        # "0"/"+1" -> "post0"/"post1". Models expect the file's own names.
        self._numeric = set(re.findall(r"\(\s*([-+]?\d+)\s", ami_text))
        self.description = getattr(self.cfg, "_description", "") or ""
        self.reserved: dict[str, Any] = dict(self.cfg.info_ami_params)
        self.specific: dict[str, Any] = dict(self.cfg.model_specific_dict)
        self._leaves: dict[tuple[str, ...], Any] = {}
        self._walk(self.specific, ())
        self.values: dict[tuple[str, ...], Any] = {
            p: _default_value(prm) for p, prm in self._leaves.items()
            if prm.pusage in ("In", "InOut")}
        for name, prm in self.reserved.items():
            if prm.pusage in ("In", "InOut"):
                self.values[(name,)] = _default_value(prm)

    def _walk(self, tree: dict, prefix: tuple[str, ...]) -> None:
        from pyibisami.ami.parameter import AMIParameter

        for name, node in tree.items():
            if name == "description":
                continue
            if isinstance(node, AMIParameter):
                self._leaves[prefix + (name,)] = node
            elif isinstance(node, dict):
                self._walk(node, prefix + (name,))

    # -- lookup ------------------------------------------------------------
    def _path(self, key: str) -> tuple[str, ...]:
        from pyibisami.ami.parser import int2tap

        parts = tuple(int2tap(k) if re.fullmatch(r"[-+]?\d+", k) else k
                      for k in key.split("."))
        if parts in self.values:
            return parts
        hits = [p for p in self.values if p[-len(parts):] == parts]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"{self.root}: no settable parameter {key!r} "
                           f"(have {', '.join('.'.join(p) for p in self.values)})")
        raise KeyError(f"{self.root}: {key!r} is ambiguous: "
                       + ", ".join(".".join(p) for p in hits))

    def definition(self, key: str) -> Any:
        """The PyAMI ``AMIParameter`` behind a settable value."""
        path = self._path(key)
        return self._leaves.get(path) or self.reserved[path[0]]

    def __getitem__(self, key: str) -> Any:
        return self.values[self._path(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        path = self._path(key)
        prm = self.definition(key)
        v = _coerce(value, prm.ptype)
        if prm.pformat == "Range" and prm.pmin is not None:
            if not (float(prm.pmin) <= float(v) <= float(prm.pmax)):
                raise ValueError(f"{self.root}.{key} = {v} is outside its "
                                 f"Range [{prm.pmin}, {prm.pmax}]")
        if prm.pformat == "List":
            legal = [_coerce(x, prm.ptype) for x in prm.pvalue]
            if v not in legal:
                raise ValueError(f"{self.root}.{key} = {v!r} is not in its "
                                 f"List {legal}")
        self.values[path] = v

    def update(self, values: dict[str, Any] | None = None, **kw: Any) -> "AmiParams":
        for k, v in {**(values or {}), **kw}.items():
            self[k] = v
        return self

    def __contains__(self, key: str) -> bool:
        try:
            self._path(key)
            return True
        except KeyError:
            return False

    def info(self, name: str, default: Any = None) -> Any:
        """A reserved parameter's value (``Ignore_Bits``, ``Rx_Noise``, ...)."""
        prm = self.reserved.get(name)
        if prm is None:
            return default
        if (name,) in self.values:
            return self.values[(name,)]
        try:
            return _default_value(prm)
        except (TypeError, ValueError):
            return default

    # -- serialization -----------------------------------------------------
    def sexpr(self) -> str:
        """The ``AMI_parameters_in`` string: ``(root (name value) ...)``."""
        def atom(v: Any) -> str:
            if isinstance(v, bool):
                return "True" if v else "False"
            if isinstance(v, str):
                return f'"{v}"'
            if isinstance(v, float):
                return repr(v)
            if isinstance(v, (list, tuple)):
                return " ".join(atom(x) for x in v)
            return str(v)

        tree: dict = {}
        for path, v in self.values.items():
            node = tree
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = v

        def emit(name: str, node: Any) -> str:
            name = self.file_name(name)
            if isinstance(node, dict):
                return f"({name} " + " ".join(emit(k, x) for k, x in node.items()) + ")"
            return f"({name} {atom(node)})"

        return f"({self.root} " + " ".join(emit(k, x) for k, x in tree.items()) + ")"

    def file_name(self, name: str) -> str:
        """A node's name as written in the .ami file (undoes PyAMI's
        pre<N>/post<N> renaming of integer tap names)."""
        m = re.fullmatch(r"(pre|post)(\d+)", name)
        if not m:
            return name
        num = ("-" if m.group(1) == "pre" else "") + m.group(2)
        for cand in (num, "+" + num if m.group(1) == "post" else num):
            if cand in self._numeric:
                return cand
        return name

    def rows(self) -> list[dict]:
        """Table of every settable parameter (for printing / UIs)."""
        out = []
        for path, v in self.values.items():
            prm = self._leaves.get(path) or self.reserved[path[0]]
            row = {"name": ".".join(path), "value": v, "type": prm.ptype,
                   "format": prm.pformat, "usage": prm.pusage,
                   "description": str(prm.pdescription).strip('"')}
            if prm.pformat == "Range":
                row["min"], row["max"] = prm.pmin, prm.pmax
            elif prm.pformat == "List":
                row["list"] = [_coerce(x, prm.ptype) for x in prm.pvalue]
                if prm.plist_tip:
                    row["tips"] = list(prm.plist_tip)
            out.append(row)
        return out


def parse_params(text: str | bytes | None) -> dict:
    """``AMI_parameters_out`` -> nested dict, via PyAMI's AMI grammar.

    ``(pf_rx (level 0.2) (dfe_taps 0.3 0.1))`` -> ``{"level": 0.2,
    "dfe_taps": [0.3, 0.1]}``. Unparseable strings come back as
    ``{"_raw": text}`` (vendor models print anything there).
    """
    if not text:
        return {}
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    _pyami()
    from pyibisami.ami.parser import ami_parse

    def val(x: Any) -> Any:
        if isinstance(x, tuple):
            return conv(x[1])
        if isinstance(x, bool):
            return x
        s = str(x)
        if s.startswith('"'):
            return s.strip('"')
        try:
            f = float(s)
            return int(f) if s.lstrip("+-").isdigit() else f
        except ValueError:
            return s

    def conv(values: list) -> Any:
        if values and all(isinstance(v, tuple) for v in values):
            return {v[0]: val(v) for v in values}
        vs = [val(v) for v in values]
        return vs[0] if len(vs) == 1 else vs

    try:
        _root, nodes = ami_parse(text.strip())
        return {n[0]: conv(n[1]) for n in nodes}
    except Exception:  # noqa: BLE001 - vendor output is free-form
        return {"_raw": text}


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------

_D = ctypes.c_double
_PD = ctypes.POINTER(ctypes.c_double)


class _CtypesExecutor:
    """In-process: the library is dlopen'ed into this interpreter."""

    where = "in-process"

    def __init__(self, dll: Path):
        try:
            self.lib = ctypes.CDLL(str(dll))
        except OSError as exc:
            raise AmiError(f"cannot load {dll}: {exc}") from exc
        self.lib.AMI_Init.restype = ctypes.c_long
        self.lib.AMI_Init.argtypes = [
            _PD, ctypes.c_long, ctypes.c_long, _D, _D, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p)]
        self.lib.AMI_Close.restype = ctypes.c_long
        self.lib.AMI_Close.argtypes = [ctypes.c_void_p]
        gw = getattr(self.lib, "AMI_GetWave", None)
        if gw is not None:
            gw.restype = ctypes.c_long
            gw.argtypes = [_PD, ctypes.c_long, _PD,
                           ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p]
        self._gw = gw
        self.handle = ctypes.c_void_p()
        self._params_in = b""

    @property
    def has_getwave(self) -> bool:
        return self._gw is not None

    def init(self, matrix: np.ndarray, ts: float, ui: float,
             params: bytes) -> tuple[int, np.ndarray, str, str]:
        buf = np.ascontiguousarray(matrix, dtype=np.float64).copy()
        rows, n = buf.shape
        self._params_in = ctypes.create_string_buffer(params)  # model may write
        po, msg = ctypes.c_char_p(), ctypes.c_char_p()
        ret = self.lib.AMI_Init(buf.ctypes.data_as(_PD), n, rows - 1, ts, ui,
                                ctypes.cast(self._params_in, ctypes.c_char_p),
                                ctypes.byref(po), ctypes.byref(self.handle),
                                ctypes.byref(msg))
        dec = (lambda s: s.decode("utf-8", "replace") if s else "")
        return int(ret), buf, dec(po.value), dec(msg.value)

    def getwave(self, wave: np.ndarray, nclk: int) -> tuple[int, np.ndarray,
                                                             np.ndarray, str]:
        w = np.ascontiguousarray(wave, dtype=np.float64).copy()
        clk = np.full(nclk, -1.0)
        po = ctypes.c_char_p()
        ret = self._gw(w.ctypes.data_as(_PD), len(w), clk.ctypes.data_as(_PD),
                       ctypes.byref(po), self.handle)
        return int(ret), w, _clock_list(clk), \
            (po.value.decode("utf-8", "replace") if po.value else "")

    def close(self) -> None:
        if self.handle.value:
            self.lib.AMI_Close(self.handle)
            self.handle = ctypes.c_void_p()


def _clock_list(clk: np.ndarray) -> np.ndarray:
    end = np.flatnonzero(clk == -1.0)
    return clk[: end[0]] if len(end) else clk


class _HostedExecutor:
    """Out-of-process: ``ami_host`` loads the library and serves AMI calls
    over stdin/stdout (protocol in models/ibis_ami/host/ami_host.c)."""

    MAGIC = b"AMIH"

    def __init__(self, argv: list[str], where: str, timeout: float | None = None):
        self.where = where
        # a vendor model can hang (e.g. a parser looping on an unexpected
        # parameter string); bound every reply instead of blocking forever
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("PHOTONFLUX_AMI_TIMEOUT", "600"))
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        head = self._read(4)
        if head != self.MAGIC:
            err = self.proc.stderr.read().decode("utf-8", "replace") if head == b"" else ""
            msg = self._read_str() if head == b"ERR!" else err.strip()
            self.proc.kill()
            raise AmiError(f"ami_host ({where}) failed to start: {msg or head!r}")
        _ver, _long, gw = struct.unpack("<III", self._read(12))
        self.has_getwave = bool(gw)

    def _read(self, n: int) -> bytes:
        fd = self.proc.stdout.fileno()
        out = b""
        while len(out) < n:
            ready, _, _ = select.select([fd], [], [], self.timeout)
            if not ready:
                self.proc.kill()
                raise AmiError(f"ami_host ({self.where}): no reply in "
                               f"{self.timeout:g} s (model hung?)")
            chunk = os.read(fd, n - len(out))
            if not chunk:
                return out
            out += chunk
        return out

    def _read_str(self) -> str:
        (n,) = struct.unpack("<Q", self._read(8))
        return self._read(n).decode("utf-8", "replace")

    def _send(self, *parts: bytes) -> None:
        try:
            self.proc.stdin.write(b"".join(parts))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise AmiError(f"ami_host ({self.where}) died: "
                           + self.proc.stderr.read().decode("utf-8", "replace")) from exc

    def _check(self, data: bytes, n: int) -> bytes:
        if len(data) < n:
            raise AmiError(f"ami_host ({self.where}) died mid-call: "
                           + self.proc.stderr.read().decode("utf-8", "replace"))
        return data

    def init(self, matrix: np.ndarray, ts: float, ui: float, params: bytes):
        buf = np.ascontiguousarray(matrix, dtype="<f8")
        rows, n = buf.shape
        self._send(b"I", struct.pack("<QQddQ", n, rows - 1, ts, ui, len(params)),
                   params, buf.tobytes())
        (ret,) = struct.unpack("<q", self._check(self._read(8), 8))
        out = np.frombuffer(self._check(self._read(buf.nbytes), buf.nbytes),
                            dtype="<f8").reshape(rows, n).copy()
        return int(ret), out, self._read_str(), self._read_str()

    def getwave(self, wave: np.ndarray, nclk: int):
        w = np.ascontiguousarray(wave, dtype="<f8")
        self._send(b"G", struct.pack("<QQ", len(w), nclk), w.tobytes())
        (ret,) = struct.unpack("<q", self._check(self._read(8), 8))
        out = np.frombuffer(self._check(self._read(w.nbytes), w.nbytes),
                            dtype="<f8").copy()
        (nc,) = struct.unpack("<Q", self._read(8))
        clk = np.frombuffer(self._check(self._read(8 * nc), 8 * nc), dtype="<f8").copy()
        return int(ret), out, clk, self._read_str()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send(b"C")
                self._read(8)
            except AmiError:
                pass
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class InitResult:
    impulse: np.ndarray            # rows x n: victim first, then aggressors
    params_out: dict
    params_out_raw: str
    message: str


@dataclass
class GetWaveResult:
    wave: np.ndarray
    clock_times: np.ndarray        # model clock ticks (sampling - UI/2)
    trace_t: np.ndarray            # end time of each GetWave call
    trace: list[dict]              # parse_params(AMI_parameters_out) per call
    trace_raw: list[str] = field(default_factory=list)

    def series(self, key: str, index: int | None = None) -> np.ndarray:
        """One adaptation variable vs time, e.g. ``series("dfe_taps", 0)``."""
        out = []
        for d in self.trace:
            v = d.get(key, np.nan)
            if index is not None:
                v = v[index] if isinstance(v, list) and index < len(v) else \
                    (v if index == 0 and not isinstance(v, list) else np.nan)
            out.append(float(v) if not isinstance(v, (list, dict, str)) else np.nan)
        return np.asarray(out)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def _host_binary(platform_: Platform | None = None) -> Path:
    from . import ami_build

    return ami_build.ami_host(platform_)


class AmiModel:
    """One IBIS-AMI model instance (Tx or Rx).

    ``host`` picks where the library runs: ``"auto"`` (in-process when the
    platform matches, else a container), ``"inproc"``, ``"process"``
    (isolated local subprocess) or ``"container"`` (Docker, the library's
    Linux platform).
    """

    def __init__(self, dll: str | Path, ami: str | Path, *, kind: str | None = None,
                 name: str | None = None, host: str = "auto",
                 ibis_model: Any = None):
        self.dll = Path(dll).resolve()
        self.ami_file = Path(ami).resolve()
        if not self.dll.exists():
            raise AmiError(f"{self.dll} not found (build it with "
                           "`python -m photonflux ami build`?)")
        self.params = AmiParams(self.ami_file.read_text(errors="replace"))
        self.name = name or self.params.root
        self.kind = kind or _guess_kind(self.params, self.dll)
        self.ibis_model = ibis_model
        self.platforms = binary_platform(self.dll)
        self.host = host
        self._exec: Any = None
        self.ts: float | None = None
        self.ui: float | None = None
        self.last_init: InitResult | None = None

    def __repr__(self) -> str:
        plats = ", ".join(map(str, self.platforms)) or "unknown platform"
        return f"<AmiModel {self.name} ({self.kind}) {self.dll.name} [{plats}]>"

    # -- reserved-parameter shortcuts ---------------------------------------
    @property
    def getwave_exists(self) -> bool:
        return bool(self.params.info("GetWave_Exists", False))

    @property
    def init_returns_impulse(self) -> bool:
        return bool(self.params.info("Init_Returns_Impulse", True))

    @property
    def ignore_bits(self) -> int:
        return int(self.params.info("Ignore_Bits", 0) or 0)

    # -- lifecycle ----------------------------------------------------------
    def _executor(self):
        if self._exec is not None:
            return self._exec
        here = host_platform()
        mode = self.host
        native = any(p == here for p in self.platforms)
        if mode == "auto":
            mode = "inproc" if native else "container"
        if mode == "inproc":
            if not native:
                raise AmiError(
                    f"{self.dll.name} is {', '.join(map(str, self.platforms)) or 'not a library'}; "
                    f"this Python is {here}. Use host='container' for Linux "
                    "builds, or a native build from source.")
            self._exec = _CtypesExecutor(self.dll)
        elif mode == "process":
            if not native:
                raise AmiError(f"host='process' needs a {here} build of {self.dll.name}")
            self._exec = _HostedExecutor([str(_host_binary()), str(self.dll)],
                                         "local subprocess")
        elif mode == "container":
            plat = next((p for p in self.platforms if p.container), None)
            if plat is None:
                raise AmiError(
                    f"{self.dll.name} ({', '.join(map(str, self.platforms))}) "
                    "has no container platform; Windows DLLs need a Windows "
                    "Python")
            from . import ami_build

            self._exec = _HostedExecutor(ami_build.container_argv(plat, self.dll),
                                         f"container {plat.container}")
        else:
            raise ValueError(f"unknown host {self.host!r}")
        return self._exec

    @property
    def where(self) -> str:
        return self._executor().where

    def close(self) -> None:
        if self._exec is not None:
            self._exec.close()
            self._exec = None

    def __enter__(self) -> "AmiModel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- AMI calls ----------------------------------------------------------
    def init(self, impulse: np.ndarray, ts: float, ui: float, *,
             aggressors: Iterable[np.ndarray] = (),
             params: dict[str, Any] | None = None) -> InitResult:
        """``AMI_Init`` on a channel impulse response (V/s samples at ``ts``).

        Returns the model's output impulse matrix (meaningful when
        ``Init_Returns_Impulse``) and its parsed ``AMI_parameters_out``.
        Calling ``init`` again re-initializes (the previous instance is
        closed), which is how parameter sweeps work.
        """
        if params:
            self.params.update(params)
        if self.ts is not None:           # re-init: drop the old instance
            if isinstance(self._exec, _CtypesExecutor):
                self._exec.close()        # AMI_Close; the library stays loaded
            else:
                self.close()              # hosted: one model per process
        rows = [np.asarray(impulse, float)] + [np.asarray(a, float) for a in aggressors]
        n = max(len(r) for r in rows)
        mat = np.zeros((len(rows), n))
        for i, r in enumerate(rows):
            mat[i, : len(r)] = r
        ex = self._executor()
        ret, out, po, msg = ex.init(mat, float(ts), float(ui),
                                    self.params.sexpr().encode())
        if ret == 0:
            raise AmiError(f"{self.name}: AMI_Init failed: {msg}")
        self.ts, self.ui = float(ts), float(ui)
        self.last_init = InitResult(out, parse_params(po), po, msg)
        return self.last_init

    def getwave(self, wave: np.ndarray, *, block_bits: int = 256) -> GetWaveResult:
        """``AMI_GetWave`` over a whole waveform in blocks of ``block_bits``
        UI, after :meth:`init`. Model state carries across blocks."""
        if self.ts is None:
            raise AmiError(f"{self.name}: call init() before getwave()")
        ex = self._executor()
        if not ex.has_getwave:
            raise AmiError(f"{self.name}: the library has no AMI_GetWave")
        nspui = max(1, int(round(self.ui / self.ts)))
        blk = max(1, int(block_bits)) * nspui
        wave = np.asarray(wave, float)
        outs, clks, tr_t, tr, raw = [], [], [], [], []
        for i in range(0, len(wave), blk):
            seg = wave[i:i + blk]
            ret, w, c, po = ex.getwave(seg, len(seg) // nspui + 16)
            if ret == 0:
                raise AmiError(f"{self.name}: AMI_GetWave failed at sample {i}")
            outs.append(w)
            clks.append(c)
            tr_t.append((i + len(seg)) * self.ts)
            tr.append(parse_params(po))
            raw.append(po)
        return GetWaveResult(np.concatenate(outs) if outs else np.zeros(0),
                             np.concatenate(clks) if clks else np.zeros(0),
                             np.asarray(tr_t), tr, raw)

    # -- PyAMI interop ------------------------------------------------------
    def pyami_model(self, impulse: np.ndarray, ts: float, ui: float):
        """(PyAMI ``AMIModel``, ``AMIModelInitializer``) for this model with
        the current parameter values — for PyAMI's own tools, e.g.
        ``m.initialize(init); m.get_responses()``. In-process only."""
        from pyibisami.ami.model import AMIModel, AMIModelInitializer

        params = {"root_name": self.params.root}
        for path, v in self.params.values.items():
            node = params
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = v
        init = AMIModelInitializer(params, info_params=self.params.reserved,
                                   bit_time=ctypes.c_double(ui),
                                   sample_interval=ctypes.c_double(ts),
                                   row_size=len(impulse))
        init.channel_response = list(map(float, impulse))
        return AMIModel(str(self.dll)), init


def _guess_kind(params: AmiParams, dll: Path) -> str:
    text = (params.root + " " + dll.name + " " + params.description).lower()
    if "_rx" in text or "receiver" in text or text.startswith("rx"):
        return "rx"
    return "tx"


# ---------------------------------------------------------------------------
# model library
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    """One IBIS-AMI executable found on disk."""

    name: str              # library key, e.g. "pf_rx", "sisoft_tx@linux-x86"
    kind: str              # "tx" | "rx"
    dll: Path
    ami: Path
    ibs: Path | None
    source: str            # "photonflux" | "public" | "vendor"
    platforms: list[Platform]
    note: str = ""

    @property
    def runs_here(self) -> bool:
        return host_platform() in self.platforms

    @property
    def runs_in_container(self) -> bool:
        return any(p.container for p in self.platforms)

    def load(self, host: str = "auto") -> AmiModel:
        return AmiModel(self.dll, self.ami, kind=self.kind, name=self.name,
                        host=host)


def _ibis_entries(ibs: Path, source: str, prefix: str = "") -> list[LibraryEntry]:
    """Every [Algorithmic Model] executable of an .ibs file that exists."""
    _pyami()
    import logging

    from pyibisami.ibis.parser import parse_ibis_file

    logging.getLogger("pyibisami").setLevel(logging.ERROR)
    _status, d = parse_ibis_file(ibs.read_text(errors="replace"))
    out = []
    for mname, model in (d.get("models") or {}).items():
        kind = "tx" if str(model.mtype).lower() in ("output", "i/o") else "rx"
        for (_os, _bits), files in model.executables:
            dll, ami = ibs.parent / files[0], ibs.parent / files[1]
            if not dll.exists() or not ami.exists():
                continue
            plats = binary_platform(dll)
            key = f"{prefix}{mname}"
            if any(e.dll == dll for e in out):
                continue
            out.append(LibraryEntry(key, kind, dll, ami, ibs, source, plats))
    return out


def library(*, include_vendor: bool = True) -> list[LibraryEntry]:
    """Discover every IBIS-AMI model available to this checkout.

    Host builds come first (``models/ibis_ami/build/<os-arch>``: the
    photonflux models and native builds of the public sources), then the
    public kits' original binaries, then ``examples/ibis_ami/vendor``. Names
    are unique: a second binary of the same model gets ``@<os-arch>``.
    """
    here = host_platform()
    # (directory, source, name prefix): public kits are prefixed by kit
    # ("sisoft.IBIS_AMI_Tx"), vendor drops by their folder
    roots: list[tuple[Path, str, str]] = [(BUILD_ROOT / here.tag, "photonflux", "")]
    pub = BUILD_ROOT / "public"
    if pub.is_dir():
        # host builds before original binaries, so the native one gets the
        # plain name and the foreign one the @os-arch suffix
        dirs = sorted((p for p in pub.iterdir() if p.is_dir()),
                      key=lambda p: (not p.name.endswith(here.tag), p.name))
        roots += [(p, "public", p.name.split("-")[0] + ".") for p in dirs]
    if include_vendor and VENDOR.is_dir():
        roots += [(p, "vendor", f"vendor.{p.name}.")
                  for p in sorted(VENDOR.iterdir()) if p.is_dir()]
    entries: list[LibraryEntry] = []
    seen: set[Path] = set()
    for root, source, prefix in roots:
        if not root.is_dir():
            continue
        for ibs in sorted(root.rglob("*.ibs")):
            for e in _ibis_entries(ibs, source, prefix):
                if e.dll in seen:
                    continue
                seen.add(e.dll)
                if any(x.name == e.name for x in entries):
                    tag = e.platforms[0].tag if e.platforms else "unknown"
                    e.name = f"{e.name}@{tag}"
                entries.append(e)
    return entries


def load(name: str, *, host: str = "auto", build: bool = True) -> AmiModel:
    """Load a library model by name (see :func:`library`), building the
    photonflux models first if they are missing."""
    entries = library()
    hit = next((e for e in entries if e.name == name), None)
    if hit is None and build:
        from . import ami_build

        ami_build.build_all(quiet=True)
        entries = library()
        hit = next((e for e in entries if e.name == name), None)
    if hit is None:
        names = ", ".join(e.name for e in entries) or "(none built yet)"
        raise AmiError(f"no IBIS-AMI model {name!r}; available: {names}")
    return hit.load(host)


def docker_available() -> bool:
    """True when a Docker daemon answers (container-hosted models)."""
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, timeout=20, text=True)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())

