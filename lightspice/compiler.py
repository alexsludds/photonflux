"""Verilog-A -> OSDI compilation with content-hash caching and a guard
against the ngspice builtin-model-type name collision.

The collision guard exists because of a real, expensive footgun: a module
named `res` compiles fine, registers fine, but a `.model X res` card binds
to ngspice's *builtin* semiconductor-resistor type and elaboration dies
with "incorrect model type! Expected OSDI device". The module name (not
the file name) is what `.model` cards reference, so it must not shadow a
builtin type.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import toolchain

__all__ = ["VaModule", "compile_va", "compile_all", "ModelNameCollision"]

# .model type names the ngspice deck parser claims before consulting the
# OSDI registry. Empirically probed on ngspice 46 (compile a trivial OSDI
# module under each name, instantiate via .model + N-card): res/r/c/l/d/
# diode/sw/csw/urc/ltra/npn/nmos all bind to the builtin and die with
# "incorrect model type! Expected OSDI device", while cap/ind/psw resolve
# to the OSDI device. The rest of the list extends the confirmed names to
# their obvious family members.
NGSPICE_BUILTIN_MODEL_TYPES = {
    "r", "res", "c", "l", "d", "diode", "sw", "csw", "urc", "ltra",
    "txl", "cpl",
    "npn", "pnp", "lpnp", "njf", "pjf", "nmf", "pmf", "nhfet", "phfet",
    "nmos", "pmos", "vdmos", "numd", "nbjt", "numos", "ndev",
}


class ModelNameCollision(ValueError):
    pass


_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\s*[(\s]", re.M)
_PARAM_RE = re.compile(
    r"^\s*parameter\s+(?:real|integer)\s+([A-Za-z_][A-Za-z0-9_$]*)\s*=\s*([^;/]+?)\s*(?:from[^;]*)?;",
    re.M,
)


def _strip_va_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


@dataclass
class VaModule:
    """A compiled Verilog-A module: source, OSDI dylib, and metadata."""

    va: Path
    osdi: Path
    name: str                      # module name == ngspice .model type name
    params: dict[str, str] = field(default_factory=dict)

    def descriptor_name(self) -> str:
        """Ground truth: read the device name from the OSDI descriptor."""
        lib = ctypes.CDLL(str(self.osdi))
        ptr = ctypes.c_void_p.from_address(
            ctypes.addressof(ctypes.c_void_p.in_dll(lib, "OSDI_DESCRIPTORS"))
        ).value
        return ctypes.c_char_p(ptr).value.decode("utf-8")


def parse_va(va_path: Path) -> tuple[str, dict[str, str]]:
    text = _strip_va_comments(va_path.read_text())
    m = _MODULE_RE.search(text)
    if not m:
        raise ValueError(f"no `module` declaration found in {va_path}")
    params = {p: d.strip() for p, d in _PARAM_RE.findall(text)}
    return m.group(1), params


def check_module_name(name: str, va_path: Path | str = "?") -> None:
    if name.lower() in NGSPICE_BUILTIN_MODEL_TYPES:
        raise ModelNameCollision(
            f"Verilog-A module {name!r} (in {va_path}) collides with ngspice's "
            f"builtin .model type {name.lower()!r}. The deck parser will bind "
            f".model cards to the builtin and fail with 'incorrect model type! "
            f"Expected OSDI device'. Rename the module (e.g. {name}_va)."
        )


def _source_fingerprint(va_path: Path, include: Path) -> str:
    h = hashlib.sha256()
    h.update(va_path.read_bytes())
    for inc in sorted(include.glob("*.h")) + sorted(include.glob("*.vams")):
        h.update(inc.name.encode())
        h.update(inc.read_bytes())
    h.update(toolchain.openvaf_version().encode())
    return h.hexdigest()


def compile_va(
    va_path: str | Path,
    include: str | Path | None = None,
    force: bool = False,
    quiet: bool = True,
) -> VaModule:
    """Compile one .va to a sibling .osdi, skipping if the cache is fresh."""
    va = Path(va_path).resolve()
    include = Path(include) if include else toolchain.include_dir()
    osdi = va.with_suffix(".osdi")
    meta = va.with_suffix(".osdi.meta.json")

    name, params = parse_va(va)
    check_module_name(name, va)

    fp = _source_fingerprint(va, include)
    if not force and osdi.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()).get("fingerprint") == fp:
                return VaModule(va, osdi, name, params)
        except (json.JSONDecodeError, OSError):
            pass

    cmd = [str(toolchain.openvaf_path()), "-I", str(include), str(va)]
    proc = subprocess.run(cmd, cwd=va.parent, capture_output=True, text=True)
    if proc.returncode != 0 or not osdi.exists():
        raise RuntimeError(
            f"openvaf-r failed on {va.name}:\n{proc.stdout}\n{proc.stderr}"
        )
    if not quiet:
        print(f"compiled {va.name} -> {osdi.name}")

    mod = VaModule(va, osdi, name, params)
    desc = mod.descriptor_name()
    if desc != name:
        raise RuntimeError(
            f"{va.name}: parsed module name {name!r} but OSDI descriptor says "
            f"{desc!r}; fix the parser assumptions"
        )
    meta.write_text(json.dumps({"fingerprint": fp, "module": name}))
    return mod


def compile_all(
    models_dir: str | Path | None = None, force: bool = False, quiet: bool = True
) -> list[VaModule]:
    models_dir = Path(models_dir) if models_dir else toolchain.MODELS_DIR
    return [
        compile_va(va, force=force, quiet=quiet)
        for va in sorted(models_dir.glob("*.va"))
    ]
