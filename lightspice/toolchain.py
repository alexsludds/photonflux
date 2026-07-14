"""Toolchain discovery: openvaf-r, libngspice, include dir, SKY130 PDK.

Everything is overridable via environment variables:

  LIGHTSPICE_OPENVAF     path to the openvaf-r binary
  LIGHTSPICE_INCLUDE     Verilog-A include directory (discipline.h, ...)
  NGSPICE_LIBRARY_PATH   libngspice shared library
  SKY130_NGSPICE_LIB     sky130.lib.spice (wins over PDK_ROOT / volare)
  PDK_ROOT               open_pdks-style root containing sky130A/
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._ngspice import find_libngspice

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
OUT_DIR = REPO / "out"


def openvaf_path() -> Path:
    env = os.environ.get("LIGHTSPICE_OPENVAF")
    if env:
        return Path(env)
    return REPO / "bin" / "openvaf-r"


def include_dir() -> Path:
    env = os.environ.get("LIGHTSPICE_INCLUDE")
    if env:
        return Path(env)
    return REPO / "include"


def openvaf_version() -> str:
    out = subprocess.run(
        [str(openvaf_path()), "--version"], capture_output=True, text=True, timeout=30
    )
    return out.stdout.strip() or out.stderr.strip()


def sky130_lib() -> Path:
    """Locate the ngspice-ready sky130.lib.spice (volare layout by default)."""
    env = os.environ.get("SKY130_NGSPICE_LIB")
    if env and Path(env).exists():
        return Path(env)
    pdk_root = os.environ.get("PDK_ROOT")
    if pdk_root:
        cand = Path(pdk_root) / "sky130A/libs.tech/ngspice/sky130.lib.spice"
        if cand.exists():
            return cand
    home = Path.home() / ".volare"
    cand = home / "sky130A/libs.tech/ngspice/sky130.lib.spice"
    if cand.exists():
        return cand
    for p in home.glob("**/sky130A/libs.tech/ngspice/sky130.lib.spice"):
        return p
    raise FileNotFoundError(
        "sky130.lib.spice not found. Install with `python3 -m pip install volare` "
        "then `python3 -m volare enable --pdk sky130 <commit>`, or set "
        "SKY130_NGSPICE_LIB / PDK_ROOT."
    )


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def doctor() -> list[Check]:
    """Verify every prerequisite of the co-sim flow on this machine."""
    checks: list[Check] = []

    p = openvaf_path()
    if p.exists():
        try:
            checks.append(Check("openvaf-r", True, f"{p}  ({openvaf_version()})"))
        except Exception as e:  # binary exists but does not run
            checks.append(Check("openvaf-r", False, f"{p} exists but failed: {e}"))
    else:
        checks.append(Check("openvaf-r", False, f"missing {p}; build it (see README)"))

    inc = include_dir()
    have = {f.name for f in inc.glob("*.h")} if inc.is_dir() else set()
    ok = {"discipline.h", "constants.h"} <= have
    checks.append(Check("VA includes", ok, f"{inc}  ({', '.join(sorted(have)) or 'empty'})"))

    try:
        lib = find_libngspice()
        from ._ngspice import NgSpice

        ng = NgSpice.get(lib)
        ver = ng.cmd("version -s", check=False)
        banner = next((ln for ln in ver if "ngspice" in ln.lower()), "loaded")
        checks.append(Check("libngspice", True, f"{lib}  ({banner.strip('* ').strip()})"))
    except Exception as e:
        checks.append(Check("libngspice", False, str(e)))

    try:
        checks.append(Check("SKY130 PDK", True, str(sky130_lib())))
    except FileNotFoundError as e:
        checks.append(Check("SKY130 PDK", False, str(e).splitlines()[0] + " (optional)"))

    return checks


def doctor_report() -> str:
    checks = doctor()
    width = max(len(c.name) for c in checks)
    lines = [
        f"  {'OK ' if c.ok else 'FAIL':<5} {c.name:<{width}}  {c.detail}" for c in checks
    ]
    return "\n".join(lines)
