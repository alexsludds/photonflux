"""Command-line entry point: `python -m lightspice <cmd>` or `lightspice <cmd>`.

  doctor      check every toolchain prerequisite
  compile     compile models/*.va -> .osdi (content-hash cached)
  smoke       compile everything, load every OSDI into ngspice, run a
              one-device sanity sim
"""
from __future__ import annotations

import argparse
import sys


def cmd_doctor(_args) -> int:
    from .toolchain import doctor, doctor_report

    print(doctor_report())
    return 0 if all(c.ok for c in doctor() if "PDK" not in c.name) else 1


def cmd_compile(args) -> int:
    from .compiler import compile_all, compile_va

    if args.files:
        mods = [compile_va(f, force=args.force, quiet=False) for f in args.files]
    else:
        mods = compile_all(force=args.force, quiet=False)
    width = max(len(m.va.name) for m in mods)
    for m in mods:
        print(f"  {m.va.name:<{width}}  module {m.name:<12} -> {m.osdi.name}")
    return 0


def cmd_smoke(_args) -> int:
    from .compiler import compile_all
    from .engine import Engine

    mods = compile_all(quiet=False)
    eng = Engine()
    for m in mods:
        eng._ng.load_osdi(m.osdi)
        print(f"  loaded {m.osdi.name}  (module {m.name!r}, OSDI ok)")

    laser = next((m for m in mods if m.name == "laser_dml"), None)
    if laser:
        from .circuit import Circuit

        ckt = Circuit("smoke: laser_dml OP")
        ckt.raw("Vdrv drv 0 1.5")
        ckt.device(laser, "ld", "drv", "0", "popt", "0")
        r = eng.op(ckt)
        p = float(r["popt"][0])
        print(f"  laser_dml OP: P_opt = {p*1e3:.2f} mW (expect 15.00)")
        if abs(p - 0.015) > 1e-6:
            print("  SMOKE FAILED")
            return 1
    print("smoke OK: compiled Verilog-A runs inside ngspice end-to-end")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lightspice")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check toolchain prerequisites")
    pc = sub.add_parser("compile", help="compile Verilog-A models to OSDI")
    pc.add_argument("files", nargs="*")
    pc.add_argument("--force", action="store_true", help="ignore the cache")
    sub.add_parser("smoke", help="end-to-end OSDI sanity check")
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "compile": cmd_compile, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
