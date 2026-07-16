"""Command-line entry point: `python -m photonflux <cmd>` or `photonflux <cmd>`.

  doctor      check every toolchain prerequisite (openvaf-ir, VA includes,
              libngspice for PDK card extraction, SKY130 PDK)
"""
from __future__ import annotations

import argparse
import sys


def cmd_doctor(_args) -> int:
    from .toolchain import doctor, doctor_report

    print(doctor_report())
    return 0 if all(c.ok for c in doctor() if "PDK" not in c.name) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="photonflux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check toolchain prerequisites")
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
