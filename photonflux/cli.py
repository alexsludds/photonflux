"""Command-line entry point: `python -m photonflux <cmd>` or `photonflux <cmd>`.

  doctor      check every toolchain prerequisite (openvaf-ir, VA includes,
              libngspice for PDK card extraction, SKY130 PDK)
  flow        run a testbench's flow (doc["flow"]) headless and report its
              checks; exits 1 when a check fails or a step errors, so saved
              testbenches double as regression tests
"""
from __future__ import annotations

import argparse
import sys


def cmd_doctor(_args) -> int:
    from .toolchain import doctor, doctor_report

    print(doctor_report())
    return 0 if all(c.ok for c in doctor() if "PDK" not in c.name) else 1


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.6g}"


def cmd_flow(args) -> int:
    """Run a saved testbench's flow in-process through the webapp engine."""
    import json
    from pathlib import Path

    webapp = Path(__file__).resolve().parents[1] / "webapp"
    if not (webapp / "flow.py").exists():
        print("photonflux flow needs a source checkout (webapp/ next to the "
              "photonflux package)", file=sys.stderr)
        return 2
    sys.path.insert(0, str(webapp))
    import jax

    jax.config.update("jax_enable_x64", True)
    import simulate

    doc = json.loads(Path(args.testbench).read_text())
    if not doc.get("flow"):
        print(f"{args.testbench}: no flow in this testbench", file=sys.stderr)
        return 2
    analysis = {"mode": "flow", "flow": doc["flow"]}
    if args.only:
        analysis["only"] = args.only
    res = simulate.run({"schematic": doc.get("schematic") or doc,
                        "analysis": analysis})
    if args.json:
        Path(args.json).write_text(json.dumps(res))
    if not res.get("ok"):
        print(f"flow failed: {res.get('error')}", file=sys.stderr)
        return 1
    fr = res["flow"]
    print(f"{doc.get('title') or args.testbench}: {doc['flow'].get('name', '')}")
    for i, s in enumerate(fr["steps"], 1):
        t = f"{s.get('time_s', 0):6.1f}s" if "time_s" in s else "       "
        print(f"  {i}. {s['title']:<34} {s['status']:<8} {t}"
              + (f"  {s['error']}" if s.get("error") else ""))
        for c in s.get("checks") or []:
            print(f"       {'pass' if c['ok'] else 'FAIL'}  {c['expr']:<24} "
                  f"{_fmt(c.get('value'))}")
    print("  variables: " + ", ".join(f"{k}={_fmt(v)}"
                                      for k, v in fr["vars"].items()))
    print("PASS" if fr["passed"] else "FAIL")
    return 0 if fr["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="photonflux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check toolchain prerequisites")
    fp = sub.add_parser("flow", help="run a testbench flow and check it")
    fp.add_argument("testbench", help="testbench JSON with a flow")
    fp.add_argument("--only", metavar="STEP",
                    help="run one step (no prior variables: use for a first "
                         "step or a self-contained one)")
    fp.add_argument("--json", metavar="OUT", help="write the full result here")
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "flow": cmd_flow}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
