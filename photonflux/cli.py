"""Command-line entry point: `python -m photonflux <cmd>` or `photonflux <cmd>`.

  doctor      check every toolchain prerequisite (openvaf-ir, VA includes,
              libngspice for PDK card extraction, SKY130 PDK)
  flow        run a testbench's flow (doc["flow"]) headless and report its
              checks; exits 1 when a check fails or a step errors, so saved
              testbenches double as regression tests
  ami         IBIS-AMI models: `build` the photonflux / public models for
              this machine, `list` every model and where it can run, `info`
              one model's parameters, `host-image` a Linux container host
"""
from __future__ import annotations

import argparse
import sys


def cmd_doctor(_args) -> int:
    from .toolchain import doctor, doctor_report

    print(doctor_report())
    return 0 if all(c.ok for c in doctor() if not c.optional) else 1


def _finite(obj):
    """JSON-safe copy: NaN / inf -> None (bare NaN is not valid JSON)."""
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


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
    from progress import PROGRESS, printer

    if not args.quiet:    # live log on stderr: steps, evaluations, checks
        PROGRESS.flow_echo = printer(sys.stderr)
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
        Path(args.json).write_text(json.dumps(_finite(res), allow_nan=False))
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
            verdict = "pass" if c["ok"] else ("warn" if c.get("info") else "FAIL")
            print(f"       {verdict}  {c['expr']:<24} {_fmt(c.get('value'))}"
                  + ("  (optimizer constraint)" if c.get("info") else ""))
    print("  variables: " + ", ".join(f"{k}={_fmt(v)}"
                                      for k, v in fr["vars"].items()))
    print("PASS" if fr["passed"] else "FAIL")
    return 0 if fr["passed"] else 1


def cmd_ami(args) -> int:
    from . import ami_build, ibisami

    if args.ami_cmd == "build":
        steps = ami_build.build_all(force=args.force)
        return 0 if steps[0].ok else 1
    if args.ami_cmd == "list":
        docker = ibisami.docker_available()
        lib = ibisami.library()
        if not lib:
            print("no IBIS-AMI models built yet: python -m photonflux ami build")
            return 1
        w = max(len(e.name) for e in lib)
        for e in lib:
            how = ("in-process" if e.runs_here else
                   "container" + ("" if docker else " (no Docker daemon)")
                   if e.runs_in_container else "not on this machine")
            plats = ", ".join(map(str, e.platforms)) or "?"
            print(f"  {e.name:<{w}}  {e.kind}  {e.source:<10}  {plats:<24}  {how}")
        return 0
    if args.ami_cmd == "info":
        m = ibisami.load(args.name)
        print(m)
        print(f"  {m.params.description}")
        print(f"  GetWave_Exists {m.getwave_exists}, Init_Returns_Impulse "
              f"{m.init_returns_impulse}, Ignore_Bits {m.ignore_bits}")
        for r in m.params.rows():
            lim = r.get("list") or (f"[{r['min']}, {r['max']}]" if "min" in r else "")
            print(f"    {r['name']:<28} {r['value']!s:<14} {r['type']:<8} {lim!s:<22} "
                  f"{r['description'][:60]}")
        print(f"  AMI_parameters_in: {m.params.sexpr()}")
        return 0
    if args.ami_cmd == "host-image":
        plat = {"linux/amd64": ibisami.Platform("linux", "x86_64"),
                "linux/386": ibisami.Platform("linux", "x86"),
                "linux/arm64": ibisami.Platform("linux", "arm64")}[args.platform]
        print(ami_build.ensure_host_image(plat, quiet=False))
        return 0
    return 2


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
    fp.add_argument("-q", "--quiet", action="store_true",
                    help="no live log on stderr while the flow runs")
    amp = sub.add_parser("ami", help="IBIS-AMI models (build / list / info)")
    asub = amp.add_subparsers(dest="ami_cmd", required=True)
    b = asub.add_parser("build", help="compile the IBIS-AMI models for this host")
    b.add_argument("--force", action="store_true", help="rebuild everything")
    asub.add_parser("list", help="every IBIS-AMI model found and where it runs")
    i = asub.add_parser("info", help="one model's parameters")
    i.add_argument("name", help="library name, e.g. pf_rx (see `ami list`)")
    h = asub.add_parser("host-image", help="build the Docker host for Linux models")
    h.add_argument("platform", choices=["linux/amd64", "linux/386", "linux/arm64"])
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "flow": cmd_flow, "ami": cmd_ami}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
