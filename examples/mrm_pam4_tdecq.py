#!/usr/bin/env python3
"""PAM-4 microring transmitter, scored by TDECQ through the reference FFE.

A PAM-4 PRBS-13Q drive (Gray-coded, rotated so its OMA_outer runs survive the
record edges) steps ``models/optical_field/ring_mod.va`` through four
resonance positions via a driver with source resistance ``--r-drv``. The
through-port power goes through the TDECQ reference receiver (4th-order
Bessel-Thomson at baud/2) and into `stateye` (``photonflux.tdec.measure_pam4``)
twice:

    unequalized     ffe_taps=0
    reference FFE   5-tap T-spaced FFE (1 pre-cursor), MMSE-designed on the
                    known PRBS-13Q, noise enhancement C_eq fed into TDECQ

A ring is a Lorentzian, so equal drive steps give unequal optical steps: the
level-mismatch ratio RLM is reported too, and ``--rlm`` pre-distorts the four
drive levels so the *static* optical levels come out equally spaced.

    python examples/mrm_pam4_tdecq.py                   # 53.125 GBd, 100G/lane
    python examples/mrm_pam4_tdecq.py --baud 26.5625e9  # 50G/lane
    python examples/mrm_pam4_tdecq.py --rlm --detune -40

    -> out/mrm_pam4_tdecq.png (unequalized vs equalized eye)

Needs the patched stateye (``pip install -e '.[eye]'`` +
docs/patches/stateye-modern-toolchain.patch), which carries the PAM-4 / TDECQ
analysis and the reference equalizer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from circulax import compile_circuit  # noqa: E402
from circulax.components.base_component import Signals, States, source  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrm_tdec_sky130 import LinkSpec, base_models  # noqa: E402

from photonflux import tdec  # noqa: E402
from photonflux.signals import pam4_gray, prbs  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
ROTATE = 64          # PRBS-13Q's only run of 6 zeros is its last 6 symbols


def prbs13q() -> np.ndarray:
    """8191 Gray-coded PAM-4 symbols (0..3), rotated for OMA_outer."""
    return np.roll(pam4_gray(np.tile(prbs(13), 2)), ROTATE)


def level_source(values, t_sym, t_rise):
    """Multi-level pattern source with C1-continuous smoothstep edges."""
    padded = jnp.concatenate([jnp.asarray(values[:1]), jnp.asarray(values)])
    n = len(values)

    @source(ports=("p1", "p2"), states=("i_src",))
    def LevelPattern(signals: Signals, s: States, t: float):
        i = jnp.clip(jnp.floor(t / t_sym).astype(jnp.int32), 0, n - 1)
        prev, cur = padded[i], padded[i + 1]
        x = jnp.clip((t - i * t_sym) / t_rise, 0.0, 1.0)
        v = prev + (cur - prev) * x * x * (3.0 - 2.0 * x)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}

    return LevelPattern


def build(spec: LinkSpec, drive_v: np.ndarray, r_drv: float):
    """Laser -> ring -> photodiode, ring electrode driven through R_drv."""
    inst = {
        "GND": {"component": "ground"},
        "LAS": {"component": "laser",
                "settings": {"wavelength_nm": spec.lambda_laser_nm,
                             "power": spec.p_laser_w}},
        "TAP": {"component": "f2ri"},
        "RING": {"component": "ring", "settings": spec.ring_settings()},
        "JOIN": {"component": "ri2f"},
        "PD": {"component": "pd"},
        "RL": {"component": "res", "settings": {"R": spec.r_pd_load}},
        "VIN": {"component": "pam4"},
        "RDRV": {"component": "res", "settings": {"R": r_drv}},
    }
    conn = {
        "LAS,p1": "TAP,c", "TAP,re": "RING,in_re", "TAP,im": "RING,in_im",
        "RING,out_re": "JOIN,re", "RING,out_im": "JOIN,im",
        "JOIN,c": "PD,po_p", "PD,cat": "RL,p1",
        "VIN,p1": "RDRV,p1", "RDRV,p2": "RING,vp",
        "GND,p1": ("LAS,p2", "RING,vn", "RING,gnd", "PD,po_n", "PD,an",
                   "RL,p2", "VIN,p2"),
    }
    mdl = base_models()
    mdl["pam4"] = level_source(drive_v, 1.0 / spec.baud, spec.t_rise)
    net = {"instances": inst, "connections": conn,
           "ports": {"vring": "RING,vp", "prx": "PD,po_p"}}
    return compile_circuit(net, mdl, backend="dense", is_complex=True,
                           max_steps=300)


def simulate(spec: LinkSpec, drive_v: np.ndarray, r_drv: float):
    """Transient over the whole pattern -> (dt, through-port power [mW])."""
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    c = build(spec, drive_v, r_drv)
    t_sym = 1.0 / spec.baud
    dt = t_sym / spec.spu
    t_max = len(drive_v) * t_sym
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=dt, y0=c.dc(),
        saveat=diffrax.SaveAt(ts=jnp.arange(len(drive_v) * spec.spu) * dt),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=int(t_max / dt) + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    if sol.result != diffrax.RESULTS.successful:
        raise RuntimeError(f"transient failed: {sol.result}")
    return dt, np.asarray(jnp.abs(c.port(sol.ys, "prx")) ** 2).real * 1e3


def static_transmission(spec: LinkSpec, v: np.ndarray) -> np.ndarray:
    """Steady-state through-port power [mW] vs DC drive -- the ring's
    static transfer, for RLM pre-distortion."""
    out = []
    for vi in v:
        c = build(spec, np.array([vi]), 50.0)
        out.append(float(jnp.abs(c.port(c.dc(), "prx")) ** 2) * 1e3)
    return np.asarray(out)


def rlm_levels(spec: LinkSpec, v_swing: float) -> np.ndarray:
    """Four drive levels in [0, v_swing] whose static optical powers are
    equally spaced (the ring's Lorentzian makes equal volts unequal mW)."""
    v = np.linspace(0.0, v_swing, 41)
    p = static_transmission(spec, v)
    if not (np.all(np.diff(p) > 0) or np.all(np.diff(p) < 0)):
        raise SystemExit("ring transfer is not monotone over the swing -- "
                         "move the lock point (--detune) off resonance")
    targets = np.linspace(p[0], p[-1], 4)
    order = np.argsort(p)
    return np.interp(targets, p[order], v[order])


def rlm(m) -> float:
    """802.3 ratio of level mismatch from stateye's _xp level means."""
    lv = [m[f"{n}_level_xp"] for n in ("zero", "one", "two", "three")]
    mid = 0.5 * (lv[0] + lv[3])
    es1 = (lv[1] - mid) / (lv[0] - mid)
    es2 = (lv[2] - mid) / (lv[3] - mid)
    return float(min(3 * es1, 3 * es2, 2 - 3 * es1, 2 - 3 * es2))


def report(tag, m):
    print(f"  {tag:13s} TDECQ {m['tdecq_outer']:6.3f} dB   "
          f"OMA_outer {m['oma_outer']:.4f} mW   RLM {rlm(m):.3f}"
          + (f"   C_eq {m['ceq']:.3f}" if "ceq" in m else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baud", type=float, default=53.125e9)
    ap.add_argument("--detune", type=float, default=-33.0,
                    help="lock point: laser - resonance [pm]")
    ap.add_argument("--gap", type=float, default=200.0,
                    help="bus gap [nm] -> kappa2 (photonflux.coupler)")
    ap.add_argument("--swing", type=float, default=1.8,
                    help="drive swing, level 0 to level 3 [V]")
    ap.add_argument("--r-drv", type=float, default=50.0,
                    help="driver source resistance [ohm]")
    ap.add_argument("--rlm", action="store_true",
                    help="pre-distort the drive levels for equal optical steps")
    ap.add_argument("--spu", type=int, default=32, help="samples per symbol")
    ap.add_argument("--s-noise", type=float, default=0.005,
                    help="TDECQ S: O/E + scope noise std [mW]")
    ap.add_argument("--ser", type=float, default=4.8e-4,
                    help="target SER (4.8e-4: 802.3 100G/lane)")
    ap.add_argument("--ffe-taps", type=int, default=5)
    args = ap.parse_args()

    spec = LinkSpec(baud=args.baud, detune_pm=args.detune, spu=args.spu,
                    gap_nm=args.gap)

    sym = prbs13q()
    levels = (rlm_levels(spec, args.swing) if args.rlm
              else np.linspace(0.0, args.swing, 4))
    print(f"PAM-4 at {spec.baud/1e9:g} GBd, PRBS-13Q ({sym.size} symbols), "
          f"{spec.spu} samples/UI; ring gap {spec.gap_nm:.0f} nm "
          f"(kappa2 {spec.kappa2:.3f}), lock "
          f"{spec.detune_pm:+.0f} pm; drive levels "
          + "/".join(f"{v:.3f}" for v in levels) + " V"
          + (" (RLM pre-distorted)" if args.rlm else ""))

    t0 = time.time()
    dt, p = simulate(spec, levels[sym], args.r_drv)
    print(f"transient: {p.size} points in {time.time() - t0:.0f} s")

    kw = dict(s_noise_mW=args.s_noise, ser=args.ser, strict=False)
    raw = tdec.measure_pam4(p, dt, spec.baud, ffe_taps=0, **kw)
    eq = tdec.measure_pam4(p, dt, spec.baud, ffe_taps=args.ffe_taps,
                           symbols=sym, **kw)
    print("\nTDECQ (reference Rx: BT4 at baud/2, SER "
          f"{args.ser:g}, S = {args.s_noise} mW)")
    report("unequalized", raw)
    report(f"{args.ffe_taps}-tap FFE", eq)
    if "ffe_taps" in eq:
        print("  FFE taps      " + "  ".join(f"{c:+.3f}" for c in eq["ffe_taps"])
              + f"   (rms error {eq['ffe_rms_error_before']:.4f} -> "
              f"{eq['ffe_rms_error_after']:.4f} mW)")

    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = []
        for m in (raw, eq):
            figs.append(m.eye.plot(show=False))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for ax, f, (tag, m) in zip(axes, figs,
                                   (("unequalized", raw),
                                    (f"{args.ffe_taps}-tap FFE", eq))):
            f.canvas.draw()
            ax.imshow(np.asarray(f.canvas.buffer_rgba()))
            ax.set_axis_off()
            ax.set_title(f"{tag}: TDECQ {m['tdecq_outer']:.2f} dB")
            plt.close(f)
        path = OUT / "mrm_pam4_tdecq.png"
        fig.savefig(path, dpi=130)
        print(f"\nwrote {path}")
    except Exception as exc:  # eye plotting is a nicety, not the result
        print(f"\neye plot unavailable ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
