#!/usr/bin/env python3
"""Co-optimize a 53.125 GBd NRZ microring link for OMA - TDEC.

The transmitter is ``models/optical_field/ring_mod.va`` driven by a SKY130
CMOS inverter. Four design freedoms are searched:

    bus gap [nm]        -> kappa2, via photonflux.coupler (evanescent fit)
    lock point [pm]     -> laser detuning from the cold resonance
    inverter W_p, W_n   -> SKY130 pfet/nfet widths [um]
    inverter L          -> SKY130 channel length [um]

and scored with `stateye <https://github.com/AyarLabs/stateye>`_ through
``photonflux.tdec``: the optical through-port power goes through a 4th-order
Bessel-Thomson reference receiver, into an eye histogram, out as IEEE 802.3
OMA / TDEC.

**Maximise OMA - TDEC, not -TDEC.** TDEC alone has a degenerate optimum: park
the laser off resonance for a tiny, spotless, useless swing.

Cost structure (measured, see docs/stateye-integration-plan.md):

* a new SKY130 (W, L) pair costs 41-80 s to compile, then 0 s forever (cached)
* the optical parameters are free to vary -- no recompile
* full PRBS-13 is 8191 bits; PRBS-9 + tdec_4140 trails it by a near-constant
  +0.01 dB bias, which cannot reorder candidates

so the search runs a cheap PRBS-9 surrogate over (gap, lock) inside a discrete
(W_p, W_n, L) grid, then rescores the winners on full PRBS-13. Every reported
number comes from PRBS-13.

    python examples/mrm_tdec_sky130.py single       # one operating point
    python examples/mrm_tdec_sky130.py sweep        # gap x lock contour
    python examples/mrm_tdec_sky130.py optimize     # the full co-optimization

    -> out/mrm_tdec_*.png, out/mrm_tdec_results.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
import jax.numpy as jnp
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source
from circulax.components.electronic import Resistor, VoltageSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _drivers import single_stage_inverter, stitch_driver  # noqa: E402
from photodiode_tia import Photodiode  # noqa: E402

from photonflux import cx, tdec  # noqa: E402
from photonflux.coupler import DEFAULT_O_BAND  # noqa: E402
from photonflux.signals import prbs  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
C0 = 2.99792458e8


# ---------------------------------------------------------------------------
# the design point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkSpec:
    """One transmitter design. Optical params are free; (w_p, w_n, l) are not."""

    # searched
    gap_nm: float = 200.0
    detune_pm: float = -33.0      # laser - cold resonance; negative = blue
    w_p_um: float = 30.0
    w_n_um: float = 15.0
    l_um: float = 0.18

    # fixed device physics
    baud: float = 53.125e9
    lambda_res_nm: float = 1310.0
    p_laser_w: float = 1e-3
    radius_um: float = 7.5
    n_g: float = 4.0
    n_eff: float = 2.4
    loss_db_m: float = 7000.0
    dl_dv_pm: float = 45.0
    cj_ff_um: float = 0.5
    v_dd: float = 1.8
    r_pd_load: float = 1e3
    t_rise: float = 7e-12

    # numerics
    spu: int = 32                 # samples per UI
    prbs_order: int = 13

    @property
    def kappa2(self) -> float:
        return float(DEFAULT_O_BAND.kappa2(self.gap_nm))

    @property
    def lambda_laser_nm(self) -> float:
        return self.lambda_res_nm + self.detune_pm * 1e-3

    def ring_settings(self) -> dict:
        return {"lambda_nm": self.lambda_laser_nm,
                "lambda_res_nm": self.lambda_res_nm,
                "radius_um": self.radius_um, "n_g": self.n_g,
                "n_eff": self.n_eff, "loss_db_m": self.loss_db_m,
                "kappa2": self.kappa2, "dl_dv_pm": self.dl_dv_pm,
                "cj_ff_um": self.cj_ff_um}

    def cmt(self) -> dict:
        """Derived cavity quantities, exactly as ring_mod.va derives them."""
        circ = 2 * np.pi * self.radius_um * 1e-6
        v_g = C0 / self.n_g
        t_rt = circ / v_g
        alpha = self.loss_db_m * np.log(10) / 10
        inv_tau_i = alpha * v_g / 2
        inv_tau_e = self.kappa2 / (2 * t_rt)
        tau = 1 / (inv_tau_i + inv_tau_e)
        w = 2 * np.pi * C0 / (self.lambda_res_nm * 1e-9)
        q_loaded = w * tau / 2
        return {"tau": tau, "q_loaded": q_loaded,
                "fwhm_pm": self.lambda_res_nm / q_loaded * 1e3,
                "f_bw": 1 / (2 * np.pi * (tau / 2)),
                "kappa2_crit": alpha * circ,
                "cj_f": self.cj_ff_um * 1e-15 * circ * 1e6}

    def energy_fj_per_bit(self) -> float:
        """Electrode dynamic energy per bit, averaged over random NRZ.

        A full charge/discharge cycle costs C*V^2/2 from the supply and random
        NRZ transitions on half the bits, so the per-bit average is C*V^2/4 --
        the figure ring-modulator papers quote.
        """
        return 0.25 * self.cmt()["cj_f"] * self.v_dd**2 * 1e15


# ---------------------------------------------------------------------------
# stimulus + circuit
# ---------------------------------------------------------------------------
def nrz_source(bits, t_bit, v0, v1, t_rise, t_fall):
    """NRZ pattern source with C1-continuous smoothstep edges (BDF2-friendly)."""
    levels = jnp.asarray(np.where(bits > 0, v1, v0), dtype=jnp.float64)
    padded = jnp.concatenate([levels[:1], levels])
    nbits = len(bits)

    @source(ports=("p1", "p2"), states=("i_src",))
    def NRZPattern(signals: Signals, s: States, t: float):
        i = jnp.clip(jnp.floor(t / t_bit).astype(jnp.int32), 0, nbits - 1)
        prev, cur = padded[i], padded[i + 1]
        t_edge = jnp.where(cur >= prev, t_rise, t_fall)
        x = jnp.clip((t - i * t_bit) / t_edge, 0.0, 1.0)
        v = prev + (cur - prev) * x * x * (3.0 - 2.0 * x)
        return {"p1": s.i_src, "p2": -s.i_src,
                "i_src": (signals.p1 - signals.p2) - v}, {}

    return NRZPattern


_MODEL_CACHE: dict = {}


def base_models() -> dict:
    """Component classes shared by every evaluation (compiled once)."""
    if not _MODEL_CACHE:
        _MODEL_CACHE.update({
            "ground": lambda: 0,
            "laser": cx.cw_laser(),
            "f2ri": cx.field_to_ri(),
            "ring": cx.va("ring_mod"),
            "ri2f": cx.ri_to_field(),
            "pd": Photodiode,
            "res": Resistor,
            "vsrc": VoltageSource,
        })
    return dict(_MODEL_CACHE)


def simulate(spec: LinkSpec, bits: np.ndarray):
    """Run the PRBS transient. Returns (dt_sec, through-port power [mW]).

    The FET geometry is what makes this expensive: `cx.sky130_fet` extracts a
    fresh BSIM4 card and compiles an OSDI binary per (device, W, L), 41-80 s
    the first time and free on every repeat.
    """
    t_bit = 1.0 / spec.baud
    n_bits = len(bits)

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
        "VDD": {"component": "vsrc", "settings": {"V": spec.v_dd}},
        "VIN": {"component": "nrz"},
    }
    conn = {
        "LAS,p1": "TAP,c",
        "TAP,re": "RING,in_re",
        "TAP,im": "RING,in_im",
        "RING,out_re": "JOIN,re",
        "RING,out_im": "JOIN,im",
        "JOIN,c": "PD,po_p",
        "PD,cat": "RL,p1",
    }
    gnd = ["LAS,p2", "RING,vn", "RING,gnd", "PD,po_n", "PD,an", "RL,p2",
           "VDD,p2", "VIN,p2"]

    mdl = base_models()
    mdl["nrz"] = nrz_source(bits, t_bit, 0.0, spec.v_dd,
                            spec.t_rise, spec.t_rise)

    parts = single_stage_inverter(w_p=spec.w_p_um, w_n=spec.w_n_um, l=spec.l_um)
    vdrv = stitch_driver(parts, inst, conn, mdl, gnd,
                         vin="VIN,p1", vdd="VDD,p1", load="RING,vp")
    conn["GND,p1"] = tuple(gnd)

    net = {"instances": inst, "connections": conn,
           "ports": {"vdrv": vdrv, "prx": "PD,po_p"}}
    c = compile_circuit(net, mdl, backend="dense", is_complex=True, max_steps=300)

    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    dt = t_bit / spec.spu
    t_max = n_bits * t_bit
    ts = jnp.arange(n_bits * spec.spu) * dt
    sol = c.transient(
        t0=0.0, t1=t_max, dt0=dt, y0=c.dc(),
        saveat=diffrax.SaveAt(ts=ts),
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver, newton_max_steps=40),
        max_steps=int(t_max / dt) + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    if sol.result != diffrax.RESULTS.successful:
        raise RuntimeError(f"transient failed: {sol.result}")

    p_mw = np.asarray(jnp.abs(c.port(sol.ys, "prx")) ** 2).real * 1e3
    return dt, p_mw


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
_PATTERN_CACHE: dict[int, np.ndarray] = {}


def pattern(order: int) -> np.ndarray:
    if order not in _PATTERN_CACHE:
        _PATTERN_CACHE[order] = prbs(order)
    return _PATTERN_CACHE[order]


def score(spec: LinkSpec, *, order: int | None = None, s_noise_mW: float = 0.01,
          ber: float = 1e-12, ref_bw_factor: float = 0.75, ref_order: int = 4):
    """Simulate + measure. Returns a tdec.Measurement, or None if it failed.

    A full-period PRBS-13 populates the `_8180` metrics; PRBS-9 only supports
    `_4140`, so the metric family follows the pattern automatically.
    """
    order = spec.prbs_order if order is None else order
    oma_type = "8180" if order >= 13 else "4140"
    try:
        dt, p = simulate(spec, pattern(order))
        m = tdec.measure(
            p, dt, spec.baud,
            s_noise_mW=s_noise_mW, ber=ber,
            ref_rx_bw_factor=ref_bw_factor, ref_rx_order=ref_order,
            oma_type=oma_type, strict=True,
        )
    except (RuntimeError, ValueError, AssertionError) as exc:
        print(f"    eval failed at gap {spec.gap_nm:.0f} nm / lock "
              f"{spec.detune_pm:+.0f} pm, W {spec.w_p_um:g}/{spec.w_n_um:g} "
              f"L {spec.l_um:g}: {exc}")
        return None
    return m


def objective(spec: LinkSpec, **kw) -> float:
    """OMA - TDEC in dBm; -inf when the point is not evaluable (to minimise, negate)."""
    m = score(spec, **kw)
    if m is None or not np.isfinite(m["oma_tdec_dbm"]):
        return -np.inf
    _FLOOR_HITS[0] += int(m["at_floor"])
    _FLOOR_HITS[1] += 1
    return m["oma_tdec_dbm"]


# [saturated, total] evaluations -- OMA - TDEC bottoms out at a floor set by
# the assumed receiver noise S, and every design at the floor scores alike.
_FLOOR_HITS = [0, 0]


def report_floor_hits(s_noise_mW: float, ber: float) -> None:
    hits, total = _FLOOR_HITS
    if not total:
        return
    floor = tdec.oma_tdec_floor_dbm(s_noise_mW, ber)
    frac = hits / total
    print(f"\n{hits}/{total} evaluations ({frac:.0%}) sat at the "
          f"{floor:+.3f} dBm saturation floor of OMA - TDEC.")
    if frac > 0.25:
        print("  Those designs are indistinguishable to the metric -- their eyes are "
              "closed enough\n  that the tolerable receiver noise R has collapsed to "
              f"S = {s_noise_mW} mW. The search sees a\n  flat plateau there. Narrow "
              "the box, or lower --s-noise to match the real reference\n  receiver and "
              "push the floor down.")


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def mode_single(spec: LinkSpec, args) -> None:
    cmt = spec.cmt()
    print(f"device: R = {spec.radius_um} um, kappa2 = {spec.kappa2:.4f} "
          f"(gap {spec.gap_nm:.0f} nm, critical = {cmt['kappa2_crit']:.4f})")
    print(f"        Q_loaded = {cmt['q_loaded']:.0f}, linewidth = "
          f"{cmt['fwhm_pm']:.0f} pm, photon-lifetime f_3dB = "
          f"{cmt['f_bw']/1e9:.1f} GHz, C_j = {cmt['cj_f']*1e15:.1f} fF")
    print(f"lock:   laser {spec.detune_pm:+.0f} pm from resonance "
          f"({spec.detune_pm/(cmt['fwhm_pm']/2):+.2f} HWHM)")
    print(f"driver: {spec.w_p_um}/{spec.w_n_um} um at L = {spec.l_um} um, "
          f"VDD = {spec.v_dd} V")
    n = len(pattern(spec.prbs_order))
    print(f"pattern: PRBS-{spec.prbs_order}, {n} bits at {spec.baud/1e9:g} GBd, "
          f"{spec.spu} samples/UI -> {n*spec.spu} timesteps")
    print(f"ref rx: Bessel order {args.ref_order} at "
          f"{args.ref_bw:.2f} x baud = {args.ref_bw*spec.baud/1e9:.2f} GHz")

    t0 = time.time()
    m = score(spec, s_noise_mW=args.s_noise, ber=args.ber,
              ref_bw_factor=args.ref_bw, ref_order=args.ref_order)
    if m is None:
        raise SystemExit("evaluation failed")
    ot = m.oma_type
    print(f"\n  evaluated in {time.time()-t0:.1f} s")
    print(f"  OMA_{ot}              {m[f'oma_{ot}']:.4f} mW  "
          f"({10*np.log10(m[f'oma_{ot}']):+.2f} dBm)")
    print(f"  TDEC_{ot}             {m[f'tdec_{ot}']:+.4f} dB")
    print(f"  OMA - TDEC            {m['oma_tdec_dbm']:+.4f} dBm"
          + ("   *** AT FLOOR ***" if m["at_floor"] else ""))
    print(f"  (floor for S={args.s_noise} mW) {m['oma_tdec_floor_dbm']:+.4f} dBm")
    print(f"  extinction ratio      {m[f'extinction_ratio_{ot}']:.2f} dB")
    print(f"  inner eye height      {m['inner_eye_height']:.4f} mW")
    # stateye reports time measurements in ps already (DEFAULT_TIME_UNITS).
    # dcd_8180 is NaN under PRBS-13 -- its filter needs a contiguous 0^8->1^8
    # window the pattern never contains -- so the _4140 variant is the one.
    print(f"  DCD (4140)            {m['dcd_4140']:.2f} ps")
    print(f"  energy (electrode)    {spec.energy_fj_per_bit():.1f} fJ/bit")
    print(f"  usable {ot} segments  {m.counts.get(f'oma_{ot}')}")

    OUT.mkdir(exist_ok=True)
    try:
        fig = m.eye.plot(show=False)
        fig.savefig(OUT / "mrm_tdec_eye.png", dpi=150)
        print(f"\nwrote {OUT/'mrm_tdec_eye.png'}")
    except Exception as exc:  # unpatched stateye 1.7 uses removed matplotlib APIs
        print(f"\neye plot unavailable ({type(exc).__name__}: {exc})")


def mode_sweep(spec: LinkSpec, args) -> None:
    """gap x lock-point contour of OMA - TDEC on the cheap surrogate."""
    gaps = np.linspace(args.gap_lo, args.gap_hi, args.n_gap)
    detunes = np.linspace(args.det_lo, args.det_hi, args.n_det)
    grid = np.full((len(detunes), len(gaps)), np.nan)

    total = len(gaps) * len(detunes)
    print(f"sweep: {len(gaps)} gaps x {len(detunes)} lock points = {total} "
          f"evaluations on PRBS-{args.surrogate_order}")
    t0 = time.time()
    k = 0
    for j, g in enumerate(gaps):
        for i, d in enumerate(detunes):
            k += 1
            s = replace(spec, gap_nm=float(g), detune_pm=float(d))
            grid[i, j] = objective(s, order=args.surrogate_order,
                                   s_noise_mW=args.s_noise, ber=args.ber,
                                   ref_bw_factor=args.ref_bw,
                                   ref_order=args.ref_order)
            print(f"  [{k:3d}/{total}] gap {g:5.0f} nm  lock {d:+6.0f} pm  "
                  f"kappa2 {s.kappa2:.4f}  OMA-TDEC {grid[i,j]:+7.3f} dBm")
    print(f"sweep took {time.time()-t0:.0f} s")

    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    print(f"best on grid: gap {gaps[best[1]]:.0f} nm, lock "
          f"{detunes[best[0]]:+.0f} pm -> {grid[best]:+.3f} dBm")

    _plot_contour(gaps, detunes, grid, spec, OUT / "mrm_tdec_sweep.png")


def _plot_contour(gaps, detunes, grid, spec, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    finite = np.isfinite(grid)
    if not finite.any():
        raise RuntimeError("every sweep point failed; nothing to plot")
    pc = ax.pcolormesh(gaps, detunes, np.where(finite, grid, np.nan),
                       shading="nearest", cmap="viridis")
    fig.colorbar(pc, ax=ax, label="OMA $-$ TDEC  [dBm]")
    if finite.sum() > 3:
        ax.contour(gaps, detunes, np.where(finite, grid, np.nanmin(grid)),
                   colors="w", linewidths=0.5, alpha=0.6)
    b = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(gaps[b[1]], detunes[b[0]], "r*", ms=16, label="best")
    crit = spec.cmt()["kappa2_crit"]
    try:
        ax.axvline(float(DEFAULT_O_BAND.gap_for_kappa2(crit)), color="r",
                   ls="--", lw=1, alpha=0.7, label="critical coupling")
    except ValueError:
        pass
    ax.axhline(0.0, color="w", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("bus gap [nm]")
    ax.set_ylabel("lock point: laser $-$ resonance [pm]")
    ax.set_title(f"{spec.baud/1e9:g} GBd NRZ microring + SKY130 "
                 f"{spec.w_p_um:g}/{spec.w_n_um:g} um inverter")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _nelder_mead(f, x0, lo, hi, *, max_evals=60, step=0.25):
    """Bounded Nelder-Mead on an n-vector. Derivative-free because the objective
    runs through stateye's Cython histogram -- there is no gradient to take."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    clip = lambda x: np.clip(x, lo, hi)  # noqa: E731
    span = (hi - lo) * step

    # Build the initial simplex from a clipped x0, stepping *away* from
    # whichever bound is nearer. Stepping blindly in +x collapses the vertex
    # onto x0 whenever x0 sits on an upper bound (reachable from the CLI with
    # e.g. --detune 150), leaving the simplex rank-deficient in that dimension
    # and the search permanently blind along it.
    x0 = clip(np.asarray(x0, float))
    simplex = [x0]
    for i in range(len(x0)):
        p = x0.copy()
        p[i] += -span[i] if x0[i] + span[i] > hi[i] else span[i]
        p = clip(p)
        if p[i] == x0[i]:  # degenerate box in this dimension
            p[i] = hi[i] if x0[i] < hi[i] else lo[i]
        simplex.append(p)
    vals = [f(p) for p in simplex]
    n_eval = len(simplex)

    while n_eval < max_evals:
        order = np.argsort(vals)          # minimising
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)

        xr = clip(centroid + (centroid - simplex[-1]))
        fr = f(xr); n_eval += 1
        if fr < vals[0]:
            xe = clip(centroid + 2.0 * (centroid - simplex[-1]))
            fe = f(xe); n_eval += 1
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            # outside contraction when the reflection at least beat the worst
            # vertex, inside otherwise -- each transient is expensive enough
            # that throwing away a usable xr is worth avoiding
            if fr < vals[-1]:
                xc = clip(centroid + 0.5 * (xr - centroid))
            else:
                xc = clip(centroid + 0.5 * (simplex[-1] - centroid))
            fc = f(xc); n_eval += 1
            if fc < min(vals[-1], fr):
                simplex[-1], vals[-1] = xc, fc
            elif fr < vals[-1]:
                simplex[-1], vals[-1] = xr, fr
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    vals[i] = f(simplex[i]); n_eval += 1
        if np.max(np.ptp(np.asarray(simplex), axis=0) / (hi - lo)) < 1e-3:
            break

    best = int(np.argmin(vals))
    return simplex[best], vals[best], n_eval


def mode_optimize(spec: LinkSpec, args) -> None:
    w_ps = [float(x) for x in args.w_p.split(",")]
    w_ns = [float(x) for x in args.w_n.split(",")]
    ls = [float(x) for x in args.l.split(",")]
    geoms = list(itertools.product(w_ps, w_ns, ls))

    print(f"outer grid: {len(geoms)} (W_p, W_n, L) points "
          f"-- each new pair costs 41-80 s of SKY130 card + OSDI compile")
    print(f"inner loop: bounded Nelder-Mead over (gap, lock) on "
          f"PRBS-{args.surrogate_order}, <= {args.max_evals} evals each\n")

    t_start = time.time()
    rows = []
    for gi, (wp, wn, lch) in enumerate(geoms, 1):
        base = replace(spec, w_p_um=wp, w_n_um=wn, l_um=lch)
        t0 = time.time()

        def neg(x):
            s = replace(base, gap_nm=float(x[0]), detune_pm=float(x[1]))
            return -objective(s, order=args.surrogate_order,
                              s_noise_mW=args.s_noise, ber=args.ber,
                              ref_bw_factor=args.ref_bw, ref_order=args.ref_order)

        x, fval, n_evals = _nelder_mead(
            neg, [spec.gap_nm, spec.detune_pm],
            [args.gap_lo, args.det_lo], [args.gap_hi, args.det_hi],
            max_evals=args.max_evals)
        best = replace(base, gap_nm=float(x[0]), detune_pm=float(x[1]))
        print(f"[{gi}/{len(geoms)}] W {wp:g}/{wn:g} L {lch:g}: "
              f"gap {x[0]:5.0f} nm  lock {x[1]:+6.0f} pm  "
              f"surrogate OMA-TDEC {-fval:+7.3f} dBm  "
              f"({n_evals} evals, {time.time()-t0:.0f} s)")
        rows.append({"w_p_um": wp, "w_n_um": wn, "l_um": lch,
                     "gap_nm": x[0], "detune_pm": x[1], "kappa2": best.kappa2,
                     "surrogate_oma_tdec_dbm": -fval,
                     "energy_fj_per_bit": best.energy_fj_per_bit(),
                     "spec": best})

    # --- final rescore of the survivors on full PRBS-13 --------------------
    rows.sort(key=lambda r: r["surrogate_oma_tdec_dbm"], reverse=True)
    top = [r for r in rows if np.isfinite(r["surrogate_oma_tdec_dbm"])][: args.rescore]
    print(f"\nrescoring top {len(top)} on full PRBS-13 "
          f"({len(pattern(13))} bits, ~16x the surrogate cost)")
    for r in top:
        m = score(r["spec"], order=13, s_noise_mW=args.s_noise, ber=args.ber,
                  ref_bw_factor=args.ref_bw, ref_order=args.ref_order)
        if m is None:
            r["oma_tdec_dbm"] = float("nan")
            continue
        r["oma_tdec_dbm"] = m["oma_tdec_dbm"]
        r["oma_mw"] = m["oma_8180"]
        r["tdec_db"] = m["tdec_8180"]
        r["er_db"] = m["extinction_ratio_8180"]
        print(f"  W {r['w_p_um']:g}/{r['w_n_um']:g} L {r['l_um']:g}  "
              f"gap {r['gap_nm']:.0f} nm  lock {r['detune_pm']:+.0f} pm  ->  "
              f"OMA-TDEC {r['oma_tdec_dbm']:+.3f} dBm  "
              f"(OMA {r['oma_mw']:.4f} mW, TDEC {r['tdec_db']:+.3f} dB, "
              f"ER {r['er_db']:.2f} dB, surrogate error "
              f"{r['oma_tdec_dbm']-r['surrogate_oma_tdec_dbm']:+.3f} dB)")

    scored = [r for r in top if np.isfinite(r.get("oma_tdec_dbm", np.nan))]
    if scored:
        win = max(scored, key=lambda r: r["oma_tdec_dbm"])
        print(f"\nBEST: W_p {win['w_p_um']:g} um, W_n {win['w_n_um']:g} um, "
              f"L {win['l_um']:g} um, gap {win['gap_nm']:.0f} nm "
              f"(kappa2 {win['kappa2']:.4f}), lock {win['detune_pm']:+.0f} pm")
        print(f"      OMA - TDEC = {win['oma_tdec_dbm']:+.3f} dBm on full PRBS-13, "
              f"{win['energy_fj_per_bit']:.1f} fJ/bit at the electrode")

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "mrm_tdec_results.csv"
    cols = ["w_p_um", "w_n_um", "l_um", "gap_nm", "detune_pm", "kappa2",
            "surrogate_oma_tdec_dbm", "oma_tdec_dbm", "oma_mw", "tdec_db",
            "er_db", "energy_fj_per_bit"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\ntotal {time.time()-t_start:.0f} s; wrote {csv_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("single", "sweep", "optimize"))
    ap.add_argument("--baud", type=float, default=53.125e9)
    ap.add_argument("--gap", type=float, default=200.0, help="bus gap [nm]")
    ap.add_argument("--detune", type=float, default=-33.0,
                    help="lock point: laser - resonance [pm]")
    ap.add_argument("--w-p", default="20,30,45", help="pfet widths [um], comma list")
    ap.add_argument("--w-n", default="10,15,22", help="nfet widths [um], comma list")
    ap.add_argument("--l", default="0.15,0.18", help="channel lengths [um]")
    ap.add_argument("--spu", type=int, default=32, help="samples per UI")
    ap.add_argument("--prbs-order", type=int, default=13)
    ap.add_argument("--surrogate-order", type=int, default=9,
                    help="cheap pattern for the search loop (9 -> tdec_4140)")
    ap.add_argument("--max-evals", type=int, default=40)
    ap.add_argument("--rescore", type=int, default=5,
                    help="how many survivors to rescore on full PRBS-13")
    ap.add_argument("--gap-lo", type=float, default=150.0)
    ap.add_argument("--gap-hi", type=float, default=320.0)
    ap.add_argument("--det-lo", type=float, default=-150.0)
    ap.add_argument("--det-hi", type=float, default=150.0)
    ap.add_argument("--n-gap", type=int, default=7)
    ap.add_argument("--n-det", type=int, default=7)
    ap.add_argument("--ref-bw", type=float, default=0.75,
                    help="reference-receiver -3 dB bandwidth / baud")
    ap.add_argument("--ref-order", type=int, default=4,
                    help="reference-receiver Bessel order")
    ap.add_argument("--s-noise", type=float, default=0.01,
                    help="TDEC S: O/E + scope noise std [mW]")
    ap.add_argument("--ber", type=float, default=1e-12)
    args = ap.parse_args()

    spec = LinkSpec(baud=args.baud, gap_nm=args.gap, detune_pm=args.detune,
                    spu=args.spu, prbs_order=args.prbs_order,
                    w_p_um=float(args.w_p.split(",")[0]),
                    w_n_um=float(args.w_n.split(",")[0]),
                    l_um=float(args.l.split(",")[0]))

    {"single": mode_single, "sweep": mode_sweep,
     "optimize": mode_optimize}[args.mode](spec, args)


if __name__ == "__main__":
    main()
