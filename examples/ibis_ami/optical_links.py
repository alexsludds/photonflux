#!/usr/bin/env python3
"""IBIS-AMI SerDes wrapped around simulated optics: where LTI stops holding.

An optical link driven and received by IBIS-AMI models, with the optics in
between simulated by circulax -- the photonflux web app's engine, run
in-process on its examples 51 and 52:

* **C-band NRZ** (example 51): ``pf_tx`` (post-cursor de-emphasis) -> MZM
  -> standard fiber (D = 17 ps/nm/km) -> PD -> TIA -> ``pf_rx`` in DSP mode
  (adaptive CTLE, 9-tap FFE, 4-tap DFE, Mueller-Muller CDR), 25 Gb/s,
  swept over fiber length.
* **106G PAM4, O-band** (example 52): ``pf_tx`` (3-tap FFE, 35 GHz driver)
  -> quadrature-biased MZM -> 2 km -> PD -> 32 GHz TIA -> ``pf_rx`` (11-tap
  FFE, 1-tap DFE, MM CDR), swept over the MZM drive swing.

The AMI models only see electrical waveforms, as in a real optical SerDes:
the Tx's ``AMI_GetWave`` output drives the circuit's modulator, and the
TIA output goes to the Rx's ``AMI_GetWave``
(``photonflux.serdes.analyze_waveform``). The statistical flow needs an LTI
channel; ``serdes.estimate_channel`` fits the best linear one to the
transient and the data. Square-law detection after dispersion, and the
MZM's sine transfer, are not linear -- the fit's residual grows with fiber
length and with drive swing, and the pure-LTI statistical BER (what an
IBIS-AMI tool's Init flow would claim) parts company with the counted one.
Treating the residual as noise, filtered by the Rx's own front end and read
at its recovered clock's phase, brings the two back together.

    python examples/ibis_ami/optical_links.py          # ~6 min
    python examples/ibis_ami/optical_links.py --quick  # ~2 min
        -> out/ibis_ami_optical_links.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import REPO, ensure_built, fmt_ber, model, save  # noqa: E402

sys.path.insert(0, str(REPO / "webapp"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import amilink  # noqa: E402
import simulate  # noqa: E402
import wavesrc  # noqa: E402

from photonflux import serdes  # noqa: E402

NSPUI = 32


def load(n: int) -> dict:
    name = {51: "51_ami_optical_cband", 52: "52_ami_optical_pam4"}[n]
    return json.loads((REPO / "webapp" / "examples" / f"{name}.json").read_text())


def run_link(doc: dict, edit) -> serdes.WaveformAnalysis:
    """Transient of the example schematic (after ``edit``), then the
    IBIS-AMI Rx analysis of its TIA output."""
    sch = json.loads(json.dumps(doc["schematic"]))
    edit(sch["instances"])
    tx = sch["instances"]["TX1"]["settings"]
    ui = 1.0 / sch["globals"]["baud"]
    t_stop = float(doc["analysis"]["t_stop"])
    npts = int(t_stop / ui * NSPUI) + 1
    res = simulate.run({"schematic": sch, "analysis": {
        "mode": "transient", "t_stop": t_stop, "points": npts,
        "max_points": npts}})
    if not res.get("ok"):
        raise RuntimeError(res.get("error"))
    tr = next(t for t in res["traces"] if t["name"] == "vrx")
    ts = ui / NSPUI
    tu = np.arange(0.0, res["x"][-1], ts)
    y = np.interp(tu, res["x"], tr["values"])
    nsym = len(y) // NSPUI
    a = 2.0 * wavesrc._symbols(dict(tx, ui=ui), nsym + 8)[:nsym] - 1.0
    blk = sch["instances"]["RX1"]["settings"]
    rx = model("pf_rx", **amilink.parse_overrides(blk["ami_params"]))
    mod = "PAM4" if tx.get("mode") == "pam4" else "NRZ"
    return serdes.analyze_waveform(rx, y, a, ui, ts, modulation=mod,
                                   noise_rms=float(blk.get("noise_rms", 0.0)),
                                   skip=min(rx.ignore_bits, nsym // 3))


def bers(wa: serdes.WaveformAnalysis) -> dict:
    c = wa.td.counted
    return {"counted": c["ber"], "bound": 1.0 / max(c["bits"], 1),
            "errors": c["bit_errors"], "qfit": wa.td.qfit.get("ber_est", np.nan),
            "stat": wa.stat.eye.bit_error_rate(),
            "lti": wa.stat_lti.eye.bit_error_rate(),
            "resid": wa.residual_rel, "dist_mv": 1e3 * wa.distortion_at_slicer}


def plot_bers(ax, x, rows, xlabel):
    floor = 1e-30
    col = lambda k: np.maximum([r[k] for r in rows], floor)  # noqa: E731
    ax.semilogy(x, col("lti"), "k:", marker=".", label="statistical, link taken as LTI")
    ax.semilogy(x, col("stat"), "C0-", marker="s", label="statistical + distortion as noise")
    ax.semilogy(x, col("qfit"), "C2--", marker="x", label="time domain, Q-fit")
    hit = [r["counted"] > 0 for r in rows]
    ax.semilogy([v for v, h in zip(x, hit) if h], [r["counted"] for r, h in zip(rows, hit) if h],
                "C3o", ms=7, label="time domain, counted")
    ax.semilogy([v for v, h in zip(x, hit) if not h], [r["bound"] for r, h in zip(rows, hit) if not h],
                "C3v", mfc="none", ms=7, label="counted: 0 errors (< 1/N)")
    ax.set_ylim(1e-30, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer sweep points")
    args = ap.parse_args(argv)
    ensure_built()

    lengths = [0.0, 12.0, 20.0, 24.0] if args.quick else [0.0, 8.0, 12.0, 16.0, 20.0, 22.0, 24.0]
    swings = [0.6, 1.0] if args.quick else [0.4, 0.6, 0.8, 1.0, 1.2]

    cband, pam4 = load(51), load(52)
    rows_c, keep_c = [], {}
    for L in lengths:
        t0 = time.time()
        wa = run_link(cband, lambda I, L=L: I["FIB1"]["settings"].update(length_km=L))
        rows_c.append(bers(wa))
        keep_c[L] = wa
        r = rows_c[-1]
        print(f"C-band {L:4.0f} km: counted {fmt_ber(r['counted'])} ({r['errors']} errors), "
              f"Q-fit {r['qfit']:.2e}, statistical {r['stat']:.2e} (LTI {r['lti']:.2e}), "
              f"linear-fit residual {r['resid']:.1%} ({time.time() - t0:.0f} s)")
    rows_p, keep_p = [], {}
    for v in swings:
        t0 = time.time()
        wa = run_link(pam4, lambda I, v=v: I["TX1"]["settings"].update(v0=-v, v1=v))
        rows_p.append(bers(wa))
        rows_p[-1]["heights"] = [wa.stat.eye.eye_height(1e-6, e) for e in range(3)]
        keep_p[v] = wa
        r = rows_p[-1]
        print(f"PAM4 +/-{v:.1f} V drive: counted {fmt_ber(r['counted'])} ({r['errors']} errors), "
              f"Q-fit {r['qfit']:.2e}, statistical {r['stat']:.2e} (LTI {r['lti']:.2e}), "
              f"residual {r['resid']:.1%}, eyes @1e-6 "
              f"{', '.join(f'{h * 1e3:.0f}' for h in r['heights'])} mV ({time.time() - t0:.0f} s)")

    fig, axs = plt.subplots(2, 4, figsize=(22, 10), layout="constrained")
    plot_bers(axs[0, 0], lengths, rows_c, "C-band fiber length [km]")
    axs[0, 0].set_title("25G NRZ, C-band: BER vs fiber length")
    axs[0, 0].legend(fontsize=7)

    ax = axs[0, 1]
    ax.plot(lengths, [100 * r["resid"] for r in rows_c], "C1o-", label="C-band NRZ vs length")
    ax.set_xlabel("C-band fiber length [km]")
    ax.set_ylabel("linear-fit residual [% of signal rms]")
    ax.set_title("how far from LTI (square-law after dispersion)")
    ax.grid(alpha=0.3)

    show_c = keep_c[max(L for L in lengths if keep_c[L].td.counted["ber"] < 1e-2)]
    serdes.plot_eye_density(axs[0, 2], show_c.td, title=(
        f"NRZ at {[k for k, v in keep_c.items() if v is show_c][0]:g} km: Rx output, "
        "CDR-referenced"))

    ax = axs[0, 3]
    td = show_c.td
    t_ui, _ = td.adaptation("level")
    nf = len(td.rx_trace.trace[-1].get("ffe_taps", []))
    for j in range(nf):
        ax.plot(t_ui, td.adaptation("ffe_taps", j)[1], lw=1, label=f"FFE w{j - 3:+d}")
    for j in range(len(td.rx_trace.trace[-1].get("dfe_taps", []))):
        ax.plot(t_ui, td.adaptation("dfe_taps", j)[1], "k", lw=1.4 - 0.2 * j,
                label=f"DFE b{j + 1}")
    ax.axvline(td.skip, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time [UI]")
    ax.set_ylabel("tap")
    ax.set_title("Rx GetWave adaptation (LMS)")
    ax.legend(fontsize=6, ncol=3)

    plot_bers(axs[1, 0], swings, rows_p, "MZM drive [+/- V, V-pi = 2 V]")
    axs[1, 0].set_title("106G PAM4, O-band: BER vs drive swing")

    ax = axs[1, 1]
    hs = np.array([r["heights"] for r in rows_p]) * 1e3
    for e in range(3):
        ax.plot(swings, hs[:, e], "o-", label=f"eye {e} @ 1e-6")
    ax2 = ax.twinx()
    ax2.plot(swings, [100 * r["resid"] for r in rows_p], "k--", label="residual")
    ax2.set_ylabel("linear-fit residual [%] (dashed)")
    ax.set_xlabel("MZM drive [+/- V]")
    ax.set_ylabel("statistical eye height [mV]")
    ax.set_title("MZM compression: outer eyes close first")
    ax.legend(fontsize=8)

    serdes.plot_eye_density(axs[1, 2], keep_p[swings[-1]].td,
                            title=f"PAM4 at +/-{swings[-1]} V: Rx output")
    nominal = keep_p[min(swings, key=lambda v: abs(v - 0.6))]
    serdes.plot_stat_eye(axs[1, 3], nominal.stat.eye,
                         title="PAM4 at +/-0.6 V: statistical eye (+ distortion)")
    fig.suptitle("IBIS-AMI Tx/Rx models around circulax-simulated optics "
                 "(photonflux web examples 51, 52)", fontsize=12)
    save(fig, "ibis_ami_optical_links.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
