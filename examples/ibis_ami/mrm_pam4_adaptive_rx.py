#!/usr/bin/env python3
"""53.125 GBd PAM4 microring link: the IBIS-AMI Rx learns its FFE + DFE in the time domain.

Web example 53, run in-process: ``pf_tx`` (30 GHz driver, no FFE) drives
``models/optical_field/ring_mod.va`` (the example-46 ring, laser 120 pm on
its red side) through 50 Ohm, then a photodiode and an 18 GHz TIA. The
receiver is ``pf_rx`` as an ADC-DSP: no CTLE, a 5-tap FFE (one pre-cursor),
one DFE tap, a Mueller-Muller CDR on the FFE input and PAM4 slicers.

``adapt_start=1`` starts ``AMI_GetWave`` cold: a pass-through FFE and a zero
DFE tap, with gain, data level and sampling phase taken from the
unequalized pulse. Decision-directed LMS then learns every tap from the
received waveform alone, as an optical DSP does at power-up. The CDR
reads the FFE input, so timing never depends on the taps and the
pre-cursor tap adapts too.

1. **Adaptation from cold**: every FFE tap, the DFE tap, the CDR phase and
   the slicer error vs time, next to ``AMI_Init``'s statistical (MMSE /
   zero-forcing) design for the same channel, and an LMS run seeded from
   that design.
2. **Which equalizer**: no EQ, DFE only, 3/5/7-tap FFE, each with and
   without the DFE tap, all adapted from cold: counted and Q-fit BER from
   the time domain, and the statistical BER of ``AMI_Init``'s design.
3. **BER vs Rx input noise**: 5-tap FFE vs 5-tap FFE + 1 DFE tap. The DFE
   cancels the first post-cursor without amplifying noise, which the FFE
   must do to cancel it.

The FFE's c(+1) and the DFE tap both act on the first post-cursor, so the
error surface is nearly flat along their trade-off: the cold start and the
Init-seeded run settle at different splits with almost the same MSE.

    python examples/ibis_ami/mrm_pam4_adaptive_rx.py           # ~3 min
    python examples/ibis_ami/mrm_pam4_adaptive_rx.py --nui 8000 --quick
        -> out/ibis_ami_mrm_pam4_adaptive.png
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

from _common import REPO, ensure_built, fmt_ber, model, plot_counted, save  # noqa: E402

sys.path.insert(0, str(REPO / "webapp"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import amilink  # noqa: E402
import simulate  # noqa: E402
import wavesrc  # noqa: E402

from photonflux import serdes  # noqa: E402

NSPUI = 32
EXAMPLE = REPO / "webapp" / "examples" / "53_ami_mrm_pam4_adaptive.json"


def transient(doc: dict, nui: int) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """The example's circuit for ``nui`` UI: the TIA output at 32 samples/UI,
    the transmitted symbols (-1..1) and the UI."""
    sch = doc["schematic"]
    tx = sch["instances"]["TX1"]["settings"]
    ui = 1.0 / sch["globals"]["baud"]
    npts = nui * NSPUI + 1
    # the web app caps a saved transient at 8192 UI of 32 samples; a script
    # may keep more
    simulate.MAX_FLOW_POINTS = max(simulate.MAX_FLOW_POINTS, npts)
    res = simulate.run({"schematic": sch, "analysis": {
        "mode": "transient", "t_stop": nui * ui, "points": npts,
        "max_points": npts}})
    if not res.get("ok"):
        raise RuntimeError(res.get("error"))
    traces = {t["name"]: np.asarray(t["values"], float) for t in res["traces"]}
    ts = ui / NSPUI
    tu = np.arange(0.0, res["x"][-1], ts)
    y = np.interp(tu, res["x"], traces["vrx"])
    nsym = len(y) // NSPUI
    a = 2.0 * wavesrc._symbols(dict(tx, ui=ui), nsym + 8)[:nsym] - 1.0
    drive = np.interp(tu, res["x"], traces["vdrv"])
    popt = np.interp(tu, res["x"], traces["popt"])
    return y, a, ui, {"vdrv": drive, "popt": popt}


def rx_run(y, a, ui, base: dict, skip: int, noise: float | None = None,
           **over) -> serdes.WaveformAnalysis:
    """pf_rx with the example's settings plus ``over``, on the waveform, with
    the example's input noise (or ``noise``, V rms per sample)."""
    with model("pf_rx", **{**base["params"], **over}) as rx:
        return serdes.analyze_waveform(
            rx, y, a, ui, ui / NSPUI, modulation="PAM4",
            noise_rms=base["noise"] if noise is None else noise, skip=skip)


def slicer_error(td: serdes.TimeDomainLink, win: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Rolling rms of (slicer input - its level for the known symbol), per
    PAM4 level spacing, vs the decision index (~UI)."""
    k = np.arange(len(td.samples))
    idx = k - td.lag
    ok = (idx >= 0) & (idx < len(td.symbols))
    s, sym = td.samples[ok], td.symbols[idx[ok]]
    tail = slice(len(s) // 2, None)
    lv = {v: s[tail][np.isclose(sym[tail], v)].mean() for v in np.unique(sym)}
    e = s - np.array([lv[v] for v in sym])
    spacing = (max(lv.values()) - min(lv.values())) / 3.0
    n = len(e) // win
    rms = np.sqrt(np.mean(e[: n * win].reshape(n, win) ** 2, axis=1)) / spacing
    return (np.arange(n) + 0.5) * win, rms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nui", type=int, default=20000, help="transient length [UI]")
    ap.add_argument("--quick", action="store_true", help="fewer noise points")
    args = ap.parse_args(argv)
    if args.nui < 3000:
        ap.error("--nui: at least 3000 UI (the Rx needs ~3000 to adapt)")
    ensure_built()

    doc = json.loads(EXAMPLE.read_text())
    blk = doc["schematic"]["instances"]["RX1"]["settings"]
    base = {"params": amilink.parse_overrides(blk["ami_params"]),
            "noise": float(blk["noise_rms"])}
    t0 = time.time()
    y, a, ui, probes = transient(doc, args.nui)
    nsym = len(a)
    skip = min(4000, nsym // 3)
    print(f"circulax transient, {nsym} UI of the MRM link: {time.time() - t0:.0f} s")

    # -- 1. cold start, and seeded from Init ---------------------------------
    cold = rx_run(y, a, ui, base, skip)
    seeded = rx_run(y, a, ui, base, skip, adapt_start=0)
    ch = cold.channel
    p = ch.pulse(ui)
    pk = int(np.argmax(p))
    cur = p[pk + np.arange(-2, 5) * NSPUI] / p[pk]
    print(f"linear-fit channel cursors (h-2..h+4): {np.round(cur, 3)}, "
          f"residual {cold.residual_rel:.1%} of the signal rms")
    print(f"Init: {cold.stat.rx_message.strip()}")
    for name, wa in (("cold start", cold), ("seeded from Init", seeded)):
        td, fin = wa.td, wa.td.rx_trace.trace[-1]
        c = td.counted
        print(f"GetWave, {name}: {c['bit_errors']}/{c['bits']} bit errors after "
              f"{td.skip} UI (BER {fmt_ber(c['ber'])}), Q-fit {td.qfit['ber_est']:.2e}; "
              f"FFE {np.round(fin['ffe_taps'], 3)}, DFE {fin['dfe_taps']:.3f}")
    design = cold.stat.rx_params
    print(f"Init design: FFE {np.round(design['ffe_taps'], 3)}, DFE "
          f"{design['dfe_taps']:.3f}; statistical BER {cold.stat.eye.bit_error_rate():.2e}")

    # -- 2. equalizer comparison (all from cold) ------------------------------
    variants = [("no EQ", dict(ffe_mode=0, dfe_mode=0)),
                ("DFE 1", dict(ffe_mode=0)),
                ("FFE 3", dict(ffe_ntaps=3, dfe_mode=0)),
                ("FFE 3 + DFE 1", dict(ffe_ntaps=3)),
                ("FFE 5", dict(dfe_mode=0)),
                ("FFE 5 + DFE 1", {}),
                ("FFE 7", dict(ffe_ntaps=7, ffe_pre=2, dfe_mode=0)),
                ("FFE 7 + DFE 1", dict(ffe_ntaps=7, ffe_pre=2))]
    table, runs = [], {}
    for label, over in variants:
        wa = runs[label] = cold if not over else rx_run(y, a, ui, base, skip, **over)
        c = wa.td.counted
        table.append((label, c["ber"], 1.0 / max(c["bits"], 1),
                      wa.td.qfit.get("ber_est", np.nan), wa.stat.eye.bit_error_rate()))
        print(f"  {label:<14} counted {fmt_ber(c['ber']):>9} ({c['bit_errors']:4d} errors), "
              f"Q-fit {table[-1][3]:.2e}, statistical (Init) {table[-1][4]:.2e}")

    # -- 3. BER vs Rx input noise --------------------------------------------
    sig = np.array([4, 6, 8, 10, 12] if args.quick else [3, 4, 5, 6, 7, 8, 10, 12, 14]) * 1e-3
    sweep = {"FFE 5": dict(dfe_mode=0), "FFE 5 + DFE 1": {}}
    noise_runs = {k: [rx_run(y, a, ui, base, skip, noise=s, **v) for s in sig]
                  for k, v in sweep.items()}
    for s, r5, r51 in zip(sig, *noise_runs.values()):
        print(f"  sigma_in {s * 1e3:4.1f} mV: FFE 5 {fmt_ber(r5.td.counted['ber'])} "
              f"(Q-fit {r5.td.qfit['ber_est']:.1e}), FFE 5 + DFE 1 "
              f"{fmt_ber(r51.td.counted['ber'])} (Q-fit {r51.td.qfit['ber_est']:.1e})")

    # -- figure -------------------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(18, 14), layout="constrained")
    ax = axs[0, 0]
    t_ui = np.arange(len(y)) / NSPUI
    sel = (t_ui >= 400) & (t_ui < 440)
    ax.plot(t_ui[sel], probes["vdrv"][sel], "C0", lw=1)
    ax2 = ax.twinx()
    norm = lambda v: (v - v.min()) / np.ptp(v)  # noqa: E731
    ax2.plot(t_ui[sel], norm(probes["popt"][sel]), "C1", lw=1, label="through-port power")
    ax2.plot(t_ui[sel], norm(y[sel]), "C2", lw=1, label="TIA output (inverting)")
    ax.set(xlabel="time [UI]", ylabel="ring drive [V] (blue)",
           title="MRM drive, optical output, TIA")
    ax2.set_ylabel("normalized")
    ax2.legend(fontsize=7, loc="lower right")

    ax = axs[0, 1]
    k = np.arange(-2, 5)
    ax.stem(k, cur, basefmt=" ")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="cursor [UI]", ylabel="normalized cursor",
           title=f"linear-fit pulse at the Rx ({cold.residual_rel:.1%} nonlinear residual)")

    serdes.plot_eye_density(axs[0, 2], runs["no EQ"].td,
                            title=f"Rx input (no EQ, CDR-referenced), "
                                  f"BER {fmt_ber(table[0][1])}")

    ax = axs[1, 0]
    td = cold.td
    t_tr, _ = td.adaptation("level")
    nf = len(design["ffe_taps"])
    for j in range(nf):
        if j == base["params"]["ffe_pre"]:
            continue
        line, = ax.plot(t_tr, td.adaptation("ffe_taps", j)[1], lw=1.3,
                        label=f"FFE c({j - base['params']['ffe_pre']:+d})")
        ax.axhline(design["ffe_taps"][j], color=line.get_color(), ls=":", lw=1)
    ax.plot(t_tr, td.adaptation("dfe_taps")[1], "k", lw=1.6, label="DFE b1")
    ax.axhline(design["dfe_taps"], color="k", ls=":", lw=1)
    ax.axvline(td.skip, color="k", ls="--", lw=0.6)
    ax.set(xlabel="time [UI]", ylabel="tap (main = 1)",
           title="GetWave LMS from a cold start (dotted: Init's design)")
    ax.legend(fontsize=7, ncol=3, loc="center right")

    ax = axs[1, 1]
    for wa, lab, sty in ((cold, "cold start", "C3-"), (seeded, "seeded from Init", "C0-")):
        x_e, e = slicer_error(wa.td)
        ax.semilogy(x_e, e, sty, lw=1, label=lab)
    ax.axvline(td.skip, color="k", ls="--", lw=0.6, label="ignored before")
    ax.set(xlabel="time [UI]", ylabel="rms slicer error / level spacing",
           title="learning curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    serdes.plot_eye_density(axs[1, 2], td, title=(
        f"Rx output, FFE 5 + DFE 1 adapted: Q-fit {td.qfit['ber_est']:.1e}"))

    ax = axs[2, 0]
    kk = np.arange(nf) - base["params"]["ffe_pre"]
    for off, w, lab, fmt in ((-0.15, design["ffe_taps"], "Init (MMSE)", "C0o"),
                             (0.0, cold.td.rx_trace.trace[-1]["ffe_taps"], "LMS, cold start", "C3s"),
                             (0.15, seeded.td.rx_trace.trace[-1]["ffe_taps"], "LMS, seeded", "C2^")):
        ax.stem(kk + off, w, linefmt=fmt[:2] + "-", markerfmt=fmt, basefmt=" ", label=lab)
    for off, b, fmt in ((-0.15, design["dfe_taps"], "C0o"),
                        (0.0, cold.td.rx_trace.trace[-1]["dfe_taps"], "C3s"),
                        (0.15, seeded.td.rx_trace.trace[-1]["dfe_taps"], "C2^")):
        ax.stem([nf - base["params"]["ffe_pre"] + 0.6 + off], [b], linefmt=fmt[:2] + "-",
                markerfmt=fmt, basefmt=" ")
    ax.set_xticks(list(kk) + [nf - base["params"]["ffe_pre"] + 0.6],
                  [f"c({v:+d})" for v in kk] + ["DFE b1"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set(ylabel="weight", title="equalizer: Init design vs LMS (c(+1) / b1 trade off)")
    ax.legend(fontsize=7)

    ax = axs[2, 1]
    labels = [r[0] for r in table]
    xx = np.arange(len(table))
    floor = 1e-14
    cnt = np.array([r[1] for r in table])
    ax.bar(xx - 0.27, np.maximum([r[3] for r in table], floor), 0.27, label="Q-fit (time domain)")
    ax.bar(xx, np.maximum(np.where(cnt > 0, cnt, [r[2] for r in table]), floor), 0.27,
           color=["C3" if c > 0 else "none" for c in cnt], edgecolor="C3",
           label="counted (hollow: 0 errors, < 1/N)")
    ax.bar(xx + 0.27, np.maximum([r[4] for r in table], floor), 0.27, color="C7",
           label="statistical (Init design)")
    ax.set_yscale("log")
    ax.set_ylim(floor, 1)
    ax.set_xticks(xx, labels, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="BER", title=f"Rx equalizer, adapted from cold "
                               f"({base['noise'] * 1e3:.0f} mV rms input noise)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axs[2, 2]
    for (lab, sweep_runs), col in zip(noise_runs.items(), ("C1", "C3")):
        ax.loglog(sig * 1e3, np.maximum([r.td.qfit["ber_est"] for r in sweep_runs], 1e-30),
                  "--", color=col, label=f"{lab}: Q-fit")
        plot_counted(ax, sig, [r.td for r in sweep_runs], f"{lab}: counted", color=col)
    ax.set_ylim(1e-12, 1)
    ax.set(xlabel="Rx input noise [mV rms / sample]", ylabel="BER (Gray)",
           title="BER vs noise: the DFE tap cancels h1 without noise gain")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    c = cold.td.counted
    fig.suptitle(f"MRM PAM4 {2 * 53.125:.1f} Gb/s (53.125 GBd) -> PD / 18 GHz TIA -> IBIS-AMI "
                 f"pf_rx: 5-tap FFE + 1-tap DFE learned in AMI_GetWave from a cold start | "
                 f"counted BER {fmt_ber(c['ber'])} over {c['bits']} bits", fontsize=11)
    save(fig, "ibis_ami_mrm_pam4_adaptive.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
