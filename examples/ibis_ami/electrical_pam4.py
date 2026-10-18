#!/usr/bin/env python3
"""106 Gb/s PAM4 (53.125 GBd) through IBIS-AMI Tx and Rx models.

The 100G-per-lane electrical case, where the receiver is a DSP: ``pf_rx``
runs with its UI-spaced FFE (8 taps, 2 pre-cursors; MMSE-designed in
``AMI_Init``, LMS-adapted in ``AMI_GetWave``), a single DFE tap (more would
propagate PAM4 errors) and a baud-rate Mueller-Muller CDR -- the
ADC-DSP receiver architecture. ``pf_tx`` adds a 3-tap FFE.

1. **Tx FFE optimization** (statistical flow): a (c-1, c+1) grid, each
   point a Tx ``AMI_Init`` + Rx ``AMI_Init`` (which re-designs its CTLE,
   FFE and DFE for that Tx) + PAM4 statistical eye; the score is the worst
   of the three eyes' heights at 1e-6.
2. **PAM4 statistical eye** with per-eye BER contours and bathtubs.
3. **Time domain** with two FEXT aggressors' worth of noise: counted bit
   errors (Gray-coded) after ``Ignore_Bits``, a CDR-referenced eye, and the
   adaptation trace of every FFE tap, the DFE tap, the data level and the
   CDR phase, for the Mueller-Muller CDR and, for comparison, bang-bang.
4. **Statistical vs counted BER** across Rx input noise, both flows
   jitter-free: the statistical flow applies Tx jitter as sampling jitter
   (the usual, pessimistic reading of ``Tx_Rj``/``Tx_Dj``), while in the
   time domain the channel filters it -- on 0.06 UI-wide PAM4 eyes that
   difference alone is several x in BER. With the Rx frozen at its Init
   design (fixed FFE/DFE, fixed phase) the counted BER lands on the
   statistical curve; adapting, it beats it: Init's MMSE design sees only
   the victim's pulse, while GetWave's LMS also minimizes the crosstalk it
   actually receives.

    python examples/ibis_ami/electrical_pam4.py
    python examples/ibis_ami/electrical_pam4.py --length 0.25 --ffe 12
        -> out/ibis_ami_pam4.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import density_of, ensure_built, fmt_ber, model, save  # noqa: E402

from photonflux import serdes  # noqa: E402


def worst_log_ber(eye):
    """log10 of the worst eye's best BER (lower is better; never saturates
    the way an eye height at a fixed BER does when every eye is closed)."""
    return float(np.log10(max(eye.min_ber(), 1e-300)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baud", type=float, default=53.125e9)
    ap.add_argument("--length", type=float, default=0.35, help="backplane [m]")
    ap.add_argument("--xtalk", type=float, default=-40.0, help="FEXT [dB]; 0 = none")
    ap.add_argument("--ffe", type=int, default=8, help="Rx FFE taps")
    ap.add_argument("--sigma", type=float, default=3e-3,
                    help="Rx input noise [V rms per sample] for the main run")
    ap.add_argument("--nsym", type=int, default=60000)
    args = ap.parse_args(argv)

    ensure_built()
    ui = 1.0 / args.baud
    nspui = 32
    ts = ui / nspui
    rj, dj = 0.008, 0.02
    ch = serdes.backplane(args.length, ts=ts, ui=ui, nfft=1 << 15).resized(256 * nspui)
    if args.xtalk:
        ch = ch.with_fext(args.xtalk, args.baud / 2, n=2)
    f, H = ch.s21()
    il = 20 * np.log10(abs(H[np.argmin(abs(f - args.baud / 2))]))
    print(f"{ch.name}: {il:.1f} dB at Nyquist ({args.baud / 2e9:.2f} GHz)")

    tx = model("pf_tx")
    rx_kw = dict(ctle_fp1=args.baud / 2, ffe_mode=2, ffe_ntaps=args.ffe,
                 ffe_pre=2, dfe_ntaps=1, cdr_mode=2, cdr_kp=2e-3)
    rx = model("pf_rx", **rx_kw)
    dens = density_of(args.sigma, ts)

    # -- 1. Tx FFE optimization -------------------------------------------
    t0 = time.time()
    pre = np.round(np.arange(0.0, -0.25 - 1e-9, -0.05), 3)
    post = np.round(np.arange(0.0, -0.30 - 1e-9, -0.05), 3)
    grid = np.zeros((len(pre), len(post)))
    for i, cm1 in enumerate(pre):
        for j, cp1 in enumerate(post):
            st = serdes.statistical_link(
                tx, ch, rx, ui, modulation="PAM4", noise_density=dens,
                rj_ui=rj, dj_ui=dj, nv=301,
                tx_params={"tx_tap_m1": cm1, "tx_tap_p1": cp1})
            grid[i, j] = worst_log_ber(st.eye)
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    tx.params.update(tx_tap_m1=pre[i], tx_tap_p1=post[j])
    print(f"Tx FFE grid ({grid.size} points) in {time.time() - t0:.1f} s: best "
          f"c-1 = {pre[i]}, c+1 = {post[j]} (worst eye's BER 1e{grid[i, j]:.1f})")

    # -- 2. statistical eye --------------------------------------------------
    st = serdes.statistical_link(tx, ch, rx, ui, modulation="PAM4",
                                 noise_density=dens, rj_ui=rj, dj_ui=dj)
    print(f"statistical: BER {st.eye.bit_error_rate():.2e}; {st.rx_message}")
    for e, d in enumerate(st.eye.summary()["eyes"]):
        print(f"  eye {e}: {d['height_v@1e-06'] * 1e3:5.1f} mV x "
              f"{d['width_ui@1e-06']:.3f} UI @ 1e-6, threshold "
              f"{d['threshold_v'] * 1e3:+.1f} mV")

    # -- 3. time domain, Mueller-Muller and bang-bang CDR --------------------
    runs = {}
    for mode, label in ((2, "Mueller-Muller"), (1, "bang-bang")):
        t0 = time.time()
        r = serdes.time_domain_link(tx, ch, rx, ui, modulation="PAM4",
                                    nsym=args.nsym, noise_rms=args.sigma,
                                    rj_ui=rj, dj_ui=dj,
                                    rx_params={"cdr_mode": mode})
        runs[label] = r
        c = r.counted
        print(f"time domain ({label} CDR): {c['bit_errors']} / {c['bits']} bit "
              f"errors (BER {fmt_ber(c['ber'])}, SER {fmt_ber(c['ser'])}), Q-fit "
              f"{r.qfit['ber_est']:.2e}, sampling at "
              f"{st.eye.relative_phase(r.sampling_phase_ui):+.3f} UI from the "
              f"pulse peak, {time.time() - t0:.1f} s")
    rx.params["cdr_mode"] = 2
    td = runs["Mueller-Muller"]

    # -- 4. statistical vs counted ------------------------------------------
    g = st.noise_rms / args.sigma
    sig = np.geomspace(args.sigma * 2, args.sigma * 10, 14)
    xt = [np.convolve(a, np.ones(nspui))[: len(a)] * ts for a in st.rx_aggressors]
    # jitter-free in both flows: the statistical flow applies Tx jitter as
    # sampling jitter (the usual, pessimistic IBIS reading) while in the
    # time domain the channel filters it, which on these narrow PAM4 eyes
    # would dominate the comparison
    stat_ber = [serdes.stat_eye(st.pulse, ts, ui, levels=4, noise_rms=g * s,
                                nv=301, crosstalk=xt).bit_error_rate() for s in sig]
    frozen = {"ffe_mode": 1, "dfe_mode": 1, "cdr_mode": 0}
    pts, counted, counted_frozen = [], [], []
    for s in sig[::3][1:]:
        r = serdes.time_domain_link(tx, ch, rx, ui, modulation="PAM4",
                                    nsym=120000, noise_rms=s, seed=5)
        rf = serdes.time_domain_link(tx, ch, rx, ui, modulation="PAM4",
                                     nsym=120000, noise_rms=s, seed=5,
                                     rx_params=frozen)
        rx.params.update(ffe_mode=2, dfe_mode=2, cdr_mode=2)
        pts.append(s)
        counted.append(r.counted["ber"])
        counted_frozen.append(rf.counted["ber"])
        print(f"  sigma_in {s * 1e3:5.1f} mV: counted {fmt_ber(r.counted['ber'])} "
              f"adaptive, {fmt_ber(rf.counted['ber'])} frozen at the Init design")

    # -- figure -------------------------------------------------------------
    fig, axs = plt.subplots(3, 4, figsize=(21, 13), layout="constrained")
    ax = axs[0, 0]
    ax.plot(f / 1e9, 20 * np.log10(np.abs(H) + 1e-12), label="victim")
    for k, a in enumerate(ch.aggressors):
        ax.plot(f / 1e9, 20 * np.log10(np.abs(np.fft.rfft(a * ts)) + 1e-12),
                color="tab:red", lw=0.8, label="FEXT" if k == 0 else None)
    ax.axvline(args.baud / 2e9, color="k", ls=":", lw=0.8)
    ax.set(xlim=(0, 2 * args.baud / 1e9), ylim=(-80, 3), xlabel="frequency [GHz]",
           ylabel="dB", title=f"channel: {il:.1f} dB at Nyquist")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    m = ax.pcolormesh(post, pre, np.maximum(grid, -40), shading="nearest",
                      cmap="viridis_r")
    ax.plot(post[j], pre[i], "r*", ms=14)
    fig.colorbar(m, ax=ax, label="log10 BER of the worst eye")
    ax.set(xlabel="Tx c(+1)", ylabel="Tx c(-1)", title="Tx FFE optimization (Init flow)")

    ax = axs[0, 2]
    pk = int(np.argmax(st.pulse))
    k = np.arange(-4, 12)
    for p, lab, mk in ((st.pulse_unequalized, "channel", "o"),
                       (st.pulse, "Tx + Rx Init", "s")):
        q = int(np.argmax(p)) if lab == "channel" else pk
        idx = q + k * nspui
        ax.plot(k, p[idx] / p[q], mk + "-", ms=4, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="cursor [UI]", ylabel="normalized cursor",
           title="pulse cursors before / after equalization")
    ax.legend(fontsize=8)

    ax = axs[0, 3]
    w_init = st.rx_params.get("ffe_taps", [])
    w_fin = td.rx_trace.trace[-1].get("ffe_taps", [])
    kk = np.arange(len(w_init)) - 2
    ax.stem(kk - 0.1, w_init, linefmt="C0-", markerfmt="C0o", basefmt=" ",
            label="Init (MMSE)")
    ax.stem(kk + 0.1, w_fin, linefmt="C1-", markerfmt="C1s", basefmt=" ",
            label="GetWave (LMS, end)")
    ax.set(xlabel="FFE tap [UI from main]", ylabel="weight",
           title="Rx FFE: Init design vs adapted")
    ax.legend(fontsize=8)

    serdes.plot_stat_eye(axs[1, 0], st.eye, title="PAM4 statistical eye")
    serdes.plot_bathtubs(axs[1, 1], axs[1, 2], st.eye)
    cdr_at = st.eye.relative_phase(td.sampling_phase_ui)
    axs[1, 1].semilogy(td.bathtub["offset_ui"] + cdr_at, td.bathtub["ber"], "ko",
                       ms=3, label="time domain (all eyes)")
    axs[1, 1].axvline(cdr_at, color="k", ls=":", lw=0.8)
    axs[1, 1].legend(fontsize=7)
    axs[1, 1].set_title("timing bathtubs")
    axs[1, 2].legend(fontsize=7)
    axs[1, 2].set_title("voltage bathtubs")

    ax = axs[1, 3]
    for label, r in runs.items():
        t_ui, ph = r.adaptation("cdr_phase_ui")
        ax.plot(t_ui, ph, label=f"{label}: BER {fmt_ber(r.counted['ber'])}")
    ax.axvline(td.skip, color="k", ls=":", lw=0.8)
    ax.set(xlabel="time [UI]", ylabel="sampling phase [UI]",
           title="CDR phase (absolute time axis)")
    ax.legend(fontsize=8)

    serdes.plot_eye_density(axs[2, 0], td, title="time domain: CDR-referenced eye")

    ax = axs[2, 1]
    t_ui, _ = td.adaptation("level")
    for n_ in range(len(w_init)):
        if n_ == 2:
            continue
        ax.plot(t_ui, td.adaptation("ffe_taps", n_)[1], lw=1, label=f"w{n_ - 2:+d}")
    ax.plot(t_ui, td.adaptation("dfe_taps", 0)[1], "k-", lw=1.5, label="DFE b1")
    ax.axvline(td.skip, color="k", ls=":", lw=0.8)
    ax.set(xlabel="time [UI]", ylabel="tap", title="Rx FFE / DFE adaptation (LMS)")
    ax.legend(fontsize=7, ncol=3)

    ax = axs[2, 2]
    ax.plot(t_ui, td.adaptation("level")[1] * 1e3, label="data level (slicer ref)")
    ax.set(xlabel="time [UI]", ylabel="mV", title="data level and AGC")
    ax2 = ax.twinx()
    ax2.plot(t_ui, td.adaptation("agc_gain")[1], "C1--", label="AGC gain")
    ax2.set_ylabel("AGC gain")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[2, 3]
    ax.loglog(sig * 1e3, stat_ber, "-", label="statistical (Init)")
    ax.loglog(np.asarray(pts) * 1e3, np.maximum(counted_frozen, 1e-7), "s",
              mfc="none", label="counted, GetWave frozen at Init's design")
    ax.loglog(np.asarray(pts) * 1e3, np.maximum(counted, 1e-7), "o",
              label="counted, GetWave adapting (LMS FFE/DFE, MM CDR)")
    ax.set(xlabel="Rx input noise [mV rms / sample]", ylabel="BER (Gray)",
           title="statistical vs time-domain BER (no jitter)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    c = td.counted
    fig.suptitle(f"IBIS-AMI PAM4 {2 * args.baud / 1e9:.1f} Gb/s: pf_tx "
                 f"(c-1 {pre[i]}, c+1 {post[j]}) -> {ch.name} -> pf_rx (CTLE + "
                 f"{args.ffe}-tap FFE + 1-tap DFE + MM CDR) | counted BER "
                 f"{fmt_ber(c['ber'])} over {c['bits']} bits", fontsize=11)
    save(fig, "ibis_ami_pam4.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
