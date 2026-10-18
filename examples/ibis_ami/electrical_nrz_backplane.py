#!/usr/bin/env python3
"""32 Gb/s NRZ over a backplane, through IBIS-AMI Tx and Rx models.

The SerDes is photonflux's open-source IBIS-AMI pair (``models/ibis_ami``,
loaded through PyAMI by ``photonflux.ibisami``):

* **Tx** ``pf_tx`` -- 6-tap FFE with the PCIe 3.0+ coefficient presets
  P0..P9, 1.0 swing.
* **Rx** ``pf_rx`` -- Init-adapted CTLE, AGC, 8-tap DFE (zero-forced in
  Init, LMS in GetWave), bang-bang CDR.

The channel is ``photonflux.serdes.backplane``: a causal 0.7 m stripline
(skin effect + Djordjevic-Sarkar dielectric) between package stubs and pad
capacitance, with a 1.5 mm via stub notch -- 27.7 dB loss at Nyquist -- plus
two FEXT aggressors (``--xtalk``, dB at Nyquist; their impulse rows ride
through both AMI_Init calls).

What it does, the way an IBIS-AMI channel simulator would:

1. **Link training (statistical flow)**: every Tx preset goes through Tx
   ``AMI_Init`` -> Rx ``AMI_Init`` (which adapts its CTLE and DFE to that
   preset) -> statistical eye; the preset with the tallest 1e-12 eye wins.
2. **Statistical eye** of the winner: BER contours, timing and voltage
   bathtubs, eye height / width at 1e-6 and 1e-12, crosstalk included.
3. **Time-domain flow**: PRBS-15 -> Tx ``AMI_GetWave`` -> channel ->
   input noise -> Rx ``AMI_GetWave``. The Rx's own CDR samples its own
   output; errors are counted after ``Ignore_Bits``, and its
   ``AMI_parameters_out`` after every call is the adaptation trace (DFE
   taps, CDR phase, data level).
   The eye is the Rx's GetWave output -- the DFE summer, i.e. the slicer
   input -- so it steps at each data edge (+/-0.5 UI) where the feedback
   switches; only its opening at the clock is meaningful, as on silicon.
4. **Cross-check of the two flows**: counted BER vs the statistical
   prediction across Rx input noise -- the GetWave adaptation and CDR land
   close to, not exactly on, Init's design, which is the classic
   statistical/time-domain correlation exercise.

    python examples/ibis_ami/electrical_nrz_backplane.py
    python examples/ibis_ami/electrical_nrz_backplane.py --length 0.9 --xtalk -35
        -> out/ibis_ami_nrz_backplane.png
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rate", type=float, default=32e9, help="bit rate [b/s]")
    ap.add_argument("--length", type=float, default=0.7, help="backplane [m]")
    ap.add_argument("--stub", type=float, default=1.5, help="via stub [mm]")
    ap.add_argument("--xtalk", type=float, default=-40.0,
                    help="FEXT at Nyquist [dB] (two aggressors); 0 = none")
    ap.add_argument("--sigma", type=float, default=4e-3,
                    help="Rx input noise [V rms per sample] for the main run")
    ap.add_argument("--nsym", type=int, default=60000)
    args = ap.parse_args(argv)

    ensure_built()
    ui = 1.0 / args.rate
    nspui = 32
    ts = ui / nspui
    rj, dj = 0.01, 0.03                    # Tx jitter: 0.01 UI rms, 0.03 UI pk-pk

    ch = serdes.backplane(args.length, ts=ts, ui=ui, via_stub_mm=args.stub,
                          nfft=1 << 15).resized(384 * nspui)
    if args.xtalk:
        ch = ch.with_fext(args.xtalk, args.rate / 2, n=2)
    f, H = ch.s21()
    il_nyq = 20 * np.log10(abs(H[np.argmin(abs(f - args.rate / 2))]))
    print(f"{ch.name}: {il_nyq:.1f} dB at Nyquist ({args.rate / 2e9:.0f} GHz)")

    tx = model("pf_tx")
    rx = model("pf_rx", ctle_fp1=args.rate / 2, dfe_ntaps=8)
    dens = density_of(args.sigma, ts)

    # -- 1. link training over the Tx presets (statistical flow) ----------
    t0 = time.time()
    sweep = []
    for preset in range(-1, 10):
        st = serdes.statistical_link(tx, ch, rx, ui, noise_density=dens,
                                     rj_ui=rj, dj_ui=dj,
                                     tx_params={"tx_preset": preset})
        sweep.append((preset, st.eye.eye_height(1e-12), st.eye.eye_width(1e-12),
                      st.rx_params.get("ctle_peaking_db", 0.0)))
    best = max(sweep, key=lambda r: (r[1], r[2]))
    print(f"link training: {len(sweep)} presets in {time.time() - t0:.1f} s; "
          f"best {'P%d' % best[0] if best[0] >= 0 else 'none'} "
          f"(eye {best[1] * 1e3:.0f} mV / {best[2]:.3f} UI @ 1e-12, "
          f"Rx CTLE {best[3]:g} dB)")
    tx.params["tx_preset"] = best[0]

    # -- 2. statistical eye of the trained link ---------------------------
    st = serdes.statistical_link(tx, ch, rx, ui, noise_density=dens,
                                 rj_ui=rj, dj_ui=dj)
    summ = st.eye.summary()
    print(f"statistical: BER {summ['ber']:.2e}, noise at slicer "
          f"{st.noise_rms * 1e3:.2f} mV rms; " + st.rx_message)

    # -- 3. time-domain flow ----------------------------------------------
    t0 = time.time()
    td = serdes.time_domain_link(tx, ch, rx, ui, nsym=args.nsym,
                                 noise_rms=args.sigma, rj_ui=rj, dj_ui=dj)
    c = td.counted
    print(f"time domain: {c['bits']} bits after {td.skip} ignored, "
          f"{c['bit_errors']} errors (BER {fmt_ber(c['ber'])}), Q-fit BER "
          f"{td.qfit['ber_est']:.2e}, {time.time() - t0:.1f} s")
    final = td.rx_trace.trace[-1]
    print("  Rx state at the end:", {k: final[k] for k in
                                      ("ctle_peaking_db", "level", "cdr_phase_ui",
                                       "dfe_taps")})

    # -- 4. statistical vs counted BER across input noise ------------------
    # slicer noise per unit input sigma, from the Rx's own linear response
    g = st.noise_rms / args.sigma
    sig_grid = []
    for target in (1e-2, 1e-3, 1e-4, 1e-5):
        lo, hi = 1e-5, 1.0
        for _ in range(40):              # bisect the statistical BER
            mid = np.sqrt(lo * hi)
            e = serdes.stat_eye(st.pulse, ts, ui, noise_rms=g * mid, rj_ui=rj,
                                dj_ui=dj, nv=301)
            lo, hi = (mid, hi) if e.bit_error_rate() < target else (lo, mid)
        sig_grid.append(np.sqrt(lo * hi))
    sig_fine = np.geomspace(sig_grid[0] * 1.1, sig_grid[-1] * 0.7, 12)
    stat_ber = [serdes.stat_eye(st.pulse, ts, ui, noise_rms=g * s, rj_ui=rj,
                                dj_ui=dj, nv=301).bit_error_rate() for s in sig_fine]
    counted, qfit = [], []
    for s in sig_grid:
        r = serdes.time_domain_link(tx, ch, rx, ui, nsym=150000, noise_rms=s,
                                    rj_ui=rj, dj_ui=dj, seed=3)
        counted.append(r.counted["ber"])
        qfit.append(r.qfit["ber_est"])
        print(f"  sigma_in {s * 1e3:6.2f} mV: counted {fmt_ber(r.counted['ber'])} "
              f"({r.counted['bit_errors']} errors), Q-fit {r.qfit['ber_est']:.2e}")

    # -- figure -------------------------------------------------------------
    fig, axs = plt.subplots(3, 3, figsize=(16, 13), layout="constrained")
    ax = axs[0, 0]
    ax.plot(f / 1e9, 20 * np.log10(np.abs(H) + 1e-12), label="victim S21")
    for k, a in enumerate(ch.aggressors):
        Ha = np.fft.rfft(a * ts)
        ax.plot(f / 1e9, 20 * np.log10(np.abs(Ha) + 1e-12), lw=0.8,
                label="FEXT" if k == 0 else None, color="tab:red", alpha=0.7)
    ax.axvline(args.rate / 2e9, color="k", ls=":", lw=0.8)
    ax.set_xlim(0, 2 * args.rate / 1e9)
    ax.set_ylim(-80, 3)
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("dB")
    ax.set_title(f"channel: {il_nyq:.1f} dB at Nyquist")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    tt = np.arange(len(st.pulse)) * ts / ui
    pk = int(np.argmax(st.pulse))
    for p, lab in ((st.pulse_unequalized, "channel"),
                   (np.convolve(st.tx_impulse, np.ones(nspui))[: len(tt)] * ts, "after Tx FFE"),
                   (st.pulse, "after Rx (Init)")):
        ax.plot(tt - pk / nspui, p / np.max(p), label=lab)
    ax.set_xlim(-4, 14)
    ax.set_xlabel("UI from equalized peak")
    ax.set_ylabel("normalized pulse response")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[0, 2]
    lab = ["off" if p < 0 else f"P{p}" for p, *_ in sweep]
    ax.bar(lab, [h * 1e3 for _, h, _, _ in sweep], color=[
        "tab:green" if r[0] == best[0] else "tab:blue" for r in sweep])
    for i, r in enumerate(sweep):
        ax.text(i, r[1] * 1e3 + 2, f"{r[3]:g}", ha="center", fontsize=7)
    ax.set_ylabel("eye height @ 1e-12 [mV]")
    ax.set_title("link training (labels: Rx CTLE dB from Init)")

    serdes.plot_stat_eye(axs[1, 0], st.eye, title="statistical eye (Init flow)")
    serdes.plot_bathtubs(axs[1, 1], axs[1, 2], st.eye, label="statistical")
    tub = td.bathtub
    cdr_at = st.eye.relative_phase(td.sampling_phase_ui)
    axs[1, 1].semilogy(tub["offset_ui"] + cdr_at, tub["ber"], "o", ms=3,
                       label="time domain (Q-fit vs phase)")
    axs[1, 1].axvline(cdr_at, color="tab:orange", ls=":", lw=1,
                      label="where the CDR samples")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].set_title("timing bathtub")
    axs[1, 2].set_title("voltage bathtub")

    serdes.plot_eye_density(axs[2, 0], td, title="time domain: CDR-referenced eye")

    ax = axs[2, 1]
    tui, _ = td.adaptation("level")
    for j in range(8):
        ax.plot(tui, td.adaptation("dfe_taps", j)[1], label=f"b{j + 1}")
    ax.axvline(td.skip, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time [UI]")
    ax.set_ylabel("DFE tap (x outer level)")
    ax2 = ax.twinx()
    ax2.plot(tui, td.adaptation("cdr_phase_ui")[1], "k--", lw=1, label="CDR phase")
    ax2.set_ylabel("CDR sampling phase [UI]")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_title("Rx GetWave adaptation (AMI_parameters_out per call)")

    ax = axs[2, 2]
    ax.loglog(np.asarray(sig_fine) * 1e3, stat_ber, "-", label="statistical (Init)")
    ax.loglog(np.asarray(sig_grid) * 1e3, np.maximum(counted, 1e-7), "o",
              label="counted (GetWave)")
    ax.loglog(np.asarray(sig_grid) * 1e3, qfit, "x", label="Q-fit (GetWave)")
    ax.set_xlabel("Rx input noise [mV rms / sample]")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("statistical vs time-domain BER")

    e0 = summ["eyes"][0]
    fig.suptitle(
        f"IBIS-AMI NRZ link {args.rate / 1e9:g} Gb/s, pf_tx "
        f"{'P%d' % best[0] if best[0] >= 0 else '(no EQ)'} -> {ch.name} -> pf_rx | "
        f"stat: H {e0['height_v@1e-12'] * 1e3:.0f} mV, W {e0['width_ui@1e-12']:.3f} UI "
        f"@1e-12 | counted BER {fmt_ber(c['ber'])} over {c['bits']} bits", fontsize=11)
    save(fig, "ibis_ami_nrz_backplane.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
