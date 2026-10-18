#!/usr/bin/env python3
"""Freely available IBIS-AMI models, run through photonflux on one channel.

Every model ``python -m photonflux ami list`` finds, side by side
(examples/ibis_ami/README.md has the provenance and licenses):

* ``pf_tx`` / ``pf_rx`` -- photonflux's own (BSD-3, models/ibis_ami).
* ``ibisami.example_tx`` / ``ibisami.example_rx`` -- David Banas's
  public-domain ibisami infrastructure (capn-freako/ibisami, BSD-3), built
  from source with a one-line fix: its Rx CTLE kept AMI_Init's filter state
  into AMI_GetWave, which rang the output to ~1e5 V.
* ``sisoft.IBIS_AMI_Tx`` -- SiSoft's 2007 IBIS-AMI Eval Toolkit Tx (the
  IBIS ATM reference model), compiled natively from the kit's C source, and
  ``sisoft.IBIS_AMI_Tx@linux-x86`` -- the kit's *original* 2008 Linux i386
  binary, run out-of-process in a ``linux/386`` container through
  ``ami_host`` (needs Docker; skipped otherwise). The two must agree.
* anything dropped into ``examples/ibis_ami/vendor/`` (vendor kits you
  download under their own licenses) is listed and characterized too.

Panels: each Tx's pulse response through the channel (``AMI_Init`` vs
``AMI_GetWave`` -- the IBIS "dual model" consistency check), and each Rx
with each Tx: statistical eye height and counted BER. The ibisami Rx has no
CDR (its clock ticks sit on the UI grid), so its runs get the channel delay
that centers its fixed clock in the eye, as a CDR-less model needs.

    python examples/ibis_ami/public_models.py
        -> out/ibis_ami_public_models.png
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ensure_built, fmt_ber, save  # noqa: E402

from photonflux import ibisami, serdes  # noqa: E402

UI = 1 / 10e9
NSPUI = 32
TS = UI / NSPUI

# per-model settings for this 10 Gb/s, ~13 dB channel
TX_SET = {
    "pf_tx": {"tx_tap_p1": -0.15},
    "ibisami.example_tx": {"tx_tap_np1": 0, "tx_tap_nm1": 4},
    "sisoft.IBIS_AMI_Tx": {"tap_filter.0": 0.85, "tap_filter.1": -0.15},
}
RX_SET = {
    "pf_rx": {"ctle_fp1": 5e9, "dfe_ntaps": 5},
    "ibisami.example_rx": {"ctle_mode": 1, "ctle_mag": 6.0, "ctle_freq": 5e9,
                           "dfe_mode": 2, "dfe_vout": 0.25, "dfe_gain": 0.05},
}


def settings(table: dict, name: str) -> dict:
    return table.get(name.split("@")[0], {})


def runnable(e: ibisami.LibraryEntry, docker: bool) -> str | None:
    if e.runs_here:
        return "auto"
    if e.runs_in_container and docker:
        return "container"
    return None


def dual_check(entry, host, ch):
    """Pulse response through the channel from AMI_Init (impulse) and from
    AMI_GetWave (a single-UI pulse) -- they should agree for a dual model."""
    m = entry.load(host)
    m.params.update(settings(TX_SET, entry.name))
    r = m.init(ch.impulse, TS, UI)
    p_init = np.convolve(r.impulse[0], np.ones(NSPUI))[: len(ch.impulse)] * TS
    p_gw = None
    if m.getwave_exists:
        x = np.zeros(64 * NSPUI)
        x[4 * NSPUI: 5 * NSPUI] = 1.0      # one UI, then quiet
        y = m.getwave(x - 0.0, block_bits=64).wave
        p_gw = np.convolve(y, ch.impulse)[: len(y)] * TS
    where = m.where
    m.close()
    return p_init, p_gw, where, r.message.strip().splitlines()[-1][:80] if r.message else ""


def main(argv=None) -> int:
    ensure_built()
    docker = ibisami.docker_available()
    lib = ibisami.library()
    print(f"{len(lib)} IBIS-AMI executables found (Docker {'up' if docker else 'not available'}):")
    for e in lib:
        how = runnable(e, docker)
        print(f"  {e.name:32s} {e.kind}  {e.source:10s} {', '.join(map(str, e.platforms)):24s} "
              f"{'runs ' + ('in-process' if how == 'auto' else 'in a container') if how else 'cannot run here'}")

    ch = serdes.backplane(0.9, ts=TS, ui=UI, nfft=1 << 14).resized(128 * NSPUI)
    f, H = ch.s21()
    il = 20 * np.log10(abs(H[np.argmin(abs(f - 5e9))]))
    print(f"channel: {ch.name}, {il:.1f} dB at 5 GHz")

    txs = [e for e in lib if e.kind == "tx" and runnable(e, docker)]
    rxs = [e for e in lib if e.kind == "rx" and runnable(e, docker)]

    # -- Tx models: Init vs GetWave, and native vs original binary ----------
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    ax = axs[0, 0]
    pulses = {}
    for k, e in enumerate(txs):
        t0 = time.time()
        p_init, p_gw, where, msg = dual_check(e, runnable(e, docker), ch)
        pulses[e.name] = (p_init, p_gw)
        t = np.arange(len(p_init)) * TS / UI
        ax.plot(t, p_init, color=f"C{k}", label=f"{e.name}: Init ({where})")
        if p_gw is not None:
            # the GetWave pulse left the Tx at UI 4; Init's impulse starts at 0
            t_gw = np.arange(len(p_gw)) * TS / UI - 4
            ax.plot(t_gw, p_gw, color=f"C{k}", ls="--", lw=1,
                    label=f"{e.name}: GetWave")
            n = min(len(p_init), len(p_gw) - 4 * NSPUI)
            dev = np.max(np.abs(p_init[:n] - p_gw[4 * NSPUI: 4 * NSPUI + n])) / np.max(np.abs(p_init))
            print(f"  {e.name}: Init vs GetWave pulse differ by {dev:.1e} of peak "
                  f"({time.time() - t0:.1f} s, {where})")
    ax.set(xlim=(0, 20), xlabel="UI", ylabel="V (1 V, 1 UI pulse)",
           title="Tx models through the channel: AMI_Init (solid) vs AMI_GetWave (dashed)")
    ax.legend(fontsize=7)

    ax = axs[0, 1]
    native, orig = "sisoft.IBIS_AMI_Tx", "sisoft.IBIS_AMI_Tx@linux-x86"
    if native in pulses and orig in pulses:
        a, b = pulses[native], pulses[orig]
        d_init = np.max(np.abs(a[0] - b[0]))
        d_gw = np.max(np.abs(a[1] - b[1])) if a[1] is not None and b[1] is not None else np.nan
        t = np.arange(len(a[0])) * TS / UI
        ax.plot(t, a[0], label="native build from the kit's C source")
        ax.plot(t, b[0], "--", label="original 2008 i386 binary (linux/386 container)")
        ax.set_title(f"SiSoft Tx: native vs original binary, max |diff| Init "
                     f"{d_init:.1e} V, GetWave {d_gw:.1e} V")
        print(f"SiSoft Tx native vs original i386 binary: max |diff| Init {d_init:.2e}, "
              f"GetWave {d_gw:.2e}")
    else:
        ax.text(0.5, 0.5, "original SiSoft i386 binary needs Docker\n"
                "(python -m photonflux ami list)", ha="center", va="center",
                transform=ax.transAxes)
    ax.set(xlim=(0, 12), xlabel="UI", ylabel="V")
    ax.legend(fontsize=8)

    # -- every Tx x Rx pairing ----------------------------------------------
    combos = []
    for tx_e in txs:
        if "@" in tx_e.name:          # the original binary is the native's twin
            continue
        for rx_e in rxs:
            tx = tx_e.load(runnable(tx_e, docker))
            rx = rx_e.load(runnable(rx_e, docker))
            tx.params.update(settings(TX_SET, tx_e.name))
            rx.params.update(settings(RX_SET, rx_e.name))
            chn = ch
            if rx_e.name.startswith("ibisami."):
                # no CDR: delay the channel so its fixed mid-UI tick is centered
                probe = serdes.time_domain_link(tx, ch, rx, UI, nsym=6000,
                                                rx_params={"dfe_mode": 0})
                shift = int(round(((0.5 - probe.sampling_phase_ui) % 1.0) * NSPUI))
                chn = serdes.Channel(np.roll(ch.impulse, shift), TS, name=ch.name)
                rx.params.update(settings(RX_SET, rx_e.name))
            st = serdes.statistical_link(tx, chn, rx, UI, noise_rms=3e-3)
            td = serdes.time_domain_link(tx, chn, rx, UI, nsym=20000, noise_rms=3e-3)
            h = st.eye.eye_height(1e-12)
            combos.append((tx_e.name, rx_e.name, h, td.counted, td))
            print(f"  {tx_e.name:22s} -> {rx_e.name:20s}: stat eye {h * 1e3:6.1f} mV @1e-12, "
                  f"counted {fmt_ber(td.counted['ber'])} over {td.counted['bits']} bits")
            tx.close()
            rx.close()

    ax = axs[1, 0]
    lab = [f"{a}\n-> {b}" for a, b, *_ in combos]
    ax.barh(lab, [c[2] * 1e3 for c in combos], color=[f"C{i}" for i in range(len(combos))])
    for i, c in enumerate(combos):
        ax.text(c[2] * 1e3, i, f"  BER {fmt_ber(c[3]['ber'])}", va="center", fontsize=8)
    ax.set_xlabel("statistical eye height @ 1e-12 [mV] (text: counted BER)")
    ax.set_title("every Tx x Rx pairing, 10 Gb/s")
    ax.tick_params(axis="y", labelsize=7)

    ax = axs[1, 1]
    for i, (a, b, _h, _c, td) in enumerate(combos):
        keys = [k for k, v in (td.rx_trace.trace[-1] if td.rx_trace.trace else {}).items()
                if "dfe" in k]
        if not keys:
            continue
        t_ui = td.rx_trace.trace_t / UI
        if keys == ["dfe_taps"]:
            ax.plot(t_ui, td.adaptation("dfe_taps", 0)[1], color=f"C{i}",
                    label=f"{a} -> {b}: b1")
        else:                         # ibisami reports dfe_tap1..dfe_tapN
            ax.plot(t_ui, td.adaptation(sorted(keys)[0])[1], color=f"C{i}", ls="--",
                    label=f"{a} -> {b}: {sorted(keys)[0]}")
    ax.set(xlabel="time [UI]", ylabel="first DFE tap (model's own units)",
           title="DFE adaptation, read from each model's AMI_parameters_out")
    ax.legend(fontsize=7)
    fig.suptitle(f"Freely available IBIS-AMI models in photonflux ({ch.name}, "
                 f"{il:.1f} dB at Nyquist)", fontsize=12)
    save(fig, "ibis_ami_public_models.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
