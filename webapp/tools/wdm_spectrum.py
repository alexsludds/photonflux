"""Bus-output optical spectrum of the DWDM CPO MRM link (example 18).

Four lasers on a 200 GHz O-band grid share one baseband reference frame
(``ref_wavelength_nm`` = grid centre), so each is a distinct tone that
coexists on the shared bus. Four microring modulators in series each align
their resonance to one channel and modulate only that carrier; the other
three (200 GHz away, ~10 linewidths) pass by untouched. All four carriers are
modulated in a SINGLE coherent solve — no per-channel superposition. A
tunable box-top filter after the bus demuxes channel 1 to a photodiode.

This tool just runs the example once and plots the two optical-spectrum
probes (bus output, and channel 1 after the demux filter) exactly as the
web app's optical-spectrum probe computes them.

    python webapp/tools/wdm_spectrum.py           # -> wdm_spectrum.png

Run from the repo root (needs webapp on the path).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEBAPP = os.path.dirname(_HERE)                 # webapp/ (server modules)
sys.path.insert(0, _WEBAPP)                      # webapp modules (simulate, catalog)
sys.path.insert(0, os.path.dirname(_WEBAPP))    # repo root (photonflux/circulax)
import simulate

EX = os.path.join(_WEBAPP, "examples", "18_wdm_oband_testbench.json")


def _channel_wavelengths(sch: dict) -> list[float]:
    """Grid channels = each ring's aligned resonance minus its flank bias.

    The example biases every resonance the same small amount above its
    channel; recover the channels as the sorted ring wavelengths rounded to
    the grid. Falls back to the lasers if needed.
    """
    lams = sorted(float(v["settings"]["wavelength_nm"])
                  for v in sch["instances"].values()
                  if v["type"] == "cw_laser")
    return [round(x, 4) for x in lams]


def run_example():
    doc = json.load(open(EX))
    r = simulate.run(doc)
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))
    lams = _channel_wavelengths(doc["schematic"])
    # the web app returns one extra_plot per spectrum probe (bus, ch1)
    names = ["bus"] + [f"ch{k + 1}" for k in range(len(lams))]
    specs, tdom = {}, {}
    for ep, name in zip(r.get("extra_plots", []), names):
        specs[name] = (np.asarray(ep["x"]),
                       np.asarray(ep["traces"][0]["values"]))
    tv = np.asarray(r["x"])
    for tr in r.get("traces", []):
        # trace values are already in physical units: optical probes carry
        # |E|^2 in mW, electrical ones volts (simulate._trace)
        tdom[tr["name"]] = (tv, np.asarray(tr["values"]))
    return lams, specs, tdom, r


def main():
    lams, specs, tdom, r = run_example()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(lams)
    colors = ["#8fd18f", "#6ecbf5", "#e6c86e", "#f08fb0"]
    fig = plt.figure(figsize=(12, 2.2 * (n + 1)), dpi=120)
    gs = fig.add_gridspec(n + 1, 2)

    # top row: the bus spectrum (all channels), spanning both columns
    axb = fig.add_subplot(gs[0, :])
    x, y = specs["bus"]; o = np.argsort(x)
    axb.plot(x[o], y[o], color="#e6862c", lw=1.0)
    for k, lam in enumerate(lams):
        axb.axvline(lam, color="#888", ls=":", lw=0.7)
        axb.text(lam, 1.5, f"ch{k+1}\n{lam:.3f}", ha="center", va="bottom",
                 fontsize=7, color="#6ecbf5")
    axb.set(xlim=(lams[0] - 0.9, lams[-1] + 0.9), ylim=(-60, 6),
            ylabel="dB", title="bus output — all 4 channels (single solve)")
    axb.grid(alpha=0.2)

    # per-channel: drop spectrum (left) + received electrical data (right)
    for k in range(n):
        ch = f"ch{k+1}"; col = colors[k % len(colors)]
        axs = fig.add_subplot(gs[k + 1, 0])
        x, y = specs[ch]; o = np.argsort(x)
        axs.plot(x[o], y[o], color=col, lw=1.0)
        for lam in lams:
            axs.axvline(lam, color="#888", ls=":", lw=0.6)
        axs.set(xlim=(lams[0] - 0.9, lams[-1] + 0.9), ylim=(-60, 6),
                ylabel="dB", title=f"{ch} drop spectrum ({lams[k]:.3f} nm)")
        axs.grid(alpha=0.2)
        axt = fig.add_subplot(gs[k + 1, 1])
        vname = f"vout{k+1}"
        if vname in tdom:                      # PD receiver output [V]
            t, v = tdom[vname]; m = t < 3e-9
            axt.plot(t[m] * 1e9, v[m] * 1e3, color=col, lw=0.8)
            axt.set(ylabel="mV", title=f"{ch} received data (PD out)")
        else:                                  # fall back: drop power [mW]
            t, p = tdom[ch]; m = t < 3e-9
            axt.plot(t[m] * 1e9, p[m], color=col, lw=0.8)
            axt.set(ylabel="mW", title=f"{ch} drop power (time)")
        if k == n - 1:
            axs.set_xlabel("wavelength [nm]")
            axt.set_xlabel("time [ns]")
    fig.suptitle("DWDM CPO link — 4-lambda microring TX + 4 add-drop filter "
                 "demux (200 GHz O-band grid)")
    fig.tight_layout()
    out = os.path.join(os.getcwd(), "wdm_spectrum.png")
    fig.savefig(out)
    print("channels (nm):", lams, "saved", out)


if __name__ == "__main__":
    main()
