#!/usr/bin/env python3
"""Characterize any IBIS-AMI model: parameters, Init and GetWave responses.

The photonflux take on PyAMI's model tester, for any model in the library
(``python -m photonflux ami list``):

* its ``.ami`` parameter tree as PyAMI parses it (with photonflux's
  correction for List defaults);
* its own impulse, step and frequency response: ``AMI_Init`` on an ideal
  channel (photonflux's executor) and ``AMI_GetWave`` on a step;
* the same responses computed by **PyAMI itself**
  (``pyibisami.ami.model.AMIModel.get_responses``) on the same model and
  parameters -- the two toolchains must agree.

Models whose ``.ami`` names tap nodes with integers (SiSoft's ``tap_filter
(-1 ...) (0 ...)``) skip the PyAMI pass: PyAMI renames those nodes
(``pre1``/``post0``) in the string it hands ``AMI_Init``, which that model's
parser cannot read; photonflux sends the file's own names.

    python examples/ibis_ami/ami_model_tester.py               # pf_tx, pf_rx
    python examples/ibis_ami/ami_model_tester.py ibisami.example_rx \\
        --param ctle_mode=1 --param ctle_mag=8
        -> out/ibis_ami_tester_<model>.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ensure_built, model, save  # noqa: E402  (model: by name)

from photonflux import serdes  # noqa: E402


def characterize(name: str, overrides: dict, ui: float, nspui: int = 32) -> None:
    ts = ui / nspui
    m = model(name)
    m.params.update({k: v for k, v in overrides.items() if k in m.params})
    print(f"\n{m}  ({m.where})")
    print(f"  {m.params.description[:110]}")
    for r in m.params.rows():
        lim = r.get("list") or (f"[{r['min']}, {r['max']}]" if "min" in r else "")
        print(f"    {r['name']:28s} = {r['value']!s:12s} {r['type']:8s} {lim}")
    n = 128 * nspui
    delta = serdes.Channel.ideal(ts, n=n, delay=0).impulse
    r = m.init(delta, ts, ui)
    print(f"  Init message: {r.message.strip()[:160]}")
    print(f"  Init params_out: {r.params_out}")
    h = r.impulse[0] * ts                          # V per sample
    t = np.arange(n) * ts / ui
    step_init = np.cumsum(h)
    f = np.fft.rfftfreq(n, ts)
    H = np.fft.rfft(h)
    step_gw = None
    if m.getwave_exists:
        x = np.concatenate([np.full(16 * nspui, -0.5), np.full(n - 16 * nspui, 0.5)])
        step_gw = m.getwave(x, block_bits=64).wave
        step_gw = step_gw - step_gw[16 * nspui - 1]
    # PyAMI's own tester on the same model and parameters
    pyami = None
    if not m.params._numeric:
        pm, init = m.pyami_model(delta, ts, ui)
        pm.initialize(init)
        pyami = pm.get_responses(calc_getw=False)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    axs[0].plot(t, h / ts * 1e-9, label="AMI_Init (photonflux)")
    axs[0].set(xlim=(0, 20), xlabel="UI", ylabel="impulse [1/ns]",
               title=f"{name}: impulse response")
    axs[1].plot(t, step_init, label="AMI_Init (photonflux)")
    if step_gw is not None:
        axs[1].plot(t - 16, step_gw, "--", label="AMI_GetWave step")
    if pyami:
        from pyibisami.ami.model import OUT_RESP_INIT

        tp, _hp, sp, *_ = pyami[OUT_RESP_INIT]
        axs[1].plot(tp / ui, sp, ":", lw=2, label="PyAMI get_responses (Init)")
        dev = np.max(np.abs(np.interp(tp / ui, t, step_init) - sp)[tp >= 0])
        print(f"  photonflux vs PyAMI step response: max |diff| {dev:.2e} V")
    axs[1].set(xlim=(-1, 20), xlabel="UI", ylabel="V", title="step response")
    axs[1].legend(fontsize=8)
    axs[2].semilogx(f[1:] / 1e9, 20 * np.log10(np.abs(H[1:]) + 1e-15))
    axs[2].axvline(0.5 / ui / 1e9, color="k", ls=":", lw=0.8)
    axs[2].set(xlabel="GHz", ylabel="dB", title="frequency response (Init)",
               xlim=(f[1] / 1e9, 0.5 / ts / 1e9))
    axs[2].grid(True, which="both", alpha=0.3)
    save(fig, f"ibis_ami_tester_{name.replace('@', '_')}.png")
    m.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*", default=["pf_tx", "pf_rx"])
    ap.add_argument("--param", action="append", default=[],
                    help="name=value override (repeatable)")
    ap.add_argument("--rate", type=float, default=25e9)
    args = ap.parse_args(argv)
    ensure_built()
    overrides = {}
    for kv in args.param:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = float(v) if any(c in v for c in ".eE") else int(v)
        except ValueError:
            overrides[k] = v
    for name in args.models:
        characterize(name, overrides, 1.0 / args.rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
