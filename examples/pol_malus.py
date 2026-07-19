#!/usr/bin/env python3
"""Polarization optics: Malus' law and a birefringent-waveguide MZI.

The two acceptance-criteria testbenches for dual-polarization coherent fields,
solved with circulax (JAX). Every optical net carries a Jones vector as two
Ereal/Eimag node pairs — X = TE, Y = TM, |Ex|^2 + |Ey|^2 = power [W]. The
scalar models (phase_shifter.va, mirror.va, ...) are unchanged; they are just
the X/TE channel, so a scalar circuit is the default-polarization special case.

Left panel — **Malus' law**. A TE-polarized field goes through a polarization
rotator set to angle theta, then a polarization beam splitter (PBS). The two
PBS ports carry cos^2(theta) and sin^2(theta) of the input power — Malus' law —
and a polarization beam combiner (PBC) puts the link back together with unit
throughput (models/optical_field/{polarization_rotator,pbs,pbc}.va).

Right panel — **birefringent-waveguide MZI**. A Mach-Zehnder interferometer
with a birefringent waveguide (n_TE != n_TM) in one arm. The cross-port fringe
is cos^2(pi * n * L / lambda) per polarization, so the TE and TM transmission
fringes are offset in wavelength by the modal birefringence dn_eff = n_TE-n_TM
(models/optical_field/birefringent_wg.va).

    python examples/pol_malus.py
        -> out/pol_malus.png
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component
from circulax.components.electronic import VoltageSource

from photonflux import cx

_JONES_IN = ("xin_re", "xin_im", "yin_re", "yin_im")
_ROT_OUT = ("xout_re", "xout_im", "yout_re", "yout_im")


def _terminator():
    """Infinite-impedance termination for a driven-but-unused (re, im) pair."""
    @component(ports=("re", "im"))
    def Terminator(signals: Signals, s: States):
        return {"re": 0.0, "im": 0.0}, {}
    return Terminator


# ---------------------------------------------------------------------------
# Malus' law: TE launch -> rotator(theta) -> PBS -> PBC
# ---------------------------------------------------------------------------
def malus_circuit():
    insts = {"GND": {"component": "ground"},
             "ROT": {"component": "rot", "settings": {"theta_deg": 0.0}},
             "PBS": {"component": "pbs"}, "PBC": {"component": "pbc"}}
    conns, gnd = {}, ["ROT,gnd", "PBS,gnd", "PBC,gnd"]
    for p, v in zip(_JONES_IN, (1.0, 0.0, 0.0, 0.0)):  # pure TE input
        insts[f"S_{p}"] = {"component": "vsrc", "settings": {"V": v}}
        conns[f"S_{p},p1"] = f"ROT,{p}"
        gnd.append(f"S_{p},p2")
    for a, b in zip(_ROT_OUT, _JONES_IN):
        conns[f"ROT,{a}"] = f"PBS,{b}"
    for a, b in [("o1x_re", "i1x_re"), ("o1x_im", "i1x_im"),
                 ("o1y_re", "i1y_re"), ("o1y_im", "i1y_im"),
                 ("o2x_re", "i2x_re"), ("o2x_im", "i2x_im"),
                 ("o2y_re", "i2y_re"), ("o2y_im", "i2y_im")]:
        conns[f"PBS,{a}"] = f"PBC,{b}"
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {"te_re": "PBS,o1x_re", "te_im": "PBS,o1x_im",
                     "tm_re": "PBS,o2y_re", "tm_im": "PBS,o2y_im",
                     "cx_re": "PBC,ox_re", "cx_im": "PBC,ox_im",
                     "cy_re": "PBC,oy_re", "cy_im": "PBC,oy_im"}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "rot": cx.va("polarization_rotator"),
              "pbs": cx.va("pbs"), "pbc": cx.va("pbc")}
    return compile_circuit(net, models)


# ---------------------------------------------------------------------------
# birefringent-waveguide MZI (one 50/50 mirror pair per polarization)
# ---------------------------------------------------------------------------
def mzi_circuit(length_um=100.0, n_te=2.40, n_tm=2.38):
    insts = {"GND": {"component": "ground"},
             "WG": {"component": "bir",
                    "settings": {"length_um": length_um, "n_te": n_te, "n_tm": n_tm}},
             "SX": {"component": "vsrc", "settings": {"V": 0.0}},
             "SY": {"component": "vsrc", "settings": {"V": 0.0}},
             "T1X": {"component": "term"}, "T1Y": {"component": "term"}}
    for b in ("B1X", "B1Y", "B2X", "B2Y"):
        insts[b] = {"component": "mir", "settings": {"refl": 0.5}}
    conns = {"SX,p1": "B1X,li_re", "SY,p1": "B1Y,li_re"}
    gnd = ["B1X,gnd", "B1Y,gnd", "B2X,gnd", "B2Y,gnd", "WG,gnd", "SX,p2", "SY,p2",
           "B1X,li_im", "B1X,ri_re", "B1X,ri_im",
           "B1Y,li_im", "B1Y,ri_re", "B1Y,ri_im"]
    conns.update({"B1X,lo_re": "WG,xin_re", "B1X,lo_im": "WG,xin_im",
                  "B1Y,lo_re": "WG,yin_re", "B1Y,lo_im": "WG,yin_im",
                  "WG,xout_re": "B2X,li_re", "WG,xout_im": "B2X,li_im",
                  "WG,yout_re": "B2Y,li_re", "WG,yout_im": "B2Y,li_im",
                  "B1X,ro_re": "B2X,ri_re", "B1X,ro_im": "B2X,ri_im",
                  "B1Y,ro_re": "B2Y,ri_re", "B1Y,ro_im": "B2Y,ri_im",
                  "T1X,re": "B2X,lo_re", "T1X,im": "B2X,lo_im",
                  "T1Y,re": "B2Y,lo_re", "T1Y,im": "B2Y,lo_im"})
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {"te_re": "B2X,ro_re", "te_im": "B2X,ro_im",
                     "tm_re": "B2Y,ro_re", "tm_im": "B2Y,ro_im"}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource, "mir": cx.va("mirror"),
              "bir": cx.va("birefringent_wg"), "term": _terminator()}
    return compile_circuit(net, models)


def main() -> None:
    # --- Malus' law ---------------------------------------------------------
    c = malus_circuit()
    thetas = np.linspace(0.0, 180.0, 181)
    y = c.dc(params={"ROT.theta_deg": thetas})
    arr = lambda p: np.asarray(c.port(y, p)).real  # noqa: E731
    p_te = arr("te_re") ** 2 + arr("te_im") ** 2
    p_tm = arr("tm_re") ** 2 + arr("tm_im") ** 2
    p_rec = (arr("cx_re") ** 2 + arr("cx_im") ** 2
             + arr("cy_re") ** 2 + arr("cy_im") ** 2)
    print(f"Malus: max |P_TE - cos^2| = {np.max(np.abs(p_te - np.cos(np.radians(thetas))**2)):.2e}")
    print(f"Malus: PBC throughput 1 +- {np.max(np.abs(p_rec - 1.0)):.2e}")

    # --- birefringent MZI ---------------------------------------------------
    L, n_te, n_tm = 100.0, 2.40, 2.38
    m = mzi_circuit(L, n_te, n_tm)
    lams = np.linspace(1540.0, 1560.0, 1001)

    def sweep(sx, sy):
        yy = m.dc(params={"SX.V": sx * np.ones_like(lams),
                          "SY.V": sy * np.ones_like(lams),
                          "WG.lambda_nm": lams})
        a = lambda p: np.asarray(m.port(yy, p)).real  # noqa: E731
        return a("te_re") ** 2 + a("te_im") ** 2, a("tm_re") ** 2 + a("tm_im") ** 2

    te_p, _ = sweep(1.0, 0.0)
    _, tm_p = sweep(0.0, 1.0)
    lam0 = float(np.median(lams))  # representative (band-centre) wavelength
    print(f"MZI: modal birefringence dn_eff = {n_te - n_tm:.3f}, "
          f"beat length = {(lam0*1e-9)/abs(n_te-n_tm)*1e3:.2f} mm")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    axl.plot(thetas, p_te, label=r"PBS TE port  $\cos^2\theta$")
    axl.plot(thetas, p_tm, label=r"PBS TM port  $\sin^2\theta$")
    axl.plot(thetas, p_rec, "k--", lw=1, label="PBC throughput")
    axl.set_xlabel(r"rotator angle $\theta$ [deg]")
    axl.set_ylabel("normalized power")
    axl.set_title("Malus' law: rotator + PBS + PBC")
    axl.legend(loc="center right", fontsize=8)

    axr.plot(lams, te_p, label=f"TE ($n$={n_te})")
    axr.plot(lams, tm_p, label=f"TM ($n$={n_tm})")
    axr.set_xlabel(r"wavelength [nm]")
    axr.set_ylabel("cross-port transmission")
    axr.set_title(r"Birefringent MZI: TE/TM fringes offset by $\Delta n_{eff}$")
    axr.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "out" / "pol_malus.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
