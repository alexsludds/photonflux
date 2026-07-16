#!/usr/bin/env python3
"""CMOS driver + CMOS-inverter TIA around a photonic link, solved with circulax.

The full link with generic
square-law MOSFETs — the PDK-free warm-up for ``link_sky130.py``, which runs
the same topology with real SKY130 BSIM4 devices.

House convention for sources: the laser is **CW only** — ``cx.cw_laser()``,
parameterised by wavelength and power, emitting a constant field
``E = sqrt(P)``. All modulation happens in a modulator (``cx.mzm()``, the
field-convention twin of ``models/optical_power/mzm.va``):

  * driver:  an NMOS common-source stage swings the MZM electrodes off a
    3.3 V rail (gate driven by a SPICE-style PULSE),
  * receiver: the ``Iph = R*|E|^2`` photodiode bridge (imported from
    ``photodiode_tia.py``) into a self-biased CMOS inverter TIA (PMOS + NMOS,
    1.8 V rail) with resistive feedback ``Rf``.

The whole thing — driver FET, MZM transfer, 3 dB fibre, detector, and the
two-FET inverter with feedback — is one nonlinear complex system, solved by a
single Newton DC + Diffrax transient, and end-to-end differentiable.

    python examples/link_cmos.py   # -> out/link_cmos.png
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.electronic import (
    NMOS,
    PMOS,
    Capacitor,
    PulseVoltageSource,
    Resistor,
    VoltageSource,
)
from circulax.components.photonic import OpticalWaveguide

# reuse the optical->electrical bridge from the example-01 port
from photodiode_tia import Photodiode

from photonflux import cx

# --- optics ------------------------------------------------------------------
WAVELENGTH_NM = 1310.0
P_LASER = 2e-3        # CW laser power [W]
VPI = 3.3             # MZM half-wave voltage [V]
FIBER_LOSS_DB = 3.0

# --- supplies / drive --------------------------------------------------------
V_LD = 3.3            # driver rail [V]
V_DDA = 1.8           # receiver inverter rail [V]
R_PULLUP = 300.0      # driver drain pull-up [ohm]
RF = 1000.0           # TIA feedback resistance [ohm]

# --- inverter sizing: Vth lowered for a 1.8 V rail, PMOS 2x wider so the ----
# --- self-bias trip point lands near mid-rail (betan == betap) --------------
VTH_N, VTH_P = 0.45, -0.45
WN, WP, LCH = 15e-6, 30e-6, 0.5e-6
LAMBDA = 0.10         # channel-length modulation -> finite inverter gain

OUT = Path(__file__).resolve().parents[1] / "out" / "link_cmos.png"


# ===========================================================================
# Netlist
# ===========================================================================
def build() -> dict:
    fiber_len_um = 1500.0
    fiber_loss_db_cm = FIBER_LOSS_DB / (fiber_len_um / 1e4)

    return {
        "instances": {
            "GND": {"component": "ground"},
            "VLD": {"component": "vsource", "settings": {"V": V_LD}},
            "VDDA": {"component": "vsource", "settings": {"V": V_DDA}},
            # gate drive: 0 -> 3.3 V NRZ-ish pulse train
            "VIN": {
                "component": "vpulse",
                "settings": {"v1": 0.0, "v2": 3.3, "td": 1e-9, "tr": 2e-10,
                             "tf": 2e-10, "pw": 2e-9, "per": 5e-9},
            },
            "RB": {"component": "resistor", "settings": {"R": R_PULLUP}},
            "MDRV": {"component": "nmos",
                     "settings": {"W": 600e-6, "L": 0.5e-6, "Vth": 0.7, "lam": LAMBDA}},
            "LAS": {"component": "laser",
                    "settings": {"wavelength_nm": WAVELENGTH_NM, "power": P_LASER}},
            # driver ON -> electrode low -> T max: light follows V_in
            "MOD": {"component": "mzm",
                    "settings": {"vpi": VPI, "vbias": 0.0, "il_db": 3.0,
                                 "er_db": 20.0, "cel": 50e-15}},
            "FIB": {"component": "waveguide",
                    "settings": {"length_um": fiber_len_um,
                                 "loss_dB_cm": fiber_loss_db_cm,
                                 "wavelength_nm": WAVELENGTH_NM,
                                 "center_wavelength_nm": WAVELENGTH_NM}},
            "PD": {"component": "photodiode"},
            "MP": {"component": "pmos",
                   "settings": {"W": WP, "L": LCH, "Vth": VTH_P, "lam": LAMBDA}},
            "MN": {"component": "nmos",
                   "settings": {"W": WN, "L": LCH, "Vth": VTH_N, "lam": LAMBDA}},
            "RF": {"component": "resistor", "settings": {"R": RF}},
            # parasitic node capacitances (drain/gate/load). Every real node
            # has some C; here they also give each electrical node a finite RC
            # so the implicit solver marches with sane steps instead of chasing
            # near-instant algebraic jumps across the square-law FET kinks.
            "CDRV": {"component": "cap", "settings": {"C": 50e-15}},
            "CIN": {"component": "cap", "settings": {"C": 20e-15}},
            "COUT": {"component": "cap", "settings": {"C": 30e-15}},
        },
        "connections": {
            "GND,p1": ("VLD,p2", "VDDA,p2", "VIN,p2", "LAS,p2", "MOD,vn",
                       "MDRV,s", "PD,po_n", "MN,s", "CDRV,p2", "CIN,p2", "COUT,p2"),
            "VLD,p1": "RB,p1",                       # 3.3 V -> pull-up
            "VDDA,p1": ("PD,an", "MP,s"),            # 1.8 V rail: PD reverse bias + PMOS source
            "VIN,p1": "MDRV,g",                      # gate drive
            "RB,p2": ("MDRV,d", "MOD,vp", "CDRV,p1"),  # driver drain = MZM electrode
            "LAS,p1": "MOD,pin",                     # CW field -> MZM
            "MOD,pout": "FIB,p1",                    # MZM -> fibre
            "FIB,p2": "PD,po_p",                     # fibre -> photodiode
            "PD,cat": ("MP,g", "MN,g", "RF,p1", "CIN,p1"),  # TIA summing node
            "MP,d": ("MN,d", "RF,p2", "COUT,p1"),    # inverter output = vout
        },
        "ports": {
            "vin": "VIN,p1",
            "vmod": "MOD,vp",
            "popt_tx": "MOD,pout",
            "popt_rx": "PD,po_p",
            "tia_in": "PD,cat",
            "vout": "MP,d",
        },
    }


MODELS = {
    "ground": lambda: 0,
    "vsource": VoltageSource,
    "vpulse": PulseVoltageSource,
    "resistor": Resistor,
    "nmos": NMOS,
    "pmos": PMOS,
    "cap": Capacitor,
    "laser": cx.cw_laser(),
    "mzm": cx.mzm(),
    "waveguide": OpticalWaveguide,
    "photodiode": Photodiode,
}


def main() -> int:
    circuit = compile_circuit(build(), MODELS, is_complex=True)
    print(f"compiled: {circuit.sys_size} complex unknowns "
          f"({2 * circuit.sys_size} real), {len(circuit.groups)} component groups")

    y0 = circuit.dc()
    vtia0 = float(circuit.port(y0, "tia_in").real)
    vout0 = float(circuit.port(y0, "vout").real)
    print(f"TIA self-bias OP: tia_in = {vtia0:.3f} V, vout = {vout0:.3f} V "
          f"(V_DDA = {V_DDA} V)")

    t_max = 20e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_max, 1200))
    sol = circuit.transient(
        t0=0.0, t1=t_max, dt0=1e-12, y0=y0, saveat=saveat,
        max_steps=400_000, throw=False,
        # dtmax keeps the adaptive stepper from striding over the 200 ps gate
        # edges (without it, the smooth RC dynamics let it take ~2 ns steps and
        # miss the pulse entirely).
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7, dtmax=20e-12),
    )
    if sol.result != diffrax.RESULTS.successful:
        print(f"transient FAILED: {sol.result}")
        return 1

    t = np.asarray(sol.ts)
    vin = np.asarray(circuit.port(sol.ys, "vin").real)
    vmod = np.asarray(circuit.port(sol.ys, "vmod").real)
    p_tx = np.asarray(jnp.abs(circuit.port(sol.ys, "popt_tx")) ** 2)
    p_rx = np.asarray(jnp.abs(circuit.port(sol.ys, "popt_rx")) ** 2)
    v_tia = np.asarray(circuit.port(sol.ys, "tia_in").real)
    v_out = np.asarray(circuit.port(sol.ys, "vout").real)
    i_pd = (v_tia - v_out) / RF

    print(f"transient OK: {len(t)} points over {t[-1] * 1e9:.2f} ns")
    print(f"  V(electrode) swing = [{vmod.min():.2f}, {vmod.max():.2f}] V (NFET drain)")
    print(f"  P_tx after MZM     = [{p_tx.min() * 1e3:.3f}, {p_tx.max() * 1e3:.2f}] mW")
    print(f"  P_rx at PD         = [{p_rx.min() * 1e3:.3f}, {p_rx.max() * 1e3:.2f}] mW")
    print(f"  TIA V_out swing    = [{v_out.min():.3f}, {v_out.max():.3f}] V")

    fig, axes = plt.subplots(5, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t * 1e9, vin, color="tab:gray");            axes[0].set_ylabel("V_in [V]")
    axes[0].set_title("CW laser + MZM link, square-law CMOS — circulax (JAX)")
    axes[1].plot(t * 1e9, vmod, color="tab:red");            axes[1].set_ylabel("V(electrode) [V]")
    axes[2].plot(t * 1e9, p_tx * 1e3, label="P_tx")
    axes[2].plot(t * 1e9, p_rx * 1e3, label="P_rx");         axes[2].set_ylabel("Optical [mW]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].plot(t * 1e9, i_pd * 1e3, color="tab:green");    axes[3].set_ylabel("I_pd [mA]")
    axes[4].plot(t * 1e9, v_out, color="tab:blue");          axes[4].set_ylabel("V_out [V]")
    axes[4].set_xlabel("time [ns]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
