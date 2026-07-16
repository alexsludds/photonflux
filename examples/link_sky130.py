#!/usr/bin/env python3
"""Photonic link with REAL SKY130 PDK transistors, solved by circulax (JAX).

The flagship example — the step
``link_cmos.py`` could not take: the transistors are the actual SKY130 BSIM4
devices from the volare PDK, not square-law stand-ins.

* **Laser** — ``cx.cw_laser()``: a CW source parameterised by wavelength and
  power, emitting a constant field ``E = sqrt(P)``. All modulation happens in
  the modulator; the laser is never modulated directly (house convention).
* **Modulator** — ``cx.mzm()``: field-convention Mach-Zehnder with the same
  intensity transfer as ``models/optical_power/mzm.va``, its electrodes driven by a SKY130
  5 V FET common-source stage.
* **Electronics** — ``cx.sky130_fet(...)``: ngspice resolves the volare model
  card (corner, `{...}` expressions, W/L bin), and the BSIM4.8 Verilog-A is
  compiled to OSDI and evaluated natively inside circulax's Newton/transient
  loop. Same physics ngspice runs, to the card.
* **Receiver** — the ``|E|^2`` photodiode bridge (from ``photodiode_tia.py``)
  into a SKY130 CMOS inverter TIA. The whole system is one complex-valued
  (coherent-field) JAX circuit.

Topology (mirrors example 02's receiver exactly):

    VPULSE -> sky130_fd_pr__nfet_g5v0d10v5 (W=50, L=0.5) common-source driver
           -> MZM electrodes;  CW laser -> MZM -> fibre -> photodiode
           -> SKY130 CMOS inverter TIA (pfet 40/0.18 + nfet 15/0.18, Rf = 1k)

    python examples/link_sky130.py
        -> out/link_sky130.png
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.electronic import (
    Capacitor,
    PulseVoltageSource,
    Resistor,
    VoltageSource,
)
from circulax.components.photonic import OpticalWaveguide

# the |E|^2 optical->electrical bridge shared by the circulax examples
from photodiode_tia import Photodiode

from photonflux import cx

# --- optics ------------------------------------------------------------------
WAVELENGTH_NM = 1310.0
P_LASER = 2e-3        # CW laser power [W]
VPI = 3.3             # MZM half-wave voltage [V] (matched to the 3.3 V driver)
FIBER_LOSS_DB = 3.0

# --- rails / passives ---------------------------------------------------------
V_LD = 3.3            # driver supply [V]
V_DDA = 1.8           # receiver rail [V]
R_PULLUP = 300.0      # driver drain pull-up [ohm]
RF = 1000.0           # TIA feedback resistance [ohm]

OUT = Path(__file__).resolve().parents[1] / "out" / "link_sky130.png"


def build_models() -> dict:
    """Compile/extract every device. All results are content-hash cached."""
    return {
        "ground": lambda: 0,
        "vsource": VoltageSource,
        "vpulse": PulseVoltageSource,
        "resistor": Resistor,
        "cap": Capacitor,
        # photonics: CW laser -> MZM -> fibre -> |E|^2 photodiode
        "laser": cx.cw_laser(),
        "mzm": cx.mzm(),
        "fiber": OpticalWaveguide,
        "photodiode": Photodiode,
        # SKY130 PDK transistors (OSDI, exact BSIM4.8)
        "nfet_drv": cx.sky130_fet("nfet_g5v0d10v5", w=50.0, l=0.5),
        "pfet_tia": cx.sky130_fet("pfet_01v8", w=40.0, l=0.18),
        "nfet_tia": cx.sky130_fet("nfet_01v8", w=15.0, l=0.18),
    }


def build() -> dict:
    # 3 dB of fibre: loss_dB_cm * (length_um / 1e4)
    fiber_len_um = 1500.0
    fiber_loss_db_cm = FIBER_LOSS_DB / (fiber_len_um / 1e4)

    return {
        "instances": {
            "GND": {"component": "ground"},
            "VLD": {"component": "vsource", "settings": {"V": V_LD}},
            "VDDA": {"component": "vsource", "settings": {"V": V_DDA}},
            # gate drive: two 0 -> 3.3 V bits
            "VIN": {"component": "vpulse",
                    "settings": {"v1": 0.0, "v2": 3.3, "td": 1e-9, "tr": 2e-10,
                                 "tf": 2e-10, "pw": 2e-9, "per": 5e-9}},
            "RB": {"component": "resistor", "settings": {"R": R_PULLUP}},
            "MDRV": {"component": "nfet_drv"},
            "LAS": {"component": "laser",
                    "settings": {"wavelength_nm": WAVELENGTH_NM,
                                 "power": P_LASER}},
            # driver ON -> electrode low -> T max: light follows V_in
            "MOD": {"component": "mzm",
                    "settings": {"vpi": VPI, "vbias": 0.0, "il_db": 3.0,
                                 "er_db": 20.0, "cel": 50e-15}},
            "FIB": {"component": "fiber",
                    "settings": {"length_um": fiber_len_um,
                                 "loss_dB_cm": fiber_loss_db_cm,
                                 "wavelength_nm": WAVELENGTH_NM,
                                 "center_wavelength_nm": WAVELENGTH_NM}},
            "PD": {"component": "photodiode",
                   "settings": {"R": 0.8, "Idk": 1e-9, "Cj": 100e-15}},
            "MP": {"component": "pfet_tia"},
            "MN": {"component": "nfet_tia"},
            "RF": {"component": "resistor", "settings": {"R": RF}},
            # node capacitances: physical (every node has some) and they give
            # the implicit stepper a finite RC per node (see link_cmos.py)
            "CDRV": {"component": "cap", "settings": {"C": 50e-15}},
            "CIN": {"component": "cap", "settings": {"C": 20e-15}},
            "COUT": {"component": "cap", "settings": {"C": 30e-15}},
        },
        "connections": {
            "GND,p1": ("VLD,p2", "VDDA,p2", "VIN,p2", "LAS,p2", "MOD,vn",
                       "MDRV,s", "MDRV,b", "PD,po_n",
                       "MN,s", "MN,b", "CDRV,p2", "CIN,p2", "COUT,p2"),
            "VLD,p1": "RB,p1",                        # 3.3 V -> pull-up
            "VDDA,p1": ("PD,an", "MP,s", "MP,b"),     # 1.8 V rail
            "VIN,p1": "MDRV,g",                       # gate drive
            "RB,p2": ("MDRV,d", "MOD,vp", "CDRV,p1"),  # driver drain = electrode
            "LAS,p1": "MOD,pin",                      # CW field -> MZM
            "MOD,pout": "FIB,p1",                     # MZM -> fibre
            "FIB,p2": "PD,po_p",                      # fibre -> photodiode
            "PD,cat": ("MP,g", "MN,g", "RF,p1", "CIN,p1"),  # TIA summing node
            "MP,d": ("MN,d", "RF,p2", "COUT,p1"),     # inverter output
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


def main() -> int:
    models = build_models()
    circuit = compile_circuit(build(), models, backend="dense",
                              is_complex=True, max_steps=300)
    print(f"compiled: {circuit.sys_size} complex unknowns "
          f"({2 * circuit.sys_size} real), {len(circuit.groups)} component groups")

    y0 = circuit.dc()
    p_tx0 = float(jnp.abs(circuit.port(y0, "popt_tx")) ** 2)
    print(f"DC: P_tx = {p_tx0 * 1e3:.3f} mW, "
          f"V(mod) = {float(circuit.port(y0, 'vmod').real):.3f} V, "
          f"tia_in = {float(circuit.port(y0, 'tia_in').real):.3f} V, "
          f"vout = {float(circuit.port(y0, 'vout').real):.3f} V")

    t_max = 8e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_max, 1200))
    # Solver note: with OSDI BSIM4 devices in a complex-valued system, every
    # *individual* BDF2 step converges (verified by fixed-dt marching), but
    # circulax 0.2.1's adaptive-controller retry path reports a spurious
    # nonlinear divergence. Fixed 20 ps steps are robust, resolve the 200 ps
    # edges with 10 points, and cost ~400 steps for the whole pattern.
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    dt_fixed = 2e-11
    sol = circuit.transient(
        t0=0.0, t1=t_max, dt0=dt_fixed, y0=y0, saveat=saveat,
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=circuit.solver, newton_max_steps=40),
        max_steps=int(t_max / dt_fixed) + 10, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    if sol.result != diffrax.RESULTS.successful:
        print(f"transient FAILED: {sol.result}")
        return 1

    t_ns = np.asarray(sol.ts) * 1e9
    vin = np.asarray(circuit.port(sol.ys, "vin").real)
    vmod = np.asarray(circuit.port(sol.ys, "vmod").real)
    p_tx = np.asarray(jnp.abs(circuit.port(sol.ys, "popt_tx")) ** 2) * 1e3  # mW
    p_rx = np.asarray(jnp.abs(circuit.port(sol.ys, "popt_rx")) ** 2) * 1e3  # mW
    tia_in = np.asarray(circuit.port(sol.ys, "tia_in").real)
    vout = np.asarray(circuit.port(sol.ys, "vout").real)

    print(f"transient OK: {len(t_ns)} points over {t_ns[-1]:.1f} ns")
    print(f"  P_tx  = [{p_tx.min():.3f}, {p_tx.max():.3f}] mW")
    print(f"  P_rx  = [{p_rx.min():.3f}, {p_rx.max():.3f}] mW")
    print(f"  vout  = [{vout.min():.3f}, {vout.max():.3f}] V")

    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)
    axes[0].plot(t_ns, vin, c="tab:gray", label="V_in")
    axes[0].plot(t_ns, vmod, c="tab:purple", ls="--", label="V(electrode)")
    axes[0].set_ylabel("drive [V]")
    axes[0].legend(loc="center right", fontsize=8)
    axes[0].set_title(
        "Photonic link on circulax — CW laser + MZM, real SKY130 BSIM4 FETs (OSDI)"
    )
    axes[1].plot(t_ns, p_tx, c="tab:orange", label="P_tx (after MZM)")
    axes[1].plot(t_ns, p_rx, c="tab:red", ls="--", label="P_rx (3 dB fibre)")
    axes[1].set_ylabel("P_opt [mW]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t_ns, tia_in, c="tab:green")
    axes[2].set_ylabel("V(tia_in) [V]")
    axes[3].plot(t_ns, vout, c="tab:blue")
    axes[3].set_ylabel("V_out [V]")
    axes[3].set_xlabel("time [ns]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
