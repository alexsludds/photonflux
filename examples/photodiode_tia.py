#!/usr/bin/env python3
"""Photodiode + ideal-TIA receiver, solved with circulax (JAX, differentiable).

The minimal receiver reproducer: a CW laser through a pulse modulator and a
short fibre into a PIN photodiode into an ideal op-amp transimpedance
amplifier, solved as one differentiable JAX system (Newton DC + Diffrax
transient).

The conventions differ, which is the interesting part:

* **photonflux** rides the *power* convention — an optical node voltage *is*
  the optical power in watts, and the photonics are compiled Verilog-A
  (``models/optical_power/photodiode.va``: ``Iph = R*V(popt)``) solved inside ngspice.

* **circulax** is a *coherent-field* simulator — every optical node carries a
  complex field amplitude ``E``, and optical power is ``|E|^2``. The whole
  complex system is assembled as a real 2N block (RR/RI/IR/II partials), so a
  *non-holomorphic* detector ``Iph = R*|E|^2`` differentiates correctly inside
  the solver.

The photodiode below is therefore a genuine mixed-domain bridge: it reads the
complex optical field on one port pair and injects a real photocurrent into
the electrical port pair, all in one matrix. Because circulax is built on JAX,
the same ``build()`` is end-to-end differentiable — you could ``jax.grad`` the
TIA output swing w.r.t. ``R``, ``Rf`` or the fibre loss for inverse design.

    python examples/photodiode_tia.py
        -> out/photodiode_tia.png
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
from circulax.components.base_component import Signals, States, component, source
from circulax.components.electronic import Resistor
from circulax.components.photonic import OpticalWaveguide

# --- receiver parameters ----------------------------------------------------
R_PD = 0.8            # photodiode responsivity [A/W]
IDK = 1e-9            # dark current [A]
CJ = 100e-15         # junction capacitance [F]
RF = 200.0           # TIA feedback resistance [ohm]
GAIN = 1e6           # ideal op-amp open-loop gain [V/V]

# optical pulse: 3 mW "0" -> 18 mW "1", same levels as example 01 post-fibre
P_OFF = 3e-3         # [W]
P_ON = 18e-3         # [W]
FIBER_LOSS_DB = 3.0  # 3 dB fibre, like ls.add_fiber(..., loss_db=3)

OUT = Path(__file__).resolve().parents[1] / "out" / "photodiode_tia.png"


# ===========================================================================
# Custom components — the photonic <-> electronic bridge lives here
# ===========================================================================
@source(ports=("pin", "pout"), states=("i_out",))
def PulseModulator(
    signals: Signals,
    s: States,
    t: float,
    p_on: float = P_ON,
    p_off: float = P_OFF,
    t0: float = 0.5e-9,
    t1: float = 2.0e-9,
    tr: float = 80e-12,
) -> tuple[dict, dict]:
    """Ideal intensity modulator carving an NRZ-like pulse out of a CW field.

    House convention: the laser itself is CW (``cx.cw_laser()``, wavelength +
    power -> constant field); *all* modulation lives in modulator elements.
    This one applies a time-varying amplitude transmission ``t(t)`` so the
    output power ramps ``p_off -> p_on -> p_off`` with sigmoid edges of time
    constant ``tr`` when fed by a CW laser of power ``p_on``. The output field
    is prescribed VCVS-style (``E_out = t(t) * E_in``) with the branch current
    ``i_out`` as the unknown; the input port draws no power (ideal).
    """
    k = 1.0 / tr
    env = jax.nn.sigmoid(k * (t - t0)) - jax.nn.sigmoid(k * (t - t1))  # 0->1->0
    trans = jnp.sqrt((p_off + (p_on - p_off) * env) / p_on)
    constraint = signals.pout - trans * signals.pin
    return {"pin": 0.0, "pout": s.i_out, "i_out": constraint}, {}


@component(ports=("po_p", "po_n", "an", "cat"))
def Photodiode(
    signals: Signals,
    s: States,
    R: float = R_PD,
    Idk: float = IDK,
    Cj: float = CJ,
    Yopt: float = 1.0,
) -> tuple[dict, dict]:
    """PIN photodiode bridging the optical (complex) and electrical (real) nets.

    Optical side (``po_p``/``po_n``): a matched absorber of reference
    admittance ``Yopt`` (=1, the normalised waveguide impedance), so the
    incident field is detected without reflection — the optical analogue of a
    50-ohm-matched RF load. Detected power is ``|E|^2``.

    Electrical side (``an``/``cat``): a photocurrent ``Iph = R*|E|^2 + Idk`` is
    sunk at the cathode and sourced at the anode (reverse photocurrent), with a
    junction capacitance ``Cj`` across the diode for the front-end bandwidth
    pole. This is the ``Iph + Cj*ddt(V)`` of ``models/optical_power/photodiode.va`` — here
    ``|E|^2`` replaces the power-domain node voltage.
    """
    e = signals.po_p - signals.po_n            # complex optical field
    i_opt = Yopt * e                           # matched-load absorption
    power = jnp.abs(e) ** 2                     # |E|^2 -> optical power [W]
    iph = R * power + Idk

    f = {"po_p": i_opt, "po_n": -i_opt, "cat": iph, "an": -iph}

    vj = signals.cat - signals.an              # junction voltage
    q = {"cat": Cj * vj, "an": -Cj * vj}       # I = Cj * dV/dt
    return f, q


@component(ports=("outp", "outn", "inp", "inn"), states=("i_out",))
def OpAmp(signals: Signals, s: States, gain: float = GAIN) -> tuple[dict, dict]:
    """Ideal VCVS op-amp: enforces ``V(outp,outn) = gain * V(inp,inn)``.

    Inputs draw no current (infinite input impedance); the output branch
    current ``i_out`` is the unknown that satisfies the voltage constraint —
    the SPICE ``E`` (VCVS) element behind the ideal TIA in example 01.
    """
    constraint = (signals.outp - signals.outn) - gain * (signals.inp - signals.inn)
    return {"outp": s.i_out, "outn": -s.i_out, "inp": 0.0, "inn": 0.0, "i_out": constraint}, {}


# ===========================================================================
# Netlist
# ===========================================================================
def build() -> dict:
    """SAX-style netlist: CW laser -> modulator -> fibre -> photodiode -> TIA.

    Inverting transimpedance config: the op-amp ``+`` input is grounded and the
    photodiode drives the ``-`` input (a virtual ground), so the photocurrent
    flows through ``Rf`` and ``V(vout) = +Iph * Rf``.
    """
    # 3 dB of fibre: loss_dB_cm * (length_um / 1e4) = 20 * 0.15 = 3.0 dB
    fiber_len_um = 1500.0
    fiber_loss_db_cm = FIBER_LOSS_DB / (fiber_len_um / 1e4)

    return {
        "instances": {
            "GND": {"component": "ground"},
            "LAS": {"component": "cw_laser",
                    "settings": {"wavelength_nm": 1310.0, "power": P_ON}},
            "MOD": {"component": "pulse_mod"},
            "FIB": {
                "component": "waveguide",
                "settings": {"length_um": fiber_len_um, "loss_dB_cm": fiber_loss_db_cm},
            },
            "PD": {"component": "photodiode"},
            "OA": {"component": "opamp"},
            "RF": {"component": "resistor", "settings": {"R": RF}},
        },
        "connections": {
            # ground rail: laser return, PD anode + optical return, op-amp + input + output return
            "GND,p1": ("LAS,p2", "PD,po_n", "PD,an", "OA,inp", "OA,outn"),
            "LAS,p1": "MOD,pin",           # CW field -> modulator
            "MOD,pout": "FIB,p1",          # modulated field -> fibre
            "FIB,p2": "PD,po_p",           # fibre -> photodiode optical input
            "PD,cat": ("RF,p1", "OA,inn"),  # photocurrent node = virtual ground (-input)
            "OA,outp": "RF,p2",            # feedback resistor closes the loop
        },
        "ports": {
            "opt_in": "MOD,pout",
            "popt_rx": "PD,po_p",  # optical field at the photodiode
            "pd_cat": "PD,cat",    # TIA summing node
            "vout": "OA,outp",     # TIA output
        },
    }


def _models() -> dict:
    from photonflux import cx

    return {
        "ground": lambda: 0,
        "cw_laser": cx.cw_laser(),
        "pulse_mod": PulseModulator,
        "waveguide": OpticalWaveguide,
        "photodiode": Photodiode,
        "opamp": OpAmp,
        "resistor": Resistor,
    }


def demo_gradient() -> None:
    """The feature ngspice can't give you: gradients through the solver.

    Replace the pulse with a CW source so there's a clean DC operating point,
    then ``jax.grad`` the TIA output w.r.t. the feedback resistance and the
    received optical power. For the ideal inverting TIA ``V_out = R*P*Rf``, so
    the gradients have closed forms — a check that AD flows correctly through
    the DC Newton solve and the non-holomorphic ``|E|^2`` detector.
    """
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SRC": {"component": "cw_laser", "settings": {"power": 9e-3}},
            "PD": {"component": "photodiode"},
            "OA": {"component": "opamp"},
            "RF": {"component": "resistor", "settings": {"R": RF}},
        },
        "connections": {
            "GND,p1": ("SRC,p2", "PD,po_n", "PD,an", "OA,inp", "OA,outn"),
            "SRC,p1": "PD,po_p",
            "PD,cat": ("RF,p1", "OA,inn"),
            "OA,outp": "RF,p2",
        },
        "ports": {"vout": "OA,outp"},
    }
    circuit = compile_circuit(net, _models(), is_complex=True)

    def vout(rf: float, power: float) -> float:
        y = circuit.dc(params={"RF.R": rf, "SRC.power": power})
        return circuit.port(y, "vout").real

    rf0, p0 = float(RF), 9e-3
    dv_drf, dv_dp = jax.grad(vout, argnums=(0, 1))(rf0, p0)
    print("differentiable solve (jax.grad through DC + |E|^2 detector):")
    print(f"  V_out               = {float(vout(rf0, p0)):.4f} V")
    print(f"  dV_out/dRf          = {float(dv_drf):.5f}  (R*P  = {R_PD * p0:.5f})")
    print(f"  dV_out/dP_opt       = {float(dv_dp):.3f}  (R*Rf = {R_PD * rf0:.3f})")


def main() -> int:
    net = build()
    circuit = compile_circuit(net, _models(), is_complex=True)
    print(f"compiled: {circuit.sys_size} complex unknowns "
          f"({2 * circuit.sys_size} real), {len(circuit.groups)} component groups")

    # DC operating point, then transient with adaptive Diffrax stepping
    t_max = 40e-9
    y0 = circuit.dc()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_max, 800))
    sol = circuit.transient(
        t0=0.0,
        t1=t_max,
        dt0=1e-12,
        y0=y0,
        saveat=saveat,
        max_steps=200_000,
        throw=False,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7),
    )
    if sol.result != diffrax.RESULTS.successful:
        print(f"transient FAILED: {sol.result}")
        return 1

    t = np.asarray(sol.ts)
    e_rx = circuit.port(sol.ys, "popt_rx")          # complex optical field
    p_rx = np.asarray(jnp.abs(e_rx) ** 2)           # optical power [W]
    v_cat = np.asarray(circuit.port(sol.ys, "pd_cat").real)
    v_out = np.asarray(circuit.port(sol.ys, "vout").real)
    i_pd = R_PD * p_rx + IDK                         # photocurrent [A]

    print(f"transient OK: {len(t)} points over {t[-1] * 1e9:.2f} ns")
    print(f"  P_rx at PD   = [{p_rx.min() * 1e3:.2f}, {p_rx.max() * 1e3:.2f}] mW")
    print(f"  PD I_ph peak = {i_pd.max() * 1e3:.3f} mA")
    print(f"  TIA V_out    = [{v_out.min():.3f}, {v_out.max():.3f}] V")
    print(f"  virtual gnd  = {np.abs(v_cat).max() * 1e6:.2f} uV (|V(pd_cat)| max)")
    # closed-form check: ideal inverting TIA gives V_out = Iph * Rf
    print(f"  V_out vs Iph*Rf: max abs err = "
          f"{np.abs(v_out - i_pd * RF).max() * 1e3:.3f} mV")

    # --- plot -------------------------------------------------------------
    t_ns = t * 1e9
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t_ns, p_rx * 1e3, color="tab:orange")
    axes[0].set_ylabel("P at PD [mW]")
    axes[0].set_title("Photodiode + ideal TIA receiver — circulax (JAX) coherent-field solve")
    axes[1].plot(t_ns, i_pd * 1e3, color="tab:green")
    axes[1].set_ylabel("I_pd [mA]")
    axes[2].plot(t_ns, v_out, color="tab:blue")
    axes[2].set_ylabel("V_out [V]")
    axes[2].set_xlabel("time [ns]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"  wrote {OUT}")

    demo_gradient()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
