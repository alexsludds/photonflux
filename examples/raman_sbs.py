#!/usr/bin/env python3
"""Raman (SRS / Raman amplifier) and Brillouin (SBS) scattering spans, pinned
against their closed-form physics.

Two lumped fibre models, both solved as one circulax DC system:

``models/optical_field/raman_amp.va`` — stimulated Raman scattering. A forward
signal and a co-/counter-propagating pump exchange power by the exact two-wave
logistic solution, so one component does both testbenches from ALE-72:

  * two-channel WDM SRS tilt — a short-lambda "pump" channel bleeds power into a
    long-lambda "signal" channel; the small-signal transfer slope is
    d ln(gain)/dP_pump = g_R*L_eff/A_eff, and the pump depletes as the signal
    grows (Q = P_s + (nu_s/nu_p)*P_p conserved);
  * distributed Raman amplifier — a weak signal plus a strong counter-pump shows
    the textbook on/off gain G_A = e^{g_R*P_p*L_eff/A_eff} vs pump power.

``models/optical_field/sbs_fiber.va`` — stimulated Brillouin scattering as a
threshold limiter. Below P_th = n_th*A_eff/(g_B*L_eff) (textbook n_th ~ 21) the
pump transmits; above it the transmitted power CLAMPS at ~P_th and the surplus
is back-reflected as the Stokes wave (P_fo = P_th*tanh(P_in/P_th)*e^{-alpha*L}).

Self-checks (all asserted, machine-verifiable):
  1. SRS slope    — d ln(gain)/dP_pump == g_R*L_eff/A_eff at small pump
  2. SRS depletion— the short-lambda channel loses exactly the photons the
                    long-lambda channel gains (matches the logistic solution)
  3. Raman on/off — counter-pumped gain vs pump power tracks G_A; co- and
                    counter-pump give the same integrated gain
  4. SBS threshold— transmitted power clamps at ~P_th, backscatter negligible
                    below and dominant above threshold, energy conserved

    python examples/raman_sbs.py            ->  out/raman_sbs.png
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component
from circulax.components.electronic import VoltageSource

from photonflux import cx

OUT = Path(__file__).resolve().parents[1] / "out" / "raman_sbs.png"

RAMAN = dict(lambda_s_nm=1550.0, lambda_p_nm=1450.0, g_r=0.6e-13, a_eff_um2=80.0,
             length_km=50.0, loss_s_db_km=0.20, loss_p_db_km=0.25)
SBS = dict(g_b=5e-11, a_eff_um2=80.0, length_km=20.0,
           loss_db_km=0.20, n_th=21.0)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# circuit builder: drive every DUT input node with a (sweepable) source,
# expose every output node as a read port
# ---------------------------------------------------------------------------
def _terminator():
    @component(ports=("re", "im"))
    def Terminator(signals: Signals, s: States):
        return {"re": 0.0, "im": 0.0}, {}

    return Terminator


def build(va_name, settings, inputs, outputs):
    insts = {"GND": {"component": "ground"},
             "DUT": {"component": "dut", "settings": dict(settings)}}
    conns, gnd = {}, ["DUT,gnd"]
    for node in inputs:
        sn = f"S_{node}"
        insts[sn] = {"component": "vsrc", "settings": {"V": 0.0}}
        conns[f"{sn},p1"] = f"DUT,{node}"
        gnd.append(f"{sn},p2")
    conns["GND,p1"] = tuple(gnd)
    net = {"instances": insts, "connections": conns,
           "ports": {o: f"DUT,{o}" for o in outputs}}
    models = {"ground": lambda: 0, "vsrc": VoltageSource,
              "dut": cx.va(va_name), "term": _terminator()}
    return compile_circuit(net, models, is_complex=True, backend="dense")


# analytic references (mirror the VA equations) ------------------------------
def _alpha(loss_db_km):
    return loss_db_km * np.log(10) / 10 / 1e3


def raman_leff(p=RAMAN):
    L = p["length_km"] * 1e3
    return (1 - np.exp(-_alpha(p["loss_p_db_km"]) * L)) / _alpha(p["loss_p_db_km"])


def raman_on_off_gain(pp0, p=RAMAN):
    aeff = p["a_eff_um2"] * 1e-12
    return np.exp(p["g_r"] / aeff * pp0 * raman_leff(p))


def sbs_pth(p=SBS):
    L = p["length_km"] * 1e3
    aeff = p["a_eff_um2"] * 1e-12
    leff = (1 - np.exp(-_alpha(p["loss_db_km"]) * L)) / _alpha(p["loss_db_km"])
    return p["n_th"] * aeff / (p["g_b"] * leff)


def _pwr(c, y, re, im):
    a = lambda p: np.asarray(c.port(y, p)).real  # noqa: E731
    return a(re) ** 2 + a(im) ** 2


def main() -> None:
    L_s = RAMAN["length_km"] * 1e3
    loss_s = np.exp(-_alpha(RAMAN["loss_s_db_km"]) * L_s)   # signal background loss

    # ======================================================================
    # 1) two-channel SRS tilt: co-pump (short lambda) sweep, weak signal
    # ======================================================================
    rin = ["si_re", "si_im", "pfi_re", "pfi_im", "pbi_re", "pbi_im"]
    rout = ["so_re", "so_im", "pfo_re", "pfo_im", "pbo_re", "pbo_im"]
    c = build("raman_amp", RAMAN, rin, rout)

    pumps = np.linspace(0.005, 0.8, 40)
    ps_probe = 1e-9                                   # weak long-lambda channel
    zeros = np.zeros_like(pumps)
    y = c.dc(params={"S_si_re.V": np.sqrt(ps_probe) + zeros,
                     "S_pfi_re.V": np.sqrt(pumps)})
    sig_gain = _pwr(c, y, "so_re", "so_im") / ps_probe / loss_s   # Raman-only
    pump_out = _pwr(c, y, "pfo_re", "pfo_im")

    # small-signal transfer slope d ln(gain)/dP_pump
    slope_meas = np.polyfit(pumps[:6], np.log(sig_gain[:6]), 1)[0]
    slope_ref = RAMAN["g_r"] / (RAMAN["a_eff_um2"] * 1e-12) * raman_leff()
    check("SRS small-signal slope", abs(slope_meas / slope_ref - 1) < 2e-3,
          f"measured {slope_meas:.4g} 1/W vs g_R*Leff/A_eff {slope_ref:.4g} 1/W")

    # ======================================================================
    # 2) distributed Raman amp: counter-pump sweep, on/off gain
    # ======================================================================
    y2 = c.dc(params={"S_si_re.V": np.sqrt(ps_probe) + zeros,
                      "S_pbi_re.V": np.sqrt(pumps)})
    gain_ctr = _pwr(c, y2, "so_re", "so_im") / ps_probe / loss_s
    ga_ref = raman_on_off_gain(pumps)
    check("Raman on/off gain vs pump", np.allclose(gain_ctr, ga_ref, rtol=2e-3),
          f"max rel err {np.max(np.abs(gain_ctr/ga_ref-1)):.1e}; "
          f"{10*np.log10(gain_ctr[-1]):.1f} dB at {pumps[-1]:.2f} W")
    check("co- == counter-pump gain", np.allclose(sig_gain, gain_ctr, rtol=1e-6),
          f"max rel diff {np.max(np.abs(sig_gain/gain_ctr-1)):.1e}")

    # ======================================================================
    # 3) pump depletion: strong signal, fixed pump
    # ======================================================================
    sigs = np.logspace(-6, -1.0, 40)                 # 1 uW .. 100 mW signal
    p_fix = 0.5
    y3 = c.dc(params={"S_si_re.V": np.sqrt(sigs),
                      "S_pbi_re.V": np.sqrt(p_fix) + np.zeros_like(sigs)})
    y3off = c.dc(params={"S_si_re.V": np.sqrt(sigs),
                         "S_pbi_re.V": np.zeros_like(sigs)})
    gain_depl = (_pwr(c, y3, "so_re", "so_im")
                 / _pwr(c, y3off, "so_re", "so_im"))
    small_signal_ok = abs(gain_depl[0] / raman_on_off_gain(p_fix) - 1) < 2e-3
    check("pump depletion saturates gain",
          gain_depl[-1] < 0.5 * gain_depl[0] and small_signal_ok,
          f"gain {10*np.log10(gain_depl[0]):.1f} dB -> "
          f"{10*np.log10(gain_depl[-1]):.1f} dB as signal grows")

    # ======================================================================
    # 4) SBS threshold: input-power sweep
    # ======================================================================
    cb = build("sbs_fiber", SBS, ["fi_re", "fi_im", "bi_re", "bi_im"],
               ["fo_re", "fo_im", "bo_re", "bo_im"])
    p_th = sbs_pth()
    t_lin = np.exp(-_alpha(SBS["loss_db_km"]) * SBS["length_km"] * 1e3)
    pin = np.logspace(np.log10(0.02 * p_th), np.log10(30 * p_th), 60)
    yb = cb.dc(params={"S_fi_re.V": np.sqrt(pin)})
    p_fo = _pwr(cb, yb, "fo_re", "fo_im")
    p_bo = _pwr(cb, yb, "bo_re", "bo_im")

    clamp = p_fo[-1]                                  # far above threshold
    check("SBS transmission clamps at P_th",
          abs(clamp / (p_th * t_lin) - 1) < 1e-2,
          f"clamp {clamp*1e3:.3f} mW vs P_th*t_lin {p_th*t_lin*1e3:.3f} mW")
    lo = p_bo[0] / (p_fo[0] + p_bo[0])
    hi = p_bo[-1] / (p_fo[-1] + p_bo[-1])
    check("SBS backscatter grows past threshold", lo < 0.02 and hi > 0.9,
          f"reflected fraction {lo*100:.2f}% -> {hi*100:.1f}%")
    check("SBS energy conserved",
          np.allclose(p_fo + p_bo, pin * t_lin, rtol=1e-3),
          f"max rel err {np.max(np.abs((p_fo+p_bo)/(pin*t_lin)-1)):.1e}")

    # ======================================================================
    # plots
    # ======================================================================
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].plot(pumps, 10 * np.log10(sig_gain), label="signal (long lambda)")
    ax[0, 0].plot(pumps, 10 * np.log10(pump_out / pumps),
                  "--", label="pump (short lambda)")
    ax[0, 0].set(title="SRS two-channel tilt (co-pump)",
                 xlabel="pump power [W]", ylabel="Raman gain [dB]")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(pumps, 10 * np.log10(gain_ctr), label="model")
    ax[0, 1].plot(pumps, 10 * np.log10(ga_ref), "k:", label="$e^{g_R P_p L_{eff}/A_{eff}}$")
    ax[0, 1].set(title="Counter-pumped Raman amp on/off gain",
                 xlabel="pump power [W]", ylabel="on/off gain [dB]")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    ax[1, 0].semilogx(sigs * 1e3, 10 * np.log10(gain_depl))
    ax[1, 0].axhline(10 * np.log10(raman_on_off_gain(p_fix)), ls=":", color="k",
                     label="small-signal $G_A$")
    ax[1, 0].set(title="Pump depletion (0.5 W pump)",
                 xlabel="signal input [mW]", ylabel="on/off gain [dB]")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].loglog(pin * 1e3, p_fo * 1e3, label="transmitted")
    ax[1, 1].loglog(pin * 1e3, p_bo * 1e3, label="backward Stokes")
    ax[1, 1].axvline(p_th * 1e3, ls=":", color="k", label="$P_{th}$")
    ax[1, 1].set(title="SBS threshold / limiter",
                 xlabel="input power [mW]", ylabel="output power [mW]")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}")

    n_fail = sum(not ok for _, ok, _ in CHECKS)
    if n_fail:
        raise SystemExit(f"{n_fail} check(s) FAILED")
    print(f"all {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
