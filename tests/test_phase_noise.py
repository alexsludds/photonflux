"""Laser phase noise / linewidth (ALE-76).

The noisy CW laser (``catalog._cw_laser_noisy``) adds a Wiener phase process on
top of the existing RIN amplitude noise: phi(t) integrates white increments of
per-step variance ``2*pi*dnu*dt_n`` so ``Var(phi(t)) = 2*pi*dnu*t``. That gives
a field-coherence decay ``exp(-pi*dnu*|tau|)`` — a Lorentzian line of FWHM =
dnu. These tests pin that physics through the compiled circulax source and pin
the webapp wiring that routes ``linewidth_hz`` to the noisy variant.

Run with:  .venv/bin/python -m pytest tests/test_phase_noise.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

circulax = pytest.importorskip("circulax")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
from circulax import compile_circuit  # noqa: E402
from circulax.components.base_component import Signals, States, component  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp"))

import catalog  # noqa: E402
import simulate  # noqa: E402
import wavesrc  # noqa: E402


# --- a tiny matched optical load so the field source has a current path ------
@component(ports=("po_p", "po_n"))
def _Absorber(signals: Signals, s: States, Yopt: float = 1.0):
    e = signals.po_p - signals.po_n
    return {"po_p": Yopt * e, "po_n": -Yopt * e}, {}


def _laser_circuit(bank, dt_n):
    """Compile ``noisy CW laser -> matched absorber`` and expose the field."""
    laser = catalog._cw_laser_noisy(bank, dt_n)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LAS": {"component": "laser"},
            "TERM": {"component": "absorber"},
        },
        "connections": {
            "GND,p1": ("LAS,p2", "TERM,po_n"),
            "LAS,p1": "TERM,po_p",
        },
        "ports": {"eout": "LAS,p1"},
    }
    models = {"ground": lambda: 0, "laser": laser, "absorber": _Absorber}
    return compile_circuit(net, models, backend="dense", is_complex=True)


def _solve_field(circuit, settings, dt_n, t_stop, seed_idx=0):
    """Complex field E(t) sampled on the fixed BDF2 step grid (= noise grid).

    The source pins ``E = field(t)`` algebraically at every step, so the saved
    samples are the exact field at those instants (no integration error)."""
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    ts = np.arange(0.0, t_stop, dt_n)
    params = {f"LAS.{k}": float(v) for k, v in settings.items()}
    params["LAS.seed_idx"] = float(seed_idx)
    y0 = circuit.dc(params=params)
    sol = circuit.transient(
        t0=0.0, t1=t_stop, dt0=dt_n, y0=y0,
        saveat=diffrax.SaveAt(ts=ts), params=params,
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=circuit.solver, newton_max_steps=40),
        max_steps=len(ts) + 16, throw=False,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    assert sol.result == diffrax.RESULTS.successful, sol.result
    return np.asarray(sol.ts), np.asarray(circuit.port(sol.ys, "eout"))


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

def test_linewidth_zero_is_deterministic():
    """dnu = 0 (and RIN off) recovers the clean constant field exactly."""
    bank, dt_n = wavesrc.noise_bank("LAS", 2, 5e-7, bw=1e9, master_seed=1)
    circuit = _laser_circuit(bank, dt_n)
    power, phase = 2e-3, 0.3
    t, E = _solve_field(circuit, {"power": power, "phase": phase,
                                  "linewidth_hz": 0.0, "rin_db": 0.0},
                        dt_n, 2e-7)
    assert np.allclose(np.abs(E), np.sqrt(power), rtol=1e-9)
    assert np.allclose(np.angle(E), phase, atol=1e-9)


def test_phase_noise_wiener_variance_sets_linewidth():
    """The phase structure function D(tau) = <(phi(t+tau)-phi(t))^2> has slope
    2*pi*dnu, i.e. a Lorentzian FWHM = dnu (here 2 MHz)."""
    dnu = 2e6
    bw = 1e9                       # >> dnu so the walk is white over the line
    seeds = 8
    t_stop = 2e-6
    bank, dt_n = wavesrc.noise_bank("LAS", seeds * 2, t_stop, bw=bw,
                                    master_seed=7)
    circuit = _laser_circuit(bank, dt_n)

    incr = []                      # phase increment over a fixed lag, per seed
    lag = 40                       # samples -> tau = lag*dt_n
    for k in range(seeds):
        _t, E = _solve_field(circuit, {"power": 1e-3, "linewidth_hz": dnu},
                             dt_n, t_stop, seed_idx=k)
        phi = np.unwrap(np.angle(E))
        incr.append(phi[lag:] - phi[:-lag])
    tau = lag * dt_n
    d_tau = np.mean(np.concatenate(incr) ** 2)
    slope = d_tau / tau
    # ensemble+time averaged over ~8*(N) increments; expect within ~20%
    assert slope == pytest.approx(2 * np.pi * dnu, rel=0.2)


def test_phase_noise_independent_of_rin_stream():
    """RIN (even rows) and phase noise (odd rows) draw from independent bank
    streams, so turning RIN on leaves the phase walk unchanged."""
    bank, dt_n = wavesrc.noise_bank("LAS", 2, 5e-7, bw=1e9, master_seed=3)
    circuit = _laser_circuit(bank, dt_n)
    common = {"power": 1e-3, "linewidth_hz": 5e6}
    _t, E0 = _solve_field(circuit, {**common, "rin_db": 0.0}, dt_n, 3e-7)
    _t, E1 = _solve_field(circuit, {**common, "rin_db": -150.0}, dt_n, 3e-7)
    # identical phase (odd stream), different magnitude (even stream now live)
    assert np.allclose(np.angle(E0), np.angle(E1), atol=1e-9)
    assert not np.allclose(np.abs(E0), np.abs(E1), atol=1e-9)


def test_osa_lineshape_is_lorentzian():
    """Blackman-Harris FFT of a long single realization shows a Lorentzian of
    FWHM ~ dnu (checked against the analytic 3 dB half-width)."""
    dnu = 4e6
    bw = 2e9
    t_stop = 8e-6                  # resolution 1/t_stop = 125 kHz << dnu
    bank, dt_n = wavesrc.noise_bank("LAS", 2, t_stop, bw=bw, master_seed=11)
    circuit = _laser_circuit(bank, dt_n)
    t, E = _solve_field(circuit, {"power": 1e-3, "linewidth_hz": dnu},
                        dt_n, t_stop)
    n = len(E)
    k = np.arange(n)
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / (n - 1))
           + 0.14128 * np.cos(4 * np.pi * k / (n - 1))
           - 0.01168 * np.cos(6 * np.pi * k / (n - 1)))
    S = np.abs(np.fft.fftshift(np.fft.fft(E * win))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, dt_n))
    # smooth the single-realization periodogram before reading the width
    box = max(int(round(dnu / (f[1] - f[0]) / 4)), 3)
    S = np.convolve(S, np.ones(box) / box, mode="same")
    half = f[S >= 0.5 * S.max()]
    fwhm = half.max() - half.min()
    assert fwhm == pytest.approx(dnu, rel=0.5)


# ---------------------------------------------------------------------------
# webapp wiring: linewidth_hz routes to the noisy variant
# ---------------------------------------------------------------------------

def _laser_sch(settings):
    return {
        "instances": {"GND": {"type": "ground"},
                      "LAS": {"type": "cw_laser", "settings": settings}},
        "wires": [["LAS,p2", "GND,p1"]],
        "probes": [{"name": "e", "at": "LAS,p1"}],
    }


def test_linewidth_selects_noisy_variant():
    noise_cfg = {"seeds": 2, "bw": 50e9, "seed": 1}
    _net, meta = simulate.schematic_to_netlist(
        _laser_sch({"linewidth_hz": 1e6}), 4e-9, noise_cfg)
    assert "_cwn:LAS" in meta["noisy"]
    assert "LAS" in meta["noise_insts"]
    # two bank rows per seed: RIN (even) + phase walk (odd)
    _kind, bank, _dt = meta["noisy"]["_cwn:LAS"]
    assert bank.shape[0] == 2 * noise_cfg["seeds"]


def test_no_noise_keeps_clean_laser():
    # linewidth = 0 and RIN = 0 under a noise run -> plain deterministic laser
    _net, meta = simulate.schematic_to_netlist(
        _laser_sch({"linewidth_hz": 0.0, "rin_db": 0.0}), 4e-9,
        {"seeds": 2, "bw": 50e9, "seed": 1})
    assert meta["noisy"] == {}
    assert "LAS" not in meta["noise_insts"]
