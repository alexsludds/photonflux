"""One optical sign convention: the engineering carrier ``e^{+j*w*t}``.

circulax's waveguides fix it (a propagation phase ``e^{-j*phi}`` and a Pade
delay ``e^{-j*w*tau}``), so every optical model follows: a laser blue of the
WDM reference is a positive envelope tone, a CMT ring detunes as
``-j*delta``, a fibre delays a tone at ``+f`` by ``beta2*L*w``, and the
optical-spectrum display maps ``+f`` to ``f0 + f``.

The lasers, CMT rings, SOA/EDFA phase, optical filter and spectrum display
used to run on the physics ``e^{-i*w*t}`` instead, which was invisible inside
either family but mirrored every WDM tone about the reference wherever the two
met. The first two pins mix the families and fail on that split; the last two
guard the laser against the ring and the spectrum display.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

C0 = 299792458.0


@pytest.fixture(scope="module")
def sim():
    import jax

    jax.config.update("jax_enable_x64", True)
    import simulate

    return simulate


def _laser(frame_nm: float, laser_nm: float | None) -> dict:
    """1 mW CW laser; a tone at ``laser_nm`` in the ``frame_nm`` baseband."""
    s = {"power": 1e-3, "wavelength_nm": laser_nm or frame_nm}
    if laser_nm:
        s["ref_wavelength_nm"] = frame_nm
    return {"type": "cw_laser", "settings": s}


def _loop_ring(frame_nm: float, laser_nm: float | None = None) -> dict:
    """Laser -> all-pass ring of a coupler and a 100 um waveguide with its
    real group delay (FSR ~750 GHz, near critical coupling, ~21 GHz wide)."""
    return {
        "instances": {
            "LAS": _laser(frame_nm, laser_nm),
            "GND": {"type": "ground", "settings": {}},
            "CPL": {"type": "dir_coupler", "settings": {"coupling": 0.088}},
            "WG": {"type": "waveguide", "settings": {
                "length_m": 100e-6, "loss_dB_cm": 40.0, "neff": 2.4,
                "n_group": 4.0, "center_wavelength_nm": 1550.0,
                "wavelength_nm": frame_nm, "transit": "delayed"}},
            "TT": {"type": "opt_term", "settings": {}},
        },
        "wires": [["LAS,p1", "CPL,p1"], ["LAS,p2", "GND,p1"],
                  ["CPL,p4", "WG,p2"], ["WG,p1", "CPL,p2"],
                  ["CPL,p3", "TT,p1"]],
        "probes": [{"name": "thru", "at": "CPL,p3"}],
    }


def _settled_power(sim, sch: dict, t_stop: float) -> float:
    res = sim.run({"schematic": sch, "analysis": {
        "mode": "transient", "t_stop": t_stop, "points": 400}})
    assert res["ok"], res.get("error")
    t = np.asarray(res["x"])
    p = np.asarray(res["traces"][0]["values"])
    return float(p[t > 0.75 * t_stop].mean())


def _mirror_nm(lam_nm: float, frame_nm: float) -> float:
    """The wavelength at the same optical-frequency offset on the other side."""
    f, f0 = C0 / (lam_nm * 1e-9), C0 / (frame_nm * 1e-9)
    return C0 / (2.0 * f0 - f) * 1e9


def test_wdm_tone_lands_on_a_delayed_waveguide_ring_resonance(sim):
    """A wavelength sweep finds the loop's notch; a WDM laser tuned to that
    wavelength, as a tone in a frame 0.4 nm to the red, must sit in the same
    notch, and its frequency mirror must not. With the lasers on the physics
    convention the tone landed at the mirror and the notch was missed."""
    sweep = sim.run({"schematic": _loop_ring(1550.0), "analysis": {
        "mode": "dcsweep", "instance": "WG", "param": "wavelength_nm",
        "start": 1547.0, "stop": 1553.0, "points": 1201}})
    assert sweep["ok"], sweep.get("error")
    lam_n = float(np.asarray(sweep["x"])[
        np.argmin(sweep["traces"][0]["values"])])
    frame = lam_n + 0.4                 # the notch sits ~50 GHz blue of it
    p_res = _settled_power(sim, _loop_ring(frame, lam_n), 200e-12)
    p_mir = _settled_power(sim, _loop_ring(frame, _mirror_nm(lam_n, frame)),
                           200e-12)
    assert p_mir > 0.5                   # mW: off resonance it passes
    assert p_res < 0.2 * p_mir


def _envelope_hz(laser_nm: float, frame_nm: float) -> float:
    """Rotation rate of cx.cw_laser's field in the frame's baseband."""
    import diffrax
    from circulax import compile_circuit

    from photonflux import cx, solvers

    net = {"instances": {"GND": {"component": "ground"},
                         "L": {"component": "laser",
                               "settings": {"wavelength_nm": laser_nm,
                                            "ref_wavelength_nm": frame_nm}},
                         "T": {"component": "term"}},
           "connections": {"L,p1": "T,p1", "GND,p1": "L,p2"},
           "ports": {"e": "L,p1"}}
    c = compile_circuit(net, {"ground": lambda: 0, "laser": cx.cw_laser(),
                              "term": cx.terminator()},
                        backend="dense", is_complex=True)
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    t1 = 2e-12
    sol = solvers.transient(
        c, t0=0.0, t1=t1, dt0=1e-15, y0=solvers.dc(c),
        saveat=diffrax.SaveAt(ts=np.array([0.0, t1])), max_steps=2100,
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=c.solver), throw=False,
        stepsize_controller=diffrax.ConstantStepSize())
    e = np.asarray(c.port(sol.ys, "e"))
    return float(np.angle(e[1] / e[0]) / (2.0 * np.pi * t1))


def test_bluer_laser_arrives_first_on_positive_dispersion():
    """On standard fibre (D > 0, beta2 < 0) the shorter wavelength travels
    faster: fiber_cd's group delay at the blue laser's envelope tone must be
    beta2*L*(w_blue - w_red) shorter than at the red one. With the lasers on
    the physics convention the blue tone sat at -f and arrived last."""
    import lti

    frame = 1550.0
    blue, red = 1549.6, 1550.4                       # ~ +-50 GHz
    f_blue, f_red = _envelope_hz(blue, frame), _envelope_hz(red, frame)
    assert f_blue > 0 > f_red
    assert f_blue == pytest.approx(C0 / (blue * 1e-9) - C0 / (frame * 1e-9),
                                   rel=1e-6)

    L_km, D = 2.0, 17.0
    log: list = []
    _, payload = lti.build("fiber", {"length_km": L_km, "D_ps": D,
                                     "lambda_nm": frame, "atten_db_km": 0.0,
                                     "fit_bw": 120e9, "n_poles": 32}, log)
    poles, res, d = (np.asarray(x) for x in payload["cplx"])

    def group_delay(f):
        w = 2.0 * np.pi * np.array([f - 1e8, f + 1e8])
        h = d + (res[None, :] / (1j * w[:, None] - poles[None, :])).sum(1)
        return -np.diff(np.unwrap(np.angle(h)))[0] / np.diff(w)[0]

    beta2 = -D * 1e-6 * (frame * 1e-9) ** 2 / (2.0 * np.pi * C0)
    want = beta2 * L_km * 1e3 * 2.0 * np.pi * (f_blue - f_red)
    got = group_delay(f_blue) - group_delay(f_red)
    assert got < 0.0
    assert got == pytest.approx(want, rel=0.05)


def test_wdm_tone_lands_on_a_cmt_ring_resonance(sim):
    """The CMT microring (ring_mod.va) and the laser share the convention: a
    tone at the ring's resonance wavelength is notched, its mirror is not."""
    frame, lam_res = 1310.0, 1309.75                 # resonance ~44 GHz blue

    def ring(laser_nm):
        return {"instances": {
            "LAS": _laser(frame, laser_nm),
            "GND": {"type": "ground", "settings": {}},
            "RING": {"type": "ring_mod", "settings": {
                "lambda_nm": frame, "lambda_res_nm": lam_res,
                "kappa2": 0.076, "loss_db_m": 7000.0}},   # ~critical, 38 GHz
            "VB": {"type": "vdc", "settings": {"V": 0.0}},
            "TT": {"type": "opt_term", "settings": {}}},
            "wires": [["LAS,p1", "RING,pin"], ["LAS,p2", "GND,p1"],
                      ["RING,pout", "TT,p1"], ["RING,vp", "VB,p1"],
                      ["RING,vn", "GND,p1"], ["RING,gnd", "GND,p1"],
                      ["VB,p2", "GND,p1"]],
            "probes": [{"name": "thru", "at": "RING,pout"}]}

    p_res = _settled_power(sim, ring(lam_res), 300e-12)
    p_mir = _settled_power(sim, ring(_mirror_nm(lam_res, frame)), 300e-12)
    assert p_mir > 0.8
    assert p_res < 0.3 * p_mir


def test_optical_spectrum_puts_a_wdm_laser_at_its_wavelength(sim):
    frame, lam = 1550.0, 1549.6
    sch = {"instances": {"LAS": _laser(frame, lam),
                         "GND": {"type": "ground", "settings": {}},
                         "TT": {"type": "opt_term", "settings": {}}},
           "wires": [["LAS,p1", "TT,p1"], ["LAS,p2", "GND,p1"]],
           "probes": [{"name": "o", "at": "TT,p1", "spectrum": True}]}
    res = sim.run({"schematic": sch, "analysis": {
        "mode": "transient", "t_stop": 100e-12, "points": 2000,
        "solver": "bdf2", "dtmax": 2e-14}})
    assert res["ok"], res.get("error")
    osa = res["extra_plots"][0]
    x = np.asarray(osa["x"])
    peak = float(x[np.argmax(osa["traces"][0]["values"])])
    assert peak == pytest.approx(lam, abs=0.02)
