"""Physics invariants of the Verilog-A photonic models, run through ngspice.

These pin the *compiled OSDI* behaviour to the analytic model equations, so
they catch model regressions and toolchain breakage alike.
"""
import numpy as np
import pytest

import lightspice as ls


def _laser_op(eng, v_drive: float, **params) -> float:
    ckt = ls.Circuit("laser op")
    ckt.raw(f"Vdrv drv 0 {v_drive}")
    ckt.device(ls.va("laser_dml"), "ld", "drv", "0", "popt", "0", **params)
    return float(eng.op(ckt)["popt"][0])


def test_laser_dml_li_curve(eng):
    # P = slope * ((V-Von)/Rs - Ith), clamped at 0
    assert _laser_op(eng, 1.2) == pytest.approx(0.0, abs=1e-9)
    assert _laser_op(eng, 1.25) == pytest.approx(0.0, abs=1e-9)      # at threshold
    assert _laser_op(eng, 1.5) == pytest.approx(0.015, rel=1e-3)     # 60mA -> 15mW
    assert _laser_op(eng, 1.5, slope=0.5, ith=20e-3) == pytest.approx(0.020, rel=1e-3)


def test_laser_dml_tau_dynamics(eng):
    """Optical step response must be first-order with time constant tau."""
    tau = 100e-12
    ckt = ls.Circuit("laser step")
    ckt.raw("Vdrv drv 0 PULSE(1.2 1.5 0.2n 1p 1p 5n 10n)")
    ckt.device(ls.va("laser_dml"), "ld", "drv", "0", "popt", "0", tau=tau)
    r = eng.tran(ckt, "2p", "1.2n")
    p = r["popt"] / 0.015
    t63 = r.t[np.argmax(p > 0.632)] - 0.2e-9
    assert t63 == pytest.approx(tau, rel=0.15)


def test_photodiode_responsivity(eng):
    ckt = ls.Circuit("pd op")
    ckt.raw("Vp popt 0 2m", "Vb cat 0 0")  # 2 mW optical in
    ckt.device(ls.va("photodiode"), "pd", "popt", "0", "0", "cat", r=0.7)
    res = eng.op(ckt)
    # photocurrent flows anode->cathode: I into Vb = R*P + Idk
    assert float(res.i("vb")[0]) == pytest.approx(0.7 * 2e-3 + 1e-9, rel=1e-6)


def test_mzm_transfer(eng):
    def t_of(v: float) -> float:
        ckt = ls.Circuit("mzm")
        ckt.raw("Vcw pin 0 1m", f"Vd vp 0 {v}")
        ckt.device(ls.va("mzm"), "m", "pin", "vp", "0", "pout", "0",
                   vpi=3.0, il_db=3.0, er_db=20.0)
        return float(eng.op(ckt)["pout"][0]) / 1e-3

    il = 10 ** (-0.3)
    assert t_of(0.0) == pytest.approx(il * (0.5 + 0.5 * 99 / 101), rel=1e-3)
    assert t_of(1.5) == pytest.approx(il * 0.5, rel=1e-3)            # quadrature
    assert t_of(0.0) / t_of(3.0) == pytest.approx(100, rel=1e-2)     # 20 dB ER


def test_laser_rate_threshold_and_dynamics(eng):
    # DC: below threshold ~nothing, above threshold mW-class output
    def p_at(v: float) -> float:
        ckt = ls.Circuit("li")
        ckt.raw(f"Vdrv drv 0 {v}")
        ckt.device(ls.va("laser_rate"), "ld", "drv", "0", "popt", "0")
        return float(eng.op(ckt)["popt"][0])

    assert p_at(1.25) < 5e-6            # 10 mA, below ~19 mA threshold
    p_hi = p_at(1.39)                   # 38 mA ~ 2x threshold
    assert 1e-3 < p_hi < 10e-3

    # transient: relaxation-oscillation overshoot then settle to the DC point
    ckt = ls.Circuit("step")
    ckt.raw("Vdrv drv 0 PULSE(1.25 1.39 0.5n 20p 20p 4n 8n)")
    ckt.device(ls.va("laser_rate"), "ld", "drv", "0", "popt", "0")
    r = eng.tran(ckt, "1p", "4n")
    p = r["popt"]
    p_ss = p[(r.t > 3.5e-9)].mean()
    assert p_ss == pytest.approx(p_hi, rel=0.05)   # tran settles to the OP
    assert p.max() > 2.0 * p_ss                    # ringing overshoot


def test_fiber_loss_and_delay(eng):
    ckt = ls.Circuit("fiber")
    ckt.raw("Vp ptx 0 PULSE(0 1m 0.1n 1p 1p 2n 4n)")
    ls.add_fiber(ckt, "f1", "ptx", "prx", loss_db=3.0, delay=100e-12)
    ckt.raw("Rload prx 0 1g")
    r = eng.tran(ckt, "1p", "1n")
    k = 10 ** (-0.3)
    assert r["prx"].max() == pytest.approx(k * 1e-3, rel=1e-3)
    # measure the delay between 50% crossings
    t_tx = r.t[np.argmax(r["ptx"] > 0.5e-3)]
    t_rx = r.t[np.argmax(r["prx"] > 0.5 * k * 1e-3)]
    assert (t_rx - t_tx) == pytest.approx(100e-12, abs=5e-12)
