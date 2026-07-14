"""End-to-end link: full analog + photonic chain solved in one matrix."""
import numpy as np
import pytest

import lightspice as ls


def build_link() -> ls.Circuit:
    ckt = ls.Circuit("ideal link")
    ckt.raw("Vdrv drv 0 PULSE(1.30 1.55 0.5n 50p 50p 0.5n 1.5n)")
    ckt.device(ls.va("laser_dml"), "laser", "drv", "0", "popt_tx", "0")
    ls.add_fiber(ckt, "fib", "popt_tx", "popt_rx", loss_db=3.0)
    ckt.device(ls.va("photodiode"), "pd", "popt_rx", "0", "pd_an", "pd_cat")
    ckt.raw("Ranch pd_an 0 1", "Eopamp vout 0 0 pd_cat 1e6", "Rf vout pd_cat 200")
    return ckt


def test_link_levels(eng):
    r = eng.tran(build_link(), "10p", "4n")
    # logic-1: 70 mA -> 18 mW; fiber 3 dB; PD 0.8 A/W; TIA -200 V/A
    assert r["popt_tx"].max() == pytest.approx(18e-3, rel=0.01)
    k = 10 ** (-0.3)
    assert r["popt_rx"].max() == pytest.approx(18e-3 * k, rel=0.01)
    i_pd_pk = 0.8 * 18e-3 * k
    assert r["vout"].min() == pytest.approx(-200 * i_pd_pk, rel=0.02)


def test_link_settles_between_bits(eng):
    r = eng.tran(build_link(), "10p", "4n")
    # late in the low half-period (>10 tau after the falling edge at 1.05 ns)
    # the laser has decayed to the logic-0 power
    lo = (r.t > 1.7e-9) & (r.t < 1.95e-9)
    p0 = 0.3 * (((1.30 - 1.2) / 5.0) - 10e-3)   # 3 mW
    assert np.median(r["popt_tx"][lo]) == pytest.approx(p0, rel=0.05)


def test_sky130_link_bias(eng, sky130_available):
    if not sky130_available:
        import pytest as _pytest

        _pytest.skip("SKY130 PDK not installed")
    ckt = ls.Circuit("sky130 tia bias")
    ckt.lib(ls.sky130_lib(), "tt")
    ckt.raw(
        """
        V_DDA v_dda 0 DC 1.8
        XM_p vout tia_in v_dda v_dda sky130_fd_pr__pfet_01v8 w=40 l=0.18 mult=1
        XM_n vout tia_in 0     0     sky130_fd_pr__nfet_01v8 w=15 l=0.18 mult=1
        R_f  vout tia_in 1k
        """
    )
    r = eng.op(ckt)
    # self-biased inverter: input and output meet near the trip point
    assert float(r["tia_in"][0]) == pytest.approx(float(r["vout"][0]), abs=1e-3)
    assert 0.5 < float(r["vout"][0]) < 1.3
