"""Netlist-level wiring of a coherent link (``webapp/simulate.py``): a QAM
source pair -> IQ modulator -> coherent receiver (with an LO laser) compiles,
and the QAM sources are recorded for the coherent report.

This pins the plumbing added for ALE-77 (QAM waveform routing + pattern
recording + component instantiation) without the expensive transient solve;
the detection physics is in ``test_coherent_components.py`` and the DSP in
``test_coherent.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "webapp"))
sys.path.insert(0, str(_ROOT))

import simulate  # noqa: E402


def _coherent_schematic(ui=40e-12, qam="qpsk"):
    base = {"order": 15, "seed": 1, "ui": ui, "sps": 8, "rrc_beta": 0.1,
            "qam": qam, "v0": -3.0, "v1": 3.0}
    instances = {
        "GND": {"type": "ground"},
        "LS": {"type": "cw_laser",
               "settings": {"power": 1e-3, "wavelength_nm": 1550.0}},
        "LO": {"type": "cw_laser",
               "settings": {"power": 1e-3, "wavelength_nm": 1550.0}},
        "SI": {"type": "prbs", "settings": {**base, "mode": "qam",
                                            "qam_drive": "i"}},
        "SQ": {"type": "prbs", "settings": {**base, "mode": "qam",
                                            "qam_drive": "q"}},
        "MOD": {"type": "iq_modulator", "settings": {"vpi": 3.0, "il_db": 0.0}},
        "RX": {"type": "coherent_rx", "settings": {"R": 0.8}},
        "RI": {"type": "resistor", "settings": {"R": 1.0}},
        "RQ": {"type": "resistor", "settings": {"R": 1.0}},
    }
    wires = [
        ["LS,p2", "GND,p1"], ["LO,p2", "GND,p1"],
        ["SI,p2", "GND,p1"], ["SQ,p2", "GND,p1"],
        ["MOD,vin", "GND,p1"], ["MOD,vqn", "GND,p1"],
        ["RX,i_n", "GND,p1"], ["RX,q_n", "GND,p1"],
        ["RI,p2", "GND,p1"], ["RQ,p2", "GND,p1"],
        ["LS,p1", "MOD,pin"], ["MOD,pout", "RX,sig"], ["LO,p1", "RX,lo"],
        ["SI,p1", "MOD,vip"], ["SQ,p1", "MOD,vqp"],
        ["RX,i_p", "RI,p1"], ["RX,q_p", "RQ,p1"],
    ]
    probes = [{"name": "I", "at": "RX,i_p"}, {"name": "Q", "at": "RX,q_p"}]
    return {"instances": instances, "wires": wires, "probes": probes}


def test_coherent_link_compiles_and_records_qam():
    sch = _coherent_schematic()
    circuit, meta, log = simulate._get_circuit(sch, wave_span=2.56e-8)
    assert circuit is not None
    # both QAM rails are recorded as patterns with the qam mode
    pats = meta.get("patterns") or {}
    assert "SI" in pats and "SQ" in pats
    assert pats["SI"]["mode"] == "qam" and pats["SI"]["qam"] == "qpsk"
    assert pats["SI"]["qam_drive"] == "i" and pats["SQ"]["qam_drive"] == "q"


@pytest.mark.parametrize("qam", ["qpsk", "qam16", "qam64"])
def test_coherent_link_all_orders_compile(qam):
    sch = _coherent_schematic(qam=qam)
    circuit, meta, _log = simulate._get_circuit(sch, wave_span=2.56e-8)
    assert circuit is not None
    assert (meta["patterns"]["SI"]["qam"]) == qam


def test_coherent_link_transient_report():
    # full path: QAM drives -> IQ modulator -> coherent RX -> coherent_report
    sch = _coherent_schematic(ui=100e-12, qam="qpsk")   # 10 GBaud
    payload = {"schematic": sch,
               "analysis": {"mode": "transient", "t_stop": 30e-9,
                            "points": 2400,
                            "coherent": {"probe_i": "I", "probe_q": "Q",
                                         "cpr": "vv"}}}
    res = simulate.run(payload)
    assert res.get("ok"), res.get("log")
    rep = res.get("coherent")
    assert rep is not None and rep["name"] == "QPSK"
    assert rep["counted"]["symbols"] > 100
    # the recovered constellation is closed and error-free at this OSNR
    assert rep["evm_rms"] < 0.35
    assert rep["counted"]["ber"] < 1e-2
    assert len(rep["rx_re"]) > 50 and len(rep["const_re"]) == 4
