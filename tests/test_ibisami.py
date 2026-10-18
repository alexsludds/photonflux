"""IBIS-AMI: models, the PyAMI-based loader and the link engine.

* the photonflux models (pf_tx, pf_rx) obey the AMI contract: Init and
  GetWave agree, GetWave state survives any block size, the Rx's DFE
  zero-forces in Init and converges by LMS in GetWave;
* ``photonflux.ibisami`` reads parameters through PyAMI correctly (List
  defaults, integer tap names), runs a model in-process and out-of-process
  with identical results, and identifies foreign binaries;
* ``photonflux.serdes``: the statistical eye matches closed-form BER, the
  statistical and time-domain flows agree, and the data-aided channel
  estimate recovers a known channel.

The models are compiled for this machine on first use (needs a C compiler).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyibisami")

from photonflux import ami_build, ibisami, serdes  # noqa: E402

UI = 1 / 25e9
NSPUI = 32
TS = UI / NSPUI


@pytest.fixture(scope="module", autouse=True)
def built():
    if ami_build.c_compiler() is None:
        pytest.skip("no C compiler to build the IBIS-AMI models")
    step = ami_build.build_photonflux()
    assert step.ok, step.detail


def two_tap_channel(post: float = 0.4, n_ui: int = 64) -> serdes.Channel:
    """Pulse response 1 at cursor 0 and ``post`` at cursor 1, built as an
    impulse (a 1-UI-wide box is a pulse of exactly those cursors)."""
    h = np.zeros(n_ui * NSPUI)
    h[4 * NSPUI] = 1.0 / TS
    h[5 * NSPUI] = post / TS
    return serdes.Channel(h, TS)


# --- parameters (PyAMI parsing) ---------------------------------------------

def test_list_defaults_and_validation():
    rx = ibisami.load("pf_rx")
    # .ami says (List 0 1 2) (Default 2) with List_Tip; PyAMI's own trait
    # default would pick 0
    assert rx.params["ctle_mode"] == 2
    assert rx.params["dfe.dfe_mode"] == 2
    assert rx.params["Modulation"] == "NRZ"
    with pytest.raises(ValueError):
        rx.params["ctle_mode"] = 7
    with pytest.raises(ValueError):
        rx.params["agc_target"] = 5.0
    with pytest.raises(KeyError):
        rx.params["no_such_param"] = 1
    s = rx.params.sexpr()
    assert s.startswith("(pf_rx ") and '(Modulation "NRZ")' in s
    assert "(ctle (ctle_mode 2)" in s


def test_parse_params_out():
    d = ibisami.parse_params("(pf_rx (level 0.2) (dfe_taps 0.3 -0.1) "
                             "(cdr (phase 0.5)) (name \"x\"))")
    assert d == {"level": 0.2, "dfe_taps": [0.3, -0.1], "cdr": {"phase": 0.5},
                 "name": "x"}
    assert ibisami.parse_params("not an s-expression")["_raw"]


def test_integer_tap_names_go_out_as_written():
    ami = ("(tx (Reserved_Parameters (Init_Returns_Impulse (Usage Info) (Type Boolean)"
           " (Value True)) (GetWave_Exists (Usage Info) (Type Boolean) (Value True)))"
           " (Model_Specific (tap_filter (-1 (Usage In) (Type Tap)"
           " (Range 0.0 -1.0 1.0)) (0 (Usage In) (Type Tap) (Range 1.0 -1.0 1.0)))))")
    p = ibisami.AmiParams(ami)
    p["tap_filter.-1"] = -0.2
    assert p["tap_filter.pre1"] == -0.2           # PyAMI's own name
    assert "(tap_filter (-1 -0.2) (0 1.0))" in p.sexpr()


def test_binary_platforms():
    assert ibisami.host_platform() in ibisami.binary_platform(
        ibisami.load("pf_rx").dll)
    elf386 = b"\x7fELF\x01\x01" + bytes(12) + struct.pack("<H", 3)
    pe64 = bytearray(256)
    pe64[:2] = b"MZ"
    pe64[0x3C:0x40] = struct.pack("<I", 0x80)
    pe64[0x80:0x86] = b"PE\0\0" + struct.pack("<H", 0x8664)
    macho = b"\xcf\xfa\xed\xfe" + struct.pack("<i", 0x0100000C)
    for blob, want in ((elf386, ("linux", "x86", "linux/386")),
                       (bytes(pe64), ("windows", "x86_64", None)),
                       (macho, ("darwin", "arm64", None))):
        f = Path(pytest.importorskip("tempfile").mkdtemp()) / "lib"
        f.write_bytes(blob + bytes(64))
        (p,) = ibisami.binary_platform(f)
        assert (p.os, p.arch, p.container) == want


# --- pf_tx ----------------------------------------------------------------

def test_tx_init_and_getwave_agree_for_any_block_size():
    tx = ibisami.load("pf_tx")
    tx.params.update(tx_tap_m1=-0.1, tx_tap_p1=-0.2, tx_bw=30e9)
    d = serdes.Channel.ideal(TS, n=64 * NSPUI, delay=0).impulse
    h = tx.init(d, TS, UI).impulse[0] * TS
    # FFE taps at UI spacing, main 2 UI behind the newest input (causal)
    assert tx.last_init.params_out["tx_taps"] == pytest.approx(
        [0, -0.1, 0.7, -0.2, 0, 0])
    x = serdes.stimulus(serdes.symbols(300), NSPUI)
    ref = np.convolve(x, h)[: len(x)]
    for bits in (7, 64, 300):
        tx.init(d, TS, UI)
        y = tx.getwave(x, block_bits=bits).wave
        assert np.max(np.abs(y - ref)) < 1e-12


# --- pf_rx ----------------------------------------------------------------

def test_rx_init_zero_forces_dfe():
    rx = ibisami.load("pf_rx")
    rx.params.update(ctle_mode=0, dfe_ntaps=2, agc_mode=0)
    r = rx.init(two_tap_channel(0.4).impulse, TS, UI)
    assert r.params_out["dfe_taps"] == pytest.approx([0.4, 0.0], abs=1e-9)
    p = np.convolve(r.impulse[0], np.ones(NSPUI))[: len(r.impulse[0])] * TS
    n0 = int(np.argmax(p))
    assert abs(p[n0 + NSPUI]) < 1e-9 * p[n0]     # post-cursor notched out


@pytest.mark.parametrize("mod", ["NRZ", "PAM4"])
def test_rx_getwave_adapts_and_decodes(mod):
    ch = two_tap_channel(0.35)
    rx = ibisami.load("pf_rx")
    rx.params.update(ctle_mode=0, dfe_ntaps=1, dfe_mu=5e-3,
                     cdr_mode=1 if mod == "NRZ" else 2)
    td = serdes.time_domain_link(None, ch, rx, UI, modulation=mod, nsym=8000,
                                 noise_rms=0.01, skip=2000)
    assert td.counted["bit_errors"] == 0
    b1 = td.rx_trace.trace[-1]["dfe_taps"]
    assert (b1[0] if isinstance(b1, list) else b1) == pytest.approx(0.35, abs=0.05)


def test_out_of_process_host_is_bit_identical():
    ch = two_tap_channel(0.3)
    x = np.convolve(serdes.stimulus(serdes.symbols(3000), NSPUI),
                    ch.impulse)[: 3000 * NSPUI] * TS
    out = {}
    for host in ("inproc", "process"):
        with ibisami.load("pf_rx", host=host) as rx:
            r = rx.init(ch.impulse, TS, UI)
            gw = rx.getwave(x, block_bits=100)
            out[host] = (r.impulse, gw.wave, gw.clock_times, gw.trace[-1])
    a, b = out["inproc"], out["process"]
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2]) and a[3] == b[3]


# --- link engine ------------------------------------------------------------

@pytest.mark.parametrize("q", [5.0, 7.0])
def test_stat_eye_matches_gaussian_ber(q):
    p = serdes.Channel.ideal(TS, n=64 * NSPUI).pulse(UI)
    e = serdes.stat_eye(p, TS, UI, noise_rms=0.5 / q)
    assert e.min_ber() == pytest.approx(float(serdes.q_to_ber(q)), rel=1e-3)
    e4 = serdes.stat_eye(p, TS, UI, levels=4, noise_rms=(0.5 / 3) / q)
    assert e4.ber[1].min() == pytest.approx(float(serdes.q_to_ber(q)), rel=1e-3)


def test_stat_eye_isi_crosstalk_and_offsets_are_exact():
    p = np.zeros(64 * NSPUI)
    p[10 * NSPUI:11 * NSPUI] = 1.0
    p[11 * NSPUI:12 * NSPUI] = 0.3
    e = serdes.stat_eye(p, TS, UI, noise_rms=0.05)
    want = 0.5 * (serdes.q_to_ber(0.35 / 0.05) + serdes.q_to_ber(0.65 / 0.05))
    assert e.min_ber() == pytest.approx(float(want), rel=0.05)
    # the same cursor as crosstalk (independent data) gives the same BER
    x = np.zeros_like(p)
    x[10 * NSPUI:11 * NSPUI] = 0.3
    p1 = p.copy()
    p1[11 * NSPUI:12 * NSPUI] = 0.0
    ex = serdes.stat_eye(p1, TS, UI, noise_rms=0.05, crosstalk=[x])
    assert ex.min_ber() == pytest.approx(e.min_ber(), rel=0.05)
    # a static offset of both levels toward 0 by 0.1 V = a 0.4 V eye
    eo = serdes.stat_eye(p1, TS, UI, noise_rms=0.05, level_offsets=[0.1, -0.1])
    assert eo.min_ber() == pytest.approx(float(serdes.q_to_ber(0.4 / 0.05)), rel=0.05)


def test_estimate_channel_recovers_a_known_channel():
    ch = serdes.backplane(0.4, ts=TS, ui=UI, nfft=1 << 14).resized(96 * NSPUI)
    a = serdes.symbols(6000)
    y = np.convolve(serdes.stimulus(a, NSPUI), ch.impulse)[: 6000 * NSPUI] * TS
    est, resid = serdes.estimate_channel(y + 0.2, TS, UI, a, span_ui=80)
    p, q = ch.pulse(UI), est.pulse(UI)
    n = min(len(p), len(q))
    assert np.max(np.abs(p[:n] - q[:n])) < 1e-3 * p.max()
    assert resid < 2e-3 * p.max()


def test_statistical_and_time_domain_flows_agree():
    ch = serdes.backplane(0.5, ts=TS, ui=UI, nfft=1 << 14).resized(128 * NSPUI)
    rx = ibisami.load("pf_rx")
    rx.params.update(ctle_fp1=12.5e9, dfe_mode=1, cdr_mode=0)  # frozen at Init
    st = serdes.statistical_link(None, ch, rx, UI, noise_density=np.sqrt(2 * TS))
    gain = st.noise_rms                 # slicer noise per volt of input noise
    lo, hi = 1e-4, 10.0                 # input noise for a statistical BER of 3e-3
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        b = serdes.stat_eye(st.pulse, TS, UI, noise_rms=gain * mid).bit_error_rate()
        lo, hi = (mid, hi) if b < 3e-3 else (lo, mid)
    td = serdes.time_domain_link(None, ch, rx, UI, nsym=40000, noise_rms=lo, skip=500)
    assert td.counted["bit_errors"] > 30
    assert td.counted["ber"] / 3e-3 == pytest.approx(1.0, abs=0.5)


# --- web app glue -------------------------------------------------------------

def test_webapp_overrides_and_catalog():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
    import amilink
    import catalog

    assert amilink.parse_overrides("a=1 b=2.5e9, Modulation=PAM4") == \
        {"a": 1, "b": 2.5e9, "Modulation": "PAM4"}
    with pytest.raises(ValueError):
        amilink.parse_overrides("oops")
    assert catalog.CATALOG["ami_tx"]["wave"] == "prbs"
    assert catalog.CATALOG["ami_rx"]["ami"] == "rx"
    assert "pf_rx" in catalog.CATALOG["ami_rx"]["params"][0]["choices"]
