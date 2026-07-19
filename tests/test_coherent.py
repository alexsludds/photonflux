"""Coherent transceiver DSP (``webapp/coherent.py`` + QAM machinery in
``webapp/wavesrc.py``), pinned against textbook analytics — the ALE-77
acceptance criteria.

Pure-numpy checks of the DSP itself (no circulax), mirroring ``test_ssfm.py``:
the in-transient coherent components (IQ modulator, 90-degree hybrid, balanced
coherent receiver) that feed this DSP are exercised for build/steady-state by
``test_photonics``-style smoke tests. Here we validate the receiver chain that
the acceptance criteria are written against:

  * back-to-back QPSK / 16-QAM: BER matches theory vs SNR, EVM = 1/sqrt(SNR);
  * QPSK over 80 km of chromatic dispersion + laser phase noise: an open
    constellation is recovered (closed) by CD compensation + carrier recovery;
  * 16-QAM: per-cluster EVM is reported by ``coherent_report``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import coherent as co  # noqa: E402
import wavesrc as w  # noqa: E402


# ===========================================================================
# QAM constellation + symbol machinery
# ===========================================================================

@pytest.mark.parametrize("mode,M", [("qpsk", 4), ("qam16", 16), ("qam64", 64)])
def test_qam_constellation_unit_power_and_gray(mode, M):
    const = w.qam_constellation(M)
    assert len(const) == M
    # unit average power
    assert np.mean(np.abs(const) ** 2) == pytest.approx(1.0, rel=1e-9)
    # Gray coded: nearest neighbours differ by exactly one bit
    d = np.abs(const[:, None] - const[None, :])
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    for i in range(M):
        for j in range(M):
            if i != j and abs(d[i, j] - dmin) < 1e-6:
                assert bin(i ^ j).count("1") == 1


@pytest.mark.parametrize("mode,M", [("qpsk", 4), ("qam16", 16), ("qam64", 64)])
def test_qam_symbols_cover_alphabet(mode, M):
    syms = w.qam_symbols({"qam": mode, "order": 15, "seed": 1}, 5000)
    const = w.qam_constellation(M)
    # every symbol lands exactly on a constellation point
    idx = co.decide(syms, const)
    assert np.allclose(syms, const[idx])
    # PRBS exercises the whole alphabet
    assert len(np.unique(idx)) == M


def test_rrc_taps_unit_energy_and_symmetric():
    h = w.rrc_taps(0.2, 4, 12)
    assert np.sum(h ** 2) == pytest.approx(1.0, rel=1e-9)
    assert np.allclose(h, h[::-1], atol=1e-12)          # even symmetry
    assert np.argmax(h) == len(h) // 2


def test_rrc_cascade_is_nyquist():
    # RRC * RRC = raised cosine -> zero ISI at nonzero symbol offsets
    sps, span = 8, 16
    h = w.rrc_taps(0.25, sps, span)
    rc = np.convolve(h, h)
    c0 = (len(rc) - 1) // 2
    for k in range(1, span):
        assert abs(rc[c0 + k * sps]) < 1e-2 * rc[c0]


# ===========================================================================
# Acceptance #1 — back-to-back QPSK / 16-QAM: BER vs theory, EVM vs SNR
# ===========================================================================

@pytest.mark.parametrize("mode,M,snr_db", [
    ("qpsk", 4, 8.0), ("qpsk", 4, 10.0),
    ("qam16", 16, 12.0), ("qam16", 16, 14.0),
])
def test_back_to_back_ber_matches_theory(mode, M, snr_db):
    rng = np.random.default_rng(11)
    const = w.qam_constellation(M)
    tx = w.qam_symbols({"qam": mode, "order": 15, "seed": 1}, 400_000)
    rx = co.awgn(tx, snr_db, rng=rng)
    counted = co.count_ber(rx, tx, const, M)
    theory = co.qam_theory_ber(snr_db, M)
    # counted BER within 25% of theory (finite-sample + Gray approximation)
    assert counted["ber"] == pytest.approx(theory, rel=0.25)


@pytest.mark.parametrize("snr_db", [10.0, 14.0, 18.0])
def test_back_to_back_evm_is_snr_limited(snr_db):
    rng = np.random.default_rng(12)
    tx = w.qam_symbols({"qam": "qpsk", "order": 15, "seed": 1}, 200_000)
    rx = co.awgn(tx, snr_db, rng=rng)
    evm = co.evm(rx, tx)
    expected = 1.0 / np.sqrt(10.0 ** (snr_db / 10.0))    # EVM = 1/sqrt(SNR)
    assert evm == pytest.approx(expected, rel=0.03)


def test_back_to_back_error_free_at_high_snr():
    rng = np.random.default_rng(13)
    const = w.qam_constellation(4)
    tx = w.qam_symbols({"qam": "qpsk", "order": 15, "seed": 1}, 100_000)
    rx = co.awgn(tx, 20.0, rng=rng)
    assert co.count_ber(rx, tx, const, 4)["bit_errors"] == 0


# ===========================================================================
# Acceptance #2 — QPSK over 80 km CD + laser phase noise: recover the eye
# ===========================================================================

def _qpsk_over_fiber(length_km, linewidth, snr_db, seed=2):
    rng = np.random.default_rng(seed)
    M, sps, beta, symrate, lam = 4, 2, 0.1, 32e9, 1550.0
    tx = w.qam_symbols({"qam": "qpsk", "order": 15, "seed": 1}, 60_000)
    sig = co.upsample_shape(tx, sps, beta)
    fs = symrate * sps
    b2L = co.beta2_of(17.0, lam) * length_km * 1e3
    sig = co.apply_cd(sig, fs, b2L)                      # chromatic dispersion
    sig = sig * np.exp(1j * co.phase_noise(len(sig), linewidth, symrate, sps,
                                           rng=rng))     # laser linewidth
    sig = co.awgn(sig, snr_db, sps=sps, rng=rng)
    base = {"sps": sps, "rrc_beta": beta, "symbol_rate": symrate, "cpr": "vv"}
    return tx, sig, base, lam


def _evm_after(sig, tx, cfg):
    dsp = co.receive(sig, w.qam_constellation(4), 4, cfg)
    rx, ref, _lag, _rot = co.sync(dsp["syms"][300:], tx[300:])
    scale = np.sqrt(np.mean(np.abs(ref) ** 2) / np.mean(np.abs(rx) ** 2))
    return co.evm(rx * scale, ref), co.count_ber(rx * scale, ref,
                                                 w.qam_constellation(4), 4)


def test_cd_compensation_recovers_qpsk():
    tx, sig, base, lam = _qpsk_over_fiber(80.0, 100e3, 18.0)
    # without CD compensation the constellation is fully closed (EVM ~ random)
    evm_open, _ = _evm_after(sig, tx, base)
    assert evm_open > 0.8
    # with CD compensation + carrier recovery it reopens to the noise floor
    cfg = {**base, "cd": {"D_ps": 17.0, "length_km": 80.0, "lam_nm": lam}}
    evm_cl, counted = _evm_after(sig, tx, cfg)
    assert evm_cl < 0.20                                 # closed constellation
    assert counted["ber"] < 1e-3


def test_carrier_phase_recovery_tracks_linewidth():
    # with a real laser linewidth, skipping CPR leaves a rotating cloud
    tx, sig, base, lam = _qpsk_over_fiber(80.0, 500e3, 20.0)
    cd = {"cd": {"D_ps": 17.0, "length_km": 80.0, "lam_nm": lam}}
    evm_nocpr, _ = _evm_after(sig, tx, {**base, **cd, "cpr": "none"})
    evm_cpr, _ = _evm_after(sig, tx, {**base, **cd, "cpr": "vv"})
    assert evm_cpr < 0.5 * evm_nocpr


def test_frequency_offset_estimation():
    rng = np.random.default_rng(7)
    symrate = 32e9
    tx = w.qam_symbols({"qam": "qpsk", "order": 15, "seed": 1}, 40_000)
    f_true = 150e6                                        # 150 MHz CFO
    n = np.arange(len(tx))
    rx = co.awgn(tx * np.exp(1j * 2 * np.pi * f_true / symrate * n), 20,
                 rng=rng)
    f_est = co.freq_offset_estimate(rx, symrate, 4)
    assert f_est == pytest.approx(f_true, abs=0.05 * symrate / len(tx) * 20
                                  + 2e6)
    corr = co.remove_freq_offset(rx, f_est, symrate)
    rec = co.viterbi_viterbi(corr)
    a, b, _l, _r = co.sync(rec[200:], tx[200:])
    sc = np.sqrt(np.mean(np.abs(b) ** 2) / np.mean(np.abs(a) ** 2))
    assert co.count_ber(a * sc, b, w.qam_constellation(4), 4)["ber"] < 1e-3


# ===========================================================================
# adaptive equalization (CMA) opens an ISI channel
# ===========================================================================

def test_cma_opens_isi_channel():
    rng = np.random.default_rng(9)
    tx = w.qam_symbols({"qam": "qpsk", "order": 15, "seed": 1}, 40_000)
    sps = 2
    up = np.zeros(len(tx) * sps, dtype=complex)
    up[::sps] = tx
    # a short multipath channel (ISI) at the fractional rate
    chan = np.array([0.2, 0.0, 1.0, 0.0, 0.35, 0.0, -0.15])
    rx = np.convolve(up, chan, mode="same")
    rx = co.awgn(rx, 22, sps=sps, rng=rng)
    syms, _w = co.cma_equalize(rx, ntaps=15, mu=2e-3, sps=sps)
    syms = co.viterbi_viterbi(syms[2000:])               # let CMA converge
    a, b, _l, _r = co.sync(syms, tx[2000:])
    sc = np.sqrt(np.mean(np.abs(b) ** 2) / np.mean(np.abs(a) ** 2))
    assert co.evm(a * sc, b) < 0.2


# ===========================================================================
# Acceptance #3 — coherent_report: constellation + per-cluster EVM
# ===========================================================================

def _iq_result(tx, rx, ui):
    t = np.arange(len(tx)) * ui
    return {"x": t.tolist(), "traces": [
        {"name": "I", "domain": "electrical", "values": rx.real.tolist()},
        {"name": "Q", "domain": "electrical", "values": rx.imag.tolist()}]}


def test_coherent_report_16qam_per_cluster_evm():
    rng = np.random.default_rng(5)
    M, symrate = 16, 25e9
    ui = 1.0 / symrate
    pat = {"mode": "qam", "qam": "qam16", "order": 15, "seed": 1, "ui": ui,
           "sps": 1, "rrc_beta": 0.0}
    tx = w.qam_symbols(pat, 20_000)
    rx = co.awgn(tx * np.exp(1j * 0.2), 22.0, rng=rng)
    result = _iq_result(tx, rx, ui)
    meta = {"patterns": {"src": pat}}
    log = []
    rep = co.coherent_report(result, meta, {"probe_i": "I", "probe_q": "Q"},
                             log)
    assert rep is not None
    assert rep["order"] == 16 and rep["name"] == "16-QAM"
    # EVM is noise-limited (~ -22 dB) and reported both ways
    assert rep["evm_db"] == pytest.approx(-22.0, abs=1.5)
    assert rep["counted"]["ber"] < 1e-3
    # per-cluster EVM: one entry per constellation point, all populated & sane
    assert len(rep["clusters"]) == 16
    assert all(c["count"] > 0 for c in rep["clusters"])
    assert all(0.0 < c["evm"] < 0.5 for c in rep["clusters"])
    # constellation payload present for the scatter plot
    assert len(rep["const_re"]) == 16 and len(rep["rx_re"]) > 100


def test_coherent_report_needs_qam_source():
    meta = {"patterns": {"p": {"mode": "nrz"}}}
    log = []
    assert co.coherent_report({"traces": []}, meta, {}, log) is None
    assert any("no QAM source" in m for m in log)


def test_coherent_report_complex_probe_path():
    rng = np.random.default_rng(6)
    symrate = 25e9
    ui = 1.0 / symrate
    pat = {"mode": "qam", "qam": "qpsk", "order": 15, "seed": 1, "ui": ui}
    tx = w.qam_symbols(pat, 20_000)
    rx = co.awgn(tx, 18.0, rng=rng)
    t = np.arange(len(tx)) * ui
    result = {"x": t.tolist(), "traces": [
        {"name": "field", "domain": "optical", "values": rx.tolist()}]}
    rep = co.coherent_report(result, {"patterns": {"src": pat}},
                             {"probe": "field"}, [])
    assert rep["name"] == "QPSK"
    assert rep["evm_db"] == pytest.approx(-18.0, abs=1.5)
