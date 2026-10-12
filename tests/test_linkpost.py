"""Link-report statistics (``webapp/linkpost.py``) against direct counting.

* Q-fit BER: the per-eye Gaussian estimate must agree with counted bit errors
  for NRZ and Gray PAM-4 (each PAM-4 eye is crossed by two of four levels and
  a slip costs one of two bits, so BER = sum over eyes / 4 -- not the SER).
* Pulse FOM: half the eye over the RMS residual ISI of random equiprobable
  data, checked against a brute-force convolution of random symbols.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

import linkpost  # noqa: E402


@pytest.mark.parametrize("nlv, sigma", [(2, 0.2), (4, 0.075)])
def test_qfit_ber_matches_counted_errors(nlv, sigma):
    rng = np.random.default_rng(0)
    tx = rng.integers(0, nlv, 400_000) / (nlv - 1)
    eq = tx + rng.normal(0.0, sigma, tx.size)
    est = linkpost._qfit(eq, tx, nlv)["ber_est"]
    counted = linkpost._count_errors(eq, tx, nlv)["ber"]
    assert est == pytest.approx(counted, rel=0.05)


@pytest.mark.parametrize("nlv", [2, 4])
def test_pulse_fom_matches_brute_force_isi(nlv):
    ui, osr = 1e-10, 32
    post = 0.25                          # one post-cursor, no equalization
    sym = np.zeros(64)
    sym[6], sym[7] = 1.0, post
    t = np.arange(sym.size * osr) * ui / osr
    rep = linkpost.pulse_report(
        {"x": t.tolist(), "traces": [{"name": "o",
                                      "values": np.repeat(sym, osr).tolist()}]},
        {"patterns": {"P": {"ui": ui}}},
        {"probe": "o", "ffe_taps": 0, "dfe_taps": 0, "levels": nlv}, [])

    rng = np.random.default_rng(1)
    a = rng.integers(0, nlv, 200_000) / (nlv - 1)
    isi = post * a[:-1]                  # what the post-cursor adds
    want = 20 * np.log10((1.0 / (2 * (nlv - 1))) / np.std(isi))
    assert rep["fom_db"] == pytest.approx(want, abs=0.05)


@pytest.mark.parametrize("sequence", ["square", "staircase"])
def test_link_report_skips_level_patterns(sequence):
    """A PAM-4 square wave never sends levels 1 and 2, so counted BER and
    TDECQ would be meaningless: the report says so instead."""
    log: list = []
    meta = {"patterns": {"PAT1": {"mode": "pam4", "sequence": sequence}}}
    assert linkpost.link_report({"traces": [], "x": []}, meta,
                                {"probe": "out"}, log) is None
    assert "level pattern" in log[0]


def test_link_report_samples_at_the_eye_centre():
    """The report's decision phase is the middle of the eye (the bathtub
    floor), and the bathtub is centred on it with walls on both sides. It
    used to sample wherever the correlation peaked (here an eye edge), plot
    the best phase at -0.5 UI, and compare half its phases with the
    neighbouring symbol (a flat BER ~ 0.5 shelf)."""
    import wavesrc

    ui, osr, nsym = 1e-10, 16, 1200
    pattern = {"mode": "nrz", "order": 9, "seed": 1, "ui": ui}
    tx = wavesrc._symbols(pattern, nsym + 16)[:nsym]
    # 16 samples per UI: a 6-sample linear edge, then the symbol holds for
    # 11 samples (6..15 and the next slot's first, still on the old value);
    # the whole record is delayed by 3 samples
    frac = (np.arange(nsym * osr) % osr) / osr
    prev = np.repeat(np.roll(tx, 1), osr)
    cur = np.repeat(tx, osr)
    ramp = np.clip(frac / (6 / osr), 0.0, 1.0)
    wave = np.roll(prev + (cur - prev) * ramp, 3)
    wave += np.random.default_rng(5).normal(0.0, 0.02, wave.size)
    t = np.arange(wave.size) * ui / osr
    rep = linkpost.link_report(
        {"x": t.tolist(), "traces": [{"name": "o", "unit": "V",
                                      "values": wave.tolist()}]},
        {"patterns": {"P": pattern}}, {"probe": "o"}, [])
    # flat top 3+6 = 9 .. 3+16 = 19 -> centre at phase 14
    assert abs(rep["sampling_phase_ui"] * osr - 14.0) <= 0.5
    assert rep["counted"]["bit_errors"] == 0
    ph = np.asarray(rep["bathtub"]["phase_ui"])
    lb = np.asarray(rep["bathtub"]["log10_ber"])
    assert ph.min() >= -0.5 and ph.max() < 0.5
    assert lb[np.argmin(np.abs(ph))] <= lb.min() + 1.0   # 0 UI on the floor
    # walls on both sides, and no BER ~ 0.5 shelf on a clean eye
    assert lb[0] > lb.min() + 3 and lb[-1] > lb.min() + 3
    assert np.sum(lb > -0.5) <= 2
