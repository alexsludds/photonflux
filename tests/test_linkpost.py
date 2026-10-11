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
