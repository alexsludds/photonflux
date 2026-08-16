"""TDEC / OMA-TDEC bridge to stateye (photonflux.tdec).

The reference-receiver tests need only scipy; the measurement tests skip when
stateye is not installed (it is the optional ``eye`` extra).
"""
import numpy as np
import pytest

from photonflux import tdec
from photonflux.signals import prbs

BAUD = 53.125e9
SPS = 32
DT = 1.0 / BAUD / SPS


def _nrz(bits, one=0.40, zero=0.09):
    return np.repeat(np.where(np.asarray(bits) > 0, one, zero), SPS)


# ---------------------------------------------------------------------------
# reference receiver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bw_factor", [0.5, 0.75, 1.0])
def test_reference_receiver_lands_on_the_requested_3db_point(bw_factor):
    """norm="mag" is what makes Wn the -3 dB frequency; scipy defaults to
    norm="phase", which would put it somewhere else entirely."""
    from scipy import signal

    fc = bw_factor * BAUD
    sos = signal.bessel(4, fc, "low", analog=False, output="sos",
                        fs=1.0 / DT, norm="mag")
    w, h = signal.sosfreqz(sos, worN=100000, fs=1.0 / DT)
    db = 20 * np.log10(np.abs(h) + 1e-300)
    f3 = np.interp(-3.0, db[::-1], w[::-1])
    assert f3 == pytest.approx(fc, rel=0.01)


def test_reference_receiver_attenuates_and_preserves_dc():
    rng = np.random.default_rng(0)
    p = _nrz(rng.integers(0, 2, 4000))
    tight = tdec.reference_receiver(p, DT, BAUD, bw_factor=0.25)
    wide = tdec.reference_receiver(p, DT, BAUD, bw_factor=1.0)
    # a tighter receiver closes the eye: less spread about the mean
    assert np.std(tight[2000:]) < np.std(wide[2000:])
    # DC gain is unity, so the settled mean survives
    assert np.mean(tight[2000:]) == pytest.approx(np.mean(p[2000:]), abs=5e-3)


def test_reference_receiver_bypass_and_override():
    p = _nrz(prbs(7))
    assert np.array_equal(tdec.reference_receiver(p, DT, BAUD, bw_factor=None), p)
    a = tdec.reference_receiver(p, DT, BAUD, bw_hz=0.5 * BAUD)
    b = tdec.reference_receiver(p, DT, BAUD, bw_factor=0.5)
    assert np.allclose(a, b)


def test_reference_receiver_rejects_bandwidth_above_nyquist():
    p = _nrz(prbs(7))
    with pytest.raises(ValueError, match="Nyquist"):
        tdec.reference_receiver(p, DT, BAUD, bw_factor=20.0)


def test_reference_receiver_is_causal():
    """sosfilt, not sosfiltfilt: a step must not move before it arrives."""
    p = np.concatenate([np.zeros(2000), np.ones(2000)])
    y = tdec.reference_receiver(p, DT, BAUD, bw_factor=0.3)
    assert np.max(np.abs(y[:1990])) < 1e-9


# ---------------------------------------------------------------------------
# oma - tdec
# ---------------------------------------------------------------------------
def test_oma_tdec_dbm_arithmetic():
    assert tdec.oma_tdec_dbm({"oma_8180": 1.0, "tdec_8180": 0.5}) == pytest.approx(-0.5)
    assert tdec.oma_tdec_dbm(
        {"oma_8180": 0.31, "tdec_8180": 0.8}) == pytest.approx(
            10 * np.log10(0.31) - 0.8)


def test_oma_tdec_dbm_is_nan_when_inputs_are_unusable():
    for m in ({"oma_8180": np.nan, "tdec_8180": 0.5},
              {"oma_8180": 0.3, "tdec_8180": np.nan},
              {"oma_8180": 0.0, "tdec_8180": 0.5}):
        assert np.isnan(tdec.oma_tdec_dbm(m))


def test_oma_tdec_cancels_oma_and_depends_only_on_addable_noise():
    """OMA - TDEC = 10*log10(2*Qinv(BER)*R): the OMA divides out.

    That is the whole point of the IEEE spec line, and it is why two designs
    with very different OMA and TDEC can score identically.
    """
    ber, q = 1e-12, tdec.q_inv(1e-12)
    for oma, r in ((0.31, 0.02), (0.05, 0.02), (0.90, 0.004)):
        t = 10 * np.log10((oma / 2) / (q * r))
        assert tdec.oma_tdec_dbm({"oma_8180": oma, "tdec_8180": t}) == pytest.approx(
            10 * np.log10(2 * q * r), abs=1e-9)
    assert tdec.q_inv(ber) == pytest.approx(7.0345, abs=1e-3)


def test_oma_tdec_floor_matches_the_observed_saturation():
    """stateye solves R = (1-M1)*sqrt(N^2 + S^2 - M2^2), so R >= S. As the eye
    closes, N -> 0 and the metric bottoms out at 10*log10(2*Qinv(BER)*S).

    -8.5174 dBm for S = 0.01 mW at 1e-12 is the plateau seen in the gap x lock
    sweep, where several unrelated designs returned the same value.
    """
    assert tdec.oma_tdec_floor_dbm(0.01, 1e-12) == pytest.approx(-8.5174, abs=1e-3)
    # halving the assumed receiver noise pushes the floor down by 3 dB
    assert tdec.oma_tdec_floor_dbm(0.005, 1e-12) == pytest.approx(
        tdec.oma_tdec_floor_dbm(0.01, 1e-12) - 10 * np.log10(2), abs=1e-6)
    # the floor carries a factor of Qinv(BER), so a tighter BER target raises
    # it: 1e-12 needs Q = 7.03 where 1e-9 needs only 6.00
    assert tdec.oma_tdec_floor_dbm(0.01, 1e-12) > tdec.oma_tdec_floor_dbm(0.01, 1e-9)
    assert tdec.oma_tdec_floor_dbm(0.01, 1e-12) - tdec.oma_tdec_floor_dbm(0.01, 1e-9) \
        == pytest.approx(10 * np.log10(tdec.q_inv(1e-12) / tdec.q_inv(1e-9)), abs=1e-9)


# ---------------------------------------------------------------------------
# stateye measurement
# ---------------------------------------------------------------------------
stateye = pytest.importorskip("stateye", reason="optional 'eye' extra")


def test_measure_prbs13_recovers_the_known_levels():
    m = tdec.measure(_nrz(prbs(13)), DT, BAUD, s_noise_mW=0.01)
    assert m["oma_8180"] == pytest.approx(0.31, abs=2e-3)
    assert m["one_level_8180"] == pytest.approx(0.40, abs=2e-3)
    assert m["zero_level_8180"] == pytest.approx(0.09, abs=2e-3)
    assert m["extinction_ratio_8180"] == pytest.approx(
        10 * np.log10(0.40 / 0.09), abs=0.1)
    assert np.isfinite(m["oma_tdec_dbm"])
    assert m.counts["oma_8180"] >= tdec.MIN_SEGMENT_COUNT


def test_measure_rejects_a_pattern_without_8180_runs():
    """PRBS-7 tops out at a run of 7 ones, so the _8180 metrics are NaN.

    That is the failure this guard exists to make loud instead of silent.
    Tiled to clear MIN_RECORD_UI so this tests the *pattern* guard and not
    the record-length one — tiling preserves the max run of 7.
    """
    short_runs = np.tile(prbs(7), 4)          # 508 bits, still max run 7
    with pytest.raises(ValueError, match="8180 segments"):
        tdec.measure(_nrz(short_runs), DT, BAUD, s_noise_mW=0.01)


def test_measure_rejects_a_record_too_short_for_stateye():
    """Below ~200 UI stateye's histogram reductions run on empty arrays and
    fail with an opaque numpy error; say what is actually wrong instead."""
    with pytest.raises(ValueError, match="UI of record|too short|Raise the transient"):
        tdec.measure(_nrz(prbs(7)), DT, BAUD, s_noise_mW=0.01, strict=False)


def test_tighter_reference_receiver_raises_tdec():
    """TDEC measures eye closure, so squeezing the reference receiver must
    monotonically worsen it."""
    p = _nrz(prbs(13))
    vals = [tdec.measure(p, DT, BAUD, s_noise_mW=0.01,
                         ref_rx_bw_factor=f)["tdec_8180"]
            for f in (1.0, 0.75, 0.5, 0.35)]
    assert all(np.isfinite(v) for v in vals)
    assert np.all(np.diff(vals) > 0), vals


def test_prbs9_4140_tracks_prbs13_8180():
    """The search surrogate: PRBS-9 + tdec_4140 must follow PRBS-13 + tdec_8180.

    This is what buys ~30 h in the optimizer. The residual is a small, nearly
    constant *bias* (~+0.01 dB across a 1.7 dB span of TDEC), not scatter --
    which is why it is safe for ranking even though it is not zero. Assert both
    the magnitude and the low spread; scatter is what would break the ranking.
    """
    errs = []
    for bw in (1.0, 0.75, 0.6, 0.5, 0.4):
        ref = tdec.measure(_nrz(prbs(13)), DT, BAUD, s_noise_mW=0.01,
                           ref_rx_bw_factor=bw)["tdec_8180"]
        sur = tdec.measure(_nrz(prbs(9)), DT, BAUD, s_noise_mW=0.01,
                           ref_rx_bw_factor=bw, oma_type="4140",
                           strict=False)["tdec_4140"]
        errs.append(sur - ref)
    errs = np.asarray(errs)
    assert np.all(np.abs(errs) < 0.02), errs
    assert np.ptp(errs) < 0.01, f"surrogate error is scattered, not a bias: {errs}"


def test_measure_rejects_a_2d_waveform():
    with pytest.raises(ValueError, match="1-D"):
        tdec.measure(np.zeros((4, 100)), DT, BAUD, s_noise_mW=0.01)


def test_measure_keeps_objects_out_of_the_metric_dict():
    """Everything in `.metrics` must be a plain number, so callers can dump it
    to CSV/JSON without tripping over the eye object or the counts dict."""
    m = tdec.measure(_nrz(prbs(13)), DT, BAUD, s_noise_mW=0.01)
    for k, v in m.metrics.items():
        assert isinstance(v, (float, int, bool, list)), f"{k} is {type(v)}"
    assert m.eye is not None and isinstance(m.counts, dict)
    assert m["oma_8180"] == m.metrics["oma_8180"]   # subscript delegates


def test_measure_flags_saturation_when_the_eye_is_closed():
    """A flat waveform has no eye at all, so R collapses to S and the metric
    must report itself as sitting on the floor rather than as a real score."""
    flat = _nrz(prbs(13), one=0.2001, zero=0.2000)
    m = tdec.measure(flat, DT, BAUD, s_noise_mW=0.01, strict=False)
    assert m["at_floor"] is True
    assert m["oma_tdec_dbm"] == pytest.approx(m["oma_tdec_floor_dbm"], abs=0.02)

    wide = tdec.measure(_nrz(prbs(13)), DT, BAUD, s_noise_mW=0.01,
                        ref_rx_bw_factor=1.0)
    assert wide["at_floor"] is False
