"""TDEC / OMA-TDEC measurement: photonflux waveform -> stateye eye diagram.

The bridge between a circulax transient and `stateye
<https://github.com/DerekK44/stateye>`_ (v1.8, the fork of AyarLabs/stateye
that adds PAM-4 and TDECQ). photonflux produces an optical through-port power
waveform; stateye draws the eye histogram and extracts the IEEE 802.3
transmitter metrics (OMA, TDEC, extinction ratio, DCD; TDECQ for PAM-4 via
:func:`measure_pam4`).

stateye is an **optional** dependency -- importing this module without it
raises a directed ImportError rather than failing at photonflux import time.

    from photonflux import tdec

    m = tdec.measure(p_thru_mW, dt_sec, baud=53.125e9, s_noise_mW=0.01)
    print(m["oma_tdec_dbm"], m["tdec_8180"], m.counts["oma_8180"])

Always check ``m["at_floor"]``. OMA - TDEC reduces to ``10*log10(2*Qinv(BER)*R)``
-- the OMA cancels -- and R cannot fall below the assumed receiver noise S, so
the metric saturates once the eye closes and every design there scores alike.
See :func:`oma_tdec_dbm` and :func:`oma_tdec_floor_dbm`.

Pattern requirements (measured; see ``docs/stateye-integration-plan.md``):

* ``oma_8180`` / ``tdec_8180`` need runs of >=8 ones and >=8 zeros, found
  **independently**. A full-period PRBS-13 supplies 16 of each.
* ``dcd_8180`` / ``rise_time_*_8180`` need a *contiguous* ``0^7.v.1^8``
  window, which PRBS-13 never contains -- they come back NaN, so the edge and
  duty-cycle diagnostics must be read from the ``_4140`` variants.
* Do not truncate PRBS-13: its first 511 bits contain no run of 8 ones, so
  ``oma_8180`` would be NaN.
* PRBS-9 + ``tdec_4140`` tracks PRBS-13 + ``tdec_8180`` to within +0.009 to
  +0.016 dB across a 1.7 dB span of TDEC -- a near-constant *bias* rather than
  scatter, so it cannot reorder candidates. That is what makes it a viable
  16x-cheaper search surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Measurement", "reference_receiver", "measure", "measure_pam4", "oma_tdec_dbm",
           "oma_tdec_floor_dbm", "q_inv", "MIN_SEGMENT_COUNT", "FLOOR_TOL_DB"]

# How close to the analytic floor counts as saturated. The histogram-derived
# N never reaches exactly zero, so the measured floor sits a few thousandths
# of a dB above the analytic one.
FLOOR_TOL_DB = 0.02

# A full PRBS-13 period yields exactly 16 runs of >=8 ones and >=8 zeros;
# stateye reports 15 usable _8180 segments (and 255 _4140). Anything materially
# below this means the pattern is wrong, not that the link is bad.
MIN_SEGMENT_COUNT = 8

# Shortest usable record, in unit intervals, after settling is discarded.
# Measured: ~150 UI of this link fails inside stateye's histogram reductions;
# ~200 UI is fine. 192 leaves margin without demanding a long transient.
MIN_RECORD_UI = 192


@dataclass
class Measurement:
    """stateye's output for one waveform.

    The numeric metrics live in :attr:`metrics` and are reachable directly by
    subscript (``m["tdec_8180"]``); ``counts`` and ``eye`` are kept out of that
    dict so callers can serialise the metrics without tripping over a dict and
    a live matplotlib-backed object.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    eye: Any = None
    oma_type: str = "8180"   # which level-estimation family the metrics came from
    # the waveform stateye scored (after the reference receiver, settling trim
    # and any equalizer) and its sample period -- for plotting that eye
    waveform: Any = None
    dt_sec: float = 0.0

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def __contains__(self, key: str) -> bool:
        return key in self.metrics

    def get(self, key: str, default=None):
        return self.metrics.get(key, default)


def _require_stateye():
    try:
        import stateye
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "photonflux.tdec needs stateye, which is an optional dependency.\n"
            "  pip install 'stateye @ git+https://github.com/DerekK44/stateye'\n"
            "That build fails on NumPy 2 (cnp.int_t / missing pyproject.toml) "
            "until docs/patches/stateye-modern-toolchain.patch is applied; see "
            "docs/stateye-integration-plan.md section 2."
        ) from exc
    return stateye


def reference_receiver(
    p,
    dt_sec: float,
    baud: float,
    *,
    bw_factor: float | None = 0.75,
    order: int = 4,
    bw_hz: float | None = None,
):
    """IEEE-style Bessel-Thomson reference receiver.

    TDEC is defined *through* a reference receiver; without one the metric
    rewards ringing and overshoot that a compliant receiver would filter away.

    bw_factor : -3 dB bandwidth as a multiple of the baud rate. 0.75 is the
        clause-95-family convention (39.84 GHz at 53.125 GBd). None bypasses
        the filter entirely.
    order : Bessel order; 4 in the standards.
    bw_hz : absolute -3 dB bandwidth [Hz]; overrides `bw_factor` when given.

    Returns the filtered waveform. Three deliberate choices:

    * ``norm="mag"`` -- scipy's default is ``norm="phase"``, which normalises
      the phase response and puts the -3 dB point somewhere else. Only "mag"
      makes ``Wn`` the -3 dB frequency (verified to -0.16% over 0.5-1.0x baud).
    * ``sosfilt``, not ``sosfiltfilt`` -- a reference receiver is causal, and
      zero-phase filtering would remove exactly the phase distortion TDEC
      exists to measure.
    * digital rather than analog+FFT -- at 32-40 samples/UI the cutoff sits
      below 4% of Nyquist so bilinear warping is negligible, and ``sos``
      avoids the circular-wrap artifact of an FFT multiply.
    """
    from scipy import signal

    if bw_hz is None and bw_factor is None:
        return np.asarray(p, dtype=float)
    fc = float(bw_hz) if bw_hz is not None else float(bw_factor) * baud
    nyquist = 0.5 / dt_sec
    if fc >= nyquist:
        raise ValueError(
            f"reference-receiver bandwidth {fc/1e9:.2f} GHz is at or above the "
            f"{nyquist/1e9:.2f} GHz Nyquist of the dt={dt_sec:.3e} s grid; "
            "increase samples per UI")
    sos = signal.bessel(order, fc, "low", analog=False, output="sos",
                        fs=1.0 / dt_sec, norm="mag")
    return signal.sosfilt(sos, np.asarray(p, dtype=float))


def q_inv(ber: float) -> float:
    """The Q corresponding to a target BER (Q = sqrt(2)*erfcinv(2*BER))."""
    from scipy.special import erfcinv

    return float(np.sqrt(2.0) * erfcinv(2.0 * ber))


def oma_tdec_floor_dbm(s_noise_mW: float, ber: float = 1e-12,
                       m1: float = 0.0, m2: float = 0.0) -> float:
    """The value OMA - TDEC saturates to once the eye is fully closed.

    stateye solves the addable receiver noise as
    ``R = (1-M1)*sqrt(N^2 + S^2 - M2^2)`` (histogram_analysis.py:214), where N
    comes from the eye histogram. As the eye closes N -> 0 and R -> S, so the
    metric bottoms out at ``10*log10(2*Qinv(BER)*R)`` with R = S.

    Any design at this floor is indistinguishable from any other design at it.
    """
    r = (1.0 - m1) * float(np.sqrt(max(s_noise_mW**2 - m2**2, 0.0)))
    if r <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(2.0 * q_inv(ber) * r))


def oma_tdec_dbm(msmts: dict, oma_type: str = "8180") -> float:
    """OMA - TDEC in dBm, the transmitter budget line item.

    OMA arrives in mW (stateye's default power unit) and TDEC is already in
    dB, so this is ``10*log10(OMA_mW) - TDEC_dB``.

    Maximise this rather than minimising TDEC alone: TDEC on its own has a
    degenerate optimum where the laser is parked far off resonance for a tiny
    but spotless swing.

    **The OMA cancels, and that is the point.** stateye computes
    ``TDEC = 10*log10((OMA/2) / (Qinv(BER)*R))``, so

        OMA_dBm - TDEC = 10*log10(2 * Qinv(BER) * R)

    depends only on R, the noise a receiver could still add and hit the target
    BER. That is exactly what the IEEE "OMA minus TDEC" spec line is for: a
    single number for transmitter quality in units of tolerable receiver
    noise. Two designs with very different OMA and TDEC can therefore score
    identically -- they are equally good, not accidentally equal.

    The corollary is that the metric **saturates**: see
    :func:`oma_tdec_floor_dbm`.
    """
    oma = msmts[f"oma_{oma_type}"]
    tdec = msmts[f"tdec_{oma_type}"]
    if not np.isfinite(oma) or not np.isfinite(tdec) or oma <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(oma) - tdec)


def _prepare(p_thru_mW, dt_sec, baud, ref_rx_bw_factor, ref_rx_order,
             ref_rx_bw_hz, settle_ui):
    """Reference-receiver filter, integer-samples/UI resampling, settling
    trim and record-length guard shared by the NRZ and PAM-4 measurements.
    Returns ``(p, dt_sec)`` on the resampled grid.

    stateye folds the record assuming a whole number of samples per UI: at
    24.93 samples/UI a clean PAM-4 eye scored 27.9 dB TDECQ instead of 4.2,
    and 24.5 crashed its bathtub fit. Any fractional grid -- a transient
    whose t_stop/points do not divide the UI -- is therefore interpolated
    onto the nearest integer rate (at least 16 samples/UI) first.
    """
    p = np.asarray(p_thru_mW, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"expected a 1-D power waveform, got shape {p.shape}")
    sps = 1.0 / (baud * dt_sec)
    sps_int = max(16, int(round(sps)))
    if abs(sps - sps_int) > 1e-9 * sps_int:
        t = np.arange(p.size) * dt_sec
        dt_sec = 1.0 / (baud * sps_int)
        p = np.interp(np.arange(0.0, t[-1], dt_sec), t, p)

    p = reference_receiver(p, dt_sec, baud, bw_factor=ref_rx_bw_factor,
                           order=ref_rx_order, bw_hz=ref_rx_bw_hz)

    sps = 1.0 / (baud * dt_sec)
    skip = int(round(settle_ui * sps))
    if skip >= p.size:
        raise ValueError(
            f"settle_ui={settle_ui} discards the whole {p.size}-sample record")
    p = p[skip:]

    # stateye needs enough symbols to lock a CDR and populate the level
    # filters; below that its histogram reductions run on empty arrays and
    # fail deep inside with an opaque "zero-size array to reduction
    # operation" instead of telling you the record is too short.
    n_ui = p.size / sps
    if n_ui < MIN_RECORD_UI:
        raise ValueError(
            f"record is only {n_ui:.0f} UI after discarding {settle_ui} UI of "
            f"settling; need >={MIN_RECORD_UI} for stateye to lock and fill "
            "the level filters. Raise the transient's t_stop.")
    return p, dt_sec


def measure(
    p_thru_mW,
    dt_sec: float,
    baud: float,
    *,
    s_noise_mW: float,
    ber: float = 1e-12,
    m1: float = 0.0,
    m2: float = 0.0,
    ref_rx_bw_factor: float | None = 0.75,
    ref_rx_order: int = 4,
    ref_rx_bw_hz: float | None = None,
    settle_ui: int = 8,
    nx: int = 512,
    ny: int = 2048,
    oma_type: str = "8180",
    strict: bool = True,
) -> Measurement:
    """Optical through-port power [mW] on a uniform grid -> stateye metrics.

    `settle_ui` unit intervals are dropped from the head of the record to skip
    both the circuit's turn-on and the reference receiver's filter transient.

    `oma_type` selects the level-estimation family (``"8180"`` needs a
    full-period PRBS-13; ``"4140"`` works on shorter patterns).

    With `strict`, raises when the selected family has too few usable segments
    -- which is what a wrong test pattern looks like (PRBS-7, or a truncated
    PRBS-13), and is otherwise a silent NaN.
    """
    stateye = _require_stateye()
    p, dt_sec = _prepare(p_thru_mW, dt_sec, baud, ref_rx_bw_factor,
                         ref_rx_order, ref_rx_bw_hz, settle_ui)

    # half_ui is mandatory for TDEC: its 0.4/0.6 UI histogram windows are
    # referenced to the eye crossing. (stateye's README says "adaptive" is the
    # default; ideal_eye.py actually defaults to half_ui -- set it explicitly.)
    eye = stateye.IdealEye(
        datarate_gbps=baud / 1e9,
        dt_sec=dt_sec,
        nx=nx,
        ny=ny,
        sampling_offset_mode="half_ui",
    )
    eye.set_tdec_s_noise(s_noise_mW)
    eye.set_tdec_m1(m1)
    eye.set_tdec_m2(m2)
    eye.set_tdec_ber(ber)
    eye.add_data(p, "mW")

    msmts = dict(eye.get_measurements())
    counts = dict(eye.get_measurement_counts())

    if strict:
        n = counts.get(f"oma_{oma_type}", 0)
        if n < MIN_SEGMENT_COUNT:
            runs = 8 if oma_type == "8180" else 4
            raise ValueError(
                f"only {n} usable {oma_type} segments (need "
                f">={MIN_SEGMENT_COUNT}); the _{oma_type} metrics need runs of "
                f">={runs} ones and >={runs} zeros. Drive this with a "
                "full-period PRBS-13 -- PRBS-7 can never produce the _8180 "
                "runs and a truncated PRBS-13 loses them.")

    msmts["oma_tdec_dbm"] = oma_tdec_dbm(msmts, oma_type)
    # The metric saturates once the eye closes (see oma_tdec_floor_dbm): every
    # design at the floor scores the same, so an optimizer sees a plateau with
    # no gradient. Flag it rather than let the search wander on it.
    floor = oma_tdec_floor_dbm(s_noise_mW, ber, m1, m2)
    msmts["oma_tdec_floor_dbm"] = floor
    msmts["at_floor"] = bool(
        np.isfinite(msmts["oma_tdec_dbm"])
        and msmts["oma_tdec_dbm"] <= floor + FLOOR_TOL_DB)
    return Measurement(metrics=msmts, counts=counts, eye=eye, oma_type=oma_type,
                       waveform=p, dt_sec=dt_sec)


def measure_pam4(
    p_thru_mW,
    dt_sec: float,
    baud: float,
    *,
    s_noise_mW: float = 0.0,
    ser: float = 4.8e-4,
    ffe_taps: int = 5,
    ffe_pre: int = 1,
    symbols=None,
    ffe_method: str = "mmse",
    ffe_mu: float = 0.05,
    ffe_passes: int = 5,
    ffe_manual=None,
    ffe_max_evals: int = 60,
    ref_rx_bw_factor: float | None = 0.5,
    ref_rx_order: int = 4,
    ref_rx_bw_hz: float | None = None,
    settle_ui: int = 8,
    nx: int = 512,
    ny: int = 2048,
    strict: bool = True,
) -> Measurement:
    """PAM-4 optical power [mW] -> stateye metrics, including TDECQ.

    Needs stateye >= 1.8 (the PAM-4 analysis landed there). Drive it with a
    Gray-coded PRBS-13Q, **rotated**::

        sym = np.roll(signals.pam4_gray(np.tile(prbs(13), 2)), 64)

    OMA_outer (and so ``tdecq_outer``) is measured on a run of 7 threes and a
    run of 6 zeros. PRBS-13Q has exactly one of each, and the zeros run is the
    last 6 symbols of the period -- unrotated it falls off the end of the
    record and every ``_outer`` metric comes back NaN.

    `ser` is the target symbol error rate: 4.8e-4 is the 802.3-2022 value for
    100G/lane (53.125 GBd) links, 4.56e-4 the 802.3dj 200G/lane one. The
    reference receiver defaults to the TDECQ 4th-order Bessel-Thomson at half
    the baud rate.

    Equalization: TDECQ is defined through the 802.3 reference equalizer, a
    T-spaced FFE after the reference receiver. ``ffe_taps`` sets its length
    (5, one pre-cursor via ``ffe_pre``, is the 802.3bs/cd reference; 0
    scores the eye unequalized). Taps are designed by MMSE with
    ``stateye.ffe_mmse`` (sum to 1) and their noise enhancement
    ``C_eq = sqrt(sum c^2)`` feeds TDECQ. Pass the transmitted PAM-4
    ``symbols`` (any rotation) whenever they are known -- as a TDECQ scope
    knows its test pattern; without them the design is decision-directed
    and only reliable while the unequalized eye is still open. The metrics
    gain ``ffe_taps`` (list), ``ceq`` and ``ffe_rms_error_before/after``.

    ``ffe_method`` picks how the taps adapt: ``"mmse"`` (block least
    squares, the default), ``"lms"`` (normalized LMS, step ``ffe_mu``,
    ``ffe_passes`` sweeps of the record), ``"optimal"`` (Nelder-Mead on
    TDECQ itself from the MMSE start, ``ffe_max_evals`` eye analyses -- the
    802.3 definition, and slow) or ``"manual"`` (``ffe_manual``, a list of
    taps, normalized to sum to 1; ``ffe_taps`` is then its length).
    MMSE taps are not the TDECQ-minimizing taps the standard asks for, so
    the result is at or slightly above the compliant value.
    """
    stateye = _require_stateye()
    if not hasattr(stateye.IdealEye, "set_tdecq_ser"):
        raise ImportError("measure_pam4 needs stateye >= 1.8; see "
                          "docs/stateye-integration-plan.md for the install.")
    p, dt_sec = _prepare(p_thru_mW, dt_sec, baud, ref_rx_bw_factor,
                         ref_rx_order, ref_rx_bw_hz, settle_ui)

    eq: dict = {}
    if ffe_taps:
        if not hasattr(stateye, "ffe_mmse"):
            raise ImportError(
                "ffe_taps needs the stateye reference equalizer "
                "(stateye.ffe_mmse); install stateye with "
                "docs/patches/stateye-modern-toolchain.patch applied, or pass "
                "ffe_taps=0.")
        sps = 1.0 / (baud * dt_sec)
        n, pre = int(ffe_taps), int(ffe_pre)
        if ffe_method == "mmse":
            taps, info = stateye.ffe_mmse(p, sps, n_taps=n, n_pre=pre,
                                          symbols=symbols)
        elif ffe_method == "lms":
            taps, info = stateye.ffe_lms(p, sps, n_taps=n, n_pre=pre,
                                         symbols=symbols, mu=float(ffe_mu),
                                         passes=int(ffe_passes))
        elif ffe_method == "optimal":
            taps, info = stateye.ffe_tdecq_optimal(
                p, dt_sec, baud / 1e9, n_taps=n, n_pre=pre, symbols=symbols,
                max_evals=int(ffe_max_evals), tdecq_s_noise=s_noise_mW,
                tdecq_ser=ser, nx=nx, ny=ny)
        elif ffe_method == "manual":
            taps = np.asarray(ffe_manual, dtype=float)
            if taps.ndim != 1 or taps.size < 1 or not pre < taps.size \
                    or taps.sum() == 0:
                raise ValueError("manual FFE taps need a non-empty list with "
                                 "more taps than pre-cursors and a nonzero sum")
            taps = taps / taps.sum()
            info = {"n_pre": pre, "ceq": stateye.noise_enhancement(taps)}
        else:
            raise ValueError(f"unknown ffe_method {ffe_method!r}; use mmse, "
                             "lms, optimal or manual")
        p = stateye.apply_ffe(p, taps, sps, n_pre=info["n_pre"])
        eq = {"ffe_taps": [float(c) for c in taps], "ceq": info["ceq"],
              "ffe_method": ffe_method}
        for key in ("rms_error_before", "rms_error_after"):
            if key in info:
                eq[f"ffe_{key}"] = info[key]
        if "evals" in info:
            eq["ffe_evals"] = info["evals"]

    eye = stateye.IdealEye(
        datarate_gbps=baud / 1e9,
        dt_sec=dt_sec,
        nx=nx,
        ny=ny,
        format="PAM4",
        sampling_offset_mode="half_ui",   # TDECQ's 0.45/0.55 UI windows need it
    )
    eye.set_tdecq_s_noise(s_noise_mW)
    eye.set_tdecq_ceq(eq.get("ceq", 1.0))
    eye.set_tdecq_ser(ser)
    eye.add_data(p, "mW")

    msmts = {k: (float(v) if isinstance(v, np.generic) else v)
             for k, v in eye.get_measurements().items()}
    counts = dict(eye.get_measurement_counts())
    msmts.update(eq)
    if strict and not np.isfinite(msmts.get("tdecq_outer", np.nan)):
        raise ValueError(
            "stateye returned a non-finite TDECQ: OMA_outer needs a run of 7 "
            "threes and a run of 6 zeros away from the record edges. Drive it "
            "with a full PRBS-13Q rotated by a few dozen symbols (its only run "
            "of 6 zeros is the last 6 symbols of the period).")
    return Measurement(metrics=msmts, counts=counts, eye=eye, oma_type="outer",
                       waveform=p, dt_sec=dt_sec)
