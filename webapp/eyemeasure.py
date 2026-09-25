"""Eye-tab measurements: any trace -> stateye -> TDECQ (PAM-4) or TDEC (NRZ).

The browser folds and draws eyes itself; this module is the scoring half,
behind ``POST /api/eyemeasure``. The payload carries the trace the Eye tab is
showing (``t``, ``values``, ``unit``), the fold UI, the modulation, the
settling time to drop, the pattern source's settings (for a known-pattern
equalizer) and the measurement settings the user picked. PAM-4 goes through
:func:`photonflux.tdec.measure_pam4` twice -- unequalized and through the
chosen reference FFE -- NRZ through :func:`photonflux.tdec.measure` (TDEC and
OMA - TDEC; 802.3 defines no equalizer for TDEC).

The scored waveform is returned decimated, so the Eye tab can draw the eye
stateye actually measured: after the reference receiver and the equalizer.
"""
from __future__ import annotations

import numpy as np

import wavesrc

_MAX_RETURN_POINTS = 20_000


def _num(v):
    """JSON-safe float: NaN/inf -> None (a bare NaN token breaks JSON.parse)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _scored(m, t0: float, t_in, v_in, ui: float) -> dict:
    """The measured waveform, time-aligned to the input trace and decimated
    to a browser-sized record.

    The reference receiver and the FFE delay the waveform by a fraction of a
    UI or more; the Eye tab folds on absolute time and reads its metrics at
    mid-UI, so the scored record is shifted back by the lag (within +-2 UI)
    that best correlates it with the input.
    """
    w = np.asarray(m.waveform, float)
    t = t0 + np.arange(w.size) * m.dt_sec
    ref = np.interp(t, t_in, v_in)
    max_lag = int(round(2 * ui / m.dt_sec))
    a, b = w - w.mean(), ref - ref.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    core = slice(max_lag, w.size - max_lag)
    corr = [np.dot(a[core], np.roll(b, lag)[core]) for lag in lags]
    t = t - lags[int(np.argmax(np.abs(corr)))] * m.dt_sec
    step = max(1, int(np.ceil(w.size / _MAX_RETURN_POINTS)))
    return {"t": t[::step].tolist(), "values": w[::step].tolist()}


def measure(payload: dict) -> dict:
    try:
        from photonflux import tdec
    except ImportError as exc:
        return {"ok": False, "error": f"TDECQ needs stateye ({exc}); install "
                "the eye extra with docs/patches/ applied"}
    t = np.asarray(payload.get("t") or [], float)
    v = np.asarray(payload.get("values") or [], float)
    ui = float(payload.get("ui") or 0.0)
    if t.size < 64 or t.size != v.size or not ui > 0:
        return {"ok": False, "error": "need a trace of >= 64 points and a UI"}
    nlv = int(payload.get("levels") or 4)
    cfg = dict(payload.get("cfg") or {})

    # uniform grid at the solver's median step, settling dropped up front
    keep = t >= float(payload.get("skip_s") or 0.0)
    t, v = t[keep], v[keep]
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    p = np.interp(tu, t, v)
    baud = 1.0 / ui

    # reference receiver: TDECQ's BT4 at baud/2, TDEC's at 0.75 x baud,
    # unless the caller sets a factor or "off"
    rx_bw = cfg.get("rx_bw", "auto")
    if rx_bw in (None, "", "auto"):
        rx_bw = 0.5 if nlv == 4 else 0.75
    common = dict(
        ref_rx_bw_factor=None if rx_bw in (0, "off") else float(rx_bw),
        ref_rx_order=int(cfg.get("rx_order", 4)),
        s_noise_mW=float(cfg.get("s_noise", 0.0)),
        settle_ui=int(cfg.get("settle_ui", 2)), strict=False,
        # stateye's bathtub fit scales with the histogram grid: 256 x 1024
        # is ~3x faster than 512 x 2048 and moved TDECQ by 0.01 dB here
        nx=int(cfg.get("nx", 256)), ny=int(cfg.get("ny", 1024)))
    log: list[str] = []
    # measure_* drop settle_ui more UI of reference-receiver turn-on
    t_scored = tu[0] + common["settle_ui"] * ui
    try:
        if nlv == 2:
            m = tdec.measure(p, dt, baud, ber=float(cfg.get("ber", 1e-12)),
                             oma_type=str(cfg.get("oma_type", "4140")),
                             **common)
            fam = m.oma_type
            out = {"format": "NRZ",
                   "tdec_db": _num(m.get(f"tdec_{fam}")),
                   "oma": _num(m.get(f"oma_{fam}")),
                   "oma_tdec_dbm": _num(m.get("oma_tdec_dbm")),
                   "at_floor": bool(m.get("at_floor")),
                   "er_db": _num(m.get(f"extinction_ratio_{fam}")),
                   "family": fam,
                   "scored": _scored(m, t_scored, tu, p, ui)}
            return {"ok": True, **out, "log": log}

        symbols = None
        pattern = payload.get("pattern")
        if cfg.get("method", "mmse") != "manual" and \
                cfg.get("use_pattern", True) and pattern:
            nsym = int(np.ceil((tu[-1] - 0.0) / ui)) + 2
            symbols = np.rint(wavesrc._symbols(pattern, nsym) * 3).astype(int)
        manual = None
        if cfg.get("method") == "manual":
            manual = [float(x) for x in str(cfg.get("manual", "")).replace(
                ",", " ").split()]
        pam = dict(common, ser=float(cfg.get("ser", 4.8e-4)))
        raw = tdec.measure_pam4(p, dt, baud, ffe_taps=0, **pam)
        n_taps = len(manual) if manual else int(cfg.get("taps", 5))
        eq = raw if n_taps == 0 else tdec.measure_pam4(
            p, dt, baud, ffe_taps=n_taps, ffe_pre=int(cfg.get("pre", 1)),
            ffe_method=str(cfg.get("method", "mmse")), symbols=symbols,
            ffe_mu=float(cfg.get("mu", 0.05)),
            ffe_passes=int(cfg.get("passes", 5)), ffe_manual=manual,
            ffe_max_evals=int(cfg.get("max_evals", 40)), **pam)
    except (ValueError, IndexError, ZeroDivisionError, ImportError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def pick(m):
        v = _num(m.get("tdecq_outer"))
        return (v, "outer") if v is not None else (_num(m.get("tdecq_xp")), "xp")

    (tq, fam), (tq_raw, _) = pick(eq), pick(raw)
    if fam == "xp":
        log.append("OMA_outer needs runs of 7 threes and 6 zeros (a full "
                   "PRBS-13Q); this record lacks them, so TDECQ uses the "
                   "crossing-point OMA")
    if symbols is None and n_taps and cfg.get("method") != "manual":
        log.append("no PRBS source on the canvas: taps adapted "
                   "decision-directed (reliable only on a mostly open eye)")
    levels = [_num(eq.get(f"{n}_level_xp"))
              for n in ("zero", "one", "two", "three")]
    return {"ok": True, "format": "PAM4", "tdecq_db": tq, "tdecq_raw_db": tq_raw,
            "family": fam, "oma": _num(eq.get(f"oma_{fam}")),
            "ceq": _num(eq.get("ceq", 1.0)),
            "taps": eq.get("ffe_taps") or [], "method": eq.get("ffe_method"),
            "evals": eq.get("ffe_evals"),
            "rms_before": _num(eq.get("ffe_rms_error_before")),
            "rms_after": _num(eq.get("ffe_rms_error_after")),
            "levels": levels, "ser_target": pam["ser"],
            "scored": _scored(eq, t_scored, tu, p, ui), "log": log}
