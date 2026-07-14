"""Serial-link post-processing: BER report, bathtub, RX EQ, pulse/COM.

Everything here is data-aided: the PRBS source's settings regenerate the
exact transmitted symbol sequence (`wavesrc._symbols`), so alignment,
equalizer design and error counting use the known data — no CDR, no blind
adaptation. Conventions follow the user's serdes codebase:

  * decision samples: best phase picked by correlation, symbol-spaced
  * RX FFE/DFE designed by data-aided least squares (the Wiener solution
    computed from the record itself); the DFE feeds back known symbols
  * Q-fit BER: per-eye Q = (mu2 - mu1)/(sigma1 + sigma2),
    BER = 0.5*erfc(Q/sqrt(2)); PAM4 sums the three Gray-coded eyes / 2
  * bathtub: Q(phase) across the UI from per-phase Gaussian fits
  * COM-style figure of merit (pulse mode): equalized cursor vs the RSS of
    residual ISI, FOM = 20*log10(cursor / sigma_isi)
"""
from __future__ import annotations

import numpy as np

import wavesrc


# math.erfc is scalar; a tiny vectorized wrapper keeps the module scipy-free
def _erfc(x):
    from math import erfc as _e
    return np.vectorize(_e)(x)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _uniform(t, v, ui, osr=16):
    """Resample onto an osr-per-UI grid; returns (dt, samples)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    dt = ui / osr
    n = int(np.floor((t[-1] - t[0]) / dt))
    tu = t[0] + np.arange(n) * dt
    return tu, np.interp(tu, t, v)


def _tx_fracs(pattern: dict, nsym: int) -> np.ndarray:
    return wavesrc._symbols(pattern, nsym)


def _levels_of(pattern: dict) -> int:
    return 4 if str(pattern.get("mode", "nrz")) == "pam4" else 2


def _align(samples: np.ndarray, tx: np.ndarray, osr: int,
           skip: int) -> tuple[int, int, float]:
    """Best (phase, symbol lag) by correlating each phase's symbol stream
    with the transmitted fractions. Returns (phase, lag, corr)."""
    nsym_rec = len(samples) // osr
    tx_z = tx[:nsym_rec] - tx[:nsym_rec].mean()
    best = (0, 0, -np.inf)
    for ph in range(osr):
        s = samples[ph::osr][:nsym_rec]
        if len(s) < skip + 32:
            continue
        s_z = s - s.mean()
        # search integer symbol lags: RX sample k corresponds to TX k - lag
        # (up to 32 UI of link latency, e.g. a dispersive fiber's transit)
        for lag in range(0, 33):
            a = s_z[skip + lag:]
            b = tx_z[skip:len(tx_z) - lag] if lag else tx_z[skip:]
            m = min(len(a), len(b))
            if m < 32:
                continue
            c = float(np.dot(a[:m], b[:m]) /
                      (np.linalg.norm(a[:m]) * np.linalg.norm(b[:m]) + 1e-30))
            if c > best[2]:
                best = (ph, lag, c)
    return best


def _ls_equalize(y: np.ndarray, a: np.ndarray, n_ffe: int, n_dfe: int,
                 skip: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data-aided least-squares FFE(+DFE): min ||a_k - w.y_k - b.a_prev||.

    y: symbol-spaced RX samples aligned to a (targets). Returns
    (equalized, ffe_taps, dfe_taps). n_ffe = 0 -> pass-through.
    """
    n = min(len(y), len(a))
    y, a = y[:n], a[:n]
    if n_ffe <= 0:
        return y, np.array([1.0]), np.zeros(0)
    pre = n_ffe // 2
    rows = []
    tgts = []
    for k in range(skip + n_ffe + n_dfe, n):
        row = [y[k - i + pre] if 0 <= k - i + pre < n else 0.0
               for i in range(n_ffe)]
        row += [a[k - 1 - j] for j in range(n_dfe)]
        rows.append(row)
        tgts.append(a[k])
    A = np.asarray(rows)
    b = np.asarray(tgts)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    eq = A @ w
    ffe, dfe = w[:n_ffe], w[n_ffe:]
    # rebuild the full-length equalized stream (aligned with `a[start:]`)
    return eq, ffe, dfe


def _qfit(eq: np.ndarray, tx: np.ndarray, nlv: int) -> dict:
    """Per-eye Gaussian Q fit on equalized samples grouped by known symbol."""
    lv_fracs = np.linspace(0, 1, nlv)
    mus, sigmas, counts = [], [], []
    for f in lv_fracs:
        sel = np.isclose(tx, f, atol=1e-6)
        s = eq[sel]
        if len(s) < 8:
            return {"ok": False, "reason": f"level {f:.2f}: too few samples"}
        mus.append(float(s.mean()))
        sigmas.append(float(s.std() + 1e-30))
        counts.append(int(len(s)))
    qs, bers = [], []
    for e in range(nlv - 1):
        q = (mus[e + 1] - mus[e]) / (sigmas[e] + sigmas[e + 1])
        qs.append(float(q))
        bers.append(float(0.5 * _erfc(max(q, 0.0) / np.sqrt(2.0))))
    # PAM4 Gray: each symbol error across one eye is one bit error in two bits
    ber = float(np.sum(bers) / 2.0) if nlv == 4 else float(bers[0])
    return {"ok": True, "levels": mus, "sigmas": sigmas, "counts": counts,
            "q": qs, "ber_est": ber}


def _count_errors(eq: np.ndarray, tx: np.ndarray, nlv: int) -> dict:
    """Slice with midpoints between fitted level means; count errors."""
    lv_fracs = np.linspace(0, 1, nlv)
    mus = [float(eq[np.isclose(tx, f, atol=1e-6)].mean()) for f in lv_fracs]
    thr = [(mus[i] + mus[i + 1]) / 2 for i in range(nlv - 1)]
    dec = np.zeros(len(eq), dtype=int)
    for t in thr:
        dec += (eq > t).astype(int)
    sym_tx = np.rint(tx * (nlv - 1)).astype(int)
    serrs = int(np.sum(dec != sym_tx))
    if nlv == 4:
        # Gray: adjacent-level slips are 1 bit; count exact bit differences
        gray = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
        bits_err = sum(int(gray[int(d)][0] != gray[int(s)][0]) +
                       int(gray[int(d)][1] != gray[int(s)][1])
                       for d, s in zip(dec, sym_tx) if d != s)
        nbits = 2 * len(eq)
    else:
        bits_err = serrs
        nbits = len(eq)
    return {"symbols": int(len(eq)), "sym_errors": serrs,
            "ser": serrs / max(len(eq), 1),
            "bits": nbits, "bit_errors": int(bits_err),
            "ber": bits_err / max(nbits, 1)}


def _bathtub(samples: np.ndarray, tx: np.ndarray, osr: int, phase: int,
             lag: int, skip: int, nlv: int) -> dict:
    """log10(BER) vs sampling phase from per-phase Gaussian Q fits (raw,
    unequalized samples — the classic scope-style bathtub)."""
    nsym = len(samples) // osr
    lv_fracs = np.linspace(0, 1, nlv)
    phases = []
    log_ber = []
    for ph in range(osr):
        s = samples[ph::osr][:nsym]
        m = min(len(s) - lag - skip, len(tx) - skip)
        if m < 64:
            continue
        s_k = s[skip + lag: skip + lag + m]
        a_k = tx[skip: skip + m]
        bers = []
        okay = True
        for e in range(nlv - 1):
            lo = s_k[np.isclose(a_k, lv_fracs[e], atol=1e-6)]
            hi = s_k[np.isclose(a_k, lv_fracs[e + 1], atol=1e-6)]
            if len(lo) < 8 or len(hi) < 8:
                okay = False
                break
            q = (hi.mean() - lo.mean()) / (lo.std() + hi.std() + 1e-30)
            bers.append(0.5 * float(_erfc(max(q, 0.0) / np.sqrt(2.0))))
        if not okay:
            continue
        tot = float(np.sum(bers) / (2.0 if nlv == 4 else 1.0))
        phases.append(((ph - phase) % osr) / osr - 0.5)
        log_ber.append(float(np.log10(max(tot, 1e-30))))
    order = np.argsort(phases)
    return {"phase_ui": [phases[i] for i in order],
            "log10_ber": [log_ber[i] for i in order]}


# ---------------------------------------------------------------------------
# BER / link report (task: transient with a PRBS source)
# ---------------------------------------------------------------------------

def link_report(result: dict, meta: dict, cfg: dict, log: list) -> dict | None:
    patterns = meta.get("patterns") or {}
    if not patterns:
        log.append("link report: no PRBS source in the schematic")
        return None
    pat_inst = cfg.get("pattern") or next(iter(patterns))
    pattern = patterns.get(pat_inst)
    if pattern is None:
        log.append(f"link report: no PRBS instance {pat_inst!r}")
        return None
    probe = cfg.get("probe")
    # multi-seed noise runs produce a family "probe#k" sharing tr["probe"]
    trs = [t for t in result["traces"]
           if t["name"] == probe or t.get("probe") == probe]
    if not trs:
        log.append(f"link report: probe {probe!r} not found")
        return None

    ui = float(pattern.get("ui", 100e-12))
    nlv = _levels_of(pattern)
    osr = 16
    t = np.asarray(result["x"], float)
    _tu, samples0 = _uniform(t, trs[0]["values"], ui, osr)
    nsym_rec = len(samples0) // osr
    if nsym_rec < 100:
        log.append(f"link report: only {nsym_rec} symbols in the record — "
                   "run a longer transient (>= ~100 UI)")
        return None
    tx = _tx_fracs(pattern, nsym_rec + 16)
    skip = int(cfg.get("skip", max(32, nsym_rec // 20)))

    phase, lag, corr = _align(samples0, tx, osr, skip)
    n_ffe = int(cfg.get("ffe_taps", 0))
    n_dfe = int(cfg.get("dfe_taps", 0))
    start = skip + (n_ffe + n_dfe if n_ffe > 0 else 0)

    eq_all, a_all = [], []
    ffe, dfe = np.array([1.0]), np.zeros(0)
    for i, tr in enumerate(trs):
        samples = samples0 if i == 0 else _uniform(t, tr["values"], ui, osr)[1]
        y = samples[phase::osr][:nsym_rec]
        m = min(len(y) - lag, nsym_rec)
        y_k = y[lag: lag + m]
        a_k = tx[:m]
        if n_ffe > 0:
            eq_i, ffe_i, dfe_i = _ls_equalize(y_k, a_k, n_ffe, n_dfe, skip)
            if i == 0:
                ffe, dfe = ffe_i, dfe_i
        else:
            eq_i = y_k[start:]
        eq_all.append(eq_i)
        a_all.append(a_k[start:start + len(eq_i)])
    eq = np.concatenate(eq_all)
    a_eq = np.concatenate(a_all)

    counted = _count_errors(eq, a_eq, nlv)
    qfit = _qfit(eq, a_eq, nlv)
    tub = _bathtub(samples0, tx, osr, phase, lag, skip, nlv)

    seed_tag = f" x {len(trs)} seeds" if len(trs) > 1 else ""
    log.append(f"link report: {counted['symbols']} symbols{seed_tag} @ phase "
               f"{phase}/{osr}, lag {lag} UI (corr {corr:.3f}); counted BER "
               f"{counted['ber']:.3g}, Q-fit BER {qfit.get('ber_est', 0):.3g}")
    return {
        "pattern": pat_inst, "probe": probe, "ui": ui, "nlv": nlv,
        "sampling_phase_ui": phase / osr, "lag_ui": lag, "corr": corr,
        "seeds": len(trs),
        "counted": counted, "qfit": qfit, "bathtub": tub,
        "ffe_taps": [float(x) for x in ffe],
        "dfe_taps": [float(x) for x in dfe],
        "skip": skip,
    }


def eye_height(t, values, ui: float, nlv: int = 2, skip_frac: float = 0.1,
               osr: int = 32) -> float:
    """Worst-case inner eye opening at the decision phase (server-side twin
    of the UI's eye metric; used as an optimizer objective)."""
    t = np.asarray(t, float)
    v = np.asarray(values, float)
    t0 = t[0] + skip_frac * (t[-1] - t[0])
    sel = t >= t0
    _tu, s = _uniform(t[sel], v[sel], ui, osr)
    if len(s) < 8 * osr:
        return 0.0
    # samples near each phase; pick the decision phase as the one with the
    # best opening (no alignment information here)
    best = 0.0
    for ph in range(0, osr, 2):
        smp = s[ph::osr]
        lv = np.quantile(smp, np.linspace(0.03, 0.97, nlv))
        for _ in range(12):        # tiny 1-D k-means
            idx = np.argmin(np.abs(smp[:, None] - lv[None, :]), axis=1)
            for c in range(nlv):
                if (idx == c).any():
                    lv[c] = smp[idx == c].mean()
            lv.sort()
        idx = np.argmin(np.abs(smp[:, None] - lv[None, :]), axis=1)
        opening = np.inf
        for e in range(nlv - 1):
            top = smp[idx == e].max() if (idx == e).any() else -np.inf
            bot = smp[idx == e + 1].min() if (idx == e + 1).any() else np.inf
            opening = min(opening, bot - top)
        if np.isfinite(opening):
            best = max(best, opening)
    return float(max(best, 0.0))


# ---------------------------------------------------------------------------
# pulse response + Wiener/COM figure of merit
# ---------------------------------------------------------------------------

def _wiener_ffe_dfe(h: np.ndarray, cursor: int, n_ffe: int, n_dfe: int,
                    snr_reg: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """MMSE FFE (+ ideal DFE) from a symbol-spaced pulse response.

    Standard construction: convolution matrix H (data symbols x taps),
    solve (H^T H + reg I) w = H^T e_cursor with the DFE-cancelled
    post-cursors excluded from the error.
    """
    L = len(h)
    pre = n_ffe // 2
    # rows: shifted copies of h -> received y_k = sum_i h_i a_{k-i}
    # FFE output cursor c = sum_j w_j y_{k+pre-j}
    n = L + n_ffe
    H = np.zeros((n, n_ffe))
    for j in range(n_ffe):
        H[j:j + L, j] = h
    d = np.zeros(n)
    cur_row = cursor + pre
    d[cur_row] = 1.0
    # residual ISI rows the DFE will cancel (the n_dfe symbols after cursor)
    mask = np.ones(n, bool)
    if n_dfe > 0:
        mask[cur_row + 1: cur_row + 1 + n_dfe] = False
    A = H[mask]
    dd = d[mask]
    w = np.linalg.solve(A.T @ A + snr_reg * np.eye(n_ffe), A.T @ dd)
    resid = H @ w
    dfe = resid[cur_row + 1: cur_row + 1 + n_dfe].copy() if n_dfe else np.zeros(0)
    return w, dfe


def pulse_report(result: dict, meta: dict, cfg: dict, log: list) -> dict | None:
    """Pulse-response extraction + Wiener EQ + COM-style FOM."""
    patterns = meta.get("patterns") or {}
    pat_inst = cfg.get("pattern") or next(iter(patterns), None)
    pattern = patterns.get(pat_inst) if pat_inst else None
    probe = cfg.get("probe")
    tr = next((t for t in result["traces"] if t["name"] == probe), None)
    if pattern is None or tr is None:
        log.append("pulse report: needs a PRBS source (mode=pulse) and a "
                   "probed output")
        return None
    ui = float(pattern.get("ui", 100e-12))
    osr = 32
    t = np.asarray(result["x"], float)
    _tu, samples = _uniform(t, tr["values"], ui, osr)
    base = np.median(samples[:2 * osr])     # runway baseline
    p = samples - base
    # pulse was emitted in symbol slot 4 (wavesrc pulse mode)
    peak = int(np.argmax(np.abs(p)))
    # symbol-spaced pulse response at the peak's phase
    ph = peak % osr
    h_full = p[ph::osr]
    cur = int(np.argmax(np.abs(h_full)))
    lo = max(0, cur - 8)
    hi = min(len(h_full), cur + 16)
    h = h_full[lo:hi]
    cursor = cur - lo
    scale = float(np.abs(h[cursor]) + 1e-30)

    n_ffe = int(cfg.get("ffe_taps", 5) or 0)
    n_dfe = int(cfg.get("dfe_taps", 1) or 0)
    if n_ffe > 0:
        w, dfe = _wiener_ffe_dfe(h / scale, cursor, n_ffe, n_dfe)
        Hc = np.zeros(len(h) + n_ffe)
        for j in range(n_ffe):
            Hc[j:j + len(h)] += w[j] * (h / scale)
        cur_row = cursor + n_ffe // 2
    else:
        w, dfe = np.array([1.0]), np.zeros(0)
        Hc = (h / scale).copy()
        cur_row = cursor
    cursor_amp = float(Hc[cur_row])
    isi = Hc.copy()
    isi[cur_row] = 0.0
    if n_dfe:
        isi[cur_row + 1: cur_row + 1 + n_dfe] = 0.0
    sigma_isi = float(np.sqrt(np.sum(isi ** 2) / 3.0))   # ~uniform data power
    fom_db = float(20.0 * np.log10(max(cursor_amp, 1e-30) /
                                   max(sigma_isi, 1e-30)))
    log.append(f"pulse report: cursor {scale:.4g} at symbol {cur}, "
               f"FOM {fom_db:.1f} dB with {n_ffe}-tap FFE + {n_dfe}-tap DFE")
    return {
        "probe": probe, "ui": ui,
        "h": [float(v) for v in h / scale], "cursor": cursor,
        "h_raw_peak": scale,
        "ffe_taps": [float(v) for v in w],
        "dfe_taps": [float(v) for v in dfe],
        "eq_pulse": [float(v) for v in Hc], "eq_cursor": cur_row,
        "sigma_isi": sigma_isi, "fom_db": fom_db,
    }
