"""Coherent transceiver DSP: CD comp, adaptive EQ, carrier recovery, EVM/BER.

The IM/DD twin of this module is ``linkpost.py``; the shared house style is
*data-aided* post-processing — the QAM source's settings regenerate the exact
transmitted symbols (``wavesrc.qam_symbols``), so alignment and metrics use the
known data. The blind DSP blocks (CD compensation, CMA / decision-directed LMS,
frequency-offset and carrier-phase recovery) are the ones a real coherent
receiver runs; the known symbols are used only to resolve the residual lag /
constellation rotation and to score EVM & BER.

Signal convention: the coherent front-end delivers a complex baseband sample
stream ``r = I + jQ`` (the balanced I and Q photocurrents). Everything is plain
numpy at report time — no scipy, matching ``linkpost``.

Pipeline (``coherent_report``):
    matched filter -> CD compensation (freq domain) -> resample to symbol rate
    -> CMA / DD-LMS (optional) -> frequency-offset removal -> carrier-phase
    recovery (Viterbi-Viterbi for QPSK, blind phase search for higher QAM)
    -> data-aided sync -> constellation / EVM / BER.
"""
from __future__ import annotations

import numpy as np

import wavesrc


def _erfc(x):
    from math import erfc as _e
    return np.vectorize(_e)(x)


# ---------------------------------------------------------------------------
# constellation decisions + bit mapping
# ---------------------------------------------------------------------------

def decide(sym: np.ndarray, const: np.ndarray) -> np.ndarray:
    """Nearest-constellation-point index for each received symbol."""
    d = np.abs(sym[:, None] - const[None, :])
    return np.argmin(d, axis=1)


def symbol_bits(idx: np.ndarray, m: int) -> np.ndarray:
    """(n, bps) bit matrix for symbol integers (MSB first) — the Gray labels."""
    bps = wavesrc.qam_bits_per_symbol(m)
    idx = np.asarray(idx, int)
    return np.stack([(idx >> (bps - 1 - b)) & 1 for b in range(bps)], axis=1)


# ---------------------------------------------------------------------------
# channel models (used to validate the DSP; a real run gets these from the
# circuit — dispersive fiber, laser phase noise, receiver noise)
# ---------------------------------------------------------------------------

def awgn(sig: np.ndarray, snr_db: float, sps: int = 1,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex AWGN for a target per-symbol SNR = Es/N0 [dB].

    ``sps`` > 1 spreads the same Es over ``sps`` samples so the post-matched-
    filter symbol SNR still equals ``snr_db``.
    """
    rng = rng or np.random.default_rng(0)
    es = np.mean(np.abs(sig) ** 2)
    n0 = es / (10.0 ** (snr_db / 10.0)) * sps
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(sig.shape)
                                 + 1j * rng.standard_normal(sig.shape))
    return sig + noise


def phase_noise(n: int, linewidth: float, symbol_rate: float, sps: int = 1,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Wiener phase walk for a combined TX+LO Lorentzian linewidth [Hz].

    Variance per sample is 2*pi*linewidth/f_s (f_s = symbol_rate*sps).
    """
    rng = rng or np.random.default_rng(0)
    if linewidth <= 0.0:
        return np.zeros(n)
    fs = symbol_rate * sps
    sigma = np.sqrt(2.0 * np.pi * linewidth / fs)
    return np.cumsum(sigma * rng.standard_normal(n))


def cd_filter(nf: int, fs: float, beta2_L: float, beta3_L: float = 0.0,
              inverse: bool = False) -> np.ndarray:
    """All-pass CD transfer over ``nf`` FFT bins at sample rate ``fs``.

    ``H(w) = exp(-j (beta2_L/2 w^2 + beta3_L/6 w^3))``; ``inverse`` conjugates
    it (the receiver's compensator). ``beta2_L`` is beta2*length [s^2].
    """
    w = 2.0 * np.pi * np.fft.fftfreq(nf, d=1.0 / fs)
    phi = 0.5 * beta2_L * w ** 2 + (beta3_L / 6.0) * w ** 3
    return np.exp((1j if inverse else -1j) * phi)


def apply_cd(sig: np.ndarray, fs: float, beta2_L: float,
             beta3_L: float = 0.0) -> np.ndarray:
    """Propagate the complex envelope through chromatic dispersion."""
    nf = len(sig)
    return np.fft.ifft(np.fft.fft(sig) * cd_filter(nf, fs, beta2_L, beta3_L))


def beta2_of(D_ps: float, lam_nm: float) -> float:
    """beta2 [s^2/m] from dispersion D [ps/nm/km] at wavelength [nm]."""
    c = 299792458.0
    lam = lam_nm * 1e-9
    D = D_ps * 1e-6                      # ps/nm/km -> s/m/m
    return -D * lam ** 2 / (2.0 * np.pi * c)


# ---------------------------------------------------------------------------
# transmit shaping (fractionally-sampled, RRC) — also a validation utility
# ---------------------------------------------------------------------------

def upsample_shape(syms: np.ndarray, sps: int, beta: float,
                   span: int = 16) -> np.ndarray:
    """Zero-stuff to ``sps``/symbol and RRC pulse-shape (unit-energy taps)."""
    up = np.zeros(len(syms) * sps, dtype=complex)
    up[::sps] = syms
    h = wavesrc.rrc_taps(beta, sps, span)
    return np.convolve(up, h, mode="same")


def matched_filter(sig: np.ndarray, sps: int, beta: float,
                   span: int = 16) -> np.ndarray:
    h = wavesrc.rrc_taps(beta, sps, span)
    return np.convolve(sig, h, mode="same")


# ---------------------------------------------------------------------------
# receiver DSP
# ---------------------------------------------------------------------------

def cd_compensate(sig: np.ndarray, fs: float, D_ps: float, length_km: float,
                  lam_nm: float = 1550.0, S_ps: float = 0.0) -> np.ndarray:
    """Frequency-domain bulk CD compensation (inverse all-pass)."""
    beta2 = beta2_of(D_ps, lam_nm)
    beta2_L = beta2 * length_km * 1e3
    # dispersion slope -> beta3 (small third-order term; 0 when S not given)
    c = 299792458.0
    lam = lam_nm * 1e-9
    beta3 = (S_ps * 1e3) * (lam ** 2 / (2 * np.pi * c)) ** 2 if S_ps else 0.0
    beta3_L = beta3 * length_km * 1e3
    nf = len(sig)
    return np.fft.ifft(np.fft.fft(sig)
                       * cd_filter(nf, fs, beta2_L, beta3_L, inverse=True))


def cma_equalize(sig: np.ndarray, ntaps: int, mu: float, sps: int = 2,
                 R2: float | None = None, warmup: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """T/sps fractionally-spaced single-pol CMA -> symbol-rate output.

    Constant-modulus (Godard p=2) blind equalizer with a centre-spike init.
    Returns (symbols, taps). Radius ``R2 = E|s|^4 / E|s|^2`` (2 for a
    normalised constellation dominated by the ring; auto if None).
    """
    sig = np.asarray(sig, complex)
    if R2 is None:
        R2 = 1.0
    w = np.zeros(ntaps, complex)
    w[ntaps // 2] = 1.0
    out = []
    nsym = (len(sig) - ntaps) // sps
    warmup = 0 if warmup is None else warmup
    for k in range(nsym):
        seg = sig[k * sps: k * sps + ntaps]
        if len(seg) < ntaps:
            break
        y = w @ seg[::-1]
        out.append(y)
        if k >= warmup:
            e = y * (R2 - np.abs(y) ** 2)
            w = w + mu * e * np.conj(seg[::-1])
    return np.asarray(out), w


def dd_lms_equalize(sig: np.ndarray, const: np.ndarray, ntaps: int, mu: float,
                    sps: int = 2, train: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Decision-directed (or trained) LMS FSE -> symbol-rate output."""
    sig = np.asarray(sig, complex)
    w = np.zeros(ntaps, complex)
    w[ntaps // 2] = 1.0
    out = []
    nsym = (len(sig) - ntaps) // sps
    for k in range(nsym):
        seg = sig[k * sps: k * sps + ntaps][::-1]
        if len(seg) < ntaps:
            break
        y = w @ seg
        out.append(y)
        d = train[k] if (train is not None and k < len(train)) \
            else const[np.argmin(np.abs(y - const))]
        e = d - y
        w = w + mu * e * np.conj(seg)
    return np.asarray(out), w


def freq_offset_estimate(sym: np.ndarray, symbol_rate: float,
                         m: int = 4) -> float:
    """Coarse carrier-frequency offset [Hz] via the 4th-power FFT peak.

    Raising to the 4th power strips the QPSK modulation (and the QAM QPSK-like
    corner ring), leaving a tone at 4*f_offset.
    """
    p = 4
    x = sym ** p
    nfft = 1 << int(np.ceil(np.log2(max(len(x), 16))))
    spec = np.abs(np.fft.fft(x, nfft))
    k = np.argmax(spec)
    f = np.fft.fftfreq(nfft, d=1.0 / symbol_rate)[k]
    return float(f / p)


def remove_freq_offset(sym: np.ndarray, f_hz: float,
                       symbol_rate: float) -> np.ndarray:
    n = np.arange(len(sym))
    return sym * np.exp(-1j * 2.0 * np.pi * f_hz / symbol_rate * n)


def viterbi_viterbi(sym: np.ndarray, window: int = 21) -> np.ndarray:
    """QPSK carrier-phase recovery: 4th-power, smooth, derotate.

    The estimate lives on the pi/2-periodic QPSK lattice, so it is unwrapped
    with period pi/2 — unwrapping the raw ±pi angle instead injects spurious
    cycle slips that corrupt long records.
    """
    p4 = sym ** 4
    if window > 1:
        k = np.ones(window) / window
        p4 = np.convolve(p4, k, mode="same")
    phi = np.angle(p4) / 4.0                     # (-pi/4, pi/4]
    phi = np.unwrap(phi, period=np.pi / 2.0)     # continuous QPSK branch
    return sym * np.exp(-1j * phi)


def blind_phase_search(sym: np.ndarray, const: np.ndarray, n_test: int = 32,
                       window: int = 25) -> np.ndarray:
    """Blind phase search CPR for square QAM (Pfau et al.)."""
    # only the first quadrant of test angles is needed (pi/2 symmetry)
    angles = np.arange(n_test) / n_test * (np.pi / 2.0)
    rot = np.exp(1j * angles)                       # (n_test,)
    trial = sym[:, None] * rot[None, :]             # (n, n_test)
    # squared distance to nearest constellation point per trial
    d = np.min(np.abs(trial[:, :, None] - const[None, None, :]) ** 2, axis=2)
    if window > 1:
        k = np.ones(window)
        d = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"),
                                0, d)
    best = np.argmin(d, axis=1)
    phi = np.unwrap(angles[best], period=np.pi / 2.0)   # pi/2 QAM lattice
    return sym * np.exp(1j * phi)


# ---------------------------------------------------------------------------
# data-aided sync + metrics
# ---------------------------------------------------------------------------

def _pair(rx, tx, d):
    """Slices aligning rx[k] with tx[k+d] (d>0: rx leads / equalizer delay)."""
    if d >= 0:
        a, b = rx, tx[d:]
    else:
        a, b = rx[-d:], tx
    m = min(len(a), len(b))
    return a[:m], b[:m]


def sync(rx: np.ndarray, tx: np.ndarray, max_lag: int = 64
         ) -> tuple[np.ndarray, np.ndarray, int, complex]:
    """Align rx to tx: find signed integer lag + constant rotation.

    Data-aided: the lag may be negative (adaptive equalizers delay the output
    by ~half their length, so the recovered stream *leads* the transmit one).
    Returns (rx_aligned, tx_aligned, lag, rot) with rx_aligned*... already
    derotated and equal lengths.
    """
    best = (0, 0.0, 1 + 0j)
    for d in range(-max_lag, max_lag + 1):
        a, b = _pair(rx, tx, d)
        m = min(len(a), len(b))
        if m < 32:
            continue
        c = np.vdot(a, b)                            # sum conj(rx)*tx
        score = np.abs(c) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
        if score > best[1]:
            best = (d, score, c / (np.abs(c) + 1e-30))
    lag, _score, rot = best
    a, b = _pair(rx, tx, lag)
    return a * rot, b, lag, rot


def evm(rx: np.ndarray, tx: np.ndarray) -> float:
    """RMS EVM (fraction of reference RMS), normalised to the reference."""
    err = rx - tx
    return float(np.sqrt(np.mean(np.abs(err) ** 2)
                         / np.mean(np.abs(tx) ** 2)))


def per_cluster_evm(rx: np.ndarray, tx: np.ndarray,
                    const: np.ndarray) -> list[dict]:
    """EVM per constellation point (16-QAM report). Normalised to overall RMS."""
    ref_rms = np.sqrt(np.mean(np.abs(const) ** 2))
    out = []
    for s, pt in enumerate(const):
        sel = np.isclose(tx, pt, atol=1e-6)
        cnt = int(np.sum(sel))
        if cnt == 0:
            out.append({"symbol": s, "count": 0, "evm": 0.0,
                        "re": float(pt.real), "im": float(pt.imag)})
            continue
        e = np.sqrt(np.mean(np.abs(rx[sel] - pt) ** 2)) / ref_rms
        out.append({"symbol": s, "count": cnt, "evm": float(e),
                    "re": float(pt.real), "im": float(pt.imag)})
    return out


def count_ber(rx: np.ndarray, tx: np.ndarray, const: np.ndarray,
              m: int) -> dict:
    """Symbol-counted SER/BER via nearest-point decisions and Gray labels."""
    dec = decide(rx, const)
    ref = decide(tx, const)                          # exact tx labels
    sym_err = int(np.sum(dec != ref))
    bits_d = symbol_bits(dec, m)
    bits_r = symbol_bits(ref, m)
    bit_err = int(np.sum(bits_d != bits_r))
    nbits = bits_r.size
    return {"symbols": int(len(rx)), "sym_errors": sym_err,
            "ser": sym_err / max(len(rx), 1),
            "bits": nbits, "bit_errors": bit_err,
            "ber": bit_err / max(nbits, 1)}


def ber_from_evm(evm_rms: float, m: int) -> float:
    """Gaussian-fit BER from EVM for square M-QAM (SNR = 1/EVM^2)."""
    if evm_rms <= 0:
        return 0.0
    snr = 1.0 / evm_rms ** 2
    L = int(round(np.sqrt(m)))
    # standard square-QAM BER(SNR) with Gray coding
    arg = np.sqrt(3.0 * snr / (m - 1.0))
    ser_axis = 2.0 * (1.0 - 1.0 / L) * 0.5 * _erfc(arg / np.sqrt(2.0))
    ber = float(ser_axis / np.log2(L))               # ~1 bit / axis slip (Gray)
    return max(min(ber, 0.5), 0.0)


def qam_theory_ber(snr_db: float, m: int) -> float:
    """Theoretical Gray square-QAM BER vs Es/N0 [dB] (validation reference)."""
    snr = 10.0 ** (snr_db / 10.0)
    return ber_from_evm(1.0 / np.sqrt(snr), m)


# ---------------------------------------------------------------------------
# full receiver chain + report
# ---------------------------------------------------------------------------

def receive(rx: np.ndarray, const: np.ndarray, m: int, cfg: dict,
            tx: np.ndarray | None = None, log: list | None = None
            ) -> dict:
    """Run the coherent DSP chain on complex samples ``rx``.

    ``cfg`` keys: sps, rrc_beta, cd (D_ps, length_km, lam_nm), eq
    ('none'/'cma'/'dd-lms'), eq_taps, eq_mu, cpr ('vv'/'bps'/'none'),
    symbol_rate, freq_recover (bool). ``tx`` (known symbols) is only used for
    the final data-aided sync + metrics. Returns the DSP outputs.
    """
    log = log if log is not None else []
    sps = int(cfg.get("sps", 1))
    beta = float(cfg.get("rrc_beta", 0.0))
    symbol_rate = float(cfg.get("symbol_rate", 1.0))
    fs = symbol_rate * sps

    sig = np.asarray(rx, complex)
    # 1. matched filter (if pulse-shaped)
    if beta > 0.0 and sps > 1:
        sig = matched_filter(sig, sps, beta)
    # 2. bulk CD compensation
    cd = cfg.get("cd")
    if cd:
        sig = cd_compensate(sig, fs, float(cd.get("D_ps", 17.0)),
                            float(cd.get("length_km", 0.0)),
                            float(cd.get("lam_nm", 1550.0)),
                            float(cd.get("S_ps", 0.0)))
        log.append(f"coherent: CD comp D={cd.get('D_ps')} ps/nm/km over "
                   f"{cd.get('length_km')} km")
    # 3. equalize / downsample to symbol rate
    eq = str(cfg.get("eq", "none"))
    ntaps = int(cfg.get("eq_taps", 15))
    mu = float(cfg.get("eq_mu", 1e-3))
    if eq == "cma" and sps >= 1:
        syms, _w = cma_equalize(sig, ntaps, mu, sps=max(sps, 1))
        log.append(f"coherent: CMA EQ {ntaps} taps, mu={mu:g}")
    elif eq == "dd-lms":
        syms, _w = dd_lms_equalize(sig, const, ntaps, mu, sps=max(sps, 1))
        log.append(f"coherent: DD-LMS EQ {ntaps} taps, mu={mu:g}")
    else:
        # sample at symbol centres (best of sps phases if pulse-shaped)
        syms = _downsample(sig, sps, const)
    # 4. frequency-offset removal
    f_off = 0.0
    if cfg.get("freq_recover"):
        f_off = freq_offset_estimate(syms, symbol_rate, m)
        syms = remove_freq_offset(syms, f_off, symbol_rate)
        log.append(f"coherent: CFO estimate {f_off:.3g} Hz")
    # 5. carrier-phase recovery
    cpr = str(cfg.get("cpr", "vv" if m == 4 else "bps"))
    if cpr == "vv":
        syms = viterbi_viterbi(syms, int(cfg.get("cpr_window", 21)))
    elif cpr == "bps":
        syms = blind_phase_search(syms, const,
                                  int(cfg.get("bps_test", 32)),
                                  int(cfg.get("cpr_window", 25)))
    return {"syms": syms, "freq_offset_hz": float(f_off), "eq": eq, "cpr": cpr}


def _downsample(sig: np.ndarray, sps: int, const) -> np.ndarray:
    """Pick the symbol-rate sampling phase with the tightest constellation."""
    if sps <= 1:
        return sig
    best = None
    best_metric = np.inf
    for ph in range(sps):
        s = sig[ph::sps]
        # tightness = mean distance to nearest point after coarse power-norm
        sc = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-30) \
            * np.sqrt(np.mean(np.abs(const) ** 2))
        d = np.mean(np.min(np.abs(sc[:2000, None] - const[None, :]), axis=1))
        if d < best_metric:
            best_metric, best = d, s
    return best


def coherent_report(result: dict, meta: dict, cfg: dict,
                    log: list) -> dict | None:
    """Coherent-link report: constellation, EVM, per-cluster EVM, BER.

    Reads the I/Q photocurrent probe(s) from a transient ``result`` (either one
    complex-labelled probe, or ``probe_i``/``probe_q`` real traces), regenerates
    the QAM source symbols, runs :func:`receive`, and scores the constellation.
    """
    patterns = meta.get("patterns") or {}
    qam_pats = {k: v for k, v in patterns.items()
                if str(v.get("mode", "")) == "qam"}
    if not qam_pats:
        log.append("coherent report: no QAM source in the schematic")
        return None
    pat_inst = cfg.get("pattern") or next(iter(qam_pats))
    pattern = patterns.get(pat_inst)
    if pattern is None:
        log.append(f"coherent report: no QAM instance {pat_inst!r}")
        return None

    rx_raw = _assemble_complex(result, cfg, log)
    if rx_raw is None:
        return None

    m = wavesrc.qam_order(pattern)
    const = wavesrc.qam_constellation(m)
    sps = int(cfg.get("sps", pattern.get("sps", 1)))
    ui = float(pattern.get("ui", 100e-12))
    symbol_rate = 1.0 / ui
    # resample the (solver-grid) I/Q traces onto a uniform sps-per-symbol grid
    rx = _resample(result.get("x"), rx_raw, ui, sps)
    nsym = len(rx) // max(sps, 1)
    if nsym < 100:
        log.append(f"coherent report: only {nsym} symbols — run a longer "
                   "transient (>= ~100 symbols)")
        return None
    tx = wavesrc.qam_symbols(pattern, nsym + 16)

    rcfg = {"sps": sps, "rrc_beta": float(pattern.get("rrc_beta", 0.0)),
            "symbol_rate": symbol_rate, **cfg}
    dsp = receive(rx, const, m, rcfg, tx=tx, log=log)
    syms = dsp["syms"]

    skip = int(cfg.get("skip", max(16, len(syms) // 20)))
    syms = syms[skip:]
    rx_al, tx_al, lag, rot = sync(syms, tx[skip:], int(cfg.get("max_lag", 64)))
    # power-normalise the received cloud to the reference constellation
    scale = np.sqrt(np.mean(np.abs(tx_al) ** 2)
                    / (np.mean(np.abs(rx_al) ** 2) + 1e-30))
    rx_al = rx_al * scale

    evm_rms = evm(rx_al, tx_al)
    clusters = per_cluster_evm(rx_al, tx_al, const)
    counted = count_ber(rx_al, tx_al, const, m)
    ber_gauss = ber_from_evm(evm_rms, m)

    # constellation points for the plot (cap for payload size)
    cap = int(cfg.get("plot_points", 4000))
    step = max(1, len(rx_al) // cap)
    pts = rx_al[::step]

    log.append(f"coherent report: {counted['symbols']} {_name(m)} symbols @ "
               f"lag {lag}, EVM {evm_rms * 100:.2f}% ({20 * np.log10(evm_rms + 1e-30):.1f} dB); "
               f"counted BER {counted['ber']:.3g}, EVM-fit BER {ber_gauss:.3g}"
               f"; CFO {dsp['freq_offset_hz']:.3g} Hz")
    return {
        "pattern": pat_inst, "order": m, "name": _name(m), "ui": ui,
        "sps": sps, "lag": lag,
        "evm_rms": float(evm_rms), "evm_pct": float(evm_rms * 100.0),
        "evm_db": float(20.0 * np.log10(evm_rms + 1e-30)),
        "snr_db": float(-20.0 * np.log10(evm_rms + 1e-30)),
        "counted": counted, "ber_evm": float(ber_gauss),
        "freq_offset_hz": float(dsp["freq_offset_hz"]),
        "eq": dsp["eq"], "cpr": dsp["cpr"],
        "clusters": clusters,
        "const_re": [float(c.real) for c in const],
        "const_im": [float(c.imag) for c in const],
        "rx_re": [float(p.real) for p in pts],
        "rx_im": [float(p.imag) for p in pts],
    }


def _resample(t, sig, ui: float, sps: int) -> np.ndarray:
    """Resample a complex trace onto a uniform ui/sps grid (identity if it
    already sits on that grid, e.g. a 1-sample/symbol synthetic record)."""
    sig = np.asarray(sig, complex)
    if t is None:
        return sig
    t = np.asarray(t, float)
    n = min(len(t), len(sig))
    t, sig = t[:n], sig[:n]
    dt = ui / max(sps, 1)
    m = int(np.floor((t[-1] - t[0]) / dt))
    if m < 2:
        return sig
    tu = t[0] + np.arange(m) * dt
    return np.interp(tu, t, sig.real) + 1j * np.interp(tu, t, sig.imag)


def _name(m: int) -> str:
    return {4: "QPSK", 16: "16-QAM", 64: "64-QAM", 256: "256-QAM"}.get(
        m, f"{m}-QAM")


def _assemble_complex(result: dict, cfg: dict, log: list):
    """Build the complex baseband stream r = I + jQ from probe trace(s)."""
    traces = result.get("traces", [])

    def _find(name):
        return next((t for t in traces
                     if t["name"] == name or t.get("probe") == name), None)

    pi = cfg.get("probe_i")
    pq = cfg.get("probe_q")
    if pi and pq:
        ti, tq = _find(pi), _find(pq)
        if ti is None or tq is None:
            log.append(f"coherent report: I/Q probes {pi!r}/{pq!r} not found")
            return None
        i = np.asarray(ti["values"], float)
        q = np.asarray(tq["values"], float)
        n = min(len(i), len(q))
        return i[:n] + 1j * q[:n]
    # single complex-valued probe (optical field trace)
    probe = cfg.get("probe")
    tr = _find(probe) if probe else None
    if tr is None:
        log.append("coherent report: set probe_i/probe_q (balanced I/Q) or a "
                   "complex probe")
        return None
    v = tr.get("complex", tr["values"])
    return np.asarray(v, complex)
