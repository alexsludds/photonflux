"""Waveform builders for the pattern (PRBS) and PWL sources.

The sources are compiled as closures over precomputed (t, v) breakpoint
arrays — a transient-friendly piecewise-linear waveform evaluated with
``jnp.interp`` (clamped at both ends). Everything here is plain numpy at
netlist-build time; nothing is traced.

PRBS conventions follow the user's serdes codebase (signaling.py):
  * LFSR tap pairs {7:[7,6], 9:[9,5], 11:[11,9], 15:[15,14], 23:[23,18],
    31:[31,28]}, x^k feedback, seed != 0.
  * PAM4 is Gray-coded: bit pairs 00,01,11,10 -> symbol levels 0,1,2,3.
  * TX FFE is a symbol-spaced 3-tap FIR [pre, main, post]; pre/post are
    given in dB as in TxFFE.from_emphasis (0 dB = tap off).
  * Jitter offsets per UI boundary: RJ (Gaussian, sigma in UI), PJ
    (sinusoidal, amplitude in UI), DCD (alternating +-d/2).

RLM predistortion: with ``rlm_vpi`` > 0 the level ladder is warped so the
*optical* eye of a quadrature-biased MZM (T = 0.5 + 0.5*sin(pi*v/vpi)) has
equally spaced levels: v_k = (vpi/pi) * asin(2*T_k - 1) with T_k linearly
spaced between the transmissions reached at v0/v1.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Feedback taps as exponents of G(x), excluding the x^0 term. Most orders take
# a primitive trinomial, but degree 13 has none — PRBS-13 is the four-term
# IEEE 802.3 polynomial (the pattern 802.3bs/cd specify for TDECQ), which is
# why the tap values are tuples of arbitrary length rather than pairs.
PRBS_TAPS = {7: (7, 6), 9: (9, 5), 11: (11, 9),
             13: (13, 12, 2, 1),          # x^13 + x^12 + x^2 + x + 1
             15: (15, 14), 23: (23, 18), 31: (31, 28)}

EDGE_PTS = 9      # samples across one raised-cosine edge


def prbs_bits(order: int, n: int, seed: int = 1) -> np.ndarray:
    """First n bits of PRBS-<order> from the standard LFSR (seed != 0)."""
    if order not in PRBS_TAPS:
        raise ValueError(f"PRBS order must be one of {sorted(PRBS_TAPS)}")
    taps = PRBS_TAPS[order]
    state = int(seed) & ((1 << order) - 1) or 1
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        fb = 0
        for t in taps:
            fb ^= state >> (order - t)
        out[i] = state & 1
        state = (state >> 1) | ((fb & 1) << (order - 1))
    return out


def _symbols(settings: dict, nsym: int) -> np.ndarray:
    """Symbol fractions in [0, 1]: NRZ 0/1, PAM4 Gray {0,1/3,2/3,1}."""
    mode = str(settings.get("mode", "nrz"))
    order = int(settings.get("order", 7))
    seed = int(settings.get("seed", 1))
    if mode == "pulse":
        f = np.zeros(nsym)
        f[min(4, nsym - 1)] = 1.0        # one-UI pulse after a short runway
        return f
    if mode == "pam4":
        bits = prbs_bits(order, 2 * nsym, seed)
        gray = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        sym = np.array([gray[(int(bits[2 * i]), int(bits[2 * i + 1]))]
                        for i in range(nsym)], dtype=float)
        return sym / 3.0
    return prbs_bits(order, nsym, seed).astype(float)   # nrz


def _levels(settings: dict, frac: np.ndarray) -> np.ndarray:
    """Fractions -> drive levels: RLM warp, then linear map v0..v1."""
    v0 = float(settings.get("v0", -0.5))
    v1 = float(settings.get("v1", 0.5))
    rlm_vpi = float(settings.get("rlm_vpi", 0.0))
    if rlm_vpi > 0.0:
        # equal *optical* spacing on T(v) = 0.5 + 0.5 sin(pi v / vpi):
        # ladder endpoints keep their drive levels, inner levels are warped
        # through the exact inverse. Requires |v0|,|v1| <= vpi/2.
        def trans(v: float) -> float:
            v = float(np.clip(v, -rlm_vpi / 2, rlm_vpi / 2))
            return 0.5 + 0.5 * np.sin(np.pi * v / rlm_vpi)

        tk = trans(v0) + frac * (trans(v1) - trans(v0))
        return (rlm_vpi / np.pi) * np.arcsin(np.clip(2.0 * tk - 1.0, -1, 1))
    return v0 + frac * (v1 - v0)


def _tx_ffe(settings: dict, levels: np.ndarray) -> np.ndarray:
    """Symbol-spaced [pre, main, post] FIR, dB knobs like TxFFE."""
    pre_db = float(settings.get("ffe_pre_db", 0.0))
    post_db = float(settings.get("ffe_post_db", 0.0))
    if pre_db <= 0.0 and post_db <= 0.0:
        return levels
    pre = -(10.0 ** (-pre_db / 20.0)) if pre_db > 0 else 0.0
    post = -(10.0 ** (-post_db / 20.0)) if post_db > 0 else 0.0
    # peak-constrained normalisation: transitions reach full swing, runs of
    # identical symbols sit de-emphasised — the standard TX FFE behaviour
    taps = np.array([pre, 1.0, post])
    taps = taps / np.abs(taps).sum()
    mid = 0.5 * (levels.min() + levels.max())
    x = levels - mid           # pre-emphasise around the ladder midpoint
    # y[k] = pre*x[k+1] + main*x[k] + post*x[k-1]
    y = np.convolve(x, taps, mode="full")[1:1 + len(x)]
    return y + mid


def _edge_offsets(settings: dict, nsym: int, ui: float) -> np.ndarray:
    """Per-boundary time offsets [s]: RJ + PJ + DCD (in UI units)."""
    rj = float(settings.get("rj_ui", 0.0))
    pj = float(settings.get("pj_ui", 0.0))
    pj_f = float(settings.get("pj_freq", 10e6))
    dcd = float(settings.get("dcd_ui", 0.0))
    if rj == 0.0 and pj == 0.0 and dcd == 0.0:
        return np.zeros(nsym + 1)
    rng = np.random.default_rng(int(settings.get("seed", 1)) + 12345)
    k = np.arange(nsym + 1, dtype=float)
    off = np.zeros(nsym + 1)
    if rj > 0:
        off += rj * rng.standard_normal(nsym + 1)
    if pj > 0:
        off += pj * np.sin(2 * np.pi * pj_f * k * ui)
    if dcd > 0:
        off += np.where(k % 2 == 0, +dcd / 2, -dcd / 2)
    return np.clip(off, -0.4, 0.4) * ui


def prbs_waveform(settings: dict, span: float) -> tuple[np.ndarray, np.ndarray]:
    """(t, v) breakpoints for the pattern source covering [0, span]."""
    ui = float(settings.get("ui", 100e-12))
    tr = min(float(settings.get("tr", 20e-12)), 0.6 * ui)
    nsym = max(4, min(int(np.ceil(span / ui)) + 2, 1_000_000))
    frac = _symbols(settings, nsym)
    lv = _tx_ffe(settings, _levels(settings, frac))
    off = _edge_offsets(settings, nsym, ui)

    # raised-cosine edge template centred on each symbol boundary
    s = np.linspace(0.0, 1.0, EDGE_PTS)
    shape = 0.5 * (1.0 - np.cos(np.pi * s))
    ts: list[float] = [0.0]
    vs: list[float] = [float(lv[0])]
    for k in range(1, nsym):
        a, b = float(lv[k - 1]), float(lv[k])
        t_edge = k * ui + off[k] - tr / 2
        if b != a:
            ts.extend(t_edge + s * tr)
            vs.extend(a + (b - a) * shape)
        else:
            ts.append(t_edge + tr / 2)
            vs.append(b)
    ts.append(nsym * ui + 10 * ui)
    vs.append(float(lv[-1]))
    t = np.asarray(ts)
    # jitter can (rarely) reorder points; enforce strict monotonicity
    t = np.maximum.accumulate(t + np.arange(len(t)) * 1e-18)
    return t, np.asarray(vs)


def pwl_waveform(settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Parse the PWL source's `data` text: 't v' or 't,v' per line/pair."""
    raw = str(settings.get("data", "") or "")
    toks = [x for x in raw.replace(",", " ").split() if x]
    if len(toks) < 4 or len(toks) % 2:
        raise ValueError(
            "PWL source needs at least two 't v' breakpoint pairs "
            "(one per line, comma or space separated)")
    vals = np.asarray([float(x) for x in toks], dtype=float)
    t, v = vals[0::2], vals[1::2]
    if np.any(np.diff(t) <= 0):
        raise ValueError("PWL breakpoint times must be strictly increasing")
    return t, v


# ---------------------------------------------------------------------------
# QAM symbol maps + RRC pulse shaping (coherent transceiver, ALE-77)
#
# Square Gray-coded QAM: the transmitted symbol integer's bits split into an
# I half (high bits) and a Q half (low bits); each half Gray-codes onto an
# odd-integer PAM ladder so nearest-neighbour points differ by exactly one
# bit. The constellation is normalised to unit average power. QPSK is the
# M = 4 case. These regenerate the exact TX symbols for the data-aided
# coherent DSP (webapp/coherent.py), the same way `_symbols` feeds linkpost.
# ---------------------------------------------------------------------------

QAM_ORDERS = {"qpsk": 4, "qam16": 16, "qam64": 64}


def qam_order(settings: dict | str | int) -> int:
    """Constellation size M from a source settings dict, mode name, or int."""
    if isinstance(settings, dict):
        mode = str(settings.get("qam", settings.get("mode", "qpsk")))
    else:
        mode = settings
    if isinstance(mode, (int, float)):
        m = int(mode)
    else:
        m = QAM_ORDERS.get(str(mode).lower(), 0)
    if m not in (4, 16, 64, 256):
        raise ValueError(f"QAM order must be one of {sorted(QAM_ORDERS)} "
                         "(qpsk/qam16/qam64)")
    return m


def qam_bits_per_symbol(m: int) -> int:
    return int(round(np.log2(m)))


def _gray(i: int) -> int:
    return i ^ (i >> 1)


def qam_constellation(m: int) -> np.ndarray:
    """Unit-average-power square QAM points indexed by transmitted symbol.

    `const[s]` is the complex point for symbol integer `s in [0, m)`; its bits
    (MSB-first, I half then Q half) place it on the Gray-coded PAM grid.
    """
    L = int(round(np.sqrt(m)))
    if L * L != m:
        raise ValueError("only square QAM (m = 4, 16, 64, 256) is supported")
    mbits = qam_bits_per_symbol(L * L) // 2      # bits per axis
    pts = np.zeros(m, dtype=complex)
    for iI in range(L):
        for iQ in range(L):
            sym = (_gray(iI) << mbits) | _gray(iQ)
            pts[sym] = (2 * iI - (L - 1)) + 1j * (2 * iQ - (L - 1))
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))    # unit average power
    return pts


def qam_symbols(settings: dict, nsym: int) -> np.ndarray:
    """Complex, unit-power QAM symbols from the source PRBS (data-aided TX)."""
    m = qam_order(settings)
    order = int(settings.get("order", 15))
    seed = int(settings.get("seed", 1))
    bps = qam_bits_per_symbol(m)
    bits = prbs_bits(order, bps * nsym, seed).astype(int)
    const = qam_constellation(m)
    idx = np.zeros(nsym, dtype=int)
    for b in range(bps):
        idx = (idx << 1) | bits[b::bps][:nsym]
    return const[idx]


def qam_drive_waveform(settings: dict,
                       span: float) -> tuple[np.ndarray, np.ndarray]:
    """(t, v) drive for one QAM rail (I or Q) — RRC-shaped, scaled to v0..v1.

    Feeds the IQ modulator's I or Q electrode: the ``qam_drive`` setting picks
    the real or imaginary part of the RRC-pulse-shaped QAM symbol stream,
    mapped to [v0, v1] around their midpoint (full-scale symbol = the rail
    peak). ``sps`` samples per UI; ``rrc_beta`` roll-off.
    """
    ui = float(settings.get("ui", 100e-12))
    sps = max(int(settings.get("sps", 16)), 2)
    beta = float(settings.get("rrc_beta", 0.1))
    v0 = float(settings.get("v0", -0.5))
    v1 = float(settings.get("v1", 0.5))
    rail = str(settings.get("qam_drive", "i")).lower()
    m = qam_order(settings)
    nsym = max(8, min(int(np.ceil(span / ui)) + 2, 2_000_000))
    syms = qam_symbols(settings, nsym)
    comp = syms.real if rail == "i" else syms.imag
    # normalise so the outermost rail level hits full swing (v0..v1)
    peak = float(np.max(np.abs(qam_constellation(m).real))) or 1.0
    frac = comp / peak                                   # in [-1, 1]
    mid = 0.5 * (v0 + v1)
    amp = 0.5 * (v1 - v0)
    up = np.zeros(nsym * sps)
    up[::sps] = frac
    h = rrc_taps(beta, sps, 12)
    shaped = np.convolve(up, h * np.sqrt(sps), mode="same")
    v = mid + amp * shaped
    t = np.arange(len(v)) * (ui / sps)
    t = np.append(t, t[-1] + 10 * ui)
    v = np.append(v, mid)
    return t, v


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine FIR (unit-energy), span symbols each side, sps/UI."""
    beta = float(np.clip(beta, 1e-6, 1.0))
    n = np.arange(-span * sps, span * sps + 1, dtype=float)
    t = n / sps                                  # time in symbols
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def wave_key(kind: str, t: np.ndarray, v: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(t.tobytes())
    h.update(v.tobytes())
    return f"{kind}:{h.hexdigest()[:16]}"


def noise_bank(inst: str, seeds: int, span: float, bw: float,
               master_seed: int = 1) -> tuple[np.ndarray, float]:
    """(bank, dt_n): unit-variance Gaussian samples, shape (seeds, N).

    Each instance gets an independent, reproducible stream (seeded from the
    master seed + a stable hash of the instance name). Samples are spaced
    dt_n = 1/(2*bw) and linearly interpolated by the components, so the
    realised noise is white to ~bw with unit variance; scale by
    sqrt(PSD/(2*dt_n)) for a target one-sided density.
    """
    dt_n = 1.0 / (2.0 * max(bw, 1e6))
    n = max(int(np.ceil(span / dt_n)) + 4, 16)
    salt = int(hashlib.sha256(inst.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng((int(master_seed) * 100003 + salt) % 2**63)
    return rng.standard_normal((max(int(seeds), 1), n)), dt_n


def span_bucket(t_stop: float) -> float:
    """Round the needed span up to a power-of-2 ns so small t_stop tweaks
    reuse the compiled circuit (the waveform is baked into the model)."""
    ns = max(t_stop, 1e-9) / 1e-9
    return (2.0 ** int(np.ceil(np.log2(ns)))) * 1e-9
