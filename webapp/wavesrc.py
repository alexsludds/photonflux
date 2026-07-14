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

PRBS_TAPS = {7: (7, 6), 9: (9, 5), 11: (11, 9), 15: (15, 14),
             23: (23, 18), 31: (31, 28)}

EDGE_PTS = 9      # samples across one raised-cosine edge


def prbs_bits(order: int, n: int, seed: int = 1) -> np.ndarray:
    """First n bits of PRBS-<order> from the standard LFSR (seed != 0)."""
    if order not in PRBS_TAPS:
        raise ValueError(f"PRBS order must be one of {sorted(PRBS_TAPS)}")
    t1, t2 = PRBS_TAPS[order]
    state = int(seed) & ((1 << order) - 1) or 1
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        fb = ((state >> (order - t1)) ^ (state >> (order - t2))) & 1
        out[i] = state & 1
        state = (state >> 1) | (fb << (order - 1))
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
