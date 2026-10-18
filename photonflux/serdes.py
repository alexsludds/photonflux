"""Serial-link analysis around IBIS-AMI models: statistical and time domain.

This is the "EDA tool" side of IBIS-AMI. It runs a Tx model, a channel and
an Rx model the way the IBIS specification lays out the two flows:

**Statistical (Init) flow** -- :func:`statistical_link`
    channel impulse -> Tx ``AMI_Init`` -> Rx ``AMI_Init`` -> pulse response
    -> :func:`stat_eye`: the exact PDF of the inter-symbol interference
    (convolution over every cursor, equiprobable symbols), Gaussian noise,
    dual-Dirac + random jitter along the phase axis. Gives BER contours,
    timing and voltage bathtubs, and eye height / width at a target BER, for
    NRZ or PAM4.

**Time-domain (GetWave) flow** -- :func:`time_domain_link`
    PRBS symbols -> +/-0.5 V stimulus (optional Tx jitter) -> Tx
    ``AMI_GetWave`` -> channel (convolution, or any callable such as a
    circulax transient of a nonlinear optical link) -> noise -> Rx
    ``AMI_GetWave``. The Rx's recovered clock samples its own equalized
    output; decisions are aligned to the transmitted symbols and errors
    counted after ``Ignore_Bits``. Also returns the per-eye Gaussian (Q) fit,
    a time-domain bathtub from re-sampling across the UI, a CDR-referenced
    eye density, and the adaptation trace (every ``AMI_parameters_out``).

Channels (:class:`Channel`) come from a causal lossy-line model
(:func:`backplane`), a Touchstone S21, a measured pulse or step response
(:meth:`Channel.from_pulse`), or a photonflux circuit.

Conventions: impulse responses are in V/s sampled at ``ts`` (IBIS's
``impulse_matrix``); data symbols are a in {-1, +1} (NRZ) or
{-1, -1/3, 1/3, 1} (PAM4, Gray-coded) driving a +/-0.5 V stimulus, so a
pulse response's main cursor ``c0`` puts the outer data level at 0.5 c0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .signals import pam4_gray, prbs

__all__ = [
    "Channel", "backplane", "symbols", "stimulus", "stat_eye", "StatEye",
    "statistical_link", "StatLink", "time_domain_link", "receive",
    "estimate_channel", "TimeDomainLink", "rx_linear_response",
    "analyze_waveform", "WaveformAnalysis",
    "q_to_ber", "ber_to_q",
]

C0 = 299_792_458.0


def q_to_ber(q):
    from scipy.special import erfc

    return 0.5 * erfc(np.asarray(q) / np.sqrt(2.0))


def ber_to_q(ber):
    from scipy.special import erfcinv

    return np.sqrt(2.0) * erfcinv(2.0 * np.asarray(ber))


def _levels(L: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, L)


def _nlevels(modulation: str) -> int:
    m = str(modulation).upper()
    if m in ("NRZ", "PAM2"):
        return 2
    if m == "PAM4":
        return 4
    raise ValueError(f"modulation must be NRZ or PAM4, not {modulation!r}")


# ---------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """A linear channel as IBIS sees it: impulse response h (V/s) at ``ts``,
    plus optional crosstalk aggressor impulse responses."""

    impulse: np.ndarray
    ts: float
    aggressors: list[np.ndarray] = field(default_factory=list)
    name: str = "channel"

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.impulse)) * self.ts

    def pulse(self, ui: float) -> np.ndarray:
        """Response to a 1 V, 1 UI pulse."""
        n = max(1, int(round(ui / self.ts)))
        return np.convolve(self.impulse, np.ones(n))[: len(self.impulse)] * self.ts

    def resized(self, n: int) -> "Channel":
        def fit(h):
            out = np.zeros(n)
            out[: min(n, len(h))] = h[:n]
            return out
        return Channel(fit(self.impulse), self.ts, [fit(a) for a in self.aggressors],
                       self.name)

    def s21(self, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(f, H(f)) of the victim path."""
        nfft = nfft or len(self.impulse)
        H = np.fft.rfft(self.impulse * self.ts, nfft)
        return np.fft.rfftfreq(nfft, self.ts), H

    def with_fext(self, xt_db: float, f_ref: float, n: int = 1) -> "Channel":
        """Add ``n`` far-end crosstalk aggressors. FEXT is the time
        derivative of the through response (inductive/capacitive coupling
        imbalance), scaled so |S41| at ``f_ref`` is ``xt_db`` dB; each
        further aggressor is skewed by 3 samples and 10 % weaker."""
        f, H = self.s21()
        i = int(np.argmin(np.abs(f - f_ref)))
        k = 10 ** (xt_db / 20) / (2 * np.pi * f[i] * max(abs(H[i]), 1e-30))
        d = np.gradient(self.impulse, self.ts) * k
        aggs = [np.roll(d, 3 * j) * (1 - 0.1 * j) for j in range(n)]
        return Channel(self.impulse, self.ts, list(self.aggressors) + aggs,
                       self.name + f" + {n} FEXT @ {xt_db:g} dB")

    @classmethod
    def from_s21(cls, f: np.ndarray, s21: np.ndarray, ts: float, *,
                 n: int | None = None, pre_ui: float = 2.0, ui: float | None = None,
                 name: str = "s21") -> "Channel":
        """Causal impulse response from S21 samples (any frequency grid):
        resampled onto the FFT grid, rolled off above the data's last
        frequency, inverse-FFT'd, and shifted so the response starts
        ``pre_ui`` UI (or 16 samples) after t = 0."""
        f = np.asarray(f, float)
        s21 = np.asarray(s21, complex)
        n = n or 1 << 15
        fg = np.fft.rfftfreq(n, ts)
        mag = np.interp(fg, f, np.abs(s21), right=0.0)
        ph = np.interp(fg, f, np.unwrap(np.angle(s21)))
        H = mag * np.exp(1j * ph)
        if f[0] > 0:                       # extrapolate DC from the first point
            H[fg < f[0]] = np.abs(s21[0])
        fmax = min(f[-1], fg[-1])
        win = np.where(fg <= 0.8 * fmax, 1.0,
                       np.where(fg >= fmax, 0.0,
                                0.5 * (1 + np.cos(np.pi * (fg - 0.8 * fmax) / (0.2 * fmax)))))
        g = np.fft.irfft(H * win, n)
        i0 = int(np.argmax(np.abs(g) > 1e-3 * np.abs(g).max()))
        lead = int(round(pre_ui * ui / ts)) if ui else 16
        g = np.roll(g, lead - i0)
        return cls(g / ts, ts, name=name)

    @classmethod
    def from_pulse(cls, pulse: np.ndarray, ts: float, ui: float,
                   name: str = "measured") -> "Channel":
        """Impulse response from a measured 1-UI pulse response (e.g. a
        photonflux transient of the whole analog link with a single-UI
        input): the step response is s[n] = p[n] + s[n - nspui]."""
        nspui = max(1, int(round(ui / ts)))
        p = np.asarray(pulse, float)
        s = p.copy()
        for i in range(nspui, len(s)):
            s[i] += s[i - nspui]
        return cls(np.diff(s, prepend=0.0) / ts, ts, name=name)

    @classmethod
    def from_step(cls, step: np.ndarray, ts: float, name: str = "measured") -> "Channel":
        return cls(np.diff(np.asarray(step, float), prepend=0.0) / ts, ts, name=name)

    @classmethod
    def ideal(cls, ts: float, n: int = 4096, delay: int = 16) -> "Channel":
        h = np.zeros(n)
        h[delay] = 1.0 / ts
        return cls(h, ts, name="ideal")


def _dj_sarkar(f: np.ndarray, er: float, tand: float, f_ref: float = 10e9,
               f1: float = 1e3, f2: float = 1e13) -> np.ndarray:
    """Djordjevic-Sarkar wideband Debye permittivity (causal by
    construction), fitted to (er, tan d) at f_ref."""
    w, w1, w2, wr = (2 * np.pi * x for x in (f, f1, f2, f_ref))
    m = np.log10(w2) - np.log10(w1)
    lg = lambda x: np.log10((w2 + 1j * x) / (w1 + 1j * x))  # noqa: E731
    d_eps = -er * tand * m / lg(wr).imag
    e_inf = er - d_eps / m * lg(wr).real
    return e_inf + d_eps / m * lg(w)


def backplane(length_m: float = 0.5, *, ts: float, z0: float = 50.0,
              er: float = 3.7, tand: float = 0.008, skin_db_m_rtghz: float = 3.0,
              pkg_mm: float = 10.0, pkg_z0: float = 42.0, c_pad: float = 120e-15,
              via_stub_mm: float = 0.0, nfft: int = 1 << 16, ui: float | None = None,
              name: str | None = None) -> Channel:
    """A causal backplane / PCB channel, built as an ABCD cascade.

    ``pad C - package line - board stripline - [via stub] - package line -
    pad C``, terminated in ``z0`` both ends. The stripline loss is a
    sqrt(f) skin effect (with its internal-inductance phase) plus a
    Djordjevic-Sarkar dielectric (``er``, ``tand`` at 10 GHz); the package
    sections' impedance step and the pad capacitance put real reflections
    (ripple) into S21, and a via stub adds its quarter-wave notch.
    """
    f = np.fft.rfftfreq(nfft, ts)
    f[0] = 1.0                                     # avoid 0/0; DC fixed below
    w = 2 * np.pi * f
    eps = _dj_sarkar(f, er, tand)
    a_skin = skin_db_m_rtghz / 8.686 * np.sqrt(f / 1e9)   # Np/m
    gamma = 1j * w * np.sqrt(eps) / C0 + a_skin * (1 + 1j)

    def line(zc, length):
        gl = gamma * length
        return np.array([[np.cosh(gl), zc * np.sinh(gl)],
                         [np.sinh(gl) / zc, np.cosh(gl)]])

    def shunt(y):
        one, zero = np.ones_like(y), np.zeros_like(y)
        return np.array([[one, zero], [y, one]])

    def mul(a, b):
        return np.einsum("ijf,jkf->ikf", a, b)

    pad = shunt(1j * w * c_pad)
    pkg = line(pkg_z0, pkg_mm * 1e-3)
    board = line(z0, length_m)
    T = mul(mul(mul(pad, pkg), board), mul(pkg, pad))
    if via_stub_mm > 0:
        gl = gamma * via_stub_mm * 1e-3
        T = mul(mul(mul(pad, pkg), mul(line(z0, length_m / 2),
                                       shunt(np.tanh(gl) / z0))),
                mul(mul(line(z0, length_m / 2), pkg), pad))
    A, B, Cm, D = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
    s21 = 2.0 / (A + B / z0 + Cm * z0 + D)
    s21[0] = np.abs(s21[1])
    f[0] = 0.0
    nm = name or f"{length_m * 100:.0f} cm backplane"
    return Channel.from_s21(f, s21, ts, n=nfft, ui=ui, name=nm)


# ---------------------------------------------------------------------------
# stimulus
# ---------------------------------------------------------------------------

def symbols(n: int, modulation: str = "NRZ", *, order: int = 15,
            seed: int = 1) -> np.ndarray:
    """n PRBS data symbols in {-1, 1} (NRZ) or Gray-coded {-1, -1/3, 1/3, 1}
    (PAM4, two PRBS bits per symbol)."""
    L = _nlevels(modulation)
    nbits = n * (1 if L == 2 else 2)
    period = (1 << order) - 1
    bits = np.resize(prbs(order, min(nbits, period), seed=seed), nbits)
    if L == 2:
        return 2.0 * bits - 1.0
    return (2.0 * pam4_gray(bits).astype(float) - 3.0) / 3.0


def stimulus(a: np.ndarray, nspui: int, *, rj_ui: float = 0.0,
             dj_ui: float = 0.0, seed: int = 7) -> np.ndarray:
    """+/-0.5 V stimulus for symbols ``a`` at ``nspui`` samples/UI.

    Edges sit on UI boundaries, shifted by Gaussian random jitter (``rj_ui``
    rms) plus dual-Dirac deterministic jitter (+/- ``dj_ui``/2); a shifted
    edge is placed to sub-sample precision (linear ramp across one sample).
    """
    a = np.asarray(a, float)
    n = len(a) * nspui
    edges = np.arange(1, len(a)) * nspui
    amp = 0.5 * np.diff(a)
    if rj_ui or dj_ui:
        rng = np.random.default_rng(seed)
        sh = rj_ui * rng.standard_normal(len(edges))
        sh += np.where(rng.random(len(edges)) < 0.5, -0.5, 0.5) * dj_ui
        pos = edges + np.clip(sh, -0.45, 0.45) * nspui
    else:
        pos = edges.astype(float)
    i = np.floor(pos).astype(int)
    fr = pos - i
    d = np.zeros(n + 2)
    np.add.at(d, i, amp * (1.0 - fr))
    np.add.at(d, i + 1, amp * fr)
    return 0.5 * a[0] + np.cumsum(d)[:n]


# ---------------------------------------------------------------------------
# statistical eye
# ---------------------------------------------------------------------------

def _shift(pdf: np.ndarray, s: float) -> np.ndarray:
    """pdf shifted by s bins (fractional, linear split)."""
    i = int(np.floor(s))
    fr = s - i
    out = np.zeros_like(pdf)
    n = len(pdf)
    for k, wgt in ((i, 1.0 - fr), (i + 1, fr)):
        if wgt == 0.0 or abs(k) >= n:
            continue
        if k >= 0:
            out[k:] += wgt * pdf[: n - k]
        else:
            out[: n + k] += wgt * pdf[-k:]
    return out


@dataclass
class StatEye:
    """BER of every (sampling phase, threshold) point of a statistical eye.

    ``ber[e]`` is eye ``e``'s conditional error probability map (bottom eye
    first; one eye for NRZ, three for PAM4) on the grid ``v`` x ``phase_ui``;
    ``phase_ui`` is relative to the pulse peak.
    """

    phase_ui: np.ndarray
    v: np.ndarray
    ber: np.ndarray                 # (n_eyes, nv, nphase)
    levels: int
    level_v: np.ndarray             # (L, nphase) mean level of each symbol
    pulse: np.ndarray
    ts: float
    ui: float
    main_cursor: float
    noise_rms: float
    rj_ui: float
    dj_ui: float

    @property
    def n_eyes(self) -> int:
        return self.levels - 1

    @property
    def peak_phase_ui(self) -> float:
        """The pulse peak (phase 0 of this eye) on the absolute time axis, as
        a fraction of the UI."""
        return float(np.argmax(self.pulse) * self.ts / self.ui) % 1.0

    def relative_phase(self, abs_phase_ui: float) -> float:
        """An absolute sampling phase (e.g. a CDR's) on this eye's axis."""
        return float((abs_phase_ui - self.peak_phase_ui + 0.5) % 1.0 - 0.5)

    def combined(self) -> np.ndarray:
        """One map for plotting: each threshold row takes the eye whose
        level band it falls in."""
        out = np.full(self.ber.shape[1:], 0.5)
        for e in range(self.n_eyes):
            lo, hi = self.level_v[e], self.level_v[e + 1]
            band = (self.v[:, None] >= lo[None, :]) & (self.v[:, None] < hi[None, :])
            out = np.where(band, self.ber[e], out)
        return out

    def _center(self, e: int) -> tuple[int, int]:
        """(phase index, threshold index) of eye e's center: the phase with
        the tallest opening at a reference BER (1e-12, or a few decades
        above the floor of a closed eye) and the middle of that opening.
        A bare argmin would land anywhere on a wide-open eye's flat floor."""
        b = self.ber[e]
        floor = float(b.min())
        ref = 1e-12 if floor < 1e-14 else float(np.clip(floor * 100, 1e-12, 0.4))
        best = (-1, 0, 0)
        for ip in range(b.shape[1]):
            open_ = b[:, ip] < ref
            if not open_.any():
                continue
            # longest run of open thresholds at this phase
            edges = np.diff(np.concatenate(([0], open_.astype(int), [0])))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            k = int(np.argmax(ends - starts))
            h = int(ends[k] - starts[k])
            if h > best[0]:
                best = (h, ip, (starts[k] + ends[k] - 1) // 2)
        if best[0] < 0:
            iv, ip = np.unravel_index(np.argmin(b), b.shape)
            return int(ip), int(iv)
        return int(best[1]), int(best[2])

    def threshold(self, e: int = 0) -> float:
        return float(self.v[self._center(e)[1]])

    def best_phase_ui(self, e: int = 0) -> float:
        return float(self.phase_ui[self._center(e)[0]])

    def bathtub_t(self, e: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(phase_ui, BER) at eye e's optimal threshold."""
        _ip, iv = self._center(e)
        return self.phase_ui, self.ber[e][iv]

    def bathtub_v(self, e: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(threshold V, BER) at eye e's optimal phase."""
        ip, _iv = self._center(e)
        return self.v, self.ber[e][:, ip]

    @staticmethod
    def _span(mask: np.ndarray, idx: int) -> tuple[int, int] | None:
        if not mask[idx]:
            return None
        lo = idx
        while lo > 0 and mask[lo - 1]:
            lo -= 1
        hi = idx
        while hi < len(mask) - 1 and mask[hi + 1]:
            hi += 1
        return lo, hi

    def eye_height(self, ber: float = 1e-12, e: int = 0) -> float:
        """Vertical opening (V) at eye e's best phase where BER < ``ber``."""
        ip, iv = self._center(e)
        sp = self._span(self.ber[e][:, ip] < ber, iv)
        return 0.0 if sp is None else float(self.v[sp[1]] - self.v[sp[0]])

    def eye_width(self, ber: float = 1e-12, e: int = 0) -> float:
        """Horizontal opening (UI) at eye e's optimal threshold."""
        ip, iv = self._center(e)
        sp = self._span(self.ber[e][iv] < ber, ip)
        return 0.0 if sp is None else float(self.phase_ui[sp[1]] - self.phase_ui[sp[0]])

    def min_ber(self, e: int | None = None) -> float:
        if e is not None:
            return float(self.ber[e].min())
        return float(max(self.ber[k].min() for k in range(self.n_eyes)))

    def ser(self) -> float:
        """Symbol error rate at every eye's optimal (phase, threshold)."""
        return float(2.0 / self.levels * sum(self.ber[e].min()
                                             for e in range(self.n_eyes)))

    def bit_error_rate(self) -> float:
        """Gray-coded BER ~ SER / bits per symbol."""
        return self.ser() / np.log2(self.levels)

    def summary(self, targets: Sequence[float] = (1e-6, 1e-12)) -> dict:
        out = {"levels": self.levels, "main_cursor_v": self.main_cursor,
               "ber": self.bit_error_rate(), "ser": self.ser(),
               "noise_rms_v": self.noise_rms, "rj_ui": self.rj_ui,
               "dj_ui": self.dj_ui, "eyes": []}
        for e in range(self.n_eyes):
            d = {"threshold_v": self.threshold(e),
                 "best_phase_ui": self.best_phase_ui(e),
                 "min_ber": float(self.ber[e].min())}
            for tb in targets:
                d[f"height_v@{tb:g}"] = self.eye_height(tb, e)
                d[f"width_ui@{tb:g}"] = self.eye_width(tb, e)
            out["eyes"].append(d)
        return out


def stat_eye(pulse: np.ndarray, ts: float, ui: float, *, levels: int = 2,
             noise_rms: float = 0.0, rj_ui: float = 0.0, dj_ui: float = 0.0,
             nv: int = 601, span_ui: float = 1.0,
             crosstalk: Sequence[np.ndarray] = (),
             level_offsets: Sequence[float] | None = None) -> StatEye:
    """Statistical eye of a pulse response (V per 1 V, 1 UI pulse).

    For each sampling phase within +/- ``span_ui`` of the pulse peak, the
    ISI PDF is the exact convolution over every cursor of the equiprobable
    symbol distribution (cursors below a quarter bin go in as their
    Gaussian variance); ``noise_rms`` (V at the slicer) is Gaussian; random
    (``rj_ui`` rms) and dual-Dirac (``dj_ui`` pk-pk) sampling jitter are
    convolved along the phase axis afterwards. ``crosstalk`` pulses
    (aggressor responses at the victim's slicer, e.g. the aggressor rows of
    an Rx ``AMI_Init`` output) add every cursor of independent aggressor
    data, sampled at the victim's phase (synchronous crosstalk).
    ``level_offsets`` (V, lowest level first) shift each symbol's level --
    static nonlinear distortion such as modulator compression; thresholds
    are still searched freely, as adapted slicers would.
    """
    from scipy.special import ndtr

    p = 0.5 * np.asarray(pulse, float)     # +/-0.5 V stimulus
    nspui = max(1, int(round(ui / ts)))
    npk = int(np.argmax(p))
    lv = _levels(levels)
    off = np.zeros(levels) if level_offsets is None else np.asarray(level_offsets, float)
    halfw = int(round(span_ui * nspui))
    jit = int(np.ceil((4 * rj_ui + dj_ui / 2) * nspui)) + 1 if (rj_ui or dj_ui) else 0
    ph = np.arange(npk - halfw - jit, npk + halfw + jit + 1)
    ph = ph[(ph >= 0) & (ph < len(p))]
    # voltage span: worst-case peak distortion over the window
    pdmax = 0.0
    xt = [0.5 * np.asarray(x, float)[: len(p)] for x in crosstalk]
    for n in ph:
        c = p[n % nspui:: nspui]
        pdmax = max(pdmax, np.abs(c).sum()
                    + sum(np.abs(x[n % nspui:: nspui]).sum() for x in xt))
    vmax = 1.15 * pdmax + 5 * noise_rms + float(np.max(np.abs(off)))
    v = np.linspace(-vmax, vmax, nv)
    dv = v[1] - v[0]
    var_a = float(np.mean(lv ** 2))
    n_eyes = levels - 1
    ber = np.zeros((n_eyes, nv, len(ph)))
    level_v = np.zeros((levels, len(ph)))
    center = nv // 2
    for j, n in enumerate(ph):
        idx = np.arange(n % nspui, len(p), nspui)
        cur = p[idx]
        k0 = int(np.flatnonzero(idx == n)[0])
        c0 = cur[k0]
        isi = np.concatenate([np.delete(cur, k0)]
                             + [x[n % nspui:: nspui] for x in xt])
        small = np.abs(isi) < dv / 4
        var_small = var_a * float(np.sum(isi[small] ** 2))
        pdf = np.zeros(nv)
        pdf[center] = 1.0
        var_split = 0.0
        for c in isi[~small]:
            acc = np.zeros(nv)
            for a in lv:
                sh = a * c / dv
                acc += _shift(pdf, sh)
                fr = sh - np.floor(sh)
                var_split += fr * (1.0 - fr) * dv * dv / levels
            pdf = acc / levels
        pdf /= max(pdf.sum(), 1e-300)
        # Gaussian part (noise + negligible cursors) in closed form: each ISI
        # bin b contributes pdf[b] Phi(+-(v - mu - v_b) / sigma), evaluated
        # with ndtr on the tail side so deep tails keep full precision.
        # Splitting a fractional shift over two bins adds f(1-f) dv^2 of
        # variance per cursor; that much is taken back out of the Gaussian
        # (exact while the noise is the larger of the two).
        var_g = noise_rms ** 2 + var_small - var_split
        sigma = np.sqrt(max(var_g, (0.3 * dv) ** 2))
        nz = np.flatnonzero(pdf > 1e-300)
        wb, vb = pdf[nz], v[nz]
        below, above = [], []        # P(y < v), P(y > v) for each level
        for li, a in enumerate(lv):
            level_v[li, j] = a * c0 + off[li]
            z = (v[:, None] - level_v[li, j] - vb[None, :]) / sigma
            below.append(ndtr(z) @ wb)
            above.append(ndtr(-z) @ wb)
        for e in range(n_eyes):
            # P(lower symbol lands above v) and P(upper lands below v)
            ber[e, :, j] = 0.5 * (above[e] + below[e + 1])
    ber = np.clip(ber, 1e-300, 0.5)
    if rj_ui or dj_ui:
        sig = rj_ui * nspui
        half = int(np.ceil(5 * sig + dj_ui * nspui / 2)) + 1
        x = np.arange(-half, half + 1, dtype=float)
        kern = np.zeros_like(x)
        for off in ((-0.5, 0.5) if dj_ui else (0.0,)):
            mu = off * dj_ui * nspui
            if sig > 0.05:
                kern += np.exp(-0.5 * ((x - mu) / sig) ** 2)
            else:
                kern += np.interp(x, [mu - 1, mu, mu + 1], [0, 1, 0])
        kern /= kern.sum()
        for e in range(n_eyes):
            padded = np.pad(ber[e], ((0, 0), (half, half)), mode="edge")
            ber[e] = np.apply_along_axis(
                lambda r: np.convolve(r, kern, mode="valid"), 1, padded)
        keep = slice(jit, len(ph) - jit) if jit else slice(None)
        ber, level_v, ph = ber[:, :, keep], level_v[:, keep], ph[keep]
    return StatEye((ph - npk) / nspui, v, ber, levels, level_v,
                   np.asarray(pulse, float), ts, ui, float(p[npk]),
                   noise_rms, rj_ui, dj_ui)


# ---------------------------------------------------------------------------
# statistical (Init) flow
# ---------------------------------------------------------------------------

@dataclass
class StatLink:
    channel: Channel
    tx_impulse: np.ndarray          # after Tx Init (channel * Tx)
    rx_impulse: np.ndarray          # after Rx Init (the equalized link)
    pulse_unequalized: np.ndarray   # channel alone
    pulse: np.ndarray               # equalized
    eye: StatEye
    tx_params: dict
    rx_params: dict
    tx_message: str = ""
    rx_message: str = ""
    noise_rms: float = 0.0
    rx_linear: np.ndarray | None = None   # Rx front end, per-sample impulse
    rx_aggressors: list = field(default_factory=list)  # crosstalk at the slicer


def rx_linear_response(rx, h_in: np.ndarray, ts: float, ui: float, *,
                       aggressors: Sequence[np.ndarray] = (),
                       params: dict | None = None) -> np.ndarray:
    """The Rx's linear front end (everything but its decision feedback) as
    a per-sample impulse response, measured through ``AMI_Init`` itself.

    Two extra Inits add a +/-eps impulse to the channel in its quiet tail
    (past the DFE's reach); the difference of the outputs is the Rx's
    response to that impulse. Adaptation that keys on the main pulse (CTLE
    choice, AGC, DFE taps) is unchanged by the tiny probe, so this works for
    any dual or Init-only Rx. Call it *before* the Init whose state you keep.
    """
    h_in = np.asarray(h_in, float)
    nspui = max(1, int(round(ui / ts)))
    e = np.cumsum(h_in ** 2)
    end = int(np.searchsorted(e, 0.9999 * e[-1]))
    # the probe needs silence: past the channel's response (else it perturbs
    # an MMSE / zero-forcing design at first order) with room after it for
    # the Rx's own response; zero-pad a short channel to make that room
    pos = end + 24 * nspui
    n = max(len(h_in), pos + 128 * nspui)
    base = np.zeros(n)
    base[: len(h_in)] = h_in
    aggs = [np.pad(np.asarray(x, float), (0, max(0, n - len(x))))[:n]
            for x in aggressors]
    eps = 1e-3 * float(np.max(np.abs(h_in)))
    outs = []
    for sgn in (1.0, -1.0):
        h = base.copy()
        h[pos] += sgn * eps
        outs.append(rx.init(h, ts, ui, aggressors=aggs, params=params).impulse[0])
    return ((outs[0] - outs[1]) / (2 * eps))[pos:]


def _noise_at_slicer(g: np.ndarray, density: float, ts: float) -> float:
    """RMS of white input noise (``density`` V/rtHz, one-sided, to 1/2ts)
    through the per-sample impulse response g."""
    return float(np.sqrt(density ** 2 / (2 * ts) * np.sum(g ** 2)))


def statistical_link(tx, channel: Channel, rx, ui: float, *,
                     modulation: str = "NRZ", noise_rms: float | None = None,
                     noise_density: float = 0.0, rj_ui: float | None = None,
                     dj_ui: float | None = None, tx_params: dict | None = None,
                     rx_params: dict | None = None, nv: int = 601,
                     level_offsets: Sequence[float] | None = None) -> StatLink:
    """IBIS statistical flow: Tx ``AMI_Init`` then Rx ``AMI_Init`` on the
    channel impulse, then :func:`stat_eye` on the equalized pulse.

    Noise at the slicer is ``noise_rms`` (default: the Rx's ``Rx_Noise``)
    plus ``noise_density`` (V/rtHz at the Rx input) through the Rx's linear
    front end (:func:`rx_linear_response`). Jitter defaults to the models' ``Tx_Rj``/``Rx_Rj`` (rms) and
    ``Tx_Dj``/``Rx_Dj`` (pk-pk) reserved parameters (seconds -> UI).
    """
    ts = channel.ts
    L = _nlevels(modulation)
    h = channel.impulse
    aggs = channel.aggressors
    tx_po: dict = {}
    rx_po: dict = {}
    tx_msg = rx_msg = ""
    if tx is not None:
        if "Modulation" in tx.params:
            tx.params["Modulation"] = modulation.upper()
        r = tx.init(h, ts, ui, aggressors=aggs, params=tx_params)
        h_tx = r.impulse[0]
        aggs = list(r.impulse[1:])
        tx_po, tx_msg = r.params_out, r.message
    else:
        h_tx = h
    g_rx = None
    if rx is not None:
        if "Modulation" in rx.params:
            rx.params["Modulation"] = modulation.upper()
        if noise_density > 0:
            g_rx = rx_linear_response(rx, h_tx, ts, ui, aggressors=aggs,
                                      params=rx_params)
        r = rx.init(h_tx, ts, ui, aggressors=aggs, params=rx_params)
        h_rx = r.impulse[0]
        rx_po, rx_msg = r.params_out, r.message
    else:
        h_rx = h_tx
    nspui = max(1, int(round(ui / ts)))

    def pulse(x):
        return np.convolve(x, np.ones(nspui))[: len(x)] * ts

    def info(model, name):
        return float(model.params.info(name, 0.0) or 0.0) if model is not None else 0.0

    if noise_rms is None:
        noise_rms = info(rx, "Rx_Noise")
    if noise_density > 0 and g_rx is None:     # no Rx: noise straight through
        g_rx = np.array([1.0])
    sig_in = _noise_at_slicer(g_rx, noise_density, ts) if noise_density > 0 else 0.0
    sig_n = float(np.hypot(noise_rms, sig_in))
    if rj_ui is None:
        rj_ui = float(np.hypot(info(tx, "Tx_Rj"), info(rx, "Rx_Rj"))) / ui
    if dj_ui is None:
        dj_ui = (info(tx, "Tx_Dj") + info(rx, "Rx_Dj")) / ui
    p = pulse(h_rx)
    rx_aggs = list(r.impulse[1:]) if rx is not None else list(aggs)
    xt = [pulse(x) for x in rx_aggs]
    eye = stat_eye(p, ts, ui, levels=L, noise_rms=sig_n, rj_ui=rj_ui,
                   dj_ui=dj_ui, nv=nv, crosstalk=xt, level_offsets=level_offsets)
    return StatLink(channel, h_tx, h_rx, pulse(h), p, eye, tx_po, rx_po,
                    tx_msg, rx_msg, sig_n, g_rx, rx_aggs)


# ---------------------------------------------------------------------------
# time-domain (GetWave) flow
# ---------------------------------------------------------------------------

@dataclass
class TimeDomainLink:
    ui: float
    ts: float
    levels: int
    symbols: np.ndarray             # transmitted, in {-1..1}
    tx_wave: np.ndarray             # Tx output (channel input)
    rx_in: np.ndarray               # channel output + noise (Rx input)
    rx_out: np.ndarray              # Rx GetWave output (slicer input)
    clock_times: np.ndarray         # Rx clock ticks (sampling - UI/2)
    samples: np.ndarray             # slicer input at each recovered clock
    lag: int                        # samples[k] <-> symbols[k - lag]
    skip: int                       # leading decisions excluded (Ignore_Bits)
    thresholds: np.ndarray
    counted: dict
    qfit: dict
    bathtub: dict
    tx_trace: Any = None            # GetWaveResult of the Tx
    rx_trace: Any = None            # GetWaveResult of the Rx (adaptation)
    rx_init: Any = None             # Rx InitResult

    def eye_density(self, nt: int | None = None, nv: int = 160, span_ui: float = 2.0,
                    vlim: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2-D histogram of the Rx output referenced to the recovered clock
        (after ``skip``): (t_ui edges, v edges, counts[nv, nt]). The default
        time bin is one sample, so the bins never alias against the grid."""
        if nt is None:
            nt = int(round(span_ui * self.ui / self.ts))
        t = np.arange(len(self.rx_out)) * self.ts
        centers = self.clock_times[self.skip:] + 0.5 * self.ui
        if len(centers) < 2:
            return np.zeros(nt + 1), np.zeros(nv + 1), np.zeros((nv, nt))
        sel = (t >= centers[0]) & (t <= centers[-1])
        t, val = t[sel], self.rx_out[sel]
        k = np.clip(np.searchsorted(centers, t), 1, len(centers) - 1)
        nearest = np.where(np.abs(t - centers[k - 1]) < np.abs(t - centers[k]),
                           centers[k - 1], centers[k])
        rel = (t - nearest) / self.ui          # in [-0.5, 0.5)
        # a span_ui-wide eye: each sample also shows one UI over
        reps = [rel]
        for sh in range(1, int(np.ceil(span_ui / 2 + 0.5))):
            reps += [rel + sh, rel - sh]
        rel = np.concatenate(reps)
        val = np.tile(val, len(reps))
        vl = vlim or 1.2 * np.max(np.abs(val))
        H, te, ve = np.histogram2d(rel, val, bins=(nt, nv),
                                   range=((-span_ui / 2, span_ui / 2), (-vl, vl)))
        return te, ve, H.T

    def adaptation(self, key: str, index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(time in UI, value) of one Rx adaptation variable. Phases in UI
        (keys ending in ``phase_ui``) are unwrapped."""
        tr = self.rx_trace
        y = tr.series(key, index)
        if key.endswith("phase_ui") and np.isfinite(y).all():
            y = np.unwrap(2 * np.pi * y) / (2 * np.pi)
        return tr.trace_t / self.ui, y

    @property
    def sampling_phase_ui(self) -> float:
        """Where the recovered clock samples, as a fraction of the UI on the
        absolute time axis (circular mean after ``skip``) -- the same axis
        as :attr:`StatEye.peak_phase_ui`."""
        ph = 2 * np.pi * ((self.clock_times[self.skip:] + 0.5 * self.ui) / self.ui)
        return float(np.angle(np.mean(np.exp(1j * ph))) / (2 * np.pi)) % 1.0


def _kmeans_levels(x: np.ndarray, L: int) -> np.ndarray:
    lv = np.quantile(x, (np.arange(L) + 0.5) / L)
    for _ in range(50):
        idx = np.argmin(np.abs(x[:, None] - lv[None, :]), axis=1)
        new = np.array([x[idx == k].mean() if np.any(idx == k) else lv[k]
                        for k in range(L)])
        if np.allclose(new, lv):
            break
        lv = np.sort(new)
    return lv


def _slice(x: np.ndarray, thr: np.ndarray, L: int) -> np.ndarray:
    return _levels(L)[np.searchsorted(thr, x)]


def _gray_bits(a: np.ndarray, L: int) -> np.ndarray:
    if L == 2:
        return (a > 0).astype(int)[:, None]
    s = np.rint((a + 1.0) * 1.5).astype(int)          # 0..3 by amplitude
    gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    return gray[s]


def _qfit(x: np.ndarray, a: np.ndarray, L: int) -> dict:
    lv = _levels(L)
    mus, sds = [], []
    for k in lv:
        s = x[np.isclose(a, k)]
        if len(s) < 4:
            return {"ok": False}
        mus.append(float(s.mean()))
        sds.append(float(s.std() + 1e-30))
    q = [(mus[e + 1] - mus[e]) / (sds[e] + sds[e + 1]) for e in range(L - 1)]
    eye_ber = [float(q_to_ber(max(qq, 0.0))) for qq in q]
    ser = 2.0 / L * sum(eye_ber)
    return {"ok": True, "levels": mus, "sigmas": sds, "q": q,
            "ber_est": ser / np.log2(L), "ser_est": ser}


def time_domain_link(tx, channel: Channel | None, rx, ui: float, *,
                     modulation: str = "NRZ", nsym: int = 20000, order: int = 15,
                     seed: int = 1, noise_rms: float = 0.0, rj_ui: float = 0.0,
                     dj_ui: float = 0.0, ts: float | None = None,
                     channel_fn: Callable[[np.ndarray], np.ndarray] | None = None,
                     init_impulse: np.ndarray | None = None,
                     block_bits: int = 256, skip: int | None = None,
                     max_lag: int = 400, tx_params: dict | None = None,
                     rx_params: dict | None = None) -> TimeDomainLink:
    """IBIS time-domain flow with a recovered clock and counted errors.

    The channel is ``channel`` (convolution) or ``channel_fn(wave) -> wave``
    on the same sample grid (e.g. a circulax transient of a nonlinear
    optical link); then ``init_impulse`` (or ``channel.impulse``) is what the
    models' ``AMI_Init`` sees -- a dual model's GetWave starts from that
    design. ``noise_rms`` is white Gaussian noise (V rms) added at the Rx
    input. Errors are counted after ``skip`` decisions (default: the Rx's
    ``Ignore_Bits``).
    """
    if channel is not None:
        ts = channel.ts
    if ts is None:
        raise ValueError("give a channel or ts")
    from scipy.signal import fftconvolve

    nspui = max(1, int(round(ui / ts)))
    a = symbols(nsym, modulation, order=order, seed=seed)
    x = stimulus(a, nspui, rj_ui=rj_ui, dj_ui=dj_ui, seed=seed + 100)
    h_init = init_impulse if init_impulse is not None else (
        channel.impulse if channel is not None else Channel.ideal(ts).impulse)

    tx_tr = None
    aggs = list(channel.aggressors) if channel is not None and channel_fn is None else []
    if tx is not None:
        if "Modulation" in tx.params:
            tx.params["Modulation"] = modulation.upper()
        r = tx.init(h_init, ts, ui, aggressors=aggs, params=tx_params)
        h_rx_init = r.impulse[0]
        aggs = list(r.impulse[1:])          # aggressor paths with the Tx FFE
        if tx.getwave_exists:
            tx_tr = tx.getwave(x, block_bits=block_bits)
            tx_out = tx_tr.wave
        else:   # Init-only Tx: its impulse is Init(delta)
            d = Channel.ideal(ts, n=64 * nspui, delay=0).impulse
            h_tx = tx.init(d, ts, ui).impulse[0]
            tx_out = np.convolve(x, h_tx)[: len(x)] * ts
    else:
        h_rx_init = h_init
        tx_out = x
    if channel_fn is not None:
        rx_in = np.asarray(channel_fn(tx_out), float)
    elif channel is not None:
        rx_in = fftconvolve(tx_out, channel.impulse)[: len(tx_out)] * ts
    else:
        rx_in = tx_out.copy()
    for j, hx in enumerate(aggs):           # crosstalk: independent data
        xa = stimulus(symbols(nsym, modulation, order=order, seed=seed + 31 * (j + 1)),
                      nspui)
        rx_in = rx_in + fftconvolve(xa, hx)[: len(rx_in)] * ts
    if noise_rms > 0:
        rx_in = rx_in + noise_rms * np.random.default_rng(seed + 200).standard_normal(len(rx_in))

    out = receive(rx, rx_in, a, ui, ts, h_rx_init, modulation=modulation,
                  block_bits=block_bits, skip=skip, max_lag=max_lag,
                  rx_params=rx_params)
    out.tx_wave = tx_out
    out.tx_trace = tx_tr
    return out


def receive(rx, rx_in: np.ndarray, a: np.ndarray, ui: float, ts: float,
            init_impulse: np.ndarray | None = None, *, modulation: str = "NRZ",
            block_bits: int = 256, skip: int | None = None, max_lag: int = 400,
            rx_params: dict | None = None) -> TimeDomainLink:
    """The Rx half of the time-domain flow, on any received waveform.

    ``rx_in`` is the Rx input on a uniform ``ts`` grid starting with symbol
    0 of ``a`` (e.g. a photonflux circuit transient, resampled); the Rx's
    ``AMI_Init`` sees ``init_impulse`` (e.g. :func:`estimate_channel` of the
    same waveform) and then ``AMI_GetWave`` recovers its clock, equalizes
    and adapts. Decisions are aligned to ``a`` and counted after ``skip``
    (default: the Rx's ``Ignore_Bits``).
    """
    L = _nlevels(modulation)
    a = np.asarray(a, float)
    rx_in = np.asarray(rx_in, float)
    nspui = max(1, int(round(ui / ts)))
    if init_impulse is None:
        init_impulse = Channel.ideal(ts, n=64 * nspui).impulse
    rx_tr = rx_init = None
    if rx is not None:
        if "Modulation" in rx.params:
            rx.params["Modulation"] = modulation.upper()
        rx_init = rx.init(init_impulse, ts, ui, params=rx_params)
        rx_tr = rx.getwave(rx_in, block_bits=block_bits)
        rx_out = rx_tr.wave
        clk = rx_tr.clock_times
        if skip is None:
            skip = rx.ignore_bits
    else:
        rx_out = rx_in
        clk = np.zeros(0)
    if len(clk) < 16:
        # no recovered clock (an Init-only or CDR-less Rx): an ideal clock at
        # the output eye's center, found data-aided -- the phase whose
        # samples track the known symbols best (correlation over lags)
        ph = _best_phase(rx_out, a, nspui, skip or 0, max_lag)
        clk = (np.arange(ph, len(rx_out) - nspui, nspui) - nspui / 2) * ts
    skip = int(skip or 0)
    tgrid = np.arange(len(rx_out)) * ts
    samples = np.interp(clk + ui / 2, tgrid, rx_out)
    skip = min(skip, max(0, len(samples) - 64))
    lv = _kmeans_levels(samples[skip:], L)
    thr = 0.5 * (lv[1:] + lv[:-1])
    dec = _slice(samples, thr, L)

    # align: decision k <-> symbol k - lag (the clock count starts at the
    # first tick, the symbol stream at t = 0)
    best = (0, -np.inf)
    seg = slice(skip, min(len(dec), skip + 4000))
    for lag in range(0, min(max_lag, len(a) - 1)):
        dd = dec[seg]
        idx = np.arange(seg.start, seg.start + len(dd)) - lag
        ok = (idx >= 0) & (idx < len(a))
        if ok.sum() < 64:
            continue
        c = float(np.mean(dd[ok] * a[idx[ok]]))
        if c > best[1]:
            best = (lag, c)
    lag = best[0]
    k = np.arange(skip, len(dec))
    idx = k - lag
    ok = (idx >= 0) & (idx < len(a))
    d_ok, a_ok = dec[k[ok]], a[idx[ok]]
    s_ok = samples[k[ok]]
    sym_err = int(np.sum(~np.isclose(d_ok, a_ok)))
    bit_err = int(np.sum(_gray_bits(d_ok, L) != _gray_bits(a_ok, L)))
    nbits = len(d_ok) * (1 if L == 2 else 2)
    counted = {"symbols": int(len(d_ok)), "symbol_errors": sym_err,
               "ser": sym_err / max(len(d_ok), 1), "bits": int(nbits),
               "bit_errors": bit_err, "ber": bit_err / max(nbits, 1)}
    qfit = _qfit(s_ok, a_ok, L)

    # time-domain bathtub: re-sample the slicer input across the UI
    offs = np.linspace(-0.5, 0.5, 33)
    tub_ber = []
    for o in offs:
        sm = np.interp(clk + ui / 2 + o * ui, tgrid, rx_out)[k[ok]]
        q = _qfit(sm, a_ok, L)
        tub_ber.append(q["ber_est"] if q["ok"] else 0.5)
    bathtub = {"offset_ui": offs, "ber": np.clip(np.asarray(tub_ber), 1e-300, 0.5)}
    return TimeDomainLink(ui, ts, L, a, rx_in.copy(), rx_in, rx_out, clk,
                          samples, lag, skip, thr, counted, qfit, bathtub, None,
                          rx_tr, rx_init)


def _best_phase(w: np.ndarray, a: np.ndarray, nspui: int, skip: int,
                max_lag: int) -> int:
    """Sample phase of ``w`` whose symbol-rate samples correlate best with
    the known symbols ``a`` (searching integer lags)."""
    n = min(len(w) // nspui, len(a))
    start = min(skip, max(0, n - 600))
    m = min(3000, n - start - max_lag - 1)
    if m < 64:
        return nspui // 2
    best = (-np.inf, nspui // 2)
    for ph in range(nspui):
        s = w[ph::nspui][:n]
        s = s - s.mean()
        for lag in range(0, max_lag):
            c = float(np.dot(s[start + lag:start + lag + m], a[start:start + m]))
            if c > best[0]:
                best = (c, ph)
    return best[1]


def estimate_channel(wave: np.ndarray, ts: float, ui: float, a: np.ndarray, *,
                     span_ui: int = 48, pre_ui: int = 2, skip: int = 16,
                     name: str = "estimated", residual: bool = False):
    """Data-aided channel estimate from a waveform and the symbols that
    made it: per sample phase, least squares for
    ``y[k nspui + ph] = 0.5 * sum_j a[k - j] p[ph + j nspui] + c``.

    Returns (Channel, rms residual), plus the residual waveform (zero
    outside the fitted span) with ``residual=True``. The pulse response p is
    in the IBIS
    stimulus convention (response to a 1 V, 1 UI pulse driving symbols of
    +/-0.5 V), so the impulse feeds ``AMI_Init`` directly. For a nonlinear
    channel (a square-law photodiode after dispersion, a compressing
    modulator) it is the best linear fit, and the residual says how far off
    linear the link is -- the error IBIS-AMI's LTI statistical flow makes.
    The first ``pre_ui`` UI of the window allow a little anticausal latency
    (pattern-source edge placement) and are dropped from the result.
    """
    y = np.asarray(wave, float)
    a = np.asarray(a, float)
    nspui = max(1, int(round(ui / ts)))
    nsym = min(len(a), len(y) // nspui)
    J = span_ui + pre_ui
    rows = np.arange(skip + J, nsym - pre_ui)
    A = np.empty((len(rows), J + 1))
    for j in range(J):
        A[:, j] = 0.5 * a[rows - j + pre_ui]
    A[:, J] = 1.0
    p = np.zeros(J * nspui)
    res_w = np.zeros(len(y))
    res2, cnt = 0.0, 0
    for ph in range(nspui):
        yy = y[rows * nspui + ph]
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        p[ph::nspui] = coef[:J]
        r = yy - A @ coef
        res_w[rows * nspui + ph] = r
        res2 += float(r @ r)
        cnt += len(r)
    # sample n of p is t = n ts - pre_ui UI: keep t >= 0 (the anticausal
    # window only soaks up edge-placement latency)
    ch = Channel.from_pulse(p[pre_ui * nspui:], ts, ui, name=name)
    rms = float(np.sqrt(res2 / max(cnt, 1)))
    return (ch, rms, res_w) if residual else (ch, rms)


# ---------------------------------------------------------------------------
# a received waveform, end to end
# ---------------------------------------------------------------------------

@dataclass
class WaveformAnalysis:
    """IBIS-AMI analysis of one received waveform (:func:`analyze_waveform`)."""

    channel: Channel                # data-aided linear estimate (Init's view)
    residual_rms: float             # what the linear fit misses, V at Rx input
    residual_rel: float             # ... relative to the waveform's rms
    distortion_at_slicer: float     # its random part through the Rx front end
    level_offsets: np.ndarray       # its static part: shift of each level, V
    stat: StatLink                  # Init flow, distortion counted as noise
    stat_lti: StatLink              # Init flow, the link taken as LTI
    td: TimeDomainLink              # GetWave flow on the waveform itself


def _split_distortion(d: np.ndarray, live: np.ndarray, a: np.ndarray,
                      skip: int, max_lag: int = 96) -> tuple[float, np.ndarray]:
    """Split distortion samples ``d`` (one per UI, at the slicer's phase)
    into a static level shift -- the mean per current symbol, e.g. a
    modulator compressing the outer PAM4 levels -- and the rest. Returns
    (rms of the rest, offset per level, lowest level first). The symbol each
    sample belongs to is the lag that explains the most variance."""
    lv = np.unique(a)
    zero = np.zeros(len(lv))
    n = min(len(d), len(a))
    idx = np.arange(skip, n)
    idx = idx[live[skip:n]]
    if len(idx) < 64:
        return 0.0, zero
    x = d[idx]
    best = (0.0, None)
    for lag in range(0, max_lag):
        k = idx - lag
        ok = k >= 0
        if ok.sum() < 64:
            break
        means = np.array([x[ok][np.isclose(a[k[ok]], v)].mean() for v in lv])
        fit = means[np.searchsorted(lv, a[k[ok]])]
        explained = float(np.var(fit))
        if explained > best[0]:
            best = (explained, (ok, fit, means))
    if best[1] is None:
        return float(np.std(x)), zero
    ok, fit, means = best[1]
    return float(np.std(x[ok] - fit)), means


def analyze_waveform(rx, wave: np.ndarray, a: np.ndarray, ui: float, ts: float, *,
                     modulation: str = "NRZ", noise_rms: float = 0.0,
                     skip: int | None = None, span_ui: int = 64, seed: int = 7,
                     nv: int = 301) -> WaveformAnalysis:
    """Run an IBIS-AMI Rx on a received waveform whose symbols are known --
    e.g. a photonflux circuit transient of a nonlinear optical link.

    1. :func:`estimate_channel` gives the LTI channel ``AMI_Init`` needs,
       and the residual it cannot explain (square-law detection after
       dispersion, modulator compression, ...).
    2. Time domain: :func:`receive` on ``wave`` + white input noise.
    3. Statistical: the Init flow on the estimate, once as pure LTI and
       once with the residual -- filtered through the Rx's linear front
       end (:func:`rx_linear_response`) and read at the recovered clock's
       phase -- split into its per-symbol mean, which moves the eye's
       levels (static distortion: modulator compression), and the rest,
       added as Gaussian noise (a Bussgang-style split of a nonlinear link
       into a linear part plus uncorrelated distortion).
    """
    y = np.asarray(wave, float)
    y = y - np.mean(y)
    nspui = max(1, int(round(ui / ts)))
    nsym = min(len(a), len(y) // nspui)
    ch, resid, res_w = estimate_channel(y, ts, ui, a[:nsym],
                                        span_ui=min(span_ui, nsym // 8),
                                        residual=True)
    if skip is None and rx is not None:
        skip = min(rx.ignore_bits, nsym // 3)
    rx_in = y + (noise_rms * np.random.default_rng(seed).standard_normal(len(y))
                 if noise_rms > 0 else 0.0)
    td = receive(rx, rx_in, a[:nsym], ui, ts, ch.impulse, modulation=modulation,
                 skip=skip)
    g = rx_linear_response(rx, ch.impulse, ts, ui) if rx is not None else np.array([1.0])
    dist = np.convolve(res_w, g)[: len(res_w)]
    ph = int(round(td.sampling_phase_ui * nspui)) % nspui
    sig_dist, offsets = _split_distortion(dist[ph::nspui], res_w[ph::nspui] != 0,
                                          a[:nsym], td.skip)
    dens = noise_rms * np.sqrt(2 * ts)
    lti = statistical_link(None, ch, rx, ui, modulation=modulation,
                           noise_density=dens, nv=nv)
    st = statistical_link(None, ch, rx, ui, modulation=modulation,
                          noise_rms=sig_dist, noise_density=dens, nv=nv,
                          level_offsets=offsets)
    return WaveformAnalysis(ch, resid, resid / max(float(np.std(y)), 1e-30),
                            sig_dist, offsets, st, lti, td)


# ---------------------------------------------------------------------------
# plotting (matplotlib axes in, nothing global)
# ---------------------------------------------------------------------------

def plot_stat_eye(ax, eye: StatEye, *, contours: Sequence[float] = (1e-3, 1e-6, 1e-9, 1e-12),
                  cmap: str = "magma", title: str | None = None) -> None:
    import matplotlib.colors as mcolors

    z = np.log10(np.clip(eye.combined(), 1e-30, 0.5))
    m = ax.pcolormesh(eye.phase_ui, eye.v * 1e3, z, cmap=cmap, shading="auto",
                      norm=mcolors.Normalize(vmin=-16, vmax=np.log10(0.5)))
    cs = ax.contour(eye.phase_ui, eye.v * 1e3, z, levels=np.log10(sorted(contours)),
                    colors="w", linewidths=0.8)
    ax.clabel(cs, fmt=lambda x: f"1e{x:.0f}", fontsize=7)
    lim = 1.6 * float(np.max(np.abs(eye.level_v))) * 1e3
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("sampling phase [UI]")
    ax.set_ylabel("threshold [mV]")
    if title:
        ax.set_title(title)
    return m


def plot_bathtubs(ax_t, ax_v, eye: StatEye, *, label: str = "") -> None:
    for e in range(eye.n_eyes):
        tag = f"{label} eye {e}" if eye.n_eyes > 1 else label
        x, b = eye.bathtub_t(e)
        ax_t.semilogy(x, b, label=tag or None)
        x, b = eye.bathtub_v(e)
        ax_v.semilogy(x * 1e3, b, label=tag or None)
    lim = 1.4 * float(np.max(np.abs(eye.level_v))) * 1e3
    ax_v.set_xlim(-lim, lim)
    for ax in (ax_t, ax_v):
        ax.set_ylim(1e-16, 1)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("BER")
    ax_t.set_xlabel("sampling phase [UI]")
    ax_v.set_xlabel("threshold [mV]")


def plot_eye_density(ax, td: TimeDomainLink, *, cmap: str = "inferno", title: str | None = None) -> None:
    te, ve, H = td.eye_density()
    ax.pcolormesh(te, ve * 1e3, np.log10(H + 1), cmap=cmap, shading="auto")
    for t in td.thresholds:
        ax.axhline(t * 1e3, color="c", lw=0.6, ls="--")
    ax.set_xlabel("time from recovered clock [UI]")
    ax.set_ylabel("Rx output [mV]")
    if title:
        ax.set_title(title)
