"""Frequency-domain RF networks: ABCD cascades for the analog channel.

An IBIS-AMI channel is the passive network between the Tx buffer's
Thevenin source and the receiver's termination: die pads, flip-chip bumps,
package traces, BGA balls and vias, board traces and connectors. Every
impedance step reflects, so the channel is the *terminated* transfer
``V_load / V_source`` of the whole cascade, not a product of per-section
S21s. :class:`Net` builds that cascade as 2x2 ABCD matrices over frequency
and returns it, or its S-parameters, input impedance and causal impulse
response.

Lines are TEM RLGC lines (:func:`line_rlgc`): a sqrt(f) conductor loss with
its internal inductance (R = X), and a Djordjevic-Sarkar dielectric whose
complex permittivity sets both C and G, so loss, dispersion and the
frequency-dependent characteristic impedance stay causal.

    from photonflux import rfnet
    f = rfnet.grid(ts, n)
    pcb = rfnet.line_rlgc(f, z0=50, er=3.6, tand=0.004, cond_db_m_rtghz=7)
    net = (rfnet.Net.shunt_c(f, 150e-15) @ rfnet.Net.line(pcb, 0.18)
           @ rfnet.Net.shunt_c(f, 100e-15))
    H = net.transfer(zs=50, zl=50)            # V_load / V_source
    h = rfnet.impulse(f, H, ts, n)            # V/s, for serdes.Channel
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Line", "Net", "line_rlgc", "grid", "impulse", "length_for_loss"]

C0 = 299_792_458.0
NP_DB = 20.0 / np.log(10.0)          # dB per neper


def grid(ts: float, n: int) -> np.ndarray:
    """The rfft frequency grid of an ``n``-sample record at ``ts``."""
    return np.fft.rfftfreq(n, ts)


def _f_safe(f: np.ndarray) -> np.ndarray:
    """Frequencies with DC nudged to 1 Hz (every formula below is regular
    there; the DC limit is reached to within ppm)."""
    return np.maximum(np.asarray(f, float), 1.0)


def _dj_sarkar(f: np.ndarray, er: float, tand: float, f_ref: float = 10e9,
               f1: float = 1e3, f2: float = 1e13) -> np.ndarray:
    """Djordjevic-Sarkar wideband Debye permittivity (causal), fitted to
    (er, tan d) at f_ref; complex, eps' - j eps''."""
    w, w1, w2, wr = (2 * np.pi * x for x in (f, f1, f2, f_ref))
    m = np.log10(w2) - np.log10(w1)

    def lg(x):
        return np.log10((w2 + 1j * x) / (w1 + 1j * x))
    d_eps = -er * tand * m / lg(wr).imag
    e_inf = er - d_eps / m * lg(wr).real
    return e_inf + d_eps / m * lg(w)


@dataclass
class Line:
    """A transmission line's per-unit-length propagation and impedance."""

    f: np.ndarray
    gamma: np.ndarray      # 1/m, alpha + j beta
    zc: np.ndarray         # Ohm, characteristic impedance

    def delay_s_m(self, f0: float) -> float:
        """Group delay per metre at ``f0``."""
        w = 2 * np.pi * _f_safe(self.f)
        k = int(np.clip(np.argmin(np.abs(self.f - f0)), 1, len(w) - 2))
        return float((self.gamma.imag[k + 1] - self.gamma.imag[k - 1])
                     / (w[k + 1] - w[k - 1]))


def line_rlgc(f: np.ndarray, *, z0: float, er: float, tand: float = 0.0,
              cond_db_m_rtghz: float = 0.0, r_dc: float = 0.0,
              f_ref: float = 10e9) -> Line:
    """TEM line: ``z0`` and ``er`` at ``f_ref`` set L and C; conductor loss
    ``cond_db_m_rtghz`` dB/m at 1 GHz scaling as sqrt(f), with an equal
    internal reactance (R = X, the skin effect); dielectric loss from a
    Djordjevic-Sarkar permittivity with ``tand`` at ``f_ref``. Z0 and the
    loss are what a matched 2D solver would report for the cross-section.
    ``r_dc`` (Ohm/m, signal + return) floors the resistance where the skin
    depth exceeds the conductor: R = sqrt(r_dc^2 + R_skin^2)."""
    fs = _f_safe(f)
    w = 2 * np.pi * fs
    eps = _dj_sarkar(fs, er, tand, f_ref) if tand > 0 else np.full(fs.shape, er + 0j)
    eps_ref = (_dj_sarkar(np.array([f_ref]), er, tand, f_ref)[0].real
               if tand > 0 else er)
    l_ext = z0 * np.sqrt(eps_ref) / C0
    c_vac = 1.0 / (z0 * np.sqrt(eps_ref) * C0)     # C per unit permittivity
    r1 = 2.0 * z0 * cond_db_m_rtghz / NP_DB                  # Ohm/m at 1 GHz
    rs = r1 * np.sqrt(fs / 1e9)
    z = np.hypot(r_dc, rs) + 1j * rs + 1j * w * l_ext
    y = 1j * w * c_vac * eps            # G = w C eps'' from the complex eps
    gamma = np.sqrt(z * y)
    # the forward wave: alpha >= 0, and beta > 0 where the line is lossless
    flip = (gamma.real < 0) | ((gamma.real <= 1e-12 * np.abs(gamma)) & (gamma.imag < 0))
    gamma = np.where(flip, -gamma, gamma)
    return Line(np.asarray(f, float), gamma, np.sqrt(z / y))


@dataclass
class Net:
    """A two-port as its ABCD matrix over frequency, shape (2, 2, nf).
    ``a @ b`` cascades (a's port 2 into b's port 1)."""

    f: np.ndarray
    abcd: np.ndarray

    @classmethod
    def _from(cls, f, a, b, c, d) -> "Net":
        f = np.asarray(f, float)
        shp = f.shape
        m = np.empty((2, 2) + shp, complex)
        m[0, 0], m[0, 1] = np.broadcast_to(a, shp), np.broadcast_to(b, shp)
        m[1, 0], m[1, 1] = np.broadcast_to(c, shp), np.broadcast_to(d, shp)
        return cls(f, m)

    @classmethod
    def series(cls, f, z) -> "Net":
        return cls._from(f, 1.0, z, 0.0, 1.0)

    @classmethod
    def shunt(cls, f, y) -> "Net":
        return cls._from(f, 1.0, 0.0, y, 1.0)

    @classmethod
    def series_l(cls, f, l: float, r: float = 0.0) -> "Net":
        return cls.series(f, r + 2j * np.pi * _f_safe(f) * l)

    @classmethod
    def shunt_c(cls, f, c: float) -> "Net":
        return cls.shunt(f, 2j * np.pi * _f_safe(f) * c)

    @classmethod
    def line(cls, ln: Line, length: float) -> "Net":
        gl = ln.gamma * length
        ch, sh = np.cosh(gl), np.sinh(gl)
        return cls._from(ln.f, ch, ln.zc * sh, sh / ln.zc, ch)

    @classmethod
    def open_stub(cls, ln: Line, length: float) -> "Net":
        """A shunt open-circuited stub (e.g. an un-backdrilled via barrel)."""
        return cls.shunt(ln.f, np.tanh(ln.gamma * length) / ln.zc)

    def __matmul__(self, other: "Net") -> "Net":
        return Net(self.f, np.einsum("ijf,jkf->ikf", self.abcd, other.abcd))

    def flipped(self) -> "Net":
        """The same network driven from port 2."""
        (a, b), (c, d) = self.abcd
        det = a * d - b * c
        return Net(self.f, np.array([[d, b], [c, a]]) / det)

    def transfer(self, zs, zl) -> np.ndarray:
        """V(load) / V(source EMF) with source impedance ``zs`` and load
        ``zl`` (scalars or arrays over f): every re-reflection included."""
        (a, b), (c, d) = self.abcd
        return zl / (a * zl + b + c * zs * zl + d * zs)

    def zin(self, zl) -> np.ndarray:
        """Impedance looking into port 1 with port 2 loaded by ``zl``."""
        (a, b), (c, d) = self.abcd
        return (a * zl + b) / (c * zl + d)

    def s(self, z0: float = 50.0) -> np.ndarray:
        """S-parameters (2, 2, nf) referenced to ``z0`` at both ports."""
        (a, b), (c, d) = self.abcd
        den = a + b / z0 + c * z0 + d
        return np.array([[(a + b / z0 - c * z0 - d) / den,
                          2 * (a * d - b * c) / den],
                         [2 / den, (-a + b / z0 - c * z0 + d) / den]])


def impulse(f: np.ndarray, H: np.ndarray, ts: float, n: int,
            f_roll: float | None = None) -> np.ndarray:
    """Causal impulse response (V/s, ``n`` samples at ``ts``) of a transfer
    sampled on :func:`grid`. Above ``f_roll`` (default: a quarter of the
    sample rate) H is rolled off with a raised cosine over one octave, where
    lumped pad / bump models stop meaning anything anyway."""
    f = np.asarray(f, float)
    if len(f) != n // 2 + 1:
        raise ValueError("H must be sampled on grid(ts, n)")
    f_roll = f_roll or 0.25 / ts
    x = np.clip((f - f_roll) / f_roll, 0.0, 1.0)
    win = 0.5 * (1 + np.cos(np.pi * x))
    return np.fft.irfft(H * win, n) / ts


def length_for_loss(ln: Line, loss_db: float, f0: float, z0: float = 50.0) -> float:
    """Length at which a line alone, between ``z0`` ports, has insertion
    loss ``loss_db`` at ``f0`` (bisection on |S21|, so the small mismatch of
    its frequency-dependent Zc is counted)."""
    k = int(np.argmin(np.abs(ln.f - f0)))
    sub = Line(ln.f[k:k + 1], ln.gamma[k:k + 1], ln.zc[k:k + 1])

    def il(length):
        return -20 * np.log10(abs(Net.line(sub, length).s(z0)[1, 0][0]))
    lo, hi = 0.0, 1.0
    while il(hi) < loss_db:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if il(mid) < loss_db else (lo, mid)
    return 0.5 * (lo + hi)


def pi_coil_pad(f: np.ndarray, c: float, z: float = 50.0) -> Net:
    """A die pad + ESD capacitance ``c`` distributed in a two-section
    pi-coil ladder (C/4, L/2, C/2, L/2, C/4 with L = z^2 c): an artificial
    line of impedance ``z``, the usual way a 100+ GHz-fT I/O hides its
    capacitance (a bridged T-coil does the same in one stage)."""
    half = Net.series_l(f, z * z * c / 2)
    return (Net.shunt_c(f, c / 4) @ half @ Net.shunt_c(f, c / 2) @ half
            @ Net.shunt_c(f, c / 4))


def host_module_link(f: np.ndarray, *, loss_db: float = 10.0, f_nyq: float = 26.5625e9,
                     c_src: float = 150e-15, c_load: float = 100e-15,
                     host_to_module: bool = True) -> dict:
    """A host-board link between a switch / SerDes package and a
    near-package optical module, in the odd mode of a 100 Ohm pair (50 Ohm
    single-ended):

    die pad (pi-coil, ``c_src``) - bump - 10 mm 45 Ohm package trace - BGA
    ball - backdrilled via - PCB stripline sized for ``loss_db`` at
    ``f_nyq`` (3.6 / 0.004 dielectric, 7 dB/m/rtGHz skin, 8 Ohm/m DC) - via
    - LGA interposer - 5 mm module substrate - bump - die pad (``c_load``).

    ``host_to_module=False`` runs the same hardware the other way (module
    TIA to host SerDes). Returns ``net`` plus ``trace_m`` and the ``pcb``
    line."""
    pcb = line_rlgc(f, z0=50.0, er=3.6, tand=0.004, cond_db_m_rtghz=7.0, r_dc=8.0)
    pkg = line_rlgc(f, z0=45.0, er=3.4, tand=0.010, cond_db_m_rtghz=14.0)
    sub = line_rlgc(f, z0=48.0, er=3.4, tand=0.010, cond_db_m_rtghz=14.0)
    lga = line_rlgc(f, z0=58.0, er=1.0)
    via_ln = line_rlgc(f, z0=45.0, er=3.6, tand=0.004)
    via = Net.line(via_ln, 2e-3) @ Net.open_stub(via_ln, 0.25e-3)
    # size the trace at f_nyq itself, whatever grid the caller passed
    pcb_nyq = line_rlgc(np.array([f_nyq]), z0=50.0, er=3.6, tand=0.004,
                        cond_db_m_rtghz=7.0, r_dc=8.0)
    trace_m = length_for_loss(pcb_nyq, loss_db, f_nyq)
    host = (Net.series_l(f, 30e-12) @ Net.line(pkg, 10e-3)          # bump, package
            @ Net.shunt_c(f, 70e-15) @ Net.series_l(f, 120e-12)     # BGA ball
            @ via)
    module = (via @ Net.line(lga, 1.2e-3) @ Net.shunt_c(f, 40e-15)  # LGA
              @ Net.line(sub, 5e-3) @ Net.series_l(f, 30e-12))      # substrate, bump
    a, b = (host, module) if host_to_module else (module.flipped(), host.flipped())
    net = (pi_coil_pad(f, c_src) @ a @ Net.line(pcb, trace_m) @ b
           @ pi_coil_pad(f, c_load))
    return {"net": net, "trace_m": trace_m, "pcb": pcb}
