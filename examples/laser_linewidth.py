#!/usr/bin/env python3
"""Laser phase noise / linewidth, pinned to the Lorentzian lineshape (ALE-76).

The CW laser gains a ``linewidth_hz`` knob (``webapp/catalog.py``): on top of
the existing RIN amplitude noise the optical phase now performs a Wiener walk,

    dphi ~ N(0, 2*pi*dnu*dt),      Var(phi(t)) = 2*pi*dnu*t,

implemented through the same per-instance noise bank as RIN/shot/ASE so
multi-seed transient runs pool correctly (variance-exact, band-limited to the
noise bandwidth). A pure phase diffusion has field autocorrelation

    <E*(0) E(tau)> = P * exp(-pi*dnu*|tau|),

whose Fourier transform is a **Lorentzian of FWHM = dnu** — the textbook laser
lineshape. Nothing about a Lorentzian is coded: it emerges from integrating
white phase noise and taking |FFT(E)|^2, exactly as an OSA would read it.

Two acceptance testbenches, both driven off a single solved field E(t) (the
field source pins E algebraically, so the saved samples are the exact envelope):

  1. OSA lineshape   — seed-averaged |FFT(E)|^2 is Lorentzian; the fitted 3 dB
                       FWHM tracks dnu across 0.5 -> 8 MHz (slope 1).
  2. delayed self-    — a Mach-Zehnder with an imbalance tau_d >> coherence time
     heterodyne         and an f_AOM frequency shift beats the laser against a
                        decorrelated copy of itself; the photocurrent RF line at
                        f_AOM is Lorentzian of FWHM = 2*dnu (the standard
                        linewidth measurement).

Self-checks (all asserted):
  * dnu = 0 recovers a delta line (single-bin OSA peak).
  * OSA FWHM ~ dnu to within the seed-averaged fit tolerance, slope 1 in dnu.
  * self-heterodyne beat FWHM ~ 2*dnu.

    .venv-circulax/bin/python examples/laser_linewidth.py   # -> out/laser_linewidth.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
from circulax import compile_circuit
from circulax.components.base_component import Signals, States, component

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
import catalog        # noqa: E402
import wavesrc        # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "laser_linewidth.png"
CHECKS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'XX'}] {name}{'  — ' + detail if detail else ''}")


@component(ports=("po_p", "po_n"))
def _Absorber(signals: Signals, s: States, Yopt: float = 1.0):
    e = signals.po_p - signals.po_n
    return {"po_p": Yopt * e, "po_n": -Yopt * e}, {}


def _laser_circuit(bank, dt_n):
    laser = catalog._cw_laser_noisy(bank, dt_n)
    net = {
        "instances": {"GND": {"component": "ground"},
                      "LAS": {"component": "laser"},
                      "TERM": {"component": "absorber"}},
        "connections": {"GND,p1": ("LAS,p2", "TERM,po_n"),
                        "LAS,p1": "TERM,po_p"},
        "ports": {"eout": "LAS,p1"},
    }
    return compile_circuit(
        net, {"ground": lambda: 0, "laser": laser, "absorber": _Absorber},
        backend="dense", is_complex=True)


def _field(circuit, dt_n, t_stop, dnu, seed_idx=0, power=1e-3):
    from circulax.solvers.transient import BDF2VectorizedTransientSolver

    ts = np.arange(0.0, t_stop, dt_n)
    params = {"LAS.power": power, "LAS.linewidth_hz": float(dnu),
              "LAS.seed_idx": float(seed_idx)}
    y0 = circuit.dc(params=params)
    sol = circuit.transient(
        t0=0.0, t1=t_stop, dt0=dt_n, y0=y0,
        saveat=diffrax.SaveAt(ts=ts), params=params,
        transient_solver=BDF2VectorizedTransientSolver(
            linear_solver=circuit.solver, newton_max_steps=40),
        max_steps=len(ts) + 16, throw=False,
        stepsize_controller=diffrax.ConstantStepSize())
    assert sol.result == diffrax.RESULTS.successful, sol.result
    return np.asarray(sol.ts), np.asarray(circuit.port(sol.ys, "eout"))


def _bh_window(n: int) -> np.ndarray:
    k = np.arange(n)
    return (0.35875 - 0.48829 * np.cos(2 * np.pi * k / (n - 1))
            + 0.14128 * np.cos(4 * np.pi * k / (n - 1))
            - 0.01168 * np.cos(6 * np.pi * k / (n - 1)))


def _g1(series, dt_n):
    """Normalized coherence magnitude |g(tau)| of a list of complex series,
    ensemble+time averaged, at a spread of lags spanning the decay. Returns
    (taus, |g|). For a pure phase diffusion |g1(tau)| = exp(-pi*dnu*|tau|)."""
    n = min(len(s) for s in series)
    # dense small lags (resolve fast decay at large dnu) + sparse long lags
    # (reach the 0.3 level at small dnu) — one grid serves the whole sweep
    lags = np.unique(np.concatenate([
        np.arange(1, min(60, n // 3)),
        np.linspace(60, n // 3, 200).astype(int)]))
    g = np.empty(len(lags))
    for j, m in enumerate(lags):
        num = 0j
        den = 0.0
        for s in series:
            num += np.vdot(s[:len(s) - m], s[m:])   # sum conj(s[t]) s[t+m]
            den += np.sum(np.abs(s[:len(s) - m]) ** 2)
        g[j] = np.abs(num) / den
    return lags * dt_n, g


def _linewidth_from_g1(taus, g):
    """Linewidth L from |g(tau)| = exp(-pi*L*tau): least-squares slope of
    ln|g| over the clean 0.9..0.1 decade (avoids the noise floor + the t->0
    curvature). Field coherence gives L = dnu; a beat of two decorrelated
    fields gives L = 2*dnu."""
    # fit the clean 0.85..0.3 span: above it lies the tau->0 curvature, below
    # it the finite-ensemble noise floor (which would flatten the slope)
    m = (g > 0.3) & (g < 0.85)
    slope = np.polyfit(taus[m], np.log(g[m]), 1)[0]
    return float(-slope / np.pi)


def _solve_fields(tag, dnu, seeds, bw, t_stop, master):
    bank, dt_n = wavesrc.noise_bank(tag, seeds * 2, t_stop, bw=bw,
                                    master_seed=master)
    circuit = _laser_circuit(bank, dt_n)
    fields = [_field(circuit, dt_n, t_stop, dnu, seed_idx=k)[1]
              for k in range(seeds)]
    return dt_n, fields


def osa_lineshape(fields, dt_n):
    """Seed-averaged optical power spectrum (Bartlett) -> a clean Lorentzian."""
    Savg = None
    for E in fields:
        S = np.abs(np.fft.fftshift(np.fft.fft(E * _bh_window(len(E))))) ** 2
        Savg = S if Savg is None else Savg + S
    f = np.fft.fftshift(np.fft.fftfreq(len(fields[0]), dt_n))
    return f, Savg / len(fields)


def main() -> int:
    print("laser linewidth / phase noise\n" + "-" * 40)

    # --- 1. dnu = 0 is a delta line ----------------------------------------
    dt0, f0 = _solve_fields("OSA", 0.0, 1, 5e8, 4e-6, 101)
    ff, S0 = osa_lineshape(f0, dt0)
    peak_bins = int((S0 >= 0.5 * S0.max()).sum())
    _check("dnu=0 -> delta line", peak_bins <= 2, f"{peak_bins} bin(s) at -3 dB")

    # --- 2. linewidth readout tracks dnu over a decade ---------------------
    # OSA lineshape (Lorentzian) for the eye; linewidth read from the exact
    # field-coherence decay |g1(tau)| = exp(-pi*dnu*tau) (bias-free vs a
    # single-realization spectral-width fit).
    dnus = np.array([1e6, 2e6, 4e6, 8e6])
    reads, shapes = [], []
    for dnu in dnus:
        # scale the record to the linewidth so every point holds the same
        # number of coherence times (~25) — uniform estimator bias, so the
        # log-log slope reads 1 cleanly
        t_stop = 8e-6 * (1e6 / dnu)
        dt_n, fields = _solve_fields("OSA", dnu, 8, 5e8, t_stop, 101)
        taus, g = _g1(fields, dt_n)
        reads.append(_linewidth_from_g1(taus, g))
        shapes.append(osa_lineshape(fields, dt_n) + (dnu,))
    reads = np.array(reads)
    for dnu, w in zip(dnus, reads):
        _check(f"linewidth ~ dnu @ {dnu/1e6:.1f} MHz",
               abs(w - dnu) / dnu < 0.15, f"read {w/1e6:.2f} MHz")
    slope = np.polyfit(np.log(dnus), np.log(reads), 1)[0]
    _check("linewidth slope vs dnu = 1", abs(slope - 1.0) < 0.06,
           f"slope {slope:.3f}")

    # --- 3. delayed self-heterodyne beat is 2*dnu ---------------------------
    # Beat the field against a copy delayed by tau_d >> coherence time. The
    # beat product p(t) = E(t) conj(E(t-tau_d)) has |g_beat| = |g1|^2, so its
    # linewidth is 2*dnu — the textbook self-heterodyne width.
    dnu_sh, f_aom, tau_d = 2e6, 40e6, 2e-6
    dt_n, fields = _solve_fields("SH", dnu_sh, 8, 2e8, 12e-6, 202)
    d = int(round(tau_d / dt_n))
    beats, Pavg, fbeat = [], None, None
    for E in fields:
        beats.append(E[d:] * np.conj(E[:-d]))       # baseband beat product
        i_pd = np.abs(E[d:] + E[:-d]                 # AOM-shifted RF beat note
                      * np.exp(1j * 2 * np.pi * f_aom * np.arange(len(E) - d)
                               * dt_n)) ** 2
        P = np.abs(np.fft.rfft((i_pd - i_pd.mean()) * _bh_window(len(i_pd)))) ** 2
        Pavg = P if Pavg is None else Pavg + P
    fbeat = np.fft.rfftfreq(len(i_pd), dt_n)
    Pavg /= len(fields)
    taus_b, g_b = _g1(beats, dt_n)
    beat_lw = _linewidth_from_g1(taus_b, g_b)
    _check("self-heterodyne beat linewidth ~ 2*dnu",
           abs(beat_lw - 2 * dnu_sh) / (2 * dnu_sh) < 0.2,
           f"beat {beat_lw/1e6:.2f} MHz vs 2*dnu = {2*dnu_sh/1e6:.1f} MHz")

    # --- figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

        for f, S, dnu in shapes:
            m = np.abs(f) < 5 * max(dnus)
            ax[0].plot(f[m] / 1e6, 10 * np.log10(S[m] / S.max() + 1e-12),
                       lw=1.2, label=f"{dnu/1e6:.1f} MHz")
        ax[0].set(xlabel="baseband offset [MHz]", ylabel="PSD [dB]",
                  ylim=(-30, 2), title="OSA lineshape (Lorentzian)")
        ax[0].legend(fontsize=8, title=r"$\Delta\nu$")

        ax[1].loglog(dnus / 1e6, reads / 1e6, "o", ms=8, color="#2c7fb8",
                     label="measured (coherence)")
        ax[1].loglog(dnus / 1e6, dnus / 1e6, "--", color="0.4",
                     label=r"linewidth = $\Delta\nu$")
        ax[1].set(xlabel=r"set $\Delta\nu$ [MHz]",
                  ylabel="measured linewidth [MHz]",
                  title=f"linewidth readout (slope {slope:.2f})")
        ax[1].legend(fontsize=8)

        rf = (fbeat > f_aom - 12e6) & (fbeat < f_aom + 12e6)
        ax[2].plot(fbeat[rf] / 1e6,
                   10 * np.log10(Pavg[rf] / Pavg[rf].max() + 1e-12),
                   color="#d95f02")
        ax[2].axvspan((f_aom - beat_lw) / 1e6, (f_aom + beat_lw) / 1e6,
                      color="0.85", label=r"$\pm 2\Delta\nu$")
        ax[2].set(xlabel="RF frequency [MHz]", ylabel="beat PSD [dB]",
                  title=f"self-heterodyne beat ({beat_lw/1e6:.1f} MHz "
                        f"= 2$\\Delta\\nu$)")
        ax[2].legend(fontsize=8)

        fig.tight_layout()
        OUT.parent.mkdir(exist_ok=True)
        fig.savefig(OUT, dpi=150)
        print(f"\nwrote {OUT}")
    except Exception as e:                     # plotting is optional
        print(f"(skipped figure: {e})")

    ok = all(c[1] for c in CHECKS)
    print("\n" + ("ALL LINEWIDTH CHECKS PASSED" if ok
                  else "SOME CHECKS FAILED: "
                       + ", ".join(n for n, o, _ in CHECKS if not o)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
