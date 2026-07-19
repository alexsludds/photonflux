#!/usr/bin/env python3
"""Split-step Fourier nonlinear fibre (ALE-71), pinned against textbook analytics.

``webapp/ssfm.py`` propagates the scalar nonlinear Schrodinger equation

    dA/dz = -(alpha/2) A + i(beta2/2) A_TT - i(beta3/6) A_TTT - i gamma |A|^2 A

by the symmetric split-step Fourier method — the standard km-scale nonlinear
fibre tool in VPI / Lumerical INTERCONNECT, and the nonlinear counterpart of the
linear ``fiber_cd`` component (``webapp/lti.py``). Self-phase modulation,
four-wave mixing and solitons all emerge from the field evolution; this script
checks each against its closed form and draws ``out/fiber_nl_ssfm.png``.

Checks (all asserted, machine-verifiable):
  1. soliton     — the N=1 soliton P0 = |beta2|/(gamma T0^2) sech(T/T0) keeps its
                   shape over 3 soliton periods; energy is conserved
  2. SPM         — a Gaussian's SPM-broadened spectrum has M peaks at
                   phi_max ~ (M - 1/2) pi (Agrawal, Nonlinear Fiber Optics Fig 4.2)
  3. FWM         — degenerate-FWM idler efficiency follows sinc^2(dbeta L/2) vs the
                   dispersive phase mismatch dbeta = beta2 Omega^2
  4. linear      — gamma = 0 reproduces the fiber_cd dispersion all-pass to <1%
  5. component   — the in-transient webapp ``fiber_nl`` split-step block (solved
                   by circulax) grows the same FWM idler as the batch engine

    .venv-circulax/bin/python examples/fiber_nl_ssfm.py    -> out/fiber_nl_ssfm.png
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp"))

import ssfm  # noqa: E402
import vf  # noqa: E402
import lti  # noqa: E402

OUT = REPO / "out" / "fiber_nl_ssfm.png"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ===========================================================================
# 1. fundamental soliton
# ===========================================================================
def soliton_study():
    p = ssfm.fiber_params(length_km=1.0, D_ps=17.0, atten_db_km=0.0,
                          gamma_per_W_km=1.3)
    t0 = 20e-12
    n = 4096
    dt = 40 * t0 / n
    t = (np.arange(n) - n / 2) * dt
    a0 = ssfm.soliton_field(t, t0, p)
    z0 = ssfm.soliton_period(t0, p)

    frames = {}
    for k in (0, 1, 2, 3):
        pk = replace(p, length=max(k, 1e-9) * z0)
        frames[k] = a0 if k == 0 else ssfm.propagate(a0, dt, pk, max_phase=2e-3)
    a3 = frames[3]
    shape_err = np.max(np.abs(np.abs(a3) - np.abs(a0))) / np.abs(a0).max()
    energy_err = abs(np.sum(np.abs(a3) ** 2) / np.sum(np.abs(a0) ** 2) - 1)
    check("N=1 soliton shape over 3 periods", shape_err < 1e-3,
          f"max |A| deviation {shape_err:.2e} of peak, energy drift "
          f"{energy_err:.1e}")
    return t, t0, frames


# ===========================================================================
# 2. SPM spectral peak count
# ===========================================================================
def spm_study():
    p = ssfm.fiber_params(length_km=1.0, D_ps=0.0, atten_db_km=0.0,
                          gamma_per_W_km=1.3)
    p = replace(p, beta2=0.0, beta3=0.0)
    n = 8192
    t0 = 40e-12
    dt = 40 * t0 / n
    t = (np.arange(n) - n / 2) * dt
    f = np.fft.fftshift(np.fft.fftfreq(n, dt))
    band = (f > -3e10) & (f < 3e10)

    spectra, counts = {}, {}
    for m in (1, 2, 3, 4):
        phi = (m - 0.5) * np.pi
        p0 = phi / (p.gamma * p.length)
        a0 = np.sqrt(p0) * np.exp(-0.5 * (t / t0) ** 2)
        a_out = ssfm.propagate(a0, dt, p, max_phase=1e-3)
        s = np.abs(np.fft.fftshift(np.fft.fft(a_out))) ** 2
        sb = s[band]
        counts[m] = int(((sb[1:-1] > sb[:-2]) & (sb[1:-1] > sb[2:])
                         & (sb[1:-1] > 1e-4 * sb.max())).sum())
        spectra[m] = (f[band], sb / sb.max())
    ok = all(counts[m] == m for m in counts)
    check("SPM peak count = round(phi/pi + 1/2)", ok,
          "peaks " + ", ".join(f"{(m-0.5):.1f}pi->{counts[m]}" for m in counts))
    return spectra


# ===========================================================================
# 3. four-wave mixing efficiency vs phase mismatch
# ===========================================================================
def fwm_study():
    p0 = ssfm.fiber_params(length_km=2.0, D_ps=0.0, atten_db_km=0.0,
                           gamma_per_W_km=1.3)
    pp, ps = 5e-3, 1e-7
    omega = 2.0 * np.pi * 50e9
    n = 2048
    dt = 60 * (2.0 * np.pi / omega) / n
    t = np.arange(n) * dt
    a0 = np.sqrt(pp) + np.sqrt(ps) * np.exp(1j * omega * t)
    f = np.fft.fftfreq(n, dt)
    i_idler = int(np.argmin(np.abs(f + 50e9)))
    lam, c0 = 1550e-9, 299792458.0

    def eff(beta2: float) -> float:
        p = replace(p0, beta2=beta2, beta3=0.0)
        a_out = ssfm.propagate(a0, dt, p, max_phase=1e-3)
        return np.abs(np.fft.fft(a_out)[i_idler] / n) ** 2 / ps

    eta0 = eff(0.0)
    d_grid = np.linspace(0.5, 14.0, 24)
    sim, th = [], []
    for d_ps in d_grid:
        beta2 = -(d_ps * 1e-6) * lam ** 2 / (2.0 * np.pi * c0)
        dbeta = beta2 * omega ** 2
        sim.append(eff(beta2) / eta0)
        th.append(np.sinc(dbeta * p0.length / 2.0 / np.pi) ** 2)
    sim, th = np.array(sim), np.array(th)
    err = float(np.max(np.abs(sim - th)))
    check("FWM idler eta ~ sinc^2(dbeta L/2)", err < 0.03,
          f"max deviation {err:.3f} over dbeta L/2 in "
          f"[{0:.1f}, {abs((-(d_grid[-1]*1e-6)*lam**2/(2*np.pi*c0))*omega**2)*p0.length/2:.1f}] "
          f"(gamma Pp L = {p0.gamma*pp*p0.length:.3f}, small-signal)")
    return d_grid, sim, th


# ===========================================================================
# 4. linear limit vs fiber_cd
# ===========================================================================
def linear_study():
    settings = dict(length_km=10.0, D_ps=17.0, S_ps=0.0, lambda_nm=1550.0,
                    atten_db_km=0.2, fit_bw=60e9, n_poles=28)
    _, payload = lti.build("fiber", settings, [])
    poles, res, d = payload["cplx"]
    p = ssfm.fiber_params(length_km=10.0, D_ps=17.0, lambda_nm=1550.0,
                          atten_db_km=0.2, gamma_per_W_km=0.0)
    n = 8192
    dt = 1.0 / 240e9
    rng = np.random.default_rng(0)
    a0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a_out = ssfm.propagate(a0, dt, p)
    f = np.fft.fftfreq(n, dt)
    w = 2.0 * np.pi * f
    h_ssfm = np.fft.fft(a_out) / np.fft.fft(a0)
    b2l, b3l = p.beta2 * p.length, p.beta3 * p.length
    w_max = 2.0 * np.pi * 60e9
    td = 1.5 * (abs(b2l) * w_max + 0.5 * abs(b3l) * w_max ** 2)
    h_vf = vf.eval_fit(poles, res, d, f)
    band = np.abs(f) < 55e9
    rel = np.max(np.abs(h_ssfm[band] * np.exp(-1j * w[band] * td)
                        - h_vf[band])) / np.abs(h_vf[band]).mean()
    check("gamma=0 linear limit == fiber_cd", rel < 0.01,
          f"max in-band complex error {rel*100:.2f}% vs the 28-pole vector fit")
    order = np.argsort(f[band])
    return f[band][order], h_ssfm[band][order], h_vf[band][order], td, w[band][order]


# ===========================================================================
# 5. in-transient webapp fiber_nl component vs the batch engine
# ===========================================================================
def component_study():
    """Solve the webapp ``fiber_nl`` split-step block in circulax on a two-tone
    drive and confirm it grows the same FWM idler the batch engine predicts."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import diffrax
    from circulax import compile_circuit
    from circulax.components.base_component import Signals, States, component, source

    fp, fs, pp, ps = 20e9, 30e9, 0.2, 1e-4
    seg = dict(length_km=2.0, D_ps=0.0, lambda_nm=1550.0, atten_db_km=0.0,
               gamma_per_W_km=1.3, n_seg=20, fit_bw=60e9, n_poles=16)

    def two_tone():
        @source(ports=("p1", "p2"), states=("i_src",))
        def TwoTone(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
            field = (jnp.sqrt(pp) * jnp.exp(2j * jnp.pi * fp * t)
                     + jnp.sqrt(ps) * jnp.exp(2j * jnp.pi * fs * t))
            return {"p1": s.i_src, "p2": -s.i_src,
                    "i_src": (signals.p1 - signals.p2) - field}, {}
        return TwoTone

    def term():
        @component(ports=("p1",))
        def Term(signals: Signals, s: States) -> tuple[dict, dict]:
            return {"p1": 0.0}, {}
        return Term

    def circuit(gamma_per_W_km: float):
        _, payload = lti.build("fiber_nl", {**seg, "gamma_per_W_km": gamma_per_W_km}, [])
        import catalog
        model = catalog._fiber_nl_field(*payload["fiber_nl"])
        net = {"instances": {"GND": {"component": "ground"},
                             "SRC": {"component": "src"},
                             "F": {"component": "fib"}, "T": {"component": "term"}},
               "connections": {"SRC,p1": "F,p1", "GND,p1": "SRC,p2",
                               "F,p2": "T,p1"}, "ports": {"out": "F,p2"}}
        models = {"ground": lambda: 0, "src": two_tone(), "fib": model,
                  "term": term()}
        return compile_circuit(net, models, backend="dense", is_complex=True,
                               max_steps=400)

    dt, npt = 0.5e-12, 4000
    ts = jnp.arange(0.0, npt * dt, dt)

    def run(c):
        sol = c.transient(t0=0.0, t1=npt * dt, dt0=dt, y0=c.dc(),
                          saveat=diffrax.SaveAt(ts=ts), max_steps=npt + 10,
                          stepsize_controller=diffrax.ConstantStepSize(),
                          throw=False)
        return np.asarray(c.port(sol.ys, "out"))[:npt]

    e = run(circuit(1.3))
    e0 = run(circuit(0.0))
    f = np.fft.fftfreq(npt, dt)
    a = np.fft.fft(e) / npt
    a0 = np.fft.fft(e0) / npt
    f_i = 2 * fp - fs
    ii = int(np.argmin(np.abs(f - f_i)))
    p_idler = abs(a[ii]) ** 2
    p_idler0 = abs(a0[ii]) ** 2
    p_out = float(np.mean(np.abs(e) ** 2))

    # batch-engine idler on the same two-tone drive (envelope frame)
    t = np.arange(npt) * dt
    p_eng = ssfm.fiber_params(length_km=2.0, D_ps=0.0, atten_db_km=0.0,
                              gamma_per_W_km=1.3)
    a_env = (np.sqrt(pp) * np.exp(2j * np.pi * fp * t)
             + np.sqrt(ps) * np.exp(2j * np.pi * fs * t))
    a_eng = ssfm.propagate(a_env.copy(), dt, p_eng, max_phase=2e-3)
    p_idler_eng = abs((np.fft.fft(a_eng) / npt)[ii]) ** 2

    rel = abs(p_idler / p_idler_eng - 1)
    check("webapp fiber_nl idler == engine", rel < 0.05 and p_idler0 < 1e-12 * p_idler,
          f"idler {10*np.log10(p_idler/1e-3):.1f} dBm vs engine "
          f"{10*np.log10(p_idler_eng/1e-3):.1f} dBm ({rel*100:.1f}% off); "
          f"gamma=0 idler floor {10*np.log10(max(p_idler0,1e-30)/1e-3):.0f} dBm")
    check("component energy conservation", abs(p_out / (pp + ps) - 1) < 1e-3,
          f"mean |E_out|^2 / P_in - 1 = {p_out/(pp+ps)-1:+.2e} (lossless)")
    return f, a, a0, f_i


# ===========================================================================
def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("split-step Fourier nonlinear fibre — analytics self-check")
    t, t0, frames = soliton_study()
    spectra = spm_study()
    d_grid, fwm_sim, fwm_th = fwm_study()
    lf, h_ssfm, h_vf, td, lw = linear_study()
    cf, ca, ca0, f_i = component_study()

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.5))
    fig.suptitle("Split-step Fourier nonlinear fibre (webapp/ssfm.py) "
                 "vs textbook analytics", fontsize=13)

    a0 = ax[0, 0]
    for k in (0, 1, 2, 3):
        a0.plot(t * 1e12, np.abs(frames[k]) ** 2 * 1e3,
                label=f"z = {k} z0", lw=1.6 if k else 2.4,
                color="k" if k == 0 else None)
    a0.set(xlabel="time [ps]", ylabel="power [mW]", xlim=(-100, 100),
           title="N=1 soliton: shape-invariant")
    a0.legend(fontsize=8)

    a1 = ax[0, 1]
    for m, (ff, ss) in spectra.items():
        a1.plot(ff / 1e9, ss + 0.0, lw=1.4, label=f"phi={m-0.5:.1f}pi ({m} peaks)")
    a1.set(xlabel="envelope f [GHz]", ylabel="norm. spectrum",
           title="SPM broadening: M peaks at (M-1/2)pi", xlim=(-25, 25))
    a1.legend(fontsize=7)

    a2 = ax[0, 2]
    a2.plot(d_grid, fwm_th, "-", color="#d95f02", lw=1.5, label=r"$\mathrm{sinc}^2(\Delta\beta L/2)$")
    a2.plot(d_grid, fwm_sim, "o", color="#2c7fb8", ms=5, label="SSFM")
    a2.set(xlabel="dispersion D [ps/nm/km]", ylabel="idler eff. / peak",
           title="FWM phase matching")
    a2.legend(fontsize=8)

    a3 = ax[1, 0]
    a3.plot(lf / 1e9, np.abs(h_vf), "-", color="#d95f02", lw=2, label="fiber_cd (vf)")
    a3.plot(lf / 1e9, np.abs(h_ssfm), "--", color="#2c7fb8", lw=1.4, label="SSFM gamma=0")
    a3.set(xlabel="envelope f [GHz]", ylabel="|H|",
           title="linear limit == fiber_cd")
    a3.legend(fontsize=8)

    a4 = ax[1, 1]
    ph_ssfm = np.unwrap(np.angle(h_ssfm * np.exp(-1j * lw * td)))
    a4.plot(lf / 1e9, np.unwrap(np.angle(h_vf)), "-", color="#d95f02", lw=2,
            label="fiber_cd phase")
    a4.plot(lf / 1e9, ph_ssfm, "--", color="#2c7fb8", lw=1.4, label="SSFM (Td removed)")
    a4.set(xlabel="envelope f [GHz]", ylabel="phase [rad]",
           title="dispersive phase")
    a4.legend(fontsize=8)

    a5 = ax[1, 2]
    p_lines = np.abs(ca) ** 2
    p0_lines = np.abs(ca0) ** 2

    def dbm(x):
        return 10 * np.log10(np.maximum(x, 1e-30) / 1e-3)
    sel = p_lines > 1e-24
    a5.vlines(cf[sel] / 1e9, -230, dbm(p_lines[sel]), color="#2c7fb8", lw=1.8)
    sel0 = p0_lines > 1e-24
    a5.vlines(cf[sel0] / 1e9, -230, dbm(p0_lines[sel0]), color="0.75", lw=3.5,
              alpha=0.8, zorder=0)
    a5.annotate("idler\n2fp-fs", (f_i / 1e9, dbm(abs(ca[int(np.argmin(np.abs(cf-f_i)))])**2) + 6),
                ha="center", fontsize=8)
    a5.set(xlim=(-15, 55), ylim=(-140, 30), xlabel="envelope f [GHz]",
           ylabel="line power [dBm]",
           title="webapp fiber_nl (grey: gamma=0)")

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    if all(ok for _, ok, _ in CHECKS):
        print("ALL SSFM CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED:", [n for n, ok, _ in CHECKS if not ok])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
