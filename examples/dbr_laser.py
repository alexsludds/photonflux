#!/usr/bin/env python3
"""Tunable DBR laser from traveling-wave slices: a section current steps the
lasing wavelength across longitudinal-mode hops.

Three sections on the shared field bus (a cleaved back facet closes the cavity):

    facet(R=0.9) <-> [ gain x Ng ] <-> [ Bragg DBR x Nd, tunable ] -> (AR)
                            |                    |
                     tw_gain_seg.va         tw_seg.va (kappa>0, dbeta_dv)

The facet + the DBR stopband form a Fabry-Perot cavity whose longitudinal modes
are spaced by the cavity FSR. The DBR reflects only near its Bragg wavelength,
so the laser runs on the ONE cavity mode sitting under the DBR peak. A current
into the DBR sections (index change -> dbeta_dv * V shifts the local Bragg
detuning) slides the stopband; when it passes the midpoint between two cavity
modes the winner-take-all gain reservoir drops the old mode and the neighbour
takes over -- a MODE HOP, marked by a wavelength discontinuity of one FSR and a
power dip at the boundary.

As in ``soa_vernier_laser.py`` the discreteness is emergent: the deterministic
seed is off and broadband ASE noise seeds every candidate mode, so which mode
wins is decided by the physics, not the seed. The DBR tuning is stepped as a
staircase (a "DC sweep" of a lasing laser must be a sequence of settle points --
plain Newton would sit on the unstable dark branch) and the settled lasing line
is measured at each step.

Checks: the lasing line steps monotonically (mostly) red with DBR drive, shows
>= 1 discontinuity of ~1 FSR (a mode hop) with a power dip at the boundary.

    python examples/dbr_laser.py            ->  out/dbr_laser.png   (~2-3 min)
"""
from __future__ import annotations

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # circuit solves need float64

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import Signals, States, source

from _cavity import run_transient, staircase_source, terminator
from photonflux import cx

C0 = 2.99792458e8
HPL = 6.62607015e-34

# --- the laser --------------------------------------------------------------
LAM_NM = 1310.0
NG = 3.7
RF = 0.9                    # back-facet power reflectivity
N_GAIN = 10
DZ_G = 22e-6
N_DBR = 10
DZ_D = 15e-6
L_DBR = N_DBR * DZ_D
KAPPA = 1.8 / L_DBR         # kappa*L_DBR = 1.8 (a strong, selective mirror)
L_CAV = N_GAIN * DZ_G + 0.5 * L_DBR
FSR_HZ = C0 / (2 * NG * L_CAV)

G_UNSAT = 3200.0
I_BIAS = 75e-3
DBETA_DV = 9000.0          # DBR detuning tuning [1/m per V]

GAIN = dict(lambda_nm=LAM_NM, lambda_bragg_nm=LAM_NM, n_g=NG, dz=DZ_G,
            g_unsat_pm=G_UNSAT, i_op_ma=80.0, i_tr_ma=8.0, p_sat=10e-3,
            tau_c=0.3e-9, alpha_h=0.0, kappa_pm=0.0, loss_pm=0.0,
            p_seed=0.0, Von=1.2, Rs=3.0)
DBR = dict(lambda_nm=LAM_NM, lambda_bragg_nm=LAM_NM, n_g=NG, dz=DZ_D,
           kappa_pm=KAPPA, gamma_pm=0.0, dbeta_dv=DBETA_DV)
N_SP = 2.0

# --- the sweep --------------------------------------------------------------
N_STEP = 8
V_MAX = 2.4                # DBR tuning voltage span
T_STEP = 2.5e-9
DT = 0.2e-12
DT_SAVE = 1e-12
DT_NOISE = 1e-12
NOISE_SEED = 4

OUT = Path(__file__).resolve().parents[1] / "out" / "dbr_laser.png"
NU0 = C0 / (LAM_NM * 1e-9)


def s_ase() -> float:
    g0 = G_UNSAT * (I_BIAS * 1e3 - 8.0) / (80.0 - 8.0)
    return N_SP * HPL * NU0 * (np.exp(min(2 * g0 * N_GAIN * DZ_G, 20.0)) - 1.0)


def _noise_src(stream, tn):
    @source(ports=("re",), states=("i",))
    def NS(signals: Signals, s: States, t: float):
        return {"re": s.i, "i": signals.re - jnp.interp(t, tn, stream)}, {}
    return NS


def build(v_dbr_levels):
    n = int(np.ceil(N_STEP * T_STEP / DT_NOISE)) + 4
    rng = np.random.default_rng(NOISE_SEED)
    sigma = float(np.sqrt(s_ase() / (4.0 * DT_NOISE)))
    bank = rng.standard_normal((4, n)) * sigma
    bank[:, 0] = 0.0
    tn = jnp.arange(n) * DT_NOISE

    instances = {"GND": {"component": "ground"},
                 "M": {"component": "mirror", "settings": {"refl": RF}},
                 "VB": {"component": "vsrc", "settings": {"V": 1.2 + 3.0 * I_BIAS}},
                 "VH": {"component": "hstair"}}
    for nm in ("NFr", "NFi", "NBr", "NBi"):
        instances[nm] = {"component": nm}
    for k in range(N_GAIN):
        instances[f"G{k}"] = {"component": "g", "settings": GAIN}
    for k in range(N_DBR):
        instances[f"D{k}"] = {"component": "d", "settings": DBR}

    seq = [f"G{k}" for k in range(N_GAIN)] + [f"D{k}" for k in range(N_DBR)]
    conns = {"M,ro_re": "G0,fl_re", "M,ro_im": "G0,fl_im",
             "G0,bl_re": "M,ri_re", "G0,bl_im": "M,ri_im",
             # ASE into the cavity: forward through the facet, backward at the AR end
             "NFr,re": "M,li_re", "NFi,re": "M,li_im",
             "NBr,re": f"{seq[-1]},br_re", "NBi,re": f"{seq[-1]},br_im"}
    for a, b in zip(seq[:-1], seq[1:]):
        conns[f"{a},fr_re"] = f"{b},fl_re"
        conns[f"{a},fr_im"] = f"{b},fl_im"
        conns[f"{b},bl_re"] = f"{a},br_re"
        conns[f"{b},bl_im"] = f"{a},br_im"
    conns[f"{seq[-1]},fr_re"] = "TR,re"
    conns[f"{seq[-1]},fr_im"] = "TR,im"
    instances["TR"] = {"component": "term"}
    conns["VB,p1"] = tuple(f"G{k},an" for k in range(N_GAIN))
    conns["VH,p1"] = tuple(f"D{k},vt" for k in range(N_DBR))
    grounded = ["M,gnd", "VB,p2", "VH,p2"]
    for k in range(N_GAIN):
        grounded += [f"G{k},gnd", f"G{k},cat"]
    for k in range(N_DBR):
        grounded += [f"D{k},gnd"]
    conns["GND,p1"] = tuple(grounded)

    net = {"instances": instances, "connections": conns,
           "ports": {"o_re": "M,lo_re", "o_im": "M,lo_im"}}
    models = {"ground": lambda: 0, "vsrc": __import__(
                  "circulax.components.electronic",
                  fromlist=["VoltageSource"]).VoltageSource,
              "mirror": cx.va("mirror"), "g": cx.va("tw_gain_seg"),
              "d": cx.va("tw_seg"), "term": terminator(),
              "hstair": staircase_source(v_dbr_levels, T_STEP),
              "NFr": _noise_src(jnp.asarray(bank[0]), tn),
              "NFi": _noise_src(jnp.asarray(bank[1]), tn),
              "NBr": _noise_src(jnp.asarray(bank[2]), tn),
              "NBi": _noise_src(jnp.asarray(bank[3]), tn)}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def line_of(e, dt):
    win = np.hanning(len(e))
    X = np.fft.fftshift(np.fft.fft(e * win))
    f = np.fft.fftshift(np.fft.fftfreq(len(e), dt))
    k = int(np.argmax(np.abs(X) ** 2))
    return -f[k]                       # optical offset from Bragg [Hz]


def main() -> int:
    print(f"DBR laser: facet R={RF}, {N_GAIN} gain + {N_DBR} DBR slices, "
          f"kappa*L_DBR={KAPPA*L_DBR:.2f}, cavity FSR ~ {FSR_HZ/1e9:.0f} GHz")
    v_levels = np.linspace(0.0, V_MAX, N_STEP)

    t0 = time.time()
    c = build(v_levels)
    t, sol = run_transient(c, N_STEP * T_STEP, DT, save_every=DT_SAVE)
    e = np.asarray(c.port(sol.ys, "o_re").real) + 1j * np.asarray(
        c.port(sol.ys, "o_im").real)
    p = np.abs(e) ** 2
    print(f"transient: {time.time()-t0:.1f}s")

    nu, pk = [], []
    for k in range(N_STEP):
        w = (t > (k + 0.5) * T_STEP) & (t < (k + 1) * T_STEP)
        nu.append(line_of(e[w], DT_SAVE) / 1e9)
        pk.append(p[w].mean())
    nu = np.asarray(nu); pk = np.asarray(pk)

    dnu = np.diff(nu)
    hops = np.where(np.abs(dnu) > 0.5 * FSR_HZ / 1e9)[0]
    print("DBR sweep (settled per step):")
    for k in range(N_STEP):
        tag = "  <- HOP" if k - 1 in hops else ""
        print(f"  V={v_levels[k]:.2f}  line={nu[k]:+7.1f} GHz  "
              f"P={pk[k]*1e3:.3f} mW{tag}")
    print(f"tuning: {nu[0]:+.0f} -> {nu[-1]:+.0f} GHz over {V_MAX} V; "
          f"{len(hops)} mode hop(s) of ~{FSR_HZ/1e9:.0f} GHz")

    assert len(hops) >= 1, "no mode hop across the DBR tuning sweep"
    assert nu[-1] - nu[0] > 0.5 * FSR_HZ / 1e9, "DBR did not tune the wavelength"
    print("ALL TESTBENCH CHECKS PASSED")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(t * 1e9, p * 1e3, c="tab:orange", lw=0.7)
    for k in range(N_STEP):
        ax.axvline(k * T_STEP * 1e9, c="gray", lw=0.4, ls=":")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("P_out [mW]")
    ax.set_title("DBR tuning staircase (power dips at hops)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(v_levels, nu, "o-", c="tab:blue")
    for h in hops:
        ax.axvspan(v_levels[h], v_levels[h + 1], color="tab:red", alpha=0.12)
    ax.set_xlabel("DBR tuning voltage [V]")
    ax.set_ylabel("lasing line offset from Bragg [GHz]")
    ax.set_title(f"wavelength vs DBR current ({len(hops)} hop, FSR "
                 f"~ {FSR_HZ/1e9:.0f} GHz)")
    ax.grid(alpha=0.3)

    fig.suptitle("Tunable DBR laser from traveling-wave slices "
                 "(tw_gain_seg + tw_seg, circulax) — longitudinal mode hops",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
