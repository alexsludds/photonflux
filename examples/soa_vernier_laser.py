#!/usr/bin/env python3
"""Vernier ring laser seeded by white ASE noise: watch the modes compete.

The tunable-laser architecture, built from the repo's Verilog-A parts and
solved closed-loop in circulax:

    SOA (models/optical_field/soa.va) -> ring A drop -> ring B drop -> feedback coupler
      '---------------------- R_fb = 0.6 ----------------------'  '-> P_out

Each ``models/optical_field/ring_filter.va`` carries FIVE longitudinal modes, so each ring
is a resonance COMB: ring A (R = 400 um) has FSR_A = 29.8 GHz, ring B
(R = 425 um) has FSR_B = 28.1 GHz. The comb mismatch (dFSR = 1.76 GHz ~ 1.5
ring linewidths) is a real tunable-laser compromise: at turn-on THREE
alignment candidates (0, +-FSR_A) sit above threshold so the laser
genuinely has to choose, yet one heater pair-step later the old pair falls
far enough below threshold to die — bigger mismatch would sterilise the
competition, smaller would leave the old mode lasing after the hop. (The comb spacing also keeps every candidate
where the fixed-step BDF2 integrator is faithful: a resonator driven at a
+-FSR_A envelope beat keeps ~95% of its true transmission at 0.25 ps steps,
so the physical gain differences — not discretization — decide the race;
at the old 200 um / 59.6 GHz geometry the penalty was 20% and could flip
the winner.)

Nothing tells it the answer. The SOA's deterministic seed is OFF
(p_seed = 0); instead a broadband ASE source injects complex white noise at
the physical level S = n_sp*h*nu*(G0-1) into the loop, the frame stays at
1310 nm throughout, and ONE continuous transient is watched like an
experiment:

1. **Turn-on from noise** (t = 0.3 ns) — the bias steps through threshold
   and the comb candidates rise out of the ASE floor together.
2. **Mode competition** — the candidate lines grow, each at ln(L)/T_rt
   with its OWN round-trip time: off-resonance light traverses the rings
   with less group delay, so poorly-filtered candidates sprint (real
   transient multimode physics) while the aligned pair — highest gain per
   pass, lowest threshold — outruns them and clamps the shared reservoir:
   the spectrogram shows the whole race.
3. **Winner takes all** — SMSR grows linearly in dB as the losers decay.
4. **A live mode hop** (t = 6 ns) — 0.50 mW steps onto the ring-B heater
   WHILE THE LASER RUNS: a 1.76 GHz comb shift drops the running mode's
   loop gain below threshold, the output collapses while the reservoir
   refills, and the newly aligned pair one FSR_A = 29.8 GHz to the red
   grows out of the ever-present ASE floor and takes over — a x17 Vernier
   tuning lever, exactly how a real tunable laser hops.

Checks: both lasing lines sit on the comb-product prediction (+-2.5 GHz)
with SMSR > 18 dB and clamped-gain reservoir power, >= 3 candidate lines
rise >= 20 dB out of their own ASE floor during buildup (the competition),
and the hop shows a real power dip.

    .venv-circulax/bin/python examples/soa_vernier_laser.py

        -> out/soa_vernier_laser.png   (runtime ~10 min: 48k BDF2 steps
                                        resolve the +-60 GHz envelope beats
                                        of the losing modes)
"""
from __future__ import annotations

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

# --- the parts ---------------------------------------------------------------
LAM0_NM = 1310.0             # cold resonance of both rings AND the fixed frame
RING_A = dict(lambda_res_nm=LAM0_NM, radius_um=400.0, n_g=4.0,
              loss_db_m=100.0, kappa2_in=0.1, kappa2_drop=0.1,
              dl_dmw_pm=20.0, r_heater=500.0)
RING_B = {**RING_A, "radius_um": 425.0}   # dFSR ~ 1.76 GHz ~ 1.5 linewidths
R_FB = 0.6                   # feedback-coupler power reflectivity
SOA = dict(g0_db=20.0, i_op_ma=80.0, i_tr_ma=8.0, p_sat=10e-3,
           tau_c=0.3e-9,
           tau_bw=5e-14,     # ~3 THz gain bandwidth: no artificial handicap
                             # for candidates 100-250 GHz off the frame
           alpha_h=0.0,
           p_seed=0.0,       # deterministic seed OFF — ASE noise seeds it
           Von=1.2, Rs=3.0)
I_BIAS = 80e-3               # operating current [A] — hard pump: the fair
                             # broadband seed per line is only ~ -80 dBm, so
                             # the buildup needs ~25 dB/ns to fit the window
N_SP = 2.0                   # SOA inversion factor for the ASE level

# --- the run ------------------------------------------------------------------
T_STOP = 12e-9               # one continuous experiment (see run_experiment)
T_ON = 0.3e-9                # bias steps up here (noise-only floor before)
T_HOP = 6e-9                 # ring-B heater steps here, DURING lasing
DT = 0.25e-12                # BDF2 step: resolves the +-60 GHz candidates
DT_SAVE = 1e-12              # saved grid -> +-500 GHz spectra
DT_NOISE = 1e-12             # ASE bank spacing (white to ~500 GHz)
NOISE_SEED = 7               # reproducible ASE realisation

OUT = Path(__file__).resolve().parents[1] / "out" / "soa_vernier_laser.png"

NU0 = C0 / (LAM0_NM * 1e-9)


# ===========================================================================
# analytic comb machinery (same equations as ring_filter.va)
# ===========================================================================
def ring_rates(ring: dict) -> tuple[float, float, float]:
    """(tau, 1/tau_e per bus, FSR [Hz]) of one add-drop ring."""
    circ = 2 * np.pi * ring["radius_um"] * 1e-6
    v_g = C0 / ring["n_g"]
    t_rt = circ / v_g
    inv_ti = ring["loss_db_m"] * np.log(10) / 10 * v_g / 2
    inv_te = ring["kappa2_in"] / (2 * t_rt)          # = drop coupling too
    tau = 1 / (inv_ti + 2 * inv_te)
    return tau, inv_te, 1 / t_rt


def drop_amp(dnu: np.ndarray, ring: dict, shift_hz: float = 0.0) -> np.ndarray:
    """Complex drop amplitude of the 5-mode comb at offset dnu from nu0."""
    tau, inv_te, fsr = ring_rates(ring)
    amp = np.zeros_like(dnu, dtype=complex)
    for m in (-2, -1, 0, 1, 2):
        delta = 2 * np.pi * (dnu - shift_hz - m * fsr)
        amp += 1j * (2 * inv_te) / (1 / tau - 1j * delta)
    return amp


def heater_shift_hz(p_heat_w: float, ring: dict) -> float:
    """Thermal red shift of the whole comb, in (negative) Hz — the exact
    reciprocal the model computes, not the -c/lambda^2 linearisation."""
    lam0 = LAM0_NM * 1e-9
    dlam = ring["dl_dmw_pm"] * 1e-12 * p_heat_w * 1e3          # [m]
    return C0 / (lam0 + dlam) - C0 / lam0


def comb_product(dnu: np.ndarray, p_heat_w: float) -> np.ndarray:
    return (np.abs(drop_amp(dnu, RING_A)) ** 2
            * np.abs(drop_amp(dnu, RING_B,
                              heater_shift_hz(p_heat_w, RING_B))) ** 2)


def predict(p_heat_w: float, span_hz: float = 300e9, n: int = 120001) -> dict:
    """Comb-product argmax: the predicted lasing line for one heater power."""
    dnu = np.linspace(-span_hz, span_hz, n)
    t_prod = comb_product(dnu, p_heat_w)
    k = np.argmax(t_prod)
    return {"nu_off": dnu[k], "t1t2": t_prod[k], "dnu": dnu, "t_prod": t_prod}


def analytic_p_out(t1t2: float) -> float:
    """Clamped-gain reservoir power for a loop closed by T1*T2*R_fb."""
    g_th = 1 / (t1t2 * R_FB)
    h_th = np.log(g_th)
    hop = SOA["g0_db"] * np.log(10) / 10
    h0 = hop * (I_BIAS * 1e3 - SOA["i_tr_ma"]) / (SOA["i_op_ma"] - SOA["i_tr_ma"])
    if h0 <= h_th:
        return 0.0
    p_fi = (h0 - h_th) * SOA["p_sat"] / (g_th - 1)
    return (1 - R_FB) / R_FB * p_fi


def s_ase() -> float:
    """One-sided ASE power spectral density of the biased SOA [W/Hz]."""
    hop = SOA["g0_db"] * np.log(10) / 10
    h0 = hop * (I_BIAS * 1e3 - SOA["i_tr_ma"]) / (SOA["i_op_ma"] - SOA["i_tr_ma"])
    return N_SP * HPL * NU0 * (np.exp(h0) - 1.0)


# ===========================================================================
# the closed loop (frame FIXED at 1310 nm -- the lasing line is emergent)
# ===========================================================================
def ring_settings(ring: dict) -> dict:
    return {"lambda_nm": LAM0_NM, **ring}


def build_loop(v_heat_levels, v_heat_tstep, v_levels, t_step):
    """The loop with the ASE adder split into its re/im quadratures (the VA
    models carry fields as re/im node pairs). The ring-B heater is a
    staircase too, so the wavelength can be STEPPED while the laser runs."""
    n = int(np.ceil(T_STOP / DT_NOISE)) + 4
    rng = np.random.default_rng(NOISE_SEED)
    sigma = float(np.sqrt(s_ase() / (4.0 * DT_NOISE)))
    bank = rng.standard_normal((2, n)) * sigma * 1.22474487
    # the initial DC solve freezes n(t=0) as an eternal constant, which
    # would pre-charge the rings' m = 0 modes with a coherent envelope-DC
    # seed ~40 dB above the fair broadband floor and rig the competition —
    # start the noise at zero so the operating point is dark
    bank[:, 0] = 0.0
    tn = jnp.arange(n) * DT_NOISE
    noise_re = jnp.asarray(bank[0])
    noise_im = jnp.asarray(bank[1])

    def noise_adder(stream):
        @source(ports=("pin", "pout"), states=("i_out",))
        def NoiseAdd(signals: Signals, s: States, t: float) -> tuple[dict, dict]:
            nval = jnp.interp(t, tn, stream)
            return {"pin": 0.0, "pout": s.i_out,
                    "i_out": signals.pout - (signals.pin + nval)}, {}
        return NoiseAdd

    instances = {
        "GND": {"component": "ground"},
        "SOA": {"component": "soa", "settings": dict(SOA)},
        "ASER": {"component": "nadd_re"},
        "ASEI": {"component": "nadd_im"},
        "RA": {"component": "ring", "settings": ring_settings(RING_A)},
        "RB": {"component": "ring", "settings": ring_settings(RING_B)},
        "CP": {"component": "mirror", "settings": {"refl": R_FB}},
        "VB": {"component": "stair"},
        "VHB": {"component": "hstair"},
        "TA": {"component": "term"}, "TB": {"component": "term"},
        "TS": {"component": "term"},
    }
    connections = {
        "SOA,fo_re": "ASER,pin", "ASER,pout": "RA,in_re",
        "SOA,fo_im": "ASEI,pin", "ASEI,pout": "RA,in_im",
        "RA,drop_re": "RB,in_re", "RA,drop_im": "RB,in_im",
        "RB,drop_re": "CP,li_re", "RB,drop_im": "CP,li_im",
        "CP,lo_re": "SOA,fi_re", "CP,lo_im": "SOA,fi_im",
        "RA,thru_re": "TA,re", "RA,thru_im": "TA,im",
        "RB,thru_re": "TB,re", "RB,thru_im": "TB,im",
        "SOA,bo_re": "TS,re", "SOA,bo_im": "TS,im",
        "VB,p1": "SOA,an",
        "VHB,p1": "RB,hp",
    }
    grounded = ["SOA,cat", "SOA,gnd", "SOA,bi_re", "SOA,bi_im",
                "RA,gnd", "RB,gnd", "RA,hp", "RA,hn", "RB,hn",
                "CP,gnd", "CP,ri_re", "CP,ri_im", "VB,p2", "VHB,p2"]
    connections["GND,p1"] = tuple(grounded)
    net = {"instances": instances, "connections": connections,
           "ports": {"pout_re": "CP,ro_re", "pout_im": "CP,ro_im"}}
    models = {"ground": lambda: 0, "soa": cx.va("soa"),
              "ring": cx.va("ring_filter"), "mirror": cx.va("mirror"),
              "term": terminator(),
              "stair": staircase_source(v_levels, t_step),
              "hstair": staircase_source(v_heat_levels, v_heat_tstep),
              "nadd_re": noise_adder(noise_re),
              "nadd_im": noise_adder(noise_im)}
    return compile_circuit(net, models, backend="dense", is_complex=True,
                           max_steps=300)


def run_experiment(p_hop_w: float):
    """THE run: noise-seeded turn-on, lase, then step the heater mid-flight.

    One transient covers the whole story — bias steps up at T_ON, the modes
    compete, the aligned pair at 1310 nm wins and clamps the gain; at T_HOP
    the ring-B heater steps to one comb-pair of shift, the running mode's
    loop gain collapses below 1, the power dips while the gain reservoir
    refills, and the NEWLY aligned pair (one FSR_A red) grows out of the
    ever-present ASE floor and takes over: a mode hop, exactly as a real
    tunable laser does it."""
    v_hop = float(np.sqrt(p_hop_w * RING_B["r_heater"]))
    v_levels = SOA["Von"] + SOA["Rs"] * np.array([5e-3, I_BIAS])
    c = build_loop(np.array([0.0, v_hop]), T_HOP, v_levels, T_ON)
    t, sol = run_transient(c, T_STOP, DT, save_every=DT_SAVE)
    er = np.asarray(c.port(sol.ys, "pout_re").real)
    ei = np.asarray(c.port(sol.ys, "pout_im").real)
    return t, er + 1j * ei


# ===========================================================================
# measurement: spectra, the winning line, SMSR
# ===========================================================================
def spectrum(e: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann periodogram of the complex envelope -> (f_env [Hz], PSD)."""
    n = len(e)
    win = np.hanning(n)
    x = np.fft.fftshift(np.fft.fft(e * win)) / np.sum(win)
    f = np.fft.fftshift(np.fft.fftfreq(n, dt))
    return f, np.abs(x) ** 2


def line_and_smsr(t, e, t_from: float, t_to: float | None = None,
                  guard_hz: float = 12e9):
    """Winning line (as an OPTICAL offset from 1310 nm), its SMSR, and the mean
    power over the window (t_from, t_to]. Envelope +f rotation = optical
    nu0 - f (red)."""
    m = t > t_from
    if t_to is not None:
        m &= t < t_to
    f, s = spectrum(e[m], float(t[1] - t[0]))
    k = np.argmax(s)
    nu_opt = -f[k]                       # optical offset from the frame
    away = np.abs(f - f[k]) > guard_hz
    smsr_db = 10 * np.log10(s[k] / s[away].max())
    power = float(np.mean(np.abs(e[m]) ** 2))
    return nu_opt, smsr_db, power, m


def spectrogram(t, e, t_win: float = 0.4e-9, hop: float = 0.1e-9):
    """STFT waterfall of the envelope: (t_centres, f_env, PSD dB)."""
    dt = float(t[1] - t[0])
    n_win = int(t_win / dt)
    n_hop = int(hop / dt)
    win = np.hanning(n_win)
    frames, centres = [], []
    for i0 in range(0, len(e) - n_win, n_hop):
        x = np.fft.fftshift(np.fft.fft(e[i0:i0 + n_win] * win))
        frames.append(np.abs(x) ** 2)
        centres.append(t[i0 + n_win // 2])
    f = np.fft.fftshift(np.fft.fftfreq(n_win, dt))
    S = np.asarray(frames)
    return np.asarray(centres), f, 10 * np.log10(S / S.max() + 1e-16)


def candidate_powers(t, e, cands_hz: np.ndarray, t_win: float = 0.4e-9,
                     hop: float = 0.1e-9, bw: float = 8e9):
    """Per-candidate line power vs time (integrated +-bw around each)."""
    tc, f, sdb = spectrogram(t, e, t_win, hop)
    s = 10 ** (sdb / 10)
    out = np.empty((len(cands_hz), len(tc)))
    for i, fc in enumerate(cands_hz):
        sel = np.abs(f - fc) < bw
        out[i] = s[:, sel].sum(axis=1)
    return tc, out


# ===========================================================================
# the testbench: report the design, measure the run, check every claim, plot
# ===========================================================================
def report_design() -> dict:
    """Print the comb geometry, the ASE seed, and the turn-on candidate gains;
    assert the competition is real; return the derived run parameters."""
    _, _, fsr_a = ring_rates(RING_A)
    _, _, fsr_b = ring_rates(RING_B)
    d_fsr = fsr_a - fsr_b
    tau, _, _ = ring_rates(RING_A)
    print(f"combs: FSR_A = {fsr_a/1e9:.2f} GHz, FSR_B = {fsr_b/1e9:.2f} GHz, "
          f"dFSR = {d_fsr/1e9:.2f} GHz ~ ring FWHM "
          f"{1/(np.pi*tau)/1e9:.2f} GHz -> Vernier lever x{fsr_a/d_fsr:.0f}")
    print(f"ASE seed: S = {s_ase():.3g} W/Hz "
          f"({10*np.log10(s_ase()/1e-3):.1f} dBm/Hz), white to "
          f"{1/(2*DT_NOISE)/1e9:.0f} GHz")

    # loop gains of the five candidates at turn-on (unsaturated)
    hop_l = SOA["g0_db"] * np.log(10) / 10
    h0 = hop_l * (I_BIAS * 1e3 - SOA["i_tr_ma"]) / (SOA["i_op_ma"] - SOA["i_tr_ma"])
    g0 = np.exp(h0)
    cands = np.array([0.0, -fsr_a, fsr_a, -2 * fsr_a, 2 * fsr_a])
    l0 = g0 * comb_product(cands, 0.0) * R_FB
    print("round-trip gains at turn-on (heater off): "
          + ", ".join(f"{c/1e9:+.0f} GHz: {v:.2f}" for c, v in zip(cands, l0)))
    assert (l0 > 1).sum() >= 3, "want >= 3 candidates above threshold"

    # heater power that re-aligns the combs on the (-1,-1) pair
    hz_per_w = -heater_shift_hz(1e-3, RING_B) / 1e-3
    p_hop = d_fsr / hz_per_w
    print(f"heater for one Vernier hop: {p_hop*1e3:.3f} mW "
          f"({np.sqrt(p_hop*RING_B['r_heater']):.3f} V)")

    return {"fsr_a": fsr_a, "d_fsr": d_fsr, "g0": g0, "cands": cands,
            "p_hop": p_hop}


def measure(t, e, geom: dict) -> dict:
    """Reduce one transient to the numbers the checks and the plot need: the
    two settled acts, the hop-dip depth, the spectrogram, and the per-candidate
    buildup that shows the mode competition."""
    cands = geom["cands"]
    pred0, pred1 = predict(0.0), predict(geom["p_hop"])

    # act 1 (before the heater step): the competition winner at 1310 nm.
    # act 2 (after the hop): the same laser, one FSR_A to the red.
    nu1, smsr1, p1, w1 = line_and_smsr(t, e, T_HOP - 1.5e-9, T_HOP - 0.05e-9)
    nu2, smsr2, p2, w2 = line_and_smsr(t, e, T_STOP - 1.5e-9)

    # the hop transient: power dips while the reservoir refills and the new
    # line grows out of the ASE floor
    p_tot = np.abs(e) ** 2
    p_dip = float(p_tot[(t > T_HOP) & (t < T_HOP + 2e-9)].min())

    # spectrogram + per-candidate powers over the whole story
    tc, fsp, sdb = spectrogram(t, e)
    tcp, pc = candidate_powers(t, e, -cands)   # envelope f = -optical offset

    # mode competition: a candidate "competed" if its line rose >= 20 dB out
    # of its own pre-bias ASE floor during the act-1 buildup (the losers rise
    # ~20-25 dB below the leader — they cross threshold later in the bias
    # ramp and grow slower — then die once the reservoir clamps)
    floor = pc[:, tcp < T_ON].mean(axis=1)
    grow = (tcp > T_ON) & (tcp < T_HOP)
    rise_db = 10 * np.log10(pc[:, grow].max(axis=1) / floor)
    k_end1 = int(np.argmin(np.abs(tcp - (T_HOP - 0.3e-9))))
    n_end1 = int((pc[:, k_end1] > pc[:, k_end1].max() * 1e-2).sum())

    return {"pred0": pred0, "pred1": pred1,
            "nu1": nu1, "smsr1": smsr1, "p1": p1, "w1": w1,
            "nu2": nu2, "smsr2": smsr2, "p2": p2, "w2": w2,
            "p_tot": p_tot, "p_dip": p_dip,
            "tc": tc, "fsp": fsp, "sdb": sdb, "tcp": tcp, "pc": pc,
            "rise_db": rise_db, "k_end1": k_end1, "n_end1": n_end1}


def check(m: dict, geom: dict) -> None:
    """Print the story and assert every claim the docstring makes."""
    fsr_a, cands = geom["fsr_a"], geom["cands"]
    pred0, pred1 = m["pred0"], m["pred1"]
    p1 = m["p1"]

    print(f"act 1 (heater off):  line {m['nu1']/1e9:+7.2f} GHz "
          f"(predicted {pred0['nu_off']/1e9:+.2f}), SMSR {m['smsr1']:.1f} dB, "
          f"P = {p1*1e3:.3f} mW (reservoir {analytic_p_out(pred0['t1t2'])*1e3:.3f})")
    print(f"act 2 (heater {geom['p_hop']*1e3:.3f} mW): line {m['nu2']/1e9:+7.2f} GHz "
          f"(predicted {pred1['nu_off']/1e9:+.2f}), SMSR {m['smsr2']:.1f} dB, "
          f"P = {m['p2']*1e3:.3f} mW (reservoir {analytic_p_out(pred1['t1t2'])*1e3:.3f})")
    print(f"mode hop: P_out dips to {m['p_dip']*1e3:.3f} mW "
          f"({m['p_dip']/p1:.1%} of act 1) while the modes swap")
    print("competition: candidate rises out of their ASE floors = "
          + ", ".join(f"{c/1e9:+.0f} GHz: {r:.0f} dB"
                      for c, r in zip(cands, m["rise_db"]))
          + f" -> {m['n_end1']} line within 20 dB at "
          f"t = {m['tcp'][m['k_end1']]*1e9:.2f} ns")

    assert abs(m["nu1"] - pred0["nu_off"]) < 2e9
    assert abs(m["nu2"] - (-fsr_a)) < 2.5e9, "did not hop one FSR_A"
    assert abs(m["nu2"] - pred1["nu_off"]) < 2.5e9
    assert m["smsr1"] > 18.0 and m["smsr2"] > 18.0, "winner not dominant"
    assert abs(p1 - analytic_p_out(pred0["t1t2"])) < 0.12 * p1
    assert abs(m["p2"] - analytic_p_out(pred1["t1t2"])) < 0.12 * m["p2"]
    assert m["p_dip"] < 0.5 * p1, "no visible power dip during the mode hop"
    assert int((m["rise_db"] > 20.0).sum()) >= 3, \
        "no visible mode competition during buildup"
    assert m["n_end1"] == 1, "competition did not resolve to one line"
    print("ALL TESTBENCH CHECKS PASSED")


def plot(t, e, geom: dict, m: dict) -> None:
    fsr_a, d_fsr, g0 = geom["fsr_a"], geom["d_fsr"], geom["g0"]
    p_hop, cands = geom["p_hop"], geom["cands"]

    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.25, 1])

    ax = fig.add_subplot(gs[0, 0])
    dnu = m["pred0"]["dnu"]
    ax.plot(dnu / 1e9, comb_product(dnu, 0.0), c="tab:blue", lw=0.9,
            label="comb product, heater off")
    ax.plot(dnu / 1e9, comb_product(dnu, p_hop), c="tab:red", lw=0.9,
            alpha=0.7, label=f"heater {p_hop*1e3:.2f} mW")
    ax.axhline(1 / (g0 * R_FB), c="k", ls=":", lw=0.8,
               label="threshold at G0 (unsaturated)")
    ax.set_xlim(-85, 85)
    ax.set_xlabel("optical offset from 1310 nm [GHz]")
    ax.set_ylabel("|T1·T2|")
    ax.set_title("the Vernier selector before/after the heater step",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    p_tot = m["p_tot"]
    ax.plot(t * 1e9, p_tot * 1e3, c="tab:orange", lw=0.8)
    ax.axvline(T_ON * 1e9, c="gray", lw=0.8, ls=":")
    ax.axvline(T_HOP * 1e9, c="tab:red", lw=0.8, ls="--")
    ax.annotate("bias on", (T_ON * 1e9, 0.92 * p_tot.max() * 1e3), fontsize=7)
    ax.annotate("heater steps", (T_HOP * 1e9 + 0.1, 0.92 * p_tot.max() * 1e3),
                fontsize=7, color="tab:red")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("P_out [mW]")
    ax.set_title("one continuous run: turn-on from ASE, lase, hop", fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    pm = ax.pcolormesh(m["tc"] * 1e9, -m["fsp"] / 1e9, m["sdb"].T, cmap="magma",
                       vmin=-55, vmax=0, shading="auto")
    ax.axvline(T_HOP * 1e9, c="w", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(-85, 85)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("optical offset [GHz]")
    ax.set_title("the whole story: candidates rise out of the ASE floor, the "
                 "winner clamps the gain; the heater step kills its loop gain "
                 f"and the laser hops {fsr_a/1e9:.1f} GHz", fontsize=9)
    fig.colorbar(pm, ax=ax, label="PSD [dB]", pad=0.01)

    ax = fig.add_subplot(gs[2, 0])
    tcp, pc = m["tcp"], m["pc"]
    for i, (fc, c_) in enumerate(zip(cands, ("tab:blue", "tab:orange",
                                             "tab:green", "tab:red",
                                             "tab:purple"))):
        ax.semilogy(tcp * 1e9, pc[i] / pc.max(), c=c_, lw=1,
                    label=f"{fc/1e9:+.0f} GHz")
    ax.axvline(T_HOP * 1e9, c="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(1e-7, 2)
    ax.set_xlabel("time [ns]"); ax.set_ylabel("line power (norm.)")
    ax.set_title("per-candidate power: competition, winner, swap", fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    for w, c_, lbl in ((m["w1"], "tab:blue", "act 1 (heater off)"),
                       (m["w2"], "tab:red", f"act 2 ({p_hop*1e3:.2f} mW)")):
        f, s = spectrum(e[w], DT_SAVE)
        ax.plot(-f / 1e9, 10 * np.log10(s / s.max() + 1e-16), c=c_, lw=0.9,
                label=lbl)
    ax.set_xlim(-85, 85); ax.set_ylim(-60, 3)
    ax.set_xlabel("optical offset [GHz]"); ax.set_ylabel("PSD [dB]")
    ax.set_title(f"settled spectra: one Vernier hop = FSR_A = "
                 f"{fsr_a/1e9:.1f} GHz for {p_hop*1e3:.2f} mW of heat "
                 f"(x{fsr_a/d_fsr:.0f} lever)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Vernier ring laser from ASE noise — mode competition, "
                 "winner-takes-all, and a live mode hop (Verilog-A, circulax)",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


def main() -> int:
    geom = report_design()
    t, e = run_experiment(geom["p_hop"])   # one continuous run: turn-on, lase, hop
    m = measure(t, e, geom)
    check(m, geom)
    plot(t, e, geom, m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
