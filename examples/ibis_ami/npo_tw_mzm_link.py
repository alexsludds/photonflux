#!/usr/bin/env python3
"""Near-package optics (NPO) at 53.125 GBd PAM4, IBIS-AMI SerDes on both ends.

The host SerDes and the optical module sit on the same board, ~18 cm apart:

    pf_tx (IBIS-AMI, FFE) -> SerDes package -> BGA/via -> 10 dB PCB trace
      -> via -> LGA interposer -> module substrate -> driver input (100 Ohm)
      -> driver: CTLE + VGA + limiting output stage, back-terminated
      -> flip-chip bumps -> traveling-wave Si MZM (two PN arms, GSSG push-pull,
         loaded line terminated on chip) -> TP2 (TDECQ)
      -> 500 m SMF -> PD -> TIA -> module substrate -> LGA -> via
      -> 10 dB PCB trace -> via/BGA -> SerDes package -> pf_rx (IBIS-AMI)

How each piece is modeled:

* **Passive RF channels** (``photonflux.rfnet``): ABCD cascades of TEM RLGC
  lines (sqrt(f) skin loss with its internal inductance, causal
  Djordjevic-Sarkar dielectric), die pads, bumps, BGA balls and vias, the
  LGA interposer, and the source and load impedances of the chips on each
  end. SerDes, driver-input and TIA-output pads hide their capacitance in a
  pi-coil ladder (as T-coils do on real 112G I/O); the driver *output* pad
  is a bare capacitance in the circuit, next to the back-termination. Each channel is the terminated transfer V(load) / V(source EMF), so
  every impedance step and every re-reflection (including the 2.3 ns
  round trip of the board trace) is in its impulse response. The PCB trace
  alone is sized for 10.0 dB at 26.5625 GHz; packages and transitions add
  their own loss and ripple on top. Differential links are solved in their
  odd mode (50 Ohm single-ended = 100 Ohm differential): the transfer
  function of the differential voltage is that of the odd-mode circuit.
* **Driver** (numpy, memoryless where real drivers are): input-referred
  noise, CTLE (1 zero, 2 poles, peaking set here), pre-driver pole, a gain
  control that sets the output swing, and a tanh-limiting output stage
  (Thevenin EMF; its back-termination, pad capacitance and bumps are
  circuit elements below).
* **Driver output network, TW-MZM, optics, PD, TIA** (circulax transient):
  per-arm back-termination, driver pad, flip-chip bump, MZM pad, then the
  ``pn_phase_shifter_tw`` part: an RLC transmission-line electrode loaded
  every 100 um by its own depletion junction (sqrt law, series resistance),
  RF/optical walk-off from per-section optical transit, terminated on chip
  to the junction bias. Reflections between driver and electrode come out
  of the circuit. Two 50/50 couplers and a lambda/4 waveguide offset make
  the MZM, at quadrature.
* **IBIS-AMI**: pf_tx runs AMI_Init on the Tx channel and AMI_GetWave on the
  Gray-coded PRBS13Q stimulus (with Tx jitter); pf_rx (CTLE, AGC, 11-tap
  FFE, 1-tap DFE, Mueller-Muller CDR) runs on the received waveform through
  ``serdes.analyze_waveform``: counted BER from its own recovered clock, and
  the statistical flow on the data-aided channel estimate.

Adaptation: the driver's CTLE peaking (0.5 dB codes, 0-12 dB) and pf_tx's
c(-1), c(+1), c(+2) taps are chosen to minimize TDECQ at TP2 (IEEE 802.3cd
method: 4th-order Bessel-Thomson at 26.5625 GHz, 5-tap reference FFE, SER
4.8e-4), guarded against over-equalization: where the reference FFE has to
low-pass the signal (C_eq < 1) the search scores TDECQ - 10 log10(C_eq),
the quantity 802.3 limits alongside TDECQ. A
transient costs minutes, so the search runs on a Wiener surrogate of the
optical core (linear FIR from the driver EMF to the optical power, then a
cubic for the MZM / junction compression) fitted to a transient, and every
reported TDECQ comes from a full circuit transient at the chosen setting;
the surrogate is refitted on that transient until the two agree.

    python examples/ibis_ami/npo_tw_mzm_link.py          # ~10 min
    python examples/ibis_ami/npo_tw_mzm_link.py --quick  # shorter search, no baseline Rx run
        -> out/ibis_ami_npo_tw_mzm.png, out/ibis_ami_npo_tw_mzm.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_toeplitz  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

from _common import OUT, REPO, ensure_built, fmt_ber, model, save  # noqa: E402

sys.path.insert(0, str(REPO / "webapp"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import simulate  # noqa: E402

from photonflux import rfnet, serdes, tdec  # noqa: E402
from photonflux.signals import pam4_gray, prbs  # noqa: E402

BAUD = 53.125e9
UI = 1.0 / BAUD
F_NYQ = BAUD / 2
NSPUI = 32
TS = UI / NSPUI
NFFT = 1 << 15                       # 18.8 ns: channel + every echo
F = rfnet.grid(TS, NFFT)
N_PRE = 384                          # UI of pattern before the scored period
PERIOD = 8191                        # PRBS13Q

# --- host SerDes (IBIS-AMI) -------------------------------------------------
TX_EMF_VPPD = 2.0        # pf_tx's +/-0.5 V stimulus -> 2 Vppd EMF (1 Vppd launched)
TX_JITTER = dict(rj_ui=0.01, dj_ui=0.02)
TX_BW = 60e9             # pf_tx internal pre-driver poles; the pad C is in the channel
RX_PARAMS = dict(ctle_fp1=26.5e9, ffe_mode=2, ffe_ntaps=11, ffe_pre=3,
                 dfe_ntaps=1, cdr_mode=2, cdr_kp=2e-3)
RX_NOISE_NV = 5.0        # SerDes Rx input-referred noise, nV/rtHz
RX_IGNORE = 2500         # symbols the Rx gets to adapt before counting

# --- driver ------------------------------------------------------------------
DRV = dict(fp1=30e9, fp2_mult=2.0, pre_bw=55e9, vsat_d=3.0, target=0.7,
           noise_nv=4.0)  # vsat_d: differential EMF limit (V); target: AGC peak / vsat
PEAK_MAX_DB = 12.0

# --- TW-MZM (silicon depletion PN, two arms, push-pull) ----------------------
VCM = 2.0                # junction reverse bias = driver output common mode
PN = dict(length_m=3.5e-3, vpil_vcm=1.5, vref=2.0, vbi=0.85, loss_dB_cm=10.0,
          dloss_dB_cm=1.0, cj0_ff_um=0.3, rs_ohm_mm=10.0, z0_ohm=60.0,
          n_rf=2.4, rf_loss_dB_cm=2.0, polarity=1.0, neff=2.4, n_group=4.0,
          center_wavelength_nm=1311.0, wavelength_nm=1311.0)
C_DRV_OUT = 80e-15       # driver output pad + output devices + ESD, per side
L_BUMP = 30e-12          # flip-chip bump, driver -> PIC
C_MZM_PAD = 20e-15       # electrode launch pad

# --- optics / receiver front end ---------------------------------------------
LASER_W = 0.016          # 12 dBm into the modulator
COUPLING_DB = 3.0        # PIC -> fiber coupling + connectors
FIBER_KM = 0.5
TIA = dict(gain_ohm=500.0, f3db=35e9)
TIA_NOISE = 16e-12       # input-referred, A/rtHz


# ---------------------------------------------------------------------------
# passive channels
# ---------------------------------------------------------------------------

def _materials():
    return dict(
        pcb=rfnet.line_rlgc(F, z0=50.0, er=3.6, tand=0.004, cond_db_m_rtghz=7.0,
                            r_dc=8.0),        # 5 mil x 0.5 oz copper
        pkg=rfnet.line_rlgc(F, z0=45.0, er=3.4, tand=0.010, cond_db_m_rtghz=14.0),
        sub=rfnet.line_rlgc(F, z0=48.0, er=3.4, tand=0.010, cond_db_m_rtghz=14.0),
        lga=rfnet.line_rlgc(F, z0=58.0, er=1.0),
        via=rfnet.line_rlgc(F, z0=45.0, er=3.6, tand=0.004),
    )


def _via(m, stub_mm=0.25):
    """Through-via (2 mm barrel, 45 Ohm) with the backdrill's residual stub."""
    return rfnet.Net.line(m["via"], 2e-3) @ rfnet.Net.open_stub(m["via"], stub_mm * 1e-3)


def _pad(c: float, z: float = 50.0) -> rfnet.Net:
    """A die pad + ESD capacitance ``c`` distributed in a two-section
    pi-coil ladder (C/4, L/2, C/2, L/2, C/4 with L = z^2 c): an artificial
    line of impedance ``z``, the usual way a 100+ GHz-fT I/O hides its
    capacitance (a bridged T-coil does the same in one stage)."""
    n = rfnet.Net
    half = n.series_l(F, z * z * c / 2)
    return (n.shunt_c(F, c / 4) @ half @ n.shunt_c(F, c / 2) @ half
            @ n.shunt_c(F, c / 4))


def channels() -> dict:
    """Tx (SerDes -> driver) and Rx (TIA -> SerDes) channels, odd mode."""
    m = _materials()
    n = rfnet.Net
    trace_m = rfnet.length_for_loss(m["pcb"], 10.0, F_NYQ)
    pcb = n.line(m["pcb"], trace_m)
    host = (n.series_l(F, 30e-12) @ n.line(m["pkg"], 10e-3)        # bump, package
            @ n.shunt_c(F, 70e-15) @ n.series_l(F, 120e-12)         # BGA ball
            @ _via(m))
    module = (_via(m) @ n.line(m["lga"], 1.2e-3) @ n.shunt_c(F, 40e-15)   # LGA
              @ n.line(m["sub"], 5e-3) @ n.series_l(F, 30e-12))     # substrate, bump
    tx = dict(net=_pad(150e-15) @ host @ pcb @ module @ _pad(100e-15),
              zs=50.0, zl=50.0, name="SerDes Tx -> driver")
    rx = dict(net=_pad(80e-15) @ module.flipped() @ pcb @ host.flipped() @ _pad(200e-15),
              zs=50.0, zl=50.0, name="TIA -> SerDes Rx")
    for ch in (tx, rx):
        ch["H"] = ch["net"].transfer(ch["zs"], ch["zl"])
        ch["h"] = rfnet.impulse(F, ch["H"], TS, NFFT)
        ch["s"] = ch["net"].s(50.0)
    k = int(np.argmin(np.abs(F - F_NYQ)))
    summary = {"trace_cm": 100 * trace_m,
               "trace_delay_ns": 1e9 * m["pcb"].delay_s_m(F_NYQ) * trace_m,
               "trace_il_db": -20 * np.log10(abs(n.line(m["pcb"], trace_m).s()[1, 0, k]))}
    for key, ch in (("tx", tx), ("rx", rx)):
        s = ch["s"][:, :, k]
        summary[key] = {"il_db": -20 * np.log10(abs(s[1, 0])),
                        "rl_in_db": -20 * np.log10(abs(s[0, 0])),
                        "rl_out_db": -20 * np.log10(abs(s[1, 1]))}
    return {"tx": tx, "rx": rx, "summary": summary, "pcb": m["pcb"]}


# ---------------------------------------------------------------------------
# TW electrode, analytically (the same RLGC + junction as the circuit part)
# ---------------------------------------------------------------------------

def electrode(f: np.ndarray, r_bt: float, r_t: float) -> dict:
    """Loaded-line Zc, gamma and the EO response (optical phase per volt of
    driver EMF, normalized at DC) of one arm with its launch network."""
    w = 2 * np.pi * np.maximum(f, 1.0)
    c0 = rfnet.C0
    lp = PN["z0_ohm"] * PN["n_rf"] / c0
    cp = PN["n_rf"] / (PN["z0_ohm"] * c0)
    rp = 2 * PN["z0_ohm"] * PN["rf_loss_dB_cm"] * 100 / rfnet.NP_DB
    cj = PN["cj0_ff_um"] * 1e-9 / np.sqrt(1 + VCM / PN["vbi"])     # F/m at bias
    tau_j = PN["rs_ohm_mm"] * 1e-3 * cj
    yj = 1j * w * cj / (1 + 1j * w * tau_j)
    z, y = rp + 1j * w * lp, 1j * w * cp + yj
    g, zc = np.sqrt(z * y), np.sqrt(z / y)
    L = PN["length_m"]
    gl = (r_t - zc) / (r_t + zc) * np.exp(-2 * g * L)
    z_line = zc * (1 + gl) / (1 - gl)
    launch = (rfnet.Net.shunt_c(f, C_DRV_OUT) @ rfnet.Net.series_l(f, L_BUMP)
              @ rfnet.Net.shunt_c(f, C_MZM_PAD))
    v0 = launch.transfer(r_bt, z_line)
    vp = v0 / (1 + gl)
    vm = gl * vp
    k_opt = w * PN["n_group"] / c0

    def integ(a):
        return (1 - np.exp(-a * L)) / (a * L)
    eo = (vp * integ(g - 1j * k_opt) + vm * integ(-g - 1j * k_opt)) / (1 + 1j * w * tau_j)
    z_load = launch.zin(z_line)
    return {"zc": zc, "gamma": g, "eo": eo / abs(eo[0]),
            "s11_drv": (z_load - r_bt) / (z_load + r_bt),
            "gamma_term": (r_t - zc) / (r_t + zc)}


def termination() -> float:
    """Back- and far-end termination: the loaded line's |Zc| at f_nyq / 2,
    rounded to the ohm (what a designer trims the on-chip resistors to)."""
    zc = electrode(np.array([F_NYQ / 2]), 40.0, 40.0)["zc"]
    return float(np.round(abs(zc[0])))


# ---------------------------------------------------------------------------
# pattern, SerDes Tx, driver
# ---------------------------------------------------------------------------

def pattern() -> tuple[np.ndarray, np.ndarray]:
    """(period, record): PRBS13Q (Gray, rotated so OMA_outer's runs fall
    inside), and N_PRE of its tail followed by one full period."""
    per = np.roll(pam4_gray(np.tile(prbs(13), 2)), 64).astype(int)
    return per, np.concatenate([per[-N_PRE:], per])


def _filt(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a transfer sampled on F (causal impulse, linear convolution)."""
    h = rfnet.impulse(F, H, TS, NFFT)
    return fftconvolve(x, h)[: len(x)] * TS


def serdes_tx(sym: np.ndarray, taps: dict, ch: dict) -> np.ndarray:
    """pf_tx AMI_Init (on the Tx channel) + AMI_GetWave -> differential EMF."""
    a = (2.0 * sym - 3.0) / 3.0
    x = serdes.stimulus(a, NSPUI, seed=11, **TX_JITTER)
    with model("pf_tx", tx_bw=TX_BW, tx_swing=1.0, **taps) as tx:
        if "Modulation" in tx.params:
            tx.params["Modulation"] = "PAM4"
        tx.init(ch["h"], TS, UI)
        y = tx.getwave(x, block_bits=512).wave
    return TX_EMF_VPPD * y


def ctle_h(peak_db: float) -> np.ndarray:
    s = 2j * np.pi * F
    wp1 = 2 * np.pi * DRV["fp1"]
    k = 10 ** (peak_db / 20)
    return (1 / k) * (1 + s * k / wp1) / ((1 + s / wp1) * (1 + s / (DRV["fp2_mult"] * wp1)))


def driver(v_in: np.ndarray, peak_db: float, seed: int = 3) -> tuple[np.ndarray, float]:
    """Driver input voltage (diff) -> output EMF (diff). Returns (emf, gain)."""
    rng = np.random.default_rng(seed)
    v = v_in + DRV["noise_nv"] * 1e-9 * np.sqrt(0.5 / TS) * rng.standard_normal(len(v_in))
    s = 2j * np.pi * F
    v = _filt(v, ctle_h(peak_db) / (1 + s / (2 * np.pi * DRV["pre_bw"])))
    pk = np.percentile(np.abs(v[N_PRE * NSPUI:]), 99.9)
    gain = DRV["target"] * DRV["vsat_d"] / max(pk, 1e-9)
    return DRV["vsat_d"] * np.tanh(gain * v / DRV["vsat_d"]), gain


def drive(sym, peak_db, taps, ch) -> dict:
    v_tx = serdes_tx(sym, taps, ch)
    v_in = fftconvolve(v_tx, ch["h"])[: len(v_tx)] * TS
    emf, gain = driver(v_in, peak_db)
    return {"v_tx": v_tx, "v_in": v_in, "emf": emf, "gain": gain}


# ---------------------------------------------------------------------------
# circuit: driver output network, TW-MZM, optics, PD, TIA
# ---------------------------------------------------------------------------

def _pwl(t, v) -> str:
    return "\n".join(f"{a:.7e} {b:.6f}" for a, b in zip(t, v))


def schematic(emf: np.ndarray, r_term: float) -> dict:
    t = np.arange(len(emf)) * TS
    inst, wires = {}, []

    def add(name, typ, **s):
        inst[name] = {"type": typ, "x": 0, "y": 0, "rot": 0, "settings": s}

    def w(a, b):
        wires.append({"from": a, "to": b})

    add("LAS", "cw_laser", power=LASER_W, wavelength_nm=1311.0)
    add("GL", "ground")
    w("LAS,p1", "DC1,p1"), w("LAS,p2", "GL,p1")
    add("DC1", "dir_coupler", coupling=0.5)
    add("DC2", "dir_coupler", coupling=0.5)
    add("OT1", "opt_term"), w("DC1,p2", "OT1,p1")
    add("OT2", "opt_term"), w("DC2,p3", "OT2,p1")
    add("VT", "vdc", V=VCM), add("GVT", "ground"), w("VT,p2", "GVT,p1")
    quarter = PN["wavelength_nm"] * 1e-9 / (4 * PN["neff"])       # quadrature
    for arm, sgn, port in (("A", 1, "p3"), ("B", -1, "p4")):
        add(f"PN_{arm}", "pn_phase_shifter_tw", **PN)
        add(f"WG_{arm}", "waveguide", length_m=20e-6 + (quarter if arm == "B" else 0.0),
            loss_dB_cm=2.0, neff=PN["neff"], n_group=PN["n_group"],
            center_wavelength_nm=1311.0, wavelength_nm=1311.0)
        w(f"DC1,{port}", f"PN_{arm},p1"), w(f"PN_{arm},p2", f"WG_{arm},p1")
        w(f"WG_{arm},p2", f"DC2,{'p1' if arm == 'A' else 'p2'}")
        # driver output: EMF -> back-termination -> pad C -> bump -> MZM pad
        add(f"E{arm}", "vpwl", data=_pwl(t, VCM + sgn * 0.5 * emf))
        add(f"GE{arm}", "ground"), w(f"E{arm},p2", f"GE{arm},p1")
        add(f"RBT{arm}", "resistor", R=r_term), w(f"E{arm},p1", f"RBT{arm},p1")
        add(f"CD{arm}", "capacitor", C=C_DRV_OUT), add(f"GCD{arm}", "ground")
        w(f"RBT{arm},p2", f"CD{arm},p1"), w(f"CD{arm},p2", f"GCD{arm},p1")
        add(f"LB{arm}", "inductor", L=L_BUMP), w(f"RBT{arm},p2", f"LB{arm},p1")
        add(f"CP{arm}", "capacitor", C=C_MZM_PAD), add(f"GCP{arm}", "ground")
        w(f"LB{arm},p2", f"CP{arm},p1"), w(f"CP{arm},p2", f"GCP{arm},p1")
        w(f"LB{arm},p2", f"PN_{arm},e_in")
        add(f"RT{arm}", "resistor", R=r_term)
        w(f"PN_{arm},e_out", f"RT{arm},p1"), w(f"RT{arm},p2", "VT,p1")
        add(f"GR{arm}", "ground"), w(f"PN_{arm},ref", f"GR{arm},p1")
    add("ATT", "opt_atten", atten_db=COUPLING_DB), w("DC2,p4", "ATT,p1")
    add("FIB", "fiber_cd", length_km=FIBER_KM, lambda_nm=1311.0, lambda0_nm=1310.0,
        S_ps=0.092, atten_db_km=0.35, fit_bw=80e9, n_poles=24)
    w("ATT,p2", "FIB,p1")
    add("PD", "photodiode", R=0.8, Cj=3e-14)
    add("GPO", "ground"), add("GPA", "ground")
    w("FIB,p2", "PD,po_p"), w("PD,po_n", "GPO,p1"), w("PD,cat", "GPA,p1")
    add("TIA", "tia", gain_ohm=TIA["gain_ohm"], f3db=TIA["f3db"], vmax=0.0, in_noise=0.0)
    w("PD,an", "TIA,inp")
    add("RL", "resistor", R=1e6), add("GRL", "ground")
    w("TIA,out", "RL,p1"), w("RL,p2", "GRL,p1")
    # TP2: in the fiber, after the PIC-to-fiber coupling
    probes = [{"name": "ptx", "at": "ATT,p2"}, {"name": "vtia", "at": "TIA,out"},
              {"name": "vlaunch", "at": "PN_A,e_in"}, {"name": "vend", "at": "PN_A,e_out"}]
    return {"instances": inst, "wires": wires, "probes": probes,
            "globals": {"baud": BAUD}}


def transient(emf: np.ndarray, r_term: float) -> dict:
    n = len(emf)
    t_stop = (n - 1) * TS
    simulate.MAX_FLOW_POINTS = max(simulate.MAX_FLOW_POINTS, n + 1)
    res = simulate.run({"schematic": schematic(emf, r_term), "analysis": {
        "mode": "transient", "t_stop": t_stop, "points": n, "max_points": n,
        "dtmax": UI / 32}})
    if not res.get("ok"):
        raise RuntimeError(res.get("error"))
    x = np.asarray(res["x"], float)
    tu = np.arange(n) * TS
    return {tr["name"]: np.interp(tu, x, np.asarray(tr["values"], float))
            for tr in res["traces"]}


# ---------------------------------------------------------------------------
# TDECQ and the surrogate
# ---------------------------------------------------------------------------

def levels(p_mw: np.ndarray, per: np.ndarray) -> np.ndarray:
    """The four PAM4 levels (mW) at TP2: through the reference receiver, at
    the best sampling phase, the mean of each symbol's samples (neighbours'
    ISI averages out over the PRBS)."""
    q = tdec.reference_receiver(p_mw[-PERIOD * NSPUI:], TS, BAUD)
    x = q.reshape(PERIOD, NSPUI)
    A = np.conj(np.fft.fft(per - 1.5))
    best = (-np.inf, 0, 0)
    for ph in range(NSPUI):
        c = np.fft.ifft(np.fft.fft(x[:, ph] - x[:, ph].mean()) * A).real
        k = int(np.argmax(np.abs(c)))
        if abs(c[k]) > best[0]:
            best = (abs(c[k]), ph, k)
    _, ph, k = best
    s = np.roll(per, k)                  # x[n, ph] carries symbol s[n]
    lv = np.array([x[8:, ph][s[8:] == v].mean() for v in range(4)])
    return lv if lv[3] > lv[0] else lv[::-1]


def tdecq(p_mw: np.ndarray, per: np.ndarray, dec: int = 1) -> dict:
    """TDECQ (802.3cd reference receiver + 5-tap FFE) of the scored period,
    with OMA_outer, ER and RLM."""
    p = p_mw[-PERIOD * NSPUI:][::dec]
    m = tdec.measure_pam4(p, TS * dec, BAUD, symbols=per, strict=False)
    lv = levels(p_mw, per)
    vm = (lv[0] + lv[3]) / 2
    es1, es2 = (lv[1] - vm) / (lv[0] - vm), (lv[2] - vm) / (lv[3] - vm)
    return {"tdecq_outer": float(m.get("tdecq_outer", np.nan)),
            "oma_outer": float(m.get("oma_outer", np.nan)),
            "er_db": float(m.get("extinction_ratio_outer", np.nan)),
            "rlm": float(min(3 * es1, 3 * es2, 2 - 3 * es1, 2 - 3 * es2)),
            "ceq": float(m.get("ceq", np.nan)),
            "ffe_taps": [float(c) for c in m.get("ffe_taps", [])]} | {"score": score(m)}


def score(m) -> float:
    """What the search minimizes: TDECQ - 10 log10(C_eq) when the reference
    FFE has to low-pass the signal (C_eq < 1), TDECQ otherwise. Plain TDECQ
    rewards over-equalizing: an overshooting drive that the reference FFE
    smooths back scores well while ER and OMA collapse, which is why 802.3
    limits TDECQ - 10 log10(C_eq) as well."""
    t, c = float(m.get("tdecq_outer", np.nan)), float(m.get("ceq", np.nan))
    return t - 10 * np.log10(min(c, 1.0))


class Surrogate:
    """Wiener model of the optical core: ptx = poly3(h * emf), fitted at
    16 samples/UI on a transient (Wiener-Hopf normal equations)."""

    DEC, PRE, SPAN = 2, 16, 14 * 16      # samples at 16/UI

    def __init__(self, emf: np.ndarray, ptx: np.ndarray):
        u, y = emf[::self.DEC], ptx[::self.DEC]
        s0 = N_PRE * NSPUI // self.DEC
        u0, y0 = u[s0:] - u[s0:].mean(), y[s0:] - y[s0:].mean()
        n = len(u0)
        nf = 1 << int(np.ceil(np.log2(2 * n)))
        U, Y = np.fft.rfft(u0, nf), np.fft.rfft(y0, nf)
        ruu = np.fft.irfft(U * U.conj(), nf)[: self.SPAN] / n
        ruy = np.fft.irfft(Y * U.conj(), nf) / n              # r[k] = E[y(t+k)u(t)]
        rhs = np.concatenate([ruy[-self.PRE:], ruy[: self.SPAN - self.PRE]])
        self.h = solve_toeplitz(ruu, rhs)                     # lags -PRE..SPAN-PRE-1
        z = self._lin(u)
        self.poly = np.polyfit(z[s0:], y[s0:], 3)
        fit = np.polyval(self.poly, z[s0:])
        self.residual_rel = float(np.std(y[s0:] - fit) / np.std(y[s0:]))

    def _lin(self, u):
        z = fftconvolve(u, self.h)
        return z[self.PRE: self.PRE + len(u)]

    def __call__(self, emf: np.ndarray) -> np.ndarray:
        """Optical power (mW) at 16 samples/UI."""
        return np.polyval(self.poly, self._lin(emf[::self.DEC]))

    def response(self, f: np.ndarray) -> np.ndarray:
        """Small-signal EMF -> optical power transfer, normalized at DC."""
        ts = TS * self.DEC
        k = np.arange(len(self.h)) - self.PRE
        H = np.exp(-2j * np.pi * np.outer(f, k * ts)) @ self.h
        return H / abs(H[0])


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

TAP_KEYS = ("tx_tap_m1", "tx_tap_p1", "tx_tap_p2")
BOUNDS = [(0.0, PEAK_MAX_DB), (-0.25, 0.0), (-0.35, 0.0), (-0.15, 0.15)]


def _taps(x) -> dict:
    return {k: round(float(v), 3) for k, v in zip(TAP_KEYS, x[1:])}


def adapt(sur: Surrogate, per, rec, ch, x0, max_iter: int) -> tuple[np.ndarray, list, float]:
    """CTLE peaking x Tx FFE minimizing the surrogate's TDECQ: a peaking
    sweep at the starting taps, then Nelder-Mead on all four."""
    cache = {}

    def f(x):
        key = (round(float(x[0]) * 2) / 2,) + tuple(_taps(x).values())  # 0.5 dB CTLE codes
        if key not in cache:
            d = drive(rec, key[0], _taps(x), ch)
            v = tdecq(np.repeat(sur(d["emf"]), sur.DEC), per, dec=sur.DEC)["score"]
            cache[key] = v if np.isfinite(v) else 30.0
        return cache[key]

    sweep = [(pk, f(np.r_[pk, x0[1:]])) for pk in np.arange(0.0, PEAK_MAX_DB + 0.1, 2.0)]
    start = np.r_[min(sweep, key=lambda r: r[1])[0], x0[1:]]
    minimize(f, start, method="Nelder-Mead", bounds=BOUNDS,
                 options={"maxfev": max_iter, "xatol": 0.02, "fatol": 0.01,
                          "initial_simplex": [start] + [
                              start + np.eye(4)[i] * step
                              * (1 if start[i] - lo < hi - start[i] else -1)
                              for i, (step, (lo, hi)) in enumerate(
                                  zip((2.0, 0.06, 0.08, 0.05), BOUNDS))]})
    # TDECQ is undefined (NaN, scored 30 dB) at some over-equalized corners,
    # so take the best setting the search actually scored
    key, val = min(cache.items(), key=lambda kv: kv[1])
    return np.array(key), sweep, val


def run_point(label, x, per, rec, ch, r_term) -> dict:
    t0 = time.time()
    d = drive(rec, float(x[0]), _taps(x), ch)
    tr = transient(d["emf"], r_term)
    out = {"label": label, "peaking_db": float(x[0]), "taps": _taps(x),
           **d, **tr}
    out["tdecq"] = tdecq(tr["ptx"], per)
    out["sec"] = time.time() - t0
    q = out["tdecq"]
    print(f"  {label}: CTLE {x[0]:.1f} dB, taps {_taps(x)} -> TDECQ {q['tdecq_outer']:.2f} dB "
          f"(C_eq {q['ceq']:.3f}, TDECQ - 10log10(min(C_eq, 1)) {q['score']:.2f} dB), "
          f"OMA_outer {10 * np.log10(q['oma_outer']):.2f} dBm, ER {q['er_db']:.2f} dB, "
          f"RLM {q['rlm']:.3f} ({out['sec']:.0f} s transient)")
    return out


def receive(pt: dict, per, rec, ch) -> serdes.WaveformAnalysis:
    """TIA output (+ its noise) -> Rx channel -> pf_rx."""
    rng = np.random.default_rng(5)
    i_n = TIA_NOISE * np.sqrt(0.5 / TS) * rng.standard_normal(len(pt["vtia"]))
    s = 2j * np.pi * F
    tau = 0.6436 / (2 * np.pi * TIA["f3db"])
    v_n = _filt(i_n, TIA["gain_ohm"] / (1 + s * tau) ** 2)
    # the TIA inverts; its differential outputs are wired swapped
    v_emf = -(pt["vtia"] - pt["vtia"].mean()) - v_n
    y = fftconvolve(v_emf, ch["h"])[: len(v_emf)] * TS
    a = (2.0 * rec - 3.0) / 3.0
    # the link's bulk latency (two 1.2 ns traces, fiber, circuit) is longer
    # than the channel estimate's window: drop it, keeping 4 UI of lead
    yu = y.reshape(-1, NSPUI).mean(axis=1)
    lag = np.arange(400)
    xc = [np.dot(yu[k + N_PRE: k + N_PRE + 4000], a[N_PRE: N_PRE + 4000]) for k in lag]
    j = int(np.argmax(np.abs(xc)))
    d = max(j - 4, 0)
    y, a = np.sign(xc[j]) * y[d * NSPUI:], a[: len(a) - d]
    rx = model("pf_rx", **RX_PARAMS)
    return serdes.analyze_waveform(rx, y, a, UI, TS, modulation="PAM4",
                                   noise_rms=RX_NOISE_NV * 1e-9 * np.sqrt(0.5 / TS),
                                   skip=RX_IGNORE)


def eye(ax, p, per_ui=2, title=""):
    p = tdec.reference_receiver(p[-PERIOD * NSPUI:], TS, BAUD)[8 * NSPUI:]
    n = per_ui * NSPUI
    seg = p[: len(p) // n * n].reshape(-1, n)
    t = np.tile(np.arange(n) / NSPUI, seg.shape[0])
    ax.hist2d(t, seg.ravel(), bins=(n, 160), cmap="inferno", cmin=1)
    ax.set_xlabel("time [UI]")
    ax.set_ylabel("optical power [mW]")
    ax.set_title(title, fontsize=9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="shorter search; skip pf_rx on the unequalized baseline")
    args = ap.parse_args(argv)
    ensure_built()
    t_all = time.time()

    ch = channels()
    cs = ch["summary"]
    print(f"PCB trace: {cs['trace_cm']:.1f} cm, {cs['trace_il_db']:.2f} dB at "
          f"{F_NYQ / 1e9:.4g} GHz, {cs['trace_delay_ns']:.2f} ns")
    for k in ("tx", "rx"):
        c = cs[k]
        print(f"{ch[k]['name']}: SDD21 {-c['il_db']:.2f} dB, SDD11 {-c['rl_in_db']:.1f} dB, "
              f"SDD22 {-c['rl_out_db']:.1f} dB at Nyquist (packages, vias, LGA, pads included)")
    r_term = termination()
    el = electrode(F, r_term, r_term)
    k = int(np.argmin(np.abs(F - F_NYQ)))
    k1 = int(np.argmin(np.abs(F - 1e9)))
    eo_db = 20 * np.log10(np.abs(el["eo"]))
    f3 = F[np.argmax(eo_db < -3)]
    print(f"TW electrode: loaded |Zc| {abs(el['zc'][k1]):.1f} Ohm @1 GHz, "
          f"{abs(el['zc'][k]):.1f} Ohm @Nyquist -> back/far-end terminations {r_term:.0f} Ohm; "
          f"|S11| at the driver {20 * np.log10(abs(el['s11_drv'][k])):.1f} dB, "
          f"EO S21 {eo_db[k]:.2f} dB at Nyquist, -3 dB at {f3 / 1e9:.1f} GHz")

    per, rec = pattern()
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    points = []
    print("transient 1: no equalization (CTLE 0 dB, Tx FFE off)")
    base = run_point("baseline", x0, per, rec, ch["tx"], r_term)
    points.append(base)
    sur, x, history = Surrogate(base["emf"], base["ptx"]), x0, []
    for it in range(2 if args.quick else 3):
        print(f"surrogate fit on '{points[-1]['label']}': residual {sur.residual_rel:.2%} of rms; "
              "searching CTLE peaking x Tx FFE ...")
        t0 = time.time()
        x, sweep, pred = adapt(sur, per, rec, ch["tx"],
                               x if it else np.array([0, -0.05, -0.1, 0.0]),
                               40 if args.quick else 80)
        history.append({"sweep": sweep, "pred": pred, "x": x.tolist()})
        print(f"  surrogate optimum: CTLE {x[0]:.1f} dB, taps {_taps(x)}, predicted score "
              f"{pred:.2f} dB ({time.time() - t0:.0f} s)")
        pt = run_point(f"adapted #{it + 1}", x, per, rec, ch["tx"], r_term)
        pt["predicted"] = pred
        points.append(pt)
        if abs(pt["tdecq"]["score"] - pred) < 0.15:
            break
        sur = Surrogate(pt["emf"], pt["ptx"])
    best = min(points[1:], key=lambda p: p["tdecq"]["score"])

    print("IBIS-AMI receive (pf_rx) after the 10 dB Rx trace:")
    rx_runs = {}
    for pt in ([points[0]] if not args.quick else []) + [best]:
        wa = receive(pt, per, rec, ch["rx"])
        c = wa.td.counted
        rx_runs[pt["label"]] = wa
        print(f"  {pt['label']}: counted BER {fmt_ber(c['ber'])} ({c['bit_errors']} errors / "
              f"{c['bits']} bits), statistical {wa.stat.eye.bit_error_rate():.2e}, "
              f"eye heights @1e-6 "
              f"{', '.join(f'{wa.stat.eye.eye_height(1e-6, e) * 1e3:.0f}' for e in range(3))} mV, "
              f"linear-fit residual {wa.residual_rel:.1%}")

    # --- figure -----------------------------------------------------------
    fig, axs = plt.subplots(2, 4, figsize=(23, 10.5), layout="constrained")
    fg = F / 1e9
    sel = fg <= 80
    ax = axs[0, 0]
    for key, ls in (("tx", "-"), ("rx", "--")):
        s = ch[key]["s"]
        ax.plot(fg[sel], 20 * np.log10(np.abs(s[1, 0][sel])), "C0" + ls, label=f"{key} SDD21")
        ax.plot(fg[sel], 20 * np.log10(np.abs(s[0, 0][sel])), "C3" + ls, lw=0.8,
                label=f"{key} SDD11")
    trace = rfnet.Net.line(ch["pcb"], cs["trace_cm"] / 100).s()[1, 0]
    ax.plot(fg[sel], 20 * np.log10(np.abs(trace[sel])), "k:", label="PCB trace alone")
    ax.axvline(F_NYQ / 1e9, color="0.5", lw=0.6)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("dB")
    ax.set_title("host channels (100 Ohm diff): loss and return loss", fontsize=9)
    ax.legend(fontsize=7)

    ax = axs[0, 1]
    ax.plot(fg[sel], np.abs(el["zc"][sel]), "C1", label="loaded line |Zc| [Ohm]")
    ax.axhline(r_term, color="C1", ls=":", lw=0.8, label=f"terminations {r_term:.0f} Ohm")
    ax.set_ylabel("Ohm")
    ax.set_xlabel("frequency [GHz]")
    ax2 = ax.twinx()
    ax2.plot(fg[sel], eo_db[sel], "C2", label="EO response (analytic)")
    sur_db = 20 * np.log10(np.abs(sur.response(F[sel])))
    ax2.plot(fg[sel], sur_db, "C2--", label="EMF -> optical, fitted to transient")
    ax2.plot(fg[sel], 20 * np.log10(np.abs(el["s11_drv"][sel])), "C3", lw=0.8,
             label="|S11| at driver")
    ax2.set_ylim(-30, 3)
    ax2.set_ylabel("dB")
    ax.set_title("TW-MZM electrode: impedance, match, EO response", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    ax = axs[0, 2]
    for i, h in enumerate(history):
        pk, v = zip(*h["sweep"])
        ax.plot(pk, v, "o-", ms=4, label=f"surrogate #{i + 1}, peaking sweep")
    for p in points:
        ax.plot(p["peaking_db"], p["tdecq"]["score"], "k*" if p is best else "kx", ms=10)
    ax.set_xlabel("driver CTLE peaking [dB]")
    ax.set_ylabel("TDECQ - 10 log10(min(C_eq, 1)) [dB]")
    ax.set_title("adaptation (x / star: full circuit transients)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axs[0, 3]
    n0 = N_PRE * NSPUI
    tt = np.arange(64 * NSPUI) * TS * 1e12
    sl = slice(n0 + 2000 * NSPUI, n0 + 2064 * NSPUI)
    ax.plot(tt, best["v_in"][sl], lw=0.8, label="driver input (after 10 dB trace)")
    ax.plot(tt, best["emf"][sl] / 4, lw=0.8, label="driver EMF / 4")
    ax.plot(tt, best["vlaunch"][sl] - VCM, lw=0.8, label="electrode launch, arm A")
    ax.plot(tt, best["vend"][sl] - VCM, lw=0.8, label="electrode far end, arm A")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("V (differential / per arm)")
    ax.set_title("adapted drive path", fontsize=9)
    ax.legend(fontsize=7)

    eye(axs[1, 0], points[0]["ptx"], title=f"TP2, no EQ: TDECQ {points[0]['tdecq']['tdecq_outer']:.2f} dB")
    eye(axs[1, 1], best["ptx"], title=(
        f"TP2, adapted: TDECQ {best['tdecq']['tdecq_outer']:.2f} dB, "
        f"OMA_outer {10 * np.log10(best['tdecq']['oma_outer']):.1f} dBm, "
        f"ER {best['tdecq']['er_db']:.1f} dB\n(BT4 26.6 GHz reference Rx, before its FFE)"))
    wa = rx_runs[best["label"]]
    serdes.plot_eye_density(axs[1, 2], wa.td, title="pf_rx output (recovered clock), adapted Tx")
    serdes.plot_stat_eye(axs[1, 3], wa.stat.eye, title="pf_rx statistical eye (+ distortion)")
    fig.suptitle("Near-package optics, 53.125 GBd PAM4: IBIS-AMI SerDes, 10 dB host traces, "
                 "CTLE driver -> traveling-wave Si MZM", fontsize=12)
    save(fig, "ibis_ami_npo_tw_mzm.png")

    summary = {
        "channels": cs, "termination_ohm": r_term,
        "electrode": {"zc_1ghz": abs(el["zc"][k1]), "zc_nyq": abs(el["zc"][k]),
                      "s11_driver_db_nyq": 20 * np.log10(abs(el["s11_drv"][k])),
                      "eo_db_nyq": eo_db[k], "eo_f3db_ghz": f3 / 1e9},
        "points": [{"label": p["label"], "peaking_db": p["peaking_db"], "taps": p["taps"],
                    "driver_gain": p["gain"], "tdecq": p["tdecq"],
                    "predicted": p.get("predicted")} for p in points],
        "adapted": best["label"],
        "rx": {lab: {"counted_ber": wa.td.counted["ber"], "bit_errors": wa.td.counted["bit_errors"],
                     "bits": wa.td.counted["bits"],
                     "stat_ber": wa.stat.eye.bit_error_rate(),
                     "eye_mv_1e6": [1e3 * wa.stat.eye.eye_height(1e-6, e) for e in range(3)],
                     "residual_rel": wa.residual_rel}
               for lab, wa in rx_runs.items()},
        "minutes": (time.time() - t_all) / 60,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "ibis_ami_npo_tw_mzm.json").write_text(json.dumps(summary, indent=1, default=float))
    q = best["tdecq"]
    print(f"\nTDECQ of the adapted transmitter: {q['tdecq_outer']:.2f} dB "
          f"(TDECQ - 10log10(min(C_eq, 1)) {q['score']:.2f} dB; "
          f"OMA_outer {10 * np.log10(q['oma_outer']):.2f} dBm, ER {q['er_db']:.2f} dB, "
          f"RLM {q['rlm']:.3f}) with driver CTLE {best['peaking_db']:.1f} dB and Tx FFE "
          f"{best['taps']}; reference FFE {[round(c, 3) for c in q['ffe_taps']]}, "
          f"C_eq {q['ceq']:.3f}. Total {summary['minutes']:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
