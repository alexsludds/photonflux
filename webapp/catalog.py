"""Component catalog for the web schematic editor.

Single source of truth mapping UI component types onto circulax/photonflux
models. Each entry declares:

  ports    ordered list of {name, domain} — domain is "optical" (complex
           coherent field, power = |E|^2) or "electrical" (real volts)
  params   UI-editable settings with defaults/units/descriptions; defaults
           mirror the underlying component signatures exactly
  expand   (composites only) how one symbol becomes several circulax
           instances — used for the ring modulator, which needs the
           field<->re/im adapters around the Verilog-A core

``build_models()`` returns the models_map for ``circulax.compile_circuit``.
Heavyweight imports (jax, circulax, photonflux) happen lazily inside it so
the catalog itself can be served instantly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# repo root (webapp/'s parent) — built-in Verilog-A sources live under models/
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# helpers to keep entries terse
# ---------------------------------------------------------------------------

def _p(name: str, default: float, unit: str, label: str, **kw: Any) -> dict:
    d = {"name": name, "default": default, "unit": unit, "label": label}
    d.update(kw)
    return d


def _ports(spec: str) -> list[dict]:
    """"pin:o pout:o vp:e vn:e" -> port dicts (o=optical, e=electrical)."""
    out = []
    for tok in spec.split():
        name, dom = tok.split(":")
        out.append({"name": name, "domain": "optical" if dom == "o" else "electrical"})
    return out


# ---------------------------------------------------------------------------
# the catalog
# ---------------------------------------------------------------------------
# NB: symbol geometry (pins' x/y offsets, drawing) lives in the frontend,
# keyed by the same type id; the backend only cares about ports/params.

CATALOG: dict[str, dict] = {
    # --- photonic sources & modulators ------------------------------------
    "cw_laser": {
        "label": "CW Laser",
        "category": "Lasers",
        "doc": "CW laser: constant field $E = \\sqrt{P}\\,e^{j\\phi}$ across "
               "p1/p2 (ground p2). All modulation belongs in modulators. For "
               "DWDM, set ref_wavelength_nm to a shared reference so several "
               "lasers at different wavelengths become distinct tones on one "
               "bus, $f_{off} = c\\,(1/\\lambda_{ref} - 1/\\lambda)$; "
               "0 = single-carrier (default). linewidth_hz > 0 adds a Wiener "
               "phase process (Lorentzian FWHM $\\Delta\\nu$) for coherent / "
               "self-heterodyne studies; needs a transient with noise seeds.",
        "ports": _ports("p1:o p2:o"),
        "params": [
            _p("wavelength_nm", 1310.0, "nm", "Wavelength"),
            _p("power", 1e-3, "W", "Power"),
            _p("phase", 0.0, "rad", "Phase"),
            _p("rin_db", 0.0, "dB/Hz", "RIN (0 = off)"),
            _p("linewidth_hz", 0.0, "Hz", "Lorentzian linewidth (0 = off)"),
            _p("ref_wavelength_nm", 0.0, "nm", "WDM ref. (0 = single-carrier)"),
        ],
    },
    "mzm": {
        "label": "Mach-Zehnder Modulator",
        "category": "Modulators",
        "doc": "Field-convention MZM: T(V) = IL*(0.5+0.5*eta*cos(pi*(V+vbias)/vpi)); "
               "electrode capacitance cel loads the driver.",
        "ports": _ports("pin:o pout:o vp:e vn:e"),
        "params": [
            _p("vpi", 3.0, "V", "V-pi"),
            _p("vbias", 0.0, "V", "Bias offset"),
            _p("il_db", 3.0, "dB", "Insertion loss"),
            _p("er_db", 20.0, "dB", "Extinction ratio"),
            _p("cel", 50e-15, "F", "Electrode cap."),
        ],
    },
    "iq_modulator": {
        "label": "IQ Modulator (nested MZM)",
        "category": "Modulators",
        "doc": "Nested-MZM IQ modulator on the coherent field: two null-biased "
               "push-pull child MZMs (I and Q arms) combined in quadrature, "
               "$t = 0.5\\,IL\\,[\\sin(\\tfrac{\\pi(V_I+b_i)}{2V_\\pi}) + "
               "e^{j(\\pi/2+q_{err})}\\sin(\\tfrac{\\pi(V_Q+b_q)}{2V_\\pi})]$, "
               "so $E_{out}=t\\,E_{in}$ maps the I/Q drives onto the complex "
               "plane — the coherent QPSK/QAM transmitter. Drive vip/vin (I) "
               "and vqp/vqn (Q); qerr is the quadrature phase error.",
        "ports": _ports("pin:o pout:o vip:e vin:e vqp:e vqn:e"),
        "params": [
            _p("vpi", 3.0, "V", "V-pi (each arm)"),
            _p("vbias_i", 0.0, "V", "I-arm bias"),
            _p("vbias_q", 0.0, "V", "Q-arm bias"),
            _p("qerr", 0.0, "rad", "Quadrature error"),
            _p("il_db", 6.0, "dB", "Insertion loss"),
            _p("cel", 50e-15, "F", "Electrode cap."),
        ],
    },
    "coherent_rx": {
        "label": "Coherent Receiver (90° hybrid + balanced PD)",
        "category": "Detectors & Bridges",
        "doc": "Single-pol coherent front-end: signal and LO beat in an ideal "
               "90° optical hybrid and two balanced photodiode pairs cancel "
               "the direct-detection terms, giving the I/Q photocurrents "
               "$i_I = R\\,\\mathrm{Re}(E_{sig}E_{lo}^*)$, "
               "$i_Q = R\\,\\mathrm{Im}(E_{sig}E_{lo}^*)$ — the complex "
               "baseband $r=i_I+j i_Q$ the coherent DSP demodulates. The lo "
               "port takes a second CW laser (its linewidth sets the phase "
               "noise the carrier recovery must track). Differential current "
               "outputs i_p/i_n (I) and q_p/q_n (Q).",
        "ports": _ports("sig:o lo:o i_p:e i_n:e q_p:e q_n:e"),
        "params": [
            _p("R", 0.8, "A/W", "Responsivity"),
        ],
    },
    "phase_shifter": {
        "label": "EO Phase Shifter (VA)",
        "category": "Modulators",
        "doc": "models/optical_field/phase_shifter.va — ideal electro-optic phase "
               "shifter: E_out = s*e^{j*phi}*E_in with phi = pi*(V+vbias)/vpi, "
               "so the electrode reaches a half-wave (pi) shift at V = vpi. "
               "The lumped building block behind every EO modulator (two make "
               "an MZM, one inside a cavity makes a ring modulator). Lossless "
               "and instantaneous by default (il_db = 0); the electrode loads "
               "the driver with cel + leakage. gnd must be grounded.",
        "ports": _ports("pin:o pout:o vp:e vn:e gnd:e"),
        "params": [
            _p("vpi", 3.0, "V", "V-pi (half-wave)"),
            _p("vbias", 0.0, "V", "Bias offset"),
            _p("il_db", 0.0, "dB", "Insertion loss"),
            _p("cel", 50e-15, "F", "Electrode cap."),
            _p("rleak", 1e8, "Ohm", "Electrode leakage"),
        ],
        # one symbol -> f2ri adapter + VA phase shifter + ri2f adapter
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ps": {"component": "_phaseshift_va", "settings": "ALL"},
                "join": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ps,in_re"),
                ("tap,im", "ps,in_im"),
                ("ps,out_re", "join,re"),
                ("ps,out_im", "join,im"),
            ],
            "port_map": {
                "pin": "tap,c", "pout": "join,c",
                "vp": "ps,vp", "vn": "ps,vn", "gnd": "ps,gnd",
            },
        },
    },
    "laser_dml": {
        "label": "DML Laser (VA)",
        "category": "Lasers",
        "doc": "models/optical_power/laser_dml.va — directly-modulated laser, static L-I: "
               "P = slope*(I - Ith) above threshold, first-order optical "
               "response tau. Drive current flows an->cat (linearised diode "
               "Von/Rs). Power-domain model: pout carries E = sqrt(P), optical "
               "phase is not modelled. gnd must be grounded.",
        "ports": _ports("an:e cat:e pout:o gnd:e"),
        "params": [
            _p("Ith", 10e-3, "A", "Threshold current"),
            _p("slope", 0.3, "W/A", "Slope efficiency"),
            _p("Rs", 5.0, "Ohm", "Series resistance"),
            _p("Von", 1.2, "V", "Turn-on voltage"),
            _p("tau", 50e-12, "s", "Optical time const."),
        ],
        "expand": {
            "instances": {
                "lsr": {"component": "_dml_va", "settings": "ALL"},
                "p2f": {"component": "_p2f"},
            },
            "connections": [("lsr,popt", "p2f,p")],
            "port_map": {"an": "lsr,an", "cat": "lsr,cat",
                         "pout": "p2f,c", "gnd": "lsr,gnd"},
        },
    },
    "laser_rate": {
        "label": "Rate-Eq. Laser (VA)",
        "category": "Lasers",
        "doc": "models/optical_power/laser_rate.va — directly-modulated laser from the "
               "single-mode rate equations: turn-on delay, relaxation "
               "oscillations, pattern-dependent ringing (what the static DML "
               "cannot show). Defaults are a generic 1.31 um DFB: Ith ~19 mA, "
               "fr ~6 GHz at ~2x threshold. Power-domain model: pout carries "
               "E = sqrt(P). gnd must be grounded.",
        # the lasing turn-on is a bifurcation: plain Newton stalls at the
        # threshold kink, so DC solves add a pseudo-transient settle
        "hard_dc": True,
        "ports": _ports("an:e cat:e pout:o gnd:e"),
        "params": [
            _p("Rs", 5.0, "Ohm", "Series resistance"),
            _p("Von", 1.2, "V", "Turn-on voltage"),
            _p("etai", 0.8, "", "Injection eff."),
            _p("eta0", 0.4, "", "Optical eff."),
            _p("Va", 1e-16, "m^3", "Active volume"),
            _p("Ntr", 1e24, "1/m^3", "Transparency dens."),
            _p("a0", 2.5e-20, "m^2", "Differential gain"),
            _p("vg", 7.5e7, "m/s", "Group velocity"),
            _p("taun", 2e-9, "s", "Carrier lifetime"),
            _p("taup", 2e-12, "s", "Photon lifetime"),
            _p("Gam", 0.3, "", "Confinement"),
            _p("beta", 1e-4, "", "Spont. emission"),
            _p("eps", 1.5e-23, "m^3", "Gain compression"),
            _p("Eph", 1.516e-19, "J", "Photon energy"),
        ],
        "expand": {
            "instances": {
                "lsr": {"component": "_rate_va", "settings": "ALL"},
                "p2f": {"component": "_p2f"},
            },
            "connections": [("lsr,popt", "p2f,p")],
            "port_map": {"an": "lsr,an", "cat": "lsr,cat",
                         "pout": "p2f,c", "gnd": "lsr,gnd"},
        },
    },
    "mzm_tw": {
        "label": "MZM Traveling-Wave (VA)",
        "category": "Modulators",
        "stiff": True,   # ps electrode ddt poles (~4.5 ps): default to BDF2
        "doc": "models/optical_power/mzm_tw.va — traveling-wave MZM. On top of the "
               "quasi-static cos() transfer it models the two effects that "
               "set a real TW electrode's EO bandwidth: frequency-dependent "
               "electrode loss (pole at f_el) and optical/RF velocity "
               "walk-off (pole at 0.443*c/(|n_rf-n_opt|*len)). Power-domain "
               "model: optical phase/chirp is discarded — keep coherent "
               "chains (dispersive fiber) on the field MZM. gnd must be "
               "grounded.",
        "ports": _ports("pin:o pout:o vp:e vn:e gnd:e"),
        "params": [
            _p("vpi", 1.5, "V", "V-pi"),
            _p("vbias", 0.0, "V", "Bias offset"),
            _p("il_db", 3.0, "dB", "Insertion loss"),
            _p("er_db", 20.0, "dB", "Extinction ratio"),
            _p("cel", 5e-15, "F", "Pad capacitance"),
            _p("len_mm", 4.0, "mm", "Electrode length",
               map={"to": "len", "scale": 1e-3}),
            _p("n_rf", 2.4, "", "RF group index"),
            _p("n_opt", 4.2, "", "Optical group index"),
            _p("f_el", 35e9, "Hz", "Electrode loss BW"),
            _p("f_el2", 0.0, "Hz", "2nd loss pole (0=off)"),
        ],
        "expand": {
            "instances": {
                "f2p": {"component": "_f2p"},
                "twm": {"component": "_tw_va", "settings": "ALL"},
                "p2f": {"component": "_p2f"},
            },
            "connections": [("f2p,p", "twm,pin"), ("twm,pout", "p2f,p")],
            "port_map": {"pin": "f2p,c", "pout": "p2f,c", "vp": "twm,vp",
                         "vn": "twm,vn", "gnd": "twm,gnd"},
        },
    },
    "ring_mod_inj": {
        "label": "Microring, Injection (VA)",
        "category": "Modulators",
        "doc": "models/optical_field/ring_mod_inj.va — carrier-INJECTION microring: "
               "forward-biased PIN shifter, resonance BLUE-shifts with the "
               "lifetime-filtered injected current (tau_c limits the "
               "modulation bandwidth to ~1/(2 pi tau_c) — drive with "
               "pre-emphasis to beat it), and free-carrier absorption adds "
               "loss with injection. Contrast with the depletion ring "
               "(ring_mod). gnd must be grounded.",
        "stiff": True,   # ps photon + ns carrier scales: default to BDF2
        "ports": _ports("pin:o pout:o vp:e vn:e gnd:e"),
        "params": [
            _p("lambda_nm", 1309.9772, "nm", "Laser wavelength"),
            _p("lambda_res_nm", 1310.0, "nm", "Resonance (0 mA)"),
            _p("radius_um", 7.5, "um", "Ring radius"),
            _p("n_g", 4.0, "", "Group index"),
            _p("loss_db_m", 3000.0, "dB/m", "Passive loss"),
            _p("kappa2", 0.05, "", "Power coupling k^2"),
            _p("tau_c", 1e-9, "s", "Carrier lifetime"),
            _p("dl_di_pm_ma", 50.0, "pm/mA", "Blue shift"),
            _p("fca_db_m_ma", 400.0, "dB/m/mA", "FCA loss"),
            _p("Von", 0.9, "V", "Diode turn-on"),
            _p("Rs", 50.0, "Ohm", "Series resistance"),
            _p("cj_ff_um", 1.0, "fF/um", "Junction cap."),
        ],
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ring": {"component": "_ring_inj_va", "settings": "ALL"},
                "join": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ring,in_re"),
                ("tap,im", "ring,in_im"),
                ("ring,out_re", "join,re"),
                ("ring,out_im", "join,im"),
            ],
            "port_map": {
                "pin": "tap,c", "pout": "join,c",
                "vp": "ring,vp", "vn": "ring,vn", "gnd": "ring,gnd",
            },
        },
    },
    "mzm_seg": {
        "label": "MZM Segmented (VA)",
        "category": "Modulators",
        "doc": "models/optical_power/mzm_seg.va — segmented-electrode MZM (optical DAC): "
               "three binary-weighted segments (4/7, 2/7, 1/7 of the "
               "electrode) driven independently synthesise up to 8 phase "
               "levels from plain digital rails — drive segments 1+2 for "
               "PAM4. vpi is the full-length half-wave voltage; short "
               "unused segments (vp = vn). Power-domain model. gnd must "
               "be grounded.",
        "ports": _ports("pin:o pout:o vp1:e vn1:e vp2:e vn2:e vp3:e vn3:e gnd:e"),
        "params": [
            _p("vpi", 3.0, "V", "V-pi (full length)"),
            _p("vbias", 1.5, "V", "Bias offset"),
            _p("il_db", 3.0, "dB", "Insertion loss"),
            _p("er_db", 20.0, "dB", "Extinction ratio"),
            _p("cel", 60e-15, "F", "Electrode cap (full)"),
        ],
        "expand": {
            "instances": {
                "f2p": {"component": "_f2p"},
                "seg": {"component": "_seg_va", "settings": "ALL"},
                "p2f": {"component": "_p2f"},
            },
            "connections": [("f2p,p", "seg,pin"), ("seg,pout", "p2f,p")],
            "port_map": {
                "pin": "f2p,c", "pout": "p2f,c",
                "vp1": "seg,vp1", "vn1": "seg,vn1",
                "vp2": "seg,vp2", "vn2": "seg,vn2",
                "vp3": "seg,vp3", "vn3": "seg,vn3",
                "gnd": "seg,gnd",
            },
        },
    },
    "pulse_mod": {
        "label": "Pulse Modulator (ideal)",
        "category": "Modulators",
        "doc": "Ideal intensity modulator carving one power pulse "
               "p_off -> p_on -> p_off (sigmoid edges) out of a CW field of "
               "power p_on.",
        "ports": _ports("pin:o pout:o"),
        "params": [
            _p("p_on", 18e-3, "W", "Pulse-top power"),
            _p("p_off", 3e-3, "W", "Baseline power"),
            _p("t0", 0.5e-9, "s", "Rise time point"),
            _p("t1", 2.0e-9, "s", "Fall time point"),
            _p("tr", 80e-12, "s", "Edge time const."),
        ],
    },
    "ring_mod": {
        "label": "Microring Modulator (VA)",
        "category": "Modulators",
        "doc": "models/optical_field/ring_mod.va — coupled-mode-theory microring modulator "
               "compiled from Verilog-A to JAX. gnd must be grounded.",
        "ports": _ports("pin:o pout:o vp:e vn:e gnd:e"),
        "params": [
            _p("lambda_nm", 1309.9772, "nm", "Laser wavelength"),
            _p("lambda_res_nm", 1310.0, "nm", "Resonance (0 V)"),
            _p("radius_um", 7.5, "um", "Ring radius"),
            _p("n_g", 4.0, "", "Group index"),
            _p("n_eff", 2.4, "", "Effective index"),
            _p("loss_db_m", 7000.0, "dB/m", "Round-trip loss"),
            _p("kappa2", 0.10, "", "Power coupling k^2"),
            _p("dl_dv_pm", 45.0, "pm/V", "Tuning slope"),
            _p("cj_ff_um", 0.5, "fF/um", "Junction cap."),
        ],
        # one symbol -> f2ri adapter + VA ring + ri2f adapter
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ring": {"component": "_ring_va", "settings": "ALL"},
                "join": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ring,in_re"),
                ("tap,im", "ring,in_im"),
                ("ring,out_re", "join,re"),
                ("ring,out_im", "join,im"),
            ],
            "port_map": {
                "pin": "tap,c", "pout": "join,c",
                "vp": "ring,vp", "vn": "ring,vn", "gnd": "ring,gnd",
            },
        },
    },
    # --- SOA + cavity building blocks (directed-wave VA models) ------------
    # These four carry DIRECTED waves: optical inputs are read at infinite
    # impedance, outputs are driven. Wire outputs to inputs (never output to
    # output); terminate dark inputs and unused outputs with the optical
    # terminator.
    "soa": {
        "label": "SOA (VA)",
        "category": "Lasers",
        "doc": "models/optical_field/soa.va — semiconductor optical amplifier: lumped "
               "Agrawal-Olsson gain reservoir tau_c*dh/dt = h0(I) - h - "
               "(G-1)*P/p_sat, G = e^h. BIDIRECTIONAL: forward (fin->fout) "
               "and backward (bin->bout) waves share the reservoir, so it "
               "drops into Fabry-Perot cavities (example 36). h0 is linear "
               "in bias current: transparent at i_tr, g0_db of gain at i_op "
               "(linearised diode Von/Rs on an/cat). tau_bw is the gain-"
               "bandwidth pole AND the cavity memory when a loop is closed "
               "around it; p_seed is a deterministic ASE stand-in that "
               "starts lasing and pins the phase. Terminate unused bin/bout "
               "with optical terminators. gnd must be grounded.",
        "hard_dc": True,
        "ports": _ports("fin:o fout:o bin:o bout:o an:e cat:e gnd:e"),
        "params": [
            _p("g0_db", 20.0, "dB", "Gain at i_op"),
            _p("i_op_ma", 80.0, "mA", "Operating current"),
            _p("i_tr_ma", 8.0, "mA", "Transparency current"),
            _p("p_sat", 10e-3, "W", "Saturation power"),
            _p("tau_c", 0.3e-9, "s", "Carrier lifetime"),
            _p("tau_bw", 1e-12, "s", "Gain-BW pole"),
            _p("alpha_h", 0.0, "", "Linewidth-enh. factor"),
            _p("p_seed", 1e-9, "W", "ASE seed power"),
            _p("Von", 1.2, "V", "Turn-on voltage"),
            _p("Rs", 3.0, "Ohm", "Series resistance"),
        ],
        "expand": {
            "instances": {
                "fi": {"component": "_f2ri"},
                "bi": {"component": "_f2ri"},
                "amp": {"component": "_soa_va", "settings": "ALL"},
                "fo": {"component": "_ri2f"},
                "bo": {"component": "_ri2f"},
            },
            "connections": [
                ("fi,re", "amp,fi_re"), ("fi,im", "amp,fi_im"),
                ("bi,re", "amp,bi_re"), ("bi,im", "amp,bi_im"),
                ("amp,fo_re", "fo,re"), ("amp,fo_im", "fo,im"),
                ("amp,bo_re", "bo,re"), ("amp,bo_im", "bo,im"),
            ],
            "port_map": {
                "fin": "fi,c", "fout": "fo,c", "bin": "bi,c", "bout": "bo,c",
                "an": "amp,an", "cat": "amp,cat", "gnd": "amp,gnd",
            },
        },
    },
    "tw_gain_seg": {
        "label": "TW gain slice (VA)",
        "category": "Lasers",
        "doc": "models/optical_field/tw_gain_seg.va — one active slice of a "
               "traveling-wave (DFB/DBR/FP) laser: the coupled-mode field "
               "stencil tau_s*dR/dt + (R - R_left) = dz*[(gamma - j*delta)R + "
               "j*kappa*S] (and the mirror for the backward wave S) with a "
               "LOCAL Agrawal-Olsson carrier reservoir tau_c*dgamma/dt = "
               "gamma0(I) - gamma*(1 + P_loc/p_sat) saturating on THIS slice's "
               "own circulating power — cascade M of them (fl/fr forward, bl/br "
               "backward) for a spatially-resolved gain region (spatial hole "
               "burning). kappa>0 makes it an index-coupled active DFB grating; "
               "alpha_h is the linewidth-enhancement chirp; the diode bias is "
               "on an/cat (Von/Rs). Feedback comes from mirrors / grating "
               "slices around it. gnd must be grounded; terminate dark inputs.",
        "hard_dc": True,
        "ports": _ports("fl:o fr:o bl:o br:o an:e cat:e gnd:e"),
        "params": [
            _p("lambda_nm", 1310.0, "nm", "Optical frame"),
            _p("lambda_bragg_nm", 1310.0, "nm", "Bragg wavelength"),
            _p("n_g", 3.7, "", "Group index"),
            _p("dz", 10e-6, "m", "Slice length"),
            _p("g_unsat_pm", 5000.0, "1/m", "Unsat amplitude gain @ i_op"),
            _p("i_op_ma", 80.0, "mA", "Operating current"),
            _p("i_tr_ma", 8.0, "mA", "Transparency current"),
            _p("p_sat", 10e-3, "W", "Saturation power"),
            _p("tau_c", 0.3e-9, "s", "Carrier lifetime"),
            _p("alpha_h", 0.0, "", "Linewidth-enh. factor"),
            _p("kappa_pm", 0.0, "1/m", "Bragg coupling"),
            _p("loss_pm", 0.0, "1/m", "Internal amplitude loss"),
            _p("p_seed", 0.0, "W", "Distributed ASE seed"),
            _p("Von", 1.2, "V", "Turn-on voltage"),
            _p("Rs", 3.0, "Ohm", "Series resistance"),
        ],
        "expand": {
            "instances": {
                "fli": {"component": "_f2ri"},
                "bri": {"component": "_f2ri"},
                "amp": {"component": "_twgain_va", "settings": "ALL"},
                "fro": {"component": "_ri2f"},
                "blo": {"component": "_ri2f"},
            },
            "connections": [
                ("fli,re", "amp,fl_re"), ("fli,im", "amp,fl_im"),
                ("bri,re", "amp,br_re"), ("bri,im", "amp,br_im"),
                ("amp,fr_re", "fro,re"), ("amp,fr_im", "fro,im"),
                ("amp,bl_re", "blo,re"), ("amp,bl_im", "blo,im"),
            ],
            "port_map": {
                "fl": "fli,c", "fr": "fro,c", "bl": "blo,c", "br": "bri,c",
                "an": "amp,an", "cat": "amp,cat", "gnd": "amp,gnd",
            },
        },
    },
    "tw_seg": {
        "label": "TW Bragg/passive slice (VA)",
        "category": "Lasers",
        "doc": "models/optical_field/tw_seg.va — one passive / fixed-gain slice "
               "of a traveling-wave laser: same coupled-mode field stencil as "
               "the gain slice with a FIXED amplitude gain/loss gamma_pm and a "
               "Bragg coupling kappa_pm. A length L of these reflects "
               "tanh(kappa*L) at the Bragg frame (delta0 = 2*pi*n_g*(1/lambda - "
               "1/lambda_bragg)) and traces the coupled-mode stopband — the "
               "DFB/DBR grating mirror. dbeta_dv shifts the local Bragg "
               "detuning with the tuning voltage vt (the DBR tuning knob); "
               "kappa_pm = 0 is a plain waveguide. Inputs fl/br read, outputs "
               "fr/bl drive; gnd must be grounded.",
        "stiff": True,
        "ports": _ports("fl:o fr:o bl:o br:o vt:e gnd:e"),
        "params": [
            _p("lambda_nm", 1310.0, "nm", "Optical frame"),
            _p("lambda_bragg_nm", 1310.0, "nm", "Bragg wavelength"),
            _p("n_g", 3.7, "", "Group index"),
            _p("dz", 10e-6, "m", "Slice length"),
            _p("kappa_pm", 0.0, "1/m", "Bragg coupling"),
            _p("gamma_pm", 0.0, "1/m", "Amplitude gain/loss"),
            _p("dbeta_dv", 0.0, "1/m/V", "Detuning tuning slope"),
            _p("r_tune", 1e6, "Ohm", "Tuning-node shunt"),
        ],
        "expand": {
            "instances": {
                "fli": {"component": "_f2ri"},
                "bri": {"component": "_f2ri"},
                "seg": {"component": "_twseg_va", "settings": "ALL"},
                "fro": {"component": "_ri2f"},
                "blo": {"component": "_ri2f"},
            },
            "connections": [
                ("fli,re", "seg,fl_re"), ("fli,im", "seg,fl_im"),
                ("bri,re", "seg,br_re"), ("bri,im", "seg,br_im"),
                ("seg,fr_re", "fro,re"), ("seg,fr_im", "fro,im"),
                ("seg,bl_re", "blo,re"), ("seg,bl_im", "blo,im"),
            ],
            "port_map": {
                "fl": "fli,c", "fr": "fro,c", "bl": "blo,c", "br": "bri,c",
                "vt": "seg,vt", "gnd": "seg,gnd",
            },
        },
    },
    "phase_pad": {
        "label": "TW phase pad / QWS defect (VA)",
        "category": "Lasers",
        "doc": "models/optical_field/phase_pad.va — lossless bidirectional "
               "phase rotation e^{-j*phi} on both the forward (fl->fr) and "
               "backward (br->bl) directed waves. phi0_rad = pi/2 is the "
               "quarter-wave defect that pulls a single QWS-DFB mode to the "
               "Bragg wavelength; phi = phi0_rad + dphi_dv*V(vt) makes it a "
               "tunable cavity-phase / DBR pad. Inputs read, outputs drive; "
               "gnd must be grounded.",
        "ports": _ports("fl:o fr:o bl:o br:o vt:e gnd:e"),
        "params": [
            _p("phi0_rad", 0.0, "rad", "Static phase"),
            _p("dphi_dv", 0.0, "rad/V", "Tuning slope"),
            _p("r_tune", 1e6, "Ohm", "Tuning-node shunt"),
        ],
        "expand": {
            "instances": {
                "fli": {"component": "_f2ri"},
                "bri": {"component": "_f2ri"},
                "pad": {"component": "_phasepad_va", "settings": "ALL"},
                "fro": {"component": "_ri2f"},
                "blo": {"component": "_ri2f"},
            },
            "connections": [
                ("fli,re", "pad,fl_re"), ("fli,im", "pad,fl_im"),
                ("bri,re", "pad,br_re"), ("bri,im", "pad,br_im"),
                ("pad,fr_re", "fro,re"), ("pad,fr_im", "fro,im"),
                ("pad,bl_re", "blo,re"), ("pad,bl_im", "blo,im"),
            ],
            "port_map": {
                "fl": "fli,c", "fr": "fro,c", "bl": "blo,c", "br": "bri,c",
                "vt": "pad,vt", "gnd": "pad,gnd",
            },
        },
    },
    "edfa": {
        "label": "EDFA (VA)",
        "category": "Lasers",
        "doc": "models/optical_field/edfa.va — erbium-doped fibre amplifier: "
               "the pumped, ms-timescale twin of the SOA. Pump-driven "
               "Agrawal-Olsson gain reservoir tau_c*dh/dt = h0(P_pump) - h - "
               "(G-1)*P/p_sat, G = e^h: h0 is linear in pump power (transparent "
               "at p_pump_tr, peak g0_db at p_pump_op). UNIDIRECTIONAL "
               "(fin->fout; EDFAs run behind isolators). The gain is "
               "WAVELENGTH-DEPENDENT: a detuned one-pole filter gives a "
               "Lorentzian gain peak at lambda_peak with -3 dB width gain_bw_nm, "
               "so on a shared DWDM bus (lambda_ref = the cw_laser WDM "
               "reference) each carrier is amplified by G/(1+(tau_bw*(w-w_pk))^2) "
               "— a real per-channel gain tilt. Set gain_bw_nm wide for a flat "
               "amplifier. p_sat saturates on the TOTAL input power, so all WDM "
               "channels share one inversion (homogeneous): drop channels and "
               "the survivors surge over ~tau_c (~10 ms). p_ase is a "
               "deterministic ASE seed (like the SOA's p_seed); for stochastic "
               "noise-figure studies cascade the ASE Noise Source at the "
               "configured PSD. gnd must be grounded.",
        "stiff": True,
        "ports": _ports("fin:o fout:o gnd:e"),
        "params": [
            _p("g0_db", 30.0, "dB", "Peak gain at p_pump_op"),
            _p("p_pump_mw", 100.0, "mW", "Pump power"),
            _p("p_pump_op_mw", 100.0, "mW", "Pump giving g0_db"),
            _p("p_pump_tr_mw", 8.0, "mW", "Transparency pump"),
            _p("p_sat", 5e-3, "W", "Saturation power"),
            _p("tau_c", 10e-3, "s", "Erbium lifetime"),
            _p("lambda_ref_nm", 1550.0, "nm", "DWDM reference"),
            _p("lambda_peak_nm", 1532.0, "nm", "Gain-peak wavelength"),
            _p("gain_bw_nm", 30.0, "nm", "Gain bandwidth (-3 dB)"),
            _p("alpha_h", 0.0, "", "Linewidth-enh. factor"),
            _p("p_ase", 1e-9, "W", "ASE seed power"),
        ],
        "expand": {
            "instances": {
                "fi": {"component": "_f2ri"},
                "amp": {"component": "_edfa_va", "settings": "ALL"},
                "fo": {"component": "_ri2f"},
            },
            "connections": [
                ("fi,re", "amp,fi_re"), ("fi,im", "amp,fi_im"),
                ("amp,fo_re", "fo,re"), ("amp,fo_im", "fo,im"),
            ],
            "port_map": {
                "fin": "fi,c", "fout": "fo,c", "gnd": "amp,gnd",
            },
        },
    },
    "ase_src": {
        "label": "ASE Noise Source",
        "category": "Lasers",
        "doc": "In-line broadband amplified-spontaneous-emission injector: "
               "E_out = E_in + n(t) with n complex white Gaussian noise of "
               "one-sided power density s_ase_dbm_hz (white to the transient "
               "noise bandwidth). Put it inside an SOA laser cavity, set the "
               "SOA's p_seed to 0, and the laser starts from NOISE — every "
               "cavity mode is seeded, so mode competition and the winning "
               "line are emergent (example 37). Physical level: S = "
               "n_sp*h*nu*(G-1) ~ -144 dBm/Hz for a 20 dB SOA. ACTIVE ONLY "
               "when the transient's noise seeds >= 1; otherwise a "
               "transparent pass-through. Directed: pin read, pout driven.",
        "ports": _ports("pin:o pout:o"),
        "params": [_p("s_ase_dbm_hz", -144.0, "dBm/Hz", "ASE density")],
    },
    "wmirror": {
        "label": "Mirror 2x2 (VA, waves)",
        "category": "Photonic Passives",
        "doc": "models/optical_field/mirror.va — partial reflector on DIRECTED waves: "
               "lo = r*e^{j*phi}*li + jt*ri, ro = jt*li + r*e^{j*phi}*ri "
               "(unitary, r = sqrt(refl)). The cavity-forming twin of the "
               "nodal Partial Mirror: use it to close loops around the SOA "
               "(li/lo face the left, ri/ro the right; phi_r_deg trims the "
               "round-trip phase). Inputs read, outputs drive — terminate "
               "dark inputs and unused outputs. gnd must be grounded.",
        "ports": _ports("li:o lo:o ri:o ro:o gnd:e"),
        "params": [
            _p("refl", 0.3, "", "Power reflectivity", min=0.0, max=1.0),
            _p("loss_db", 0.0, "dB", "Excess loss"),
            _p("phi_r_deg", 0.0, "deg", "Reflection phase"),
        ],
        "expand": {
            "instances": {
                "l_in": {"component": "_f2ri"},
                "r_in": {"component": "_f2ri"},
                "mir": {"component": "_mirror_va", "settings": "ALL"},
                "l_out": {"component": "_ri2f"},
                "r_out": {"component": "_ri2f"},
            },
            "connections": [
                ("l_in,re", "mir,li_re"), ("l_in,im", "mir,li_im"),
                ("r_in,re", "mir,ri_re"), ("r_in,im", "mir,ri_im"),
                ("mir,lo_re", "l_out,re"), ("mir,lo_im", "l_out,im"),
                ("mir,ro_re", "r_out,re"), ("mir,ro_im", "r_out,im"),
            ],
            "port_map": {
                "li": "l_in,c", "lo": "l_out,c",
                "ri": "r_in,c", "ro": "r_out,c", "gnd": "mir,gnd",
            },
        },
    },
    "circulator": {
        "label": "Circulator (VA)",
        "category": "Photonic Passives",
        "doc": "models/optical_field/circulator.va — non-reciprocal 3-port optical "
               "circulator: light routes 1 -> 2 -> 3 -> 1 only. The element "
               "that separates counter-propagating waves on one bidirectional "
               "fibre — transmit into p1i, put the shared line on p2 (p2o out, "
               "p2i the returning wave), drop the received wave at p3o. Each "
               "physical port is split into a directed input (p*i, a matched "
               "receiver) and a directed output (p*o, driven — never tie two "
               "outputs to one node). il_db is the routed-path loss, iso_db the "
               "finite isolation (reverse leakage into the wrong port; raise it "
               "for a more ideal circulator). Terminate any unused input with "
               "the optical terminator. gnd must be grounded.",
        "ports": _ports("p1i:o p1o:o p2i:o p2o:o p3i:o p3o:o gnd:e"),
        "params": [
            _p("il_db", 0.6, "dB", "Insertion loss"),
            _p("iso_db", 40.0, "dB", "Isolation"),
            _p("phi_deg", 0.0, "deg", "Transmission phase"),
        ],
        "expand": {
            "instances": {
                "i1": {"component": "_f2ri_m"},
                "i2": {"component": "_f2ri_m"},
                "i3": {"component": "_f2ri_m"},
                "circ": {"component": "_circ_va", "settings": "ALL"},
                "o1": {"component": "_ri2f"},
                "o2": {"component": "_ri2f"},
                "o3": {"component": "_ri2f"},
            },
            "connections": [
                ("i1,re", "circ,p1i_re"), ("i1,im", "circ,p1i_im"),
                ("i2,re", "circ,p2i_re"), ("i2,im", "circ,p2i_im"),
                ("i3,re", "circ,p3i_re"), ("i3,im", "circ,p3i_im"),
                ("circ,p1o_re", "o1,re"), ("circ,p1o_im", "o1,im"),
                ("circ,p2o_re", "o2,re"), ("circ,p2o_im", "o2,im"),
                ("circ,p3o_re", "o3,re"), ("circ,p3o_im", "o3,im"),
            ],
            "port_map": {
                "p1i": "i1,c", "p1o": "o1,c",
                "p2i": "i2,c", "p2o": "o2,c",
                "p3i": "i3,c", "p3o": "o3,c",
                "gnd": "circ,gnd",
            },
        },
    },
    "ring_comb": {
        "label": "Ring Filter Comb (VA)",
        "category": "Photonic Passives",
        "doc": "models/optical_field/ring_filter.va — add-drop microring with FIVE "
               "longitudinal modes (m = -2..+2 at FSR = c/(n_g*2*pi*R)), so "
               "the resonance COMB is modelled — the Vernier-laser building "
               "block (example 37). Heater hp/hn is a plain resistor whose "
               "power red-shifts the whole comb by dl_dmw_pm per mW. "
               "wavelength_nm is the probe/frame reference: DC-sweep it with "
               "instance \"*\" to trace the comb. Directed 3-port (in -> "
               "thru, in -> drop). gnd must be grounded.",
        "ports": _ports("pin:o thru:o drop:o hp:e hn:e gnd:e"),
        "params": [
            _p("wavelength_nm", 1310.0, "nm", "Probe wavelength",
               map={"to": "lambda_nm"}),
            _p("lambda_res_nm", 1310.0, "nm", "Cold m=0 resonance"),
            _p("radius_um", 100.0, "um", "Ring radius"),
            _p("n_g", 4.0, "", "Group index"),
            _p("loss_db_m", 100.0, "dB/m", "Propagation loss"),
            _p("kappa2_in", 0.05, "", "Bus coupling k^2"),
            _p("kappa2_drop", 0.05, "", "Drop coupling k^2"),
            _p("dl_dmw_pm", 20.0, "pm/mW", "Thermal shift"),
            _p("r_heater", 500.0, "Ohm", "Heater resistance"),
        ],
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ring": {"component": "_ringcomb_va", "settings": "ALL"},
                "jt": {"component": "_ri2f"},
                "jd": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ring,in_re"), ("tap,im", "ring,in_im"),
                ("ring,thru_re", "jt,re"), ("ring,thru_im", "jt,im"),
                ("ring,drop_re", "jd,re"), ("ring,drop_im", "jd,im"),
            ],
            "port_map": {
                "pin": "tap,c", "thru": "jt,c", "drop": "jd,c",
                "hp": "ring,hp", "hn": "ring,hn", "gnd": "ring,gnd",
            },
        },
    },
    "ring_kerr": {
        "label": "Kerr FWM Ring (VA)",
        "category": "Photonic Passives",
        "doc": "models/optical_field/ring_kerr.va — add-drop microring whose FIVE modes "
               "(m = -2..+2 at the FSR) mix through the intracavity chi(3) "
               "Kerr nonlinearity: the modal (Lugiato-Lefever) form of "
               "four-wave mixing in a resonator. Pump one resonance and "
               "seed the next (two WDM lasers one FSR apart): the idler "
               "grows in the mode one FSR on the other side (2f_p = f_s + "
               "f_i, momentum matched), cascading into modes +-2 at mW "
               "drive — a comb seed (example 40). SPM, XPM (x2) and FWM "
               "all come from one momentum-matched triple sum; d2_hz (comb "
               "dispersion) detunes the idler mode off the FWM frequency — "
               "the ring's phase-matching knob. Pure Kerr, no TPA/FCA "
               "(that is ring_nl). Directed 3-port (in -> thru, in -> "
               "drop). gnd must be grounded.",
        "ports": _ports("pin:o thru:o drop:o gnd:e"),
        "params": [
            _p("wavelength_nm", 1310.0, "nm", "Probe wavelength",
               map={"to": "lambda_nm"}),
            _p("lambda_res_nm", 1310.0, "nm", "Cold m=0 resonance"),
            _p("radius_um", 2000.0, "um", "Ring radius"),
            _p("n_g", 4.0, "", "Group index"),
            _p("loss_db_m", 30.0, "dB/m", "Propagation loss"),
            _p("kappa2_in", 0.035, "", "Bus coupling k^2"),
            _p("kappa2_drop", 0.035, "", "Drop coupling k^2"),
            _p("a_eff_um2", 0.1, "um^2", "Effective area"),
            _p("n2_kerr", 4.5e-18, "m^2/W", "Kerr index"),
            _p("d2_hz", 0.0, "Hz", "Comb dispersion /mode^2"),
        ],
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ring": {"component": "_ringkerr_va", "settings": "ALL"},
                "jt": {"component": "_ri2f"},
                "jd": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ring,in_re"), ("tap,im", "ring,in_im"),
                ("ring,thru_re", "jt,re"), ("ring,thru_im", "jt,im"),
                ("ring,drop_re", "jd,re"), ("ring,drop_im", "jd,im"),
            ],
            "port_map": {
                "pin": "tap,c", "thru": "jt,c", "drop": "jd,c",
                "gnd": "ring,gnd",
            },
        },
    },
    "ring_nl": {
        "label": "Nonlinear Ring TPA/FCA (VA)",
        "category": "Photonic Passives",
        "doc": "models/optical_field/ring_nl.va — high-Q all-pass ring whose intrinsic "
               "loss grows with the stored field: two-photon absorption of "
               "the circulating intensity + free-carrier absorption of the "
               "TPA-generated carriers (lifetime tau_fc), plus Kerr red / "
               "free-carrier blue dispersive shifts (set n2_kerr = dn_dn = 0 "
               "for a pure absorption study). This is why a 1e6-Q silicon "
               "ring stops being 1e6-Q above ~0.1 mW in the bus (example "
               "38). Defaults: Q_i ~ 2.8e6, kappa2 = 6e-4 -> loaded Q ~ "
               "1.2e6 mildly overcoupled. DC-sweep wavelength_nm with "
               "instance \"*\". Directed 2-port. gnd must be grounded. "
               "(Solved cold per sweep point — with the dispersive shifts "
               "enabled the pulled line is bistable and a cold Newton picks "
               "one branch.)",
        "ports": _ports("pin:o pout:o gnd:e"),
        "params": [
            _p("wavelength_nm", 1310.0, "nm", "Probe wavelength",
               map={"to": "lambda_nm"}),
            _p("lambda_res_nm", 1310.0, "nm", "Cold resonance"),
            _p("radius_um", 10.0, "um", "Ring radius"),
            _p("n_g", 4.0, "", "Group index"),
            _p("loss_db_m", 30.0, "dB/m", "Linear intrinsic loss"),
            _p("kappa2", 6e-4, "", "Bus coupling k^2"),
            _p("a_eff_um2", 0.1, "um^2", "Mode area"),
            _p("beta_tpa", 8e-12, "m/W", "TPA coefficient"),
            _p("sigma_fca", 1.45e-21, "m^2", "FCA cross-section"),
            _p("tau_fc", 1e-9, "s", "Carrier lifetime"),
            _p("n2_kerr", 4.5e-18, "m^2/W", "Kerr index"),
            _p("dn_dn", -4e-27, "m^3", "FCD dn/dN"),
        ],
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ring": {"component": "_ringnl_va", "settings": "ALL"},
                "join": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ring,in_re"), ("tap,im", "ring,in_im"),
                ("ring,out_re", "join,re"), ("ring,out_im", "join,im"),
            ],
            "port_map": {"pin": "tap,c", "pout": "join,c", "gnd": "ring,gnd"},
        },
    },
    "ring_selfheat": {
        "label": "Self-Heating Ring (VA)",
        "category": "Photonic Passives",
        "stiff": True,   # ps photon + us thermal scales: default to BDF2
        "doc": "models/optical_field/ring_selfheat.va — high-Q all-pass ring "
               "with OPTICAL SELF-HEATING (thermo-optic bistability): the "
               "fraction heat_frac of the intrinsic loss that is absorbed "
               "heats a one-pole thermal reservoir (R_th, tau_th), silicon's "
               "dn/dT > 0 red-shifts the resonance by dl_dt_pm per kelvin, "
               "and that shift changes the stored power — a nonlinear "
               "feedback loop. The laser wavelength is the ELECTRICAL node "
               "`lam` (V(lam) = wavelength in nm), so a PWL/ramp source can "
               "sweep it as a waveform. Ramp it slowly across the cold "
               "resonance and back (a transient) and the through-port traces "
               "a HYSTERESIS loop: to the red the heated resonance is dragged "
               "along with the laser (locked, high circulating power, a "
               "triangular thermal-locking line) then snaps back; to the blue "
               "the ring stays cold until the cold resonance. The bistable "
               "window widens with power (~90 pm at 50 uW). Held at a fixed "
               "lam it reduces to the all-pass Lorentzian. Directed 2-port; "
               "lam and gnd are electrical. (Pinned physics study: "
               "examples/ring_selfheat.py — example 41.)",
        "ports": _ports("pin:o pout:o lam:e gnd:e"),
        "params": [
            _p("lambda_res_nm", 1310.0, "nm", "Cold resonance"),
            _p("radius_um", 8.0, "um", "Ring radius"),
            _p("n_g", 4.0, "", "Group index"),
            _p("n_eff", 2.4, "", "Effective index"),
            _p("loss_db_m", 300.0, "dB/m", "Intrinsic loss"),
            _p("kappa2", 0.004, "", "Bus coupling k^2"),
            _p("heat_frac", 1.0, "", "Absorbed fraction of loss"),
            _p("rth_k_w", 3.0e4, "K/W", "Thermal resistance"),
            _p("tau_th_s", 1.0e-6, "s", "Thermal time constant"),
            _p("dl_dt_pm", 80.0, "pm/K", "Thermo-optic shift"),
        ],
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "ring": {"component": "_ringselfheat_va", "settings": "ALL"},
                "join": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "ring,in_re"), ("tap,im", "ring,in_im"),
                ("ring,out_re", "join,re"), ("ring,out_im", "join,im"),
            ],
            "port_map": {
                "pin": "tap,c", "pout": "join,c",
                "lam": "ring,lam_nm", "gnd": "ring,gnd",
            },
        },
    },
    "wg_nl": {
        "label": "Nonlinear Waveguide (VA)",
        "category": "Photonic Passives",
        "doc": "models/optical_field/waveguide_nl.va — silicon-wire segment with two-"
               "photon absorption (exact closed form), free-carrier "
               "absorption of the TPA carriers (lifetime tau_fc), Kerr SPM "
               "and free-carrier dispersion phase. Lumped single segment: "
               "cascade several for long/high-power spans. Directed 2-port "
               "(pin -> pout). gnd must be grounded.",
        "ports": _ports("pin:o pout:o gnd:e"),
        "params": [
            _p("wavelength_nm", 1310.0, "nm", "Wavelength",
               map={"to": "lambda_nm"}),
            _p("length_m", 1e-3, "m", "Length",
               map={"to": "length_um", "scale": 1e6}),
            _p("loss_db_m", 200.0, "dB/m", "Linear loss"),
            _p("a_eff_um2", 0.1, "um^2", "Effective area"),
            _p("beta_tpa", 8e-12, "m/W", "TPA coefficient"),
            _p("sigma_fca", 1.45e-21, "m^2", "FCA cross-section"),
            _p("tau_fc", 1e-9, "s", "Carrier lifetime"),
            _p("n2_kerr", 4.5e-18, "m^2/W", "Kerr index"),
            _p("dn_dn", -4e-27, "m^3", "FCD dn/dN"),
        ],
        "expand": {
            "instances": {
                "tap": {"component": "_f2ri"},
                "wg": {"component": "_wgnl_va", "settings": "ALL"},
                "join": {"component": "_ri2f"},
            },
            "connections": [
                ("tap,re", "wg,in_re"), ("tap,im", "wg,in_im"),
                ("wg,out_re", "join,re"), ("wg,out_im", "join,im"),
            ],
            "port_map": {"pin": "tap,c", "pout": "join,c", "gnd": "wg,gnd"},
        },
    },
    # --- photonic passives -------------------------------------------------
    "waveguide": {
        "label": "Waveguide",
        "category": "Photonic Passives",
        "doc": "Single-mode integrated waveguide with first-order dispersion "
               "and propagation loss (circulax OpticalWaveguide). For a long "
               "fibre span with real chromatic dispersion use Fiber "
               "(dispersion) instead. Length in "
               "meters (SI suffixes work: 100u, 1.5m, 2). wavelength_nm is "
               "a live parameter — DC-sweep it with instance \"*\" to trace "
               "spectral responses; interferometer FSR follows the *group* "
               "index: $\\text{FSR} = \\lambda^2/(n_g\\,\\Delta L)$.",
        "ports": _ports("p1:o p2:o"),
        "params": [
            # UI is SI meters; circulax wants um -> mapped in simulate.py
            _p("length_m", 1e-4, "m", "Length",
               map={"to": "length_um", "scale": 1e6}),
            _p("loss_dB_cm", 1.0, "dB/cm", "Loss"),
            _p("neff", 2.4, "", "n_eff"),
            _p("n_group", 4.0, "", "n_group"),
            _p("center_wavelength_nm", 1310.0, "nm", "Center wavelength"),
            _p("wavelength_nm", 1310.0, "nm", "Operating wavelength"),
        ],
    },
    "splitter": {
        "label": "Y-Splitter",
        "category": "Photonic Passives",
        "doc": "Lossless Y-junction; split_ratio of input power goes to p2, "
               "the rest to p3. Reciprocal: used in reverse (feed p2/p3, "
               "take p1) it is the coherent combiner of an MZI — "
               "$E_{p1} = \\sqrt{r}\\,E_{p2} + j\\sqrt{1-r}\\,E_{p3}$.",
        "ports": _ports("p1:o p2:o p3:o"),
        "params": [_p("split_ratio", 0.5, "", "Split ratio -> p2", min=0.0, max=1.0)],
    },
    "dir_coupler": {
        "label": "Directional Coupler",
        "category": "Photonic Passives",
        "doc": "2x2 beamsplitter: inputs p1/p2 (left), outputs p3/p4 "
               "(right). Bar p1->p3 / p2->p4 with sqrt(1-coupling), cross "
               "p1->p4 / p2->p3 with j*sqrt(coupling) (unitary). Reciprocal "
               "— feed any port. Terminate unused ports with the optical "
               "terminator: an open optical port reflects.",
        "ports": _ports("p1:o p2:o p3:o p4:o"),
        "params": [_p("coupling", 0.5, "", "Power coupling", min=0.0, max=1.0)],
    },
    "photodiode": {
        "label": "Photodiode",
        "category": "Detectors & Bridges",
        "doc": "PIN photodiode bridge: matched optical absorber (po_p/po_n), "
               "photocurrent $I_{ph} = R\\,|E|^2 + I_{dk}$ into an/cat with "
               "junction capacitance Cj. Optional intrinsic (transit-time) "
               "bandwidth f_3dB (two real poles; 0 = unlimited) and soft "
               "output saturation current (0 = off). Shot noise $2qI_{ph}$ "
               "is used by the noise analyses.",
        "ports": _ports("po_p:o po_n:o an:e cat:e"),
        "params": [
            _p("R", 0.8, "A/W", "Responsivity"),
            _p("Idk", 1e-9, "A", "Dark current"),
            _p("Cj", 100e-15, "F", "Junction cap."),
            _p("f3db", 0.0, "Hz", "Transit BW (0 = inf)"),
            _p("isat", 0.0, "A", "Saturation (0 = off)"),
        ],
    },
    "apd": {
        "label": "APD Photodiode",
        "category": "Detectors & Bridges",
        "doc": "Avalanche photodiode bridge: a PIN detector followed by "
               "avalanche multiplication. The primary photocurrent $R\\,|E|^2$ "
               "and the bulk dark current are multiplied by the avalanche gain "
               "$M$; the surface dark current bypasses the multiplication "
               "region, so the DC output is $I = M(R P + I_{dk,bulk}) + "
               "I_{dk,surf}$. Multiplication amplifies the shot noise "
               "SUPER-linearly: the excess-noise factor $F(M) = kM + "
               "(2 - 1/M)(1 - k)$ (McIntyre, ionization ratio $k$) makes the "
               "shot-noise PSD $2q\\,I_{prim}\\,M^2 F(M)$ — so a receiver "
               "sensitivity sweep vs $M$ shows a noise-optimal $M^*$ (thermal-"
               "limited below, excess-noise-limited above). A gain-bandwidth "
               "product gbp caps the effective bandwidth to $\\sim$gbp$/M$ "
               "above the transit-time corner $M_0 = $gbp$/f_{3dB}$ (0 = off). "
               "Shot noise feeds the transient noise seeds (like the PIN "
               "photodiode); use it in place of the photodiode in a receiver "
               "link.",
        "ports": _ports("po_p:o po_n:o an:e cat:e"),
        "params": [
            _p("R", 0.8, "A/W", "Responsivity (unmult.)"),
            _p("M", 10.0, "", "Avalanche gain M", min=1.0),
            _p("k_ion", 0.3, "", "Ionization ratio k", min=0.0, max=1.0),
            _p("Idk_bulk", 1e-9, "A", "Dark current (mult.)"),
            _p("Idk_surf", 1e-9, "A", "Surface dark (unmult.)"),
            _p("Cj", 100e-15, "F", "Junction cap."),
            _p("f3db", 0.0, "Hz", "Transit BW (0 = inf)"),
            _p("gbp", 0.0, "Hz", "Gain-BW product (0 = off)"),
            _p("isat", 0.0, "A", "Saturation (0 = off)"),
        ],
    },
    "channel": {
        "label": "Copper Channel",
        "category": "Channels",
        "doc": "Parametric PCB/cable channel: sqrt(f) skin-effect loss "
               "(loss_db at f_nyq) with minimum phase, vector-fitted to a "
               "compact state-space at compile time (parameter edits "
               "recompile). Voltage-transfer block: high-Z input, driven "
               "output — add your own source/termination resistors. The "
               "same shape as the user's serdes channel.py.",
        "ports": _ports("inp:e out:e"),
        "params": [
            _p("loss_db", 10.0, "dB", "Loss @ f_nyq", rebuild=True),
            _p("f_nyq", 14e9, "Hz", "Nyquist freq", rebuild=True),
            _p("n_poles", 10, "", "Fit order", rebuild=True),
        ],
        "lti": "chan",
    },
    "s2p_channel": {
        "label": "S2P Channel (file)",
        "category": "Channels",
        "doc": "Two-port from a Touchstone .s2p file: the matched-"
               "termination insertion transfer S21/2 is vector-fitted to a "
               "state-space (fit error logged). Input presents a z0 shunt, "
               "output is driven. Upload the file with the inspector "
               "button; the matched approximation ignores re-reflections "
               "with your actual source/load.",
        "ports": _ports("inp:e out:e"),
        "params": [
            _p("file", "", "", "Touchstone file", rebuild=True, kind="file"),
            _p("z0", 50.0, "Ohm", "Reference Z", rebuild=True),
            _p("n_poles", 12, "", "Fit order", rebuild=True),
        ],
        "lti": "s2p",
    },
    "fiber_cd": {
        "label": "Fiber (dispersion)",
        "category": "Channels",
        "doc": "Chromatic dispersion on the coherent field: the all-pass "
               "$\\exp(-j(\\tfrac{\\beta_2}{2}\\omega^2 + "
               "\\tfrac{\\beta_3}{6}\\omega^3)L)$ (+ flat attenuation) "
               "vector-fitted over $\\pm$fit_bw with a causal transit delay "
               "(waveforms arrive later — that's the fiber's latency). "
               "C-band: set D (and optionally slope S). O-band / near the "
               "zero-dispersion wavelength: set lambda0_nm > 0 and S (read "
               "as S0) — $D(\\lambda)$ then follows the G.652 profile "
               "$\\tfrac{S_0}{4}(\\lambda - \\lambda_0^4/\\lambda^3)$ and the "
               "beta3 slope term dominates, so "
               "the model stays correct where D ~ 0. Compile-time fit; the "
               "log reports D, beta2*L, beta3*L and fit error. Keep fit_bw "
               "~3x the signal bandwidth.",
        "ports": _ports("p1:o p2:o"),
        "params": [
            _p("length_km", 10.0, "km", "Length", rebuild=True),
            _p("D_ps", 17.0, "ps/nm/km", "Dispersion D", rebuild=True),
            _p("S_ps", 0.0, "ps/nm^2/km", "Slope S (S0 if l0 set)",
               rebuild=True),
            _p("lambda0_nm", 0.0, "nm", "Zero-disp. l0 (0 = use D)",
               rebuild=True),
            _p("lambda_nm", 1550.0, "nm", "Wavelength", rebuild=True),
            _p("atten_db_km", 0.2, "dB/km", "Attenuation", rebuild=True),
            _p("fit_bw", 60e9, "Hz", "Fit bandwidth", rebuild=True),
            _p("n_poles", 28, "", "Fit order", rebuild=True),
        ],
        "lti": "fiber",
    },
    "fiber_nl": {
        "label": "Fiber (nonlinear, split-step)",
        "category": "Channels",
        "doc": "Nonlinear fibre by the split-step Fourier method on the "
               "coherent field: attenuation, dispersion (beta2/beta3) and the "
               "Kerr effect ($\\gamma\\,[1/\\mathrm{W/km}]$) solved as a cascade "
               "of $n_{seg}$ dispersion segments interleaved with the "
               "instantaneous nonlinear phase $\\exp(-j\\gamma|E|^2 dz)$ — SPM, "
               "XPM and four-wave mixing emerge from the field evolution. Set "
               "$\\gamma$ = 0 to recover the linear fiber_cd all-pass; raise "
               "n_seg for accuracy (each segment adds n_poles states, so keep "
               "it modest for long transients). D/S and lambda0 follow the same "
               "G.652 convention as fiber_cd. The batch reference solver "
               "webapp/ssfm.py shares the physics and pins the soliton / SPM / "
               "FWM analytics. Keep fit_bw ~3x the signal bandwidth.",
        "ports": _ports("p1:o p2:o"),
        "params": [
            _p("length_km", 10.0, "km", "Length", rebuild=True),
            _p("D_ps", 17.0, "ps/nm/km", "Dispersion D", rebuild=True),
            _p("S_ps", 0.0, "ps/nm^2/km", "Slope S (S0 if l0 set)",
               rebuild=True),
            _p("lambda0_nm", 0.0, "nm", "Zero-disp. l0 (0 = use D)",
               rebuild=True),
            _p("gamma_per_W_km", 1.3, "1/W/km", "Kerr gamma", rebuild=True),
            _p("lambda_nm", 1550.0, "nm", "Wavelength", rebuild=True),
            _p("atten_db_km", 0.2, "dB/km", "Attenuation", rebuild=True),
            _p("n_seg", 20, "", "Split-step segments", rebuild=True),
            _p("fit_bw", 60e9, "Hz", "Fit bandwidth", rebuild=True),
            _p("n_poles", 12, "", "Fit order / segment", rebuild=True),
        ],
        "lti": "fiber_nl",
    },
    "raman_amp": {
        "label": "Raman Fiber / Amp (VA)",
        "category": "Channels",
        "doc": "models/optical_field/raman_amp.va — stimulated Raman scattering "
               "(SRS) span + distributed Raman amplifier. A forward signal "
               "(sin->sout) and a pump that may be co-propagating (pcin->pcout) "
               "and/or counter-propagating (pctin->pctout); Raman gain transfers "
               "power from the shorter-wavelength pump to the longer-wavelength "
               "signal with the pump depleting by nu_s/nu_p. Exact two-wave "
               "logistic solution: small-signal on/off gain is the textbook "
               "$e^{g_R P_p L_{eff}/A_{eff}}$, saturating as the signal grows "
               "(pump depletion). Two uses: (1) two-channel WDM SRS tilt — wire "
               "the short-lambda channel into pcin and the long-lambda channel "
               "into sin; (2) a Raman amplifier — a weak signal plus a strong "
               "(usually counter-propagating) pump into pctin. Total pump "
               "$P_p = |E_{pcin}|^2 + |E_{pctin}|^2$; co/counter give the same "
               "integrated on/off gain. Terminate unused pump ports. gnd "
               "grounded.",
        "ports": _ports("sin:o sout:o pcin:o pcout:o pctin:o pctout:o gnd:e"),
        "params": [
            _p("lambda_s_nm", 1550.0, "nm", "Signal wavelength"),
            _p("lambda_p_nm", 1450.0, "nm", "Pump wavelength"),
            _p("g_r", 0.6e-13, "m/W", "Peak Raman gain"),
            _p("a_eff_um2", 80.0, "um^2", "Effective area"),
            _p("length_km", 50.0, "km", "Span length"),
            _p("loss_s_db_km", 0.20, "dB/km", "Signal loss"),
            _p("loss_p_db_km", 0.25, "dB/km", "Pump loss"),
        ],
        "expand": {
            "instances": {
                "si": {"component": "_f2ri"},
                "pc": {"component": "_f2ri"},
                "px": {"component": "_f2ri"},
                "amp": {"component": "_raman_va", "settings": "ALL"},
                "so": {"component": "_ri2f"},
                "pco": {"component": "_ri2f"},
                "pxo": {"component": "_ri2f"},
            },
            "connections": [
                ("si,re", "amp,si_re"), ("si,im", "amp,si_im"),
                ("pc,re", "amp,pfi_re"), ("pc,im", "amp,pfi_im"),
                ("px,re", "amp,pbi_re"), ("px,im", "amp,pbi_im"),
                ("amp,so_re", "so,re"), ("amp,so_im", "so,im"),
                ("amp,pfo_re", "pco,re"), ("amp,pfo_im", "pco,im"),
                ("amp,pbo_re", "pxo,re"), ("amp,pbo_im", "pxo,im"),
            ],
            "port_map": {
                "sin": "si,c", "sout": "so,c",
                "pcin": "pc,c", "pcout": "pco,c",
                "pctin": "px,c", "pctout": "pxo,c",
                "gnd": "amp,gnd",
            },
        },
    },
    "sbs_fiber": {
        "label": "SBS Fiber (VA)",
        "category": "Channels",
        "doc": "models/optical_field/sbs_fiber.va — stimulated Brillouin "
               "scattering span (threshold + backscatter). A forward pump "
               "(fin->fout) drives a counter-propagating Stokes wave out the "
               "backward port (bout). Below the SBS threshold "
               "$P_{th}=n_{th}A_{eff}/(g_B L_{eff})$ (textbook $n_{th}\\approx21$) "
               "almost everything transmits; above it the transmitted power "
               "CLAMPS at ~P_th and the surplus is reflected as the backward "
               "Stokes — the classic SBS power limiter. Energy conserving "
               "($P_{fout}=P_{th}\\tanh(P_{in}/P_{th})e^{-\\alpha L}$, the rest "
               "to bout). NOTE the shared baseband envelope cannot represent the "
               "~11 GHz Stokes shift / ~20-50 MHz linewidth — this is a "
               "power-domain limiter. Terminate the unused bin. gnd grounded.",
        "ports": _ports("fin:o fout:o bin:o bout:o gnd:e"),
        "params": [
            _p("g_b", 5e-11, "m/W", "Brillouin gain"),
            _p("a_eff_um2", 80.0, "um^2", "Effective area"),
            _p("length_km", 20.0, "km", "Span length"),
            _p("loss_db_km", 0.20, "dB/km", "Linear loss"),
            _p("n_th", 21.0, "", "Threshold factor"),
        ],
        "expand": {
            "instances": {
                "fi": {"component": "_f2ri"},
                "bi": {"component": "_f2ri"},
                "amp": {"component": "_sbs_va", "settings": "ALL"},
                "fo": {"component": "_ri2f"},
                "bo": {"component": "_ri2f"},
            },
            "connections": [
                ("fi,re", "amp,fi_re"), ("fi,im", "amp,fi_im"),
                ("bi,re", "amp,bi_re"), ("bi,im", "amp,bi_im"),
                ("amp,fo_re", "fo,re"), ("amp,fo_im", "fo,im"),
                ("amp,bo_re", "bo,re"), ("amp,bo_im", "bo,im"),
            ],
            "port_map": {
                "fin": "fi,c", "fout": "fo,c", "bin": "bi,c", "bout": "bo,c",
                "gnd": "amp,gnd",
            },
        },
    },
    # --- passive integrated optics (coherent field, wavelength-aware) -------
    "grating": {
        "label": "Grating Coupler",
        "category": "Photonic Passives",
        "doc": "Grating coupler: Gaussian spectral response — insertion loss "
               "grows quadratically with detuning from center_wavelength_nm "
               "(bandwidth_1dB = full 1 dB width) — plus a finite "
               "back-reflection at the fiber interface (back_refl_db return "
               "loss, ~25 dB for a typical uniform GC; clamped to keep the "
               "2-port passive). Two gratings on one chip therefore form a "
               "weak parasitic Fabry-Perot — the ripple real measurements "
               "show. wavelength_nm is live: include it in a \"*\" "
               "wavelength sweep to see the passband.",
        "ports": _ports("grating:o waveguide:o"),
        "params": [
            _p("center_wavelength_nm", 1310.0, "nm", "Center wavelength"),
            _p("peak_loss_dB", 1.5, "dB", "Peak insertion loss"),
            _p("bandwidth_1dB", 20.0, "nm", "1 dB bandwidth"),
            _p("back_refl_db", 25.0, "dB", "Back-reflection (RL)"),
            _p("wavelength_nm", 1310.0, "nm", "Operating wavelength"),
        ],
    },
    "opt_mirror": {
        "label": "Partial Mirror",
        "category": "Photonic Passives",
        "doc": "Partially reflective element (facet, DBR, loop mirror): "
               "S11 = S22 = √R, S12 = S21 = j√(1-R) — the j keeps the "
               "lossless 2-port unitary; il_db adds excess loss to both "
               "paths. R = 0 is a transparent thru, R -> 1 a hard mirror "
               "(clamped at 0.995 for conditioning). Two of these around a "
               "waveguide make a Fabry-Perot cavity solved self-consistently "
               "by the nodal field solve (example 35): "
               "$\\text{FSR} = \\lambda^2/(2 n_g L)$, "
               "$\\mathcal{F} = \\pi\\sqrt{r_1 r_2 a}/(1 - r_1 r_2 a)$.",
        "ports": _ports("p1:o p2:o"),
        "params": [
            _p("R", 0.9, "", "Power reflectivity"),
            _p("il_db", 0.0, "dB", "Excess loss"),
        ],
    },
    "opt_term": {
        "label": "Opt. Terminator",
        "category": "Photonic Passives",
        "doc": "Optical absorber terminating a port. Real terminations are "
               "not perfectly matched: return_loss_db sets the residual "
               "reflection (50 dB default — a good index-matched absorber; "
               "raise it to 60+ dB for an ideal load). Terminate unused "
               "splitter/coupler ports and probe transmission at its node — "
               "an open optical port reflects like an open transmission "
               "line.",
        "ports": _ports("p1:o"),
        "params": [
            _p("return_loss_db", 50.0, "dB", "Return loss"),
        ],
    },
    "opt_filter": {
        "label": "Tunable Add-Drop Filter",
        "category": "Photonic Passives",
        "doc": "Tunable optical add-drop filter acting on the coherent field: a "
               "Butterworth (maximally-flat, flat-top) response of the given "
               "order, mapped to the baseband envelope around the carrier and "
               "realised as a compile-time state-space filter, so it filters "
               "the modulation sidebands (narrowing bandwidth_nm below the "
               "signal bandwidth cuts the sidebands and closes the eye). Three "
               "ports: pin (input), drop (the selected channel, Butterworth "
               "lowpass in the shifted frame), and thru (the power-"
               "complementary same-pole highpass, "
               "$|H_{drop}|^2 + |H_{thru}|^2 = 1$ — "
               "a unitary add-drop like a real lossless filter). FWHM = "
               "bandwidth_nm at -3 dB; higher order = flatter top / steeper "
               "skirts. Tune center_nm relative to lambda_nm (the reference/"
               "carrier). Cascade them (thru -> next pin), each tuned to a "
               "different wavelength, to demultiplex a WDM bus. il_db applies "
               "to the drop path.",
        "ports": _ports("pin:o drop:o thru:o"),
        "params": [
            _p("center_nm", 1310.0, "nm", "Center (tunable)", rebuild=True),
            _p("bandwidth_nm", 0.6, "nm", "Bandwidth (FWHM)", rebuild=True),
            _p("order", 3.0, "", "Butterworth order", rebuild=True),
            _p("il_db", 0.0, "dB", "Insertion loss", rebuild=True),
            _p("lambda_nm", 1310.0, "nm", "Carrier wavelength", rebuild=True),
        ],
        "lti": "filter",
    },
    "tia": {
        "label": "TIA",
        "category": "Amplifiers & EQ",
        "doc": "Behavioural transimpedance amplifier: `inp` is a virtual-"
               "ground current input, `out` drives -gain*Iin through a two-"
               "real-pole response (-3 dB at f3db) with an optional tanh "
               "swing limit (0 = off). in_noise is the input-referred "
               "current noise density used by the noise analyses. Parameter "
               "names follow the user's behavioural TIA conventions.",
        "ports": _ports("inp:e out:e"),
        "params": [
            _p("gain_ohm", 10e3, "Ohm", "Transimpedance"),
            _p("f3db", 20e9, "Hz", "Bandwidth"),
            _p("vmax", 0.0, "V", "Swing limit (0 = off)"),
            _p("in_noise", 0.0, "A/rtHz", "Input noise"),
        ],
    },
    "ctle": {
        "label": "CTLE",
        "category": "Amplifiers & EQ",
        "doc": "Continuous-time linear equalizer, 1 zero / 2 poles: "
               "$$H(s) = \\frac{A_{dc}\\,(1 + s/\\omega_z)}"
               "{(1 + s/\\omega_{p1})(1 + s/\\omega_{p2})}$$ "
               "The zero is placed peaking_db below $f_{p1}$ and "
               "$A_{dc} = 1/\\text{peaking}$ so the peaked band sits at "
               "~0 dB while low frequencies are attenuated — the standard "
               "receive-side ISI equalizer shape. "
               "High-impedance input, driven output.",
        "ports": _ports("inp:e out:e"),
        "params": [
            _p("peaking_db", 6.0, "dB", "Peaking"),
            _p("f_p1", 10e9, "Hz", "First pole"),
            _p("fp2_mult", 2.0, "x", "2nd pole mult."),
            _p("A_out", 1.0, "V/V", "Output scale"),
        ],
    },
    "rx_ffe": {
        "label": "Rx FFE",
        "category": "Amplifiers & EQ",
        "doc": "Receiver feed-forward equalizer. Drop it inline on the "
               "receive path: during the transient it is an ideal unity "
               "buffer (high-impedance input, driven output — no loading, "
               "waveform unchanged). The Link BER report reads its tap count "
               "and adaptation rate and applies a data-aided FFE to the "
               "selected “BER vs” probe. adapt_rate = 0 uses the "
               "one-shot least-squares (Wiener) solution; > 0 adapts the taps "
               "with normalized LMS at that step size.",
        "ports": _ports("inp:e out:e"),
        "params": [
            _p("n_taps", 7, "", "FFE taps"),
            _p("adapt_rate", 0.0, "", "Adaptation rate"),
        ],
        "eq": "ffe",
    },
    "rx_dfe": {
        "label": "Rx DFE",
        "category": "Amplifiers & EQ",
        "doc": "Receiver decision-feedback equalizer. Drop it inline on the "
               "receive path: during the transient it is an ideal unity "
               "buffer (high-impedance input, driven output — no loading, "
               "waveform unchanged). The Link BER report reads its tap count "
               "and adaptation rate and cancels that many post-cursor symbols "
               "(data-aided) on the selected “BER vs” probe. "
               "adapt_rate = 0 uses the one-shot least-squares solution; "
               "> 0 adapts with normalized LMS at that step size.",
        "ports": _ports("inp:e out:e"),
        "params": [
            _p("n_taps", 1, "", "DFE taps"),
            _p("adapt_rate", 0.0, "", "Adaptation rate"),
        ],
        "eq": "dfe",
    },
    # --- electrical sources -------------------------------------------------
    "vdc": {
        "label": "DC Voltage",
        "category": "Sources",
        "doc": "Step voltage source: V after `delay`, 0 before.",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("V", 1.0, "V", "Voltage"), _p("delay", 0.0, "s", "Delay")],
    },
    "vpulse": {
        "label": "Pulse Voltage",
        "category": "Sources",
        "doc": "SPICE-style PULSE(v1 v2 td tr tf pw per) source.",
        "ports": _ports("p1:e p2:e"),
        "params": [
            _p("v1", 0.0, "V", "Low level"),
            _p("v2", 1.0, "V", "High level"),
            _p("td", 1e-9, "s", "Delay"),
            _p("tr", 2e-10, "s", "Rise"),
            _p("tf", 2e-10, "s", "Fall"),
            _p("pw", 2e-9, "s", "Pulse width"),
            _p("per", 5e-9, "s", "Period"),
        ],
    },
    "vsin": {
        "label": "Sine Voltage",
        "category": "Sources",
        "doc": "V*sin(2*pi*freq*t + phase) after `delay`.",
        "ports": _ports("p1:e p2:e"),
        "params": [
            _p("V", 1.0, "V", "Amplitude"),
            _p("freq", 1e9, "Hz", "Frequency"),
            _p("phase", 0.0, "rad", "Phase"),
            _p("delay", 0.0, "s", "Delay"),
        ],
    },
    "prbs": {
        "label": "PRBS / Pattern Source",
        "category": "Sources",
        "doc": "Serial-data pattern source: PRBS7/9/11/15/23/31 (LFSR), NRZ "
               "or Gray-coded PAM4 levels between v0/v1, raised-cosine edges. "
               "Optional TX FFE pre/post-cursor de-emphasis (dB), RLM "
               "predistortion for a quadrature-biased MZM (set rlm_vpi to "
               "its V-pi), and RJ/PJ/DCD jitter on the edge times. "
               "mode=pulse emits one isolated UI for pulse-response runs. "
               "mode=qam emits an RRC-shaped I or Q drive (pick qam=qpsk/"
               "qam16/qam64, qam_drive=i/q) for the IQ modulator — one source "
               "per rail, sharing order/seed. "
               "The unit interval is set globally by the top-bar baud rate "
               "(UI = 1/baud), not per source. "
               "The waveform is baked at compile time: parameter edits "
               "recompile the circuit (seconds).",
        "ports": _ports("p1:e p2:e"),
        "params": [
            _p("mode", "nrz", "", "Mode", rebuild=True, kind="enum",
               choices=["nrz", "pam4", "pulse", "qam"]),
            _p("order", 7, "", "PRBS order", rebuild=True, kind="enum",
               choices=[7, 9, 11, 13, 15, 23, 31]),
            _p("v0", -0.5, "V", "Low level", rebuild=True),
            _p("v1", 0.5, "V", "High level", rebuild=True),
            _p("tr", 20e-12, "s", "Edge time (20-80%)", rebuild=True),
            _p("seed", 1, "", "PRBS seed", rebuild=True),
            _p("qam", "qpsk", "", "QAM order", rebuild=True, kind="enum",
               choices=["qpsk", "qam16", "qam64"]),
            _p("qam_drive", "i", "", "QAM rail", rebuild=True, kind="enum",
               choices=["i", "q"]),
            _p("rrc_beta", 0.1, "", "RRC roll-off", rebuild=True),
            _p("sps", 16, "", "QAM samples/UI", rebuild=True),
            _p("ffe_pre_db", 0.0, "dB", "TX FFE pre-cursor", rebuild=True),
            _p("ffe_post_db", 0.0, "dB", "TX FFE post-cursor", rebuild=True),
            _p("rlm_vpi", 0.0, "V", "RLM V-pi (0 = off)", rebuild=True),
            _p("rj_ui", 0.0, "UI", "Random jitter (rms)", rebuild=True),
            _p("pj_ui", 0.0, "UI", "Periodic jitter (amp)", rebuild=True),
            _p("pj_freq", 10e6, "Hz", "PJ frequency", rebuild=True),
            _p("dcd_ui", 0.0, "UI", "Duty-cycle distortion", rebuild=True),
        ],
        "wave": "prbs",
    },
    "vpwl": {
        "label": "PWL Source",
        "category": "Sources",
        "doc": "Piecewise-linear voltage source. `data` holds 't v' break-"
               "point pairs (one per line, comma or space separated; SI "
               "notation like 1e-9 works). The value holds before the first "
               "and after the last point. Use the inspector's Load CSV "
               "button to fill it from a file (e.g. a waveform exported "
               "from another simulator).",
        "ports": _ports("p1:e p2:e"),
        "params": [
            _p("data", "0 0\n1e-9 0\n1.1e-9 1\n3e-9 1", "", "Breakpoints",
               rebuild=True, kind="text"),
        ],
        "wave": "pwl",
    },
    "idc": {
        "label": "DC Current",
        "category": "Sources",
        "doc": "Constant current source, p1 -> p2.",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("I", 1e-3, "A", "Current")],
    },
    # --- electrical passives ------------------------------------------------
    "resistor": {
        "label": "Resistor",
        "category": "Electrical",
        "doc": "Ohm's law.",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("R", 1e3, "Ohm", "Resistance")],
    },
    "capacitor": {
        "label": "Capacitor",
        "category": "Electrical",
        "doc": "Linear capacitor.",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("C", 1e-12, "F", "Capacitance")],
    },
    "inductor": {
        "label": "Inductor",
        "category": "Electrical",
        "doc": "Linear inductor.",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("L", 1e-9, "H", "Inductance")],
    },
    "diode": {
        "label": "Diode",
        "category": "Electrical",
        "doc": "Shockley diode: $I = I_s\\,(e^{V_d/(n V_t)} - 1)$. p1 = anode.",
        "ports": _ports("p1:e p2:e"),
        "params": [
            _p("Is", 1e-12, "A", "Saturation current"),
            _p("n", 1.0, "", "Ideality"),
            _p("Vt", 0.02585, "V", "Thermal voltage"),
        ],
    },
    "nmos": {
        "label": "NMOS (square-law)",
        "category": "Electrical",
        "doc": "Square-law N-channel MOSFET with channel-length modulation.",
        "ports": _ports("d:e g:e s:e"),
        "params": [
            _p("Kp", 2e-5, "A/V^2", "Transconductance"),
            _p("W", 1e-5, "m", "Width"),
            _p("L", 1e-6, "m", "Length"),
            _p("Vth", 1.0, "V", "Threshold"),
            _p("lam", 0.0, "1/V", "Lambda (CLM)"),
        ],
    },
    "pmos": {
        "label": "PMOS (square-law)",
        "category": "Electrical",
        "doc": "Square-law P-channel MOSFET (Vth negative).",
        "ports": _ports("d:e g:e s:e"),
        "params": [
            _p("Kp", 1e-5, "A/V^2", "Transconductance"),
            _p("W", 2e-5, "m", "Width"),
            _p("L", 1e-6, "m", "Length"),
            _p("Vth", -1.0, "V", "Threshold"),
            _p("lam", 0.0, "1/V", "Lambda (CLM)"),
        ],
    },
}


def _sky130_fet_entry(device: str, label: str, doc: str,
                      w: float, l: float) -> dict:  # noqa: E741
    return {
        "label": label,
        "category": "SKY130 FETs",
        "doc": f"Real sky130_fd_pr__{device}, BSIM4.8 via OSDI — exact PDK "
               f"physics. {doc} Transient auto-switches to the fixed-step "
               "BDF2 solver. First use of a new geometry extracts + compiles "
               "the model card (slow once, cached forever).",
        "ports": _ports("d:e g:e s:e b:e"),
        "params": [
            _p("w_um", w, "um", "Width", rebuild=True),
            _p("l_um", l, "um", "Length", rebuild=True),
        ],
        "sky130": {"kind": "fet", "device": device},
    }


def _sky130_res_entry(cell: str, label: str, doc: str,
                      params: list[dict]) -> dict:
    return {
        "label": label,
        "category": "SKY130 Passives",
        "doc": f"{doc} The R value is measured from the real PDK model by "
               "ngspice (body tied to ground) and backs an ideal resistor — "
               "first use of a new geometry parses the sky130 library "
               "(slow once, cached forever).",
        "ports": _ports("p1:e p2:e"),
        "params": params,
        "sky130": {"kind": "res", "cell": cell},
    }


CATALOG.update({
    # --- SKY130 FETs (BSIM4 via OSDI) --------------------------------------
    "sky130_nfet": _sky130_fet_entry(
        "nfet_01v8", "SKY130 NFET 1.8V", "Standard-Vt 1.8 V core device.",
        1.0, 0.15),
    "sky130_nfet_lvt": _sky130_fet_entry(
        "nfet_01v8_lvt", "SKY130 NFET 1.8V LVT", "Low-Vt 1.8 V device.",
        1.0, 0.35),
    "sky130_nfet_5v": _sky130_fet_entry(
        "nfet_g5v0d10v5", "SKY130 NFET 5V", "5.0 V (10.5 V drain) I/O device.",
        1.0, 0.5),
    "sky130_nfet_nvt": _sky130_fet_entry(
        "nfet_05v0_nvt", "SKY130 NFET 5V native", "Native (~0 Vt) 5 V device.",
        1.0, 0.9),
    "sky130_pfet": _sky130_fet_entry(
        "pfet_01v8", "SKY130 PFET 1.8V", "Standard-Vt 1.8 V core device.",
        2.0, 0.15),
    "sky130_pfet_lvt": _sky130_fet_entry(
        "pfet_01v8_lvt", "SKY130 PFET 1.8V LVT", "Low-Vt 1.8 V device.",
        2.0, 0.35),
    "sky130_pfet_hvt": _sky130_fet_entry(
        "pfet_01v8_hvt", "SKY130 PFET 1.8V HVT", "High-Vt 1.8 V device.",
        2.0, 0.15),
    "sky130_pfet_5v": _sky130_fet_entry(
        "pfet_g5v0d10v5", "SKY130 PFET 5V", "5.0 V (10.5 V drain) I/O device.",
        2.0, 0.5),
    # --- SKY130 passives (values measured from the PDK by ngspice) ---------
    "sky130_res_po": _sky130_res_entry(
        "res_generic_po", "SKY130 Poly Res",
        "Generic poly resistor, ~48.2 Ohm/sq; free W/L.",
        [_p("w_um", 1.0, "um", "Width", rebuild=True),
         _p("l_um", 10.0, "um", "Length", rebuild=True)]),
    "sky130_res_nd": _sky130_res_entry(
        "res_generic_nd", "SKY130 N-Diff Res",
        "Generic n-diffusion resistor, ~120 Ohm/sq; free W/L.",
        [_p("w_um", 1.0, "um", "Width", rebuild=True),
         _p("l_um", 10.0, "um", "Length", rebuild=True)]),
    "sky130_res_high_po": _sky130_res_entry(
        "res_high_po_0p69", "SKY130 High-R Poly",
        "Precision P+ poly resistor, ~320 Ohm/sq, fixed 0.69 um width.",
        [_p("l_um", 5.0, "um", "Length", rebuild=True)]),
    "sky130_res_xhigh_po": _sky130_res_entry(
        "res_xhigh_po_0p69", "SKY130 X-High-R Poly",
        "Precision poly resistor, ~2 kOhm/sq, fixed 0.69 um width.",
        [_p("l_um", 5.0, "um", "Length", rebuild=True)]),
    "sky130_cap_mim": {
        "label": "SKY130 MiM Cap (M3)",
        "category": "SKY130 Passives",
        "doc": "Metal-insulator-metal capacitor between met3/capm. The C "
               "value is measured from the real PDK model by ngspice and "
               "backs an ideal capacitor — first use of a new geometry "
               "parses the sky130 library (slow once, cached forever).",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("w_um", 10.0, "um", "Width", rebuild=True),
                   _p("l_um", 10.0, "um", "Length", rebuild=True)],
        "sky130": {"kind": "cap", "cell": "cap_mim_m3_1"},
    },
    "sky130_cap_mim2": {
        "label": "SKY130 MiM Cap (M4)",
        "category": "SKY130 Passives",
        "doc": "Metal-insulator-metal capacitor between met4/cap2m. The C "
               "value is measured from the real PDK model by ngspice and "
               "backs an ideal capacitor — first use of a new geometry "
               "parses the sky130 library (slow once, cached forever).",
        "ports": _ports("p1:e p2:e"),
        "params": [_p("w_um", 10.0, "um", "Width", rebuild=True),
                   _p("l_um", 10.0, "um", "Length", rebuild=True)],
        "sky130": {"kind": "cap", "cell": "cap_mim_m3_2"},
    },
    # --- analog blocks -------------------------------------------------------
    "opamp": {
        "label": "Op-Amp (ideal)",
        "category": "Amplifiers & EQ",
        "doc": "Ideal VCVS op-amp: V(out_p,out_m) = A * V(in_p,in_m); inputs "
               "draw no current.",
        "ports": _ports("out_p:e out_m:e in_p:e in_m:e"),
        "params": [_p("A", 1e6, "V/V", "Open-loop gain")],
    },
    # --- optical <-> electrical power-domain bridges -------------------------
    "opt_f2p": {
        "label": "Field -> Power tap",
        "category": "Detectors & Bridges",
        "doc": "Bridge for power-domain models (e.g. uploaded .va with a "
               "popt-style port): absorbs the coherent field (matched, no "
               "reflection) and drives an electrical node with V = |E|^2 "
               "in watts.",
        "ports": _ports("c:o p:e"),
        "params": [],
    },
    "opt_p2f": {
        "label": "Power -> Field source",
        "category": "Detectors & Bridges",
        "doc": "Bridge from a power-domain node (V = watts) back into the "
               "coherent-field world: E = sqrt(max(V, 0)), zero phase.",
        "ports": _ports("p:e c:o"),
        "params": [],
    },
    # --- reference ------------------------------------------------------------
    "ground": {
        "label": "Ground",
        "category": "Reference",
        "doc": "0 V / zero-field reference. Required in every circuit.",
        "ports": _ports("p1:e"),
        "params": [],
    },
    # Subcircuit boundary marker. Only meaningful inside a subcircuit
    # definition sheet: its single pin `p` is the net exposed as the named
    # external port. The netlist flattener (webapp/subcircuit.py) splices it
    # onto the parent net and drops the marker, so `port` never reaches
    # circulax — hence "pseudo".
    "port": {
        "label": "Subcircuit Port",
        "category": "Subcircuit",
        "doc": "Boundary port of a subcircuit definition. Name it and pick its "
               "domain (optical/electrical); wire pin $p$ to the internal net "
               "you want to expose. Instances of the subcircuit show one pin "
               "per port.",
        "ports": _ports("p:e"),
        "pseudo": True,
        "params": [
            _p("name", "", "", "Port name", kind="text"),
            _p("domain", "optical", "", "Domain", kind="enum",
               choices=["optical", "electrical"]),
        ],
    },
})


# ---------------------------------------------------------------------------
# models_map construction (lazy, heavyweight)
# ---------------------------------------------------------------------------

def _photodiode():
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("po_p", "po_n", "an", "cat"), states=("x1", "x2"))
    def Photodiode(
        signals: Signals,
        s: States,
        R: float = 0.8,
        Idk: float = 1e-9,
        Cj: float = 100e-15,
        f3db: float = 0.0,
        isat: float = 0.0,
        Yopt: float = 1.0,
    ) -> tuple[dict, dict]:
        e = signals.po_p - signals.po_n
        i_opt = Yopt * e
        power = jnp.abs(e) ** 2
        iph = R * power + Idk
        # intrinsic (transit-time) bandwidth: two identical real poles with
        # -3 dB at f3db; f3db = 0 collapses tau to 0 and the states become
        # algebraic copies of iph (no extra dynamics)
        tau = jnp.where(f3db > 0.0,
                        0.6436 / (2.0 * jnp.pi * jnp.maximum(f3db, 1.0)), 0.0)
        f_x1 = s.x1 - iph
        f_x2 = s.x2 - s.x1
        i_bw = s.x2.real
        # soft output saturation (space-charge screening); isat = 0 -> off
        i_out = jnp.where(
            isat > 0.0,
            jnp.maximum(isat, 1e-30) * jnp.tanh(
                i_bw / jnp.maximum(isat, 1e-30)),
            i_bw)
        f = {"po_p": i_opt, "po_n": -i_opt, "cat": i_out, "an": -i_out,
             "x1": f_x1, "x2": f_x2}
        vj = signals.cat - signals.an
        q = {"cat": Cj * vj, "an": -Cj * vj,
             "x1": tau * s.x1, "x2": tau * s.x2}
        return f, q

    return Photodiode


def _apd_tau(f3db, gbp, Mc):
    """Effective transit/gain-bandwidth pole time constant for the APD.

    Two identical real poles give a -3 dB corner at the smaller of the
    transit-time bandwidth ``f3db`` and the gain-bandwidth-limited ``gbp/M``
    (each 0 = unlimited). Above the corner gain M0 = gbp/f3db the effective
    bandwidth falls as gbp/M; below it the transit time dominates. tau = 0
    (no extra dynamics) when both are unset.
    """
    import jax.numpy as jnp

    f_gb = jnp.where(gbp > 0.0, gbp / Mc, jnp.inf)
    f_tr = jnp.where(f3db > 0.0, f3db, jnp.inf)
    f_eff = jnp.minimum(f_gb, f_tr)
    return jnp.where(jnp.isfinite(f_eff),
                     0.6436 / (2.0 * jnp.pi * jnp.maximum(f_eff, 1.0)), 0.0)


def _apd():
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("po_p", "po_n", "an", "cat"), states=("x1", "x2"))
    def APD(
        signals: Signals,
        s: States,
        R: float = 0.8,
        M: float = 10.0,
        k_ion: float = 0.3,
        Idk_bulk: float = 1e-9,
        Idk_surf: float = 1e-9,
        Cj: float = 100e-15,
        f3db: float = 0.0,
        gbp: float = 0.0,
        isat: float = 0.0,
        Yopt: float = 1.0,
    ) -> tuple[dict, dict]:
        e = signals.po_p - signals.po_n
        i_opt = Yopt * e
        power = jnp.abs(e) ** 2
        # avalanche gain multiplies the primary photocurrent and the bulk dark
        # current; the surface dark current bypasses the multiplication region
        Mc = jnp.maximum(M, 1.0)
        i_prim = R * power + Idk_bulk
        i_mult = Mc * i_prim + Idk_surf
        # gain-bandwidth tradeoff: effective bandwidth ~ gbp/M above the corner
        tau = _apd_tau(f3db, gbp, Mc)
        f_x1 = s.x1 - i_mult
        f_x2 = s.x2 - s.x1
        i_bw = s.x2.real
        # soft output saturation (space-charge screening); isat = 0 -> off
        i_out = jnp.where(
            isat > 0.0,
            jnp.maximum(isat, 1e-30) * jnp.tanh(
                i_bw / jnp.maximum(isat, 1e-30)),
            i_bw)
        f = {"po_p": i_opt, "po_n": -i_opt, "cat": i_out, "an": -i_out,
             "x1": f_x1, "x2": f_x2}
        vj = signals.cat - signals.an
        q = {"cat": Cj * vj, "an": -Cj * vj,
             "x1": tau * s.x1, "x2": tau * s.x2}
        return f, q

    return APD


def _iq_modulator():
    """Nested-MZM IQ modulator on the coherent field (ALE-77).

    Two child MZMs in parallel (I and Q arms), each null-biased and driven
    push-pull so its field transmission is ``sin(pi*(V+bias)/(2*vpi))`` (bipolar
    drive -> bipolar field), with the Q arm phase-shifted 90 degrees before the
    combiner. The composite field transmission is

        t = 0.5 * il * [ sin(pi*(V_I+bias_i)/(2*vpi))
                         + e^{j*(pi/2 + qerr)} * sin(pi*(V_Q+bias_q)/(2*vpi)) ]

    so ``E_out = t * E_in`` maps the (I, Q) drives onto the complex plane — the
    coherent transmitter behind QPSK/QAM. ``qerr`` is the quadrature (90-degree
    hybrid) phase error. Uses the same S->Y 2-port assembly as ``cx.mzm`` so the
    optical path is matched (no reflection); ``cel`` loads each driver.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component
    from circulax.s_transforms import s_to_y

    @component(ports=("pin", "pout", "vip", "vin", "vqp", "vqn"))
    def IQModulator(
        signals: Signals,
        s: States,
        vpi: float = 3.0,        # half-wave voltage of each child MZM [V]
        vbias_i: float = 0.0,    # I-arm bias offset [V]
        vbias_q: float = 0.0,    # Q-arm bias offset [V]
        qerr: float = 0.0,       # quadrature (90 deg) phase error [rad]
        il_db: float = 6.0,      # excess field insertion loss [dB]
        cel: float = 50e-15,     # electrode capacitance [F]
    ) -> tuple[dict, dict]:
        il = 10.0 ** (-il_db / 20.0)                 # amplitude (field) loss
        vi = (signals.vip - signals.vin).real
        vq = (signals.vqp - signals.vqn).real
        ti = jnp.sin(jnp.pi * (vi + vbias_i) / (2.0 * vpi))
        tq = jnp.sin(jnp.pi * (vq + vbias_q) / (2.0 * vpi))
        t = 0.5 * il * (ti + jnp.exp(1j * (jnp.pi / 2.0 + qerr)) * tq)

        S = jnp.array([[0.0 * t, t], [t, 0.0 * t]], dtype=jnp.complex128)
        Y = s_to_y(S)
        v_vec = jnp.array([signals.pin, signals.pout], dtype=jnp.complex128)
        i_vec = Y @ v_vec

        f = {"pin": i_vec[0], "pout": i_vec[1],
             "vip": 0.0, "vin": 0.0, "vqp": 0.0, "vqn": 0.0}
        qi = cel * (signals.vip - signals.vin)
        qq = cel * (signals.vqp - signals.vqn)
        q = {"vip": qi, "vin": -qi, "vqp": qq, "vqn": -qq}
        return f, q

    return IQModulator


def _coherent_rx():
    """Single-pol coherent receiver front-end: 90-degree hybrid + balanced PDs.

    The signal and LO fields beat in an ideal 90-degree optical hybrid; the two
    balanced photodiode pairs cancel the direct-detection (|E|^2) terms and
    deliver the in-phase and quadrature photocurrents

        i_I = R * Re(E_sig * conj(E_lo)),   i_Q = R * Im(E_sig * conj(E_lo)),

    i.e. the complex baseband ``r = i_I + j*i_Q = R * E_sig * conj(E_lo)`` that
    the coherent DSP (``webapp/coherent.py``) demodulates. Both optical inputs
    are matched absorbers (like the photodiode, no reflection); the LO port
    takes a second ``cw_laser``. Differential current outputs ``i_p/i_n`` (I) and
    ``qp/qn`` (Q).
    """
    from circulax.components.base_component import Signals, States, component

    @component(ports=("sig", "lo", "i_p", "i_n", "q_p", "q_n"))
    def CoherentRx(
        signals: Signals,
        s: States,
        R: float = 0.8,          # responsivity [A/W]
        Yopt: float = 1.0,       # matched-absorber admittance
    ) -> tuple[dict, dict]:
        es = signals.sig
        el = signals.lo
        i_sig = Yopt * es                            # matched hybrid inputs
        i_lo = Yopt * el
        # E_sig * conj(E_lo), written via re/im (non-holomorphic, like the PD)
        re = es.real * el.real + es.imag * el.imag
        im = es.imag * el.real - es.real * el.imag
        i_i = R * re
        i_q = R * im
        # sign matches the photodiode bridge: a positive beat sources current
        # out of the '+' terminal (V(i_p) = +i_i across a cathode-side load)
        f = {"sig": i_sig, "lo": i_lo,
             "i_p": -i_i, "i_n": i_i, "q_p": -i_q, "q_n": i_q}
        return f, {}

    return CoherentRx


def _tia():
    """Behavioural TIA macro, parameter names mirroring the user's tia.py.

    ``in`` is a virtual-ground current input (V = 0, like an ideal feedback
    summing node); ``out`` drives ``-gain_ohm * i_in`` through a two-real-
    pole low-pass (-3 dB at f3db) with an optional tanh swing limit.
    ``in_noise`` (A/sqrt(Hz), input-referred) is used by the noise analyses.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("inp", "out"), states=("i_vg", "x1", "x2", "i_out"))
    def TIAMacro(
        signals: Signals,
        s: States,
        gain_ohm: float = 10e3,
        f3db: float = 20e9,
        vmax: float = 0.0,
        in_noise: float = 0.0,
    ) -> tuple[dict, dict]:
        tau = 0.6436 / (2.0 * jnp.pi * jnp.maximum(f3db, 1.0))
        v_tgt = -gain_ohm * s.i_vg
        f_x1 = s.x1 - v_tgt
        f_x2 = s.x2 - s.x1
        v_lp = s.x2.real
        v_out = jnp.where(
            vmax > 0.0,
            jnp.maximum(vmax, 1e-30) * jnp.tanh(
                v_lp / jnp.maximum(vmax, 1e-30)),
            v_lp)
        return {
            "inp": s.i_vg,
            "i_vg": signals.inp,                 # V(inp) = 0 (virtual ground)
            "x1": f_x1, "x2": f_x2,
            "out": s.i_out,
            "i_out": signals.out - v_out,        # drive the output node
        }, {"x1": tau * s.x1, "x2": tau * s.x2}

    return TIAMacro


def _ctle():
    """CTLE: A_dc (1 + s/wz) / ((1 + s/wp1)(1 + s/wp2)) — the exact
    1-zero/2-pole shape of the user's ctle.py. The zero is placed so
    20*log10(wp1/wz) = peaking_db and A_dc = wz/wp1, giving ~unity gain in
    the peaked band. High-impedance input, driven output.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("inp", "out"), states=("x1", "x2", "i_out"))
    def CTLE(
        signals: Signals,
        s: States,
        peaking_db: float = 6.0,
        f_p1: float = 10e9,
        fp2_mult: float = 2.0,
        A_out: float = 1.0,
    ) -> tuple[dict, dict]:
        wp1 = 2.0 * jnp.pi * jnp.maximum(f_p1, 1.0)
        wp2 = fp2_mult * wp1
        k = 10.0 ** (peaking_db / 20.0)
        wz = wp1 / k
        a_dc = 1.0 / k
        vin = (signals.inp).real
        # x1 = vin/(1+s/wp1), x2 = x1/(1+s/wp2); y needs x2 + s x2/wz where
        # s x2 = wp2 (x1 - x2)
        f_x1 = s.x1 - vin
        f_x2 = s.x2 - s.x1
        y = A_out * a_dc * (s.x2 + (wp2 / wz) * (s.x1 - s.x2)).real
        return {
            "inp": 0.0,
            "x1": f_x1, "x2": f_x2,
            "out": s.i_out,
            "i_out": signals.out - y,
        }, {"x1": s.x1 / wp1, "x2": s.x2 / wp2}

    return CTLE


def _rx_eq_buffer():
    """Rx FFE / DFE placeholder: an ideal unity buffer.

    The receive-side equalizers are *measurement-time* post-processing (see
    linkpost.py) rather than a filter in the transient solve, so the inline
    schematic block is electrically transparent: a high-impedance input
    (draws no current, does not load the channel) driving an output held at
    the input voltage. ``n_taps``/``adapt_rate`` are read by the Link BER
    report, not by this model — they are accepted here only because every
    catalog param is passed through as a component setting.
    """
    from circulax.components.base_component import Signals, States, component

    @component(ports=("inp", "out"), states=("i_out",))
    def RxEqBuffer(
        signals: Signals,
        s: States,
        n_taps: float = 7.0,
        adapt_rate: float = 0.0,
    ) -> tuple[dict, dict]:
        vin = signals.inp.real
        return {
            "inp": 0.0,                      # high-Z input (no current drawn)
            "out": s.i_out,
            "i_out": signals.out - vin,      # drive output to the input volts
        }, {}

    return RxEqBuffer


def _pulse_mod():
    import jax
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    @source(ports=("pin", "pout"), states=("i_out",))
    def PulseModulator(
        signals: Signals,
        s: States,
        t: float,
        p_on: float = 18e-3,
        p_off: float = 3e-3,
        t0: float = 0.5e-9,
        t1: float = 2.0e-9,
        tr: float = 80e-12,
    ) -> tuple[dict, dict]:
        k = 1.0 / tr
        env = jax.nn.sigmoid(k * (t - t0)) - jax.nn.sigmoid(k * (t - t1))
        trans = jnp.sqrt((p_off + (p_on - p_off) * env) / p_on)
        constraint = signals.pout - trans * signals.pin
        return {"pin": 0.0, "pout": s.i_out, "i_out": constraint}, {}

    return PulseModulator


def _waveform_source(wt, wv):
    """Voltage source following precomputed (t, v) breakpoints (jnp.interp).

    The arrays are baked as closure constants — not JAX leaves — so pattern
    parameters are compile-time (rebuild=True in the catalog). Values clamp
    to the first/last breakpoint outside the time range.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    wt_c = jnp.asarray(wt)
    wv_c = jnp.asarray(wv)

    @source(ports=("p1", "p2"), states=("i_src",), amplitude_param="scale")
    def WaveformSource(
        signals: Signals,
        s: States,
        t: float,
        scale: float = 1.0,
    ) -> tuple[dict, dict]:
        v_val = scale * jnp.interp(t, wt_c, wv_c)
        constraint = (signals.p1 - signals.p2) - v_val
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}

    return WaveformSource


def _power_to_field():
    """Adapter: power-domain optical node (V = watts) -> coherent field node.

    The repo's power-convention VA models (laser_dml, laser_rate, mzm_tw)
    carry optical power as a node voltage; the webapp's optical domain is the
    coherent field E with |E|^2 = power. This source enforces
    ``V(c) = sqrt(max(V(p), 0))`` (zero phase — power models carry none),
    drawing nothing from the power node.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("p", "c"), states=("i_c",))
    def PowerToField(signals: Signals, s: States) -> tuple[dict, dict]:
        power = jnp.maximum(signals.p.real, 0.0)
        field = jnp.sqrt(power + 1e-30)
        return {"p": 0.0, "c": s.i_c, "i_c": signals.c - field}, {}

    return PowerToField


def _field_to_power():
    """Adapter: coherent field node -> power-domain optical node.

    Matched absorber on the field side (like the photodiode: i = Yopt*E, no
    reflection) that re-emits ``V(p) = |E|^2`` for a power-convention VA
    model's optical input.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("c", "p"), states=("i_p",))
    def FieldToPower(signals: Signals, s: States) -> tuple[dict, dict]:
        e = signals.c
        return {"c": e, "p": s.i_p,
                "i_p": signals.p - jnp.abs(e) ** 2}, {}

    return FieldToPower


def _field_to_ri_matched():
    """Adapter: complex field node -> (re, im) pair, MATCHED absorber.

    Like :func:`cx.field_to_ri`, but the field side is a matched load
    (``i = E``, Yopt = 1, the photodiode/terminator convention) instead of a
    zero-current tap. A directed-wave VA model's *input* port is a receiver:
    fed from a nodal reciprocal element (a waveguide/modulator output), a
    high-Z tap leaves that port open and the S-matrix reflects, doubling the
    field; the matched absorber terminates it so ``V(re)/V(im)`` read the true
    transmitted field. Fed from another directed output (an ideal ``ri_to_field``
    source) the extra load is harmless — the source fixes the node either way.
    """
    from circulax.components.base_component import Signals, States, component

    @component(ports=("c", "re", "im"), states=("i_re", "i_im"))
    def FieldToRIMatched(signals: Signals, s: States) -> tuple[dict, dict]:
        e = signals.c
        return {
            "c": e,                       # matched absorber: no reflection
            "re": s.i_re,
            "im": s.i_im,
            "i_re": signals.re - e.real,
            "i_im": signals.im - e.imag,
        }, {}

    return FieldToRIMatched


def _opt_term():
    """Optical terminator with finite return loss.

    One-port with S11 = r = 10^(-RL/20): the equivalent nodal admittance is
    Y = (1-r)/(1+r) (r = 0 recovers the matched absorber i = E). Real
    absorbers reflect a little; 50 dB is a good index-matched load.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    @component(ports=("p1",))
    def OptTerm(signals: Signals, s: States,
                return_loss_db: float = 50.0) -> tuple[dict, dict]:
        r = 10.0 ** (-jnp.abs(return_loss_db) / 20.0)
        return {"p1": (1.0 - r) / (1.0 + r) * signals.p1}, {}

    return OptTerm


def _grating_r():
    """Grating coupler: circulax's Gaussian passband + fiber-side
    back-reflection.

    Same spectral model as circulax.components.photonic.Grating, with
    S11 = 10^(-back_refl_db/20) on the grating (fiber) port, clamped so the
    lossy 2-port stays passive (|S11|^2 + |S21|^2 <= 1).
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component
    from circulax.s_transforms import s_to_y

    @component(ports=("grating", "waveguide"))
    def GratingR(
        signals: Signals,
        s: States,
        center_wavelength_nm: float = 1310.0,
        peak_loss_dB: float = 0.0,
        bandwidth_1dB: float = 20.0,
        back_refl_db: float = 25.0,
        wavelength_nm: float = 1310.0,
    ) -> tuple[dict, dict]:
        delta = wavelength_nm - center_wavelength_nm
        loss_dB = peak_loss_dB + (delta / (0.5 * bandwidth_1dB)) ** 2
        T = jnp.minimum(10.0 ** (-loss_dB / 20.0), 0.9999)
        rb = 10.0 ** (-jnp.abs(back_refl_db) / 20.0)
        rb = jnp.minimum(rb, jnp.sqrt(jnp.maximum(1.0 - T ** 2, 1e-8)))
        S = jnp.array([[rb + 0j, T + 0j], [T + 0j, 0j]],
                      dtype=jnp.complex128)
        Y = s_to_y(S)
        v = jnp.array([signals.grating, signals.waveguide],
                      dtype=jnp.complex128)
        iv = Y @ v
        return {"grating": iv[0], "waveguide": iv[1]}, {}

    return GratingR


def _opt_mirror():
    """Partially reflective element: S = [[r, jt], [jt, r]], r² + t² = 1.

    The j on the transmitted path keeps the lossless 2-port unitary (same
    convention as the couplers); il_db scales both paths. R is clamped to
    0.995 so I + S stays well-conditioned in the S->Y conversion.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component
    from circulax.s_transforms import s_to_y

    @component(ports=("p1", "p2"))
    def OptMirror(signals: Signals, s: States, R: float = 0.9,
                  il_db: float = 0.0) -> tuple[dict, dict]:
        a = 10.0 ** (-jnp.abs(il_db) / 20.0)
        Rc = jnp.clip(R, 0.0, 0.995)
        r = a * jnp.sqrt(Rc)
        t = a * jnp.sqrt(1.0 - Rc)
        S = jnp.array([[r + 0j, 1j * t], [1j * t, r + 0j]],
                      dtype=jnp.complex128)
        Y = s_to_y(S)
        v = jnp.array([signals.p1, signals.p2], dtype=jnp.complex128)
        iv = Y @ v
        return {"p1": iv[0], "p2": iv[1]}, {}

    return OptMirror


def _lti_vt(A, B, C, D, zin: float = 0.0):
    """Electrical LTI voltage-transfer block from a real state-space.

    High-impedance input (or a zin shunt when zin > 0, e.g. a matched
    S-parameter port), driven output: out = C x + D vin, dx/dt = A x + B vin.
    Matrices are baked per instance (vector-fitted channels).
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    A_c = jnp.asarray(A)
    B_c = jnp.asarray(B)
    C_c = jnp.asarray(C)
    D_c = float(D)
    n = A_c.shape[0]
    st = tuple(f"x{i}" for i in range(n)) + ("i_out",)

    @component(ports=("inp", "out"), states=st)
    def LTIBlock(signals: Signals, s: States) -> tuple[dict, dict]:
        x = jnp.stack([getattr(s, f"x{i}") for i in range(n)])
        vin = signals.inp
        dx = A_c @ x + B_c * vin
        y = (C_c @ x + D_c * vin).real
        f = {"inp": signals.inp / zin if zin > 0 else 0.0,
             "out": s.i_out, "i_out": signals.out - y}
        for i in range(n):
            f[f"x{i}"] = -dx[i]
        q = {f"x{i}": getattr(s, f"x{i}") for i in range(n)}
        return f, q

    return LTIBlock


def _lti_field(poles, res, d):
    """Optical (coherent-field) LTI block from a complex diagonal fit.

    Matched absorber at p1 (i = E_in, no reflection), driven field at p2:
    E_out = sum_k c_k x_k + d E_in with dx_k/dt = p_k x_k + E_in. States are
    complex — legal inside an is_complex circuit. Used for chromatic
    dispersion, whose baseband envelope response is not conjugate-symmetric.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    p_c = jnp.asarray(poles)
    c_c = jnp.asarray(res)
    d_c = complex(d)
    n = len(poles)
    st = tuple(f"x{i}" for i in range(n)) + ("i_out",)

    @component(ports=("p1", "p2"), states=st)
    def FieldLTI(signals: Signals, s: States) -> tuple[dict, dict]:
        x = jnp.stack([getattr(s, f"x{i}") for i in range(n)])
        ein = signals.p1
        y = jnp.sum(c_c * x) + d_c * ein
        f = {"p1": ein, "p2": s.i_out, "i_out": signals.p2 - y}
        for i in range(n):
            f[f"x{i}"] = -(p_c[i] * x[i] + ein)
        q = {f"x{i}": getattr(s, f"x{i}") for i in range(n)}
        return f, q

    return FieldLTI


def _lti_field_drop(poles, res_d, res_t):
    """Add-drop optical filter: coherent-field LTI with a DROP and a THRU port.

    Three ports sharing one set of complex states (dx_k/dt = p_k x_k + E_in):
    ``pin`` absorbs the input (matched, no reflection), ``drop`` carries the
    selected channel E_drop = sum(res_d x), and ``thru`` carries the
    power-complementary remainder E_thru = E_in + sum(res_t x) — the same-pole
    highpass, so |H_drop|^2 + |H_thru|^2 = 1 at every frequency like a real
    lossless (unitary) add-drop. (A naive thru of E_in - E_drop is NOT
    passive: its skirt gains +4 dB where H_drop's phase rotates.) Cascading
    these (each tuned to a different channel, thru -> next pin) demultiplexes
    a WDM bus: on resonance the channel drops fully and thru nulls; off
    resonance it passes to thru untouched.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    p_c = jnp.asarray(poles)
    cd_c = jnp.asarray(res_d)
    ct_c = jnp.asarray(res_t)
    n = len(poles)
    st = tuple(f"x{i}" for i in range(n)) + ("i_drop", "i_thru")

    @component(ports=("pin", "drop", "thru"), states=st)
    def FieldDrop(signals: Signals, s: States) -> tuple[dict, dict]:
        x = jnp.stack([getattr(s, f"x{i}") for i in range(n)])
        ein = signals.pin
        y_drop = jnp.sum(cd_c * x)                 # lowpass:  wc^n / B(s)
        y_thru = ein + jnp.sum(ct_c * x)           # highpass: s^n / B(s)
        f = {"pin": ein,
             "drop": s.i_drop, "i_drop": signals.drop - y_drop,
             "thru": s.i_thru, "i_thru": signals.thru - y_thru}
        for i in range(n):
            f[f"x{i}"] = -(p_c[i] * x[i] + ein)
        q = {f"x{i}": getattr(s, f"x{i}") for i in range(n)}
        return f, q

    return FieldDrop


def _fiber_nl_field(poles, res, d, gamma: float, dz: float, n_seg: int):
    """Nonlinear-fibre split-step block on the coherent field (p1 -> p2).

    A causal, in-transient split-step Fourier method: ``n_seg`` identical
    dispersion segments (each the linear all-pass over ``dz``, realised by the
    complex diagonal state-space ``poles``/``res``/``d`` from ``lti._fit_dispersion``)
    interleaved with the instantaneous Kerr phase ``exp(-j*gamma*|E|^2*dz)``.
    Every segment owns ``len(poles)`` complex states ``dx/dt = p x + u_seg``
    (``u_seg`` its input field); its output ``sum(res*x) + d*u_seg`` is Kerr-
    rotated and fed to the next segment. SPM/XPM/FWM emerge from the field
    evolution, and ``gamma = 0`` collapses to ``n_seg`` cascaded dispersion
    segments == the full-length ``fiber_cd``. This is a first-order (Lie) split
    — dispersion then Kerr per segment — so accuracy is set by ``n_seg`` (raise
    it for strong nonlinearity); the batch reference solver ``webapp/ssfm.py``
    uses the symmetric split and shares the beta2/beta3/gamma definitions, and
    is what the soliton / SPM / FWM analytics are pinned against.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, component

    p_c = jnp.asarray(poles)
    c_c = jnp.asarray(res)
    d_c = complex(d)
    n = len(poles)
    g_dz = float(gamma * dz)
    st = tuple(f"x{j}_{i}" for j in range(n_seg) for i in range(n)) + ("i_out",)

    @component(ports=("p1", "p2"), states=st)
    def FiberNL(signals: Signals, s: States) -> tuple[dict, dict]:
        u = signals.p1
        f = {"p1": u}                                  # matched input tap
        q: dict = {}
        for j in range(n_seg):
            x = jnp.stack([getattr(s, f"x{j}_{i}") for i in range(n)])
            v = jnp.sum(c_c * x) + d_c * u             # dispersion over dz
            for i in range(n):
                f[f"x{j}_{i}"] = -(p_c[i] * x[i] + u)
                q[f"x{j}_{i}"] = x[i]
            u = v * jnp.exp(-1j * g_dz * (v * jnp.conj(v)).real)   # Kerr over dz
        f["p2"] = s.i_out
        f["i_out"] = signals.p2 - u
        return f, q

    return FiberNL


# ---------------------------------------------------------------------------
# transient-noise variants: same physics + a baked per-instance noise bank
# (unit-variance Gaussian rows, one per seed; runtime param seed_idx picks
# the row so N seeds re-run without recompiling). Scale for a one-sided PSD
# S is sqrt(S/(2*dt_n)) — see wavesrc.noise_bank.
# ---------------------------------------------------------------------------

def _noise_reader(bank, dt_n):
    import jax.numpy as jnp

    # sqrt(3/2) restores the variance lost to linear interpolation between
    # samples (mean of f^2 + (1-f)^2 over a segment is 2/3)
    bank_c = jnp.asarray(bank) * 1.22474487
    tn = jnp.arange(bank.shape[1]) * dt_n

    def nval(t, seed_idx):
        row = jnp.take(bank_c,
                       jnp.clip(jnp.asarray(seed_idx).astype(jnp.int32),
                                0, bank_c.shape[0] - 1), axis=0)
        return jnp.interp(t, tn, row)

    return nval


def _cw_laser_noisy(bank, dt_n):
    """CW laser with RIN (amplitude) and/or linewidth (phase) noise.

    The bank carries TWO independent rows per seed (even = RIN amplitude,
    odd = phase-noise increments) so intensity and phase fluctuations are
    uncorrelated — see the cwn branch of simulate._make_noisy.

    RIN drives a multiplicative amplitude fluctuation exactly as before.
    Linewidth drives a Wiener phase process: phi(t) is the running integral
    of white increments with per-step variance 2*pi*dnu*dt_n, so
    Var(phi(t)) = 2*pi*dnu*t. That gives a field-coherence decay
    exp(-pi*dnu*|tau|), i.e. a Lorentzian lineshape of FWHM = dnu.
    """
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    # two rows per seed (RIN even, phase odd) — _make_noisy allocates seeds*2
    assert bank.shape[0] % 2 == 0, "cwn noise bank must have an even row count"
    n_seeds = max(bank.shape[0] // 2, 1)
    tn = jnp.arange(bank.shape[1]) * dt_n
    # RIN: white samples, sqrt(3/2) restores the variance lost to linear
    # interpolation between bank samples (see _noise_reader).
    rin_c = jnp.asarray(bank) * 1.22474487
    # Phase walk: cumulative sum of the *raw* unit-variance increments — the
    # sampled walk points are exact, so no interp-variance restoration is
    # applied. Anchor each walk at phi(0) = 0 so the DC solve is phase-clean.
    walk = jnp.cumsum(jnp.asarray(bank), axis=1)
    walk = walk - walk[:, :1]
    c0 = 299792458.0

    @source(ports=("p1", "p2"), states=("i_src",))
    def CWLaserNoisy(
        signals: Signals,
        s: States,
        t: float,
        wavelength_nm: float = 1310.0,
        power: float = 1e-3,
        phase: float = 0.0,
        rin_db: float = 0.0,
        linewidth_hz: float = 0.0,
        ref_wavelength_nm: float = 0.0,
        seed_idx: float = 0.0,
    ) -> tuple[dict, dict]:
        k = jnp.clip(jnp.asarray(seed_idx).astype(jnp.int32), 0, n_seeds - 1)
        # RIN amplitude noise on the even row
        sigma = jnp.where(rin_db < 0.0,
                          jnp.sqrt(10.0 ** (rin_db / 10.0) / (2.0 * dt_n)),
                          0.0)
        n_rin = jnp.interp(t, tn, jnp.take(rin_c, 2 * k, axis=0))
        rel = jnp.maximum(1.0 + sigma * n_rin, 1e-6)
        # Phase noise (Wiener walk) on the odd row; step std sqrt(2*pi*dnu*dt_n)
        phi_scale = jnp.sqrt(2.0 * jnp.pi * jnp.maximum(linewidth_hz, 0.0)
                             * dt_n)
        phi_n = phi_scale * jnp.interp(t, tn, jnp.take(walk, 2 * k + 1, axis=0))
        # WDM carrier offset in the shared baseband frame (see cx.cw_laser);
        # select the never-zero reference before dividing
        ref_safe = jnp.where(ref_wavelength_nm > 0.0,
                             ref_wavelength_nm, wavelength_nm)
        w_off = 2.0 * jnp.pi * c0 * (1.0 / (ref_safe * 1e-9)
                                     - 1.0 / (wavelength_nm * 1e-9))
        field = jnp.sqrt(power * rel) * jnp.exp(
            1j * (w_off * t + phase + phi_n))
        constraint = (signals.p1 - signals.p2) - field
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}

    return CWLaserNoisy


def _photodiode_noisy(bank, dt_n):
    import jax
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    nval = _noise_reader(bank, dt_n)
    q_e = 1.602176634e-19

    @source(ports=("po_p", "po_n", "an", "cat"), states=("x1", "x2"))
    def PhotodiodeShot(
        signals: Signals,
        s: States,
        t: float,
        R: float = 0.8,
        Idk: float = 1e-9,
        Cj: float = 100e-15,
        f3db: float = 0.0,
        isat: float = 0.0,
        Yopt: float = 1.0,
        seed_idx: float = 0.0,
    ) -> tuple[dict, dict]:
        e = signals.po_p - signals.po_n
        i_opt = Yopt * e
        power = jnp.abs(e) ** 2
        iph = R * power + Idk
        tau = jnp.where(f3db > 0.0,
                        0.6436 / (2.0 * jnp.pi * jnp.maximum(f3db, 1.0)), 0.0)
        f_x1 = s.x1 - iph
        f_x2 = s.x2 - s.x1
        i_bw = s.x2.real
        # shot noise on the instantaneous current: S_i = 2 q I  ->  scale
        # sqrt(2 q I/(2 dt_n)) = sqrt(q I/dt_n). The amplitude is held out
        # of the Jacobian (stop_gradient): d(sqrt I)/dI diverges at I -> 0
        # and knocks Newton over, while physically the noise amplitude is
        # quasi-static within a step.
        i_qs = jax.lax.stop_gradient(jnp.maximum(i_bw, 0.0))
        i_shot = jnp.sqrt(q_e * i_qs / dt_n) * nval(t, seed_idx)
        i_tot = i_bw + i_shot
        i_out = jnp.where(
            isat > 0.0,
            jnp.maximum(isat, 1e-30) * jnp.tanh(
                i_tot / jnp.maximum(isat, 1e-30)),
            i_tot)
        f = {"po_p": i_opt, "po_n": -i_opt, "cat": i_out, "an": -i_out,
             "x1": f_x1, "x2": f_x2}
        vj = signals.cat - signals.an
        q = {"cat": Cj * vj, "an": -Cj * vj,
             "x1": tau * s.x1, "x2": tau * s.x2}
        return f, q

    return PhotodiodeShot


def _apd_noisy(bank, dt_n):
    import jax
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    nval = _noise_reader(bank, dt_n)
    q_e = 1.602176634e-19

    @source(ports=("po_p", "po_n", "an", "cat"), states=("x1", "x2"))
    def APDShot(
        signals: Signals,
        s: States,
        t: float,
        R: float = 0.8,
        M: float = 10.0,
        k_ion: float = 0.3,
        Idk_bulk: float = 1e-9,
        Idk_surf: float = 1e-9,
        Cj: float = 100e-15,
        f3db: float = 0.0,
        gbp: float = 0.0,
        isat: float = 0.0,
        Yopt: float = 1.0,
        seed_idx: float = 0.0,
    ) -> tuple[dict, dict]:
        e = signals.po_p - signals.po_n
        i_opt = Yopt * e
        power = jnp.abs(e) ** 2
        Mc = jnp.maximum(M, 1.0)
        i_prim = R * power + Idk_bulk
        i_mult = Mc * i_prim + Idk_surf
        tau = _apd_tau(f3db, gbp, Mc)
        f_x1 = s.x1 - i_mult
        f_x2 = s.x2 - s.x1
        i_bw = s.x2.real
        # McIntyre excess noise: multiplication amplifies the shot noise
        # super-linearly. The multiplied primary shot-noise PSD is
        # S_i = 2 q I_prim M^2 F(M), F(M) = k M + (2 - 1/M)(1 - k); the surface
        # dark current is not multiplied and adds plain 2 q Idk_surf. Scale to
        # sqrt(S_i/(2 dt_n)) per unit-variance sample -> sqrt(q S/(2q) / dt_n).
        # The amplitude is held out of the Jacobian (stop_gradient) exactly as
        # the PIN photodiode: d(sqrt I)/dI diverges at I -> 0 and stalls Newton.
        F = k_ion * Mc + (2.0 - 1.0 / Mc) * (1.0 - k_ion)
        i_prim_qs = jax.lax.stop_gradient(jnp.maximum(i_prim, 0.0))
        s_over_2q = i_prim_qs * Mc * Mc * F + jnp.maximum(Idk_surf, 0.0)
        i_shot = jnp.sqrt(q_e * s_over_2q / dt_n) * nval(t, seed_idx)
        i_tot = i_bw + i_shot
        i_out = jnp.where(
            isat > 0.0,
            jnp.maximum(isat, 1e-30) * jnp.tanh(
                i_tot / jnp.maximum(isat, 1e-30)),
            i_tot)
        f = {"po_p": i_opt, "po_n": -i_opt, "cat": i_out, "an": -i_out,
             "x1": f_x1, "x2": f_x2}
        vj = signals.cat - signals.an
        q = {"cat": Cj * vj, "an": -Cj * vj,
             "x1": tau * s.x1, "x2": tau * s.x2}
        return f, q

    return APDShot


def _tia_noisy(bank, dt_n):
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    nval = _noise_reader(bank, dt_n)

    @source(ports=("inp", "out"), states=("i_vg", "x1", "x2", "i_out"))
    def TIAMacroNoisy(
        signals: Signals,
        s: States,
        t: float,
        gain_ohm: float = 10e3,
        f3db: float = 20e9,
        vmax: float = 0.0,
        in_noise: float = 0.0,
        seed_idx: float = 0.0,
    ) -> tuple[dict, dict]:
        tau = 0.6436 / (2.0 * jnp.pi * jnp.maximum(f3db, 1.0))
        i_n = in_noise * jnp.sqrt(1.0 / (2.0 * dt_n)) * nval(t, seed_idx)
        v_tgt = -gain_ohm * (s.i_vg + i_n)
        f_x1 = s.x1 - v_tgt
        f_x2 = s.x2 - s.x1
        v_lp = s.x2.real
        v_out = jnp.where(
            vmax > 0.0,
            jnp.maximum(vmax, 1e-30) * jnp.tanh(
                v_lp / jnp.maximum(vmax, 1e-30)),
            v_lp)
        return {
            "inp": s.i_vg,
            "i_vg": signals.inp,
            "x1": f_x1, "x2": f_x2,
            "out": s.i_out,
            "i_out": signals.out - v_out,
        }, {"x1": tau * s.x1, "x2": tau * s.x2}

    return TIAMacroNoisy


def _ase_src_plain():
    """ase_src without transient noise: a transparent pass-through (the doc
    tells users the injector only fires when noise seeds >= 1)."""
    from circulax.components.base_component import Signals, States, source

    @source(ports=("pin", "pout"), states=("i_out",))
    def ASESrcOff(
        signals: Signals,
        s: States,
        t: float,
        s_ase_dbm_hz: float = -144.0,
    ) -> tuple[dict, dict]:
        return {"pin": 0.0, "pout": s.i_out,
                "i_out": signals.pout - signals.pin}, {}

    return ASESrcOff


def _ase_src_noisy(bank, dt_n):
    """Broadband ASE injector: E_out = E_in + n(t), n complex white Gaussian
    noise of one-sided power density s_ase_dbm_hz split between quadratures
    (sigma_q = sqrt(S/(4*dt_n)) per sample). The bank carries TWO rows per
    seed (real/imag streams) — see the ase branch of _make_noisy."""
    import jax.numpy as jnp
    from circulax.components.base_component import Signals, States, source

    bank_c = jnp.asarray(bank) * 1.22474487   # interp variance restoration
    tn = jnp.arange(bank.shape[1]) * dt_n
    n_seeds = max(bank.shape[0] // 2, 1)

    @source(ports=("pin", "pout"), states=("i_out",))
    def ASESrc(
        signals: Signals,
        s: States,
        t: float,
        s_ase_dbm_hz: float = -144.0,
        seed_idx: float = 0.0,
    ) -> tuple[dict, dict]:
        sigma = jnp.sqrt(10.0 ** (s_ase_dbm_hz / 10.0) * 1e-3 / (4.0 * dt_n))
        k = jnp.clip(jnp.asarray(seed_idx).astype(jnp.int32), 0, n_seeds - 1)
        nr = jnp.interp(t, tn, jnp.take(bank_c, 2 * k, axis=0))
        ni = jnp.interp(t, tn, jnp.take(bank_c, 2 * k + 1, axis=0))
        nval = sigma * (nr + 1j * ni)
        return {"pin": 0.0, "pout": s.i_out,
                "i_out": signals.pout - (signals.pin + nval)}, {}

    return ASESrc


_NOISY_BUILDERS = {"cwn": _cw_laser_noisy, "pdn": _photodiode_noisy,
                   "apdn": _apd_noisy,
                   "ase": _ase_src_noisy,
                   "tian": _tia_noisy}


# ---------------------------------------------------------------------------
# user-uploaded Verilog-A models (webapp/models_user/*.va)
# ---------------------------------------------------------------------------

USER_VA_DIR = Path(__file__).resolve().parent / "models_user"


def _veriloga_file(entry: dict) -> Path | None:
    """Absolute path to the Verilog-A source backing a catalog entry, else None.

    User uploads carry a ``user_va`` stem under ``models_user/``; built-in VA
    cards begin their ``doc`` with the repo-relative ``models/…/<name>.va``
    path (see the catalog above). Non-VA components return None.
    """
    stem = entry.get("user_va")
    if stem:
        return USER_VA_DIR / f"{stem}.va"
    m = re.match(r"\s*(models/\S+\.va)\b", entry.get("doc", ""))
    if m:
        return REPO_ROOT / m.group(1)
    return None


def _resolved_va_file(entry: dict) -> Path | None:
    """Resolved ``.va`` path for an entry, confined to the ``models/`` and
    ``models_user/`` trees (so a ``..``/symlink escape returns None), or None
    when the entry is not VA-backed or the file is missing."""
    f = _veriloga_file(entry)
    if f is None:
        return None
    f = f.resolve()
    roots = ((REPO_ROOT / "models").resolve(), USER_VA_DIR.resolve())
    return f if any(r in f.parents for r in roots) and f.is_file() else None


def _va_relpath(f: Path) -> str:
    return str(f.relative_to(REPO_ROOT)) if REPO_ROOT in f.parents else f.name


def _annotate_veriloga(entry: dict) -> None:
    """Tag one entry with a ``veriloga`` display path (repo-relative) when it is
    backed by a readable, in-tree ``.va`` file, else None — the UI keys its
    source-viewer button off this field, so this must agree with what
    ``veriloga_source`` will actually serve."""
    f = _resolved_va_file(entry)
    entry["veriloga"] = _va_relpath(f) if f else None


def veriloga_source(key: str) -> tuple[str, str] | None:
    """``(display_path, source_text)`` for a VA-backed catalog key, else None.

    Reads are confined to the ``models/`` and ``models_user/`` trees so a
    crafted ``type`` can never escape into arbitrary files.
    """
    entry = CATALOG.get(key)
    if entry is None:
        return None
    f = _resolved_va_file(entry)
    return (_va_relpath(f), f.read_text()) if f else None


def _doc_from_va(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("//"):
            lines.append(s.lstrip("/ ").strip())
        elif s and not s.startswith("`"):
            break
    return " ".join(lines)[:400] or "User-uploaded Verilog-A model."


def register_user_va(stem: str) -> dict:
    """Compile models_user/<stem>.va and register a catalog entry.

    Ports come from the lowered component; parameters (with defaults) from
    the VA text. All ports are electrical — coherent-field models need
    Ereal/Eimag node pairs bridged manually; power-domain optical ports can
    be bridged with the opt_f2p / opt_p2f adapter components.
    """
    from photonflux import cx

    path = USER_VA_DIR / f"{stem}.va"
    comp = cx.va(path)
    text = path.read_text()
    defaults = cx._va_literal_defaults(text)
    # localparams are not user-facing: keep only `parameter` declarations
    param_names = __import__("re").findall(
        r"^\s*parameter\s+real\s+(\w+)", text, __import__("re").M)
    key = f"uva_{stem}"
    CATALOG[key] = {
        "label": f"{stem} (.va)",
        "category": "User VA",
        "doc": _doc_from_va(text) + " [uploaded Verilog-A — all ports "
               "electrical; bridge power-domain optical ports with the "
               "Field->Power / Power->Field adapters]",
        "ports": [{"name": p, "domain": "electrical"} for p in comp.ports],
        "params": [_p(n, defaults.get(n, 0.0), "", n)
                   for n in param_names],
        "user_va": stem,
    }
    _annotate_veriloga(CATALOG[key])
    return CATALOG[key]


def load_user_va() -> list[str]:
    """Scan models_user/ at startup; returns the stems that registered."""
    if not USER_VA_DIR.is_dir():
        return []
    out = []
    for p in sorted(USER_VA_DIR.glob("*.va")):
        try:
            register_user_va(p.stem)
            out.append(p.stem)
        except Exception as e:  # a broken upload must not kill the server
            print(f"user VA {p.name}: failed to load — {e}")
    return out


# tag every static card up front so /api/components tells the UI which
# components can show their Verilog-A source (uploads are tagged on register).
for _entry in CATALOG.values():
    _annotate_veriloga(_entry)
del _entry


def build_models(sky130_geoms: dict[str, tuple[str, float, float]] | None = None,
                 waveforms: dict[str, tuple] | None = None,
                 noisy: dict[str, tuple] | None = None,
                 ltis: dict[str, dict] | None = None) -> dict:
    """models_map for compile_circuit.

    ``sky130_geoms`` maps a model key (e.g. ``"sky130_nfet:1x0.15"``) to
    (device, w_um, l_um) — each distinct FET flavor+geometry is its own OSDI
    descriptor. ``waveforms`` maps a model key (``"prbs:<hash>"``) to its
    (t, v) breakpoint arrays for the pattern/PWL sources.
    """
    from circulax.components.electronic import (
        NMOS,
        PMOS,
        Capacitor,
        CurrentSource,
        Diode,
        IdealOpAmp,
        Inductor,
        PulseVoltageSource,
        Resistor,
        VoltageSource,
        VoltageSourceAC,
    )
    from circulax.components.photonic import (
        DirectionalCoupler,
        OpticalWaveguide,
        Splitter,
    )

    from photonflux import cx

    models: dict[str, Any] = {
        "ground": lambda: 0,
        "cw_laser": cx.cw_laser(),
        "mzm": cx.mzm(),
        "pulse_mod": _pulse_mod(),
        "waveguide": OpticalWaveguide,
        "splitter": Splitter,
        "dir_coupler": DirectionalCoupler,
        "grating": _grating_r(),
        "opt_mirror": _opt_mirror(),
        "opt_term": _opt_term(),
        "photodiode": _photodiode(),
        "apd": _apd(),
        "iq_modulator": _iq_modulator(),
        "coherent_rx": _coherent_rx(),
        "_f2ri": cx.field_to_ri(),
        "_f2ri_m": _field_to_ri_matched(),
        "_ri2f": cx.ri_to_field(),
        "_ring_va": cx.va("ring_mod"),
        "_phaseshift_va": cx.va("phase_shifter"),
        "_p2f": _power_to_field(),
        "_f2p": _field_to_power(),
        "_dml_va": cx.va("laser_dml"),
        "_rate_va": cx.va("laser_rate"),
        "_tw_va": cx.va("mzm_tw"),
        "_ring_inj_va": cx.va("ring_mod_inj"),
        "_seg_va": cx.va("mzm_seg"),
        "_soa_va": cx.va("soa"),
        "_edfa_va": cx.va("edfa"),
        "_raman_va": cx.va("raman_amp"),
        "_sbs_va": cx.va("sbs_fiber"),
        "_mirror_va": cx.va("mirror"),
        "_circ_va": cx.va("circulator"),
        "_ringcomb_va": cx.va("ring_filter"),
        "_ringnl_va": cx.va("ring_nl"),
        "_ringkerr_va": cx.va("ring_kerr"),
        "_ringselfheat_va": cx.va("ring_selfheat"),
        "_wgnl_va": cx.va("waveguide_nl"),
        "_twseg_va": cx.va("tw_seg"),
        "_twgain_va": cx.va("tw_gain_seg"),
        "_phasepad_va": cx.va("phase_pad"),
        "vdc": VoltageSource,
        "vpulse": PulseVoltageSource,
        "vsin": VoltageSourceAC,
        "idc": CurrentSource,
        "resistor": Resistor,
        "capacitor": Capacitor,
        "inductor": Inductor,
        "diode": Diode,
        "nmos": NMOS,
        "pmos": PMOS,
        "opamp": IdealOpAmp,
        "tia": _tia(),
        "ctle": _ctle(),
        "rx_ffe": _rx_eq_buffer(),
        "rx_dfe": _rx_eq_buffer(),
        "opt_f2p": _field_to_power(),
        "opt_p2f": _power_to_field(),
        "ase_src": _ase_src_plain(),
    }
    for key, entry in CATALOG.items():
        if entry.get("user_va"):
            models[key] = cx.va(USER_VA_DIR / f"{entry['user_va']}.va")
    for key, (device, w_um, l_um) in (sky130_geoms or {}).items():
        models[key] = cx.sky130_fet(device, w=w_um, l=l_um)
    for key, (wt, wv) in (waveforms or {}).items():
        models[key] = _waveform_source(wt, wv)
    for key, (kind, bank, dt_n) in (noisy or {}).items():
        models[key] = _NOISY_BUILDERS[kind](bank, dt_n)
    for key, payload in (ltis or {}).items():
        if "real" in payload:
            A, B, C, D, zin = payload["real"]
            models[key] = _lti_vt(A, B, C, D, zin)
        elif "cplx_drop" in payload:
            poles, res_d, res_t = payload["cplx_drop"]
            models[key] = _lti_field_drop(poles, res_d, res_t)
        elif "fiber_nl" in payload:
            poles, res, d, gamma, dz, n_seg = payload["fiber_nl"]
            models[key] = _fiber_nl_field(poles, res, d, gamma, dz, n_seg)
        else:
            poles, res, d = payload["cplx"]
            models[key] = _lti_field(poles, res, d)
    return models
