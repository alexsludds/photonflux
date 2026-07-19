# Standalone examples

Each script builds a circuit on
[circulax](https://github.com/gdsfactory/circulax) (differentiable JAX/Diffrax)
and writes a figure to `out/`. **Each script's module docstring is the full
write-up** — the table below is just a one-line index.

Convention: the only lasers are **CW lasers** (`cx.cw_laser()`, `E = √P`); all
modulation happens in modulators (`cx.mzm()`, or the ideal `PulseModulator` in
`photodiode_tia.py`), never in the laser.

| script | what it shows | also in the web app |
|---|---|---|
| [`photodiode_tia.py`](photodiode_tia.py) | CW laser → modulator → fibre → PD → TIA, with `jax.grad` of V_out through the DC solve | example 01 |
| [`link_cmos.py`](link_cmos.py) | full CW + MZM link, transistor driver + CMOS-inverter TIA (no PDK needed) | example 02 |
| [`link_sky130.py`](link_sky130.py) | the same link with **real SKY130 BSIM4 FETs** (`cx.sky130_fet`) | — |
| [`ring_mod_sky130.py`](ring_mod_sky130.py) | Verilog-A microring modulator driven by a SKY130 inverter; DC tuning + optical eye (`--baud`, `--kappa2`) | example 03 |
| [`ring_eo_response.py`](ring_eo_response.py) | small-signal **electro-optic frequency response** of the same ring — swept-tone lock-in vs the analytic photon-lifetime rolloff, f_3dB ~ 44 GHz (`--kappa2`, `--rs`) | — |
| [`soa_fp_laser.py`](soa_fp_laser.py) | Fabry-Perot laser from an SOA between two mirrors — lasing emerges from the loop | "SOA Fabry-Perot laser" |
| [`soa_vernier_laser.py`](soa_vernier_laser.py) | Vernier laser seeded by ASE noise, with a live one-FSR mode hop | "Vernier laser mode hop" |
| [`edfa_wdm.py`](edfa_wdm.py) | **EDFA gain dynamics** (`edfa.va`): drop 7 of 8 WDM channels and the surviving channel surges as the shared ms-lifetime erbium reservoir refills — settled gains, surge, and the recovery time constant (∝ `tau_c`) pinned to the analytic reservoir; C-band gain-tilt spectrum + ASE/NF floor alongside | — |
| [`ring_tpa_q.py`](ring_tpa_q.py) | TPA + free-carrier absorption capping a high-Q ring | "TPA-limited high-Q ring" |
| [`wg_fwm.py`](wg_fwm.py) | χ(3) four-wave mixing in a waveguide, pinned to every textbook scaling | "χ(3) four-wave mixing" |
| [`ring_fwm.py`](ring_fwm.py) | four-wave mixing **inside** a Kerr ring resonator | "Kerr ring four-wave mixing" |
| [`ring_selfheat.py`](ring_selfheat.py) | **thermo-optic bistability** of a self-heating ring — the laser wavelength ramped up then down traces a hysteresis loop | — |
| [`mzm_tw_transient.py`](mzm_tw_transient.py) | **electro-optic bandwidth** of the traveling-wave MZM (`mzm_tw.va`): step-response rise time and a full-swing NRZ eye, velocity-matched vs walk-off electrode | `mzm_tw` |
| [`mzm_tw_eo_bw.py`](mzm_tw_eo_bw.py) | **electro-optic frequency response** of the traveling-wave MZM — a network-analyser-style tone sweep pins |H(f)|, phase, the -3 dB EO bandwidth, velocity-match / walk-off (`f_w ∝ 1/ℓ`) and pole-count roll-off against the model's pole cascade | — |
| [`pol_malus.py`](pol_malus.py) | **polarization optics**: a rotator + PBS + PBC chain splitting a TE launch by **Malus' law** (`cos²θ`/`sin²θ`), and a **birefringent-waveguide MZI** whose TE/TM fringes are offset by the modal birefringence `Δn_eff` — the two dual-polarization (Jones-vector) acceptance testbenches (see [`docs/polarization.md`](../docs/polarization.md)) | — |
| [`laser_linewidth.py`](laser_linewidth.py) | **laser phase noise / linewidth** (`cw_laser` `linewidth_hz`): a Wiener phase walk gives the CW laser a **Lorentzian line of FWHM = Δν**. The OSA lineshape, the field-coherence linewidth readout tracking Δν across 1–8 MHz (slope 1), and a **delayed self-heterodyne** beat of width 2·Δν — all read off one solved envelope | — |
| [`eo_comb.py`](eo_comb.py) | **electro-optic frequency comb** from a microring modulator built out of **sub-components** — a directional coupler, an EO phase shifter, and a cavity-mode loop (the temporal-CMT block diagram), asserted equal to the monolithic `ring_mod.va` to machine precision. A strong RF tone on the phase shifter sweeps the resonance across a slope-parked laser and the through port grows a comb spaced by `f_RF`, pinned line-by-line to an independent CMT integration; the comb is **cavity-shaped** and its bandwidth **saturates at the photon-lifetime limit** `f_cav ≈ 44 GHz` as `f_RF` rises (`--frf`, `--swing`, `--kappa2`) | example 42 |

The last seven have a browser twin (the named web-app examples 36–42): the
script is the pinned physics study, the web-app example is the same circuit to
click through. Example 41 drives the ring's wavelength node with a PWL source
so the transient sweeps forward then back — the through-port hysteresis loop is
the same one this script plots. Example 42 drives the ring modulator's electrode
with a strong `vsin` tone; its through-port spectrum probe shows the EO comb.

```bash
.venv-circulax/bin/python examples/photodiode_tia.py    # -> out/photodiode_tia.png
.venv-circulax/bin/python examples/soa_fp_laser.py      # -> out/soa_fp_laser.png
.venv-circulax/bin/python examples/ring_selfheat.py     # -> out/ring_selfheat.png
.venv-circulax/bin/python examples/mzm_tw_transient.py  # -> out/mzm_tw_transient.png
.venv-circulax/bin/python examples/mzm_tw_eo_bw.py      # -> out/mzm_tw_eo_bw.png
.venv-circulax/bin/python examples/eo_comb.py           # -> out/eo_comb.png
# ...each script prints its own output path
```

## Analysis notebooks

[`notebooks/`](notebooks/) holds Jupyter companions to the web-app
testbenches: each pulls a built-in example through the notebook bridge
(`photonflux.nb`), runs it on the dev server, and pins the result against
theory — SKY130 FET characterization ($g_m/I_D$, $f_T$), the EO comb vs
coupled-mode theory, four-wave-mixing scalings, the Vernier and Fabry-Perot
design spaces, and diff-pair / ring-oscillator pen-and-paper analysis. See
[`notebooks/README.md`](notebooks/README.md); `notebook_live_bench.ipynb`
in this directory is the interactive tour of the bridge itself.

## Shared helpers (not runnable examples)

- [`_cavity.py`](_cavity.py) — staircase bias source, an infinite-impedance
  terminator for driven-but-unused optical outputs, and the fixed-step BDF2
  transient runner. Used by the SOA-cavity examples.
- [`_drivers.py`](_drivers.py) — SKY130 CMOS-inverter driver fragments (plain,
  Miller-neutralized, two-stage) for the ring-modulator testbench.

## Two things worth knowing

- **Lasers are solved by transient settling, not DC.** Above threshold the dark
  state is still a valid stationary point that Newton converges to; the lasing
  operating point is reached by stepping the bias in a transient (which is also
  just how lasers turn on). `soa_fp_laser.py` collects a whole L-I curve in one
  transient staircase.
- **The lasing wavelength is emergent.** In the baseband-envelope frame the
  laser line is an envelope tone at its offset from the frame reference;
  `soa_vernier_laser.py` measures the true line as the residual rotation
  `d(arg E)/dt` of the settled field.

## Setup

```bash
python3 -m venv .venv-circulax
.venv-circulax/bin/pip install "circulax[verilog-a]" openvaf-py
.venv-circulax/bin/pip install -e .           # photonflux (for cx)
.venv-circulax/bin/python examples/photodiode_tia.py
```
