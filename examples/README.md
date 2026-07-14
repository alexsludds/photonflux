# Photonic-link examples

Verilog-A photonics and SKY130 electronics on
[circulax](https://github.com/gdsfactory/circulax) — a differentiable circuit
simulator on JAX/Diffrax/Optimistix — instead of ngspice + OSDI Verilog-A.

**Source convention:** the only lasers in this environment are **CW lasers** —
`cx.cw_laser()`, parameterised by wavelength and power, emitting a constant
field `E = √P`. All modulation happens in modulator elements (`cx.mzm()`, the
field-convention twin of `models/mzm.va`, or the ideal `PulseModulator` in
example 01); a laser is never modulated directly.

| script | what it shows |
|---|---|
| [`photodiode_tia.py`](photodiode_tia.py) | CW laser → pulse modulator → PD → ideal op-amp TIA; `jax.grad` of V_out for inverse design |
| [`link_cmos.py`](link_cmos.py) | full CW-laser + MZM link with a **transistor** electrode driver and a **CMOS-inverter** TIA (square-law stand-ins, no PDK needed) |
| [`link_sky130.py`](link_sky130.py) | the same link with **real SKY130 PDK BSIM4 transistors** (`cx.sky130_fet`, OSDI evaluated natively in circulax) |
| [`ring_mod_sky130.py`](ring_mod_sky130.py) | self-checking testbench: **Verilog-A microring modulator** (`models/ring_mod.va` via `cx.va`) driven by a SKY130 inverter; DC tuning curve pinned to the analytic CMT Lorentzian, physical device parameters (R = 7.5 µm, 7000 dB/m junction loss, κ² = 10 % overridable with `--kappa2`, 45 pm/V, 0.5 fF/µm), PRBS NRZ at a configurable baud rate (`--baud`, open eyes 5–50 Gbd), optical **eye diagram** of the through port |
| [`soa_fp_laser.py`](soa_fp_laser.py) | **Fabry-Perot laser from parts**: the bidirectional `models/soa.va` gain reservoir between two `models/mirror.va` partial reflectors. Lasing emerges from the closed loop: L-I staircase transient (threshold at the analytic `I_th`, clamped-gain slope to <2 %), turn-on with relaxation-oscillation overshoot, and the round-trip gain visibly clamping to `R1·R2·G² = 1` |
| [`soa_vernier_laser.py`](soa_vernier_laser.py) | **Vernier laser from ASE noise, with a live mode hop**: SOA behind two `models/ring_filter.va` five-mode combs (FSR 29.8 vs 28.1 GHz) — the SOA's deterministic seed is OFF and broadband complex white noise at the physical ASE level `n_sp·hν·(G−1)` seeds every comb line. One continuous 12 ns transient in a fixed 1310 nm frame: three alignment candidates rise out of the noise floor and compete (each grows at ln L/T_rt with its *own* ring group delay), the aligned pair clamps the shared gain (SMSR 65 dB), then 0.5 mW steps onto the ring-B heater *while lasing* — the running mode falls below threshold, the output collapses, and the newly aligned pair one FSR (29.8 GHz, ×17 lever) to the red rebuilds from the ASE floor (SMSR 71 dB). Spectrogram of the whole story; lines, powers and the hop pinned against the comb-product/reservoir analytics |
| [`ring_tpa_q.py`](ring_tpa_q.py) | **nonlinear absorption capping a high-Q ring**: `models/ring_nl.va` spectra vs input power pinned against an independent root-solve; loaded Q vs power for 30/100/300 dB/m devices collapsing onto the TPA+FCA ceiling (10× apart at 1 µW, 2.5× at 10 mW); FCD/Kerr back on for the asymmetric blue-pulled line via a warm-started continuation sweep |
| [`wg_fwm.py`](wg_fwm.py) | **chi(3) four-wave mixing, emergent and pinned to every textbook scaling**: pump (+20 GHz) and signal (+30 GHz) tones through `models/waveguide_nl.va` with only the Kerr term on. FWM is coded nowhere — the instantaneous phase `e^{−jk\|E(t)\|²}` mixes the beating envelope and the idler appears at exactly 2f_p−f_s. Checks: the simulated spectrum equals the exact map to 1e-9 over 190 dB; idler slope 2.0000 in pump / 1.0000 in signal / 2.0000 in length; absolute η = T(kP_p)²; XPM/SPM phase ratio 2.000; the idler asymmetry P_p/P_s; the strong-drive Bessel comb `T(P_pJ_m² + P_sJ_{m−1}²)` line-by-line at x ≈ 2 (pump depleted to J_0² = 4 %); N cascaded segments converging on the distributed-NLSE γL_eff at 1/N²; energy conserved to 1e-15 |
| [`ring_fwm.py`](ring_fwm.py) | **four-wave mixing INSIDE a ring resonator, driven at its FSR**: `models/ring_kerr.va` puts the chi(3) into a five-mode add-drop ring (the modal Lugiato–Lefever equations, momentum-matched so SPM/XPM×2/FWM all come from one triple sum). Pump on mode 0 + signal on mode +1 (one FSR blue) → the idler grows in mode −1, exactly one FSR red, all three waves resonantly enhanced: ~×1000 the conversion of the same 12.6 mm of straight waveguide. Checks: every bus line vs an independent scipy integration of the same 5 ODEs; absolute η = (g_eff τ P_p)²(κ²τ)⁶; slopes 2/1; the conversion collapsing as the Lorentzian **to the 4th power** when the comb slides under the lasers; comb dispersion d2 detuning the idler mode = ring-style phase-matching loss; the comb spreading to modes ±2 at 0.5 mW (2f_s−f_p + a true second-order cascade line) |

```bash
.venv-circulax/bin/python examples/photodiode_tia.py   # -> out/circulax_photodiode_tia.png
.venv-circulax/bin/python examples/link_cmos.py         # -> out/circulax_link_cmos.png
.venv-circulax/bin/python examples/link_sky130.py       # -> out/circulax_link_sky130.png
.venv-circulax/bin/python examples/ring_mod_sky130.py   # -> out/circulax_ring_mod.png
.venv-circulax/bin/python examples/soa_fp_laser.py      # -> out/soa_fp_laser.png
.venv-circulax/bin/python examples/soa_vernier_laser.py # -> out/soa_vernier_laser.png
.venv-circulax/bin/python examples/ring_tpa_q.py        # -> out/ring_tpa_q.png
.venv-circulax/bin/python examples/wg_fwm.py             # -> out/wg_fwm.png
.venv-circulax/bin/python examples/ring_fwm.py           # -> out/ring_fwm.png
```

## The SOA / cavity examples

`soa_fp_laser.py`, `soa_vernier_laser.py` and `ring_tpa_q.py` share
[`cavity.py`](cavity.py) (staircase bias source, infinite-impedance
terminator for driven-but-unused optical outputs, fixed-step BDF2 runner).
Two conventions worth knowing before building your own cavities:

* **Lasers are solved by transient settling, not DC.** Above threshold the
  dark state is still a mathematically valid stationary point and Newton
  converges to it; the lasing operating point is reached by stepping the
  bias in a transient (which is also just… how lasers turn on). The L-I
  staircase pattern in `soa_fp_laser.py` collects a whole L-I curve in one
  transient.
* **The lasing wavelength is emergent.** In the baseband-envelope frame the
  laser line appears as an envelope tone at its offset from the frame
  reference; `soa_vernier_laser.py` re-references the frame to the comb
  alignment predicted analytically, then *measures* the true line as the
  residual rotation `d(arg E)/dt` of the settled field.

## `link_sky130.py` — the real PDK inside circulax

`photonflux.cx` (see the module docstring for details) provides:

* `cx.cw_laser()` / `cx.mzm()` — the CW source and Mach-Zehnder modulator
  (intensity transfer identical to `models/mzm.va`, electrodes present a real
  capacitive load to the driver).
* `cx.sky130_fet("nfet_01v8", w=1.0, l=0.15)` — ngspice resolves the volare
  model card (corner stitching, `{...}` card expressions, W/L bin selection,
  read back with `showmod`), and the BSIM4.8 Verilog-A is compiled to OSDI by
  the ChipFlow openvaf fork (`bin/openvaf-ir`) and evaluated natively inside
  circulax's Newton/transient loop. Off-state exact to gmin; on-state matches
  an ngspice run of the same card with `version=4.8` to ~1e-4, and true
  sky130 (BSIM4v5) within the 3–10 % version-skew band — pinned by
  `tests/test_cx.py`.
* `cx.va("photodiode")` — any `models/*.va` compiled by bosdi into a pure-JAX
  differentiable component (exact: DC and `jax.grad` validated to 1e-9).

Solver caveat (documented in the script): with OSDI BSIM4 devices in the
complex-valued system, every *individual* BDF2 step converges (verified by
fixed-dt marching), but circulax 0.2.1's adaptive-controller retry path
reports a spurious nonlinear divergence — the script uses fixed 20 ps steps
with `BDF2VectorizedTransientSolver`, which is robust and cheap (~400 steps
for the 8 ns pattern).

Known limitation: `cx.sky130_fet(..., backend="jax")` (fully differentiable
pure-JAX BSIM4) exists but is experimental — bosdi 0.1.5's optimized-MIR
ingestion miscompiles BSIM4's nested conditionals (nfet subthreshold broken,
pfet unusable). The fixed ingestion path (`--dump-unopt-mir-with-split`) is
not yet in any published openvaf build.

## photonflux vs circulax, same circuit

|                | photonflux/ngspice (reference flow)     | circulax (`photodiode_tia.py`)                    |
|----------------|-----------------------------------------|---------------------------------------------------|
| solver         | ngspice matrix, native libngspice       | JAX: Newton DC + Diffrax adaptive transient       |
| photonics      | compiled Verilog-A → OSDI               | Python `@component` functions                     |
| optical node   | **power** convention (V = power in W)   | **coherent field** (V = complex amplitude E)      |
| photocurrent   | `Iph = R*V(popt)` in `photodiode.va`    | `Iph = R*|E|^2` bridge component                  |
| extra          | PRBS/BER/eye analysis, PDK transistors  | end-to-end `jax.grad` for inverse design          |

Because circulax carries the optical **field**, the photodiode is a genuine
mixed-domain bridge: it reads the complex field on its optical ports and
injects a *real* photocurrent `R·|E|²` into its electrical ports. `|E|²` is
non-holomorphic, but circulax assembles the complex system as a real 2N block
(separate `∂f_re/∂v_re`, `∂f_re/∂v_im`, … partials), so the detector
differentiates correctly inside the solve. The MZM is the transpose case —
complex optical currents depending on the (real) electrode voltage — and rides
the same machinery.

## What `photodiode_tia.py` shows

1. **Transient** — `CW laser → pulse modulator → 3 dB fibre → photodiode →
   inverting TIA`. The fibre halves the optical power (3 → 1.5 mW low,
   18 → 9 mW high); the photodiode turns 9 mW into 7.2 mA (R = 0.8 A/W); the
   TIA closes the loop to `V_out = Iph·Rf = 1.44 V` while holding the summing
   node at a ~1.4 µV virtual ground. Output figure mirrors example 01's
   waveform stack.

2. **Gradients** — `jax.grad` of the steady-state `V_out` through the DC solve
   and the `|E|²` detector returns the exact analytic sensitivities
   `∂V_out/∂Rf = R·P` and `∂V_out/∂P = R·Rf`. This is the capability the
   ngspice path doesn't have: the same netlist is differentiable for
   gradient-based inverse design.

## `link_cmos.py` — electrode driver + CMOS-inverter TIA

The PDK-free warm-up for `link_sky130.py`, using circulax's own square-law
MOSFETs:

* **driver** — an NMOS common-source stage (gate driven by a SPICE-style
  `PULSE`) swings the MZM electrodes off a 3.3 V rail through a pull-up,
* **link** — CW laser → MZM (driver ON → electrode low → transmission max, so
  light follows V_in) → 3 dB fibre,
* **receiver** — a self-biased CMOS inverter (PMOS + NMOS, 1.8 V) with feedback
  `Rf` is the transimpedance amplifier around the photodiode summing node.

Everything — driver FET, MZM transfer, fibre, detector, and the two-FET
inverter with feedback — is one nonlinear complex system solved as a single
Newton DC + BDF2 transient, end-to-end differentiable.

### Solver note (in the script)

**Parasitic node caps + `dtmax`.** circulax's implicit BDF2 crawls if it has
to chase near-instant algebraic jumps across the square-law FET kinks. Small
drain/gate/load capacitances (every real node has them) give each node a
finite RC, and a `dtmax = 20 ps` on the step controller keeps the adaptive
stepper from striding over the 200 ps gate edges.

## Components

- `cx.cw_laser()` — **the** laser: CW, `{wavelength_nm, power, phase}`,
  emits `E = √P·e^{jφ}` (in `photonflux/cx.py`).
- `cx.mzm()` — Mach-Zehnder modulator, field-convention twin of
  `models/mzm.va` with capacitive drive electrodes (in `photonflux/cx.py`).
- `PulseModulator` (`@source`, `photodiode_tia.py`) — ideal intensity
  modulator carving an NRZ-like pulse out of a CW field.
- `Photodiode` (`@component`, `photodiode_tia.py`) — optical matched absorber
  + `Iph = R·|E|² + Idk` photocurrent + junction capacitance `Cj`. The bridge
  between domains.
- `OpAmp` (`@component`, `photodiode_tia.py`) — ideal VCVS, the TIA gain core.
- `OpticalWaveguide`, `Resistor`, `NMOS`, `PMOS`, `Capacitor`,
  `VoltageSource`, `PulseVoltageSource` are circulax built-ins.

## Setup

circulax needs Python ≥ 3.12 and pulls in JAX/Diffrax/SAX. To keep it out of
the photonflux/ngspice environment it was installed into a local venv:

```bash
python3 -m venv .venv-circulax
.venv-circulax/bin/pip install "circulax[verilog-a]" openvaf-py
.venv-circulax/bin/pip install -e .                # photonflux (for cx)
.venv-circulax/bin/python examples/photodiode_tia.py
```
