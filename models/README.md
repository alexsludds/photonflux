# Verilog-A model library

Every model is lowered to a differentiable JAX component by `cx.va("<name>")`
and solved by circulax. Call it by **bare name** — the subfolders below are for
humans; the loader finds a model wherever it lives:

```python
from photonflux import cx
RING = cx.va("ring_mod")          # -> models/optical_field/ring_mod.va
```

Compilation and content-hash caching (under `models/__jax__/`, regenerated on
first use) are automatic.

## `optical_power/` — power-domain optical (node voltage = optical power [W])

| model | what it is |
|---|---|
| `laser_dml.va` | DML, static L-I (`Ith`, `slope`, `Rs`, `Von`) + first-order optical response `tau` |
| `laser_rate.va` | DML from the full rate equations: turn-on delay, relaxation oscillations, pattern-dependent ringing |
| `photodiode.va` | PIN PD: responsivity `r`, dark current `idk`, junction cap `cj` |
| `mzm.va` | Mach-Zehnder modulator: `vpi`, `vbias`, insertion loss, finite ER, electrode cap |
| `mzm_seg.va` | segmented-electrode MZM ("optical DAC"): three binary-weighted phase segments |
| `mzm_tw.va` | traveling-wave MZM: differential RF electrode with distributed drive |

## `optical_field/` — coherent-field optical (paired Ereal/Eimag nodes)

| model | what it is |
|---|---|
| `laser_cw.va` | CW source emitting an E-field with thermally-driven phase |
| `ring_mod.va` | Si microring modulator: coupled-mode-theory all-pass ring from the physical device (radius, group index, junction dB/m, bus κ², pm/V, fF/µm) — coupling condition + photon-lifetime bandwidth. **Hierarchical**: instantiates `directional_coupler.va` + `ring_phase_shifter.va` + `ring_waveguide.va` on a shared cavity node (flattened source-level by `cx.va`, see below) |
| `directional_coupler.va` | bus↔ring point coupler (CMT sub-component): couples the bus field into the shared cavity node at rate 1/τ_e and taps the through port `s_out = s_in + j·A` |
| `ring_phase_shifter.va` | intracavity EO phase shifter (CMT sub-component): electrode voltage sets the cavity detuning δ(V) — the *cavity-detuning* element, distinct from the memoryless field rotator `phase_shifter.va` — plus the junction cap + leakage load the driver sees |
| `ring_waveguide.va` | ring waveguide loop (CMT sub-component): photon storage `dA/dt` + round-trip loss at rate 1/τ_i — the element that keeps the photon-lifetime dynamics |
| `ring_mod_inj.va` | carrier-injection twin of `ring_mod.va`: forward-biased PIN, lifetime-limited bandwidth, FCD blue shift + FCA loss |
| `waveguide_nl.va` | nonlinear waveguide segment: TPA (exact closed form), FCA (lumped population, lifetime `tau_fc`), Kerr SPM + free-carrier dispersion — the instantaneous phase makes multi-tone inputs **four-wave-mix** at the textbook efficiency (`examples/wg_fwm.py` pins every scaling). Trapezoidal z-lumping, 1/N² NLSE convergence. Unidirectional |
| `soa.va` | semiconductor optical amplifier: lumped Agrawal-Olsson gain reservoir, **bidirectional** (forward/backward share the reservoir), α_H chirp, deterministic ASE seed for lasing start-up |
| `edfa.va` | erbium-doped fibre amplifier: the pumped, ms-timescale, **wavelength-shaped** twin of the SOA. Pump-driven log-gain reservoir (transparent at `p_pump_tr`, peak `g0_db` at `p_pump_op`) with **homogeneous** saturation on the total input power — so dropping WDM channels surges the survivors over `tau_c` (~10 ms). A detuned one-pole filter gives a Lorentzian gain peak at `lambda_peak` (width `gain_bw_nm`), so each carrier on a shared DWDM bus is amplified differently. Unidirectional (runs behind isolators); steady-state pins in `tests/test_edfa.py`, the channel-drop transient in `examples/edfa_wdm.py` |
| `raman_amp.va` | stimulated Raman scattering (SRS) span + distributed Raman amplifier: a forward signal and a co-/counter-propagating pump exchange power by the **exact two-wave logistic solution** — small-signal on/off gain `e^{g_R·P_p·Leff/A_eff}` saturating with pump depletion (`Q = P_s + (ν_s/ν_p)·P_p` conserved). One component does both the two-channel WDM SRS tilt and the Raman-amp on/off gain; steady-state pins in `tests/test_raman_sbs.py`, sweeps in `examples/raman_sbs.py` |
| `sbs_fiber.va` | stimulated Brillouin scattering (SBS) span as a **threshold limiter** on directed waves: below `P_th = n_th·A_eff/(g_B·Leff)` (textbook `n_th ≈ 21`) the pump transmits; above it the transmitted power **clamps** at ~`P_th` and the surplus back-reflects as the Stokes wave (`P_fo = P_th·tanh(P_in/P_th)·e^{−αL}`, energy conserving). Power-domain model — the shared baseband envelope cannot carry the ~11 GHz Stokes shift |
| `mirror.va` | partially reflective mirror / beamsplitter on directed waves (`lo = r·e^{jφ}·li + jt·ri`), unitary + excess loss — turns unidirectional chains into standing-wave cavities |
| `ring_filter.va` | add-drop ring **filter with a five-mode comb** (m = −2…+2 at the device FSR) and a resistive heater that shifts the whole comb — the Vernier-laser building block |
| `ring_nl.va` | high-Q all-pass ring with **intracavity nonlinear absorption**: TPA of the stored energy, FCA of its carriers, Kerr red / FCD blue shifts — the model of power-limited Q |
| `ring_kerr.va` | add-drop ring whose **five FSR-spaced modes mix through the intracavity χ(3)** — the modal Lugiato-Lefever equations (machine-generated triple sum), so SPM, XPM and **four-wave mixing between resonances** emerge from one Kerr coefficient (`examples/ring_fwm.py` pins everything) |
| `ring_selfheat.va` | self-heating twin of `ring_mod.va`: the CMT all-pass ring plus a one-pole thermal reservoir (`tau_th·dΔT/dt + ΔT = R_th·P_absorbed`) — the absorbed circulating power heats the ring and thermo-optically (dn/dT>0) red-shifts the resonance, a nonlinear feedback loop that is **bistable**. The laser wavelength is an input node (`lam_nm`) so a testbench can ramp it; a slow forward+backward sweep traces a hysteresis loop (`examples/ring_selfheat.py`) |

### Traveling-wave multi-section laser slices (DFB / DBR / FP)

Cascadable **bidirectional** directed-wave sections that build DFB, DBR and
Fabry-Perot lasers by the method of lines applied to the coupled-mode
traveling-wave equations — each slice carries its own forward/backward field
state (a transit-time `ddt`, so the cavity has real group delay and real
longitudinal modes) and, for the active slice, its own carrier reservoir (so
**spatial hole burning** and **longitudinal mode competition** emerge). Chain M
in the netlist; the lasing wavelength, threshold, SMSR and mode hops are all
emergent. Ports are the field pairs `fl/fr` (forward in/out) and `bl/br`
(backward out/in).

| model | what it is |
|---|---|
| `tw_seg.va` | passive / fixed-gain traveling-wave slice: coupled-mode stencil `tau_s·dR/dt + (R−R_left) = dz·[(γ−jδ)R + jκS]` (and the mirror for S). `kappa` is the Bragg coupling — a length `L` of these reflects `tanh(κL)` at the Bragg frame and traces the coupled-mode stopband (`tests/test_tw_laser.py`); `dbeta_dv` shifts the local Bragg detuning with a tuning voltage (the **DBR** knob). `kappa=0` is a plain waveguide |
| `tw_gain_seg.va` | active traveling-wave gain slice: the `tw_seg` field stencil with a **local** Agrawal-Olsson carrier reservoir (`tau_c·dγ/dt = γ0(I) − γ(1+P_loc/p_sat)`) saturating on that slice's OWN circulating power, plus α_H chirp and a diode bias port. Cascade for a segmented gain region (spatial hole burning); give it `kappa>0` for an index-coupled active **DFB** grating |
| `phase_pad.va` | lossless bidirectional phase rotation `e^{−jφ}` on both waves: the **quarter-wave defect** of a QWS-DFB (`φ=π/2` pulls a single lasing mode to the exact Bragg wavelength) and a tunable cavity-phase / DBR pad (`φ = φ0 + dφ_dv·V`) |

Examples: `examples/tw_fp_laser.py` (segmented FP — threshold + slope efficiency
vs the lumped-`soa` analytic, plus the spatial-hole-burning profile),
`examples/dfb_laser.py` (quarter-wave-shifted DFB — single-mode lasing at the
Bragg wavelength, SMSR > 30 dB in the OSA panel), `examples/dbr_laser.py`
(tunable DBR — a section current steps the lasing wavelength across a
longitudinal mode hop). All are noise-seeded and self-checking, in the spirit of
`soa_fp_laser.py` / `soa_vernier_laser.py`.

### Dual-polarization (Jones-vector) field models

Every optical net is a Jones vector carried as **two** Ereal/Eimag pairs —
`X = (x*_re, x*_im)` is the TE component, `Y = (y*_re, y*_im)` the TM
component, with `|Ex|² + |Ey|² = power [W]`. The scalar models above are the
X/TE channel of this convention and keep working unchanged (a Y net a scalar
component never touches stays at 0). See `docs/polarization.md` for the design.

| model | what it is |
|---|---|
| `polarization_rotator.va` | fixed-angle Jones rotation `[Ex;Ey] ← R(θ)[Ex;Ey]` — a half-wave plate / TE↔TM mode rotator, lossless by default (`theta_deg`, `il_db`) |
| `pbs.va` | polarization beam splitter: TE (X) → port 1, TM (Y) → port 2, with a finite extinction ratio `er_db`. An X input rotated by θ upstream splits by Malus' law, `cos²θ`/`sin²θ` |
| `pbc.va` | polarization beam combiner: the reciprocal of `pbs.va` — X taken from port 1, Y from port 2, multiplexed onto one net |
| `birefringent_wg.va` | birefringent straight waveguide: TE/TM see different `n_eff`, so `Δφ = 2π·Δn_eff·L/λ` — the static core of PMD; in one MZI arm it offsets the TE and TM fringes (`examples/pol_malus.py`) |
| `pdl.va` | polarization-dependent loss: a common `il_db` plus a differential `pdl_db` on the TM axis (a partial polarizer) |

The `examples/pol_malus.py` study and `tests/test_polarization.py` pin the two
acceptance-criteria testbenches: a rotator + PBS + PBC Malus-law split, and a
birefringent-waveguide MZI whose TE/TM fringes are offset by `Δn_eff`.

The laser examples (`examples/soa_fp_laser.py`, `examples/soa_vernier_laser.py`)
compose `soa.va` + `mirror.va` (+ `ring_filter.va`) into cavities: lasing —
threshold, gain clamping, Vernier mode selection — emerges from the closed loop
rather than being coded into any model. Steady-state physics pins for all five
live in `tests/test_nonlinear.py`; the transient turn-on studies are the
examples themselves.

## `util/` — converters + electrical test models

| model | what it is |
|---|---|
| `cart2pol.va`, `pol2cart.va` | field representation converters (cartesian ↔ polar) |
| `res_va.va`, `capacitor.va` | trivial R / C used to validate the toolchain |

## Writing a new model

The **module name** is the component type name — keep it unique. (Historically
it also had to avoid ngspice builtin model types like `res`/`nmos`, which is why
the resistor model is `res_va`; the circulax flow no longer uses ngspice to
simulate, but the convention is kept so the models stay portable.)

### Hierarchical models

A model may instantiate other library models (see `ring_mod.va`). openvaf
parses child instances but silently drops their physics, so `cx.va` flattens
the hierarchy at the source level first (`photonflux/va_hier.py`) — the
children are inlined into one flat module before compilation. The supported
subset is deliberately small and fails loudly outside it: one module per file
(name = file stem), **named** port connections and parameter overrides only
(`mod #(.p(expr)) inst(.port(net), ...);`), every child port connected, and
`localparam real` chains for deriving override expressions from the parent's
parameters. Contributions accumulate (`<+`), so children sharing an internal
node Kirchhoff-sum into it — that is how `ring_mod.va`'s three sub-components
form the cavity ODE.
