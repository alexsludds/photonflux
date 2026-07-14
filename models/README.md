# Verilog-A model library

Compiled to OSDI by the native arm64 `openvaf-r`:

```
python3 -m lightspice compile          # all models, content-hash cached
python3 -m lightspice compile models/mzm.va --force
```

## Power-domain models (node voltage = optical power [W])

| model | what it is |
|---|---|
| `laser_dml.va` | DML, static L-I (Ith, slope, Rs, Von) + first-order optical response `tau` |
| `laser_rate.va` | DML from the full rate equations: turn-on delay, relaxation oscillations, pattern-dependent ringing |
| `photodiode.va` | PIN PD: responsivity `r`, dark current `idk`, junction cap `cj` |
| `mzm.va` | Mach-Zehnder modulator: `vpi`, `vbias`, insertion loss, finite ER, electrode cap |

## Coherent-field models (paired Ereal/Eimag nodes)

| model | what it is |
|---|---|
| `laser_cw.va` | CW source emitting E-field with thermally-driven phase |
| `ring_mod.va` | Si microring modulator: coupled-mode-theory all-pass ring parameterised by the physical device (radius, group index, junction dB/m, bus κ², pm/V, fF/µm) — captures coupling condition and photon-lifetime bandwidth |
| `ring_mod_inj.va` | carrier-injection twin of `ring_mod.va`: forward-biased PIN, lifetime-limited bandwidth, FCD blue shift + FCA loss |
| `waveguide_nl.va` | nonlinear waveguide segment: TPA (exact closed form), FCA of the TPA-generated carriers (lumped population, lifetime `tau_fc`), Kerr SPM + free-carrier dispersion phase — the phase acts on the *instantaneous* envelope, so multi-tone inputs **four-wave-mix** at the textbook phase-matched efficiency (`examples/wg_fwm.py` pins every scaling). Trapezoidal z-lumping, second-order accurate: N cascaded instances converge on the distributed NLSE at 1/N². Unidirectional |
| `soa.va` | semiconductor optical amplifier: lumped Agrawal-Olsson gain reservoir `tau_c*dh/dt = h0(I) - h - (G-1)P/p_sat`, **bidirectional** (forward/backward share the reservoir), bias-current pins, gain-bandwidth pole `tau_bw` (= cavity memory in laser loops), α_H chirp, deterministic ASE seed for lasing start-up |
| `mirror.va` | partially reflective mirror / beamsplitter on directed waves: `lo = r·e^{jφ}·li + jt·ri`, unitary, plus excess loss — turns unidirectional chains into standing-wave cavities |
| `ring_filter.va` | add-drop ring **filter with a five-mode comb** (m = −2…+2 at the device FSR) and a resistive heater that thermally shifts the whole comb — the Vernier-laser building block |
| `ring_nl.va` | high-Q all-pass ring with **intracavity nonlinear absorption**: TPA of the stored energy, FCA of its carriers, Kerr red / FCD blue shifts — the model of power-limited Q |
| `ring_kerr.va` | add-drop ring whose **five FSR-spaced modes mix through the intracavity chi(3)** — the modal Lugiato-Lefever equations with the momentum-matched triple sum (85 terms, machine-generated), so SPM, XPM (×2) and **four-wave mixing between resonances** all emerge from one Kerr coefficient; `d2_hz` comb dispersion is the phase-matching knob (`examples/ring_fwm.py` pins everything) |
| `cart2pol.va`, `pol2cart.va` | representation converters |

The laser examples (`examples/soa_fp_laser.py`, `examples/soa_vernier_laser.py`)
compose `soa.va` + `mirror.va` (+ `ring_filter.va`) into cavities: lasing —
threshold, gain clamping, Vernier mode selection — emerges from the closed
loop rather than being coded into any model. Physics pins for all five live
in `tests/test_nonlinear.py`.

## Electrical test models

`res_va.va`, `capacitor.va` — trivial R/C used to validate the toolchain.

## Naming rule (important)

The **module name** is what ngspice `.model` cards reference. It must not
collide with a builtin ngspice model type (`res`, `r`, `c`, `l`, `d`,
`diode`, `sw`, `npn`, `nmos`, ...) or the deck parser binds the builtin
and elaboration fails with `incorrect model type! Expected OSDI device`.
`lightspice` refuses to compile colliding names — that is why the
resistor model is `res_va`, not `res`.
