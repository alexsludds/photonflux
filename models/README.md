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
| `ring_mod.va` | Si microring modulator: coupled-mode-theory all-pass ring from the physical device (radius, group index, junction dB/m, bus κ², pm/V, fF/µm) — coupling condition + photon-lifetime bandwidth |
| `ring_mod_inj.va` | carrier-injection twin of `ring_mod.va`: forward-biased PIN, lifetime-limited bandwidth, FCD blue shift + FCA loss |
| `waveguide_nl.va` | nonlinear waveguide segment: TPA (exact closed form), FCA (lumped population, lifetime `tau_fc`), Kerr SPM + free-carrier dispersion — the instantaneous phase makes multi-tone inputs **four-wave-mix** at the textbook efficiency (`examples/wg_fwm.py` pins every scaling). Trapezoidal z-lumping, 1/N² NLSE convergence. Unidirectional |
| `soa.va` | semiconductor optical amplifier: lumped Agrawal-Olsson gain reservoir, **bidirectional** (forward/backward share the reservoir), α_H chirp, deterministic ASE seed for lasing start-up |
| `mirror.va` | partially reflective mirror / beamsplitter on directed waves (`lo = r·e^{jφ}·li + jt·ri`), unitary + excess loss — turns unidirectional chains into standing-wave cavities |
| `ring_filter.va` | add-drop ring **filter with a five-mode comb** (m = −2…+2 at the device FSR) and a resistive heater that shifts the whole comb — the Vernier-laser building block |
| `ring_nl.va` | high-Q all-pass ring with **intracavity nonlinear absorption**: TPA of the stored energy, FCA of its carriers, Kerr red / FCD blue shifts — the model of power-limited Q |
| `ring_kerr.va` | add-drop ring whose **five FSR-spaced modes mix through the intracavity χ(3)** — the modal Lugiato-Lefever equations (machine-generated triple sum), so SPM, XPM and **four-wave mixing between resonances** emerge from one Kerr coefficient (`examples/ring_fwm.py` pins everything) |

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
