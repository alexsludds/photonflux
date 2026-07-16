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
| [`soa_fp_laser.py`](soa_fp_laser.py) | Fabry-Perot laser from an SOA between two mirrors — lasing emerges from the loop | "SOA Fabry-Perot laser" |
| [`soa_vernier_laser.py`](soa_vernier_laser.py) | Vernier laser seeded by ASE noise, with a live one-FSR mode hop | "Vernier laser mode hop" |
| [`ring_tpa_q.py`](ring_tpa_q.py) | TPA + free-carrier absorption capping a high-Q ring | "TPA-limited high-Q ring" |
| [`wg_fwm.py`](wg_fwm.py) | χ(3) four-wave mixing in a waveguide, pinned to every textbook scaling | "χ(3) four-wave mixing" |
| [`ring_fwm.py`](ring_fwm.py) | four-wave mixing **inside** a Kerr ring resonator | "Kerr ring four-wave mixing" |

The last five have a browser twin (the named web-app examples 36–40): the
script is the pinned physics study, the web-app example is the same circuit to
click through.

```bash
.venv-circulax/bin/python examples/photodiode_tia.py    # -> out/photodiode_tia.png
.venv-circulax/bin/python examples/soa_fp_laser.py      # -> out/soa_fp_laser.png
# ...each script prints its own output path
```

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
