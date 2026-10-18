# photonflux IBIS-AMI models: `pf_tx`, `pf_rx`

Open-source (BSD-3), dependency-free C99 IBIS-AMI models. Each is dual mode:
`AMI_Init` (statistical flow) and `AMI_GetWave` (time-domain flow) share one
implementation of every filter, so the two flows agree sample for sample.
`python -m photonflux ami build` compiles them for the host into
`build/<os>-<arch>/`, next to copies of the `.ami` and `.ibs` files.

| file | |
|---|---|
| `src/pf_tx.c` | transmitter |
| `src/pf_rx.c` | receiver |
| `src/pf_ami.{h,c}` | parameter-string tokenizer, string buffers, biquad and delay-line blocks |
| `pf_tx.ami`, `pf_rx.ami` | parameter definitions (IBIS 7.1); `python -m photonflux ami info pf_rx` prints them |
| `pf_serdes.ibs` | IBIS wrapper: 50 Ohm buffers, one Tx and one Rx model |
| `host/ami_host.c` | serves any AMI library over a pipe, for out-of-process / container runs |

## `pf_tx`

6-tap UI-spaced FFE (c-2, c-1, main, c+1, c+2, c+3), linear and signed. The
main tap takes the rest of the swing (c0 = 1 - sum|c_k|) so the peak output
stays at `tx_swing`. `tx_preset` = 0..9 selects the PCIe 3.0+ presets P0..P9.
`tx_bw` adds a two-pole driver response. The FIR is causal: the main cursor
sits 2 UI behind the newest input in both Init and GetWave.

## `pf_rx`

```
in -> CTLE -> AGC -> UI-spaced FFE -(+)-> out (slicer input)
                                    ^ DFE feedback
                         CDR (bang-bang / Mueller-Muller), NRZ / PAM4 slicer
```

* **Init**: picks the CTLE peaking (`ctle_mode` 2: 0..`ctle_max_db`) and the
  sampling phase that maximize half-eye over sqrt(residual ISI + shaped
  input noise `rx_noise_nv`); MMSE-designs the FFE around a zero-forcing
  DFE; sets the AGC so the outer level is `agc_target`; returns the
  equalized impulse response (DFE post-cursors notched out) and seeds
  GetWave with that design.
* **GetWave**: the same CTLE / AGC / FFE sample by sample. The DFE feedback
  updates half a UI after each decision. The CDR runs a proportional-integral
  loop on an Alexander (bang-bang, edge samples taken before the DFE
  summer) or Mueller-Muller detector. LMS adapts the DFE and FFE (the main
  tap is fixed; under an adaptive CDR so is the first pre-cursor tap, since
  the CDR owns timing), tracks the data level, and optionally the AGC.
  A bang-bang loop locks midway between data crossings; with a strong DFE
  the best data phase is elsewhere, so Init calibrates the offset from the
  equalized pulse (`cdr_calibrate`, up to a quarter UI) plus any manual
  `cdr_offset_ui`.
  `AMI_parameters_out` after every call reports CTLE, AGC, level, CDR phase
  and frequency, and the FFE / DFE taps.
* `Modulation` (reserved, In): `NRZ` or `PAM4` (Gray, thresholds at 0 and
  +/-2/3 of the adapted outer level).
* Clock ticks follow the IBIS convention (sampling instant - UI/2), with
  `-1` after the last.
