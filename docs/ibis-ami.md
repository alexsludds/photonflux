# IBIS-AMI in photonflux

IBIS-AMI is how SerDes vendors ship equalization and clock recovery: a
shared library exporting `AMI_Init`, `AMI_GetWave` and `AMI_Close`, a
`.ami` file declaring its parameters, and an `.ibs` file naming the
executables per platform. photonflux plays the role of the EDA tool. It
loads the models, feeds them channels (analytic, Touchstone-like, or
circulax-simulated optics), and runs both IBIS flows. Model catalog and
examples: [examples/ibis_ami/README.md](../examples/ibis_ami/README.md).

## Pieces

| module | role |
|---|---|
| [`photonflux/ibisami.py`](../photonflux/ibisami.py) | loading and running models: `AmiParams` (PyAMI-parsed parameter tree and values), `AmiModel` (`init`, `getwave`), executors, `library()` / `load()` |
| [`photonflux/ami_build.py`](../photonflux/ami_build.py) | compiling models for the host; container hosts |
| [`photonflux/serdes.py`](../photonflux/serdes.py) | the link engine: channels, statistical eye, both flows, waveform analysis, plotting |
| [`photonflux/rfnet.py`](../photonflux/rfnet.py) | analog channels as ABCD cascades: causal RLGC lines, pads, bumps, vias, stubs; terminated transfer V(load)/V(source) with every re-reflection, S-parameters, causal impulse |
| [`models/ibis_ami/`](../models/ibis_ami/) | photonflux's own models (`pf_tx`, `pf_rx`) and `ami_host` |
| [`webapp/amilink.py`](../webapp/amilink.py) | the web app's IBIS-AMI Tx source and Rx report |

### What PyAMI does, and what photonflux does itself

PyAMI ([capn-freako/PyAMI](https://github.com/capn-freako/PyAMI), pinned in
`pyproject.toml`) parses `.ibs` files (`pyibisami.ibis.parser`), `.ami`
parameter trees (`AMIParamConfigurator`, `AMIParameter`) and the
`AMI_parameters_out` strings models return (`ami_parse`). photonflux adds:

* **Parameter values**. `AmiParams` starts from each parameter's own
  default (PyAMI's trait default skips `(Default …)` for `List` parameters
  with `List_Tip`), validates against `Range` / `List`, and builds the
  `AMI_parameters_in` string itself. That string carries reserved `In`
  parameters such as `Modulation`, and integer tap names exactly as the
  `.ami` file writes them (PyAMI renames them `pre1` / `post0`).
* **Execution**. `AmiModel.getwave` reads each call's clock ticks up to the
  `-1` terminator and records `AMI_parameters_out` per call: the adaptation
  trace. `AmiModel.pyami_model()` still hands back a PyAMI `AMIModel` for
  PyAMI's own tools (`examples/ibis_ami/ami_model_tester.py` checks the two
  against each other).

### Where a model runs

`binary_platform()` reads the library header (ELF / Mach-O incl. universal /
PE) and `AmiModel` picks an executor:

| host | when | how |
|---|---|---|
| `inproc` | same OS / CPU as the Python process | ctypes |
| `process` | same platform, but isolated (a crashing model cannot take Python down) | `ami_host` as a subprocess |
| `container` | Linux i386 / x86-64 / arm64 libraries elsewhere | `ami_host` in a `photonflux-ami-host:<arch>` Docker image; `--network none`, model folder read-only |

`ami_host` speaks a small binary protocol on stdin/stdout (see the header
of `models/ibis_ami/host/ami_host.c`). The model's own `stdout` is redirected
to `stderr` first, so models that `printf` cannot corrupt the stream.
Replies are bounded by `PHOTONFLUX_AMI_TIMEOUT` (600 s), so a hung vendor
model raises instead of blocking. SiSoft's original 2008 i386 binary,
container-hosted, matches its native rebuild bit for bit.

## The two flows

**Statistical (Init)**, `serdes.statistical_link`: channel impulse ->
Tx `AMI_Init` -> Rx `AMI_Init` -> pulse response -> `stat_eye`.

* ISI: the exact PDF over every cursor of equiprobable symbols (NRZ or
  Gray PAM4), discretized on a voltage grid. Splitting a fractional shift
  over two bins adds f(1-f)dv^2 of variance per cursor, and that much is
  subtracted back from the Gaussian part, so the BER is grid-independent.
* Noise and negligible cursors: Gaussian, in closed form (`ndtr` on the tail
  side, accurate to 1e-300). Input-referred noise reaches the slicer
  through the Rx's linear front end, measured by `rx_linear_response`: two
  extra `AMI_Init` calls with a +/-eps impulse in the channel's quiet tail.
  This works for any model whose adaptation keys on the main pulse.
* Crosstalk: aggressor rows of the impulse matrix go through both Inits and
  their cursors join the ISI convolution as independent data.
* Jitter: random (rms) and dual-Dirac convolved along the phase axis.
  Defaults come from `Tx_Rj` / `Tx_Dj` / `Rx_Rj` / `Rx_Dj`, applied as
  sampling jitter: the usual, pessimistic reading of Tx jitter.
* Output: per-eye BER maps (contours), timing and voltage bathtubs, eye
  height / width at a target BER, SER and Gray BER.

**Time domain (GetWave)**, `serdes.time_domain_link` / `serdes.receive`:
PRBS symbols -> +/-0.5 V stimulus (with Tx jitter) -> Tx `AMI_GetWave` ->
channel (convolution plus crosstalk aggressors with independent data, or
any callable) -> input noise -> Rx `AMI_GetWave`. The Rx's own recovered
clock samples its own output. Decisions are aligned to the transmitted
symbols, and errors are counted after `Ignore_Bits`. Also: per-eye Q fit, a
bathtub from re-sampling across the UI, a CDR-referenced eye density, and
the adaptation trace. A model without a clock gets an ideal one at the eye
center, found data-aided.

The two flows agree (examples `electrical_nrz_backplane.py`,
`electrical_pam4.py`, `tests/test_ibisami.py`):

| link | noise for statistical BER | counted BER |
|---|---|---|
| 32G NRZ, 27.7 dB + FEXT, adapting Rx | 1e-2 / 1e-3 / 1e-4 / 1e-5 | 9.8e-3 / 9.8e-4 / 1.6e-4 / 1.4e-5 |
| 106G PAM4, Rx frozen at its Init design | ~2e-4 at 18 mV | ~2.6e-4 |

An adapting PAM4 receiver beats the statistical prediction under
crosstalk: Init's MMSE design sees only the victim's pulse, while LMS also
minimizes the crosstalk it actually receives.

## Adapting from a cold start

`pf_rx` with `adapt_start=1` starts `AMI_GetWave` from a pass-through FFE
and zero DFE taps; decision-directed LMS learns the equalizer from the
received waveform, as an optical DSP does at power-up. With the
Mueller-Muller CDR on the FFE input (`cdr_mode=3`) timing never depends on
the taps, so every FFE tap but the main one adapts (under `cdr_mode` 1 or 2
the first pre-cursor tap keeps Init's value; an FFE or DFE fixed at Init's
design keeps the whole start there). On a 0.3 m backplane
the learned 5-tap FFE lands within 0.03 of `AMI_Init`'s MMSE design
(`tests/test_ibisami.py`).

`examples/ibis_ami/mrm_pam4_adaptive_rx.py` (web example 53) runs it on
a 53.125 GBd PAM4 microring link (30 GHz driver, ring 120 pm red of the
laser, 18 GHz TIA; pulse cursors 0.12 / 1 / 0.22, 5.5 % nonlinear
residual), 6 mV rms Rx noise, 20000 UI:

| Rx, adapted from cold | counted BER | Q-fit |
|---|---|---|
| no EQ | 3.9e-2 | 3.6e-2 |
| DFE 1 tap | 1.6e-3 | 9.8e-3 |
| FFE 3 / 5 / 7 | 0 / 0 / 0 | 2.8e-6 / 2.5e-7 / 5.7e-8 |
| FFE 3 / 5 / 7 + DFE 1 | 0 / 0 / 0 | 1.1e-7 / 5.0e-8 / 5.1e-9 |

Every tap converges within ~3000 UI. The FFE's c(+1) and the DFE tap both
cancel the first post-cursor, so the error surface is nearly flat along
their trade-off: the cold start settles at c(+1) = -0.05, b1 = 0.15, while
`AMI_Init`'s design (and LMS seeded from it) uses c(+1) = +0.2, b1 = 0.4,
at nearly the same MSE (Q-fit 5e-8 vs 1.4e-8). Without an FFE the 12 %
pre-cursor stays, and the DFE's error propagation makes it worse than
no EQ at higher noise.

## Nonlinear channels: the optical case

IBIS-AMI's statistical flow assumes an LTI channel. A direct-detection
optical link is not one: the photodiode squares the field after chromatic
dispersion, and an MZM compresses large swings. `serdes.analyze_waveform`
handles a received waveform whose symbols are known (a circulax transient
of the whole optical link, with the Tx's `AMI_GetWave` output as the drive):

1. `estimate_channel`: per sample phase, least squares for the best
   linear pulse response. That is the LTI channel `AMI_Init` gets; the
   residual is what no LTI channel explains.
2. Time domain: the Rx's `AMI_GetWave` on the waveform itself (the ground
   truth).
3. Statistical: the residual is filtered through the Rx's linear front end,
   read at the recovered clock's phase, and split. Its per-symbol mean
   moves the eye's levels (static distortion such as MZM compression; slicer
   thresholds re-optimize). The rest is added as Gaussian noise. This is a
   Bussgang-style split into a linear part plus uncorrelated distortion.

On 25G NRZ over C-band fiber (web example 51) the pure-LTI statistical BER
says 1e-300 at every length, while the split tracks the time domain:

| fiber | linear-fit residual | statistical (split) | time domain |
|---|---|---|---|
| 16 km | 25 % | 6e-15 | Q-fit 3e-15 |
| 20 km | 28.5 % | 8e-6 | Q-fit 1.3e-4 |
| 22 km | 29.6 % | 4.2e-2 | counted 4.6e-2 |

## Near-package optics: host traces, driver, traveling-wave MZM

`examples/ibis_ami/npo_tw_mzm_link.py` is a 53.125 GBd PAM4 link with the
optics on the host board: pf_tx -> 10 dB PCB trace -> linear driver with a
CTLE -> traveling-wave Si MZM -> 500 m SMF -> PD / TIA -> 10 dB PCB trace
-> pf_rx. Each part is modeled where its physics lives:

* **Host channels** (`photonflux.rfnet`, frequency domain, odd mode of the
  100 Ohm pair): pi-coil-compensated die pads, bumps, a 10 mm 45 Ohm
  package trace, BGA ball, backdrilled vias, an 18.4 cm stripline sized for
  10.0 dB at 26.5625 GHz (sqrt(f) skin loss with a DC floor, causal
  Djordjevic-Sarkar dielectric), an LGA interposer and the module substrate,
  between the real source and load impedances. The terminated transfer
  keeps every re-reflection, including the 2.3 ns trace round trip. Total
  SDD21 at Nyquist is -12.3 dB; SDD11 / SDD22 are -10.6 to -22 dB.
* **Driver** (numpy): input-referred noise, CTLE (0-12 dB in 0.5 dB codes),
  pre-driver pole, a gain control that sets the output swing, and a tanh
  output stage driving its EMF into the circuit.
* **Driver output network to TIA** (circulax transient): 40 Ohm
  back-termination, pad capacitance, flip-chip bump, then two
  `pn_phase_shifter_tw` arms (3.5 mm, push-pull GSSG). Each is a
  distributed RLC electrode loaded every 100 um by a sqrt-law depletion
  junction, with walk-off and an on-chip 40 Ohm termination to the 2 V
  junction bias. The terminations equal the loaded line's |Zc| at
  f_nyq / 2 (47 Ohm at 1 GHz, 40.8 Ohm at Nyquist; |S11| at the driver
  -12.5 dB). The EO response is -3.3 dB at Nyquist. Couplers and a
  lambda/4 offset bias the MZM at quadrature. Then the fiber, PD and TIA.

The search sets the driver CTLE and pf_tx's c(-1), c(+1), c(+2) for
minimum TDECQ at TP2, the fiber after the PIC coupling (802.3cd method).
A transient of the full PRBS13Q period costs ~100 s, so the search runs on
a Wiener surrogate fitted to a transient: an FIR from the driver EMF to
the optical power, then a cubic. The surrogate matches the circuit to
0.25 % rms. Every reported number comes from a full transient at the
chosen setting.

Plain TDECQ can be gamed by over-equalizing. An overshooting drive that
the reference FFE low-passes back (C_eq < 1) scored 0.25 dB while ER fell
to 1.7 dB. The search therefore minimizes TDECQ - 10 log10(min(C_eq, 1)),
which 802.3 also limits.

| | CTLE | Tx FFE c(-1) / c(+1) / c(+2) | TDECQ | OMA_outer | ER | RLM | pf_rx counted BER | pf_rx statistical BER |
|---|---|---|---|---|---|---|---|---|
| no equalization | 0 dB | 0 / 0 / 0 | 16.05 dB | 3.0 dBm | 4.9 dB | 0.958 | 5.1e-4 | 2.0e-4 |
| adapted | 12 dB (max) | -0.161 / -0.142 / +0.023 | **0.88 dB** (C_eq 0.975) | 0.1 dBm | 2.3 dB | 0.934 | 0 / 11868 bits | 3.7e-12 |

The CTLE ends at its 12 dB limit and the Tx FFE covers the rest. After
the 10 dB Rx trace, pf_rx opens eyes of 37 / 49 / 37 mV
at 1e-6.

## The web app

* **IBIS-AMI Tx** (`ami_tx`, Sources): the PRBS / PAM4 pattern, jitter
  included, as the IBIS stimulus, through the model's `AMI_GetWave` at
  compile time, then scaled by v0 / v1 (so level edits do not recompile).
* **IBIS-AMI Rx** (`ami_rx`, Amplifiers & EQ): a unity buffer in the
  transient. With the "BER vs" probe at its input, the Link tab gets an
  IBIS-AMI card: counted BER from the model's own clock, statistical BER
  (split and pure-LTI), eye height / width, linear-fit residual, a
  statistical-eye map, bathtubs and every adapted quantity vs time. A
  transient with an AMI Rx is saved at 32 samples/UI automatically.
* `Parameters` on both blocks takes `name=value` overrides of the `.ami`
  defaults.
* **Host PCB Link (NPO)** (`pcb_link`, Channels): `rfnet.host_module_link`
  terminated at both ends (source R and pad C inside the block, so drive it
  with the source EMF), vector-fitted with the bulk flight delay removed.
  The fit runs from 100 kHz, log-spaced so the DC level holds, to 8 f_nyq,
  so its instant feedthrough stays ~1e-4. Its error is ~0.1 %, and the
  2.3 ns trace echo stays in the response.
* **Modulator Driver (CTLE)** (`mod_driver`, Amplifiers & EQ): CTLE,
  pre-driver pole, gain and a tanh-limited differential EMF output; the
  back-terminations and pads are separate parts.

Example 54 is `npo_tw_mzm_link.py` in the GUI, at the adapted settings:
its Link tab reports pf_rx with 0 errors in 8190 bits, a statistical BER
of 2.4e-12 and eyes of 40 / 49 / 37 mV at 1e-6. On its ~6k-UI prbs+oma
record the Eye tab's Measure gives TDECQ 0.99 dB (OMA_outer 0.99 mW); the
script's full PRBS13Q gives 0.88 dB.

## Building

`python -m photonflux ami build` (also run on first use):

* `pf_tx`, `pf_rx`, `ami_host`: any C compiler (`CC`);
* ibisami examples: a C++ compiler and Boost headers (`BOOST_ROOT`, or
  Homebrew / system include dirs);
* SiSoft Tx: native from the kit's source, plus the original i386 binary
  extracted for container runs.

The Docker image builds all of them for Linux in its `ami-models` stage.
`python -m photonflux doctor` reports PyAMI, the model build, Boost and
Docker; Boost and Docker are optional.
