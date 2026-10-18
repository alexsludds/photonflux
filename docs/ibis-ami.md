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
| 22 km | 29.6 % | 4.6e-2 | counted 4.6e-2 |

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
