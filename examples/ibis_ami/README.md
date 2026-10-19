# IBIS-AMI examples and freely available models

photonflux runs IBIS-AMI transmitter and receiver models on its channels,
including circulax-simulated optics. PyAMI ([capn-freako/PyAMI](https://github.com/capn-freako/PyAMI))
parses the `.ibs` / `.ami` files; `photonflux.ibisami` runs the models;
`photonflux.serdes` is the link engine (statistical eyes, BER contours,
bathtubs, counted BER, adaptation traces). Design notes are in
[docs/ibis-ami.md](../../docs/ibis-ami.md).

```bash
python -m photonflux ami build           # compile every model this machine can build
python -m photonflux ami list            # what was found, and where each one runs
python -m photonflux ami info pf_rx      # one model's parameters (as PyAMI parses them)
```

## Examples

| script | what it shows |
|---|---|
| [`electrical_nrz_backplane.py`](electrical_nrz_backplane.py) | 32 Gb/s NRZ over a 27.7 dB backplane with FEXT: Tx preset link training in the statistical flow, BER contours, bathtubs, GetWave DFE/CDR adaptation, counted BER, and statistical vs counted BER across four decades |
| [`electrical_pam4.py`](electrical_pam4.py) | 106 Gb/s PAM4: Tx FFE (c-1, c+1) optimization, three-eye statistical analysis, Rx FFE + DFE + Mueller-Muller CDR adaptation, MM vs bang-bang CDR, frozen vs adapting Rx against the statistical prediction |
| [`optical_links.py`](optical_links.py) | IBIS-AMI Tx/Rx around circulax optics (web examples 51/52): 25G NRZ over C-band fiber swept in length, 106G PAM4 over O-band swept in MZM drive. Where IBIS-AMI's LTI assumption breaks, and the linear-fit + distortion split that repairs the statistical flow |
| [`npo_tw_mzm_link.py`](npo_tw_mzm_link.py) | near-package optics, 53.125 GBd PAM4: pf_tx -> 10 dB host trace (packages, BGA, vias, LGA, pads, every reflection) -> CTLE driver -> traveling-wave Si MZM (distributed PN electrode, terminations, walk-off) -> 500 m -> PD/TIA -> 10 dB trace -> pf_rx. Adapts the driver CTLE peaking and the Tx FFE for minimum TDECQ, then reports TDECQ from the full circuit transient and the Rx's BER |
| [`public_models.py`](public_models.py) | every library model on one channel: Init vs GetWave consistency, every Tx x Rx pairing, and SiSoft's original 2008 i386 binary (in a container) against its native rebuild |
| [`ami_model_tester.py`](ami_model_tester.py) | any model's parameters and impulse / step / frequency response, cross-checked against PyAMI's own `AMIModel.get_responses` |

The web app has the same blocks, **IBIS-AMI Tx** (a pattern source) and
**IBIS-AMI Rx** (on the receive path). Examples 50 (electrical), 51
(C-band optical) and 52 (106G PAM4 optical) use them; the Link tab shows the
IBIS-AMI report.

## Freely available IBIS-AMI models

Most vendor IBIS-AMI kits sit behind an account, an NDA or a click-through
evaluation license, and nearly all of them ship only Windows DLLs or Linux
`.so` files. In October 2026 these were public:

### Included: runs out of the box

| model (library name) | source | license | what it is | how it runs |
|---|---|---|---|---|
| `pf_tx`, `pf_rx` | this repo, [`models/ibis_ami`](../../models/ibis_ami) | BSD-3 | Tx: 6-tap FFE, PCIe presets, driver bandwidth. Rx: adaptive CTLE, AGC, FFE, DFE, bang-bang / Mueller-Muller CDR, NRZ / PAM4 | built from C99 source for any OS |
| `ibisami.example_tx`, `ibisami.example_rx` | [capn-freako/ibisami](https://github.com/capn-freako/ibisami) @ `74555dd`, vendored in [`public/ibisami`](public/ibisami) | BSD-3 | David Banas's public-domain model infrastructure: Tx FIR, Rx CTLE + adaptive DFE (no CDR) | built from source (needs Boost headers), with a one-line fix: the Rx CTLE's filter state leaked from `AMI_Init` into `AMI_GetWave` |
| `sisoft.IBIS_AMI_Tx` | SiSoft IBIS-AMI Eval Toolkit v2.21 (2009), posted to the [IBIS ATM work archive](https://ibis.org/macromodel_wip/archive/20090930/toddwesterhoff/) and mirrored by [IBIS-Library/IBIS-AMI-test-kits](https://github.com/IBIS-Library/IBIS-AMI-test-kits); the original ZIP is in [`public/sisoft_eval_v2.21`](public/sisoft_eval_v2.21) | SiSoft's permissive license (redistribution with all original files and the notice) | 4-tap FFE Tx: the IBIS ATM reference model | built natively from the kit's C source (its header lacked a struct field the 2008 `.c` uses) |
| `sisoft.IBIS_AMI_Tx@linux-x86` | the same kit's original 2008 Linux i386 `libIBIS_AMI_Tx.so` | as above | as above | out-of-process in a `linux/386` container (Docker); output identical to the native build |

### Public, but not bundled

| kit | where | why it is not bundled | to use it |
|---|---|---|---|
| Cadence IBIS AMI Evaluation Toolkit v3.1 (2011): `IBIS_AMI_RX_CTF` (FIR/IIR Rx), `IBIS_AMI_CHGAIN`; Linux i386 + Win32 binaries, C source | [IBIS-Library/IBIS-AMI-test-kits](https://github.com/IBIS-Library/IBIS-AMI-test-kits) | Cadence's license allows internal use for evaluating Cadence products only, and no redistribution | if your use fits that license, copy `Linux/MODELS/` into `vendor/cadence/`; the i386 binaries run in a `linux/386` container |
| AMD (Xilinx) transceivers: Spartan-6 GTP, Virtex-5 GTP/GTX, Virtex-6 GTX kits; UltraScale(+), Versal "IBIS-AMI lounges" | [AMD device models](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/device-models/simulation-models.html) | AMD account + simulation-model evaluation license (click-through); lounges need access approval | download, unpack into `vendor/<kit>/` |
| Intel FPGAs: Arria II GX, Stratix IV GX transceiver IBIS-AMI models | [Arria II GX](https://www.intel.com/content/www/us/en/content-details/676119/arria-ii-gx-transceiver-ibis-ami-model.html), [Stratix IV GX](https://www.intel.com/content/www/us/en/content-details/676098/stratix-iv-gx-transceiver-ibis-ami-model.html) | public pages, but the CDN refuses scripted downloads; license per kit | download in a browser into `vendor/<kit>/` |
| TI: KeyStone DSP HyperLink / PCIe SerDes ([SPRABW2](https://www.ti.com/lit/pdf/sprabw2), [SPRABW3](https://www.ti.com/lit/pdf/sprabw3): Windows 32-bit + MATLAB runtime); AM64x ([SPRACV3](https://www.ti.com/lit/pdf/spracv3): Windows 64 + Linux 64, Cadence Torrent16 PHY) | ti.com product pages | myTI login and export screening | Linux x86-64 `.so` runs in a `linux/amd64` container; Windows DLLs need a Windows Python |
| SerDesDesign.com free Tx FFE and Rx CTLE/CDR/DFE model generators | [serdesdesign.com](https://www.serdesdesign.com/) | free account; models are generated per request | generated Win64 DLL / Linux64 `.so` into `vendor/<name>/` |
| MathWorks SerDes Toolbox (PCIe, USB, DDR example models) | MathWorks | not free (MATLAB license) | the generated DLL / `.so` work like any vendor model |

Anything with an `.ibs` file under `examples/ibis_ami/vendor/` (gitignored)
shows up in `python -m photonflux ami list` as `vendor.<folder>.<model>`.
`photonflux.ibisami` reads each executable's header to find its platform:

* the same OS/CPU as Python: loaded in-process (ctypes);
* Linux i386 / x86-64 / arm64 elsewhere: run in a Docker container through
  `ami_host` (`python -m photonflux ami host-image linux/386` builds the
  host; networking is off and the model folder is mounted read-only);
* Windows DLLs: only from a Windows Python of the same bitness.

## What the public models taught us

* PyAMI (as of `e00a010`) gives a `List` parameter with `List_Tip` its first
  list entry, not its `(Default …)` (it compares the raw default string to
  the parsed integers). It also renames integer tap nodes (`-1` to `pre1`,
  `0` to `post0`) in the parameter string it builds for `AMI_Init`, and
  SiSoft's parser hangs on those names. `photonflux.ibisami` resolves
  defaults itself and sends each `.ami` file's own names.
* `pyibisami.ami.model.AMIModel.getWave` concatenates every call's whole
  clock buffer, padding included, so recovered clock times drift out of step
  after the first call. photonflux reads each call's ticks up to the `-1`
  terminator.
* ibisami's example Rx reuses one CTLE filter object across `AMI_Init` and
  `AMI_GetWave` without clearing it. A realistic channel's impulse leaves
  ~1e10 V/s in its state, and the first GetWave block then rings to ~1e5 V.
  It also recovers no clock: its ticks sit on the UI grid, so the channel
  delay decides where it samples.
