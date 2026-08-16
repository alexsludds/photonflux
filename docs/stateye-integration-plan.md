# Plan: photonflux as a dependency of `AyarLabs/stateye`, with a 53.125 GBd MRM TDEC optimizer

Goal: make photonflux installable from within
[`AyarLabs/stateye`](https://github.com/AyarLabs/stateye), and ship an example there
that optimizes **OMA − TDEC** for a 53.125 GBd NRZ silicon microring modulator driven
by a SKY130 CMOS inverter, over the MRM bus gap, the laser locking point, and the
inverter (W, L).

Everything in the "Measured" tables below was run on this machine against the real
repos; nothing there is an estimate.

## Implementation status

| Phase | Status | Landed as |
|---|---|---|
| 0 — stateye packaging (B1–B6) | **done**, needs upstreaming | [`docs/patches/stateye-modern-toolchain.patch`](patches/stateye-modern-toolchain.patch) |
| 1 — PRBS-13 generator | **done** | [`photonflux/signals.py`](../photonflux/signals.py), [`tests/test_signals.py`](../tests/test_signals.py) |
| 1 — stateye bridge, OMA−TDEC | **done** | [`photonflux/tdec.py`](../photonflux/tdec.py), [`tests/test_tdec.py`](../tests/test_tdec.py) |
| 2 — gap → κ² map | **done** | [`photonflux/coupler.py`](../photonflux/coupler.py), [`tests/test_coupler.py`](../tests/test_coupler.py) |
| 2 — scipy Bessel reference receiver | **done** | `photonflux.tdec.reference_receiver` |
| 2 — junction Δλ(V) / C_j(V) law | **not done** | see 4.3; the (W, L) optimum is biased until this lands |
| 3 — optimizer + report | **done** | [`examples/mrm_tdec_sky130.py`](../examples/mrm_tdec_sky130.py) |
| 4 — corners, energy Pareto, thermal | **not done** | see section 6 |

The dependency direction landed as photonflux ← stateye (an optional `eye` extra in
`pyproject.toml`), not the reverse. Section 1 explains why that was the right call and
what the mirror-image change would be; the bridge module is symmetric either way.

---

## 0. What I verified before planning

**stateye** was cloned at `main` (v1.7, last CHANGELOG entry 04/28/2024) and built.
**photonflux** was exercised through `examples/ring_mod_sky130.py` at the target baud.

### Blockers found in stateye (all reproduced, all fixable)

| # | Problem | Evidence | Fix |
|---|---|---|---|
| B1 | No `pyproject.toml`, so PEP 517 build isolation runs `setup.py` in a clean env where `import numpy` (line 2) fails | `pip install .` → `ModuleNotFoundError: No module named 'numpy'` | Add `pyproject.toml` with `[build-system] requires = ["setuptools","wheel","Cython","numpy"]` |
| B2 | `utilities.pyx` uses `cnp.int_t`, removed for NumPy 2 | `utilities.pyx:75:22: Invalid type.` under Cython 3.2.9 | Two-line change to `cnp.int64_t` — **verified: builds and imports clean against NumPy 2.5.2 + Cython 3.2.9** |
| B3 | Building against NumPy 1 then resolving `requirements.txt` to NumPy 2 yields a broken install | `ValueError: numpy.dtype size changed... Expected 96 from C header, got 88` | Falls out of B2; do not pin NumPy 1 |
| B4 | `requirements.txt` is the `install_requires`, and it contains `black`, `pre-commit`, `pytest` | Installing stateye drags in a formatter and a git-hook manager | Split runtime vs `[dev]` extra |
| B5 | `python_requires=">=3.6"`; `setup.py install` is the documented path | README install section | Bump to `>=3.10`, document `pip install .` |
| B6 | `eye.py` calls `matplotlib.cm.get_cmap`, removed in matplotlib 3.9 | `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` on every `eye.plot()` / `plot_bathtub()` | `matplotlib.colormaps["inferno"]` — 3 sites (`eye.py:590,595,659`) |

B1–B3 are the load-bearing ones: **today, `pip install git+https://github.com/AyarLabs/stateye`
fails outright on a modern toolchain.** That has to be fixed before anything can depend
on stateye or ship alongside it.

### Cost of a photonflux evaluation

`examples/ring_mod_sky130.py --baud 53.125e9`, single-neut SKY130 inverter driving
`models/optical_field/ring_mod.va`, 40 samples/UI:

| Operation | Wall time |
|---|---|
| First-ever run (VA→OSDI compile, SKY130 card extraction, JAX JIT) | **132 s** |
| Warm run, 127 bits | **12.0 s** |
| Warm run, 508 bits | **14.5 s** |
| Warm run, 508 bits at a *new* `kappa2` | **14.5 s** — no recompile |
| **New `(W, L)` FET pair** (ngspice `showmod` + BSIM4→OSDI) | **41 s** (nfet), **80 s** (pfet) |
| Repeat `(W, L)` (content-hash cache hit) | **0.00 s** |

Fitting the two warm points: ≈ **11.2 s fixed per process** (interpreter + JIT) plus
**6.6 ms per bit** of transient at 40 samples/UI (**5.3 ms/bit at 32 sps**). So inside
one long-lived process a 511-bit PRBS-9 record costs ~**2.7 s** and a full 8191-bit
PRBS-13 record ~**43 s**, with the 11 s paid once rather than per evaluation.

This is the single most important number in the plan and it inverts the naive
architecture: **transient length is cheap; changing transistor geometry is not.**

### Physics baseline at the target rate

The 53.125 GBd run reports:

```
device: R = 7.5 um, 7000 dB/m, kappa^2 = 0.100 (critical = 0.076)
        -> Q_i = 11903, Q_e = 9041, Q_loaded = 5138, f_3dB = 44.5 GHz
operating point: linewidth = 255 pm, swing = 81 pm, laser detune = -33 pm, cj = 23.6 fF
eye: height = 0.1596 mW, levels 0.3997 / 0.0898 mW, ER = 6.5 dB
driver: electrode 20-80% edge = 16.0 ps (UI = 19 ps), Cgd overshoot = +253/-185 mV
note: 53.125 Gbd exceeds the SKY130 inverter edge rate (16 ps 20-80% vs 19 ps UI
      — a 130 nm/1.8 V technology limit)
```

**Predicted outcome of the optimization, stated up front so it can be falsified:** the
ring is not the bottleneck (44.5 GHz photon-lifetime bandwidth against a 53.125 GBd
UI), the SKY130 inverter is. The optimizer will spend its budget buying electrode edge
rate, and TDEC will be dominated by driver-induced ISI and duty-cycle distortion rather
than by cavity response. Expect the useful design freedom to be in **(W, L) and the
lock point**, with the gap mattering mainly through the κ²↔OMA↔bandwidth trade.

### stateye's TDEC path, confirmed working

Against a synthetic 53.125 GBd NRZ waveform through a 35 GHz single-pole channel, on
the patched NumPy 2 build:

```
oma_8180                   1.0000
tdec_8180                  0.8075        (dimensionless — dB)
inner_eye_height           0.8715
units oma: mW | tdec: None
```

TDEC is computed as `10*log10((OMA/2) / (Qinv(BER) * R))` where `R` is the addable
receiver noise solved per IEEE 802.3-2015 Eq. (95-3), sampling four vertical histograms
at 0.4 and 0.6 UI (`histogram_analysis.py:148`). It needs `set_tdec_s_noise`,
`set_tdec_m1`, `set_tdec_m2`, `set_tdec_ber`.

### PRBS-13 run statistics vs. stateye's `_8180` filters

Generated with the IEEE 802.3 polynomial `x^13 + x^12 + x^2 + x + 1` and counted
cyclically over the full period:

| Pattern | Length | max run of 1s | max run of 0s | `0^8→1^8` | `1^8→0^8` |
|---|---|---|---|---|---|
| PRBS-7 | 127 | 7 | 6 | 0 | 0 |
| PRBS-9 | 511 | 9 | 8 | 0 | 0 |
| PRBS-11 | 2047 | 11 | 10 | 0 | 1 |
| **PRBS-13** | **8191** | **13** | **12** | **0** | **1** |
| PRBS-15 | 32767 | 15 | 14 | 0 | 1 |

This splits stateye's `_8180` measurements into two groups, confirmed by running a real
PRBS-13 waveform through `IdealEye` (53.125 GBd, 32 sps, BT4 at 0.75 × baud):

| Measurement | Result | Count | Why |
|---|---|---|---|
| `oma_8180` | 0.3100 mW ✅ | 15 | `compute_x1x0_oma` finds 8-runs of 1s and of 0s **independently** |
| `tdec_8180` | −0.3383 dB ✅ | — | depends only on `oma_8180` and `R` |
| `extinction_ratio_8180` | 6.478 dB ✅ | 15 | same |
| `dcd_8180` | **nan** ❌ | 0 | `OMA_FILTER_MAP["8180"]` needs the *adjacent* 16-bit run |
| `rise_time_20-80_8180` | **nan** ❌ | 0 | same |

So PRBS-13 delivers the headline OMA/TDEC numbers but **not** the `_8180` edge-rate,
DCD or overshoot diagnostics — those need the `_4140` variants (count 255 on PRBS-13).

**Truncating PRBS-13 is not safe.** Runs of ≥8 in the first N bits:

| First N bits | runs ≥8 ones | runs ≥8 zeros |
|---|---|---|
| 511 | **0** | 3 |
| 1023 | 2 | 3 |
| 2047 | 6 | 4 |
| 4095 | 7 | 6 |
| **8191 (full period)** | **16** | **16** |

The full period is required; a 511-bit prefix yields `oma_8180 = nan`.

### `tdec_4140` on PRBS-9 is a near-exact, 16× cheaper surrogate

Same synthetic link, sweeping the reference-receiver bandwidth to close the eye.
Measured through the shipped `photonflux.tdec.measure` path (which discards 8 settle
UI), not a prototype:

| BT4 bandwidth | PRBS-13 `tdec_8180` (ref) | PRBS-9 `tdec_4140` | error |
|---|---|---|---|
| 1.00 × baud | −0.4139 dB | −0.4055 | +0.0085 |
| 0.75 × baud | −0.3383 dB | −0.3294 | +0.0089 |
| 0.60 × baud | −0.0753 dB | −0.0656 | +0.0098 |
| 0.50 × baud | +0.2385 dB | +0.2486 | +0.0101 |
| 0.40 × baud | +0.7776 dB | +0.7884 | +0.0108 |
| 0.35 × baud | +1.3239 dB | +1.3401 | +0.0161 |

The important property is not that the error is small but that it is a **bias, not
scatter**: +0.009 to +0.016 dB across a 1.7 dB span of TDEC. A near-constant offset
cannot reorder candidates, which is all the surrogate is asked to do — and the spread
(< 0.01 dB) is far below the differences the optimizer resolves. `tests/test_tdec.py`
asserts both the magnitude and the low spread, so a regression to *scattered* error
fails the suite rather than silently corrupting the ranking. Note PRBS-9 gives
`oma_8180 = nan` (count 0) — the surrogate is `_4140`, and only `_4140`.

### scipy Bessel reference receiver, `norm="mag"` verified

`scipy.signal.bessel(4, bw_factor*baud, "low", fs=1/dt, output="sos", norm="mag")`,
−3 dB point measured back off the realized `sosfreqz` response:

| `bw_factor` | requested f₃dB | measured | error |
|---|---|---|---|
| 0.50 | 26.56 GHz | 26.52 GHz | −0.16 % |
| 0.75 | 39.84 GHz | 39.78 GHz | −0.16 % |
| 1.00 | 53.12 GHz | 53.04 GHz | −0.16 % |

`norm="mag"` is mandatory — scipy's default is `norm="phase"`, which normalizes the
phase response instead and puts the −3 dB point somewhere else entirely.

---

## 1. Which way should the dependency actually point?

The request is photonflux-as-dependency-**of**-stateye. Taken literally that is
backwards in weight class: stateye is a ~2 kLOC NumPy/Cython measurement library whose
users feed it scope captures; photonflux pulls in JAX, circulax, diffrax, `libngspice`,
the SKY130 PDK via volare, and a Rust/LLVM-18-built `openvaf-ir` binary. Making that a
hard `install_requires` of stateye would break stateye for every one of its existing
users.

**Recommendation — put the example in stateye, but make photonflux an optional extra
with lazy imports.** This satisfies the request (the example lives in stateye, and
`pip install stateye[photonflux]` gets you photonflux) without regressing the base
install.

```
stateye/
  pyproject.toml            # B1 fix; declares the extras below
  stateye/                  # unchanged, no photonflux import anywhere
  stateye_sources/          # NEW optional subpackage — waveform *producers*
    __init__.py             # raises a clear ImportError if photonflux is absent
    photonflux_link.py      # the bridge (section 3)
    ref_receiver.py         # IEEE reference receiver (section 4.2)
  examples/
    mrm_sky130_tdec_53g/    # the deliverable (section 5)
```

```toml
[project.optional-dependencies]
photonflux = ["photonflux @ git+https://github.com/<owner>/photonflux@<tag>"]
dev = ["pytest", "black", "pre-commit"]
```

Two things to be honest about with this layout:

- **photonflux is not on PyPI and cannot be made a normal wheel.** `bin/openvaf-ir` is a
  native, arch-specific binary built by `scripts/build-openvaf.sh` (Rust + LLVM 18), and
  the SKY130 PDK arrives through `volare`. A pip extra can install the Python package;
  it cannot install the toolchain. The extra must therefore be paired with
  `python -m photonflux doctor` in the example's preflight, and the example's README
  must point at photonflux's `Dockerfile` as the reproducible path. Anything that runs
  in stateye's CI must be the *replay* test (section 6), not a live solve.
- Tag photonflux and depend on the tag, not `main`. Recent history on `main` moves fast
  (APD, traveling-wave lasers, Raman/Brillouin all landed in the last few commits), and
  an example pinned to a moving branch will rot.

If the intent was ever the other direction, the mirror-image change in photonflux is
one line in `pyproject.toml` (`[project.optional-dependencies] eye = ["stateye @ ..."]`)
plus the same bridge module — the bridge is symmetric and the work below is unchanged.

---

## 2. Phase 0 — make stateye installable (½ day)

Upstream to stateye, independent of everything else:

1. Add `pyproject.toml` (B1) with build requires and `requires-python = ">=3.10"`.
2. `sed -i 's/cnp\.int_t/cnp.int64_t/g' stateye/measurements/utilities.pyx` (B2).
   Verified sufficient; `dtype=int` is int64 on both macOS-arm64 and Linux-x86_64, but
   add a `cnp.import_array()`-adjacent assertion or use `cnp.npy_intp` if Windows
   support matters, where `dtype=int` is int32.
3. Move `black`/`pre-commit`/`pytest` out of `install_requires` into `[dev]` (B4).
4. Fix the dead `url=` (points at `noether.ayarlabs.com`).
5. Add a CI matrix job that does a clean `pip install .` on 3.10/3.11/3.12 with NumPy 2
   — the thing that would have caught B1–B3.

Exit criterion: `pip install git+https://github.com/AyarLabs/stateye` succeeds in a
fresh venv and `python -m pytest tests/` passes.

---

## 3. Phase 1 — the bridge: photonflux waveform → stateye measurement (1 day)

One module, `stateye_sources/photonflux_link.py`, with a narrow surface:

```python
def measure(p_thru_mW, dt_sec, baud, *, s_noise_mW, ber=1e-12, m1=0.0, m2=0.0,
            ref_rx_bw_factor=0.75, ref_rx_order=4, ref_rx_bw_hz=None,
            nx=512, ny=2048) -> dict:
    """Optical through-port power [mW] on a uniform dt grid -> stateye measurements.

    ref_rx_bw_factor : reference-receiver -3 dB bandwidth as a multiple of the
        baud rate (0.75 -> 39.84 GHz at 53.125 GBd). Set None to bypass.
    ref_rx_bw_hz : absolute -3 dB bandwidth [Hz]; overrides ref_rx_bw_factor.
    """
```

It applies the reference receiver (4.2), builds an `IdealEye(datarate_gbps=baud/1e9,
dt_sec=dt)`, sets the four TDEC parameters, calls `add_data(p, "mW")`, and returns
`get_measurements()` alongside `get_measurement_counts()`.

Four details that are easy to get wrong and each cost a silent `nan`:

- **`sampling_offset_mode` must stay `"half_ui"`.** TDEC's 0.4/0.6 UI histogram windows
  are referenced to the crossing, and `compute_tdec_r` says as much in its own comment.
  Note the stateye README states adaptive is the default; `ideal_eye.py:16` shows the
  default is actually `"half_ui"`. Set it explicitly rather than relying on either.
- **Use a full-period PRBS-13, and expect two of the `_8180` metrics to be `nan`.**
  The pattern is the IEEE 802.3 PRBS13 polynomial `x^13 + x^12 + x^2 + x + 1`, all
  **8191 bits**. Per the measurements in section 0, that populates `oma_8180`,
  `tdec_8180` and `extinction_ratio_8180` (count 15), but leaves `dcd_8180` and
  `rise_time_*_8180` at `nan`, because those go through `OMA_FILTER_MAP["8180"]`, which
  demands a contiguous `0^7 · v · 1^8` window that PRBS-13 contains zero times. Take the
  edge-rate and DCD diagnostics from the `_4140` variants instead. Do not truncate the
  pattern: a 511-bit prefix has no run of 8 ones at all. The PRBS-7 used by
  `examples/ring_mod_sky130.py` today (max run 7) cannot produce any `_8180` metric.
- **Always check `get_measurement_counts()`.** Gate the objective on a minimum
  `oma_8180` count (15 for full PRBS-13) and raise rather than silently optimizing
  against a `nan`. Note the counts dict does not track `tdec_*` or `inner_eye_height`
  at all — they are histogram-derived, and their counts always read 0. Gate on
  `oma_8180`, never on `tdec_8180`.
- **Feed absolute mW, not normalized power.** `extinction_ratio_8180` came back `nan` in
  my synthetic check precisely because the zero level was 0 (`log10(p1/p0)`). The ring's
  overcoupled dip floor gives a real nonzero P0 (0.0898 mW measured above), so this is
  fine for the real link — but it is a trap if anyone normalizes the trace first.

### Prerequisite: photonflux cannot generate PRBS-13 today

`photonflux/signals.py:prbs()` is a Fibonacci LFSR hard-coded to a **two-tap**
trinomial `x^n + x^k + 1`, with `_PRBS_TAPS = {5, 6, 7, 9, 11, 15, 23, 31}`. Order 13 is
absent and cannot simply be added, because **degree 13 has no primitive trinomial** —
the standard PRBS13 is the four-term `x^13 + x^12 + x^2 + x + 1` from IEEE 802.3
(the same pattern 802.3bs/cd specify for TDECQ, which is why it is the right choice
here).

The fix is small: generalize the tap table to a list of exponents and XOR over all of
them, keeping the existing left-shift convention so the existing orders reproduce
bit-for-bit.

```python
_PRBS_TAPS = {7: [7, 6], 9: [9, 5], 11: [11, 9], 13: [13, 12, 2, 1], 15: [15, 14], ...}
fb = 0
for t in taps:
    fb ^= (reg >> (t - 1)) & 1
reg = ((reg << 1) | fb) & ((1 << order) - 1)
```

Add a maximal-length assertion to the tests (`sum(bits) == 2**(n-1)` over one period)
— that is what caught a wrong shift direction while checking this.

### The objective: OMA − TDEC

stateye reports `oma_*` and `tdec_*` separately; the composite is not implemented.
Define it in the bridge, matching the IEEE spec line ("OMA minus TDEC", in dBm):

```python
oma_tdec_dbm = 10*np.log10(msmts["oma_8180"]) - msmts["tdec_8180"]   # oma in mW -> dBm
```

**Optimize this, not TDEC alone.** Minimizing TDEC by itself has a degenerate optimum:
park the laser far off resonance, get a tiny but beautifully clean swing, and TDEC → 0
with an unusable link. OMA − TDEC is the quantity that actually carries the transmitter
budget, and it is what makes the gap and lock-point axes non-trivial. Report bare TDEC
as a diagnostic and as a spec constraint, not as the thing being minimized.

#### The OMA cancels — and the metric saturates

Found by running the first real sweep, where several unrelated designs returned
*identical* scores to three decimals. Substituting stateye's definition:

```
TDEC        = 10·log10( (OMA/2) / (Qinv(BER)·R) )
OMA − TDEC  = 10·log10( 2 · Qinv(BER) · R )
```

**The OMA divides out entirely.** OMA − TDEC depends only on `R`, the noise a receiver
could still add and hit the target BER. That is not a bug — it is exactly what the IEEE
spec line is for, a single transmitter-quality number in units of tolerable receiver
noise — but it means two designs with very different OMA and TDEC scoring the same are
genuinely equally good, not accidentally equal.

The consequence that *does* bite: stateye solves
`R = (1−M1)·sqrt(N² + S² − M2²)` (`histogram_analysis.py:214`), where `N` comes from the
eye histogram. `R` therefore has a hard floor at `S`, so as the eye closes `N → 0` and

```
OMA − TDEC  →  10·log10( 2 · Qinv(BER) · S )   =   −8.517 dBm   for S = 0.01 mW, BER 1e-12
```

which is precisely the plateau observed. **Every closed-eye design scores identically,
so the optimizer sees a flat region with no gradient.** In the 7×7 sweep, 6 of 49 points
sat on it.

Two things follow. First, `S` is not a cosmetic input: it sets the floor and therefore
how much of the design space is distinguishable, so it has to match the real reference
receiver rather than be picked for convenience. Second, saturation must be *detected* —
`photonflux.tdec` returns `oma_tdec_floor_dbm` and an `at_floor` flag, and the example
reports what fraction of evaluations landed there and warns above 25 %.

---

## 4. Phase 2 — physics photonflux needs before the example is honest (2–3 days)

### 4.1 Bus gap → κ², which does not exist today

`models/optical_field/ring_mod.va:71` is parameterized by `kappa2` (bus **power**
coupling), not by a gap. The user-facing knob has to be a gap in nm, so a coupler model
is required. Add `photonflux/coupler.py` (pure Python — this is a design-space map, not
device physics that belongs in the solver):

```
kappa(g)  = kappa0 * exp(-(g - g0) / g_d)          # evanescent overlap
kappa2(g) = kappa(g)**2
```

with `g_d` ≈ 100–130 nm for a 220 nm SOI strip at 1310 nm. Ship it as a **calibration
dataclass with defaults and a `from_points()` fitter**, so it can be replaced by an
FDTD/Lumerical sweep or measured data without touching the example.

Be explicit in the docstring that **this fit is the dominant source of absolute error**
in the whole study. Relative comparisons across gaps are trustworthy; the mapping from
an optimal κ² back to a mask-layer gap in nm is only as good as the calibration. Also
note the second-order effects the model drops: gap changes coupler excess loss and
slightly loads the resonance, and the point-coupler idealization gets worse as the gap
closes.

Sanity anchor from the measured baseline: κ² = 0.100 against a critical-coupling value
of 0.076, so the device ships mildly overcoupled — the gap axis will want to explore
both sides of critical, and the objective is not unimodal there (undercoupled gives high
ER and low OMA; overcoupled gives the reverse).

### 4.2 The scipy Bessel reference receiver — configurable, and TDEC is wrong without it

TDEC is defined through a reference receiver with a **fourth-order Bessel–Thomson
response**, conventionally at **0.75 × the signaling rate** — here ≈ 39.8 GHz. stateye
does not apply it (it measures whatever you hand it), and photonflux does not have it.
Without it the optimizer will happily reward ringing and overshoot that a compliant
receiver would filter away.

Implement it in the bridge with scipy, and expose the bandwidth as a user knob rather
than burning 0.75 in:

```python
from scipy import signal

def reference_receiver(p, dt, baud, *, bw_factor=0.75, order=4, bw_hz=None):
    """Order-N Bessel-Thomson reference receiver, -3 dB at bw_factor * baud.

    bw_hz overrides bw_factor when given. bw_factor=None bypasses the filter.
    """
    if bw_factor is None and bw_hz is None:
        return p
    fc = bw_hz if bw_hz is not None else bw_factor * baud
    sos = signal.bessel(order, fc, "low", analog=False, output="sos",
                        fs=1.0 / dt, norm="mag")
    return signal.sosfilt(sos, p)
```

Three choices in there that are load-bearing:

- **`norm="mag"`, not scipy's default `norm="phase"`.** Only `"mag"` makes `Wn` the
  −3 dB point, which is how a reference receiver is specified. Verified in section 0:
  −0.16 % across `bw_factor` 0.5–1.0. With the default `"phase"` the realized bandwidth
  is a different number and every TDEC in the study would be quietly mis-referenced.
- **`sosfilt`, not `sosfiltfilt`.** A reference receiver is causal; zero-phase filtering
  would remove the phase distortion that is precisely what TDEC is meant to capture,
  understating ISI. Discard the first few UI to let the filter's transient settle, and
  note the ~12 ps group delay is absorbed by stateye's CDR lock.
- **Digital, not analog + FFT.** At 53.125 GBd with 32–40 samples/UI the sample rate is
  ~1.7–2.1 THz, so `fc` sits below 4 % of Nyquist and bilinear warping is negligible;
  `sos` avoids the circular-wrap artifact of the FFT approach used in stateye's own
  `tests/sample_signals/generate_signals.py:filter_waveform`.

Surface `bw_factor` and `order` all the way up to the example's CLI (`--ref-rx-bw 0.75
--ref-rx-order 4`, `--ref-rx-bw-hz` for an absolute override, `--no-ref-rx` to bypass),
since sweeping the reference bandwidth is itself a useful study — the table in section 0
is exactly that sweep, and it doubles as the sensitivity analysis for how much of the
reported TDEC is the reference receiver rather than the transmitter.

If the filter should also be visible *inside* the circuit rather than applied in post,
photonflux's `webapp/lti.py` + `webapp/vf.py` can vector-fit it into a state-space
component.

Confirm the exact clause and bandwidth factor against the standard being targeted before
publishing compliance numbers — 0.75 × baud is the clause-95-family convention, and I
have not verified which clause governs a 53.125 GBd NRZ link. Making it a parameter
rather than a constant is partly insurance against that.

### 4.3 Junction model fidelity — two idealizations that bias the (W, L) answer

`models/optical_field/ring_phase_shifter.va` implements:

```
lambda_res(V) = lambda_res_nm + dl_dv_pm * V     # linear
I(vp,vn) <+ cj * ddt(V(vp,vn)) + V(vp,vn)/rleak  # cj constant
```

A real depletion junction is neither. Δλ goes as `(V_bi + V)^m` with m ≈ 0.3–0.5, and
`C_j` goes as `(V_bi + V)^-m`. Both matter *specifically* for this study:

- Constant `C_j` makes the driver's load voltage-independent, so the optimizer's
  W-sizing answer is biased — the real load is heaviest exactly at the low-reverse-bias
  end of the swing.
- Linear Δλ(V) overstates modulation efficiency at large reverse bias, which flatters
  OMA and therefore flatters the lock point that chases it.

Fix: add `vbi` and `m` parameters to `ring_phase_shifter.va` (and the `ring_mod.va`
instance override), defaulting to the present linear/constant behaviour so existing
tests and the `tests/test_ring_decomposition.py` machine-precision pin do not move.
This is a contained change and worth doing before the optimization, not after.

---

## 5. Phase 3 — the example (3–4 days)

`stateye/examples/mrm_sky130_tdec_53g/`, built by lifting the already-working structure
of photonflux's `examples/ring_mod_sky130.py`.

### Design space

| Parameter | Range | Type | Cost to change |
|---|---|---|---|
| MRM bus gap `g` | 150–350 nm → κ² ≈ 0.03–0.20 | continuous | free (VA runtime param) |
| Lock point `λ_laser − λ_res` | −150 … +150 pm (linewidth is 255 pm) | continuous | free |
| Inverter `W_p` | 5–60 µm | discrete (layout grid) | **41–80 s per new value** |
| Inverter `W_n` | 3–40 µm | discrete | **41–80 s per new value** |
| Inverter `L` | {0.15, 0.18, 0.25} µm | discrete (SKY130 bins) | **41–80 s per new value** |

`L` must be a small discrete set, not a continuous axis: SKY130 BSIM4 cards are
**binned** by W/L, so `cx.sky130_fet` extracts a different card per geometry — there is
no meaningful interpolation between them, and every distinct pair is a fresh ngspice
`showmod` + BSIM4→OSDI compile.

### Optimizer architecture — follow the measured cost structure

Gradient descent is off the table, and for a sharper reason than photonflux's
`webapp/optimize.py` already gives ("no JAX gradient exists through a recompile"):
**the objective itself is non-differentiable**, because TDEC is computed inside
stateye's NumPy/Cython histogram code. `jax.grad` through the circuit is irrelevant when
the last mile is a Cython 2D histogram. So:

- **Outer loop, discrete `(W_p, W_n, L)`** — enumerate a coarse grid (say 4 × 4 × 3 = 48
  pairs) once. At 41–80 s per *new* pair that is **≈ 30–60 min, paid once**; every later
  hit is 0.00 s from the content-hash cache. Precompile the whole grid in a warmup pass
  before the optimizer starts, so the inner loop never stalls on a compile.
- **Inner loop, continuous `(gap, lock point)`** — bounded Nelder–Mead, reusing the
  existing `webapp/optimize.py` pattern. Free to re-evaluate.
- **Two-tier scoring, and this is what makes PRBS-13 affordable.** Search on
  **PRBS-9 + `tdec_4140`** (511 bits, ~2.7 s), then re-score only the surviving
  candidates on **full PRBS-13 + `tdec_8180`** (8191 bits, ~43 s). Section 0 measures
  the surrogate error at **< 0.01 dB** across the usable range and 0.034 dB at a badly
  closed eye — far below the differences the optimizer is resolving. Every *reported*
  number comes from PRBS-13; the surrogate only ranks.
- **One long-lived process.** This is where the 11.2 s fixed cost is won or lost:
  in-process a PRBS-9 evaluation is ~2.7 s, as a subprocess ~14 s. Do not shell out per
  evaluation.
- **Parallelize the outer loop.** Each `(W, L)` point is independent and the FET cache
  is on disk and shared, so the outer sweep is embarrassingly parallel across processes.

### Runtime budget

At the measured 6.6 ms/bit/40-sps — i.e. **5.3 ms per bit at 32 samples/UI**:

Measured end-to-end, in-process, at 32 samples/UI (the sweep below is a real run, not a
projection):

| Stage | Pattern | Per eval | Count | Wall time |
|---|---|---|---|---|
| `(W, L)` grid warmup (48 pairs) | — | 41–80 s | 48 | 30–60 min, once |
| Inner Nelder–Mead per `(W, L)` point | PRBS-9, 511 b | **6.8 s** | ~40 | ~4.5 min |
| Full 48-point outer sweep | PRBS-9 | — | ~1900 | **~3.6 h** |
| Final rescore of top candidates | **PRBS-13, 8191 b** | **35 s** | ~5 | ~3 min |
| **Total** | | | | **~4 h** |

The measured 6.8 s/eval (49-point sweep in 332 s) is higher than the 2.7 s the transient
alone would cost, because `simulate()` rebuilds and re-JITs the circuit on every
evaluation — changing a Verilog-A *setting* does not recompile the model, but it does
retrace the netlist. That is the obvious next optimization: hoisting the settings into
traced arrays so one compiled circuit serves the whole inner loop would take the search
back toward the transient-limited ~2.7 s.

For contrast, running full PRBS-13 in the inner loop is 48 × 40 × 35 s ≈ **19 h** — the
surrogate is worth about 15 hours. The other order-of-magnitude saving is process reuse:
one subprocess per evaluation adds ~11 s of interpreter and JIT startup each time.

Sanity-check the surrogate once per study rather than trusting the table above blindly:
rescore ~20 random points on PRBS-13 and confirm the ranking is unchanged.

### Deliverables — landed as one script with three modes

[`examples/mrm_tdec_sky130.py`](../examples/mrm_tdec_sky130.py):

- `single` — one operating point: the full metric set on full PRBS-13 plus the stateye
  eye diagram to `out/mrm_tdec_eye.png`.
- `sweep` — the gap × lock-point contour of OMA − TDEC on the surrogate, to
  `out/mrm_tdec_sweep.png`.
- `optimize` — the two-tier search: bounded Nelder–Mead over (gap, lock) inside the
  discrete (W_p, W_n, L) grid, survivors rescored on full PRBS-13, results to
  `out/mrm_tdec_results.csv` (which carries both the surrogate and PRBS-13 scores so
  the surrogate error is auditable after the fact).

Reference-receiver bandwidth and order, the TDEC `S`/BER, the pattern orders and the
whole search box are CLI flags. Reproducing any of it needs the photonflux toolchain
(`openvaf-ir` + SKY130 PDK) or the repo's `Dockerfile` — not just `pip install`.

---

## 6. What else belongs in this, beyond the four requested knobs

Ranked by how much they change the answer.

1. **A drive-strength constraint on the input of the inverter.** The current testbench
   drives the inverter from an *ideal* voltage source with a fixed 7 ps edge. Under that
   model, **W has no cost and the optimizer will run it to the upper bound** — the
   answer will be "as wide as allowed", which is not a design. Either fix a predriver of
   given strength (so W is traded against the load it presents backwards), or constrain
   the ratio to a fanout-of-4 chain. Without this the (W_p, W_n) axes are meaningless.
   This is the single most important addition on this list, and it is **still open** —
   `examples/mrm_tdec_sky130.py` inherits the ideal source, so treat its (W_p, W_n)
   result as an upper bound on what sizing can buy, not as a sizing recommendation.

2. **Energy per bit as a co-objective.** `E ≈ C_j·V_DD²·activity`; at the measured
   `C_j = 23.6 fF` and 1.8 V that is ~76 fJ per full swing before the driver's own
   switching energy. Report a Pareto front rather than a single OMA − TDEC optimum —
   the widest inverter will always win on TDEC alone.

3. **Duty-cycle distortion, which couples directly to W_p/W_n.** The inverter's trip
   point is set by the P/N ratio; trip-point offset becomes DCD, and DCD closes the eye
   horizontally. This makes the W_p/W_n *ratio* a genuine optimization axis (edge rate
   vs. trip symmetry) rather than a scaling factor, and it is the most interesting
   coupling in the whole problem. Read it from **`dcd_4140`, not `dcd_8180`** — the
   latter is `nan` under PRBS-13 (section 0), and it is the one metric where the pattern
   choice costs you something real.

4. **Laser RIN and receiver noise.** TDEC's `S` parameter is the O/E + scope noise in
   y-units; it has to be chosen deliberately since it directly scales the result. Add
   RIN as a vertical-noise kernel via `eye.add_vertical_noise()`, and TX jitter via
   `eye.add_random_jitter()` / `add_flat_bounded_jitter()` — stateye convolves both into
   the histogram, so they cost nothing per evaluation.

5. **Thermal lock stability.** `models/optical_field/ring_selfheat.va` exists and
   demonstrates thermo-optic bistability. With ~1 mW in a Q≈5100 ring this is not
   negligible, and it constrains the lock point in a way pure CMT does not: with
   dn/dT > 0 the red-detuned side is the thermally self-stabilizing lock, while maximum
   modulation slope often sits on the blue side. If the optimizer picks a blue lock
   point, flag whether it is thermally holdable — otherwise the result is unbuildable.

6. **Process corners and temperature.** `cx.sky130_fet(..., corner=...)` already
   supports this. A tt-only optimum is not a design; at minimum re-evaluate the winner
   at ss/ff and 0/85 °C. Note each corner is a fresh card extraction (~40–80 s per
   geometry per corner), so budget the grid accordingly.

7. **Extinction ratio as a constraint, not an objective.** OMA − TDEC alone does not
   pin ER, and the baseline already sits at 6.5 dB. Add a floor so the optimizer cannot
   trade ER away for OMA.

8. **Statistical convergence — and note that reseeding is not a convergence test here.**
   A full PRBS-13 period gives `oma_8180` a count of 15: adequate, not large. But every
   nonzero LFSR seed produces a *cyclic rotation of the same m-sequence*, so over a full
   period all seeds carry identical run statistics and will return essentially identical
   TDEC. Reseeding proves nothing. The variance that actually matters comes from
   elsewhere: the RIN and jitter kernel realizations, the eye grid resolution
   (`nx`/`ny`), and the CDR's chosen sampling phase. Sweep those. If a genuinely
   independent pattern is wanted, change the polynomial (e.g. a second primitive
   degree-13 generator) rather than the seed.

9. **A replay-based regression test for stateye CI.** Check in one recorded
   `(t, P_thru)` waveform as `.npz` and assert stateye reproduces its TDEC/OMA. This
   keeps the example under test in an environment that has no openvaf, no PDK, and no
   JAX — which is what stateye's CI will actually be.

10. **A driver-topology axis, given the measured 16 ps edge vs 19 ps UI.** If the
    optimization confirms the driver is the binding constraint, sizing alone will not
    close it. The levers worth opening up: the existing `--driver two-stage` and
    `single-neut` flavors in `examples/_drivers.py`, C_neut as a continuous parameter
    (it is already one), and CMOS pre-emphasis / a segmented electrode. Worth scoping as
    a follow-on rather than folding into the first example.

---

## 7. Risks and open questions

- **The gap → κ² calibration is unvalidated** (4.1). Relative trends are safe; absolute
  gap numbers are only as good as the fit. Highest-value thing to replace with real data.
- **The reference-receiver bandwidth factor** (4.2) is asserted from the clause-95
  convention; confirm against the governing standard for 53.125 GBd NRZ before quoting
  compliance numbers.
- **photonflux cannot be a pip-installable dependency in the full sense** (§1). The
  extra installs the Python package; the openvaf/PDK toolchain still needs
  `scripts/build-openvaf.sh` + volare, or the container.
- **stateye v1.7 is ~2 years stale** and its CI evidently does not do a clean install.
  Phase 0 should land as its own PR upstream before any of this depends on it.
- **`main` in photonflux moves fast.** Tag before depending.
- The linear-Δλ / constant-C_j idealization (4.3) biases the W-sizing result. If Phase 2
  gets cut for time, say so explicitly in the example's README rather than presenting
  the (W, L) optimum as physical.
- **The PRBS-9 surrogate was validated on a synthetic Bessel-filtered NRZ waveform**,
  not on the real ring + driver, whose distortion is nonlinear and pattern-dependent in
  a way a lowpass is not. The < 0.01 dB agreement is strong but it is not proof. Run the
  20-point rescore check (section 5) before relying on the ranking, and fall back to
  full PRBS-13 in the inner loop — at ~34 h, parallelized across the outer grid — if the
  ranking moves.

---

## 8. Sequence

| Phase | Work | Effort | Blocks |
|---|---|---|---|
| 0 | Fix stateye packaging (B1–B5), upstream PR | ½ day | everything |
| 1 | Bridge module + OMA−TDEC + PRBS-13 generator + replay test | 1 day | Phase 0 |
| 2 | gap→κ² model, scipy Bessel reference receiver, junction law | 2–3 days | Phase 1 |
| 3 | Optimizer + report + README | 3–4 days | Phase 2 |
| 4 | Corners, energy Pareto, thermal check | 2 days | Phase 3 |

Phase 0 and Phase 2 are independent and can run in parallel; Phase 2 is photonflux-side
work and does not touch stateye at all.
