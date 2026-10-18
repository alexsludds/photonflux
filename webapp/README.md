# photonflux web UI — schematic editor + simulator in the browser

Drag-and-drop photonic/electronic schematic capture with circulax (JAX)
simulation behind a Run button. No build step, no extra Python dependencies —
the server is pure stdlib, the frontend is vanilla JS + SVG with a vendored
[uPlot](https://github.com/leeoniya/uPlot) for waveforms.

```bash
.venv-circulax/bin/python webapp/server.py             # http://localhost:8642
.venv-circulax/bin/python webapp/server.py --no-reload # single plain process
```

Static assets (`static/`) and example JSON are read from disk on every
request, so editing them is already live — just refresh the page. Editing the
Python engine (`simulate.py`, `catalog.py`, the `photonflux` package) would
otherwise need a restart, so **auto-reload is on by default**: the server
supervises a child and restarts it whenever a `.py` source file changes, and a
browser refresh always shows the latest code. Pass `--no-reload` (or set
`PHOTONFLUX_RELOAD=0`) to run a single plain process with no supervisor — the
container image does this, so production runs exactly one process.

![architecture] frontend (static/) --JSON--> server.py --> simulate.py --> circulax

## Using the editor

* **Place**: click a part in the left palette, then click the canvas
  (Shift-click to place several; `R` rotates the ghost, `F` flips it,
  `Shift+F` flips vertically; `Esc` cancels).
* **Wire**: click a port (circle), then click the destination port.
  Optical ports/wires are amber, electrical are blue; domain mismatches are
  rejected (bridge domains with a photodiode/modulator, ground is exempt).
* **Orient**: a selected component can be rotated (`R` or the inspector
  ⟲ / ⟳ buttons) and mirrored (`F` / `Shift+F` or the ↔ / ↕ buttons) — the
  four rotations and a flip give all eight orientations, so signal flow can
  run whichever way keeps the wires clean (e.g. a mirrored splitter becomes
  an MZI combiner). Reference designators and pin labels always stay upright.
* **Edit**: select a component — the right panel shows its parameters with
  SPICE-style suffix input (`4n`, `1.5u`, `2k`, `3meg`). The headline value is
  drawn under the reference designator.
* **Probe**: `P` (or the palette Probe tool), then click a port. Probes are
  what gets recorded and plotted: optical probes plot |E|² in mW, electrical
  probes plot volts. Selecting a probe offers a **Hide from plots** toggle —
  the probe stays in the netlist (so AC `in_<x>`/`out_<x>` pairing and
  operating-point math are unaffected) but its traces are left out of the
  plots, the operating-point table, and CSV export. Toggling it re-renders the
  cached results instantly, with no re-run.
* **Optical spectrum probe**: an optical probe has an extra **Optical
  spectrum** toggle. With it on, a transient adds an OSA-style plot beside
  the time-domain traces — |FFT(E)|² in dB versus wavelength (nm), centred
  on the reference optical frequency (Welch-averaged for a smooth envelope).
  It shows every carrier present on that node and its modulation sidebands (a
  ring modulator's chirp shows as asymmetric sidebands, an MZM's as symmetric
  ones). A coherent envelope carries a full baseband band, so several
  wavelengths coexist as distinct tones — see **multi-carrier WDM** below.
  Example 18's bus probe has it on and shows all four DWDM carriers at once.
  With the toggle on, two **Window start** / **Window stop** fields set the
  time span the FFT runs over (SI suffixes like `10n` accepted). They default
  to the full simulation; shorten the window to isolate a settled span (e.g.
  skip the turn-on transient) at the cost of coarser resolution.
* **Multi-carrier WDM**: a CW laser has a **WDM reference** field
  (`ref_wavelength_nm`). Leave it 0 for an ordinary single-carrier bus
  (constant envelope at the laser wavelength). Set it to a shared reference
  and the laser instead emits its true tone in that common baseband frame,
  `E(t) = √P · exp(j·2π·f_off·t)` with `f_off = c·(1/ref − 1/λ)`, so several
  lasers at different wavelengths become distinct tones that coexist and beat
  on one bus. Wavelength-selective devices referenced to the same frame
  (`lambda_nm = ref`) — the microring modulators and the add-drop filters —
  see each tone at its own offset and act only on the one on resonance. That is a
  genuine DWDM simulation in a **single coherent solve** (example 18), not a
  per-channel superposition. The only cost is bandwidth: the solver must
  resolve the whole grid, so a 200 GHz × 4-channel bus needs sub-picosecond
  steps (fine sampling, ~16k points).
* **Analyses** (toolbar): transient (adaptive Diffrax PID or fixed-step BDF2 —
  auto-selects BDF2 for SKY130/OSDI devices, stiff lasers/injection rings and
  pattern-driven circuits), DC operating point, a vectorized DC sweep of any
  instance parameter (with an optional stepped second parameter for curve
  families, SPICE `.step` style; instance `* (all)` sweeps the parameter in
  lockstep on *every* instance that has it — pick `wavelength_nm` to trace
  the spectral response of an interferometer, examples 15/16), an AC
  S-parameter sweep, a small-signal
  **noise** analysis, a **pulse/COM** analysis, and an **optimizer** (all
  described below). AC pairs probes by name (`in_<x>` drives, `out_<x>`
  responds) and reports |S21| plus |h21| — the 0 dB crossing of |h21| is f_T,
  logged per pair. Electrical-only circuits compile as real-valued systems;
  AC/noise are not available for optical circuits yet.
  * *Fixture-free f_T*: place the `in_<x>`/`out_<x>` probes on the terminals
    of the DC **sources** that bias the device (gate source and drain source,
    connected directly — no bias resistor or choke needed). The analysis then
    uses the SPICE `V… AC 1` idiom: the gate's source is driven with 1 V AC,
    every other source is an AC short (so the drain's source is the
    short-circuit ammeter), and h21 is the ratio of the two sources' branch
    currents — the pure device current gain at every frequency, with nothing
    in the fixture to influence the response. Probes on plain circuit nodes
    fall back to the z0-terminated S-parameter method (useful for amplifier
    transfer functions, but the measured two-port then includes whatever bias
    network hangs on the probed nodes).
* **AC parameter sweep**: the AC toolbar has an optional `sweep instance /
  param / values` — it overlays the |h21| family across those values (shaded
  by value) and logs f_T for each, so you can watch bandwidth track a device
  parameter (e.g. sweep a SKY130 NFET's `w_um`). Runtime params (R, ideal-FET
  W, …) and rebuild params (SKY130 `w_um`/`l_um`, baked into the OSDI model)
  are handled uniformly by recompiling the circuit per value through the
  cached path. A first-time SKY130 `w_um`/`l_um` sweep batch-extracts all its
  new BSIM4 cards in a *single* volare library parse (`sky130_cards.py`) —
  seconds, not one slow parse per width — and thereafter re-runs in seconds.
  Example 08 sweeps NFET width.
* **Plots**: drag left-right on any plot to zoom the x-axis; the per-plot
  **reset** button (or a double-click) restores the full range. Each plot's
  y-axis (and x, where valid) switches between linear / log / dBm, the results
  panel is resizable, and cursors are synced across plots.

## Testbench flows

A **flow** is a testbench's pre-configured chain of analyses, run as one job
(the Flow card at the top of the Run panel; saved with the testbench as
`doc["flow"]` next to `doc["analysis"]`). Switch the card to **Flow** and the
Analysis / Measure / Vary cards edit the *selected step*; **Single analysis**
brings the tab's own analysis back. Steps share **flow variables**:

* **vars** (the `vars` button): knobs with a starting value
  (`detune_pm = 1.5`), and the **design mapping** every step applies
  (`PN_A.length_m = pn_um*1e-6`, `PRBS.v0 = vbias - vswing/2`).
* **Analysis step**: any single-run analysis (DC, DC sweep, transient, AC,
  noise, pulse, with sweeps and fx expressions) plus
  *set for this step* (`*.wavelength_nm = lam_res - detune_pm*1e-3`; `*`
  writes every instance that has the parameter), *outputs* (`lam_res =
  xat_min(thru)`, `rlm = rlm(thru)`) that become variables, and *checks*
  (`notch_db < -3`). A DC sweep step can carry `"warm_start": true` (examples
  48/49's resonance steps do): every point starts from the converged
  operating point at the first sweep value instead of from zero, which cuts a
  large circuit's sweep ~4x. It is opt-in because on a circuit with several DC
  solutions (a latch, a bistable ring) the starting point picks the branch.
  Output expressions see the probes, `x`/`t`, the
  variables, the fx vocabulary plus `xat_min`, `xat_max`, `cross`, and the
  stateye figures of merit `rlm`, `er_db`, `tdecq`, `tdec`, `oma_dbm`,
  `rel_oma_db(out, in)`, `il_db(out, in)` (scored exactly like the Eye tab's
  Measure button, with its settings).
* **Optimize step**: searches variables and/or `INST.param` ranges,
  re-running the chosen *inner* steps per evaluation (so a resonance found by
  step 1 is re-found for every candidate). The objective is an output name or
  expression; constraints (`rlm >= 0.96`) are penalties (100x the relative
  shortfall, in objective units). The optimum carries into later steps.
  `method` picks the search:
  * `bayes` (new steps, examples 48/49): Bayesian optimization with BoTorch.
    The start design and a Sobol space-filling set, then one design per
    evaluation chosen by qLogNEI. Each argument of a top-level `min(...)` /
    `max(...)` objective and each constraint slack gets its own GP, and the
    acquisition samples the min/max of those -- the worst-margin objective is
    kinked, its terms are smooth. Needs `pip install -e '.[bo]'`.
  * `nelder-mead` (steps saved without a `method`): a bounded simplex; repeated
    points (a reflection clipped onto a bound) are not re-simulated. Needs far
    more evaluations than `bayes` once there are more than ~3 knobs.
* **Evaluate step**: re-runs the inner steps at the current design (the
  optimum) with the transient stretched over a **full PRBS period** at
  16 samples/UI, so TDECQ uses OMA<sub>outer</sub>, and scores the built-in
  figures of merit against specs: relOMA = OMA<sub>outer</sub> / P<sub>in</sub>
  (P<sub>in</sub> = mean power at the *mod in* probe), OMA [dBm], IL =
  P<sub>in</sub> / top level, ER, TDECQ (TDEC for NRZ), RLM.

**Run flow** (Cmd+Enter) runs every step; the caret runs from the selected
step or just that step (Shift / Alt + Cmd+Enter), reusing the last run's
variables. The progress bar names the stage (`step 3/4, eval 12/50, <step title>`) and
Stop cancels the whole flow. The **Flow** results tab shows the step chain,
a mini plot per step (sweep with located points marked, eye, optimizer
history), the scorecard (value the earlier steps measured vs. the evaluated
one), the variables and every check; **Apply design to schematic** writes the
final design (flow mapping + optimum + the evaluated step's set) as one
undoable edit, and **Export report** downloads it as Markdown. The `step`
select in the results bar picks which step the Waveforms / Eye / Link tabs
show. Flows run at one process corner.

Headless: `python -m photonflux flow webapp/examples/48_cmr_flow_106g.json`
(exit 1 when a check fails, so saved testbenches double as regression tests)
or `Session().flow()` in a notebook (a `FlowResult` with `.vars`,
`.scorecard()`, `.step(id)`, `.design`).

Example 48 is the worked case: the coupling-modulated ring of example 47 at
106.25 GBd PAM4, optimized over laser detuning, drive bias, arm imbalance, PN
length and ring length against RLM > 0.95, TDECQ < 1.5 dB, relOMA > -2 dB,
ER > 3.5 dB. Its device assumptions are flow variables (see the canvas note):
with lumped junctions driven through 50 Ohm the junction RC caps it well
short of the spec (relOMA -4.9 dB, TDECQ 2.4 dB; scored through a 5-tap FFE
with S = 0.005 mW, before the flows switched to the 802.3 reference settings
and S = 0). Example 49 runs the flow on traveling-wave PN arms with real
optical transit (`transit` = delayed), a 10 um / 0.3 dB/cm feedback delay
line, 30 Ohm electrodes driven through 30 Ohm, and the input coupler and
the PRBS source's TX FFE pre/post-cursor taps as optimizer knobs. Before
the 30 Ohm / FFE change (50 Ohm drive, both couplers as knobs) it scored
relOMA -5.6 dB, TDECQ 2.3 dB, RLM 0.92, ER 1.9 dB -- all miss; the ring's
photon lifetime (~6 UI) limits. With quasi-static optics that design had
scored near relOMA -3.6 dB / TDECQ 1.4 dB.

## Serial-link (SERDES) workflow

* **IBIS-AMI Tx / Rx** (examples 50–52): the **IBIS-AMI Tx** source runs
  its pattern through a Tx model's `AMI_GetWave` when the circuit compiles;
  the **IBIS-AMI Rx** block is a unity buffer whose Link-tab report runs the
  Rx model on the "BER vs" probe: `AMI_Init` on the channel estimated from
  the probe (statistical eye, BER, bathtubs; a nonlinear residual such as
  square-law detection after dispersion is counted as distortion) and
  `AMI_GetWave` on the waveform (its own CDR, counted BER, adaptation
  trace). `Parameters` overrides the `.ami` defaults (`name=value`). See
  [docs/ibis-ami.md](../docs/ibis-ami.md).

* **PRBS / pattern source**: PRBS7/9/11/15/23/31 (same LFSR taps as the
  companion time-domain serdes codebase), NRZ or Gray-coded PAM4, raised-
  cosine edges, 3-tap TX FFE (`ffe_pre`/`ffe_post`: side taps relative
  to the main cursor, linear and signed), RLM predistortion
  for a quadrature-biased MZM (set `rlm_vpi`), edge jitter (RJ Gaussian
  rms, SJ sinusoidal peak at `sj_freq`, DJ dual-Dirac peak-to-peak, DCD),
  and a single-pulse mode. `sequence` picks the pattern: `prbs`;
  `prbs+oma` writes a top-level then bottom-level run over the PRBS every
  512 UI (PAM4 >= 7 threes / 6 zeros, NRZ >= 8 / 8), so TDECQ's OMA_outer
  and TDEC's OMA_8180 are measurable on records far shorter than a PRBS
  period; `square` (top/bottom runs of `run_ui`); `staircase` (PAM4
  0,1,2,3,2,1 for RLM). Coherent QAM drive is a separate **QAM Source**
  (one per I/Q rail). The waveform is precomputed and baked at
  compile time (parameter edits recompile, seconds). A **PWL source** plays
  arbitrary `t v` breakpoints (paste or Load CSV) — the simplest way to
  replay waveforms from another simulator.
* **Eye tab**: folds any transient probe at a UI (prefilled from the PRBS
  source), scope-style persistence render, with levels (1-D k-means), per-eye
  height, and width (guard-banded clear phase span). Multi-seed noise
  families fold together.
  Its **stateye** row scores whatever eye is shown (`POST /api/eyemeasure`,
  `webapp/eyemeasure.py`): PAM4 -> IEEE TDECQ, raw and through the reference
  equalizer; NRZ -> TDEC and OMA - TDEC. Settings: reference-receiver
  bandwidth (x baud, or off) and order, FFE taps and pre-cursors, DFE taps,
  how the taps adapt (noise-aware MMSE block solve -- what P802.3dj D3.1+
  specifies --, normalized LMS with step mu and passes, TDECQ-optimal
  Nelder-Mead on TDECQ itself -- the Clause 121 definition, slow, with live
  "evaluation i/n" progress -- or manual taps), the P802.3dj D3.2 Table
  180-16 reference-equalizer limits (w(0) >= 0.9 - b(1), the pre/post
  difference including the DFE tap, 0 <= b <= 0.33; on by default at
  200G/lane -- 802.3cd sets none at 100G/lane -- the search stays inside
  them, MMSE/LMS taps are projected into them), training
  on the PRBS source's pattern or decision-directed,
  target SER/BER, scope noise S, and the histogram grid. FFE taps,
  pre-cursors, DFE taps and target SER default to "auto": the lane rate's
  standard -- 802.3dj at 200G/lane (15 taps, 3 pre, 1 DFE, SER 4.56e-4),
  802.3cd at 100G/lane (5 taps, 1 pre, no DFE, SER 4.8e-4). S defaults to
  0: 802.3 credits scope noise back to the transmitter, and a simulated
  trace has none. "show scored eye"
  draws the eye stateye measured (after the reference receiver and FFE,
  time-aligned to the probe).
* **Link tab / BER report**: pick a received probe in the Link tab's
  "BER vs" select. Alignment and error counting are data-aided
  against the PRBS source's known sequence: best sampling phase + lag by
  correlation, optional RX **FFE/DFE** — configured by dropping **Rx FFE** /
  **Rx DFE** blocks on the receive path (each carries a tap count and an
  adaptation rate: 0 = one-shot least-squares/Wiener, > 0 = normalized-LMS
  adaptation) — counted BER/SER, per-eye Gaussian **Q-fit BER**, and a
  **bathtub** curve. Multi-seed noise runs pool their statistics.
* **Pulse / COM analysis**: reruns the PRBS source in single-pulse mode,
  extracts the symbol-spaced pulse response at a probe, designs a Wiener
  FFE(+DFE) and reports an ISI-limited FOM = 20 log10(half eye / sigma_ISI)
  (half eye = cursor/(2(L-1)), sigma_ISI the RMS residual ISI of random NRZ or
  PAM4 data) with the tap sets and the equalized pulse — a "best achievable EQ" oracle for
  any schematic link.
* **Transient noise**: set `noise seeds` (+ bandwidth) in the transient
  toolbar to enable photodiode **shot noise** (2qI on the instantaneous
  current), laser **RIN** (set `rin_db` < 0 on a CW laser, dB/Hz), and TIA
  input-referred noise. N seeds re-run the compiled circuit (one compile,
  N solves); the eye and BER report pool all seeds. The noise PSD equals the
  set density at low frequency and rolls off as sinc^4 (-1.8 dB at half the
  noise bandwidth), so set the bandwidth to at least twice the receiver's.
* **Noise analysis (small-signal)**: output-referred V/sqrt(Hz) at a probe
  via one adjoint solve per frequency; sources are resistor thermal 4kT/R
  (incl. PDK resistors), TIA `in_noise`, and diode shot at the operating
  point, with a per-source breakdown and integrated Vrms. (SKY130 FET
  channel noise is not modelled yet; optical chains: use transient seeds.)
* **Optimize**: Nelder-Mead over up to 8 instance parameters
  (`INST.param=min:max, ...`), objective `expr:<name>` (a scalar from the
  fx panel), `eye:<probe>`, `ber:<probe>` (minimized) or `fom`; runs the
  inner transient per evaluation (compile-time parameters recompile through
  the caches), reports the optimum + finite-difference sensitivities, and
  can apply the result back to the schematic. Derivative-free by design:
  eye/BER objectives are noisy and rebuild-parameters have no gradient
  through a recompile.
* **Process corners**: the top-bar **Corner** menu sets the SKY130 corner
  (`tt`/`ss`/`ff`/`sf`/`fs`) for every FET. `all` runs any plot-producing
  analysis once per corner and overlays the runs (it composes with the
  parameter-sweep pane), and makes Optimize corner-robust: each design is
  scored by its worst corner (or the mean, per the optimize `corners`
  select), the per-corner values are reported, and the final plots show the
  worst corner.
* **Expressions (fx button)**: derived traces evaluated server-side per
  run — `iph [A] = (1.8 - vout)/500`, `gain_db [dB] = db(vout/vin)`,
  `spectrum = spec(vout)` (extra log-f plot), scalars (`vpp = pk2pk(vout)`)
  land in the log and feed the optimizer. Whitelisted-AST evaluation.

## Passive integrated optics

All passives work on the coherent field (S-matrix -> Y-matrix, matched and
reflection-free), and their `wavelength_nm` is a *live* runtime parameter:
run a DC sweep with instance `* (all)` over `wavelength_nm` and every
waveguide/grating on the canvas sweeps in lockstep — the vectorized DC solve
traces a 2001-point spectrum in about a second.

* **Waveguide**: first-order dispersion via the group index
  (`n(lam) = neff + (neff - n_group)(lam - lam_c)/lam_c`) plus dB/cm loss.
  Interferometer FSR comes out as lam^2/(n_group dL) — the group index, not
  n_eff, which is the standard trap. `transit` = quasi-static (default)
  applies the transmission instantly; `delayed` adds the real group delay
  L n_g / c in both directions (Pade all-pass stages of <= 0.5 ps, group
  delay within ~1 % to 60 GHz; past 64 stages, ~2.4 mm, the stages grow
  and the run log warns), so a ring closed by such guides has its round
  trip and photon lifetime (`tests/test_transit.py`). Identical at DC;
  about 3x the transient cost in example 49.
* **Y-splitter**: lossless 1x2, `split_ratio` to p2. It is reciprocal —
  wire a second one in reverse (feed p2/p3, take p1) and it is the coherent
  combiner of an MZI; power missing at destructive interference is the
  junction's radiated mode.
* **Directional coupler**: 2x2 beamsplitter, inputs p1/p2 left, outputs
  p3/p4 right; bar sqrt(1-c), cross j sqrt(c) (unitary).
* **Grating coupler**: Gaussian passband — insertion loss grows
  quadratically with detuning from its center wavelength.
* **Tunable filter**: a flat-top (Butterworth) optical bandpass that acts
  on the coherent field — the passband is mapped to the baseband envelope
  around the laser carrier and realised as a compile-time state-space
  filter, so it filters the *modulation sidebands*, not just the carrier
  amplitude. `bandwidth_nm` is the -3 dB FWHM and `order` sets the flatness
  (1 = one-pole, 3-5 = boxy). Because it is a real filter: a bandwidth much
  wider than the signal passes it unchanged; one comparable to the signal
  rings and adds ISI; and one narrower than the signal strips the modulation
  (a 1 pm filter turns a modulated channel into a near-flat carrier and the
  eye closes). Tune `center_nm` relative to `lambda_nm` (the carrier) to
  select or reject a channel off the bus.
* **Optical terminator**: matched absorber. Terminate unused coupler ports
  and detection points — an *open* optical port reflects like an open
  transmission line stub.

Example 15 builds an imbalanced MZI (dL = 214.6 um) from
splitter + two waveguides + reversed splitter and sweeps 1304-1316 nm:
fringes with exactly the analytic FSR (2.00 nm) and >50 dB nulls.
Example 16 closes the loop — literally — with a ring add-drop filter from
two directional couplers and two 100 um half-rings: Lorentzian through-
notches and drop-peaks at FSR = 2.145 nm; the drop peak (0.980 mW) matches
the textbook add-drop formula to four decimals. Both are pure structural
compositions — no dedicated MZI/ring component, just the building blocks
solved through the same complex Newton DC as everything else.

Example 18 is a **DWDM co-packaged-optics microring-modulator (CPO MRM)
link**: four lasers on a 200 GHz O-band grid (≈ 1308.28 / 1309.43 / 1310.57
/ 1311.72 nm — the 1.145 nm that 200 GHz works out to at 1310 nm) share one
baseband reference frame (`ref_wavelength_nm = 1310`, the grid centre), so
each is a distinct tone; their phases are staggered because real WDM lasers
are mutually incoherent (equal phases would coherently align all four tones
at t = 0 into an unphysical 16× start-up pulse). A 4→1 combiner tree merges
them onto one bus, and four microring modulators sit *in series on that
shared waveguide* (combiner -> ring1 -> ring2 -> ring3 -> ring4). Each ring
is resonance-aligned to one channel and modulates only its own carrier; the
other three, 200 GHz (~9 linewidths) away for the high-Q rings
(`kappa2 = 0.03`, FWHM ~120 pm), pass by untouched. **All four carriers are
modulated in one coherent solve** — this is the multi-carrier WDM path, not
a per-channel superposition.

The ring bias is chosen the way a real depletion MRM is driven: **the
resonance never crosses the carrier**. Parked at `lambda_res_nm = λ_k +
100 pm` (red flank) and driven bit-0 = −2 V / bit-1 = 0 V through the
45 pm/V depletion shifter, the detuning swings +100 pm (transmit) to +10 pm
(extinguish) — a monotonic transfer with ~8 dB ER. (Biasing inside the
swing, so the resonance sweeps *through* the line on every edge, punches a
transient notch into each transition — a classic MRM design error the
simulator happily reproduces if you ask for it.)

The receiver is the classic DWDM demux: a **cascade of four tunable add-drop
filters, each dropping its channel onto its own photodiode**. Each
`opt_filter` is a 3-port flat-top (Butterworth, 200 pm, order 3) add-drop:
`pin` in, `drop` = the channel it is tuned to (`center_nm = λ_k`), `thru` =
everything it did *not* drop (E_thru = E_in − E_drop), wired `thru -> next
pin`. Three views per run:

* **The bus spectrum** — the bus-output probe has the optical-spectrum toggle
  on: four carriers exactly on the 200 GHz grid, each with its own PRBS data
  sidebands (rings driven by decorrelated PRBS 7/9/11/15 at 10 Gb/s). This is
  what a real OSA would see at the modulator-bank output — one OSA trace from
  one simulation.
* **Per-channel optics** — every drop port carries an optical-spectrum probe:
  a time-domain optical-power trace (that channel's NRZ data) plus the
  drop-port spectrum (one isolated carrier; the other three rejected below
  −60 dB).
* **Per-channel electronics** — each drop feeds a photodiode + load
  (`vout1..4`), so all four channels' received eyes come out of the single
  solve side by side.

`python webapp/tools/wdm_spectrum.py` runs the example once and renders the bus
spectrum plus each channel's drop spectrum and received waveform.

A note on the modelling: coherent-field envelopes are baseband, so putting
several wavelengths on one node means representing each as its own tone in a
*common* reference frame (that is what a laser's `ref_wavelength_nm` does).
The ring modulators are coupled-mode-theory ODEs, so — being frequency-
selective in the time domain — each interacts only with the tone on its
resonance while passing the rest, exactly as a real ring does. The price is
temporal resolution: the solver must not only *sample* the whole optical
grid (Nyquist above the outermost carrier, else it aliases inward) but also
*integrate* it accurately — BDF2's per-step amplitude error on a tone at
offset f grows like (2πf·Δt)², so an under-resolved outer carrier is
silently damped and the channels tilt. Example 18 pins the step with
`dtmax = 50 fs` (ω·Δt ≈ 0.09 at ±300 GHz → <2 % tilt); the simulator logs a
warning if a WDM run's step is too coarse for its grid.

Example 19 turns the same machinery into a **spectral-crosstalk
demonstration**: a *signal* channel at 1310.0 nm and a 3 dB-hotter
*aggressor* 100 GHz away (1310.5727 nm, half of example 18's grid), each
modulated by its own microring with decorrelated PRBS-7/PRBS-9 at 10 Gb/s on
one bus. A 50/50 tap feeds the same bus to two receivers **both tuned to the
signal channel** — one behind the example-18 demux filter (order 3, 200 pm:
the aggressor lands ~43 dB down) and one behind a deliberately weak filter
(order 1, 2 nm FWHM: the aggressor comes through at ~−1 dB, i.e. *above* the
signal, since it runs hot). One solve, side by side: `vout_clean` is an open
10 Gb/s NRZ eye, while `vout_leaky` is corrupted beyond recovery — the
aggressor's leaked intensity modulation lands in-band on the photodiode and
its carrier beats against the signal at the 100 GHz spacing (partially
averaged by the PD RC). The two drop-port spectrum probes show why at a
glance: the same aggressor carrier is absent from one drop and at full
strength in the other.

## Channels

* **Copper channel**: parametric sqrt(f) skin-effect loss (dB at f_nyq)
  with minimum phase, vector-fitted at compile time to a compact real
  state-space (fit error logged).
* **S2P channel**: upload a Touchstone .s2p (inspector button); the
  matched-termination insertion transfer S21/2 is vector-fitted; the input
  presents a z0 shunt.
* **Fiber (dispersion)**: chromatic dispersion on the coherent field —
  exp(-j (beta2/2 w^2 + beta3/6 w^3) L) plus attenuation, vector-fitted
  with free complex poles over `fit_bw` and a causal transit latency
  covering the worst in-band group delay (the impulse response of pure CD
  is non-causal without it). **C-band**: set `D_ps` (and optionally the
  slope `S_ps`). **O-band / near the zero-dispersion wavelength**: set
  `lambda0_nm` > 0 — D(lambda) then follows the G.652 Sellmeier profile
  S0/4 (l - l0^4/l^3) with `S_ps` read as S0 (0.092 default), and the
  beta3 slope term keeps the model honest where D ~ 0 (beta3 ~ 0.08
  ps^3/km at 1310, matching SMF-28). The fit target is extended past the
  band edge with a flat-|H|, frozen-group-delay continuation so the
  nearly-flat zero-dispersion response cannot fit to a degenerate huge-
  feed-through solution. Validated to <1% against FFT pulse propagation at
  1550, 1311 and exactly at lambda0. Example 12 shows 25 Gb/s closing over
  20 km at 1550 — and that a linear RX FFE cannot undo dispersion after
  square-law detection, which is real physics, not a bug; example 17 runs
  the *same* link at 1311 nm where near-zero dispersion re-opens the eye.
* **Laser & modulator models (Verilog-A)**: besides the ideal CW laser and
  field-convention MZM there are five compiled-from-VA device models:
  `laser_dml` (static L-I with first-order response), `laser_rate` (full
  single-mode rate equations — turn-on delay, relaxation oscillations,
  pattern-dependent ringing; see example 09), `mzm_tw` (traveling-wave MZM
  whose EO bandwidth is set by electrode loss — one or two poles — and
  velocity walk-off), `mzm_seg` (three binary-weighted electrode segments =
  optical DAC, drive two of them for PAM4) and `ring_mod_inj` (carrier-
  injection microring: forward-diode drive, resonance blue-shifts with the
  lifetime-filtered current — tau_c caps the bandwidth unless you drive it
  with the PRBS source's pre-emphasis — plus free-carrier absorption; see
  example 14). The lasers, mzm_tw and mzm_seg are power-domain models
  bridged with sqrt(P)/|E|^2 adapters (optical phase is not carried
  through); ring_mod_inj is coherent-field like ring_mod. Circuits with
  `laser_rate` automatically get the fixed-step BDF2 transient solver and a
  pseudo-transient DC settle (lasing turn-on is a bifurcation plain Newton
  cannot cross); injection rings and mzm_tw default to BDF2 too (mzm_tw's
  distributed electrode adds ~4.5 ps ddt poles). Example 42 is the mzm_tw
  browser twin: a 50 Gb/s NRZ PRBS drives the traveling-wave MZM at quadrature
  and a photodiode detects it, so the electrode-loss pole (f_el = 35 GHz) and
  velocity-walk-off pole (f_w ~ 18 GHz for the default n_rf=2.4 vs n_opt=4.2
  electrode) close the optical and received eyes (EO -3 dB ~ 15 GHz). Set
  `MZM.n_rf = 4.2` to velocity-match: the walk-off pole vanishes, only f_el
  limits (-3 dB ~ 35 GHz), and the eye reopens — the same EO bandwidth
  `examples/mzm_tw_transient.py` measures as a rise time and
  `examples/mzm_tw_eo_bw.py` measures as an |H(f)| network-analyser sweep.
* **SOA + cavity building blocks (Verilog-A, directed waves)**: `soa`
  (bidirectional Agrawal-Olsson gain reservoir with bias pins and an ASE
  seed — hard-DC like laser_rate, so a cavity closed around it lases in DC
  sweeps and transients), `ase_src` (in-line broadband ASE injector: with
  transient noise seeds >= 1 it adds complex white field noise at a set
  dBm/Hz, so a laser cavity starts from NOISE and its mode choice is
  emergent; a pass-through otherwise), `wmirror` (2x2 partial reflector on
  directed waves), `ring_comb` (add-drop ring with a FIVE-mode resonance
  comb and a resistive heater) and `ring_nl`/`wg_nl` (high-Q ring /
  waveguide segment with two-photon + free-carrier absorption and Kerr/FCD
  shifts). These carry directed waves: optical inputs read, outputs drive —
  wire outputs to inputs and terminate dark inputs / unused outputs.
  Example 36 is a Fabry-Perot laser (SOA between two wmirrors, bias stepped
  through threshold); example 37 is the noise-seeded Vernier ring laser —
  comb candidates rise out of the ASE floor, compete for the shared gain,
  the aligned pair wins, and a mid-run heater step hops the lasing line one
  29.8 GHz FSR (x17 lever) with the classic power dip while the modes swap;
  example 38 is the TPA/FCA-limited high-Q ring against its beta_tpa = 0
  twin; example 39 is chi(3) four-wave mixing in `wg_nl` — two WDM lasers
  10 GHz apart beat through the instantaneous Kerr phase and the idlers
  emerge on the OSA at exactly 2f_p - f_s and 2f_s - f_p with the textbook
  T*(k*P_p)^2*P_s efficiency (the full scaling proof lives in
  `examples/wg_fwm.py`); example 40 moves the chi(3) INSIDE a resonator —
  `ring_kerr` (models/optical_field/ring_kerr.va, five FSR-spaced modes sharing the Kerr
  nonlinearity, the modal Lugiato-Lefever equations) pumped by two lasers
  exactly one FSR apart: the idler emerges in the next resonance over,
  ~1000x the conversion of the same length of straight waveguide, and the
  comb spreads to modes +-2 (`examples/ring_fwm.py` pins the physics);
  example 41 is `ring_selfheat` (models/optical_field/ring_selfheat.va, an
  all-pass ring with a one-pole thermal reservoir) driven by a `vpwl` source
  that ramps the ring's electrical `lam` node — the laser wavelength — across
  the cold resonance and back in one transient: the through-port traces a
  thermo-optic HYSTERESIS loop (the heated resonance locks to the laser on the
  way to the red, then snaps back; cold on the way to the blue), the classic
  bistability fingerprint (`examples/ring_selfheat.py` pins the physics);
  example 42 is the ring-modulator **EO frequency comb** — `ring_mod`
  (models/optical_field/ring_mod.va) with a CW laser parked on its resonance
  slope while a strong 10 GHz `vsin` drives the depletion electrode: sweeping
  the resonance across the laser grows a comb on the through-port spectrum
  probe, spaced by the drive and shaped (bandwidth-limited) by the ring's
  photon lifetime (`examples/eo_comb.py` pins every tooth to the ring CMT and
  plots the bandwidth-vs-f_RF rolloff); example 47 is a **coupling-modulated
  ring** built from parts — two 50/50 `dir_coupler`s whose arms carry
  push-pull `pn_phase_shifter`s (depletion PN junction: sqrt-law n_eff and
  loss, nonlinear junction charge) form an MZI coupler closed by
  `waveguide`s. The junctions change the coupling, not the round-trip phase,
  so the resonance stays put while a 53.125 GBd PAM4 PRBS (2 Vpp, series
  push-pull) rides the under-coupled flank: the arm phase bias and laser
  detuning are chosen for evenly spaced levels (RLM 0.98, ER 4 dB), and the
  ring's Q ~ 80k sets the catch — RLM >= 0.94 only within +-0.5 pm of
  drift. The optics are quasi-static (steady-state waveguides), so the eye
  shows level placement, not photon-lifetime bandwidth
  (`tests/test_pn_phase_shifter.py` pins it); set `transit` = delayed on
  the ring's parts for the real round trip, as example 49 does.
* **PN phase shifter, traveling-wave** (`pn_phase_shifter_tw`, its own
  long symbol with the electrode strip): the lumped part's junction physics
  distributed under a transmission-line electrode. Drive `e_in` (at the `p1`
  end), terminate `e_out` into the bias, connect `ref` to the junction's
  other side (`polarity` +1: the electrode is the n side). The electrode is
  an RLC ladder (unloaded `z0_ohm`, `n_rf`, flat conductor loss) loaded
  section by section by the junction -- `rs_ohm_mm / length` in series with
  the depletion charge -- so the loaded impedance (~40 Ohm for the defaults
  at 2 V), the RF slow-down (n ~ 3.6) and the junction-RC line loss come out
  of the physics. Each section's optical phase reaches `p2` after its optical
  transit (a Pade all-pass chain at `n_group`), so RF/optical walk-off is
  modelled rather than approximated by a pole. `n_sections` 0 picks one
  section per <= 100 um (accurate to ~60 GHz). At DC it equals the lumped
  part; `tests/test_pn_phase_shifter_tw.py` pins the line impedance and delay
  and the walk-off. `transit` = delayed carries the field itself through
  every section (instead of only the phase change), so the part adds its
  full group delay, as Waveguide does. Example 49 is example 48 with
  traveling-wave arms and real optical transit.
* **Bring your own .va**: the palette's "Upload .va" button compiles any
  Verilog-A file through openvaf/bosdi into a placeable JAX component
  (ports from the module, parameters with their defaults). Uploads persist
  in `webapp/models_user/`. All ports are electrical; bridge power-domain
  optical ports with the Field->Power / Power->Field adapter components.
  Platform limits (clearly diagnosed at compile): `absdelay()` is not
  supported (model delay with poles or the vector-fitted LTI blocks);
  `white_noise()`/`flicker_noise()` are dropped by the lowering — use the
  webapp's transient-noise seeds / noise-analysis sources instead.
* **SKY130 devices**: eight FET flavors (1.8 V std/LVT/HVT, 5 V, 5 V native —
  real BSIM4.8 via OSDI), poly/diffusion/precision resistors and MiM caps.
  Passive R/C values are *measured from the real PDK by ngspice* (DC current
  for R, 1 MHz AC current for C) and cached in
  `models/__jax__/sky130_passives.json`; FET model cards are extracted with
  `showmod` exactly like `cx.sky130_fet`. First use of any new geometry parses
  the sky130 library (about a minute) — after that it's instant.
* `Cmd+Z`/`Shift+Cmd+Z` undo/redo, `Del` delete, `Cmd+Enter` run,
  `Cmd+S`/Save downloads the circuit JSON, Load restores it. The schematic
  autosaves to localStorage; the Examples menu has 34 ready-made circuits
  grouped by topic — photonic links, lasers & modulators, SKY130 device
  studies, channels & equalization, passive photonics, WDM & crosstalk, and
  standard analog (RLC, rectifier, CS amp, diff pair, Sallen-Key, envelope
  detector) and digital (inverter VTC, NAND2, SR latch, ring oscillator,
  T-gate mux, Schmitt trigger) electronics.

## Notebook bridge (photonflux.nb)

The server keeps a **live mirror of the active tab** (`/api/schematic`): the
editor debounce-pushes it on every edit, and notebook clients read it, edit
it, and subscribe to changes (`/api/schematic/events`, SSE) — so a Jupyter
kernel and the canvas are two views of one document. The browser applies
notebook pushes through the normal undo stack (`Cmd+Z` reverts them), and
notebook parameter writes are revision-checked so they never clobber a racing
canvas edit.

```python
from photonflux.nb import Session, Builder

s = Session()                        # http://127.0.0.1:8642
s.pull()                             # the live schematic (browser's active tab)
s["CPL.coupling"] = 0.088            # canvas updates live, undoably
res = s.dcsweep("*", "wavelength_nm", 1304, 1316, points=2001)
res.plot()                           # numpy traces, matplotlib

for sch in s.watch():                # yields on every canvas edit —
    ...                              # a live analysis bench for the schematic
```

`s.run()` with no arguments mirrors the Run button (canvas schematic +
toolbar analysis); `transient/dc/dcsweep/ac/noise/pulse` helpers cover the
toolbar modes, all accepting SI suffixes ("20n", "50G"). Progress streams
from `/api/progress`; interrupting the kernel cancels the run server-side.
`Builder` composes schematics programmatically (an N-channel WDM bank is a
loop) and `s.push(builder)` drops the result on the canvas for hand-tidying.
Importing `photonflux.nb` needs only numpy — no JAX in the kernel; runs
execute server-side through the same caches as the Run button, serialized
behind it. `examples/notebook_live_bench.ipynb` is the guided tour, and
`examples/notebooks/` holds analysis notebooks that pin the built-in
testbenches against theory through this bridge (FET gm/Id + f_T extraction,
the EO comb vs CMT, FWM scalings, the Vernier and Fabry-Perot design
spaces, diff-pair and ring-oscillator hand analysis).

The mirror is one process-global document — right for the single-user local
server, wrong for a shared host (it would leak schematics between visitors),
so like VA upload it has a deployment knob: `PHOTONFLUX_ENABLE_BRIDGE=0`
disables the three endpoints (the container image does).

## How it maps onto circulax

The frontend posts `{schematic: {instances, wires, probes}, analysis}` to
`/api/run`. `simulate.py` turns wires into nets (union-find), probes into
circulax `ports`, expands composites (the microring modulator becomes
field→re/im adapter + `cx.va("ring_mod")` + re/im→field adapter), bakes the
catalog defaults into every instance's settings, and calls
`circulax.compile_circuit(..., backend="dense", is_complex=True)`. Compiled
circuits are cached on the schematic hash, so re-running with different
analysis settings skips the JAX compile.

All numeric settings are coerced to float server-side: JSON drops the
int/float distinction, and an integer-traced JAX leaf would silently truncate
float sweep overrides.

Component catalog lives in `catalog.py` (ports, parameter metadata, domains);
symbol drawings live in `static/symbols.js` keyed by the same type ids. To add
a component: add a catalog entry + a models_map entry in `build_models()` + a
symbol glyph.

## Module map

The modules group by role:

```
Server core
  server.py          stdlib HTTP server (static + /api/components, /api/run,
                     /api/upload, /api/upload_va, /api/schematic; runs
                     serialized by a lock)
  session.py         live schematic mirror + SSE change feed (the notebook
                     bridge; client side is photonflux/nb.py)
  simulate.py        schematic JSON -> circulax netlist -> dc/dcsweep/
                     transient/ac/noise/pulse/optimize   (the hub)
  catalog.py         component catalog + models_map builder (noisy variants,
                     LTI blocks, user VA)
  warmup.py          build-time cache warm (lower every VA, compile the FETs)

Signals, waveforms & link post-processing
  wavesrc.py         PRBS/PWL waveform builders + noise banks
  linkpost.py        BER/Q/bathtub link report, RX LS-EQ, pulse/COM, eye metric
  exprs.py           derived-trace expression engine (whitelisted AST)
  optimize.py        Nelder-Mead parameter optimizer + FD sensitivities
  bayesopt.py        BoTorch Bayesian optimizer (flow Optimize, method bayes)
  flow.py            testbench flows: step chains sharing variables, the flow
                     Optimize / Evaluate steps, stateye figures of merit

LTI channel construction
  lti.py             channel / s2p / fiber-CD model construction
  vf.py              vector fitting + state-space realisation

SKY130 PDK extraction (via photonflux.cx + libngspice)
  sky130_cards.py    batch BSIM4 FET model-card pre-extraction
  sky130_passives.py real R / MiM-C values from the PDK

Tools (standalone, not part of the server)
  tools/wdm_spectrum.py   run example 18 and plot its optical spectra

Frontend & data
  static/            index.html, app.js (editor), flow.js (flow editor +
                     results), symbols.js (glyphs), style.css
  static/vendor/     uPlot 1.6.31 (vendored, no CDN needed)
  examples/*.json    example circuits (schematic + layout + analysis)
  models_user/       uploaded .va models (auto-loaded at startup)
  uploads/           uploaded data files (Touchstone)
```
