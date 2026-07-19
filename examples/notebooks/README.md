# Analysis notebooks

Jupyter companions to the web-app testbenches. Each notebook pulls a built-in
example through the notebook bridge (`photonflux.nb`, see `webapp/README.md`),
runs it **server-side** — the same engine as the browser's Run button — and
pins the results against theory or pen-and-paper analysis in numpy.

They need the web app running (`python webapp/server.py`, or set
`PHOTONFLUX_URL`) and a kernel with `numpy` + `matplotlib` — no JAX in the
notebook process. Every run passes its schematic explicitly, so nothing here
touches the schematic mirrored from your browser canvas.

| notebook | testbench(es) | what gets pinned |
|---|---|---|
| [`01_sky130_fet_characterization`](01_sky130_fet_characterization.ipynb) | 04–07 | Id–Vds families, Vth (max-gm), subthreshold swing, the gm/Id design chart, ro/λ, f_T vs bias from \|h21\| — real BSIM4 physics, extraction saved to `out/fet_extraction.json` |
| [`02_eo_comb_analysis`](02_eo_comb_analysis.ipynb) | 42 (EO comb) | comb spacing = f_RF, every line vs an independent CMT integration (<1.5 dB), comb bandwidth saturating at the photon-lifetime limit f_cav |
| [`03_fwm_analysis`](03_fwm_analysis.ipynb) | 39 + 40 (FWM) | idler at 2ν_p−ν_s, pump-slope 2 with the absolute (γP_pL_nl)² prefactor, ring CMT conversion, ×~3000 resonant enhancement over the same length of straight guide |
| [`04_vernier_design_space`](04_vernier_design_space.ipynb) | 33 (Vernier) | FSRs, Vernier FSR, interstitial suppression vs the zero-fit Airy product, and the Δm / κ² design charts (FSR-extension vs rejection, linewidth vs loss) |
| [`05_diff_pair_hand_analysis`](05_diff_pair_hand_analysis.ipynb) | 23 (+ 04) | gm/Id-method gain prediction (few %), square-law transfer overlay and where it cracks, ±R_D·I_T plateaus |
| [`06_fabry_perot_airy`](06_fabry_perot_airy.ipynb) | 35 (Fabry-Perot) | full-sweep Airy overlay (<1 dB RMS), FSR/finesse/peak-T/×10 buildup across a mirror-R design sweep |
| [`07_cmos_inverter_ring_osc`](07_cmos_inverter_ring_osc.ipynb) | 26 + 29 | square-law calibration, V_M switching-threshold formula to mV across a skew sweep, ring-oscillator frequency: four-constant ODE to ~1 % (and why the CΔV/I napkin is ×2 fast) |

Suggested order: `01` first (it writes the extraction that `05` reuses —
though `05` falls back to re-measuring inline), everything else standalone.
`notebook_live_bench.ipynb` one directory up is the *interactive* tour of the
bridge itself (live canvas mirror, `watch()`, the Builder).

Each notebook ends with a **Things to try** cell — edits to make in the
browser or from Python that extend the analysis. Assertions with generous
tolerances run throughout, so `jupyter execute` (or nbclient) doubles as a
physics regression suite for the testbenches.
