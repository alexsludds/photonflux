# Polarization modeling — design

photonflux was strictly scalar / single-polarization: every coherent optical
net was one complex field `E`, carried as an `Ereal`/`Eimag` node pair, with
`|E|² = power [W]`. This document describes the dual-polarization extension —
the Jones-vector field convention, the polarization-aware component set, how
scalar circuits stay a special case, and what is deliberately left to a later
change (webapp wiring, statistical/temporal PMD).

This is one of the two architectural items from the VPI/Lumerical gap analysis
(the other being multi-band signals). It ships the physics-model layer and its
testbenches; the browser catalog is a scoped follow-up (see *Roadmap*).

## 1. Field representation: a Jones vector per net

The scalar convention represents `E` as two real nodes. The natural extension
to polarization is a **Jones vector** `[Ex; Ey]` — two complex components — so
every coherent optical net now carries **two** `Ereal`/`Eimag` pairs:

| slot | nodes | meaning |
|---|---|---|
| X | `x*_re`, `x*_im` | TE component `Ex` |
| Y | `y*_re`, `y*_im` | TM component `Ey` |

with the total optical power `|Ex|² + |Ey|² = power [W]`. This keeps the
existing lowering unchanged: the Jones vector is just *two* of the same
real-node field pairs the solver already assembles (the real-2N complex
assembly, RR/RI/IR/II partials). No new node discipline, no per-net "mode
dimension" — a polarization-aware component simply declares four input field
nodes and four output field nodes and writes the 2×2 Jones matrix out as
real/imag contributions, exactly the way `mirror.va` writes its 2×2 scattering
matrix today.

### Why not a complex 2-vector node type

circulax nodes are scalar unknowns; the repo already carries complex fields as
real pairs and converts at the boundary (`cx.field_to_ri` / `cx.ri_to_field`).
Extending that to a Jones vector is four real nodes — mechanically identical to
what works today — whereas a first-class "vector node" would need new discipline
support in bosdi/circulax for no physical gain. The four-node convention is the
low-risk choice and is what the models below use.

### Convention: X = TE is the default polarization

The scalar models (`phase_shifter.va`, `mirror.va`, `directional_coupler.va`,
the rings, the SOA, …) are, unchanged, the **X/TE channel** of this convention.
A scalar circuit is the special case where the Y (TM) net is never populated:
any Y net a scalar component doesn't touch simply stays at 0. So:

* **existing examples/tests keep passing bit-for-bit** — nothing in the scalar
  path changed (the entire pre-existing suite is the regression test), and
* a polarization-aware and a scalar component compose on the **shared X pair**:
  feed a `birefringent_wg` X output into a scalar `phase_shifter` input and it
  just works; the TM channel rides alongside on the Y pair.

## 2. Component set (`models/optical_field/*.va`)

All are memoryless linear field transforms — a Jones matrix per element — so a
DC operating point gives the steady field (no transient needed). Each carries
the one-line reactive dummy node (`I(dmy,gnd) <+ 1e-12·ddt(V)+V`) that
`mirror.va` documents: a module with no `ddt()` on the optics needs it to keep
the reactive side dtype-honest for the complex assembly.

| model | Jones matrix | key params |
|---|---|---|
| `polarization_rotator.va` | real rotation `R(θ)` | `theta_deg`, `il_db` |
| `pbs.va` | routes X→port1, Y→port2 | `er_db`, `il_db` |
| `pbc.va` | X from port1, Y from port2 | `er_db`, `il_db` |
| `birefringent_wg.va` | `diag(e^{jφx}, e^{jφy})`, `φ=2πnL/λ` | `n_te`, `n_tm`, `length_um`, `lambda_nm`, `loss_db_m` |
| `pdl.va` | `diag(10^{-il/20}, 10^{-(il+pdl)/20})` | `il_db`, `pdl_db` |

The PBS/PBC use `lk = 10^{-er_db/20}` for the cross-leakage amplitude and
`tx = √(1-lk²)` so each port is power-normalized; `er_db → ∞` is the ideal
element. `birefringent_wg` gives the differential phase
`Δφ = 2π·Δn_eff·L/λ` (with `Δn_eff = n_te − n_tm`) — the static, deterministic
core of polarization-mode dispersion.

Polarization-dependent loss on the *existing* passives is expressed by dropping
a `pdl.va` in series (rather than duplicating every passive as a PDL variant);
that keeps the scalar models untouched.

## 3. Acceptance-criteria testbenches

Both live in `tests/test_polarization.py` (pinned) and `examples/pol_malus.py`
(a plotted study):

* **Malus' law** — a TE launch → `polarization_rotator(θ)` → `pbs` → `pbc`. The
  two PBS ports carry `cos²θ` and `sin²θ` of the input power (Malus), power is
  conserved at every split, and the PBC reconstructs the link with unit
  throughput. The solve matches analytics to machine precision (the example
  prints ~1e-16; the test asserts a conservative 1e-9).
* **Birefringent MZI** — a Jones MZI (a 50/50 `mirror` pair per polarization,
  a `birefringent_wg` in one arm, a balanced reference arm). The cross-port
  fringe is `cos²(π·n·L/λ)` per polarization, so the TE and TM transmission
  fringes are **offset in wavelength** by the modal birefringence `Δn_eff`;
  with `Δn_eff = 0` the two fringes coincide exactly. The per-polarization
  fringe matches the closed form to ~1e-13 (the test asserts 1e-9).

## 4. PMD and the fiber: why it belongs in the LTI path

True polarization-mode dispersion is a **differential group delay (DGD)**
between the two axes — a *time delay*, not just a static phase. The VA→JAX
lowering has no transport delay: `cx._check_va_support` rejects `absdelay()`
outright because the solvers keep no signal history. So a DGD element cannot be
a Verilog-A model; like chromatic dispersion (`webapp/lti.py::fiber_cd`) it must
be realized as a **vector-fitted state-space (pole/residue) block**.

The planned shape, mirroring `fiber_cd`:

* a `fiber_pmd` builder in `webapp/lti.py` that fits two frequency responses —
  one per principal state of polarization — with a differential group delay
  `Δτ` (mean DGD `⟨Δτ⟩ = PMD·√L`), plus a random Jones rotation `R(α,δ)` in
  front of them for the launch-to-PSP mapping;
* the fine-structure statistical model (a concatenation of many birefringent +
  random-rotation sections giving a Maxwellian DGD distribution) built from
  `birefringent_wg` + `polarization_rotator` sections, for studies that need
  the full second-order-PMD statistics rather than a single DGD.

`birefringent_wg.va` already provides the deterministic per-section
birefringence that the concatenation model is made of; the DGD *delay* and the
`fiber_cd` co-integration are the follow-up in the LTI path.

## 5. Roadmap (scoped follow-ups)

1. **Webapp catalog + wiring.** The browser represents an optical port as a
   single node mapped through the `_f2ri`/`_ri2f` adapters. A dual-pol port is
   two field pairs (X and Y), so it needs either a distinct `"optical-jones"`
   port domain (four nodes) or an X/Y port split, plus Jones adapters and the
   `app.js` domain-compatibility rule (scalar-optical ↔ Jones is the TE
   channel). Catalog entries for the five models above, wired like the existing
   `phase_shifter` composite, then follow.
2. **Fiber PMD** in `webapp/lti.py` (`fiber_pmd`), co-integrated with
   `fiber_cd`, per §4.
3. **PDL on the built-in passives** — either an optional `pdl_db` on the
   existing components or the drop-in `pdl.va` (shipped) in series.

The physics-model layer here is the foundation those build on; it is complete
and self-contained (models + testbenches + example) and changes nothing in the
scalar path.
