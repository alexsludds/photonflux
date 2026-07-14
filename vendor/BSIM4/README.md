# BSIM4.8 Verilog-A (vendored)

`bsim4.va` is the cogenda **VA-BSIM48** port of BSIM4 version 4.8
(https://github.com/cogenda/VA-BSIM48; see `readme` and `LICENSE` alongside).
It supplies the device physics for `photonflux.cx.sky130_fet(...)`: the
SKY130 model card (extracted from the volare PDK via ngspice `showmod`) is
applied to this model, compiled to OSDI with `bin/openvaf-ir`, and evaluated
natively inside circulax.

Notes:

* The port marks "parameter not given" with in-band sentinels (`-12345789`
  for reals) and resolves defaults at runtime — the OSDI NaN-not-given
  mechanism drives that ladder exactly like ngspice's model setup.
* Compile this file with `bin/openvaf-ir` (ChipFlow openvaf fork), **not**
  `bin/openvaf-r` — the latter miscompiles it to OSDI (segfault in
  setup_model).
