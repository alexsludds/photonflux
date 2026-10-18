# Vendored public IBIS-AMI kits

Third-party IBIS-AMI material redistributed here under its own license.
`python -m photonflux ami build` compiles it for the host into
`models/ibis_ami/build/public/` (gitignored); nothing here is modified.

| folder | provenance | license |
|---|---|---|
| `ibisami/` | [capn-freako/ibisami](https://github.com/capn-freako/ibisami) at commit `74555dd7445b38790d777f23491a39d16bfdc4df` (2025-02-13): `LICENSE`, `README.md`, `include/`, `src/`, and `example/example_{tx,rx}.{cpp,ami,ibs,exp}` | BSD-3-Clause, see `ibisami/LICENSE` |
| `sisoft_eval_v2.21/SiSoft_IBIS-AMI_Eval_Toolkit_v2.21.zip` | SiSoft IBIS-AMI Eval Toolkit v2.21 (2009-09-30), from the [IBIS ATM work archive](https://ibis.org/macromodel_wip/archive/20090930/toddwesterhoff/SiSoft%20IBIS-AMI%20Eval%20Toolkit%20v2.21/SiSoft_IBIS-AMI_Eval_Toolkit_v2.21.zip) via [IBIS-Library/IBIS-AMI-test-kits](https://github.com/IBIS-Library/IBIS-AMI-test-kits) at `5d02011`; SHA-256 `0d16a2188af327683ac419a85ef695af18499808fa9ceda3142028d9cd482d25` | SiSoft's permissive license (`license.txt` inside the ZIP): use, copy, modify and distribute with the notice, *with all original files of the kit*, so the ZIP is kept whole |

The build applies two small fixes, in `photonflux/ami_build.py` and not in
these files:

* SiSoft: `IBIS_AMI_Tx.h` lacks the `wr_ptr` field that the 2008
  `IBIS_AMI_Tx.c` uses (the original binary has it), so the build adds it.
* ibisami: `AmiRx::proc_imp` runs the CTLE over the impulse response in
  `AMI_Init` and keeps that filter state for `AMI_GetWave`, so the build
  clears it after Init.
