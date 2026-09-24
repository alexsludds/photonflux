# Patches

## `stateye-modern-toolchain.patch`

Makes [AyarLabs/stateye](https://github.com/AyarLabs/stateye) v1.7 build and run on a
current Python toolchain. Without it, `photonflux.tdec` has no backend: upstream
stateye does not install at all on NumPy 2 / Cython 3, and its plotting is broken on
matplotlib ≥ 3.9.

Three independent breakages, each reproduced before patching (see
[`../stateye-integration-plan.md`](../stateye-integration-plan.md) §0):

| Fix | Symptom without it |
|---|---|
| add `pyproject.toml` | `pip install .` → `ModuleNotFoundError: No module named 'numpy'` — PEP 517 build isolation runs `setup.py`, whose line 2 imports numpy, in a clean env |
| `cnp.int_t` → `cnp.int64_t` (`utilities.pyx`) | `utilities.pyx:75:22: Invalid type.` — NumPy 2 removed `int_t` |
| `matplotlib.cm.get_cmap` → `matplotlib.colormaps[...]` (`eye.py`, 3 sites) | `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` on every `eye.plot()` |

### Applying it

```bash
git clone https://github.com/AyarLabs/stateye && cd stateye
git apply /path/to/photonflux/docs/patches/stateye-modern-toolchain.patch
pip install --no-build-isolation --no-deps .
```

`--no-deps` keeps stateye's `requirements.txt` from pulling `black` and `pre-commit`
into your environment — they are dev tools that upstream lists as runtime
`install_requires`. The actual runtime needs are `numpy scipy matplotlib h5py Pint
tqdm colorama`. `--no-build-isolation` needs `setuptools`, `wheel`, `Cython` and
`numpy` already present in the target environment.

Verified against Python 3.12.2 / NumPy 2.5.1 / Cython 3.2.9 / matplotlib 3.11.0,
coexisting with JAX 0.7.2 + circulax 0.2.1 in one venv.

### Caveats

* **Windows is untouched by this patch and probably still broken.** `dtype=int` is
  int32 there while `cnp.int64_t` is not, and the surrounding `cdef long` / `long[:]`
  declarations are 32-bit on MSVC. A portable fix would use `cnp.npy_intp` and match
  the `cdef long` declarations to it. Not attempted — untested here.
* This is a compatibility patch, not a feature change: no measurement behaviour is
  altered. It should go upstream as its own PR rather than living here.
