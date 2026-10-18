"""Shared helpers for the IBIS-AMI examples (not a runnable example)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "out"
sys.path.insert(0, str(REPO))

from photonflux import ami_build, ibisami  # noqa: E402


def model(name: str, **params) -> ibisami.AmiModel:
    """Library model by name (built on first use), parameters applied."""
    m = ibisami.load(name)
    if params:
        m.params.update(params)
    return m


def ensure_built() -> None:
    steps = ami_build.build_all(quiet=True)
    for s in steps:
        if not s.ok:
            print(f"  note: {s.name}: {s.detail}")


def density_of(sigma_in: float, ts: float) -> float:
    """White noise of rms ``sigma_in`` per sample (bandwidth 1/(2 ts)) as a
    density in V/rtHz, so the statistical flow sees what the time-domain
    flow adds."""
    return sigma_in * np.sqrt(2.0 * ts)


def save(fig, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=130)
    print(f"-> {path.relative_to(REPO)}")
    return path


def fmt_ber(x: float) -> str:
    return "0" if x <= 0 else f"{x:.2e}"


def plot_counted(ax, sigma, results, label, marker="o", **kw):
    """Counted BER points; zero-error runs drawn as upper bounds (< 1/N)."""
    sigma = np.asarray(sigma) * 1e3
    ber = np.array([r.counted["ber"] for r in results])
    bound = np.array([1.0 / max(r.counted["bits"], 1) for r in results])
    hit = ber > 0
    line = ax.loglog(sigma[hit], ber[hit], marker, ls="none", label=label, **kw)
    if (~hit).any():
        ax.loglog(sigma[~hit], bound[~hit], "v", ls="none",
                  color=line[0].get_color(), mfc="none",
                  label=f"{label}: 0 errors (< 1/N)")
