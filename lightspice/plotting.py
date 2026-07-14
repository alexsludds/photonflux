"""Matplotlib helpers for link plots: waveform stacks, eyes, BER curves."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["waveform_stack", "eye_plot", "ber_curve", "save"]


def waveform_stack(
    t: np.ndarray,
    panels: Sequence[tuple[str, Sequence[tuple[np.ndarray, str]]]],
    title: str | None = None,
    xunit: float = 1e-9,
    xlabel: str = "time [ns]",
):
    """Stacked, time-aligned waveform panels.

    panels = [(ylabel, [(array, label), ...]), ...]
    """
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.2 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (ylabel, traces) in zip(axes, panels):
        for arr, label in traces:
            ax.plot(t / xunit, arr, lw=0.9, label=label or None)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if any(lbl for _, lbl in traces):
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def eye_plot(ax, phase: np.ndarray, v: np.ndarray, ui: float, title: str = ""):
    ax.plot(phase * 1e12, v, ".", ms=0.6, alpha=0.5)
    ax.set_xlabel(f"phase [ps]  ({phase.max() / ui:.0f}-UI fold)")
    ax.set_ylabel("V")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)


def ber_curve(ax, p_dbm: np.ndarray, ber: np.ndarray, title: str = ""):
    ax.semilogy(p_dbm, np.clip(ber, 1e-16, 1), "o-")
    ax.axhline(4e-3, ls=":", c="r", lw=1, label="HD-FEC (4e-3)")
    ax.axhline(1e-9, ls=":", c="k", lw=1, label="1e-9")
    ax.invert_xaxis()
    ax.set_xlabel("P_rx ('1' level) [dBm]")
    ax.set_ylabel("BER")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def save(fig, path: str | Path, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    print(f"plot saved: {path}")
    return path
