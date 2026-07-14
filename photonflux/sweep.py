"""Parallel parameter sweeps.

libngspice holds global state — one simulator per process — so parallel
sweep points each run in their own worker process. The worker function must
be importable (top-level in a module, not a closure), per multiprocessing
spawn semantics.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

P = TypeVar("P")
R = TypeVar("R")

__all__ = ["sweep"]


def sweep(
    worker: Callable[[P], R],
    points: Sequence[P] | Iterable[P],
    jobs: int | None = None,
) -> list[R]:
    """Run `worker(point)` for every point, in parallel worker processes.

    Results come back in input order. With jobs=1 the sweep runs serially
    in-process (handy for debugging — full tracebacks, one ngspice).
    """
    points = list(points)
    if jobs == 1 or len(points) <= 1:
        return [worker(p) for p in points]
    jobs = jobs or min(len(points), max(1, (os.cpu_count() or 2) - 2))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        return list(ex.map(worker, points))
