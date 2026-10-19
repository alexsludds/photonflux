"""Bayesian optimization for the flow Optimize step (BoTorch).

Why not Nelder-Mead for these flows: one evaluation is a whole inner chain
(tens of seconds), the budget is tens of evaluations over up to 10 knobs, and
the usual objective is a worst-case margin ``min(m1, ..., mk)``. A simplex
needs hundreds of evaluations in 8-D, and it stalls on the kinks where the
binding spec changes.

Here every objective *term* (each argument of a top-level ``min``/``max``)
and every constraint slack gets its own Gaussian process -- the terms are
smooth even where their min is not -- and the acquisition (qLogNEI) scores a
candidate by Monte-Carlo samples of the *composite*: the min/max of the
sampled terms minus the constraint penalty, i.e. the same J the flow ranks
evaluations by. The model therefore knows which spec binds where.

Budget: the start design plus scrambled-Sobol points, ``max(d + 1, 4)`` in
all (capped at a third of the budget), then one acquisition-optimized point
per evaluation. A GP fit + acquisition search costs ~1 s, nothing next to a
simulation.

Outcomes are modelled on a bilog scale, sign(y) log(1 + |y|), so a collapsed
eye's huge TDECQ margin does not flatten the model near the optimum; samples
are mapped back before the composite. Undefined outcomes (NaN TDECQ, a
failed chain) are imputed past the worst value seen for that outcome (and an
undefined constraint slack as infeasible), steering the search away from the
region rather than letting the model ignore it.
"""
from __future__ import annotations

import warnings

import numpy as np

# L-BFGS iterations per acquisition restart: past ~60 the kinked composite
# only polishes (same optimum on 8-D min-of-margins tests, 35% faster)
ACQ_MAXITER = 60
INSTALL_HINT = ("Bayesian optimization needs BoTorch: pip install botorch "
                "(or pip install -e '.[bo]')")


def require():
    """Import torch + BoTorch (quietly: they warn about deprecations on
    import) and return torch; ImportError with the install hint if absent."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
            import torch
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return torch


def initial_count(d: int, iters: int) -> int:
    """Evaluations in the space-filling start (the start design included)."""
    return int(min(max(d + 1, 4), max(2, iters // 3)))


def _bilog(y):
    return np.sign(y) * np.log1p(np.abs(y))


def _impute(Y: np.ndarray, sense: np.ndarray, n_terms: int) -> np.ndarray:
    """Replace non-finite outcomes with a value one observed range past the
    worst finite one in their column (worst: lowest when higher is better,
    ``sense`` +1), so an undefined design scores below every defined one.
    Slack columns (from ``n_terms`` on) are also forced negative: the flow
    scores an undefined constraint as infeasible (J = inf)."""
    Y = np.array(Y, float)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        ok = np.isfinite(col)
        if ok.all():
            continue
        if not ok.any():
            col[:] = -1.0 if j >= n_terms else 0.0
            continue
        good = col[ok]
        worst = good.min() if sense[j] > 0 else good.max()
        pad = np.ptp(good) or max(abs(worst), 1.0)
        fill = worst - sense[j] * pad
        col[~ok] = min(fill, -pad) if j >= n_terms else fill
    return Y


def bayes_opt(f, x0, lo, hi, iters: int, *, n_terms: int,
              combine: str | None, maximize: bool, n_slack: int = 0,
              penalty: float = 100.0, seed: int = 0, say=None):
    """Optimize over the [lo, hi] box in exactly ``iters`` calls of ``f``.

    ``f(x)`` returns ``n_terms + n_slack`` outcomes, NaN where undefined: the
    objective terms, then the constraint slacks (relative margins, >= 0 when
    met). The objective is ``combine`` ("min", "max" or None for a single
    term) over the terms, maximized or minimized, minus ``penalty`` times the
    summed slack shortfalls -- the flow's J. Returns the evaluated points and
    their raw outcomes as arrays."""
    torch = require()
    from torch.quasirandom import SobolEngine

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d, m = len(lo), n_terms + n_slack
    span = hi - lo
    s = 1.0 if maximize else -1.0
    # the composite is increasing in each slack and, for a maximized
    # objective, in each term: the direction a missing outcome is "worst" in
    sense = np.array([s] * n_terms + [1.0] * n_slack)

    def composite(Z):
        """Scalar to maximize from bilog-scale samples [..., m]."""
        Y = torch.sign(Z) * torch.expm1(Z.abs().clamp_max(30.0))
        t = Y[..., :n_terms]
        v = (t.amin(-1) if combine == "min" else
             t.amax(-1) if combine == "max" else t[..., 0])
        out = s * v
        if n_slack:
            out = out - penalty * (-Y[..., n_terms:]).clamp_min(0.0).sum(-1)
        return out

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []

    def run(x):
        x = np.clip(np.asarray(x, float), lo, hi)
        y = np.asarray(f(x), float).reshape(-1)
        if y.size != m:
            raise ValueError(f"bayes_opt: f returned {y.size} outcomes, "
                             f"expected {m}")
        X.append(x)
        Y.append(y)

    sobol = SobolEngine(d, scramble=True, seed=seed)
    n_init = min(initial_count(d, iters), iters)
    if say:
        say(f"  bayes: start design + {n_init - 1} Sobol points, then "
            f"{iters - n_init} model-guided (a GP per "
            + (f"term of {combine}()" if combine else "objective")
            + (" and per constraint" if n_slack else "") + ", qLogNEI)")
    run(x0)
    for u in sobol.draw(n_init - 1).double().numpy():
        run(lo + u * span)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        while len(X) < iters:
            U = (np.asarray(X) - lo) / span
            Z = _bilog(_impute(np.asarray(Y), sense, n_terms))
            try:
                u = _propose(torch, U, Z, composite)
            except Exception as exc:  # noqa: BLE001 -- keep a long run going
                if say:
                    say(f"  bayes: model step failed ({type(exc).__name__}: "
                        f"{exc}); sampling a Sobol point instead")
                u = None
            # a repeat of an evaluated design teaches the model nothing
            if u is None or np.min(np.abs(U - u).max(axis=1)) < 1e-9:
                u = sobol.draw(1).double().numpy()[0]
            run(lo + np.clip(u, 0.0, 1.0) * span)
    return np.asarray(X), np.asarray(Y)


def _propose(torch, U: np.ndarray, Z: np.ndarray, composite) -> np.ndarray:
    """The next point in the unit cube: fit a GP per outcome column of ``Z``
    and maximize qLogNEI of the composite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

    d = U.shape[1]
    tx = torch.tensor(U, dtype=torch.float64)
    ty = torch.tensor(Z, dtype=torch.float64)
    bounds = torch.stack([torch.zeros(d, dtype=torch.float64),
                          torch.ones(d, dtype=torch.float64)])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)     # tiny GP matrices: one thread is as fast
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(
                tx, ty, outcome_transform=Standardize(m=ty.shape[-1]))
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(model.likelihood, model))
            acq = qLogNoisyExpectedImprovement(
                model, X_baseline=tx,
                objective=GenericMCObjective(
                    lambda Z_, X=None: composite(Z_)),
                prune_baseline=True)
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8,
                                    raw_samples=512,
                                    options={"maxiter": ACQ_MAXITER})
    finally:
        torch.set_num_threads(threads)
    u = cand.detach().numpy().reshape(-1)
    if not np.all(np.isfinite(u)):
        raise ValueError("non-finite candidate")
    return u
