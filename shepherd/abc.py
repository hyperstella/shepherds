"""Approximate Bayesian computation with a simulated reference table (NumPy).

Workflow
1. `sobol_prior` draws parameter sets (log-uniform or uniform per parameter).
2. `simulate_table` runs the windowed dog-replay harness for every set and
   stores per-window summary statistics (n_sets, n_windows, n_stats), so any
   subset of windows (e.g. leave-one-session-out) can be aggregated later
   without re-simulating.
3. `fit` keeps the closest `accept` fraction of simulations to the observed
   summaries (Euclidean on standardized statistics) and applies Beaumont et
   al.'s (2002) local-linear regression adjustment in a transformed space.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from . import data
from . import protocol as PR
from . import scenario as SC
from .model import Params


@dataclass(frozen=True)
class Prior:
    lo: float
    hi: float
    log: bool = True

    def to_z(self, x):
        """Map to an unbounded space (logit of the scaled position)."""
        u = (np.log(x) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo)) if self.log else (x - self.lo) / (self.hi - self.lo)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.log(u / (1 - u))

    def from_z(self, z):
        u = 1 / (1 + np.exp(-z))
        return np.exp(np.log(self.lo) + u * (np.log(self.hi) - np.log(self.lo))) if self.log else self.lo + u * (self.hi - self.lo)

    def from_unit(self, u):
        return np.exp(np.log(self.lo) + u * (np.log(self.hi) - np.log(self.lo))) if self.log else self.lo + u * (self.hi - self.lo)


def sobol_prior(priors, n, seed=0, box=None):
    """n parameter sets from the priors (optionally truncated to box = {name: (lo, hi)})."""
    names = list(priors)
    u = qmc.Sobol(len(names), scramble=True, seed=seed).random(n)
    out = {}
    for j, k in enumerate(names):
        pr = priors[k]
        if box is not None:
            lo, hi = box[k]
            pr = Prior(lo, hi, pr.log)
        out[k] = pr.from_unit(u[:, j])
    return out


# ---------------------------------------------------------------- simulation

_DRIVES = None


def _drives():
    global _DRIVES
    if _DRIVES is None:
        _DRIVES = data.load_drives(data.load_sessions())
    return _DRIVES


def _summarize(segs, n, stats):
    out = np.empty((n, len(segs), len(stats)), np.float32)
    for w, s in enumerate(segs):
        sm = PR.summaries(s)
        out[:, w, :] = np.stack([sm[k] for k in stats], axis=-1)
    return out


def _job(args):
    theta, base, dt, seed, stats, window_kw = args
    n = len(next(iter(theta.values())))
    segs = SC.replay_windows(_drives(), base, n_runs=n, seed=seed, dt=dt, per_run=theta, **window_kw)
    return _summarize(segs, n, stats)


def _job_kernel(args):
    theta, model, c_base, dt, seed, stats, window_kw = args
    from . import kernels as KN

    n = len(next(iter(theta.values())))
    coef = KN.apply_gains(model, c_base, {k: v for k, v in theta.items() if k != "sigma"})
    segs = SC.replay_windows_kernel(_drives(), model, coef, theta["sigma"], n_runs=n, seed=seed, dt=dt, **window_kw)
    return _summarize(segs, n, stats)


def window_meta(dt, **window_kw):
    """Labels, sessions and durations of the replay windows, plus the data's
    per-window summaries (n_windows, n_stats) for `PR.SUMMARY_KEYS`."""
    segs = SC.window_segments(_drives(), dt=dt, **window_kw)
    obs = np.stack([np.stack([PR.summaries(s)[k] for k in PR.SUMMARY_KEYS]) for s in segs]).astype(float)
    return {
        "labels": np.array([s.label for s in segs]),
        "sessions": np.array([s.session for s in segs]),
        "durations": np.array([s.duration for s in segs]),
        "obs": obs,
    }


def simulate_table(theta, base=Params(), dt=1.0, seed=0, chunk=250, workers=None, stats=PR.SUMMARY_KEYS,
                   window_kw=None):
    """Per-window summaries for every parameter set: (n_sets, n_windows, n_stats) float32."""
    n = len(next(iter(theta.values())))
    jobs = [({k: v[i:i + chunk] for k, v in theta.items()}, base, dt, seed + i, list(stats), window_kw or {})
            for i in range(0, n, chunk)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1:
        parts = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_job, jobs))
    return np.concatenate(parts, axis=0)


def simulate_table_kernel(theta, model, c_base, dt=1.0, seed=0, chunk=128, workers=None, stats=PR.SUMMARY_KEYS,
                          window_kw=None):
    """Like `simulate_table` for the learned kernel model: theta holds per-run
    gains for kernel terms (e.g. "E", "D", "DV") and the noise level "sigma"."""
    n = len(next(iter(theta.values())))
    jobs = [({k: v[i:i + chunk] for k, v in theta.items()}, model, np.asarray(c_base), dt, seed + i, list(stats),
             window_kw or {}) for i in range(0, n, chunk)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1:
        parts = [_job_kernel(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_job_kernel, jobs))
    return np.concatenate(parts, axis=0)


def sanitize(table, stats=PR.SUMMARY_KEYS):
    """Apply the current degenerate-fit rule for Fig-4 slopes to a stored table."""
    table = np.array(table, dtype=float, copy=True)
    for k in ("slope_4a", "slope_4b"):
        if k in stats:
            j = list(stats).index(k)
            v = table[..., j]
            v[~(np.abs(v) <= PR.SLOPE_LIMIT)] = np.nan
    return table


def aggregate(table, durations, mask=None):
    """Duration-weighted mean over windows (ignoring NaN): (..., W, K) -> (..., K)."""
    w = np.asarray(durations, float)
    if mask is not None:
        w = np.where(mask, w, 0.0)
    ok = np.isfinite(table)
    ww = np.where(ok, w[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(np.where(ok, table, 0.0) * ww, axis=-2) / ww.sum(axis=-2)


# ---------------------------------------------------------------- inference

@dataclass
class Posterior:
    names: list
    samples: dict  # adjusted, name -> (m,)
    raw: dict  # accepted before adjustment
    idx: np.ndarray  # accepted table rows
    dist: np.ndarray
    tolerance: float

    def summary(self, priors):
        out = {}
        for k in self.names:
            s = self.samples[k]
            z_post, z_prior = priors[k].to_z(s), priors[k].to_z(priors[k].from_unit(np.linspace(0.001, 0.999, 2001)))
            out[k] = {
                "median": float(np.median(s)),
                "ci90": [float(v) for v in np.percentile(s, [5, 95])],
                "contraction": float(1 - np.var(z_post) / np.var(z_prior)),
            }
        return out


def fit(theta, sims, obs, scale, priors, accept=0.01, adjust=True):
    """Rejection ABC with local-linear regression adjustment.

    theta: {name: (n,)}; sims: (n, K) aggregated summaries; obs: (K,);
    scale: (K,) per-statistic scale (e.g. MAD of the prior predictive).
    """
    names = list(theta)
    ok_cols = np.isfinite(obs) & np.isfinite(scale) & (scale > 0)
    S = sims[:, ok_cols] / scale[ok_cols]
    o = obs[ok_cols] / scale[ok_cols]
    S = np.where(np.isfinite(S), S, np.nanmedian(S, axis=0))  # rare NaN: neutral value
    d = np.sqrt(((S - o) ** 2).sum(1))
    m = max(20, int(round(accept * len(d))))
    idx = np.argsort(d)[:m]
    tol = d[idx[-1]]
    raw = {k: theta[k][idx] for k in names}
    Z = np.stack([priors[k].to_z(raw[k]) for k in names], axis=1)
    if adjust and m > 5 * (S.shape[1] + 1):  # too few accepted draws would overfit the regression
        w = 1 - (d[idx] / tol) ** 2
        X = np.hstack([np.ones((m, 1)), S[idx] - o])
        W = np.sqrt(np.clip(w, 1e-12, None))[:, None]
        beta, *_ = np.linalg.lstsq(X * W, Z * W, rcond=None)
        Z = Z - (S[idx] - o) @ beta[1:]
    samples = {k: priors[k].from_z(Z[:, j]) for j, k in enumerate(names)}
    return Posterior(names, samples, raw, idx, d[idx], float(tol))


def mad_scale(sims):
    med = np.nanmedian(sims, axis=0)
    return 1.4826 * np.nanmedian(np.abs(sims - med), axis=0)


def coverage(theta, sims, priors, n_test=50, accept=0.01, level=0.9, seed=0):
    """Use table rows as pseudo-observed data: share of true values inside the
    posterior `level` interval, per parameter (should be close to `level`)."""
    rng = np.random.default_rng(seed)
    n = len(sims)
    test = rng.choice(n, n_test, replace=False)
    rest = np.setdiff1d(np.arange(n), test)
    th_rest = {k: v[rest] for k, v in theta.items()}
    scale = mad_scale(sims[rest])
    hits = {k: 0 for k in theta}
    a = (1 - level) / 2 * 100
    for t in test:
        post = fit(th_rest, sims[rest], sims[t], scale, priors, accept)
        for k in theta:
            lo, hi = np.percentile(post.samples[k], [a, 100 - a])
            hits[k] += lo <= theta[k][t] <= hi
    return {k: v / n_test for k, v in hits.items()}


def truncated_box(post, priors, margin=0.1):
    """Bounding box of accepted draws, widened by `margin` (in the prior's scale)."""
    box = {}
    for k in post.names:
        pr, s = priors[k], post.raw[k]
        if pr.log:
            lo, hi = np.log(s.min()), np.log(s.max())
            w = hi - lo
            box[k] = (max(pr.lo, np.exp(lo - margin * w)), min(pr.hi, np.exp(hi + margin * w)))
        else:
            lo, hi = s.min(), s.max()
            w = hi - lo
            box[k] = (max(pr.lo, lo - margin * w), min(pr.hi, hi + margin * w))
    return box
