"""Shared helpers for the experiment scripts: loading tables and fitting posteriors."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from shepherd import abc as ABC, protocol as PR  # noqa: E402
from build_table import FIT_STATS, MODELS  # noqa: E402

TABLES = ROOT / "results" / "tables"
COLS = [PR.SUMMARY_KEYS.index(k) for k in FIT_STATS]


def load(name):
    """Load a table, applying the current degenerate-slope rule to stored summaries."""
    t = dict(np.load(TABLES / f"{name}.npz", allow_pickle=False))
    for k in list(t):
        if (k.startswith("tab") or k == "meta_obs") and t[k].ndim >= 2:
            t[k] = ABC.sanitize(t[k])
    return t


def observed(tab, mask=None, lmask=None):
    """Observed summary vector: short-window statistics (+ long-drive ones for multi-horizon tables)."""
    obs = ABC.aggregate(tab["meta_obs"][None, :, COLS], tab["meta_durations"], mask)[0]
    if "ltab1" in tab:
        obs = np.concatenate([obs, ABC.aggregate(tab["lmeta_obs"][None], tab["lmeta_durations"], lmask)[0]])
    return obs


def simulated(tab, rnd, mask=None, lmask=None):
    S = ABC.aggregate(tab[f"tab{rnd}"][..., COLS], tab["meta_durations"], mask)
    if f"ltab{rnd}" in tab:
        S = np.concatenate([S, ABC.aggregate(tab[f"ltab{rnd}"], tab["lmeta_durations"], lmask)], axis=1)
    return S


def posterior(tab, name, rnd=2, accept=0.01, mask=None, lmask=None):
    """ABC posterior from round `rnd` of a model table (optionally on a subset of windows/drives)."""
    names = [str(k) for k in tab["names"]]
    th = {k: tab[f"th{rnd}_{k}"] for k in names}
    S = simulated(tab, rnd, mask, lmask)
    return ABC.fit(th, S, observed(tab, mask, lmask), ABC.mad_scale(S), MODELS[name]["priors"], accept)


def draws(post, n, seed=0):
    names = post.names
    pick = np.random.default_rng(seed).choice(len(post.samples[names[0]]), min(n, len(post.samples[names[0]])),
                                              replace=False)
    return {k: post.samples[k][pick] for k in names}
