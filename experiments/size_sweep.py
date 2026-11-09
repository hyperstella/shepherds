"""Does the cross-perception condition N <= 2K + 1 mark where a driven Jadhav flock stops holding together?

Counting fact (docs/steer_and_shape.md §11): split a flock into two parts; if the smaller part has at most
K = 10 sheep, each of its sheep perceives at least one sheep of the other part. For two halves that
holds iff floor(N/2) <= K, i.e. N <= 21. From N = 22 two halves of 11 or more can lose sight of each
other. This script tests that boundary two ways, for N = 14, 16, 19, 21, 22, 24, 26, 30.

1. One step (as in experiments/shape_stress.py, topological test): compact synthetic flocks, every
   sheep in range (r_dog 40), split into halves across the heading by 4, 8 and 12 m. Share of perceived
   neighbours in the other half, kappa and E[dM2 | state] with bootstrap intervals.
2. Trajectories: the same compact flocks (disc of radius 0.9 sqrt(N) m, aligned, dog 7.5 m behind),
   published weights and rule, r_dog 20, driven for 300 steps by a dog that only drives, or that also
   collects (Strömbom, l_sep = 2 N^(2/3) m). Measured: the largest group linked at 6 m, a split (two or
   more sheep outside it), a straggler (exactly one), the step of the first split, the share of sheep
   moving at the end, and sqrt(M2).

Prediction if the condition controls fragmentation: rare splits up to N = 21, a jump from 22.

3. Where the boundary sits instead: the drive-only trajectory test for N = 14 to 24, varying the
   attraction weight (1.5, 3) and the dog's standoff (7.5, 12 m; its lateral spreading falls as 1/d).
   If fragmentation is set by how much each half still perceives of the other against the dog's
   spreading, the boundary should move to larger N with stronger attraction or a farther dog.

    .venv/bin/python experiments/size_sweep.py
Writes results/size_sweep.json and results/figures/size_sweep.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shape_stress as SS  # noqa: E402
from shape_trajectories import largest_group  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
SIZES = (14, 16, 19, 21, 22, 24, 26, 30)
GAPS = (4.0, 8.0, 12.0)
T, RUNS, SEED, EVERY = 300, 30, 71, 5
POLICIES = {"drive only": dict(l_sep=1e6), "with collecting": dict()}
COL = SS.COL


def compact(rng, S, N):
    R0 = 0.9 * np.sqrt(N)
    pts = []
    for _ in range(S):
        x = np.empty((0, 2))
        while len(x) < N:
            a = 2 * np.pi * rng.random()
            c = R0 * np.sqrt(rng.random()) * np.array([np.cos(a), np.sin(a)])
            if len(x) == 0 or np.linalg.norm(x - c, axis=1).min() >= 1.0:
                x = np.vstack([x, c])
        pts.append(x)
    return np.array(pts)


def trajectories(rng):
    out = []
    for N in SIZES:
        c0 = np.array([SS.FAR, 0.0])
        pos = c0 + compact(rng, RUNS, N)
        hd = SS.unit(np.array([-1.0, 0.0]) + 0.1 * rng.standard_normal((RUNS, N, 2)))
        for pol, extra in POLICIES.items():
            per = dict(w_att=np.full(RUNS, 1.5), w_ali=np.full(RUNS, 1.3), r_dog=np.full(RUNS, 20.0), l_drive=np.full(RUNS, SS.D_STAND))
            per.update({k: np.full(RUNS, v) for k, v in extra.items()})
            res = simulate(Params(n_sheep=N, n_steps=T + 1), n_runs=RUNS, seed=SEED + N, per_run=per,
                           init=dict(pos_s=pos, vel_s=hd, pos_d=np.tile(c0 + [SS.D_STAND, 0.0], (RUNS, 1)), vel_d=np.tile([-1.0, 0.0], (RUNS, 1))))
            X = res.pos_s
            ts = list(range(0, T + 1, EVERY))
            grp = np.stack([largest_group(X[..., t]) for t in ts], 1)  # (RUNS, len(ts))
            outside = N - grp
            split_t = outside >= 2
            first_split = np.where(split_t.any(1), np.array(ts)[split_t.argmax(1)], np.nan)
            moving_end = (np.linalg.norm(X[..., -1] - X[..., -2], axis=-1) > 1e-9).mean(1)
            sq = np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
            row = dict(N=N, policy=pol, split_end=float((outside[:, -1] >= 2).mean()), straggler_end=float((outside[:, -1] == 1).mean()),
                       ever_split=float(split_t.any(1).mean()), first_split_median=float(np.nanmedian(first_split)) if np.isfinite(first_split).any() else None,
                       largest_group_frac_end=float(np.median(grp[:, -1] / N)), moving_end=float(moving_end.mean()),
                       sqrtM2_end_median=float(np.median(sq[:, -1])), sqrtM2_over_sqrtN_cohesive=float(np.median((sq[:, -1] / np.sqrt(N))[outside[:, -1] == 0])) if (outside[:, -1] == 0).any() else None,
                       split_ci=wilson(int((outside[:, -1] >= 2).sum()), RUNS))
            out.append(row)
            print(f"  N {N:2d} {pol:15s}: split at end {row['split_end']:.2f} {[round(v, 2) for v in row['split_ci']]} (ever {row['ever_split']:.2f}, first at "
                  f"{row['first_split_median']}) straggler {row['straggler_end']:.2f} | largest group {row['largest_group_frac_end']:.2f} of N | moving {row['moving_end']:.2f} | "
                  f"√M2 {row['sqrtM2_end_median']:.1f}", flush=True)
    return out


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


BOUNDARY_SIZES = (14, 16, 17, 18, 19, 21, 24)
BOUNDARY_CASES = {"published, dog 7.5 m": dict(w_att=1.5, l_drive=7.5), "w_att 3, dog 7.5 m": dict(w_att=3.0, l_drive=7.5),
                  "published, dog 12 m": dict(w_att=1.5, l_drive=12.0)}


def boundary(rng):
    out = []
    for name, case in BOUNDARY_CASES.items():
        for N in BOUNDARY_SIZES:
            c0 = np.array([SS.FAR, 0.0])
            pos = c0 + compact(rng, RUNS, N)
            hd = SS.unit(np.array([-1.0, 0.0]) + 0.1 * rng.standard_normal((RUNS, N, 2)))
            per = dict(w_att=np.full(RUNS, case["w_att"]), w_ali=np.full(RUNS, 1.3), r_dog=np.full(RUNS, 20.0),
                       l_drive=np.full(RUNS, case["l_drive"]), l_sep=np.full(RUNS, 1e6))
            res = simulate(Params(n_sheep=N, n_steps=T + 1), n_runs=RUNS, seed=SEED + 100 + N, per_run=per,
                           init=dict(pos_s=pos, vel_s=hd, pos_d=np.tile(c0 + [case["l_drive"], 0.0], (RUNS, 1)), vel_d=np.tile([-1.0, 0.0], (RUNS, 1))))
            outside = N - largest_group(res.pos_s[..., -1])
            k = int((outside >= 2).sum())
            out.append(dict(case=name, N=N, split_end=k / RUNS, split_ci=wilson(k, RUNS)))
            print(f"  {name:22s} N {N:2d}: split {k}/{RUNS} {[round(v, 2) for v in wilson(k, RUNS)]}", flush=True)
    return out


def figure(one, traj, bnd=None):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    for g, col in zip(GAPS, ("blue", "orange", "red")):
        rs = [r for r in one if r["gap"] == g]
        ax.plot([r["N"] for r in rs], [r["other_half_neighbours"] for r in rs], "o-", color=COL[col], lw=2, label=f"gap {g:g} m")
    ax.axvline(21.5, color=COL["gray"], lw=1, ls="--")
    ax.text(21.7, 0.2, "N = 2K + 1", fontsize=8, color="#3a3a38")
    ax.set(xlabel="sheep in the flock", ylabel="share of perceived neighbours in the other half", title="Halves split apart: can they still see each other?")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    for g, col in zip(GAPS, ("blue", "orange", "red")):
        rs = [r for r in one if r["gap"] == g]
        y = np.array([r["kappa_median"] for r in rs])
        lo = np.array([r["kappa_median_ci"][0] for r in rs])
        hi = np.array([r["kappa_median_ci"][1] for r in rs])
        ax.errorbar([r["N"] for r in rs], y, yerr=[y - lo, hi - y], fmt="o-", color=COL[col], lw=2, capsize=3, label=f"gap {g:g} m")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(21.5, color=COL["gray"], lw=1, ls="--")
    ax.set(xlabel="sheep in the flock", ylabel="κ, median with 95% CI", title="Inward relative heading across the gap (one step)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    for pol, col in (("drive only", "orange"), ("with collecting", "blue")):
        rs = [r for r in traj if r["policy"] == pol]
        y = np.array([r["split_end"] for r in rs])
        lo = np.array([r["split_ci"][0] for r in rs])
        hi = np.array([r["split_ci"][1] for r in rs])
        ax.errorbar([r["N"] for r in rs], y, yerr=[y - lo, hi - y], fmt="o-", color=COL[col], lw=2, capsize=3, label=f"split after {T} steps, {pol}")
    ax.axvline(21.5, color=COL["gray"], lw=1, ls="--")
    for (name, col) in zip(BOUNDARY_CASES, ("red", "green", "purple")) if bnd else []:
        rs = [r for r in bnd if r["case"] == name]
        ax.plot([r["N"] for r in rs], [r["split_end"] for r in rs], "s--", color=COL[col], lw=1.5, label=f"drive only, {name}")
    ax.set(xlabel="sheep in the flock", ylabel="fraction of runs (95% Wilson CI)", ylim=(-0.03, 1.03), title=f"Compact flocks under the drive: split after {T} steps")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(SIZES)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "size_sweep.png", dpi=130)


def main():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    SS.TOPO.update(sizes=SIZES, gaps=GAPS)
    print("one step, split halves:", flush=True)
    one = SS.topological(rng)
    print("trajectories from compact starts:", flush=True)
    traj = trajectories(rng)
    print("where the boundary sits (drive only):", flush=True)
    bnd = boundary(rng)
    out = dict(boundary=bnd, setup=dict(sizes=SIZES, gaps=GAPS, T=T, runs=RUNS, K=SS.P0.k_perceive, policies=POLICIES, rule="published", weights="published",
                          r_dog_trajectories=20.0, r_dog_one_step=SS.TOPO["r_dog"]), one_step=one, trajectories=traj, seconds=time.time() - tic)
    (OUT / "size_sweep.json").write_text(json.dumps(out, indent=1))
    figure(one, traj, bnd)
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
