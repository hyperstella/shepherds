"""Seed-robustness check of the scorecard row "B. DL metres: Jadhav sheep, fixed goal" (no rule or controller change).

    .venv/bin/python experiments/follower_swap_validation.py   # writes results/follower_swap_validation.json

The configuration is taken from hybrid_scorecard.scenarios() unchanged (Jadhav sheep, 14 Di Lorenzo herders,
uniform start, 24 m torus, r_goal 6 m). 50 fresh seeds (5000–5049) for 600 s; the first 6 of them, chosen before
any result, continue to 6000 s (same seed, so the first 600 s are the same run). dt = 1 s and positions are stored
every step, so occupancy is checked at every step of the simulation, as an integer count of sheep.

Late window = the scorecard's: last 25 % of the run, endpoints included. Density error is Metrics.density_error's
functional (relative L² against the smoothed target, the same grid and bandwidth): instantaneous = mean over the window of the
error at each step; time-averaged = error of the window-averaged density. Initial transients are excluded by the window.
"""

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from shepherd import bridge as B  # noqa: E402
from shepherd.scorecard import score  # noqa: E402
import hybrid_scorecard as HS  # noqa: E402

NAME = "B. DL metres: Jadhav sheep, fixed goal"
SEEDS, T_SHORT = list(range(5000, 5050)), 600.0
LONG_SEEDS, T_LONG, BLOCK = SEEDS[:6], 6000.0, 600.0
RADII = (2.0, 3.0, 4.0, 5.0, 6.0)


def window(r, met, t0, t1):
    Tr, D, g = met.T, r.domain, met.grid
    c = np.array(Tr.centre, float)
    k = np.where((r.t >= t0 - 1e-9) & (r.t <= t1 + 1e-9))[0]
    xs = r.pos_s[k]
    N = xs.shape[1]
    dist = np.linalg.norm(D.wrap(xs - c), axis=-1)  # (steps, sheep)
    count = (dist <= Tr.r_goal).sum(1)
    h2 = g.h ** 2
    avg, inst = np.zeros_like(met.rho_b), []
    for x in xs:
        rho = g.convolve(met.Kb, g.deposit(D.wrap(x), np.full(N, 1.0 / N)))
        avg += rho
        inst.append(float(((rho - met.rho_b) ** 2).sum() * h2 / met.norm))
    avg /= len(xs)
    G = np.array([D.centroid(x) for x in xs])
    cd = np.linalg.norm(D.wrap(G - c), axis=-1)
    rms = np.sqrt((np.linalg.norm(D.wrap(xs - G[:, None]), axis=-1) ** 2).mean(1))
    nearest = np.linalg.norm(D.wrap(r.pos_s[k][:, :, None] - r.pos_d[k][:, None]), axis=-1).min(axis=(1, 2))
    return dict(steps=len(k), full=bool((count == N).all()), frac_steps_full=float((count == N).mean()), min_count=int(count.min()),
                mean_occupancy=float(count.mean() / N), max_dist=float(dist.max()), p99_of_step_max=float(np.percentile(dist.max(1), 99)),
                occ_by_radius={str(R): float((dist <= R).mean()) for R in RADII},
                inst_err=float(np.mean(inst)), inst_err_lo=float(np.min(inst)), inst_err_hi=float(np.max(inst)),
                timeavg_err=float(((avg - met.rho_b) ** 2).sum() * h2 / met.norm),
                centroid_mean=float(cd.mean()), centroid_max=float(cd.max()), rms_mean=float(rms.mean()), rms_max=float(rms.max()),
                nearest_dog_min=float(nearest.min()), nearest_dog_mean=float(nearest.mean()))


def one(job):
    seed, T_end = job
    F, H, T, D, kw, _ = HS.scenarios()[NAME]
    r = B.simulate_bridge(F, H, T, D, seed=seed, n_steps=int(round(T_end / F.dt)), **kw)
    met = B.Metrics(T.resolve(D), D, F.n)
    assert float(r.t[1] - r.t[0]) == 1.0
    s = score(r)
    dist = np.linalg.norm(D.wrap(r.pos_s - np.array(met.T.centre)), axis=-1)
    out_t = r.t[(dist > met.T.r_goal).any(1)]
    disp = np.linalg.norm(D.wrap(np.diff(r.pos_s[-151:], axis=0)), axis=-1)
    largest = np.array([np.bincount(connected_components(np.linalg.norm(D.wrap(x[:, None] - x[None]), axis=-1) < 5.0, directed=False)[1]).max() / F.n
                        for x in r.pos_s])  # same link and share as scorecard.DEFAULTS
    split_t = r.t[largest < 0.9]
    row = dict(seed=seed, T=T_end, late=window(r, met, 0.75 * T_end, T_end), from100=window(r, met, 100.0, T_end),
               t_last_any_out=float(out_t.max()) if len(out_t) else None, n_steps_any_out_after_60=int((out_t >= 60).sum()),
               time_to_success=s["time_to_success"], escapes=s["escapes"], splits=s["splits"], splits_unrecovered=s["splits_unrecovered"],
               largest_group_min=s["largest_group_min"], t_last_split=float(split_t.max()) if len(split_t) else None, late_sheep_speed=float(disp.mean()),
               active_late=float(np.mean(r.metrics["active_fraction"][-151:])), pos600=r.pos_s[600].tobytes().hex()[:64])
    if T_end > T_SHORT:
        row["blocks"] = [dict(t0=float(a), **{k: v for k, v in window(r, met, a, a + BLOCK).items() if k != "occ_by_radius"})
                         for a in np.arange(0.0, T_end, BLOCK)]
    return row


def q(v):
    v = np.array(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), q05=float(np.percentile(v, 5)), q25=float(np.percentile(v, 25)),
                median=float(np.median(v)), q75=float(np.percentile(v, 75)), q95=float(np.percentile(v, 95)), max=float(v.max()))


def main():
    F, _, T, D, _, _ = HS.scenarios()[NAME]
    Tr = T.resolve(D)
    met = B.Metrics(Tr, D, F.n)
    g = met.grid
    r_grid = np.hypot(D.wrap(g.X - Tr.centre[0]), D.wrap(g.Y - Tr.centre[1]))
    raw = g.target(Tr, D)[0]
    target = dict(floor=float(met.floor), mass_in_goal_smoothed=float(met.rho_b[r_grid <= Tr.r_goal].sum() * g.h ** 2),
                  mass_in_goal_unsmoothed=float(raw[r_grid <= Tr.r_goal].sum() * g.h ** 2), r_goal=float(Tr.r_goal),
                  mass_by_radius={str(R): float(met.rho_b[r_grid <= R].sum() * g.h ** 2) for R in RADII})
    print("target", target, flush=True)
    with Pool(10) as p:
        short = p.map(one, [(s, T_SHORT) for s in SEEDS], chunksize=1)
        print("short done", flush=True)
        long = p.map(one, [(s, T_LONG) for s in LONG_SEEDS], chunksize=1)
    res = dict(config=NAME, seeds=SEEDS, T=T_SHORT, long_seeds=LONG_SEEDS, T_long=T_LONG, target=target, short=short, long=long)
    res["summary"] = {k: q([x["late"][k] for x in short]) for k in ("inst_err", "timeavg_err", "max_dist", "centroid_mean", "centroid_max", "rms_mean", "mean_occupancy", "frac_steps_full")}
    res["summary"]["timeavg_err_from100"] = q([x["from100"]["timeavg_err"] for x in short])
    res["summary"]["inst_err_from100"] = q([x["from100"]["inst_err"] for x in short])
    res["summary"]["full_late"] = int(sum(x["late"]["full"] for x in short))
    res["determinism_first600"] = [a["pos600"] == b["pos600"] for a, b in zip(short[:6], long)]
    (ROOT / "results" / "follower_swap_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    print("deterministic first 600 s:", res["determinism_first600"])


if __name__ == "__main__":
    main()
