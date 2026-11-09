"""A two-sided linear bound on the flock centre's distance to the target (published Jadhav model).

    .venv/bin/python experiments/linear_approach_bound.py   # writes results/linear_approach_bound.json

Published sheep and Strömbom dog at the published 250 m start, 300 runs (seed 2024) continued to 500 s. With rho_t the
centre's distance to the target, a_t the unit vector toward it, and the centre's step dG_t = v (p_t a_t + q_t a_t^perp):

  exact, no assumptions (the triangle inequality and Pythagoras)
      rho_{t+1}^2 = (rho_t - v p_t)^2 + v^2 q_t^2,   |rho_{t+1} - rho_t| <= v
      -v p_t <= rho_{t+1} - rho_t <= -v p_t + v^2 q_t^2 / (2 (rho_t - v p_t))     when rho_t > v p_t
      so rho_n >= rho_0 - v n, and rho_n = rho_0 - v sum_{t<n} p_t + E_n with E_n >= 0

  conditional on the alignment. m(x) = E[p_t | x_t = x] is computed exactly (shepherd/expected_heading.py, no Monte Carlo).
      Given the state the sheep draw independently, so p_t - m(x_t) has conditional sub-Gaussian proxy 1/N (Hoeffding),
      and the running sum is a supermartingale after exponential tilting (Azuma, with Ville's maximal inequality).
      If (1/n) sum_{t<n} m(x_t) >= p* for all n >= n0, then with probability >= 1 - delta, for n0 <= n <= T until rho <= rho_min,
          rho_n <= rho_0 - c n + v sqrt(2 T ln(1/delta) / N),   c = v p* - v^2 / (2 (rho_min - v)).

The drive window is that of experiments/closed_loop_convergence.py: from the dog reaching its drive point until the centre is
within rho_min = 10 m. The assumption on m is checked on the states the runs visit; it is not proved for all states.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import closed_loop_convergence as CL  # noqa: E402
from shepherd.expected_heading import expected_headings  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

RUNS, SEED, STEPS = 300, 2024, 500
RHO_MIN, DELTA, BURN_IN, LINK = 10.0, 0.01, 30, 5.0
P_STARS = (0.65, 0.70, 0.75, 0.80)


def largest_linked(x):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return np.bincount(connected_components(d < LINK, directed=False)[1]).max() / len(x)


def main():
    p = Params(n_steps=STEPS)
    v, N = p.v_sheep, p.n_sheep
    sim = simulate(p, n_runs=RUNS, seed=SEED)
    G = sim.pos_s.mean(axis=1)  # (R, 2, T)
    rho = np.linalg.norm(G, axis=1)  # (R, T)
    a = -G / rho[:, None]
    perp = np.stack([-a[:, 1], a[:, 0]], axis=1)
    dG = np.diff(G, axis=2)
    pt = (a[:, :, :-1] * dG).sum(1) / v
    qt = (perp[:, :, :-1] * dG).sum(1) / v
    change, rt = np.diff(rho, axis=1), rho[:, :-1]
    usable = rt - v * pt > 0
    upper = -v * pt + v ** 2 * qt ** 2 / (2 * np.where(usable, rt - v * pt, 1.0))
    exact = dict(steps=int(pt.size), identity_residual_max=float(np.abs(rho[:, 1:] ** 2 - (rt - v * pt) ** 2 - (v * qt) ** 2).max()),
                 speed_limit_violations=int((np.abs(change) > v + 1e-9).sum()), lower_violations=int((change[usable] < -v * pt[usable] - 1e-9).sum()),
                 upper_violations=int((change[usable] > upper[usable] + 1e-9).sum()), steps_with_rho_above_vp=int(usable.sum()))

    g = CL.geometry(sim)
    t_dog = CL.pursuit(g, sim.mode, CL.reach_times(g))["t_dog"]
    windows = [(r, int(t_dog[r]), int(t_dog[r] + np.flatnonzero(rho[r, t_dog[r]:] < RHO_MIN)[0])) for r in range(RUNS) if (rho[r, t_dog[r]:] < RHO_MIN).any()]

    per_run, M, surprise = [], {}, []
    for r, t0, t1 in windows:
        n = np.arange(1, t1 - t0 + 1)
        y, tt = rho[r, t0:t1], np.arange(t0, t1, dtype=float)
        S = np.cumsum(pt[r, t0:t1])
        E = rho[r, t0 + 1:t1 + 1] - rho[r, t0] + v * S
        b, c0 = np.polyfit(tt, y, 1)
        k, ln_a = np.polyfit(tt, np.log(y), 1)
        rmse = float(np.sqrt(np.mean((y - (c0 + b * tt)) ** 2)))
        m = np.concatenate([((lambda mu, mv: ((mv[..., None] * mu).mean(1) * a[r, :, s:min(s + 400, t1)].T).sum(1))(
            *expected_headings(sim.pos_s[r, :, :, s:min(s + 400, t1)].transpose(2, 0, 1), sim.vel_s[r, :, :, s:min(s + 400, t1)].transpose(2, 0, 1),
                               sim.pos_d[r, :, s:min(s + 400, t1)].T, p))) for s in range(t0, t1, 400)])
        M[r] = m
        surprise.append(pt[r, t0:t1] - m)
        bar_p = (S / n)[n >= BURN_IN]
        per_run.append(dict(length=int(t1 - t0), slope=float(-b), predicted=float(v * pt[r, t0:t1].mean()), rmse_linear=rmse,
                            rmse_exponential=float(np.sqrt(np.mean((np.exp(ln_a + k * tt) - y) ** 2))), largest_miss=float(np.abs(y - (c0 + b * tt)).max()),
                            E_max=float(E.max()), E_min=float(E.min()), running_p_min=float(bar_p.min()), running_p_max=float(bar_p.max()),
                            running_m_min=float((np.cumsum(m) / n)[n >= BURN_IN].min())))
    q = lambda key: [row[key] for row in per_run]
    all_m, all_s = np.concatenate(list(M.values())), np.concatenate(surprise)
    lengths = np.array(q("length"))
    drive = dict(runs=len(windows), steps=int(lengths.sum()), window_median=float(np.median(lengths)), window_min=int(lengths.min()), window_max=int(lengths.max()),
                 slope_median=float(np.median(q("slope"))), slope_min=float(min(q("slope"))), slope_max=float(max(q("slope"))), predicted_median=float(np.median(q("predicted"))),
                 rmse_median=float(np.median(q("rmse_linear"))), largest_miss_median=float(np.median(q("largest_miss"))), largest_miss_max=float(max(q("largest_miss"))),
                 exp_over_linear_rmse_median=float(np.median(np.array(q("rmse_exponential")) / np.array(q("rmse_linear")))),
                 E_max_median=float(np.median(q("E_max"))), E_max=float(max(q("E_max"))), E_min=float(min(q("E_min"))),
                 running_p_min=float(min(q("running_p_min"))), running_p_max=float(max(q("running_p_max"))), burn_in_s=BURN_IN)
    expected = dict(min=float(all_m.min()), p01=float(np.percentile(all_m, 0.1)), p1=float(np.percentile(all_m, 1)), median=float(np.median(all_m)),
                    share_at_least={str(x): float((all_m >= x).mean()) for x in (0.5, 0.6, 0.7, 0.75, 0.8)}, surprise_mean=float(all_s.mean()),
                    surprise_se=float(all_s.std() / np.sqrt(len(all_s))), surprise_sd=float(all_s.std()), hoeffding_proxy_sd=float(1 / np.sqrt(N)), running_m_min=float(min(q("running_m_min"))))

    band = {}
    for pstar in P_STARS:
        c = v * pstar - v ** 2 / (2 * (RHO_MIN - v))
        held, broke_up, broke_low = 0, 0, 0
        for r, t0, t1 in windows:
            n = np.arange(1, t1 - t0 + 1)
            if not ((np.cumsum(M[r]) / n)[n >= BURN_IN] >= pstar).all():
                continue
            held += 1
            y = rho[r, t0 + 1:t1 + 1]
            top = rho[r, t0] - c * n + v * np.sqrt(2 * len(n) * np.log(1 / DELTA) / N)
            broke_up += bool((y[n >= BURN_IN] > top[n >= BURN_IN]).any())
            broke_low += bool((y < rho[r, t0] - v * n - 1e-9).any())
        band[str(pstar)] = dict(slope_c=float(c), assumption_held_runs=held, upper_broken=broke_up, lower_broken=broke_low)
    fluctuation = {f"T={T}": float(v * np.sqrt(2 * T * np.log(1 / DELTA) / N)) for T in (240, int(lengths.max()))}

    bins = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 200))
    startup = dict(dog_in_position_median_s=float(np.median(t_dog)), dog_in_position_max_s=float(np.max(t_dog)),
                   closing_speed_by_time={f"{lo}-{hi}s": float(-np.diff(rho[:, lo:hi + 1], axis=1).mean()) for lo, hi in bins})

    spread, unlinked, align = [], [], []
    for r, t0, t1 in windows:
        x = sim.pos_s[r, :, :, t0:t1].transpose(2, 0, 1)
        spread.append(np.linalg.norm(x - x.mean(1, keepdims=True), axis=-1).max(1))
        unlinked.append(np.array([largest_linked(z) < 1 for z in x]))
        align.append(pt[r, t0:t1])
    spread, unlinked, align = map(np.concatenate, (spread, unlinked, align))
    whole = [min(largest_linked(sim.pos_s[r, :, :, k].T.T) for k in range(0, rho.shape[1], 2)) for r in range(RUNS)]
    cohesion = dict(farthest_sheep_p99=float(np.percentile(spread, 99)), farthest_sheep_max=float(spread.max()), steps_with_a_sheep_unlinked=int(unlinked.sum()),
                    alignment_by_farthest_sheep={f"{lo}-{hi}m": float(align[(spread >= lo) & (spread < hi)].mean()) for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 12))},
                    runs_ever_below_90pct_linked_whole_500s=int(np.sum(np.array(whole) < 0.9)))

    out = dict(setup=dict(start_m=p.box_length, steps=STEPS, runs=RUNS, seed=SEED, rho_min=RHO_MIN, delta=DELTA, link_m=LINK), exact=exact, drive=drive,
               expected_alignment=expected, band=band, fluctuation_m=fluctuation, startup=startup, cohesion=cohesion)
    (ROOT / "results" / "linear_approach_bound.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
