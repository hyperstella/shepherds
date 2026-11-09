"""Bounded validation of the time-averaged coverage objective in one continuous simulation.

    .venv/bin/python experiments/goal_schedule_validation.py   # writes results/goal_schedule_validation.json

Configuration: `bridge.dilorenzo_scenario("jadhav")` unchanged (36 published Jadhav sheep, 14 Di Lorenzo
herders, their design and flow, 100 flow sub-steps per 1 s step, 24 m torus, von Mises goal). Only the
goal schedule differs, through `simulate_bridge(..., goal=...)` (shepherd/goal_schedule.py):

  fixed   the goal stays at the target centre
  hop     every 20 s the goal jumps to the next waypoint
  glide   the same waypoints and 20 s, the goal moving at constant speed between them (never jumps)

Matched across schedules: resources, target, duration (1200 s), seeds 0–3, the simulation's random
stream and therefore the initial conditions, and (hop vs glide) the waypoint sequence. The flock shape S
and the requested centre distribution p_c = ρ̄ ⊘ S come from separate fixed-goal runs (seeds 100–102).

Every quantity uses the same estimator: unit-mass deposit on the 64-grid, Gaussian smoothing with the
target's bandwidth b, samples every 1 s. Errors are relative L², ‖f − ρ̄_b‖²/‖ρ̄_b‖² (as Metrics). The
time-averaged density error is the error OF the averaged field, never an average of instantaneous errors.
Two windows are reported: t ∈ [0, 1200] s (includes the initial gathering) and t ∈ [100, 1200] s
(excludes only the initial gathering; every goal change, travel and settling stays in).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import bridge as B  # noqa: E402
from shepherd import goal_schedule as GS  # noqa: E402

OUT = ROOT / "results"
SEEDS = (0, 1, 2, 3)
T_END, DWELL, T0 = 1200, 20, 100
BURST = 3.0  # cohesion (m) above which the flock counts as burst: 1.5 × the held flock's ≈ 2.0 m median
SPLIT_LINK, SPLIT_SHARE = 5.0, 0.9  # split: the largest 5 m-linked group holds < 90 % of the sheep


def episodes(flag, t):
    """(start time, duration, ended?) for each maximal run of True."""
    out, k = [], 0
    while k < len(flag):
        if flag[k]:
            j = k
            while j < len(flag) and flag[j]:
                j += 1
            out.append((float(t[k]), float(t[min(j, len(t) - 1)] - t[k]), j < len(flag)))
            k = j
        else:
            k += 1
    return out


def rel_l2(f, ref, h2):
    return float(((f - ref) ** 2).sum() * h2 / ((ref ** 2).sum() * h2))


def analyse(r, met, g, Kb, P_dens, S_ref, D, N):
    h2 = g.h ** 2
    t = r.t
    G = np.array([D.centroid(x) for x in r.pos_s])
    kde = np.array([g.convolve(Kb, g.deposit(x, np.full(N, 1.0 / N))) for x in r.pos_s])
    inst = np.array([rel_l2(k, met.rho_b, h2) for k in kde])
    coh = np.array([met.cohesion(x) for x in r.pos_s])
    rms = np.array([np.sqrt((np.linalg.norm(D.wrap(x - D.centroid(x)), axis=-1) ** 2).mean()) for x in r.pos_s])
    aniso = []
    for x in r.pos_s:
        ev = np.linalg.eigvalsh(np.cov(D.wrap(x - D.centroid(x)).T))
        aniso.append(float(np.sqrt(ev[1] / max(ev[0], 1e-12))))
    aniso = np.array(aniso)
    largest = []
    for x in r.pos_s:
        d = np.linalg.norm(D.wrap(x[:, None] - x[None]), axis=-1)
        _, lab = connected_components(d < SPLIT_LINK, directed=False)
        largest.append(np.bincount(lab).max() / N)
    largest = np.array(largest)
    target_occ = g.convolve(Kb, P_dens)  # the requested centre distribution, smoothed like everything else
    out = {}
    for name, lo in (("from_0", 0.0), ("from_100", float(T0))):
        w = t >= lo
        avg = kde[w].mean(0)
        occ = g.convolve(Kb, g.deposit(G[w], np.full(w.sum(), 1.0 / w.sum())))
        goal_occ = g.convolve(Kb, g.deposit(r.goal[w], np.full(w.sum(), 1.0 / w.sum())))
        # fixed-shape approximation: predicted time-averaged density = S ∗ actual occupation
        occ_raw = g.deposit(G[w], np.full(w.sum(), 1.0 / w.sum()))  # already a density (mass / h²)
        S_run = GS.flock_shape(r.pos_s[w], D, g)
        conv = lambda S_, o: g.convolve(Kb, np.real(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(S_)) * np.fft.fft2(o))) * h2)
        out[name] = dict(
            instantaneous_error_mean=float(inst[w].mean()), instantaneous_error_median=float(np.median(inst[w])),
            time_averaged_error=rel_l2(avg, met.rho_b, h2),
            occupation_error=rel_l2(occ, target_occ, h2), goal_path_occupation_error=rel_l2(goal_occ, target_occ, h2),
            fixed_shape_ref_vs_measured=rel_l2(conv(S_ref, occ_raw), avg, h2),
            fixed_shape_run_vs_measured=rel_l2(conv(S_run, occ_raw), avg, h2),
            planned_S_pc_vs_measured=rel_l2(conv(S_ref, P_dens), avg, h2),
            centroid_to_target_mean=float(np.linalg.norm(D.wrap(G[w] - np.array(met.c)), axis=-1).mean()),
            centroid_to_goal_mean=float(np.linalg.norm(D.wrap(G[w] - r.goal[w]), axis=-1).mean()),
            cohesion_mean=float(coh[w].mean()), cohesion_sd=float(coh[w].std()), cohesion_p99=float(np.percentile(coh[w], 99)),
            rms_radius_mean=float(rms[w].mean()), rms_radius_cv=float(rms[w].std() / rms[w].mean()),
            anisotropy_mean=float(aniso[w].mean()),
        )
    b = [e for e in episodes(coh > BURST, t) if e[0] >= T0]
    s = [e for e in episodes(largest < SPLIT_SHARE, t) if e[0] >= T0]
    out["events_after_100"] = dict(
        bursts=len(b), burst_recovery_mean=float(np.mean([e[1] for e in b])) if b else 0.0,
        burst_recovery_max=float(max([e[1] for e in b])) if b else 0.0, bursts_unrecovered=sum(not e[2] for e in b),
        splits=len(s), split_duration_mean=float(np.mean([e[1] for e in s])) if s else 0.0,
        split_duration_max=float(max([e[1] for e in s])) if s else 0.0, splits_unrecovered=sum(not e[2] for e in s),
        time_split_share=float(np.mean(largest[t >= T0] < SPLIT_SHARE)),
    )
    return out


def main():
    F, H, T, D, _ = B.dilorenzo_scenario("jadhav")
    T = T.resolve(D)
    g = B.Grid(D.size, H.grid)
    met = B.Metrics(T, D, F.n, g)
    Kb = met.Kb
    t_start = time.time()
    # shape and requested centre distribution from separate fixed-goal runs
    ref = []
    for s in (100, 101, 102):
        r = B.simulate_bridge(F, H, T, D, seed=s, init="uniform", n_steps=300)
        ref += [r.pos_s[k] for k in range(100, 301)]
    S_ref = GS.flock_shape(ref, D, g)
    P, neg = GS.centre_distribution(S_ref, T, D, g)
    P_dens = P / g.h ** 2
    h2 = g.h ** 2
    info = dict(S_rms=float(np.sqrt((S_ref * (g.X ** 2 + g.Y ** 2)).sum() * h2)), pc_negative_share=neg,
                pc_rms=float(np.sqrt((P * (g.X ** 2 + g.Y ** 2)).sum())), floor=float(met.floor), burst_cohesion=BURST,
                split_rule=f"largest {SPLIT_LINK} m-linked group < {SPLIT_SHARE:.0%} of sheep", seeds=list(SEEDS),
                t_end=T_END, dwell=DWELL, substeps=H.substeps, bandwidth=T.bandwidth, sample_interval_s=F.dt)
    print(f"S RMS {info['S_rms']:.2f} m, p_c RMS {info['pc_rms']:.2f} m, negative share {neg*100:.2f}% "
          f"[{time.time()-t_start:.0f}s]", flush=True)
    res = {"info": info, "runs": {}}
    for seed in SEEDS:
        wp = GS.draw_waypoints(P, g, T_END // DWELL + 2, np.random.default_rng(1000 + seed))
        schedules = {"fixed": GS.FixedGoal(T.centre), "hop": GS.HopSchedule(wp, DWELL), "glide": GS.GlideSchedule(wp, DWELL, D)}
        for name, goal in schedules.items():
            r = B.simulate_bridge(F, H, T, D, seed=seed, init="uniform", n_steps=T_END, goal=goal)
            res["runs"].setdefault(name, []).append(analyse(r, met, g, Kb, P_dens, S_ref, D, F.n))
            a = res["runs"][name][-1]
            print(f"seed {seed} {name:5s}: time-avg error {a['from_100']['time_averaged_error']:.3f}, "
                  f"inst {a['from_100']['instantaneous_error_mean']:.2f}, occupation {a['from_100']['occupation_error']:.3f}, "
                  f"bursts {a['events_after_100']['bursts']} [{time.time()-t_start:.0f}s]", flush=True)
    # summary: mean and range over seeds
    summ = {}
    for name, rows in res["runs"].items():
        summ[name] = {}
        for win in ("from_0", "from_100", "events_after_100"):
            summ[name][win] = {k: dict(mean=float(np.mean([r_[win][k] for r_ in rows])),
                                       lo=float(np.min([r_[win][k] for r_ in rows])), hi=float(np.max([r_[win][k] for r_ in rows])))
                               for k in rows[0][win]}
    res["summary"] = summ
    OUT.mkdir(exist_ok=True)
    (OUT / "goal_schedule_validation.json").write_text(json.dumps(res, indent=1, default=float))
    print("done", f"[{time.time()-t_start:.0f}s]")


if __name__ == "__main__":
    main()
