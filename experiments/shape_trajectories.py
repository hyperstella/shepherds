"""Trajectories from stressed flock states: does the flock recover, or leave the drift lemma's region first?

The drift lemma (docs/steer_and_shape.md §10) is a one-step statement on the region G (every sheep
moves, weighted inward relative heading kappa >= kappa*, mean-heading polarization P >= P*). This
script runs the full published model forward from the stressed states of experiments/shape_stress.py,
with two dog policies (the published dog driving at 7.5 m with collecting off, and with Strömbom
collecting on), and records for each trajectory:

  sigma  first step with sqrt(M2) <= rho (recovery into the ball where the lemma stops promising
         negative drift; rho = (b + eps)/a with a = 2 kappa* gamma*, b = 1 - P*^2, from the
         observed states of the setting)
  tau    first evaluated step outside G, and why (some sheep frozen / kappa < kappa* / P < P*);
         tau = 0 when the stressed state already starts outside G. kappa and P are Monte Carlo
         estimates, so an exit is counted when the estimate is below its threshold by more than
         two standard errors (the strict count, on point estimates, is reported too)
  which comes first, recovery times, and, after an exit, what follows by the end of the run:
  renewed cohesion (recovers later), freezing (most sheep stopped at the end), continued
  spreading (sqrt(M2) up by more than 1 m since the exit, or split), or neither
  splits at the end: the largest group linked at 6 m holds fewer than N - 1 sheep

kappa, P and E[dM2 | state] (with its Monte Carlo standard error) are estimated every EVERY steps
from K one-step draws, as in experiments/shape_drift.py.

    .venv/bin/python experiments/shape_trajectories.py
Writes results/shape_trajectories.json and results/figures/shape_trajectories.png.
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
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
T, EVERY, K_TRAJ, SEED = 150, 3, 32, 61
EPS = 0.02  # drift margin defining the ball
LINK = 6.0  # m, linking distance for groups
SETTINGS = [s for s in SS.SETTINGS if s["N"] == 14]
CONDITIONS = [("dilated", None, 1), ("stretch across", "stretch_perp", 3.0), ("stretch along", "stretch_par", 3.0),
              ("split across", "split_perp", 8.0), ("split across", "split_perp", 12.0), ("split along", "split_par", 12.0),
              ("outlier to the side", "out_side", 14.0), ("outlier ahead", "out_front", 10.0), ("heading noise", "head_noise", 3.0),
              ("half reversed", "head_reverse", 1)]
POLICIES = {"drive only": dict(l_sep=1e6), "with collecting": dict()}
COL = SS.COL


def dilate2(pos):
    c = pos.mean(1, keepdims=True)
    return c + 2.0 * (pos - c)


def largest_group(pos):
    """Size of the largest group linked at LINK m (transitive closure of the proximity graph). pos (S, N, 2)."""
    d = np.linalg.norm(pos[:, None] - pos[:, :, None], axis=-1)
    reach = d <= LINK
    for _ in range(4):  # 2^4 > N hops
        reach = (reach.astype(np.int32) @ reach.astype(np.int32)) > 0
    return reach.sum(-1).max(-1)


def ball_radius(st, pos, hd, dog, rng):
    """rho = (b + EPS) / a from the observed states: kappa* = SS.KAPPA_STAR, P* and gamma* their 10th percentiles."""
    f = SS.analyse(pos, SS.one_step(st, pos, hd, dog, SEED, {}))
    xi = pos - pos.mean(1, keepdims=True)
    r = np.linalg.norm(xi, axis=-1)
    gamma = r.mean(1) / np.sqrt((r ** 2).mean(1))
    P_star, g_star = float(np.percentile(f["P"], 10)), float(np.percentile(gamma, 10))
    a, b = 2 * SS.KAPPA_STAR * g_star * SS.V, SS.V ** 2 * (1 - P_star ** 2)
    return dict(rho=(b + EPS) / a, a=a, b=b, P_star=P_star, gamma_star=g_star, sqrtM2_observed=float(np.median(np.sqrt((r ** 2).mean(1)))))


def run_condition(st, pos0, hd0, dog0, ball, kind, level, policy, rng, seed):
    S, N = pos0.shape[:2]
    if kind is None:
        pos, hd = dilate2(pos0), hd0.copy()
    else:
        pos, hd = SS.deform(kind, level, pos0, hd0, rng)
    per = dict(w_att=np.full(S, st["w_att"]), w_ali=np.full(S, st["w_ali"]), r_dog=np.full(S, st["r_dog"]), l_drive=np.full(S, SS.D_STAND))
    per.update({k: np.full(S, v) for k, v in POLICIES[policy].items()})
    vel_d = SS.unit(pos.mean(1) - dog0)
    res = simulate(Params(n_sheep=N, n_steps=T + 1), n_runs=S, seed=seed, per_run=per, init=dict(pos_s=pos, vel_s=hd, pos_d=dog0, vel_d=vel_d))
    X, H, Dg = res.pos_s, res.vel_s, res.pos_d
    sqrtM2 = np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum(2).mean(1))  # (S, T+1)
    moving = np.linalg.norm(np.diff(X, axis=-1), axis=2) > 1e-9  # (S, N, T)
    group = np.stack([largest_group(X[..., t]) for t in range(0, T + 1, EVERY)], 1)
    ev = list(range(0, T + 1, EVERY))
    kap, kap_se, P, P_se, fm, dM2, dM2_se = (np.full((S, len(ev)), np.nan) for _ in range(7))
    for j, t in enumerate(ev):
        f = SS.analyse(X[..., t], one_step_k(st, X[..., t], H[..., t], Dg[..., t], seed * 31 + t))
        kap[:, j], kap_se[:, j], P[:, j], P_se[:, j], fm[:, j], dM2[:, j], dM2_se[:, j] = (
            f["kappa"], f["kappa_se"], f["P"], f["P_se"], f["f_m"], f["dM2"], f["dM2_se"])
    rho = ball["rho"]
    rec = sqrtM2 <= rho
    sigma = np.where(rec.any(1), rec.argmax(1), np.nan)
    out_cov = fm < 0.999
    out_kap = ~out_cov & (kap + 2 * kap_se < SS.KAPPA_STAR)
    out_P = P + 2 * P_se < ball["P_star"]
    outside = out_cov | out_kap | out_P
    strict = out_cov | (~out_cov & (kap < SS.KAPPA_STAR)) | (P < ball["P_star"])
    ev = np.array(ev)
    tau = np.where(outside.any(1), ev[outside.argmax(1)], np.nan)
    tau_strict = np.where(strict.any(1), ev[strict.argmax(1)], np.nan)
    rows = []
    for s in range(S):
        sg, ta = sigma[s], tau[s]
        if np.isfinite(sg) and (not np.isfinite(ta) or sg <= ta):
            first = "recovered first" if sg > 0 else "starts inside the ball"
        elif np.isfinite(ta):
            first = "starts outside G" if ta == 0 else "left G first"
        else:
            first = "neither by the end"
        cause = None
        if np.isfinite(ta) and first in ("left G first", "starts outside G"):
            j = int(np.searchsorted(ev, ta))
            cause = "+".join(n for n, flag in (("frozen sheep", out_cov[s, j]), ("kappa", out_kap[s, j]), ("polarization", out_P[s, j])) if flag)
        after = None
        if first in ("left G first", "starts outside G"):
            ta_i = int(ta)
            later = rec[s, ta_i:]
            if later.any():
                after = "renewed cohesion"
            elif moving[s, :, -20:].mean() < 0.5:
                after = "freezing"
            elif sqrtM2[s, -1] > sqrtM2[s, ta_i] + 1.0 or group[s, -1] < N - 1:
                after = "continued spreading"
            else:
                after = "neither (hovering)"
        ts = tau_strict[s]
        first_strict = ("recovered first" if sg > 0 else "starts inside the ball") if np.isfinite(sg) and (not np.isfinite(ts) or sg <= ts) else \
            (("starts outside G" if ts == 0 else "left G first") if np.isfinite(ts) else "neither by the end")
        inside0 = first == "starts inside the ball"
        rows.append(dict(first=first, first_strict=first_strict, sigma=None if not np.isfinite(sg) else int(sg), tau=None if not np.isfinite(ta) else int(ta), cause=cause, after=after,
                         left_ball=bool(inside0 and (sqrtM2[s] > rho).any()), excursion=float(sqrtM2[s].max() - rho) if inside0 else None,
                         split_end=bool(group[s, -1] < N - 1), straggler_end=bool(group[s, -1] == N - 1), sqrtM2_start=float(sqrtM2[s, 0]),
                         sqrtM2_end=float(sqrtM2[s, -1]), recovered_ever=bool(rec[s].any()),
                         recovery_time=None if not rec[s].any() else int(rec[s].argmax())))
    drift0 = dict(median=float(np.median(dM2[:, 0])), se_median=float(np.median(dM2_se[:, 0])), ci=SS.boot_ci(dM2[:, 0], rng))
    return rows, dict(sqrtM2_mean=np.nanmean(sqrtM2, 0)[::EVERY].tolist(), moving_mean=moving.mean((0, 1)).tolist()), drift0


def one_step_k(st, pos, hd, dog, seed):
    """SS.one_step with K_TRAJ draws."""
    old = SS.K
    SS.K = K_TRAJ
    try:
        return SS.one_step(st, pos, hd, dog, seed, {})
    finally:
        SS.K = old


def summarize(rows):
    n = len(rows)
    count = lambda key, val: sum(r[key] == val for r in rows) / n
    rt = [r["sigma"] for r in rows if r["first"] == "recovered first"]
    rt_all = [r["recovery_time"] for r in rows if r["recovery_time"] is not None and r["first"] != "starts inside the ball"]
    exits = [r for r in rows if r["first"] in ("left G first", "starts outside G")]
    causes = {}
    for r in exits:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    after = {}
    for r in exits:
        after[r["after"]] = after.get(r["after"], 0) + 1
    return dict(n=n, **{f"frac_{k.replace(' ', '_')}": count("first", k) for k in ("recovered first", "left G first", "starts outside G", "neither by the end", "starts inside the ball")},
                recovery_time_median=float(np.median(rt)) if rt else None, recovery_time_p90=float(np.percentile(rt, 90)) if rt else None,
                recovered_ever=sum(r["recovered_ever"] for r in rows) / n,
                recovery_time_any_median=float(np.median(rt_all)) if rt_all else None,
                recovery_time_any_p90=float(np.percentile(rt_all, 90)) if rt_all else None,
                frac_left_G_first_strict=sum(r["first_strict"] == "left G first" for r in rows) / n,
                frac_starts_outside_G_strict=sum(r["first_strict"] == "starts outside G" for r in rows) / n,
                inside_left_ball=(sum(r["left_ball"] for r in rows) / max(1, sum(r["first"] == "starts inside the ball" for r in rows))),
                inside_excursion_median=float(np.median([r["excursion"] for r in rows if r["excursion"] is not None])) if any(r["excursion"] is not None for r in rows) else None,
                exit_causes=causes, after_exit={k: v / max(len(exits), 1) for k, v in after.items()}, n_exits=len(exits),
                split_end=sum(r["split_end"] for r in rows) / n, straggler_end=sum(r["straggler_end"] for r in rows) / n)


def main():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    out = dict(setup=dict(T=T, every=EVERY, K=K_TRAJ, eps=EPS, link_m=LINK, kappa_star=SS.KAPPA_STAR, policies=POLICIES,
                          rule="published (stop when no dog within r_dog)", conditions=[(c, k, l) for c, k, l in CONDITIONS]), settings=[])
    seed = SEED * 100
    for st in SETTINGS:
        pos0, hd0, dog0 = SS.observed_states(st)
        ball = ball_radius(st, pos0, hd0, dog0, rng)
        print(f"{st['name']}: rho {ball['rho']:.2f} m (observed √M2 {ball['sqrtM2_observed']:.2f}; a {ball['a']:.3f}, b {ball['b']:.3f}, P* {ball['P_star']:.2f})", flush=True)
        srow = dict(setting=st["name"], ball=ball, conditions=[])
        for cname, kind, level in CONDITIONS:
            for pol in POLICIES:
                seed += 1
                rows, series, drift0 = run_condition(st, pos0, hd0, dog0, ball, kind, level, pol, rng, seed)
                sm = summarize(rows)
                srow["conditions"].append(dict(condition=cname, level=level, policy=pol, summary=sm, drift_at_start=drift0, series=series, runs=rows))
                print(f"  {cname:20s} {level:>5} {pol:15s}: start E[dM2] {drift0['median']:+.2f} {[round(v, 2) for v in drift0['ci']]} | "
                      f"in ball {sm['frac_starts_inside_the_ball']:.2f} (left it {sm['inside_left_ball']:.2f}) rec first {sm['frac_recovered_first']:.2f} "
                      f"left G first {sm['frac_left_G_first']:.2f} [strict {sm['frac_left_G_first_strict']:.2f}] start out {sm['frac_starts_outside_G']:.2f} "
                      f"[strict {sm['frac_starts_outside_G_strict']:.2f}] neither {sm['frac_neither_by_the_end']:.2f} | rec time {sm['recovery_time_any_median']} p90 {sm['recovery_time_any_p90']} "
                      f"(ever {sm['recovered_ever']:.2f}) | causes {sm['exit_causes']} | after {({k: round(v, 2) for k, v in sm['after_exit'].items()})} | "
                      f"split {sm['split_end']:.2f} straggler {sm['straggler_end']:.2f}", flush=True)
        out["settings"].append(srow)
        print(f"  ({time.time() - tic:.0f} s)", flush=True)
    out["thirty"] = thirty(rng)
    out["seconds"] = time.time() - tic
    (OUT / "shape_trajectories.json").write_text(json.dumps(out, indent=1))
    figure(out)
    print(f"done in {time.time() - tic:.0f} s")


def thirty(rng, S=30, N=30):
    """Synthetic compact 30-sheep flocks (disc of radius 0.9 sqrt(N) m, aligned, dog 7.5 m behind), published weights, r_dog 20."""
    out = []
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
    c0 = np.array([SS.FAR, 0.0])
    pos = c0 + np.array(pts)
    hd = SS.unit(np.array([-1.0, 0.0]) + 0.1 * rng.standard_normal((S, N, 2)))
    for pol in POLICIES:
        per = dict(w_att=np.full(S, 1.5), w_ali=np.full(S, 1.3), r_dog=np.full(S, 20.0), l_drive=np.full(S, SS.D_STAND))
        per.update({k: np.full(S, v) for k, v in POLICIES[pol].items()})
        res = simulate(Params(n_sheep=N, n_steps=T + 1), n_runs=S, seed=SEED + 7, per_run=per,
                       init=dict(pos_s=pos, vel_s=hd, pos_d=np.tile(c0 + [SS.D_STAND, 0.0], (S, 1)), vel_d=np.tile([-1.0, 0.0], (S, 1))))
        X = res.pos_s
        sq = np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
        moving = np.linalg.norm(np.diff(X, axis=-1), axis=2) > 1e-9
        grp = largest_group(X[..., -1])
        row = dict(policy=pol, sqrtM2_at=[float(np.median(sq[:, t])) for t in (0, 25, 50, 100, T)], moving_at=[float(moving[..., t].mean()) for t in (0, 25, 50, 100, T - 1)],
                   split_end=float((grp < N - 1).mean()), largest_group_median=float(np.median(grp)))
        out.append(row)
        print(f"  30 sheep, {pol:15s}: median √M2 at t=0,25,50,100,{T}: {[round(v, 1) for v in row['sqrtM2_at']]} | moving {[round(v, 2) for v in row['moving_at']]} | "
              f"split {row['split_end']:.2f}, largest group {row['largest_group_median']:.0f}", flush=True)
    return out


def outcome(r):
    if r["first"] == "starts inside the ball":
        return "started in the ball"
    if r["first"] == "recovered first":
        return "recovered before leaving G"
    if r["after"] == "renewed cohesion":
        return "outside G first, then recovered"
    if r["after"] == "freezing":
        return "outside G first, then froze"
    if r["after"] == "continued spreading":
        return "outside G first, then kept spreading"
    return "other"


OUTCOMES = [("started in the ball", "#c9c8c4"), ("recovered before leaving G", COL["green"]), ("outside G first, then recovered", COL["blue"]),
            ("outside G first, then froze", COL["orange"]), ("outside G first, then kept spreading", COL["red"]), ("other", COL["gray"])]


def figure(out):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, srow in zip(axs, out["settings"]):
        labels, data = [], []
        for c in srow["conditions"]:
            labels.append(f"{c['condition']} {c['level']:g}, {'collect' if c['policy'] == 'with collecting' else 'drive only'}")
            oc = [outcome(r) for r in c["runs"]]
            data.append([oc.count(k) / len(oc) for k, _ in OUTCOMES])
        data = np.array(data)
        left = np.zeros(len(data))
        for j, (k, col) in enumerate(OUTCOMES):
            ax.barh(range(len(data)), data[:, j], left=left, color=col, label=k, height=0.8)
            left += data[:, j]
        for i, c in enumerate(srow["conditions"]):
            ax.text(1.02, i, f"split {c['summary']['split_end']:.2f}", va="center", fontsize=6.5, color="#3a3a38")
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1.22), title=f"{srow['setting']} (ball ρ = {srow['ball']['rho']:.1f} m)", xlabel="fraction of trajectories (150 steps)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "figures" / "shape_trajectories.png", dpi=130)


if __name__ == "__main__":
    main()
