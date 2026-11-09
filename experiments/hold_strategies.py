"""Two single-dog ways to hold a flock at the goal, tested on the published sheep.

A. Reach-and-release. `model.Params.hold_radius` adds a hold mode to the published dog:
   once the centroid is within hold_radius of the target the dog withdraws to
   hold_standoff from the centroid (beyond the sheep's notice radius) and waits; it
   re-engages when the centroid drifts beyond hold_release. Tested on the closed-loop task
   of experiments/closed_loop_convergence.py (start 100 m away), with the published sheep,
   which stop when no dog is in range, and with the paper's modified sheep
   (`social_always`), which keep moving, so the rule has to re-engage.

B. A single dog touring Di Lorenzo's ring. The herder density the deconvolution gives for
   the Gaussian goal (experiments/target_density.py) is a ring; by symmetry its density is
   uniform on the ring, so a dog running round it at constant speed is the time-multiplexed
   version of that herder density. Tested on the Jadhav sheep (published rules, the dog's
   path replayed) and on Di Lorenzo's diffusing targets, against lap speed and against the
   static ring of 10 dogs.

    .venv/bin/python experiments/hold_strategies.py
Writes results/hold_strategies.json, results/hold_replay.json and
results/figures/hold_strategies.png.
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

import closed_loop_convergence as C  # noqa: E402
import target_density as TD  # noqa: E402
from shepherd.model import HOLD, SLOW, Params, simulate  # noqa: E402

OUT = ROOT / "results"
SEED = 11
R_A, T_A, HOLD_R, HOLD_REL, AFTER = 200, 450, 5.0, 10.0, 300
R_B, T_B = TD.R, TD.T
TOUR_SPEEDS = (0.0, 1.5, 3.0, 6.0, 12.0)
COL = TD.COL


def fl(x):
    return TD.fl(x)


# ------------------------------------------------------------------ A. reach-and-release

def episodes(flags):
    """Number of runs of True along the last axis, per row."""
    f = flags.astype(int)
    return (np.diff(f, axis=1) == 1).sum(1) + f[:, 0]


def part_a():
    out, series, replay = {}, {}, None
    for sheep in ("published", "always_moving"):
        for ctrl in ("published_dog", "hold_rule"):
            kw = dict(n_steps=T_A, social_always=(sheep == "always_moving"))
            if ctrl == "hold_rule":
                kw.update(hold_radius=HOLD_R, hold_release=HOLD_REL)
            res = simulate(C.base(**kw), n_runs=R_A, seed=SEED)
            g, mode = C.geometry(res), res.mode
            t_reach = C.reach_times(g)
            rho, Tn = g["rho"], T_A
            t = np.arange(Tn)[None]
            post = (t >= t_reach[:, None] + 20) & (t_reach[:, None] < Tn)
            step = np.linalg.norm(np.diff(res.pos_s, axis=-1), axis=2) > 1e-9  # (R, N, T-1)
            aligned = np.full((R_A, AFTER), np.nan)
            for r in np.flatnonzero(t_reach < Tn):
                k = min(AFTER, Tn - t_reach[r])
                aligned[r, :k] = rho[r, t_reach[r] : t_reach[r] + k]
            hold = mode == HOLD
            key = f"{sheep}/{ctrl}"
            out[key] = dict(
                success_rate=fl((t_reach < Tn).mean()),
                t_reach_median=fl(np.median(t_reach[t_reach < Tn])) if (t_reach < Tn).any() else None,
                post_rho_mean=fl(rho[post].mean()) if post.any() else None,
                post_rho_p90=fl(np.percentile(rho[post], 90)) if post.any() else None,
                post_inside_frac=fl((rho[post] < C.ARRIVE).mean()) if post.any() else None,
                post_sheep_moving_frac=fl(step[np.broadcast_to(post[:, None, 1:], step.shape)].mean()) if post.any() else None,
                post_hold_frac=fl(hold[post].mean()) if post.any() else None,
                post_slow_frac=fl((mode == SLOW)[post].mean()) if post.any() else None,
                post_maxspread_median=fl(np.median(g["maxspread"][post])) if post.any() else None,
                hold_episodes_mean=fl(episodes(hold).mean()),
                reengagements_mean=fl(np.maximum(episodes(hold) - 1, 0).mean()),
                final_rho_median=fl(np.median(rho[:, -1])),
            )
            series[key] = dict(median=np.nanmedian(aligned, 0).tolist(), p25=np.nanpercentile(aligned, 25, 0).tolist(), p75=np.nanpercentile(aligned, 75, 0).tolist())
            o = out[key]
            print(f"  A {key:28s} reached {o['success_rate']:.2f} at {o['t_reach_median']}; after: inside {o['post_inside_frac']:.2f}, "
                  f"rho mean {o['post_rho_mean']:.1f} p90 {o['post_rho_p90']:.1f}, sheep moving {o['post_sheep_moving_frac']:.2f}, "
                  f"hold episodes {o['hold_episodes_mean']:.2f}, re-engagements {o['reengagements_mean']:.2f}", flush=True)
            if sheep == "published" and ctrl == "hold_rule":
                r0 = int(np.flatnonzero(t_reach < Tn)[0])
                pos, _ = res.run(r0)
                G0 = pos[:14, :, 0].mean(0)
                ang = np.arctan2(-G0[1], -G0[0])
                c, sn = np.cos(-ang), np.sin(-ang)
                pos = np.einsum("ij,njt->nit", np.array([[c, -sn], [sn, c]]), pos)
                replay = dict(key="hold_rule", label="Reach-and-release: the dog withdraws at 5 m and waits", N=14, n_dogs=1, static=False,
                              r_goal=HOLD_R, l_sep=fl(g["p"].l_sep), l_drive=fl(g["p"].l_drive), r_dog=fl(g["p"].r_dog), v_dog=fl(g["p"].v_dog),
                              v_sheep=1.0, arrive=C.ARRIVE, target=[0.0, 0.0], t_reach=int(t_reach[r0]),
                              pos=np.round(np.moveaxis(pos, -1, 0), 2).tolist(), mode=mode[r0].astype(int).tolist(),
                              rho=np.round(rho[r0], 2).tolist(), spread=np.round(g["maxspread"][r0], 2).tolist(), ppar=np.round(g["P_par"][r0], 3).tolist())
    return out, series, replay


# ------------------------------------------------------------------ B. one dog touring the ring

def tour_path(r_ring, v, T, phase0=np.pi):
    th = phase0 + (v / r_ring) * np.arange(T + 1)
    return np.stack([r_ring * np.cos(th), r_ring * np.sin(th)])  # (2, T+1)


def part_b(ring, s, grid, floor):
    rows, series, replay = [], {}, []
    r_ring = ring["r_ring"]
    for start in ("scattered", "goal"):
        for v in TOUR_SPEEDS:
            path = tour_path(r_ring, v, T_B)
            # Jadhav sheep: published rules, the dog's path replayed
            x0, h0 = TD.starts(np.random.default_rng(SEED + 5), start, r_ring)
            res = simulate(Params(), n_runs=R_B, seed=SEED + 6, init={"pos_s": x0, "vel_s": h0}, dog_path=path)
            m = TD.metrics(res.pos_s, s, grid, floor)
            sees = np.linalg.norm(res.pos_s - res.pos_d[:, None], axis=2) <= TD.R_DOG
            r_end = np.linalg.norm(res.pos_s[..., -50:], axis=2)
            rows.append(dict(model="jadhav_sheep", start=start, v_tour=v, lap_s=fl(2 * np.pi * r_ring / v) if v else None,
                             final_frac_in=m["final_frac_in"], final_rms=m["final_rms"], final_l2=m["final_l2"],
                             escaped_frac=fl((r_end > r_ring + 2).mean()), sees_frac=fl(sees[..., -100:].mean()), t_10pct=m["t_10pct"]))
            # Di Lorenzo targets under the same single dog (full push)
            dogs_path = np.moveaxis(path, 0, -1)[:, None, :]
            pos = TD.run_level(TD.LEVELS[0][1], dogs_path, np.random.default_rng(SEED + 7), start, r_ring)
            m0 = TD.metrics(pos, s, grid, floor)
            r_end0 = np.linalg.norm(pos[..., -50:], axis=2)
            rows.append(dict(model="targets", start=start, v_tour=v, lap_s=fl(2 * np.pi * r_ring / v) if v else None,
                             final_frac_in=m0["final_frac_in"], final_rms=m0["final_rms"], final_l2=m0["final_l2"],
                             escaped_frac=fl((r_end0 > r_ring + 2).mean()), sees_frac=None, t_10pct=m0["t_10pct"]))
            print(f"  B {start:9s} v={v:4.1f}: sheep inside {m['final_frac_in']:.2f} rms {m['final_rms']:.2f} escaped {rows[-2]['escaped_frac']:.2f} "
                  f"sees {rows[-2]['sees_frac']:.2f} | targets inside {m0['final_frac_in']:.2f} rms {m0['final_rms']:.2f} escaped {rows[-1]['escaped_frac']:.2f}", flush=True)
            if v == 1.5:
                series[f"{start}/sheep"] = m["frac_in"]
                series[f"{start}/targets"] = m0["frac_in"]
                if start == "scattered":
                    T1 = T_B + 1
                    for key, label, P, rd in (("tour_jadhav", "One dog touring the ring at 1.5 m/s: Jadhav sheep", res.pos_s[0], TD.R_DOG),
                                              ("tour_targets", "One dog touring the ring at 1.5 m/s: Di Lorenzo's targets", pos[0], None)):
                        allp = np.concatenate([np.moveaxis(P, -1, 0), np.moveaxis(path, 0, -1)[:, None, :]], 1)
                        replay.append(dict(key=key, label=label, N=14, n_dogs=1, static=True, r_goal=TD.R_GOAL, l_sep=None, l_drive=None, r_dog=rd,
                                           pos=np.round(allp, 2).tolist(), mode=[6] * T1,
                                           rho=np.round(np.linalg.norm(P.mean(0), axis=0), 2).tolist(),
                                           spread=np.round(np.linalg.norm(P - P.mean(0, keepdims=True), axis=1).max(0), 2).tolist()))
    # references: the static ring of 10 dogs
    ref = {}
    for key, level in (("sheep", TD.LEVELS[5][1]), ("targets", TD.LEVELS[0][1])):
        for start in ("scattered", "goal"):
            pos = TD.run_level(level, TD.ring_dogs(TD.M_LADDER, r_ring), np.random.default_rng(SEED + 8), start, r_ring)
            m = TD.metrics(pos, s, grid, floor)
            ref[f"{start}/{key}"] = dict(final_frac_in=m["final_frac_in"], final_rms=m["final_rms"])
    return rows, series, ref, replay


# ------------------------------------------------------------------ figure

def figure(a, a_series, b_rows, b_series, b_ref, ring, out_path):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    k = np.arange(AFTER)
    for ax, sheep, title in ((axs[0, 0], "published", "Published sheep (stop without a dog in range)"), (axs[0, 1], "always_moving", "Modified sheep (always moving)")):
        for ctrl, c, lab in (("published_dog", "orange", "published dog"), ("hold_rule", "blue", "reach-and-release")):
            sr = a_series[f"{sheep}/{ctrl}"]
            ax.fill_between(k, sr["p25"], sr["p75"], color=COL[c], alpha=0.18, lw=0)
            ax.plot(k, sr["median"], color=COL[c], lw=2, label=lab)
        ax.axhline(C.ARRIVE, color=COL["gray"], lw=1, ls=":")
        ax.set(xlabel="steps after first reaching 5 m", ylabel="centroid distance to target (m)", title=title)
        ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 2]
    keys = ["published/published_dog", "published/hold_rule", "always_moving/published_dog", "always_moving/hold_rule"]
    labels = ["published sheep\npublished dog", "published sheep\nhold rule", "moving sheep\npublished dog", "moving sheep\nhold rule"]
    vals = [a[kk]["post_inside_frac"] for kk in keys]
    ax.bar(range(4), vals, color=[COL["orange"], COL["blue"], COL["orange"], COL["blue"]], width=0.6)
    for i, kk in enumerate(keys):
        ax.text(i, vals[i] + 0.02, f"re-engage {a[kk]['reengagements_mean']:.1f}×", ha="center", fontsize=8, color=COL["gray"])
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set(ylabel="fraction of post-reach time within 5 m", ylim=(0, 1.1), title="A. Does the flock stay at the goal?")

    ax = axs[1, 0]
    for model, c, mk in (("jadhav_sheep", "red", "o"), ("targets", "blue", "s")):
        for start, ls in (("scattered", "-"), ("goal", "--")):
            rows = [r for r in b_rows if r["model"] == model and r["start"] == start]
            ax.plot([r["v_tour"] for r in rows], [r["final_frac_in"] for r in rows], mk + ls, color=COL[c], label=f"{'Jadhav sheep' if model == 'jadhav_sheep' else 'their targets'}, {start} start")
    ax.axhline(b_ref["scattered/sheep"]["final_frac_in"], color=COL["red"], lw=1, ls=":", label="static ring of 10 dogs, sheep")
    ax.axhline(b_ref["scattered/targets"]["final_frac_in"], color=COL["blue"], lw=1, ls=":", label="static ring of 10 dogs, targets")
    ax.set(xlabel="tour speed of the single dog (m/s); 0 = standing on the ring", ylabel="final fraction inside r_goal", ylim=(0, 1.05), title="B. One dog touring the ring")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1, 1]
    t = np.arange(T_B + 1)
    for key, c, ls, lab in (("scattered/sheep", "red", "-", "Jadhav sheep, scattered start"), ("scattered/targets", "blue", "-", "their targets, scattered start"),
                            ("goal/sheep", "red", "--", "Jadhav sheep, goal start"), ("goal/targets", "blue", "--", "their targets, goal start")):
        ax.plot(t, b_series[key], color=COL[c], ls=ls, lw=1.8, label=lab)
    ax.axhline(0.95, color=COL["gray"], lw=1, ls=":")
    ax.set(xlabel="time (s)", ylabel="fraction inside r_goal", ylim=(0, 1.05), title=f"B. Tour at 1.5 m/s (a lap every {2 * np.pi * ring['r_ring'] / 1.5:.0f} s)")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[1, 2]
    for model, c, mk in (("jadhav_sheep", "red", "o"), ("targets", "blue", "s")):
        rows = [r for r in b_rows if r["model"] == model and r["start"] == "scattered"]
        ax.plot([r["v_tour"] for r in rows], [r["escaped_frac"] for r in rows], mk + "-", color=COL[c], label=f"{'Jadhav sheep' if model == 'jadhav_sheep' else 'their targets'}: escaped beyond the ring")
    rows = [r for r in b_rows if r["model"] == "jadhav_sheep" and r["start"] == "scattered"]
    ax.plot([r["v_tour"] for r in rows], [r["sees_frac"] for r in rows], "^-", color=COL["green"], label="Jadhav sheep: share that sees the dog")
    ax.set(xlabel="tour speed (m/s)", ylabel="fraction", ylim=(0, 1.05), title="B. Leakage and coverage (scattered start)")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Holding a flock at the goal with one dog: reach-and-release (A) and touring Di Lorenzo's ring (B)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    print("A. reach-and-release", flush=True)
    a, a_series, replay_a = part_a()
    print("B. touring dog", flush=True)
    ring = TD.herder_ring()
    s, grid = TD.goal_sigma(), TD.kde_grid()
    floor = float(TD.l2_error(s * np.random.default_rng(SEED).standard_normal((200, 14, 2)), s, grid).mean())
    b_rows, b_series, b_ref, replay_b = part_b(ring, s, grid, floor)
    for kk, v in b_ref.items():
        print(f"  B reference static ring, {kk}: inside {v['final_frac_in']:.2f} rms {v['final_rms']:.2f}", flush=True)
    out = dict(setup=dict(seed=SEED, runs_a=R_A, steps_a=T_A, hold_radius=HOLD_R, hold_release=HOLD_REL, hold_standoff=fl(Params().r_dog + 3),
                          runs_b=R_B, steps_b=T_B, r_goal=TD.R_GOAL, tour_speeds=list(TOUR_SPEEDS), ring=ring),
               reach_and_release=a, reach_and_release_series=a_series, tour=b_rows, tour_series={k: list(v) for k, v in b_series.items()},
               tour_static_ring_reference=b_ref, l2_floor=floor, seconds=fl(time.time() - tic))
    (OUT / "hold_strategies.json").write_text(json.dumps(out, indent=1))
    (OUT / "hold_replay.json").write_text(json.dumps([replay_a] + replay_b, separators=(",", ":")))
    figure(a, a_series, b_rows, b_series, b_ref, ring, OUT / "figures" / "hold_strategies.png")
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
