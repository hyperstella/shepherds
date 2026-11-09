"""One Jadhav dog under a Di Lorenzo-inspired density controller (shepherd/density_control.py).

The sheep are the published model; the dog is placed each step at the candidate position
that most lowers the expected one-step change of a density-error functional, computed
exactly from the sheep's rule, and moves toward it at v_dog. Two functionals: the second
moment about the goal ('moment', the potential energy of the Gaussian target; exact) and
Di Lorenzo's L² density error ('l2'; first order in the step).

A. The closed-loop task of experiments/hold_strategies.py: start 100 m from the goal, 450
   steps, with the published sheep (they stop without a dog in range) and the modified sheep
   (`social_always`, they keep moving). Compared with the published dog and the
   reach-and-release rule (results/hold_strategies.json). The L² functional has no gradient
   100 m from a 5 m goal (its goal term is a Gaussian), so it is run only briefly there.
B. At the goal, the task of experiments/target_density.py: the Gaussian goal (95% within
   5 m), scattered and displaced-cluster starts, 200 steps, one dog starting on the ring
   where their 10 static dogs stand. Compared with that ring (ladder rung L5) and with the
   published dog started at the goal (L6), on their metrics: fraction inside 5 m, RMS
   distance to the goal, relative L² error of the kernel density estimate.
C. The placement landscape: E[ΔV | x, h] over dog positions h around one driven flock.

Diagnostics on every step: the best score (is a descent direction available?), the score of
the position the dog actually took (the motion layer's lag), the realized change of V against
the expected one, and whether the dog has released the flock (no sheep sees it).

Three controllers: 'moment' and 'l2' as above, and 'hybrid', the second moment while the
centroid is more than 5 m from the goal and the L² error inside (drive, then regulate).

    .venv/bin/python experiments/density_controller.py [--part A|B|C|fig] [--only KEY] [--quick]
Parts and variants merge into results/density_controller.json, so they can run in parallel;
`--part fig` draws results/figures/density_controller.png from it.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
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
from shepherd import density_control as DC  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
JSON_PATH, FIG_PATH = OUT / "density_controller.json", OUT / "figures" / "density_controller.png"
SEED = 11
R_A, T_A, AFTER = 48, 450, 300  # hold_strategies used 200 runs; the exact placement costs ~1.3 s per step for 48
T_A_L2, R_A_L2 = 100, 16
R_B, T_B = TD.R, TD.T
R_GOAL = TD.R_GOAL
S_GOAL = DC.goal_sigma(R_GOAL)
COL = TD.COL


def fl(x):
    return TD.fl(x)


def functional(kind):
    return DC.Functional(kind, goal=(0.0, 0.0), s=S_GOAL, b=TD.KDE_H)


def diagnostics(rec, t_reach=None):
    """How the placement and motion layers did, over all steps or after reaching."""
    dV = rec["dV"]  # realized change of V over the step, under the functional that placed the dog
    sel = np.ones_like(rec["V"], bool)
    if t_reach is not None:
        sel = np.arange(rec["V"].shape[1])[None] >= t_reach[:, None] + 20
    return dict(
        descent_available_frac=fl((rec["score_best"] < 0)[sel].mean()),
        descent_taken_frac=fl((rec["score_taken"] < 0)[sel].mean()),
        released_frac=fl(rec["released"][sel].mean()),
        coverage_mean=fl(rec["coverage"][sel].mean()),
        gap_mean=fl(rec["gap"][sel].mean()),
        gap_p90=fl(np.percentile(rec["gap"][sel], 90)),
        expected_dV_mean=fl(rec["score_taken"][sel].mean()),
        realized_dV_mean=fl(dV[sel].mean()),
        expected_dV_at_hstar_mean=fl(rec["score_taken"][sel & (rec["gap"] < 1e-9)].mean()) if (sel & (rec["gap"] < 1e-9)).any() else None,
        realized_dV_at_hstar_mean=fl(dV[sel & (rec["gap"] < 1e-9)].mean()) if (sel & (rec["gap"] < 1e-9)).any() else None,
        at_hstar_frac=fl((rec["gap"] < 1e-9)[sel].mean()),
        near_active_frac=fl(rec["which"][sel].mean()),
    )


HORIZON = 8  # steps of extrapolation for 'moment8'


V_DOG_FAST = 3.0  # the '_fast' variants: the dog speed the tuned designs of continuification_driving §12 used


def controller(kind):
    """'moment', 'l2', 'moment8' (the second moment of positions extrapolated HORIZON steps along the headings),
    or 'hybrid': the second moment beyond R_GOAL of the goal and the L² error inside. A '_fast' suffix
    only changes the dog's speed (see params_for)."""
    kind = kind.removesuffix("_fast")
    if kind == "hybrid":
        return DC.Controller(functional("moment"), near=functional("l2"), switch_radius=R_GOAL)
    if kind == "moment8":
        return DC.Controller(DC.Functional("moment", goal=(0.0, 0.0), s=S_GOAL, horizon=HORIZON))
    return DC.Controller(functional(kind))


def params_for(kind, p):
    return replace(p, v_dog=V_DOG_FAST) if kind.endswith("_fast") else p


# ------------------------------------------------------------------ A. the closed-loop task from 100 m

A_VARIANTS = (("published", "moment"), ("published", "hybrid"), ("always_moving", "moment"), ("always_moving", "hybrid"), ("published", "l2"),
              ("published", "moment8"), ("always_moving", "moment8"), ("published", "moment_fast"), ("always_moving", "moment_fast"))


def part_a(quick=False, only=None):
    R, T = (12, 160) if quick else (R_A, T_A)
    out, series = {}, {}
    grid = TD.kde_grid()
    for sheep, kind in A_VARIANTS:
        key = f"{sheep}/density_{kind}"
        if only and only not in key:
            continue
        Rk, Tk = (min(R, R_A_L2), min(T, T_A_L2)) if kind == "l2" else (R, T)
        tic = time.time()
        p = C.base(n_steps=1, social_always=(sheep == "always_moving"))
        st = simulate(p, n_runs=Rk, seed=SEED)
        init = dict(pos_s=st.pos_s[..., 0], vel_s=st.vel_s[..., 0], pos_d=st.pos_d[..., 0])
        pos, _, rec = DC.run(controller(kind), params_for(kind, p), init, Tk, SEED)
        rho = np.linalg.norm(pos.mean(1), axis=1)[:, :Tk]  # (R, T), the centroid's distance to the goal
        t_reach = C.first_true(rho < C.ARRIVE, Tk)
        t = np.arange(Tk)[None]
        post = (t >= t_reach[:, None] + 20) & (t_reach[:, None] < Tk)
        step = np.linalg.norm(np.diff(pos, axis=-1), axis=2)[..., :Tk] > 1e-9  # (R, N, T) sheep moved this step
        aligned = np.full((Rk, AFTER), np.nan)
        for r in np.flatnonzero(t_reach < Tk):
            k = min(AFTER, Tk - t_reach[r])
            aligned[r, :k] = rho[r, t_reach[r] : t_reach[r] + k]
        rad = np.linalg.norm(pos[..., -20:], axis=2)  # (R, N, 20) distance of each sheep to the goal, last 20 steps
        reached = (t_reach < Tk).any()
        out[key] = dict(
            runs=Rk, steps=Tk, success_rate=fl((t_reach < Tk).mean()),
            t_reach_median=fl(np.median(t_reach[t_reach < Tk])) if (t_reach < Tk).any() else None,
            post_rho_mean=fl(rho[post].mean()) if post.any() else None,
            post_rho_p90=fl(np.percentile(rho[post], 90)) if post.any() else None,
            post_inside_frac=fl((rho[post] < C.ARRIVE).mean()) if post.any() else None,
            post_sheep_moving_frac=fl(step[np.broadcast_to(post[:, None], step.shape)].mean()) if post.any() else None,
            post_released_frac=fl(rec["released"][post].mean()) if post.any() else None,
            final_rho_median=fl(np.median(rho[:, -1])),
            final_frac_in=fl((rad <= R_GOAL).mean()), final_rms=fl(np.sqrt((rad ** 2).mean())),
            final_l2=fl(np.mean([TD.l2_error(pos[..., -1 - k], S_GOAL, grid).mean() for k in range(0, 20, 5)])),
            diagnostics_all=diagnostics(rec), diagnostics_post=diagnostics(rec, t_reach) if post.any() else None,
            seconds=fl(time.time() - tic),
        )
        nan = [None] * AFTER
        series[key] = dict(median=np.nanmedian(aligned, 0).tolist() if reached else nan, p25=np.nanpercentile(aligned, 25, 0).tolist() if reached else nan,
                           p75=np.nanpercentile(aligned, 75, 0).tolist() if reached else nan, rho_median=np.median(rho, 0).tolist(),
                           released=rec["released"].mean(0).tolist())
        o = out[key]
        print(f"  A {key:28s} reached {o['success_rate']:.2f} at {o['t_reach_median']}; after: inside {o['post_inside_frac']}, rho mean {o['post_rho_mean']} "
              f"p90 {o['post_rho_p90']}, moving {o['post_sheep_moving_frac']}, released {o['post_released_frac']}; final in {o['final_frac_in']:.2f} "
              f"rms {o['final_rms']:.1f} l2 {o['final_l2']:.2f}; descent avail {o['diagnostics_all']['descent_available_frac']:.2f} "
              f"taken {o['diagnostics_all']['descent_taken_frac']:.2f}; {o['seconds']:.0f} s", flush=True)
    return dict(reach=out, reach_series=series)


# ------------------------------------------------------------------ B. at the goal, against the static ring

B_VARIANTS = tuple((start, kind) for start in TD.STARTS for kind in ("moment", "l2", "moment8", "moment_fast"))


def part_b(quick=False, only=None):
    R, T = (10, 60) if quick else (R_B, T_B)
    TD.T = T  # TD.metrics reads the horizon from its module
    grid, floor = TD.kde_grid(), None
    ring = TD.herder_ring()
    rng = np.random.default_rng(TD.SEED)
    goal_sample = S_GOAL * rng.standard_normal((400, TD.N, 2))
    floor = fl(TD.l2_error(goal_sample, S_GOAL, grid).mean())
    out = {}
    for start, kind in B_VARIANTS:
        key = f"{start}/density_{kind}"
        if only and only not in key:
            continue
        tic = time.time()
        rng = np.random.default_rng(TD.SEED + 1)
        x, h = TD.starts(rng, start, ring["r_ring"])  # draws TD.R starts; the first R are used
        x, h = x[:R], h[:R]
        init = dict(pos_s=x, vel_s=h, pos_d=np.tile(np.array([[-ring["r_ring"], 0.0]]), (R, 1)))
        p = params_for(kind, Params(n_steps=T + 1))
        pos, _, rec = DC.run(controller(kind), p, init, T, SEED + 3)
        m = TD.metrics(pos, S_GOAL, grid, floor)
        m.update(runs=R, steps=T, diagnostics=diagnostics(rec), seconds=fl(time.time() - tic),
                 released_frac_series=rec["released"].mean(0).tolist(), coverage_series=rec["coverage"].mean(0).tolist(),
                 final_released_frac=fl(rec["released"][:, -20:].mean()), final_centroid_off=fl(np.linalg.norm(pos[..., -20:].mean(1), axis=1).mean()))
        out[key] = m
        print(f"  B {key:24s} inside {m['final_frac_in']:.2f} rms {m['final_rms']:.2f} l2 {m['final_l2']:.2f} rate {m['rate_per_s']} "
              f"t10 {m['t_10pct']}; released {m['final_released_frac']:.2f} centroid off {m['final_centroid_off']:.1f}; "
              f"descent avail {m['diagnostics']['descent_available_frac']:.2f}; {m['seconds']:.0f} s", flush=True)
    return dict(goal=out, l2_floor=floor, ring=ring)


# ------------------------------------------------------------------ C. the placement landscape

def part_c():
    res = simulate(Params(n_steps=45, box_length=60.0), n_runs=1, seed=5)  # mid-drive, about 30 m from the goal
    pos, hd, dog = res.pos_s[..., -1], res.vel_s[..., -1], res.pos_d[..., -1]
    c = pos.mean(1)
    out = {}
    for kind in ("moment", "l2"):
        ctrl = DC.Controller(functional(kind), radii=tuple(np.arange(2.0, 18.1, 1.0)), n_angles=36)
        resp = DC.ExpectedResponse(pos, hd)
        h, _, sc = ctrl.place(resp, dog)
        cand = ctrl.candidates(c, dog)
        out[kind] = dict(cand=(cand[0] - c[0]).round(2).tolist(), score=sc[0].round(5).tolist(), h_star=(h[0] - c[0]).round(2).tolist(),
                         V=fl(ctrl.functional.value(pos)[0]))
    return dict(landscape=dict(pos=(pos[0] - c[0]).round(2).tolist(), hd=hd[0].round(3).tolist(), dog=(dog[0] - c[0]).round(2).tolist(),
                               to_goal=(-c[0] / np.linalg.norm(c[0])).round(3).tolist(), rho=fl(np.linalg.norm(c[0])),
                               l_drive=fl(Params().resolved().l_drive), **out))


# ------------------------------------------------------------------ figure

def figure(out, out_path):
    hold = json.load(open(OUT / "hold_strategies.json"))
    td = json.load(open(OUT / "target_density.json"))
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    k = np.arange(AFTER)
    for ax, sheep, title in ((axs[0, 0], "published", "A. Published sheep (stop without a dog in range)"), (axs[0, 1], "always_moving", "A. Modified sheep (always moving)")):
        for key, c, lab in ((f"{sheep}/published_dog", "orange", "published dog"), (f"{sheep}/hold_rule", "gray", "reach-and-release rule")):
            sr = hold["reach_and_release_series"][key]
            ax.plot(k, sr["median"], color=COL[c], lw=1.6, ls="--", label=lab)
        for kind, c, lab in (("moment", "blue", "density controller: second moment"), ("moment8", "green", f"second moment, {HORIZON}-step lookahead"),
                             ("moment_fast", "teal", f"second moment, v_dog = {V_DOG_FAST:g}"), ("hybrid", "purple", "moment, then L² inside 5 m")):
            sr = out["reach_series"].get(f"{sheep}/density_{kind}")
            if sr is None or sr["median"][0] is None:
                continue
            ax.fill_between(k, sr["p25"], sr["p75"], color=COL[c], alpha=0.18, lw=0)
            ax.plot(k, sr["median"], color=COL[c], lw=2, label=lab)
        ax.axhline(C.ARRIVE, color=COL["gray"], lw=1, ls=":")
        ax.set(xlabel="steps after first reaching 5 m", ylabel="centroid distance to goal (m)", title=title, ylim=(0, 40))
        ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 2]
    L = out["landscape"]
    cand, sc = np.array(L["moment"]["cand"]), np.array(L["moment"]["score"])
    lim = max(abs(sc.min()), abs(sc.max()))
    im = ax.scatter(cand[:-1, 0], cand[:-1, 1], c=sc[:-1], cmap="RdBu_r", vmin=-lim, vmax=lim, s=14)
    ax.scatter(*np.array(L["pos"]).T, color="k", s=12, zorder=3)
    ax.quiver(*np.array(L["pos"]).T, *np.array(L["hd"]).T, color="k", scale=25, width=0.004, zorder=3)
    ax.plot(*L["dog"], "^", color=COL["orange"], ms=9, zorder=4, label="published dog")
    ax.plot(*L["moment"]["h_star"], "*", color=COL["blue"], ms=14, zorder=4, label="h* (second moment)")
    g = np.array(L["to_goal"])
    ax.annotate("", xy=g * 6, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=COL["gray"]))
    ax.text(*(g * 6.5), "goal", color=COL["gray"], fontsize=8)
    ax.set(aspect="equal", xlabel="m from the centroid", ylabel="m", title=f"C. Placement: E[ΔV | x, h] over dog positions h, {L['rho']:.0f} m from the goal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="E[ΔV] per step (m²), second moment about the goal")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    t = np.arange(T_B + 1)
    for ax, start, ylab, key in ((axs[1, 0], "scattered", "fraction of sheep inside r_goal", "frac_in"), (axs[1, 1], "scattered", "RMS distance to the goal (m)", "rms")):
        l5 = td["ladder"][start]["L5 + social terms (Jadhav sheep)"]
        ax.plot(t, l5[key], color=COL["red"], lw=1.6, ls="--", label="static ring of 10 dogs (L5)")
        ax.plot(t, td["published_at_goal"][key], color=COL["orange"], lw=1.6, ls="--", label="published dog at the goal (L6)")
        for kind, c, lab in (("moment", "blue", "second moment"), ("moment8", "green", f"second moment, {HORIZON}-step lookahead"),
                             ("moment_fast", "teal", f"second moment, v_dog = {V_DOG_FAST:g}"), ("l2", "purple", "L² error")):
            m = out["goal"].get(f"{start}/density_{kind}")
            if m is None:
                continue
            ax.plot(np.arange(len(m[key])), m[key], color=COL[c], lw=2, label=f"density controller: {lab}")
            m2 = out["goal"].get(f"cluster/density_{kind}")
            if m2 is not None:
                ax.plot(np.arange(len(m2[key])), m2[key], color=COL[c], lw=1.2, alpha=0.5)
        ax.set(xlabel="time (s)", ylabel=ylab, title=f"B. At the goal, {start} start (faint: displaced cluster)")
        if key == "frac_in":
            ax.axhline(0.95, color=COL["gray"], lw=1, ls=":")
            ax.set_ylim(0, 1.02)
            ax.legend(frameon=False, fontsize=7, loc="center right")
        else:
            ax.axhline(np.sqrt(2) * S_GOAL, color=COL["gray"], lw=1, ls=":")
            ax.set_ylim(0, 12)
    ax = axs[1, 2]
    for kind, c, lab in (("moment", "blue", "second moment"), ("moment8", "green", f"{HORIZON}-step lookahead"), ("moment_fast", "teal", f"v_dog = {V_DOG_FAST:g}"),
                         ("l2", "purple", "L² error")):
        m = out["goal"].get(f"scattered/density_{kind}")
        if m is not None:
            ax.plot(m["l2_times"], m["l2"], color=COL[c], lw=2, label=f"density controller: {lab}")
    l5 = td["ladder"]["scattered"]["L5 + social terms (Jadhav sheep)"]
    ax.plot(l5["l2_times"], l5["l2"], color=COL["red"], lw=1.6, ls="--", label="static ring (L5)")
    ax.plot(td["published_at_goal"]["l2_times"], td["published_at_goal"]["l2"], color=COL["orange"], lw=1.6, ls="--", label="published dog (L6)")
    ax.axhline(out["l2_floor"], color=COL["gray"], lw=1, ls=":")
    ax.set(xlabel="time (s)", ylabel="relative L² error of the KDE vs n̄", title="B. Their density error, scattered start (dotted: 14-sample floor)")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("One Jadhav dog under a density controller: placed each step where the exact expected descent of the density error is largest, "
                 "then moved there at v_dog", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)


# ------------------------------------------------------------------ main

def merge(new):
    """Merge this run's parts into the JSON on disk, one level deep, so parallel parts and variants keep each other's keys."""
    cur = json.load(open(JSON_PATH)) if JSON_PATH.exists() else {}
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    JSON_PATH.write_text(json.dumps(cur, indent=1))
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "fig"], default=None, help="run one part and merge it into the JSON (default: all)")
    ap.add_argument("--only", default=None, help="run only the variants of the part whose key contains this text")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    tic = time.time()
    out = dict(setup=dict(seed=SEED, runs_a=R_A, steps_a=T_A, runs_b=R_B, steps_b=T_B, r_goal=R_GOAL, s=fl(S_GOAL), kde_h=TD.KDE_H,
                          radii=list(DC.Controller().radii), n_angles=DC.Controller().n_angles, tol=DC.Controller().tol))
    parts = [a.part] if a.part else ["A", "B", "C", "fig"]
    if "A" in parts:
        out.update(part_a(a.quick, a.only))
    if "B" in parts:
        out.update(part_b(a.quick, a.only))
    if "C" in parts:
        out.update(part_c())
    out["seconds"] = {",".join(parts) + (f":{a.only}" if a.only else ""): fl(time.time() - tic)}
    if not a.quick:
        out = merge(out)
    if "fig" in parts and all(k in out for k in ("reach_series", "goal", "landscape")):
        figure(out, FIG_PATH if not a.quick else Path(os.environ.get("TMPDIR", "/tmp")) / "density_controller_quick.png")
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
