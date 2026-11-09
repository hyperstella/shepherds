"""Di Lorenzo's target density inside the Jadhav simulation: which components give convergence?

Goal density n̄: a Gaussian with 95% of the sheep within r_goal, the plane analogue of their
von Mises profile. Herders: the feedforward arrangement their deconvolution gives for this
goal, from shepherd.feasibility.minimum_dogs_local (non-negative least squares on rings) with
the exponential push kernel at the project's measured amplitude and diffusion, discretized
as M dogs equally spaced on the ring it finds.

Convergence is measured their way and in finite-N ways: relative L2 error of a kernel
density estimate against n̄ (with its 14-sample floor), the fraction of sheep inside r_goal
(their confinement metric), the RMS distance to the goal against the target's sqrt(2)·s, and
an exponential rate fitted to the RMS excess.

Ladder from their targets to the Jadhav sheep, one component per rung, same static ring:
  L0  overdamped targets, exponential push summed over dogs, diffusion D, nothing else
  L1  fixed speed 1 m/s: the heading follows the summed push; Jadhav's heading noise
  L2  the push is Jadhav's: a unit vector away from each dog within r_dog, summed, normalized
  L3  freeze: a sheep with no dog within r_dog stands still
  L4  heading inertia 0.5
  L5  the social terms (repulsion, attraction to 4 of 10 nearest, alignment): the Jadhav sheep
  L5b the Jadhav sheep without the freeze (their modified model: social terms always on)
  L6  the published model with its own closed-loop dog, started at the goal, no ring
Then L0 and L5 against the number of dogs on the ring, and L2, L3 and L5 against the sheep's
notice radius r_dog: a unit push has divergence 1/r > 0, so a ring of Jadhav dogs can only confine
through the edge of the notice disc. The sweep compares where the sheep settle with the circle
rho*(r_dog) on which the summed visible pushes turn from outward to inward.

    .venv/bin/python experiments/target_density.py
Writes results/target_density.json, results/target_density_replay.json,
results/figures/target_density.png and results/figures/target_density_notice.png.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import feasibility as FE  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
A_PUSH, L_PUSH, D_DIFF = 3.57, 3.0, 0.25  # measured dog push (L = 3 m row of results/feasibility.json) and D at 1 s
N, R, T, SEED = 14, 50, 200, 7
R_GOAL, R_DOG, V_SHEEP = 5.0, 12.0, 1.0
W = dict(inertia=0.5, rep=2.0, dog=1.0, att=1.5, noise=0.3, ali=1.3)  # published weights
R_REP, K_PERCEIVE, N_ATT, N_ALI = 2.0, 10, 4, 1
DT_SUB, KDE_H, L2_EVERY = 0.1, 1.0, 5
M_LADDER = 10
M_SWEEP = (5, 10, 20, 40)
R_DOG_SWEEP = (6.0, 9.0, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 22.0)
NOTICE_LEVELS = ("L2", "L3", "L5")
STARTS = ("scattered", "cluster")
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a", "teal": "#1b9aaa"}

LEVELS = [
    ("L0 Di Lorenzo targets", dict(speed="overdamped", kernel="exp", freeze=False, inertia=0.0, social=False)),
    ("L1 + fixed speed, heading noise", dict(speed="fixed", kernel="exp", freeze=False, inertia=0.0, social=False)),
    ("L2 + unit push within r_dog", dict(speed="fixed", kernel="unit", freeze=False, inertia=0.0, social=False)),
    ("L3 + freeze without a dog in range", dict(speed="fixed", kernel="unit", freeze=True, inertia=0.0, social=False)),
    ("L4 + heading inertia", dict(speed="fixed", kernel="unit", freeze=True, inertia=0.5, social=False)),
    ("L5 + social terms (Jadhav sheep)", dict(speed="fixed", kernel="unit", freeze=True, inertia=0.5, social=True)),
    ("L5b Jadhav sheep, no freeze", dict(speed="fixed", kernel="unit", freeze=False, inertia=0.5, social=True)),
]
LEVEL_COLORS = ["blue", "teal", "green", "orange", "purple", "red", "gray"]


def fl(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def f_push(r):
    return A_PUSH * np.exp(-r / L_PUSH)


def goal_sigma(r_goal=R_GOAL, coverage=0.95):
    return r_goal / np.sqrt(-2 * np.log(1 - coverage))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def herder_ring():
    """Their deconvolution for this goal: candidate dogs on rings, non-negative counts; the mass-weighted ring radius."""
    res = FE.minimum_dogs_local(f_push, None, D_DIFF, N, R_GOAL)
    C, m = res["dog_positions"], res["dog_counts"]
    radii = np.hypot(C[:, 0], C[:, 1])
    keep = m > 1e-6
    return dict(r_ring=float(np.average(radii[keep], weights=m[keep])), n_dogs=float(res["n_dogs"]), drift_error=float(res["relative_drift_error"]))


def ring_dogs(M, r_ring):
    th = 2 * np.pi * np.arange(M) / M
    return np.stack([r_ring * np.cos(th), r_ring * np.sin(th)], -1)


def starts(rng, kind, r_ring):
    if kind == "goal":  # sampled from the goal density itself
        x = goal_sigma() * rng.standard_normal((R, N, 2))
        th = 2 * np.pi * rng.random((R, N))
        return x, np.stack([np.cos(th), np.sin(th)], -1)
    if kind == "scattered":
        rad = 0.8 * r_ring * np.sqrt(rng.random((R, N)))
        th = 2 * np.pi * rng.random((R, N))
        x = np.stack([rad * np.cos(th), rad * np.sin(th)], -1)
    else:
        x = np.array([0.65 * r_ring, 0.0]) + 1.0 * rng.standard_normal((R, N, 2))
    th = 2 * np.pi * rng.random((R, N))
    return x, np.stack([np.cos(th), np.sin(th)], -1)


# ------------------------------------------------------------------ the ladder simulator

def run_level(level, dogs, rng, start, r_ring, push_scale=1.0, r_dog=R_DOG):
    """Positions (R, N, 2, T+1) of N sheep under dogs, static (M, 2) or on a path (T+1, M, 2);
    the rules are the Jadhav rules at the top rung."""
    x, h = starts(rng, start, r_ring)
    pos = np.empty((R, N, 2, T + 1))
    pos[..., 0] = x
    ar, an = np.arange(R)[:, None], np.arange(N)[None, :]
    eye = np.eye(N, dtype=bool)
    dogs = np.asarray(dogs, float)
    moving = dogs.ndim == 3
    for t in range(T):
        if level["speed"] == "overdamped":
            n_sub = int(round(1 / DT_SUB))
            for k in range(n_sub):
                d_now = dogs[t] + (dogs[t + 1] - dogs[t]) * (k / n_sub) if moving else dogs
                rd = x[:, :, None, :] - d_now[None, None]
                dd = np.linalg.norm(rd, axis=-1) + 1e-9
                F = push_scale * ((f_push(dd) / dd)[..., None] * rd).sum(2)
                x = x + F * DT_SUB + np.sqrt(2 * D_DIFF * DT_SUB) * rng.standard_normal(x.shape)
            pos[..., t + 1] = x
            continue
        d_now = dogs[t] if moving else dogs
        rd = x[:, :, None, :] - d_now[None, None]
        dd = np.linalg.norm(rd, axis=-1) + 1e-9
        e = rd / dd[..., None]
        if level["kernel"] == "exp":
            dog_dir = (np.exp(-dd / L_PUSH)[..., None] * e).sum(2)
            sees = np.ones((R, N), bool)
        else:
            within = dd <= r_dog
            dog_dir = unit(np.where(within[..., None], e, 0.0).sum(2))
            sees = within.any(2)
        th = 2 * np.pi * rng.random((R, N))
        eta = np.stack([np.cos(th), np.sin(th)], -1)
        terms = level["inertia"] * h + W["dog"] * dog_dir + W["noise"] * eta
        has_rep = np.zeros((R, N), bool)
        grazing = h
        if level["social"]:
            r_ij = x[:, None, :, :] - x[:, :, None, :]  # [r, i, j] = x_j - x_i
            dist = np.linalg.norm(r_ij, axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                u_ij = np.nan_to_num(r_ij / dist[..., None])
            near = (dist < R_REP) & ~eye
            has_rep = near.any(-1)
            rep = np.where(has_rep[..., None], -unit(np.where(near[..., None], u_ij, 0.0).sum(2)), 0.0)
            dist_s = np.where(eye, np.inf, dist)
            knn = np.argsort(dist_s, axis=-1, kind="stable")[..., :K_PERCEIVE]
            att = np.take_along_axis(knn, np.argsort(rng.random((R, N, K_PERCEIVE)), -1)[..., :N_ATT], -1)
            ali = np.take_along_axis(att, np.argsort(rng.random((R, N, N_ATT)), -1)[..., :N_ALI], -1)
            r_att = r_ij[ar[..., None], an[..., None], att]
            s_att = unit(unit(r_att).sum(2))
            s_ali = unit(h[ar[..., None], ali].sum(2))
            terms = terms + W["rep"] * rep + W["att"] * s_att + W["ali"] * s_ali
            grazing = unit(level["inertia"] * h + W["rep"] * rep)
        active = unit(terms)
        if level["freeze"]:
            moves = sees | has_rep
            new_h = np.where(sees[..., None], active, grazing)
            new_h = np.where(moves[..., None], new_h, h)
        else:
            moves = np.ones((R, N), bool)
            new_h = active
        x = x + np.where(moves[..., None], V_SHEEP * new_h, 0.0)
        h = new_h
        pos[..., t + 1] = x
    return pos


def run_published_at_goal(rng, s, r_ring):
    """L6: the published model, sheep sampled from n̄ at the goal, its own dog starting on the ring."""
    pos_s = s * rng.standard_normal((R, N, 2))
    th = 2 * np.pi * rng.random((R, N))
    vel_s = np.stack([np.cos(th), np.sin(th)], -1)
    pos_d = np.tile(np.array([[-r_ring, 0.0]]), (R, 1))
    res = simulate(Params(n_steps=T + 1), n_runs=R, seed=SEED + 99, init={"pos_s": pos_s, "vel_s": vel_s, "pos_d": pos_d})
    return res.pos_s, res.pos_d


# ------------------------------------------------------------------ metrics

def kde_grid():
    a = np.arange(-3 * R_GOAL, 3 * R_GOAL + 1e-9, 0.5)
    X, Y = np.meshgrid(a, a, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], -1)


def l2_error(x, s, grid):
    """Relative L2 distance between the KDE of x (R, N, 2) and n̄ smoothed with the same kernel."""
    d2 = ((x[:, :, None, :] - grid[None, None]) ** 2).sum(-1)  # (R, N, G)
    n_hat = np.exp(-d2 / (2 * KDE_H ** 2)).mean(1) / (2 * np.pi * KDE_H ** 2)
    v = s ** 2 + KDE_H ** 2
    n_bar = np.exp(-(grid ** 2).sum(-1) / (2 * v)) / (2 * np.pi * v)
    return np.sqrt(((n_hat - n_bar[None]) ** 2).sum(1)) / np.sqrt((n_bar ** 2).sum())


def metrics(pos, s, grid, floor):
    r = np.linalg.norm(pos, axis=2)  # (R, N, T+1)
    frac_in = (r <= R_GOAL).mean(1).mean(0)
    rms = np.sqrt((r ** 2).mean(axis=(0, 1)))
    r_star = np.sqrt(2) * s
    times = np.arange(0, T + 1, L2_EVERY)
    l2 = np.array([l2_error(pos[..., t], s, grid).mean() for t in times])
    excess = rms - r_star
    e0 = excess[0]
    below = np.flatnonzero(excess < 0.1 * e0) if e0 > 0 else np.array([])
    t10 = int(below[0]) if len(below) else None
    end = t10 if t10 is not None and t10 >= 3 else T  # fit up to the 10% crossing; a crossing within 3 s is fitted on those steps
    tt = np.arange(0, end + 1)
    ok = excess[: end + 1] > 0
    rate = fl(-np.polyfit(tt[ok], np.log(excess[: end + 1][ok]), 1)[0]) if ok.sum() >= 3 else None
    return dict(
        frac_in=frac_in.tolist(), rms=rms.tolist(), l2_times=times.tolist(), l2=l2.tolist(), r_star=fl(r_star), l2_floor=fl(floor),
        final_frac_in=fl(frac_in[-20:].mean()), final_rms=fl(rms[-20:].mean()), final_l2=fl(l2[-4:].mean()),
        rms_excess_initial=fl(e0), rms_excess_final=fl(excess[-20:].mean()), t_10pct=t10, rate_per_s=rate,
    )


# ------------------------------------------------------------------ the notice radius under the ring

def unit_push_radial(rho, phi, dogs, r_dog):
    """Radial component (outward > 0) of the summed unit pushes of the dogs within r_dog of a sheep at (rho, phi)."""
    e = np.array([np.cos(phi), np.sin(phi)])
    rd = rho * e - dogs
    dd = np.linalg.norm(rd, axis=-1)
    vis = dd <= r_dog
    return float((rd[vis] / dd[vis, None]).sum(0) @ e)


def stable_radius(dogs, r_ring, r_dog):
    """Where the summed unit push turns from outward to inward, on a dog's bearing and midway between two dogs.

    Each dog's unit push has divergence 1/r > 0 inside its notice disc, so with every dog in view the
    sum points outward everywhere inside the ring (it is the gradient of the summed distances, which is
    smallest at the centre). The only inward turn comes where far dogs drop out of view.
    """
    rhos = np.linspace(0.01, r_ring - 0.01, 2000)
    out = []
    for phi in (0.0, np.pi / len(dogs)):
        F = np.array([unit_push_radial(r, phi, dogs, r_dog) for r in rhos])
        idx = np.flatnonzero((F[:-1] > 0) & (F[1:] <= 0))
        out.append(fl(rhos[idx[0]]) if len(idx) else None)
    centre_blind = r_dog < r_ring  # no dog in view at the centre: a neutral disc of radius r_ring - r_dog
    return dict(on_dog=out[0], between=out[1], centre_blind=centre_blind, blind_radius=fl(r_ring - r_dog) if centre_blind else None)


def notice_sweep(dogs, r_ring, s, grid, floor):
    """L2, L3 and L5 under the same ring, with the sheep's notice radius swept."""
    rows = []
    levels = {n.split(" ")[0]: lv for n, lv in LEVELS}
    for r_dog in R_DOG_SWEEP:
        pred = stable_radius(dogs, r_ring, r_dog)
        row = dict(r_dog=r_dog, predicted=pred, levels={})
        for key in NOTICE_LEVELS:
            pos = run_level(levels[key], dogs, np.random.default_rng(SEED + 4), "scattered", r_ring, r_dog=r_dog)
            r = np.linalg.norm(pos[..., -20:], axis=2)  # (R, N, 20)
            G = np.linalg.norm(pos[..., -20:].mean(1), axis=1)  # centroid distance (R, 20)
            m = metrics(pos, s, grid, floor)
            row["levels"][key] = dict(final_frac_in=m["final_frac_in"], final_rms=m["final_rms"], final_l2=m["final_l2"],
                                      median_radius=fl(np.median(r)), frac_centre_1m=fl((r <= 1.0).mean()),
                                      frac_escaped=fl((r > r_ring).mean()), centroid_dist=fl(G.mean()),
                                      radial_hist=np.histogram(r.ravel(), bins=np.arange(0, 2 * r_ring + 0.5, 0.5))[0].tolist())
            lv = row["levels"][key]
            print(f"  r_dog {r_dog:4.1f} {key}: predicted rho* {pred['on_dog']}/{pred['between']}  median radius {lv['median_radius']:.2f}  "
                  f"inside {lv['final_frac_in']:.2f}  escaped {lv['frac_escaped']:.2f}  within 1 m {lv['frac_centre_1m']:.2f}", flush=True)
        rows.append(row)
    return rows


def notice_figure(rows, r_ring, s, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    rd = np.array([r["r_dog"] for r in rows])
    cols = dict(L2="green", L3="orange", L5="red")
    names = dict(L2="L2 unit push", L3="L3 + freeze", L5="L5 Jadhav sheep")
    ax = axs[0]
    for key in NOTICE_LEVELS:
        ax.plot(rd, [r["levels"][key]["median_radius"] for r in rows], "o-", color=COL[cols[key]], lw=2, label=f"{names[key]}: median sheep radius")
    pred = np.array([np.nan if r["predicted"]["on_dog"] is None else np.mean([r["predicted"]["on_dog"], r["predicted"]["between"]]) for r in rows])
    ax.plot(rd, pred, "s--", color="k", ms=6, lw=1.2, label="predicted stable circle ρ*")
    ax.axhline(s * np.sqrt(2 * np.log(2)), color=COL["gray"], lw=1, ls=":", label="goal median radius")
    ax.axhline(r_ring, color=COL["blue"], lw=1, ls=":", label=f"the ring ({r_ring:.1f} m)")
    ax.axvline(r_ring, color=COL["gray"], lw=0.8)
    esc = [r["r_dog"] for r in rows if r["levels"]["L2"]["frac_escaped"] > 0.5]
    if esc:
        ax.axvspan(min(esc) - 0.5, rd.max() + 0.5, color=COL["gray"], alpha=0.12, lw=0)
        ax.text(min(esc) + 0.2, 6.0, "all escape\n(median radius\n> 25 m)", fontsize=8, color="#3a3a38", va="top")
    ax.set(xlabel="sheep notice radius r_dog (m)", ylabel="radius (m)", ylim=(0, 12), title="Where the sheep settle under the ring")
    ax.legend(loc="upper left", fontsize=7.5, facecolor="white", edgecolor="none", framealpha=1)
    ax = axs[1]
    for key in NOTICE_LEVELS:
        ax.plot(rd, [r["levels"][key]["frac_escaped"] for r in rows], "o-", color=COL[cols[key]], lw=2, label=names[key])
    ax.axvline(r_ring, color=COL["gray"], lw=0.8)
    ax.set(xlabel="sheep notice radius r_dog (m)", ylabel="fraction of sheep beyond the ring", ylim=(-0.02, 1.02),
           title="Escape: every dog in view pushes outward")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    edges = np.arange(0, 2 * r_ring + 0.5, 0.5)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for r_dog, c in ((12.0, "green"), (13.0, "teal"), (14.0, "purple")):
        row = next(r for r in rows if r["r_dog"] == r_dog)
        h = np.array(row["levels"]["L2"]["radial_hist"], float)
        dens = h / h.sum() / (2 * np.pi * mid * 0.5)  # per unit area
        ax.plot(mid, dens, color=COL[c], lw=2, label=f"L2, r_dog = {r_dog:g} m (ρ* {np.mean([row['predicted']['on_dog'], row['predicted']['between']]):.1f} m)")
    ax.plot(mid, np.exp(-mid ** 2 / (2 * s ** 2)) / (2 * np.pi * s ** 2), color="k", lw=1.2, ls="--", label="goal density n̄ / N")
    ax.set(xlabel="distance from the goal (m)", ylabel="sheep per m² (fraction)", xlim=(0, r_ring), title="Radial profile: a hollow goal")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


# ------------------------------------------------------------------ figure

def figure(ring, s, out, sweep, out_path):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    t = np.arange(T + 1)
    for (name, _), c in zip(LEVELS, LEVEL_COLORS):
        m = out["ladder"]["scattered"][name]
        axs[0, 0].plot(t, m["rms"], color=COL[c], lw=1.8, label=name)
        axs[0, 1].plot(t, m["frac_in"], color=COL[c], lw=1.8)
        axs[0, 2].plot(m["l2_times"], m["l2"], color=COL[c], lw=1.8)
        axs[1, 0].plot(t, out["ladder"]["cluster"][name]["rms"], color=COL[c], lw=1.8)
    m6 = out["published_at_goal"]
    axs[0, 0].plot(t, m6["rms"], color="k", lw=1.5, ls="--", label="L6 published model, own dog, started at goal")
    axs[0, 1].plot(t, m6["frac_in"], color="k", lw=1.5, ls="--")
    axs[0, 2].plot(m6["l2_times"], m6["l2"], color="k", lw=1.5, ls="--")
    r_star = np.sqrt(2) * s
    for ax in (axs[0, 0], axs[1, 0]):
        ax.axhline(r_star, color=COL["gray"], lw=1, ls=":")
        ax.set(xlabel="time (s)", ylabel="RMS distance to the goal (m)")
    axs[0, 0].set_title("Scattered start: RMS radius (dotted: target √2·s)")
    axs[1, 0].set_title("Displaced-cluster start: RMS radius")
    axs[0, 0].legend(frameon=False, fontsize=7.5)
    axs[0, 1].axhline(0.95, color=COL["gray"], lw=1, ls=":")
    axs[0, 1].set(xlabel="time (s)", ylabel="fraction of sheep inside r_goal", ylim=(0, 1.02), title="Their confinement metric (dotted: 95%)")
    axs[0, 2].axhline(out["l2_floor"], color=COL["gray"], lw=1, ls=":")
    axs[0, 2].set(xlabel="time (s)", ylabel="relative L2 error of the KDE vs n̄", title="Their density error (dotted: 14-sample floor)")
    ax = axs[1, 1]
    names = [n for n, _ in LEVELS] + ["L6 published"]
    rates = [out["ladder"]["scattered"][n]["rate_per_s"] for n, _ in LEVELS] + [m6["rate_per_s"]]
    vals = [r if r is not None else 0 for r in rates]
    ax.barh(range(len(names)), vals, color=[COL[c] for c in LEVEL_COLORS] + ["k"])
    ax.set_yticks(range(len(names)), [n.split(" ", 1)[0] for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(2 * D_DIFF / s ** 2, color=COL["gray"], lw=1, ls=":")
    ax.set(xlabel="fitted decay rate of the RMS excess (1/s); dotted: 2D/s² of the OU bowl", title="Rate per rung (scattered start)")
    for i, (r, n) in enumerate(zip(rates, names)):
        ax.text(max(vals) * 0.02, i, "no fit" if r is None else f"{r:.3f}", va="center", fontsize=8)
    ax = axs[1, 2]
    for key, c, lab in (("L0_fixed_mass", "blue", "L0 targets, total push fixed (continuum limit)"), ("L0_real_dogs", "teal", "L0 targets, each dog a full dog"), ("L5_real_dogs", "red", "L5 Jadhav sheep, each dog a full dog")):
        rows = sweep[key]
        ax.plot([r["M"] for r in rows], [r["final_frac_in"] for r in rows], "o-", color=COL[c], label=lab)
    ax.axvline(ring["n_dogs"], color=COL["gray"], lw=1, ls=":", label=f"continuum dog mass {ring['n_dogs']:.1f}")
    ax.set(xlabel="dogs on the ring M", ylabel="final fraction inside r_goal", ylim=(0, 1.02), xscale="log", title="Discreteness of the enclosure")
    ax.set_xticks(list(M_SWEEP), [str(m) for m in M_SWEEP])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Di Lorenzo's goal density (95% within {R_GOAL:g} m) under their herder ring ({M_LADDER} dogs at {ring['r_ring']:.1f} m), "
                 "from their targets to the Jadhav sheep one component at a time", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    s = goal_sigma()
    ring = herder_ring()
    print(f"goal: r_goal {R_GOAL} m, s {s:.2f} m; their ring: radius {ring['r_ring']:.1f} m, dog mass {ring['n_dogs']:.1f}, drift error {ring['drift_error']:.2f}", flush=True)
    grid = kde_grid()
    floor = float(l2_error(s * rng.standard_normal((200, N, 2)), s, grid).mean())
    dogs = ring_dogs(M_LADDER, ring["r_ring"])
    out = dict(setup=dict(N=N, runs=R, steps=T, r_goal=R_GOAL, s=fl(s), r_dog=R_DOG, a=A_PUSH, L=L_PUSH, D=D_DIFF, weights=W, M_ladder=M_LADDER),
               ring=ring, l2_floor=floor, ladder={}, sweep={})
    replay = []
    for start in STARTS:
        out["ladder"][start] = {}
        for name, level in LEVELS:
            pos = run_level(level, dogs, np.random.default_rng(SEED + 1), start, ring["r_ring"])
            m = metrics(pos, s, grid, floor)
            out["ladder"][start][name] = m
            print(f"  [{start:9s}] {name:38s} rate {m['rate_per_s'] if m['rate_per_s'] is None else round(m['rate_per_s'], 3)}  "
                  f"t10 {m['t_10pct']}  final inside {m['final_frac_in']:.2f}  rms {m['final_rms']:.2f}  L2 {m['final_l2']:.2f} (floor {floor:.2f})", flush=True)
            if start == "scattered" and name.split(" ")[0] in ("L0", "L5"):
                replay.append(dict(key=name.split(" ")[0].lower() + "_ring", label=f"{name} under their ring of {M_LADDER} dogs", N=N, n_dogs=M_LADDER, static=True,
                                   r_goal=R_GOAL, l_sep=None, l_drive=None, r_dog=R_DOG if level["kernel"] == "unit" else None,
                                   pos=np.round(np.concatenate([np.moveaxis(pos[0], -1, 0), np.tile(dogs[None], (T + 1, 1, 1))], 1), 2).tolist(),
                                   rho=m and np.round(np.linalg.norm(pos[0].mean(0), axis=0), 2).tolist(),
                                   spread=np.round(np.linalg.norm(pos[0] - pos[0].mean(0, keepdims=True), axis=1).max(0), 2).tolist()))
    pos6, dog6 = run_published_at_goal(np.random.default_rng(SEED + 2), s, ring["r_ring"])
    m6 = metrics(pos6, s, grid, floor)
    out["published_at_goal"] = m6
    print(f"  L6 published model at goal: final inside {m6['final_frac_in']:.2f} rms {m6['final_rms']:.2f} L2 {m6['final_l2']:.2f}", flush=True)
    replay.append(dict(key="l6_published_at_goal", label="L6 published model with its own dog, started at the goal", N=N, n_dogs=1, static=False,
                       r_goal=R_GOAL, l_sep=fl(Params().resolved().l_sep), l_drive=fl(Params().resolved().l_drive), r_dog=R_DOG,
                       pos=np.round(np.concatenate([np.moveaxis(pos6[0], -1, 0), np.moveaxis(dog6[0], -1, 0)[:, None]], 1), 2).tolist(),
                       rho=np.round(np.linalg.norm(pos6[0].mean(0), axis=0), 2).tolist(),
                       spread=np.round(np.linalg.norm(pos6[0] - pos6[0].mean(0, keepdims=True), axis=1).max(0), 2).tolist()))
    print("dog-count sweep", flush=True)
    for key, level, scale in (("L0_fixed_mass", LEVELS[0][1], True), ("L0_real_dogs", LEVELS[0][1], False), ("L5_real_dogs", LEVELS[5][1], False)):
        rows = []
        for M in M_SWEEP:
            d = ring_dogs(M, ring["r_ring"])
            pos = run_level(level, d, np.random.default_rng(SEED + 3), "scattered", ring["r_ring"], push_scale=(ring["n_dogs"] / M if scale else 1.0))
            m = metrics(pos, s, grid, floor)
            rows.append(dict(M=M, push_scale=fl(ring["n_dogs"] / M if scale else 1.0), final_frac_in=m["final_frac_in"], final_rms=m["final_rms"], rate_per_s=m["rate_per_s"]))
            print(f"  {key:14s} M={M:2d}: inside {m['final_frac_in']:.2f} rms {m['final_rms']:.2f} rate {m['rate_per_s']}", flush=True)
        out["sweep"][key] = rows
    print("notice-radius sweep under the ring", flush=True)
    out["notice"] = notice_sweep(dogs, ring["r_ring"], s, grid, floor)
    notice_figure(out["notice"], ring["r_ring"], s, OUT / "figures" / "target_density_notice.png")
    out["seconds"] = fl(time.time() - tic)
    (OUT / "target_density.json").write_text(json.dumps(out, indent=1))
    (OUT / "target_density_replay.json").write_text(json.dumps(replay, separators=(",", ":")))
    figure(ring, s, out, out["sweep"], OUT / "figures" / "target_density.png")
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
