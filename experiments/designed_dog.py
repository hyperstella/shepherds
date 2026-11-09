"""A dog designed from Di Lorenzo's deconvolution, in the dog-control comparison.

The sheep are the Jadhav sheep (published rules, and the calibrated M5tau sheep with the
learned dog and pair kernels); the dog's drive geometry comes from shepherd/dog_design.py,
which applies Di Lorenzo's herder-density design to the project's measured kernels:

- "Designed, first-order arc": standoff and lateral sweep read off the one-sided arc the
  translating-goal deconvolution gives (section 6 of docs/continuification_driving.md
  predicts this standoff to be too close, since in a first-order model the dog's push
  must supply the flock's speed).
- "Designed, direction-matched": the kinetic reading, where the sheep take the dog's
  direction, which puts the dog as far back as the notice radius allows.

Both sit beside the published Strömbom dog, the data-derived dog and a heuristic tuned on
the same flock (the grid of experiments/dog_control.py, tuned on separate seeds); same
task, starts, seeds and scores as there: drive the flock 60 m to a target within 300 s.

    .venv/bin/python experiments/designed_dog.py
Writes results/designed_dog.json and results/figures/designed_dog.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dog_control as DC  # noqa: E402
from dog_response import learned_dog_kernel  # noqa: E402
from pair_attraction import learned_pair_kernel  # noqa: E402
from shepherd import data, dog_design as DD, dog_policy as DP, feasibility as FE  # noqa: E402
from shepherd.kernels import HatBasis  # noqa: E402
from shepherd.model import COLLECT, SLOW, Params, simulate  # noqa: E402
from build_table import V_SHEEP  # noqa: E402

OUT = ROOT / "results"
L_PUSH, G_MAX = 3.0, 3.6
SIZES = (14, 30)
V_DOG_DESIGN = 1.5
COL = {"Strömbom (published)": "#eb6834", "Data-derived dog": "#2a78d6", "Tuned heuristic": "#1baf7a",
       "Designed, first-order (one dog)": "#7b61c4", "Designed, direction-matched": "#d63a3a"}


def fl(x):
    return None if x is None else float(x)


def measured_kernels(drives):
    a, cg, r2 = FE.fit_exponential(drives, L_PUSH, cohesion=True)
    pair = HatBasis(FE.PAIR_KNOTS_FO)
    f = lambda r, a=a: a * np.exp(-r / L_PUSH)
    g = lambda r, cg=cg: np.where(r <= G_MAX, pair.curve(cg, r), 0.0)
    D = json.loads((OUT / "feasibility.json").read_text())["D_by_window_s"]["1.0"]
    return f, g, float(D), float(a), float(r2)


def designs(N, f, g, D, c, r_dog):
    r_goal = 2.5 * np.sqrt(N / 14)
    arc = DD.translating_arc(f, g, D, N, r_goal, c, r_dog=r_dog)
    one = DD.single_dog_first_order(f, g, D, N, r_goal, c, r_dog=r_dog)
    dm = DD.direction_match(N, r_goal, c, D, g, r_dog)
    s = DD.goal_sigma(r_goal)
    common = {"noise_dog": 0.3, "l_collect": 2.0, "l_sep": 2.0 * N ** (2 / 3), "tau_d": "instant", "v_dog": V_DOG_DESIGN}
    dogs = {
        "Designed, first-order (one dog)": {**common, "l_drive": max(one["standoff"], 0.5), "flank_sigma": 0.0},
        "Designed, direction-matched": {**common, "l_drive": dm["standoff"], "flank_sigma": 0.0},
    }
    summary = dict(r_goal=fl(r_goal), s=fl(s), c=fl(c), r_dog=fl(r_dog),
                   arc=dict(standoff=arc["standoff"], lateral_sd=arc["lateral_sd"], radius=arc["radius"], angle_sd_deg=float(np.degrees(arc["angle_sd"])),
                            mass=arc["mass"], residual=arc["residual"], behind_mass_frac=arc["behind_mass_frac"],
                            single_dog_standoff=one["standoff"], single_dog_lateral=one["lateral"], single_dog_error=one["error_best"]),
                   direction=dict(standoff=dm["standoff"], d_cov=dm["d_cov"], J_best=dm["J_best"], d_unconstrained=dm["d_unconstrained"],
                                  J_unconstrained=dm["J_unconstrained"], d_grid=dm["d_grid"].tolist(), J=dm["J"].tolist(),
                                  y_grid=dm["y_grid"].tolist(), J_lat=dm["J_lat"].tolist()))
    return dogs, summary, arc


def run_job(job):
    """One (flock, N, dog) setting, same starts and scoring as dog_control.run, plus the model extensions."""
    N, dog, sheep, seed, dt, R, v_sheep, base_kw = job
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, R)
    start = DC.START * np.stack([np.cos(ang), np.sin(ang)], -1)
    r0 = 0.6 * np.sqrt(N) * np.sqrt(rng.random((R, N)))
    a0 = rng.uniform(0, 2 * np.pi, (R, N))
    pos_s = start[:, None] + np.stack([r0 * np.cos(a0), r0 * np.sin(a0)], -1)
    pos_d = start * (1 + (8 + 0.6 * np.sqrt(N)) / DC.START) + rng.normal(0, 2, (R, 2))
    th = rng.uniform(0, 2 * np.pi, (R, N))
    init = {"pos_s": pos_s, "vel_s": np.stack([np.cos(th), np.sin(th)], -1), "pos_d": pos_d}
    p = Params(n_sheep=N, dt=dt, v_sheep=v_sheep, n_steps=int(DC.T_MAX / dt) + 1, box_length=DC.START,
               flank_sigma=1e-9 if dog.get("flank_sigma", 0) > 0 else 0.0, dog_speed_gain=dog.get("dog_speed_gain"), **base_kw)
    dog = {**dog, "tau_d": dt if dog.get("tau_d") == "instant" else dog.get("tau_d")}
    per_run = {k: np.broadcast_to(np.asarray(v, float), (R,)).copy() for k, v in {**sheep, **dog}.items() if v is not None}
    res = simulate(p, R, seed + 1, per_run=per_run, init=init)
    G = res.pos_s.mean(1)
    dist = np.linalg.norm(G, axis=1)
    arrived = dist < DC.ARRIVE
    t_arr = np.where(arrived.any(1), arrived.argmax(1), dist.shape[1])
    C = np.linalg.norm(res.pos_s - G[:, None], axis=2).mean(1)
    d_sd = np.linalg.norm(res.pos_s - res.pos_d[:, None], axis=2)
    r_dog = per_run.get("r_dog", np.full(R, p.r_dog))[:, None, None]
    mask = np.arange(dist.shape[1])[None] < t_arr[:, None]
    m = lambda x: (x * mask).sum(1) / np.maximum(mask.sum(1), 1)
    return {
        "success": arrived.any(1), "time_s": t_arr * dt, "cohesion": m(C),
        "max_spread": np.where(mask, np.linalg.norm(res.pos_s - G[:, None], axis=2).max(1), 0).max(1),
        "pressure": m((d_sd.min(1) < 3.0).astype(float)), "collect_frac": m((res.mode == COLLECT).astype(float)),
        "slow_frac": m((res.mode == SLOW).astype(float)), "sees_frac": m((d_sd <= r_dog).all(1).astype(float)),
        "dog_behind_m": m(np.einsum("rct,rct->rt", res.pos_d - G, -G / np.maximum(dist[:, None], 1e-9))),
    }


def summarize(m):
    s = DC.summarize(m)
    s["slow_frac"] = float(np.mean(m["slow_frac"]))
    s["all_see_dog_frac"] = float(np.mean(m["sees_frac"]))
    s["dog_behind_centroid_m_median"] = float(np.median(m["dog_behind_m"]))
    return s


def main():
    tic = time.time()
    drives = data.load_drives(data.load_sessions())
    f, g, D, a, r2 = measured_kernels(drives)
    measured = DP.estimate(drives)
    m3t = DC.sheep_posterior(model="M3tau")
    rep = {k: np.repeat(v, DC.REPS) for k, v in m3t.items()}
    r_dog_cal = float(np.median(m3t["r_dog"]))
    kern_dog, kern_pair = learned_dog_kernel(), learned_pair_kernel()
    flocks = {  # name: (sheep per-run, dt, v_sheep, base Params kwargs, r_dog for the design)
        "published sheep (M0)": ({}, 1.0, 1.0, {}, 12.0),
        "calibrated M5tau (learned kernels), grazing": (rep, 0.2, V_SHEEP[0.2], dict(dog_kernel=kern_dog, g_dogk=2.0, pair_kernel=kern_pair, g_pairk=2.0), r_dog_cal),
    }
    print(f"kernels: a {a:.2f} m/s at L {L_PUSH} m (cv R2 {r2:.2f}), D {D:.3f}; calibrated r_dog median {r_dog_cal:.2f} m", flush=True)
    out = {"kernels": dict(a=a, L=L_PUSH, D=D, cv_r2=r2, g_cutoff=G_MAX), "measured_dog": measured, "r_dog_calibrated_median": r_dog_cal, "flocks": {}}
    R = DC.N_DRAW * DC.REPS
    pool = ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2))
    # tune the heuristic on each flock and size, on separate seeds, as dog_control does
    tjobs, tkeys = [], []
    for flock, (sheep, dt, v_sheep, base_kw, r_dog) in flocks.items():
        for N in SIZES:
            for name, dog in DC.tuning_grid(N):
                tjobs.append((N, dog, sheep, 10_000 + N, dt, R, v_sheep, base_kw))
                tkeys.append((flock, N, name, dog))
    tuned = {}
    for (flock, N, name, dog), m in zip(tkeys, pool.map(run_job, tjobs)):
        s = DC.summarize(m)
        if (flock, N) not in tuned or DC.score(s) < DC.score(tuned[(flock, N)][2]):
            tuned[(flock, N)] = (name, dog, s)
    for (flock, N), (name, dog, s) in tuned.items():
        print(f"[{flock[:28]}] N={N:2d} tuned: {name} -> success {s['success_rate']:.2f} time {s['time_to_target_s_median']}", flush=True)
    jobs, keys = [], []
    for flock, (sheep, dt, v_sheep, base_kw, r_dog) in flocks.items():
        out["flocks"][flock] = {"designs": {}, "results": {}, "tuned": {}}
        for N in SIZES:
            out["flocks"][flock]["tuned"][str(N)] = tuned[(flock, N)][0]
            designed, summary, _ = designs(N, f, g, D, v_sheep, r_dog)
            out["flocks"][flock]["designs"][str(N)] = summary
            d = summary
            print(f"[{flock}] N={N}: first-order arc: mass {d['arc']['mass']:.1f} dogs, {d['arc']['behind_mass_frac']:.0%} behind, centre {d['arc']['standoff']:.1f} m behind, "
                  f"residual {d['arc']['residual']:.2f}; best single dog {d['arc']['single_dog_standoff']:.1f} m behind, {d['arc']['single_dog_lateral']:+.1f} m aside, "
                  f"residual {d['arc']['single_dog_error']:.2f} | direction-matched standoff {d['direction']['standoff']:.1f} m (coverage limit {d['direction']['d_cov']:.1f}, "
                  f"unconstrained best {d['direction']['d_unconstrained']:.1f})", flush=True)
            dogs = DC.dog_settings(N, measured)
            dogs["Tuned heuristic"] = tuned[(flock, N)][1]
            dogs.update(designed)
            for name, dog in dogs.items():
                jobs.append((N, dog, sheep, 20_000 + N, dt, R, v_sheep, base_kw))
                keys.append((flock, N, name, {k: fl(v) if isinstance(v, (int, float, np.floating)) else v for k, v in dog.items()}))
    with pool as ex:
        for (flock, N, name, dog), m in zip(keys, ex.map(run_job, jobs)):
            s = summarize(m)
            s["dog"] = dog
            out["flocks"][flock]["results"][f"N{N} | {name}"] = s
            print(f"[{flock[:28]}] N={N:2d} {name:36s} success {s['success_rate']:.2f} time {s['time_to_target_s_median']} C {s['cohesion_mean']:.2f} "
                  f"dog<3m {s['dog_within_3m_frac']:.2f} slow {s['slow_frac']:.2f} all-see {s['all_see_dog_frac']:.2f} behind {s['dog_behind_centroid_m_median']:.1f}", flush=True)
    out["seconds"] = time.time() - tic
    (OUT / "designed_dog.json").write_text(json.dumps(out, indent=1, default=float))
    plot(out)
    print(f"done in {time.time() - tic:.0f} s")


def plot(out):
    flocks = list(out["flocks"])
    dogs = list(COL)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax = axs[0, 0]
    for flock, c in zip(flocks, ("#eb6834", "#2a78d6")):
        d = out["flocks"][flock]["designs"]["14"]["direction"]
        ax.plot(d["d_grid"], d["J"], color=c, lw=2, label=f"{flock[:30]}: r_dog {out['flocks'][flock]['designs']['14']['r_dog']:.0f} m")
        ax.axvline(d["d_cov"], color=c, ls=":", lw=1)
        ax.plot([d["standoff"]], [d["J_best"]], "o", color=c)
    md = out["measured_dog"]["l_drive"]
    ax.axvline(md, color="#8a8a8a", ls="--", lw=1, label=f"real collie, median {md:.1f} m behind")
    ax.axvline(2 * np.sqrt(14), color="#8a8a8a", ls="-.", lw=1, label="published drive point 2√N")
    ax.set(xlabel="dog's distance behind the flock centre on the axis (m)", ylabel="direction mismatch J", title="Direction-matched design, N = 14 (dots: best within coverage)")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[0, 1]
    for flock, c in zip(flocks, ("#eb6834", "#2a78d6")):
        dd = out["flocks"][flock]["designs"]["14"]["direction"]
        ax.plot(dd["y_grid"], dd["J_lat"], color=c, lw=2, label=f"{flock[:30]} at its standoff")
    ax.set(xlabel="lateral offset of the dog (m)", ylabel="direction mismatch J", title="A lateral position does not help the flock as a whole")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[0, 2]
    for flock, c in zip(flocks, ("#eb6834", "#2a78d6")):
        a_ = out["flocks"][flock]["designs"]["14"]["arc"]
        ax.bar([f"{flock[:14]}\nfirst-order", f"{flock[:14]}\ndirection"], [a_["single_dog_standoff"], out["flocks"][flock]["designs"]["14"]["direction"]["standoff"]], color=c, width=0.6)
    ax.axhline(md, color="#8a8a8a", ls="--", lw=1)
    ax.axhline(2 * np.sqrt(14), color="#8a8a8a", ls="-.", lw=1)
    ax.set(ylabel="designed standoff behind the centre (m)", title="Designed standoffs, N = 14 (dashed: collie, dash-dot: published)")
    ax.tick_params(axis="x", labelsize=7.5)
    metrics = [("success_rate", "success within 300 s"), ("time_to_target_s_median", "median time to target (s)"), ("cohesion_mean", "mean cohesion C (m)")]
    groups = [(flock, N) for flock in flocks for N in SIZES]
    w = 0.16
    for ax, (k, lab) in zip(axs[1], metrics):
        for j, dname in enumerate(dogs):
            vals = []
            for flock, N in groups:
                r = out["flocks"][flock]["results"].get(f"N{N} | {dname}")
                vals.append(np.nan if r is None or r[k] is None else r[k])
            ax.bar(np.arange(len(groups)) + (j - 2) * w, vals, w, color=COL[dname], label=dname)
        ax.set_xticks(range(len(groups)), [f"{flock[:11]}…\nN = {N}" for flock, N in groups], fontsize=7.5)
        ax.set_title(lab, fontsize=10)
        if k == "success_rate":
            ax.set_ylim(0, 1.05)
    axs[1, 0].legend(frameon=False, fontsize=7)
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A dog designed from Di Lorenzo's deconvolution with the measured kernels, against the published, data-derived and tuned dogs", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "figures" / "designed_dog.png", dpi=120)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "designed_dog.json").read_text()))
    else:
        main()
