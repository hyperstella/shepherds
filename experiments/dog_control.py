"""Phase 6 (Q4): closed-loop herding with different dog policies.

The flock follows the calibrated sheep rules (M2tau posterior draws, dt =
0.2 s, constant measured speed). The task, as in the experiment: drive the
flock to a target 60 m away. Dogs:

- Strömbom heuristic as published (drive point 2*sqrt(N) m behind, 1.5 m/s,
  instant turning, no flanking).
- Data-derived dog: drive distance, flanking (OU), turning time and speed law
  measured from the real collie (shepherd/dog_policy.py).
- Tuned heuristic: the best of a small grid, chosen on separate runs.

    .venv/bin/python experiments/dog_control.py
Writes results/dog_control.json and results/figures/dog_control.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, dog_policy as DP, protocol as PR  # noqa: E402
from shepherd.model import COLLECT, Params, simulate  # noqa: E402
from build_table import FIT_STATS, V_SHEEP  # noqa: E402

TABLES, OUT = ROOT / "results" / "tables", ROOT / "results"
COLS = [PR.SUMMARY_KEYS.index(k) for k in FIT_STATS]
DT, T_MAX, START, ARRIVE = 0.2, 300.0, 60.0, 5.0
N_DRAW, REPS = 16, 4
SIZES = (14, 50, 100)


def sheep_posterior(n=N_DRAW, seed=0, model="M2tau"):
    """Posterior draws of the sheep parameters (both horizons for multi-horizon tables)."""
    import common

    tab = common.load(model)
    return common.draws(common.posterior(tab, model), n, seed)


def dog_settings(N, measured):
    s = np.sqrt(N / 14)  # flock radius grows like sqrt(N)
    common = {"noise_dog": 0.3, "l_collect": 2.0, "l_sep": 2.0 * N ** (2 / 3)}
    return {
        "Strömbom (published)": {**common, "l_drive": 2.0 * np.sqrt(N), "v_dog": 1.5, "tau_d": "instant",
                                 "flank_sigma": 0.0},
        "Data-derived dog": {**common, "l_drive": measured["l_drive"] * s, "v_dog": measured["v_dog"],
                             "tau_d": measured["tau_d"], "flank_sigma": measured["flank_sigma"] * s,
                             "flank_tau": measured["flank_tau"], "dog_speed_gain": measured["dog_speed_gain"]},
    }


def tuning_grid(N):
    s = np.sqrt(N / 14)
    common = {"noise_dog": 0.3, "l_collect": 2.0, "l_sep": 2.0 * N ** (2 / 3), "tau_d": "instant"}
    for l, v, fl in itertools.product((2.5, 4.0, 6.0, 9.0), (1.5, 3.0), (0.0, 2.5)):
        yield f"l_drive {l:g}·√(N/14), v {v:g}, flank {fl:g}", {**common, "l_drive": l * s, "v_dog": v,
                                                                 "flank_sigma": fl * s, "flank_tau": 2.7}


def run(job):
    """Simulate R runs for one (N, dog) setting; return per-run metrics."""
    N, dog, sheep, seed, social_always, dt, R, v_sheep = job
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, R)
    start = START * np.stack([np.cos(ang), np.sin(ang)], -1)
    r0 = 0.6 * np.sqrt(N) * np.sqrt(rng.random((R, N)))
    a0 = rng.uniform(0, 2 * np.pi, (R, N))
    pos_s = start[:, None] + np.stack([r0 * np.cos(a0), r0 * np.sin(a0)], -1)
    pos_d = start * (1 + (8 + 0.6 * np.sqrt(N)) / START) + rng.normal(0, 2, (R, 2))  # behind the flock
    th = rng.uniform(0, 2 * np.pi, (R, N))
    init = {"pos_s": pos_s, "vel_s": np.stack([np.cos(th), np.sin(th)], -1), "pos_d": pos_d}
    p = Params(n_sheep=N, dt=dt, v_sheep=v_sheep, n_steps=int(T_MAX / dt) + 1, box_length=START,
               flank_sigma=1e-9 if dog.get("flank_sigma", 0) > 0 else 0.0,
               dog_speed_gain=dog.get("dog_speed_gain"), social_always=social_always)
    dog = {**dog, "tau_d": dt if dog.get("tau_d") == "instant" else dog.get("tau_d")}
    per_run = {k: np.broadcast_to(np.asarray(v, float), (R,)).copy() for k, v in {**sheep, **dog}.items()
               if v is not None}
    res = simulate(p, R, seed + 1, per_run=per_run, init=init)
    G = res.pos_s.mean(1)  # (R, 2, T)
    dist = np.linalg.norm(G, axis=1)
    arrived = dist < ARRIVE
    t_arr = np.where(arrived.any(1), arrived.argmax(1), dist.shape[1])
    C = np.linalg.norm(res.pos_s - G[:, None], axis=2).mean(1)
    near = np.linalg.norm(res.pos_s - res.pos_d[:, None], axis=2).min(1) < 3.0
    mask = np.arange(dist.shape[1])[None] < t_arr[:, None]
    m = lambda x: (x * mask).sum(1) / np.maximum(mask.sum(1), 1)
    return {
        "success": arrived.any(1),
        "time_s": t_arr * dt,
        "cohesion": m(C),
        "max_spread": np.where(mask, np.linalg.norm(res.pos_s - G[:, None], axis=2).max(1), 0).max(1),
        "pressure": m(near.astype(float)),
        "collect_frac": m((res.mode == COLLECT).astype(float)),
    }


def summarize(m):
    ok = m["success"]
    return {
        "success_rate": float(ok.mean()),
        "time_to_target_s_median": float(np.median(m["time_s"][ok])) if ok.any() else None,
        "cohesion_mean": float(np.mean(m["cohesion"])),
        "max_spread_median": float(np.median(m["max_spread"])),
        "dog_within_3m_frac": float(np.mean(m["pressure"])),
        "collect_frac": float(np.mean(m["collect_frac"])),
    }


def score(s):
    """Lower is better: expected time, counting failures as 2 * T_MAX."""
    t = s["time_to_target_s_median"] or T_MAX
    return s["success_rate"] * t + (1 - s["success_rate"]) * 2 * T_MAX


def main():
    drives = data.load_drives(data.load_sessions())
    measured = DP.estimate(drives)
    rep = lambda d: {k: np.repeat(v, REPS) for k, v in d.items()}
    m2t, m3, m3t = (rep(sheep_posterior(model=m)) for m in ("M2tau", "M3", "M3tau"))
    out = {"measured_dog": measured}
    R = N_DRAW * REPS
    flocks = {  # name: (sheep per-run parameters, social always on, dt, sheep speed)
        "published sheep (M0)": ({}, False, 1.0, 1.0),
        "calibrated M2tau (short horizon), grazing": (m2t, False, 0.2, V_SHEEP[0.2]),
        "calibrated M3 (multi-horizon), grazing": (m3, False, 1.0, V_SHEEP[1.0]),
        "calibrated M3tau (multi-horizon), grazing": (m3t, False, 0.2, V_SHEEP[0.2]),
        "calibrated M3tau (multi-horizon), social always on": (m3t, True, 0.2, V_SHEEP[0.2]),
    }
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as ex:
        for flock, (sheep_rep, social, dt, v_sheep) in flocks.items():
            res, tun = {}, {}
            # tuning on separate seeds
            jobs, keys = [], []
            for N in SIZES:
                for name, dog in tuning_grid(N):
                    jobs.append((N, dog, sheep_rep, 10_000 + N, social, dt, R, v_sheep))
                    keys.append((N, name, dog))
            tuned = {}
            for (N, name, dog), m in zip(keys, ex.map(run, jobs)):
                s = summarize(m)
                tun[f"N{N} | {name}"] = s
                if N not in tuned or score(s) < score(tuned[N][2]):
                    tuned[N] = (name, dog, s)
            # evaluation on fresh seeds
            jobs, keys = [], []
            for N in SIZES:
                dogs = dog_settings(N, measured)
                dogs[f"Tuned heuristic ({tuned[N][0]})"] = tuned[N][1]
                for name, dog in dogs.items():
                    jobs.append((N, dog, sheep_rep, 20_000 + N, social, dt, R, v_sheep))
                    keys.append((N, name))
            for (N, name), m in zip(keys, ex.map(run, jobs)):
                s = res[f"N{N} | {name}"] = summarize(m)
                print(f"[{flock}] N={N:3d} {name[:46]:46s} success {s['success_rate']:.2f}  time "
                      f"{s['time_to_target_s_median']}  C {s['cohesion_mean']:.2f}  dog<3m {s['dog_within_3m_frac']:.2f}  "
                      f"collect {s['collect_frac']:.2f}")
            out[flock] = {"results": res, "tuning": tun}
    (OUT / "dog_control.json").write_text(json.dumps(out, indent=2, default=float))
    plot(out)


def plot(out):
    dogs = ["Strömbom (published)", "Data-derived dog", "Tuned heuristic"]
    cols = {"Strömbom (published)": "#eb6834", "Data-derived dog": "#2a78d6", "Tuned heuristic": "#1baf7a"}
    metrics = [("success_rate", "Success within 300 s"), ("time_to_target_s_median", "Median time to target (s)"),
               ("cohesion_mean", "Mean cohesion C (m)"), ("dog_within_3m_frac", "Share of time dog < 3 m from a sheep")]
    flocks = [k for k in out if k != "measured_dog"]
    fig, axs = plt.subplots(len(flocks), 4, figsize=(18, 3.8 * len(flocks)))
    w = 0.26
    for row, flock in enumerate(flocks):
        res = out[flock]["results"]
        for ax, (k, lab) in zip(axs[row], metrics):
            for j, dname in enumerate(dogs):
                vals = []
                for N in SIZES:
                    key = next(kk for kk in res if kk.startswith(f"N{N} |") and dname in kk)
                    vals.append(res[key][k] if res[key][k] is not None else np.nan)
                ax.bar(np.arange(len(SIZES)) + (j - 1) * w, vals, w, color=cols[dname], label=dname)
            ax.set_xticks(range(len(SIZES)), [f"N = {n}" for n in SIZES])
            ax.set_title(f"{lab}\n(flock: {flock})", fontsize=9)
            if k in ("success_rate", "dog_within_3m_frac"):
                ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Q4  Closed-loop herding: drive the flock 60 m to a target (empty bars: no successful run)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "dog_control.png", dpi=110)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((OUT / "dog_control.json").read_text()))
    else:
        main()
