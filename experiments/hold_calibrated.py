"""The hold rule (model.Params.hold_radius) under the calibrated sheep.

experiments/hold_strategies.py tests reach-and-release on the published sheep. Here the
same task and rule run on the calibrated model: M3tau posterior draws of the sheep weights,
the learned dog kernel (position, velocity and looming terms, gain 2) and the learned pair
attraction (gain 2), dt = 0.2 s, measured flock speed. The dog is the published Strömbom
dog, with and without the hold rule, on two tasks: the 100 m drive of closed_loop_convergence
(reaching, then holding) and a start at the goal (holding alone: sheep sampled from the Gaussian
goal density of experiments/target_density.py, dog 11 m away).

    .venv/bin/python experiments/hold_calibrated.py
Writes results/hold_calibrated.json and results/hold_calibrated_replay.json (one run of each
start-at-goal case for viz/closed_loop_replay.html).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import pair_attraction as PA  # noqa: E402
from common import draws, load, posterior  # noqa: E402
from shepherd.model import HOLD, SLOW, simulate  # noqa: E402

OUT = ROOT / "results"
SEED, N_DRAW, REPS, DT, T_S, START, HOLD_R, HOLD_REL, AFTER_S = 11, 16, 4, 0.2, 450.0, 100.0, 5.0, 10.0, 300.0


def fl(x):
    return None if x is None else float(x)


def score(res, p, label):
    T, dt = res.pos_s.shape[-1], p.dt
    G = res.pos_s.mean(1)
    rho = np.linalg.norm(G - np.asarray(p.target)[None, :, None], axis=1)
    reached = rho < HOLD_R
    t_reach = np.where(reached.any(1), reached.argmax(1), T)
    ok = t_reach < T
    after = int(AFTER_S / dt)
    rows = []
    for r in np.flatnonzero(ok):
        t0 = t_reach[r]
        if t0 + after > T:
            continue
        sl = slice(t0, t0 + after)
        spread = np.linalg.norm(res.pos_s[r, :, :, sl] - G[r, :, sl][None], axis=1)
        moving = (np.linalg.norm(np.diff(res.pos_s[r, :, :, sl], axis=-1), axis=1) > 1e-9).mean()
        inside = (np.linalg.norm(res.pos_s[r, :, :, sl], axis=1) <= HOLD_R).mean()
        m = res.mode[r, sl]
        hold_eps = int(((m == HOLD)[1:] & (m != HOLD)[:-1]).sum() + (m[0] == HOLD))
        rows.append(dict(rho=rho[r, sl].mean(), rho_p90=np.percentile(rho[r, sl], 90), inside=inside, moving=moving,
                         maxspread=np.median(spread.max(0)), hold=(m == HOLD).mean(), slow=(m == SLOW).mean(), eps=hold_eps, final=rho[r, -1]))
    k = lambda key: fl(np.mean([q[key] for q in rows])) if rows else None
    out = dict(label=label, success_rate=fl(ok.mean()), t_reach_median_s=fl(np.median(t_reach[ok]) * dt) if ok.any() else None, n_scored=len(rows),
               post_rho_mean=k("rho"), post_rho_p90=k("rho_p90"), post_inside_frac=k("inside"), post_sheep_moving_frac=k("moving"),
               post_maxspread_median=k("maxspread"), post_hold_frac=k("hold"), post_slow_frac=k("slow"),
               hold_episodes_mean=k("eps"), reengagements_mean=fl(np.mean([max(0, q["eps"] - 1) for q in rows])) if rows else None,
               final_rho_median=fl(np.median([q["final"] for q in rows])) if rows else None)
    print(f"  {label:28s} reached {out['success_rate']:.2f} at {out['t_reach_median_s']} s; after: rho {out['post_rho_mean']}, "
          f"inside {out['post_inside_frac']}, moving {out['post_sheep_moving_frac']}, re-engagements {out['reengagements_mean']}", flush=True)
    return out


def score_hold(res, p, label, after_s=AFTER_S):
    """Started at the goal: does the flock stay? Metrics over the whole run."""
    after = min(int(after_s / p.dt), res.pos_s.shape[-1])
    G = res.pos_s.mean(1)
    rho = np.linalg.norm(G - np.asarray(p.target)[None, :, None], axis=1)[:, :after]
    r = np.linalg.norm(res.pos_s[..., :after], axis=2)
    moving = (np.linalg.norm(np.diff(res.pos_s[..., :after], axis=-1), axis=2) > 1e-9).mean()
    spread = np.linalg.norm(res.pos_s[..., :after] - G[:, None, :, :after], axis=2).max(1)
    m = res.mode[:, :after]
    out = dict(label=label, rho_mean=fl(rho.mean()), rho_p90=fl(np.percentile(rho, 90)), rho_final_median=fl(np.median(rho[:, -1])),
               inside_frac=fl((r <= HOLD_R).mean()), inside_final=fl((r[..., -1] <= HOLD_R).mean()), sheep_moving_frac=fl(moving),
               maxspread_median=fl(np.median(spread)), hold_frac=fl((m == HOLD).mean()), slow_frac=fl((m == SLOW).mean()),
               reengagements_mean=fl(((m[:, 1:] == HOLD) & (m[:, :-1] != HOLD)).sum(1).mean() - 1) if (m == HOLD).any() else None)
    print(f"  {label:28s} at goal: rho {out['rho_mean']:.1f} (final median {out['rho_final_median']:.1f}), inside {out['inside_frac']:.2f} "
          f"(final {out['inside_final']:.2f}), moving {out['sheep_moving_frac']:.2f}, hold {out['hold_frac']:.2f}, re-engagements {out['reengagements_mean']}", flush=True)
    return out


def main():
    tic = time.time()
    sheep = draws(posterior(load("M3tau"), "M3tau"), N_DRAW)
    R = N_DRAW * REPS
    per = {k: np.repeat(np.asarray(v)[:N_DRAW], REPS) for k, v in sheep.items()}
    base = replace(PA.base_params(2.0), dt=DT, n_steps=int(T_S / DT) + 1, box_length=START)
    out = dict(setup=dict(seed=SEED, runs=R, dt=DT, seconds=T_S, start_m=START, hold_radius=HOLD_R, hold_release=HOLD_REL,
                          per_run_keys=sorted(per), r_dog_median=fl(np.median(per["r_dog"])) if "r_dog" in per else None), results={}, at_goal={})
    for label, kw in (("calibrated/published_dog", {}), ("calibrated/hold_rule", dict(hold_radius=HOLD_R, hold_release=HOLD_REL))):
        res = simulate(replace(base, **kw), R, seed=SEED, per_run=per)
        out["results"][label] = score(res, base, label)
    # holding alone: start at the goal density (s = 2.04 m, 95% within 5 m), dog 11 m away
    rng = np.random.default_rng(SEED + 1)
    s_goal = HOLD_R / np.sqrt(-2 * np.log(0.05))
    pos_s = s_goal * rng.standard_normal((R, 14, 2))
    th = 2 * np.pi * rng.random((R, 14))
    init = {"pos_s": pos_s, "vel_s": np.stack([np.cos(th), np.sin(th)], -1), "pos_d": np.tile([[-11.0, 0.0]], (R, 1))}
    base_g = replace(base, n_steps=int(AFTER_S / DT) + 1)
    replay = []
    for label, kw in (("calibrated/published_dog", {}), ("calibrated/hold_rule", dict(hold_radius=HOLD_R, hold_release=HOLD_REL))):
        res = simulate(replace(base_g, **kw), R, seed=SEED + 2, per_run=per, init=init)
        out["at_goal"][label] = score_hold(res, base_g, label)
        pos, _ = res.run(0)
        G = pos[:14].mean(0)
        step = max(1, int(round(1.0 / DT)))  # one frame per second
        sl = slice(0, pos.shape[-1], step)
        replay.append(dict(key="calib_" + label.split("/")[1], N=14, n_dogs=1, static=False, r_goal=HOLD_R,
                           l_sep=fl(base_g.resolved().l_sep), l_drive=fl(base_g.resolved().l_drive), r_dog=fl(per["r_dog"][0]) if "r_dog" in per else None,
                           label=("Calibrated sheep at the goal: " + ("published dog" if "published" in label else "hold rule")),
                           note=("The calibrated sheep (learned dog and attraction kernels, M3τ weights), sampled from the goal density with the "
                                 "dog 11 m away. " + ("The published drive rule has no stop and chases the flock away." if "published" in label else
                                                       "With the hold rule the dog withdraws beyond the notice radius and the flock stays: 95% inside 5 m for 300 s.")),
                           pos=np.round(np.moveaxis(pos, -1, 0)[sl], 2).tolist(), mode=res.mode[0][sl].astype(int).tolist(),
                           rho=np.round(np.linalg.norm(G, axis=0)[sl], 2).tolist(),
                           spread=np.round(np.linalg.norm(pos[:14] - G[None], axis=1).max(0)[sl], 2).tolist(),
                           ppar=[0.0] * len(range(*sl.indices(pos.shape[-1]))), t_reach=0, v_sheep=fl(base_g.v_sheep), v_dog=fl(base_g.v_dog)))
    (OUT / "hold_calibrated_replay.json").write_text(json.dumps(replay, separators=(",", ":")))
    out["seconds"] = fl(time.time() - tic)
    (OUT / "hold_calibrated.json").write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
