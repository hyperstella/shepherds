"""Fix the calibrated model's dog response with the learned dog kernels.

The multi-horizon fit (M3tau) cannot identify how strongly sheep react to the
dog: in real drives the flock already moves the right way. The learned
kernels (Phase 4) estimate that response directly from the tracks. Here the
learned dog acceleration enters the heading rule (model.Params.dog_kernel)
with one gain g, swept and chosen on the fit statistics.

    .venv/bin/python experiments/dog_response.py
Writes results/dog_response.json and results/figures/dog_response.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from common import COLS, FIT_STATS, ROOT, draws, load, posterior  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import abc as ABC, data, kernels as K, protocol as PR, scenario as SC  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402
from build_table import V_SHEEP, measured_dog  # noqa: E402

OUT = ROOT / "results"
GAINS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
N_DRAW, DT = 16, 0.2


def learned_dog_kernel():
    fit = np.load(OUT / "tables" / "kernel_fit.npz")
    m = K.KernelModel(terms=tuple(str(t) for t in fit["terms"]))
    sl, c = m.slices(), fit["c"]
    return (tuple(m.dog.knots), tuple(c[sl["D"]]), tuple(c[sl["DV"]]), tuple(c[sl["DL"]]))


def turn_test(base, sheep, dog, n=N_DRAW, reps=4, t_max=120.0, seed=5):
    """Flock moving east, target 40 m north: share of runs that arrive within t_max."""
    R = n * reps
    rng = np.random.default_rng(seed)
    pos_s = rng.normal(0, 1.0, (R, 14, 2))
    vel_s = np.tile([1.0, 0.0], (R, 14, 1))
    pos_d = np.tile([-6.0, 0.0], (R, 1)) + rng.normal(0, 0.5, (R, 2))
    p = replace(base, dt=DT, n_steps=int(t_max / DT) + 1, target=(0.0, 40.0), flank_sigma=1e-9,
                dog_speed_gain=dog["dog_speed_gain"])
    per = {**{k: np.repeat(np.asarray(v)[:n], reps) for k, v in sheep.items()},
           **{k: np.full(R, v) for k, v in dog.items() if v is not None}}
    res = simulate(p, R, seed, per_run=per, init={"pos_s": pos_s, "vel_s": vel_s, "pos_d": pos_d,
                                                   "vel_d": np.tile([1.0, 0.0], (R, 1))})
    G = res.pos_s.mean(1)
    d = np.linalg.norm(G - np.array([0.0, 40.0])[None, :, None], axis=1)
    ok = (d < 5).any(1)
    t = np.where(ok, (d < 5).argmax(1) * DT, np.nan)
    C = np.linalg.norm(res.pos_s - G[:, None], axis=2).mean(1)
    return {"success": float(ok.mean()), "time_s": float(np.nanmedian(t)) if ok.any() else None,
            "cohesion": float(C.mean())}


def main():
    drives = data.load_drives(data.load_sessions())
    dog = measured_dog()
    tab = load("M3tau")
    sheep = draws(posterior(tab, "M3tau"), N_DRAW)
    meta = {k[5:]: v for k, v in tab.items() if k.startswith("meta_")}
    obs_w = meta["obs"][:, COLS]
    obs = ABC.aggregate(obs_w[None], meta["durations"])[0]
    scale = np.nanstd(obs_w, axis=0)
    real_long = SC.real_long_segments(drives, dt=DT)
    lw = np.array([s.duration for s in real_long])
    long_obs = {k: float(sum(np.nanmean(PR.long_summaries(s)[k]) * w for s, w in zip(real_long, lw)) / lw.sum())
                for k in PR.LONG_KEYS}
    kern = learned_dog_kernel()
    out = {"data": {"short": dict(zip(FIT_STATS, map(float, obs))), "long": long_obs}, "gains": {}}
    for g in GAINS:
        base = Params(v_sheep=V_SHEEP[DT], dog_kernel=kern if g > 0 else None, g_dogk=g)
        segs = SC.replay_windows(drives, base, N_DRAW, 7, DT, per_run=sheep, speed_replay=True)
        short = np.stack([np.stack([np.nanmean(PR.summaries(s)[k]) for k in FIT_STATS]) for s in segs])
        sim = ABC.aggregate(short[None], meta["durations"])[0]
        z = (sim - obs) / scale
        longs = SC.closed_loop_drives(drives, base, N_DRAW, 3, DT, per_run=sheep, dog=dog)
        long = {k: float(sum(np.nanmean(PR.long_summaries(s)[k]) * w for s, w in zip(longs, lw)) / lw.sum())
                for k in PR.LONG_KEYS}
        turn = turn_test(base, sheep, dog)
        out["gains"][g] = {"short_rms_z": float(np.sqrt(np.nanmean(z ** 2))), "slope_4a": float(sim[FIT_STATS.index("slope_4a")]),
                           "long": long, "turn_test": turn}
        print(f"g={g:4.1f}  short RMS z {out['gains'][g]['short_rms_z']:.2f}  slope_4a {out['gains'][g]['slope_4a']:+.2f} "
              f"(data {obs[FIT_STATS.index('slope_4a')]:+.2f})  long C {long['C_med']:.2f} P {long['P_med']:.2f} "
              f"drift {long['C_drift']:.2f} far {long['frac_far']:.2f}  turn test {turn['success']:.2f} "
              f"in {turn['time_s']} s (C {turn['cohesion']:.1f})")
    (OUT / "dog_response.json").write_text(json.dumps(out, indent=2, default=float))
    plot(out)


def plot(out):
    gs = [float(g) for g in out["gains"]]
    G = list(out["gains"].values())
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    axs[0].plot(gs, [g["short_rms_z"] for g in G], "o-", color="#2a78d6")
    axs[0].set_ylabel("Short-window error (RMS z)")
    axs[1].plot(gs, [g["slope_4a"] for g in G], "o-", color="#2a78d6")
    axs[1].axhline(out["data"]["short"]["slope_4a"], color="#52514e", ls="--", label="real flock")
    axs[1].set_ylabel("Flock turns away from a flanking dog (Fig 4a slope)"), axs[1].legend(fontsize=8)
    axs[2].plot(gs, [g["long"]["C_med"] for g in G], "o-", color="#2a78d6")
    axs[2].axhline(out["data"]["long"]["C_med"], color="#52514e", ls="--")
    axs[2].set_ylabel("Long drives: cohesion C (m)")
    axs[3].plot(gs, [g["turn_test"]["success"] for g in G], "o-", color="#1baf7a")
    axs[3].set_ylabel("Turn test: share reaching a target 90° off"), axs[3].set_ylim(0, 1.05)
    for ax in axs:
        ax.set_xscale("symlog", linthresh=0.5), ax.set_xlabel("Gain g on the learned dog response")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Calibrated flock (M3τ) + learned dog response", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "dog_response.png", dpi=110)


if __name__ == "__main__":
    main()
