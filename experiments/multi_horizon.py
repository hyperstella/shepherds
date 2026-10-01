"""Evaluate the multi-horizon calibration (M3, M3tau) against single-horizon fits.

1. Posteriors (spacing, dog radius, relaxation time, weight shares).
2. Held-out sessions: short-window statistics (as in compare_models.py), with
   multi-horizon models fitted on both horizons of the training sessions only.
3. Closed-loop check: each posterior re-runs the 20 long drives with the
   measured collie policy; long-horizon statistics vs the real flock.
4. Hierarchy (fair protocol, dt = 0.2 s windows) for M3tau.

    .venv/bin/python experiments/multi_horizon.py
Writes results/multi_horizon.json and results/figures/multi_horizon.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402

import numpy as np  # noqa: E402

from common import COLS, FIT_STATS, MODELS, ROOT, draws, load, posterior  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import abc as ABC, data, protocol as PR, scenario as SC  # noqa: E402
from shepherd.model import Params  # noqa: E402
from build_table import V_SHEEP, WEIGHTS, measured_dog  # noqa: E402

OUT = ROOT / "results"
N_DRAW = 16


def heldout(tab, name):
    """Leave-one-session-out prediction of the short-window statistics."""
    sess, dur = tab["meta_sessions"], tab["meta_durations"]
    obs = tab["meta_obs"][:, COLS]
    lsess = tab.get("lmeta_sessions")
    scale = np.nanstd(obs, axis=0)
    z, w = [], []
    for s in np.unique(sess):
        post = posterior(tab, name, rnd=1, mask=sess != s, lmask=None if lsess is None else lsess != s)
        pred = np.nanmedian(ABC.aggregate(tab["tab1"][post.idx][..., COLS], dur, sess == s), axis=0)
        z.append((pred - ABC.aggregate(obs[None], dur, sess == s)[0]) / scale)
        w.append(dur[sess == s].sum())
    z, w = np.array(z), np.array(w)
    per = np.sqrt(np.nansum(w[:, None] * z ** 2, 0) / w.sum())
    return {"rms_z": float(np.sqrt(np.nanmean(per ** 2))), "per_stat": dict(zip(FIT_STATS, map(float, per)))}


def long_check(drives, sheep, dt, dog, n=N_DRAW):
    """Closed-loop long drives with the measured dog: pooled long-horizon statistics."""
    base = Params(v_sheep=V_SHEEP[dt])
    per = None if sheep is None else {k: np.asarray(v)[:n] for k, v in sheep.items()}
    segs = SC.closed_loop_drives(drives, base if sheep is not None else Params(), n, 3, dt, per_run=per, dog=dog)
    sm = [PR.long_summaries(s) for s in segs]
    w = np.array([s.duration for s in segs])
    return {k: float(sum(np.nanmean(x[k]) * ww for x, ww in zip(sm, w)) / w.sum()) for k in PR.LONG_KEYS}


def main():
    drives = data.load_drives(data.load_sessions())
    dog = measured_dog()
    out = {"posteriors": {}, "heldout": {}, "closed_loop": {}, "hierarchy": {}}
    for dt, label in ((1.0, "real flock"), (0.2, "real flock, dt 0.2")):
        real = SC.real_long_segments(drives, dt=dt)
        sm = [PR.long_summaries(s) for s in real]
        w = np.array([s.duration for s in real])
        out["closed_loop"][label] = {k: float(sum(np.nanmean(x[k]) * ww for x, ww in zip(sm, w)) / w.sum())
                                     for k in PR.LONG_KEYS}
    out["closed_loop"]["published M0"] = long_check(drives, None, 1.0, dog)

    for name in ("M2", "M3", "M2tau", "M3tau"):
        tab = load(name)
        dt = float(tab["dt"])
        post = posterior(tab, name)
        s = post.summary(MODELS[name]["priors"])
        out["posteriors"][name] = {k: v for k, v in s.items() if k not in WEIGHTS}
        out["posteriors"][name]["weight_shares"] = ABC.shares(post.samples, list(WEIGHTS))
        out["heldout"][name] = heldout(tab, name)
        out["closed_loop"][name] = long_check(drives, draws(post, N_DRAW), dt, dog)
        print(f"{name:6s} held-out RMS z {out['heldout'][name]['rms_z']:.2f} | closed loop "
              + " ".join(f"{k}={v:.2f}" for k, v in out["closed_loop"][name].items())
              + " | shares " + " ".join(f"{k[2:] if k.startswith('w_') else k}={v['median']:.2f}"
                                       for k, v in out["posteriors"][name]["weight_shares"].items()))
        if name == "M3tau":
            per = draws(post, N_DRAW)
            segs = SC.replay_windows(drives, Params(v_sheep=V_SHEEP[0.2]), N_DRAW, 7, 0.2, per_run=per, speed_replay=True)
            recs = PR.hierarchy(SC.unbatch(segs), 2.0, 0.5, True, min_duration_s=8.0)
            st = PR.hierarchy_stats(recs)
            ci = PR.cluster_bootstrap(recs, lambda r: PR.hierarchy_stats(r)["spearman"], 300, 0)["ci95"]
            out["hierarchy"]["M3tau posterior"] = {"rho": st["spearman"], "ci95": ci, "dog_rank": st.get("dog_rank")}
            print("M3tau hierarchy:", out["hierarchy"]["M3tau posterior"])
    print("real flock closed-loop stats:", {k: round(v, 2) for k, v in out["closed_loop"]["real flock"].items()})
    print("published M0:               ", {k: round(v, 2) for k, v in out["closed_loop"]["published M0"].items()})
    (OUT / "multi_horizon.json").write_text(json.dumps(out, indent=2, default=float))
    plot(out)


def plot(out):
    cl = out["closed_loop"]
    names = ["real flock", "published M0", "M2", "M3", "M2tau", "M3tau"]
    cols = {"real flock": "#52514e", "published M0": "#eb6834", "M2": "#86b6ef", "M3": "#2a78d6",
            "M2tau": "#9ed9bf", "M3tau": "#1baf7a"}
    keys = [("C_med", "Cohesion C (m)"), ("C_drift", "Cohesion drift, end − start (m)"),
            ("frac_far", "Share of sheep > 5 m from centre"), ("P_med", "Polarization P")]
    fig, axs = plt.subplots(1, len(keys), figsize=(17, 3.8))
    for ax, (k, lab) in zip(axs, keys):
        ax.bar(range(len(names)), [cl[n][k] for n in names], color=[cols[n] for n in names])
        ax.set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_title(lab, fontsize=10), ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Long closed-loop drives with the measured collie policy: single-horizon (M2) vs multi-horizon (M3) fits",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "multi_horizon.png", dpi=110)


if __name__ == "__main__":
    main()
