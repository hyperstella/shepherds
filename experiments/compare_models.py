"""Phase 5: held-out comparison of sheep models (Q2) and the hierarchy test (Q3).

All models face the same external inputs: the real dog's path and the real
flock's mean speed, in 15 s windows that restart from the real flock state.

    .venv/bin/python experiments/compare_models.py
Needs results/tables/{M0,M0s,M0b,M0s_dt02,M1,M1tau,hybrid}.npz (build_table.py).
Writes results/compare_models.json and results/figures/{heldout,hierarchy_models}.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import abc as ABC, data, kernels as K, protocol as PR, scenario as SC  # noqa: E402
from shepherd.model import Params  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from build_table import FIT_STATS, HYBRID_PRIORS, MODELS, V_SHEEP  # noqa: E402

TABLES, OUT = ROOT / "results" / "tables", ROOT / "results"
COLS = [PR.SUMMARY_KEYS.index(k) for k in FIT_STATS]
ACCEPT = 0.01


def load(name):
    return dict(np.load(TABLES / f"{name}.npz", allow_pickle=False))


def data_scale(meta):
    """Window-to-window SD of each statistic in the data (units for errors)."""
    return np.nanstd(meta["obs"][:, COLS], axis=0)


def heldout_param(tab, name):
    """Leave-one-session-out for a calibrated parametric model (round-1 table, full prior)."""
    names = [str(n) for n in tab["names"]]
    priors = MODELS[name]["priors"]
    th = {k: tab[f"th1_{k}"] for k in names}
    sims = tab["tab1"][..., COLS]
    sess, dur, obs = tab["meta_sessions"], tab["meta_durations"], tab["meta_obs"][:, COLS]
    out = {}
    for s in np.unique(sess):
        tr, te = sess != s, sess == s
        S_tr = ABC.aggregate(sims, dur, tr)
        post = ABC.fit(th, S_tr, ABC.aggregate(obs[None], dur, tr)[0], ABC.mad_scale(S_tr), priors, ACCEPT)
        out[int(s)] = ABC.aggregate(sims[post.idx], dur, te).mean(0)
    return out


def heldout_fixed(tab):
    sess, dur = tab["meta_sessions"], tab["meta_durations"]
    return {int(s): ABC.aggregate(tab["tab"][..., COLS], dur, sess == s).mean(0) for s in np.unique(sess)}


def heldout_hybrid(tab):
    sess, dur, obs = tab["meta_sessions"], tab["meta_durations"], tab["meta_obs"][:, COLS]
    out = {}
    for s in np.unique(sess):
        f = str(int(s))
        th = {k: tab[f"th_{f}_{k}"] for k in HYBRID_PRIORS}
        sims = tab[f"tab_{f}"][..., COLS]
        tr, te = sess != s, sess == s
        S_tr = ABC.aggregate(sims, dur, tr)
        post = ABC.fit(th, S_tr, ABC.aggregate(obs[None], dur, tr)[0], ABC.mad_scale(S_tr), HYBRID_PRIORS, 0.03)
        out[int(s)] = ABC.aggregate(sims[post.idx], dur, te).mean(0)
    return out


def heldout_train_mean(meta):
    """Reference predictor: the training sessions' own average."""
    sess, dur, obs = meta["sessions"], meta["durations"], meta["obs"][:, COLS]
    return {int(s): ABC.aggregate(obs[None], dur, sess != s)[0] for s in np.unique(sess)}


def score(pred, meta):
    sess, dur, obs = meta["sessions"], meta["durations"], meta["obs"][:, COLS]
    scale = data_scale(meta)
    z = np.array([(pred[int(s)] - ABC.aggregate(obs[None], dur, sess == s)[0]) / scale for s in np.unique(sess)])
    w = np.array([dur[sess == s].sum() for s in np.unique(sess)])
    per_stat = np.sqrt(np.nansum(w[:, None] * z ** 2, 0) / w.sum())
    return {"rms_z": float(np.sqrt(np.nanmean(per_stat ** 2))), "per_stat": dict(zip(FIT_STATS, map(float, per_stat)))}


def posterior_all(tab, name, n=48, seed=0):
    names = [str(k) for k in tab["names"]]
    priors = MODELS[name]["priors"]
    th = {k: tab[f"th2_{k}"] for k in names}
    sims = ABC.aggregate(tab["tab2"][..., COLS], tab["meta_durations"])
    obs = ABC.aggregate(tab["meta_obs"][None, :, COLS], tab["meta_durations"])[0]
    post = ABC.fit(th, sims, obs, ABC.mad_scale(sims), priors, ACCEPT)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(post.samples[names[0]]), n, replace=False)
    return post, {k: post.samples[k][pick] for k in names}


def hierarchy_of(segs, key="session"):
    recs = PR.hierarchy(segs, 2.0, 0.5, True, min_duration_s=8.0)
    st = PR.hierarchy_stats(recs)
    st["spearman_ci95"] = PR.cluster_bootstrap(recs, lambda r: PR.hierarchy_stats(r)["spearman"], 500, 0, key)["ci95"]
    return st


def main():
    drives = data.load_drives(data.load_sessions())
    res = {"heldout": {}, "posteriors": {}, "hierarchy": {}}

    # ---- Q2: held-out sessions ----
    m1, m1t, hyb = load("M1"), load("M1tau"), load("hybrid")
    meta1 = {k[5:]: v for k, v in m1.items() if k.startswith("meta_")}
    meta02 = {k[5:]: v for k, v in m1t.items() if k.startswith("meta_")}
    preds = {
        ("dt=1 s", "training-data average"): (heldout_train_mean(meta1), meta1),
        ("dt=1 s", "M0 published"): (heldout_fixed(load("M0")), meta1),
        ("dt=1 s", "M0s published + flock speed"): (heldout_fixed(load("M0s")), meta1),
        ("dt=1 s", "M0b + body-length spacing"): (heldout_fixed(load("M0b")), meta1),
        ("dt=1 s", "M1 calibrated"): (heldout_param(m1, "M1"), meta1),
        ("dt=0.2 s", "training-data average"): (heldout_train_mean(meta02), meta02),
        ("dt=0.2 s", "M0s published + flock speed"): (heldout_fixed(load("M0s_dt02")), meta02),
        ("dt=0.2 s", "M1tau calibrated"): (heldout_param(m1t, "M1tau"), meta02),
        ("dt=0.2 s", "Learned kernels + gains"): (heldout_hybrid(hyb), meta02),
    }
    for (dt, name), (pred, meta) in preds.items():
        res["heldout"][f"{dt} | {name}"] = score(pred, meta)
        print(f"{dt:8s} {name:32s} held-out RMS z = {res['heldout'][f'{dt} | {name}']['rms_z']:.2f}")

    # ---- posteriors on all data ----
    post1, draws1 = posterior_all(m1, "M1")
    post1t, draws1t = posterior_all(m1t, "M1tau")
    res["posteriors"]["M1"] = post1.summary(MODELS["M1"]["priors"])
    res["posteriors"]["M1tau"] = post1t.summary(MODELS["M1tau"]["priors"])
    thh = {k: hyb[f"th_all_{k}"] for k in HYBRID_PRIORS}
    simsh = ABC.aggregate(hyb["tab_all"][..., COLS], meta02["durations"])
    obsh = ABC.aggregate(meta02["obs"][None, :, COLS], meta02["durations"])[0]
    posth = ABC.fit(thh, simsh, obsh, ABC.mad_scale(simsh), HYBRID_PRIORS, 0.03)
    res["posteriors"]["hybrid"] = posth.summary(HYBRID_PRIORS)
    res["coverage_M1"] = ABC.coverage({k: m1[f"th1_{k}"] for k in m1["names"]},
                                      ABC.aggregate(m1["tab1"][..., COLS], meta1["durations"]),
                                      MODELS["M1"]["priors"], n_test=50, accept=ACCEPT)
    print("coverage of 90% intervals (M1):", {k: round(v, 2) for k, v in res["coverage_M1"].items()})

    # ---- Q3: hierarchy under the fair protocol, dt = 0.2 s windows ----
    real = SC.window_segments(drives, dt=0.2)
    res["hierarchy"]["data"] = hierarchy_of(real)
    sur = PR.timeshift_surrogate([s for s in real if s.duration >= 8.0], n_surr=200, seed=1)
    res["hierarchy"]["data_surrogate_p"] = float((1 + np.sum(sur["spearman"] >= res["hierarchy"]["data"]["spearman"]))
                                                / (1 + len(sur["spearman"])))
    base02 = Params(v_sheep=V_SHEEP[0.2])
    n_draw = 16
    per = {k: v[:n_draw] for k, v in draws1t.items()}
    res["hierarchy"]["M0s published"] = hierarchy_of(SC.unbatch(SC.replay_windows(drives, Params(), n_draw, 7, 0.2,
                                                                                   speed_replay=True)))
    res["hierarchy"]["M1tau posterior"] = hierarchy_of(SC.unbatch(SC.replay_windows(drives, base02, n_draw, 7, 0.2,
                                                                                    per_run=per, speed_replay=True)))
    fit = np.load(TABLES / "kernel_fit.npz")
    km = K.KernelModel(terms=tuple(str(t) for t in fit["terms"]))
    c_all = hyb["coef_all"]
    gi = np.random.default_rng(0).choice(len(posth.samples["E"]), n_draw, replace=False)
    g = {k: posth.samples[k][gi] for k in HYBRID_PRIORS}
    coef = K.apply_gains(km, c_all, {"E": g["E"], "D": g["dog"], "DV": g["dog"], "DL": g["dog"]})
    res["hierarchy"]["learned posterior"] = hierarchy_of(SC.unbatch(SC.replay_windows_kernel(
        drives, km, coef, g["sigma"], n_draw, 7, 0.2)))
    no_dog = K.apply_gains(km, c_all, {"E": g["E"], "D": 0 * g["dog"], "DV": 0 * g["dog"], "DL": 0 * g["dog"]})
    segs_null = SC.replay_windows_kernel(drives, km, no_dog, g["sigma"], n_draw, 7, 0.2)
    res["hierarchy"]["learned, dog removed (polarized null)"] = hierarchy_of(SC.unbatch(segs_null))
    res["hierarchy"]["learned, dog removed: polarization"] = float(np.nanmean(
        [np.nanmean(PR.summaries(s)["P_med"]) for s in segs_null]))
    for k, v in res["hierarchy"].items():
        if isinstance(v, dict):
            print(f"{k:40s} rho {v['spearman']:+.3f} CI {np.round(v['spearman_ci95'], 3)}  dog rank {v.get('dog_rank')}")

    (OUT / "compare_models.json").write_text(json.dumps(res, indent=2, default=float))
    plot(res)


def plot(res):
    C = {"training-data average": "#b8b7b1", "M0": "#eb6834", "M1": "#2a78d6", "Learned": "#1baf7a"}
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    names = list(res["heldout"])
    vals = [res["heldout"][n]["rms_z"] for n in names]
    cols = [next((c for k, c in C.items() if k in n.split("| ")[1]), "#52514e") for n in names]
    y = np.arange(len(names))[::-1]
    axs[0].barh(y, vals, color=cols)
    axs[0].set_yticks(y, [n.replace(" | ", ": ") for n in names], fontsize=8)
    axs[0].set_xlabel("Held-out error, RMS over statistics (units: window-to-window SD)")
    axs[0].set_title("Q2  Predicting unseen sessions (leave one session out)", fontsize=10)
    h = {k: v for k, v in res["hierarchy"].items() if isinstance(v, dict)}
    labels = list(h)
    vals = [h[k]["spearman"] for k in labels]
    ci = np.array([h[k]["spearman_ci95"] for k in labels])
    x = np.arange(len(labels))
    axs[1].bar(x, vals, color=["#52514e", "#eb6834", "#2a78d6", "#1baf7a", "#b8b7b1"][: len(labels)])
    axs[1].errorbar(x, vals, yerr=np.abs(ci.T - vals), fmt="none", ecolor="#17201a", capsize=3)
    axs[1].axhline(0, color="#52514e", lw=0.8)
    axs[1].set_xticks(x, [l.replace(" (", "\n(").replace(", ", ",\n") for l in labels], fontsize=8)
    axs[1].set_ylabel("Sheep-level ρ(followers, front position)")
    axs[1].set_title("Q3  Front-to-back leadership, same windows and protocol", fontsize=10)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "heldout_hierarchy.png", dpi=110)


if __name__ == "__main__":
    main()
