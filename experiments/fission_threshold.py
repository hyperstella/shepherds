"""A model for the size at which a driven Jadhav flock splits in two, and a test of its predictions.

Mechanism (docs/steer_and_shape.md §11.5). In a polarized flock with the dog behind at distance d, the
left and right halves, with centroids at +-Y/2, get opposite lateral pushes from the dog,
w_dog (Y/2) / sqrt(d^2 + (Y/2)^2), and are pulled together by attraction to the perceived neighbours
across the midline, w_att lambda s, with s the share of perceived neighbours that are across and lambda the
mean lateral component of a unit vector to one of them. Both sit in the same normalized heading sum,
so normalization, heading memory and alignment change the rate, not the sign. Once a gap opens, each
sheep's own half is nearer than the other, and a half of n sheep keeps at least K + 1 - n cross
neighbours among its K perceived:

    s_gap(N) = max(0, K + 1 - N/2) / K.

Fission when

    Phi(N) = w_dog (Y0/2) / sqrt(d^2 + (Y0/2)^2) / (w_att lambda s_gap(N)) > 1,
    Y0 = 1.6 sd_perp(N),  sd_perp(N) = sd_perp(N_ref) sqrt(N/N_ref),

with sd_perp(N_ref) the lateral standard deviation of a driven flock of N_ref = max(14, K + 1) sheep at the
setting (measured; the model needs at least K + 1 sheep), and
1.6 the separation of half-centroids for a symmetric unimodal profile (Gaussian: 2 sqrt(2/pi)).
s_gap = 0 at N >= 2K + 2, so that is a ceiling on N_c for any attraction.

lambda is calibrated once, on the published setting's measured 50% point (experiments/size_sweep.py).
Then N_c is predicted for the two other measured settings (out of sample) and for new ones: K = 7 and
15, w_dog 0.5, w_ali 0 and 3, the dog at 5 m. Each new prediction is checked by driving compact flocks
of sizes around it (drive only, published rule, r_dog 20, 30 runs, 300 steps).

    .venv/bin/python experiments/fission_threshold.py
Writes results/fission_threshold.json and results/figures/fission_threshold.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shape_stress as SS  # noqa: E402
from shape_trajectories import largest_group  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402
from size_sweep import compact, wilson  # noqa: E402

OUT = ROOT / "results"
T, RUNS, SEED = 300, 30, 83
SD_WINDOW = (60, 120)  # steps over which the 14-sheep lateral sd is measured
BASE = dict(w_att=1.5, w_ali=1.3, w_dog=1.0, d=7.5, K=10)
SETTINGS = {  # name: changes from BASE; "measured" ones were run in size_sweep.py
    "published, dog 7.5 m (calibration)": dict(),
    "published, dog 12 m": dict(d=12.0),
    "w_att 3, dog 7.5 m": dict(w_att=3.0),
    "K = 7": dict(K=7),
    "K = 15": dict(K=15),
    "w_dog 0.5": dict(w_dog=0.5),
    "w_ali 0": dict(w_ali=0.0),
    "w_ali 3": dict(w_ali=3.0),
    "dog 5 m": dict(d=5.0),
}
COL = SS.COL


def setting(name):
    return dict(BASE, **SETTINGS[name])


def run_flocks(st, N, rng, steps=T, seed=SEED):
    c0 = np.array([SS.FAR, 0.0])
    pos = c0 + compact(rng, RUNS, N)
    hd = SS.unit(np.array([-1.0, 0.0]) + 0.1 * rng.standard_normal((RUNS, N, 2)))
    per = dict(w_att=np.full(RUNS, st["w_att"]), w_ali=np.full(RUNS, st["w_ali"]), w_dog=np.full(RUNS, st["w_dog"]),
               r_dog=np.full(RUNS, 20.0), l_drive=np.full(RUNS, st["d"]), l_sep=np.full(RUNS, 1e6))
    return simulate(Params(n_sheep=N, n_steps=steps + 1, k_perceive=st["K"]), n_runs=RUNS, seed=seed + N, per_run=per,
                    init=dict(pos_s=pos, vel_s=hd, pos_d=np.tile(c0 + [st["d"], 0.0], (RUNS, 1)), vel_d=np.tile([-1.0, 0.0], (RUNS, 1))))


def lateral_sd(st, rng, N=14):
    """Median lateral sd (about the centroid, across the flock's heading) of driven N-sheep flocks over SD_WINDOW, unsplit runs only."""
    res = run_flocks(st, N, rng, steps=SD_WINDOW[1], seed=SEED + 500)
    vals = []
    for t in range(*SD_WINDOW, 5):
        X, H = res.pos_s[..., t], res.vel_s[..., t]
        e = SS.unit(H.mean(1))
        el = np.stack([-e[:, 1], e[:, 0]], -1)
        lat = np.einsum("rnc,rc->rn", X - X.mean(1, keepdims=True), el)
        ok = (N - largest_group(X)) < 2
        vals.append(np.where(ok, np.sqrt((lat ** 2).mean(1)), np.nan))
    return float(np.nanmedian(vals))


def s_gap(N, K):
    return np.maximum(0.0, K + 1 - N / 2) / K


def n_ref(st):
    return max(14, st["K"] + 1)


def phi(N, st, sd14, lam):
    y = 0.8 * sd14 * np.sqrt(N / n_ref(st))  # Y0 / 2
    push = st["w_dog"] * y / np.sqrt(st["d"] ** 2 + y ** 2)
    pull = st["w_att"] * lam * s_gap(N, st["K"])
    return np.where(pull > 0, push / np.maximum(pull, 1e-12), np.inf)


def predicted_Nc(st, sd14, lam):
    f = lambda N: phi(N, st, sd14, lam) - 1
    hi = 2 * st["K"] + 2 - 1e-6
    lo = 4.0
    return float(brentq(f, lo, hi)) if f(lo) < 0 < f(hi) else (lo if f(lo) >= 0 else hi)


def measured_Nc(rows):
    """Interpolated size at which half the runs split (rows sorted by N)."""
    Ns = np.array([r["N"] for r in rows], float)
    fr = np.array([r["split"] for r in rows])
    above = np.flatnonzero(fr >= 0.5)
    if not len(above):
        return None, "no 50% crossing in range"
    i = above[0]
    if i == 0:
        return float(Ns[0]), "at or below the smallest size run"
    return float(Ns[i - 1] + (0.5 - fr[i - 1]) / (fr[i] - fr[i - 1]) * (Ns[i] - Ns[i - 1])), "interpolated"


def cross_share(X, H, K):
    """Share of each sheep's K nearest neighbours lying across the flock's lateral median line (in its own frame).
    X, H: (R, N, 2) positions and headings at one step. Returns (R,)."""
    N = X.shape[1]
    e = SS.unit(H.mean(1))
    el = np.stack([-e[:, 1], e[:, 0]], -1)
    lat = np.einsum("rnc,rc->rn", X - X.mean(1, keepdims=True), el)
    side = lat > np.median(lat, axis=1, keepdims=True)
    d = np.linalg.norm(X[:, None] - X[:, :, None], axis=-1)
    d = np.where(np.eye(N, dtype=bool), np.inf, d)
    knn = np.argsort(d, axis=-1)[..., :K]
    nb_side = np.take_along_axis(np.repeat(side[:, None, :], N, 1), knn, -1)
    return (nb_side != side[..., None]).mean((1, 2))


EARLY = (10, 30)  # steps over which the intact flock's cross share is measured (before most splits start)


def split_scan(st, sizes, rng):
    """Split counts after T steps; from the same runs (no extra random draws), the cross share and lateral width."""
    rows = []
    for N in sizes:
        res = run_flocks(st, N, rng)
        outside = N - largest_group(res.pos_s[..., -1])
        k = int((outside >= 2).sum())
        ts = list(range(0, T + 1, 5))
        s_t = np.stack([cross_share(res.pos_s[..., t], res.vel_s[..., t], st["K"]) for t in ts], 1)  # (R, len(ts))
        intact = np.stack([(N - largest_group(res.pos_s[..., t])) == 0 for t in ts], 1)
        split_t = np.stack([(N - largest_group(res.pos_s[..., t])) >= 2 for t in ts], 1)
        ts_a = np.array(ts)
        early = (ts_a >= EARLY[0]) & (ts_a <= EARLY[1])
        s_early = np.where(intact[:, early], s_t[:, early], np.nan)
        first = np.where(split_t.any(1), split_t.argmax(1), -1)
        s_pre = np.array([s_t[r, max(f - 1, 0)] if f > 0 else np.nan for r, f in enumerate(first)])
        lat_sd = []
        for t in range(*SD_WINDOW, 5):
            X, H = res.pos_s[..., t], res.vel_s[..., t]
            e = SS.unit(H.mean(1))
            el = np.stack([-e[:, 1], e[:, 0]], -1)
            lat = np.einsum("rnc,rc->rn", X - X.mean(1, keepdims=True), el)
            ok = (N - largest_group(X)) < 2
            lat_sd.append(np.where(ok, np.sqrt((lat ** 2).mean(1)), np.nan))
        lat_sd = np.array(lat_sd)
        rows.append(dict(N=int(N), k=k, runs=RUNS, split=k / RUNS, split_ci=wilson(k, RUNS),
                         s_early_median=float(np.nanmedian(s_early)) if np.isfinite(s_early).any() else None,
                         s_pre_split_median=float(np.nanmedian(s_pre)) if np.isfinite(s_pre).any() else None,
                         s_gap=float(s_gap(N, st["K"])),
                         sd_perp_N=float(np.nanmedian(lat_sd)) if np.isfinite(lat_sd).any() else None,
                         first_split_median=float(np.median(ts_a[first[first >= 0]])) if (first >= 0).any() else None))
    return rows


def main():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    sw = json.loads((OUT / "size_sweep.json").read_text())
    cal_rows = sorted([dict(N=r["N"], split=r["split_end"]) for r in sw["boundary"] if r["case"] == "published, dog 7.5 m"], key=lambda r: r["N"])
    Nc_cal, _ = measured_Nc(cal_rows)
    out = dict(setup=dict(base=BASE, settings=SETTINGS, T=T, runs=RUNS, sd_window=SD_WINDOW), settings={})
    sds = {name: lateral_sd(setting(name), rng, N=n_ref(setting(name))) for name in SETTINGS}
    st0 = setting("published, dog 7.5 m (calibration)")
    lam = float(brentq(lambda l: phi(Nc_cal, st0, sds["published, dog 7.5 m (calibration)"], l) - 1, 1e-3, 10))
    print(f"calibration: published measured N_c50 = {Nc_cal:.2f}, sd_perp(14) = {sds['published, dog 7.5 m (calibration)']:.2f} m -> lambda = {lam:.3f}", flush=True)
    out["lambda"], out["Nc_calibration"] = lam, Nc_cal
    for name in SETTINGS:
        st = setting(name)
        pred = predicted_Nc(st, sds[name], lam)
        ceiling = 2 * st["K"] + 2
        lo = max(6, st["K"] + 1, int(np.floor(pred)) - 3)
        hi = min(int(np.ceil(pred)) + 3, ceiling + 2)
        sizes = list(range(lo, hi + 1))
        rows = split_scan(st, sizes, rng)
        meas, how = measured_Nc(rows)
        out["settings"][name] = dict(setting=st, sd_perp_ref=sds[name], N_ref=n_ref(st), predicted_Nc=pred, ceiling=ceiling, measured_Nc=meas, measured_how=how, scan=rows)
        print(f"  {name:36s} sd⊥({n_ref(st)}) {sds[name]:.2f} m | predicted N_c {pred:5.1f} (ceiling {ceiling}) | measured {meas if meas is None else round(meas, 1)} ({how}) | "
              + " ".join(f"{r['N']}:{int(round(r['split'] * RUNS))}" for r in rows), flush=True)
    out["seconds"] = time.time() - tic
    (OUT / "fission_threshold.json").write_text(json.dumps(out, indent=1))
    figure(out)
    print(f"done in {time.time() - tic:.0f} s")


def figure(out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    names = list(out["settings"])
    pred = [out["settings"][n]["predicted_Nc"] for n in names]
    meas = [out["settings"][n]["measured_Nc"] for n in names]
    for i, n in enumerate(names):
        col = COL["gray"] if "calibration" in n else (COL["blue"] if n in ("published, dog 12 m", "w_att 3, dog 7.5 m") else COL["orange"])
        if meas[i] is not None:
            ax.scatter(pred[i], meas[i], s=70 if "calibration" not in n else 110, color=col, edgecolor="white", lw=1.2, zorder=3 if "calibration" not in n else 4)
            near = [j for j in range(i) if meas[j] is not None and abs(pred[j] - pred[i]) < 1.0]
            ax.annotate(n, (pred[i], meas[i]), xytext=(6, -10 + 11 * len(near)), textcoords="offset points", fontsize=7.5, color="#3a3a38")
    lim = [min(pred + [m for m in meas if m]) - 1, max(pred + [m for m in meas if m]) + 1]
    ax.plot(lim, lim, color=COL["gray"], lw=1, ls=":")
    ax.set(xlabel="predicted N_c (Φ = 1)", ylabel="measured N_c (50% of runs split after 300 steps)",
           title="Fission threshold: grey calibration, blue measured before, orange new")
    ax = axs[1]
    palette = ["#8a8a8a", "#2a78d6", "#1baf7a", "#eb6834", "#7b61c4", "#d63a3a", "#1b9aaa", "#b8860b", "#3a3a38"]
    for n, col in zip(names, palette):
        rows = out["settings"][n]["scan"]
        ax.plot([r["N"] - out["settings"][n]["predicted_Nc"] for r in rows], [r["split"] for r in rows], "o-", color=col, lw=1.5, ms=4, label=n)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="N − predicted N_c", ylabel="fraction of runs split after 300 steps", ylim=(-0.03, 1.03), title="Split fraction against distance from the prediction")
    ax.legend(frameon=False, fontsize=7)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fission_threshold.png", dpi=130)


if __name__ == "__main__":
    main()
