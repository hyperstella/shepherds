"""Why does the fission-threshold model predict N_c about 0.6 sheep too low? Four candidate causes, tested.

Context: docs/steer_and_shape.md §11.5-§11.6. The model Phi(N) = 1 uses the lower bound
s_gap(N) = max(0, K + 1 - N/2)/K for the cross-perception share of two equal halves; with one
constant fitted to the published setting it predicts the 50% split size of eight other settings
with bias -0.6 sheep. Fitting lambda to each setting separately shows the bias is concentrated where
N_c lies close to 2K + 2 (s_gap near 0). Candidates:

  H1 horizon: N_c is defined by a split within 300 steps; if fission near the threshold is slow,
     longer runs lower the measured N_c toward the prediction.
  H2 unequal parts: the model assumes equal halves; if flocks split into unequal parts, the smaller
     part keeps K + 1 - n_small cross neighbours, more than s_gap allows near 2K + 2.
  H3 dog geometry: the model uses the nominal standoff d with the dog straight behind; the real dog
     sits at another distance or off-axis.
  H4 quantile: the model's balance may match a split probability other than 50% (existing fits only).

Protocol as in experiments/fission_threshold.py (code parameters, published rule, drive-only Strömbom
dog, r_dog 20, compact aligned start, 30 runs per size), new seeds, runs of 1000 steps, sizes around
each setting's measured N_c. From the same runs: split by step 150, 300, 600 and 1000; group sizes at
the first split; the dog's distance to the centroid and lateral offset before any split.

    .venv/bin/python experiments/fission_bias.py
Writes results/fission_bias.json and results/figures/fission_bias.png.
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

import fission_provenance as FP  # noqa: E402
import shape_stress as SS  # noqa: E402
from shepherd.model import simulate, Params  # noqa: E402
from size_sweep import compact  # noqa: E402

OUT = ROOT / "results"
T_LONG, RUNS, SEED, EVERY = 1000, 30, 97, 5
HORIZONS = (150, 300, 600, 1000)
CAL = FP.CAL
LINK = 6.0


def groups(X):
    """Sizes of the two largest groups linked at LINK m. X (R, N, 2) -> (largest, second)."""
    d = np.linalg.norm(X[:, None] - X[:, :, None], axis=-1)
    reach = d <= LINK
    for _ in range(5):
        reach = (reach.astype(np.int32) @ reach.astype(np.int32)) > 0
    sizes = reach.sum(-1)  # component size of each sheep
    largest = sizes.max(-1)
    second = np.where(sizes < largest[:, None], sizes, 0).max(-1)
    return largest, second


def run(st, N, rng):
    c0 = np.array([SS.FAR, 0.0])
    pos = c0 + compact(rng, RUNS, N)
    hd = SS.unit(np.array([-1.0, 0.0]) + 0.1 * rng.standard_normal((RUNS, N, 2)))
    per = dict(w_att=np.full(RUNS, st["w_att"]), w_ali=np.full(RUNS, st["w_ali"]), w_dog=np.full(RUNS, st["w_dog"]),
               r_dog=np.full(RUNS, 20.0), l_drive=np.full(RUNS, st["d"]), l_sep=np.full(RUNS, 1e6))
    res = simulate(Params(n_sheep=N, n_steps=T_LONG + 1, k_perceive=st["K"]), n_runs=RUNS, seed=SEED + N, per_run=per,
                   init=dict(pos_s=pos, vel_s=hd, pos_d=np.tile(c0 + [st["d"], 0.0], (RUNS, 1)), vel_d=np.tile([-1.0, 0.0], (RUNS, 1))))
    ts = np.arange(0, T_LONG + 1, EVERY)
    big, sec = zip(*[groups(res.pos_s[..., t]) for t in ts])
    big, sec = np.stack(big, 1), np.stack(sec, 1)  # (R, len(ts))
    split = (N - big) >= 2
    first = np.where(split.any(1), ts[split.argmax(1)], np.inf)  # first sampled step with a split
    # group sizes at the first split
    j = split.argmax(1)
    at_big = np.where(np.isfinite(first), big[np.arange(RUNS), j], np.nan)
    at_sec = np.where(np.isfinite(first), sec[np.arange(RUNS), j], np.nan)
    # dog geometry before any split (steps 20 .. min(first, 200))
    dist, lat = [], []
    for t in range(20, 201, 5):
        X, H, D = res.pos_s[..., t], res.vel_s[..., t], res.pos_d[..., t]
        c = X.mean(1)
        e = SS.unit(H.mean(1))
        el = np.stack([-e[:, 1], e[:, 0]], -1)
        ok = t < first
        rel = D - c
        dist.append(np.where(ok, np.linalg.norm(rel, axis=-1), np.nan))
        lat.append(np.where(ok, np.abs((rel * el).sum(-1)), np.nan))
    return dict(first=first, at_big=at_big, at_sec=at_sec, dog_dist=np.nanmedian(dist), dog_lat=np.nanmedian(lat))


def bias_of(preds, meas, names):
    e = np.array([preds[n] - meas[n] for n in names if preds.get(n) is not None and meas.get(n) is not None])
    return float(e.mean()), float(np.sqrt((e ** 2).mean())), int(len(e))


def main():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    ft = json.loads((OUT / "fission_threshold.json").read_text())
    pv = json.loads((OUT / "fission_provenance.json").read_text())
    out = dict(setup=dict(T=T_LONG, horizons=HORIZONS, runs=RUNS, seed=SEED, link_m=LINK, protocol="as experiments/fission_threshold.py, new seeds"),
               settings={})
    # H4: quantiles from the existing fits (no new runs)
    h4 = {}
    for q in (0.1, 0.25, 0.5, 0.75):
        Nq = {n: c["fit"]["Nc"] + c["fit"]["width"] * np.log(q / (1 - q)) for n, c in pv["curves"].items() if n in ft["settings"]}
        cal_pool = pv["calibration_consistency"]["pooled"]["fit"]
        Nq[CAL] = cal_pool["Nc"] + cal_pool["width"] * np.log(q / (1 - q))
        dcal = ft["settings"][CAL]
        phi_cal = FP.make_phi(dcal["setting"], dcal["sd_perp_ref"], dcal["N_ref"], dcal["scan"], "gap", "scaled")
        lam = brentq(lambda l: phi_cal(Nq[CAL], l) - 1, 1e-3, 50)
        preds = {}
        for n, d in ft["settings"].items():
            phi = FP.make_phi(d["setting"], d["sd_perp_ref"], d["N_ref"], d["scan"], "gap", "scaled")
            preds[n] = FP.solve_Nc(phi, lam, 4.0, 2 * d["setting"]["K"] + 2)
        b, r, n_ = bias_of(preds, Nq, [n for n in ft["settings"] if n != CAL])
        h4[str(q)] = dict(lam=float(lam), bias=b, rms=r, n=n_)
        print(f"H4 quantile {q:.2f}: λ {lam:.3f} bias {b:+.2f} RMS {r:.2f}", flush=True)
    out["H4_quantile"] = h4
    # H1-H3: new runs
    for name, d in ft["settings"].items():
        st = d["setting"]
        Nc0 = (pv["calibration_consistency"]["pooled"]["fit"] if name == CAL else pv["curves"][name]["fit"])["Nc"]
        sizes = list(range(max(6, st["K"] + 1, int(np.floor(Nc0)) - 3), int(np.ceil(Nc0)) + 4))
        per_N = {}
        for N in sizes:
            per_N[N] = run(st, N, rng)
        curves = {}
        for T in HORIZONS:
            k = [int((per_N[N]["first"] <= T).sum()) for N in sizes]
            curves[T] = dict(N=sizes, k=k, fit=FP.logistic_fit(sizes, k, [RUNS] * len(sizes)))
        nb = [N for N in sizes if abs(N - Nc0) <= 1.5]
        sec = np.concatenate([per_N[N]["at_sec"][np.isfinite(per_N[N]["at_sec"])] / N for N in nb])
        smaller_part = np.concatenate([np.minimum(per_N[N]["at_sec"], N - per_N[N]["at_big"])[np.isfinite(per_N[N]["at_sec"])] for N in nb])
        first_near = np.concatenate([per_N[N]["first"][np.isfinite(per_N[N]["first"])] for N in nb])
        out["settings"][name] = dict(setting=st, sizes=sizes, curves={str(T): c for T, c in curves.items()},
                                     dog_dist_median=float(np.nanmedian([per_N[N]["dog_dist"] for N in nb])),
                                     dog_lat_median=float(np.nanmedian([per_N[N]["dog_lat"] for N in nb])),
                                     second_group_frac_median=float(np.median(sec)) if len(sec) else None,
                                     second_group_size_median=float(np.median(smaller_part)) if len(smaller_part) else None,
                                     first_split_median=float(np.median(first_near)) if len(first_near) else None,
                                     first_split_p90=float(np.percentile(first_near, 90)) if len(first_near) else None)
        s = out["settings"][name]
        print(f"  {name:36s} N_c by T=150/300/600/1000: " + " / ".join(f"{curves[T]['fit']['Nc']:.2f}" for T in HORIZONS)
              + f" | dog dist {s['dog_dist_median']:.2f} (d {st['d']}) lat {s['dog_lat_median']:.2f} | second group at first split {s['second_group_size_median']} "
              f"({s['second_group_frac_median']}) | first split median {s['first_split_median']} p90 {s['first_split_p90']}  ({time.time() - tic:.0f} s)", flush=True)
    # predictions (as in fission_provenance V5) against N_c at each horizon, recalibrating on the calibration setting's own N_c(T)
    for T in HORIZONS:
        meas = {n: out["settings"][n]["curves"][str(T)]["fit"]["Nc"] for n in out["settings"]}
        dcal = ft["settings"][CAL]
        phi_cal = FP.make_phi(dcal["setting"], dcal["sd_perp_ref"], dcal["N_ref"], dcal["scan"], "gap", "scaled")
        lam = brentq(lambda l: phi_cal(meas[CAL], l) - 1, 1e-3, 50)
        preds = {}
        for n, d in ft["settings"].items():
            phi = FP.make_phi(d["setting"], d["sd_perp_ref"], d["N_ref"], d["scan"], "gap", "scaled")
            preds[n] = FP.solve_Nc(phi, lam, 4.0, 2 * d["setting"]["K"] + 2)
        b, r, n_ = bias_of(preds, meas, [n for n in ft["settings"] if n != CAL])
        out.setdefault("H1_horizon", {})[str(T)] = dict(lam=float(lam), bias=b, rms=r, n=n_, predictions=preds, measured=meas)
        print(f"H1 horizon {T}: λ {lam:.3f} bias {b:+.2f} RMS {r:.2f}", flush=True)
    # H3: predictions with the measured dog distance in place of d (recalibrated)
    def phi_dog(n, lam, N):
        d = ft["settings"][n]
        st = dict(d["setting"], d=out["settings"][n]["dog_dist_median"])
        return FP.make_phi(st, d["sd_perp_ref"], d["N_ref"], d["scan"], "gap", "scaled")(N, lam)
    meas300 = {n: out["settings"][n]["curves"]["300"]["fit"]["Nc"] for n in out["settings"]}
    lam = brentq(lambda l: phi_dog(CAL, l, meas300[CAL]) - 1, 1e-3, 50)
    preds = {n: FP.solve_Nc(lambda N, l, n=n: phi_dog(n, l, N), lam, 4.0, 2 * ft["settings"][n]["setting"]["K"] + 2) for n in ft["settings"]}
    b, r, n_ = bias_of(preds, meas300, [n for n in ft["settings"] if n != CAL])
    out["H3_dog_distance"] = dict(lam=float(lam), bias=b, rms=r, n=n_, predictions=preds)
    print(f"H3 measured dog distance (T=300): λ {lam:.3f} bias {b:+.2f} RMS {r:.2f}", flush=True)
    # H2: unequal parts, using the measured smaller-part size at the first split
    def phi_uneq(n, lam, N):
        d = ft["settings"][n]
        st = d["setting"]
        f_small = out["settings"][n]["second_group_frac_median"] or 0.5
        f_small = min(f_small, 0.5)
        n_small, n_large = f_small * N, (1 - f_small) * N
        s = (max(0.0, st["K"] + 1 - n_small) + max(0.0, st["K"] + 1 - n_large)) / (2 * st["K"])
        y = 0.8 * d["sd_perp_ref"] * np.sqrt(N / d["N_ref"])
        push = st["w_dog"] * y / np.sqrt(st["d"] ** 2 + y ** 2)
        pull = st["w_att"] * lam * s
        return push / pull if pull > 0 else np.inf
    lam = brentq(lambda l: phi_uneq(CAL, l, meas300[CAL]) - 1, 1e-3, 50)
    preds = {n: FP.solve_Nc(lambda N, l, n=n: phi_uneq(n, l, N), lam, 4.0, 2 * ft["settings"][n]["setting"]["K"] + 6) for n in ft["settings"]}
    b, r, n_ = bias_of(preds, meas300, [n for n in ft["settings"] if n != CAL])
    out["H2_unequal_parts"] = dict(lam=float(lam), bias=b, rms=r, n=n_, predictions=preds)
    print(f"H2 unequal parts (T=300): λ {lam:.3f} bias {b:+.2f} RMS {r:.2f}", flush=True)
    # combined: 1000-step horizon with the lateral width measured at every N (instead of sqrt(N) scaling)
    meas1000 = {n: out["settings"][n]["curves"]["1000"]["fit"]["Nc"] for n in out["settings"]}
    dcal = ft["settings"][CAL]
    phi_cal = FP.make_phi(dcal["setting"], dcal["sd_perp_ref"], dcal["N_ref"], dcal["scan"], "gap", "measured")
    lam = brentq(lambda l: phi_cal(meas1000[CAL], l) - 1, 1e-3, 50)
    preds = {n: FP.solve_Nc(FP.make_phi(d["setting"], d["sd_perp_ref"], d["N_ref"], d["scan"], "gap", "measured"), lam, 4.0, 2 * d["setting"]["K"] + 2)
             for n, d in ft["settings"].items()}
    b, r, n_ = bias_of(preds, meas1000, [n for n in ft["settings"] if n != CAL])
    out["H1_1000_measured_width"] = dict(lam=float(lam), bias=b, rms=r, n=n_, predictions=preds, measured=meas1000)
    print(f"T=1000 with measured widths: λ {lam:.3f} bias {b:+.2f} RMS {r:.2f}", flush=True)
    out["seconds"] = time.time() - tic
    (OUT / "fission_bias.json").write_text(json.dumps(out, indent=1, default=float))
    figure(out)
    print(f"done in {time.time() - tic:.0f} s")


def figure(out):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    names = list(out["settings"])
    for n in names:
        ys = [out["settings"][n]["curves"][str(T)]["fit"]["Nc"] for T in HORIZONS]
        ax.plot(HORIZONS, ys, "o-", lw=1.5, ms=4, label=n.split(" (")[0])
    ax.set(xscale="log", xlabel="horizon: split by step T", ylabel="measured N_c (logistic 50% point)", title="H1: does N_c fall with a longer horizon?")
    ax.minorticks_off()
    ax.set_xticks(HORIZONS, [str(T) for T in HORIZONS])
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax = axs[1]
    labels = [f"H1 T={T}" for T in HORIZONS] + ["T=1000 + widths measured", "H2 unequal parts", "H3 dog distance"] + [f"H4 q={q}" for q in out["H4_quantile"]]
    vals = [out["H1_horizon"][str(T)]["bias"] for T in HORIZONS] + [out["H1_1000_measured_width"]["bias"], out["H2_unequal_parts"]["bias"], out["H3_dog_distance"]["bias"]] + [v["bias"] for v in out["H4_quantile"].values()]
    rms = [out["H1_horizon"][str(T)]["rms"] for T in HORIZONS] + [out["H1_1000_measured_width"]["rms"], out["H2_unequal_parts"]["rms"], out["H3_dog_distance"]["rms"]] + [v["rms"] for v in out["H4_quantile"].values()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vals, width=0.38, color=FP.COL["orange"], label="bias (mean predicted − measured)")
    ax.bar(x + 0.2, rms, width=0.38, color=FP.COL["gray"], label="RMS error")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels, rotation=50, fontsize=8)
    ax.set(ylabel="sheep (8 validation settings)", title="Bias under each candidate fix\n(each recalibrated on the calibration setting only)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    for n in names:
        s = out["settings"][n]
        ax.scatter(s["setting"]["d"], s["dog_dist_median"], s=40, color=FP.COL["blue"])
    ax.annotate("seven settings at d = 7.5 m", (7.5, 8.0), xytext=(8, -14), textcoords="offset points", fontsize=7.5)
    ax.annotate("dog 5 m", (5.0, out["settings"]["dog 5 m"]["dog_dist_median"]), xytext=(6, -4), textcoords="offset points", fontsize=7.5)
    ax.annotate("dog 12 m", (12.0, out["settings"]["published, dog 12 m"]["dog_dist_median"]), xytext=(-50, 4), textcoords="offset points", fontsize=7.5)
    ax.plot([4, 13], [4, 13], color=FP.COL["gray"], lw=1, ls=":")
    ax.set(xlabel="nominal standoff d (m)", ylabel="measured dog–centroid distance before a split (m)", title="H3: where the dog actually is")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fission_bias.png", dpi=125)


if __name__ == "__main__":
    main()
