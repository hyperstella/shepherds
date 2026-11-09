"""Provenance, split curves and checks for the fission-threshold analysis (docs/steer_and_shape.md §11.4-§11.6).

Runs no simulations. It reads:
  results/size_sweep.json         experiments/size_sweep.py: the calibration data (published setting,
                                  "boundary" scan) and the first size sweep ("trajectories")
  results/fission_threshold.json  experiments/fission_threshold.py: the validation scans, with the cross
                                  share and lateral width measured in the same runs

and produces:
  1. split fraction against N for every setting, with 95% Wilson intervals, a logistic fit by maximum
     likelihood and N_c = its 50% point with a profile-likelihood 95% interval (also the linear
     interpolation used before);
  2. a check of the cross-neighbour approximation: the measured cross share of intact driven flocks
     against the lower bound s_gap(N) = max(0, K + 1 - N/2)/K used in the threshold model, and whether
     the lower bound explains why the predictions run low. Variants, each recalibrated on the
     calibration setting only:
       V0  s_gap, sd_perp scaled as sqrt(N), lambda from the interpolated calibration N_c (as published)
       V1  as V0, lambda from the logistic calibration N_c
       V2  measured cross share of the intact flock (steps 10-30) instead of s_gap
       V3  measured cross share just before the first split
       V4  s_gap, with sd_perp measured at every N instead of scaled
     scored on the eight validation settings: bias and RMS error against the logistic N_c.

    .venv/bin/python experiments/fission_provenance.py
Writes results/fission_provenance.json and results/figures/fission_curves.png, fission_check.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "results"
CAL = "published, dog 7.5 m (calibration)"
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------ logistic fit with a profile-likelihood interval

def nll(c, w, N, k, n):
    p = 1 / (1 + np.exp(-(N - c) / w))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()


def profile(c, N, k, n):
    r = minimize_scalar(lambda lw: nll(c, np.exp(lw), N, k, n), bounds=(np.log(0.05), np.log(10)), method="bounded")
    return r.fun, float(np.exp(r.x))


def logistic_fit(N, k, n):
    """N_c (50% point) by maximum likelihood, with a 95% profile-likelihood interval; the width w is profiled out
    (bounded below at 0.05 sheep, since perfectly separated data push it to 0)."""
    N, k, n = (np.asarray(x, float) for x in (N, k, n))
    grid = np.arange(N.min() - 1.5, N.max() + 1.5, 0.01)
    prof = np.array([profile(c, N, k, n)[0] for c in grid])
    i = int(prof.argmin())
    inside = grid[2 * (prof - prof[i]) <= 3.84]
    return dict(Nc=float(grid[i]), ci=[float(inside.min()), float(inside.max())], width=profile(grid[i], N, k, n)[1])


def interpolated(N, frac):
    N, frac = np.asarray(N, float), np.asarray(frac)
    above = np.flatnonzero(frac >= 0.5)
    if not len(above) or above[0] == 0:
        return None
    i = above[0]
    return float(N[i - 1] + (0.5 - frac[i - 1]) / (frac[i] - frac[i - 1]) * (N[i] - N[i - 1]))


# ------------------------------------------------------------------ threshold model variants

def s_gap(N, K):
    return np.maximum(0.0, K + 1 - N / 2) / K


def interp_or_none(N, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys)
    if ok.sum() < 2:
        return None
    return float(np.interp(N, xs[ok], ys[ok]))


def make_phi(st, sd_ref, n_ref, scan, s_mode, sd_mode):
    Ns = [r["N"] for r in scan]

    def s_of(N):
        if s_mode == "gap":
            return float(s_gap(N, st["K"]))
        key = {"early": "s_early_median", "pre": "s_pre_split_median"}[s_mode]
        v = interp_or_none(N, Ns, [np.nan if r[key] is None else r[key] for r in scan])
        return v if v is not None else float(s_gap(N, st["K"]))

    def sd_of(N):
        if sd_mode == "scaled":
            return sd_ref * np.sqrt(N / n_ref)
        v = interp_or_none(N, Ns, [np.nan if r["sd_perp_N"] is None else r["sd_perp_N"] for r in scan])
        return v if v is not None else sd_ref * np.sqrt(N / n_ref)

    def phi(N, lam):
        y = 0.8 * sd_of(N)
        push = st["w_dog"] * y / np.sqrt(st["d"] ** 2 + y ** 2)
        pull = st["w_att"] * lam * s_of(N)
        return push / pull if pull > 0 else np.inf

    return phi


def solve_Nc(phi, lam, lo, hi):
    f = lambda N: phi(N, lam) - 1
    grid = np.linspace(lo, hi, 400)
    vals = np.array([f(x) for x in grid])
    cross = np.flatnonzero((vals[:-1] < 0) & (vals[1:] >= 0))
    if not len(cross):
        return None
    j = cross[0]
    return float(brentq(f, grid[j], grid[j + 1])) if np.isfinite(vals[j + 1]) else float(grid[j + 1])


VARIANTS = {"V0 s_gap, scaled width, interpolated calibration (as published)": ("gap", "scaled", "interp"),
            "V1 s_gap, scaled width, logistic calibration": ("gap", "scaled", "logistic"),
            "V2 measured cross share (intact, steps 10-30)": ("early", "scaled", "logistic"),
            "V3 measured cross share just before the split": ("pre", "scaled", "logistic"),
            "V4 s_gap, width measured at every N": ("gap", "measured", "logistic"),
            "V5 s_gap, scaled width, calibrated on all three batches pooled": ("gap", "scaled", "pooled")}


def main():
    sw = json.loads((OUT / "size_sweep.json").read_text())
    ft = json.loads((OUT / "fission_threshold.json").read_text())
    out = dict(sources=dict(calibration="results/size_sweep.json, key boundary, case 'published, dog 7.5 m' (experiments/size_sweep.py)",
                            first_sweep="results/size_sweep.json, key trajectories (experiments/size_sweep.py)",
                            validation="results/fission_threshold.json, key settings (experiments/fission_threshold.py)"),
               curves={}, variants={})
    # 1. curves
    cal_rows = sorted([r for r in sw["boundary"] if r["case"] == "published, dog 7.5 m"], key=lambda r: r["N"])
    Nn = [r["N"] for r in cal_rows]
    kk = [int(round(r["split_end"] * 30)) for r in cal_rows]
    cal_fit = logistic_fit(Nn, kk, [30] * len(Nn))
    out["curves"]["calibration data (size_sweep boundary)"] = dict(N=Nn, k=kk, n=30, fit=cal_fit, interpolated=interpolated(Nn, np.array(kk) / 30))
    sweep_rows = sorted([r for r in sw["trajectories"] if r["policy"] == "drive only"], key=lambda r: r["N"])
    out["curves"]["first sweep (size_sweep trajectories, drive only)"] = dict(N=[r["N"] for r in sweep_rows], k=[int(round(r["split_end"] * 30)) for r in sweep_rows], n=30,
                                                                            fit=logistic_fit([r["N"] for r in sweep_rows], [int(round(r["split_end"] * 30)) for r in sweep_rows], [30] * len(sweep_rows)))
    for name, d in ft["settings"].items():
        N = [r["N"] for r in d["scan"]]
        k = [r["k"] for r in d["scan"]]
        out["curves"][name] = dict(N=N, k=k, n=30, fit=logistic_fit(N, k, [30] * len(N)), interpolated=interpolated(N, np.array(k) / 30),
                                   predicted_as_published=d["predicted_Nc"])
    for name, c in out["curves"].items():
        f = c["fit"]
        print(f"{name:52s} N_c {f['Nc']:.2f} [{f['ci'][0]:.2f}, {f['ci'][1]:.2f}] width {f['width']:.2f}"
              + (f" | interpolated {c['interpolated']:.2f}" if c.get("interpolated") else "") + (f" | predicted (as published) {c['predicted_as_published']:.2f}" if "predicted_as_published" in c else ""))
    # 1b. between-batch consistency of the calibration setting: three independent batches of 30 runs
    from scipy.stats import fisher_exact
    batches = {"calibration (size_sweep boundary)": out["curves"]["calibration data (size_sweep boundary)"],
               "first sweep (size_sweep trajectories)": out["curves"]["first sweep (size_sweep trajectories, drive only)"],
               "rerun (fission_threshold)": out["curves"][CAL]}
    cells = [(b, N, k) for b, c in batches.items() for N, k in zip(c["N"], c["k"])]
    sizes = sorted({N for _, N, _ in cells})
    pooled_N = sizes
    pooled_k = [sum(k for _, n_, k in cells if n_ == N) for N in sizes]
    pooled_n = [30 * sum(1 for _, n_, _ in cells if n_ == N) for N in sizes]
    pooled = logistic_fit(pooled_N, pooled_k, pooled_n)
    # goodness of fit of one curve to every batch cell (deviance against the saturated model)
    p_c = lambda N: 1 / (1 + np.exp(-(N - pooled["Nc"]) / pooled["width"]))
    dev = 0.0
    for _, N, k in cells:
        pc = np.clip(p_c(N), 1e-12, 1 - 1e-12)
        dev += 2 * ((k * np.log(k / (30 * pc)) if k > 0 else 0) + ((30 - k) * np.log((30 - k) / (30 * (1 - pc))) if k < 30 else 0))
    df = len(cells) - 2
    disp = max(1.0, dev / df)
    # interval widened by the dispersion (quasi-likelihood: deviance threshold 3.84 * dispersion)
    grid = np.arange(min(pooled_N) - 1.5, max(pooled_N) + 1.5, 0.01)
    prof = np.array([profile(c, np.array(pooled_N, float), np.array(pooled_k, float), np.array(pooled_n, float))[0] for c in grid])
    inside = grid[2 * (prof - prof.min()) <= 3.84 * disp]
    pooled["ci_dispersion"] = [float(inside.min()), float(inside.max())]
    k17 = {b: next((k for N, k in zip(c["N"], c["k"]) if N == 17), None) for b, c in batches.items()}
    fisher_p = float(fisher_exact([[k17["calibration (size_sweep boundary)"], 30 - k17["calibration (size_sweep boundary)"]],
                                   [k17["rerun (fission_threshold)"], 30 - k17["rerun (fission_threshold)"]]])[1])
    out["calibration_consistency"] = dict(batches={b: dict(N=c["N"], k=c["k"], Nc=c["fit"]["Nc"], ci=c["fit"]["ci"]) for b, c in batches.items()},
                                          pooled=dict(N=pooled_N, k=pooled_k, n=pooled_n, fit=pooled), deviance=float(dev), df=int(df), dispersion=float(disp),
                                          N17=k17, fisher_p_N17=fisher_p)
    print(f"\ncalibration setting, three batches: N=17 splits {k17}; Fisher exact p (batch 1 vs rerun) = {fisher_p:.4f}")
    print(f"  pooled fit N_c {pooled['Nc']:.2f} [{pooled['ci'][0]:.2f}, {pooled['ci'][1]:.2f}] (binomial); deviance {dev:.1f} on {df} df -> dispersion {disp:.2f}; "
          f"interval allowing for it [{pooled['ci_dispersion'][0]:.2f}, {pooled['ci_dispersion'][1]:.2f}]")
    # 2. cross-share check
    print("\ncross share at the sizes scanned: measured (intact, steps 10-30) / just before the split / lower bound s_gap")
    check = {}
    for name, d in ft["settings"].items():
        rows = []
        for r in d["scan"]:
            rows.append(dict(N=r["N"], s_early=r["s_early_median"], s_pre=r["s_pre_split_median"], s_gap=r["s_gap"], sd_N=r["sd_perp_N"],
                             sd_scaled=d["sd_perp_ref"] * np.sqrt(r["N"] / d["N_ref"])))
        check[name] = rows
        near = min(rows, key=lambda r: abs(r["N"] - out["curves"][name]["fit"]["Nc"]))
        print(f"  {name:36s} at N {near['N']}: {near['s_early']:.3f} / {near['s_pre'] if near['s_pre'] is None else round(near['s_pre'], 3)} / {near['s_gap']:.3f} "
              f"| width measured {near['sd_N']:.2f} vs scaled {near['sd_scaled']:.2f}")
    out["cross_share_check"] = check
    # 3. variants
    print("\nvariant: lambda | predicted N_c per validation setting | bias, RMS error vs logistic N_c (8 validation settings)")
    cal_targets = {"interp": out["curves"]["calibration data (size_sweep boundary)"]["interpolated"], "logistic": cal_fit["Nc"], "pooled": pooled["Nc"]}
    for vname, (s_mode, sd_mode, cal_mode) in VARIANTS.items():
        dcal = ft["settings"][CAL]
        phi_cal = make_phi(dcal["setting"], dcal["sd_perp_ref"], dcal["N_ref"], dcal["scan"], s_mode, sd_mode)
        target = cal_targets[cal_mode]
        lam = float(brentq(lambda l: phi_cal(target, l) - 1, 1e-3, 20))
        preds, errs = {}, []
        for name, d in ft["settings"].items():
            phi = make_phi(d["setting"], d["sd_perp_ref"], d["N_ref"], d["scan"], s_mode, sd_mode)
            pred = solve_Nc(phi, lam, 4.0, 2 * d["setting"]["K"] + 2)
            meas = out["curves"][name]["fit"]["Nc"]
            preds[name] = dict(predicted=pred, measured=meas, ci=out["curves"][name]["fit"]["ci"], error=None if pred is None else pred - meas)
            if name != CAL and pred is not None:
                errs.append(pred - meas)
        errs = np.array(errs)
        out["variants"][vname] = dict(lam=lam, calibration_target=target, predictions=preds, bias=float(errs.mean()), rms=float(np.sqrt((errs ** 2).mean())),
                                      max_abs=float(np.abs(errs).max()), n=int(len(errs)))
        print(f"  {vname:62s} λ {lam:.3f} | bias {errs.mean():+.2f} RMS {np.sqrt((errs ** 2).mean()):.2f} max|e| {np.abs(errs).max():.2f}")
    (OUT / "fission_provenance.json").write_text(json.dumps(out, indent=1))
    figures(out, ft)


def figures(out, ft):
    names = list(ft["settings"])
    fig, axs = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
    xx_all = np.linspace(4, 34, 600)
    for ax, name in zip(axs.ravel(), names):
        c = out["curves"][name]
        N, k = np.array(c["N"]), np.array(c["k"])
        lo, hi = zip(*[wilson(x, 30) for x in k])
        ax.errorbar(N, k / 30, yerr=[k / 30 - np.array(lo), np.array(hi) - k / 30], fmt="o", color=COL["blue"], capsize=3, ms=5,
                    label="simulated: fraction split (30 runs, 95% Wilson)")
        f = c["fit"]
        xx = xx_all[(xx_all >= N.min() - 1) & (xx_all <= N.max() + 1)]
        ax.plot(xx, 1 / (1 + np.exp(-(xx - f["Nc"]) / f["width"])), color=COL["blue"], lw=1.2, label="logistic fit (ML)")
        ax.axvspan(*f["ci"], color=COL["blue"], alpha=0.12, lw=0, label=f"measured N_c {f['Nc']:.1f} [{f['ci'][0]:.1f}, {f['ci'][1]:.1f}]")
        ax.axvline(c["predicted_as_published"], color=COL["orange"], lw=1.8, ls="--", label=f"predicted N_c {c['predicted_as_published']:.1f} (analytic, V0)")
        if name == CAL:
            cc = out["curves"]["calibration data (size_sweep boundary)"]
            Nc_, kc = np.array(cc["N"]), np.array(cc["k"])
            ax.plot(Nc_, kc / 30, "s", color=COL["gray"], ms=6, mfc="none", label="calibration runs (size_sweep, other seeds)")
            ax.set_title("CALIBRATION: published weights, dog 7.5 m\n(λ fitted to the grey squares; blue = rerun)", fontsize=9, color=COL["red"])
        else:
            ax.set_title(f"validation: {name}", fontsize=9)
        ax.axhline(0.5, color=COL["gray"], lw=0.6, ls=":")
        ax.set(xlabel="sheep in the flock N", ylim=(-0.04, 1.04))
        ax.legend(frameon=False, fontsize=6.3, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("fraction of runs split after 300 steps")
    fig.suptitle("Fission threshold: simulated split fractions (Jadhav model, code parameters, published rule, drive-only dog, r_dog 20, compact aligned start, "
                 "300 steps, 30 runs per size;\nsplit = ≥ 2 sheep outside the largest group linked at 6 m at step 300) against the analytic prediction (orange)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "figures" / "fission_curves.png", dpi=120)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    for name, col in zip((CAL, "K = 7", "K = 15", "w_att 3, dog 7.5 m"), ("gray", "orange", "purple", "green")):
        rows = out["cross_share_check"][name]
        N = [r["N"] for r in rows]
        ax.plot(N, [r["s_early"] for r in rows], "o-", color=COL[col], lw=2, label=f"{name.split(' (')[0]}: measured, intact flock")
        ax.plot(N, [np.nan if r["s_pre"] is None else r["s_pre"] for r in rows], "^:", color=COL[col], lw=1, ms=4)
        ax.plot(N, [r["s_gap"] for r in rows], "--", color=COL[col], lw=1.2)
    ax.set(xlabel="sheep in the flock N", ylabel="share of perceived neighbours across the midline",
           title="Cross share: measured in intact flocks (solid),\njust before a split (▲), and the bound s_gap (dashed)")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[1]
    for name in names:
        rows = out["cross_share_check"][name]
        ax.plot([r["N"] for r in rows], [r["sd_N"] / r["sd_scaled"] if r["sd_N"] else np.nan for r in rows], "o-", lw=1, ms=4, label=name.split(" (")[0])
    ax.axhline(1, color="k", lw=0.8)
    ax.set(xlabel="sheep in the flock N", ylabel="measured σ⊥(N) / σ⊥(N_ref)·√(N/N_ref)", title="Is the lateral width ∝ √N?\n(unsplit runs, steps 60–120)")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax = axs[2]
    for (vname, v), col, mk in zip(out["variants"].items(), ("orange", "red", "green", "purple", "blue", "gray"), ("o", "s", "^", "v", "D", "P")):
        P = [v["predictions"][n] for n in names if n != CAL and v["predictions"][n]["predicted"] is not None]
        ax.errorbar([p["predicted"] for p in P], [p["measured"] for p in P], yerr=[[p["measured"] - p["ci"][0] for p in P], [p["ci"][1] - p["measured"] for p in P]],
                    fmt=mk, color=COL[col], ms=5, capsize=2, alpha=0.85, label=f"{vname.split(' ')[0]}: bias {v['bias']:+.2f}, RMS {v['rms']:.2f}")
    lim = [11, 26]
    ax.plot(lim, lim, color=COL["gray"], lw=1, ls=":")
    ax.set(xlabel="predicted N_c", ylabel="measured N_c (logistic, 95% profile CI)", xlim=lim, ylim=lim,
           title="8 validation settings; each variant\nrecalibrated on the calibration setting only")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fission_check.png", dpi=120)


if __name__ == "__main__":
    main()
