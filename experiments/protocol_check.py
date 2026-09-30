"""Phase 1: a fair protocol, and a check of three preliminary findings.

F1  The published model's front-to-back hierarchy comes from the start-up
    transient, not from steady driving.
F2  The paper's r = 0.87 is a correlation of bin means; the association
    for individual sheep is much weaker (but real).
F3  Leader-follower lags in the data are sub-second, so resampling to the
    model's 1 s step loses the effect.

    .venv/bin/python experiments/protocol_check.py [--runs 300] [--surrogates 500]
Writes results/protocol_check.json and results/figures/protocol_check.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, protocol as PR  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
MIN_DUR = 8.0  # s; shorter drives are too short for a 2 s lag window
C_DATA, C_MODEL, C_ALT, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def rho_of(recs):
    return PR.hierarchy_stats(recs)["spearman"] if recs else np.nan


def summarize(recs, boot_key="session", n_boot=1000, seed=0):
    st = PR.hierarchy_stats(recs)
    st["spearman_ci95"] = PR.cluster_bootstrap(recs, rho_of, n_boot=n_boot, seed=seed, key=boot_key)["ci95"]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--surrogates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    drives = data.load_drives(data.load_sessions())
    res = {}

    # --- data: legacy pipeline (paper) and fair protocol --------------------
    res["data_legacy"] = summarize(PR.hierarchy(PR.segments_legacy(drives), 2.0, 0.5, True))
    segs = PR.segments_from_drives(drives, 0.1)
    recs = PR.hierarchy(segs, 2.0, 0.5, True, MIN_DUR)
    res["data_fair"] = summarize(recs)
    lags = np.concatenate([r["lags_s"] for r in recs])

    sur = PR.timeshift_surrogate(segs, n_surr=args.surrogates, seed=args.seed, min_duration_s=MIN_DUR)
    obs = res["data_fair"]["spearman"]
    res["data_fair_surrogate"] = {
        "n": args.surrogates,
        "spearman_null_mean": float(np.nanmean(sur["spearman"])),
        "spearman_null_95": float(np.nanpercentile(sur["spearman"], 95)),
        "p_value": float((1 + np.sum(sur["spearman"] >= obs)) / (1 + np.isfinite(sur["spearman"]).sum())),
        "edges_null_mean": float(np.mean(sur["edges_per_segment"])),
        "edges_observed": res["data_fair"]["edges_per_segment"],
    }

    # --- sensitivity grid on the data (F3 + robustness) ---------------------
    grid = []
    for dt in (0.1, 0.2, 0.5, 1.0):
        sg = PR.segments_from_drives(drives, dt)
        for maxlag in (1.0, 2.0, 4.0):
            for cmin in (0.3, 0.5, 0.7, 0.9):
                st = PR.hierarchy_stats(PR.hierarchy(sg, maxlag, cmin, True, MIN_DUR, with_dog=False))
                grid.append({"dt": dt, "maxlag_s": maxlag, "cmin": cmin, "spearman": st["spearman"],
                             "binmean_pearson": st["binmean_pearson"], "edges_per_segment": st["edges_per_segment"]})
    res["data_sensitivity"] = grid

    # --- published model: start-up vs steady driving (F1) ------------------
    sim = simulate(Params(), n_runs=args.runs, seed=args.seed)
    windows = {"whole_run": (1, None), "startup_1_100": (1, 100), "steady_100_370": (100, None)}
    model = {}
    for name, (t0, t1) in windows.items():
        leg = PR.hierarchy(PR.segments_from_result(sim, 1.0, t0=t0, t1=t1, legacy=True), 20.0, 0.5, signed=False)
        fair = PR.hierarchy(PR.segments_from_result(sim, 1.0, t0=t0, t1=t1), 2.0, 0.5, signed=True)
        model[name] = {"legacy_absCCF_20s": summarize(leg, n_boot=500), "fair_signed_2s": summarize(fair, n_boot=500)}
    res["model_M0"] = model

    (OUT / "protocol_check.json").write_text(json.dumps(res, indent=2, default=float))

    # --- figure ------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].hist(lags, bins=np.arange(0.05, 2.1, 0.1), color=C_DATA)
    axs[0].axvline(np.median(lags), color=C_MUTED, ls="--")
    axs[0].set_xlabel("Leader→follower lag (s)"), axs[0].set_ylabel("Edges")
    axs[0].set_title(f"F3  Real lags are short (median {np.median(lags):.1f} s)", fontsize=10)

    dts, mls = (0.1, 0.2, 0.5, 1.0), (1.0, 2.0, 4.0)
    H = np.array([[next(g["spearman"] for g in grid if g["dt"] == dt and g["maxlag_s"] == ml and g["cmin"] == 0.5)
                   for dt in dts] for ml in mls])
    im = axs[1].imshow(H, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    axs[1].set_xticks(range(4), [f"{d:g}" for d in dts]), axs[1].set_yticks(range(3), [f"{m:g}" for m in mls])
    axs[1].set_xlabel("Sampling step dt (s)"), axs[1].set_ylabel("Max lag (s)")
    for i in range(3):
        for j in range(4):
            axs[1].text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=9)
    axs[1].set_title("Sheep-level ρ(indegree, d_i) in data, c_min 0.5", fontsize=10)
    fig.colorbar(im, ax=axs[1], fraction=0.04)

    labels, vals, cis, cols = [], [], [], []
    for lab, st, col in [
        ("Data (fair)", res["data_fair"], C_DATA),
        ("M0 start-up\n(legacy)", model["startup_1_100"]["legacy_absCCF_20s"], C_MODEL),
        ("M0 steady\n(legacy)", model["steady_100_370"]["legacy_absCCF_20s"], C_MODEL),
        ("M0 steady\n(fair)", model["steady_100_370"]["fair_signed_2s"], C_ALT),
    ]:
        labels.append(lab), vals.append(st["spearman"]), cis.append(st["spearman_ci95"]), cols.append(col)
    x = np.arange(len(vals))
    axs[2].bar(x, vals, color=cols, width=0.6)
    axs[2].errorbar(x, vals, yerr=np.abs(np.array(cis).T - np.array(vals)), fmt="none", ecolor=C_MUTED, capsize=3)
    axs[2].axhline(0, color=C_MUTED, lw=0.8)
    axs[2].set_xticks(x, labels, fontsize=8), axs[2].set_ylabel("Sheep-level Spearman ρ")
    axs[2].set_title("F1  Model hierarchy: start-up vs steady driving", fontsize=10)

    axs[3].hist(sur["spearman"], bins=30, color="#b8b7b1")
    axs[3].axvline(obs, color=C_DATA, lw=2, label=f"observed ρ = {obs:.2f}")
    axs[3].set_xlabel("Sheep-level ρ under time-shift surrogates"), axs[3].legend(fontsize=8)
    axs[3].set_title(f"Data vs surrogate null (p = {res['data_fair_surrogate']['p_value']:.3f})", fontsize=10)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "protocol_check.png", dpi=110)

    brief = {
        "data_legacy_binmean_r": res["data_legacy"]["binmean_pearson"],
        "data_legacy_sheep_spearman": res["data_legacy"]["spearman"],
        "data_fair_sheep_spearman": [obs, res["data_fair"]["spearman_ci95"]],
        "data_surrogate_p": res["data_fair_surrogate"]["p_value"],
        "data_median_lag_s": res["data_fair"]["median_lag_s"],
        "data_rho_dt1_lag2": next(g["spearman"] for g in grid if g["dt"] == 1.0 and g["maxlag_s"] == 2.0 and g["cmin"] == 0.5),
        **{f"M0_{w}_{a}": [model[w][a]["spearman"], model[w][a]["spearman_ci95"], model[w][a]["binmean_pearson"]]
           for w in windows for a in ("legacy_absCCF_20s", "fair_signed_2s")},
    }
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
