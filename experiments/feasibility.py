"""How many dogs does a real flock need? Continuification feasibility with measured kernels.

    .venv/bin/python experiments/feasibility.py
Writes results/feasibility.json and results/figures/feasibility.png.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, feasibility as FE  # noqa: E402
from shepherd.kernels import HatBasis  # noqa: E402

OUT = ROOT / "results"
GRID = FE.Grid(L=40.0, n=256)
RANGES = (2.0, 3.0, 5.0, 8.0)  # dog kernel range L (m): not identifiable from driving data
SIZES = (14, 30, 50, 100, 200)
COHESION = {"none (their assumption)": 0.0, "measured": 1.0, "measured x4 (shrinkage-corrected)": 4.0}


def goal_radius(n):
    """A goal as tight as the real flock: 95% of sheep within r, scaled from 14 sheep (r = 2.5 m)."""
    return 2.5 * np.sqrt(n / 14)


G_MAX = 3.6  # m: 95% of observed sheep-sheep distances; the attraction kernel is not extrapolated beyond


def kernels_for(drives, L, k):
    a, cg, r2 = FE.fit_exponential(drives, L, cohesion=k > 0)
    pair = HatBasis(FE.PAIR_KNOTS_FO)
    f = lambda r, a=a: a * np.exp(-r / L)
    g = (lambda r, cg=cg, k=k: np.where(r <= G_MAX, k * pair.curve(cg, r), 0.0)) if k else None
    return f, g, a, r2


def main():
    drives = data.load_drives(data.load_sessions())
    fo = FE.fit_first_order(drives, cohesion=True)
    D_main, D_hi = fo.D[1.0], fo.D[4.0]
    out = {"D_by_window_s": fo.D, "g_cutoff_m": G_MAX, "rows": [], "micro_checks": []}
    for L in RANGES:
        for cname, k in COHESION.items():
            f, g, a, r2 = kernels_for(drives, L, k)
            for D in (D_main, D_hi):
                for n in SIZES:
                    rg = goal_radius(n)
                    res = FE.minimum_dogs_local(f, g, D, n, rg)
                    out["rows"].append({"L": L, "cohesion": cname, "a": a, "cv_r2": r2, "D": D, "n_sheep": n,
                                        "r_goal": rg, "n_dogs": res["n_dogs"], "drift_error": res["relative_drift_error"]})
            r = [x for x in out["rows"] if x["L"] == L and x["cohesion"] == cname and x["D"] == D_main]
            print(f"L={L:g} {cname:34s} a={a:.2f}: " + "  ".join(
                f"N{x['n_sheep']}: {x['n_dogs']:.1f} (err {x['drift_error']:.2f})" for x in r))
    # microscopic checks: do the predicted dogs actually hold the flock?
    L = 3.0
    for n in (() if "--no-micro" in sys.argv else (14, 50)):
        rg = goal_radius(n)
        for cname, k in (("none (their assumption)", 0.0), ("measured", 1.0)):
            f, g, _, _ = kernels_for(drives, L, k)
            res = FE.minimum_dogs_local(f, g, D_main, n, rg)
            for dogs in sorted({max(1, round(res["n_dogs"] * fr)) for fr in (0.5, 1.0, 2.0)} | {1}):
                share = FE.micro_check_local(f, g, D_main, n, rg, res["dog_positions"], res["dog_counts"], dogs) \
                    if res["dog_counts"].sum() > 0 else float("nan")
                out["micro_checks"].append({"n_sheep": n, "cohesion": cname, "predicted": res["n_dogs"],
                                            "dogs": dogs, "share_inside_goal": share})
                print(f"micro N={n} {cname[:8]:8s} predicted {res['n_dogs']:.1f}: {dogs} dogs -> "
                      f"{100 * share:.0f}% of sheep inside the goal")
    (OUT / "feasibility.json").write_text(json.dumps(out, indent=2, default=float))
    plot(out, D_main)


def plot(out, D_main):
    cols = {"none (their assumption)": "#eb6834", "measured": "#2a78d6", "measured x4 (shrinkage-corrected)": "#1baf7a"}
    fig, axs = plt.subplots(1, len(RANGES), figsize=(17, 4), sharey=True)
    for ax, L in zip(axs, RANGES):
        for cname, col in cols.items():
            rows = [r for r in out["rows"] if r["L"] == L and r["cohesion"] == cname and r["D"] == D_main]
            hi = [r for r in out["rows"] if r["L"] == L and r["cohesion"] == cname and r["D"] != D_main]
            ax.plot([r["n_sheep"] for r in rows], [r["n_dogs"] for r in rows], "o-", color=col, label=cname)
            ax.plot([r["n_sheep"] for r in hi], [r["n_dogs"] for r in hi], ":", color=col)
        ax.axhline(1, color="#52514e", lw=0.8, ls="--")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_title(f"Dog kernel range L = {L:g} m", fontsize=10), ax.set_xlabel("Sheep in the flock")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Minimum number of dogs to hold the flock\nin a goal as tight as the real flock")
    axs[0].legend(fontsize=8, title="Sheep cohesion", title_fontsize=8)
    fig.suptitle("Continuification feasibility with measured kernels (solid: D from 1 s windows; dotted: from 4 s windows)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "feasibility.png", dpi=110)


if __name__ == "__main__":
    main()
