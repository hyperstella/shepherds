"""What cohesion does to Di Lorenzo's design: it shapes the flock, but the dogs still hold its position.

Pair forces cancel in the centroid (the pair kernel is antisymmetric), so a rigidly displaced
flock is pulled back by the dogs alone, at the stiffness k_dog = -1/2 <div u_dog> under the goal
density. The design asks the dogs for u_dog = D grad ln n* - K_g * n*, so

    k_dog = D/s^2 - k_coh,      k_coh = -1/2 <div K_g * n*>

and k_coh / (D/s^2) is the share of the confinement cohesion takes over. This script measures
that share for the measured pair kernel (and four times it, the shrinkage-corrected variant of
the feasibility table), and checks the table's dog saving from cohesion by comparing designs on
a common ring: the non-negative least squares of minimum_dogs_local minimizes the drift error,
not the dog count, so two of its solutions can differ in count because they sit at different
radii and errors.

    .venv/bin/python experiments/cohesion_stiffness.py
Writes results/cohesion_stiffness.json and results/figures/cohesion_stiffness.png.
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
RANGES = (3.0, 5.0, 8.0)
SIZES = (14, 30, 50)
SCALES = {"none": 0.0, "measured": 1.0, "measured x4": 4.0}
G_MAX = 3.6  # m, as in experiments/feasibility.py
TOL = (0.05, 0.10)  # drift-error budgets for the cheapest single ring
COL = {"none": "#eb6834", "measured": "#2a78d6", "measured x4": "#1baf7a"}


def goal_radius(n):
    return 2.5 * np.sqrt(n / 14)


def design(f, g, D, n, rg):
    """NNLS design of minimum_dogs_local, with its mass-weighted radius and the stiffness it realizes."""
    res = FE.minimum_dogs_local(f, g, D, n, rg)
    C, m = res["dog_positions"], res["dog_counts"]
    P, w, _, _, s = FE.goal_points(n, rg)
    rd = P[:, None] - C[None]
    dd = np.maximum(np.linalg.norm(rd, axis=-1), 1e-6)
    u = ((m * f(dd) / dd)[..., None] * rd).sum(1)
    r = np.hypot(*C.T)
    return dict(n_dogs=res["n_dogs"], drift_error=res["relative_drift_error"],
                radius=float(np.average(r, weights=m)) if m.sum() > 0 else None, k_dog=FE.centroid_stiffness(P, w, u, s))


def cheapest(rings, tol):
    ok = [r for r in rings if r["drift_error"] <= tol and r["mass"] > 0]
    return min(ok, key=lambda r: r["mass"]) if ok else None


def main():
    drives = data.load_drives(data.load_sessions())
    D = json.loads((OUT / "feasibility.json").read_text())["D_by_window_s"]["1.0"]
    # what spreads the flock's shape is each sheep's own residual, not the part the whole flock shares
    # (experiments/residual_persistence.py); its diffusivity, averaged over the two flock axes
    rp = json.loads((OUT / "residual_persistence.json").read_text())["fits_pooled"]
    D_ind = (rp["ind_par"]["D_inf"] + rp["ind_perp"]["D_inf"]) / 2
    pair = HatBasis(FE.PAIR_KNOTS_FO)
    out = {"D": D, "D_individual": D_ind, "g_cutoff_m": G_MAX, "rows": []}
    for L in RANGES:
        a_g, cg, _ = FE.fit_exponential(drives, L, cohesion=True)
        a_0, _, _ = FE.fit_exponential(drives, L, cohesion=False)
        for n in SIZES:
            rg = goal_radius(n)
            P, w, n_star, step, s = FE.goal_points(n, rg)
            radii = np.arange(rg + 1.0, rg + 10.01, 0.25)
            row = {"L": L, "n_sheep": n, "r_goal": rg, "s": s, "k_bowl": D / s ** 2, "cases": {}}
            for name, k in SCALES.items():
                a = a_g if k else a_0
                f = lambda r, a=a: a * np.exp(-r / L)
                g = (lambda r, k=k: np.where(r <= G_MAX, k * pair.curve(cg, r), 0.0)) if k else None
                k_coh = FE.centroid_stiffness(P, w, FE.cohesion_drift(g, P, n_star, step), s) if g else 0.0
                rings = FE.single_rings(f, g, D, n, rg, radii)
                row["cases"][name] = dict(a=a, k_coh=k_coh, coh_share=k_coh / (D / s ** 2), coh_share_D_individual=k_coh / (D_ind / s ** 2),
                                          nnls=design(f, g, D, n, rg),
                                          rings=rings, cheapest={str(t): cheapest(rings, t) for t in TOL})
            # the same rings, with and without cohesion: what cohesion saves at a fixed radius
            base = row["cases"]["none"]["rings"]
            for name in ("measured", "measured x4"):
                ratio = [r["mass"] / b["mass"] for r, b in zip(row["cases"][name]["rings"], base) if b["mass"] > 0 and r["mass"] > 0]
                row["cases"][name]["mass_ratio_same_ring"] = dict(median=float(np.median(ratio)), min=float(np.min(ratio)), max=float(np.max(ratio))) if ratio else None
            out["rows"].append(row)
            c = row["cases"]
            print(f"L={L:g} N={n:3d} D/s2={D / s ** 2:.3f}  " + "  ".join(
                f"{nm}: coh share {c[nm]['coh_share']:.2f} ({c[nm]['coh_share_D_individual']:.2f} at D_ind), NNLS {c[nm]['nnls']['n_dogs']:.1f} dogs at "
                f"{c[nm]['nnls']['radius'] or float('nan'):.1f} m (err {c[nm]['nnls']['drift_error']:.2f}, k_dog {c[nm]['nnls']['k_dog']:.3f})"
                + (f", same-ring mass x{c[nm]['mass_ratio_same_ring']['median']:.2f}" if c[nm].get("mass_ratio_same_ring") else "")
                for nm in SCALES), flush=True)
    (OUT / "cohesion_stiffness.json").write_text(json.dumps(out, indent=1))
    figure(out)


def figure(out):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    row = next(r for r in out["rows"] if r["L"] == 3.0 and r["n_sheep"] == 14)
    for name, c in row["cases"].items():
        rr = np.array([r["radius"] for r in c["rings"]])
        mass = np.array([r["mass"] for r in c["rings"]])
        err = np.array([r["drift_error"] for r in c["rings"]])
        ok = (mass > 0) & (err < 0.5)  # a ring near the push range cannot shape the drift at any mass
        nn = c["nnls"]
        axs[0].plot(rr[ok], mass[ok], color=COL[name], lw=2, label=f"cohesion {name} (NNLS: {nn['n_dogs']:.1f} dogs)")
        axs[1].plot(rr[ok], err[ok], color=COL[name], lw=2, label=name)
        if nn["radius"] is not None:
            axs[0].plot(nn["radius"], nn["n_dogs"], "o", ms=9, color=COL[name], mec="white", mew=2)
            axs[1].plot(nn["radius"], nn["drift_error"], "o", ms=9, color=COL[name], mec="white", mew=2)
    axs[0].set(xlabel="ring radius (m)", ylabel="dog mass on the ring", ylim=(0, 25),
               title="Dogs needed on one ring, 14 sheep, L = 3 m\n(dots: the NNLS designs of the feasibility table)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].set(xlabel="ring radius (m)", ylabel="relative drift error", ylim=(0, 0.3), title="Drift error of that ring")
    axs[1].legend(frameon=False, fontsize=8)
    ax = axs[2]
    for name in ("measured", "measured x4"):
        for L, ls in zip(RANGES, ("-", "--", ":")):
            rows = [r for r in out["rows"] if r["L"] == L]
            ax.plot([r["n_sheep"] for r in rows], [r["cases"][name]["coh_share"] for r in rows], ls, marker="o", color=COL[name],
                    lw=2, label=f"{name}, L = {L:g} m")
    ax.axhline(1, color="#8a8a8a", lw=1, ls=":", label="1: cohesion alone holds the shape")
    ax.set(xlabel="sheep in the flock", ylabel="k_coh / (D/s²)", xticks=SIZES,
           title="Share of the goal's curvature cohesion supplies\n(the dogs supply the rest, and all of the centroid's)")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cohesion_stiffness.png", dpi=130)


if __name__ == "__main__":
    main()
