"""Phase 4: learn interaction kernels from the tracks (Q1 identifiability, Q3 mechanism).

    .venv/bin/python experiments/fit_kernels.py            # real data + synthetic study
    .venv/bin/python experiments/fit_kernels.py --quick    # fewer synthetic repeats

Writes results/kernels.json, results/tables/kernel_fit.npz and
results/figures/{kernels,identifiability}.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, kernels as K  # noqa: E402

OUT = ROOT / "results"
H, WIN = 1.0, 2.1  # integration window and smoothing window (s), chosen on synthetic data
LAMS = np.logspace(-3, 2, 11)
MAIN = ("E", "A", "D", "DV", "DL")  # headline model: flock speed is an external input
VARIANTS = {
    "full (with self-propulsion)": dict(terms=("self", "E", "A", "D", "DV", "DL")),
    "main: E+A+D+DV+DL": dict(terms=MAIN),
    "no alignment": dict(terms=("E", "D", "DV", "DL")),
    "no attraction/repulsion": dict(terms=("A", "D", "DV", "DL")),
    "dog position only": dict(terms=("E", "A", "D")),
    "no dog terms": dict(terms=("E", "A")),
    "topological, 4 nearest": dict(terms=MAIN, topo_k=4),
    "topological, 7 nearest": dict(terms=MAIN, topo_k=7),
}
C = {"E": "#2a78d6", "A": "#1baf7a", "D": "#eb6834", "DV": "#e87ba4", "DL": "#4a3aa7"}
LABEL = {"E": "φ_E attraction(+)/repulsion(−)", "A": "φ_A alignment", "D": "φ_D flight from dog",
         "DV": "ψ_V response to dog velocity (−: against it)", "DL": "ψ_L flight × dog closing speed"}


def split_r2(model, lam, drives, normals):
    """Leave-one-session-out R2, total and split into flock-mean vs within-flock parts."""
    acc = np.zeros(4)
    for s in sorted({d.session for d in drives}):
        c = K.solve(model, [n for n in normals if n.session != s], lam)
        for d in drives:
            if d.session != s:
                continue
            pos, vel = K.smooth_drive(d.pos[: K.N + 1], window_s=WIN)
            x, v = np.moveaxis(pos, -1, 0), np.moveaxis(vel, -1, 0)
            F = model.features(x[:, :K.N], v[:, :K.N], x[:, K.N], v[:, K.N])
            Cm = cumulative_trapezoid(F, dx=data.DT, axis=0, initial=0)
            h = int(round(H / data.DT))
            t0 = np.arange(0, x.shape[0] - h, 2)
            y, p = v[t0 + h, :K.N] - v[t0, :K.N], (Cm[t0 + h] - Cm[t0]) @ c
            ym, pm = y.mean(1, keepdims=True), p.mean(1, keepdims=True)
            acc += [((ym - pm) ** 2).sum() * K.N, (ym ** 2).sum() * K.N,
                    (((y - ym) - (p - pm)) ** 2).sum(), ((y - ym) ** 2).sum()]
    return {"flock_mean_r2": float(1 - acc[0] / acc[1]), "within_flock_r2": float(1 - acc[2] / acc[3]),
            "flock_mean_share": float(acc[1] / (acc[1] + acc[3]))}


def fit_real(drives):
    res = {}
    for name, kw in VARIANTS.items():
        m = K.KernelModel(**kw)
        nrm = K.normal_equations(m, drives, H=H, window_s=WIN)
        lam, rows = K.cross_validate(m, nrm, LAMS)
        res[name] = {"lambda": lam, "cv_r2": max(r for _, r in rows), **split_r2(m, lam, drives, nrm),
                     "coercivity": K.coercivity(m, nrm)}
        print(f"{name:32s} CV R2 {res[name]['cv_r2']:.3f}  flock-mean {res[name]['flock_mean_r2']:.3f}  "
              f"within-flock {res[name]['within_flock_r2']:.3f}")
    return res


def main_kernels(drives):
    m = K.KernelModel(terms=MAIN)
    nrm = K.normal_equations(m, drives, H=H, window_s=WIN)
    lam, _ = K.cross_validate(m, nrm, LAMS)
    c = K.solve(m, nrm, lam)
    folds = [K.solve(m, [n for n in nrm if n.session != s], lam) for s in sorted({d.session for d in drives})]
    return m, c, folds, lam, K.noise_level(nrm, c, H=H)


def synthetic_study(drives, m, c_true, sigma, reps, meas_sigmas=(0.1, 0.2), sizes=(5, 10, 20, 29), seed=0):
    """Recover known kernels (the real-data fit) from synthetic flocks with tag noise."""
    W = K.distance_weights(drives)
    rng = np.random.default_rng(seed)
    rows = []
    for meas in meas_sigmas:
        for n in sizes:
            for r in range(reps if n < len(drives) else max(1, reps // 2)):
                pick = sorted(rng.choice(len(drives), n, replace=False)) if n < len(drives) else range(len(drives))
                sub = [drives[i] for i in pick]
                pieces, tracks = K.synthetic_pieces(m, c_true, sigma, sub, seed=int(rng.integers(1 << 31)),
                                                    meas_sigma=meas, window_s=WIN, speed_replay=True)
                nn = K.normal_equations(m, pieces, H=H, tracks=tracks)
                lam, _ = K.cross_validate(m, nn, LAMS) if len({p.session for p in pieces}) > 1 else (1.0, None)
                err = K.curve_error(m, c_true, K.solve(m, nn, lam), W)
                rows.append({"meas_sigma": meas, "n_drives": n, "rep": r, **err})
    return rows


def plot_kernels(m, c, folds, drives, out):
    W = K.distance_weights(drives)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    groups = [(axs[0], ("E",), "pair", "Sheep–sheep distance r (m)"), (axs[1], ("A",), "pair", "Sheep–sheep distance r (m)"),
              (axs[2], ("D", "DV", "DL"), "dog", "Sheep–dog distance r (m)")]
    for ax, terms, kind, xl in groups:
        r, w = W[kind]
        ax2 = ax.twinx()
        ax2.fill_between(r, 0, w, color="#d8d7d0", alpha=0.6, lw=0)
        ax2.set_yticks([]), ax2.set_ylim(0, w.max() * 3)
        for t in terms:
            basis = m.pair if kind == "pair" else m.dog
            sl = m.slices()[t]
            ys = np.array([basis.curve(f[sl], r) for f in folds])
            ax.fill_between(r, ys.min(0), ys.max(0), color=C[t], alpha=0.2, lw=0)
            ax.plot(r, basis.curve(c[sl], r), color=C[t], lw=2, label=LABEL[t])
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xlabel(xl), ax.set_ylabel("Kernel value (m/s² per unit)")
        ax.set_zorder(ax2.get_zorder() + 1), ax.patch.set_visible(False)
        ax.legend(fontsize=8, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].axvline(1.2, color="#52514e", ls=":", lw=1)
    axs[0].text(1.25, axs[0].get_ylim()[1] * 0.9, "body length", fontsize=8, color="#52514e")
    fig.suptitle("Learned interaction kernels (bands: range over leave-one-session-out fits; grey: where data lie)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def plot_identifiability(rows, out):
    terms = [t for t in ("E", "A", "D", "DV", "DL") if t in rows[0]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, meas in zip(axs, sorted({r["meas_sigma"] for r in rows})):
        for t in terms:
            sizes = sorted({r["n_drives"] for r in rows})
            med = [np.median([r[t] for r in rows if r["n_drives"] == n and r["meas_sigma"] == meas]) for n in sizes]
            ax.plot(sizes, med, "o-", color=C[t], label=LABEL[t])
        ax.axhline(1, color="#52514e", ls=":", lw=1)
        ax.set_xlabel("Drives used for fitting"), ax.set_title(f"Tag noise σ = {meas:g} m", fontsize=10)
        ax.set_yscale("log"), ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Kernel error (relative, weighted by data density)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Q1  Recovering known kernels from synthetic flocks driven by the real dog", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    drives = data.load_drives(data.load_sessions())

    variants = fit_real(drives)
    m, c, folds, lam, sigma_resid = main_kernels(drives)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tables" / "kernel_fit.npz", c=c, folds=np.array(folds), terms=np.array(m.terms), lam=lam,
             sigma_resid=sigma_resid, H=H, window_s=WIN)
    plot_kernels(m, c, folds, drives, OUT / "figures" / "kernels.png")

    rows = synthetic_study(drives, m, c, sigma=0.1, reps=2 if args.quick else 4)
    plot_identifiability(rows, OUT / "figures" / "identifiability.png")
    summary = {}
    for meas in sorted({r["meas_sigma"] for r in rows}):
        for n in sorted({r["n_drives"] for r in rows}):
            sel = [r for r in rows if r["meas_sigma"] == meas and r["n_drives"] == n]
            summary[f"meas{meas:g}_n{n}"] = {t: float(np.median([r[t] for r in sel])) for t in m.slices()}
    cur = K.kernel_curves(m, c)
    rE, yE = cur["E"]
    cross = rE[1:][np.diff(np.sign(yE)) > 0]
    out = {"H_s": H, "smooth_window_s": WIN, "variants": variants, "main_lambda": lam,
           "residual_sigma": sigma_resid, "E_zero_crossing_m": [float(v) for v in cross],
           "synthetic_recovery_median_error": summary}
    (OUT / "kernels.json").write_text(json.dumps(out, indent=2))
    print("E kernel changes sign (repulsion -> attraction) at r =", np.round(cross, 2), "m")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
