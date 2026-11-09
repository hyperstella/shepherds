"""Is the first-order model's leftover motion diffusion? Persistence and coherence of the residual.

Di Lorenzo's targets move by drift plus independent white noise, so the density obeys a
Fokker-Planck equation. Fit the first-order model of shepherd/feasibility.py (dog push, pair
kernel) to the tracks and look at what it leaves over. If the residual displacement over a
window w comes from a velocity that forgets itself in a time tau, the apparent diffusivity is

    D(w) = D_inf [1 - (tau/w)(1 - exp(-w/tau))] + sigma^2 / w

(sigma: position noise). The density is a valid state only if tau is short against the flock's
own relaxation time s^2/D_inf, i.e. if the persistence length sqrt(D_inf tau) is small against
the flock radius s (a Knudsen number Kn = sqrt(D_inf tau)/s well below 1).

The residual is split into the flock-mean part (common to all sheep: unmodelled coherent motion)
and each sheep's deviation from it (individual noise), both resolved in the flock frame (along and
across the flock's heading). For independent noise the flock-mean variance is 1/N of a sheep's,
so the coherence ratio chi = N var(mean residual) / var(individual residual) is 1; in general
chi - 1 = N var(shared part) / var(individual part), so chi = N + 1 when the shared part is as large
as each sheep's own.

    .venv/bin/python experiments/residual_persistence.py
Writes results/residual_persistence.json and results/figures/residual_persistence.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, feasibility as FE, kernels as K  # noqa: E402

OUT = ROOT / "results"
N = data.N_SHEEP
WINDOWS = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)  # s
MIN_DRIVE_S = 20.0  # per-drive fits only on drives at least this long
# The individual residual saturates beyond about 4 s (relative positions are bounded by cohesion the
# first-order pair kernel underestimates), so its persistence is fitted below that.
FIT_MAX = {"ind_par": 4.0, "ind_perp": 4.0, "com_par": 8.0, "com_perp": 8.0}
PARTS = ("ind_par", "ind_perp", "com_par", "com_perp")
COL = {"ind_par": "#2a78d6", "ind_perp": "#1baf7a", "com_par": "#eb6834", "com_perp": "#7b61c4"}
LABEL = {"ind_par": "individual, along the heading", "ind_perp": "individual, across",
         "com_par": "flock mean ×N, along", "com_perp": "flock mean ×N, across"}


def d_of_w(w, D_inf, tau, sig2):
    return D_inf * (1 - (tau / w) * (1 - np.exp(-w / tau))) + sig2 / w


def fit_dw(ws, Ds, w_max=np.inf):
    ws, Ds = np.asarray(ws), np.asarray(Ds)
    ok = np.isfinite(Ds) & (ws <= w_max)
    if ok.sum() < 4:
        return None
    try:
        p, _ = curve_fit(d_of_w, ws[ok], Ds[ok], p0=(Ds[ok].max(), 1.0, 0.001), sigma=Ds[ok],  # relative errors
                         bounds=([0, 0.02, 0], [50, 100, 1]), maxfev=20000)
    except RuntimeError:
        return None
    return dict(D_inf=float(p[0]), tau=float(p[1]), sigma2=float(p[2]))


def residual_variances(fit, model, d):
    """Per window: summed squares and counts of the residual parts, plus the flock's size."""
    c = np.concatenate([fit.c_g, fit.c_f])
    pos = d.pos[: N + 1]
    sm, vel = K.smooth_drive(pos)
    x = np.moveaxis(sm, -1, 0)  # (T, N+1, 2)
    F = model.features(x[:, :N], None, x[:, N]) @ c  # (T, N, 2) first-order drift
    drift = np.cumsum(F, axis=0) * data.DT
    raw = np.moveaxis(pos[:N], -1, 0)
    V = np.moveaxis(vel[:N], -1, 0).mean(1)  # flock velocity (T, 2)
    e_par = V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-9)
    e_perp = np.stack([-e_par[:, 1], e_par[:, 0]], -1)
    rel = x[:, :N] - x[:, :N].mean(1, keepdims=True)
    s_par = float(np.sqrt(np.mean(np.einsum("tnc,tc->tn", rel, e_par) ** 2)))
    s_perp = float(np.sqrt(np.mean(np.einsum("tnc,tc->tn", rel, e_perp) ** 2)))
    out = {}
    for w in WINDOWS:
        h = int(round(w / data.DT))
        if raw.shape[0] <= h + 1:
            continue
        t0 = np.arange(0, raw.shape[0] - h, max(1, h // 2))
        r = (raw[t0 + h] - raw[t0]) - (drift[t0 + h] - drift[t0])  # (k, N, 2)
        rm = r.mean(1)
        ri = r - rm[:, None]
        acc = {}
        for name, e in (("par", e_par[t0]), ("perp", e_perp[t0])):
            acc[f"ind_{name}"] = (float((np.einsum("knc,kc->kn", ri, e) ** 2).sum() * N / (N - 1)), ri.shape[0] * N)
            acc[f"com_{name}"] = (float((np.einsum("kc,kc->k", rm, e) ** 2).sum() * N), rm.shape[0])  # x N: per-sheep scale
        out[w] = acc
    return out, s_par, s_perp


def to_D(acc):
    """D(w) = variance / (2 w) per part, from accumulated squares."""
    D = {p: [] for p in PARTS}
    for w in WINDOWS:
        for p in PARTS:
            if w in acc and acc[w][p][1] > 0:
                D[p].append(acc[w][p][0] / acc[w][p][1] / (2 * w))
            else:
                D[p].append(np.nan)
    return D


def main():
    drives = data.load_drives(data.load_sessions())
    fit = FE.fit_first_order(drives, cohesion=True)
    model = K.KernelModel(terms=("E", "D"))
    pooled = {w: {p: [0.0, 0] for p in PARTS} for w in WINDOWS}
    per_drive = []
    for d in drives:
        acc, s_par, s_perp = residual_variances(fit, model, d)
        for w, a in acc.items():
            for p in PARTS:
                pooled[w][p][0] += a[p][0]
                pooled[w][p][1] += a[p][1]
        T_s = d.pos.shape[-1] * data.DT
        row = dict(session=d.session, index=d.index, seconds=T_s, s_par=s_par, s_perp=s_perp, s=float(np.sqrt((s_par ** 2 + s_perp ** 2) / 2)))
        if T_s >= MIN_DRIVE_S:
            D = to_D(acc)
            row["D_w"] = D
            row["fits"] = {p: fit_dw(WINDOWS, D[p], FIT_MAX[p]) for p in PARTS}
        per_drive.append(row)
    Dp = to_D(pooled)
    fits = {p: fit_dw(WINDOWS, Dp[p], FIT_MAX[p]) for p in PARTS}
    s_all = float(np.median([r["s"] for r in per_drive]))
    s_perp = float(np.median([r["s_perp"] for r in per_drive]))
    summary = {}
    for p in ("ind_par", "ind_perp"):
        f = fits[p]
        s_ref = s_perp if p == "ind_perp" else float(np.median([r["s_par"] for r in per_drive]))
        summary[p] = dict(**f, persistence_length=float(np.sqrt(f["D_inf"] * f["tau"])), s=s_ref,
                          knudsen=float(np.sqrt(f["D_inf"] * f["tau"]) / s_ref), relax_time=float(s_ref ** 2 / f["D_inf"]))
    chi = {w: {ax: Dp[f"com_{ax}"][i] / Dp[f"ind_{ax}"][i] for ax in ("par", "perp")} for i, w in enumerate(WINDOWS)}
    pd = [r for r in per_drive if "fits" in r and r["fits"]["ind_perp"] is not None]
    kn = [np.sqrt(r["fits"]["ind_perp"]["D_inf"] * r["fits"]["ind_perp"]["tau"]) / r["s_perp"] for r in pd]
    out = dict(windows=WINDOWS, first_order_cv_r2=fit.cv_r2, D_pooled=Dp, fits_pooled=fits, summary=summary,
               coherence_ratio=chi, shared_over_individual={w: {ax: (c[ax] - 1) / N for ax in c} for w, c in chi.items()},
               centroid={ax: dict(D_inf=fits[f"com_{ax}"]["D_inf"] / N, tau=fits[f"com_{ax}"]["tau"],
                                  persistence_length=float(np.sqrt(fits[f"com_{ax}"]["D_inf"] / N * fits[f"com_{ax}"]["tau"])),
                                  independent_noise_D=fits[f"ind_{ax}"]["D_inf"] / N,
                                  # a tau beyond half the longest window means still ballistic there: a lower bound only
                                  resolved=bool(fits[f"com_{ax}"]["tau"] < max(WINDOWS) / 2)) for ax in ("par", "perp")},
               fit_max_s=FIT_MAX, flock_s_median=s_all, per_drive=per_drive,
               per_drive_knudsen_perp=dict(n=len(kn), median=float(np.median(kn)), p10=float(np.percentile(kn, 10)),
                                           p90=float(np.percentile(kn, 90)), frac_above_1=float(np.mean(np.array(kn) > 1))),
               per_drive_tau_perp=dict(median=float(np.median([r["fits"]["ind_perp"]["tau"] for r in pd])),
                                       p10=float(np.percentile([r["fits"]["ind_perp"]["tau"] for r in pd], 10)),
                                       p90=float(np.percentile([r["fits"]["ind_perp"]["tau"] for r in pd], 90))))
    (OUT / "residual_persistence.json").write_text(json.dumps(out, indent=1, default=float))
    for p, f in fits.items():
        print(f"{p:9s} D_inf {f['D_inf']:.3f} m²/s  tau {f['tau']:.2f} s  sigma² {f['sigma2']:.4f} m²" if f else f"{p}: no fit")
    for p, sm in summary.items():
        print(f"{p}: persistence length {sm['persistence_length']:.2f} m vs s {sm['s']:.2f} m -> Kn {sm['knudsen']:.2f}; "
              f"tau {sm['tau']:.2f} s vs s²/D {sm['relax_time']:.2f} s")
    print("centroid:", out["centroid"])
    print("coherence ratio chi (1 = independent):", {w: {k: round(v, 1) for k, v in c.items()} for w, c in chi.items()})
    print("per-drive Kn (across):", out["per_drive_knudsen_perp"], "tau:", out["per_drive_tau_perp"])
    figure(out)


def figure(out):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ws = np.array(out["windows"])
    wf = np.geomspace(ws.min(), ws.max(), 200)
    ax = axs[0]
    for p in PARTS:
        ax.plot(ws, out["D_pooled"][p], "o", color=COL[p], ms=6)
        f = out["fits_pooled"][p]
        if f:
            tau = f"τ {f['tau']:.1f} s, D∞ {f['D_inf']:.2f}" if f["tau"] < max(ws) / 2 else f"τ > {max(ws) / 2:g} s (still ballistic)"
            ax.plot(wf, d_of_w(wf, f["D_inf"], f["tau"], f["sigma2"]), color=COL[p], lw=2, label=f"{LABEL[p]}: {tau}")
    ax.set(xscale="log", yscale="log", xlabel="window w (s)", ylabel="apparent diffusivity D(w) (m²/s)",
           title="First-order residual, pooled over the 29 drives")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[1]
    chi = out["coherence_ratio"]
    for axn, c in (("par", "#eb6834"), ("perp", "#7b61c4")):
        ax.plot(ws, [chi[w][axn] if w in chi else chi[str(w)][axn] for w in ws], "o-", color=c, lw=2, label=f"{'along' if axn == 'par' else 'across'} the heading")
    ax.axhline(1, color="#8a8a8a", lw=1, ls=":", label="1: independent noise (Di Lorenzo)")
    ax.axhline(N + 1, color="#8a8a8a", lw=1, ls="--", label=f"{N + 1}: shared part as large as each sheep's own")
    ax.set(xscale="log", yscale="log", xlabel="window w (s)", ylabel="χ = N var(flock-mean residual) / var(individual)",
           title="How much of the residual is shared")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax = axs[2]
    pd = [r for r in out["per_drive"] if "fits" in r and r["fits"]["ind_perp"] is not None]
    relax = np.array([r["s_perp"] ** 2 / r["fits"]["ind_perp"]["D_inf"] for r in pd])
    tau = np.array([r["fits"]["ind_perp"]["tau"] for r in pd])
    ax.loglog(relax, tau, "o", color=COL["ind_perp"], ms=7, mec="white", mew=1.5, label=f"one drive ≥ {MIN_DRIVE_S:g} s (n = {len(pd)})")
    lim = [min(relax.min(), tau.min()) / 1.5, max(relax.max(), tau.max()) * 1.5]
    ax.plot(lim, lim, color="#8a8a8a", lw=1, ls=":", label="τ = s²/D∞ (Kn = 1)")
    ax.set(xlabel="flock relaxation time s⊥²/D∞ (s)", ylabel="persistence τ of the individual residual (s)",
           title="Across the heading, per drive:\ndiffusive below the diagonal")
    ax.legend(frameon=False, fontsize=7.5)
    ticks = [0.2, 0.5, 1, 2, 4, 8]
    for ax in axs[:2]:
        ax.minorticks_off()
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "residual_persistence.png", dpi=130)


if __name__ == "__main__":
    main()
