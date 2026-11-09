"""The second moment of a Jadhav flock, one step ahead: exact decomposition and a drift bound.

With xi_i = x_i - c, M2 = (1/N) sum |xi_i|^2, a_i the actual one-step displacements and abar their
mean, exactly
    dM2 = (2/N) sum xi_i . a_i + (1/N) sum |a_i - abar|^2.
Under Jadhav's rule a_i = v m_i h_i', with m_i in {0, 1} set by the current state (a sheep moves
when it sees the dog or is repelled; the published stop rule) and h_i' = unit(S_i) its new
heading. Given the full current state the random draws (neighbour subsets, noise) are independent
across sheep, so with mu_i = E[h_i' | state], f_m = (1/N) sum m_i and Mv = (1/N) sum m_i mu_i,
    E[dM2 | state] = C + G,
    C = (2v/N) sum m_i xi_i . mu_i                                  (radial drift)
    G = v^2 [ f_m - |Mv|^2 - (1/N^2) sum m_i (1 - |mu_i|^2) ]        (heading dispersion)
      <= v^2 (f_m - f_m^2 P^2),  P = |Mv| / f_m  (= v^2 (1 - P^2) when every sheep moves)
and C = C_rel + C_frz with mubar = Mv / f_m,
    C_rel = (2v/N) sum m_i xi_i . (mu_i - mubar),   C_frz = -(2v/N) (sum_frozen xi_i) . mubar,
C_rel split along and across mubar. Nothing here assumes an additive or diffusive model.

The conditional means are estimated by Monte Carlo: each state is replicated K times and advanced
one step with independent draws (simulate with a fixed replayed dog). States come from driven
flocks (published rule; two weight sets, three standoffs, two notice radii), from a paper-text
parameter variant (n_att 5, noise 0.5, as in the paper's Methods; the code and everything else
here use 4 and 0.3), and from dog-free flocks (the null model, i.e. the modified rule). Each state
is also dilated about its centroid by 1.5, 2 and 3 to probe large spreads.

Checked: the sample identity; whether E[dM2 | state] <= -a sqrt(M2) + b (constant-speed
restoring, what unit-vector attraction can give) or <= -a M2 + b holds, and where; the
sufficient condition of docs/steer_and_shape.md §11 (all sheep move; every sheep beyond r0 has an
inward mean heading relative to the flock's, by at least c); and what breaks it.

    .venv/bin/python experiments/shape_drift.py
Writes results/shape_drift.json and results/figures/shape_drift.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
N, T_RUN, SNAPS, REPS, K, SEED = 14, 300, (150, 180, 210, 240, 270), 6, 128, 41
FAR = 2000.0
V = 1.0
DIL = (1.0, 1.5, 2.0, 3.0)
CHUNK = 24000
KAPPA_STAR = 0.05  # the weighted inward relative heading the drift lemma asks for, in panel (d)
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ------------------------------------------------------------------ states

def settings():
    rows = []
    for (wa, wl), d, rd in itertools.product(((1.5, 1.3), (3.0, 3.0)), (5.0, 7.5, 10.0), (12.0, 20.0)):
        rows.append(dict(kind="driven", w_att=wa, w_ali=wl, l_drive=d, r_dog=rd, n_att=4, noise=0.3))
    rows.append(dict(kind="driven", w_att=1.5, w_ali=1.3, l_drive=7.5, r_dog=12.0, n_att=5, noise=0.5))  # paper-text values
    for wa, wl in ((1.5, 1.3), (3.0, 3.0)):
        rows.append(dict(kind="free", w_att=wa, w_ali=wl, l_drive=None, r_dog=None, n_att=4, noise=0.3))
    return rows


def base_states(row):
    """Snapshots (pos, heading, dog) of REPS runs of one setting."""
    p = Params(n_sheep=N, box_length=FAR, n_steps=T_RUN, n_att=row["n_att"], dog_present=row["kind"] == "driven")
    per = dict(w_att=np.full(REPS, row["w_att"]), w_ali=np.full(REPS, row["w_ali"]), noise=np.full(REPS, row["noise"]), noise_dog=np.full(REPS, 0.3))
    if row["kind"] == "driven":
        per.update(l_drive=np.full(REPS, row["l_drive"]), r_dog=np.full(REPS, row["r_dog"]), l_sep=np.full(REPS, 1e6))
    res = simulate(p, n_runs=REPS, seed=SEED, per_run=per)
    out = []
    for t in SNAPS:
        for r in range(REPS):
            out.append((res.pos_s[r, :, :, t], res.vel_s[r, :, :, t], res.pos_d[r, :, t]))
    return out


def dilate(pos, s):
    c = pos.mean(0)
    return c + s * (pos - c)


# ------------------------------------------------------------------ one-step Monte Carlo

def one_step(row, pos, hd, dog, seed):
    """K replicates of one step from each of the given states. pos (S, N, 2) -> new positions (S, K, N, 2)."""
    S = len(pos)
    R = S * K
    p = Params(n_sheep=N, n_steps=2, n_att=row["n_att"], dog_present=row["kind"] == "driven")
    per = dict(w_att=np.full(R, row["w_att"]), w_ali=np.full(R, row["w_ali"]), noise=np.full(R, row["noise"]))
    init = dict(pos_s=np.repeat(pos, K, 0), vel_s=np.repeat(hd, K, 0))
    if row["kind"] == "driven":
        per["r_dog"] = np.full(R, row["r_dog"])
        dp = np.repeat(dog, K, 0)
        res = simulate(p, n_runs=R, seed=seed, per_run=per, init=init, dog_path=np.stack([dp, dp], -1))
    else:
        res = simulate(p, n_runs=R, seed=seed, per_run=per, init=init)
    return res.pos_s[..., -1].reshape(S, K, N, 2)


def analyse(pos, new):
    """Exact decomposition for states pos (S, N, 2) and their replicates new (S, K, N, 2)."""
    c = pos.mean(1, keepdims=True)
    xi = pos - c  # (S, N, 2)
    M2 = (xi ** 2).sum(-1).mean(-1)
    a = new - pos[:, None]  # (S, K, N, 2)
    abar = a.mean(2, keepdims=True)
    lhs = ((new - new.mean(2, keepdims=True)) ** 2).sum(-1).mean(-1) - M2[:, None]
    rhs = 2 * np.einsum("snc,sknc->sk", xi, a) / N + ((a - abar) ** 2).sum(-1).mean(-1)
    identity_err = np.abs(lhs - rhs).max()
    step = np.linalg.norm(a, axis=-1)  # (S, K, N)
    m = (step > 1e-9).mean(1) > 0.5  # moving is set by the state
    h = np.where(step[..., None] > 1e-9, a / np.maximum(step[..., None], 1e-12), 0.0)
    mu = h.mean(1)  # (S, N, 2): conditional mean heading (zero for frozen sheep)
    mu_sq = (mu ** 2).sum(-1)
    f_m = m.mean(1)
    Mv = (m[..., None] * mu).mean(1)  # (S, 2)
    # unbiased |Mv|^2: subtract the Monte Carlo variance of the mean
    var_mu = (m * (1 - mu_sq)).sum(1) / N ** 2 / K
    G = V ** 2 * (f_m - ((Mv ** 2).sum(-1) - var_mu) - (m * (1 - mu_sq)).sum(1) / N ** 2)
    C = 2 * V * (m * (xi * mu).sum(-1)).mean(1)
    mubar = Mv / np.maximum(f_m, 1e-12)[:, None]
    e = mubar / np.maximum(np.linalg.norm(mubar, axis=-1, keepdims=True), 1e-12)
    e_l = np.stack([-e[:, 1], e[:, 0]], -1)
    rel = mu - mubar[:, None]
    C_rel_par = 2 * V * (m * np.einsum("snc,sc->sn", xi, e) * np.einsum("snc,sc->sn", rel, e)).mean(1)
    C_rel_perp = 2 * V * (m * np.einsum("snc,sc->sn", xi, e_l) * np.einsum("snc,sc->sn", rel, e_l)).mean(1)
    C_frz = -2 * V * (((~m)[..., None] * xi).sum(1) * mubar).sum(-1) / N
    # inward relative heading of sheep beyond the RMS radius: q_i = xi_hat . (mu_i - mubar)
    r = np.linalg.norm(xi, axis=-1)
    q = np.einsum("snc,snc->sn", xi / np.maximum(r[..., None], 1e-12), rel)
    outer = r >= np.sqrt(M2)[:, None]
    q_out_max = np.where(outer & m, q, -np.inf).max(1)
    frac_out_inward = (np.where(outer & m, q < 0, False).sum(1) / np.maximum((outer & m).sum(1), 1))
    gamma = r.mean(1) / np.sqrt(M2)
    # weighted inward relative heading: with every sheep moving, C = -2 v kappa gamma sqrt(M2) exactly
    mean_r = r.mean(1)
    kappa = -(C_rel_par + C_rel_perp) / (2 * V * mean_r)
    kappa_par = -C_rel_par / (2 * V * mean_r)
    kappa_perp = -C_rel_perp / (2 * V * mean_r)
    dM2 = lhs.mean(1)
    se = lhs.std(1) / np.sqrt(K)
    P_mu = np.linalg.norm(mubar, axis=-1)
    return dict(M2=M2, dM2=dM2, se=se, C=C, G=G, C_rel_par=C_rel_par, C_rel_perp=C_rel_perp, C_frz=C_frz, f_m=f_m, P_mu=P_mu,
                q_out_max=q_out_max, frac_out_inward=frac_out_inward, gamma=gamma, dispersion_bound=V ** 2 * (f_m - f_m ** 2 * P_mu ** 2),
                kappa=kappa, kappa_par=kappa_par, kappa_perp=kappa_perp,
                identity_err=identity_err)


def run():
    tic = time.time()
    rows = settings()
    recs = []
    seed = SEED * 100
    for row in rows:
        bs = base_states(row)
        pos0 = np.stack([b[0] for b in bs])
        hd = np.stack([b[1] for b in bs])
        dog = np.stack([b[2] for b in bs])
        for s in DIL:
            pos = np.stack([dilate(p, s) for p in pos0])
            res = {}
            for i in range(0, len(pos), max(1, CHUNK // K)):
                sl = slice(i, i + max(1, CHUNK // K))
                seed += 1
                new = one_step(row, pos[sl], hd[sl], dog[sl], seed)
                part = analyse(pos[sl], new)
                for k, v in part.items():
                    res.setdefault(k, []).append(np.atleast_1d(v))
            res = {k: np.concatenate(v) for k, v in res.items()}
            recs.append(dict(row=row, dilation=s, **{k: v for k, v in res.items()}))
        print(f"  {row['kind']:6s} w_att {row['w_att']} w_ali {row['w_ali']} d {row['l_drive']} r_dog {row['r_dog']} n_att {row['n_att']} "
              f"noise {row['noise']}: done ({time.time() - tic:.0f} s)", flush=True)
    return recs


# ------------------------------------------------------------------ fits and summaries

def fit_forms(sqrtM2, y):
    """Least-squares fits y = -a sqrt(M2) + b and y = -a M2 + b, with residual RMS."""
    out = {}
    for name, x in (("sqrt", sqrtM2), ("linear", sqrtM2 ** 2)):
        A = np.stack([-x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[name] = dict(a=float(coef[0]), b=float(coef[1]), rms=float(np.sqrt(np.mean((A @ coef - y) ** 2))))
    return out


def summarize(recs):
    groups = {}
    for r in recs:
        key = json.dumps(r["row"], sort_keys=True)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        row = json.loads(key)
        allv = {k: np.concatenate([r[k] for r in rs]) for k in ("M2", "dM2", "C", "G", "C_rel_par", "C_rel_perp", "C_frz", "f_m", "P_mu", "q_out_max",
                                                               "frac_out_inward", "gamma", "dispersion_bound", "se", "kappa", "kappa_par", "kappa_perp")}
        dil = np.concatenate([np.full(len(r["M2"]), r["dilation"]) for r in rs])
        allmove = allv["f_m"] >= 0.999
        region = allmove & (allv["q_out_max"] < 0)
        sq = np.sqrt(allv["M2"])
        fits = fit_forms(sq[allmove], allv["dM2"][allmove]) if allmove.sum() > 5 else None
        fits_C = fit_forms(sq[allmove], allv["C"][allmove]) if allmove.sum() > 5 else None
        by_dil = []
        for s in DIL:
            sel = dil == s
            by_dil.append(dict(dilation=s, n=int(sel.sum()), M2_median=fl(np.median(allv["M2"][sel])), dM2_median=fl(np.median(allv["dM2"][sel])),
                               dM2_positive_frac=fl((allv["dM2"][sel] > 0).mean()), C_median=fl(np.median(allv["C"][sel])), G_median=fl(np.median(allv["G"][sel])),
                               C_rel_par_median=fl(np.median(allv["C_rel_par"][sel])), C_rel_perp_median=fl(np.median(allv["C_rel_perp"][sel])),
                               C_frz_median=fl(np.median(allv["C_frz"][sel])), all_move_frac=fl(allmove[sel].mean()), region_frac=fl(region[sel].mean()),
                               P_mu_median=fl(np.median(allv["P_mu"][sel])), frac_out_inward_median=fl(np.median(allv["frac_out_inward"][sel])),
                               gamma_median=fl(np.median(allv["gamma"][sel])),
                               kappa_median=fl(np.median(allv["kappa"][sel & allmove])) if (sel & allmove).any() else None,
                               kappa_p10=fl(np.percentile(allv["kappa"][sel & allmove], 10)) if (sel & allmove).any() else None,
                               kappa_par_median=fl(np.median(allv["kappa_par"][sel & allmove])) if (sel & allmove).any() else None,
                               kappa_perp_median=fl(np.median(allv["kappa_perp"][sel & allmove])) if (sel & allmove).any() else None,
                               kappa_nonpos_frac=fl((allv["kappa"][sel & allmove] <= 0).mean()) if (sel & allmove).any() else None))
        # failure: states with positive expected growth, what they have in common
        pos = allv["dM2"] > 2 * allv["se"]
        fail = dict(n=int(pos.sum()), frozen_share=fl((~allmove[pos]).mean()) if pos.any() else None,
                    outward_outer_share=fl((allv["q_out_max"][pos] >= 0).mean()) if pos.any() else None,
                    dilation_of_failures=[float(x) for x in np.unique(dil[pos])])
        at1 = (dil == 1) & allmove
        stationary = dict(sqrtM2=fl(np.mean(sq[dil == 1])), G=fl(np.mean(allv["G"][dil == 1])), C=fl(np.mean(allv["C"][dil == 1])),
                          kappa_gamma=fl(np.mean((allv["kappa"] * allv["gamma"])[at1])) if at1.any() else None,
                          P_mu=fl(np.mean(allv["P_mu"][dil == 1])))
        out.append(dict(**row, fits_dM2=fits, fits_C=fits_C, by_dilation=by_dil, failures=fail, stationary=stationary,
                        G_at_1=fl(np.median(allv["G"][dil == 1])), dispersion_bound_at_1=fl(np.median(allv["dispersion_bound"][dil == 1]))))
    return out


def figure(recs, summ):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.6))
    def pick(kind, wa, wl, d=None, rd=None, n_att=4):
        return [r for r in recs if r["row"]["kind"] == kind and r["row"]["w_att"] == wa and r["row"]["w_ali"] == wl
                and r["row"]["l_drive"] == d and r["row"]["r_dog"] == rd and r["row"]["n_att"] == n_att]
    ax = axs[0, 0]
    for (lab, rs, col) in (("driven, r_dog 20, d 7.5", pick("driven", 1.5, 1.3, 7.5, 20.0), "blue"),
                           ("driven, r_dog 12, d 7.5", pick("driven", 1.5, 1.3, 7.5, 12.0), "orange"),
                           ("dog-free (modified rule)", pick("free", 1.5, 1.3), "gray")):
        sq = np.concatenate([np.sqrt(r["M2"]) for r in rs])
        y = np.concatenate([r["dM2"] for r in rs])
        ax.scatter(sq, y, s=10, alpha=0.5, color=COL[col], label=lab, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="√M2 (m), states and their 1.5×, 2×, 3× dilations", ylabel="E[ΔM2 | state] (m² per step)",
           title="(a) Published weights: expected one-step change of M2")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 1]
    for (lab, rs, col) in (("driven, r_dog 20, d 7.5", pick("driven", 3.0, 3.0, 7.5, 20.0), "green"),
                           ("driven, r_dog 12, d 7.5", pick("driven", 3.0, 3.0, 7.5, 12.0), "orange"),
                           ("dog-free (modified rule)", pick("free", 3.0, 3.0), "gray")):
        sq = np.concatenate([np.sqrt(r["M2"]) for r in rs])
        y = np.concatenate([r["dM2"] for r in rs])
        ax.scatter(sq, y, s=10, alpha=0.5, color=COL[col], label=lab, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="√M2 (m)", ylabel="E[ΔM2 | state] (m² per step)", title="(b) Strong cohesion and alignment (w_att 3, w_ali 3)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    names = ("C_rel_par", "C_rel_perp", "C_frz", "G")
    labels = ("along: relative heading", "across: relative heading", "frozen sheep", "dispersion G")
    cols = ("blue", "green", "orange", "red")
    sels = [("pub r20 d7.5", pick("driven", 1.5, 1.3, 7.5, 20.0)), ("pub r12 d7.5", pick("driven", 1.5, 1.3, 7.5, 12.0)),
            ("pub r12 d10", pick("driven", 1.5, 1.3, 10.0, 12.0)), ("strong r20 d7.5", pick("driven", 3.0, 3.0, 7.5, 20.0)),
            ("free pub", pick("free", 1.5, 1.3)), ("free strong", pick("free", 3.0, 3.0))]
    x = np.arange(len(sels))
    for j, (nm, lab, col) in enumerate(zip(names, labels, cols)):
        vals = [np.median(next(r for r in rs if r["dilation"] == 1.0)[nm]) for _, rs in sels]
        ax.bar(x + (j - 1.5) * 0.2, vals, width=0.19, color=COL[col], label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [s[0] for s in sels], fontsize=8, rotation=20)
    ax.set(ylabel="median term at the actual states (m² per step)", title="(c) Exact decomposition of E[ΔM2 | state] = C + G (undilated)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 1]
    for (lab, rs, col) in (("pub, r_dog 20", pick("driven", 1.5, 1.3, 7.5, 20.0), "blue"), ("pub, r_dog 12", pick("driven", 1.5, 1.3, 7.5, 12.0), "orange"),
                           ("strong, r_dog 20", pick("driven", 3.0, 3.0, 7.5, 20.0), "green"), ("strong, r_dog 12", pick("driven", 3.0, 3.0, 7.5, 12.0), "purple")):
        ax.plot(DIL, [np.mean(r["f_m"] >= 0.999) for r in sorted(rs, key=lambda r: r["dilation"])], "o-", color=COL[col], lw=2, label=f"all sheep move, {lab}")
        ax.plot(DIL, [np.mean((r["f_m"] >= 0.999) & (r["kappa"] >= KAPPA_STAR)) for r in sorted(rs, key=lambda r: r["dilation"])], "s--", color=COL[col], lw=1.2)
    ax.set(xlabel="dilation of the observed state", ylabel="fraction of states", ylim=(-0.02, 1.02),
           title=f"(d) Where the drift lemma's region holds (d 7.5)\n● every sheep moves; ■ and κ ≥ {KAPPA_STAR:g}")
    ax.legend(frameon=False, fontsize=8)
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "shape_drift.png", dpi=130)


def main():
    tic = time.time()
    recs = run()
    err = max(float(r["identity_err"].max()) for r in recs)
    summ = summarize(recs)
    for s in summ:
        f = s["fits_dM2"]
        print(f"{s['kind']:6s} att {s['w_att']} ali {s['w_ali']} d {s['l_drive']} r_dog {s['r_dog']} n_att {s['n_att']} noise {s['noise']}: "
              f"G(1x) {s['G_at_1']:.3f} vs v²(f_m - f_m²P²) {s['dispersion_bound_at_1']:.3f} | "
              + (f"fit sqrt: a {f['sqrt']['a']:.3f} b {f['sqrt']['b']:.3f} rms {f['sqrt']['rms']:.3f}; linear: a {f['linear']['a']:.3f} b {f['linear']['b']:.3f} rms {f['linear']['rms']:.3f}" if f else "no all-moving states"))
        for b in s["by_dilation"]:
            print(f"     x{b['dilation']}: M2 {b['M2_median']:.2f} E[dM2] {b['dM2_median']:+.3f} (>0 in {b['dM2_positive_frac']:.2f}) | C_par {b['C_rel_par_median']:+.3f} "
                  f"C_perp {b['C_rel_perp_median']:+.3f} C_frz {b['C_frz_median']:+.3f} G {b['G_median']:.3f} | all move {b['all_move_frac']:.2f} region {b['region_frac']:.2f} "
                  f"P {b['P_mu_median']:.2f} outer inward {b['frac_out_inward_median']:.2f} gamma {b['gamma_median']:.2f}")
        print(f"     failures (E[dM2] > 2 se): {s['failures']}")
    out = dict(setup=dict(N=N, K=K, reps=REPS, snapshots=SNAPS, dilations=DIL, v=V, identity_max_abs_error=err,
                          rules=dict(driven="published (stop when no dog within r_dog)", free="null model (modified rule)")),
               summary=summ, seconds=time.time() - tic)
    (OUT / "shape_drift.json").write_text(json.dumps(out, indent=1))
    figure(recs, summ)
    print(f"identity max abs error {err:.2e}; done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
