"""Deriving the restoring condition C4 in the driving region (baseline parameters, published rule).

C4 (docs/jadhav_dilorenzo_bridge.md §2) asks for kappa gamma >= k* > 0 outside a ball, where, with every
sheep moving,
    E[dM2 | x] = -2 v kappa gamma sqrt(M2) + G,   kappa = -(1/S) sum_i xi_i . mu_i,   S = sum_i |xi_i|.
Here mu_i = E[h_i' | x] is computed exactly (shepherd/expected_heading.py): normalization, the random
attraction subset and the random copied neighbour are kept, and there is no Monte Carlo.

In the frame (e, n) of the flock's mean expected heading, xi_i = x_i e + y_i n and, exactly,
    kappa_par  = (1/S) sum_i x_i Delta_i,   Delta_i = 1 - mu_i . e        (front sheep slower: restoring)
    kappa_perp = -(1/S) sum_i y_i mu_i . n                                (turning inward: restoring)
Normalization identity: with omega = m(noise/|c|)/|c| for each realization c of the drive before
normalization (the noise factor m is in closed form),
    mu_i = omegabar_i F_i + Cov(omega_i, c_i),   F_i = E c_i,
so kappa_perp splits exactly into one omegabar-weighted term per force plus the normalization covariance.
kappa_par does not split usefully: the common forward drive and the attraction's front-back pull nearly
cancel after normalization, and what is left is the speed-deficit asymmetry.

Parts:
 1. Driving region (dog driving, every sheep sees it): the decomposition, interpretable bounds on each
    term, and what they certify, on states of the baseline closed loop dilated about the centroid.
 2. Uniform one-step C4: headings fanned outward by phi; the critical phi of each state.
 3. Multi-step: exactly dM2 = 2 v Q_{t+1} - v^2 (1 - P_{t+1}^2), Q = (1/N) sum xi . h. If the virial
    V = (1/N) sum xi . mu satisfies V <= lambda Q - k mean|xi| (a memory inequality), then
    W = M2 + beta Q, beta = 2 v lambda / (1 - lambda), drifts down outside a ball. Checked exactly on the
    states and fans; and the m-step drift of M2 from fanned states, by Monte Carlo under the published dog.

    .venv/bin/python experiments/restoring_condition.py
Writes results/restoring_condition.json and results/figures/restoring_condition.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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

from shepherd.expected_heading import drives, noise_shrink, realizations  # noqa: E402
from shepherd.model import DRIVE, Params, simulate  # noqa: E402

OUT = ROOT / "results"
P0 = Params()  # baseline
N, R_RUNS, SEED = P0.n_sheep, 96, 5
SNAPS = list(range(100, 300, 10))
DIL = (1.0, 1.25, 1.5, 1.75, 2.0)
RHO = 4.46  # ball radius of the shape lemma at baseline (bridge §4.2): C4 is needed for sqrt(M2) >= RHO
FANS = (0, 5, 10, 15, 20, 30, 40, 60, 90)
MC_FANS, MC_STEPS, MC_K = (0, 20, 40), 8, 64
CHUNK = 150
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def q(x, p):
    return fl(np.percentile(x, p)) if len(x) else None


def rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


# ------------------------------------------------------------------ states

def closed_loop_states():
    """Snapshots of the baseline closed loop (published Strömbom dog, collecting on) far from the target."""
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=300), n_runs=R_RUNS, seed=SEED)
    idx = [(r, t) for t in SNAPS for r in range(R_RUNS)]
    pick = lambda a: np.stack([a[r, ..., t] for r, t in idx])
    return dict(pos=pick(res.pos_s), hd=pick(res.vel_s), dog=pick(res.pos_d), dog_v=pick(res.vel_d),
                mode=np.array([res.mode[r, t] for r, t in idx]))


def dilate(pos, s):
    c = pos.mean(1, keepdims=True)
    return c + s * (pos - c)


# ------------------------------------------------------------------ exact one-step analysis

def analyse(pos, hd, dog):
    """Exact kappa, its decomposition and the bounds, for states (S, N, 2). Only meaningful where every sheep sees the dog."""
    out = {}
    for i0 in range(0, len(pos), CHUNK):
        part = _analyse(pos[i0:i0 + CHUNK], hd[i0:i0 + CHUNK], dog[i0:i0 + CHUNK])
        for k, v in part.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def _analyse(pos, hd, dog):
    p = P0
    dr = drives(pos, hd, dog, p)
    c = realizations(dr, p)  # (S, N, n_real, 2), equally likely
    nc = np.linalg.norm(c, axis=-1)
    om = noise_shrink(p.noise / nc) / nc
    obar = om.mean(2)
    F = c.mean(2)
    cov = (om[..., None] * (c - F[:, :, None])).mean(2)
    mu = obar[..., None] * F + cov  # exact E[h' | x] for sheep that see the dog
    xi = pos - pos.mean(1, keepdims=True)
    r = np.linalg.norm(xi, axis=-1)
    S = r.sum(1)
    mubar = mu.mean(1)
    e = mubar / np.linalg.norm(mubar, axis=-1, keepdims=True)
    n = np.stack([-e[:, 1], e[:, 0]], -1)
    x, y = np.einsum("snc,sc->sn", xi, e), np.einsum("snc,sc->sn", xi, n)
    perp = lambda v: np.einsum("snc,sc->sn", v, n)
    Delta = 1 - np.einsum("snc,sc->sn", mu, e)
    kpar = (x * Delta).sum(1) / S
    kperp = -(y * perp(mu)).sum(1) / S
    abar = dr["att"].mean(2)
    hnb = dr["ali"].reshape(*hd.shape[:2], -1, 2).mean(2)  # the copied neighbour is uniform over the K nearest
    forces = dict(attraction=p.w_att * abar, dog=p.w_dog * dr["away"], repulsion=p.w_rep * dr["rep"],
                  own_heading=p.w_inertia * hd, neighbour_headings=p.w_ali * hnb)
    # kappa_perp = sum of these (sign: positive = restoring)
    terms = {k: -(y * obar * perp(v)).sum(1) / S for k, v in forces.items()}
    terms["normalization"] = -(y * perp(cov)).sum(1) / S
    # coarse bounds: each force term with omegabar_i replaced by its range over the flock (sign-aware)
    w_lo, w_hi = obar.min(1), obar.max(1)
    coarse = lambda v: (w_lo * np.clip(-y * perp(v), 0, None).sum(1) - w_hi * np.clip(y * perp(v), 0, None).sum(1)) / S
    A_lb = coarse(forces["attraction"])
    D_ub = -coarse(forces["dog"])
    R_ub = -coarse(forces["repulsion"])
    H_lb = coarse(forces["own_heading"]) + coarse(forces["neighbour_headings"])
    rel_d = dog - pos.mean(1)
    y_d = np.einsum("sc,sc->s", rel_d, n)
    x_d = np.einsum("sc,sc->s", rel_d, e)
    d_min = dr["dog_dist"].min(1)
    crowded = np.linalg.norm(dr["rep"], axis=-1) > 0
    sd_om = om.std(2)
    sd_cperp = np.sqrt(((np.einsum("snkc,sc->snk", c, n) - perp(F)[..., None]) ** 2).mean(2))
    E_cs = (np.abs(y) * sd_om * sd_cperp).sum(1) / S
    H = terms["own_heading"] + terms["neighbour_headings"]  # the heading term (restoring if positive)
    forces_sum = sum(terms[k] for k in forces)
    lb_exact = forces_sum - E_cs + np.minimum(kpar, 0)  # exact force terms, normalization by Cauchy-Schwarz
    lb_coarse = A_lb - D_ub - R_ub + H_lb - E_cs + np.minimum(kpar, 0)
    # the naive reading: attraction's front-back pull before normalization, with one common normalizer
    naive_par = -(x * p.w_att * np.einsum("snc,sc->sn", abar, e)).sum(1) / (S * np.linalg.norm(F, axis=-1).mean(1))
    # multi-step quantities
    V = np.einsum("snc,snc->sn", xi, mu).mean(1)
    Q = np.einsum("snc,snc->sn", xi, hd).mean(1)
    EP2 = (mubar ** 2).sum(-1) + (1 - (mu ** 2).sum(-1)).sum(1) / N ** 2
    out = dict(M2=(r ** 2).mean(1), S=S, m=r.mean(1), gamma=r.mean(1) / np.sqrt((r ** 2).mean(1)), sees_all=dr["sees"].all(1),
               kappa=kpar + kperp, kpar=kpar, kperp=kperp, naive_par=naive_par, A_lb=A_lb, D_ub=D_ub,
               R_ub=R_ub, H_lb=H_lb, E_cs=E_cs, H=H, lb_exact=lb_exact, lb_coarse=lb_coarse, crowded_frac=crowded.mean(1), d_min=d_min, y_d=y_d, x_d=x_d, w_lo=w_lo, w_hi=w_hi,
               fan_obs=np.degrees(np.arcsin(np.clip((y * perp(hd)).sum(1) / np.abs(y).sum(1), -1, 1))), V=V, Q=Q, EP2=EP2)
    out.update({f"term_{k}": v for k, v in terms.items()})
    return out


def fanned(st, phi_deg, which="all"):
    """Headings turned outward (away from the flock axis) by phi; which: all, inner (inner two thirds by |y|), or alternate."""
    hd = st["hd"]
    hbar = hd.mean(1)
    e = hbar / np.linalg.norm(hbar, axis=-1, keepdims=True)
    n = np.stack([-e[:, 1], e[:, 0]], -1)
    y = np.einsum("snc,sc->sn", st["pos"] - st["pos"].mean(1, keepdims=True), n)
    w = np.sign(y)
    if which == "inner":
        w = w * (np.abs(y) < np.percentile(np.abs(y), 67, axis=1, keepdims=True))
    elif which == "alternate":
        alt = np.zeros_like(y, dtype=bool)
        np.put_along_axis(alt, np.argsort(y, axis=1)[:, ::2], True, axis=1)
        w = w * alt
    return rot(hd, np.deg2rad(phi_deg) * w)


# ------------------------------------------------------------------ multi-step Monte Carlo

def m_step_drift(st, hd, seed):
    """E[M2(t+s)] - M2(t), s = 1..MC_STEPS, under the published dog (rules on), MC_K replicates per state."""
    S = len(st["pos"])
    rep = lambda a: np.repeat(a, MC_K, 0)
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=MC_STEPS + 1), n_runs=S * MC_K, seed=seed,
                   init=dict(pos_s=rep(st["pos"]), vel_s=rep(hd), pos_d=rep(st["dog"]), vel_d=rep(st["dog_v"])))
    pos = res.pos_s.reshape(S, MC_K, N, 2, -1)
    M2 = ((pos - pos.mean(2, keepdims=True)) ** 2).sum(3).mean(2)  # (S, K, T)
    d = M2.mean(1)[:, 1:] - M2[:, 0, :1]
    se = M2[:, :, 1:].std(1) / np.sqrt(MC_K)
    collect = (res.mode.reshape(S, MC_K, -1)[:, :, 1:] != DRIVE).mean((1, 2))
    return d, se, collect


# ------------------------------------------------------------------ main

def summarize(a, sel):
    keys = ["kappa", "kpar", "kperp", "naive_par", "A_lb", "D_ub", "R_ub", "H_lb", "E_cs", "H", "lb_exact", "lb_coarse", "crowded_frac", "d_min", "fan_obs",
            "gamma"] + [k for k in a if k.startswith("term_")]
    return {k: dict(median=q(a[k][sel], 50), p10=q(a[k][sel], 10), p90=q(a[k][sel], 90), min=fl(a[k][sel].min()) if sel.any() else None) for k in keys}


def main():
    tic = time.time()
    st0 = closed_loop_states()
    drive = st0["mode"] == DRIVE
    print(f"{len(drive)} states, {drive.mean():.2f} with the dog driving")
    by_dil, keep = {}, {}
    for s in DIL:
        st = dict(st0, pos=dilate(st0["pos"], s))
        a = analyse(st["pos"], st["hd"], st["dog"])
        sel = drive & a["sees_all"]
        out_ball = sel & (np.sqrt(a["M2"]) >= RHO)
        by_dil[s] = dict(states=int(sel.sum()), outside_ball=int(out_ball.sum()), sqrtM2_median=q(np.sqrt(a["M2"][sel]), 50),
                         frac_kappa_neg=fl((a["kappa"][sel] < 0).mean()), frac_kappa_neg_outside=fl((a["kappa"][out_ball] < 0).mean()) if out_ball.any() else None,
                         certified_exact_terms=fl((a["lb_exact"][out_ball] > 0).mean()) if out_ball.any() else None,
                         certified_coarse=fl((a["lb_coarse"][out_ball] > 0).mean()) if out_ball.any() else None,
                         certified_exact_terms_all=fl((a["lb_exact"][sel] > 0).mean()) if sel.any() else None,
                         gamma_min=fl(a["gamma"][sel].min()) if sel.any() else None, summary=summarize(a, sel),
                         summary_outside=summarize(a, out_ball) if out_ball.any() else None)
        keep[s] = (st, a, sel, out_ball)
        b = by_dil[s]
        if not sel.any():
            print(f"x{s}: no driving state keeps every sheep within r_dog of the dog")
            continue
        print(f"x{s}: driving & every sheep sees the dog {b['states']} (sqrt M2 median {b['sqrtM2_median']}), outside the ball {b['outside_ball']}; "
              f"kappa<0 {b['frac_kappa_neg']:.2f} (outside the ball {b['frac_kappa_neg_outside']}); certified outside the ball: exact force terms {b['certified_exact_terms']}, "
              f"coarse {b['certified_coarse']}; certified, all sizes {b['certified_exact_terms_all']}")
        sm = b["summary"]
        print("     " + ", ".join(f"{k.replace('term_', '')} {sm[k]['median']:+.3f}" for k in sm if k.startswith("term_")) +
              f" | kperp {sm['kperp']['median']:+.3f} kpar {sm['kpar']['median']:+.3f} (naive {sm['naive_par']['median']:+.3f}) | "
              f"E_cs {sm['E_cs']['median']:.3f} | lower bounds: exact terms {sm['lb_exact']['median']:+.3f} (p10 {sm['lb_exact']['p10']:+.3f}), "
              f"coarse {sm['lb_coarse']['median']:+.3f}")

    # the states the lemma needs: outside the ball, driving, every sheep sees the dog (pooled over dilations)
    pool = [(keep[s][0], keep[s][3]) for s in DIL if keep[s][3].any()]
    stO = {k: np.concatenate([st[k][sel] for st, sel in pool]) for k in ("pos", "hd", "dog", "dog_v")}
    print(f"outside-ball driving states pooled: {len(stO['pos'])}")

    # 2. fans
    fan_k, fan_rows = [], []
    for phi in FANS:
        a = analyse(stO["pos"], fanned(stO, phi), stO["dog"])
        ok = a["sees_all"]
        fan_k.append(np.where(ok, a["kappa"], np.nan))
        fan_rows.append(dict(phi=phi, kappa_median=q(a["kappa"][ok], 50), kappa_min=fl(a["kappa"][ok].min()), frac_neg=fl((a["kappa"][ok] < 0).mean()),
                             Q_median=q(a["Q"][ok], 50)))
        print(f"   fan {phi:2d} deg: kappa median {fan_rows[-1]['kappa_median']:+.3f} min {fan_rows[-1]['kappa_min']:+.3f}, negative {fan_rows[-1]['frac_neg']:.2f}")
    fan_k = np.array(fan_k)  # (n_phi, S)
    crit = np.full(fan_k.shape[1], np.inf)
    for j in range(fan_k.shape[1]):
        kk = fan_k[:, j]
        neg = np.flatnonzero(kk < 0)
        if len(neg):
            i = neg[0]
            crit[j] = FANS[i - 1] + (FANS[i] - FANS[i - 1]) * kk[i - 1] / (kk[i - 1] - kk[i]) if i > 0 else 0.0
    obs = np.concatenate([keep[1.0][1]["fan_obs"][keep[1.0][2]]])
    print(f"   critical fan: median {np.median(crit):.0f} deg, p10 {np.percentile(crit, 10):.0f}, min {crit.min():.1f}, never negative up to 90 deg in {np.isinf(crit).mean():.2f}; "
          f"observed fan in undisturbed driving: median {np.median(obs):+.1f} deg, p90 {np.percentile(obs, 90):+.1f}, max {obs.max():+.1f}")

    # 3. memory inequality V <= lambda Q - k m over the states and all fan families
    VQm, ratio = [], {}
    base = analyse(stO["pos"], stO["hd"], stO["dog"])
    for which, phis in (("all", FANS), ("inner", (10, 20, 40, 60)), ("alternate", (10, 20, 40, 60))):
        for phi in phis:
            a = analyse(stO["pos"], fanned(stO, phi, which), stO["dog"])
            ok = a["sees_all"]
            if phi == 20:  # memory ratio dV/dQ against the unfanned state
                rr = ((a["V"] - base["V"]) / (a["Q"] - base["Q"]))[ok & base["sees_all"]]
                ratio[which] = dict(median=q(rr, 50), p90=q(rr, 90), max=fl(rr.max()))
            VQm.append(np.stack([a["V"][ok], a["Q"][ok], a["m"][ok], a["EP2"][ok], a["gamma"][ok], np.full(ok.sum(), phi), np.full(ok.sum(), ["all", "inner", "alternate"].index(which))], 1))
    VQm = np.concatenate(VQm)
    V, Qv, m, EP2, gam = VQm[:, 0], VQm[:, 1], VQm[:, 2], VQm[:, 3], VQm[:, 4]
    lams = np.round(np.arange(0.0, 0.96, 0.05), 2)
    k_of = np.array([np.min((lam * Qv - V) / m) for lam in lams])
    D_typ, D = 1 - np.median(EP2), 1 - EP2.min()
    gmin = gam.min()
    radius = np.where(k_of > 0, P0.v_sheep * D_typ * (1 + lams) / (2 * np.maximum(k_of, 1e-12) * gmin), np.inf)
    radius_worst = np.where(k_of > 0, P0.v_sheep * D * (1 + lams) / (2 * np.maximum(k_of, 1e-12) * gmin), np.inf)
    best = int(np.argmin(radius))
    # the drift bound for W itself, state by state: -(2v/(1-l)) k m + v^2 D_s (1+l)/(1-l) < 0
    l_b, k_b = lams[best], k_of[best]
    w_neg = (-(2 * P0.v_sheep / (1 - l_b)) * k_b * m + P0.v_sheep ** 2 * (1 - EP2) * (1 + l_b) / (1 - l_b)) < 0
    print("   memory ratio dV/dQ at 20 deg: " + ", ".join(f"{k} median {v['median']:.2f} max {v['max']:.2f}" for k, v in ratio.items()))
    print(f"   memory inequality over {len(V)} states (fans of three kinds): k(lambda) = " + ", ".join(f"{l:.2f}:{k:+.3f}" for l, k in zip(lams, k_of)))
    print(f"   best lambda {l_b:.2f}: k {k_b:+.3f}; W-ball sqrt M2 about {radius[best]:.2f} m with typical 1 - E P'^2 = {D_typ:.3f}, {radius_worst[best]:.2f} m with the worst {D:.3f}; "
          f"W drift bound negative on {w_neg.mean():.2f} of the tested states; gamma min {gmin:.3f}")

    # 4. m-step drift by Monte Carlo under the published dog
    mc = {}
    for phi in MC_FANS:
        d, se, col = m_step_drift(stO, fanned(stO, phi), seed=SEED * 1000 + phi)
        sig = d > 2 * se
        mc[phi] = dict(drift_median=[fl(v) for v in np.median(d, 0)], drift_max=[fl(v) for v in d.max(0)], n_positive=[int(v) for v in (d > 0).sum(0)],
                       n_positive_2se=[int(v) for v in sig.sum(0)], n_states=int(len(d)), se_median=[fl(v) for v in np.median(se, 0)],
                       all_negative_from=int(np.argmax(np.r_[(d.max(0) < 0), True]) + 1), not_driving_frac=fl(col.mean()))
        print(f"   m-step drift, fan {phi:2d}: median " + " ".join(f"{v:+.2f}" for v in np.median(d, 0)) + " | states positive " + " ".join(str(v) for v in (d > 0).sum(0)) +
              " | positive by > 2 SE " + " ".join(str(v) for v in sig.sum(0)) + f" (of {len(d)}) | every state negative from step {mc[phi]['all_negative_from']} | dog not driving {col.mean():.3f}")

    out = dict(setup=dict(params="baseline (code values), published rule", states=f"closed loop, {R_RUNS} runs, steps {SNAPS[0]}-{SNAPS[-1]} every 10, seed {SEED}",
                          dilations=list(DIL), ball_radius=RHO, fans_deg=list(FANS), mc=dict(fans=list(MC_FANS), steps=MC_STEPS, K=MC_K)),
               by_dilation={str(k): v for k, v in by_dil.items()}, outside_ball_states=int(len(stO["pos"])),
               fans=fan_rows, critical_fan=dict(median=fl(np.median(crit)), p10=fl(np.percentile(crit, 10)), min=fl(crit.min()), never_frac=fl(np.isinf(crit).mean())),
               observed_fan=dict(median=fl(np.median(obs)), p90=fl(np.percentile(obs, 90)), max=fl(obs.max())),
               memory=dict(n=int(len(V)), lambdas=lams.tolist(), k=[fl(v) for v in k_of], W_radius=[fl(v) for v in radius], W_radius_worst=[fl(v) for v in radius_worst],
                           best_lambda=fl(l_b), best_k=fl(k_b), best_radius=fl(radius[best]), best_radius_worst=fl(radius_worst[best]), D_typical=fl(D_typ), D_max=fl(D),
                           gamma_min=fl(gmin), W_drift_negative_frac=fl(w_neg.mean()), ratio_dV_dQ_at_20deg=ratio),
               m_step=mc, seconds=time.time() - tic)
    (OUT / "restoring_condition.json").write_text(json.dumps(out, indent=1))
    figure(by_dil, fan_rows, crit, obs, lams, k_of, radius, mc)
    print(f"done in {time.time() - tic:.0f} s")


def figure(by_dil, fan_rows, crit, obs, lams, k_of, radius, mc):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axs[0, 0]
    names = [("term_attraction", "attraction"), ("term_dog", "dog"), ("term_repulsion", "repulsion"), ("term_own_heading", "own heading"),
             ("term_neighbour_headings", "neighbours' headings"), ("term_normalization", "normalization"), ("kperp", "κ⊥ (sum)"), ("kpar", "κ∥")]
    xs = np.arange(len(names))
    for j, (s, col) in enumerate(((1.0, "blue"), (1.5, "orange"))):
        sm = by_dil[s]["summary"]
        med = [sm[k]["median"] for k, _ in names]
        lo = [sm[k]["median"] - sm[k]["p10"] for k, _ in names]
        hi = [sm[k]["p90"] - sm[k]["median"] for k, _ in names]
        ax.bar(xs + (j - 0.5) * 0.38, med, width=0.36, color=COL[col], yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8), label=f"dilation ×{s:g}")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set_xticks(xs, [lab for _, lab in names], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="contribution to κ (restoring > 0)", title="Exact decomposition, driving states with every sheep seeing the dog\n(medians, 10th–90th percentiles)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 1]
    ph = [r["phi"] for r in fan_rows]
    ax.plot(ph, [r["kappa_median"] for r in fan_rows], "o-", color=COL["blue"], lw=2, label="κ, median over states")
    ax.plot(ph, [r["kappa_min"] for r in fan_rows], "v:", color=COL["red"], lw=1.2, label="κ, smallest")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax2 = ax.twinx()
    fin = np.sort(crit[np.isfinite(crit)])
    ax2.step(np.r_[0, fin, 90], np.r_[0, np.arange(1, len(fin) + 1), len(fin)] / len(crit), where="post", color=COL["green"], lw=1.5,
             label="share of states whose κ < 0 at fan ≤ φ")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("share of states", color=COL["green"])
    ax.set(xlabel="headings turned outward by φ (deg)", ylabel="κ", title=f"Uniform one-step C4 fails: outward fans (outside the ball)\n"
           f"(undisturbed driving: observed fan median {np.median(obs):+.1f}°, max {obs.max():+.1f}°)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axs[1, 0]
    ax.plot(lams, k_of, "o-", color=COL["purple"], lw=2, label="k(λ): largest k with V ≤ λQ − k·mean|ξ| on every tested state")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="memory factor λ", ylabel="k", title="Memory inequality, outside the ball, with all fan families")
    axr = ax.twinx()
    axr.plot(lams, np.where(np.isfinite(radius), radius, np.nan), "s--", color=COL["orange"], lw=1, ms=4, label="ball radius for W = M₂ + βQ (m)")
    axr.set_ylabel("√M₂ radius (m)", color=COL["orange"])
    axr.set_ylim(0, 15)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    axr.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[1, 1]
    for phi, col in zip(MC_FANS, ("blue", "orange", "red")):
        sv = np.arange(1, MC_STEPS + 1)
        ax.plot(sv, mc[phi]["drift_median"], "o-", color=COL[col], lw=2, label=f"fan {phi}°: median over states")
        ax.plot(sv, mc[phi]["drift_max"], "^:", color=COL[col], lw=1, ms=4)
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="steps s", ylabel="E[M₂(t+s)] − M₂(t)  (m²)", title="Multi-step drift from fanned states, published dog\n(▲: largest over states)")
    ax.legend(frameon=False, fontsize=8)
    for a in axs.ravel():
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "restoring_condition.png", dpi=130)


if __name__ == "__main__":
    main()
