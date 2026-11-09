"""Tests the hypotheses of the edge-exit lemma (docs/confinement_edge.md) on the follower-swap validation runs.

    .venv/bin/python experiments/confinement_edge.py   # writes results/confinement_edge.json

Configuration unchanged from follower_swap_validation.py: Jadhav sheep, 14 Di Lorenzo herders, 24 m torus, seeds
5000-5049, 600 s, late window 450-600 s. No rule or controller is changed.

Given a stored state, the law of one sheep's next step is finite apart from the noise angle: 210 attraction subsets,
4 alignment choices, and the noise angle on a 48-point midpoint rule. So E[dr | state] and E[exp(lambda dr) | state]
are computed by enumeration rather than sampled.
"""

import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from shepherd import bridge as B  # noqa: E402
import hybrid_scorecard as HS  # noqa: E402

NAME = "B. DL metres: Jadhav sheep, fixed goal"
SEEDS = list(range(5000, 5050))
T_END, T0 = 600, 450
R_MIN = 3.5  # states below this radius are not enumerated
R_AS = (4.0, 4.5, 5.0, 5.5)
R_BS = (6.0, 7.0, 8.0, 10.0)
LAMS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
SUB = np.array(list(itertools.combinations(range(10), 4)))
TH = 2 * np.pi * (np.arange(48) + 0.5) / 48
ETA = np.stack([np.cos(TH), np.sin(TH)], -1)
W_WORST = 0.5 + 2.0 + 1.5 + 1.3 + 0.3  # inertia, repulsion, attraction, alignment, noise


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def next_headings(F, D, x, h, dogs, i):
    """All equally likely next headings of sheep i, shape (210, 4, 48, 2). Mirrors simulate_bridge's heading rule."""
    r_ij = D.wrap(x - x[i])
    d = np.linalg.norm(r_ij, axis=-1)
    d[i] = np.inf
    knn = np.argsort(d, kind="stable")[:F.k_perceive]
    near = d < F.r_rep
    rep = -unit(np.where(near[:, None], r_ij / np.maximum(d, 1e-12)[:, None], 0).sum(0)) if near.any() else np.zeros(2)
    to = D.wrap(dogs - x[i])
    dd = np.linalg.norm(to, axis=-1)
    sees = dd <= F.r_dog
    if not sees.any():
        raise ValueError("sheep sees no herder; this state needs the grazing rule")
    away = (-to[sees] / dd[sees][:, None]).sum(0)
    u = r_ij[knn] / (d[knn][:, None] + B.EPS)
    s_att = unit(u[SUB].sum(1))
    base = F.w_inertia * h[i] + F.w_rep * rep + F.w_dog * away
    W = base[None, None, None] + F.w_att * s_att[:, None, None] + F.w_ali * h[knn][SUB][:, :, None] + F.e_noise * ETA[None, None]
    return unit(W), bool(near.any())


def field_margin(F, D, dogs, R):
    """Inward herder push p on the circle of radius R, and the margin of the sure one-step barrier."""
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    n = np.stack([np.cos(th), np.sin(th)], -1)
    to = D.wrap(dogs[None] - (R * n)[:, None])
    d = np.linalg.norm(to, axis=-1)
    away = np.where((d <= F.r_dog)[..., None], -to / np.maximum(d, 1e-12)[..., None], 0.0).sum(1)
    p = -(away * n).sum(1)
    margin = p - W_WORST - F.v * F.dt / (2 * R) * (np.linalg.norm(away, axis=-1) + W_WORST)
    return float(p.max()), float(p.min()), float(margin.max())


def one(seed):
    F, H, T, D, kw, _ = HS.scenarios()[NAME]
    r = B.simulate_bridge(F, H, T, D, seed=seed, n_steps=T_END, **kw)
    F = F.resolved()
    v = F.v * F.dt
    rows, exits = [], []
    for k in range(T0, T_END):
        x, h, dogs = r.pos_s[k], r.hd_s[k], r.pos_d[k]
        rad = np.linalg.norm(D.wrap(x), axis=-1)
        rad1 = np.linalg.norm(D.wrap(r.pos_s[k + 1]), axis=-1)
        for i in np.where(rad >= R_MIN)[0]:
            hn, rep = next_headings(F, D, x, h, dogs, i)
            dr = np.linalg.norm(D.wrap(x[i]) + v * hn, axis=-1) - rad[i]
            real = rad1[i] - rad[i]
            rows.append([rad[i], dr.mean(), (dr > 0).mean(), float(rep), real, dr.max(), dr.min(), *np.exp(LAMS[:, None, None, None] * dr).mean((1, 2, 3))])
            if rad1[i] > 6.0 >= rad[i]:
                exits.append(dict(seed=seed, t=k, r=float(rad[i]), r_next=float(rad1[i]), drift=float(dr.mean()), p_outward=float((dr > 0).mean()), repelled=rep))
    rad = np.linalg.norm(D.wrap(r.pos_s[T0:T_END + 1]), axis=-1)
    starts = {str(a): int(((rad[:-1] < a) & (rad[1:] >= a)).sum() + (rad[0] >= a).sum()) for a in R_AS}
    start_ok = {str(a): bool((rad[0] < a + v).all()) for a in R_AS}
    fields = np.array([[field_margin(F, D, r.pos_d[k], R) for R in (5.0, 6.0)] for k in range(T0, T_END + 1, 30)])
    return dict(rows=np.array(rows), exits=exits, starts=starts, start_ok=start_ok, fields=fields, herders_move=float(np.linalg.norm(D.wrap(r.pos_d[T0 + 1:] - r.pos_d[T0:-1]), axis=-1).max()),
                n=F.n)


def main():
    with Pool(8) as p:
        out = p.map(one, SEEDS)
    R = np.vstack([o["rows"] for o in out])
    rad, m, pp, _, real, up, lo = R.T[:7]
    mg = R[:, 7:]
    res = dict(states=len(R), check=dict(predicted_mean=float(m.mean()), realized_mean=float(real.mean()), realized_se=float(real.std() / np.sqrt(len(real))),
                                         outside_support=int(((real > up + 1e-6) | (real < lo - 1e-6)).sum())))
    bins = [(3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 7.0)]
    res["by_radius"] = [dict(lo=a, hi=b, n=int(k.sum()), mean_drift=float(m[k].mean()), max_drift=float(m[k].max()), share_nonpositive=float((m[k] <= 0).mean()), max_p_outward=float(pp[k].max()))
                        for a, b in bins if (k := (rad >= a) & (rad < b)).any()]
    N_T = out[0]["n"] * (T_END - T0)
    res["lemma"] = []
    for a in R_AS:
        k = rad >= a
        sup = mg[k].max(0)
        ok = LAMS[sup <= 1.0]
        lam = float(ok.max()) if len(ok) else None
        c = float(-m[k].max())
        lam_h = 2 * c if c > 0 else None
        starts = np.mean([o["starts"][str(a)] for o in out])
        row = dict(r_a=a, n=int(k.sum()), worst_drift=float(m[k].max()), share_nonpositive=float((m[k] <= 0).mean()), sup_mgf={str(l): float(s) for l, s in zip(LAMS, sup)}, best_lambda_on_grid=lam,
                   hoeffding_lambda=lam_h, starts_per_run=float(starts), runs_start_ok=int(sum(o["start_ok"][str(a)] for o in out)))
        row["bound"] = {str(b): (None if lam is None else min(1.0, float(starts * np.exp(-lam * max(b - a - 1.0, 0.0))))) for b in R_BS}
        row["bound_crude_NT"] = {str(b): (None if lam is None else min(1.0, float(N_T * np.exp(-lam * max(b - a - 1.0, 0.0))))) for b in R_BS}
        res["lemma"].append(row)
    ex = [e for o in out for e in o["exits"]]
    res["exits"] = ex
    F5, F6 = np.concatenate([o["fields"][:, 0] for o in out]), np.concatenate([o["fields"][:, 1] for o in out])
    res["sure_barrier"] = {"W_worst": W_WORST, "p_max_on_5m": float(F5[:, 0].max()), "p_max_on_6m": float(F6[:, 0].max()), "p_min_on_6m_lowest": float(F6[:, 1].min()), "best_margin_5m": float(F5[:, 2].max()), "best_margin_6m": float(F6[:, 2].max())}
    res["herder_step_max_late"] = float(max(o["herders_move"] for o in out))
    Path(ROOT / "results" / "confinement_edge.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "lemma"}, indent=1)[:3500])
    for r_ in res["lemma"]:
        print({k: (v if k != "sup_mgf" else {a: round(b, 3) for a, b in v.items()}) for k, v in r_.items()})


if __name__ == "__main__":
    main()
