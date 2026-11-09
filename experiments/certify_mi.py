"""A rigorous certificate for the memory inequality MI+ on the baseline driving states outside the ball.

MI+ (docs/jadhav_dilorenzo_bridge.md §11.4) asks, for lambda = 0.35, that
    V(h) - lambda Q+(h) <= -k mean|xi|        for the heading fields h of the region,
where V = (1/N) sum_i xi_i . mu_i (mu exact, shepherd/expected_heading.py) and Q = (1/N) sum_i xi_i . h_i.
For any alpha in [0, lambda], -lambda Q+ <= -alpha Q, so it suffices that sup over ALL heading fields of
    F_alpha(theta) = V(theta) - alpha Q(theta)
be at most -k mean|xi|. No polarization condition is needed for that.

F_alpha is a sum of terms each depending on at most two headings. With J the random attraction subset and
k the copied neighbour,
    V = sum_i sum_{k in N(i)} t_ik(theta_i, theta_k),
    t_ik = (1/(10 N)) mean over the 84 subsets J containing k of xi_i . g(G_iJ + 0.5 h(theta_i) + 1.3 h(theta_k)),
with g(c) = m(r/|c|) unit(c) and G_iJ = 2 rep_i + away_i + 1.5 a_i(J), which depends on positions only.

Certificate:
 1. Cut each heading angle into M cells. Over a cell pair, c lies within eps = 1.8 * 2 sin(delta/4) of its
    value at the cell centres. So |c| and the angle between c and xi_i lie in known intervals, and since m
    decreases in q = r/|c|, interval arithmetic bounds xi_i . g(c) from above. Unary terms -alpha xi_j . h_j
    have exact cell maxima. Every heading field lies in some cell tuple, so the maximum of the discrete
    problem with these cell bounds is at least sup F_alpha.
 2. Bound the discrete maximum by the dual of its linear-programming relaxation, decreased by MPLP
    message passing. Any value of that dual is a valid upper bound.
 3. Take the smallest bound over alpha in a grid. A state is certified with k = -U / mean|xi| if U < 0.
Floating point: the closed form for m is accurate to 1.5e-9 (checked against 2^24-node quadrature), and
a margin of 2e-9 is added to every cell bound of m.

States not certified at M = 72 are refined to M = 144. The adversary's best heading fields give lower
bounds on the same supremum, so the gap of the certificate is measured too.

    .venv/bin/python experiments/certify_mi.py
Writes results/certify_mi.json and results/figures/certify_mi.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd.expected_heading import drives, noise_shrink  # noqa: E402
from shepherd.model import Params  # noqa: E402

OUT = ROOT / "results"
P0 = Params()
N, R_NOISE = P0.n_sheep, P0.noise
LAM = 0.35
ALPHAS = (0.20, 0.25, 0.30, 0.35)
M_LEVELS = (72, 144)
SWEEPS = 400
WORKERS = 10
M_SAFE = 2e-9
SUBS = np.array(list(itertools.combinations(range(P0.k_perceive), P0.n_att)))
# m on a fine grid in q; m decreases, so the grid value at the left end bounds m from above, the right from below
QG = np.concatenate([np.linspace(0.0, 4.0, 400_001), np.geomspace(4.0, 1e6, 100_001)[1:]])
MG = noise_shrink(QG)
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "gray": "#8a8a8a", "red": "#d63a3a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def m_upper(q):
    """An upper bound on m(q): its value at the grid point at or below q."""
    i = np.clip(np.searchsorted(QG, q, side="right") - 1, 0, len(QG) - 1)
    return MG[i] + M_SAFE


def m_lower(q):
    """A lower bound on m(q): its value at the grid point at or above q (0 beyond the grid)."""
    i = np.searchsorted(QG, q, side="left")
    return np.where(i < len(QG), MG[np.minimum(i, len(QG) - 1)] - M_SAFE, 0.0)


def cos_max(lo, hi):
    """The maximum of cos over the angle interval [lo, hi] (hi - lo < 2 pi), elementwise."""
    k = np.ceil(lo / (2 * np.pi))
    return np.where(2 * np.pi * k <= hi, 1.0, np.maximum(np.cos(lo), np.cos(hi)))


# ------------------------------------------------------------------ the certificate for one state

def edge_terms(pos, dog):
    """xi, and for each directed edge (i, k in N(i)) the drives G_iJ of the 84 subsets J that contain k."""
    dr = drives(pos[None], np.broadcast_to(np.array([1.0, 0.0]), (1, N, 2)), dog[None], P0)
    if not dr["sees"].all():
        raise ValueError("every sheep must see the dog")
    pf = (P0.w_rep * dr["rep"] + P0.w_dog * dr["away"])[0]
    att, knn = dr["att"][0], dr["knn"][0]
    xi = pos - pos.mean(0)
    edges = []
    for i in range(N):
        for p in range(P0.k_perceive):
            Js = np.flatnonzero((SUBS == p).any(1))
            edges.append((i, int(knn[i, p]), pf[i][None] + P0.w_att * att[i, Js]))
    return xi, edges


def edge_table(xi_i, G, M):
    """Cell upper bound of t_ik on an M x M grid (rows theta_i, columns theta_k)."""
    d = 2 * np.pi / M
    th = -np.pi + (np.arange(M) + 0.5) * d
    h = np.stack([np.cos(th), np.sin(th)], -1)
    c0 = G[:, None, None, :] + P0.w_inertia * h[None, :, None, :] + P0.w_ali * h[None, None, :, :]
    eps = (P0.w_inertia + P0.w_ali) * 2 * np.sin(d / 4)
    r0 = np.linalg.norm(c0, axis=-1)
    beta = np.arctan2(c0[..., 1], c0[..., 0]) - np.arctan2(xi_i[1], xi_i[0])
    spread = np.arcsin(np.minimum(1.0, eps / np.maximum(r0, 1e-300)))
    cm = cos_max(beta - spread, beta + spread)
    m_hi = m_upper(R_NOISE / (r0 + eps))
    m_lo = m_lower(R_NOISE / np.maximum(r0 - eps, 1e-300))
    nx = np.linalg.norm(xi_i)
    ub = np.where(r0 > eps, nx * cm * np.where(cm >= 0, m_hi, m_lo), nx * m_hi)
    return ub.mean(0) / (10 * N)


def pair_tables(xi, edges, M):
    pair = {}
    for i, k, G in edges:
        T = edge_table(xi[i], G, M)
        key, T = ((i, k), T) if i < k else ((k, i), T.T)
        pair[key] = pair.get(key, 0) + T
    return pair


def dual_bound(xi, pair, M, alpha):
    """MPLP on the complete graph of 14 headings; returns the smallest dual value seen (each is a valid bound)."""
    d = 2 * np.pi / M
    th = -np.pi + (np.arange(M) + 0.5) * d
    phi = np.arctan2(xi[:, 1], xi[:, 0])
    nx = np.linalg.norm(xi, axis=1)
    u = (alpha / N) * nx[:, None] * cos_max(th[None] - d / 2 - phi[:, None] - np.pi, th[None] + d / 2 - phi[:, None] - np.pi)
    edges = [(a, b, pair.get((a, b), np.zeros((M, M)))) for a in range(N) for b in range(a + 1, N)]
    la = [np.zeros(M) for _ in edges]
    lb = [np.zeros(M) for _ in edges]
    bel = [u[j].copy() for j in range(N)]

    def value():
        return sum(b.max() for b in bel) + sum((T - la[e][:, None] - lb[e][None, :]).max() for e, (_, _, T) in enumerate(edges))

    best = value()
    for sw in range(SWEEPS):
        for e, (a, b, T) in enumerate(edges):
            ba, bb = bel[a] - la[e], bel[b] - lb[e]
            la[e] = -0.5 * ba + 0.5 * (T + bb[None, :]).max(1)
            lb[e] = -0.5 * bb + 0.5 * (T + ba[:, None]).max(0)
            bel[a], bel[b] = ba + la[e], bb + lb[e]
        if sw % 25 == 24:
            v = value()
            if v > best - 1e-7 * abs(best) - 1e-12:
                best = min(best, v)
                break
            best = v
    return best


def certify_state(args):
    s, pos, dog, M = args
    tic = time.time()
    xi, edges = edge_terms(pos, dog)
    m = np.linalg.norm(xi, axis=1).mean()
    pair = pair_tables(xi, edges, M)
    bounds = {a: dual_bound(xi, pair, M, a) / m for a in ALPHAS}
    a_best = min(bounds, key=bounds.get)
    return dict(state=int(s), M=M, bound_over_m=bounds[a_best], alpha=a_best, bounds={str(k): v for k, v in bounds.items()}, seconds=time.time() - tic)


# ------------------------------------------------------------------ lower bounds: the same supremum, attacked

def attack(st, alphas, seeds=3, iters=15):
    """Best heading fields found for V - alpha Q (no polarization condition): lower bounds on the supremum."""
    import memory_inequality as mi
    out = {}
    for a in alphas:
        Lh = mi.memory("scalar", a, st["pos"])  # with Lhat = alpha I, Pi = (V - alpha Q) / (1 - alpha)
        best = None
        rng = np.random.default_rng(11)
        for s in range(seeds):
            if s == 0:
                hd = st["hd"]
            else:
                hb = st["hd"].mean(1)
                th0 = np.arctan2(hb[:, 1], hb[:, 0])
                ang = th0[:, None] + np.deg2rad(rng.uniform(30, 180)) * rng.uniform(-1, 1, size=(len(th0), N))
                hd = np.stack([np.cos(ang), np.sin(ang)], -1)
            mi.ADV_ITERS = iters
            mi.P_MIN = -1.0  # no polarization condition
            _, fb, _ = mi.adversary(dict(st, hd=hd), Lh, 7 + s)
            fb = fb * (1 - a)
            best = fb if best is None else np.maximum(best, fb)
        out[a] = best
    return out


# ------------------------------------------------------------------ main

def main():
    import memory_inequality as mi
    tic = time.time()
    st = mi.outside_ball_states()
    S = len(st["pos"])
    print(f"{S} driving states outside the ball", flush=True)
    results = {}
    todo = list(range(S))
    for M in M_LEVELS:
        with Pool(WORKERS) as pool:
            for r in pool.imap_unordered(certify_state, [(s, st["pos"][s], st["dog"][s], M) for s in todo], chunksize=1):
                results[r["state"]] = r
        todo = [s for s in todo if results[s]["bound_over_m"] >= 0]
        cert = np.array([results[s]["bound_over_m"] for s in range(S)])
        print(f"M {M}: certified {np.sum(cert < 0)} of {S}; k = -bound/mean|xi|: median {np.median(-cert):.4f}, smallest {(-cert).min():+.4f}; "
              f"alpha used {dict(zip(*np.unique([results[s]['alpha'] for s in range(S)], return_counts=True)))} ({time.time() - tic:.0f} s)", flush=True)
        if not todo:
            break
    cert = np.array([results[s]["bound_over_m"] for s in range(S)])
    # gap: attack the same objective at each state's best alpha
    lower = np.full(S, -np.inf)
    alpha_of = np.array([results[s]["alpha"] for s in range(S)])
    for a in ALPHAS:
        sel = np.flatnonzero(alpha_of == a)
        if len(sel):
            lb = attack({k: v[sel] for k, v in st.items()}, [a])[a]
            lower[sel] = lb
    gap = cert - lower
    print(f"gap between certificate and best attack: median {np.median(gap):.4f}, largest {gap.max():.4f}; attack's largest value {lower.max():+.4f}", flush=True)
    out = dict(setup=dict(lam=LAM, alphas=list(ALPHAS), M_levels=list(M_LEVELS), sweeps=SWEEPS, safety_m=M_SAFE,
                          statement="certified: sup over all heading fields of (V - alpha Q) <= bound * mean|xi|, which implies MI+ with k = -bound"),
               states=S, certified=int((cert < 0).sum()), k_certified=dict(median=fl(np.median(-cert)), p10=fl(np.percentile(-cert, 10)), min=fl((-cert).min())),
               per_state=[results[s] for s in range(S)], attack_lower=[fl(v) for v in lower], gap=dict(median=fl(np.median(gap)), max=fl(gap.max())),
               seconds=time.time() - tic)
    (OUT / "certify_mi.json").write_text(json.dumps(out, indent=1))
    figure(cert, lower, results, S)
    print(f"done in {time.time() - tic:.0f} s")


def figure(cert, lower, results, S):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    order = np.argsort(cert)
    ax.plot(np.arange(S), cert[order], color=COL["blue"], lw=2, label="certified upper bound on sup (V − αQ) / mean|ξ|")
    ax.plot(np.arange(S), lower[order], color=COL["orange"], lw=1.2, label="best attack (lower bound)")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="state (sorted by certificate)", ylabel="per mean|ξ|", title="MI⁺ certificate over all heading fields, 255 states")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    Ms = np.array([results[s]["M"] for s in range(S)])
    for M, col in zip(M_LEVELS, ("blue", "orange")):
        sel = Ms == M
        ax.hist(-cert[sel], bins=30, color=COL[col], alpha=0.8, label=f"final M = {M}: {sel.sum()} states, {np.sum(sel & (cert < 0))} certified")
    ax.axvline(0, color=COL["gray"], lw=0.8)
    ax.axvspan(ax.get_xlim()[0], 0, color=COL["gray"], alpha=0.12, lw=0)
    ax.set(xlabel="k = −bound / mean|ξ|  (shaded, k ≤ 0: not certified)", ylabel="states", title="Certified restoring constants")
    ax.legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "certify_mi.png", dpi=130)


if __name__ == "__main__":
    main()
