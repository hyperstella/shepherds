"""Exact one-step expected headings of the Jadhav/Strömbom sheep (no Monte Carlo).

Given the state (positions, headings, dog), a sheep that sees the dog takes the heading
    h' = unit(B + w_att a(J) + w_ali h_j + noise e(psi)),   B = w_inertia h + w_rep rep + w_dog away,
where J is a uniformly random n_att-subset of its K nearest neighbours, a(J) = unit(sum_{k in J}
unit(x_k - x)), j is uniform in J, and psi is uniform on the circle. The three draws are independent
of each other and across sheep. Averaging over psi only shrinks the unit vector radially:
    E_psi[unit(c + r e(psi))] = m(r / |c|) unit(c),   m(q) = E[(1 + q cos psi) / |1 + q e(psi)|],
so the conditional mean mu = E[h' | state] is an exact finite sum over the C(K, n_att) n_att
subset/copy choices, with m in closed form (complete elliptic integrals). A sheep that does not see the
dog but is repelled turns deterministically to unit(w_inertia h + w_rep rep) (published rule);
any other sheep stands still. Only the published rule (dog present, not social_always) is covered.
"""

import itertools

import numpy as np
from scipy.special import ellipe, ellipkm1

from .model import Params

def noise_shrink(q):
    """m(q) = E_psi[cos of the angle between c + r e(psi) and c], q = r / |c|.

    Closed form: m = [(1 + q) E(k) + (1 - q) K(k)] / pi with k = 2 sqrt(q) / (1 + q), from
    1 + q cos psi = (|1 + q e(psi)|^2 + 1 - q^2) / 2 and the standard integrals of |1 + q e(psi)|^{+-1}.
    K is evaluated from the complementary parameter 1 - k^2 = ((1 - q)/(1 + q))^2 (ellipkm1), which
    keeps the logarithmic singularity at q = 1 accurate. m(0) = 1, m(1) = 2/pi, m(q) ~ 1/(2q) for
    large q, and m decreases in q."""
    q = np.asarray(q, float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        p = ((1 - q) / (1 + q)) ** 2
        lo = np.where(q == 1.0, 0.0, (1 - q) * ellipkm1(p))
        out = ((1 + q) * ellipe(1 - p) + lo) / np.pi
    return np.where(np.isinf(q), 0.0, out)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def drives(pos, hd, dog, p=Params()):
    """The ingredients of the heading rule for states pos, hd (S, N, 2), dog (S, 2).

    Returns a dict with B (S, N, 2), att (S, N, n_sub, 2) = a(J) for every subset J, ali (S, N, n_sub,
    n_att, 2) = h_j for every j in J, sees (S, N), rep_only (S, N) (repelled but not seeing the dog),
    graze (S, N, 2) = their deterministic heading, and knn (S, N, K) neighbour indices."""
    pos, hd, dog = (np.asarray(a, float) for a in (pos, hd, dog))
    S, N, _ = pos.shape
    K = p.k_perceive
    r_ij = pos[:, None, :, :] - pos[:, :, None, :]  # [s, i, j] = x_j - x_i
    dist = np.linalg.norm(r_ij, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_ij = r_ij / dist[..., None]
    near = (dist < p.r_rep) & ~np.eye(N, dtype=bool)
    has_rep = near.any(-1)
    rep = np.where(has_rep[..., None], -_unit(np.where(near[..., None], u_ij, 0.0).sum(2)), 0.0)
    to_dog = dog[:, None, :] - pos
    dd = np.linalg.norm(to_dog, axis=-1)
    sees = dd <= p.r_dog
    away = np.where(sees[..., None], -to_dog / dd[..., None], 0.0)
    knn = np.argsort(dist, axis=-1, kind="stable")[..., 1:K + 1]
    subs = np.array(list(itertools.combinations(range(K), p.n_att)))  # (n_sub, n_att) positions in knn
    nb = np.take_along_axis(knn, subs.reshape(1, 1, -1).repeat(S, 0).repeat(N, 1), -1).reshape(S, N, len(subs), p.n_att)
    ss, ii = np.arange(S)[:, None, None, None], np.arange(N)[None, :, None, None]
    d_nb = r_ij[ss, ii, nb]  # (S, N, n_sub, n_att, 2)
    att = _unit((d_nb / (np.linalg.norm(d_nb, axis=-1, keepdims=True) + np.finfo(float).eps)).sum(3))
    ali = hd[ss, nb]  # copied heading for each j in J
    B = p.w_inertia * hd + p.w_rep * rep + p.w_dog * away
    graze = _unit(p.w_inertia * hd + p.w_rep * rep)
    return dict(B=B, att=att, ali=ali, sees=sees, rep_only=has_rep & ~sees, graze=graze, knn=knn, away=away, rep=rep,
                dog_dist=dd)


def realizations(dr, p=Params()):
    """c(J, j) = B + w_att a(J) + w_ali h_j for every subset/copy choice, (S, N, n_sub * n_att, 2), equally likely."""
    c = dr["B"][:, :, None, None, :] + p.w_att * dr["att"][:, :, :, None, :] + p.w_ali * dr["ali"]
    S, N = c.shape[:2]
    return c.reshape(S, N, -1, 2)


def expected_headings(pos, hd, dog, p=Params()):
    """mu (S, N, 2) = E[new heading | state] exactly, and moves (S, N) (published stop rule)."""
    dr = drives(pos, hd, dog, p)
    c = realizations(dr, p)
    nc = np.linalg.norm(c, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(nc[..., None] > 0, noise_shrink(p.noise / nc)[..., None] * c / nc[..., None], 0.0)
    mu_see = g.mean(2)
    mu = np.where(dr["sees"][..., None], mu_see, np.where(dr["rep_only"][..., None], dr["graze"], 0.0))
    return mu, dr["sees"] | dr["rep_only"]


def expected_spread_change(pos, hd, dog, p=Params()):
    """E[M2(t+1) - M2(t) | state] (S,) exactly, M2 = (1/N) sum_i |x_i - xbar|^2, published stop rule included.

    With d_i = v h_i' for a sheep that moves and d_i = 0 otherwise, M2' - M2 = (2/N) sum xi_i . d_i + (1/N) sum |d_i|^2
    - |dbar|^2. Given the state the d_i are independent with E d_i = v mu_i and E|d_i|^2 = v^2 [moves_i], so
        E[M2' - M2] = 2v (1/N) sum xi_i . mu_i + v^2 (n_moving/N - |mubar|^2 - (1/N^2) sum_i ([moves_i] - |mu_i|^2))."""
    mu, mv = expected_headings(pos, hd, dog, p)
    N, v = mu.shape[1], p.v_sheep * p.dt
    xi = np.asarray(pos, float) - np.asarray(pos, float).mean(1, keepdims=True)
    mubar = mu.mean(1)
    D = mv.mean(1) - (mubar ** 2).sum(-1) - (mv - (mu ** 2).sum(-1)).sum(1) / N ** 2
    return 2 * v * np.einsum("snc,snc->s", xi, mu) / N + v ** 2 * D
