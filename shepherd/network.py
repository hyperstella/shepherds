"""Leader-follower networks from directional cross-correlation (Nagy et al. 2010).

Port of figure_5de.m (data) and model_network_ana.m (model).
"""

import numpy as np
from scipy import stats


def xcorr_unbiased(x, y, maxlag):
    """MATLAB xcorr(x, y, maxlag, 'unbiased'): R(m) = sum_n x[n+m] y[n] / (N-|m|)."""
    n = len(x)
    lags = np.arange(-maxlag, maxlag + 1)
    r = np.empty(len(lags))
    for k, m in enumerate(lags):
        if m >= 0:
            r[k] = np.dot(x[m:], y[: n - m]) / (n - m)
        else:
            r[k] = np.dot(x[: n + m], y[-m:]) / (n + m)
    return lags, r


def directional_ccf(ux, uy, maxlag):
    """C_ij(tau) for every ordered pair; ux, uy are (n, T) unit-heading components.

    Returns lags (L,) and C (n, n, L), with C[i, j] = xcorr(u_i, u_j).
    """
    n, T = ux.shape
    lags = np.arange(-maxlag, maxlag + 1)
    C = np.zeros((n, n, len(lags)))
    for k, m in enumerate(lags):
        if m >= 0:
            a, b = slice(m, T), slice(0, T - m)
        else:
            a, b = slice(0, T + m), slice(-m, T)
        C[:, :, k] = (ux[:, a] @ ux[:, b].T + uy[:, a] @ uy[:, b].T) / (T - abs(m))
    return lags, C


def leader_follower(ux, uy, maxlag=20, cmin=0.5, use_abs=False):
    """Directed leader-follower edges.

    For each ordered pair (i, j), take the lag tau* maximizing C_ij. If
    C_ij(tau*) >= cmin and tau* < 0, i leads j by |tau*| samples. The data
    script maximizes C_ij; the model script maximizes |C_ij| (use_abs=True).

    Returns (indegree (n,), list of (leader, follower, |lag|)).
    """
    lags, C = directional_ccf(ux, uy, maxlag)
    score = np.abs(C) if use_abs else C
    best = score.argmax(axis=2)
    peak = np.take_along_axis(score, best[..., None], axis=2)[..., 0]
    tau = lags[best]
    n = ux.shape[0]
    edges = [
        (i, j, -tau[i, j])
        for i in range(n)
        for j in range(n)
        if i != j and peak[i, j] >= cmin and tau[i, j] < 0
    ]
    indeg = np.zeros(n, dtype=int)
    for leader, _, _ in edges:
        indeg[leader] += 1
    return indeg, edges


def front_back_position(pos, ux, uy, n_sheep):
    """d_i = mean_j mean_t (x_i - x_j) . v_hat_flock  (>0: ahead of the others).

    v_flock is the normalized mean of the sheep unit headings.
    pos is (n, 2, T) aligned with ux/uy (n, T).
    """
    vg = np.stack([ux[:n_sheep].mean(0), uy[:n_sheep].mean(0)])
    with np.errstate(invalid="ignore", divide="ignore"):
        vg = vg / np.linalg.norm(vg, axis=0)
    proj = (pos * vg[None]).sum(axis=1)  # (n, T): x_i . v_hat
    m = np.nanmean(proj, axis=1)  # time steps with no moving sheep have no flock direction
    dij = m[:, None] - m[None, :]
    n = pos.shape[0]
    np.fill_diagonal(dij, np.nan)
    return np.nanmean(dij, axis=1)[:n]


def indegree_vs_position(indegrees, positions, min_count=0):
    """Mean d_i (and s.e.) for each indegree value, plus correlations.

    indegrees, positions: flat arrays over (segment, individual).
    Returns dict with k, mean, se, count, pearson (r, p), spearman (rho, p).
    """
    indegrees = np.asarray(indegrees)
    positions = np.asarray(positions)
    ks, means, ses, counts = [], [], [], []
    for k in range(int(indegrees.max()) + 1):
        v = positions[indegrees == k]
        if v.size <= min_count or v.size == 0:
            continue
        ks.append(k)
        means.append(v.mean())
        ses.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
        counts.append(v.size)
    ks, means = np.array(ks), np.array(means)
    return dict(
        k=ks,
        mean=means,
        se=np.array(ses),
        count=np.array(counts),
        pearson=stats.pearsonr(ks, means),
        spearman=stats.spearmanr(ks, means),
    )
