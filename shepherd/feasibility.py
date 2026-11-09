"""How many dogs does a real flock need? A continuification feasibility analysis.

Follows Di Lorenzo, Maffettone & di Bernardo (arXiv:2411.04791), but with
interaction kernels measured from the tracks and with cohesive targets.

First-order model (theirs, plus target-target interaction g):
    dx_i/dt = sum_dogs f(|x_i - y|) e(y -> x_i) + sum_j g(r_ij) e(x_i -> x_j) + sqrt(2D) xi
Continuum (number densities n_T, n_H in agents / m^2, periodic box):
    d_t n_T + div(n_T [f * n_H + g * n_T]) = D lap n_T
A desired steady flock density n_T* needs the dogs to supply the drift
    v*(x) = D grad ln n_T*(x) - (g * n_T*)(x)          (their Eq. 9, plus cohesion)
Deconvolving v* by f gives the dog density; shifting it to be non-negative
gives the minimum number of dogs (their Eqs. 10-12).
"""

from dataclasses import dataclass

import numpy as np

from . import data
from . import kernels as K

N = data.N_SHEEP


# ------------------------------------------------------------ measuring f, g, D

@dataclass
class FirstOrderFit:
    pair: K.HatBasis
    dog: K.HatBasis
    c_g: np.ndarray  # sheep-sheep kernel coefficients (attraction > 0)
    c_f: np.ndarray  # dog kernel coefficients (push away > 0)
    lam: float
    cv_r2: float
    D: dict  # diffusion estimates by window length (s)

    def g(self, r):
        return self.pair.curve(self.c_g, r)

    def f(self, r):
        return self.dog.curve(self.c_f, r)


def fit_first_order(drives, stride_s=0.5, window_s=2.1, lams=np.logspace(-3, 2, 11), cohesion=True):
    """Regress each sheep's (smoothed) velocity on the dog and neighbour kernels."""
    model = K.KernelModel(terms=("E", "D") if cohesion else ("D",))
    s = max(1, int(round(stride_s / data.DT)))
    normals, rows = [], []
    for d in drives:
        pos, vel = K.smooth_drive(d.pos[: N + 1], window_s=window_s)
        x, v = np.moveaxis(pos, -1, 0)[::s], np.moveaxis(vel, -1, 0)[::s]
        X = model.features(x[:, :N], v[:, :N], x[:, N])  # (T, N, 2, P)
        y = v[:, :N]
        X, y = X.reshape(-1, X.shape[-1]), y.reshape(-1)
        normals.append(K.Normal(X.T @ X, X.T @ y, float(y @ y), len(y), d.session, ""))
        rows.append((d, pos, x))
    lam, cv = K.cross_validate(model, normals, lams)
    c = K.solve(model, normals, lam)
    sl = model.slices()
    c_g = c[sl["E"]] if cohesion else np.zeros(model.pair.size)
    c_f = c[sl["D"]]
    fit = FirstOrderFit(model.pair, model.dog, c_g, c_f, lam, max(r for _, r in cv), {})
    fit.D = diffusion(drives, fit, model)
    return fit


def diffusion(drives, fit, model, windows=(1.0, 2.0, 4.0)):
    """D from the scatter of displacements around the model's drift over a window."""
    c = np.concatenate([fit.c_g, fit.c_f]) if "E" in model.terms else fit.c_f
    out = {}
    for w in windows:
        h = int(round(w / data.DT))
        res = []
        for d in drives:
            pos = d.pos[: N + 1]
            if pos.shape[-1] <= h + 1:
                continue
            sm, _ = K.smooth_drive(pos)
            x = np.moveaxis(sm, -1, 0)
            F = model.features(x[:, :N], None, x[:, N]) @ c  # (T, N, 2) drift
            drift = np.cumsum(F, axis=0) * data.DT
            disp = np.moveaxis(pos[:N], -1, 0)
            t0 = np.arange(0, x.shape[0] - h, h)
            r = (disp[t0 + h] - disp[t0]) - (drift[t0 + h] - drift[t0])
            res.append(r.reshape(-1, 2))
        r = np.concatenate(res)
        out[w] = float(np.var(r, axis=0).mean() / (2 * w))
    return out


# ------------------------------------------------------------ continuum feasibility

@dataclass
class Grid:
    L: float  # box side (m)
    n: int  # cells per side

    @property
    def dx(self):
        return self.L / self.n

    def coords(self):
        a = (np.arange(self.n) - self.n // 2) * self.dx
        return np.meshgrid(a, a, indexing="ij")

    def wrapped(self):
        """Displacements from the origin, wrapped to the box (for convolution kernels)."""
        a = np.fft.fftfreq(self.n, d=1.0 / self.n) * self.dx
        return np.meshgrid(a, a, indexing="ij")


def vector_kernel(grid, radial, outward=True):
    """Vector field k(x) = radial(|x|) x/|x| on the wrapped grid (outward from the origin)."""
    X, Y = grid.wrapped()
    R = np.hypot(X, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = np.where(R > 0, X / R, 0.0), np.where(R > 0, Y / R, 0.0)
    m = radial(R) * (1 if outward else -1)
    return m * ux, m * uy


def conv(grid, kx, ky, density):
    """(k * density)(x) for a vector kernel, as a (2, n, n) field (circular convolution)."""
    Fd = np.fft.fft2(density)
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(kx) * Fd)), np.real(np.fft.ifft2(np.fft.fft2(ky) * Fd))]) * grid.dx ** 2


def target_density(grid, n_sheep, r_goal, coverage=0.95):
    """Isotropic Gaussian flock with `coverage` of its sheep inside r_goal."""
    s = r_goal / np.sqrt(-2 * np.log(1 - coverage))
    X, Y = grid.coords()
    rho = np.exp(-(X ** 2 + Y ** 2) / (2 * s ** 2))
    return np.fft.ifftshift(n_sheep * rho / (rho.sum() * grid.dx ** 2)), s


def required_drift(grid, n_target, D, g=None):
    """v* = D grad ln n_T* - g * n_T*  (2, n, n)."""
    # grad ln n for the Gaussian, computed numerically for generality
    ln = np.log(np.maximum(n_target, 1e-300))
    gx = (np.roll(ln, -1, 0) - np.roll(ln, 1, 0)) / (2 * grid.dx)
    gy = (np.roll(ln, -1, 1) - np.roll(ln, 1, 1)) / (2 * grid.dx)
    v = D * np.stack([gx, gy])
    if g is not None:
        kx, ky = vector_kernel(grid, g, outward=False)  # attraction pulls x toward the neighbour
        v = v - conv(grid, kx, ky, n_target)
    return v


def deconvolve(grid, f, v, reg=1e-3):
    """Dog density n_H with f * n_H ~ v (Tikhonov-regularized least squares per Fourier mode)."""
    kx, ky = vector_kernel(grid, f, outward=True)  # a dog pushes sheep away from itself
    Fx, Fy = np.fft.fft2(kx) * grid.dx ** 2, np.fft.fft2(ky) * grid.dx ** 2
    Vx, Vy = np.fft.fft2(v[0]), np.fft.fft2(v[1])
    den = np.abs(Fx) ** 2 + np.abs(Fy) ** 2
    num = np.conj(Fx) * Vx + np.conj(Fy) * Vy
    H = num / (den + reg * den.max())
    H[0, 0] = 0.0  # a uniform dog density exerts no net drift
    return np.real(np.fft.ifft2(H))


def minimum_dogs(grid, f, n_target, D, g=None, reg=1e-3, mask_r=None):
    """Minimum number of dogs (their Eq. 11), with residual of the achieved drift.

    The drift only matters where sheep are; `mask_r` restricts the residual
    check to that region (default: 3x the flock's spread).
    """
    v = required_drift(grid, n_target, D, g)
    h = deconvolve(grid, f, v, reg)
    n_dog = h - h.min()  # minimal non-negative density
    total = float(n_dog.sum() * grid.dx ** 2)
    kx, ky = vector_kernel(grid, f, outward=True)
    achieved = conv(grid, kx, ky, n_dog)
    w = n_target / n_target.max()
    err = float(np.sqrt(np.sum(w * ((achieved - v) ** 2).sum(0)) / np.sum(w * (v ** 2).sum(0))))
    return {"n_dogs": total, "relative_drift_error": err, "dog_density": n_dog, "required_drift": v}


# ------------------------------------------------------------ exponential dog kernel (their form)

PAIR_KNOTS_FO = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fit_exponential(drives, L, cohesion=True, stride_s=0.5, window_s=2.1, lam=0.1):
    """Sheep velocity = a exp(-r_dog / L) (away from the dog) + sum_j g(r_ij) e_ij.

    The data cannot identify the range L (all driving happens 3-7 m from the
    dog), so L is set by the caller; a and g are fitted for that L.
    Returns (a, g coefficients, session-CV R2).
    """
    pair = K.HatBasis(PAIR_KNOTS_FO)
    mE = K.KernelModel(terms=("E",), pair=pair)
    rows = []
    s = max(1, int(round(stride_s / data.DT)))
    for d in drives:
        pos, vel = K.smooth_drive(d.pos[: N + 1], window_s=window_s)
        x, v = np.moveaxis(pos, -1, 0)[::s], np.moveaxis(vel, -1, 0)[::s]
        rd = x[:, :N] - x[:, N][:, None]
        dd = np.linalg.norm(rd, axis=-1)
        fD = (np.exp(-dd / L) / dd)[..., None, None] * rd[..., None]
        X = np.concatenate([mE.features(x[:, :N], None, x[:, N]), fD], -1) if cohesion else fD
        rows.append((d.session, X.reshape(-1, X.shape[-1]), v[:, :N].reshape(-1)))
    P = rows[0][1].shape[1]
    Dm = np.zeros((P, P))
    if cohesion:
        D2 = np.diff(np.eye(P - 1), 2, 0)
        Dm[: P - 1, : P - 1] = D2.T @ D2

    def solve(sel):
        X = np.concatenate([r[1] for r in sel]); y = np.concatenate([r[2] for r in sel])
        G = X.T @ X
        return np.linalg.solve(G + lam * np.trace(G) / P * Dm + 1e-8 * np.eye(P), X.T @ y)

    err = tot = 0.0
    for sess in sorted({r[0] for r in rows}):
        c = solve([r for r in rows if r[0] != sess])
        for r in rows:
            if r[0] == sess:
                err += np.sum((r[2] - r[1] @ c) ** 2); tot += np.sum(r[2] ** 2)
    c = solve(rows)
    g = c[:-1] if cohesion else np.zeros(pair.size)
    return float(c[-1]), g, float(1 - err / tot)


# ------------------------------------------------------------ microscopic check

def micro_check(f, g, D, n_sheep, dog_density, grid, r_goal, n_dogs, t_end=200.0, dt=0.05, runs=8, seed=0):
    """First-order agent simulation with fixed dogs sampled from the dog density.

    Returns the mean share of sheep inside r_goal over the last quarter of the run.
    """
    rng = np.random.default_rng(seed)
    L, out = grid.L, []
    p = (dog_density / dog_density.sum()).ravel()
    X, Y = grid.wrapped()
    wrap = lambda d: (d + L / 2) % L - L / 2
    for _ in range(runs):
        k = max(1, int(round(n_dogs)))
        idx = rng.choice(p.size, k, p=p)
        dogs = np.stack([X.ravel()[idx], Y.ravel()[idx]], -1) + rng.uniform(-grid.dx / 2, grid.dx / 2, (k, 2))
        x = rng.normal(0, r_goal / 2, (n_sheep, 2))
        inside = []
        steps = int(t_end / dt)
        for t in range(steps):
            rd = wrap(x[:, None] - dogs[None])  # dog -> sheep
            dd = np.linalg.norm(rd, axis=-1) + 1e-9
            v = ((f(dd) / dd)[..., None] * rd).sum(1)
            if g is not None:
                rij = wrap(x[None] - x[:, None])  # [i, j] = x_j - x_i
                dij = np.linalg.norm(rij, axis=-1)
                np.fill_diagonal(dij, np.inf)
                v = v + ((g(dij) / dij)[..., None] * rij).sum(1)
            x = wrap(x + v * dt + np.sqrt(2 * D * dt) * rng.standard_normal(x.shape))
            if t >= 3 * steps // 4:
                inside.append(np.mean(np.linalg.norm(x, axis=-1) <= r_goal))
        out.append(np.mean(inside))
    return float(np.mean(out))


# ------------------------------------------------------------ local version (no periodic-box artifact)

def minimum_dogs_local(f, g, D, n_sheep, r_goal, coverage=0.95, ring_step=0.5, n_angles=36, ring_max=None,
                       eval_step=0.25):
    """Minimum dogs as a non-negative least-squares problem on an open plane.

    Candidate dog positions sit on rings outside the goal. We look for
    non-negative dog counts m_k so that the drift sum_k m_k f(.) e(dog -> x)
    matches the required drift v*(x) = D grad ln n* - g * n* where the sheep
    are (weighted by the desired density n*). N_dogs = sum m_k.
    """
    from scipy.optimize import nnls

    s = r_goal / np.sqrt(-2 * np.log(1 - coverage))
    R = 2.0 * r_goal
    eval_step = max(eval_step, r_goal / 12)  # about 12 cells per goal radius keeps the pairwise term small
    a = np.arange(-R, R + 1e-9, eval_step)
    X, Y = np.meshgrid(a, a, indexing="ij")
    keep = np.hypot(X, Y) <= R
    P = np.stack([X[keep], Y[keep]], -1)
    w = np.exp(-(P ** 2).sum(1) / (2 * s ** 2))
    n_star = n_sheep * w / (2 * np.pi * s ** 2)  # agents / m^2
    v = -D * P / s ** 2  # D grad ln n*
    if g is not None:
        d = P[None] - P[:, None]  # [i, j] = p_j - p_i
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        v = v - ((g(dist) / dist * n_star[None] * eval_step ** 2)[..., None] * d).sum(1)
    ring_max = ring_max or (r_goal + 12.0)
    radii = np.arange(r_goal + ring_step, ring_max + 1e-9, ring_step)
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    C = np.stack([np.outer(radii, np.cos(th)).ravel(), np.outer(radii, np.sin(th)).ravel()], -1)
    rd = P[:, None] - C[None]  # dog -> eval point
    dd = np.maximum(np.linalg.norm(rd, axis=-1), 1e-6)  # a dog exactly on a point pushes nowhere (rd = 0)
    A = (f(dd) / dd)[..., None] * rd  # (n_eval, n_cand, 2)
    sw = np.sqrt(w)[:, None]
    Amat = np.concatenate([A[..., 0] * sw, A[..., 1] * sw])
    b = np.concatenate([v[:, 0] * sw[:, 0], v[:, 1] * sw[:, 0]])
    m, _ = nnls(Amat, b, maxiter=20 * Amat.shape[1])
    resid = np.linalg.norm(Amat @ m - b) / np.linalg.norm(b)
    return {"n_dogs": float(m.sum()), "relative_drift_error": float(resid), "dog_positions": C, "dog_counts": m}


def micro_check_local(f, g, D, n_sheep, r_goal, dog_positions, dog_counts, n_dogs, t_end=200.0, dt=0.05,
                      runs=4, seed=0):
    """Agent simulation on the open plane with `n_dogs` fixed dogs placed in proportion to dog_counts."""
    rng = np.random.default_rng(seed)
    p = dog_counts / dog_counts.sum()
    out = []
    for _ in range(runs):
        k = max(1, int(round(n_dogs)))
        dogs = dog_positions[rng.choice(len(p), k, p=p)]
        x = rng.normal(0, r_goal / 2.45, (n_sheep, 2))
        inside, steps = [], int(t_end / dt)
        for t in range(steps):
            rd = x[:, None] - dogs[None]
            dd = np.linalg.norm(rd, axis=-1) + 1e-9
            v = ((f(dd) / dd)[..., None] * rd).sum(1)
            if g is not None:
                rij = x[None] - x[:, None]
                dij = np.linalg.norm(rij, axis=-1)
                np.fill_diagonal(dij, np.inf)
                v = v + ((g(dij) / dij)[..., None] * rij).sum(1)
            x = x + v * dt + np.sqrt(2 * D * dt) * rng.standard_normal(x.shape)
            if t >= 3 * steps // 4:
                inside.append(np.mean(np.linalg.norm(x, axis=-1) <= r_goal))
        out.append(np.mean(inside))
    return float(np.mean(out))
