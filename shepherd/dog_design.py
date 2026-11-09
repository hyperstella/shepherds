"""Where should one dog stand? Di Lorenzo's deconvolution applied to a driven flock.

Two readings of the same design problem, both with the project's measured kernels
(dog push f(r) = a e^(-r/L), pair kernel g, diffusion D) and the Gaussian goal density
n̄ of the feasibility analysis (95% of the sheep within r_goal):

- First-order, Di Lorenzo's own: the sheep velocity is the sum of the dogs' pushes plus
  cohesion and diffusion, so a flock driven at speed c needs the dogs to supply
  u* = c ê + D ∇ln n̄ − g * n̄ wherever there are sheep. Non-negative least squares over
  candidate dog positions around the flock (`translating_arc`) gives a one-sided arc
  behind it: its mean distance is the standoff, its lateral spread the sweep, its mass
  the number of dogs.

- Direction-matched, the kinetic reading: the Jadhav sheep take the dog's direction, not
  its push. A dog position is scored by how well the unit vector away from it matches
  the direction of u* over the flock (`direction_match`), subject to every sheep seeing
  it (within r_dog). A single dog on the axis does best as far back as coverage allows;
  a lateral position helps one side and hurts the other.
"""

import numpy as np
from scipy.optimize import nnls


def goal_sigma(r_goal, coverage=0.95):
    return r_goal / np.sqrt(-2 * np.log(1 - coverage))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def flock_grid(n_sheep, r_goal, step=0.25):
    """Evaluation points over the goal density: positions, weights n̄/max, densities, s, step."""
    s = goal_sigma(r_goal)
    R = 2.0 * r_goal
    step = max(step, r_goal / 12)
    a = np.arange(-R, R + 1e-9, step)
    X, Y = np.meshgrid(a, a, indexing="ij")
    keep = np.hypot(X, Y) <= R
    P = np.stack([X[keep], Y[keep]], -1)
    w = np.exp(-(P ** 2).sum(1) / (2 * s ** 2))
    return P, w, n_sheep * w / (2 * np.pi * s ** 2), s, step


def required_drift(P, n_star, s, D, c, g, step):
    """u*(x) = c ê_x + D ∇ln n̄ − (g * n̄)(x) on the evaluation points; the flock moves along +x."""
    v = -D * P / s ** 2
    v[:, 0] += c
    if g is not None:
        d = P[None] - P[:, None]  # [i, j] = p_j - p_i
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        v = v - ((g(dist) / dist * n_star[None] * step ** 2)[..., None] * d).sum(1)
    return v


def covered(C, r_goal, r_dog, hover=0.5):
    """Candidate dog positions from which every sheep of the 95% disc is within r_dog (after a hover margin)."""
    edge = np.array([[r_goal, 0.0], [-r_goal, 0.0], [0.0, r_goal], [0.0, -r_goal]])
    return (np.linalg.norm(C[:, None] - edge[None], axis=-1) <= r_dog - hover).all(1)


def candidate_dogs(r_goal, r_dog=None, ring_step=0.5, n_angles=48, ring_max=None):
    radii = np.arange(r_goal + ring_step, (ring_max or r_goal + 12.0) + 1e-9, ring_step)
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    C = np.stack([np.outer(radii, np.cos(th)).ravel(), np.outer(radii, np.sin(th)).ravel()], -1)
    if r_dog is not None:  # keep only positions every sheep can see from; if none exists, use them all
        keep = covered(C, r_goal, r_dog)
        if keep.any():
            return C[keep]
    return C


def translating_arc(f, g, D, n_sheep, r_goal, c, r_dog=None):
    """First-order design: non-negative dog counts on candidates so that the summed push matches u* on the flock."""
    P, w, n_star, s, step = flock_grid(n_sheep, r_goal)
    v = required_drift(P, n_star, s, D, c, g, step)
    C = candidate_dogs(r_goal, r_dog)
    rd = P[:, None] - C[None]  # dog -> evaluation point
    dd = np.maximum(np.linalg.norm(rd, axis=-1), 1e-6)
    A = (f(dd) / dd)[..., None] * rd
    sw = np.sqrt(w)[:, None]
    Amat = np.concatenate([A[..., 0] * sw, A[..., 1] * sw])
    b = np.concatenate([v[:, 0] * sw[:, 0], v[:, 1] * sw[:, 0]])
    m, _ = nnls(Amat, b, maxiter=20 * Amat.shape[1])
    mass = float(m.sum())
    mw = m / mass if mass > 0 else np.full_like(m, 1 / len(m))
    angle = np.arctan2(C[:, 1], -C[:, 0])  # 0 = straight behind
    return dict(
        positions=C, counts=m, mass=mass, residual=float(np.linalg.norm(Amat @ m - b) / np.linalg.norm(b)),
        standoff=float(-(mw * C[:, 0]).sum()), lateral_sd=float(np.sqrt((mw * C[:, 1] ** 2).sum())),
        radius=float((mw * np.hypot(C[:, 0], C[:, 1])).sum()), angle_sd=float(np.sqrt((mw * angle ** 2).sum())),
        behind_mass_frac=float(mw[C[:, 0] < 0].sum()), s=float(s), c=float(c), D=float(D),
    )


def single_dog_first_order(f, g, D, n_sheep, r_goal, c, r_dog=None):
    """First-order design for one real dog of amplitude a: the candidate position whose push best matches u*."""
    P, w, n_star, s, step = flock_grid(n_sheep, r_goal)
    v = required_drift(P, n_star, s, D, c, g, step)
    C = candidate_dogs(r_goal, r_dog, n_angles=72)
    rd = P[:, None] - C[None]
    dd = np.maximum(np.linalg.norm(rd, axis=-1), 1e-6)
    push = (f(dd) / dd)[..., None] * rd  # (n_eval, n_cand, 2)
    err = (w[:, None] * ((push - v[:, None]) ** 2).sum(-1)).sum(0) / (w * (v ** 2).sum(-1)).sum()
    k = int(np.argmin(err))
    return dict(positions=C, error=err, best=C[k], standoff=float(-C[k, 0]), lateral=float(C[k, 1]), error_best=float(err[k]),
                error_none=1.0, s=float(s))


def direction_mismatch(P, w, e_req, dogs):
    """n̄-weighted mean |ê_away − ê_req|² for each dog position in `dogs` (K, 2)."""
    e_away = unit(P[None] - dogs[:, None])  # (K, n_eval, 2)
    return (w[None] * ((e_away - e_req[None]) ** 2).sum(-1)).sum(1) / w.sum()


def direction_match(n_sheep, r_goal, c, D, g, r_dog, hover=0.5, d_step=0.25, d_extra=12.0):
    """Kinetic design: the best single position on the axis behind the flock, within coverage."""
    P, w, n_star, s, step = flock_grid(n_sheep, r_goal)
    e_req = unit(required_drift(P, n_star, s, D, c, g, step))
    d_cov = r_dog - hover - r_goal  # on the axis, the front sheep is the first to lose sight of the dog
    d_grid = np.arange(0.5, max(d_cov, 0.5) + d_extra + 1e-9, d_step)
    J = direction_mismatch(P, w, e_req, np.stack([-d_grid, np.zeros_like(d_grid)], -1))
    within = d_grid <= d_cov
    best = float(d_grid[within][np.argmin(J[within])]) if within.any() else float(d_grid[0])
    y_grid = np.linspace(-best, best, 17)
    J_lat = direction_mismatch(P, w, e_req, np.stack([np.full_like(y_grid, -best), y_grid], -1))
    return dict(
        d_grid=d_grid, J=J, d_cov=float(d_cov), standoff=best, J_best=float(J[within][np.argmin(J[within])]) if within.any() else float(J[0]),
        d_unconstrained=float(d_grid[np.argmin(J)]), J_unconstrained=float(J.min()), y_grid=y_grid, J_lat=J_lat, s=float(s),
    )
