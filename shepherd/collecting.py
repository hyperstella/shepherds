"""The published dog's collecting phase: its decision rule, and an exact bound on a two-phase collecting potential.

Decision (shepherd/model.py, read from the state before the step): slow if the dog is within r_rep of a sheep,
else collect if the farthest sheep is at least l_sep from the centroid, else drive. Collecting aims at
A = x_k + l_collect e_k, beyond the farthest sheep k on the ray from the centroid (e_k = xi_k / |xi_k|).

Potential, for the current target k:
    Psi_k(x) = |xi_k| + lam (|D - x_k| - d0)_+,
the target's distance from the centroid plus the dog's distance from it beyond d0. Two bounds give, at every state,
an upper bound U(x) >= E[Psi_k(x') - Psi_k(x) | x] with the same k on both sides (no Monte Carlo):

 1. The sheep (exact first moment, bounded second moment). With d_j = v h_j' for a sheep that moves and 0 otherwise,
    Delta = xi_k' - xi_k = sum_j w_j d_j, w_k = v(1 - 1/N), w_j = -v/N. Since |xi + Delta| <= |xi| + e.Delta +
    |Delta_perp|^2 / (2(|xi| - |Delta|max)),
        E|xi_k'| - |xi_k| <= E[e.Delta] + (E|Delta|^2 - (E e.Delta)^2) / (2(|xi_k| - |Delta|max)),
    with E Delta = sum_j w_j mu_j and E|Delta|^2 = |E Delta|^2 + sum_j w_j^2 ([moves_j] - |mu_j|^2), both exact
    (shepherd.expected_heading).
 2. The dog. (a) Every draw: its heading unit(u + noise e(psi)) lies within arcsin(noise) of u, the direction to
    A, and u lies within theta of the direction to x_k. So with phi = theta + arcsin(noise),
        |D' - x_k'| <= sqrt(d^2 - 2 v_dog d cos(phi) + v_dog^2) + v [k moves],
    and in the slow mode |D' - x_k'| <= d + dog_slow_step + v [k moves].
    (b) In expectation: the noise average of the dog's heading is m(noise) u (the same closed form as the sheep's),
    and the dog's step is independent of the sheep's, so E|D' - x_k'|^2 is exact. If the distance cannot fall below
    d0 within the step (d - dog step - v >= d0), then E(|D' - x_k'| - d0)_+ = E|D' - x_k'| - d0 <= sqrt(E|D' - x_k'|^2)
    - d0 (Jensen). The smaller of (a) and (b) is used.

A target switch (another sheep becomes the farthest) is not covered: Psi then jumps to the new target's value, and
that jump is counted separately.
"""

import numpy as np

from .expected_heading import expected_headings, noise_shrink
from .model import Params

DRIVE_, COLLECT_, SLOW_ = 0, 1, 2


def dog_decision(pos, dog, p=Params()):
    """The dog's next decision from states pos (..., N, 2), dog (..., 2): 0 drive, 1 collect, 2 slow."""
    p = p.resolved()
    pos, dog = np.asarray(pos, float), np.asarray(dog, float)
    xi = pos - pos.mean(-2, keepdims=True)
    slow = np.linalg.norm(pos - dog[..., None, :], axis=-1).min(-1) <= p.r_rep
    collect = np.linalg.norm(xi, axis=-1).max(-1) >= p.l_sep
    return np.where(slow, SLOW_, np.where(collect, COLLECT_, DRIVE_))


def target(pos):
    """Farthest sheep from the centroid, its distance R and unit direction e, for states (S, N, 2)."""
    xi = pos - pos.mean(1, keepdims=True)
    r = np.linalg.norm(xi, axis=-1)
    k = r.argmax(1)
    s = np.arange(len(pos))
    return k, r[s, k], xi[s, k] / r[s, k, None]


def psi(pos, dog, k, d0=4.0, lam=1.0):
    """Psi_k for a given target index k (S,)."""
    s = np.arange(len(pos))
    xi = pos[s, k] - pos.mean(1)
    return np.linalg.norm(xi, axis=-1) + lam * np.clip(np.linalg.norm(dog - pos[s, k], axis=-1) - d0, 0, None)


def collecting_drift_bound(pos, hd, dog, dog_v, p=Params(), d0=4.0, lam=1.0):
    """U(x) >= E[Psi_k(x') - Psi_k(x) | x] for the current target k, at states (S, N, 2) with the dog at dog and
    heading dog_v (S, 2) (the heading matters only in the slow mode); see the module docstring.

    Returns a dict of (S,) arrays: U, its sheep part U_sheep and dog part U_dog, the target k, R = |xi_k|, the
    dog's distance d to the target, E[e.Delta] (the target's expected radial step relative to the centroid), the
    curvature term, whether the target sees the dog (awake) or moves, and whether the dog is in the slow mode."""
    pr = p.resolved()
    pos, hd, dog, dog_v = (np.asarray(a, float) for a in (pos, hd, dog, dog_v))
    S, N, _ = pos.shape
    s = np.arange(S)
    v = pr.v_sheep * pr.dt
    mu, mv = expected_headings(pos, hd, dog, p)
    mv = mv.astype(float)
    k, R, e = target(pos)
    w = np.full((S, N), -v / N)
    w[s, k] += v
    EDelta = (w[..., None] * mu).sum(1)
    Ee = (EDelta * e).sum(-1)
    E2 = (EDelta ** 2).sum(-1) + (w ** 2 * (mv - (mu ** 2).sum(-1))).sum(1)
    dmax = (np.abs(w) * mv).sum(1)
    curv = np.maximum(E2 - Ee ** 2, 0.0) / (2 * (R - dmax))
    U_sheep = Ee + curv
    x = pos[s, k]
    A = x + pr.l_collect * e
    d = np.linalg.norm(dog - x, axis=-1)
    ua = (A - dog) / np.linalg.norm(A - dog, axis=-1, keepdims=True)
    us = (x - dog) / d[:, None]
    theta = np.arccos(np.clip((ua * us).sum(-1), -1.0, 1.0))
    noise_dog = pr.noise if pr.noise_dog is None else pr.noise_dog
    phi = np.minimum(theta + np.arcsin(min(noise_dog, 1.0)), np.pi)
    vd = pr.v_dog * pr.dt
    slow = np.linalg.norm(pos - dog[:, None], axis=-1).min(1) <= pr.r_rep
    reach = np.sqrt(np.maximum(d ** 2 - 2 * vd * d * np.cos(phi) + vd ** 2, 0.0))
    d_next = np.where(slow, d + pr.dog_slow_step * pr.dt, reach) + v * mv[s, k]
    U_draw = np.clip(d_next - d0, 0, None) - np.clip(d - d0, 0, None)
    # (b) exact second moment of |D' - x_k'|, then Jensen, where the distance stays above d0 surely
    D_slow = dog + pr.dog_slow_step * pr.dt * dog_v
    ED = np.where(slow[:, None], D_slow, dog + vd * noise_shrink(noise_dog) * ua)
    ED2 = np.where(slow, ((D_slow - x) ** 2).sum(-1), d ** 2 + vd ** 2 + 2 * vd * ((dog - x) * (ED - dog)).sum(-1) / vd)
    E2_dog = ED2 - 2 * v * ((ED - x) * mu[s, k]).sum(-1) + v ** 2 * mv[s, k]
    step_d = np.where(slow, pr.dog_slow_step * pr.dt, vd)
    clear = d - step_d - v * mv[s, k] >= d0
    U_mom = np.where(clear, np.sqrt(np.maximum(E2_dog, 0.0)) - d, np.inf)
    U_dog = np.minimum(U_draw, U_mom)
    return dict(U=U_sheep + lam * U_dog, U_sheep=U_sheep, U_dog=U_dog, target=k, R=R, d=d, Ee=Ee, curv=curv,
                awake=d <= pr.r_dog, moves=mv[s, k] > 0, slow=slow)
