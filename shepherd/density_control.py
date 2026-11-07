"""A Di Lorenzo-inspired density controller for one Jadhav dog.

Di Lorenzo's control has two layers: a herder density computed from the target density
(the deconvolution of the required drift), and herder dynamics that converge onto it. The
Jadhav sheep's response to a dog is not additive (a unit vector within the notice radius,
normalized together with the social terms), so there is no drift to deconvolve. This module
keeps his two layers and replaces the deconvolution by an exact computation on the sheep's
own rule:

  placement   h*(x) = argmin over candidate positions h of E[ΔV | x, h], the expected
              one-step change of a density-error functional V when the dog stands at h,
              with the expectation over the sheep's neighbour choices and noise taken
              exactly (shepherd/expected_heading.py), not by Monte Carlo;
  motion      the dog moves toward h* by at most v_dog per step.

So the placement is feedback in the sheep positions where his herder density is
feedforward in the target, and the dog is one, rate-limited, where his herders are many.

Two functionals of the sheep positions x_i, for a Gaussian target n̄ at the goal g with
scale s:

  'l2'      V = ||ρ̂_b − n̄_b||², the L² density error of his Theorem 2: ρ̂_b is the Gaussian
            kernel density estimate of bandwidth b and n̄_b the target smoothed by the same
            kernel, so V is a closed-form sum of Gaussians of pair and goal distances. E[ΔV]
            is taken to first order in the sheep's step, plus an isotropic second-order term.
  'moment'  V = (1/N) Σ |x_i − g|², the potential energy of n̄ (−2s² ln n̄ up to a constant);
            E[ΔV] = (1/N) Σ m_i [2v (x_i − g)·μ_i + v²] exactly, where μ_i is the expected
            new heading and m_i whether the sheep moves. Its minimizer does not depend on s,
            and the v² term is the price of a step: a sheep is worth moving only where its
            expected displacement toward the goal exceeds half a step.
            With a horizon H > 0 the positions are extrapolated H steps along the current
            headings first, V_H = (1/N) Σ |x_i − g + H v h_i|², which charges the flock's
            momentum (Di Lorenzo's targets have none); E[ΔV_H] is still exact, since a
            mover's new heading is a unit vector and a sheep that stands still keeps its
            heading.

Only the published rule (dog present) and the paper's modified rule (social_always) are
covered, with the sheep at fixed speed and no learned kernels.
"""

from dataclasses import dataclass, replace

import numpy as np

from .expected_heading import drives, noise_shrink, realizations
from .model import Params, simulate


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def goal_sigma(r_goal, coverage=0.95):
    return r_goal / np.sqrt(-2 * np.log(1 - coverage))


# ------------------------------------------------------------------ the functionals

@dataclass(frozen=True)
class Functional:
    """A density-error functional V(x) of the sheep positions and its first two derivatives.

    kind: 'l2' or 'moment'. goal: the target's centre. s: the target's scale (95% of the
    density within r_goal = s sqrt(-2 ln 0.05)). b: the KDE bandwidth of the L² error."""

    kind: str = "l2"
    goal: tuple = (0.0, 0.0)
    s: float = goal_sigma(5.0)
    b: float = 1.0
    horizon: float = 0.0  # 'moment' only: extrapolate positions this many steps along the headings
    v_sheep: float = 1.0  # the step length used by that extrapolation

    def _gauss(self, z, a):
        """K_a(z) = exp(−|z|²/2a²) / (2π a²), its gradient and Laplacian, for z (..., 2)."""
        r2 = (z ** 2).sum(-1)
        K = np.exp(-r2 / (2 * a ** 2)) / (2 * np.pi * a ** 2)
        return K, -(z / a ** 2) * K[..., None], (r2 / a ** 4 - 2 / a ** 2) * K

    def terms(self, pos):
        """Value (S,), gradient (S, N, 2) and Laplacian trace (S, N) of V at pos (S, N, 2)."""
        pos = np.asarray(pos, float)
        S, N, _ = pos.shape
        zg = pos - np.asarray(self.goal, float)
        if self.kind == "moment":
            return (zg ** 2).sum(-1).mean(1), 2 * zg / N, np.full((S, N), 4.0 / N)
        if self.kind != "l2":
            raise ValueError(self.kind)
        a, sig = np.sqrt(2) * self.b, np.sqrt(self.s ** 2 + 2 * self.b ** 2)
        Kp, gp, lp = self._gauss(pos[:, :, None] - pos[:, None], a)  # pair terms [s, i, j] = x_i − x_j
        off = ~np.eye(N, dtype=bool)
        Kg, gg, lg = self._gauss(zg, sig)
        const = 1 / (2 * np.pi * (2 * self.s ** 2 + 4 * self.b ** 2))  # ∫ n̄_b²
        value = Kp.sum((1, 2)) / N ** 2 - 2 * Kg.sum(1) / N + const
        grad = 2 * (gp * off[None, :, :, None]).sum(2) / N ** 2 - 2 * gg / N
        lap = 2 * (lp * off[None]).sum(2) / N ** 2 - 2 * lg / N
        return value, grad, lap

    def value(self, pos, hd=None):
        if self.kind == "moment" and self.horizon > 0:
            z = np.asarray(pos, float) - np.asarray(self.goal, float) + self.horizon * self.v_sheep * np.asarray(hd, float)
            return (z ** 2).sum(-1).mean(1)
        return self.terms(pos)[0]

    def expected_change(self, pos, mu, moves, v_sheep=1.0, terms=None, hd=None):
        """E[ΔV | x, dog] from the expected headings mu (S, N, 2) and movers (S, N).

        First order in the step plus the isotropic second-order term (v²/4) tr ∇²V per mover,
        which is exact for 'moment'. `terms` can pass a precomputed self.terms(pos). With a
        horizon the change is exact and needs the current headings hd."""
        if self.kind == "moment" and self.horizon > 0:
            H, N = self.horizon, pos.shape[1]
            zg = np.asarray(pos, float) - np.asarray(self.goal, float)
            after = (zg ** 2).sum(-1) + 2 * (1 + H) * v_sheep * (zg * mu).sum(-1) + ((1 + H) * v_sheep) ** 2
            before = ((zg + H * v_sheep * np.asarray(hd, float)) ** 2).sum(-1)
            return (moves * (after - before)).sum(1) / N
        _, grad, lap = self.terms(pos) if terms is None else terms
        return (moves * (v_sheep * (grad * mu).sum(-1) + v_sheep ** 2 / 4 * lap)).sum(1)


# ------------------------------------------------------------------ the exact sheep response, for many dog positions

class ExpectedResponse:
    """E[new heading | state, dog] exactly, for one state batch and any number of dog positions.

    The parts of the heading rule that do not involve the dog (inertia, repulsion, and every
    attraction subset and alignment copy) are computed once; each dog position only adds
    w_dog × the unit vector away from it to every realization and re-averages the noise."""

    def __init__(self, pos, hd, p=Params()):
        self.p, self.pos, self.hd = p, np.asarray(pos, float), np.asarray(hd, float)
        far = self.pos.mean(1) + np.array([1e6, 0.0])  # no sheep sees a dog this far: away = 0
        dr = drives(self.pos, hd, far, p)
        self.c0 = realizations(dr, p)  # (S, N, M, 2), the dog-free realizations
        self.c0n2 = (self.c0 ** 2).sum(-1)
        self.has_rep, self.graze = dr["rep_only"], dr["graze"]

    def __call__(self, dog):
        """mu (S, N, 2) and moves (S, N) with the dog at dog (S, 2)."""
        p = self.p
        to_dog = np.asarray(dog, float)[:, None, :] - self.pos
        dd = np.linalg.norm(to_dog, axis=-1)
        sees = dd <= p.r_dog
        away = np.where(sees[..., None], -to_dog / np.maximum(dd, 1e-12)[..., None], 0.0)
        n2 = self.c0n2 + 2 * p.w_dog * np.einsum("snmc,snc->snm", self.c0, away) + p.w_dog ** 2 * sees[..., None]
        nc = np.sqrt(np.maximum(n2, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            wgt = np.where(nc > 0, noise_shrink(p.noise / nc) / nc, 0.0)  # (S, N, M)
        M = wgt.shape[-1]
        mu_see = (np.einsum("snm,snmc->snc", wgt, self.c0) + p.w_dog * wgt.sum(-1)[..., None] * away) / M
        if p.social_always:
            return mu_see, np.ones(sees.shape, bool)
        mu = np.where(sees[..., None], mu_see, np.where(self.has_rep[..., None], self.graze, 0.0))
        return mu, sees | self.has_rep


# ------------------------------------------------------------------ the controller

@dataclass(frozen=True)
class Controller:
    """Placement over a ring of candidates around the centroid, then rate-limited motion.

    With `near` set, that functional replaces `functional` while the centroid is within
    switch_radius of its goal: a drive-then-regulate pair, for instance the second moment
    (whose basin is global) far from the goal and the L² error (whose basin is the target's
    own width) inside it."""

    functional: Functional = Functional()
    near: Functional | None = None
    switch_radius: float = 5.0
    radii: tuple = (4.0, 6.0, 8.0, 10.0, 15.0, 18.0)
    n_angles: int = 12
    tol: float = 0.02  # candidates within this fraction of the best score tie; the nearest to the dog wins

    def active(self, pos):
        """Which functional each run uses now: 0 for `functional`, 1 for `near`."""
        if self.near is None:
            return np.zeros(len(pos), int)
        rho = np.linalg.norm(pos.mean(1) - np.asarray(self.near.goal, float), axis=-1)
        return (rho < self.switch_radius).astype(int)

    def functionals(self):
        return (self.functional,) if self.near is None else (self.functional, self.near)

    def value(self, pos, hd=None, which=None):
        which = self.active(pos) if which is None else which
        vals = np.stack([f.value(pos, hd) for f in self.functionals()], -1)
        return vals[np.arange(len(pos)), which]

    def candidates(self, centroid, dog):
        """(S, K, 2): rings around the centroid plus the dog's current position."""
        th = 2 * np.pi * np.arange(self.n_angles) / self.n_angles
        ring = np.stack([np.outer(self.radii, np.cos(th)).ravel(), np.outer(self.radii, np.sin(th)).ravel()], -1)
        return np.concatenate([centroid[:, None] + ring[None], dog[:, None]], 1)

    def scores(self, resp, cand, v_sheep=1.0, which=None):
        """E[ΔV | x, h] for every candidate h, (S, K), under each run's active functional."""
        which = self.active(resp.pos) if which is None else which
        fns = self.functionals()
        terms = [f.terms(resp.pos) for f in fns]
        ar = np.arange(len(which))
        out = np.empty((len(which), cand.shape[1]))
        for k in range(cand.shape[1]):
            mu, mv = resp(cand[:, k])
            out[:, k] = np.stack([f.expected_change(resp.pos, mu, mv, v_sheep, tm, resp.hd) for f, tm in zip(fns, terms)], -1)[ar, which]
        return out

    def place(self, resp, dog, v_sheep=1.0, which=None):
        """h* (S, 2), its score (S,), and the scores of all candidates (S, K)."""
        cand = self.candidates(resp.pos.mean(1), dog)
        sc = self.scores(resp, cand, v_sheep, which)
        best = sc.min(1)
        near = sc <= best[:, None] + self.tol * np.abs(best[:, None])
        dist = np.where(near, np.linalg.norm(cand - dog[:, None], axis=-1), np.inf)
        k = dist.argmin(1)
        ar = np.arange(len(dog))
        return cand[ar, k], sc[ar, k], sc

    def move(self, dog, target, v_dog):
        step = target - dog
        n = np.linalg.norm(step, axis=-1, keepdims=True)
        return dog + step * np.minimum(1.0, v_dog / np.maximum(n, 1e-12))


def run(ctrl, params, init, n_steps, seed):
    """Close the loop: the controller places and moves the dog, `simulate` moves the sheep.

    init: dict with pos_s (R, N, 2), vel_s (R, N, 2), pos_d (R, 2). Returns positions
    pos_s (R, N, 2, T+1), pos_d (R, 2, T+1) and per-step diagnostics (R, T): V, the best
    score, the score of the position the dog actually took, its distance to h*, the fraction
    of sheep that see the dog, and whether no sheep sees it."""
    p = params.resolved()
    st = dict(pos_s=np.array(init["pos_s"], float), vel_s=np.array(init["vel_s"], float))
    dog = np.array(init["pos_d"], float)
    R, N = st["pos_s"].shape[:2]
    p1 = replace(params, n_steps=2)
    pos_s, pos_d = np.empty((R, N, 2, n_steps + 1)), np.empty((R, 2, n_steps + 1))
    pos_s[..., 0], pos_d[..., 0] = st["pos_s"], dog
    rec = {k: np.empty((R, n_steps)) for k in ("V", "score_best", "score_taken", "gap", "coverage", "released", "which")}
    ar = np.arange(R)
    for t in range(n_steps):
        resp = ExpectedResponse(st["pos_s"], st["vel_s"], p)
        which = ctrl.active(st["pos_s"])
        h_star, s_star, _ = ctrl.place(resp, dog, p.v_sheep, which)
        h_new = ctrl.move(dog, h_star, p.v_dog * p.dt)
        mu, mv = resp(h_new)
        sees = np.linalg.norm(st["pos_s"] - h_new[:, None], axis=-1) <= p.r_dog
        rec["V"][:, t], rec["which"][:, t] = ctrl.value(st["pos_s"], st["vel_s"], which), which
        rec["score_best"][:, t] = s_star
        rec["score_taken"][:, t] = np.stack([f.expected_change(st["pos_s"], mu, mv, p.v_sheep, None, st["vel_s"]) for f in ctrl.functionals()], -1)[ar, which]
        rec["gap"][:, t] = np.linalg.norm(h_new - h_star, axis=-1)
        rec["coverage"][:, t], rec["released"][:, t] = sees.mean(1), ~sees.any(1)
        res = simulate(p1, n_runs=R, seed=seed * 1000 + t, init=st, dog_path=np.stack([h_new, h_new], -1))
        st = dict(pos_s=res.pos_s[..., -1], vel_s=res.vel_s[..., -1])
        dog = h_new
        pos_s[..., t + 1], pos_d[..., t + 1] = st["pos_s"], dog
        rec["V_next"] = ctrl.value(st["pos_s"], st["vel_s"], which)  # V after the step, under the functional that chose it
        rec.setdefault("dV", np.empty((R, n_steps)))[:, t] = rec["V_next"] - rec["V"][:, t]
    return pos_s, pos_d, rec
