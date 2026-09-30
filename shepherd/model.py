"""Shepherding model of Jadhav et al. (2024), after Strömbom et al. (2014).

`simulate` is a vectorized port of model/herding_model.m that runs many
independent replicates at once. `simulate_reference` is a line-by-line
translation of the MATLAB loop, kept only to test `simulate` against.
Both consume random numbers in the same order, so for the same seed they
produce identical trajectories.

Parameter defaults are those in model/simulation_hm.m (the code that
produced the paper's figures), which differ in places from the paper's
Methods text; see REPORT.md.
"""

from dataclasses import dataclass, field, replace

import numpy as np

# Dog behaviour modes recorded per step.
INIT, DRIVE, COLLECT, SLOW = 0, 1, 2, 3


@dataclass(frozen=True)
class Params:
    n_sheep: int = 14
    box_length: float = 250.0  # start distance from the target (origin)
    r_rep: float = 2.0  # sheep-sheep repulsion radius
    r_dog: float = 12.0  # radius within which sheep react to the dog
    k_perceive: int = 10  # nearest neighbours perceived (K)
    n_att: int = 4  # of which this many attract (random each step)
    n_ali: int = 1  # of which this many are copied for alignment
    v_sheep: float = 1.0
    v_dog: float = 1.5
    w_inertia: float = 0.5  # h / alpha
    w_rep: float = 2.0  # rho_a
    w_dog: float = 1.0  # rho_d
    noise: float = 0.3  # e (also used for the dog)
    w_att: float = 1.5  # c
    w_ali: float = 1.3
    dog_slow_step: float = 0.05  # dog step length when touching a sheep
    n_steps: int = 370
    l_sep: float | None = None  # default r_rep * N^(2/3)
    l_drive: float | None = None  # default r_rep * sqrt(N)
    l_collect: float | None = None  # default r_rep
    target: tuple = field(default=(0.0, 0.0))
    # Null model (paper Supp. Sec.): no dog; sheep always interact socially and
    # move at constant speed. Only supported by `simulate`.
    dog_present: bool = True

    def resolved(self):
        n = self.n_sheep
        return replace(
            self,
            l_sep=self.r_rep * n ** (2 / 3) if self.l_sep is None else self.l_sep,
            l_drive=self.r_rep * np.sqrt(n) if self.l_drive is None else self.l_drive,
            l_collect=self.r_rep if self.l_collect is None else self.l_collect,
        )


@dataclass
class Result:
    params: Params
    pos_s: np.ndarray  # (R, N, 2, T)
    vel_s: np.ndarray  # (R, N, 2, T) unit headings
    pos_d: np.ndarray  # (R, 2, T)
    vel_d: np.ndarray  # (R, 2, T)
    mode: np.ndarray  # (R, T) dog mode: INIT/DRIVE/COLLECT/SLOW

    def run(self, r):
        """(pos, vel) of run r as (N+1, 2, T) arrays with the dog last."""
        pos = np.concatenate([self.pos_s[r], self.pos_d[r][None]], axis=0)
        vel = np.concatenate([self.vel_s[r], self.vel_d[r][None]], axis=0)
        return pos, vel


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _draw_init(rng, R, p):
    return dict(
        theta_pos=rng.random(R),
        off_s=rng.random((R, p.n_sheep, 2)),
        off_d=rng.random((R, 2)),
        theta_s=rng.random((R, p.n_sheep)),
        theta_d=rng.random(R),
    )


def _draw_step(rng, R, p):
    return dict(
        att_keys=rng.random((R, p.n_sheep, p.k_perceive)),
        ali_keys=rng.random((R, p.n_sheep, p.n_att)),
        noise_s=rng.random((R, p.n_sheep)),
        noise_d=rng.random(R),
    )


def _init_state(d, p):
    side = p.box_length * np.stack([np.cos(2 * np.pi * d["theta_pos"]), np.sin(2 * np.pi * d["theta_pos"])], -1)
    side = side + np.asarray(p.target)
    pos_s = side[:, None] - 3 * p.r_rep * d["off_s"]
    pos_d = side - 3 * p.r_dog * d["off_d"]
    ts, td = 2 * np.pi * d["theta_s"], 2 * np.pi * d["theta_d"]
    vel_s = np.stack([np.cos(ts), np.sin(ts)], -1)
    vel_d = np.stack([np.cos(td), np.sin(td)], -1)
    return pos_s, vel_s, pos_d, vel_d


def _allocate(R, p):
    N, T = p.n_sheep, p.n_steps
    return (
        np.full((R, N, 2, T), np.nan),
        np.full((R, N, 2, T), np.nan),
        np.full((R, 2, T), np.nan),
        np.full((R, 2, T), np.nan),
        np.zeros((R, T), dtype=np.int8),
    )


def simulate(params=Params(), n_runs=1, seed=None):
    """Run `n_runs` independent replicates of the herding model."""
    p = params.resolved()
    rng = np.random.default_rng(seed)
    R, N = n_runs, p.n_sheep
    target = np.asarray(p.target, dtype=float)
    ar, an = np.arange(R)[:, None], np.arange(N)[None, :]

    pos_s, vel_s, pos_d, vel_d = _init_state(_draw_init(rng, R, p), p)
    P_s, V_s, P_d, V_d, mode = _allocate(R, p)
    P_s[..., 0], V_s[..., 0], P_d[..., 0], V_d[..., 0] = pos_s, vel_s, pos_d, vel_d

    for t in range(1, p.n_steps):
        d = _draw_step(rng, R, p)

        # ---- sheep (all read state at t-1) ----
        r_ij = pos_s[:, None, :, :] - pos_s[:, :, None, :]  # [r, i, j] = x_j - x_i
        dist = np.linalg.norm(r_ij, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            u_ij = r_ij / dist[..., None]
        near = (dist < p.r_rep) & ~np.eye(N, dtype=bool)
        has_rep = near.any(-1)
        with np.errstate(invalid="ignore"):
            rep = -_unit(np.where(near[..., None], u_ij, 0).sum(2))
        rep = np.where(has_rep[..., None], rep, 0.0)

        to_dog = pos_d[:, None, :] - pos_s
        dog_dist = np.linalg.norm(to_dog, axis=-1)
        sees_dog = dog_dist <= p.r_dog
        away = -to_dog / dog_dist[..., None]
        if not p.dog_present:
            sees_dog = np.ones_like(sees_dog)
            away = np.zeros_like(away)

        knn = np.argsort(dist, axis=-1, kind="stable")[..., 1 : p.k_perceive + 1]
        att = np.take_along_axis(knn, np.argsort(d["att_keys"], -1)[..., : p.n_att], -1)
        ali = np.take_along_axis(att, np.argsort(d["ali_keys"], -1)[..., : p.n_ali], -1)
        r_att = r_ij[ar[..., None], an[..., None], att]
        s_att = _unit((r_att / (np.linalg.norm(r_att, axis=-1, keepdims=True) + np.finfo(float).eps)).sum(2))
        s_ali = _unit(vel_s[ar[..., None], ali].sum(2))
        th = 2 * np.pi * d["noise_s"]
        noise = np.stack([np.cos(th), np.sin(th)], -1)

        active = (
            p.w_inertia * vel_s + p.w_rep * rep + p.w_dog * away
            + p.w_att * s_att + p.noise * noise + p.w_ali * s_ali
        )
        grazing = p.w_inertia * vel_s + p.w_rep * rep
        with np.errstate(invalid="ignore"):
            new_v = np.where(sees_dog[..., None], _unit(active), _unit(grazing))
        moves = sees_dog | has_rep
        new_v = np.where(moves[..., None], new_v, vel_s)
        new_pos_s = pos_s + np.where(moves[..., None], p.v_sheep * new_v, 0.0)

        # ---- dog (reads sheep state at t-1) ----
        min_d = np.linalg.norm(pos_s - pos_d[:, None], axis=-1).min(-1)
        slow = min_d <= p.r_rep
        G = pos_s.mean(1)
        from_g = pos_s - G[:, None]
        spread = np.linalg.norm(from_g, axis=-1)
        far = spread.argmax(-1)
        collect = ~slow & (spread.max(-1) >= p.l_sep)
        drive = ~slow & ~collect

        far_vec = from_g[np.arange(R), far]
        far_dist = spread[np.arange(R), far]
        p_collect = G + (far_dist + p.l_collect)[:, None] * far_vec / far_dist[:, None]
        away_t = G - target
        g_norm = np.linalg.norm(away_t, axis=-1, keepdims=True)
        p_drive = target + (g_norm + p.l_drive) * away_t / g_norm
        aim = np.where(collect[:, None], p_collect, p_drive)
        th_d = 2 * np.pi * d["noise_d"]
        dog_dir = _unit(_unit(aim - pos_d) + p.noise * np.stack([np.cos(th_d), np.sin(th_d)], -1))

        new_pos_d = np.where(slow[:, None], pos_d + p.dog_slow_step * vel_d, pos_d + p.v_dog * dog_dir)
        vel_d = np.where(slow[:, None], vel_d, dog_dir)

        pos_s, vel_s, pos_d = new_pos_s, new_v, new_pos_d
        P_s[..., t], V_s[..., t], P_d[..., t], V_d[..., t] = pos_s, vel_s, pos_d, vel_d
        mode[:, t] = np.select([slow, collect, drive], [SLOW, COLLECT, DRIVE])

    return Result(p, P_s, V_s, P_d, V_d, mode)


def simulate_reference(params=Params(), n_runs=1, seed=None):
    """Literal translation of herding_model.m (slow; for testing only)."""
    p = params.resolved()
    rng = np.random.default_rng(seed)
    R, N = n_runs, p.n_sheep
    target = np.asarray(p.target, dtype=float)
    eps = np.finfo(float).eps
    norm = np.linalg.norm

    pos_s0, vel_s0, pos_d0, vel_d0 = _init_state(_draw_init(rng, R, p), p)
    P_s, V_s, P_d, V_d, mode = _allocate(R, p)
    P_s[..., 0], V_s[..., 0], P_d[..., 0], V_d[..., 0] = pos_s0, vel_s0, pos_d0, vel_d0
    pos_s1, vel_s1, pos_d1, vel_d1 = pos_s0.copy(), vel_s0.copy(), pos_d0.copy(), vel_d0.copy()

    for t in range(1, p.n_steps):
        d = _draw_step(rng, R, p)
        for r in range(R):
            pos_s = pos_s1[r].copy()
            for i in range(N):
                r_sd = pos_d1[r] - pos_s1[r, i]
                dist_sd = norm(r_sd)
                r_sd = r_sd / dist_sd
                r_ij = pos_s1[r] - pos_s1[r, i]
                mag = norm(r_ij, axis=1)
                rep_j = [j for j in np.flatnonzero(mag < p.r_rep) if j != i]
                if rep_j:
                    rep = -(r_ij[rep_j] / mag[rep_j, None]).sum(0)
                    rep = rep / norm(rep)
                if dist_sd > p.r_dog:
                    if rep_j:
                        v = p.w_inertia * vel_s1[r, i] + p.w_rep * rep
                        v = v / norm(v)
                        pos_s[i] = pos_s[i] + p.v_sheep * v
                        V_s[r, i, :, t] = v
                    else:
                        V_s[r, i, :, t] = vel_s1[r, i]
                else:
                    knn = np.argsort(mag, kind="stable")[1 : p.k_perceive + 1]
                    lcm = knn[np.argsort(d["att_keys"][r, i])[: p.n_att]]
                    a = (r_ij[lcm] / (mag[lcm, None] + eps)).sum(0)
                    a = a / norm(a)
                    l_alg = lcm[np.argsort(d["ali_keys"][r, i])[: p.n_ali]]
                    al = vel_s1[r, l_alg].sum(0)
                    al = al / norm(al)
                    th = 2 * np.pi * d["noise_s"][r, i]
                    err = np.array([np.cos(th), np.sin(th)])
                    v = p.w_inertia * vel_s1[r, i] + p.w_dog * (-r_sd) + p.w_att * a + p.noise * err + p.w_ali * al
                    if rep_j:
                        v = v + p.w_rep * rep
                    v = v / norm(v)
                    pos_s[i] = pos_s[i] + p.v_sheep * v
                    V_s[r, i, :, t] = v

            dist_ds = norm(pos_s1[r] - pos_d1[r], axis=1)
            if dist_ds.min() <= p.r_rep:
                pos_d = pos_d1[r] + p.dog_slow_step * vel_d1[r]
                V_d[r, :, t] = vel_d1[r]
                mode[r, t] = SLOW
            else:
                G = pos_s1[r].mean(0)
                rg = pos_s1[r] - G
                dg = norm(rg, axis=1)
                th = 2 * np.pi * d["noise_d"][r]
                err = np.array([np.cos(th), np.sin(th)])
                if dg.max() >= p.l_sep:
                    sp = dg.argmax()
                    aim = G + (dg[sp] + p.l_collect) * rg[sp] / dg[sp]
                    mode[r, t] = COLLECT
                else:
                    gt = G - target
                    aim = target + (norm(gt) + p.l_drive) * gt / norm(gt)
                    mode[r, t] = DRIVE
                rdc = (aim - pos_d1[r]) / norm(aim - pos_d1[r])
                v = rdc + p.noise * err
                v = v / norm(v)
                pos_d = pos_d1[r] + p.v_dog * v
                vel_d1[r] = v
                V_d[r, :, t] = v
            P_s[r, :, :, t] = pos_s
            P_d[r, :, t] = pos_d
        pos_s1 = P_s[..., t].copy()
        vel_s1 = V_s[..., t].copy()
        pos_d1 = P_d[..., t].copy()

    return Result(p, P_s, V_s, P_d, V_d, mode)
