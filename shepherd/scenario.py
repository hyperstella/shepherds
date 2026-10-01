"""Simulated drives matched to the real ones.

Each simulated drive starts from the first frame of a real drive (sheep
positions and headings) and replays the real dog's recorded path, so
model and data face exactly the same dog. Comparisons then test only the
sheep rules.
"""

from dataclasses import replace

import numpy as np

from . import data
from . import protocol as PR
from .model import Params, simulate

N = data.N_SHEEP


def drive_start(drive, dt):
    """Initial sheep state and the dog path of a real drive, sampled every dt.

    Headings come from each sheep's displacement over the first step; a sheep
    that has not moved takes the flock's mean heading (or +x if nobody moved).
    Returns (init dict for `simulate`, dog path (2, T)).
    """
    k = int(round(dt / data.DT))
    frames = drive.pos[: N + 1, :, ::k]
    disp = frames[:N, :, 1] - frames[:N, :, 0]
    norm = np.linalg.norm(disp, axis=-1, keepdims=True)
    moving = norm[:, 0] > 1e-6
    mean = disp[moving].sum(0) if moving.any() else np.array([1.0, 0.0])
    mean = mean / np.linalg.norm(mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        head = np.where(moving[:, None], disp / norm, mean)
    return {"pos_s": frames[:N, :, 0][None], "vel_s": head[None]}, frames[N]


def replay(drive, params=Params(), n_runs=1, seed=None, dt=1.0, per_run=None):
    """Simulate `n_runs` replicates of one real drive with the dog replayed."""
    init, dog_path = drive_start(drive, dt)
    return simulate(replace(params, n_sheep=N, dt=dt), n_runs, seed, per_run=per_run, init=init, dog_path=dog_path)


def replay_segments(drives, params=Params(), n_runs=1, seed=0, dt=1.0, per_run=None, min_speed=PR.MIN_SPEED):
    """One batched Segment (runs, N+1, 2, T) per real drive."""
    out = []
    for i, d in enumerate(drives):
        res = replay(d, params, n_runs, seed=None if seed is None else seed + i, dt=dt, per_run=per_run)
        pos = np.concatenate([res.pos_s, res.pos_d[:, None]], axis=1)
        out.append(PR.from_positions(pos, dt, N, min_speed, d.session, f"s{d.session}d{d.index + 1}"))
    return out


def _window_bounds(T, L, min_len):
    """Non-overlapping windows of L samples; a tail of at least L/2 (or a
    short drive of at least min_len) becomes its own window."""
    if T < L:
        return [(0, T)] if T >= min_len else []
    bounds = [(s, s + L) for s in range(0, T - L + 1, L)]
    tail = bounds[-1][1]
    if T - tail >= max(L // 2, min_len):
        bounds.append((tail, T))
    return bounds


def _drive_windows(drive, dt, horizon, min_len_s):
    k = int(round(dt / data.DT))
    frames = drive.pos[: N + 1, :, ::k]
    L, m = int(round(horizon / dt)), int(round(min_len_s / dt))
    return frames, _window_bounds(frames.shape[-1], L, m)


def window_segments(drives, dt=1.0, horizon=15.0, burn=3.0, min_len_s=6.0, min_speed=PR.MIN_SPEED):
    """Real drives cut into the same windows as `replay_windows`, minus the burn-in."""
    b = int(round(burn / dt))
    out = []
    for d in drives:
        frames, bounds = _drive_windows(d, dt, horizon, min_len_s)
        for w, (s, e) in enumerate(bounds):
            seg = PR.from_positions(frames[..., s:e], dt, N, min_speed, d.session, f"s{d.session}d{d.index + 1}w{w}")
            out.append(_trim(seg, b))
    return out


def replay_windows(drives, params=Params(), n_runs=1, seed=0, dt=1.0, horizon=15.0, burn=3.0, min_len_s=6.0,
                   per_run=None, min_speed=PR.MIN_SPEED, speed_replay=False):
    """Multiple shooting: every `horizon` seconds the model restarts from the
    real flock state and the real dog path is replayed, so model flock and
    dog cannot drift apart. The first `burn` seconds of each window (the
    flock relaxing from the real configuration to the model's) are dropped.
    With speed_replay, sheep move at the real flock's mean speed of each step
    instead of the constant v_sheep.
    Returns one batched Segment (runs, N+1, 2, T) per window.
    """
    b = int(round(burn / dt))
    out, i = [], 0
    for d in drives:
        frames, bounds = _drive_windows(d, dt, horizon, min_len_s)
        for w, (s, e) in enumerate(bounds):
            win = frames[..., s:e]
            disp = win[:N, :, 1] - win[:N, :, 0]
            norm = np.linalg.norm(disp, axis=-1, keepdims=True)
            moving = norm[:, 0] > 1e-6
            mean = disp[moving].sum(0) if moving.any() else np.array([1.0, 0.0])
            with np.errstate(invalid="ignore", divide="ignore"):
                head = np.where(moving[:, None], disp / norm, mean / np.linalg.norm(mean))
            speed = None
            if speed_replay:
                speed = np.linalg.norm(np.diff(win[:N], axis=-1), axis=1).mean(0) / dt
                speed = np.concatenate([speed[:1], speed])  # step t moves at the speed from t-1 to t
            res = simulate(replace(params, n_sheep=N, dt=dt), n_runs, None if seed is None else seed + i,
                           per_run=per_run, init={"pos_s": win[:N, :, 0][None], "vel_s": head[None]},
                           dog_path=win[N], speed_path=speed)
            i += 1
            pos = np.concatenate([res.pos_s, res.pos_d[:, None]], axis=1)
            seg = PR.from_positions(pos, dt, N, min_speed, d.session, f"s{d.session}d{d.index + 1}w{w}")
            out.append(_trim(seg, b))
    return out


def _trim(seg, b):
    if b <= 0:
        return seg
    return PR.Segment(seg.pos[..., b:], seg.vel[..., b:], seg.head[..., b:], seg.dt, seg.n_sheep, seg.session, seg.label)


def unbatch(segs):
    """Split batched Segments into one Segment per run (for hierarchy analysis)."""
    out = []
    for s in segs:
        for r in range(s.pos.shape[0]):
            out.append(PR.Segment(s.pos[r], s.vel[r], s.head[r], s.dt, s.n_sheep, s.session, f"{s.label}r{r}"))
    return out


def replay_windows_kernel(drives, model, coef, sigma, n_runs=1, seed=0, dt=1.0, horizon=15.0, burn=3.0,
                          min_len_s=6.0, window_s=2.1, min_speed=PR.MIN_SPEED, speed_replay=True,
                          velocity_replay=False, init_fn=None):
    """Same windows as `replay_windows`, for a learned second-order kernel model.

    The model is integrated at the 0.1 s tracking step from the smoothed real
    sheep positions and velocities at each window start, with the smoothed
    real dog path replayed, then sampled every dt. init_fn(pos, vel) -> (pos, vel)
    may alter the starting state ((N, 2) arrays), e.g. to test what is inherited
    from the real configuration.
    """
    from . import kernels as KN

    k = int(round(dt / data.DT))
    b = int(round(burn / dt))
    out, i = [], 0
    for d in drives:
        pos, vel = KN.smooth_drive(d.pos[: N + 1], window_s=window_s)
        _, bounds = _drive_windows(d, dt, horizon, min_len_s)
        for w, (s, e) in enumerate(bounds):
            s10, e10 = s * k, (e - 1) * k + 1
            speed = np.linalg.norm(vel[:N, :, s10:e10], axis=1).mean(0) if speed_replay and not velocity_replay else None
            mean_vel = vel[:N, :, s10:e10].mean(0) if velocity_replay else None
            p0, v0 = pos[:N, :, s10], vel[:N, :, s10]
            if init_fn is not None:
                p0, v0 = init_fn(p0, v0)
            P, _ = KN.simulate(model, coef, sigma, np.repeat(p0[None], n_runs, 0),
                               np.repeat(v0[None], n_runs, 0), pos[N, :, s10:e10],
                               seed=None if seed is None else seed + i, speed_path=speed, mean_vel_path=mean_vel)
            i += 1
            seg = PR.from_positions(P[..., ::k], dt, N, min_speed, d.session, f"s{d.session}d{d.index + 1}w{w}")
            out.append(_trim(seg, b))
    return out
