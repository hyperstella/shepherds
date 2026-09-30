"""Collective-behaviour observables from Jadhav et al. (2024).

Ports of the computations in figure_2ae.m, figure_2fj.m, figure_3.m,
figure_4.m and model_results.m. All functions take MATLAB-layout arrays:
positions/velocities (n_agents, 2, T) with sheep first, then the dog.
"""

import numpy as np


def _unit(v, axis):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def signed_angle(a, b):
    """Angle turning 2D vector(s) a onto b, in (-pi, pi]. Shapes (..., 2).

    Equivalent to MATLAB's acos(dot(a_hat, b_hat)) * sign(cross_z(a, b)),
    including returning 0 when either vector is zero.
    """
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    return np.arctan2(cross, dot)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def speeds(vel):
    return np.linalg.norm(vel, axis=1)  # (n, T)


def barycenter(pos, n_sheep):
    return pos[:n_sheep].mean(axis=0)  # (2, T)


def cohesion(pos, n_sheep):
    """C(t): mean distance of sheep to the barycenter."""
    b = barycenter(pos, n_sheep)
    return np.linalg.norm(pos[:n_sheep] - b, axis=1).mean(axis=0)


def polarization_from_phi(phi, n_sheep):
    """P(t) from heading angles, as in figure_2fj.m."""
    return np.hypot(np.cos(phi[:n_sheep]).mean(0), np.sin(phi[:n_sheep]).mean(0))


def polarization_from_vel(vel, n_sheep):
    """P(t) from normalized velocities, as in model_results.m."""
    return np.linalg.norm(_unit(vel[:n_sheep], axis=1).mean(0), axis=0)


def flock_frame(pos, vel, n_sheep):
    """Coordinates (x_bar, y_bar) of every agent in the barycenter frame.

    The flock velocity v_B = mean sheep velocity points along +y_bar, and
    x_bar > 0 is to the right of the direction of motion.
    Returns two arrays of shape (n_agents, T).
    """
    b = barycenter(pos, n_sheep)
    vb = vel[:n_sheep].mean(axis=0)
    ang = np.arctan2(vb[1], vb[0])
    c, s = np.cos(ang), np.sin(ang)
    rel = pos - b
    along = c * rel[:, 0] + s * rel[:, 1]
    left = -s * rel[:, 0] + c * rel[:, 1]
    return -left, along


def frame_observables(pos, vel, n_sheep, dog=None):
    """Elongation E, dog lateral offset x_D, and dog-to-rear-sheep distance y_RD."""
    dog = n_sheep if dog is None else dog
    xb, yb = flock_frame(pos, vel, n_sheep)
    xs, ys = xb[:n_sheep], yb[:n_sheep]
    elong = (ys.max(0) - ys.min(0)) / (xs.max(0) - xs.min(0))
    x_d = xb[dog]
    y_rd = ys.min(0) - yb[dog]  # >0 when the dog is behind the rear sheep
    return elong, x_d, y_rd


def dog_barycenter_distance(pos, n_sheep, dog=None):
    dog = n_sheep if dog is None else dog
    return np.linalg.norm(pos[dog] - barycenter(pos, n_sheep), axis=0)


def viewing_angles(pos, vel, n_sheep, dog=None, dog_vel=None):
    """psi_BD, psi_DB and phi_BD (radians), as in figure_3.m.

    psi_BD: angle v_B must turn to point at the dog.
    psi_DB: angle v_D must turn to point at the barycenter.
    phi_BD: heading difference phi_D - phi_B.
    """
    dog = n_sheep if dog is None else dog
    b = barycenter(pos, n_sheep).T
    vb = vel[:n_sheep].mean(axis=0).T
    vd = (vel[dog] if dog_vel is None else dog_vel).T
    r_bd = pos[dog].T - b
    return signed_angle(vb, r_bd), signed_angle(vd, -r_bd), signed_angle(vb, vd)


def turning_vs_viewing(pos, vel, n_sheep, lag, dog=None):
    """Turning angle over `lag` samples of barycenter and dog, paired with the
    viewing angle at the start of the window (figure_4.m).

    Returns dict with dphi_B, psi_BD, dphi_D, psi_DB (radians).
    Uses samples 1: onwards (sample 0 has no velocity), like the MATLAB code.
    """
    dog = n_sheep if dog is None else dog
    pos, vel = pos[:, :, 1:], vel[:, :, 1:]
    vb = vel[:n_sheep].mean(axis=0)
    phi_b = np.arctan2(vb[1], vb[0])
    phi_d = np.arctan2(vel[dog, 1], vel[dog, 0])
    psi_bd, psi_db, _ = viewing_angles(pos, vel, n_sheep, dog)
    return {
        "dphi_B": wrap(phi_b[lag:] - phi_b[:-lag]),
        "psi_BD": psi_bd[:-lag],
        "dphi_D": wrap(phi_d[lag:] - phi_d[:-lag]),
        "psi_DB": psi_db[:-lag],
    }


PSI_EDGES = np.arange(-np.pi - np.pi / 12, np.pi + np.pi / 12 + 1e-9, np.pi / 6)


def binned_stats(x, y, edges=PSI_EDGES):
    """Median, quartiles, mean and s.e. of y within bins of x."""
    idx = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for k in range(len(centers)):
        v = y[(idx == k) & np.isfinite(y)]
        if v.size == 0:
            rows.append((np.nan,) * 6)
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append((med, q1, q3, v.mean(), v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan, v.size))
    med, q1, q3, mean, se, n = map(np.array, zip(*rows))
    return dict(center=centers, median=med, q1=q1, q3=q3, mean=mean, se=se, n=n)
