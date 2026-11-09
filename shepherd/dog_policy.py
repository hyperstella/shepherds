"""A dog policy whose parameters are measured from the real collie.

The policy keeps the Strömbom structure (aim at a point behind the flock on
the far side from the target), with every free quantity taken from the
tracks:

- l_drive: how far the dog works behind the flock centre, along the flock's
  direction of motion (median of -y_D in the flock frame).
- flank_sigma, flank_tau: the dog's sideways position relative to the flock
  (x_D) treated as an Ornstein-Uhlenbeck process; SD and correlation time
  from its autocorrelation.
- tau_d: heading relaxation time, from how fast the dog turns toward the
  flock (turning rate vs the angle psi_DB, Fig 4b).
- v_dog, dog_speed_gain: speed cap and the slope of speed vs distance to the
  inferred aim point.
"""

import numpy as np
from scipy.optimize import curve_fit

from . import data, metrics as M
from .kernels import smooth_drive

N = data.N_SHEEP


def _pooled(drives, window_s=1.1):
    rows = []
    for d in drives:
        pos, vel = smooth_drive(d.pos[: N + 1], window_s=window_s)
        xb, yb = M.flock_frame(pos, vel, N)
        _, psi_db, _ = M.viewing_angles(pos, vel, N)
        phi_d = np.unwrap(np.arctan2(vel[N, 1], vel[N, 0]))
        rows.append({"xD": xb[N], "yD": yb[N], "psi_db": psi_db, "phi_d": phi_d,
                     "speed_d": np.linalg.norm(vel[N], axis=0), "T": pos.shape[-1]})
    return rows


def estimate(drives, lag_turn_s=1.0, max_lag_s=4.0):
    dt = data.DT
    rows = _pooled(drives)
    yD = np.concatenate([r["yD"] for r in rows])
    xD = np.concatenate([r["xD"] for r in rows])
    l_drive = float(np.median(-yD))

    # x_D autocorrelation pooled over drives (mean removed per drive), fit exp(-lag/tau)
    lags = np.arange(0, int(max_lag_s / dt) + 1)
    num, den = np.zeros(len(lags)), np.zeros(len(lags))
    for r in rows:
        x = r["xD"] - r["xD"].mean()
        for i, L in enumerate(lags):
            if len(x) > L + 1:
                num[i] += np.sum(x[: len(x) - L] * x[L:])
                den[i] += len(x) - L
    ac = (num / den) / (num[0] / den[0])
    (tau,), _ = curve_fit(lambda l, t: np.exp(-l / t), lags * dt, ac, p0=[2.0])
    flank_sigma = float(np.std(xD))

    # turning rate toward the flock vs psi_DB (small-angle slope = 1 / tau_d)
    L = int(round(lag_turn_s / dt))
    x = np.concatenate([r["psi_db"][:-L] for r in rows])
    y = np.concatenate([(r["phi_d"][L:] - r["phi_d"][:-L]) / lag_turn_s for r in rows])
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < np.pi / 2)
    k = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2))

    # speed vs distance to the inferred aim (behind the flock centre by l_drive, on the flock axis)
    dist = np.hypot(xD, yD + l_drive)
    sp = np.concatenate([r["speed_d"] for r in rows])
    gain = float(np.sum(dist * sp) / np.sum(dist ** 2))
    return {
        "l_drive": l_drive,
        "flank_sigma": flank_sigma,
        "flank_tau": float(tau),
        "turn_gain_per_s": k,
        "tau_d": float(1 / k) if k > 0 else None,
        "v_dog": float(np.percentile(sp, 95)),
        "dog_speed_gain": gain,
        "mean_dog_speed": float(sp.mean()),
        "xD_autocorr": {f"{l * dt:.1f}s": float(a) for l, a in zip(lags[::5], ac[::5])},
    }
