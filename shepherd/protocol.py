"""One analysis protocol for real and simulated drives.

The original study analysed data and model differently (headings from
velocity vs model state, signed vs absolute cross-correlation, a 2 s vs
20 s lag window, bin-mean correlations). Here both go through the same
steps:

1. `Segment`: sheep + dog positions sampled every `dt` seconds, with
   headings from displacement over `dt`. Samples slower than
   `min_speed` get a zero heading and are left out of heading statistics.
2. `summaries`: per-segment statistics for calibration (never the
   leadership hierarchy, which is what we test).
3. `hierarchy` / `hierarchy_stats`: leader-follower networks with lags in
   seconds, tested at the level of individual sheep, with session-level
   bootstrap and time-shift surrogates.

Arrays keep the MATLAB layout (..., n_agents, 2, T) with the dog last;
leading batch axes (simulation runs) are allowed in `Segment` and
`summaries`.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import data
from . import metrics as M
from . import network as NW

MIN_SPEED = 0.1  # m/s


@dataclass
class Segment:
    pos: np.ndarray  # (..., N+1, 2, T), dog last
    vel: np.ndarray  # (..., N+1, 2, T) displacement / dt
    head: np.ndarray  # (..., N+1, 2, T) unit headings, 0 where slower than min_speed
    dt: float
    n_sheep: int
    session: int = -1
    label: str = ""

    @property
    def T(self):
        return self.pos.shape[-1]

    @property
    def duration(self):
        return self.T * self.dt

    @property
    def moving(self):
        return np.any(self.head != 0, axis=-2)  # (..., N+1, T)


def from_positions(pos, dt, n_sheep, min_speed=MIN_SPEED, session=-1, label=""):
    """Build a Segment from positions; headings from backward displacement
    (forward difference at t = 0)."""
    d = np.diff(pos, axis=-1)
    disp = np.concatenate([d[..., :1], d], axis=-1)
    vel = disp / dt
    speed = np.linalg.norm(vel, axis=-2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        head = np.where(speed >= min_speed, vel / speed, 0.0)
    return Segment(pos, vel, np.nan_to_num(head), dt, n_sheep, session, label)


def segments_from_drives(drives, dt=0.1, min_speed=MIN_SPEED):
    """Real drives resampled to `dt` (a multiple of the 0.1 s tracking step)."""
    k = int(round(dt / data.DT))
    if abs(k * data.DT - dt) > 1e-9:
        raise ValueError(f"dt must be a multiple of {data.DT}")
    n = data.N_SHEEP
    return [
        from_positions(d.pos[: n + 1, :, ::k], dt, n, min_speed, d.session, f"s{d.session}d{d.index + 1}")
        for d in drives
    ]


def segments_legacy(drives):
    """Exactly the inputs of figure_5de.m: headings from phi, sample 0 dropped."""
    n = data.N_SHEEP
    out = []
    for d in drives:
        phi = d.phi[: n + 1, 1:]
        head = np.stack([np.cos(phi), np.sin(phi)], axis=-2)
        out.append(Segment(d.pos[: n + 1, :, 1:], d.vel[: n + 1, :, 1:], head, data.DT, n, d.session,
                           f"s{d.session}d{d.index + 1}"))
    return out


def segments_from_result(res, model_dt=1.0, dt=None, t0=0, t1=None, legacy=False, min_speed=MIN_SPEED):
    """One Segment per simulation run, using steps [t0, t1) sampled every dt.

    legacy=True uses the model's internal heading vectors (as in
    model_network_ana.m) instead of displacement headings.
    """
    dt = model_dt if dt is None else dt
    k = int(round(dt / model_dt))
    n = res.params.n_sheep
    out = []
    for r in range(res.mode.shape[0]):
        pos, vel = res.run(r)
        pos, vel = pos[..., t0:t1:k], vel[..., t0:t1:k]
        if legacy:
            out.append(Segment(pos, vel, vel / np.linalg.norm(vel, axis=-2, keepdims=True), dt, n, r, f"run{r}"))
        else:
            out.append(from_positions(pos, dt, n, min_speed, r, f"run{r}"))
    return out


# ------------------------------------------------------------------ summaries

SUMMARY_KEYS = [
    "nnd_med", "C_med", "E_med", "P_med", "P_p25",
    "turn_sheep_med", "turn_sheep_p90", "turn_dog_med", "slope_4a", "slope_4b",
    "dBD_med", "dBD_iqr", "yRD_med", "xD_iqr", "rear_frac", "xD_ac2", "xD_ac6",
]


def _nanq(x, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows give NaN
        return np.nanpercentile(x, q, axis=-1)


def _turn(head, lag):
    """|angle| between headings lag samples apart; NaN where either is zero."""
    a, b = head[..., :-lag], head[..., lag:]
    ok = np.any(a != 0, axis=-2) & np.any(b != 0, axis=-2)
    ang = np.abs(M.signed_angle(np.swapaxes(a, -1, -2), np.swapaxes(b, -1, -2)))
    return np.where(ok, ang, np.nan)


SLOPE_LIMIT = 10.0  # the data's Fig-4 slopes are about -0.8 and +1.0


def _slope0(x, y, min_n=5, min_mean_sq=1e-2):
    """Least-squares slope through the origin along the last axis, ignoring NaN.

    NaN when the fit is degenerate: fewer than min_n samples, almost no spread
    in x (e.g. the dog exactly behind the flock for a whole window), or a
    slope beyond +-SLOPE_LIMIT.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(-1)
    x, y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    sxx = (x * x).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (x * y).sum(-1) / sxx
    bad = (n < min_n) | (sxx < min_mean_sq * np.maximum(n, 1)) | ~(np.abs(b) <= SLOPE_LIMIT)
    return np.where(bad, np.nan, b)


def _autocorr(x, lag):
    if x.shape[-1] <= lag + 2:
        return np.full(x.shape[:-1], np.nan)
    x = x - np.nanmean(x, axis=-1, keepdims=True)
    a, b = x[..., :-lag], x[..., lag:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(a * b, -1) / np.nanmean(x * x, -1)


def summaries(seg, chunk=64):
    """Calibration statistics for a Segment (batch axes allowed).

    Returns a dict of arrays with the segment's batch shape.
    """
    batch = seg.pos.shape[:-3]
    if batch and batch[0] > chunk:
        parts = [summaries(Segment(seg.pos[i:i + chunk], seg.vel[i:i + chunk], seg.head[i:i + chunk], seg.dt,
                                   seg.n_sheep), chunk) for i in range(0, batch[0], chunk)]
        return {k: np.concatenate([p[k] for p in parts]) for k in SUMMARY_KEYS}

    N, dt = seg.n_sheep, seg.dt
    pos, vel, head = seg.pos, seg.vel, seg.head
    sp = pos[..., :N, :, :]
    out = {}

    diff = sp[..., :, None, :, :] - sp[..., None, :, :, :]  # (..., N, N, 2, T)
    dist = np.linalg.norm(diff, axis=-2)
    dist[..., np.arange(N), np.arange(N), :] = np.inf
    nnd = dist.min(axis=-2)  # (..., N, T)
    out["nnd_med"] = np.median(nnd.reshape(*nnd.shape[:-2], -1), axis=-1)

    out["C_med"] = np.median(M.cohesion(pos, N), axis=-1)
    E, xD, yRD = M.frame_observables(pos, vel, N)
    out["E_med"] = _nanq(np.where(np.isfinite(E), E, np.nan), 50)

    hs = head[..., :N, :, :]
    n_mov = np.any(hs != 0, axis=-2).sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.linalg.norm(hs.sum(-3), axis=-2) / n_mov
    P = np.where(n_mov >= 2, P, np.nan)
    out["P_med"], out["P_p25"] = _nanq(P, 50), _nanq(P, 25)

    L1 = max(1, int(round(1.0 / dt)))
    ts = _turn(hs, L1)
    ts = ts.reshape(*ts.shape[:-2], -1)
    out["turn_sheep_med"], out["turn_sheep_p90"] = _nanq(ts, 50), _nanq(ts, 90)
    out["turn_dog_med"] = _nanq(_turn(head[..., N, :, :], L1), 50)

    L3 = max(1, int(round(3.0 / dt)))
    psi_bd, psi_db, _ = M.viewing_angles(pos, vel, N)
    vb = vel[..., :N, :, :].mean(-3)
    phi_b = np.arctan2(vb[..., 1, :], vb[..., 0, :])
    phi_d = np.arctan2(vel[..., N, 1, :], vel[..., N, 0, :])
    dphi_b = M.wrap(phi_b[..., L3:] - phi_b[..., :-L3])
    dphi_d = M.wrap(phi_d[..., L3:] - phi_d[..., :-L3])
    dog_moving = np.any(head[..., N, :, :] != 0, axis=-2)
    ok_d = dog_moving[..., L3:] & dog_moving[..., :-L3] & (np.abs(psi_db[..., :-L3]) < np.pi / 2)
    out["slope_4a"] = _slope0(np.sin(psi_bd[..., :-L3]), dphi_b)
    out["slope_4b"] = _slope0(np.where(ok_d, psi_db[..., :-L3], np.nan), dphi_d)

    dBD = M.dog_barycenter_distance(pos, N)
    q25, q50, q75 = _nanq(dBD, 25), _nanq(dBD, 50), _nanq(dBD, 75)
    out["dBD_med"], out["dBD_iqr"] = q50, q75 - q25
    out["yRD_med"] = _nanq(yRD, 50)
    out["xD_iqr"] = _nanq(xD, 75) - _nanq(xD, 25)
    out["rear_frac"] = np.mean(np.abs(psi_bd) > np.radians(150), axis=-1)
    out["xD_ac2"] = _autocorr(xD, max(1, int(round(2.0 / dt))))
    out["xD_ac6"] = _autocorr(xD, max(1, int(round(6.0 / dt))))
    return out


LONG_KEYS = ["C_med", "nnd_med", "P_med", "P_p25", "E_med", "C_drift", "frac_far"]


def long_summaries(seg, far=5.0):
    """Long-horizon flock statistics (batch axes allowed).

    C_drift: median cohesion in the last third minus the first third (m);
    a flock that falls apart has C_drift >> 0. frac_far: share of sheep-time
    farther than `far` m from the flock centre. Plus the usual C, nnd, P, E.
    """
    base = summaries(seg)
    N = seg.n_sheep
    C = M.cohesion(seg.pos, N)
    T = C.shape[-1]
    k = max(1, T // 3)
    out = {key: base[key] for key in ("C_med", "nnd_med", "P_med", "P_p25", "E_med")}
    out["C_drift"] = np.median(C[..., -k:], axis=-1) - np.median(C[..., :k], axis=-1)
    b = M.barycenter(seg.pos, N)
    d = np.linalg.norm(seg.pos[..., :N, :, :] - b[..., None, :, :], axis=-2)
    out["frac_far"] = (d > far).mean(axis=(-2, -1))
    return out


def aggregate(summary_list, durations):
    """Duration-weighted mean over segments, ignoring NaN, per key."""
    w = np.asarray(durations, float)
    out = {}
    for k in SUMMARY_KEYS:
        v = np.stack([np.asarray(s[k], float) for s in summary_list], axis=0)  # (S, ...)
        ww = np.where(np.isfinite(v), w.reshape(-1, *([1] * (v.ndim - 1))), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.nansum(np.nan_to_num(v) * ww, axis=0) / ww.sum(axis=0)
    return out


# ------------------------------------------------------------------ hierarchy

def _network(seg, maxlag, cmin, signed, n_nodes):
    ux, uy = seg.head[:n_nodes, 0], seg.head[:n_nodes, 1]
    return NW.leader_follower(ux, uy, maxlag=maxlag, cmin=cmin, use_abs=not signed)


def hierarchy(segs, maxlag_s=2.0, cmin=0.5, signed=True, min_duration_s=0.0, with_dog=True):
    """Per-segment leader-follower records.

    Sheep-only networks give each sheep's indegree (number of followers);
    d_i is its mean front(+)/back(-) offset. With the dog included, the
    network gives the dog's indegree for ranking.
    """
    recs = []
    for s in segs:
        N = s.n_sheep
        lag = max(1, int(round(maxlag_s / s.dt)))
        if s.duration < min_duration_s or s.T <= lag + 1:
            continue
        k, edges = _network(s, lag, cmin, signed, N)
        di = NW.front_back_position(s.pos[:N], s.head[:N, 0], s.head[:N, 1], N)
        rec = {"session": s.session, "label": s.label, "indeg": k, "di": di, "n_edges": len(edges),
               "lags_s": np.array([l * s.dt for *_, l in edges])}
        if with_dog:
            rec["indeg_dog"] = _network(s, lag, cmin, signed, N + 1)[0]
        recs.append(rec)
    return recs


def hierarchy_stats(recs):
    """Sheep-level and legacy (bin-mean) statistics pooled over records."""
    k = np.concatenate([r["indeg"] for r in recs]).astype(float)
    di = np.concatenate([r["di"] for r in recs])
    ok = np.isfinite(di)
    k, di = k[ok], di[ok]
    lags = np.concatenate([r["lags_s"] for r in recs]) if recs else np.array([])
    out = {
        "n_segments": len(recs),
        "edges_per_segment": float(np.mean([r["n_edges"] for r in recs])) if recs else np.nan,
        "spearman": float(stats.spearmanr(k, di)[0]) if np.ptp(k) > 0 else np.nan,
        "pearson": float(stats.pearsonr(k, di)[0]) if np.ptp(k) > 0 else np.nan,
        "median_lag_s": float(np.median(lags)) if lags.size else np.nan,
        "lag_iqr_s": [float(v) for v in np.percentile(lags, [25, 75])] if lags.size else [np.nan, np.nan],
    }
    levels = np.unique(k)
    if levels.size >= 3:
        means = np.array([di[k == v].mean() for v in levels])
        out["binmean_pearson"] = float(stats.pearsonr(levels, means)[0])
    else:
        out["binmean_pearson"] = np.nan
    if recs and "indeg_dog" in recs[0]:
        kd = np.array([r["indeg_dog"] for r in recs], float).mean(0)
        out["dog_mean_indegree"] = float(kd[-1])
        out["sheep_mean_indegree"] = float(kd[:-1].mean())
        out["dog_rank"] = int(1 + (kd[:-1] > kd[-1]).sum())  # 1 = most followed
    return out


def cluster_bootstrap(recs, fn, n_boot=2000, seed=0, key="session"):
    """Percentile CI of fn(records) resampling whole sessions (or runs)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for r in recs:
        groups.setdefault(r[key], []).append(r)
    ids = list(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        pooled = [r for i in pick for r in groups[ids[i]]]
        vals.append(fn(pooled))
    vals = np.asarray(vals, float)
    return {"ci95": [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])], "sd": float(np.nanstd(vals))}


def timeshift_surrogate(segs, n_surr=500, seed=0, maxlag_s=2.0, cmin=0.5, signed=True, min_duration_s=0.0):
    """Null distribution: each sheep's heading series is circularly shifted by
    an independent random offset (at least twice the maximum lag), which keeps
    every individual's own dynamics but breaks who-follows-whom. Front/back
    positions d_i come from the unshifted data.

    Returns arrays of the sheep-level Spearman rho and edges per segment.
    """
    rng = np.random.default_rng(seed)
    base = hierarchy(segs, maxlag_s, cmin, signed, min_duration_s, with_dog=False)
    di = {r["label"]: r["di"] for r in base}
    rhos, edges = [], []
    for _ in range(n_surr):
        ks, ds, ne = [], [], []
        for s in segs:
            if s.label not in di:
                continue
            N, T = s.n_sheep, s.T
            lag = max(1, int(round(maxlag_s / s.dt)))
            margin = min(2 * lag, T // 3)
            head = s.head.copy()
            for i in range(N):
                head[i] = np.roll(s.head[i], rng.integers(margin, T - margin + 1), axis=-1)
            k, e = NW.leader_follower(head[:N, 0], head[:N, 1], maxlag=lag, cmin=cmin, use_abs=not signed)
            ks.append(k), ds.append(di[s.label]), ne.append(len(e))
        k, d = np.concatenate(ks).astype(float), np.concatenate(ds)
        ok = np.isfinite(d)
        rhos.append(stats.spearmanr(k[ok], d[ok])[0] if np.ptp(k[ok]) > 0 else np.nan)
        edges.append(np.mean(ne))
    return {"spearman": np.array(rhos), "edges_per_segment": np.array(edges)}
