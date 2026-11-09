"""Closed-loop convergence of the published model (Strömbom/Jadhav dog).

Realizes the cascade argument of docs/continuification_driving.md on the published
model (Params() defaults, flock started 100 m from the target, 300 steps):

1. Pursuit: the dog's gap to the drive point closes at about v_dog - v_flock and
   the dog then hovers near the drive point.
2. Reaching: the centroid's distance to the target falls linearly at
   v_sheep * P_par (finite time), not exponentially.
3. Cohesion: for the lateral-most sheep, the inward attraction component
   w_att * kappa against the dog's outward unit-vector component plus noise, and
   whether that margin predicts the sheep's realized inward/outward step.
4. Switching: dwell times of collect/drive and whether V = rho + lambda * maxspread
   decreases in both modes.
5. Release: with the dog moved beyond r_dog the flock freezes.
6. Sweeps of the four sufficient conditions: dog speed, notice radius, attraction
   weight, and flock size with K = 10, n_att = 4 and the published notice radius.
7. Switching where collects occur (w_att = 0.9), with r_dog = 12 and 20.

    .venv/bin/python experiments/closed_loop_convergence.py
Writes results/closed_loop.json and results/figures/closed_loop.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd.model import COLLECT, DRIVE, SLOW, Params, simulate  # noqa: E402

OUT = ROOT / "results"
START, T, R, SEED = 100.0, 300, 200, 2024
ARRIVE, NEAR_DRIVE, FIT_END, MIN_SEG = 5.0, 3.0, 10.0, 20
REPS, T_SWEEP = 100, 400
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
SWEEPS = {
    "v_dog": [0.6, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0],
    "r_dog": [4.0, 6.7, 9.0, 10.0, 11.0, 12.0, 14.0, 20.0],
    "w_att": [0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0],
}
SIZES = (14, 30, 50, 100)
W_ATT_SWITCH = 0.9  # attraction weak enough that collect episodes occur
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a"}


def base(**kw):
    kw.setdefault("box_length", START)
    kw.setdefault("n_steps", T)
    return Params(**kw)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def fl(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def first_true(mask, default):
    return np.where(mask.any(1), mask.argmax(1), default)


# ------------------------------------------------------------------ geometry

def geometry(res, r_dog=None):
    """Per-step quantities. Shapes: (R, T) unless noted."""
    p = res.params.resolved()
    pos, vel, dog = res.pos_s, res.vel_s, res.pos_d
    target = np.asarray(p.target, float)
    G = pos.mean(1)  # (R, 2, T)
    to_t = target[None, :, None] - G
    rho = np.linalg.norm(to_t, axis=1)
    axis = to_t / np.maximum(rho[:, None], 1e-12)  # forward unit vector (R, 2, T)
    lat = np.stack([-axis[:, 1], axis[:, 0]], 1)
    p_drive = G - p.l_drive * axis  # l_drive behind the centroid on the target line
    gap = np.linalg.norm(dog - p_drive, axis=1)
    rel = pos - G[:, None]
    spread = np.linalg.norm(rel, axis=2)  # (R, N, T)
    y = np.einsum("rnct,rct->rnt", rel, lat)  # lateral offsets
    x_ax = np.einsum("rnct,rct->rnt", rel, axis)  # axial offsets (ahead > 0)
    P_par = np.einsum("rnct,rct->rnt", vel, axis).mean(1)
    d_behind = -np.einsum("rct,rct->rt", dog - G, axis)
    rd = p.r_dog if r_dog is None else np.asarray(r_dog, float)[:, None, None]
    sees = np.linalg.norm(pos - dog[:, None], axis=2) <= rd  # (R, N, T)
    return dict(p=p, G=G, rho=rho, axis=axis, lat=lat, gap=gap, spread=spread, maxspread=spread.max(1),
                y=y, x_ax=x_ax, P_par=P_par, d_behind=d_behind, sees=sees)


def reach_times(g):
    return first_true(g["rho"] < ARRIVE, g["rho"].shape[1])


# ------------------------------------------------------------------ 1. pursuit

def pursuit(g, mode, t_reach):
    gap, Tn = g["gap"], g["gap"].shape[1]
    t = np.arange(Tn)[None]
    arrived = (gap < NEAR_DRIVE) & (mode == DRIVE)
    t_dog = first_true(arrived, Tn)
    dgap = np.diff(gap, axis=1)
    approach = (mode[:, :-1] == DRIVE) & (gap[:, :-1] > NEAR_DRIVE) & (t[:, :-1] < t_dog[:, None])
    dG = np.linalg.norm(np.diff(g["G"], axis=2), axis=1)  # centroid speed (R, T-1)
    hover_sel = (mode == DRIVE) & (t >= t_dog[:, None]) & (t < t_reach[:, None])
    p = g["p"]
    return dict(
        t_dog=t_dog,
        t_dog_median=fl(np.median(t_dog[t_dog < Tn])) if (t_dog < Tn).any() else None,
        dog_arrival_rate=fl((t_dog < Tn).mean()),
        closing_rate_m_per_step=fl(-dgap[approach].mean()) if approach.any() else None,
        closing_predicted=fl(p.v_dog - dG[approach].mean()) if approach.any() else None,
        centroid_speed_on_approach=fl(dG[approach].mean()) if approach.any() else None,
        hover_gap_median=fl(np.median(gap[hover_sel])) if hover_sel.any() else None,
        hover_gap_p90=fl(np.percentile(gap[hover_sel], 90)) if hover_sel.any() else None,
        slow_frac_before_dog_arrival=fl((mode[:, :-1][t[:, :-1] < t_dog[:, None]] == SLOW).mean()),
    )


# ------------------------------------------------------------------ 2. reaching

def reaching(g, mode, t_reach, t_dog):
    rho, P, p = g["rho"], g["P_par"], g["p"]
    Rn, Tn = rho.shape
    slopes, pred, rmse_lin, rmse_exp, t_pred, t_act, P_seg = [], [], [], [], [], [], []
    for r in range(Rn):
        t0 = t_dog[r]
        if t0 >= Tn:
            continue
        below = np.flatnonzero(rho[r, t0:] < FIT_END)
        t1 = t0 + below[0] if len(below) else Tn
        if t1 - t0 < MIN_SEG:
            continue
        tt, yy = np.arange(t0, t1), rho[r, t0:t1]
        b, a = np.polyfit(tt, yy, 1)
        k, lnA = np.polyfit(tt, np.log(yy), 1)
        slopes.append(b)
        rmse_lin.append(np.sqrt(np.mean((a + b * tt - yy) ** 2)))
        rmse_exp.append(np.sqrt(np.mean((np.exp(lnA + k * tt) - yy) ** 2)))
        Pm = P[r, t0:t1].mean()
        P_seg.append(Pm)
        pred.append(-p.v_sheep * Pm)
        if t_reach[r] < Tn:
            t_pred.append(t0 + (rho[r, t0] - ARRIVE) / (p.v_sheep * Pm))
            t_act.append(t_reach[r])
    slopes, pred = np.array(slopes), np.array(pred)
    return dict(
        n_segments=len(slopes),
        slope_median=fl(np.median(slopes)),
        slope_predicted_median=fl(np.median(pred)),
        slope_ratio_median=fl(np.median(slopes / pred)) if len(slopes) else None,
        P_par_on_drive_median=fl(np.median(P_seg)),
        rmse_linear_median=fl(np.median(rmse_lin)),
        rmse_exponential_median=fl(np.median(rmse_exp)),
        exp_over_lin_rmse_median=fl(np.median(np.array(rmse_exp) / np.array(rmse_lin))) if len(slopes) else None,
        t_reach_predicted_median=fl(np.median(t_pred)) if t_pred else None,
        t_reach_actual_median=fl(np.median(t_act)) if t_act else None,
        t_reach_error_median_steps=fl(np.median(np.array(t_act) - np.array(t_pred))) if t_pred else None,
        success_rate=fl((t_reach < Tn).mean()),
    )


def post_reach(g, mode, t_reach, n_after=50):
    rho, G, axis, Tn = g["rho"], g["G"], g["axis"], g["rho"].shape[1]
    target = np.asarray(g["p"].target, float)
    sel = np.flatnonzero(t_reach <= Tn - n_after)
    if len(sel) == 0:
        return dict(n_runs=0)
    rho_post, cross, modes = [], [], []
    for r in sel:
        tr = t_reach[r]
        a0 = axis[r, :, tr]
        s = ((G[r, :, tr:] - target[:, None]) * a0[:, None]).sum(0)  # axial coordinate, + = past the target
        cross.append(int((np.diff(np.sign(s)) != 0).sum()))
        rho_post.append(rho[r, tr:])
        modes.append(mode[r, tr:])
    rho_post = np.concatenate(rho_post)
    modes = np.concatenate(modes)
    return dict(
        n_runs=int(len(sel)),
        rho_mean=fl(rho_post.mean()),
        rho_p90=fl(np.percentile(rho_post, 90)),
        axis_crossings_mean=fl(np.mean(cross)),
        drive_frac=fl((modes == DRIVE).mean()),
        collect_frac=fl((modes == COLLECT).mean()),
        slow_frac=fl((modes == SLOW).mean()),
    )


# ------------------------------------------------------------------ 3. cohesion

def edge_terms(res, g, mode, t_reach, rng, n_sub=24):
    """Lateral-most sheep at each step: dog outward term, kappa, realized lateral step."""
    p, pos, dog = g["p"], res.pos_s, res.pos_d
    Rn, _, _, Tn = pos.shape
    y, G = g["y"], g["G"]
    i_edge = np.abs(y).argmax(1)  # (R, T)
    ar = np.arange(Rn)
    K, na = p.k_perceive, p.n_att
    out = {k: np.full((Rn, Tn), np.nan) for k in ("dog_term", "kappa_K", "R_lat", "d", "realized", "front")}
    out["kappa_sub"] = np.full((Rn, Tn, n_sub), np.nan)
    valid = np.zeros((Rn, Tn), bool)
    for t in range(Tn - 1):
        i = i_edge[:, t]
        x = pos[ar, i, :, t]
        n_out = np.sign(y[ar, i, t])[:, None] * g["lat"][:, :, t]  # outward lateral unit
        away = unit(x - dog[:, :, t])
        out["dog_term"][:, t] = (away * n_out).sum(1)
        rel = pos[:, :, :, t] - x[:, None, :]
        dist = np.linalg.norm(rel, axis=2)
        dist[ar, i] = np.inf
        knn = np.argsort(dist, axis=1, kind="stable")[:, :K]
        u = rel / np.maximum(dist[..., None], 1e-12)
        u_k = u[ar[:, None], knn]  # (R, K, 2) unit vectors toward the K nearest
        out["kappa_K"][:, t] = -(unit(u_k.sum(1)) * n_out).sum(1)
        for s in range(n_sub):
            sub = np.argsort(rng.random((Rn, K)), axis=1)[:, :na]
            out["kappa_sub"][:, t, s] = -(unit(u_k[ar[:, None], sub].sum(1)) * n_out).sum(1)
        dx = (pos[ar, i, :, t + 1] - x) - (G[:, :, t + 1] - G[:, :, t])
        out["realized"][:, t] = (dx * n_out).sum(1)
        out["R_lat"][:, t] = np.abs(y[ar, i, t])
        out["d"][:, t] = g["d_behind"][:, t]
        out["front"][:, t] = g["x_ax"][:, :, t].max(1)
        valid[:, t] = (mode[:, t] == DRIVE) & (g["d_behind"][:, t] > 0) & g["sees"][ar, i, t] & (t < t_reach)
    out["valid"] = valid
    return out


def cohesion(edge, p):
    v = edge["valid"]
    dog_t, kK, ks = edge["dog_term"][v], edge["kappa_K"][v], edge["kappa_sub"][v]
    realized = edge["realized"][v]
    margin_typ = p.w_att * kK - p.w_dog * dog_t
    margin_worst = margin_typ - p.noise
    bins = np.array([-np.inf, -0.5, 0.0, 0.5, 1.0, np.inf])
    idx = np.digitize(margin_typ, bins) - 1
    inward_by_bin = [fl((realized[idx == b] < 0).mean()) if (idx == b).any() else None for b in range(len(bins) - 1)]
    count_by_bin = [int((idx == b).sum()) for b in range(len(bins) - 1)]
    dog_med, k_med = np.median(dog_t), np.median(ks)
    return dict(
        n_steps=int(v.sum()),
        R_lat_median=fl(np.median(edge["R_lat"][v])),
        d_behind_median=fl(np.median(edge["d"][v])),
        front_depth_median=fl(np.median(edge["front"][v])),
        dog_term_median=fl(dog_med),
        dog_term_p90=fl(np.percentile(dog_t, 90)),
        kappa_K_median=fl(np.median(kK)),
        kappa_sub_median=fl(k_med),
        kappa_sub_p05=fl(np.percentile(ks, 5)),
        kappa_sub_frac_below_0=fl((ks < 0).mean()),
        kappa_required_worst=fl((p.w_dog * dog_med + p.noise) / p.w_att),
        kappa_required_typical=fl(p.w_dog * dog_med / p.w_att),
        w_att_required_worst=fl((p.w_dog * dog_med + p.noise) / k_med),
        w_att_required_typical=fl(p.w_dog * dog_med / k_med),
        margin_worst_frac_positive=fl((margin_worst > 0).mean()),
        margin_typical_frac_positive=fl((margin_typ > 0).mean()),
        realized_inward_frac=fl((realized < 0).mean()),
        realized_inward_by_margin_bin=inward_by_bin,
        margin_bins=[fl(b) for b in bins],
        count_by_margin_bin=count_by_bin,
        corr_margin_realized=fl(np.corrcoef(margin_typ, realized)[0, 1]),
    )


# ------------------------------------------------------------------ 4. switching

def switching(g, mode, t_reach):
    rho, ms, Tn = g["rho"], g["maxspread"], g["rho"].shape[1]
    t = np.arange(Tn)[None]
    pre = t < t_reach[:, None]
    m, pre1 = mode[:, :-1], pre[:, :-1]
    drho, dms = np.diff(rho, axis=1), np.diff(ms, axis=1)
    out = {"lyapunov": {}, "mean_step": {}}
    for name, code in (("drive", DRIVE), ("collect", COLLECT), ("slow", SLOW)):
        sel = (m == code) & pre1
        out["mean_step"][name] = dict(d_rho=fl(drho[sel].mean()) if sel.any() else None,
                                      d_maxspread=fl(dms[sel].mean()) if sel.any() else None,
                                      frac_of_steps=fl(sel.sum() / pre1.sum()))
    for lam in LAMBDAS:
        dV = drho + lam * dms
        row = {}
        for name, code in (("drive", DRIVE), ("collect", COLLECT), ("slow", SLOW)):
            sel = (m == code) & pre1
            row[name] = fl((dV[sel] <= 0).mean()) if sel.any() else None
        row["all"] = fl((dV[pre1] <= 0).mean())
        out["lyapunov"][str(lam)] = row
    n_col, d_drive, d_col = [], [], []
    for r in range(mode.shape[0]):
        seq = mode[r, : t_reach[r]]
        if len(seq) < 2:
            continue
        change = np.flatnonzero(np.diff(seq) != 0) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(seq)]
        codes, lens = seq[starts], ends - starts
        n_col.append(int((codes == COLLECT).sum()))
        d_drive += lens[codes == DRIVE].tolist()
        d_col += lens[codes == COLLECT].tolist()
    out["collect_episodes_mean"] = fl(np.mean(n_col))
    out["dwell_drive_median"] = fl(np.median(d_drive)) if d_drive else None
    out["dwell_collect_median"] = fl(np.median(d_col)) if d_col else None
    out["dwell_collect_mean"] = fl(np.mean(d_col)) if d_col else None
    out["maxspread_over_l_sep_frac"] = fl((ms[pre] >= g["p"].l_sep).mean())
    return out


# ------------------------------------------------------------------ 5. release

def release(res, g, t_reach, n_after=60):
    Tn = g["rho"].shape[1]
    sel = np.flatnonzero(t_reach < Tn)
    if len(sel) == 0:
        return dict(n_runs=0)
    pos_s = np.stack([res.pos_s[r, :, :, t_reach[r]] for r in sel])
    vel_s = np.stack([res.vel_s[r, :, :, t_reach[r]] for r in sel])
    far = np.asarray(g["p"].target, float) + np.array([0.0, 300.0])
    dog_path = np.broadcast_to(far[None, :, None], (len(sel), 2, n_after)).copy()
    r2 = simulate(base(), n_runs=len(sel), seed=SEED + 1, init={"pos_s": pos_s, "vel_s": vel_s}, dog_path=dog_path)
    step = np.linalg.norm(np.diff(r2.pos_s, axis=-1), axis=2)  # (R', N, T'-1)
    moving = (step > 1e-9).any(1)  # (R', T'-1)
    last_move = np.where(moving.any(1), moving.shape[1] - 1 - np.argmax(moving[:, ::-1], axis=1), -1)
    disp = np.linalg.norm(r2.pos_s[..., -1] - r2.pos_s[..., 0], axis=2)  # (R', N)
    return dict(
        n_runs=int(len(sel)),
        frozen_by_end_rate=fl((last_move < moving.shape[1] - 1).mean()),
        steps_until_frozen_median=fl(np.median(last_move + 1)),
        steps_until_frozen_p90=fl(np.percentile(last_move + 1, 90)),
        displacement_mean_m=fl(disp.mean()),
        displacement_max_m=fl(disp.max()),
        sheep_moved_frac=fl((disp > 1e-9).mean()),
    )


# ------------------------------------------------------------------ 6. sweeps

def summarize(res, g, mode, edge=None):
    t_reach = reach_times(g)
    Tn = g["rho"].shape[1]
    ok = t_reach < Tn
    pre = np.arange(Tn)[None] < t_reach[:, None]
    ms = g["maxspread"]
    out = dict(
        success_rate=fl(ok.mean()),
        t_reach_median=fl(np.median(t_reach[ok])) if ok.any() else None,
        final_maxspread_median=fl(np.median(ms[:, -1])),
        split_rate=fl((ms[:, -1] >= g["p"].l_sep).mean()),
        collect_frac=fl((mode[pre] == COLLECT).mean()) if pre.any() else None,
        slow_frac=fl((mode[pre] == SLOW).mean()) if pre.any() else None,
        coverage_frac=fl(g["sees"].all(1)[pre].mean()) if pre.any() else None,
        P_par_median=fl(np.median(g["P_par"][pre & (mode == DRIVE)])) if (pre & (mode == DRIVE)).any() else None,
    )
    if edge is not None:
        v = edge["valid"]
        if v.any():
            out["kappa_sub_p05"] = fl(np.percentile(edge["kappa_sub"][v], 5))
            out["kappa_sub_median"] = fl(np.median(edge["kappa_sub"][v]))
            out["dog_term_median"] = fl(np.median(edge["dog_term"][v]))
    return out


def sweep_per_run(field, values, rng):
    vals = np.repeat(np.asarray(values, float), REPS)
    res = simulate(base(n_steps=T_SWEEP), n_runs=len(vals), seed=SEED + 2, per_run={field: vals})
    rows = []
    for v in values:
        sel = np.flatnonzero(vals == v)
        sub = _subset(res, sel)
        gs = geometry(sub, r_dog=vals[sel] if field == "r_dog" else None)
        t_reach = reach_times(gs)
        edge = edge_terms(sub, gs, sub.mode, t_reach, rng, n_sub=8) if field == "w_att" else None
        rows.append({field: float(v), **summarize(sub, gs, sub.mode, edge)})
    return rows


def _subset(res, sel):
    from dataclasses import replace

    return replace(res, pos_s=res.pos_s[sel], vel_s=res.vel_s[sel], pos_d=res.pos_d[sel], vel_d=res.vel_d[sel], mode=res.mode[sel])


def sweep_sizes(rng, r_dog=None, n_steps=T_SWEEP):
    rows = []
    for N in SIZES:
        kw = dict(n_sheep=N, n_steps=n_steps)
        if r_dog is not None:
            kw["r_dog"] = r_dog
        res = simulate(base(**kw), n_runs=REPS, seed=SEED + 3)
        g = geometry(res)
        t_reach = reach_times(g)
        edge = edge_terms(res, g, res.mode, t_reach, rng, n_sub=8)
        rows.append({"n_sheep": N, "l_sep": fl(g["p"].l_sep), "l_drive": fl(g["p"].l_drive), **summarize(res, g, res.mode, edge)})
        print(f"  N={N:3d} r_dog={g['p'].r_dog:g} success {rows[-1]['success_rate']:.2f} split {rows[-1]['split_rate']:.2f} "
              f"coverage {rows[-1]['coverage_frac']:.2f} kappa05 {rows[-1].get('kappa_sub_p05', float('nan')):.2f}", flush=True)
    return rows


def switching_rich(rng):
    """Switching statistics where collects actually happen (weak attraction), with the
    published notice radius and a larger one that keeps the main flock engaged during a collect."""
    out = {}
    for r_dog in (12.0, 20.0):
        res = simulate(base(w_att=W_ATT_SWITCH, r_dog=r_dog, n_steps=T_SWEEP), n_runs=R, seed=SEED + 4)
        g, mode = geometry(res), res.mode
        t_reach = reach_times(g)
        out[str(r_dog)] = {"summary": summarize(res, g, mode), **switching(g, mode, t_reach)}
        sw = out[str(r_dog)]
        print(f"  w_att={W_ATT_SWITCH} r_dog={r_dog:g}: success {sw['summary']['success_rate']:.2f}, collect frac "
              f"{sw['mean_step']['collect']['frac_of_steps']:.2f}, d_rho collect {sw['mean_step']['collect']['d_rho']}, "
              f"d_maxspread collect {sw['mean_step']['collect']['d_maxspread']}", flush=True)
    return out


# ------------------------------------------------------------------ figure

def figure(g, mode, t_reach, pur, rea, coh, edge, sw, sizes, swr, out_path):
    p = g["p"]
    fig, axs = plt.subplots(3, 3, figsize=(15.5, 12.5))
    t = np.arange(g["rho"].shape[1])

    ax = axs[0, 0]
    for r in range(0, g["rho"].shape[0], 7):
        ax.plot(t, g["rho"][r], color=COL["gray"], lw=0.5, alpha=0.5)
    med = np.median(g["rho"], axis=0)
    ax.plot(t, med, color=COL["blue"], lw=2, label="median of runs")
    t0 = int(pur["t_dog_median"] or 0)
    below = np.flatnonzero(med[t0:] < FIT_END)
    t1 = t0 + (below[0] if len(below) else len(med) - t0)
    if t1 - t0 >= MIN_SEG:
        tt = np.arange(t0, t1)
        b, a = np.polyfit(tt, med[t0:t1], 1)
        k, lnA = np.polyfit(tt, np.log(med[t0:t1]), 1)
        ax.plot(tt, a + b * tt, "--", color=COL["orange"], lw=2, label=f"linear fit, slope {b:.2f} m/step")
        ax.plot(tt, np.exp(lnA + k * tt), ":", color=COL["green"], lw=2, label="exponential fit")
    ax.axhline(ARRIVE, color=COL["gray"], lw=0.8, ls="--")
    ax.set(xlabel="time step (s)", ylabel="centroid distance to target (m)", title="Reaching is linear in time")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[0, 1]
    q = np.nanpercentile(g["gap"], [25, 50, 75], axis=0)
    ax.fill_between(t, q[0], q[2], color=COL["blue"], alpha=0.2, lw=0)
    ax.plot(t, q[1], color=COL["blue"], lw=2, label="dog gap to drive point (median, IQR)")
    ax.axhline(NEAR_DRIVE, color=COL["gray"], lw=0.8, ls="--")
    if pur["t_dog_median"] is not None:
        ax.axvline(pur["t_dog_median"], color=COL["orange"], lw=1, ls=":", label=f"median dog arrival, step {pur['t_dog_median']:.0f}")
    ax.set(xlabel="time step (s)", ylabel="distance (m)", title="Pursuit of the drive point", ylim=(0, 40))
    ax.legend(frameon=False, fontsize=8)

    ax = axs[0, 2]
    v = edge["valid"]
    ax.hist(edge["kappa_sub"][v].ravel(), bins=40, range=(-1, 1), color=COL["blue"], alpha=0.7, density=True, label="κ, random 4 of 10 neighbours")
    ax.hist(edge["kappa_K"][v], bins=40, range=(-1, 1), color=COL["orange"], histtype="step", lw=1.5, density=True, label="κ, all 10 neighbours")
    ax.axvline(coh["kappa_required_worst"], color=COL["green"], lw=1.5, ls="--", label=f"κ needed, worst-case noise ({coh['kappa_required_worst']:.2f})")
    ax.axvline(coh["kappa_required_typical"], color=COL["green"], lw=1.5, ls=":", label=f"κ needed, no noise ({coh['kappa_required_typical']:.2f})")
    ax.set(xlabel="inward component κ of the edge sheep's attraction", ylabel="density", title="Cohesion inequality: κ of the lateral-most sheep")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axs[1, 0]
    labels = ["< −0.5", "−0.5–0", "0–0.5", "0.5–1", "> 1"]
    vals = [x if x is not None else 0 for x in coh["realized_inward_by_margin_bin"]]
    ax.bar(labels, vals, color=COL["blue"], width=0.6)
    for i, (val, n) in enumerate(zip(vals, coh["count_by_margin_bin"])):
        ax.text(i, val + 0.02, f"n={n}", ha="center", fontsize=7.5, color=COL["gray"])
    ax.axhline(0.5, color=COL["gray"], lw=0.8, ls="--")
    ax.set(xlabel="predicted lateral margin  w_att·κ − w_dog·(dog term)", ylabel="fraction of steps the edge sheep moved inward", ylim=(0, 1.08),
           title="Does the inequality predict the realized step?")

    ax = axs[1, 1]
    lam = [float(k) for k in sw["lyapunov"]]
    ax.plot(lam, [sw["lyapunov"][str(l)]["drive"] for l in lam], "o-", color=COL["blue"], label="published: drive steps (no collects occur)")
    for r_dog, ls in (("12.0", "-"), ("20.0", "--")):
        L = swr[r_dog]["lyapunov"]
        ax.plot(lam, [L[str(l)]["drive"] for l in lam], "s" + ls, color=COL["green"], label=f"w_att = {W_ATT_SWITCH}, r_dog = {float(r_dog):g}: drive steps")
        ax.plot(lam, [L[str(l)]["collect"] for l in lam], "^" + ls, color=COL["orange"], label=f"w_att = {W_ATT_SWITCH}, r_dog = {float(r_dog):g}: collect steps")
    ax.set(xlabel="λ in  V = ρ + λ·max spread", ylabel="fraction of steps with ΔV ≤ 0", ylim=(0, 1.02), title="Switching: a common Lyapunov candidate")
    ax.legend(frameon=False, fontsize=7)

    ax = axs[1, 2]
    rows = sw["sweeps"]["v_dog"]
    ax.plot([r["v_dog"] for r in rows], [r["success_rate"] for r in rows], "o-", color=COL["blue"], label="reached target")
    ax.plot([r["v_dog"] for r in rows], [r["split_rate"] for r in rows], "s-", color=COL["orange"], label="split at end (max spread ≥ l_sep)")
    ax.plot([r["v_dog"] for r in rows], [r["coverage_frac"] for r in rows], "^-", color=COL["green"], label="steps where every sheep sees the dog")
    ax.plot([r["v_dog"] for r in rows], [r["P_par_median"] for r in rows], "v-", color=COL["purple"], label="flock polarization P∥ (drive steps)")
    ax.axvline(p.v_sheep * (rea["P_par_on_drive_median"] or 1), color=COL["gray"], lw=1, ls=":", label="flock speed v_sheep·P∥")
    ax.set(xlabel="dog speed v_dog (m/s)", ylabel="fraction of runs", ylim=(-0.02, 1.05), title="Condition 1: speed advantage")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[2, 0]
    rows = sw["sweeps"]["r_dog"]
    ax.plot([r["r_dog"] for r in rows], [r["success_rate"] for r in rows], "o-", color=COL["blue"], label="reached target")
    ax.plot([r["r_dog"] for r in rows], [r["coverage_frac"] for r in rows], "^-", color=COL["green"], label="steps where every sheep sees the dog")
    need = p.l_drive + (coh["front_depth_median"] or 0) + (pur["hover_gap_median"] or 0)
    ax.axvline(need, color=COL["gray"], lw=1, ls=":", label=f"l_drive + front depth + hover ({need:.1f} m)")
    ax.set(xlabel="notice radius r_dog (m)", ylabel="fraction", ylim=(-0.02, 1.05), title="Condition 2: coverage")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[2, 1]
    rows = sw["sweeps"]["w_att"]
    ax.plot([r["w_att"] for r in rows], [r["success_rate"] for r in rows], "o-", color=COL["blue"], label="reached target")
    ax.plot([r["w_att"] for r in rows], [r["split_rate"] for r in rows], "s-", color=COL["orange"], label="split at end")
    ax.axvline(coh["w_att_required_worst"], color=COL["gray"], lw=1, ls=":", label=f"w_att needed, worst-case noise ({coh['w_att_required_worst']:.2f})")
    ax.axvline(coh["w_att_required_typical"], color=COL["gray"], lw=1, ls="--", label=f"w_att needed, no noise ({coh['w_att_required_typical']:.2f})")
    ax.set(xlabel="attraction weight w_att", ylabel="fraction of runs", ylim=(-0.02, 1.05), title="Condition 3: cohesion inequality")
    ax.legend(frameon=False, fontsize=8)

    ax = axs[2, 2]
    N = [r["n_sheep"] for r in sizes]
    ax.plot(N, [r["success_rate"] for r in sizes], "o-", color=COL["blue"], label="reached target")
    ax.plot(N, [r["coverage_frac"] for r in sizes], "^-", color=COL["green"], label="steps where every sheep sees the dog")
    ax.plot(N, [r["split_rate"] for r in sizes], "s-", color=COL["orange"], label="split at end")
    ax.plot(N, [r.get("kappa_sub_p05", np.nan) for r in sizes], "v-", color=COL["purple"], label="κ, 5th percentile")
    ax.plot(N, [min(1.0, r["l_drive"] / g["p"].r_dog) for r in sizes], ":", color=COL["gray"], label="l_drive / r_dog")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="flock size N (K = 10, n_att = 4, r_dog = 12 fixed)", ylabel="fraction / cosine", ylim=(-0.05, 1.05), title="Condition 4: large flocks fail by coverage, not splitting")
    ax.legend(frameon=False, fontsize=7.5)

    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Closed-loop convergence of the published model: the cascade argument and its four conditions", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    rng = np.random.default_rng(SEED + 7)
    print("base run", flush=True)
    res = simulate(base(), n_runs=R, seed=SEED)
    g, mode = geometry(res), res.mode
    t_reach = reach_times(g)
    pur = pursuit(g, mode, t_reach)
    rea = reaching(g, mode, t_reach, pur["t_dog"])
    post = post_reach(g, mode, t_reach)
    edge = edge_terms(res, g, mode, t_reach, rng)
    coh = cohesion(edge, g["p"])
    sw = switching(g, mode, t_reach)
    rel = release(res, g, t_reach)
    print(f"  success {rea['success_rate']:.2f}, dog arrival step {pur['t_dog_median']}, closing {pur['closing_rate_m_per_step']} "
          f"(pred {pur['closing_predicted']}), slope {rea['slope_median']} (pred {rea['slope_predicted_median']})", flush=True)
    print("sweeps", flush=True)
    sweeps = {k: sweep_per_run(k, v, rng) for k, v in SWEEPS.items()}
    for k, rows in sweeps.items():
        print("  " + k + ": " + ", ".join(f"{r[k]:g}->{r['success_rate']:.2f}" for r in rows), flush=True)
    sizes = sweep_sizes(rng)
    print("switching-rich runs", flush=True)
    swr = switching_rich(rng)
    sw["sweeps"] = sweeps
    p = g["p"]
    out = dict(
        setup=dict(start_m=START, steps=T, runs=R, seed=SEED, arrive_m=ARRIVE, params=dict(
            n_sheep=p.n_sheep, v_sheep=p.v_sheep, v_dog=p.v_dog, r_dog=p.r_dog, w_att=p.w_att, w_dog=p.w_dog, noise=p.noise,
            k_perceive=p.k_perceive, n_att=p.n_att, l_drive=fl(p.l_drive), l_sep=fl(p.l_sep))),
        pursuit={k: v for k, v in pur.items() if k != "t_dog"},
        reaching=rea,
        post_reach=post,
        cohesion=coh,
        switching={k: v for k, v in sw.items() if k != "sweeps"},
        release=rel,
        switching_rich=swr,
        sweeps=sweeps,
        sizes=sizes,
        seconds=fl(time.time() - tic),
    )
    (OUT / "closed_loop.json").write_text(json.dumps(out, indent=1))
    (OUT / "figures").mkdir(exist_ok=True)
    figure(g, mode, t_reach, pur, rea, coh, edge, sw, sizes, swr, OUT / "figures" / "closed_loop.png")
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
