"""A real dog steering a Jadhav flock along a straight line and a circular arc.

The sheep are the published model (they stop when no dog is within r_dog). The dog moves at
most v_dog per step and is controlled by a cascade built from the measured heading law of
experiments/steering_gain.py:

  guidance   theta_d = theta_path - atan(e / LOOK), e the signed cross-track error (left > 0)
  heading    omega_cmd = V kappa + K_THETA wrap(theta_d - theta)
  bearing    phi_d = delta^-1(omega_cmd), from the measured delta(phi) (|phi_d| <= 60°)
  dog        aim h_d = c + d e_beta with beta = theta + pi + phi_d, moved toward
             h_d + (flock step) by at most v_dog

The bearing phi is measured from directly behind the flock, positive when the dog is behind
on the right, so the flock turns left: theta_{t+1} - theta_t = delta(phi_t).

Actuator feasibility (derived in docs/steer_and_shape.md §8): holding standoff d and bearing
phi through a steady turn needs a dog speed
    |h'| = sqrt(V^2 + 2 V d omega sin(phi) + d^2 omega^2),  omega = delta(phi), V = v P(phi),
so a path of radius R is feasible in steady state only if V/delta(phi) <= R for some phi with
|h'| <= v_dog. This gives R_min(v_dog, d, w_ali); tighter arcs are predicted to leave an
outward error of about R_min - R.

Compared with what the measured laws predict: the heading one step ahead, the polarization
P_eq(phi), and the shape moments (sd along and across the heading, centroid second moment);
and checked that the flock stays where the laws were measured (|phi| <= 60°, polarization not
collapsing, every sheep in the dog's range).

    .venv/bin/python experiments/path_tracking.py
Writes results/path_tracking.json and results/figures/path_tracking.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
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

from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
N, T_DRIVE, T, REPS, SEED = 14, 150, 170, 16, 31
FAR = 2000.0
L1, L2 = 25.0, 40.0  # straight before and after the arc (m)
D_STAND, W_ATT = 7.5, 1.5  # the cell of the steering-gain table the controller uses
W_ALI = (0.0, 1.3, 3.0)
V_DOG = (1.5, 3.0)
R_PATH = (3.0, 5.0, 8.0, 12.0, 20.0)
LOOK, K_THETA, PHI_LIM = 6.0, 0.3, np.deg2rad(60.0)
SPLIT_M = 15.0
P0 = Params()
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ the measured laws

def laws():
    """delta(phi), V(phi) = flock speed, P(phi), shape moments, per w_ali, from the steering-gain table (phi in [0, 150°])."""
    g = json.loads((OUT / "steering_gain.json").read_text())
    out = {}
    for a in W_ALI:
        rs = sorted([r for r in g["rows"] if r["w_ali"] == a and r["d"] == D_STAND and r["w_att"] == W_ATT], key=lambda r: r["phi_deg"])
        out[a] = {k: np.array([r[k] for r in rs], float) for k in ("rate", "speed", "polarization", "s_par", "s_perp")}
        out[a]["phi"] = np.deg2rad([r["phi_deg"] for r in rs])
    return out


def odd(table, phi, key):
    """Interpolate an odd (rate) or even (everything else) function of the bearing."""
    x = np.interp(np.abs(phi), table["phi"], table[key])
    return np.sign(phi) * x if key == "rate" else x


def inverse_rate(table, omega):
    """Bearing that gives turn rate omega, from the monotone part of delta up to 60°."""
    keep = table["phi"] <= PHI_LIM + 1e-9
    ph, rt = table["phi"][keep], np.maximum.accumulate(table["rate"][keep])
    return np.sign(omega) * np.interp(np.abs(omega), rt, ph)  # saturates at 60°


def feasibility(table, d, v_dog):
    """Smallest steady turning radius with standoff d and dog speed v_dog, and the bearing it needs."""
    g = np.linspace(0, PHI_LIM, 600)
    om, V = odd(table, g, "rate"), odd(table, g, "speed")
    need = np.sqrt(V ** 2 + 2 * V * d * om * np.sin(g) + d ** 2 * om ** 2)
    ok = need <= v_dog
    i = np.flatnonzero(ok)[-1] if ok.any() else 0
    ideal = V[-1] / om[-1]
    return dict(phi_max=float(np.rad2deg(g[i])), omega_max=float(om[i]), R_min=float(V[i] / max(om[i], 1e-9)), R_min_ideal=float(ideal),
                bearing_rate_margin_at_0=float(np.sqrt(max(v_dog ** 2 - V[0] ** 2, 0)) / d))


# ------------------------------------------------------------------ the path

def path_frame(p, Rp, seg_min=None):
    """Signed cross-track error (left > 0), path tangent angle, curvature and arc length for points p (R, 2).

    Segments only advance: seg_min (each run's previous segment) keeps a centroid that drifts more
    than R to the left of the final straight from being read as back on the first straight."""
    x, y = p[:, 0], p[:, 1]
    lo = np.zeros(len(p), int) if seg_min is None else seg_min
    e, th, k, s = np.empty(len(p)), np.empty(len(p)), np.zeros(len(p)), np.empty(len(p))
    seg1 = (x <= 0) & (lo <= 0)
    seg3 = ~seg1 & ((y >= Rp) | (lo >= 2))
    arc = ~seg1 & ~seg3
    e[seg1], th[seg1], s[seg1] = y[seg1], 0.0, x[seg1] + L1
    rho = np.hypot(x[arc], y[arc] - Rp[arc])
    alpha = np.arctan2(y[arc] - Rp[arc], x[arc])
    e[arc], th[arc], k[arc], s[arc] = Rp[arc] - rho, alpha + np.pi / 2, 1 / Rp[arc], L1 + Rp[arc] * (alpha + np.pi / 2)
    e[seg3], th[seg3], s[seg3] = Rp[seg3] - x[seg3], np.pi / 2, L1 + np.pi * Rp[seg3] / 2 + (y[seg3] - Rp[seg3])
    return e, th, k, s, np.where(seg1, 0, np.where(arc, 1, 2))


# ------------------------------------------------------------------ simulation

def start_states(rows, reps=REPS, seed=SEED):
    """A flock driven for T_DRIVE steps by the published dog, moved to (-L1, 0) heading +x with the dog behind it."""
    R = len(rows) * reps
    per = {"w_ali": np.repeat([r["w_ali"] for r in rows], reps), "w_att": np.full(R, W_ATT)}
    drive = simulate(Params(n_sheep=N, box_length=FAR, n_steps=T_DRIVE), n_runs=R, seed=seed, per_run=dict(per, l_sep=np.full(R, 1e6)))
    pos, vel = drive.pos_s[..., -1], drive.vel_s[..., -1]
    G = pos.mean(1)
    e1 = unit(vel.mean(1))
    c, s = e1[:, 0], -e1[:, 1]  # rotation taking e1 to (1, 0)
    Rm = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1)
    pos = np.einsum("rij,rnj->rni", Rm, pos - G[:, None]) + np.array([-L1, 0.0])
    vel = np.einsum("rij,rnj->rni", Rm, vel)
    return per, dict(pos_s=pos, vel_s=vel), pos.mean(1) + np.array([-D_STAND, 0.0])


def run(tab, rows=None, reps=REPS, seed=SEED):
    """The closed loop for every setting in rows (default: the full grid), reps runs each."""
    rows = rows or [dict(w_ali=a, v_dog=v, R_path=rp) for a, v, rp in itertools.product(W_ALI, V_DOG, R_PATH)]
    R = len(rows) * reps
    per, st, h = start_states(rows, reps, seed)
    w_ali = per["w_ali"]
    v_dog = np.repeat([r["v_dog"] for r in rows], reps)
    Rp = np.repeat([r["R_path"] for r in rows], reps)
    p1 = Params(n_sheep=N, n_steps=2)
    rec = {k: np.full((R, T), np.nan) for k in ("e", "theta", "theta_path", "phi", "phi_cmd", "P", "V", "s_par", "s_perp", "pred_theta",
                                                "pred_P", "coverage", "seg", "progress", "dog_speed", "spread")}
    cen = np.empty((R, 2, T + 1))
    c_prev = st["pos_s"].mean(1)
    seg_min = np.zeros(R, int)
    for t in range(T):
        x, hv = st["pos_s"], st["vel_s"]
        c = x.mean(1)
        m = hv.mean(1)
        P = np.linalg.norm(m, axis=-1)
        th = np.arctan2(m[:, 1], m[:, 0])
        V = np.linalg.norm(c - c_prev, axis=-1) if t else P * P0.v_sheep
        e, thp, kap, s, seg = path_frame(c, Rp, seg_min)
        seg_min = seg
        # cascade: guidance -> turn rate -> bearing -> dog aim
        th_d = thp - np.arctan(e / LOOK)
        omega = V * kap + K_THETA * wrap(th_d - th)
        phi_d = np.empty(R)
        for a in W_ALI:
            sel = w_ali == a
            phi_d[sel] = inverse_rate(tab[a], omega[sel])
        beta = th + np.pi + phi_d
        aim = c + D_STAND * np.stack([np.cos(beta), np.sin(beta)], -1) + V[:, None] * np.stack([np.cos(th), np.sin(th)], -1)
        step = aim - h
        n = np.linalg.norm(step, axis=-1, keepdims=True)
        h_new = h + step * np.minimum(1.0, v_dog[:, None] / np.maximum(n, 1e-9))
        # the bearing the sheep actually see, relative to the state they update from
        rel = h_new - c
        phi = wrap(np.arctan2(rel[:, 1], rel[:, 0]) - th - np.pi)
        e_h = np.stack([np.cos(th), np.sin(th)], -1)
        e_l = np.stack([-e_h[:, 1], e_h[:, 0]], -1)
        xi = x - c[:, None]
        for k, v in (("e", e), ("theta", th), ("theta_path", thp), ("phi", phi), ("phi_cmd", phi_d), ("P", P), ("V", V), ("seg", seg), ("progress", s),
                     ("dog_speed", np.linalg.norm(h_new - h, axis=-1)),
                     ("s_par", np.sqrt((np.einsum("rnc,rc->rn", xi, e_h) ** 2).mean(1))),
                     ("s_perp", np.sqrt((np.einsum("rnc,rc->rn", xi, e_l) ** 2).mean(1))),
                     ("coverage", (np.linalg.norm(x - h_new[:, None], axis=-1) <= P0.r_dog).mean(1)),
                     ("spread", np.linalg.norm(xi, axis=-1).max(1))):
            rec[k][:, t] = v
        for a in W_ALI:
            sel = w_ali == a
            rec["pred_theta"][sel, t] = th[sel] + odd(tab[a], phi[sel], "rate")
            rec["pred_P"][sel, t] = odd(tab[a], phi[sel], "polarization")
        cen[..., t] = c
        res = simulate(p1, n_runs=R, seed=seed * 1000 + t, per_run=per, init=st, dog_path=np.stack([h_new, h_new], -1))
        c_prev, h = c, h_new
        st = dict(pos_s=res.pos_s[..., -1], vel_s=res.vel_s[..., -1])
    rec["cx"], rec["cy"] = cen[:, 0, :T], cen[:, 1, :T]
    return rows, rec


def summarize(rows, rec, tab):
    """Per setting, over the runs that start as one flock (a few split during the warm-up drive, which has no collecting)."""
    out = []
    one_step = wrap(rec["theta"][:, 1:] - rec["pred_theta"][:, :-1])  # actual minus predicted heading, one step on
    turn = wrap(rec["theta"][:, 1:] - rec["theta"][:, :-1])
    M2 = rec["s_par"] ** 2 + rec["s_perp"] ** 2
    for i, r in enumerate(rows):
        sl = np.arange(i * REPS, (i + 1) * REPS)
        start_split = rec["spread"][sl, 0] > SPLIT_M
        sl = sl[~start_split]
        f = feasibility(tab[r["w_ali"]], D_STAND, r["v_dog"])
        on_arc = rec["seg"][sl] == 1
        after = rec["seg"][sl] == 2
        straight = (rec["seg"][sl] == 0) & (np.arange(T) >= 20)
        e = np.abs(rec["e"][sl])
        done = rec["progress"][sl, -1] >= L1 + np.pi * r["R_path"] / 2 + 0.5 * L2
        arc_steps = on_arc[:, :-1]
        valid = (np.abs(rec["phi"][sl]) <= PHI_LIM) & (rec["coverage"][sl] >= 0.999)
        mean_abs_phi_arc = np.nanmean(np.where(on_arc, np.abs(rec["phi"][sl]), np.nan))
        per_run = lambda x, m: np.nanmean(np.where(m, x, np.nan), 1)
        out.append(dict(**r, **f, n_runs=int(len(sl)), split_before_start=int(start_split.sum()),
                        predicted_max_error=max(0.0, f["R_min"] - r["R_path"]),
                        max_error_median=fl(np.median(np.max(np.where(on_arc | after, e, 0), 1))),
                        max_error_p90=fl(np.percentile(np.max(np.where(on_arc | after, e, 0), 1), 90)),
                        rms_error_straight=fl(np.sqrt(np.nanmean(np.where(straight, e, np.nan) ** 2))),
                        completed=float(done.mean()), split_during=float((rec["spread"][sl, -1] > SPLIT_M).mean()),
                        heading_err_rms_deg=fl(np.rad2deg(np.sqrt(np.nanmean(one_step[sl] ** 2)))),
                        heading_err_rms_arc_deg=fl(np.rad2deg(np.sqrt(np.nanmean(np.where(arc_steps, one_step[sl], np.nan) ** 2)))),
                        turn_ratio_arc=fl(np.nansum(np.where(arc_steps, turn[sl], 0)) / np.nansum(np.where(arc_steps, rec["pred_theta"][sl, :-1] - rec["theta"][sl, :-1], 0))),
                        P_err_rms=fl(np.sqrt(np.nanmean((rec["P"][sl, 1:] - rec["pred_P"][sl, :-1]) ** 2))),
                        P_arc=fl(np.nanmean(np.where(on_arc, rec["P"][sl], np.nan))), P_arc_pred=fl(np.nanmean(np.where(on_arc, rec["pred_P"][sl], np.nan))),
                        mean_abs_phi_arc_deg=fl(np.rad2deg(mean_abs_phi_arc)),
                        s_par_arc=fl(np.nanmedian(per_run(rec["s_par"][sl], on_arc))), s_perp_arc=fl(np.nanmedian(per_run(rec["s_perp"][sl], on_arc))),
                        s_par_straight=fl(np.nanmedian(per_run(rec["s_par"][sl], straight))), s_perp_straight=fl(np.nanmedian(per_run(rec["s_perp"][sl], straight))),
                        s_par_arc_pred=fl(odd(tab[r["w_ali"]], mean_abs_phi_arc, "s_par")), s_perp_arc_pred=fl(odd(tab[r["w_ali"]], mean_abs_phi_arc, "s_perp")),
                        M2_straight=fl(np.nanmedian(per_run(M2[sl], straight))), M2_arc=fl(np.nanmedian(per_run(M2[sl], on_arc))),
                        M2_max=fl(np.median(np.max(np.where(on_arc | after, M2[sl], 0), 1))),  # median over runs of each run's largest
                        M2_max_p90=fl(np.percentile(np.max(np.where(on_arc | after, M2[sl], 0), 1), 90)),
                        M2_max_all=fl(np.max(np.where(on_arc | after, M2[sl], 0))),  # largest over all runs and steps
                        valid_frac=float(valid.mean()), dog_saturated_frac=float((rec["dog_speed"][sl] >= 0.999 * r["v_dog"]).mean()),
                        # for the mean-square tracking bound (docs/jadhav_dilorenzo_bridge.md §4.1)
                        rms_error_arc=fl(np.sqrt(np.nanmean(np.where(on_arc, e, np.nan) ** 2))),
                        rms_error_after=fl(np.sqrt(np.nanmean(np.where(after, e, np.nan) ** 2))),
                        heading_err_mean_arc_rad=fl(np.nanmean(np.where(arc_steps, one_step[sl], np.nan))),
                        heading_err_mean_straight_rad=fl(np.nanmean(np.where(straight[:, :-1], one_step[sl], np.nan))),
                        heading_err_sd_rad=fl(np.nanstd(one_step[sl])),
                        speed_range=[fl(np.nanpercentile(rec["V"][sl][:, 5:], 5)), fl(np.nanpercentile(rec["V"][sl][:, 5:], 95))]))
    return out


def figure(rows, rec, summ):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10))
    ax = axs[0, 0]
    for a, col in zip(W_ALI, ("orange", "blue", "green")):
        ss = [s for s in summ if s["w_ali"] == a]
        ax.scatter([s["predicted_max_error"] for s in ss], [s["max_error_median"] for s in ss], s=[40 + 40 * (s["v_dog"] == 3.0) for s in ss],
                   color=COL[col], edgecolor="white", lw=0.8, label=f"w_ali {a:g}")
    m = max(max(s["max_error_median"] for s in summ), max(s["predicted_max_error"] for s in summ)) * 1.05
    ax.plot([0, m], [0, m], color=COL["gray"], lw=0.8, ls=":")
    ax.set(xlabel="predicted outward error max(0, R_min − R_path) (m)", ylabel="max cross-track error after the arc starts, median (m)",
           title="Arcs tighter than the actuator allows (small dots v_dog 1.5, big 3)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 1]
    for (a, v, rp), col in zip(((1.3, 1.5, 3.0), (1.3, 1.5, 8.0), (1.3, 1.5, 20.0), (3.0, 1.5, 5.0)), ("orange", "blue", "green", "purple")):
        i = next(j for j, r in enumerate(rows) if r["w_ali"] == a and r["v_dog"] == v and r["R_path"] == rp)
        sl = slice(i * REPS, (i + 1) * REPS)
        th = np.linspace(-np.pi / 2, 0, 50)
        ax.plot(np.r_[-L1, rp * np.cos(th), rp, rp], np.r_[0, rp + rp * np.sin(th), rp, rp + L2], color=COL[col], lw=1, ls="--")
        for k in range(3):
            ax.plot(rec["cx"][sl][k], rec["cy"][sl][k], color=COL[col], lw=1.2, alpha=0.8)
        ax.plot([], [], color=COL[col], label=f"w_ali {a:g}, R {rp:g} m")
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Centroid paths (3 runs each, dashed: the path), v_dog 1.5")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    for a, col in zip(W_ALI, ("orange", "blue", "green")):
        ss = [s for s in summ if s["w_ali"] == a]
        ax.scatter([s["P_arc_pred"] for s in ss], [s["P_arc"] for s in ss], color=COL[col], marker="o", s=45, edgecolor="white", label=f"polarization, w_ali {a:g}")
        ax.scatter([s["s_perp_arc_pred"] / 3 for s in ss], [s["s_perp_arc"] / 3 for s in ss], color=COL[col], marker="s", s=35, edgecolor="white", alpha=0.6)
    ax.plot([0, 1], [0, 1], color=COL["gray"], lw=0.8, ls=":")
    ax.set(xlabel="predicted from the bearing (steering-gain table)", ylabel="measured on the arc",
           title="On the arc: polarization (●) and sd across the heading / 3 m (■)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 1]
    for a, col in zip(W_ALI, ("orange", "blue", "green")):
        ss = sorted([s for s in summ if s["w_ali"] == a and s["v_dog"] == 1.5], key=lambda s: s["R_path"])
        ax.plot([s["R_path"] for s in ss], [s["M2_arc"] for s in ss], "o-", color=COL[col], lw=2, label=f"on the arc, w_ali {a:g}")
        ax.plot([s["R_path"] for s in ss], [s["M2_max"] for s in ss], "^:", color=COL[col], lw=1, ms=5)
        ax.axhline(np.median([s["M2_straight"] for s in ss]), color=COL[col], lw=1, ls="--")
    ax.set(xlabel="path radius (m)", ylabel="centroid second moment M2 = sd∥² + sd⊥² (m²)", ylim=(0, None),
           title="Shape through the turn, v_dog 1.5\n(dashed: straight; ▲: largest after the arc starts; medians)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "path_tracking.png", dpi=130)


def main():
    tic = time.time()
    tab = laws()
    rows, rec = run(tab)
    summ = summarize(rows, rec, tab)
    for s in summ:
        print(f"w_ali {s['w_ali']:.1f} v_dog {s['v_dog']:.1f} R {s['R_path']:4.0f}: R_min {s['R_min']:.1f} (ideal {s['R_min_ideal']:.1f}) -> pred err {s['predicted_max_error']:.1f}, "
              f"max err {s['max_error_median']:.1f} (p90 {s['max_error_p90']:.1f}), done {s['completed']:.2f}, split {s['split_during']:.2f} (excl {s['split_before_start']}) | "
              f"1-step heading err {s['heading_err_rms_deg']:.1f}° (arc {s['heading_err_rms_arc_deg']:.1f}°), turn ratio {s['turn_ratio_arc']:.2f} | "
              f"P arc {s['P_arc']:.2f} vs {s['P_arc_pred']:.2f} | sd arc {s['s_par_arc']:.2f}/{s['s_perp_arc']:.2f} vs {s['s_par_arc_pred']:.2f}/{s['s_perp_arc_pred']:.2f} "
              f"(straight {s['s_par_straight']:.2f}/{s['s_perp_straight']:.2f}), M2 {s['M2_straight']:.1f}/{s['M2_arc']:.1f}/{s['M2_max']:.1f}, "
              f"|phi| {s['mean_abs_phi_arc_deg']:.0f}° | valid {s['valid_frac']:.2f} sat {s['dog_saturated_frac']:.2f}", flush=True)
    out = dict(setup=dict(N=N, t_drive=T_DRIVE, steps=T, reps=REPS, L1=L1, L2=L2, standoff=D_STAND, w_att=W_ATT, look=LOOK, k_theta=K_THETA,
                          phi_lim_deg=float(np.rad2deg(PHI_LIM)), sheep_rule="published (stop when no dog within r_dog)"),
               summary=summ, seconds=time.time() - tic)
    (OUT / "path_tracking.json").write_text(json.dumps(out, indent=1))
    figure(rows, rec, summ)
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
