"""The steering gain: how fast one dog turns a Jadhav flock, per unit of its bearing.

A driven, polarized flock (150 steps behind the published dog) is handed to a dog held at a fixed
bearing phi from directly behind the flock and a fixed standoff d, in the flock's own frame
(the dog is re-placed every step from the flock's mean heading and centroid, by one-step
`simulate` calls with a replayed dog position; the model itself is unchanged). The flock then
turns away from the dog's side at a steady rate. Measured: that rate, the step response after
the dog jumps to phi, the flock's speed and its turning radius, and splits.

Prediction from the heading rule. In a flock turning together, each sheep's own heading (weight
w_inertia) and the copied neighbour's heading (w_ali) both lie along the flock's heading e, and
the dog's push (w_dog) along e rotated by phi. Attraction and repulsion average out along e. So
the heading turns by

    delta(phi) = atan2(w_dog sin phi, w_inertia + w_ali + w_dog cos phi)    per step,

a linear gain K = w_dog / (w_inertia + w_ali + w_dog) near phi = 0, and a maximum of
asin(w_dog / (w_inertia + w_ali)) when w_dog < w_inertia + w_ali.

    .venv/bin/python experiments/steering_gain.py
Writes results/steering_gain.json and results/figures/steering_gain.png.
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
N, T_DRIVE, T_HOLD, T_FIT, REPS, SEED = 14, 150, 50, 10, 16, 23
FAR = 2000.0
SPLIT_M = 15.0
P_STEADY = 0.8  # a turn counts as steady only while the flock keeps this share of its polarization at phi = 0
PHI_DEG = (0, 10, 20, 30, 45, 60, 90, 120, 150)
D_STAND = (5.0, 7.5, 10.0)
W_ALI = (0.0, 1.3, 3.0)
W_ATT = (1.5, 3.0)
P0 = Params()
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def predicted(phi, w_ali, w_inertia=P0.w_inertia, w_dog=P0.w_dog):
    return np.arctan2(w_dog * np.sin(phi), w_inertia + w_ali + w_dog * np.cos(phi))


def run():
    rows = [dict(phi=np.deg2rad(p), phi_deg=p, d=d, w_ali=a, w_att=c) for p, d, a, c in itertools.product(PHI_DEG, D_STAND, W_ALI, W_ATT)]
    R = len(rows) * REPS
    per = {k: np.repeat([r[k] for r in rows], REPS) for k in ("w_ali", "w_att")}
    drive = simulate(Params(n_sheep=N, box_length=FAR, n_steps=T_DRIVE), n_runs=R, seed=SEED,
                     per_run=dict(per, l_sep=np.full(R, 1e6)))
    st = dict(pos_s=drive.pos_s[..., -1], vel_s=drive.vel_s[..., -1])
    phi = np.repeat([r["phi"] for r in rows], REPS)
    d = np.repeat([r["d"] for r in rows], REPS)
    p1 = Params(n_sheep=N, n_steps=2)
    heading = np.empty((R, 2, T_HOLD + 1))
    pol = np.empty((R, T_HOLD + 1))
    G = np.empty((R, 2, T_HOLD + 1))
    spread = np.empty((R, T_HOLD + 1))
    sees = np.empty((R, T_HOLD))
    X = np.empty((R, N, 2, T_HOLD + 1))
    for t in range(T_HOLD + 1):
        X[..., t] = st["pos_s"]
        m = st["vel_s"].mean(1)
        pol[:, t] = np.linalg.norm(m, axis=-1)
        e = unit(m)
        heading[..., t], G[..., t] = e, st["pos_s"].mean(1)
        spread[:, t] = np.linalg.norm(st["pos_s"] - G[:, None, :, t], axis=-1).max(-1)
        if t == T_HOLD:
            break
        e_left = np.stack([-e[:, 1], e[:, 0]], -1)
        dog = G[..., t] + d[:, None] * (-np.cos(phi)[:, None] * e + np.sin(phi)[:, None] * e_left)
        sees[:, t] = (np.linalg.norm(st["pos_s"] - dog[:, None], axis=-1) <= P0.r_dog).mean(-1)
        res = simulate(p1, n_runs=R, seed=SEED * 1000 + t, per_run=per, init=st, dog_path=np.stack([dog, dog], -1))
        st = dict(pos_s=res.pos_s[..., -1], vel_s=res.vel_s[..., -1])
    # turn rate toward the side away from the dog (the dog sits to the left for phi > 0, so the flock turns right)
    ang = np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]), axis=-1)
    rate = -np.diff(ang, axis=-1)  # (R, T_HOLD), rad/step, positive = away from the dog
    speed = np.linalg.norm(np.diff(G, axis=-1), axis=1)
    split = spread[:, -1] > SPLIT_M
    # shape in the flock's own frame (along / across its heading), and each sheep's lateral motion relative to the flock
    xi = X - G[:, None]
    e_l = np.stack([-heading[:, 1], heading[:, 0]], 1)
    par = np.einsum("rnct,rct->rnt", xi, heading)
    perp = np.einsum("rnct,rct->rnt", xi, e_l)
    v_rel = np.diff(X, axis=-1) - np.diff(G, axis=-1)[:, None]
    v_perp = np.einsum("rnct,rct->rnt", v_rel, e_l[..., :-1])[..., T_FIT:]
    tau_perp = lag_tau(v_perp).mean(1)  # integrated autocorrelation time (steps)
    D_perp = (v_perp ** 2).mean(axis=(1, 2)) * tau_perp  # m^2 / step, per axis
    out = []
    for i, r in enumerate(rows):
        sl = slice(i * REPS, (i + 1) * REPS)
        ok = ~split[sl]
        rr = rate[sl][ok]
        steady = rr[:, T_FIT:].mean() if ok.any() else np.nan
        sp = speed[sl][ok][:, T_FIT:].mean() if ok.any() else np.nan
        resp = rr.mean(0) if ok.any() else np.full(T_HOLD, np.nan)
        # step response: steps until the mean rate first reaches 63% of its steady value
        t63 = int(np.argmax(np.abs(resp) >= 0.63 * abs(steady))) if ok.any() and abs(steady) > 1e-3 else None
        out.append(dict(phi_deg=r["phi_deg"], d=r["d"], w_ali=r["w_ali"], w_att=r["w_att"], rate=fl(steady), rate_sd=fl(rr[:, T_FIT:].mean(1).std()) if ok.any() else None,
                        predicted=fl(predicted(r["phi"], r["w_ali"])), speed=fl(sp), radius=fl(sp / steady) if steady > 1e-3 else None,
                        coverage=fl(sees[sl][ok].mean()) if ok.any() else None, split_rate=float(split[sl].mean()), t63=t63,
                        polarization=fl(pol[sl][ok][:, T_FIT:].mean()) if ok.any() else None,
                        polarization_response=[fl(x) for x in pol[sl][ok].mean(0)[:21]] if ok.any() else None,
                        s_par=fl(np.sqrt((par[sl][ok][..., T_FIT:] ** 2).mean())) if ok.any() else None,
                        s_perp=fl(np.sqrt((perp[sl][ok][..., T_FIT:] ** 2).mean())) if ok.any() else None,
                        tau_perp=fl(tau_perp[sl][ok].mean()) if ok.any() else None, D_perp=fl(D_perp[sl][ok].mean()) if ok.any() else None,
                        response=[fl(x) for x in resp[:20]]))
    return out


def lag_tau(x, max_lag=15):
    """Integrated autocorrelation time (steps) of x (R, N, T), summed up to the first zero crossing."""
    x = x - x.mean(-1, keepdims=True)
    c0 = np.maximum((x * x).mean(-1), 1e-12)
    tau, alive = np.full(x.shape[:2], 0.5), np.ones(x.shape[:2], bool)
    for k in range(1, max_lag):
        c = (x[..., :-k] * x[..., k:]).mean(-1) / c0
        alive &= c > 0
        tau += np.where(alive, c, 0.0)
    return tau


def polarization_lag(rows):
    """Steps for the flock's polarization to cover 63% of its change after the dog jumps from behind to phi."""
    out = []
    for r in rows:
        pr = r.get("polarization_response")
        if not pr or r["phi_deg"] == 0:
            continue
        p = np.array(pr)
        dp = r["polarization"] - p[0]
        t63 = int(np.argmax(np.abs(p - p[0]) >= 0.63 * abs(dp))) if abs(dp) > 0.02 else None
        out.append(dict(phi_deg=r["phi_deg"], d=r["d"], w_ali=r["w_ali"], w_att=r["w_att"], dP=fl(dp), t63=t63))
    return out


def gains(rows):
    """Linear gain: slope of the steady rate against phi through the origin, fitted on phi <= 30 degrees."""
    out = []
    for d, a, c in itertools.product(D_STAND, W_ALI, W_ATT):
        rs = [r for r in rows if r["d"] == d and r["w_ali"] == a and r["w_att"] == c and r["phi_deg"] <= 30 and r["rate"] is not None]
        ph = np.deg2rad([r["phi_deg"] for r in rs])
        y = np.array([r["rate"] for r in rs])
        k = float((ph * y).sum() / (ph * ph).sum())
        allr = [r for r in rows if r["d"] == d and r["w_ali"] == a and r["w_att"] == c and r["rate"] is not None]
        p0 = next(r["polarization"] for r in allr if r["phi_deg"] == 0)
        steady = [r for r in allr if r["polarization"] >= P_STEADY * p0]  # beyond these the flock reverses rather than turns
        mx = max(steady, key=lambda r: r["rate"])
        out.append(dict(d=d, w_ali=a, w_att=c, gain=k, gain_predicted=float(P0.w_dog / (P0.w_inertia + a + P0.w_dog)),
                        max_rate=mx["rate"], max_rate_phi=mx["phi_deg"], max_rate_predicted=float(predicted(np.deg2rad(np.arange(0, 181)), a).max()),
                        min_radius=mx["radius"], max_steady_phi=max(r["phi_deg"] for r in steady),
                        reverses_from=min([r["phi_deg"] for r in allr if r["polarization"] < P_STEADY * p0], default=None),
                        polarization_at_0=p0,
                        drift_at_0=next(r["rate"] for r in allr if r["phi_deg"] == 0)))
    return out


def figure(rows, gs):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ph = np.linspace(0, 150, 151)
    ax = axs[0]
    for a, col in zip(W_ALI, ("orange", "blue", "green")):
        rs = sorted([r for r in rows if r["w_ali"] == a and r["d"] == 7.5 and r["w_att"] == 1.5], key=lambda r: r["phi_deg"])
        ax.plot([r["phi_deg"] for r in rs], [np.rad2deg(r["rate"]) for r in rs], "o", color=COL[col], ms=7, label=f"measured, w_ali {a:g}")
        ax.plot(ph, np.rad2deg(predicted(np.deg2rad(ph), a)), color=COL[col], lw=1.6, ls="--")
    ax.set(xlabel="dog's bearing from directly behind (deg)", ylabel="flock turn rate away from the dog (deg/step)",
           xlim=(-5, 155), ylim=(0, 140),
           title="Turn rate vs bearing (standoff 7.5 m, w_att 1.5)\ndashed: atan2(w_dog sin φ, w_in + w_ali + w_dog cos φ)")
    ax.axvspan(75, 155, color=COL["gray"], alpha=0.12, lw=0)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    x = np.arange(len(W_ALI))
    for j, (d, col) in enumerate(zip(D_STAND, ("purple", "blue", "gray"))):
        g = [np.mean([r["gain"] for r in gs if r["d"] == d and r["w_ali"] == a]) for a in W_ALI]
        ax.bar(x + (j - 1) * 0.26, g, width=0.24, color=COL[col], label=f"measured, standoff {d:g} m")
    ax.plot(x, [P0.w_dog / (P0.w_inertia + a + P0.w_dog) for a in W_ALI], "k_", ms=40, mew=2, label="predicted w_dog/(w_in + w_ali + w_dog)")
    ax.set_xticks(x, [f"w_ali {a:g}" for a in W_ALI])
    ax.set(ylabel="linear steering gain (rad/step per rad of bearing)", ylim=(0, 0.95), title="Alignment lowers the steering gain")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axs[2]
    for a, col in zip(W_ALI, ("orange", "blue", "green")):
        rs = sorted([r for r in rows if r["w_ali"] == a and r["d"] == 7.5 and r["w_att"] == 1.5], key=lambda r: r["phi_deg"])
        ax.plot([r["phi_deg"] for r in rs], [r["polarization"] for r in rs], "o-", color=COL[col], lw=2, label=f"w_ali {a:g}")
    ax.axvspan(75, 155, color=COL["gray"], alpha=0.12, lw=0)
    ax.text(80, 0.08, "the flock reverses\nrather than turns", fontsize=8, color="#3a3a38")
    ax.set(xlabel="dog's bearing from directly behind (deg)", ylabel="polarization = flock speed / sheep speed", ylim=(0, 1),
           title="The cost of steering: polarization (and speed) falls")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "steering_gain.png", dpi=130)


def main():
    tic = time.time()
    rows = run()
    gs = gains(rows)
    for g in gs:
        print(f"d {g['d']:4.1f} w_ali {g['w_ali']:.1f} w_att {g['w_att']:.1f}: gain {g['gain']:.3f} (pred {g['gain_predicted']:.3f}), "
              f"max {np.rad2deg(g['max_rate']):.1f}°/step at {g['max_rate_phi']}° (pred {np.rad2deg(g['max_rate_predicted']):.1f}°), "
              f"min radius {g['min_radius']:.1f} m, polarized up to {g['max_steady_phi']}°, reverses from {g['reverses_from']}°, "
              f"drift at 0 {np.rad2deg(g['drift_at_0']):.2f}°/step", flush=True)
    out = dict(setup=dict(N=N, t_drive=T_DRIVE, t_hold=T_HOLD, t_fit_from=T_FIT, reps=REPS, phi_deg=PHI_DEG, standoff=D_STAND, w_ali=W_ALI,
                          w_att=W_ATT, r_dog=P0.r_dog, w_inertia=P0.w_inertia, w_dog=P0.w_dog),
               rows=rows, gains=gs, polarization_lag=polarization_lag(rows), seconds=time.time() - tic)
    (OUT / "steering_gain.json").write_text(json.dumps(out, indent=1))
    figure(rows, gs)
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
