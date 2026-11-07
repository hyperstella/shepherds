"""The hold radius of the density controller is the tightest steady turn the dog can hold.

Under the second-moment controller (shepherd/density_control.py) a flock at the goal does not
stop: it is turned round and round, and the radius of the circle it walks sets how far the
centroid wanders. The steady-turn feasibility of docs/steer_and_shape.md §8 (bridge §4.1) gives
that radius from the measured heading law: with the dog at standoff d and bearing φ the flock
turns at ω = δ(φ) and moves at V = v P_eq(φ), so it circles at R = V/ω, and holding that
position on the circle needs a dog speed

    s(φ, d) = sqrt(V² + 2 V d ω sin φ + d² ω²).

The smallest radius a dog of speed v_dog can hold is therefore
    R_min(v_dog, d) = min { V(φ)/δ(φ) : s(φ, d) ≤ v_dog, |φ| ≤ 60° },
computed from the steering-gain table (results/steering_gain.json, baseline weights, standoffs
5, 7.5 and 10 m) by experiments/path_tracking.py's `feasibility`. Prediction: in a hold the
flock's turning radius V/|ω|, and with it the centroid's distance to the goal, is about
R_min(v_dog, d) at the standoff the controller actually keeps, and never much below it.

Test: holds of 300 steps at five dog speeds, published sheep and the paper's always-moving
sheep, 24 runs each, from a driven flock translated to the goal. Measured over steps 60–300:
the centroid's distance ρ, its speed V and turn rate ω per step, the dog's standoff and bearing,
and how far the flock circles (accumulated turn).

    .venv/bin/python experiments/hold_condition.py
Writes results/hold_condition.json and results/figures/hold_condition.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import path_tracking as PT  # noqa: E402
from shepherd import density_control as DC  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
SPEEDS = (1.5, 2.0, 2.5, 3.0, 4.0)
SHEEP = ("published", "always_moving")
R, T, T0, SEED = 24, 300, 60, 21
STANDOFFS = (5.0, 7.5, 10.0)
W_ALI, W_ATT = 1.3, 1.5
R_GOAL = 5.0
S_GOAL = DC.goal_sigma(R_GOAL)
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a", "teal": "#1b9aaa"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ the prediction

def tables():
    """The measured heading law per standoff: δ(φ), V(φ), P(φ) at the baseline weights."""
    g = json.loads((OUT / "steering_gain.json").read_text())
    out = {}
    for d in STANDOFFS:
        rs = sorted([r for r in g["rows"] if r["d"] == d and r["w_ali"] == W_ALI and r["w_att"] == W_ATT], key=lambda r: r["phi_deg"])
        out[d] = {k: np.array([r[k] for r in rs], float) for k in ("rate", "speed", "polarization")}
        out[d]["phi"] = np.deg2rad([r["phi_deg"] for r in rs])
    return out


def r_min(tabs, d, v_dog):
    """R_min(v_dog, d), with the law interpolated linearly between the measured standoffs."""
    ds = np.array(STANDOFFS)
    d = float(np.clip(d, ds[0], ds[-1]))
    vals = np.array([PT.feasibility(tabs[s], s, v_dog)["R_min"] for s in STANDOFFS])
    return float(np.interp(d, ds, vals))


def speed_for_radius(tabs, d, radius):
    """The smallest dog speed whose R_min(v, d) is at most `radius` (bisection on the monotone curve)."""
    lo, hi = 0.5, 8.0
    if r_min(tabs, d, hi) > radius:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if r_min(tabs, d, mid) > radius else (lo, mid)
    return float(hi)


# ------------------------------------------------------------------ the holds

def start_state(social_always):
    """A driven flock after 60 steps, translated so that its centroid sits at the goal; the dog 7.5 m behind."""
    res = simulate(Params(n_steps=60, box_length=200.0, social_always=social_always), n_runs=R, seed=SEED)
    pos, hd = res.pos_s[..., -1], res.vel_s[..., -1]
    pos = pos - pos.mean(1, keepdims=True)
    m = hd.mean(1)
    th = np.arctan2(m[:, 1], m[:, 0])
    return dict(pos_s=pos, vel_s=hd, pos_d=-7.5 * np.stack([np.cos(th), np.sin(th)], -1))


def hold(args):
    sheep, v_dog = args
    tic = time.time()
    p = replace(Params(social_always=(sheep == "always_moving")), v_dog=v_dog)
    ctrl = DC.Controller(DC.Functional("moment", s=S_GOAL))
    pos, dog, rec = DC.run(ctrl, p, start_state(sheep == "always_moving"), T, SEED + 1)
    c = pos.mean(1)  # (R, 2, T+1)
    rho = np.linalg.norm(c, axis=1)
    step = np.diff(c, axis=2)  # centroid displacement t -> t+1, (R, 2, T)
    V = np.linalg.norm(step, axis=1)
    th = np.arctan2(step[:, 1], step[:, 0])
    om = wrap(np.diff(th, axis=1))  # turn between consecutive displacements, (R, T-1)
    rel = dog[..., 1:] - c[..., 1:]
    stand = np.linalg.norm(rel, axis=1)
    phi = wrap(np.arctan2(rel[:, 1], rel[:, 0]) - th - np.pi)  # 0 = straight behind the moving flock
    sel = slice(T0, None)
    om_s, V_s = om[:, sel], V[:, 1:][:, sel]
    sgn = np.sign(om_s)
    hd = pos[..., 1:]  # headings are not stored; polarization from the displacement field instead
    out = dict(
        sheep=sheep, v_dog=v_dog, runs=R, steps=T, from_step=T0,
        rho_median=fl(np.median(rho[:, sel])), rho_mean=fl(rho[:, sel].mean()), rho_p10=fl(np.percentile(rho[:, sel], 10)), rho_p90=fl(np.percentile(rho[:, sel], 90)),
        inside_frac=fl((rho[:, sel] < R_GOAL).mean()),
        V_median=fl(np.median(V[:, sel])), turn_median_deg=fl(np.rad2deg(np.median(np.abs(om_s)))), turn_mean_deg=fl(np.rad2deg(np.abs(om_s).mean())),
        turning_radius_median=fl(np.median(V_s / np.maximum(np.abs(om_s), 1e-6))),
        standoff_median=fl(np.median(stand[:, sel])), standoff_p10=fl(np.percentile(stand[:, sel], 10)), standoff_p90=fl(np.percentile(stand[:, sel], 90)),
        bearing_median_deg=fl(np.rad2deg(np.median(np.abs(phi[:, sel])))), bearing_le60_frac=fl((np.abs(phi[:, sel]) <= np.pi / 3).mean()),
        same_sign_turn_frac=fl((sgn[:, 1:] * sgn[:, :-1] > 0).mean()), laps_median=fl(np.median(np.abs(om_s.sum(1))) / (2 * np.pi)),
        at_hstar_frac=fl((rec["gap"][:, sel] < 1e-9).mean()), gap_mean=fl(rec["gap"][:, sel].mean()), released_frac=fl(rec["released"][:, sel].mean()),
        rho_series_median=np.median(rho, 0).tolist(), seconds=fl(time.time() - tic),
    )
    return out


# ------------------------------------------------------------------ figure

def figure(out, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    tabs = tables()
    v = np.linspace(1.0, 5.0, 81)
    ax = axs[0]
    for d, c in zip(STANDOFFS, ("gray", "blue", "purple")):
        ax.plot(v, [r_min(tabs, d, x) for x in v], color=COL[c], lw=1.5, label=f"R_min(v_dog, d = {d:g} m), from the measured law")
    for sheep, mk, lab in (("published", "o", "published sheep"), ("always_moving", "s", "always-moving sheep")):
        rows = [r for r in out["holds"] if r["sheep"] == sheep]
        x = [r["v_dog"] for r in rows]
        ax.errorbar(x, [r["rho_median"] for r in rows], yerr=[[r["rho_median"] - r["rho_p10"] for r in rows], [r["rho_p90"] - r["rho_median"] for r in rows]],
                    fmt=mk, color=COL["red"], mfc=COL["red"] if sheep == "published" else "white", ms=7, capsize=3, lw=1, label=f"ρ median (bars: 10–90%), {lab}")
        ax.plot(x, [r["turning_radius_median"] for r in rows], "x" if sheep == "published" else "+", color=COL["teal"], ms=8, mew=1.6, label=f"turning radius V/|ω| median, {lab}")
    ax.axhline(R_GOAL, color=COL["gray"], lw=1, ls=":")
    ax.set(xlabel="dog speed v_dog (m/step)", ylabel="radius (m)", ylim=(0, 14), title="Hold radius against dog speed")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    for sheep, mk in (("published", "o"), ("always_moving", "s")):
        rows = [r for r in out["holds"] if r["sheep"] == sheep]
        pred = [r["R_min_at_standoff"] for r in rows]
        ax.plot(pred, [r["rho_median"] for r in rows], mk, color=COL["red"], mfc=COL["red"] if sheep == "published" else "white", ms=7, label=f"ρ median, {sheep.replace('_', ' ')}")
        ax.plot(pred, [r["turning_radius_median"] for r in rows], "x" if sheep == "published" else "+", color=COL["teal"], ms=8, mew=1.6, label=f"V/|ω| median, {sheep.replace('_', ' ')}")
        for r in rows:
            ax.annotate(f"{r['v_dog']:g}", (r["R_min_at_standoff"], r["rho_median"]), textcoords="offset points", xytext=(5, 3), fontsize=7, color=COL["gray"])
    ax.plot([0, 8], [0, 8], color=COL["gray"], lw=1, ls="--")
    ax.set(xlabel="predicted R_min(v_dog, d) at the standoff the dog kept (m)", ylabel="measured (m)", xlim=(0, 8), ylim=(0, 8), aspect="equal", title="Measured against predicted (labels: v_dog)")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[2]
    t = np.arange(T + 1)
    for r, c in zip([r for r in out["holds"] if r["sheep"] == "published"], ("gray", "purple", "blue", "teal", "green")):
        ax.plot(t, r["rho_series_median"], color=COL[c], lw=1.6, label=f"v_dog = {r['v_dog']:g}: median ρ, R_min {r['R_min_at_standoff']:.1f} m")
        ax.axhline(r["R_min_at_standoff"], color=COL[c], lw=0.8, ls=":")
    ax.axvline(T0, color=COL["gray"], lw=0.8, ls="--")
    ax.set(xlabel="step", ylabel="centroid distance to goal (m)", ylim=(0, 12), title="Published sheep: ρ(t) (dotted: predicted R_min)")
    ax.legend(frameon=False, fontsize=7)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The density controller's hold is the tightest steady turn the dog can keep up with", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    tabs = tables()
    pred = {f"{d:g}": {f"{v:g}": PT.feasibility(tabs[d], d, v) for v in SPEEDS} for d in STANDOFFS}
    jobs = [(s, v) for s in SHEEP for v in SPEEDS]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 2)) as ex:
        holds = list(ex.map(hold, jobs))
    for h in holds:
        h["R_min_at_7p5"] = fl(r_min(tabs, 7.5, h["v_dog"]))
        h["R_min_at_standoff"] = fl(r_min(tabs, h["standoff_median"], h["v_dog"]))
        f = PT.feasibility(tabs[7.5], 7.5, h["v_dog"])
        h["phi_max_deg_at_7p5"], h["omega_max_deg_at_7p5"] = fl(f["phi_max"]), fl(np.rad2deg(f["omega_max"]))
        print(f"  {h['sheep']:14s} v_dog {h['v_dog']:.1f}: rho median {h['rho_median']:.2f} (p10 {h['rho_p10']:.2f}, p90 {h['rho_p90']:.2f}) inside {h['inside_frac']:.2f}; "
              f"V {h['V_median']:.2f}, turn {h['turn_median_deg']:.1f} deg/step, V/|w| {h['turning_radius_median']:.2f}; standoff {h['standoff_median']:.1f}, "
              f"bearing {h['bearing_median_deg']:.0f} deg (<=60: {h['bearing_le60_frac']:.2f}); laps {h['laps_median']:.1f}, same-sign {h['same_sign_turn_frac']:.2f}; "
              f"R_min at standoff {h['R_min_at_standoff']:.2f} (at 7.5: {h['R_min_at_7p5']:.2f}, phi_max {h['phi_max_deg_at_7p5']:.0f}, omega_max {h['omega_max_deg_at_7p5']:.1f}); {h['seconds']:.0f} s", flush=True)
    v5 = {f"{d:g}": speed_for_radius(tabs, d, R_GOAL) for d in STANDOFFS}
    print("  dog speed for R_min = 5 m:", {k: round(v, 2) for k, v in v5.items() if v})
    out = dict(setup=dict(speeds=list(SPEEDS), sheep=list(SHEEP), runs=R, steps=T, from_step=T0, standoffs=list(STANDOFFS), seed=SEED, r_goal=R_GOAL),
               prediction=pred, speed_for_5m=v5, holds=holds, seconds=fl(time.time() - tic))
    (OUT / "hold_condition.json").write_text(json.dumps(out, indent=1))
    figure(out, OUT / "figures" / "hold_condition.png")
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
