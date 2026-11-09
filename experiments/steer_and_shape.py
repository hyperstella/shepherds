"""Can one dog steer a flock and hold its shape? The Jadhav sheep in the co-moving frame.

The effective model to test (docs/steer_and_shape.md) is

    d_t rho + div( rho (b_collective + u_dog) ) = D_ind lap rho,      rho*(x, t) = q*(x - c(t)),

with the matching condition u_dog* = c' + D_ind grad log rho* - b_collective[rho*]. Jadhav's
sheep do not respond additively: the heading is unit(S), with S the weighted sum of unit
vectors (inertia, repulsion, dog, attraction, alignment) plus noise of weight eta. This script
measures three things.

1. Response curve: the mean heading of one sheep in a constant stimulus of strength F with
   noise eta, alone and with heading inertia. Additive (linear in F, gain 1/(2 eta)) only while
   F << eta; saturated (direction only) once F >> eta.
2. Driven flocks in the co-moving frame. The published dog drives the flock toward a target
   far away, with collecting switched off so the dog's standoff l_drive and flanking amplitude
   are the control knobs. The grid covers heading noise eta, attraction w_att, alignment w_ali
   and notice radius r_dog. Measured over the second half of each run: the flock's shape along
   and across the drive (sd of the sheep about the centroid), its speed, the fraction of sheep
   that see the dog, splits, and the Knudsen number of each sheep's lateral motion relative to
   the flock (persistence length over the flock's lateral sd; the real flock's is 0.2).
3. The same flocks with no dog (sheep always social and moving: the paper's null model, i.e. its
   MODIFIED rule, since under the published rule a sheep with no dog in range stands still): the
   shape cohesion holds on its own. At r_dog = 20 every sheep of a driven flock sees the dog, so
   there the published and modified rules coincide and the comparison is within one rule.
4. Turning and stopping, under the MODIFIED rule (social_always: sheep out of the dog's range keep
   moving). Under the published rule a released flock freezes (continuification_driving.md §9.2).

    .venv/bin/python experiments/steer_and_shape.py
Writes results/steer_and_shape.json and results/figures/steer_and_shape.png.
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
N, T, T0, REPS, SEED = 14, 300, 150, 10, 11
FAR = 2000.0  # the target is this far away: a straight drive that never arrives
SPLIT_M = 15.0  # a sheep this far from the centroid at the end means the flock split
ETA = (0.3, 1.0, 2.0)  # heading noise weight (published 0.3)
W_ATT = (0.75, 1.5, 3.0)  # attraction (published 1.5)
W_ALI = (0.0, 1.3, 3.0)  # alignment (published 1.3)
R_DOG = (12.0, 20.0)  # notice radius (published 12)
L_DRIVE = (3.0, 5.0, 7.5, 10.0)  # dog standoff behind the centroid (published 2 sqrt(N) = 7.5)
FLANK = (0.0, 3.0)  # sd (m) of the dog's sideways swing
MAX_LAG = 25
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ------------------------------------------------------------------ 1. response curve

def response_curve(rng, ratios=np.geomspace(0.05, 30, 40), alpha=0.5, steps=4000):
    """Mean heading component along a constant stimulus F (in units of eta), without and with inertia."""
    eta = 1.0
    th = 2 * np.pi * np.arange(4000) / 4000
    zeta = np.stack([np.cos(th), np.sin(th)], -1)
    no_mem = []
    for x in ratios:  # exact average over the noise direction
        s = np.array([x, 0.0]) + zeta
        no_mem.append(float((s[:, 0] / np.linalg.norm(s, axis=1)).mean()))
    mem = []
    h = np.tile([1.0, 0.0], (len(ratios), 1))
    acc = np.zeros(len(ratios))
    for t in range(steps):
        a = 2 * np.pi * rng.random(len(ratios))
        s = alpha * h + np.stack([ratios, np.zeros_like(ratios)], -1) + eta * np.stack([np.cos(a), np.sin(a)], -1)
        h = s / np.linalg.norm(s, axis=1, keepdims=True)
        if t >= 200:
            acc += h[:, 0]
    mem = acc / (steps - 200)
    return dict(F_over_eta=ratios.tolist(), mean_heading=no_mem, mean_heading_inertia=mem.tolist(), alpha_over_eta=alpha / eta)


# ------------------------------------------------------------------ 2-3. flocks

def settings():
    dog = [dict(noise=e, w_att=a, w_ali=l, r_dog=r, l_drive=d, flank_sigma=f)
           for e, a, l, r, d, f in itertools.product(ETA, W_ATT, W_ALI, R_DOG, L_DRIVE, FLANK)]
    free = [dict(noise=e, w_att=a, w_ali=l) for e, a, l in itertools.product(ETA, W_ATT, W_ALI)]
    return dog, free


def lag_tau(x):
    """Integrated autocorrelation time (steps) of x (R, N, T), up to the first zero crossing."""
    x = x - x.mean(-1, keepdims=True)
    c0 = (x * x).mean(-1)
    tau = np.zeros(x.shape[:2])
    alive = np.ones(x.shape[:2], bool)
    for k in range(MAX_LAG):
        c = (x[..., : x.shape[-1] - k] * x[..., k:]).mean(-1) / np.maximum(c0, 1e-12) if k else np.ones(x.shape[:2])
        alive &= c > 0
        tau += np.where(alive, c, 0.0) * (1.0 if k else 0.5)
    return tau


def measure(pos, dog, axis_mode, r_dog=None):
    """Shape and motion in the co-moving frame over steps T0..T. pos (R, N, 2, T); dog (R, 2, T) or None."""
    x = pos[..., T0:]
    G = x.mean(1)  # (R, 2, t)
    vel = np.diff(x, axis=-1)  # actual displacements per step (frozen sheep: 0)
    if axis_mode == "target":
        e = -G / np.linalg.norm(G, axis=1, keepdims=True)  # toward the target at the origin
    else:  # the flock's own heading
        V = vel.mean(1)
        V = np.concatenate([V, V[..., -1:]], -1)
        e = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    e_perp = np.stack([-e[:, 1], e[:, 0]], 1)
    xi = x - G[:, None]
    par = np.einsum("rnct,rct->rnt", xi, e)
    perp = np.einsum("rnct,rct->rnt", xi, e_perp)
    spread_end = np.linalg.norm(xi[..., -1], axis=-1).max(-1)
    split = spread_end > SPLIT_M
    s_par = np.sqrt((par ** 2).mean(axis=(1, 2)))
    s_perp = np.sqrt((perp ** 2).mean(axis=(1, 2)))
    speed = np.einsum("rct,rct->rt", np.diff(G, axis=-1), e[..., :-1]).mean(-1)
    v_rel = vel - vel.mean(1, keepdims=True)
    v_perp = np.einsum("rnct,rct->rnt", v_rel, e_perp[..., :-1])
    tau = lag_tau(v_perp).mean(1)
    rms = np.sqrt((v_perp ** 2).mean(axis=(1, 2)))
    kn = rms * tau / np.maximum(s_perp, 1e-9)
    out = dict(s_par=s_par, s_perp=s_perp, speed=speed, split=split, kn_perp=kn, tau_perp=tau, rms_vperp=rms)
    if dog is not None:
        dd = np.linalg.norm(x - dog[:, None, :, T0:], axis=2)
        out["coverage"] = (dd <= r_dog[:, None, None]).mean(axis=(1, 2))
        out["standoff"] = np.einsum("rct,rct->rt", G - dog[..., T0:], e).mean(-1)
    return out


def run_grid(rows, dog_present):
    R = len(rows) * REPS
    per = {k: np.repeat([r[k] for r in rows], REPS) for k in rows[0]}
    per["noise_dog"] = np.full(R, 0.3)  # the dog's own heading noise stays the published one
    if dog_present:
        per["l_sep"] = np.full(R, 1e6)  # no collecting: standoff and flanking are the only knobs
    p = Params(n_sheep=N, box_length=FAR, n_steps=T, dog_present=dog_present)
    res = simulate(p, n_runs=R, seed=SEED, per_run=per)
    m = measure(res.pos_s, res.pos_d if dog_present else None, "target" if dog_present else "heading",
                per.get("r_dog") if dog_present else None)
    out = []
    for i, r in enumerate(rows):
        sl = slice(i * REPS, (i + 1) * REPS)
        ok = ~m["split"][sl]
        agg = {k: fl(np.mean(v[sl][ok])) if ok.any() else None for k, v in m.items() if k != "split"}
        out.append(dict(**r, **agg, split_rate=float(m["split"][sl].mean())))
    return out


# ------------------------------------------------------------------ 4. turning and stopping

TURN_ATT, TURN_ALI, TURN_RDOG = (1.5, 3.0), (0.0, 1.3, 3.0), (12.0, 20.0)
TURN_VDOG = (1.5, 3.0)  # published dog, and one fast enough that its repositioning is not the bottleneck
TURN_REPS, T_DRIVE, T_TURN, T_STOP = 24, 150, 200, 100
HEAD_OK = np.deg2rad(20.0)


def to_turn_frame(res):
    """Rotate and shift each run so the flock sits at (-FAR, 0) heading +y: a target at the origin is then 90° to its right."""
    G = res.pos_s[..., -1].mean(1)  # (R, 2)
    V = (res.pos_s[..., -1] - res.pos_s[..., -6]).mean(1)
    e1 = V / np.linalg.norm(V, axis=1, keepdims=True)
    # rotation taking e1 to (0, 1)
    c, s = e1[:, 1], -e1[:, 0]
    Rm = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1)  # (R, 2, 2)
    move = lambda p: np.einsum("rij,r...j->r...i", Rm, p - G[:, None]) + np.array([-FAR, 0.0])
    rot = lambda v: np.einsum("rij,r...j->r...i", Rm, v)
    return dict(pos_s=move(res.pos_s[..., -1]), vel_s=rot(res.vel_s[..., -1]),
                pos_d=move(res.pos_d[:, None, :, -1])[:, 0], vel_d=rot(res.vel_d[:, None, :, -1])[:, 0])


def turn_and_stop():
    """Drive straight, then (a) ask for a 90° turn, (b) remove the dog from sheep that keep moving."""
    rows = [dict(w_att=a, w_ali=l, r_dog=r, v_dog=v) for a, l, r, v in itertools.product(TURN_ATT, TURN_ALI, TURN_RDOG, TURN_VDOG)]
    R = len(rows) * TURN_REPS
    per = {k: np.repeat([r[k] for r in rows], TURN_REPS) for k in rows[0]}
    per.update(l_sep=np.full(R, 1e6), noise_dog=np.full(R, 0.3))
    # social_always: sheep out of the dog's range keep moving socially, so stopping is the flock's own doing
    drive = simulate(Params(n_sheep=N, box_length=FAR, n_steps=T_DRIVE, social_always=True), n_runs=R, seed=SEED + 1, per_run=per)
    init = to_turn_frame(drive)
    turn = simulate(Params(n_sheep=N, n_steps=T_TURN, social_always=True), n_runs=R, seed=SEED + 2, per_run=per, init=init)
    stop = simulate(Params(n_sheep=N, n_steps=T_STOP, dog_present=False), n_runs=R, seed=SEED + 3,
                    per_run={k: v for k, v in per.items() if k not in ("r_dog", "l_sep", "noise_dog", "v_dog")}, init=init)
    G = turn.pos_s.mean(1)  # (R, 2, T)
    V = np.diff(G, axis=-1)
    ang = np.arctan2(V[:, 1], V[:, 0])  # 0 = toward the target (east)
    ok = np.abs(ang) < HEAD_OK
    # turned: heading within 20° of the target for 10 consecutive steps
    win = np.lib.stride_tricks.sliding_window_view(ok, 10, axis=-1).all(-1)
    t_turn = np.where(win.any(-1), win.argmax(-1), np.nan)
    overshoot = G[:, 1].max(-1) - G[:, 1, 0]  # how far the flock carried on in its old direction
    spread = np.linalg.norm(turn.pos_s - G[:, None], axis=2).max(1)  # (R, T)
    Gs = stop.pos_s.mean(1)
    run_on = np.linalg.norm(Gs[..., -1] - Gs[..., 0], axis=-1)
    Vs = np.diff(stop.pos_s, axis=-1)
    pol = np.linalg.norm(Vs.mean(1), axis=1)  # (R, T-1): flock speed = polarization at unit speed
    pol0 = pol[:, :5].mean(-1)
    half = np.array([np.argmax(p < 0.5 * p0) if (p < 0.5 * p0).any() else np.nan for p, p0 in zip(pol, pol0)], float)
    out = []
    for i, r in enumerate(rows):
        sl = slice(i * TURN_REPS, (i + 1) * TURN_REPS)
        tt = t_turn[sl]
        out.append(dict(**r, turned=float(np.isfinite(tt).mean()), t_turn_median=fl(np.nanmedian(tt)) if np.isfinite(tt).any() else None,
                        overshoot_median=fl(np.median(overshoot[sl])), max_spread_median=fl(np.median(spread[sl].max(-1))),
                        split_rate=float((spread[sl, -1] > SPLIT_M).mean()),
                        stop_run_on_median=fl(np.median(run_on[sl])), stop_speed0=fl(pol0[sl].mean()),
                        stop_half_life_median=fl(np.nanmedian(half[sl])) if np.isfinite(half[sl]).any() else None,
                        stop_never_halved=float(np.isnan(half[sl]).mean())))
    return out


def main():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    curve = response_curve(rng)
    dog_rows, free_rows = settings()
    free = run_grid(free_rows, dog_present=False)
    print(f"no-dog reference: {len(free)} settings ({time.time() - tic:.0f} s)", flush=True)
    driven = []
    for eta in ETA:  # in chunks, to bound memory
        driven += run_grid([r for r in dog_rows if r["noise"] == eta], dog_present=True)
        print(f"  eta {eta}: done ({time.time() - tic:.0f} s)", flush=True)
    turns = turn_and_stop()
    for r in turns:
        print(f"  turn/stop w_att {r['w_att']} w_ali {r['w_ali']} r_dog {r['r_dog']:.0f} v_dog {r['v_dog']}: turned {r['turned']:.2f} in {r['t_turn_median']} steps, "
              f"overshoot {r['overshoot_median']:.1f} m, max spread {r['max_spread_median']:.1f} m, split {r['split_rate']:.2f} | "
              f"dog removed: runs on {r['stop_run_on_median']:.1f} m in {T_STOP} steps, speed {r['stop_speed0']:.2f}, "
              f"half-life {r['stop_half_life_median']} (never halves {r['stop_never_halved']:.2f})", flush=True)
    out = dict(setup=dict(N=N, steps=T, window_from=T0, reps=REPS, far=FAR, split_m=SPLIT_M, eta=ETA, w_att=W_ATT, w_ali=W_ALI,
                          r_dog=R_DOG, l_drive=L_DRIVE, flank=FLANK,
                          rules=dict(driven="published (stop when no dog within r_dog)",
                                     free="null model: no dog, social terms always on, sheep keep moving (the paper's modified rule)",
                                     turn_stop="modified (social_always): sheep out of the dog's range keep moving; stop phase has no dog"),
                          turn=dict(reps=TURN_REPS, t_drive=T_DRIVE, t_turn=T_TURN, t_stop=T_STOP,
                                                                              heading_ok_deg=20.0, social_always=True)),
               response=curve, free=free, driven=driven, summary=summarize(free, driven), turn_stop=turns, seconds=time.time() - tic)
    (OUT / "steer_and_shape.json").write_text(json.dumps(out, indent=1))
    report(out)
    figure(out)


def key(r):
    return (r["noise"], r["w_att"], r["w_ali"])


def summarize(free, driven):
    """Per flock (eta, w_att, w_ali) and notice radius: the shapes the dog's knobs reach, against the dog-free shape."""
    ref = {key(r): r for r in free}
    rows = []
    for k in sorted(ref):
        for rd in R_DOG:
            ds = [r for r in driven if key(r) == k and r["r_dog"] == rd]
            good = [r for r in ds if r["s_par"] is not None and r["split_rate"] <= 0.2 and r["speed"] is not None and r["speed"] > 0.5]
            f = ref[k]
            row = dict(noise=k[0], w_att=k[1], w_ali=k[2], r_dog=rd, free_s_par=f["s_par"], free_s_perp=f["s_perp"], free_kn=f["kn_perp"],
                       free_split=f["split_rate"], n_policies=len(ds), n_good=len(good))
            if good:
                sp = np.array([r["s_perp"] for r in good])
                sa = np.array([r["s_par"] for r in good])
                row.update(s_perp_min=fl(sp.min()), s_perp_max=fl(sp.max()), s_par_min=fl(sa.min()), s_par_max=fl(sa.max()),
                           perp_ratio_min=fl(sp.min() / f["s_perp"]) if f["s_perp"] else None,
                           par_ratio_min=fl(sa.min() / f["s_par"]) if f["s_par"] else None,
                           speed_range=[fl(min(r["speed"] for r in good)), fl(max(r["speed"] for r in good))],
                           kn_driven=fl(np.mean([r["kn_perp"] for r in good])))
            rows.append(row)
    return rows


def report(out):
    print("flock (eta, w_att, w_ali) r_dog | no dog: s_par s_perp Kn | driven, usable policies: s_perp range, s_par range, speed")
    for r in out["summary"]:
        if r["n_good"]:
            print(f"  ({r['noise']:.1f}, {r['w_att']:.2f}, {r['w_ali']:.1f}) r_dog {r['r_dog']:4.0f} | {r['free_s_par'] or float('nan'):5.2f} {r['free_s_perp'] or float('nan'):5.2f} "
                  f"{r['free_kn'] or float('nan'):5.2f} | {r['n_good']}/{r['n_policies']}: perp {r['s_perp_min']:.2f}-{r['s_perp_max']:.2f} "
                  f"par {r['s_par_min']:.2f}-{r['s_par_max']:.2f} speed {r['speed_range'][0]:.2f}-{r['speed_range'][1]:.2f}")
        else:
            print(f"  ({r['noise']:.1f}, {r['w_att']:.2f}, {r['w_ali']:.1f}) r_dog {r['r_dog']:4.0f} | no usable policy")


def figure(out):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
    c = out["response"]
    ax = axs[0, 0]
    x = np.array(c["F_over_eta"])
    ax.plot(x, c["mean_heading"], color=COL["blue"], lw=2, label="one step, no memory")
    ax.plot(x, c["mean_heading_inertia"], color=COL["orange"], lw=2, label=f"with inertia α = {c['alpha_over_eta']:g} η")
    ax.plot(x, np.minimum(x / 2, 1), color=COL["gray"], lw=1, ls=":", label="linear response |S|/(2η)")
    ax.set(xscale="log", xlabel="stimulus |S| / heading noise η", ylabel="mean heading along the stimulus",
           title="(a) One sheep: additive only while |S| ≲ η, a direction beyond")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ref = {key(r): r for r in out["free"]}
    ax = axs[0, 1]
    for w_att, col in zip(W_ATT, ("orange", "blue", "green")):
        pts = [r for r in out["driven"] if r["noise"] == 0.3 and r["w_att"] == w_att and r["w_ali"] == 1.3
               and r["s_par"] is not None and r["split_rate"] <= 0.2]
        ax.scatter([r["s_par"] for r in pts], [r["s_perp"] for r in pts], s=[22 + 28 * (r["r_dog"] == 20) for r in pts],
                   color=COL[col], alpha=0.8, edgecolor="white", lw=0.8, label=f"driven, w_att {w_att:g}")
        f = ref[(0.3, w_att, 1.3)]
        ax.scatter([f["s_par"]], [f["s_perp"]], marker="*", s=240, color=COL[col], edgecolor="k", lw=0.8)
    m = 5.5
    ax.plot([0, m], [0, m], color=COL["gray"], lw=0.8, ls=":")
    ax.set(xlabel="sd along the drive (m)", ylabel="sd across the drive (m)", xlim=(0, m), ylim=(0, m),
           title="(b) Shapes one dog reaches (published η, w_ali)\n★ same flock, no dog (modified rule); small dots r_dog 12, big 20")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[1, 0]
    for w_ali, col in zip(W_ALI, ("orange", "blue", "green")):
        pts = sorted([r for r in out["driven"] if r["noise"] == 0.3 and r["w_att"] == 3.0 and r["w_ali"] == w_ali
                      and r["r_dog"] == 20.0 and r["flank_sigma"] == 0.0], key=lambda r: r["l_drive"])
        ax.plot([r["l_drive"] for r in pts], [r["s_perp"] for r in pts], "o-", color=COL[col], lw=2, label=f"driven, w_ali {w_ali:g}")
        ax.axhline(ref[(0.3, 3.0, w_ali)]["s_perp"], color=COL[col], lw=1.2, ls="--")
    ax.set(xlabel="dog standoff behind the centroid (m)", ylabel="sd across the drive (m)", ylim=(0, None),
           title="(c) Across the drive (w_att 3, r_dog 20, where both rules coincide):\na nearer dog widens; dashed: the dog-free flock")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 1]
    ts = out["turn_stop"]
    alis = sorted({r["w_ali"] for r in ts})
    xs = np.arange(len(alis))
    for j, (w_att, col) in enumerate(zip(sorted({r["w_att"] for r in ts}), ("blue", "green"))):
        vals = [np.median([r["stop_run_on_median"] for r in ts if r["w_att"] == w_att and r["w_ali"] == a]) for a in alis]
        ax.bar(xs + (j - 0.5) * 0.36, vals, width=0.34, color=COL[col], label=f"w_att {w_att:g}")
        for xx, v in zip(xs + (j - 0.5) * 0.36, vals):
            ax.text(xx, v + 0.8, f"{v:.0f} m", ha="center", fontsize=8, color="#3a3a38")
    ax.set_xticks(xs, [f"w_ali {a:g}" for a in alis])
    ax.set(ylabel=f"distance the flock carries on in {T_STOP} steps (m)",
           title="(d) Stopping, MODIFIED rule (sheep keep moving without a dog):\nrun-on after the dog leaves grows with alignment")
    ax.legend(frameon=False, fontsize=8)
    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "steer_and_shape.png", dpi=130)


if __name__ == "__main__":
    main()
