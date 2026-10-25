"""Run the phase labeller on simulated flocks and compare with the real collie.

The task is the experiment's: drive the flock between A = (0, 0) and B = (60, 0)
and back, for 10 minutes. The target switches when the flock centre is within
5 m of it. The model has one global target, so each run is simulated in 5 s
chunks, shifted so that its own target is at the origin (the model is
translation invariant). The dog's flanking offset restarts at zero each chunk.

Models:
- Published: the model as in the paper's code (dt = 1 s).
- Calibrated (M4tau): M3tau posterior draws, the measured collie and the
  learned dog response with gain 2 (Phase 3d), dt = 0.2 s.

    .venv/bin/python experiments/label_simulations.py
Writes results/label_simulations.json and results/figures/label_simulations.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from common import ROOT, draws, load, posterior  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, phases as PH  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402
from build_table import V_SHEEP, measured_dog  # noqa: E402
from dog_response import learned_dog_kernel  # noqa: E402

OUT = ROOT / "results"
A, B = np.array([0.0, 0.0]), np.array([60.0, 0.0])
T_TOTAL, CHUNK, ARRIVE, N_RUNS = 600.0, 5.0, 5.0, 16
N = data.N_SHEEP
COLORS = {"drive": "#2a78d6", "turn": "#eb6834", "flank": "#1baf7a", "idle": "#c9c7c1", "away": "#8a6fd1"}


def models():
    dog = measured_dog()
    sheep = draws(posterior(load("M3tau"), "M3tau"), N_RUNS)
    calibrated = Params(dt=0.2, v_sheep=V_SHEEP[0.2], dog_kernel=learned_dog_kernel(), g_dogk=2.0,
                        flank_sigma=1e-9, dog_speed_gain=dog["dog_speed_gain"])
    per = {**{k: np.asarray(v)[:N_RUNS] for k, v in sheep.items()},
           **{k: np.full(N_RUNS, v) for k, v in dog.items() if v is not None}}
    return {"Published": (Params(), None), "Calibrated (M4τ)": (calibrated, per)}


def back_and_forth(params, per_run, seed=0):
    """(R, N+1, 2, T) positions (dog last), trips completed per run, and each run's
    target during each chunk, (R, n_chunks, 2)."""
    dt, R = params.dt, N_RUNS
    rng = np.random.default_rng(seed)
    pos_s = A + rng.normal(0, 1.0, (R, N, 2))
    vel_s = np.tile([1.0, 0.0], (R, N, 1))
    pos_d = A + np.array([-6.0, 0.0]) + rng.normal(0, 0.5, (R, 2))
    vel_d = np.tile([1.0, 0.0], (R, 1))
    target = np.tile(B, (R, 1))
    trips = np.zeros(R, int)
    steps = int(round(CHUNK / dt))
    p = replace(params, n_sheep=N, n_steps=steps + 1, target=(0.0, 0.0))
    frames = [np.concatenate([pos_s, pos_d[:, None]], 1)[..., None]]
    log = []
    for c in range(int(round(T_TOTAL / CHUNK))):
        log.append(target.copy())
        sh = target[:, None, :]
        init = {"pos_s": pos_s - sh, "vel_s": vel_s, "pos_d": pos_d - target, "vel_d": vel_d}
        res = simulate(p, R, seed + 1 + c, per_run=per_run, init=init)
        ps = res.pos_s + target[:, None, :, None]
        pd = res.pos_d + target[:, :, None]
        frames.append(np.concatenate([ps, pd[:, None]], 1)[..., 1:])
        pos_s, vel_s = ps[..., -1], res.vel_s[..., -1]
        pos_d, vel_d = pd[..., -1], res.vel_d[..., -1]
        arrived = np.linalg.norm(pos_s.mean(1) - target, axis=1) < ARRIVE
        trips += arrived
        target[arrived] = np.where((target[arrived] == B).all(1, keepdims=True), A, B)
    return np.concatenate(frames, -1), trips, np.stack(log, 1)


def sharpness(tracks, rules=PH.Rules()):
    """Percentiles of the 10 s heading change (degrees) while the flock moves."""
    h = []
    for tr in tracks:
        f = PH.features(tr, rules)
        h.append(f["heading_change"][f["flock_speed"] >= rules.idle_speed])
    return dict(zip(("p50", "p90", "p99"), np.percentile(np.concatenate(h), [50, 90, 99]).round(1).tolist()))


def summarize(labs, segs, dt):
    shares = PH.time_shares(labs)
    moving = sum(shares[k] for k in ("drive", "turn", "flank"))
    minutes = sum(len(l) for l in labs) * dt / 60
    by = {}
    for k in PH.LABELS:
        g = [s for s in segs if s.label == k]
        by[k] = {"segments_per_10min": len(g) / minutes * 10,
                 "median_duration_s": float(np.median([s.end_s - s.start_s for s in g])) if g else None,
                 **{f: float(np.median([getattr(s, f) for s in g])) if g else None
                    for f in ("flock_speed", "dog_dist", "psi_bd", "cohesion")}}
    return {"minutes": minutes, "shares": shares,
            "moving_shares": {k: shares[k] / moving for k in ("drive", "turn", "flank")} if moving else None,
            "by_label": by}


def main():
    out = {}
    tracks = [PH.Track.from_session(s) for s in data.load_sessions()]
    res = [PH.label_track(t) for t in tracks]
    out["Real collie"] = summarize([r[0] for r in res], [s for r in res for s in r[1]], data.DT)
    out["Real collie"]["heading_change_moving"] = sharpness(tracks)
    examples = {"Real collie": (res[7][0], data.DT)}  # session 8: mostly driving
    for name, (params, per) in models().items():
        pos, trips, _ = back_and_forth(params, per)
        sim_tracks = [PH.Track(f"{name} run {r}", pos[r], params.dt, N, N) for r in range(N_RUNS)]
        res = [PH.label_track(t) for t in sim_tracks]
        out[name] = summarize([r[0] for r in res], [s for r in res for s in r[1]], params.dt)
        out[name]["heading_change_moving"] = sharpness(sim_tracks)
        out[name]["runs_with_away"] = int(sum((r[0] == PH.LABELS.index("away")).any() for r in res))
        out[name]["trips_per_10min"] = float(trips.mean() * 600 / T_TOTAL)
        away = [(r[0] == PH.LABELS.index("away")).mean() for r in res]
        examples[name] = (res[int(np.argsort(away)[len(away) // 2])][0], params.dt)  # run with the median away share
    out["Real collie"]["trips_per_10min"] = None  # not defined: sessions include rests between trips
    (OUT / "label_simulations.json").write_text(json.dumps(out, indent=2))
    plot(out, examples)
    for name, o in out.items():
        sh = "  ".join(f"{k} {100 * v:4.1f}%" for k, v in o["shares"].items())
        mv = "  ".join(f"{k} {100 * v:4.1f}%" for k, v in (o["moving_shares"] or {}).items())
        print(f"{name:18s} {o['minutes']:6.1f} min | {sh}\n{'':18s} moving only: {mv}  trips/10 min {o['trips_per_10min']}"
              f"  10 s heading change while moving {o['heading_change_moving']}")


def plot(out, examples):
    names = list(out)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    for row, name in enumerate(names):
        left = 0.0
        for k in PH.LABELS:
            v = out[name]["shares"][k]
            ax.barh(row, v, left=left, color=COLORS[k], height=0.6, label=k if row == 0 else None)
            if v > 0.04:
                ax.text(left + v / 2, row, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of time")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax = axes[1]
    for row, (name, (lab, dt)) in enumerate(examples.items()):
        t = np.arange(len(lab)) * dt / 60
        for i, k in enumerate(PH.LABELS):
            m = lab == i
            ax.fill_between(t, row - 0.3, row + 0.3, where=m, color=COLORS[k], step="post", lw=0)
    ax.set_yticks(range(len(examples)), [f"{n} (one 10 min track)" for n in examples])
    ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (min)")
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "label_simulations.png", dpi=130)


if __name__ == "__main__":
    main()
