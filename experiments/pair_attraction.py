"""Add the learned sheep-sheep attraction/repulsion kernel to the calibrated model (M4tau).

Phase 3d left the calibrated flock unable to hold together over repeated
reversals and pointed at its weak random-neighbour attraction. Here the learned
pair kernel phi_E (shepherd/kernels.py) enters the heading rule
(model.Params.pair_kernel) with one gain g_pair, on top of M4tau (M3tau draws,
measured collie, learned dog response with gain 2). Each gain is scored like
experiments/dog_response.py, plus the back-and-forth task of
experiments/label_simulations.py with phase labels.

    .venv/bin/python experiments/pair_attraction.py
Writes results/pair_attraction.json and results/figures/pair_attraction.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402

import numpy as np  # noqa: E402

from common import ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, kernels as K, phases as PH  # noqa: E402
from shepherd.model import Params  # noqa: E402
from build_table import V_SHEEP  # noqa: E402
import dog_response as DR  # noqa: E402
import label_simulations as LS  # noqa: E402

OUT = ROOT / "results"
GAINS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
G_DOG = 2.0


def learned_pair_kernel():
    fit = np.load(OUT / "tables" / "kernel_fit.npz")
    m = K.KernelModel(terms=tuple(str(t) for t in fit["terms"]))
    return (tuple(m.pair.knots), tuple(fit["c"][m.slices()["E"]]))


def base_params(g):
    return Params(v_sheep=V_SHEEP[DR.DT], dog_kernel=DR.learned_dog_kernel(), g_dogk=G_DOG,
                  pair_kernel=learned_pair_kernel() if g > 0 else None, g_pairk=g)


def task(base, ctx):
    """The back-and-forth task with the same sheep draws and collie as label_simulations."""
    dog = ctx["dog"]
    p = Params(**{**base.__dict__, "dt": 0.2, "flank_sigma": 1e-9, "dog_speed_gain": dog["dog_speed_gain"]})
    per = {**{k: np.asarray(v)[: LS.N_RUNS] for k, v in ctx["sheep"].items()},
           **{k: np.full(LS.N_RUNS, v) for k, v in dog.items() if v is not None}}
    pos, trips, _ = LS.back_and_forth(p, per)
    res = [PH.label_track(PH.Track(r, pos[r], p.dt, data.N_SHEEP, data.N_SHEEP)) for r in range(LS.N_RUNS)]
    shares = PH.time_shares([r[0] for r in res])
    G = pos[:, : data.N_SHEEP].mean(1)
    C = np.linalg.norm(pos[:, : data.N_SHEEP] - G[:, None], axis=2).mean(1)
    return {"trips_per_10min": float(trips.mean() * 600 / LS.T_TOTAL), "trips_per_run": trips.tolist(),
            "shares": shares, "runs_with_away": int(sum((r[0] == PH.LABELS.index("away")).any() for r in res)),
            "cohesion_median": float(np.median(C)), "cohesion_p90": float(np.percentile(C, 90))}


def main():
    ctx = DR.context()
    out = {"data": {"long": ctx["long_obs"]}, "g_dog": G_DOG, "gains": {}}
    for g in GAINS:
        base = base_params(g)
        r = DR.evaluate(base, ctx)
        r["task"] = task(base, ctx)
        out["gains"][g] = r
        L, T, B = r["long"], r["turn_test"], r["task"]
        print(f"g_pair={g:4.1f}  short RMS z {r['short_rms_z']:.2f}  long C {L['C_med']:.2f} drift {L['C_drift']:.2f} "
              f"P {L['P_med']:.2f}  turn {T['success']:.2f}  trips/10min {B['trips_per_10min']:.2f}  "
              f"away {100 * B['shares']['away']:.0f}% ({B['runs_with_away']}/16 runs)  task C {B['cohesion_median']:.2f}")
    (OUT / "pair_attraction.json").write_text(json.dumps(out, indent=2, default=float))
    plot(out)


def plot(out):
    g = list(out["gains"])
    R = out["gains"]
    x = np.arange(len(g))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    panels = [
        ("Short-window error", [R[k]["short_rms_z"] for k in g], None),
        ("Long drives: cohesion C (m)", [R[k]["long"]["C_med"] for k in g], out["data"]["long"]["C_med"]),
        ("Back-and-forth trips / 10 min", [R[k]["task"]["trips_per_10min"] for k in g], None),
        ("Back-and-forth: dog away share", [R[k]["task"]["shares"]["away"] for k in g], None),
    ]
    for ax, (title, y, ref) in zip(axes, panels):
        ax.plot(x, y, "o-", color="#2a78d6")
        if ref is not None:
            ax.axhline(ref, color="#52514e", ls="--", lw=1, label="real flock")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(x, [f"{k:g}" for k in g])
        ax.set_xlabel("attraction gain g_pair")
        ax.set_title(title, fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "pair_attraction.png", dpi=130)


if __name__ == "__main__":
    main()
