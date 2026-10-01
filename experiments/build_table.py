"""Phase 3: reference tables for calibrating the Strömbom/Jadhav sheep rules.

Every parameter set is run through the windowed dog-replay harness (the
real dog, restarts from the real flock every 15 s). Two rounds:
round 1 samples the full prior; round 2 samples a box around the round-1
posterior. Fixed-parameter baselines are simulated too.

    .venv/bin/python experiments/build_table.py --model M1 --n1 50000 --n2 25000
    .venv/bin/python experiments/build_table.py --model M1tau --n1 30000 --n2 15000
    .venv/bin/python experiments/build_table.py --baselines

Writes results/tables/<model>.npz (gitignored).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import abc as ABC, protocol as PR  # noqa: E402
from shepherd.model import Params  # noqa: E402

TABLES = ROOT / "results" / "tables"

# Dog-only statistics are fixed by the replayed dog, so they are not fitted.
FIT_STATS = [k for k in PR.SUMMARY_KEYS if k not in ("turn_dog_med", "slope_4b")]

# Mean displacement speed of moving sheep in the drives at each step (m/s).
V_SHEEP = {1.0: 1.245, 0.2: 1.289}

# Wide log-uniform priors. Runs with weights capped at 5 and then 50 piled up
# at the alignment bound (relative to the fixed w_rep = 2) and the noise floor.
SHEEP_PRIORS = {
    "r_rep": ABC.Prior(0.2, 3.0),
    "w_inertia": ABC.Prior(0.05, 1000.0),
    "w_att": ABC.Prior(0.05, 1000.0),
    "w_ali": ABC.Prior(0.01, 1000.0),
    "w_dog": ABC.Prior(0.05, 1000.0),
    "noise": ABC.Prior(1e-4, 5.0),
    "r_dog": ABC.Prior(3.0, 25.0, log=False),
}

# All calibrated models replay the real flock's mean speed (an external input,
# like the dog): neither the published model (constant speed) nor the learned
# model identifies how the flock's overall speed is regulated.
REPLAY = {"speed_replay": True}

# The six terms of the sheep's heading rule only matter through their ratios.
WEIGHTS = ("w_inertia", "w_rep", "w_dog", "w_att", "w_ali", "noise")
SHARE_PRIORS = {
    "r_rep": ABC.Prior(0.1, 3.0),
    "r_dog": ABC.Prior(3.0, 25.0, log=False),
    **{k: ABC.Prior(exp=True) for k in WEIGHTS},  # uniform prior on the weight shares
}

MODELS = {
    # First calibration: w_rep = 2 fixed as the scale reference (alignment ran to the prior edge).
    "M1": {"dt": 1.0, "priors": SHEEP_PRIORS},
    "M1tau": {"dt": 0.2, "priors": {**SHEEP_PRIORS, "tau_s": ABC.Prior(0.2, 3.0)}},
    # Reparametrized: all six weights free with a uniform prior on their shares.
    "M2": {"dt": 1.0, "priors": SHARE_PRIORS},
    "M2tau": {"dt": 0.2, "priors": {**SHARE_PRIORS, "tau_s": ABC.Prior(0.05, 3.0)}},
    # Multi-horizon: M2 plus long closed-loop drives with the measured collie policy.
    "M3": {"dt": 1.0, "priors": SHARE_PRIORS, "multi": True},
    "M3tau": {"dt": 0.2, "priors": {**SHARE_PRIORS, "tau_s": ABC.Prior(0.05, 3.0)}, "multi": True},
}


def measured_dog():
    """The real collie's policy parameters (shepherd/dog_policy.py), as per-run model fields."""
    from shepherd import data, dog_policy as DP

    m = DP.estimate(data.load_drives(data.load_sessions()))
    return {"l_drive": m["l_drive"], "v_dog": m["v_dog"], "tau_d": m["tau_d"], "flank_sigma": m["flank_sigma"],
            "flank_tau": m["flank_tau"], "dog_speed_gain": m["dog_speed_gain"], "noise_dog": 0.3,
            "l_sep": 11.6, "l_collect": 2.0}

BASELINES = {
    "M0": (1.0, Params(), {}),  # exactly as published (constant 1 m/s)
    "M0s": (1.0, Params(), REPLAY),  # published rules, real flock speed
    "M0b": (1.0, Params(r_rep=1.2), REPLAY),  # + personal space = body length
    "M0s_dt02": (0.2, Params(), REPLAY),
}


def build_model(name, n1, n2, seed, workers):
    spec = MODELS[name]
    dt, priors = spec["dt"], spec["priors"]
    base = Params(v_sheep=V_SHEEP[dt])
    meta = ABC.window_meta(dt)
    cols = [PR.SUMMARY_KEYS.index(k) for k in FIT_STATS]
    obs = ABC.aggregate(meta["obs"][:, cols], meta["durations"])

    multi = spec.get("multi", False)
    dog = measured_dog() if multi else None
    extra = {}
    if multi:
        lmeta = ABC.long_meta(dt)
        obs = np.concatenate([obs, ABC.aggregate(lmeta["obs"][None], lmeta["durations"])[0]])
        extra.update({f"lmeta_{k}": v for k, v in lmeta.items()})

    def table(theta, s):
        if not multi:
            return ABC.simulate_table(theta, base, dt, seed=s, workers=workers, window_kw=REPLAY), None
        return ABC.simulate_table_multi(theta, base, dt, seed=s, workers=workers, window_kw=REPLAY, dog=dog)

    def combined(tab, ltab):
        S = ABC.aggregate(ABC.sanitize(tab)[..., cols], meta["durations"])
        return S if ltab is None else np.concatenate([S, ABC.aggregate(ltab, lmeta["durations"])], axis=1)

    t = time.time()
    th1 = ABC.sobol_prior(priors, n1, seed)
    tab1, ltab1 = table(th1, seed * 1_000_000)
    print(f"{name}: round 1, {n1} sets x {tab1.shape[1]} windows in {time.time() - t:.0f}s")
    sims1 = combined(tab1, ltab1)
    post1 = ABC.fit(th1, sims1, obs, ABC.mad_scale(sims1), priors, accept=0.02)
    box = ABC.truncated_box(post1, priors)
    print("  round-1 posterior:", {k: [round(v["median"], 3), [round(c, 3) for c in v["ci90"]]]
                                   for k, v in post1.summary(priors).items()})

    t = time.time()
    th2 = ABC.sobol_prior(priors, n2, seed + 1, box=box)
    tab2, ltab2 = table(th2, (seed + 1) * 1_000_000)
    if multi:
        extra.update({"ltab1": ltab1, "ltab2": ltab2, "long_stats": np.array(PR.LONG_KEYS)})
        post2 = ABC.fit(th2, combined(tab2, ltab2), obs, ABC.mad_scale(combined(tab2, ltab2)), priors, accept=0.01)
        print("  round-2 posterior:", {k: [round(v["median"], 3), [round(c, 3) for c in v["ci90"]]]
                                       for k, v in post2.summary(priors).items()})
    print(f"{name}: round 2, {n2} sets in {time.time() - t:.0f}s; box {box}")

    TABLES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TABLES / f"{name}.npz",
        dt=dt, v_sheep=base.v_sheep, stats=np.array(PR.SUMMARY_KEYS), fit_stats=np.array(FIT_STATS),
        names=np.array(list(priors)), **{f"th1_{k}": v for k, v in th1.items()}, tab1=tab1,
        **{f"th2_{k}": v for k, v in th2.items()}, tab2=tab2,
        box=np.array([box[k] if box[k] is not None else (np.nan, np.nan) for k in priors]),
        **{f"meta_{k}": v for k, v in meta.items()}, **extra,
    )


HYBRID_PRIORS = {
    "E": ABC.Prior(0.2, 10.0),  # gain on the learned attraction/repulsion kernel
    "dog": ABC.Prior(0.2, 10.0),  # common gain on the three learned dog kernels
    "sigma": ABC.Prior(0.02, 1.0),  # velocity noise (m/s per sqrt(s))
}


def build_hybrid(n, seed, workers, dt=0.2):
    """Learned kernel shapes x calibrated gains. One table per held-out session
    (shapes learned without it) plus one for all sessions."""
    from shepherd import data, kernels as K

    fit = np.load(ROOT / "results" / "tables" / "kernel_fit.npz")
    model = K.KernelModel(terms=tuple(str(t) for t in fit["terms"]))
    drives = data.load_drives(data.load_sessions())
    normals = K.normal_equations(model, drives, H=float(fit["H"]), window_s=float(fit["window_s"]))
    meta = ABC.window_meta(dt)
    folds = ["all"] + sorted({int(s) for s in meta["sessions"]})
    tables = {}
    for f in folds:
        train = [nn for nn in normals if f == "all" or nn.session != f]
        c = K.solve(model, train, float(fit["lam"]))
        th = ABC.sobol_prior(HYBRID_PRIORS, n, seed + (0 if f == "all" else f))
        gains = {"E": th["E"], "D": th["dog"], "DV": th["dog"], "DL": th["dog"], "sigma": th["sigma"]}
        t = time.time()
        tab = ABC.simulate_table_kernel(gains, model, c, dt, seed=seed * 1_000_000, workers=workers, chunk=64)
        print(f"hybrid fold {f}: {n} sets x {tab.shape[1]} windows in {time.time() - t:.0f}s")
        tables[f"coef_{f}"], tables[f"tab_{f}"] = c, tab
        tables.update({f"th_{f}_{k}": v for k, v in th.items()})
    np.savez_compressed(TABLES / "hybrid.npz", dt=dt, folds=np.array([str(f) for f in folds]),
                        names=np.array(list(HYBRID_PRIORS)), terms=np.array(model.terms), stats=np.array(PR.SUMMARY_KEYS),
                        **tables, **{f"meta_{k}": v for k, v in meta.items()})


def build_baselines(reps, seed, workers):
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, (dt, p, kw) in BASELINES.items():
        theta = {"r_rep": np.full(reps, p.r_rep)}  # replicate the fixed parameter set
        tab = ABC.simulate_table(theta, p, dt, seed=seed, workers=workers, chunk=max(1, reps // 4), window_kw=kw)
        meta = ABC.window_meta(dt)
        np.savez_compressed(TABLES / f"{name}.npz", dt=dt, tab=tab, stats=np.array(PR.SUMMARY_KEYS),
                            **{f"meta_{k}": v for k, v in meta.items()})
        print(f"{name}: {reps} replicates x {tab.shape[1]} windows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--n1", type=int, default=50000)
    ap.add_argument("--n2", type=int, default=25000)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--n", type=int, default=2048, help="parameter sets per hybrid fold")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.baselines:
        build_baselines(args.reps, args.seed, args.workers)
    if args.model:
        build_model(args.model, args.n1, args.n2, args.seed, args.workers)
    if args.hybrid:
        build_hybrid(args.n, args.seed, args.workers)


if __name__ == "__main__":
    main()
