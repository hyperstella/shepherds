"""Counterexamples to MI+, simulated forward under the baseline dog, and a multi-step drift condition.

MI+ (docs/jadhav_dilorenzo_bridge.md §11.4) asks V <= lambda Q+ - k mean|xi| (lambda = 0.35). A multi-start
adversary over the 14 headings (polarization >= 0.5) finds heading fields that violate it at some of the
baseline driving states outside the ball. At those states V > 0 while Q is about 0, so every drift function
of the form M2 + f(Q), f convex and nondecreasing, increases in expectation (Jensen).

This script:
 1. finds the counterexamples (single-start adversary on all states, then 12 starts on the 30 tightest) and
    checks each exactly;
 2. simulates each counterexample forward under the baseline closed loop (published Strömbom dog with
    collecting, published sheep rule), REPS replicates for T steps, against the same positions with their
    natural headings; records M2, return to the ball, coverage, dog mode, polarization, splits, and the
    exact one-step virial V and covariance Q along the mean path;
 3. tests the m-step drift condition E[M2(t+m) | x] - M2(t) < 0 for m = 1..M_MAX on the counterexamples,
    on all 30 adversarial fields, and on the 255 natural states.

    .venv/bin/python experiments/mi_counterexamples.py
Writes results/mi_counterexamples.json and results/figures/mi_counterexamples.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import memory_inequality as mi  # noqa: E402
from shepherd.expected_heading import expected_headings  # noqa: E402
from shepherd.model import DRIVE, Params, simulate  # noqa: E402

OUT = ROOT / "results"
P0 = Params()
N, V_S = P0.n_sheep, P0.v_sheep
LAM, RHO, LINK = 0.35, 4.46, 6.0
TIGHT, STARTS, ITERS_STRONG, ITERS_FIRST = 30, 12, 30, 15
WORKERS = 10
REPS, T = 256, 80
M_MAX, REPS_M = 10, 128
SEED = 23
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ------------------------------------------------------------------ 1. counterexamples

def _first(args):
    sub, seed = args
    mi.ADV_ITERS = ITERS_FIRST
    Lh = mi.memory("scalar+", LAM, sub["pos"])
    _, fb, th = mi.adversary(sub, Lh, seed, plus=LAM)
    return fb, th


def _strong(args):
    """STARTS starts (the natural headings, then random spreads of 20-90 degrees) of ITERS_STRONG iterations on one state."""
    one, seed = args
    mi.ADV_ITERS = ITERS_STRONG
    Lh = mi.memory("scalar+", LAM, one["pos"])
    rng = np.random.default_rng(seed)
    best, bth = -np.inf, None
    for s in range(STARTS):
        hd = one["hd"]
        if s:
            hb = one["hd"].mean(1)
            th0 = np.arctan2(hb[:, 1], hb[:, 0])
            ang = th0[:, None] + np.deg2rad(rng.uniform(20, 90)) * rng.uniform(-1, 1, size=(1, N))
            hd = np.stack([np.cos(ang), np.sin(ang)], -1)
        _, fb, th = mi.adversary(dict(one, hd=hd), Lh, seed + s, plus=LAM)
        if fb[0] > best:
            best, bth = fb[0], th[0]
    return best, bth


def exact(pos, hd, dog):
    """V, Q, mean|xi|, P, D and whether every sheep sees the dog, exactly."""
    mu, _ = expected_headings(pos, hd, dog, P0)
    xi = pos - pos.mean(1, keepdims=True)
    mubar = mu.mean(1)
    return dict(V=np.einsum("snc,snc->sn", xi, mu).mean(1), Q=np.einsum("snc,snc->sn", xi, hd).mean(1), m=np.linalg.norm(xi, axis=-1).mean(1),
                P=np.linalg.norm(hd.mean(1), axis=-1), D=1 - ((mubar ** 2).sum(-1) + (1 - (mu ** 2).sum(-1)).sum(1) / N ** 2),
                sees_all=np.linalg.norm(pos - dog[:, None], axis=-1).max(1) <= P0.r_dog, M2=(xi ** 2).sum(-1).mean(1))


def find_counterexamples(st):
    S = len(st["pos"])
    chunks = np.array_split(np.arange(S), WORKERS * 2)
    with Pool(WORKERS) as pool:
        parts = pool.map(_first, [({k: v[c] for k, v in st.items()}, SEED) for c in chunks])
    first = np.concatenate([p[0] for p in parts])
    tight = np.argsort(-first)[:TIGHT]
    with Pool(WORKERS) as pool:
        strong = pool.map(_strong, [({k: v[[i]] for k, v in st.items()}, SEED * 100 + int(i)) for i in tight])
    th = np.stack([s[1] for s in strong])
    hd = np.stack([np.cos(th), np.sin(th)], -1)
    ex = exact(st["pos"][tight], hd, st["dog"][tight])
    viol = (ex["V"] - LAM * np.clip(ex["Q"], 0, None)) / ex["m"]
    return dict(first=first, tight=tight, hd=hd, ex=ex, viol=viol)


# ------------------------------------------------------------------ 2. forward simulation

def largest_group(pos):
    d = np.linalg.norm(pos[:, None] - pos[:, :, None], axis=-1)
    reach = d <= LINK
    for _ in range(4):
        reach = (reach.astype(np.int32) @ reach.astype(np.int32)) > 0
    return reach.sum(-1).max(-1)


def forward(pos, hd, dog, dog_v, reps, steps, seed):
    """reps replicates of the baseline closed loop from each state; returns per-step arrays (S, reps, steps + 1)."""
    S = len(pos)
    rep = lambda a: np.repeat(a, reps, 0)
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=steps + 1), n_runs=S * reps, seed=seed,
                   init=dict(pos_s=rep(pos), vel_s=rep(hd), pos_d=rep(dog), vel_d=rep(dog_v)))
    P = res.pos_s  # (R, N, 2, T)
    xi = P - P.mean(1, keepdims=True)
    M2 = (xi ** 2).sum(2).mean(1)
    sees = (np.linalg.norm(P - res.pos_d[:, None], axis=2) <= P0.r_dog).all(1)
    pol = np.linalg.norm(res.vel_s.mean(1), axis=1)
    groups = np.stack([largest_group(np.moveaxis(P[..., t], 0, 0)) for t in range(steps + 1)], -1)
    shape = lambda a: a.reshape(S, reps, -1)
    # where the out-of-range sheep are: share of them ahead of the centroid along the flock's mean heading
    hb = res.vel_s.mean(1)
    ahead = np.einsum("rnct,rct->rnt", xi, hb / np.maximum(np.linalg.norm(hb, axis=1, keepdims=True), 1e-12)) > 0
    out = np.linalg.norm(P - res.pos_d[:, None], axis=2) > P0.r_dog
    n_out = out.sum(1)
    ahead_share = np.where(n_out > 0, (out & ahead).sum(1) / np.maximum(n_out, 1), np.nan)
    return dict(M2=shape(M2), sees=shape(sees), P=shape(pol), drive=shape(res.mode == DRIVE), group=shape(groups), ahead_share=shape(ahead_share),
                pos=res.pos_s.reshape(S, reps, N, 2, -1), hd=res.vel_s.reshape(S, reps, N, 2, -1), dog=res.pos_d.reshape(S, reps, 2, -1))


def first_step(flag):
    """First step at which flag holds (inf if never), per replicate."""
    t = np.argmax(flag, -1).astype(float)
    t[~flag.any(-1)] = np.inf
    return t


def first_return(M2):
    """First step with sqrt(M2) < RHO (inf if never), per replicate."""
    inside = np.sqrt(M2) < RHO
    t = np.argmax(inside, -1).astype(float)
    t[~inside.any(-1)] = np.inf
    return t


def mean_path_virial(fw, steps):
    """Exact V/mean|xi| and Q along the path, averaged over replicates, every step (states of all replicates)."""
    S, R = fw["pos"].shape[:2]
    out_V, out_Q = np.empty((S, steps + 1)), np.empty((S, steps + 1))
    for t in range(steps + 1):
        pos = fw["pos"][..., t].reshape(S * R, N, 2)
        hd = fw["hd"][..., t].reshape(S * R, N, 2)
        dog = fw["dog"][..., t].reshape(S * R, 2)
        e = exact(pos, hd, dog)
        out_V[:, t] = (e["V"] / e["m"]).reshape(S, R).mean(1)
        out_Q[:, t] = (e["Q"] / e["m"]).reshape(S, R).mean(1)
    return out_V, out_Q


# ------------------------------------------------------------------ 3. multi-step drift

def m_step(pos, hd, dog, dog_v, seed):
    fw = forward(pos, hd, dog, dog_v, REPS_M, M_MAX, seed)
    M2 = fw["M2"]
    d = M2[..., 1:].mean(1) - M2[..., :1].mean(1)
    se = M2[..., 1:].std(1) / np.sqrt(REPS_M)
    collect = (~fw["drive"][..., 1:]).mean(1)  # share of replicates not driving at each step
    lost = (~fw["sees"][..., 1:]).mean(1)
    return d, se, collect, lost


def summarize_m(d, se, sqrtM2):
    rows = []
    for j in range(M_MAX):
        rows.append(dict(m=j + 1, n_positive=int((d[:, j] > 0).sum()), n_positive_2se=int((d[:, j] > 2 * se[:, j]).sum()), max=fl(d[:, j].max()),
                         median=fl(np.median(d[:, j])), max_over_sqrtM2=fl((d[:, j] / sqrtM2).max())))
    neg_from = next((r["m"] for r in rows if r["max"] < 0), None)
    return rows, neg_from


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    st = mi.outside_ball_states()
    S = len(st["pos"])
    print(f"{S} driving states outside the ball", flush=True)
    ce = find_counterexamples(st)
    viol, ex, tight = ce["viol"], ce["ex"], ce["tight"]
    is_ce = (viol > 0) & (ex["P"] >= 0.5) & ex["sees_all"]
    idx = tight[is_ce]
    print(f"counterexamples: {is_ce.sum()} of the {TIGHT} tightest states (V - {LAM} Q+ > 0, P >= 0.5, every sheep seeing the dog); "
          f"violation / mean|xi|: {np.round(np.sort(viol[is_ce])[::-1], 4).tolist()} ({time.time() - tic:.0f} s)", flush=True)
    jensen = (ex["V"][is_ce] > 0) & (ex["Q"][is_ce] <= ex["V"][is_ce] + V_S * ex["D"][is_ce])
    print(f"   Jensen obstruction (V > 0 and Q <= V + vD): {jensen.sum()} of {is_ce.sum()}", flush=True)

    # forward: counterexamples and their natural-heading controls
    cs = {k: v[idx] for k, v in st.items()}
    hd_ce = ce["hd"][is_ce]
    fw_ce = forward(cs["pos"], hd_ce, cs["dog"], cs["dog_v"], REPS, T, SEED)
    fw_nat = forward(cs["pos"], cs["hd"], cs["dog"], cs["dog_v"], REPS, T, SEED)
    V_ce, Q_ce = mean_path_virial({k: v[:, :32, ..., :21] for k, v in fw_ce.items() if k in ("pos", "hd", "dog")}, 20)
    rows = []
    for j, s in enumerate(idx):
        tr_ce, tr_nat = first_return(fw_ce["M2"][j]), first_return(fw_nat["M2"][j])
        M2m = fw_ce["M2"][j].mean(0)
        peak = int(np.argmax(M2m))
        rows.append(dict(state=int(s), violation=fl(viol[is_ce][j]), P0=fl(ex["P"][is_ce][j]), Q0=fl(ex["Q"][is_ce][j]), V0_over_m=fl(ex["V"][is_ce][j] / ex["m"][is_ce][j]),
                         sqrtM2_0=fl(np.sqrt(ex["M2"][is_ce][j])),
                         returned=fl(np.isfinite(tr_ce).mean()), return_median=fl(np.median(tr_ce)), return_p90=fl(np.percentile(tr_ce, 90)),
                         returned_natural=fl(np.isfinite(tr_nat).mean()), return_median_natural=fl(np.median(tr_nat)), return_p90_natural=fl(np.percentile(tr_nat, 90)),
                         expansion_steps=int(np.argmax(np.r_[np.diff(M2m) < 0, True])), M2_peak_rise=fl(M2m[peak] - M2m[0]), M2_peak_step=peak,
                         M2_mean_at=[fl(M2m[t]) for t in (0, 1, 2, 3, 5, 10, 20, 40, 80)],
                         coverage_lost_ever=fl((~fw_ce["sees"][j]).any(-1).mean()), collect_ever=fl((~fw_ce["drive"][j][:, 1:]).any(-1).mean()),  # step 0 is labelled INIT
                         coverage_lost_first_median=fl(np.median(first_step(~fw_ce["sees"][j]))), collect_first_median=fl(np.median(first_step(~fw_ce["drive"][j][:, 1:]) + 1)),
                         coverage_lost_first_median_natural=fl(np.median(first_step(~fw_nat["sees"][j]))),
                         collect_first_median_natural=fl(np.median(first_step(~fw_nat["drive"][j][:, 1:]) + 1)),
                         collect_within_3=fl((~fw_ce["drive"][j][:, 1:4]).any(-1).mean()), coverage_lost_within_3=fl((~fw_ce["sees"][j][:, 1:4]).any(-1).mean()),
                         collect_within_3_natural=fl((~fw_nat["drive"][j][:, 1:4]).any(-1).mean()),
                         coverage_lost_within_3_natural=fl((~fw_nat["sees"][j][:, 1:4]).any(-1).mean()),
                         out_of_range_ahead_share_steps_1_3=fl(np.nanmean(fw_ce["ahead_share"][j][:, 1:4])),
                         out_of_range_ahead_share_steps_1_3_natural=fl(np.nanmean(fw_nat["ahead_share"][j][:, 1:4])),
                         returned_without_collect_by_return=fl(np.mean([np.isfinite(tr_ce[q]) and not (~fw_ce["drive"][j][q, 1:int(tr_ce[q]) + 1]).any()
                                                                       for q in range(len(tr_ce))])),
                         split_end=fl((fw_ce["group"][j][:, -1] < N).mean()), split_ever=fl((fw_ce["group"][j] < N).any(-1).mean()),
                         P_path=[fl(v) for v in fw_ce["P"][j].mean(0)[:21]], V_over_m_path=[fl(v) for v in V_ce[j]], Q_over_m_path=[fl(v) for v in Q_ce[j]],
                         M2_path=[fl(v) for v in M2m], M2_path_natural=[fl(v) for v in fw_nat["M2"][j].mean(0)]))
        r = rows[-1]
        print(f"   state {s:3d}: violation {r['violation']:+.4f}, sqrt M2 {r['sqrtM2_0']:.2f}, P {r['P0']:.2f} | back in the ball: {r['returned']:.2f} of runs, "
              f"median {r['return_median']} steps (natural headings {r['returned_natural']:.2f}, {r['return_median_natural']}) | M2 grows for {r['expansion_steps']} steps, "
              f"peak +{r['M2_peak_rise']:.2f} m2 at step {r['M2_peak_step']} | V/m path {np.round(V_ce[j][:6], 3).tolist()} | coverage lost {r['coverage_lost_ever']:.2f}, "
              f"collect {r['collect_ever']:.2f}, split ever {r['split_ever']:.2f}, at end {r['split_end']:.2f}", flush=True)
        print(f"      first coverage loss median step {r['coverage_lost_first_median']} (natural {r['coverage_lost_first_median_natural']}), first collect {r['collect_first_median']} "
              f"(natural {r['collect_first_median_natural']}); within steps 1-3: coverage lost {r['coverage_lost_within_3']:.2f} (natural {r['coverage_lost_within_3_natural']:.2f}), "
              f"collect {r['collect_within_3']:.2f} (natural {r['collect_within_3_natural']:.2f}); back in the ball before any collecting {r['returned_without_collect_by_return']:.2f}; "
              f"out-of-range sheep ahead of the centroid (steps 1-3): {r['out_of_range_ahead_share_steps_1_3']:.2f} (natural {r['out_of_range_ahead_share_steps_1_3_natural']:.2f})", flush=True)

    # multi-step drift on three sets
    msets = {}
    for name, pos, hd, dog, dv in (("counterexamples", cs["pos"], hd_ce, cs["dog"], cs["dog_v"]),
                                   ("adversarial fields, 30 tightest", st["pos"][tight], ce["hd"], st["dog"][tight], st["dog_v"][tight]),
                                   ("natural headings, all 255", st["pos"], st["hd"], st["dog"], st["dog_v"])):
        d, se, collect, lost = m_step(pos, hd, dog, dv, SEED + 1)
        sq = np.sqrt(((pos - pos.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))
        rows_m, neg_from = summarize_m(d, se, sq)
        for j, r in enumerate(rows_m):
            r.update(collect_share=fl(collect[:, j].mean()), coverage_lost_share=fl(lost[:, j].mean()))
        msets[name] = dict(n=int(len(pos)), rows=rows_m, every_state_negative_from=neg_from)
        print(f"   m-step drift, {name}: states with E[M2(t+m)] - M2(t) > 0 for m = 1..{M_MAX}: " + " ".join(str(r["n_positive"]) for r in rows_m) +
              f" (by > 2 SE: " + " ".join(str(r["n_positive_2se"]) for r in rows_m) + f"); every state negative from m = {neg_from}; "
              f"largest at m = 3: {rows_m[2]['max']:+.3f} m2; share of replicates collecting at steps 1-3: " + " ".join(f"{r['collect_share']:.2f}" for r in rows_m[:3]) +
              ", with a sheep out of range: " + " ".join(f"{r['coverage_lost_share']:.2f}" for r in rows_m[:3]), flush=True)

    out = dict(setup=dict(lam=LAM, rho=RHO, link_m=LINK, tight=TIGHT, starts=STARTS, iters=[ITERS_FIRST, ITERS_STRONG], reps=REPS, steps=T, m_max=M_MAX, reps_m=REPS_M,
                          seed=SEED, controller="baseline closed loop: published Strömbom dog (drive and collect), published sheep rule"),
               adversary=dict(first_pass_worst=fl(ce["first"].max()), tight_states=tight.tolist(), violations=[fl(v) for v in viol], P=[fl(v) for v in ex["P"]],
                              Q=[fl(v) for v in ex["Q"]], counterexamples=idx.tolist(), jensen_obstruction=int(jensen.sum()),
                              fields_theta=[np.arctan2(h[:, 1], h[:, 0]).tolist() for h in ce["hd"]]),
               forward=rows, multi_step=msets, seconds=time.time() - tic)
    (OUT / "mi_counterexamples.json").write_text(json.dumps(out, indent=1))
    figure(rows, msets)
    print(f"done in {time.time() - tic:.0f} s")


def figure(rows, msets):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    for r, c in zip(rows, plt.cm.viridis(np.linspace(0, 0.9, len(rows)))):
        ax.plot(np.sqrt(r["M2_path"]), color=c, lw=1.6, label=f"state {r['state']}")
        ax.plot(np.sqrt(r["M2_path_natural"]), color=c, lw=0.9, ls=":")
    ax.axhline(RHO, color=COL["gray"], lw=0.8, ls="--")
    ax.set(xlabel="step", ylabel="√ of mean M₂ over replicates (m)", title="Counterexamples forward, baseline dog\n(solid: adversarial headings; dotted: natural; dashed: ball)")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    for r, c in zip(rows, plt.cm.viridis(np.linspace(0, 0.9, len(rows)))):
        ax.plot(r["V_over_m_path"], color=c, lw=1.6)
        ax.plot(r["Q_over_m_path"], color=c, lw=0.9, ls="--")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="step", ylabel="per mean|ξ| (mean over replicates)", title="Virial V (solid) and covariance Q (dashed) along the path")
    ax = axs[2]
    for (name, ms), col in zip(msets.items(), ("red", "orange", "blue")):
        ax.plot([r["m"] for r in ms["rows"]], [r["max"] for r in ms["rows"]], "o-", color=COL[col], lw=1.8, label=f"{name}: largest")
        ax.plot([r["m"] for r in ms["rows"]], [r["median"] for r in ms["rows"]], ":", color=COL[col], lw=1.2)
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="horizon m (steps)", ylabel="E[M₂(t+m)] − M₂(t)  (m²)", title="m-step drift (solid: largest over states; dotted: median)")
    ax.legend(frameon=False, fontsize=7)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "mi_counterexamples.png", dpi=130)


if __name__ == "__main__":
    main()
