"""Where does the shape drift lemma's region hold? Stressed flock states, ablations, a geometric certificate.

docs/steer_and_shape.md §10 proved, from the exact second-moment identity, that on the set G of
states in which every sheep moves, the weighted inward relative heading kappa >= kappa* and the
mean-heading polarization P >= P*, E[dM2 | state] <= -a sqrt(M2) + b. It checked G only on
observed driven states dilated about their centroid. Here observed states (published rule, the
dog 7.5 m behind, its observed position kept) are deformed instead:

  stretch along / across the heading, shear, split into two halves (across or along the
  heading, by a gap), one sheep moved out (to the side, ahead, behind), heading noise, half
  the headings reversed,

and, separately, synthetic compact flocks of 14 and 30 sheep split into halves (a half of 15 is
larger than the 10 neighbours each sheep perceives). For every state the conditional mean headings come from K one-step Monte Carlo draws
(as in experiments/shape_drift.py). Measured: whether every sheep moves, kappa, P, E[dM2 | state].
Ablations (attraction, dog push or alignment weight set to 0, same states and draws) show which
term supplies or opposes kappa.

Geometric certificate (proved in the doc): if all K perceived neighbours of a sheep lie in a cone
of half-angle alpha < 90° about the inward direction u, every random attraction subset gives
A . u >= cos(alpha). A sheep at the side of the flock then heads inward on *every* draw if
    w_att cos(alpha) + w_in h.u + w_ali min_j h_j.u + w_dog e_dog.u - noise - w_rep [repelled] > 0.
The script reports how often lateral outer sheep are so certified.

    .venv/bin/python experiments/shape_stress.py
Writes results/shape_stress.json and results/figures/shape_stress.png.
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

from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
T_RUN, SNAPS, REPS, K, SEED = 300, (150, 180, 210, 240, 270), 6, 128, 53
FAR, D_STAND, V = 2000.0, 7.5, 1.0
KAPPA_STAR = 0.05
CHUNK_RUNS = 24000
P0 = Params()
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}

FAMILIES = [
    ("observed", None, (1,)),
    ("stretch along", "stretch_par", (2.0, 3.0)),
    ("stretch across", "stretch_perp", (2.0, 3.0)),
    ("shear", "shear", (0.5, 1.0)),
    ("split across", "split_perp", (4.0, 8.0, 12.0)),
    ("split along", "split_par", (4.0, 8.0, 12.0)),
    ("outlier to the side", "out_side", (6.0, 10.0, 14.0)),
    ("outlier ahead", "out_front", (6.0, 10.0)),
    ("outlier behind", "out_back", (6.0, 10.0)),
    ("heading noise", "head_noise", (0.5, 1.0, 3.0)),
    ("half reversed", "head_reverse", (1,)),
]
SETTINGS = [
    dict(name="published, r_dog 20", N=14, w_att=1.5, w_ali=1.3, r_dog=20.0, ablate=True),
    dict(name="published, r_dog 12", N=14, w_att=1.5, w_ali=1.3, r_dog=12.0, ablate=False),
    dict(name="strong, r_dog 20", N=14, w_att=3.0, w_ali=3.0, r_dog=20.0, ablate=True),
]
# Topological test on synthetic compact flocks: a disc of radius 0.9 sqrt(N) m (about the observed 14-sheep
# density), aligned headings, the dog 7.5 m behind with r_dog 40 so every sheep moves, split into halves
# across the heading. A half larger than the K = 10 perceived neighbours cannot see the other half.
TOPO = dict(sizes=(14, 30), gaps=(0.0, 4.0, 8.0, 12.0), w_att=1.5, w_ali=1.3, r_dog=40.0, n_states=30)
ABLATIONS = {"full": {}, "no attraction": {"w_att": 0.0}, "no dog push": {"w_dog": 0.0}, "no alignment": {"w_ali": 0.0}}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def boot_ci(x, rng, n=2000):
    """95% bootstrap interval of the median over states."""
    if len(x) < 3:
        return None
    meds = np.median(rng.choice(x, (n, len(x))), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ states

def observed_states(st):
    """Snapshots of driven flocks. The 30-sheep flocks start compact and aligned (radius 4 m, heading to the
    target, dog 7.5 m behind): from the model's spread-out default start, with collecting off, they never cohere."""
    N = st["N"]
    p = Params(n_sheep=N, box_length=FAR, n_steps=T_RUN)
    per = dict(w_att=np.full(REPS, st["w_att"]), w_ali=np.full(REPS, st["w_ali"]), r_dog=np.full(REPS, st["r_dog"]),
               l_drive=np.full(REPS, D_STAND), l_sep=np.full(REPS, 1e6))
    init = None
    if N > 14:
        rng = np.random.default_rng(SEED + N)
        c0 = np.array([FAR, 0.0])
        rad, th = 4.0 * np.sqrt(rng.random((REPS, N))), 2 * np.pi * rng.random((REPS, N))
        pos = c0 + np.stack([rad * np.cos(th), rad * np.sin(th)], -1)
        hd = unit(np.array([-1.0, 0.0]) + 0.2 * rng.standard_normal((REPS, N, 2)))
        init = dict(pos_s=pos, vel_s=hd, pos_d=np.tile(c0 + np.array([D_STAND, 0.0]), (REPS, 1)), vel_d=np.tile([-1.0, 0.0], (REPS, 1)))
    res = simulate(p, n_runs=REPS, seed=SEED + N, per_run=per, init=init)
    pos = np.concatenate([res.pos_s[..., t] for t in SNAPS])  # (S, N, 2)
    hd = np.concatenate([res.vel_s[..., t] for t in SNAPS])
    dog = np.concatenate([res.pos_d[..., t] for t in SNAPS])
    return pos, hd, dog


def deform(kind, level, pos, hd, rng):
    """Apply one deformation to states (S, N, 2); the dog stays where it was."""
    if kind is None:
        return pos.copy(), hd.copy()
    c = pos.mean(1, keepdims=True)
    e = unit(hd.mean(1))[:, None, :]  # flock heading
    el = np.concatenate([-e[..., 1:], e[..., :1]], -1)
    xi = pos - c
    par, perp = (xi * e).sum(-1, keepdims=True), (xi * el).sum(-1, keepdims=True)
    new_h = hd.copy()
    if kind == "stretch_par":
        xi = level * par * e + perp * el
    elif kind == "stretch_perp":
        xi = par * e + level * perp * el
    elif kind == "shear":
        xi = par * e + (perp + level * par) * el
    elif kind in ("split_perp", "split_par"):
        coord = perp if kind == "split_perp" else par
        axis = el if kind == "split_perp" else e
        rank = np.argsort(np.argsort(coord[..., 0], axis=1), axis=1)  # equal halves, so the centroid stays
        side = np.where(rank >= pos.shape[1] // 2, 1.0, -1.0)[..., None]
        xi = xi + side * (level / 2) * axis
    elif kind in ("out_side", "out_front", "out_back"):
        coord = {"out_side": perp, "out_front": par, "out_back": -par}[kind][..., 0]
        k = coord.argmax(1)
        s = np.arange(len(pos))
        if kind == "out_side":
            xi[s, k] = par[s, k] * e[:, 0] + level * el[:, 0]
        else:
            sign = 1.0 if kind == "out_front" else -1.0
            xi[s, k] = sign * level * e[:, 0] + perp[s, k] * el[:, 0]
    elif kind == "head_noise":
        new_h = unit(hd + level * rng.standard_normal(hd.shape))
    elif kind == "head_reverse":
        flip = rng.random(hd.shape[:2]) < 0.5
        new_h = np.where(flip[..., None], -hd, hd)
    return c + xi, new_h


# ------------------------------------------------------------------ one-step Monte Carlo and the exact decomposition

def one_step(st, pos, hd, dog, seed, override):
    S, N = pos.shape[:2]
    R = S * K
    per = dict(w_att=np.full(R, st["w_att"]), w_ali=np.full(R, st["w_ali"]), r_dog=np.full(R, st["r_dog"]))
    per.update({k: np.full(R, v) for k, v in override.items()})
    dp = np.repeat(dog, K, 0)
    res = simulate(Params(n_sheep=N, n_steps=2), n_runs=R, seed=seed, per_run=per,
                   init=dict(pos_s=np.repeat(pos, K, 0), vel_s=np.repeat(hd, K, 0)), dog_path=np.stack([dp, dp], -1))
    return res.pos_s[..., -1].reshape(S, K, N, 2)


def analyse(pos, new):
    _, N = pos.shape[:2]
    xi = pos - pos.mean(1, keepdims=True)
    M2 = (xi ** 2).sum(-1).mean(-1)
    a = new - pos[:, None]
    dM2_k = ((new - new.mean(2, keepdims=True)) ** 2).sum(-1).mean(-1) - M2[:, None]  # (S, K)
    dM2 = dM2_k.mean(1)
    dM2_se = dM2_k.std(1, ddof=1) / np.sqrt(dM2_k.shape[1])
    step = np.linalg.norm(a, axis=-1)
    m = (step > 1e-9).mean(1) > 0.5
    h = np.where(step[..., None] > 1e-9, a / np.maximum(step[..., None], 1e-12), 0.0)
    mu = h.mean(1)
    f_m = m.mean(1)
    Mv = (m[..., None] * mu).mean(1)
    mubar = Mv / np.maximum(f_m, 1e-12)[:, None]
    r = np.linalg.norm(xi, axis=-1)
    rel = mu - mubar[:, None]
    kappa = -(m * (xi * rel).sum(-1)).sum(1) / r.sum(1)
    G = V ** 2 * (f_m - (Mv ** 2).sum(-1) - (m * (1 - (mu ** 2).sum(-1))).sum(1) / N ** 2)
    # Monte Carlo standard error of kappa from 4 batches of the K draws
    kb = []
    for hb in np.array_split(h, 4, axis=1):
        mub = hb.mean(1)
        mubar_b = (m[..., None] * mub).mean(1) / np.maximum(f_m, 1e-12)[:, None]
        kb.append(-(m * (xi * (mub - mubar_b[:, None])).sum(-1)).sum(1) / r.sum(1))
    kappa_se = np.std(kb, axis=0, ddof=1) / 2.0
    Pb = [np.linalg.norm((m[..., None] * hb.mean(1)).mean(1) / np.maximum(f_m, 1e-12)[:, None], axis=-1) for hb in np.array_split(h, 4, axis=1)]
    P_se = np.std(Pb, axis=0, ddof=1) / 2.0
    return dict(M2=M2, dM2=dM2, dM2_se=dM2_se, f_m=f_m, kappa=kappa, kappa_se=kappa_se, P=np.linalg.norm(mubar, axis=-1), P_se=P_se, G=G)


def certificate(st, pos, hd, dog):
    """Fraction of lateral outer sheep that head inward on every draw (the cone certificate)."""
    S, N = pos.shape[:2]
    w = dict(att=st["w_att"], ali=st["w_ali"], ine=P0.w_inertia, dog=P0.w_dog, rep=P0.w_rep, eta=P0.noise)
    c = pos.mean(1, keepdims=True)
    xi = pos - c
    r = np.linalg.norm(xi, axis=-1)
    M2 = (r ** 2).mean(1)
    e = unit(hd.mean(1))
    u = -xi / np.maximum(r[..., None], 1e-12)  # inward
    lateral = np.abs((u * e[:, None]).sum(-1)) < 0.5
    outer = (r >= np.sqrt(M2)[:, None]) & lateral
    rij = pos[:, None, :, :] - pos[:, :, None, :]  # [s, i, j] = x_j - x_i
    dist = np.linalg.norm(rij, axis=-1)
    dist_s = np.where(np.eye(N, dtype=bool), np.inf, dist)
    knn = np.argsort(dist_s, axis=-1)[..., : P0.k_perceive]  # (S, N, K)
    s_idx = np.arange(S)[:, None, None]
    i_idx = np.arange(N)[None, :, None]
    dn = unit(rij[s_idx, i_idx, knn])  # unit vectors to the perceived neighbours
    cos_min = (dn * u[:, :, None, :]).sum(-1).min(-1)  # cos of the widest neighbour angle from u
    in_cone = cos_min > 0
    h_u = (hd * u).sum(-1)
    ali_min = (hd[s_idx, knn] * u[:, :, None, :]).sum(-1).min(-1)
    to_dog = dog[:, None, :] - pos
    dd = np.linalg.norm(to_dog, axis=-1)
    sees = dd <= st["r_dog"]
    dog_u = np.where(sees, (-to_dog / np.maximum(dd[..., None], 1e-12) * u).sum(-1), 0.0)
    repelled = (dist_s < P0.r_rep).any(-1)
    lower = w["att"] * cos_min + w["ine"] * h_u + w["ali"] * ali_min + w["dog"] * dog_u - w["eta"] - w["rep"] * repelled
    cert = outer & in_cone & (lower > 0)
    n_out = outer.sum(1)
    return np.where(n_out > 0, cert.sum(1) / np.maximum(n_out, 1), np.nan), np.where(n_out > 0, (outer & in_cone).sum(1) / np.maximum(n_out, 1), np.nan)


# ------------------------------------------------------------------ main

def run():
    tic = time.time()
    rng = np.random.default_rng(SEED)
    out = []
    seed = SEED * 1000
    for st in SETTINGS:
        pos0, hd0, dog0 = observed_states(st)
        N = st["N"]
        per_call = max(1, int(CHUNK_RUNS * (14 / N) ** 2) // K)
        for fam, kind, levels in FAMILIES:
            for lev in levels:
                pos, hd = deform(kind, lev, pos0, hd0, rng)
                res = {}
                for abl, override in ABLATIONS.items():
                    if abl != "full" and not st["ablate"]:
                        continue
                    parts = []
                    for i in range(0, len(pos), per_call):
                        seed += 1
                        sl = slice(i, i + per_call)
                        parts.append(analyse(pos[sl], one_step(st, pos[sl], hd[sl], dog0[sl], seed, override)))
                    res[abl] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                cert, cone = certificate(st, pos, hd, dog0)
                f = res["full"]
                allmove = f["f_m"] >= 0.999
                in_G = allmove & (f["kappa"] >= KAPPA_STAR)
                row = dict(setting=st["name"], N=N, family=fam, level=lev, n=int(len(pos)),
                           sqrtM2_median=fl(np.median(np.sqrt(f["M2"]))), dM2_median=fl(np.median(f["dM2"])),
                           dM2_positive=fl((f["dM2"] > 0).mean()), all_move=fl(allmove.mean()), in_G=fl(in_G.mean()),
                           kappa_median=fl(np.median(f["kappa"][allmove])) if allmove.any() else None,
                           kappa_p10=fl(np.percentile(f["kappa"][allmove], 10)) if allmove.any() else None,
                           kappa_nonpos=fl((f["kappa"][allmove] <= 0).mean()) if allmove.any() else None,
                           P_median=fl(np.median(f["P"])), G_median=fl(np.median(f["G"])),
                           dM2_se_median=fl(np.median(f["dM2_se"])), kappa_se_median=fl(np.median(f["kappa_se"])),
                           dM2_median_ci=boot_ci(f["dM2"], rng), kappa_median_ci=boot_ci(f["kappa"][allmove], rng) if allmove.sum() > 2 else None,
                           dM2_sig_positive=fl((f["dM2"] > 2 * f["dM2_se"]).mean()), dM2_sig_negative=fl((f["dM2"] < -2 * f["dM2_se"]).mean()),
                           cert_frac=fl(np.nanmean(cert)) if np.isfinite(cert).any() else None,
                           cone_frac=fl(np.nanmean(cone)) if np.isfinite(cone).any() else None)
                for abl in ABLATIONS:
                    if abl in res and abl != "full":
                        row[f"dkappa_{abl.replace(' ', '_')}"] = fl(np.median(f["kappa"][allmove] - res[abl]["kappa"][allmove])) if allmove.any() else None
                out.append(row)
                ci = row["dM2_median_ci"]
                print(f"  {st['name']:30s} {fam:20s} {lev:>5}: √M2 {row['sqrtM2_median']:.2f} E[dM2] {row['dM2_median']:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] "
                      f"se {row['dM2_se_median']:.3f} sig+ {row['dM2_sig_positive']:.2f} sig- {row['dM2_sig_negative']:.2f} κ-se {row['kappa_se_median']:.3f} | "
                      f"all move {row['all_move']:.2f} in G {row['in_G']:.2f} κ {fmt(row['kappa_median'], 3)} "
                      f"[p10 {fmt(row['kappa_p10'], 3)}] P {row['P_median']:.2f} | cert {fmt(row['cert_frac'])} cone {fmt(row['cone_frac'])}"
                      + (f" | Δκ att {row['dkappa_no_attraction']:+.3f} dog {row['dkappa_no_dog_push']:+.3f} ali {row['dkappa_no_alignment']:+.3f}"
                         if st["ablate"] and row.get("dkappa_no_attraction") is not None else ""), flush=True)
        print(f"  ({time.time() - tic:.0f} s)", flush=True)
    return out


def topological(rng):
    out = []
    seed = SEED * 7777
    for N in TOPO["sizes"]:
        S = TOPO["n_states"]
        st = dict(N=N, w_att=TOPO["w_att"], w_ali=TOPO["w_ali"], r_dog=TOPO["r_dog"])
        R0 = 0.9 * np.sqrt(N)
        pts = []
        for _ in range(S):  # rejection-sample a disc with no pair closer than 1 m
            x = np.empty((0, 2))
            while len(x) < N:
                c = R0 * np.sqrt(rng.random()) * np.array([np.cos(a := 2 * np.pi * rng.random()), np.sin(a)])
                if len(x) == 0 or np.linalg.norm(x - c, axis=1).min() >= 1.0:
                    x = np.vstack([x, c])
            pts.append(x)
        pos0 = np.array(pts)
        hd0 = unit(np.array([1.0, 0.0]) + 0.1 * rng.standard_normal((S, N, 2)))
        dog = np.tile([-D_STAND, 0.0], (S, 1))
        for g in TOPO["gaps"]:
            pos, hd = deform("split_perp", g, pos0, hd0, rng) if g > 0 else (pos0.copy(), hd0.copy())
            # do the halves see each other? share of perceived neighbours in the other half
            perp = pos[..., 1] - pos[..., 1].mean(1, keepdims=True)
            half = perp > 0
            d = np.linalg.norm(pos[:, None] - pos[:, :, None], axis=-1)
            d = np.where(np.eye(N, dtype=bool), np.inf, d)
            knn = np.argsort(d, axis=-1)[..., : P0.k_perceive]
            other = (np.take_along_axis(np.repeat(half[:, None, :], N, 1), knn, -1) != half[..., None]).mean(-1).mean(-1)
            seed += 1
            per_call = max(1, int(CHUNK_RUNS * (14 / N) ** 2) // K)
            parts = [analyse(pos[i:i + per_call], one_step(st, pos[i:i + per_call], hd[i:i + per_call], dog[i:i + per_call], seed + i, {}))
                     for i in range(0, S, per_call)]
            f = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
            allmove = f["f_m"] >= 0.999
            row = dict(N=N, gap=g, other_half_neighbours=fl(other.mean()), sqrtM2_median=fl(np.median(np.sqrt(f["M2"]))),
                       dM2_median=fl(np.median(f["dM2"])), dM2_median_ci=boot_ci(f["dM2"], rng), dM2_se_median=fl(np.median(f["dM2_se"])),
                       kappa_median=fl(np.median(f["kappa"][allmove])) if allmove.any() else None,
                       kappa_median_ci=boot_ci(f["kappa"][allmove], rng) if allmove.sum() > 2 else None,
                       kappa_se_median=fl(np.median(f["kappa_se"])), all_move=fl(allmove.mean()), P_median=fl(np.median(f["P"])))
            out.append(row)
            print(f"  topological N {N} gap {g:4.1f}: other-half neighbours {row['other_half_neighbours']:.2f} | √M2 {row['sqrtM2_median']:.2f} "
                  f"E[dM2] {row['dM2_median']:+.3f} {[round(v, 2) for v in row['dM2_median_ci']]} | κ {fmt(row['kappa_median'], 3)} "
                  f"{[round(v, 3) for v in row['kappa_median_ci']] if row['kappa_median_ci'] else '-'} | all move {row['all_move']:.2f} P {row['P_median']:.2f}", flush=True)
    return out


def figure(rows):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    order = []
    for fam, _, levels in FAMILIES:
        for lev in levels:
            order.append((fam, lev))
    for ax, key, title in ((axs[0], "in_G", "fraction of states in the lemma's region\n(every sheep moves and κ ≥ 0.05)"),
                           (axs[1], "dM2_positive", "fraction of states with E[ΔM2 | state] > 0")):
        for st, col in zip(SETTINGS, ("blue", "orange", "green", "purple")):
            ys = [next((r[key] for r in rows if r["setting"] == st["name"] and r["family"] == f and r["level"] == l), np.nan) for f, l in order]
            ax.plot(range(len(order)), ys, "o", color=COL[col], ms=6, label=st["name"])
        ax.set_xticks(range(len(order)), [f"{f} {l:g}" if f not in ("observed", "half reversed") else f for f, l in order], rotation=75, fontsize=7.5)
        ax.set(ylim=(-0.03, 1.03), title=title)
        ax.legend(frameon=False, fontsize=7.5)
    ax = axs[2]
    fams = [f for f, _, _ in FAMILIES]
    x = np.arange(len(fams))
    for j, (key, lab, col) in enumerate((("dkappa_no_attraction", "attraction", "green"), ("dkappa_no_dog_push", "dog push", "orange"),
                                          ("dkappa_no_alignment", "alignment", "blue"))):
        vals = [np.mean([r[key] for r in rows if r["family"] == f and r.get(key) is not None]) for f in fams]
        ax.bar(x + (j - 1) * 0.27, vals, width=0.26, color=COL[col], label=f"κ(full) − κ(no {lab})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, fams, rotation=60, fontsize=8)
    ax.set(ylabel="change in median κ (mean over levels and both r_dog 20 settings)", title="Which term supplies the inward relative heading")
    ax.legend(frameon=False, fontsize=7.5)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "shape_stress.png", dpi=130)


def main():
    tic = time.time()
    rows = run()
    topo = topological(np.random.default_rng(SEED + 1))
    out = dict(topological=topo, topological_setup=TOPO, setup=dict(K=K, reps=REPS, snapshots=SNAPS, standoff=D_STAND, kappa_star=KAPPA_STAR, families=[(f, k, list(l)) for f, k, l in FAMILIES],
                          settings=SETTINGS, rule="published (stop when no dog within r_dog); the dog stays at its observed position"),
               rows=rows, seconds=time.time() - tic)
    (OUT / "shape_stress.json").write_text(json.dumps(out, indent=1))
    figure(rows)
    print(f"done in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
