"""A high-confidence certificate for the 3-step drift condition D3 (docs/jadhav_dilorenzo_bridge.md §12.6).

D_m(x) = E[M2(t+m) | x_t = x] - M2(t) under the baseline closed loop: the published sheep rule with its stop rule,
and the published Strömbom dog (drive and collect, heading noise). §12.5 found D3 <= -0.12 m^2 on every tested
state outside the ball, from 128 replicates. This script turns that into a statistical certificate.

Estimator. D1 is exact (shepherd.expected_heading.expected_spread_change). For m = 2, 3,
    Y_m = D1(x) + M2(x_m) - M2(x_1)
is unbiased for D_m: the realized first step is replaced by its exact conditional mean. No sheep moves more than
v per step, so |sqrt M2(x_{k+1}) - sqrt M2(x_k)| <= v surely, and with s = sqrt M2(x),
    Y_m - D1  lies in  [max(s - (m - 2) v, 0)^2 - (s + v)^2,  (s + m v)^2 - (s + v)^2],
a range fixed before sampling.

Bound. Empirical Bernstein (Maurer & Pontil 2009, Thm 4): for n i.i.d. samples in an interval of width R, with
probability >= 1 - delta,
    E Y <= mean + sqrt(2 V ln(2/delta) / n) + 7 R ln(2/delta) / (3 (n - 1)),
V the unbiased sample variance; the lower bound is symmetric. Replicates come in cumulative stages of 2^13, 2^15,
2^17 and 2^19. At every stage each state gets four bounds (upper and lower, m = 2 and 3), each with its own share
of delta, so all of them hold together (union bound) and stopping early is allowed. A state stops once D3 is
decided against both thresholds 0 and -C1, or at the last stage.

State sets.
 1. Tested: the 255 driving states outside the ball with natural headings (memory_inequality.outside_ball_states)
    and the 30 adversarial heading fields of experiments/mi_counterexamples.py, the 6 MI+ counterexamples among
    them. The adversarial ones run every stage, to decide the sign of D2. One delta covers all 285.
 2. Visited (PAC): independent recovery paths. A baseline closed-loop snapshot (independent run, random time) is
    deformed, by a dilation or one of the experiments/shape_stress.py deformations chosen uniformly, and the
    closed loop runs T_PATH steps. Before the flock first enters the ball, each state is driving (the dog's next
    decision is drive), collecting or slow. One state of the wanted kind is drawn uniformly per path, so each
    family's states are i.i.d. Let B = {D3 > -c} and C = {certified at -c}. Then P(B) <= P(not C) + P(B and C),
    where P(B and C) is at most the state's share of delta, and Clopper-Pearson bounds P(not C). The states
    certified positive give a lower bound on P(D3 > 0) the same way. Families (VISITED): driving and collecting
    states from all deformations; then, on fresh paths, driving states after non-lateral and after lateral
    deformations (LATERAL), to test on new data what the first family showed. Each family has its own delta.

    .venv/bin/python experiments/certify_d3.py
Writes results/certify_d3.json and results/figures/certify_d3.png.
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
from scipy.stats import beta  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import memory_inequality as mi  # noqa: E402
import shape_stress as ss  # noqa: E402
from shepherd.expected_heading import expected_spread_change  # noqa: E402
from shepherd.model import COLLECT, Params, simulate  # noqa: E402

OUT = ROOT / "results"
P0 = Params().resolved()
N, V_S = P0.n_sheep, P0.v_sheep * P0.dt
RHO = 4.46
STAGES = (2 ** 13, 2 ** 15, 2 ** 17, 2 ** 19)
CHUNK = 2 ** 13
C1 = 0.1  # second threshold: D3 <= -C1
DELTA = 1e-3  # per claim: the tested certificate, and each PAC family (half to its states, half to Clopper-Pearson)
MARGIN = 1e-8  # floating-point margin on the exact D1
WORKERS = 10
SEED = 41
T_PATH = 150
DEFORMS = [("dilation", "dilate", (1.25, 2.0))] + [f for f in ss.FAMILIES if f[1] is not None]
LATERAL = ("split across", "stretch across", "outlier to the side")  # deformations that move sheep apart across the heading
# Visited families: name, the dog's next decision (0 drive, 1 collect), deformations drawn, seed, paths, states wanted.
# The first two share their paths. The last two are fresh samples that test, on new paths, what the first one showed:
# every driving failure came from a lateral deformation.
VISITED = [("driving", 0, "all", SEED, 3000, 800),
           ("collecting", 1, "all", SEED, 3000, 300),
           ("driving after a non-lateral deformation", 0, "non-lateral", SEED + 1000, 5000, 600),
           ("driving after a lateral deformation", 0, "lateral", SEED + 2000, 1500, 600)]
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def m2(pos):
    """M2 of positions (..., N, 2)."""
    xi = pos - pos.mean(-2, keepdims=True)
    return (xi ** 2).sum(-1).mean(-1)


def exact_d1(pos, hd, dog, chunk=200):
    return np.concatenate([expected_spread_change(pos[i:i + chunk], hd[i:i + chunk], dog[i:i + chunk], P0) for i in range(0, len(pos), chunk)])


# ------------------------------------------------------------------ one state

def ranges(s, v=V_S):
    """Sure ranges of M2(x_m) - M2(x_1), m = 2, 3, from sqrt M2(x_0) = s."""
    lo = lambda m: max(s - (m - 2) * v, 0.0) ** 2 - (s + v) ** 2
    hi = lambda m: (s + m * v) ** 2 - (s + v) ** 2
    return {m: (lo(m), hi(m)) for m in (2, 3)}


def blocks(pos, hd, dog, dog_v, n, seed):
    """n independent replicates of 3 closed-loop steps from one state: M2(x2) - M2(x1), M2(x3) - M2(x1), whether the dog
    collects, whether a sheep is out of the dog's range at x1 or x2."""
    rep = lambda a: np.repeat(a[None], n, 0)
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=4), n_runs=n, seed=seed,
                   init=dict(pos_s=rep(pos), vel_s=rep(hd), pos_d=rep(dog), vel_d=rep(dog_v)))
    M = m2(np.moveaxis(res.pos_s, -1, 1))  # (n, 4)
    out = (np.linalg.norm(res.pos_s[..., 1:3] - res.pos_d[:, None, :, 1:3], axis=2) > P0.r_dog).any(1).any(-1)
    return M[:, 2] - M[:, 1], M[:, 3] - M[:, 1], (res.mode[:, 1:] == COLLECT).any(1), out


def eb_slack(var, n, width, delta):
    L = np.log(2 / delta)
    return np.sqrt(2 * var * L / n) + 7 * width * L / (3 * (n - 1))


class Moments:
    def __init__(self):
        self.n, self.mean, self.ss = 0, 0.0, 0.0

    def add(self, x):  # Chan et al. merge
        nb, mb = len(x), x.mean()
        sb = ((x - mb) ** 2).sum()
        n = self.n + nb
        d = mb - self.mean
        self.mean += d * nb / n
        self.ss += sb + d * d * self.n * nb / n
        self.n = n

    @property
    def var(self):
        return self.ss / (self.n - 1)


def decided(U3, L3):
    return U3 < -C1 or L3 > 0 or (L3 > -C1 and U3 < 0)


def certify_one(task):
    key, pos, hd, dog, dog_v, d1, delta, run_all = task
    s = float(np.sqrt(m2(pos)))
    rg = ranges(s)
    mom = {2: Moments(), 3: Moments()}
    n_col = n_out = 0
    stages = []
    for n_target in STAGES:
        while mom[3].n < n_target:
            k = min(CHUNK, n_target - mom[3].n)
            seed = int(np.random.SeedSequence([SEED, *key, mom[3].n]).generate_state(1)[0])
            y2, y3, col, out = blocks(pos, hd, dog, dog_v, k, seed)
            for m, y in ((2, y2), (3, y3)):
                lo, hi = rg[m]
                assert y.min() >= lo - 1e-9 and y.max() <= hi + 1e-9, (key, m, y.min(), y.max(), lo, hi)
                mom[m].add(y)
            n_col += int(col.sum())
            n_out += int(out.sum())
        row = dict(n=mom[3].n)
        for m in (2, 3):
            est = d1 + mom[m].mean
            sl = eb_slack(mom[m].var, mom[m].n, rg[m][1] - rg[m][0], delta) + MARGIN
            row.update({f"D{m}": est, f"U{m}": est + sl, f"L{m}": est - sl, f"sd{m}": float(np.sqrt(mom[m].var))})
        stages.append(row)
        if not run_all and decided(row["U3"], row["L3"]):
            break
    last = stages[-1]
    return dict(key=list(key), sqrtM2=s, D1=float(d1), **{k: float(v) for k, v in last.items()}, stages=len(stages),
                collect_share=n_col / last["n"], out_of_range_share=n_out / last["n"], ranges={m: list(r) for m, r in rg.items()})


def run_tasks(tasks):
    order = sorted(range(len(tasks)), key=lambda i: not tasks[i][-1])  # run-all states first
    with Pool(WORKERS) as pool:
        res = pool.map(certify_one, [tasks[i] for i in order], chunksize=1)
    out = [None] * len(tasks)
    for i, r in zip(order, res):
        out[i] = r
    return out


# ------------------------------------------------------------------ state sets

def tested_states():
    st = mi.outside_ball_states()
    js = json.loads((OUT / "mi_counterexamples.json").read_text())["adversary"]
    tight, th = np.array(js["tight_states"]), np.array(js["fields_theta"])
    adv = dict(pos=st["pos"][tight], hd=np.stack([np.cos(th), np.sin(th)], -1), dog=st["dog"][tight], dog_v=st["dog_v"][tight])
    return st, adv, tight, np.array(js["counterexamples"])


def dog_decision(pos, dog):
    """The dog's next decision from states (..., N, 2), dog (..., 2): 0 drive, 1 collect, 2 slow."""
    xi = pos - pos.mean(-2, keepdims=True)
    slow = np.linalg.norm(pos - dog[..., None, :], axis=-1).min(-1) <= P0.r_rep
    collect = np.linalg.norm(xi, axis=-1).max(-1) >= P0.l_sep
    return np.where(slow, 2, np.where(collect, 1, 0))


def recovery_paths(seed, which, n_base, picks):
    """n_base independent deformed snapshots (deformations: all, lateral or non-lateral) run forward T_PATH steps. For each
    (name, code, quota) in picks, one state per path is drawn uniformly from the states before the flock first enters
    the ball at which the dog's next decision is code, from the first quota paths that have one."""
    allowed = [f for f, d in enumerate(DEFORMS) if which == "all" or (d[0] in LATERAL) == (which == "lateral")]
    rng = np.random.default_rng(seed)
    base = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=300), n_runs=n_base, seed=seed)
    t0 = rng.integers(100, 300, n_base)
    r = np.arange(n_base)
    pos, hd = base.pos_s[r, :, :, t0], base.vel_s[r, :, :, t0]
    dog, dog_v = base.pos_d[r, :, t0], base.vel_d[r, :, t0]
    fam = np.asarray(allowed)[rng.integers(0, len(allowed), n_base)]
    level = np.full(n_base, np.nan)
    for f, (name, kind, levels) in enumerate(DEFORMS):
        sel = np.flatnonzero(fam == f)
        if kind == "dilate":
            lv = rng.uniform(*levels, len(sel))
            c = pos[sel].mean(1, keepdims=True)
            pos[sel] = c + lv[:, None, None] * (pos[sel] - c)
        else:
            lv = np.asarray(levels, float)[rng.integers(0, len(levels), len(sel))]
            for L in np.unique(lv):
                s2 = sel[lv == L]
                pos[s2], hd[s2] = ss.deform(kind, L, pos[s2], hd[s2], rng)
        level[sel] = lv
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=T_PATH + 1), n_runs=n_base, seed=seed + 1,
                   init=dict(pos_s=pos, vel_s=hd, pos_d=dog, vel_d=dog_v))
    P = np.moveaxis(res.pos_s, -1, 1)  # (R, T, N, 2)
    D = np.moveaxis(res.pos_d, -1, 1)
    sq = np.sqrt(m2(P))
    inside = sq < RHO
    sigma = np.where(inside.any(1), inside.argmax(1), T_PATH + 1)
    before = np.arange(T_PATH + 1)[None] < sigma[:, None]
    dec = dog_decision(P, D)
    fams = {}
    for name, code, quota in picks:
        cand = before & (dec == code)
        has = np.flatnonzero(cand.any(1))
        i, t = np.array([(k, int(rng.choice(np.flatnonzero(cand[k])))) for k in has[:quota]]).T
        later = np.array([bool((before[k, s + 1:] & (dec[k, s + 1:] == 1)).any()) for k, s in zip(i, t)])
        fams[name] = dict(pos=res.pos_s[i, :, :, t], hd=res.vel_s[i, :, :, t], dog=res.pos_d[i, :, t], dog_v=res.vel_d[i, :, t],
                          path=i, t=t, family=fam[i], level=level[i], available=len(has), collects_later=later,
                          steps_to_ball=np.where(sigma[i] <= T_PATH, sigma[i] - t, -1))
    stats = dict(paths=n_base, deformations=which, started_outside=int((sigma > 0).sum()), never_entered=int((sigma == T_PATH + 1).sum()))
    return fams, stats


def describe(st):
    """Coverage, polarization, the farthest sheep's distance from the centroid and whether it is out of the dog's range."""
    pos, dog = st["pos"], st["dog"]
    out = np.linalg.norm(pos - dog[:, None], axis=-1) > P0.r_dog
    r = np.linalg.norm(pos - pos.mean(1, keepdims=True), axis=-1)
    far = r.argmax(1)
    return dict(sees_all=~out.any(1), n_out=out.sum(1), P=np.linalg.norm(st["hd"].mean(1), axis=-1), max_xi=r.max(1),
                far_out=out[np.arange(len(pos)), far])


# ------------------------------------------------------------------ summaries

def clopper_pearson_upper(k, n, delta):
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def clopper_pearson_lower(k, n, delta):
    return 0.0 if k <= 0 else float(beta.ppf(delta, k, n - k + 1))


def summarize(rows, delta_test):
    U3 = np.array([r["U3"] for r in rows])
    L3 = np.array([r["L3"] for r in rows])
    return dict(n=len(rows), certified_negative=int((U3 < 0).sum()), certified_below_C1=int((U3 < -C1).sum()), positive=int((L3 > 0).sum()),
                undecided_at_0=int(((U3 >= 0) & (L3 <= 0)).sum()), largest_upper_bound=fl(U3.max()), median_upper_bound=fl(np.median(U3)),
                largest_estimate=fl(max(r["D3"] for r in rows)), delta_per_bound=delta_test,
                stages_used=np.bincount([r["stages"] for r in rows], minlength=len(STAGES) + 1)[1:].tolist())


def pac(rows, delta_test):
    """Over the family's distribution, each at DELTA/6: upper bounds on P(D3 > -c) for c = 0 and C1, and a lower bound on
    P(D3 > 0) from the states certified positive."""
    U3 = np.array([r["U3"] for r in rows])
    L3 = np.array([r["L3"] for r in rows])
    n, slack = len(rows), len(STAGES) * delta_test
    out = {}
    for c in (0.0, C1):
        k = int((U3 >= -c).sum())
        out[f"c={c:g}"] = dict(not_certified=k, n=n, eps=clopper_pearson_upper(k, n, DELTA / 6) + slack)
    k = int((L3 > 0).sum())
    out["positive"] = dict(certified_positive=k, n=n, lower=max(clopper_pearson_lower(k, n, DELTA / 6) - slack, 0.0))
    return out


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    st, adv, tight, ces = tested_states()
    S, A = len(st["pos"]), len(adv["pos"])
    d1_nat, d1_adv = exact_d1(st["pos"], st["hd"], st["dog"]), exact_d1(adv["pos"], adv["hd"], adv["dog"])
    delta_T = DELTA / ((S + A) * len(STAGES) * 4)
    print(f"tested: {S} natural states, {A} adversarial fields; delta per bound {delta_T:.2e}, ln(2/delta) {np.log(2 / delta_T):.1f} "
          f"({time.time() - tic:.0f} s)", flush=True)
    tasks = [((0, i), st["pos"][i], st["hd"][i], st["dog"][i], st["dog_v"][i], d1_nat[i], delta_T, False) for i in range(S)]
    tasks += [((1, j), adv["pos"][j], adv["hd"][j], adv["dog"][j], adv["dog_v"][j], d1_adv[j], delta_T, True) for j in range(A)]
    res = run_tasks(tasks)
    nat, advr = res[:S], res[S:]
    for j, r in enumerate(advr):
        r.update(state=int(tight[j]), counterexample=bool(tight[j] in ces))
    for i, r in enumerate(nat):
        r.update(state=i)
    tested = dict(natural=dict(summary=summarize(nat, delta_T), D1_positive=int((d1_nat > 0).sum()), rows=nat),
                  adversarial=dict(summary=summarize(advr, delta_T), D1_positive=int((d1_adv > 0).sum()),
                                   D2_positive=int(sum(r["L2"] > 0 for r in advr)), D2_negative=int(sum(r["U2"] < 0 for r in advr)), rows=advr))
    allU = max(r["U3"] for r in res)
    print(f"   natural: {tested['natural']['summary']}; D1 > 0 exactly at {tested['natural']['D1_positive']}", flush=True)
    print(f"   adversarial: {tested['adversarial']['summary']}; D1 > 0 exactly at {tested['adversarial']['D1_positive']}, "
          f"D2 > 0 certified at {tested['adversarial']['D2_positive']}, D2 < 0 at {tested['adversarial']['D2_negative']}", flush=True)
    for r in advr:
        if r["counterexample"]:
            print(f"      counterexample {r['state']}: D1 {r['D1']:+.3f}, D2 {r['D2']:+.3f} [{r['L2']:+.3f}, {r['U2']:+.3f}], "
                  f"D3 {r['D3']:+.3f} [{r['L3']:+.3f}, {r['U3']:+.3f}], n {r['n']}", flush=True)
    print(f"   all 285 tested states: D3 <= {allU:+.4f} m2 simultaneously, with probability >= {1 - DELTA} ({time.time() - tic:.0f} s)", flush=True)

    fams, pstats = {}, {}
    for seed, which, n_base in sorted({(v[3], v[2], v[4]) for v in VISITED}):
        picks = [(v[0], v[1], v[5]) for v in VISITED if (v[3], v[2], v[4]) == (seed, which, n_base)]
        got, pstats[which] = recovery_paths(seed, which, n_base, picks)
        fams.update(got)
        print(f"recovery paths: {pstats[which]}; available states: " + ", ".join(f"{k} {got[k]['available']}" for k in got), flush=True)
    visited = {}
    for f, name in enumerate(v[0] for v in VISITED):
        fm = fams[name]
        n = len(fm["pos"])
        d1 = exact_d1(fm["pos"], fm["hd"], fm["dog"])
        delta_V = (DELTA / 2) / (n * len(STAGES) * 4)
        rows = run_tasks([((2 + f, i), fm["pos"][i], fm["hd"][i], fm["dog"][i], fm["dog_v"][i], d1[i], delta_V, False) for i in range(n)])
        desc = describe(fm)
        for i, r in enumerate(rows):
            r.update(path=int(fm["path"][i]), t=int(fm["t"][i]), deformation=DEFORMS[fm["family"][i]][0], level=fl(fm["level"][i]),
                     sees_all=bool(desc["sees_all"][i]), n_out=int(desc["n_out"][i]), P=fl(desc["P"][i]), max_xi=fl(desc["max_xi"][i]),
                     far_out=bool(desc["far_out"][i]), collects_later=bool(fm["collects_later"][i]),
                     steps_to_ball=int(fm["steps_to_ball"][i]) if fm["steps_to_ball"][i] >= 0 else None)
        bad = [r for r in rows if r["U3"] >= -C1]
        visited[name] = dict(summary=summarize(rows, delta_V), pac=pac(rows, delta_V), available=fm["available"],
                             by_deformation={d[0]: int(sum(r["deformation"] == d[0] for r in rows)) for d in DEFORMS},
                             not_certified_by_deformation={d[0]: int(sum(r["deformation"] == d[0] for r in bad)) for d in DEFORMS},
                             not_certified_collects_later=int(sum(r["collects_later"] for r in bad)),
                             certified_collects_later=int(sum(r["collects_later"] for r in rows if r["U3"] < -C1)),
                             full_coverage=int(desc["sees_all"].sum()), at_start=int(sum(r["t"] == 0 for r in rows)), rows=rows)
        print(f"   {name}: {visited[name]['summary']}", flush=True)
        print(f"      PAC: {visited[name]['pac']}; full coverage {visited[name]['full_coverage']} of {n}, at the start state {visited[name]['at_start']}", flush=True)
        print(f"      not certified at -{C1}, by deformation: { {k: v for k, v in visited[name]['not_certified_by_deformation'].items() if v} }; "
              f"the dog later collects on the path for {visited[name]['not_certified_collects_later']} of {len(bad)} "
              f"(certified states: {visited[name]['certified_collects_later']} of {n - len(bad)})", flush=True)
        for r in sorted(bad, key=lambda r: -r["U3"])[:12]:
            print(f"      not certified at -{C1}: {r['deformation']} {r['level']}, t {r['t']}, sqrt M2 {r['sqrtM2']:.2f}, max|xi| {r['max_xi']:.1f}, "
                  f"out of range {r['n_out']} (farthest {r['far_out']}), "
                  f"D1 {r['D1']:+.3f}, D3 {r['D3']:+.3f} [{r['L3']:+.3f}, {r['U3']:+.3f}], collect share {r['collect_share']:.2f}", flush=True)
        print(f"      ({time.time() - tic:.0f} s)", flush=True)

    out = dict(setup=dict(rho=RHO, stages=list(STAGES), C1=C1, delta=DELTA, margin=MARGIN, seed=SEED, path_steps=T_PATH, paths=pstats,
                          visited=[dict(name=v[0], dog_decision=["drive", "collect"][v[1]], deformations=v[2], seed=v[3], paths=v[4], wanted=v[5]) for v in VISITED],
                          lateral=list(LATERAL),
                          deformations=[dict(name=d[0], kind=d[1], levels=list(d[2])) for d in DEFORMS],
                          bound="Maurer-Pontil empirical Bernstein, one-sided, union over states, stages and the four bounds per stage",
                          estimator="Y_m = D1(x) + M2(x_m) - M2(x_1), D1 exact"),
               tested=dict(largest_upper_bound=fl(allU), **tested), visited=visited, seconds=time.time() - tic)
    (OUT / "certify_d3.json").write_text(json.dumps(out, indent=1))
    figure(tested, visited)
    print(f"done in {time.time() - tic:.0f} s")


def figure(tested, visited):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    for rows, col, lab in ((tested["natural"]["rows"], "blue", "natural headings (255)"),
                           ([r for r in tested["adversarial"]["rows"] if not r["counterexample"]], "orange", "adversarial fields (24)"),
                           ([r for r in tested["adversarial"]["rows"] if r["counterexample"]], "red", "MI⁺ counterexamples (6)")):
        x = [r["sqrtM2"] for r in rows]
        ax.errorbar(x, [r["D3"] for r in rows], yerr=[np.zeros(len(rows)), [r["U3"] - r["D3"] for r in rows]], fmt="o", ms=3.5, color=COL[col],
                    ecolor=COL[col], elinewidth=1, capsize=0, label=lab)
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="√M₂ (m)", ylabel="D₃ = E[M₂(t+3)] − M₂(t)  (m²)",
           title=f"Tested states: estimate and certified upper bound\n(all 285 at once, probability ≥ {1 - DELTA})")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    for r in tested["adversarial"]["rows"]:
        col, lw, z = (COL["red"], 1.6, 3) if r["counterexample"] else (COL["gray"], 0.8, 2)
        ax.plot([1, 2, 3], [r["D1"], r["D2"], r["D3"]], color=col, lw=lw, zorder=z)
        ax.vlines([2, 3], [r["L2"], r["L3"]], [r["U2"], r["U3"]], color=col, lw=lw + 1.5, zorder=z)
    ax.axhline(0, color=COL["gray"], lw=0.8)
    half = max(max(r["U2"] - r["D2"], r["U3"] - r["D3"]) for r in tested["adversarial"]["rows"])
    ax.set(xticks=[1, 2, 3], xlabel="horizon m (steps)", ylabel="D_m (m²)",
           title=f"The 30 adversarial fields by horizon (red: counterexamples)\nD₁ exact; D₂, D₃ certified to ±{half:.3f} m²")
    ax = axs[2]
    for name, col, lab, z in ((VISITED[2][0], "blue", "driving, after non-lateral deformations", 3), (VISITED[3][0], "orange", "driving, after lateral deformations", 2),
                              (VISITED[1][0], "purple", "collecting", 1)):
        v = visited[name]
        rows = v["rows"]
        ax.scatter([r["sqrtM2"] for r in rows], [r["U3"] for r in rows], s=9, color=COL[col], alpha=0.6, zorder=z, linewidths=0,
                   label=f"{lab} ({len(rows)}): P(D₃ > −{C1:g}) ≤ {v['pac'][f'c={C1:g}']['eps']:.3f}")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.axhline(-C1, color=COL["gray"], lw=0.8, ls="--")
    ax.set(xlabel="√M₂ (m)", ylabel="certified upper bound on D₃ (m²)", title="States visited on recovery paths, before the ball\n(fresh paths for the two driving families)")
    ax.legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "certify_d3.png", dpi=130)


if __name__ == "__main__":
    main()
