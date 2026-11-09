"""The collecting chain (docs/collecting_chain.md): from a disturbed flock, through the published dog's collecting,
back to driving and into the ball.

Paths, as in experiments/certify_d3.py: an independent baseline closed-loop snapshot (random step 100-299) is deformed
by one of 11 deformations chosen uniformly, then the closed loop (published sheep, published Strömbom dog) runs T
steps. tau is the first entry into the ball (sqrt M2 < RHO). A state is in the collecting phase while the farthest
sheep is at least l_sep from the centroid, and in the driving phase otherwise. The seed is not the one used to choose
D0 and LAM, so every number here is from fresh paths.

A. Episode structure [S]: collecting and driving episodes before tau, their lengths, how they end, target switches.
B. Collecting drift: shepherd.collecting.collecting_drift_bound gives, at every collecting state, an upper bound
   U(x) on the one-step expected change of Psi_k = |xi_k| + LAM (|D - x_k| - D0)_+ for the current target k,
   exactly (no Monte Carlo). Measured: the share of collecting states with U <= -C and U <= 0, by sub-phase;
   whether the target's own radial term is positive (then E[delta R] > 0 exactly, since R' >= |xi_k'| and
   E|xi_k'| >= R + E[e.Delta]); one state drawn uniformly per path for Clopper-Pearson bounds; target switches
   and the jump of Psi at each; a one-step Monte Carlo check of U at random collecting states; and the realized
   change of Psi_k against U along the paths.
C. The gap [S]: sheep frozen while the dog drives, until they move again or collecting starts.
D. Handover [S, with confidence]: one state per path, drawn uniformly from the states at which a collecting cycle
   has just ended. The 3-step drift D3 is certified there as in experiments/certify_d3.py. REPS replicates from each
   give p(x) = P(ball before the next collecting cycle), with a Clopper-Pearson lower bound, and E[driving cycle
   length], with an empirical-Bernstein upper bound.
E. Collecting starts [S, with confidence]: one state per path, drawn uniformly from the starts of collecting cycles;
   REPS_C replicates give E[collecting cycle length], with an empirical-Bernstein upper bound.
G. Target switching [P for the jump bound; S]: switches inside collecting episodes, across the flock or not, served or
   not; the deterministic jump bound J <= 4v + 2 LAM R'; R at successive across switches; the number of across switches
   per episode against the start distance of the rival on the opposite side, with Clopper-Pearson bounds.
F. The cycle bound: if every collecting cycle has expected length at most L_C, every driving cycle that starts at a
   handover has expected length at most L_D and ends in the ball with probability at least p, then from a handover
   E[tau] <= (L_C + L_D)/p (geometric number of cycles). Evaluated with the sample's worst bounds, and with its
   medians for comparison.
Cycles: the dog's rule has no hysteresis, so a handover with the farthest sheep just under l_sep can switch straight
back. The analysis therefore ends a collecting cycle only when R < l_sep - H_BAND; the rule itself is unchanged.
H_BAND = 3 m was chosen on an exploration seed among 0-5 m (0: some handovers reach the ball with probability 0.004;
4-5 m: almost every collecting cycle ends in the ball, so the cycles no longer separate the two modes).

Each of B, D and E has its own delta = DELTA: per-state bounds share half (union bound), the Clopper-Pearson
statements the other half.

    .venv/bin/python experiments/collecting_chain.py
Writes results/collecting_chain.json and results/figures/collecting_chain.png.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
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

import certify_d3 as cd  # noqa: E402
import shape_stress as ss  # noqa: E402
from shepherd.collecting import DRIVE_, collecting_drift_bound, dog_decision, psi  # noqa: E402
from shepherd.expected_heading import expected_headings  # noqa: E402
from shepherd.model import Params, simulate  # noqa: E402

OUT = ROOT / "results"
P0 = Params().resolved()
N, V_S, L_SEP = P0.n_sheep, P0.v_sheep * P0.dt, P0.l_sep
RHO = 4.46
SEED = 61
N_PATHS, T = 3000, 200
D0, LAM, C = 4.0, 1.0, 0.1
H_BAND = 3.0  # analysis hysteresis: a collecting cycle ends only when R < l_sep - H_BAND (the dog's rule is unchanged)
DELTA = 1e-3
REPS, HORIZON, CHUNK_STEPS = 1024, 150, 25  # handover states
REPS_C, HORIZON_C = 4096, 600  # collecting starts: their cycles are longer and heavy-tailed
N_CHECK, REPS_CHECK = 40, 8192
WORKERS = 10
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "purple": "#7b61c4", "gray": "#8a8a8a", "red": "#d63a3a"}


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def cp_upper(k, n, delta):
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def cp_lower(k, n, delta):
    return 0.0 if k <= 0 else float(beta.ppf(delta, k, n - k + 1))


def quant(x, qs=(10, 50, 90)):
    x = np.asarray(x, float)
    return [fl(np.percentile(x, q)) for q in qs] if len(x) else None


# ------------------------------------------------------------------ paths

def make_paths(seed, n):
    """Deformed snapshots (as in certify_d3.recovery_paths) run T steps; full trajectories and derived quantities."""
    rng = np.random.default_rng(seed)
    base = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=300), n_runs=n, seed=seed)
    t0 = rng.integers(100, 300, n)
    r = np.arange(n)
    pos, hd = base.pos_s[r, :, :, t0], base.vel_s[r, :, :, t0]
    dog, dog_v = base.pos_d[r, :, t0], base.vel_d[r, :, t0]
    fam = rng.integers(0, len(cd.DEFORMS), n)
    for f, (_, kind, levels) in enumerate(cd.DEFORMS):
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
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=T + 1), n_runs=n, seed=seed + 1,
                   init=dict(pos_s=pos, vel_s=hd, pos_d=dog, vel_d=dog_v))
    P = np.moveaxis(res.pos_s, -1, 1)  # (n, T+1, N, 2)
    D = np.moveaxis(res.pos_d, -1, 1)
    xi = P - P.mean(2, keepdims=True)
    rr = np.linalg.norm(xi, axis=-1)
    sq = np.sqrt((rr ** 2).mean(-1))
    inside = sq < RHO
    tau = np.where(inside.any(1), inside.argmax(1), T + 1)
    R = rr.max(-1)
    return dict(P=P, H=np.moveaxis(res.vel_s, -1, 1), D=D, DV=np.moveaxis(res.vel_d, -1, 1), fam=fam, rng=rng, R=R,
                target=rr.argmax(-1), rr=rr, sq=sq, tau=tau, phaseC=R >= L_SEP, cycleC=cycles(R), dec=dog_decision(P, D))


def cycles(R, h=None):
    """Analysis cycles with a hysteresis band: a collecting cycle starts when R >= l_sep and ends when R < l_sep - h."""
    h = H_BAND if h is None else h
    out = np.zeros(R.shape, bool)
    cur = R[:, 0] >= L_SEP
    out[:, 0] = cur
    for t in range(1, R.shape[1]):
        cur = np.where(cur, R[:, t] >= L_SEP - h, R[:, t] >= L_SEP)
        out[:, t] = cur
    return out


def runs(flags):
    """Maximal runs of a boolean sequence: (value, start, end)."""
    out, cur, s0 = [], None, 0
    for t, x in enumerate(flags):
        if x != cur:
            if cur is not None:
                out.append((bool(cur), s0, t))
            cur, s0 = x, t
    if cur is not None:
        out.append((bool(cur), s0, len(flags)))
    return out


# ------------------------------------------------------------------ A. episode structure

def episode_structure(pt):
    n = len(pt["tau"])
    drive_ball = drive_collect = 0
    d_len, c_len, c_switch, c_reached, eps_per_path = [], [], [], [], []
    for i in range(n):
        tau = pt["tau"][i]
        if tau == 0:
            continue
        end = min(tau, T + 1)
        eps = runs(pt["cycleC"][i, :end])
        eps_per_path.append(sum(e[0] for e in eps))
        for j, (isC, a, b) in enumerate(eps):
            if isC:
                c_len.append(b - a)
                c_switch.append(int((np.diff(pt["target"][i, a:b]) != 0).sum()))
                k = pt["target"][i, a:b]
                dd = np.linalg.norm(pt["D"][i, a:b] - pt["P"][i, np.arange(a, b), k], axis=-1)
                c_reached.append(bool((dd <= D0).any()))
            else:
                last = j == len(eps) - 1
                if last and tau <= T:
                    drive_ball += 1
                elif not last:
                    drive_collect += 1
                d_len.append(b - a)
    pre = np.concatenate([pt["dec"][i, :min(pt["tau"][i], T + 1)] for i in range(n) if pt["tau"][i] > 0])
    started = pt["tau"] > 0
    tau = pt["tau"][started]
    return dict(paths=n, started_outside=int(started.sum()), never_entered=int((pt["tau"] > T).sum()),
                tau=dict(mean=fl(tau[tau <= T].mean()), quantiles=quant(tau[tau <= T], (50, 90, 99)), max=int(tau.max())),
                paths_with_collecting=int(sum(e > 0 for e in eps_per_path)), collecting_episodes_per_path=quant([e for e in eps_per_path if e > 0], (50, 90, 100)),
                collecting=dict(n=len(c_len), length_mean=fl(np.mean(c_len)), length=quant(c_len, (50, 90, 100)), switches=quant(c_switch, (50, 90, 100)),
                                dog_within_D0=fl(np.mean(c_reached))),
                driving=dict(n=drive_ball + drive_collect, ends_in_ball=drive_ball, ends_in_collecting=drive_collect, p=fl(drive_ball / max(drive_ball + drive_collect, 1)),
                             length_mean=fl(np.mean(d_len)), length=quant(d_len, (50, 90, 100))),
                decisions_before_tau=dict(drive=fl((pre == 0).mean()), collect=fl((pre == 1).mean()), slow=fl((pre == 2).mean())))


# ------------------------------------------------------------------ B. collecting drift

def bounds_at(pt, ii, tt, chunk=400):
    out = {}
    for j in range(0, len(ii), chunk):
        a, b = ii[j:j + chunk], tt[j:j + chunk]
        part = collecting_drift_bound(pt["P"][a, b], pt["H"][a, b], pt["D"][a, b], pt["DV"][a, b], P0, D0, LAM)
        for k, v in part.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def _check_one(task):
    key, pos, hd, dog, dog_v, k = task
    rep = lambda a: np.repeat(a[None], REPS_CHECK, 0)
    seed = int(np.random.SeedSequence([SEED, 7, *key]).generate_state(1)[0])
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=2), n_runs=REPS_CHECK, seed=seed,
                   init=dict(pos_s=rep(pos), vel_s=rep(hd), pos_d=rep(dog), vel_d=rep(dog_v)))
    kk = np.full(REPS_CHECK, k)
    d = psi(res.pos_s[..., 1], res.pos_d[..., 1], kk, D0, LAM) - psi(pos[None], dog[None], kk[:1], D0, LAM)[0]
    return d.mean(), d.std() / np.sqrt(REPS_CHECK)


def collecting_drift(pt):
    n = len(pt["tau"])
    idx = np.array([(i, t) for i in range(n) for t in range(min(pt["tau"][i], T + 1)) if pt["phaseC"][i, t]])
    ii, tt = idx[:, 0], idx[:, 1]
    b = bounds_at(pt, ii, tt)
    U = b["U"]
    cls = np.where(b["slow"], "slow", np.where(b["awake"], "awake", "frozen"))
    by = {}
    for c in ("frozen", "awake", "slow"):
        m = cls == c
        by[c] = dict(n=int(m.sum()), le_0=fl((U[m] <= 0).mean()) if m.any() else None, le_minus_C=fl((U[m] <= -C).mean()) if m.any() else None,
                     median=fl(np.median(U[m])) if m.any() else None, R_drift_positive=fl((b["Ee"][m] > 0).mean()) if m.any() else None)
    bad = U > -C
    fail = dict(n=int(bad.sum()), by_class=dict(Counter(cls[bad])), d=quant(b["d"][bad]), Ee=quant(b["Ee"][bad]), U=quant(U[bad]))
    # one state per path, uniformly: i.i.d.
    rng = np.random.default_rng(SEED + 5)
    pick = []
    for i in np.unique(ii):
        js = np.flatnonzero(ii == i)
        pick.append(int(rng.choice(js)))
    pick = np.array(pick)
    m = len(pick)
    pac = {f"U>{-c:g}": dict(k=int((U[pick] > -c).sum()), n=m, eps=cp_upper(int((U[pick] > -c).sum()), m, DELTA / 4)) for c in (0.0, C)}
    Rpos = int((b["Ee"][pick] > 0).sum())
    pac["E[dR]>0"] = dict(k=Rpos, n=m, lower=cp_lower(Rpos, m, DELTA / 4))
    # target switches and jumps of Psi, and the realized change of Psi_k against U (consecutive collecting states)
    nxt = (tt + 1 <= T) & pt["phaseC"][ii, np.minimum(tt + 1, T)] & (tt + 1 < pt["tau"][ii])
    a, t1 = ii[nxt], tt[nxt] + 1
    k0 = b["target"][nxt]
    psi_k1 = psi(pt["P"][a, t1], pt["D"][a, t1], k0, D0, LAM)
    psi_k0 = psi(pt["P"][a, tt[nxt]], pt["D"][a, tt[nxt]], k0, D0, LAM)
    psi_new = psi(pt["P"][a, t1], pt["D"][a, t1], pt["target"][a, t1], D0, LAM)
    sw = pt["target"][a, t1] != k0
    J = psi_new - psi_k1
    gap = (psi_k1 - psi_k0) - U[nxt]
    switches = dict(steps=int(nxt.sum()), share=fl(sw.mean()), jump=quant(J[sw], (10, 50, 90, 100)), jump_mean=fl(J[sw].mean()),
                    jump_negative_share=fl((J[sw] < 0).mean()))
    realized = dict(mean_realized_minus_bound=fl(gap.mean()), se=fl(gap.std() / np.sqrt(len(gap))), n=int(len(gap)),
                    mean_bound=fl(U[nxt].mean()), mean_realized=fl((psi_k1 - psi_k0).mean()))
    # Monte Carlo check at random collecting states
    sel = rng.choice(len(ii), size=min(N_CHECK, len(ii)), replace=False)
    tasks = [((int(j),), pt["P"][ii[j], tt[j]], pt["H"][ii[j], tt[j]], pt["D"][ii[j], tt[j]], pt["DV"][ii[j], tt[j]], int(b["target"][j])) for j in sel]
    with Pool(WORKERS) as pool:
        mc = pool.map(_check_one, tasks)
    mcm = np.array([x[0] for x in mc])
    mcs = np.array([x[1] for x in mc])
    check = dict(n=len(sel), reps=REPS_CHECK, exceed_4se=int((mcm > U[sel] + 4 * mcs).sum()), slack=quant(U[sel] - mcm, (10, 50, 90)),
                 max_z=fl(((mcm - U[sel]) / mcs).max()))
    summary = dict(states=int(len(ii)), le_0=fl((U <= 0).mean()), le_minus_C=fl((U <= -C).mean()), by_class=by, failures=fail, pac=pac,
                   switches=switches, realized=realized, mc_check=check, R_drift_positive=fl((b["Ee"] > 0).mean()))
    return summary, dict(U=U, d=b["d"], cls=cls, Ee=b["Ee"])


# ------------------------------------------------------------------ C. the gap

def gap_events(pt):
    """Sheep frozen while the dog drives (not seeing it, not repelled), followed until they move or collecting starts."""
    P, D = pt["P"], pt["D"]
    n = len(P)
    rows = []
    for i in range(n):
        s = min(pt["tau"][i], T + 1)
        if s <= 1:
            continue
        dsh = np.linalg.norm(P[i, :s] - D[i, :s, None], axis=-1)
        dd = np.linalg.norm(P[i, :s, :, None] - P[i, :s, None], axis=-1)
        dd[:, np.arange(N), np.arange(N)] = np.inf
        frozen = (dsh > P0.r_dog) & (dd.min(-1) >= P0.r_rep)
        drive = pt["dec"][i, :s] == DRIVE_
        active = np.zeros(N, bool)
        for t in range(s):
            new = frozen[t] & drive[t] & ~active
            for k in np.flatnonzero(new):
                u, how = t, "ball first"
                while u < s:
                    if pt["phaseC"][i, u]:
                        how = "collecting"
                        break
                    if not frozen[u, k]:
                        how = "woken"
                        break
                    u += 1
                G = P[i, t].mean(0)
                g = -G / np.linalg.norm(G)
                xi = P[i, t, k] - G
                rows.append((how, u - t, float(np.linalg.norm(xi)), float(xi @ g), float(abs(xi @ np.array([-g[1], g[0]])))))
            active = frozen[t] & drive[t]
    how = np.array([r[0] for r in rows])
    L = np.array([r[1] for r in rows])
    lat = np.array([r[4] for r in rows])
    out = dict(events=len(rows), how=dict(Counter(how.tolist())))
    for lab, m in (("all", np.ones(len(rows), bool)), ("lateral >= 9.4 m", lat >= 9.4)):
        out[lab] = {h: dict(n=int((m & (how == h)).sum()), steps=quant(L[m & (how == h)], (50, 90, 100))) for h in ("woken", "collecting")}
    return out, L, how


# ------------------------------------------------------------------ D, E. episodes from sampled states

def episode_mc(task):
    """reps replicates from one state until its cycle ends (see cycles) or the flock enters the ball, at most
    horizon steps."""
    key, pos, hd, dog, dog_v, startC, reps, horizon = task
    rep = lambda a: np.repeat(a[None], reps, 0)
    cur = dict(pos_s=rep(pos), vel_s=rep(hd), pos_d=rep(dog), vel_d=rep(dog_v))
    length = np.full(reps, -1)
    ball = np.zeros(reps, bool)
    alive = np.arange(reps)
    done = chunk = 0
    while len(alive) and done < horizon:
        n = min(CHUNK_STEPS, horizon - done)
        seed = int(np.random.SeedSequence([SEED, 11, *key, chunk]).generate_state(1)[0])
        res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=n + 1), n_runs=len(alive), seed=seed,
                       init={k: v[alive] for k, v in cur.items()})
        Pp = np.moveaxis(res.pos_s, -1, 1)[:, 1:]
        r = np.linalg.norm(Pp - Pp.mean(2, keepdims=True), axis=-1)
        inball = np.sqrt((r ** 2).mean(-1)) < RHO
        Rr = r.max(-1)
        end = inball | ((Rr < L_SEP - H_BAND) if startC else (Rr >= L_SEP))
        hit = end.any(1)
        first = end.argmax(1)
        length[alive[hit]] = done + first[hit] + 1
        ball[alive[hit]] = inball[hit, first[hit]]
        keep = ~hit
        for k, src in (("pos_s", res.pos_s), ("vel_s", res.vel_s), ("pos_d", res.pos_d), ("vel_d", res.vel_d)):
            cur[k][alive[keep]] = src[keep][..., -1]
        alive = alive[keep]
        done += n
        chunk += 1
    censored = length < 0
    length[censored] = horizon
    return dict(key=list(key), length_mean=float(length.mean()), length_var=float(length.var(ddof=1)), p_ball=float(ball.mean()),
                k_ball=int(ball.sum()), censored=int(censored.sum()))


def sample_states(pt, kind, salt):
    """One state per path, uniformly among handovers (kind 'handover': the first state of a driving cycle after a
    collecting cycle) or collecting starts (kind 'collect_start'), before tau. Cycles use the band H_BAND."""
    rng = np.random.default_rng(SEED + salt)
    picks = []
    for i in range(len(pt["tau"])):
        s = min(pt["tau"][i], T + 1)
        ph = pt["cycleC"][i, :s]
        if kind == "handover":
            ts = np.flatnonzero(~ph[1:] & ph[:-1]) + 1
        else:
            ts = np.flatnonzero(ph & np.r_[True, ~ph[:-1]])
        if len(ts):
            picks.append((i, int(rng.choice(ts))))
    return np.array(picks)


def state_of(pt, i, t):
    return pt["P"][i, t], pt["H"][i, t], pt["D"][i, t], pt["DV"][i, t]


def handover(pt):
    pk = sample_states(pt, "handover", 1)
    n = len(pk)
    pos = np.stack([pt["P"][i, t] for i, t in pk])
    hd = np.stack([pt["H"][i, t] for i, t in pk])
    dog = np.stack([pt["D"][i, t] for i, t in pk])
    dog_v = np.stack([pt["DV"][i, t] for i, t in pk])
    d1 = cd.exact_d1(pos, hd, dog)
    delta3 = (DELTA / 2) / (n * len(cd.STAGES) * 4)
    with Pool(WORKERS) as pool:
        d3 = pool.map(cd.certify_one, [((40, j), pos[j], hd[j], dog[j], dog_v[j], d1[j], delta3, False) for j in range(n)], chunksize=1)
        mc = pool.map(episode_mc, [((41, j), pos[j], hd[j], dog[j], dog_v[j], False, REPS, HORIZON) for j in range(n)], chunksize=1)
    sees = (np.linalg.norm(pos - dog[:, None], axis=-1) <= P0.r_dog).all(1)
    dp = (DELTA / 2) / n
    p_lo = np.array([cp_lower(m["k_ball"], REPS, dp) for m in mc])
    L_hi = np.array([m["length_mean"] + cd.eb_slack(m["length_var"], REPS, HORIZON, dp) for m in mc])
    U3 = np.array([r["U3"] for r in d3])
    L3 = np.array([r["L3"] for r in d3])
    rows = [dict(path=int(i), t=int(t), sqrtM2=fl(pt["sq"][i, t]), R=fl(pt["R"][i, t]), sees_all=bool(sees[j]), D3=r["D3"], U3=r["U3"], L3=r["L3"],
                 p_ball=m["p_ball"], p_lower=fl(p_lo[j]), L_D=m["length_mean"], L_D_upper=fl(L_hi[j]), censored=m["censored"])
            for j, ((i, t), r, m) in enumerate(zip(pk, d3, mc))]
    kneg = int((U3 >= -cd.C1).sum())
    summary = dict(n=n, full_coverage=int(sees.sum()), D3_certified_below_C1=int((U3 < -cd.C1).sum()), D3_positive=int((L3 > 0).sum()),
                   pac_D3=dict(eps=cp_upper(kneg, n, DELTA / 4) + len(cd.STAGES) * delta3, k=kneg),
                   p_ball=quant([m["p_ball"] for m in mc], (0, 10, 50, 90)), p_lower_min=fl(p_lo.min()), p_lower=quant(p_lo, (10, 50)),
                   L_D=quant([m["length_mean"] for m in mc], (50, 90, 100)), L_D_upper_max=fl(L_hi.max()), censored=int(sum(m["censored"] for m in mc)))
    return summary, rows


def collect_starts(pt):
    pk = sample_states(pt, "collect_start", 2)
    n = len(pk)
    with Pool(WORKERS) as pool:
        mc = pool.map(episode_mc, [((42, j), *state_of(pt, i, t), True, REPS_C, HORIZON_C) for j, (i, t) in enumerate(pk)], chunksize=1)
    dl = (DELTA / 2) / n
    L_hi = np.array([m["length_mean"] + cd.eb_slack(m["length_var"], REPS_C, HORIZON_C, dl) for m in mc])
    b = collecting_drift_bound(np.stack([pt["P"][i, t] for i, t in pk]), np.stack([pt["H"][i, t] for i, t in pk]),
                               np.stack([pt["D"][i, t] for i, t in pk]), np.stack([pt["DV"][i, t] for i, t in pk]), P0, D0, LAM)
    psi0 = b["R"] + LAM * np.clip(b["d"] - D0, 0, None)
    rows = [dict(path=int(i), t=int(t), R=fl(pt["R"][i, t]), psi0=fl(psi0[j]), L_C=m["length_mean"], L_C_upper=fl(L_hi[j]), ends_in_ball=m["p_ball"],
                 censored=m["censored"]) for j, ((i, t), m) in enumerate(zip(pk, mc))]
    summary = dict(n=n, L_C=quant([m["length_mean"] for m in mc], (50, 90, 100)), L_C_upper_max=fl(L_hi.max()), censored=int(sum(m["censored"] for m in mc)),
                   corr_psi0_length=fl(np.corrcoef(psi0, [m["length_mean"] for m in mc])[0, 1]))
    return summary, rows


# ------------------------------------------------------------------ G. target switching

RIVAL_BINS = (0.0, 9.0, 10.0, 11.0, L_SEP, np.inf)


def switching(pt):
    """Target switches inside collecting episodes (maximal runs with R >= l_sep, before tau).

    Per switch: across the flock (old and new target on opposite sides, cos < 0) or not; served (the old target saw
    the dog) or premature; the jump of Psi and the deterministic bound J <= 4v + 2 LAM R' (the first term because
    |xi_k| >= |xi_k'| before the step and each moves by at most 2v; the second because the dog term changes by at
    most LAM |x_k' - x_k| <= LAM (|xi_k'| + |xi_k|)). Per episode: the number of across switches against the start
    distance of the farthest sheep on the opposite side of the target (the rival), with Clopper-Pearson bounds, and
    R at successive across switches. Per collecting state with an awake target: the rival's exact expected radial
    change to first order, -E[delta G].e_rival (the rival is frozen when it is out of the dog's range)."""
    P, D = pt["P"], pt["D"]
    G = P.mean(2)
    xi = P - G[:, :, None]
    rr = pt["rr"]
    sw_rows, ep_rows = [], []
    for i in range(len(P)):
        s = min(pt["tau"][i], T + 1)
        for isC, a, b in runs(pt["phaseC"][i, :s]):
            if not isC:
                continue
            ks = pt["target"][i, a:b]
            k0 = ks[0]
            c0 = xi[i, a] @ xi[i, a, k0] / (rr[i, a] * rr[i, a, k0])
            rival0 = float(np.where(c0 < 0, rr[i, a], -np.inf).max())
            across_R = []
            n_across = 0
            for q in np.flatnonzero(np.diff(ks) != 0):
                t = a + q
                k, k2 = ks[q], ks[q + 1]
                across = bool(xi[i, t + 1, k] @ xi[i, t + 1, k2] < 0)
                served = bool(np.linalg.norm(D[i, t] - P[i, t, k]) <= P0.r_dog)
                J = float(psi(P[i, t + 1][None], D[i, t + 1][None], np.array([k2]), D0, LAM)[0] - psi(P[i, t + 1][None], D[i, t + 1][None], np.array([k]), D0, LAM)[0])
                Jb = 4 * V_S + 2 * LAM * float(rr[i, t + 1].max())
                sw_rows.append((across, served, J, Jb))
                if across:
                    n_across += 1
                    across_R.append(float(rr[i, t + 1].max()))
            ep_rows.append(dict(rival0=rival0, n_across=n_across, n_switch=int((np.diff(ks) != 0).sum()), length=int(b - a),
                                R_at_across=across_R))
    sw = np.array(sw_rows, dtype=float)
    across, served, J, Jb = sw[:, 0] > 0, sw[:, 1] > 0, sw[:, 2], sw[:, 3]
    drops = np.array([r1 - r2 for e in ep_rows for r1, r2 in zip(e["R_at_across"], e["R_at_across"][1:])])
    riv = np.array([e["rival0"] for e in ep_rows])
    nac = np.array([e["n_across"] for e in ep_rows])
    L = np.array([e["length"] for e in ep_rows])
    bins = []
    for lo, hi in zip(RIVAL_BINS[:-1], RIVAL_BINS[1:]):
        m = (riv >= lo) & (riv < hi)
        k1, k2 = int((nac[m] >= 1).sum()), int((nac[m] >= 2).sum())
        bins.append(dict(rival=[fl(lo), fl(hi)], episodes=int(m.sum()), across_mean=fl(nac[m].mean()) if m.any() else None,
                         P_any=fl(k1 / max(m.sum(), 1)), P_any_upper=cp_upper(k1, int(m.sum()), DELTA / (2 * len(RIVAL_BINS))) if m.any() else None,
                         P_two=fl(k2 / max(m.sum(), 1)), length_mean=fl(L[m].mean()) if m.any() else None))
    by_n = {str(c): dict(episodes=int(m.sum()), length_mean=fl(L[m].mean())) for c, m in
            (("0", nac == 0), ("1", nac == 1), ("2", nac == 2), (">=3", nac >= 3)) if m.any()}
    # rival drift while the target is served
    idx = np.array([(i, t) for i in range(len(P)) for t in range(min(pt["tau"][i], T + 1)) if pt["phaseC"][i, t]])
    ii, tt = idx[:, 0], idx[:, 1]
    k = pt["target"][ii, tt]
    awake = np.linalg.norm(D[ii, tt] - P[ii, tt, k], axis=-1) <= P0.r_dog
    e = xi[ii, tt, k] / rr[ii, tt, k][:, None]
    cosj = np.einsum("snc,sc->sn", xi[ii, tt], e) / rr[ii, tt]
    j = np.where(cosj < 0, rr[ii, tt], -np.inf).argmax(1)
    rival_frozen = np.linalg.norm(D[ii, tt] - P[ii, tt, j], axis=-1) > P0.r_dog
    mu = np.concatenate([expected_headings(P[ii[a:a + 400], tt[a:a + 400]], pt["H"][ii[a:a + 400], tt[a:a + 400]], D[ii[a:a + 400], tt[a:a + 400]], P0)[0]
                         for a in range(0, len(ii), 400)])
    EdG = V_S * mu.mean(1)
    rival_drift = -(EdG * xi[ii, tt, j]).sum(-1) / rr[ii, tt, j]
    m = awake & rival_frozen
    return dict(switches=len(sw), across_share=fl(across.mean()), served_share=fl(served.mean()), served_share_across=fl(served[across].mean()),
                jump_bound_holds=bool((J <= Jb + 1e-9).all()), jump_over_bound=quant(J / Jb, (50, 90, 100)),
                R_change_between_across=dict(n=len(drops), drop_quantiles=quant(drops, (10, 25, 50, 75, 90)), share_not_lower=fl((drops <= 0).mean()) if len(drops) else None),
                episodes=len(ep_rows), across_per_episode=quant(nac, (50, 90, 99, 100)), by_rival=bins, length_by_across=by_n,
                rival_drift_served=dict(states=int(m.sum()), share_negative=fl((rival_drift[m] < 0).mean()), median=fl(np.median(rival_drift[m]))))


# ------------------------------------------------------------------ main

def main():
    tic = time.time()
    pt = make_paths(SEED, N_PATHS)
    A = episode_structure(pt)
    print(f"A. paths: {A['paths']}, started outside the ball {A['started_outside']}, never entered in {T} steps {A['never_entered']}; tau mean {A['tau']['mean']:.1f}, "
          f"quantiles 50/90/99 {A['tau']['quantiles']}, max {A['tau']['max']} ({time.time() - tic:.0f} s)", flush=True)
    print(f"   collecting: {A['paths_with_collecting']} paths, episodes per path {A['collecting_episodes_per_path']}; {A['collecting']}", flush=True)
    print(f"   driving: {A['driving']}; decisions before tau {A['decisions_before_tau']}", flush=True)

    B, Bdat = collecting_drift(pt)
    print(f"B. collecting states {B['states']}: U <= 0 {B['le_0']:.3f}, U <= -{C} {B['le_minus_C']:.3f}; E[dR] > 0 exactly at {B['R_drift_positive']:.3f}", flush=True)
    for c, v in B["by_class"].items():
        print(f"   {c}: {v}", flush=True)
    print(f"   failures: {B['failures']}", flush=True)
    print(f"   PAC (one state per path): {B['pac']}", flush=True)
    print(f"   switches: {B['switches']}", flush=True)
    print(f"   realized change of Psi_k minus bound: {B['realized']}", flush=True)
    print(f"   Monte Carlo check: {B['mc_check']} ({time.time() - tic:.0f} s)", flush=True)

    Cg, gapL, gapHow = gap_events(pt)
    print(f"C. gap: {Cg} ({time.time() - tic:.0f} s)", flush=True)

    Dh, Drows = handover(pt)
    print(f"D. handover: {Dh} ({time.time() - tic:.0f} s)", flush=True)
    Ec, Erows = collect_starts(pt)
    print(f"E. collecting starts: {Ec} ({time.time() - tic:.0f} s)", flush=True)

    worst = dict(L_C=Ec["L_C_upper_max"], L_D=Dh["L_D_upper_max"], p=Dh["p_lower_min"])
    worst["tau_bound"] = fl((worst["L_C"] + worst["L_D"]) / worst["p"]) if worst["p"] > 0 else None
    med = dict(L_C=Ec["L_C"][0], L_D=Dh["L_D"][0], p=Dh["p_ball"][2])
    med["tau_bound"] = fl((med["L_C"] + med["L_D"]) / med["p"])
    F = dict(worst=worst, median=med, observed_tau_after_handover=None)
    # observed: time from each sampled handover to tau, along its own path
    obs = [min(pt["tau"][r["path"]], T + 1) - r["t"] for r in Drows]
    F["observed_tau_after_handover"] = dict(mean=fl(np.mean(obs)), quantiles=quant(obs, (50, 90, 100)))
    print(f"F. cycle bound: worst {worst}; medians {med}; observed {F['observed_tau_after_handover']}", flush=True)
    Gs = switching(pt)
    print(f"G. switching: {Gs} ({time.time() - tic:.0f} s)", flush=True)

    out = dict(setup=dict(seed=SEED, paths=N_PATHS, steps=T, rho=RHO, d0=D0, lam=LAM, C=C, delta=DELTA, reps=REPS, horizon=HORIZON,
                          reps_collect=REPS_C, horizon_collect=HORIZON_C, band=H_BAND,
                          deformations=[d[0] for d in cd.DEFORMS], l_sep=L_SEP),
               A=A, B=B, C=Cg, D=dict(summary=Dh, rows=Drows), E=dict(summary=Ec, rows=Erows), F=F, G=Gs, seconds=time.time() - tic)
    (OUT / "collecting_chain.json").write_text(json.dumps(out, indent=1))
    figure(Bdat, gapL, gapHow, Drows, Erows)
    print(f"done in {time.time() - tic:.0f} s")


def figure(Bdat, gapL, gapHow, Drows, Erows):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    for c, col, z in (("frozen", "blue", 2), ("awake", "orange", 3), ("slow", "gray", 4)):
        m = Bdat["cls"] == c
        ax.scatter(Bdat["d"][m], Bdat["U"][m], s=6, color=COL[col], alpha=0.5, linewidths=0, zorder=z, label=f"target {c} ({m.sum()})")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.axhline(-C, color=COL["gray"], lw=0.8, ls="--")
    ax.axvline(P0.r_dog, color=COL["gray"], lw=0.6, ls=":")
    ax.set(xlabel="dog's distance to the farthest sheep (m)", ylabel="bound on E[ΔΨ] (m per step)",
           title=f"Collecting: exact bound on the potential's drift\n(U ≤ −{C:g} at {(Bdat['U'] <= -C).mean():.1%} of collecting states)")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    bins = np.arange(0, max(gapL.max(), 1) + 2) - 0.5
    for h, col in (("woken", "green"), ("collecting", "purple")):
        ax.hist(gapL[gapHow == h], bins=bins, color=COL[col], alpha=0.75, label=f"{h} ({(gapHow == h).sum()})")
    ax.set(xlabel="steps frozen", ylabel="frozen-sheep events while driving", title="The gap: how long a sheep stays frozen while the dog drives")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    x = [r["sqrtM2"] for r in Drows]
    cols = [COL["red"] if r["U3"] >= -cd.C1 else COL["blue"] for r in Drows]
    ax.scatter(x, [r["p_ball"] for r in Drows], s=10, c=cols, alpha=0.7, linewidths=0)
    ax.scatter([], [], s=10, color=COL["blue"], label="D₃ ≤ −0.1 certified")
    ax.scatter([], [], s=10, color=COL["red"], label="D₃ not certified")
    ax.set(xlabel="√M₂ at the handover (m)", ylabel="P(ball before the next collecting cycle)", ylim=(-0.03, 1.03),
           title="Handover states: the driving cycle that follows")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "collecting_chain.png", dpi=130)


if __name__ == "__main__":
    main()
