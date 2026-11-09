"""A mean-square bound on the centroid's cross-track error under the path-following cascade.

Baseline sheep (published stop rule, w_ali 1.3), baseline dog speed (v_dog 1.5), and the cascade
controller and straight-arc-straight paths of experiments/path_tracking.py.

Reduction (an identity, no closure). With x = (psi, e), psi the heading relative to the path
tangent and e the signed cross-track error (left > 0), the linearized geometry of one step is
    A(V, kappa) = [[1 - g k, -(g k / LOOK + V kappa^2)], [V (1 - g k), 1 - g k V / LOOK]]
    c(V, kappa) = ((g - 1) V kappa, V^2 kappa (g - 1/2))
(the centroid steps along the new heading, so a step on an arc is a chord). Define the disturbance
w_t = x_{t+1} - A(V_t, kappa_t) x_t - c(V_t, kappa_t). Then x_{t+1} = A_t x_t + c_t + w_t holds
exactly; the closures (heading law, polarization) are what keep w small. Split w_t into its
predictable part b_t = E[w_t | F_t] and a martingale difference z_t.

Bound (proved in docs/jadhav_dilorenzo_bridge.md §9). If one quadratic norm |.|_P is contracted by
lambda < 1 by every A(V, kappa) with V in [V_LO, 1] and kappa in [0, 1/3], then, up to the first time
V_t < V_LO (V_LO = 0.25: reached in one of the 79 published runs),
    RMS(e_t) <= c_e (a_t + s_t),   a_{t+1} = lambda a_t + B_t,   s_{t+1}^2 = lambda^2 s_t^2 + S_t^2,
with a_0 the RMS of |x_0|_P, B_t the RMS of |c_t + b_t|_P, S_t^2 the mean square of |z_t|_P and
c_e^2 = (P^-1)_ee.

The constants are estimated on fresh calibration runs (seed CAL_SEED, CAL_REPS per radius), b_t by
regressing w_t on quantities known at time t. The bound is then compared with the published runs
of path_tracking.py (seed 31), and with the exact stationary moments of the nominal recursion.
A narrower speed box (V_LO_ALT = 0.5) contracts faster but stops the bound at the first slower step;
it is reported as a sensitivity.

    .venv/bin/python experiments/tracking_bound.py
Writes results/tracking_bound.json and results/figures/tracking_bound.png.
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
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.linalg import solve_discrete_lyapunov  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import path_tracking as pt  # noqa: E402

OUT = ROOT / "results"
W_ALI, V_DOG = 1.3, 1.5  # baseline
K, LOOK = pt.K_THETA, pt.LOOK
CAL_SEED, CAL_REPS = 131, 64
V_LO, V_LO_ALT, KAP_MAX = 0.25, 0.5, 1 / 3.0
FEATURES = ("1", "kappa", "sat", "sat*kappa", "psi", "e", "psi^3", "e^3", "V-0.85", "|phi|", "1-coverage", "sat(t-1)", "sat(t-1)*w_psi(t-1)", "sat(t-1)*w_e(t-1)")
COL = pt.COL
RCOL = {3.0: "purple", 5.0: "orange", 8.0: "blue", 12.0: "green", 20.0: "gray"}


def fl(x):
    return pt.fl(x)


# ------------------------------------------------------------------ the reduced recursion

def A_mat(g, V, kap):
    return np.array([[1 - g * K, -(g * K / LOOK + V * kap ** 2)], [V * (1 - g * K), 1 - g * K * V / LOOK]])


def apply_A(g, V, kap, x):
    psi, e = x[..., 0], x[..., 1]
    return np.stack([(1 - g * K) * psi - (g * K / LOOK + V * kap ** 2) * e, V * (1 - g * K) * psi + (1 - g * K * V / LOOK) * e], -1)


def c_vec(g, V, kap):
    return np.stack([(g - 1) * V * kap, V ** 2 * kap * (g - 0.5)], -1)


def pnorm2(P, v):
    return np.einsum("...i,ij,...j->...", v, P, v)


def contraction(P, g, v_lo=V_LO):
    """Largest |A|_P over the corners of V in [v_lo, 1], kappa^2 in [0, KAP_MAX^2]; A is multi-affine, so the corners suffice."""
    w, U = np.linalg.eigh(P)
    Ph = U @ np.diag(np.sqrt(w)) @ U.T
    Pi = np.linalg.inv(Ph)
    return max(np.linalg.norm(Ph @ A_mat(g, V, kap) @ Pi, 2) for V, kap in itertools.product((v_lo, 1.0), (0.0, KAP_MAX)))


def P_of(p):
    """Positive definite P with P_ee = 1 (the bound is invariant to scaling P)."""
    L = np.array([[np.exp(p[0]), 0.0], [p[1], 1.0]])
    P = L @ L.T
    return P / P[1, 1]


# ------------------------------------------------------------------ data

def extract(rows, rec, tab, reps):
    """Per radius: x = (psi, e), V, kappa, saturation, commanded and realized turn, bearing, coverage, arc entry and exit."""
    a = tab[W_ALI]
    rmax = np.maximum.accumulate(a["rate"][a["phi"] <= pt.PHI_LIM + 1e-9]).max()
    out = {}
    for i, r in enumerate(rows):
        if r["w_ali"] != W_ALI or r["v_dog"] != V_DOG:
            continue
        sl = np.arange(i * reps, (i + 1) * reps)
        sl = sl[rec["spread"][sl, 0] <= pt.SPLIT_M]  # as in path_tracking: runs that start as one flock
        th, thp, e, V, seg = (rec[k][sl] for k in ("theta", "theta_path", "e", "V", "seg"))
        kap = np.where(seg == 1, 1 / r["R_path"], 0.0)
        om = V * kap + K * pt.wrap(thp - np.arctan(e / LOOK) - th)  # the controller's command
        sat = (np.abs(om) > rmax) | (rec["dog_speed"][sl] >= 0.999 * V_DOG)
        out[r["R_path"]] = dict(x=np.stack([pt.wrap(th - thp), e], -1), V=V, kap=kap, seg=seg, sat=sat.astype(float),
                                om=np.sign(om) * np.minimum(np.abs(om), rmax), turn=pt.wrap(th[:, 1:] - th[:, :-1]),
                                phi=np.abs(rec["phi"][sl]), cov=rec["coverage"][sl], entry=np.argmax(seg == 1, 1),
                                exit=np.argmax(seg == 2, 1), n=len(sl))
    return out


def fit_gain(D):
    """Realized turn against commanded turn on unsaturated steps with every sheep moving: turn = g om + beta + noise."""
    X, Y, per = [], [], {}
    for R, d in D.items():
        t = np.arange(d["om"].shape[1] - 1)[None, :]
        m = (d["sat"][:, :-1] == 0) & (d["cov"][:, :-1] >= 0.999) & (t >= 5)
        X.append(d["om"][:, :-1][m])
        Y.append(d["turn"][m])
        (g, b), *_ = np.linalg.lstsq(np.c_[X[-1], np.ones_like(X[-1])], Y[-1], rcond=None)
        per[R] = dict(g=float(g), beta=float(b))
    X, Y = np.concatenate(X), np.concatenate(Y)
    (g, b), *_ = np.linalg.lstsq(np.c_[X, np.ones_like(X)], Y, rcond=None)
    res = Y - g * X - b
    return dict(g=float(g), beta=float(b), sd=float(res.std()), per_radius=per, n=int(len(X)))


def disturbance(d, g):
    """w_t = x_{t+1} - A_t x_t - c_t (exact), and the features known at time t."""
    x, V, kap = d["x"], d["V"], d["kap"]
    w = x[:, 1:] - apply_A(g, V[:, :-1], kap[:, :-1], x[:, :-1]) - c_vec(g, V[:, :-1], kap[:, :-1])
    w[..., 0] = pt.wrap(w[..., 0])
    psi, e = x[:, :-1, 0], x[:, :-1, 1]
    s = d["sat"][:, :-1]
    prev = np.concatenate([np.full_like(w[:, :1], np.nan), w[:, :-1]], 1)
    s_prev = np.concatenate([np.full_like(s[:, :1], np.nan), s[:, :-1]], 1)
    F = np.stack([np.ones_like(psi), kap[:, :-1], s, s * kap[:, :-1], psi, e, psi ** 3, e ** 3, V[:, :-1] - 0.85, d["phi"][:, :-1],
                  1 - d["cov"][:, :-1], s_prev, s_prev * prev[..., 0], s_prev * prev[..., 1]], -1)  # a saturation deficit persists; noise does not
    return w, F


def straight_parts(D, P):
    """RMS of the predictable and unpredictable disturbance in the P-norm, on the first straight after warm-up."""
    Bs, Ss = [], []
    for d in D.values():
        t = np.arange(d["w"].shape[1])[None, :]
        m = (d["seg"][:, :-1] == 0) & (t >= 15)
        Bs.append(pnorm2(P, d["b"])[m])
        Ss.append(pnorm2(P, d["z"])[m])
    return np.sqrt(np.mean(np.concatenate(Bs))), np.sqrt(np.mean(np.concatenate(Ss)))


def choose_norm(cal, g, v_lo):
    """The P minimizing the stationary bound on the straight, subject to contraction over the whole box."""
    def objective(p):
        P = P_of(p)
        lam = contraction(P, g, v_lo)
        if lam >= 1:
            return 1e3 * (1 + lam)
        B, S = straight_parts(cal, P)
        return np.sqrt(np.linalg.inv(P)[1, 1]) * (B / (1 - lam) + S / np.sqrt(1 - lam ** 2))

    A0 = A_mat(g, 0.85, 0.0)
    P0 = solve_discrete_lyapunov(A0.T, np.eye(2))
    L0 = np.linalg.cholesky(P0 / P0[1, 1])
    p0 = np.array([np.log(L0[0, 0]), L0[1, 0]])
    best = min((minimize(objective, p0 + dp, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=5000))
                for dp in ([0, 0], [1, 0], [-1, 0], [0.5, 0.5])), key=lambda r: r.fun)
    P = P_of(best.x)
    return P, float(contraction(P, g, v_lo)), float(np.sqrt(np.linalg.inv(P)[1, 1]))


def martingale_check(val):
    """Autocorrelation of z, and its correlation with the time-t features, on the published runs (out of sample)."""
    zs = [d["z"][:, 5:] for d in val.values()]
    acf = {}
    for j, name in enumerate(("psi", "e")):
        num = [np.nanmean(np.concatenate([(z[:, lag:, j] * z[:, :z.shape[1] - lag, j]).ravel() for z in zs])) for lag in range(1, 9)]
        var = np.nanmean(np.concatenate([(z[..., j] ** 2).ravel() for z in zs]))
        acf[name] = [float(v / var) for v in num]
    n_lag = sum(z.shape[0] * z.shape[1] for z in zs)
    Xv = np.concatenate([d["F"][:, 1:].reshape(-1, len(FEATURES)) for d in val.values()])
    Zv = np.concatenate([d["z"][:, 1:].reshape(-1, 2) for d in val.values()])
    okv = np.isfinite(Xv).all(1) & np.isfinite(Zv).all(1)
    feat_corr = {f: [fl(np.corrcoef(Xv[okv, i], Zv[okv, j])[0, 1]) for j in range(2)] for i, f in enumerate(FEATURES) if Xv[okv, i].std() > 0}
    return acf, n_lag, feat_corr


def bound_curves(cal, val, g, P, lam, ce, feas, v_lo, verbose=False):
    """The bound c_e (a_t + s_t) per radius at fixed times t from the start (x_0 = 0 by construction), against the published runs."""
    curves, summary = {}, []
    for R in pt.R_PATH:
        dc, dv = cal[R], val[R]
        B_t = np.sqrt(np.nanmean(pnorm2(P, c_vec(g, dc["V"][:, :-1], dc["kap"][:, :-1]) + dc["b"]), 0))
        S_t = np.sqrt(np.nanmean(pnorm2(P, dc["z"]), 0))
        X0 = float(np.sqrt(np.mean(pnorm2(P, dv["x"][:, 0]))))  # the flock starts on the path, heading along it
        a_t, s_t = np.zeros(len(B_t) + 1), np.zeros(len(B_t) + 1)
        a_t[0] = X0
        for t in range(len(B_t)):
            a_t[t + 1] = lam * a_t[t] + B_t[t]
            s_t[t + 1] = np.sqrt(lam ** 2 * s_t[t] ** 2 + S_t[t] ** 2)
        bound = ce * (a_t + s_t)
        e_v = dv["x"][..., 1]
        meas = np.sqrt(np.mean(e_v ** 2, 0))
        ts = np.arange(len(meas))
        entry, exit_ = int(np.median(dv["entry"])), int(np.median(dv["exit"]))
        on_arc = (dv["seg"] == 1)
        arc_win = (ts >= entry) & (ts < exit_ + 10)
        late = ts >= exit_ + 20
        pre = (ts >= 15) & (ts < entry)
        curves[R] = dict(t=ts.tolist(), measured=[fl(v) for v in meas], bound=[fl(v) for v in bound], transient_and_bias=[fl(ce * v) for v in a_t],
                         noise=[fl(ce * v) for v in s_t], B=[fl(v) for v in B_t], S=[fl(v) for v in S_t], entry=entry, exit=exit_)
        summary.append(dict(R_path=R, feasible=bool(R >= feas["R_min"]), runs_validation=int(dv["n"]), runs_calibration=int(dc["n"]),
                            entry_median=entry, arc_steps_median=exit_ - entry, X0=X0, runs_leaving_box=int((np.nanmin(dv["V"][:, 1:], 1) < v_lo).sum()),
                            sat_frac_arc_cal=fl(dc["sat"][dc["seg"] == 1].mean()),
                            measured_pre=fl(np.sqrt(np.mean(e_v[:, pre] ** 2))), bound_pre=fl(np.max(bound[pre])),
                            measured_arc_rms=fl(np.sqrt(np.mean(e_v[on_arc] ** 2))), measured_arc_peak=fl(np.max(meas[arc_win])),
                            bound_arc_peak=fl(np.max(bound[arc_win])),
                            measured_late=fl(np.sqrt(np.mean(e_v[:, late] ** 2))), bound_late=fl(np.max(bound[late])),
                            bound_over_measured_min=fl(np.min(bound[1:] / meas[1:])), bound_over_measured_median=fl(np.median(bound[1:] / meas[1:])),
                            B_arc=fl(np.sqrt(np.mean(pnorm2(P, c_vec(g, dc["V"][:, :-1], dc["kap"][:, :-1]) + dc["b"])[dc["seg"][:, :-1] == 1]))),
                            B_straight=fl(np.sqrt(np.mean(B_t[pre[:-1]] ** 2))), S_arc=fl(np.sqrt(np.mean(pnorm2(P, dc["z"])[dc["seg"][:, :-1] == 1])))))
        s = summary[-1]
        if verbose:
            print(f"R {R:4.0f} ({'feasible' if s['feasible'] else 'R < R_min'}, arc steps {entry}-{exit_}, dog saturated on {s['sat_frac_arc_cal']:.2f} of arc steps): "
                  f"RMS e before the arc {s['measured_pre']:.2f} (bound {s['bound_pre']:.2f}); peak on the arc {s['measured_arc_peak']:.2f} (bound {s['bound_arc_peak']:.2f}); "
                  f"after {s['measured_late']:.2f} (bound {s['bound_late']:.2f}); bound/measured min {s['bound_over_measured_min']:.2f} median {s['bound_over_measured_median']:.2f}; "
                  f"B straight {s['B_straight']:.3f} arc {s['B_arc']:.3f}, S arc {s['S_arc']:.3f}; x0 {X0:.1e}; runs leaving the V box {s['runs_leaving_box']}")
    return curves, summary


# ------------------------------------------------------------------ main analysis

def main():
    tic = time.time()
    tab = pt.laws()
    rows = [dict(w_ali=W_ALI, v_dog=V_DOG, R_path=R) for R in pt.R_PATH]
    feas = pt.feasibility(tab[W_ALI], pt.D_STAND, V_DOG)
    cal = extract(rows, pt.run(tab, rows, CAL_REPS, CAL_SEED)[1], tab, CAL_REPS)
    val = extract(*pt.run(tab), tab, pt.REPS)  # the published runs: the full grid at seed 31, baseline rows kept

    # 1. closed-loop heading law, then the exact disturbance and its predictable part
    gain = fit_gain(cal)
    g = gain["g"]
    for D in (cal, val):
        for d in D.values():
            d["w"], d["F"] = disturbance(d, g)
    Xc = np.concatenate([d["F"][:, 1:].reshape(-1, len(FEATURES)) for d in cal.values()])
    Wc = np.concatenate([d["w"][:, 1:].reshape(-1, 2) for d in cal.values()])
    ok = np.isfinite(Xc).all(1) & np.isfinite(Wc).all(1)
    coef, *_ = np.linalg.lstsq(Xc[ok], Wc[ok], rcond=None)
    for D in (cal, val):
        for d in D.values():
            d["b"] = np.nan_to_num(d["F"]) @ coef
            d["b"][:, 0] = 0.0  # no lagged disturbance at the first step
            d["z"] = d["w"] - d["b"]
    zc = Wc[ok] - Xc[ok] @ coef
    r2 = 1 - zc.var(0) / Wc[ok].var(0)

    # 2. the norm, and 3. the martingale diagnostics
    P, lam, ce = choose_norm(cal, g, V_LO)
    print(f"heading law: g {g:.3f}, beta {gain['beta']:+.4f}, residual sd {gain['sd']:.4f} rad (n {gain['n']}); "
          f"per radius {[round(v['g'], 3) for v in gain['per_radius'].values()]}")
    B_str, S_str = straight_parts(cal, P)
    floor = dict(B=float(B_str), S=float(S_str), bias_part=float(ce * B_str / (1 - lam)), noise_part=float(ce * S_str / np.sqrt(1 - lam ** 2)))
    floor["bound"] = floor["bias_part"] + floor["noise_part"]
    print(f"disturbance: regression R^2 psi {r2[0]:.3f}, e {r2[1]:.3f}; norm P {P.round(3).tolist()}, lambda {lam:.4f}, c_e {ce:.3f}; "
          f"stationary bound on the straight {floor['bound']:.2f} m (bias {floor['bias_part']:.2f} + noise {floor['noise_part']:.2f})")
    acf, n_lag, feat_corr = martingale_check(val)
    print(f"published runs: lag-1..3 autocorrelation of z: psi {np.round(acf['psi'][:3], 3)}, e {np.round(acf['e'][:3], 3)} (2/sqrt(n) = {2 / np.sqrt(n_lag):.3f}); "
          f"largest |corr(z, feature)| {max(abs(v) for vs in feat_corr.values() for v in vs if v is not None):.3f}")

    # 4. the bound along the path against the published runs, and with a narrower speed box
    curves, summary = bound_curves(cal, val, g, P, lam, ce, feas, V_LO, verbose=True)
    P2, lam2, ce2 = choose_norm(cal, g, V_LO_ALT)
    _, summ2 = bound_curves(cal, val, g, P2, lam2, ce2, feas, V_LO_ALT)
    narrow = dict(V_LO=V_LO_ALT, P=P2.tolist(), lam=float(lam2), c_e=ce2, summary=summ2)
    print(f"sensitivity, V box [{V_LO_ALT}, 1]: lambda {lam2:.4f}; arc peak bounds {[round(x['bound_arc_peak'], 2) for x in summ2]}, "
          f"runs reaching V < {V_LO_ALT}: {[x['runs_leaving_box'] for x in summ2]}")

    # 5. the nominal recursion with frozen parameters: exact stationary moments
    def stationary(kap, mask_fn):
        Vm = np.nanmean(np.concatenate([d["V"][mask_fn(d)] for d in cal.values() if kap == 0 or abs(d["kap"].max() - kap) < 1e-12]))
        sel = [d for d in cal.values() if kap == 0 or abs(d["kap"].max() - kap) < 1e-12]
        bm = np.nanmean(np.concatenate([d["b"][mask_fn(d)[:, :-1]] for d in sel]), 0)
        zz = np.concatenate([d["z"][mask_fn(d)[:, :-1]] for d in sel])
        Q = np.cov(zz[np.isfinite(zz).all(1)].T)
        A = A_mat(g, Vm, kap)
        mean = np.linalg.solve(np.eye(2) - A, c_vec(g, Vm, kap) + bm)
        Sig = solve_discrete_lyapunov(A, Q)
        mean0 = np.linalg.solve(np.eye(2) - A, c_vec(g, Vm, kap))
        return dict(kappa=kap, V=float(Vm), mean_e=float(mean[1]), mean_e_geometry_only=float(mean0[1]), sd_e=float(np.sqrt(Sig[1, 1])),
                    sd_psi=float(np.sqrt(Sig[0, 0])), rms_e=float(np.sqrt(mean[1] ** 2 + Sig[1, 1])),
                    spectral_radius=float(max(abs(np.linalg.eigvals(A)))), noise_sd=[float(v) for v in np.sqrt(np.diag(Q))], noise_corr=float(Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1])))

    nominal = {"straight": stationary(0.0, lambda d: (d["seg"] == 0) & (np.arange(d["seg"].shape[1])[None, :] >= 15))}
    for R in (8.0, 12.0, 20.0):
        nominal[f"arc R {R:g}"] = stationary(1 / R, lambda d: (d["seg"] == 1) & (np.arange(d["seg"].shape[1])[None, :] >= d["entry"][:, None] + 6))
    meas_stat = {}
    for name, D in (("published", val), ("calibration", cal)):
        straight = [(d["seg"] == 0) & (np.arange(d["seg"].shape[1])[None, :] >= 15) for d in D.values()]
        ms = {"straight": dict(rms_e=fl(np.sqrt(np.mean(np.concatenate([d["x"][..., 1][m] for d, m in zip(D.values(), straight)]) ** 2))),
                               sd_psi=fl(np.std(np.concatenate([d["x"][..., 0][m] for d, m in zip(D.values(), straight)]))))}
        for R in (8.0, 12.0, 20.0):
            d = D[R]
            m = (d["seg"] == 1) & (np.arange(d["seg"].shape[1])[None, :] >= d["entry"][:, None] + 6)
            ms[f"arc R {R:g}"] = dict(mean_e=fl(np.mean(d["x"][..., 1][m])), rms_e=fl(np.sqrt(np.mean(d["x"][..., 1][m] ** 2))), steps=int(m.sum()))
        meas_stat[name] = ms
    for k, v in nominal.items():
        print(f"nominal {k}: stationary RMS e {v['rms_e']:.3f} (mean {v['mean_e']:+.3f}, geometry only {v['mean_e_geometry_only']:+.3f}; sd {v['sd_e']:.3f}), sd psi {v['sd_psi']:.3f}, "
              f"spectral radius {v['spectral_radius']:.3f} | measured, published {meas_stat['published'][k]}, calibration {meas_stat['calibration'][k]}")

    out = dict(setup=dict(w_ali=W_ALI, v_dog=V_DOG, k_theta=K, look=LOOK, standoff=pt.D_STAND, calibration=dict(seed=CAL_SEED, reps=CAL_REPS),
                          validation="published runs of path_tracking.py (seed 31, 16 per radius)", V_box=[V_LO, 1.0], kappa_box=[0.0, KAP_MAX],
                          sheep_rule="published (stop when no dog within r_dog)"),
               feasibility=feas, heading_law=gain, features=list(FEATURES), regression_coef=coef.T.tolist(), regression_R2=[float(v) for v in r2],
               norm=dict(P=P.tolist(), lam=float(lam), c_e=ce), straight_floor=floor, narrow_speed_box=narrow, martingale_check=dict(acf_lags_1_8=acf, two_over_sqrt_n=float(2 / np.sqrt(n_lag)), corr_with_features=feat_corr),
               nominal_stationary=nominal, measured_stationary=meas_stat, summary=summary, curves=curves, seconds=time.time() - tic)
    (OUT / "tracking_bound.json").write_text(json.dumps(out, indent=1))
    figure(curves, acf, n_lag, feas)
    print(f"done in {time.time() - tic:.0f} s")


def figure(curves, acf, n_lag, feas):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1.0, 0.8]))
    for ax, radii, title in ((axs[0], (8.0, 12.0, 20.0), f"Radii above R_min ≈ {feas['R_min']:.1f} m"), (axs[1], (3.0, 5.0), "Radii below R_min (dog saturated on the arc)")):
        for R in radii:
            c = curves[R]
            col = COL[RCOL[R]]
            ax.plot(c["t"], c["measured"], color=col, lw=2, label=f"R {R:g} m: measured")
            ax.plot(c["t"], c["bound"], color=col, lw=1.3, ls="--", label=f"R {R:g} m: bound")
            ax.axvspan(c["entry"], c["exit"], color=col, alpha=0.08, lw=0)
        ax.set(xlabel="step (shaded: on the arc, median run)", ylabel="RMS cross-track error (m)", title=title, ylim=(0, None), xlim=(0, None))
        ax.legend(frameon=False, fontsize=8, ncol=1)
    ax = axs[2]
    lags = np.arange(1, 9)
    ax.bar(lags - 0.18, acf["psi"], width=0.34, color=COL["blue"], label="heading component")
    ax.bar(lags + 0.18, acf["e"], width=0.34, color=COL["orange"], label="cross-track component")
    ax.axhspan(-2 / np.sqrt(n_lag), 2 / np.sqrt(n_lag), color=COL["gray"], alpha=0.2, lw=0)
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="lag (steps)", ylabel="autocorrelation", title="Unpredictable part of the disturbance\n(published runs; band ±2/√n)", ylim=(-0.15, 0.15))
    ax.legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "tracking_bound.png", dpi=130)


if __name__ == "__main__":
    main()
