"""The multi-step restoring inequality: a drift function for M2 that charges current headings (baseline).

With xi_i = x_i - c, u_i = h_i - hbar, mu_i = E[h_i' | x] (exact, shepherd/expected_heading.py) and any
memory matrix Lhat(x) known at time t with spectral radius < 1, let Rhat = Lhat (I - Lhat)^-1 and
    Phi(x) = (1/N) sum_i xi_i . (Rhat u)_i,     W = M2 + 2 v Phi.
If every sheep moves, then exactly (docs/jadhav_dilorenzo_bridge.md §11)
    E[dW | x] = 2 v Pi + v^2 D + 2 v (Err_a + Err_b),
    Pi    = (1/N) xi^T (I - Lhat)^-1 (mu - mubar - Lhat u)          the restoring term
    D     = 1 - E[P'^2 | x]                                          heading dispersion
    Err_a = (v/N) E[u'^T Rhat(x') u']        |Err_a| <= v lambda_max(sym Rhat(x')) D
    Err_b = (1/N) E[xi^T (Rhat(x') - Rhat(x)) u']                    zero when Rhat is constant.
Lhat = 0 is the one-step lemma; Lhat = lambda I is the scalar memory inequality (bridge §10.5);
Lhat = w0 (0.5 I + 1.3 A), A the average over each sheep's K nearest, follows the alignment graph.
A positive-part variant W+ = M2 + beta Q+ (beta = 2 v lambda / (1 - lambda)) charges only outward headings:
if V <= lambda Q+ - k mean|xi| then E[dW+ | x] <= -2 v k mean|xi| + v^2 D (1 + lambda)/(1 - lambda) + beta sigma / 2,
sigma = (1/N) sqrt(sum |xi_i|^2 (1 - |mu_i|^2)) + v/2 >= sd(Q' | x).

Checked here, on the baseline driving states outside the shape lemma's ball (as in restoring_condition.py):
 1. the identity, sample by sample and in mean (Monte Carlo under the published dog);
 2. the restoring term on the tested heading families (fans, random headings);
 3. uniformity over heading fields: an adversary searches the 14 headings (polarization >= P_MIN) to make
    Pi >= 0, for each memory matrix; what the failing fields look like.

    .venv/bin/python experiments/memory_inequality.py
Writes results/memory_inequality.json and results/figures/memory_inequality.png.
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
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import restoring_condition as rc  # noqa: E402
from shepherd.expected_heading import drives, expected_headings  # noqa: E402
from shepherd.model import DRIVE, Params, simulate  # noqa: E402

OUT = ROOT / "results"
P0 = Params()
N, V_S = P0.n_sheep, P0.v_sheep
I2 = np.eye(N)[None]
MEMORIES = [("one-step", "scalar", 0.0), ("scalar λ 0.35", "scalar", 0.35), ("scalar λ 0.50", "scalar", 0.50),
            ("graph ω0 0.35", "graph", 0.35), ("graph ω0 0.40", "graph", 0.40), ("scalar⁺ λ 0.35", "scalar+", 0.35)]
MC_MEMORIES = ("scalar λ 0.35", "graph ω0 0.40")
MC_K, SEED = 64, 7
P_MIN = 0.5
ADV_ITERS, ADV_DELTA, ADV_STEP = 15, 0.05, 0.6
CHUNK = 1500
COL = rc.COL
MCOL = {"one-step": "gray", "scalar λ 0.35": "blue", "scalar λ 0.50": "purple", "graph ω0 0.35": "orange", "graph ω0 0.40": "red", "scalar⁺ λ 0.35": "green"}


def fl(x):
    return rc.fl(x)


# ------------------------------------------------------------------ memory matrices and the drift terms

def graph_A(pos):
    """Row-stochastic average over each sheep's K nearest neighbours (positions only)."""
    dr = drives(pos, np.broadcast_to(np.array([1.0, 0.0]), pos.shape), pos.mean(1), P0)
    A = np.zeros((len(pos), N, N))
    np.put_along_axis(A, dr["knn"], 1.0 / P0.k_perceive, axis=2)
    return A


def memory(kind, w, pos):
    if kind in ("scalar", "scalar+"):
        return w * np.repeat(I2, len(pos), 0)
    return w * (0.5 * I2 + 1.3 * graph_A(pos))


def terms(pos, hd, dog, Lh, plus=None):
    """Pi, Phi, V, Q, mean|xi|, M2, D, polarization and sigma (bound on sd Q') for states (S, N, 2), chunked.
    plus = lambda: Pi is the positive-part version (V - lambda Q+) / (1 - lambda)."""
    out = {}
    for i in range(0, len(pos), CHUNK):
        sl = slice(i, i + CHUNK)
        mu, _ = expected_headings(pos[sl], hd[sl], dog[sl], P0)
        xi = pos[sl] - pos[sl].mean(1, keepdims=True)
        u = hd[sl] - hd[sl].mean(1, keepdims=True)
        mubar = mu.mean(1, keepdims=True)
        Rinv = np.linalg.inv(I2 - Lh[sl])
        dot = lambda a, b: np.einsum("snc,snc->sn", a, b).mean(1)
        part = dict(Pi=dot(xi, np.einsum("sij,sjc->sic", Rinv, mu - mubar - np.einsum("sij,sjc->sic", Lh[sl], u))),
                    Phi=dot(xi, np.einsum("sij,sjc->sic", Rinv - I2, u)), V=dot(xi, mu), Q=dot(xi, hd[sl]),
                    m=np.linalg.norm(xi, axis=-1).mean(1), M2=(xi ** 2).sum(-1).mean(1),
                    D=1 - ((mubar[:, 0] ** 2).sum(-1) + (1 - (mu ** 2).sum(-1)).sum(1) / N ** 2),
                    P=np.linalg.norm(hd[sl].mean(1), axis=-1),
                    sigma=np.sqrt(((xi ** 2).sum(-1) * (1 - (mu ** 2).sum(-1))).sum(1)) / N + V_S / 2)
        if plus is not None:
            part["Pi"] = (part["V"] - plus * np.clip(part["Q"], 0, None)) / (1 - plus)
        for k, v in part.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------ 1. the identity, by Monte Carlo

def mc_check(st, hd, kind, w, seed):
    pos, dog = st["pos"], st["dog"]
    S = len(pos)
    Lh = memory(kind, w, pos)
    R = np.linalg.inv(I2 - Lh) - I2
    t = terms(pos, hd, dog, Lh)
    mu, _ = expected_headings(pos, hd, dog, P0)
    xi = pos - pos.mean(1, keepdims=True)
    rep = lambda a: np.repeat(a, MC_K, 0)
    res = simulate(Params(n_sheep=N, box_length=2000.0, n_steps=2), n_runs=S * MC_K, seed=seed,
                   init=dict(pos_s=rep(pos), vel_s=rep(hd), pos_d=rep(dog), vel_d=rep(st["dog_v"])))
    p1, h1 = res.pos_s[..., 1], res.vel_s[..., 1]
    xi1 = p1 - p1.mean(1, keepdims=True)
    u1 = h1 - h1.mean(1, keepdims=True)
    R1 = np.linalg.inv(I2 - memory(kind, w, p1)) - I2
    dot = lambda a, b: np.einsum("snc,snc->sn", a, b).mean(1).reshape(S, MC_K)
    Phi1 = dot(xi1, np.einsum("sij,sjc->sic", R1, u1))
    lin = dot(rep(xi), np.einsum("sij,sjc->sic", rep(R), u1))
    ea = V_S * dot(u1, np.einsum("sij,sjc->sic", R1, u1))
    eb = dot(rep(xi), np.einsum("sij,sjc->sic", R1 - rep(R), u1))
    M2_1 = (xi1 ** 2).sum(-1).mean(1).reshape(S, MC_K)
    lin_exact = np.einsum("snc,snc->sn", xi, np.einsum("sij,sjc->sic", R, mu - mu.mean(1, keepdims=True))).mean(1)
    z = lambda x, mean: (x.mean(1) - mean) / (x.std(1) / np.sqrt(MC_K))
    z_lin, z_M2 = z(lin, lin_exact), z(M2_1, t["M2"] + 2 * V_S * t["V"] + V_S ** 2 * t["D"])
    lam_sym = np.linalg.eigvalsh(0.5 * (R1 + np.swapaxes(R1, 1, 2))).max(-1).reshape(S, MC_K).max(1)
    Ea, Eb = ea.mean(1), eb.mean(1)
    dW = 2 * V_S * t["Pi"] + V_S ** 2 * t["D"] + 2 * V_S * (Ea + Eb)
    W1 = M2_1 + 2 * V_S * Phi1
    dW_mc = W1.mean(1) - (t["M2"] + 2 * V_S * t["Phi"])
    z_dW = (dW_mc - dW) / (W1.std(1) / np.sqrt(MC_K))
    return dict(identity_max_err=fl(np.abs(Phi1 - lin - ea - eb).max()), z_lin_rms=fl(np.sqrt(np.mean(z_lin ** 2))), z_M2_rms=fl(np.sqrt(np.mean(z_M2 ** 2))),
                z_dW_rms=fl(np.sqrt(np.mean(z_dW ** 2))), k_min=fl((-t["Pi"] / t["m"]).min()), k_median=fl(np.median(-t["Pi"] / t["m"])),
                Err_a_median=fl(np.median(Ea)), Err_a_bound_median=fl(np.median(V_S * lam_sym * t["D"])),
                Err_b_over_m_median=fl(np.median(Eb / t["m"])), Err_b_over_m_max=fl((Eb / t["m"]).max()),
                dW_negative_frac=fl((dW < 0).mean()), dW_median=fl(np.median(dW)), dW_max=fl(dW.max()),
                parts_median=dict(restoring=fl(np.median(2 * V_S * t["Pi"])), dispersion=fl(np.median(V_S ** 2 * t["D"])),
                                  err_a=fl(np.median(2 * V_S * Ea)), err_b=fl(np.median(2 * V_S * Eb))))


# ------------------------------------------------------------------ 3. adversary over heading fields

def objective(st, theta, Lh, plus=None):
    hd = np.stack([np.cos(theta), np.sin(theta)], -1)
    t = terms(st["pos"], hd, st["dog"], Lh, plus)
    return t["Pi"] / t["m"] - 10 * np.clip(P_MIN - t["P"], 0, None)


def adversary(st, Lh, seed, plus=None):
    """Projected finite-difference ascent of Pi/mean|xi| over the 14 heading angles, with a random-direction fallback."""
    rng = np.random.default_rng(seed)
    S = len(st["pos"])
    th = np.arctan2(st["hd"][..., 1], st["hd"][..., 0])
    best = objective(st, th, Lh, plus)
    f0 = best.copy()
    step = ADV_STEP
    many = lambda a: np.repeat(a, 2 * N, 0)
    big = dict(pos=many(st["pos"]), dog=many(st["dog"]))
    for it in range(ADV_ITERS):
        TH = np.repeat(th[:, None, :], 2 * N, 1)
        for j in range(N):
            TH[:, 2 * j, j] += ADV_DELTA
            TH[:, 2 * j + 1, j] -= ADV_DELTA
        fs = objective(big, TH.reshape(-1, N), many(Lh), plus).reshape(S, 2 * N)
        grad = (fs[:, 0::2] - fs[:, 1::2]) / (2 * ADV_DELTA)
        grad = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12)
        cand = [th + s * grad for s in (step, step / 3, step / 9)] + [th + step * rng.normal(size=th.shape) / np.sqrt(N)]
        fc = np.stack([objective(st, c, Lh, plus) for c in cand], 1)
        k = fc.argmax(1)
        better = fc[np.arange(S), k] > best
        th = np.where(better[:, None], np.stack(cand, 1)[np.arange(S), k], th)
        best = np.maximum(best, fc[np.arange(S), k])
        if (it + 1) % 5 == 0:
            step *= 0.6
    return f0, best, th


def describe(st, th_adv):
    """What an adversarial heading field looks like, against the natural one."""
    hd = np.stack([np.cos(th_adv), np.sin(th_adv)], -1)
    hb = st["hd"].mean(1)
    e = hb / np.linalg.norm(hb, axis=-1, keepdims=True)
    n = np.stack([-e[:, 1], e[:, 0]], -1)
    xi = st["pos"] - st["pos"].mean(1, keepdims=True)
    y = np.einsum("snc,sc->sn", xi, n)
    th_nat = np.arctan2(st["hd"][..., 1], st["hd"][..., 0])
    turn = (th_adv - th_nat + np.pi) % (2 * np.pi) - np.pi
    outward = turn * np.sign(y)  # positive: turned away from the flock axis
    r = np.abs(y) / np.abs(y).max(1, keepdims=True)
    return dict(P=np.linalg.norm(hd.mean(1), axis=-1), mean_abs_turn_deg=np.degrees(np.abs(turn).mean(1)),
                max_abs_turn_deg=np.degrees(np.abs(turn).max(1)),
                outer_outward_deg=np.degrees(np.where(r > 0.6, outward, np.nan)),
                inner_outward_deg=np.degrees(np.where(r <= 0.6, outward, np.nan)),
                Q=np.einsum("snc,snc->sn", xi, hd).mean(1), Q_nat=np.einsum("snc,snc->sn", xi, st["hd"]).mean(1))


# ------------------------------------------------------------------ main

def outside_ball_states():
    st0 = rc.closed_loop_states()
    drive = st0["mode"] == DRIVE
    pool = []
    for s in rc.DIL:
        st = dict(st0, pos=rc.dilate(st0["pos"], s))
        a = rc.analyse(st["pos"], st["hd"], st["dog"])
        sel = drive & a["sees_all"] & (np.sqrt(a["M2"]) >= rc.RHO)
        if sel.any():
            pool.append({k: st[k][sel] for k in ("pos", "hd", "dog", "dog_v")})
    return {k: np.concatenate([p[k] for p in pool]) for k in pool[0]}


def main():
    tic = time.time()
    st = outside_ball_states()
    S = len(st["pos"])
    print(f"{S} driving states outside the ball, every sheep seeing the dog")
    fams = [("natural", st["hd"])] + [(f"fan {w} {p}°", rc.fanned(st, p, w)) for w in ("all", "inner", "alternate") for p in (20, 40, 60, 90)]
    rng = np.random.default_rng(0)
    hb = st["hd"].mean(1)
    th0 = np.arctan2(hb[:, 1], hb[:, 0])
    for a in (30, 60, 90):
        ang = th0[:, None] + np.deg2rad(a) * rng.uniform(-1, 1, size=(S, N))
        fams.append((f"random ±{a}°", np.stack([np.cos(ang), np.sin(ang)], -1)))

    out = dict(setup=dict(states=S, memories=[dict(name=m[0], kind=m[1], w=m[2]) for m in MEMORIES], P_min=P_MIN, mc=dict(K=MC_K, seed=SEED),
                          adversary=dict(iters=ADV_ITERS, delta=ADV_DELTA, step=ADV_STEP)), memories={})
    for name, kind, w in MEMORIES:
        Lh = memory(kind, w, st["pos"])
        plus = w if kind == "scalar+" else None
        rho = np.abs(np.linalg.eigvals(Lh)).max(-1)
        fam = {}
        for fname, hd in fams:
            t = terms(st["pos"], hd, st["dog"], Lh, plus)
            k = -t["Pi"] / t["m"]
            fam[fname] = dict(k_min=fl(k.min()), k_median=fl(np.median(k)), frac_nonrestoring=fl((k <= 0).mean()))
            if plus is not None:  # the rigorous drift bound for W+, state by state
                lam, beta = plus, 2 * V_S * plus / (1 - plus)
                kk = (lam * np.clip(t["Q"], 0, None) - t["V"]) / t["m"]
                bound = -2 * V_S * kk * t["m"] + V_S ** 2 * t["D"] * (1 + lam) / (1 - lam) + beta * t["sigma"] / 2
                fam[fname].update(W_plus_bound_negative=fl((bound < 0).mean()), sigma_term_median=fl(np.median(beta * t["sigma"] / 2)),
                                  restoring_median=fl(np.median(2 * V_S * kk * t["m"])))
        worst = min(fam.values(), key=lambda d: d["k_min"])
        rec = dict(rho_max=fl(rho.max()), families=fam, k_min_over_families=worst["k_min"])
        print(f"{name}: spectral radius {rho.max():.3f}; restoring margin -Pi/mean|xi| on the tested families: natural min {fam['natural']['k_min']:+.3f}, "
              f"worst family min {worst['k_min']:+.3f}", flush=True)
        if plus is not None:
            print("   W+ rigorous bound negative on: " + ", ".join(f"{f} {d['W_plus_bound_negative']:.2f}" for f, d in fam.items() if f in ("natural", "fan all 40°", "fan alternate 60°", "random ±60°")), flush=True)
        if name in MC_MEMORIES:
            rec["mc"] = {}
            for fname, hd in (("natural", st["hd"]), ("fan all 40°", rc.fanned(st, 40)), ("fan alternate 60°", rc.fanned(st, 60, "alternate"))):
                r = mc_check(st, hd, kind, w, SEED)
                rec["mc"][fname] = r
                print(f"   MC {fname:18s}: identity err {r['identity_max_err']:.1e}, z rms lin {r['z_lin_rms']:.2f} M2 {r['z_M2_rms']:.2f} dW {r['z_dW_rms']:.2f} | "
                      f"Err_a {r['Err_a_median']:+.3f} (bound {r['Err_a_bound_median']:.3f}), Err_b/m {r['Err_b_over_m_median']:+.4f} (max {r['Err_b_over_m_max']:+.4f}) | "
                      f"E dW < 0 on {r['dW_negative_frac']:.2f}, max {r['dW_max']:+.2f}", flush=True)
        f0, fb, th = adversary(st, Lh, SEED, plus)
        d = describe(st, th)
        fail = fb >= 0
        rec["adversary"] = dict(natural_max=fl(f0.max()), adversarial_max=fl(fb.max()), adversarial_median=fl(np.median(fb)), frac_fail=fl(fail.mean()), n_fail=int(fail.sum()),
                                P_median=fl(np.median(d["P"])), P_fail_median=fl(np.median(d["P"][fail])) if fail.any() else None,
                                mean_turn_fail_deg=fl(np.median(d["mean_abs_turn_deg"][fail])) if fail.any() else None,
                                max_turn_fail_deg=fl(np.median(d["max_abs_turn_deg"][fail])) if fail.any() else None,
                                outer_outward_fail_deg=fl(np.nanmedian(d["outer_outward_deg"][fail])) if fail.any() else None,
                                inner_outward_fail_deg=fl(np.nanmedian(d["inner_outward_deg"][fail])) if fail.any() else None,
                                Q_fail_median=fl(np.median(d["Q"][fail])) if fail.any() else None, Q_nat_fail_median=fl(np.median(d["Q_nat"][fail])) if fail.any() else None,
                                curve=np.sort(fb).tolist())
        a = rec["adversary"]
        print(f"   adversary: Pi >= 0 in {a['frac_fail']:.3f} of states (n {a['n_fail']}), max Pi/m {a['adversarial_max']:+.3f}; failing fields: P {a['P_fail_median']}, "
              f"mean turn {a['mean_turn_fail_deg']}°, outer sheep outward {a['outer_outward_fail_deg']}°, inner outward {a['inner_outward_fail_deg']}°, Q {a['Q_fail_median']} (natural {a['Q_nat_fail_median']}) "
              f"({time.time() - tic:.0f} s)", flush=True)
        out["memories"][name] = rec
    out["seconds"] = time.time() - tic
    (OUT / "memory_inequality.json").write_text(json.dumps(out, indent=1))
    figure(out)
    print(f"done in {time.time() - tic:.0f} s")


def figure(out):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    names = [m[0] for m in MEMORIES]
    ax = axs[0]
    xs = np.arange(len(names))
    nat = [out["memories"][n]["families"]["natural"]["k_min"] for n in names]
    fam = [out["memories"][n]["k_min_over_families"] for n in names]
    adv = [-out["memories"][n]["adversary"]["adversarial_max"] for n in names]
    ax.bar(xs - 0.27, nat, 0.26, color=COL["blue"], label="natural headings")
    ax.bar(xs, fam, 0.26, color=COL["orange"], label="worst tested family (fans, random)")
    ax.bar(xs + 0.27, adv, 0.26, color=COL["red"], label="adversarial headings (P ≥ 0.5)")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set_xticks(xs, names, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="smallest restoring margin −Π / mean|ξ| over states", title="Restoring term of W = M₂ + 2vΦ, by memory matrix")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    for n in names:
        c = np.array(out["memories"][n]["adversary"]["curve"])
        ax.plot(c, 1 - np.arange(1, len(c) + 1) / len(c), color=COL[MCOL[n]], lw=1.6, label=n)
    ax.axvline(0, color=COL["gray"], lw=0.8)
    ax.set(xlabel="adversarial Π / mean|ξ|  (≥ 0: restoring fails)", ylabel="share of states above", title="Adversary over the 14 headings, per state", yscale="log", ylim=(1 / 300, 1.05))
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    parts = ["restoring", "dispersion", "err_a", "err_b"]
    labels = ["2vΠ", "v²D", "2v Err_a", "2v Err_b"]
    xs = np.arange(len(parts))
    for j, (n, fname) in enumerate([(MC_MEMORIES[0], "natural"), (MC_MEMORIES[0], "fan all 40°"), (MC_MEMORIES[1], "natural"), (MC_MEMORIES[1], "fan all 40°")]):
        pm = out["memories"][n]["mc"][fname]["parts_median"]
        ax.bar(xs + (j - 1.5) * 0.2, [pm[p] for p in parts], 0.19, color=COL[MCOL[n]], alpha=1.0 if fname == "natural" else 0.55, label=f"{n}, {fname}")
    ax.axhline(0, color=COL["gray"], lw=0.8)
    ax.set_xticks(xs, labels)
    ax.set(ylabel="contribution to E[ΔW] (m², median over states)", title="The drift identity, term by term (Monte Carlo check)")
    ax.legend(frameon=False, fontsize=7)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "memory_inequality.png", dpi=130)


if __name__ == "__main__":
    main()
