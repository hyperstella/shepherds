"""The shared experiment: how social interactions and herder numbers change the density a flock
reaches, for Di Lorenzo's followers, Jadhav's sheep, and the two hybrids.

    .venv/bin/python experiments/bridge_compare.py            # writes results/bridge_compare.json + figures
    .venv/bin/python experiments/bridge_compare.py --quick    # fewer seeds and dog counts

Every comparison varies one component at a time, with identical sheep starting positions
(the same seed draws the sheep before the dogs) and reports the tail mean of the last quarter
of each run against the finite-population floor of the density error.

E1  First-order followers on the torus (Di Lorenzo's units): dog count sweep for non-interacting
    followers (their 2024 model), Morse-cohesive followers (their 2026 model) and cohesive followers
    with the alignment extension, with the feasibility calculation's N̂_H beside each.
E2  Jadhav's sheep on the open plane with Strömbom dog(s): dog count × attraction × alignment,
    published activation, with and without the reach-and-release hold.
E3  Hybrid on the torus: Jadhav heading sheep (always active) under Di Lorenzo's feedforward
    herders, dog count × attraction, next to first-order sheep under the same herders.
E4  Matched resources: the density-feedback dogs against Strömbom-with-hold, Di Lorenzo's herders and
    no dogs, holding a gathered flock at the goal and reaching it from 30–40 m away.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import bridge as B  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"


def tail(r, frac=0.25):
    m = r.tail_mean(frac)
    m["floor"] = r.floor
    m["t_end"] = float(r.t[-1])
    return m


def run_many(F, H, T, D, seeds, **kw):
    rows = [tail(B.simulate_bridge(F, H, T, D, seed=s, **kw)) for s in seeds]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {
        k + "_sd": float(np.std([r[k] for r in rows])) for k in ("occupancy", "density_error", "cohesion")}


# ---------------------------------------------------------------------------- E1

def e1(quick, only=None):
    """First-order followers, torus, their units; sweep N_H for three follower variants."""
    N = 200
    dom = B.Domain(True, 2 * np.pi, images=1)
    T = B.Target("vonmises", k=6 / np.pi, r_goal=np.pi / 2, bandwidth=0.3)
    base = B.Followers("first_order", n=N, dt=0.05, D=0.01, a_dog=1.0, L_dog=np.pi, population_scaling=True)
    # the 2026 paper's "weak Morse" kernel: (1/ℓ_r) e^(−r/ℓ_r) − (ζ/ℓ_a) e^(−r/ℓ_a), ℓ_r = π/2, ℓ_a = π, ζ = 1
    morse = dict(g_rep=2 / np.pi, l_rep=np.pi / 2, g_att=1 / np.pi, l_att=np.pi)
    variants = {
        "non-interacting (2024)": base,
        "Morse cohesion (2026)": replace(base, **morse),
        "Morse cohesion + alignment (extension)": replace(base, **morse, ali_fo=0.5, r_ali=1.0),
        "attraction only (the Morse kernel's attractive part)": replace(base, g_att=1 / np.pi, l_att=np.pi),
    }
    counts = [10, 25, 50, 80, 120] if quick else [5, 10, 20, 30, 40, 50, 65, 80, 100, 120, 160]
    seeds = [0] if quick else [0, 1]
    n_steps = 1200 if quick else 2000
    out = {"n_sheep": N, "dog_counts": counts, "variants": {}}
    for name, F in variants.items():
        if only is not None and name not in only:
            continue
        scan = B.feasibility_scan(F, T, dom, masses=np.linspace(0.02, 0.98, 97), grid=64)
        m1 = scan["min_feasible_mass"]
        rows = []
        for M in counts:
            t0 = time.time()
            H = B.Herders("dilorenzo", n=M, v=np.inf, K_H=10.0, bandwidth=0.4, grid=64)
            row = run_many(F, H, T, dom, seeds, n_steps=n_steps, init="uniform", store_every=50)
            d = B.design_herders(F, H, T, dom)
            row |= {"n_dogs": M, "mass": M / (N + M), "mass_needed": d.mass_needed, "feasible": d.feasible,
                    "design_drift_error": d.drift_error}
            rows.append(row)
            print(f"E1 {name:40s} N_H={M:4d} occ={row['occupancy']:.3f} err={row['density_error']:.3f} "
                  f"(floor {row['floor']:.3f}) feasible={d.feasible} [{time.time() - t0:.0f}s]", flush=True)
        out["variants"][name] = {
            "rows": rows, "min_feasible_mass": m1,
            "n_dogs_needed": None if m1 is None else B.herders_from_mass(N, m1),
            "scan": {"mass": scan["mass"].tolist(), "needed": scan["needed"].tolist()},
        }
    return out


# ---------------------------------------------------------------------------- E2

def e2(quick):
    """Jadhav's sheep on the plane: dog count × attraction × alignment, with and without the hold."""
    N = 14
    T = B.Target("gaussian", sigma=2.5)
    dom = B.Domain(False)
    seeds = list(range(4 if quick else 20))
    counts = [1, 2, 3] if quick else [1, 2, 3, 4]
    atts = [0.5, 1.5, 2.5]
    alis = [0.0, 1.3, 2.6]
    kw = dict(n_steps=400, init="jadhav", box_length=60.0, store_every=4)
    rows = []
    for hold in (None, 6.0):
        for M in counts:
            for wa in atts:
                for wl in alis:
                    if wa != 1.5 and wl != 1.3:
                        continue  # vary one social weight at a time around the baseline
                    F = B.Followers(n=N, w_att=wa, w_ali=wl)
                    H = B.Herders("strombom", n=M, hold_radius=hold)
                    row = run_many(F, H, T, dom, seeds, **kw)
                    row |= {"n_dogs": M, "w_att": wa, "w_ali": wl, "hold": hold is not None}
                    rows.append(row)
                    print(f"E2 hold={hold is not None!s:5s} dogs={M} w_att={wa} w_ali={wl}: occ={row['occupancy']:.2f} "
                          f"err={row['density_error']:.2f} (floor {row['floor']:.2f}) coh={row['cohesion']:.2f}", flush=True)
    return {"n_sheep": N, "rows": rows}


# ---------------------------------------------------------------------------- E3

def e3(quick):
    """Hybrid on the torus: heading sheep and first-order sheep under Di Lorenzo herders.

    Their von Mises goal on an 80 m torus (k = 3/r*, r* = 10 m), as a Gaussian is negligible over most
    of the box and gives the design no far-field drift. All followers carry D = 0.5 m²/s (within the
    0.12–0.68 measured in feasibility.py; for the heading sheep a labelled hybrid, since the design needs
    a diffusion to confine and with D = 0 it is empty). The push the design assumes is a weak one,
    0.8 e^(−r/12 m) m/s, so that the feasibility count (about 7 dogs) lies inside the sweep: a real
    dog's 2.3 m/s is 50× the drift the design needs and a few such dogs scatter the flock (§4.3).
    """
    N = 60
    dom = B.Domain(True, 80.0)
    T = B.Target("vonmises", k=3 / (2 * np.pi * 10 / 80), r_goal=10.0, bandwidth=1.5)
    seeds = [0] if quick else [0, 1, 2]
    counts = [4, 8, 16, 32] if quick else [2, 4, 6, 8, 12, 16, 24, 32, 48]
    kw = dict(init="uniform", store_every=10)
    heading = B.Followers("heading", n=N, activation="always", a_dog=0.8, L_dog=12.0, population_scaling=False, D=0.5)
    first = B.Followers("first_order", n=N, dt=0.5, D=0.5, a_dog=0.8, L_dog=12.0, population_scaling=False)
    variants = {
        "heading sheep, w_att 0.5": replace(heading, w_att=0.5),
        "heading sheep, w_att 1.5 (baseline)": heading,
        "heading sheep, w_att 2.5": replace(heading, w_att=2.5),
        "heading sheep, w_ali 0": replace(heading, w_ali=0.0),
        "heading sheep, published activation": replace(heading, activation="published"),
        "first-order sheep (design assumption)": first,
        "first-order sheep + Morse cohesion": replace(first, g_rep=0.5, l_rep=1.5, g_att=0.05, l_att=6.0),
    }
    out = {"n_sheep": N, "dog_counts": counts, "variants": {}}
    for name, F in variants.items():
        rows = []
        for M in counts:
            H = B.Herders("dilorenzo", n=M, v=1.5, K_H=1.0, grid=64)
            row = run_many(F, H, T, dom, seeds, n_steps=int(round(600 / F.dt)), **kw)  # 600 s for every rule
            d = B.design_herders(F, H, T, dom)
            row |= {"n_dogs": M, "n_dogs_needed": d.mass_needed, "feasible": d.feasible, "design_drift_error": d.drift_error}
            rows.append(row)
            print(f"E3 {name:40s} N_H={M:3d} occ={row['occupancy']:.3f} err={row['density_error']:.3f} "
                  f"(floor {row['floor']:.3f}) coh={row['cohesion']:.2f} needed={d.mass_needed:.1f}", flush=True)
        out["variants"][name] = {"rows": rows}
    return out


# ---------------------------------------------------------------------------- E4

def _series_stats(r, occ_level=0.6):
    """Time to first reach occupancy ≥ occ_level, and the share of the second half of the run held there."""
    occ = r.metrics["occupancy"]
    hit = np.flatnonzero(occ >= occ_level)
    half = len(occ) // 2
    return {"t_reach": float(r.t[hit[0]]) if len(hit) else float("nan"),
            "held_fraction": float((occ[half:] >= occ_level).mean())}


def run_many4(F, H, T, D, seeds, init_fn, **kw):
    rows = []
    for s in seeds:
        r = B.simulate_bridge(F, H, T, D, seed=s, init_state=init_fn(s), **kw)
        rows.append(tail(r) | _series_stats(r))
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys} | {
        "reached_fraction": float(np.mean([np.isfinite(r["t_reach"]) for r in rows])),
        "occupancy_sd": float(np.std([r["occupancy"] for r in rows]))}


def e4(quick):
    """Matched resources: the density-feedback dogs against Strömbom-with-hold, Di Lorenzo's herders and
    no dogs, on two protocols: 'hold' (the flock starts gathered at the goal; can the controller keep it
    there under the sheep's own motion?) and 'reach' (the flock starts 40 m away, cohesive). Same dog
    count, speed, seeds and starts for every controller."""
    rng0 = np.random.default_rng
    out = {}
    # ---- plane: Jadhav's 14 sheep
    N = 14
    T = B.Target("gaussian", sigma=2.5)
    dom = B.Domain(False)
    seeds = list(range(5 if quick else 20))
    counts = [1, 2] if quick else [1, 2, 3]
    followers = {"published activation": B.Followers(n=N), "always active (adaptation)": B.Followers(n=N, activation="always")}

    def init_plane(seed, offset):
        rng = rng0(seed)
        pos = np.array([offset, 0.0]) + 2.0 * rng.standard_normal((N, 2))
        hd = B._unit(rng.standard_normal((N, 2)))
        return pos, hd, None, None

    def controllers(M, floor):
        c = {"no dogs": B.Herders("none", n=0),
             "Strömbom + hold": B.Herders("strombom", n=M, hold_radius=6.0),
             "density feedback": B.Herders("density_feedback", n=M),
             "density feedback + release": B.Herders("density_feedback", n=M, fb_release=2 * floor)}
        return c

    floor = B.finite_population_floor(T.resolve(dom), dom, N)
    out["plane"] = {"n_sheep": N, "floor": floor, "rows": []}
    for fname, F in followers.items():
        for proto, offset in (("hold", 0.0), ("reach", 40.0)):
            for M in counts:
                for cname, H in controllers(M, floor).items():
                    if cname == "no dogs" and M != counts[0]:
                        continue
                    def init_fn(seed, offset=offset, M=M):
                        pos, hd, _, _ = init_plane(seed, offset)
                        rng = rng0(1000 + seed)
                        ang = 2 * np.pi * rng.random(M)
                        pd = pos.mean(0) + 20.0 * np.stack([np.cos(ang), np.sin(ang)], -1)
                        return pos, hd, pd, np.tile([[1.0, 0.0]], (M, 1))
                    row = run_many4(F, H, T, dom, seeds, init_fn, n_steps=600, store_every=4)
                    row |= {"followers": fname, "protocol": proto, "n_dogs": 0 if cname == "no dogs" else M, "controller": cname}
                    out["plane"]["rows"].append(row)
                    print(f"E4 plane {fname:28s} {proto:5s} M={M} {cname:28s} occ={row['occupancy']:.2f} err={row['density_error']:.2f} "
                          f"held={row['held_fraction']:.2f} t_reach={row['t_reach']:.0f}", flush=True)
    # ---- torus: the E3 setting
    N = 60
    dom = B.Domain(True, 80.0)
    T = B.Target("vonmises", k=3 / (2 * np.pi * 10 / 80), r_goal=10.0, bandwidth=1.5)
    seeds = [0] if quick else [0, 1, 2]
    counts = [2, 8] if quick else [1, 2, 4, 8, 16]
    heading = B.Followers("heading", n=N, activation="always", a_dog=0.8, L_dog=12.0, population_scaling=False, D=0.5)
    first = B.Followers("first_order", n=N, dt=0.5, D=0.5, a_dog=0.8, L_dog=12.0, population_scaling=False)
    followers = {"heading, always + D (hybrid)": heading, "first-order": first}
    floor = B.finite_population_floor(T.resolve(dom), dom, N)
    out["torus"] = {"n_sheep": N, "floor": floor, "rows": []}
    for fname, F in followers.items():
        for proto, offset in (("hold", 0.0), ("reach", 30.0)):
            for M in counts:
                ctrl = {"no dogs": B.Herders("none", n=0),
                        "Di Lorenzo feedforward": B.Herders("dilorenzo", n=M, v=1.5, K_H=1.0, grid=64),
                        "Strömbom + hold": B.Herders("strombom", n=M, v=1.5, hold_radius=5.0),
                        "density feedback": B.Herders("density_feedback", n=M, v=1.5),
                        "density feedback + release": B.Herders("density_feedback", n=M, v=1.5, fb_release=2 * floor)}
                for cname, H in ctrl.items():
                    if cname == "no dogs" and M != counts[0]:
                        continue
                    def init_fn(seed, offset=offset, M=M):
                        rng = rng0(seed)
                        pos = np.array([offset, 0.0]) + 5.0 * rng.standard_normal((N, 2))
                        hd = B._unit(rng.standard_normal((N, 2)))
                        rng = rng0(1000 + seed)
                        ang = 2 * np.pi * (np.arange(M) + rng.random()) / M
                        pd = pos.mean(0) + 25.0 * np.stack([np.cos(ang), np.sin(ang)], -1)
                        return dom.wrap(pos), hd, dom.wrap(pd), np.tile([[1.0, 0.0]], (M, 1))
                    row = run_many4(F, H, T, dom, seeds, init_fn, n_steps=int(600 / F.dt), store_every=int(round(5 / F.dt)))
                    row |= {"followers": fname, "protocol": proto, "n_dogs": 0 if cname == "no dogs" else M, "controller": cname}
                    out["torus"]["rows"].append(row)
                    print(f"E4 torus {fname:28s} {proto:5s} M={M:2d} {cname:28s} occ={row['occupancy']:.2f} err={row['density_error']:.2f} "
                          f"held={row['held_fraction']:.2f} t_reach={row['t_reach']:.0f}", flush=True)
    return out


def figures4(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e4 = res["E4"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row_i, (dom_name, fnames) in enumerate((("plane", ["published activation", "always active (adaptation)"]),
                                                 ("torus", ["heading, always + D (hybrid)", "first-order"]))):
        rows = e4[dom_name]["rows"]
        for col, (fname, proto) in enumerate([(f, p) for f in fnames for p in ("hold", "reach")]):
            ax = axes[row_i, col]
            for cname in sorted({r["controller"] for r in rows}):
                sel = sorted([r for r in rows if r["followers"] == fname and r["protocol"] == proto and r["controller"] == cname], key=lambda r: r["n_dogs"])
                if cname == "no dogs":
                    ax.axhline(sel[0]["occupancy"], color="k", lw=0.8, ls="--", label="no dogs")
                else:
                    ax.plot([r["n_dogs"] for r in sel], [r["occupancy"] for r in sel], "o-", label=cname)
            ax.set(title=f"{dom_name}: {fname}\n{proto}", xlabel="dogs", ylabel="occupancy (tail mean)", ylim=(0, 1.02))
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bridge_controllers.png", dpi=120)


# ---------------------------------------------------------------------------- figures

def figures(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, v in res["E1"]["variants"].items():
        x = [r["n_dogs"] for r in v["rows"]]
        ax[0].plot(x, [r["occupancy"] for r in v["rows"]], "o-", label=name)
        ax[1].plot(x, [r["density_error"] for r in v["rows"]], "o-", label=name)
        if v["n_dogs_needed"] is not None and v["n_dogs_needed"] <= 1.5 * max(x):  # off-scale counts are in the JSON
            for a in ax[:2]:
                a.axvline(v["n_dogs_needed"], ls=":", color=ax[0].lines[-1].get_color())
    ax[1].axhline(res["E1"]["variants"]["non-interacting (2024)"]["rows"][0]["floor"], color="k", lw=0.8, ls="--",
                  label="finite-N floor")
    ax[0].set(xlabel="herders N_H", ylabel="occupancy (tail mean)", title=f"E1: first-order followers, N_T={res['E1']['n_sheep']}, torus")
    ax[1].set(xlabel="herders N_H", ylabel="relative L² density error", yscale="log", title="dotted: feasibility N̂_H")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    for name, v in res["E3"]["variants"].items():
        x = [r["n_dogs"] for r in v["rows"]]
        ax[2].plot(x, [r["occupancy"] for r in v["rows"]], "o-", label=name)
    ax[2].set(xlabel="herders N_H", ylabel="occupancy (tail mean)", title="E3: hybrid, Di Lorenzo herders, torus 80 m")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bridge_compare.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rows = res["E2"]["rows"]
    for hold, a in zip((False, True), ax):
        for wa in sorted({r["w_att"] for r in rows}):
            sel = [r for r in rows if r["hold"] == hold and r["w_att"] == wa and r["w_ali"] == 1.3]
            a.plot([r["n_dogs"] for r in sel], [r["occupancy"] for r in sel], "o-", label=f"w_att={wa}")
        for wl in (0.0, 2.6):
            sel = [r for r in rows if r["hold"] == hold and r["w_att"] == 1.5 and r["w_ali"] == wl]
            a.plot([r["n_dogs"] for r in sel], [r["occupancy"] for r in sel], "s--", label=f"w_ali={wl}")
        a.set(xlabel="dogs", ylabel="occupancy (tail mean)", ylim=(0, 1.02),
              title=f"E2: Jadhav sheep, Strömbom dog(s), {'with hold' if hold else 'published loop'}")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bridge_compare_jadhav.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=["E1", "E2", "E3"])
    args = ap.parse_args()
    path = OUT / "bridge_compare.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    for name, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)):
        if name in args.only:
            res[name] = fn(args.quick)
            path.write_text(json.dumps(res, indent=1))
    if all(k in res for k in ("E1", "E2", "E3")):
        figures(res)
    if "E4" in res:
        figures4(res)
    print(f"done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
