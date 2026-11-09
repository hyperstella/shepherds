"""Di Lorenzo's scenario with Jadhav's sheep: the herders keep every mechanism, only the followers change.

    .venv/bin/python experiments/dilorenzo_jadhav_sheep.py            # writes results/dilorenzo_jadhav_sheep.json + figure
    .venv/bin/python experiments/dilorenzo_jadhav_sheep.py --quick    # 2 seeds
    .venv/bin/python experiments/dilorenzo_jadhav_sheep.py --release  # only the release test, added to the JSON
    .venv/bin/python experiments/dilorenzo_jadhav_sheep.py --timeavg  # only the time-averaged density, added to the JSON

Scenario (`bridge.dilorenzo_scenario`): their 2411.04791 Sec. IV-A setting in metres, 1 m per 12/π m
(their push range L = π is the sheep's 12 m notice radius, box 24 m, r* = 6 m), 1 s per time unit,
36 sheep and 14 herders (their mass fraction 0.28), uniform starts, T = 200 s, herder flow at their
dt = 0.01 with K_H = 10, uncapped, designed by deconvolution from their follower model.

Rows, with identical sheep starts per seed:
  reference     their first-order followers (design and sheep agree)
  jadhav        Jadhav's heading sheep at the published baseline under the same herders
  undesigned    the same herders with a flat design (D = 0 in the herders' model): dogs present, no design
  no dogs       Jadhav sheep alone
  capped        jadhav with the herders capped at the Jadhav dog's 1.5 m/s
and a herder-count sweep for both follower types (feasibility count: M ≥ M̂ (N + M), M̂ = 0.241 → 12),
a box-size sweep, and a release test: the designed herders work until the flock is gathered (first
step with every sheep inside r*, or t = 200 s), then disappear, and the flock runs alone for 120 s.
A published sheep stands only with no herder within r_dog and no neighbour within r_rep, so this is
the only way the flock can come to rest; it shows what shape it rests in.

Time average (--timeavg): the flock keeps moving inside the goal disc, so the instantaneous density
and its time average differ. Runs of 600 s; over t ∈ [100, 600] s the kernel density estimate is
averaged in time and compared with the smoothed target (same relative L² error), the average density
inside r* and inside r*/2 is compared with the target's, and the distance of the flock's centroid from
the goal is summarized.

Reported: tail mean over the last quarter of occupancy within r*, the L² density error (against its
finite-N floor), cohesion (mean distance to the centroid; the target's own value is listed), centroid
error, active fraction, herder speed.
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
KEYS = ("occupancy", "density_error", "cohesion", "centroid_error", "active_fraction", "dog_speed")


def run(cfg, seeds, keep=False):
    F, H, T, D, kw = cfg
    rows, last = [], None
    for s in seeds:
        r = B.simulate_bridge(F, H, T, D, seed=s, store_every=5, **kw)
        m = r.tail_mean(0.25)
        rows.append(m)
        if keep and last is None:
            last = r
    out = {k: float(np.mean([m[k] for m in rows])) for k in KEYS}
    out |= {k + "_sd": float(np.std([m[k] for m in rows])) for k in ("occupancy", "density_error", "cohesion")}
    out["floor"] = float(r.floor)
    out["mass_needed"] = float(r.design.mass_needed) if r.design is not None else None
    out["mass_available"] = float(r.design.mass_available) if r.design is not None else None
    return out, last


def target_cohesion(T, D):
    """Mean distance from the goal under the target density (what a matched flock's cohesion would be)."""
    g = B.Grid(D.size, 128)
    rho, _ = g.target(T.resolve(D), D)
    r = np.hypot(D.wrap(g.X - T.centre[0]), D.wrap(g.Y - T.centre[1]))
    return float((rho * r).sum() / rho.sum())


def hex_cohesion(n, d):
    """Cohesion (mean distance to the centre) of n points hexagonally packed at spacing d in a disc:
    the tightest a standing flock can be, since standing needs every pair at least r_rep apart."""
    R = np.sqrt(n * np.sqrt(3) / 2 * d * d / np.pi)
    return float(2 * R / 3)


def release(seeds, after=120.0):
    """Designed herders until `when`, then none: where and how a released flock comes to rest."""
    F, H, T, D, kw = B.dilorenzo_scenario("jadhav")
    out = {}
    for when in ("gathered", "t200"):
        rows = []
        for s in seeds:
            r = B.simulate_bridge(F, H, T, D, seed=s, **kw)
            if when == "gathered":
                k = int(np.argmax(r.metrics["occupancy"] >= 1.0)) if (r.metrics["occupancy"] >= 1.0).any() else len(r.t) - 1
            else:
                k = len(r.t) - 1
            st = (r.pos_s[k], r.hd_s[k], r.pos_d[k], np.zeros((H.n, 2)))
            q = B.simulate_bridge(F, replace(H, controller="none"), T, D, seed=s, init_state=st,
                                  n_steps=int(round(after / F.dt)))
            mv = np.linalg.norm(D.wrap(np.diff(q.pos_s, axis=0)), axis=-1).max(1)  # largest step at each time
            moving = np.nonzero(mv > 0)[0]
            still_after = float(q.t[moving[-1] + 1]) if len(moving) else 0.0
            if moving.size and moving[-1] == len(mv) - 1:
                still_after = float("nan")  # still moving at the end
            nn = np.linalg.norm(D.wrap(q.pos_s[-1][:, None] - q.pos_s[-1][None]), axis=-1)
            np.fill_diagonal(nn, np.inf)
            a, b = {kk: float(v[k]) for kk, v in r.metrics.items()}, q.final()
            rows.append(dict(t_release=float(r.t[k]), still_after=still_after, min_spacing=float(nn.min()),
                             **{f"{kk}_at_release": a[kk] for kk in KEYS[:4]}, **{f"{kk}_rest": b[kk] for kk in KEYS[:4]}))
        keys = rows[0].keys()
        out[when] = {kk: float(np.nanmean([r_[kk] for r_ in rows])) for kk in keys}
        out[when] |= {kk + "_range": [float(np.nanmin([r_[kk] for r_ in rows])), float(np.nanmax([r_[kk] for r_ in rows]))]
                      for kk in ("still_after", "occupancy_rest", "density_error_rest", "cohesion_rest")}
        print(when, {kk: round(v, 2) for kk, v in out[when].items() if not kk.endswith("_range")}, flush=True)
    out["hex_cohesion"] = hex_cohesion(F.n, F.r_rep)
    out["after_s"] = after
    return out


def time_average(seeds, t_end=600.0, t0=100.0):
    """Time-averaged density and centroid offset over t ∈ [t0, t_end], per case, pooled over seeds."""
    jad = B.dilorenzo_scenario("jadhav")
    F, H, T, D, kw = jad
    cases = {"reference": B.dilorenzo_scenario("first_order"), "jadhav": jad,
             "flat design": (F, replace(H, design_model=replace(H.design_model, D=0.0)), T, D, kw)}
    Tr = T.resolve(D)
    met = B.Metrics(Tr, D, F.n)
    g = met.grid
    r_grid = np.hypot(D.wrap(g.X - Tr.centre[0]), D.wrap(g.Y - Tr.centre[1]))
    h2 = g.h ** 2
    in_goal, in_core = r_grid <= Tr.r_goal, r_grid <= Tr.r_goal / 2
    target = met.rho_b * F.n  # smoothed target as sheep per m²
    edges = np.linspace(0, D.size / 2 * np.sqrt(2), 25)
    rb = np.digitize(r_grid, edges) - 1

    def radial(field):
        return [float(field[rb == i].mean()) if (rb == i).any() else None for i in range(len(edges) - 1)]

    def rel_err(rho):  # same functional as Metrics.density_error, on a unit-mass field
        return float(((rho - met.rho_b) ** 2).sum() * h2 / met.norm)

    def mean_dens(field, mask):
        return float(field[mask].mean())

    out = {"window_s": [t0, t_end], "r_goal": Tr.r_goal, "radial_edges": edges.tolist(),
           "target": {"density_in_goal": mean_dens(target, in_goal), "density_in_core": mean_dens(target, in_core),
                      "mass_in_goal": float(met.rho_b[in_goal].sum() * h2), "radial": radial(target),
                      "mean_distance_from_goal": float((met.rho_b * r_grid).sum() / met.rho_b.sum())},
           "cases": {}}
    maps = {"target": target}
    for name, (Fc, Hc, Tc, Dc, kwc) in cases.items():
        every = max(int(round(1.0 / Fc.dt)), 1)  # sample once per second
        acc, inst, cerr, spread, coh, occ = np.zeros_like(target), [], [], [], [], []
        per_seed = []
        for s in seeds:
            r = B.simulate_bridge(Fc, Hc, Tc, Dc, seed=s, init=kwc["init"], n_steps=int(round(t_end / Fc.dt)),
                                  store_every=every, metric_every=every)
            k0 = int(np.searchsorted(r.t, t0))
            seed_acc = np.zeros_like(target)
            for k in range(k0, len(r.t)):
                x = r.pos_s[k]
                rho = g.convolve(met.Kb, g.deposit(x, np.full(len(x), 1.0 / len(x))))
                seed_acc += rho
                inst.append(rel_err(rho))
                cerr.append(met.centroid_error(x))
                spread.append(float(np.linalg.norm(D.wrap(x - np.array(Tr.centre)), axis=-1).mean()))
                coh.append(met.cohesion(x))
                occ.append(met.occupancy(x))
            seed_acc /= len(r.t) - k0
            per_seed.append(rel_err(seed_acc))
            acc += seed_acc
        acc /= len(seeds)
        cerr = np.array(cerr)
        maps[name] = acc * Fc.n
        out["cases"][name] = {
            "time_avg_error": rel_err(acc), "time_avg_error_per_seed": per_seed,
            "instantaneous_error_mean": float(np.mean(inst)),
            "density_in_goal": mean_dens(acc * Fc.n, in_goal), "density_in_core": mean_dens(acc * Fc.n, in_core),
            "mass_in_goal": float(acc[in_goal].sum() * h2), "occupancy_mean": float(np.mean(occ)),
            "radial": radial(acc * Fc.n),
            "centroid_distance": {"mean": float(cerr.mean()), "median": float(np.median(cerr)),
                                  "p95": float(np.percentile(cerr, 95)), "rms": float(np.sqrt((cerr ** 2).mean())),
                                  "max": float(cerr.max())},
            "mean_distance_from_goal": float(np.mean(spread)), "cohesion_mean": float(np.mean(coh)),
        }
        c = out["cases"][name]
        print(f"{name:12s} time-avg error {c['time_avg_error']:.3f} (instantaneous {c['instantaneous_error_mean']:.3f}); "
              f"density in r* {c['density_in_goal']:.3f} /m² (target {out['target']['density_in_goal']:.3f}), "
              f"in r*/2 {c['density_in_core']:.3f} (target {out['target']['density_in_core']:.3f}); centroid from goal "
              f"mean {c['centroid_distance']['mean']:.2f} m, p95 {c['centroid_distance']['p95']:.2f} m; "
              f"sheep from goal {c['mean_distance_from_goal']:.2f} m (target {out['target']['mean_distance_from_goal']:.2f})", flush=True)
    figure_timeavg(out, maps, D)
    return out


def figure_timeavg(out, maps, D):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    L = D.size
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.4))
    vmax = max(m.max() for m in maps.values())
    for a, name in zip(ax[:4], ("target", "reference", "jadhav", "flat design")):
        im = a.imshow(maps[name].T, origin="lower", extent=(-L / 2, L / 2, -L / 2, L / 2), cmap="viridis", vmin=0, vmax=vmax)
        a.add_patch(plt.Circle((0, 0), out["r_goal"], fill=False, ls="--", color="w"))
        title = {"target": "target (smoothed)", "reference": "their followers, time-averaged",
                 "jadhav": "Jadhav sheep, time-averaged", "flat design": "Jadhav sheep, flat design, time-averaged"}[name]
        if name != "target":
            title += f"\nL² error {out['cases'][name]['time_avg_error']:.2f} (instantaneous {out['cases'][name]['instantaneous_error_mean']:.2f})"
        a.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax[3], label="sheep per m²", fraction=0.046)
    e = np.array(out["radial_edges"])
    mid = (e[:-1] + e[1:]) / 2
    ax[4].plot(mid, out["target"]["radial"], "k--", label="target")
    for name, c in (("reference", "tab:blue"), ("jadhav", "tab:orange"), ("flat design", "tab:purple")):
        ax[4].plot(mid, out["cases"][name]["radial"], color=c, label=name)
    ax[4].axvline(out["r_goal"], color="tab:green", ls=":", lw=1, label="r*")
    ax[4].set_xlabel("distance from goal (m)"), ax[4].set_ylabel("time-averaged sheep per m²"), ax[4].legend(fontsize=8)
    ax[4].set_title(f"radial profile, t ∈ [{out['window_s'][0]:.0f}, {out['window_s'][1]:.0f}] s", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "dilorenzo_jadhav_timeavg.png", dpi=120)


def main(quick=False):
    seeds = range(2) if quick else range(5)
    jad = B.dilorenzo_scenario("jadhav")
    ref = B.dilorenzo_scenario("first_order")
    F, H, T, D, kw = jad
    flat = replace(H, design_model=replace(H.design_model, D=0.0))
    configs = {
        "reference": ref,
        "jadhav": jad,
        "undesigned": (F, flat, T, D, kw),
        "no dogs": (F, replace(H, controller="none"), T, D, kw),  # sheep are drawn first: same starts
        "capped": (F, replace(H, v=1.5), T, D, kw),
    }
    t0 = time.time()
    res = {"setting": dict(box=D.size, r_goal=T.resolve(D).r_goal, n_sheep=F.n, n_dogs=H.n, T_s=kw["n_steps"] * F.dt,
                           seeds=len(seeds), target_cohesion=target_cohesion(T, D)), "rows": {}, "sweep": {}}
    examples = {}
    for name, cfg in configs.items():
        res["rows"][name], examples[name] = run(cfg, seeds, keep=True)
        print(f"{name:11s} " + " ".join(f"{k}={res['rows'][name][k]:.3f}" for k in KEYS), f"[{time.time()-t0:.0f}s]", flush=True)
    counts = (4, 8, 12, 14, 20, 28)
    for sheep in ("first_order", "jadhav"):
        res["sweep"][sheep] = {}
        for m in counts:
            res["sweep"][sheep][m], _ = run(B.dilorenzo_scenario(sheep, n_dogs=m), seeds)
            r = res["sweep"][sheep][m]
            print(f"{sheep:11s} M={m:2d} occ={r['occupancy']:.2f} err={r['density_error']:.2f} coh={r['cohesion']:.2f}"
                  f" feasible={r['mass_available'] >= r['mass_needed']}", flush=True)
    # Box size relative to the notice radius: their L = π maps to k·r_dog (box 2k·r_dog), sheep density and
    # herder fraction fixed (n_sheep = 36 k²). At k = 1 every sheep sees a herder; larger k leaves the goal
    # region outside every herder's notice disc.
    res["scale"] = {}
    for sheep in ("first_order", "jadhav"):
        res["scale"][sheep] = {}
        for k in (1.0, 1.5, 2.0):
            cfg = B.dilorenzo_scenario(sheep, n_sheep=int(round(36 * k * k)), scale=k * 12.0 / np.pi)
            res["scale"][sheep][k], _ = run(cfg, seeds)
            r = res["scale"][sheep][k]
            r["box"] = cfg[3].size
            print(f"{sheep:11s} box={cfg[3].size:.0f} m N={cfg[0].n} M={cfg[1].n} occ={r['occupancy']:.2f}"
                  f" err={r['density_error']:.2f} floor={r['floor']:.2f} coh={r['cohesion']:.2f}"
                  f" active={r['active_fraction']:.2f} cerr={r['centroid_error']:.1f}", flush=True)
    res["release"] = release(seeds)
    res["time_average"] = time_average(seeds)
    OUT.mkdir(exist_ok=True)
    (OUT / "dilorenzo_jadhav_sheep.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res, examples)
    return res


def figure(res, ex):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
    names = ("reference", "jadhav", "undesigned", "no dogs")
    titles = ("their followers (reference)", "Jadhav sheep", "Jadhav sheep, flat design", "Jadhav sheep, no dogs")
    for a, name, title in zip(ax[0], names, titles):
        r = ex[name]
        L = r.domain.size
        if r.design is not None:
            a.imshow(r.design.rho_H.T, origin="lower", extent=(-L / 2, L / 2, -L / 2, L / 2), cmap="Greys", alpha=0.5)
        a.add_patch(plt.Circle(r.target.centre, r.target.r_goal, fill=False, ls="--", color="tab:green"))
        a.scatter(*r.pos_s[-1].T, s=14, color="tab:blue", label="sheep")
        if r.pos_d.shape[1]:
            a.scatter(*r.pos_d[-1].T, s=30, marker="^", color="tab:red", label="herders")
        a.set_xlim(-L / 2, L / 2), a.set_ylim(-L / 2, L / 2), a.set_aspect("equal")
        a.set_title(f"{title}\nt = {r.t[-1]:.0f} s (seed 0); grey: designed herder density", fontsize=9)
    ax[0, 0].legend(loc="upper right", fontsize=8)
    colors = {"reference": "tab:blue", "jadhav": "tab:orange", "undesigned": "tab:purple", "no dogs": "tab:grey"}
    for name in names:
        r = ex[name]
        ax[1, 0].plot(r.t, r.metrics["occupancy"], color=colors[name], label=name)
        ax[1, 1].plot(r.t, r.metrics["density_error"], color=colors[name])
        ax[1, 2].plot(r.t, r.metrics["cohesion"], color=colors[name])
    ax[1, 1].axhline(ex["reference"].floor, color="k", ls=":", lw=1, label="finite-N floor")
    ax[1, 2].axhline(res["setting"]["target_cohesion"], color="tab:green", ls="--", lw=1, label="target's cohesion")
    ax[1, 0].set_ylabel("fraction within r*"), ax[1, 1].set_ylabel("relative L² density error")
    ax[1, 2].set_ylabel("cohesion (m)"), ax[1, 1].set_yscale("log")
    for a in ax[1, :3]:
        a.set_xlabel("t (s)")
    ax[1, 0].legend(fontsize=8), ax[1, 1].legend(fontsize=8), ax[1, 2].legend(fontsize=8)
    for sheep, c in (("first_order", "tab:blue"), ("jadhav", "tab:orange")):
        sw = res["sweep"][sheep]
        m = sorted(int(k) for k in sw)
        ax[1, 3].errorbar(m, [sw[k]["occupancy"] for k in m], [sw[k]["occupancy_sd"] for k in m], color=c, marker="o",
                          label="their followers" if sheep == "first_order" else "Jadhav sheep", capsize=2)
    need = res["rows"]["reference"]["mass_needed"]
    n = res["setting"]["n_sheep"]
    ax[1, 3].axvline(need * n / (1 - need), color="k", ls=":", lw=1, label="feasibility count")
    ax[1, 3].set_xlabel("herders"), ax[1, 3].set_ylabel("tail occupancy within r*"), ax[1, 3].legend(fontsize=8)
    ax[1, 3].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "dilorenzo_jadhav_sheep.png", dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--release", action="store_true", help="run only the release test and add it to the existing JSON")
    ap.add_argument("--timeavg", action="store_true", help="run only the time-averaged density and add it to the existing JSON")
    a = ap.parse_args()
    if a.release or a.timeavg:
        path = OUT / "dilorenzo_jadhav_sheep.json"
        res = json.loads(path.read_text())
        seeds = range(2) if a.quick else range(5)
        if a.release:
            res["release"] = release(seeds)
        if a.timeavg:
            res["time_average"] = time_average(seeds)
        path.write_text(json.dumps(res, indent=1, default=float))
    else:
        main(a.quick)
