"""Hybrid B's sheep under Di Lorenzo's herders instead of Strömbom's dog (a component swap, no rule changes).

    .venv/bin/python experiments/hybrid_b_herders.py   # writes results/hybrid_b_herders.json

Sheep: the `hybrid_first_order_sheep_strombom_dogs` preset exactly (50 first-order sheep, dog push
2.3 e^(−r/5 m) m/s, Morse pair force 1·e^(−r/1) − 0.2·e^(−r/4), D = 0.12 m²/s, dt = 0.1 s), Gaussian goal
σ = 3 m, starts scattered around the goal (sd 8 m) with the dogs on a ring, seeds 0–3, 600 s.

Configurations:
  strombom, plane      the published preset
  strombom, torus 40   the same dog in a 40 m periodic box: isolates the domain change the design needs
  dilorenzo ×1, ×2     Di Lorenzo's herders (design from these sheep's own model, K_H = 10, uncapped, flow at
                       their dt = 0.01: 10 sub-steps) at their minimum count and at twice it, in the same box

Window t ∈ [100, 600] s (excludes the first 100 s). The time-averaged density is the KDE averaged over the
window, compared with the smoothed target (relative L², as Metrics); samples every 1 s. The plane run is
evaluated on an 80 m periodic grid at the same resolution (stray sheep beyond ±40 m wrap; noted).
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import bridge as B  # noqa: E402

OUT = ROOT / "results"
SEEDS, T_END, T0 = (0, 1, 2, 3), 600.0, 100.0
BOX = 40.0


def summarize(r, eval_dom, n_grid):
    F, T = r.followers, r.target
    met = B.Metrics(T, eval_dom, F.n, B.Grid(eval_dom.size, n_grid))
    g, h2 = met.grid, met.grid.h ** 2
    w = r.t >= T0
    xs = r.pos_s[w]
    kde = np.array([g.convolve(met.Kb, g.deposit(eval_dom.wrap(x), np.full(F.n, 1.0 / F.n))) for x in xs])
    inst = [float(((k - met.rho_b) ** 2).sum() * h2 / met.norm) for k in kde]
    avg = kde.mean(0)
    G = np.array([x.mean(0) for x in xs]) if not r.domain.periodic else np.array([r.domain.centroid(x) for x in xs])
    cdist = np.linalg.norm(r.domain.wrap(G - np.array(T.centre)), axis=1)
    occ = [met.occupancy(eval_dom.wrap(x)) for x in xs]
    coh = [float(np.linalg.norm(r.domain.wrap(x - (x.mean(0) if not r.domain.periodic else r.domain.centroid(x))), axis=-1).mean()) for x in xs]
    return dict(occupancy=float(np.mean(occ)), centroid_inside=float(np.mean(cdist <= T.r_goal)),
                centroid_dist_mean=float(cdist.mean()), instantaneous_error=float(np.mean(inst)),
                time_averaged_error=float(((avg - met.rho_b) ** 2).sum() * h2 / met.norm), floor=float(met.floor),
                cohesion=float(np.mean(coh)), cohesion_p99=float(np.percentile(coh, 99)))


def main():
    F, H, T, D, kw = B.preset("hybrid_first_order_sheep_strombom_dogs")
    T = T.resolve(D)
    torus = B.Domain(True, BOX)
    Hd = B.Herders("dilorenzo", n=1, v=np.inf, K_H=10.0, grid=64, substeps=int(round(F.dt / 0.01)))
    need = B.design_herders(F.resolved(), Hd, T, torus).mass_needed
    m1 = int(np.ceil(need))
    configs = {
        "strombom, plane": (H, D, B.Domain(True, 80.0), 128),
        f"strombom, torus {BOX:.0f}": (H, torus, torus, 64),
        f"dilorenzo ×1 ({m1})": (replace(Hd, n=m1), torus, torus, 64),
        f"dilorenzo ×2 ({2 * m1})": (replace(Hd, n=2 * m1), torus, torus, 64),
    }
    res = {"herders_needed": need, "box": BOX, "rows": {}}
    print(f"Di Lorenzo's count for these sheep in a {BOX:.0f} m box: M̂ = {need:.2f} → {m1}", flush=True)
    for name, (Hc, dom, eval_dom, n_grid) in configs.items():
        rows = []
        for s in SEEDS:
            r = B.simulate_bridge(F, Hc, T, dom, seed=s, init=kw["init"], box_length=kw["box_length"],
                                  n_steps=int(T_END / F.dt), store_every=10)
            rows.append(summarize(r, eval_dom, n_grid))
        res["rows"][name] = {k: dict(mean=float(np.mean([x[k] for x in rows])), lo=float(np.min([x[k] for x in rows])),
                                     hi=float(np.max([x[k] for x in rows]))) for k in rows[0]}
        m = res["rows"][name]
        print(f"{name:22s}: occupancy {m['occupancy']['mean']:.2f} ({m['occupancy']['lo']:.2f}–{m['occupancy']['hi']:.2f}), "
              f"centre inside r_goal {m['centroid_inside']['mean']*100:.0f}%, centre dist {m['centroid_dist_mean']['mean']:.1f} m, "
              f"inst error {m['instantaneous_error']['mean']:.2f}, time-avg error {m['time_averaged_error']['mean']:.3f} "
              f"(floor {m['floor']['mean']:.2f}), cohesion {m['cohesion']['mean']:.1f} m (p99 {m['cohesion_p99']['mean']:.1f})", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "hybrid_b_herders.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
