"""The shared scorecard (shepherd/scorecard.py) for every hybrid studied in docs/bridge_simulation.md §4.

    .venv/bin/python experiments/hybrid_scorecard.py   # writes results/hybrid_scorecard.json + figure

Every run: seeds 0–3, 600 s, its scenario's own goal circle and preset start, scored every 1 s with the same
thresholds (scorecard.DEFAULTS: success = occupancy ≥ 0.8 for 30 s; final window = last 150 s; escape =
occupancy < 0.5 for ≥ 5 s after success; split = largest 5 m-linked group < 90 %). Agents' rules are the
presets'; only the controller (and, for §4.7, the goal schedule) differs within a scenario family. Starts,
goal circles and sheep counts differ between families, so times compare controllers within a family only.
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import bridge as B  # noqa: E402
from shepherd import goal_schedule as GS  # noqa: E402
from shepherd.scorecard import DEFAULTS, score  # noqa: E402

OUT, FIG = ROOT / "results", ROOT / "results" / "figures"
SEEDS, T_END = (0, 1, 2, 3), 600.0


def scenarios():
    out = {}
    # A. Jadhav's sheep on the plane, Strömbom's dog (published start 250 m out)
    F, H, T, D, _ = B.preset("jadhav_reference")
    out["A. Jadhav: Strömbom dog (published)"] = (F, H, T, D, dict(init="jadhav", box_length=250.0), None)
    out["A. Jadhav: Strömbom + reach-and-release (ext., 6 m)"] = (F, replace(H, hold_radius=6.0), T, D,
                                                                  dict(init="jadhav", box_length=250.0), None)
    # B. Di Lorenzo's scenario in metres (§4.5, §4.7)
    Fm, Hm, Tm, Dm, _ = B.preset("dilorenzo_metres")
    out["B. DL metres: their followers"] = (Fm, Hm, Tm, Dm, dict(init="uniform"), None)
    Fj, Hj, Tj, Dj, _ = B.preset("dilorenzo_jadhav_sheep")
    for name in ("fixed goal", "hop schedule", "glide schedule"):
        out[f"B. DL metres: Jadhav sheep, {name}"] = (Fj, Hj, Tj, Dj, dict(init="uniform"), name)
    # C. Hybrid A
    Fa, Ha, Ta, Da, _ = B.preset("hybrid_jadhav_sheep_dilorenzo_dogs")
    out["C. Hybrid A (as preset)"] = (Fa, Ha, Ta, Da, dict(init="uniform"), None)
    # D. Hybrid B, and its sheep under Di Lorenzo's herders in a 40 m box (§4.8)
    Fb, Hb, Tb, Db, _ = B.preset("hybrid_first_order_sheep_strombom_dogs")
    out["D. Hybrid B: Strömbom dog, plane (preset)"] = (Fb, Hb, Tb, Db, dict(init="spread", box_length=8.0), None)
    torus = B.Domain(True, 40.0)
    Hd = B.Herders("dilorenzo", n=7, v=np.inf, K_H=10.0, grid=64, substeps=int(round(Fb.dt / 0.01)))
    out["D. Hybrid B sheep: Di Lorenzo, 7 herders, 40 m box"] = (Fb, Hd, Tb, torus, dict(init="spread", box_length=8.0), None)
    return out


def goal_for(name, seed, ctx):
    if name is None or name == "fixed goal":
        return None
    _, D, g, P = ctx
    wp = GS.draw_waypoints(P, g, int(T_END // 20) + 2, np.random.default_rng(1000 + seed))
    return GS.HopSchedule(wp, 20) if name == "hop schedule" else GS.GlideSchedule(wp, 20, D)


def main():
    t0 = time.time()
    # requested centre distribution for §4.7's schedules (separate seeds, as goal_schedule_validation.py)
    Fj, Hj, Tj, Dj, _ = B.preset("dilorenzo_jadhav_sheep")
    Tj = Tj.resolve(Dj)
    g = B.Grid(Dj.size, Hj.grid)
    ref = []
    for s in (100, 101, 102):
        r = B.simulate_bridge(Fj, Hj, Tj, Dj, seed=s, init="uniform", n_steps=300)
        ref += [r.pos_s[k] for k in range(100, 301)]
    P, _ = GS.centre_distribution(GS.flock_shape(ref, Dj, g), Tj, Dj, g)
    ctx = (Tj, Dj, g, P)
    res = {"thresholds": DEFAULTS, "seeds": list(SEEDS), "t_end": T_END, "rows": {}}
    for name, (F, H, T, D, kw, sched) in scenarios().items():
        rows = []
        for s in SEEDS:
            r = B.simulate_bridge(F, H, T, D, seed=s, n_steps=int(round(T_END / F.dt)), goal=goal_for(sched, s, ctx), **kw)
            rows.append(score(r))
        res["rows"][name] = rows
        succ = [x["time_to_success"] for x in rows]
        print(f"{name:52s} success {sum(v is not None for v in succ)}/4, t {np.mean([v for v in succ if v is not None]) if any(v is not None for v in succ) else float('nan'):5.0f} s, "
              f"final occ {np.mean([x['final_occupancy'] for x in rows]):.2f}, escapes {np.mean([x['escapes'] for x in rows]):.1f}, "
              f"RMS {np.mean([x['rms_spread_final'] for x in rows]):.1f} m, largest {np.mean([x['largest_group_final'] for x in rows]):.2f}, "
              f"unrec. splits {sum(x['splits_unrecovered'] for x in rows)}, dogs {rows[0]['dogs']}, dist/dog {np.mean([x['distance_per_dog'] for x in rows]):.0f} m "
              f"[{time.time()-t0:.0f}s]", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "hybrid_scorecard.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res)


def figure(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fams = sorted({k.split(".")[0] for k in res["rows"]})
    titles = {"A": "Jadhav's sheep, plane (start 250 m out)", "B": "Di Lorenzo's scenario in metres (24 m box)",
              "C": "Hybrid A (80 m box)", "D": "Hybrid B's sheep"}
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4), sharey=True)
    for ax, f in zip(axes, fams):
        for name, rows in res["rows"].items():
            if not name.startswith(f + "."):
                continue
            t = np.array(rows[0]["t"])
            occ = np.array([r["occupancy"] for r in rows])
            line, = ax.plot(t, occ.mean(0), label=name.split(": ", 1)[-1], lw=1.4)
            ax.fill_between(t, occ.min(0), occ.max(0), color=line.get_color(), alpha=0.15, lw=0)
        ax.axhline(res["thresholds"]["success"], color="k", ls=":", lw=1)
        ax.axvspan(res["t_end"] * (1 - res["thresholds"]["final_frac"]), res["t_end"], color="0.9", zorder=-1)
        ax.set_title(titles.get(f, f), fontsize=10), ax.set_xlabel("t (s)"), ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("fraction of sheep inside the goal circle")
    fig.suptitle("Occupancy over time (mean of 4 seeds, band = range); dotted: success threshold; grey: final window", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "hybrid_scorecard.png", dpi=120)


if __name__ == "__main__":
    main()
