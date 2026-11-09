"""Diagnostics behind three things the bridge page shows (docs/bridge_simulation.md §4.6).

    .venv/bin/python experiments/bridge_diagnostics.py                   # writes results/bridge_diagnostics.json
    .venv/bin/python experiments/bridge_diagnostics.py --only multi_dog  # one section, merged into the JSON

A  first-order followers (preset dilorenzo_metres): how the per-step jitter compares with the drift the
   design supplies, and how the herders move (uncapped flow, feedforward).
B  Hybrid A (preset hybrid_jadhav_sheep_dilorenzo_dogs): how often the heading flock is split, where
   the split-off groups are, and whether the herders' push, as the heading sheep feel it (unit vectors
   from herders within r_dog), points to the goal, against the design's own summed push.
C  Hybrid B (preset hybrid_first_order_sheep_strombom_dogs): how far the Morse cohesion reaches against
   the Strömbom collect threshold l_sep, and what the dog does while a sheep is cut off.
E  Several Strömbom dogs over the published sheep (preset jadhav_reference, start 60 m out, 600 s): the
   arc of the flock the dogs cover, how often the flock crosses the goal, and what it ends as.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import bridge as B  # noqa: E402

OUT = ROOT / "results"


def first_order_look(seed=1):
    F, H, T, D, kw = B.preset("dilorenzo_metres")
    Tr = T.resolve(D)
    r = B.simulate_bridge(F, H, T, D, seed=seed, init=kw["init"], n_steps=kw["n_steps"])
    dt = F.dt
    step_speed = np.linalg.norm(D.wrap(np.diff(r.pos_s, axis=0)), axis=-1).mean(1) / dt  # (S−1,)
    dog_speed = np.linalg.norm(D.wrap(np.diff(r.pos_d, axis=0)), axis=-1) / dt  # (S−1, M)
    t = r.t[1:]
    out = {
        "dt": dt, "D": F.D,
        "follower_step_speed_mean": float(step_speed.mean()),
        "follower_step_speed_brownian": float(np.sqrt(np.pi * F.D / dt)),  # E|N(0, 2D dt I₂)|/dt
        "max_design_drift": float(F.D * Tr.k * 2 * np.pi / D.size),  # max |D ∇ln ρ̄| for the von Mises goal
        "push_per_herder_at_contact": float(F.a_dog / (F.n + H.n)),
        "herder_speed_first_0p2s_max": float(dog_speed[t <= 0.2 + 1e-9].max()),
        "herder_speed_after_10s_max": float(dog_speed[t >= 10].max()),
        "occupancy": {f"{tt:.0f}": float(r.metrics["occupancy"][np.searchsorted(r.t, tt)]) for tt in (0, 50, 100, 150, 200)},
    }
    out["drift_equals_diffusion_after_s"] = 4 * F.D / out["max_design_drift"] ** 2  # v t = sqrt(4 D t)
    return out


def hybrid_a(seeds=range(4)):
    F, H, T, D, kw = B.preset("hybrid_jadhav_sheep_dilorenzo_dogs")
    F, Tr, L = F.resolved(), T.resolve(D), D.size
    g = np.linspace(-L / 2, L / 2, 121)[:-1] + L / 240
    P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    to_goal = -D.wrap(P - np.array(Tr.centre))
    rg = np.linalg.norm(to_goal, axis=1)
    out = {"seeds": [], "bands": {}}
    bands = ((10, 20), (20, 30), (30, 40), (40, 57))
    acc = {b: {"unseen": [], "jad_goalward": [], "jad_cos": [], "des_goalward": [], "des_cos": []} for b in bands}
    for seed in seeds:
        r = B.simulate_bridge(F, H, T, D, seed=seed, init=kw["init"], n_steps=kw["n_steps"])
        ngroups, sec_dist, sec_size, sec_in = [], [], [], []
        for k in range(0, len(r.t), 10):
            x = r.pos_s[k]
            d = np.linalg.norm(D.wrap(x[:, None] - x[None]), axis=-1)
            _, lab = connected_components(d < 5.0, directed=False)
            sizes = np.bincount(lab)
            big = [c for c in np.argsort(-sizes) if sizes[c] >= 3]
            ngroups.append(len(big))
            for c in big[1:]:
                Gc = D.centroid(x[lab == c])
                dist = float(np.linalg.norm(D.wrap(Gc - np.array(Tr.centre))))
                sec_dist.append(dist), sec_size.append(int(sizes[c])), sec_in.append(dist < Tr.r_goal)
        dogs = r.pos_d[-1]
        z = D.wrap(P[:, None] - dogs[None])
        dist = np.linalg.norm(z, axis=-1)
        unit = np.where((dist <= F.r_dog)[..., None], z / dist[..., None], 0).sum(1)  # the heading sheep's dog term
        des = ((F.a_dog * np.exp(-dist / F.L_dog) / dist)[..., None] * z).sum(1)  # the design's push (minimum image)
        seen = np.linalg.norm(unit, axis=1) > 0

        def cos(v):
            n = np.linalg.norm(v, axis=1)
            return np.where(n > 1e-9, (v * to_goal).sum(1) / np.maximum(n * rg, 1e-12), np.nan)

        cj, cd = cos(unit), cos(des)
        for b in bands:
            m = (rg >= b[0]) & (rg < b[1])
            acc[b]["unseen"].append(float(np.mean(~seen[m])))
            acc[b]["jad_goalward"].append(float(np.nanmean(cj[m & seen] > 0)))
            acc[b]["jad_cos"].append(float(np.nanmean(cj[m & seen])))
            acc[b]["des_goalward"].append(float(np.nanmean(cd[m] > 0)))
            acc[b]["des_cos"].append(float(np.nanmean(cd[m])))
        cover = float(np.mean(dist.min(1) <= F.r_dog))
        out["seeds"].append(dict(seed=seed, final_occupancy=r.final()["occupancy"], groups_mean=float(np.mean(ngroups)),
                                 split_fraction=float(np.mean(np.array(ngroups) > 1)),
                                 secondary_size=float(np.mean(sec_size)) if sec_size else None,
                                 secondary_distance=float(np.mean(sec_dist)) if sec_dist else None,
                                 secondary_inside_goal=float(np.mean(sec_in)) if sec_in else None,
                                 notice_cover_of_box=cover))
        s = out["seeds"][-1]
        print(f"A seed {seed}: occ {s['final_occupancy']:.2f}, groups {s['groups_mean']:.1f}, split {s['split_fraction']*100:.0f}%, "
              f"secondary {s['secondary_size']:.1f} sheep at {s['secondary_distance']:.1f} m, inside r* {s['secondary_inside_goal']*100:.0f}%, "
              f"notice cover {cover*100:.0f}%", flush=True)
    for b in bands:
        out["bands"][f"{b[0]}-{b[1]}"] = {k: float(np.mean(v)) for k, v in acc[b].items()}
        print(f"A {b[0]}–{b[1]} m: " + ", ".join(f"{k} {np.mean(v):+.2f}" for k, v in acc[b].items()), flush=True)
    return out


def hybrid_b(seeds=range(4), cut=12.0):
    F, H, T, D, kw = B.preset("hybrid_first_order_sheep_strombom_dogs")
    F = F.resolved()
    Hr = H.resolve(F, D, T.resolve(D))
    out = {"l_sep": Hr.l_sep, "dog_speed": Hr.v, "cut_off_m": cut,
           "flock_pull_m_s": {str(d): float(-F.n * F.g(d)) for d in (5, 10, 15, 20, 27)},
           "dog_push_m_s": {str(d): float(F.a_dog * np.exp(-d / F.L_dog)) for d in (5, 10, 15, 20, 27)},
           "sheep_outruns_dog_within_m": float(F.L_dog * np.log(F.a_dog / Hr.v)), "seeds": []}
    for seed in seeds:
        r = B.simulate_bridge(F, H, T, D, seed=seed, init=kw["init"], box_length=kw["box_length"], n_steps=kw["n_steps"])
        x, y, mode = r.pos_s, r.pos_d[:, 0], r.mode[:, 0]
        S = len(r.t)
        out_steps, collect, drive, between, far, switches, prev, n_out = 0, 0, 0, 0, 0, 0, None, []
        farthest = np.zeros(S)
        for k in range(S):
            G = x[k].mean(0)
            dd = np.linalg.norm(x[k] - G, axis=1)
            farthest[k] = dd.max()
            i = int(dd.argmax())
            if prev is not None and i != prev and mode[k] == B.COLLECT:
                switches += 1
            prev = i
            n_out.append(int((dd > cut).sum()))
            if n_out[-1]:
                out_steps += 1
                collect += mode[k] == B.COLLECT
                drive += mode[k] == B.DRIVE
                u = (x[k][i] - G) / dd[i]
                v = y[k] - G
                if 0 < v @ u < dd[i] and abs(u[0] * v[1] - u[1] * v[0]) < 6:
                    between += 1
                far += np.linalg.norm(y[k] - x[k][i]) > 3 * F.L_dog
        drv = mode == B.DRIVE
        n = max(out_steps, 1)
        s = dict(seed=seed, cut_off_fraction=out_steps / S, max_out=int(max(n_out)), collecting=collect / n, driving=drive / n,
                 dog_between=between / n, dog_far=far / n, target_switches=switches,
                 farthest_while_driving_median=float(np.median(farthest[drv])) if drv.any() else None,
                 farthest_while_driving_p90=float(np.percentile(farthest[drv], 90)) if drv.any() else None)
        out["seeds"].append(s)
        print(f"C seed {seed}: cut off {s['cut_off_fraction']*100:.0f}% (max {s['max_out']}), collecting {s['collecting']*100:.0f}%, "
              f"driving {s['driving']*100:.0f}%, between {s['dog_between']*100:.0f}%, far {s['dog_far']*100:.0f}%, switches {switches}, "
              f"farthest while driving median {s['farthest_while_driving_median']:.1f} m", flush=True)
    return out


def multi_dog(counts=(1, 2, 4, 6), seeds=range(5)):
    """The multi-dog Strömbom fan: each dog runs the single-dog rule, drive aims rotated by spread_angle."""
    F, H, T, D, _ = B.preset("jadhav_reference")
    c = np.asarray(T.centre, float)
    out = {"spread_angle_deg": float(np.degrees(H.spread_angle)), "counts": {}}
    for M in counts:
        spans, far_side, crossings, occ, coh = [], [], [], [], []
        for seed in seeds:
            r = B.simulate_bridge(F, replace(H, n=M), T, D, seed=seed, init="jadhav", box_length=60.0, n_steps=600)
            Gs = r.pos_s.mean(1)
            u0 = (Gs[0] - c) / np.linalg.norm(Gs[0] - c)  # initial approach axis, goal at 0
            proj = (Gs - c) @ u0
            sgn = np.sign(proj[proj != 0])
            crossings.append(int(np.sum(sgn[1:] != sgn[:-1])))
            for k in range(len(r.t)):
                if M > 1:
                    a = np.sort(np.mod(np.arctan2(*(r.pos_d[k] - Gs[k]).T[::-1]), 2 * np.pi))
                    spans.append(2 * np.pi - np.diff(np.r_[a, a[0] + 2 * np.pi]).max())  # arc the dogs occupy
                g = Gs[k] - c
                if np.linalg.norm(g) > 1e-9:
                    far_side.append(np.mean((r.pos_d[k] - Gs[k]) @ (g / np.linalg.norm(g)) > 0))
            tm = r.tail_mean(0.5)
            occ.append(tm["occupancy"]), coh.append(tm["cohesion"])
        row = dict(arc_median_deg=float(np.degrees(np.median(spans))) if spans else 0.0,
                   arc_p90_deg=float(np.degrees(np.percentile(spans, 90))) if spans else 0.0,
                   dogs_on_far_side=float(np.mean(far_side)), goal_crossings=float(np.mean(crossings)),
                   occupancy_last_half=float(np.mean(occ)), cohesion_last_half=float(np.mean(coh)))
        out["counts"][str(M)] = row
        print(f"E {M} dogs: arc {row['arc_median_deg']:.0f}° (90%: {row['arc_p90_deg']:.0f}°), far side {row['dogs_on_far_side']*100:.0f}%, "
              f"crossings {row['goal_crossings']:.1f}, occupancy {row['occupancy_last_half']:.2f}, cohesion {row['cohesion_last_half']:.1f} m", flush=True)
    return out


SECTIONS = {"first_order_look": first_order_look, "hybrid_a": hybrid_a, "hybrid_b": hybrid_b, "multi_dog": multi_dog}

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SECTIONS), help="run one section and merge it into the existing JSON")
    a = ap.parse_args()
    path = OUT / "bridge_diagnostics.json"
    res = json.loads(path.read_text()) if (a.only and path.exists()) else {}
    for name, fn in SECTIONS.items():
        if a.only and name != a.only:
            continue
        res[name] = fn()
        if name == "first_order_look":
            print("first-order:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=float))
