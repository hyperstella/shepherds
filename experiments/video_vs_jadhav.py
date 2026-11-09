"""Compare flock-dog observables measured from drone video with the Jadhav et al. UWB drives.

    .venv/bin/python experiments/video_vs_jadhav.py TRACKS_DIR
TRACKS_DIR holds <video>_tracks.csv from experiments/label_video.py (sheep only, metres) for the
videos listed in results/icaerus_dog_tracks.csv, whose dog positions were found by a white-blob
tracker and checked by eye. Each frame with the dog and at least MIN_SHEEP moving sheep in view is
measured the same way as each 0.1 s sample of the UWB drives (data.load_drives). Writes
results/video_vs_jadhav.json.

Caveats: positions are in the camera frame (the drone hovers), metres come from sheep box size
(about +-20%), the flock is often partly out of shot, and these clips are sheep following a
shepherd through a gate with a livestock dog nearby, not a dog driving the flock. Scale-free
observables (polarization, angles, distance ratios) are the fairest comparison.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import data  # noqa: E402

MIN_SHEEP = 8
MOVING = 0.2  # m/s; slower sheep have no meaningful heading
DT = data.DT


def frame_stats(sheep, sheep_v, dog, dog_v):
    """Observables for one frame. sheep, sheep_v: (n, 2); dog, dog_v: (2,)."""
    sp = np.linalg.norm(sheep_v, axis=1)
    mov = sp > MOVING
    if mov.sum() < MIN_SHEEP:
        return None
    b, vb = sheep.mean(0), sheep_v.mean(0)
    head = sheep_v[mov] / sp[mov, None]
    d = np.linalg.norm(sheep[:, None] - sheep[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nnd = float(np.median(d.min(1)))
    r_bd = dog - b
    psi = np.degrees(np.arctan2(vb[0] * r_bd[1] - vb[1] * r_bd[0], vb @ r_bd))
    # flock frame: y along v_B, x to the right
    u = vb / max(np.linalg.norm(vb), 1e-9)
    along = (sheep - b) @ u
    across = (sheep - b) @ np.array([u[1], -u[0]])
    dog_along = r_bd @ u
    return {
        "P": float(np.linalg.norm(head.mean(0))),
        "nnd": nnd,
        "C_over_nnd": float(np.linalg.norm(sheep - b, axis=1).mean() / nnd),
        "E": float(np.ptp(along) / max(np.ptp(across), 1e-9)),
        "speed_sheep": float(np.median(sp[mov])),
        "speed_bary": float(np.linalg.norm(vb)),
        "speed_dog": float(np.linalg.norm(dog_v)),
        "dBD": float(np.linalg.norm(r_bd)),
        "dBD_over_nnd": float(np.linalg.norm(r_bd) / nnd),
        "abs_psiBD": float(abs(psi)),
        "dog_behind_rear": bool(dog_along < along.min()),
        "n": int(len(sheep)),
        # per-sheep distance to dog and speed, for the response-range estimate
        "d_dog": np.linalg.norm(sheep - dog, axis=1).tolist(),
        "sp": sp.tolist(),
    }


def summarise(frames):
    keys = ["P", "nnd", "C_over_nnd", "E", "speed_sheep", "speed_bary", "speed_dog", "dBD", "dBD_over_nnd", "abs_psiBD"]
    out = {"frames": len(frames), "seconds": round(len(frames) * DT, 1)}
    for k in keys:
        v = np.array([f[k] for f in frames])
        out[k] = {"median": round(float(np.median(v)), 3), "p25": round(float(np.percentile(v, 25)), 3),
                  "p75": round(float(np.percentile(v, 75)), 3)}
    out["frac_dog_behind_rear"] = round(float(np.mean([f["dog_behind_rear"] for f in frames])), 3)
    out["frac_psiBD_over_120"] = round(float(np.mean([f["abs_psiBD"] > 120 for f in frames])), 3)
    # sheep speed against distance to the dog, normalised by each frame's median sheep speed
    d = np.concatenate([f["d_dog"] for f in frames]); s = np.concatenate([np.array(f["sp"]) / max(f["speed_sheep"], 1e-9) for f in frames])
    edges = [0, 2, 4, 6, 8, 10, 15, 25]
    out["relative_speed_by_dog_distance"] = {
        f"{a}-{b} m": (round(float(np.median(s[(d >= a) & (d < b)])), 3) if ((d >= a) & (d < b)).sum() >= 20 else None)
        for a, b in zip(edges[:-1], edges[1:])}
    return out


def uwb_frames():
    frames = []
    for dr in data.load_drives():
        N = data.N_SHEEP
        for t in range(dr.pos.shape[-1]):
            f = frame_stats(dr.pos[:N, :, t], dr.vel[:N, :, t], dr.pos[data.DOG, :, t], dr.vel[data.DOG, :, t])
            if f:
                frames.append(f)
    return frames


def load_dog(path):
    dogs = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            dogs[r["video"]].append(r)
    return dogs


def video_frames(tracks_csv, dog_rows):
    """Per-frame stats for one video. Dog pixels are converted with the tracks' own scale."""
    rows = list(csv.DictReader(open(tracks_csv)))
    sheep = defaultdict(dict)  # t index -> track -> (x_m, y_m, x_norm, y_norm)
    for r in rows:
        if r["kind"] == "sheep":
            sheep[round(float(r["video_t_s"]) / DT)][r["track"]] = (float(r["x_m"]), float(r["y_m"]), float(r["x_norm"]), float(r["y_norm"]))
    # metres per normalised unit from the tracks themselves (x_m = x_norm * W / px_per_m)
    pairs = np.array([(v[0], v[2], v[1], v[3]) for t in sheep.values() for v in t.values()])
    mx = np.polyfit(pairs[:, 1], pairs[:, 0], 1)  # x_m = a * x_norm + c
    my = np.polyfit(pairs[:, 3], pairs[:, 2], 1)  # y_m = a * y_norm + c (a < 0: y up)
    W, H, fps = float(dog_rows[0]["W"]), float(dog_rows[0]["H"]), float(dog_rows[0]["fps"])
    dog = {}
    for r in dog_rows:
        k = round(int(r["frame"]) / fps / DT)
        xn, yn = (float(r["x_px"]) + float(r["w_px"]) / 2) / W, (float(r["y_px"]) + float(r["h_px"]) / 2) / H
        dog[k] = np.array([np.polyval(mx, xn), np.polyval(my, yn)])
    ks = sorted(dog)
    out = []
    for k in ks:
        if k - 1 not in dog and k + 1 not in dog:
            continue
        ids = [i for i in sheep.get(k, {}) if i in sheep.get(k - 1, {}) and i in sheep.get(k + 1, {})]
        if len(ids) < MIN_SHEEP:
            continue
        pos = np.array([sheep[k][i][:2] for i in ids])
        vel = np.array([(np.array(sheep[k + 1][i][:2]) - np.array(sheep[k - 1][i][:2])) / (2 * DT) for i in ids])
        a, b = (k - 1 if k - 1 in dog else k), (k + 1 if k + 1 in dog else k)
        dv = (dog[b] - dog[a]) / ((b - a) * DT)
        f = frame_stats(pos, vel, dog[k], dv)
        if f:
            out.append(f)
    return out


def main(argv=None):
    tracks_dir = Path((argv or sys.argv[1:])[0])
    dogs = load_dog(ROOT / "results" / "icaerus_dog_tracks.csv")
    vid, per_video = [], {}
    for v, rows in sorted(dogs.items()):
        tr = tracks_dir / f"{v}_tracks.csv"
        if not tr.exists():
            print(f"skip {v}: no {tr.name}")
            continue
        fr = video_frames(tr, rows)
        per_video[v] = len(fr)
        vid += fr
        print(f"{v}: {len(fr)} usable frames")
    res = {"video": summarise(vid), "uwb_jadhav": summarise(uwb_frames()), "frames_per_video": per_video,
           "min_sheep": MIN_SHEEP, "moving_threshold_m_s": MOVING}
    (ROOT / "results" / "video_vs_jadhav.json").write_text(json.dumps(res, indent=2))
    for k in ["P", "E", "C_over_nnd", "dBD_over_nnd", "abs_psiBD", "nnd", "speed_sheep", "speed_bary", "speed_dog", "dBD"]:
        a, b = res["video"][k], res["uwb_jadhav"][k]
        print(f"{k:13s} video {a['median']:7.2f} [{a['p25']:.2f}-{a['p75']:.2f}]   UWB {b['median']:7.2f} [{b['p25']:.2f}-{b['p75']:.2f}]")
    for k in ["frac_dog_behind_rear", "frac_psiBD_over_120", "seconds"]:
        print(f"{k:22s} video {res['video'][k]}   UWB {res['uwb_jadhav'][k]}")
    print("relative sheep speed by distance to dog")
    for k in res["video"]["relative_speed_by_dog_distance"]:
        print(f"   {k:8s} video {res['video']['relative_speed_by_dog_distance'][k]}   UWB {res['uwb_jadhav']['relative_speed_by_dog_distance'][k]}")


if __name__ == "__main__":
    main()
