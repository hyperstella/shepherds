"""Build viz/phase-replay.html: particle replays of every draft phase segment,
for the real tracks and for the simulated runs of experiments/label_simulations.py.

    .venv/bin/python viz/build_phases.py
"""

import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy.spatial import ConvexHull  # noqa: E402

from shepherd import data, metrics as M, phases as PH  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
import label_simulations as LS  # noqa: E402

HERE = Path(__file__).resolve().parent


def field_layout(sessions):
    """What the tracks say about the pasture: none of it was mapped, all of it is inferred.

    hull: convex hull of every tracked position (all animals, all sessions).
    ends: where the flock turned around, as the median over sessions of the flock
    centre's furthest point at each end of the drive axis (x). Session 6 is
    skipped because it never reaches either end.
    """
    pts = np.concatenate([s.pos.transpose(0, 2, 1).reshape(-1, 2) for s in sessions])
    hull = pts[ConvexHull(pts).vertices]
    west, east = [], []
    for s in sessions:
        if s.number == 6:
            continue
        b = M.barycenter(s.pos, data.N_SHEEP)
        west.append(b[:, b[0].argmin()])
        east.append(b[:, b[0].argmax()])
    ends = [np.median(west, axis=0), np.median(east, axis=0)]
    return {
        "hull": np.round(hull, 2).tolist(),
        "ends": [{"x": round(float(e[0]), 1), "y": round(float(e[1]), 1), "name": n} for e, n in zip(ends, ("W", "E"))],
        "ends_spread": [round(float(np.ptp([w[0] for w in west])), 1), round(float(np.ptp([e[0] for e in east])), 1)],
    }


def _pack(pos, scale):
    """(n, 2, T) metres -> (T, 16, 2) int16 in units of 1/scale m; missing rows stay zero."""
    out = np.zeros((pos.shape[-1], data.N_IND, 2), "<i2")
    q = np.round(pos.transpose(2, 0, 1) * scale)
    assert np.abs(q).max() < 32767, "positions overflow int16; lower the scale"
    out[:, : pos.shape[0]] = q
    return out.ravel()


def _segments(key, segs, real):
    out = []
    for g in segs:
        d = {k: (None if np.isnan(v) else round(v, 2)) if isinstance(v, float) else v for k, v in asdict(g).items()}
        d["track"] = key
        d["candidate"] = bool(real and g.label == "drive" and g.hand_drive_overlap < 0.1 and g.end_s - g.start_s >= 20)
        out.append(d)
    return out


def main():
    sessions = data.load_sessions()
    tracks, chunks, segs, offset = [], [], [], 0

    def add(track, key, group, name, scale, extra):
        nonlocal offset
        F = track.pos.shape[-1]
        chunks.append(_pack(track.pos, scale))
        tracks.append({"key": key, "group": group, "name": name, "dt": track.dt, "frames": F, "offset": offset,
                       "scale": scale, "nSheep": track.n_sheep, "dog": track.dog, "shepherd": track.shepherd, **extra})
        offset += F
        _, ss = PH.label_track(track)
        segs.extend(_segments(key, ss, group == "real"))

    for s in sessions:
        add(PH.Track.from_session(s), f"S{s.number}", "real", f"Session {s.number}", 100,
            {"drives": [list(w) for w in data.DRIVES.get(s.number, [])]})
    for group, (label, (params, per)) in zip(("published", "calibrated"), LS.models().items()):
        pos, trips, log = LS.back_and_forth(params, per)
        for r in range(LS.N_RUNS):
            tr = PH.Track(f"{group}{r + 1}", pos[r], params.dt, data.N_SHEEP, data.N_SHEEP)
            # target switches as (time s, x, y), one entry per change
            sw = [[0.0, *log[r, 0].tolist()]] + [[c * LS.CHUNK, *log[r, c].tolist()]
                                                 for c in range(1, log.shape[1]) if (log[r, c] != log[r, c - 1]).any()]
            add(tr, f"{group}{r + 1}", group, f"{label} · run {r + 1}", 10, {"targets": sw, "trips": int(trips[r])})

    payload = {"nInd": data.N_IND, "tracks": tracks, "segments": segs, "rules": asdict(PH.Rules()),
               "field": field_layout(sessions),
               "task": {"A": LS.A.tolist(), "B": LS.B.tolist(), "arrive": LS.ARRIVE, "minutes": LS.T_TOTAL / 60}}
    pos = base64.b64encode(np.concatenate(chunks).tobytes()).decode()
    html = (HERE / "phases_app.html").read_text()
    html = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))).replace("__POS__", pos)
    out = HERE / "phase-replay.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(tracks)} tracks, {len(segs)} segments)")


if __name__ == "__main__":
    main()
