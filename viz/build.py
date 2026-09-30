"""Build viz/sheep-and-collie.html: real drives + live model in one page.

    .venv/bin/python viz/build.py
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import data, metrics as M  # noqa: E402
import reproduce  # noqa: E402

HERE = Path(__file__).resolve().parent
N = data.N_SHEEP

# Histogram bins shared by the data PDFs and the in-page simulation.
BINS = {
    "C": (0.0, 5.0, 0.1),
    "P": (0.0, 1.0, 0.025),
    "E": (0.0, 4.0, 0.1),
    "yRD": (-4.0, 10.0, 0.25),
    "xD": (-8.0, 8.0, 0.4),
    "psiBD": (0.0, 180.0, 5.0),
}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def density(x, lo, hi, w):
    """Density over [lo, hi); values outside still count in the total, as in the page."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    edges = np.linspace(lo, hi, int(round((hi - lo) / w)) + 1)
    h, _ = np.histogram(x, bins=edges)
    return (h / (x.size * w)).round(5).tolist()


def main():
    sessions = data.load_sessions()
    D = reproduce.analyse_data(sessions)
    nets = {(d.session, d.index): edges for d, edges in D["nets"]}

    drives, pos_chunks, obs_chunks = [], [], []
    for d in D["drives"]:
        F = d.pos.shape[2]
        pos_cm = np.round(d.pos[: N + 2].transpose(2, 0, 1) * 100).astype("<i2")  # (F, 16, 2)
        e, xd, yrd = M.frame_observables(d.pos, d.vel, N)
        obs = np.stack([
            np.linalg.norm(d.vel[:N].mean(0), axis=0),  # flock speed
            M.speeds(d.vel)[data.DOG],  # dog speed
            M.cohesion(d.pos, N),
            M.polarization_from_phi(d.phi, N),
            e, M.dog_barycenter_distance(d.pos, N), yrd, xd,
        ], axis=1).astype("<f4")  # (F, 8)
        edges = nets[(d.session, d.index)]
        indeg = np.zeros(N + 1, int)
        for leader, _, _ in edges:
            indeg[leader] += 1
        start = data.DRIVES[d.session][d.index][0]
        drives.append({
            "session": d.session, "index": d.index + 1, "frames": F, "start": start,
            "edges": [[int(a), int(b), int(l)] for a, b, l in edges],
            "indegree": indeg.tolist(),
        })
        pos_chunks.append(pos_cm.ravel())
        obs_chunks.append(obs.ravel())

    stats = {
        "C": D["C"], "P": D["P"], "E": D["E"], "yRD": D["yRD"], "xD": D["xD"],
        "psiBD": np.degrees(np.abs(D["psi_BD"])),
    }
    pdfs = {k: {"lo": lo, "hi": hi, "w": w, "density": density(stats[k], lo, hi, w)} for k, (lo, hi, w) in BINS.items()}
    means = {k: float(np.nanmean(v)) for k, v in stats.items()}
    means["xDabs"] = float(np.nanmean(np.abs(stats["xD"])))
    means["psiRear"] = float((stats["psiBD"] > 150).mean())

    payload = {
        "dt": data.DT, "nSheep": N, "agents": N + 2,
        "drives": drives,
        "pos": b64(np.concatenate(pos_chunks)),
        "obs": b64(np.concatenate(obs_chunks)),
        "obsKeys": ["vB", "vD", "C", "P", "E", "dBD", "yRD", "xD"],
        "pdfs": pdfs, "means": means,
    }
    page = (HERE / "app.html").read_text()
    page = page.replace("/*__DATA__*/", "window.SHEEP_DATA = " + json.dumps(payload, separators=(",", ":")) + ";")
    page = page.replace("/*__SIM__*/", (HERE / "sim.js").read_text())
    out = HERE / "sheep-and-collie.html"
    out.write_text(page)
    # Local preview copy: the publish step wraps the page in a UTF-8 skeleton, a plain file server does not.
    (HERE / "preview.html").write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n' + page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(drives)} drives")


if __name__ == "__main__":
    main()
