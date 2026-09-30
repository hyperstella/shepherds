"""Phase 2 check: the published model with the real dog replayed.

    .venv/bin/python experiments/replay_check.py
Writes results/replay_check.json and results/figures/replay_overlay.png.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, protocol as PR, scenario as SC  # noqa: E402
from shepherd.model import Params  # noqa: E402

OUT = ROOT / "results"
C_DATA, C_MODEL, C_DOG, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def pooled(segs):
    summ = [PR.summaries(s) for s in segs]
    agg = PR.aggregate(summ, [s.duration for s in segs])
    return {k: float(np.nanmean(v)) for k, v in agg.items()}


def main():
    drives = data.load_drives(data.load_sessions())
    real = PR.segments_from_drives(drives, 1.0)
    out = {"data_full_drives": pooled(real)}
    variants = {
        "M0_published": Params(),
        "M0_sheep_speed_1.29": Params(v_sheep=1.29),
    }
    sims = {}
    for name, p in variants.items():
        sims[name] = SC.replay_segments(drives, p, n_runs=20, seed=0, dt=1.0)
        out[name + "_full"] = pooled(sims[name])
    # multiple shooting: restart from the real state every 15 s, drop 3 s burn-in
    for dt in (1.0, 0.2):
        out[f"data_windows_dt{dt:g}"] = pooled(SC.window_segments(drives, dt=dt))
        for name, p in variants.items():
            out[f"{name}_windows_dt{dt:g}"] = pooled(SC.replay_windows(drives, p, n_runs=20, seed=0, dt=dt))
    (OUT / "replay_check.json").write_text(json.dumps(out, indent=2))

    keys = ["nnd_med", "C_med", "E_med", "P_med", "dBD_med", "yRD_med", "xD_iqr", "rear_frac", "slope_4a"]
    print(f"{'':34s}" + "".join(f"{k:>10s}" for k in keys))
    for name, v in out.items():
        print(f"{name:34s}" + "".join(f"{v[k]:10.2f}" for k in keys))

    # overlay: real vs one simulated run for three drives
    pick = [i for i, d in enumerate(drives) if (d.session, d.index + 1) in {(4, 5), (8, 5), (9, 2)}]
    fig, axs = plt.subplots(1, len(pick), figsize=(6 * len(pick), 5))
    sim = sims["M0_sheep_speed_1.29"]
    for ax, i in zip(np.atleast_1d(axs), pick):
        r, s = real[i], sim[i]
        for j in range(14):
            ax.plot(r.pos[j, 0], r.pos[j, 1], color=C_DATA, lw=0.7, alpha=0.6)
            ax.plot(s.pos[0, j, 0], s.pos[0, j, 1], color=C_MODEL, lw=0.7, alpha=0.6)
        ax.plot(r.pos[14, 0], r.pos[14, 1], color=C_DOG, lw=2)
        ax.plot([], [], color=C_DATA, label="Real sheep"), ax.plot([], [], color=C_MODEL, label="Model sheep")
        ax.plot([], [], color=C_DOG, lw=2, label="Real dog (replayed)")
        ax.set_aspect("equal"), ax.set_title(f"{r.label}: {r.duration:.0f} s", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.suptitle("Published model (sheep speed 1.29 m/s) driven by the real dog's path", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "replay_overlay.png", dpi=110)


if __name__ == "__main__":
    main()
