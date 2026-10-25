"""Draft phase labels for all 76 tracked minutes, checked against the 29 hand-labelled drives.

    .venv/bin/python experiments/label_phases.py
Writes results/labels/draft_phases.csv (one row per segment, for review),
results/label_phases.json and results/figures/label_phases.png.
"""

import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shepherd import data, phases as PH  # noqa: E402

OUT = ROOT / "results"
COLORS = {"drive": "#2a78d6", "turn": "#eb6834", "flank": "#1baf7a", "idle": "#c9c7c1", "away": "#8a6fd1"}


def main():
    sessions = data.load_sessions()
    labels, segs = {}, []
    for s in sessions:
        labels[s.number], ss = PH.label_session(s)
        segs += ss

    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    with open(OUT / "labels" / "draft_phases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, [f.name for f in fields(PH.Segment)] + ["reviewed_label", "notes"])
        w.writeheader()
        for g in segs:
            row = {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(g).items()}
            w.writerow(row | {"reviewed_label": "", "notes": ""})

    # Agreement with the hand-labelled drives.
    all_lab = np.concatenate([labels[s.number] for s in sessions])
    hand = np.concatenate([PH.hand_drive_mask(s) for s in sessions])
    names = np.array(PH.LABELS)[all_lab]
    in_hand = {k: float((names[hand] == k).mean()) for k in PH.LABELS}
    minutes = {k: float((names == k).sum() * data.DT / 60) for k in PH.LABELS}
    outside = {k: float(((names == k) & ~hand).sum() * data.DT / 60) for k in PH.LABELS}
    per_drive = []
    for s in sessions:
        for k, sl in enumerate(s.drive_slices()):
            lab = np.array(PH.LABELS)[labels[s.number][sl]]
            per_drive.append({"session": s.number, "drive": k, "drive_frac": float((lab == "drive").mean()),
                              "drive_or_turn_frac": float(np.isin(lab, ["drive", "turn"]).mean())})
    drive_frames = names == "drive"
    out = {
        "rules": asdict(PH.Rules()),
        "minutes_total": float(len(names) * data.DT / 60),
        "minutes_by_label": minutes,
        "minutes_outside_hand_drives": outside,
        "hand_drive_frames_by_label": in_hand,
        "drive_recall": in_hand["drive"],
        "drive_precision": float(hand[drive_frames].mean()),
        "hand_drives_mostly_drive": int(sum(d["drive_frac"] >= 0.5 for d in per_drive)),
        "hand_drives_mostly_drive_or_turn": int(sum(d["drive_or_turn_frac"] >= 0.5 for d in per_drive)),
        "n_hand_drives": len(per_drive),
        "per_drive": per_drive,
        "n_segments": len(segs),
        "n_segments_by_label": {k: sum(g.label == k for g in segs) for k in PH.LABELS},
    }
    (OUT / "label_phases.json").write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(len(sessions), 1, figsize=(11, 1.0 * len(sessions) + 1), sharex=True)
    for ax, s in zip(axes, sessions):
        for g in (g for g in segs if g.track == s.number):
            ax.axvspan(g.start_s / 60, g.end_s / 60, 0.0, 0.62, color=COLORS[g.label], lw=0)
        for a, b in data.DRIVES.get(s.number, []):
            ax.axvspan(a / 60, b / 60, 0.72, 1.0, color="#222222", lw=0)
        ax.set_yticks([])
        ax.set_ylabel(f"S{s.number}", rotation=0, ha="right", va="center")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("time (min)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#222222"))
    fig.legend(handles, list(COLORS) + ["hand-labelled drive"], loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "figures" / "label_phases.png", dpi=130)

    print(json.dumps({k: v for k, v in out.items() if k not in ("per_drive", "rules")}, indent=2))


if __name__ == "__main__":
    main()
