"""Rule-based draft labels for herding phases in the tracked sessions.

Only 14 of the 76 tracked minutes are hand-labelled (the 29 drives in
data.DRIVES). This module proposes a phase for every sample so the rest can be
reviewed by hand. Labels, in priority order:

    away   the dog is far from the flock (outrun, recall, not working it)
    idle   the flock is (nearly) standing still
    turn   the flock's heading changes by a lot within a few seconds
    drive  the flock moves and the dog is behind it
    flank  the flock moves and the dog is beside or ahead of it
           (covering, collecting or repositioning)

These are geometric proxies. The shepherd's commands were not recorded, so
the labels say where the dog was, not what it was told to do.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import data, metrics as M

LABELS = ("drive", "turn", "flank", "idle", "away")
FEATURES = ("flock_speed", "dog_dist", "psi_bd", "heading_change", "cohesion", "shepherd_dist")


@dataclass(frozen=True)
class Rules:
    smooth_s: float = 2.0  # moving-average window for velocities
    idle_speed: float = 0.25  # m/s, flock speed below this is idle
    away_dist: float = 15.0  # m, dog-barycenter distance above this is away
    behind_deg: float = 120.0  # |psi_BD| at or above this means the dog is behind
    turn_window_s: float = 10.0  # window over which the heading change is measured
    turn_deg: float = 120.0  # heading change over turn_window_s that counts as a turn
    min_segment_s: float = 5.0  # shorter runs are absorbed into their neighbours


def features(pos, vel, n_sheep=data.N_SHEEP, dog=data.DOG, shepherd=data.SHEPHERD, rules=Rules()):
    """Per-sample features from (n, 2, T) arrays. All angles in degrees."""
    bary = M.barycenter(pos, n_sheep)
    return flock_features(vel[:n_sheep].mean(0), bary, pos[dog], pos[shepherd], M.cohesion(pos, n_sheep), rules)


def flock_features(flock_vel, bary, dog_pos, shepherd_pos, cohesion, rules=Rules()):
    """features() from flock-level (2, T) series, e.g. cleaned video tracks."""
    w = max(1, round(rules.smooth_s / data.DT))
    vb = uniform_filter1d(flock_vel, w, axis=-1)
    r_bd = dog_pos - bary
    heading = np.unwrap(np.arctan2(vb[1], vb[0]))
    half = max(1, round(rules.turn_window_s / data.DT / 2))
    padded = np.pad(heading, half, mode="edge")
    dturn = np.abs(padded[2 * half :] - padded[: -2 * half])
    return {
        "flock_speed": np.linalg.norm(vb, axis=0),
        "dog_dist": np.linalg.norm(r_bd, axis=0),
        "psi_bd": np.degrees(np.abs(M.signed_angle(vb.T, r_bd.T))),
        "heading_change": np.degrees(dturn),
        "cohesion": cohesion,
        "shepherd_dist": np.linalg.norm(shepherd_pos - bary, axis=0),
    }


def classify(f, rules=Rules()):
    """Per-sample label indices into LABELS, before segment cleanup."""
    lab = np.where(f["psi_bd"] >= rules.behind_deg, LABELS.index("drive"), LABELS.index("flank"))
    lab[f["heading_change"] >= rules.turn_deg] = LABELS.index("turn")
    lab[f["flock_speed"] < rules.idle_speed] = LABELS.index("idle")
    lab[f["dog_dist"] > rules.away_dist] = LABELS.index("away")
    return lab


def _runs(lab):
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(lab)]
    return starts, ends


def clean(lab, min_len):
    """Absorb runs shorter than min_len samples into the longer neighbour, shortest first."""
    lab = lab.copy()
    while True:
        starts, ends = _runs(lab)
        lengths = ends - starts
        if len(lengths) == 1 or lengths.min() >= min_len:
            return lab
        k = int(np.argmin(lengths))
        if k == 0:
            nb = 1
        elif k == len(lengths) - 1:
            nb = k - 1
        else:
            nb = k - 1 if lengths[k - 1] >= lengths[k + 1] else k + 1
        lab[starts[k] : ends[k]] = lab[starts[nb]]


@dataclass
class Segment:
    session: int
    start_s: float
    end_s: float
    label: str
    flock_speed: float
    dog_dist: float
    psi_bd: float
    heading_change: float
    cohesion: float
    shepherd_dist: float  # m; the hand-labelled drives keep the shepherd further back
    hand_drive_overlap: float  # fraction of the segment inside a hand-labelled drive


def label_session(session, rules=Rules()):
    """Cleaned per-sample labels and the segment list for one Session."""
    f = features(session.pos, session.vel, rules=rules)
    lab = clean(classify(f, rules), max(1, round(rules.min_segment_s / data.DT)))
    hand = hand_drive_mask(session)
    segs = []
    for a, b in zip(*_runs(lab)):
        segs.append(
            Segment(
                session.number,
                round(a * data.DT, 1),
                round(b * data.DT, 1),
                LABELS[lab[a]],
                *(float(np.median(f[k][a:b])) for k in FEATURES),
                float(hand[a:b].mean()),
            )
        )
    return lab, segs


def hand_drive_mask(session):
    mask = np.zeros(session.pos.shape[-1], bool)
    for sl in session.drive_slices():
        mask[sl] = True
    return mask


def read_reviewed(path, session):
    """Per-sample label indices from a reviewed CSV (draft_phases.csv or viz/labeler.html export).

    Uses `reviewed_label` where filled and `label` otherwise. Samples no row
    covers are -1. Times are in tracking seconds, as in draft_phases.csv.
    """
    import csv

    lab = np.full(session.pos.shape[-1], -1)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["session"]) != session.number:
                continue
            name = (row.get("reviewed_label") or row["label"]).strip()
            if name not in LABELS:
                continue
            a = max(0, round(float(row["start_s"]) / data.DT))
            b = min(len(lab), round(float(row["end_s"]) / data.DT))
            lab[a:b] = LABELS.index(name)
    return lab
