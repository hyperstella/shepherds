"""Load the Jadhav et al. (2024) UWB tracking data.

Port of `pre_ana.m`, `pre_ini_ana.m` and `drivs_dat.m` from
github.com/tee-lab/collective-responses-of-flocking-sheep-to-herding-dog.

Array layout follows the MATLAB code: positions and velocities are
(n_individuals, 2, T); headings are (n_individuals, T). Individuals 0-13 are
sheep, 14 is the dog and 15 is the shepherd.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "upstream"
RAW_FILE = DATA_DIR / "sheepR.dat"

DT = 0.1  # s between samples
N_IND = 16
N_SHEEP = 14
DOG = 14
SHEPHERD = 15

# Hand-labelled drive windows (seconds) from drivs_dat.m, keyed by session
# number (1-9, the order of trials in sheepR.dat). Session 7 has no drives.
DRIVES = {
    1: [(140, 160), (453, 465)],
    2: [(151, 177), (196, 232), (258, 290), (700, 730)],
    3: [(163, 193)],
    4: [(23, 35), (50, 77), (130, 170), (191, 198), (210, 240), (272, 286), (293, 298)],
    5: [(0.1, 24), (48, 68), (110, 128), (207, 227), (283, 288)],
    6: [(81, 89)],
    8: [(0.1, 33), (55, 90), (170, 220), (332, 387), (438, 530), (585, 593)],
    9: [(0.1, 32), (99, 210), (303, 324)],
}


@dataclass
class Session:
    """One trial: pos/vel are (16, 2, T), phi is (16, T)."""

    number: int
    raw_trial_id: int
    pos: np.ndarray
    vel: np.ndarray
    phi: np.ndarray

    def drive_slices(self):
        """Index slices for each drive, matching MATLAB's inclusive st:et."""
        return [slice(round(s / DT) - 1, round(e / DT)) for s, e in DRIVES.get(self.number, [])]


@dataclass
class Drive:
    session: int
    index: int  # 0-based drive number within the session
    pos: np.ndarray
    vel: np.ndarray
    phi: np.ndarray


def _kinematics(pos):
    """Backward-difference velocity and heading in [0, 2pi); t=0 is zero, as in pre_ini_ana.m."""
    vel = np.zeros_like(pos)
    vel[:, :, 1:] = np.diff(pos, axis=2) / DT
    phi = np.arctan2(vel[:, 1, :], vel[:, 0, :])
    phi[phi < 0] += 2 * np.pi
    phi[:, 0] = 0.0
    return vel, phi


def load_sessions(path=RAW_FILE):
    """Parse sheepR.dat into a list of 9 Sessions."""
    raw = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :5]
    raw.columns = ["trial", "id", "t", "x", "y"]
    sessions = []
    for number, (trial_id, g) in enumerate(raw.groupby("trial", sort=True), start=1):
        counts = g.groupby("id").size()
        if counts.nunique() != 1 or len(counts) != N_IND:
            raise ValueError(f"trial {trial_id}: unequal samples per individual {counts.to_dict()}")
        T = counts.iloc[0]
        pos = np.empty((N_IND, 2, T))
        for i in range(N_IND):
            rows = g[g["id"] == i + 1]
            pos[i, 0] = rows["x"].to_numpy()
            pos[i, 1] = rows["y"].to_numpy()
        vel, phi = _kinematics(pos)
        sessions.append(Session(number, int(trial_id), pos, vel, phi))
    return sessions


def load_drives(sessions=None):
    """All 29 hand-labelled drives as separate segments."""
    sessions = sessions or load_sessions()
    drives = []
    for s in sessions:
        for k, sl in enumerate(s.drive_slices()):
            drives.append(Drive(s.number, k, s.pos[:, :, sl], s.vel[:, :, sl], s.phi[:, sl]))
    return drives


def concat_drives_by_session(sessions=None):
    """Drives concatenated per session, reproducing drives_data.mat (8 entries).

    The figure_2fj / figure_3 / figure_4 scripts operate on these concatenated
    arrays, so windowed statistics can straddle drive boundaries; we keep that
    behaviour for a faithful reproduction.
    """
    sessions = sessions or load_sessions()
    out = []
    for s in sessions:
        sls = s.drive_slices()
        if not sls:
            continue
        pos = np.concatenate([s.pos[:, :, sl] for sl in sls], axis=2)
        vel = np.concatenate([s.vel[:, :, sl] for sl in sls], axis=2)
        phi = np.concatenate([s.phi[:, sl] for sl in sls], axis=1)
        out.append(Drive(s.number, -1, pos, vel, phi))
    return out
