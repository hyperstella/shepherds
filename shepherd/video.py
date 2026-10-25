"""Clean herding tracks from video: detect, track, clean, then draft phase labels.

    detections = detect("drive.mp4")          # YOLO + ByteTrack, needs requirements-video.txt
    tracks = clean_tracks(detections)         # metres, 10 Hz, gaps interpolated, smoothed
    flock = flock_series(tracks)              # flock centre/velocity, dog, shepherd
    segs = label_phases(flock)                # same rules as phases.label_session

Detection uses a COCO-pretrained model, whose classes include sheep, dog and
person (taken to be the shepherd). Only detect() needs the model; the other
steps run on a detections table, so they can be re-run without it.

Positions are converted to metres with a homography (pixel -> ground plane),
a fixed pixels-per-metre scale for top-down footage, or, failing both, a
scale guessed from the median sheep box (a sheep is about SHEEP_LENGTH_M long).
The guess is only sensible for roughly top-down video.
"""

import csv
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import data, phases as PH

COCO = {"sheep": 18, "dog": 16, "shepherd": 0}  # shepherd = COCO "person"
KINDS = tuple(COCO)
# Model class names that map onto each kind; other classes are ignored. Matching by name lets
# single-class sheep detectors (e.g. ones trained on drone footage) drop in for the COCO model.
NAMES = {"sheep": "sheep", "ewe": "sheep", "lamb": "sheep", "dog": "dog", "person": "shepherd", "human": "shepherd"}


def class_kinds(names):
    """{class id: kind} for a model's names dict."""
    return {int(i): NAMES[n.lower()] for i, n in names.items() if n.lower() in NAMES}
SHEEP_LENGTH_M = 1.2
DET_FIELDS = ("frame", "track", "kind", "x", "y", "w", "h", "conf")


@dataclass
class Detections:
    """Tracked boxes in pixels; (x, y) is the box centre, image y pointing down."""

    fps: float
    width: int
    height: int
    frame: np.ndarray
    track: np.ndarray
    kind: np.ndarray  # str
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray
    conf: np.ndarray
    n_frames: int = 0  # video length; 0 means up to the last detection

    @property
    def t(self):
        return self.frame / self.fps


@dataclass
class Tracks:
    """Cleaned tracks on a regular grid; NaN where a track is not seen."""

    t: np.ndarray  # (T,) video seconds
    ids: list
    kinds: list
    xy: np.ndarray  # (n, 2, T) metres, y up
    norm: np.ndarray  # (n, 2, T) position in the frame, 0..1, y down
    px_per_m: float | None = None
    notes: list = field(default_factory=list)


@dataclass
class Flock:
    t: np.ndarray
    centre: np.ndarray  # (2, T)
    vel: np.ndarray  # (2, T) mean of the sheep's own velocities
    cohesion: np.ndarray  # (T,)
    n_sheep: np.ndarray  # (T,) sheep seen
    dog: np.ndarray  # (2, T), NaN when not seen
    shepherd: np.ndarray  # (2, T), NaN when not seen


# --- detection -------------------------------------------------------------


def detect(video, model="icaerus-n", conf=None, imgsz=None, stride=1, tracker="bytetrack.yaml", device=None, progress=False):
    """Run a YOLO detector with ByteTrack over a video file. Needs `ultralytics`.

    model is a .pt path, an ultralytics name, or a shortcut from weights.SHORTCUTS
    ("icaerus-n", the default); conf and imgsz default to the shortcut's settings.
    """
    import cv2
    from ultralytics import YOLO

    from . import weights

    sz, c = weights.defaults(model)
    imgsz, conf = imgsz or sz or 1280, conf or c or 0.25
    model = weights.resolve(model)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    yolo = YOLO(model)
    kind_of = class_kinds(yolo.names)
    if not kind_of:
        raise ValueError(f"{model} has none of the classes {sorted(set(NAMES))}: {yolo.names}")
    rows = []
    results = yolo.track(
        source=str(video), stream=True, persist=True, conf=conf, imgsz=imgsz, vid_stride=stride,
        classes=list(kind_of), tracker=tracker, device=device, verbose=False,
    )
    for k, r in enumerate(results):
        f = k * stride
        if progress and k % 100 == 0:
            print(f"  frame {f}/{n_frames}", flush=True)
        b = r.boxes
        if b is None or b.id is None:
            continue
        for (x, y, w, h), c, i, p in zip(b.xywh.cpu().numpy(), b.cls.cpu().numpy(), b.id.cpu().numpy(), b.conf.cpu().numpy()):
            rows.append((f, int(i), kind_of[int(c)], x, y, w, h, p))
    return _make_detections(fps, width, height, rows, n_frames)


def _make_detections(fps, width, height, rows, n_frames=0):
    cols = list(zip(*rows)) if rows else [()] * len(DET_FIELDS)
    arr = {k: np.array(v, dtype=object if k == "kind" else float) for k, v in zip(DET_FIELDS, cols)}
    arr["frame"] = arr["frame"].astype(int)
    arr["track"] = arr["track"].astype(int)
    arr["kind"] = arr["kind"].astype(str)
    return Detections(float(fps), int(width), int(height), **arr, n_frames=int(n_frames))


def write_detections(det, path):
    with open(path, "w", newline="") as fh:
        fh.write(f"# fps={det.fps} width={det.width} height={det.height} frames={det.n_frames}\n")
        w = csv.writer(fh)
        w.writerow(DET_FIELDS)
        for row in zip(det.frame, det.track, det.kind, *(np.round(getattr(det, k), 2) for k in "xywh"), np.round(det.conf, 3)):
            w.writerow(row)


def read_detections(path):
    with open(path, newline="") as fh:
        meta = dict(kv.split("=") for kv in fh.readline().lstrip("# ").split())
        rows = [(int(r["frame"]), int(r["track"]), r["kind"], *(float(r[k]) for k in ("x", "y", "w", "h", "conf")))
                for r in csv.DictReader(fh)]
    return _make_detections(float(meta["fps"]), int(meta["width"]), int(meta["height"]), rows, int(meta.get("frames", 0)))


# --- cleaning --------------------------------------------------------------


def estimate_px_per_m(det, sheep_length_m=SHEEP_LENGTH_M):
    """Scale from the median long side of sheep boxes. Rough, and only for top-down video."""
    sheep = det.kind == "sheep"
    if not sheep.any():
        return None
    return float(np.median(np.maximum(det.w[sheep], det.h[sheep]))) / sheep_length_m


def to_metres(x, y, height, px_per_m=None, homography=None):
    """Pixel centres to ground metres with y up."""
    if homography is not None:
        H = np.asarray(homography, float)
        p = H @ np.stack([x, y, np.ones_like(x)])
        return p[0] / p[2], p[1] / p[2]
    return x / px_per_m, (height - y) / px_per_m


def _nan_smooth(a, w):
    """Moving average over the last axis that ignores NaNs and keeps them where they were."""
    if w <= 1:
        return a
    from scipy.ndimage import uniform_filter1d

    ok = np.isfinite(a)
    s = uniform_filter1d(np.where(ok, a, 0.0), w, axis=-1, mode="nearest")
    n = uniform_filter1d(ok.astype(float), w, axis=-1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[~ok] = np.nan
    return out


def _resample(t_obs, v_obs, grid, max_gap):
    """Linear interpolation onto grid; NaN outside the track and across gaps longer than max_gap."""
    out = np.interp(grid, t_obs, v_obs, left=np.nan, right=np.nan)
    k = np.clip(np.searchsorted(t_obs, grid), 1, len(t_obs) - 1)
    gap = t_obs[k] - t_obs[k - 1]
    exact = np.isin(grid, t_obs)
    out[(gap > max_gap) & ~exact] = np.nan
    return out


def clean_tracks(det, px_per_m=None, homography=None, dt=data.DT, min_track_s=1.0, max_gap_s=1.0, smooth_s=0.5, min_conf=0.0):
    """Regular-grid tracks in metres from raw detections.

    Tracks seen for less than min_track_s are dropped (mostly false positives
    and ID-switch fragments). Each track takes the class it was most often
    detected as.
    """
    notes = []
    keep = det.conf >= min_conf
    if homography is None and px_per_m is None:
        px_per_m = estimate_px_per_m(det)
        if px_per_m is None:
            raise ValueError("no sheep detected, so no scale; pass px_per_m or a homography")
        notes.append(f"scale guessed from sheep boxes: {px_per_m:.1f} px/m (top-down footage only)")
    t_all = det.t
    end = max(t_all.max() if len(t_all) else 0.0, (det.n_frames - 1) / det.fps)
    grid = np.round(np.arange(0.0, end + dt / 2, dt), 6)
    w = max(1, round(smooth_s / dt))

    ids, kinds, xy, norm = [], [], [], []
    for i in np.unique(det.track[keep]):
        m = keep & (det.track == i)
        t = t_all[m]
        if t.max() - t.min() < min_track_s:
            continue
        order = np.argsort(t, kind="stable")
        t, x, y = t[order], det.x[m][order], det.y[m][order]
        t, first = np.unique(t, return_index=True)
        x, y = x[first], y[first]
        if len(t) < 2:
            continue
        values, counts = np.unique(det.kind[m], return_counts=True)
        ids.append(int(i))
        kinds.append(str(values[np.argmax(counts)]))
        mx, my = to_metres(x, y, det.height, px_per_m, homography)
        xy.append(np.stack([_resample(t, mx, grid, max_gap_s), _resample(t, my, grid, max_gap_s)]))
        norm.append(np.stack([_resample(t, x / det.width, grid, max_gap_s), _resample(t, y / det.height, grid, max_gap_s)]))
    shape = (0, 2, len(grid))
    xy = _nan_smooth(np.array(xy), w) if xy else np.empty(shape)
    norm = _nan_smooth(np.array(norm), w) if norm else np.empty(shape)
    return Tracks(grid, ids, kinds, xy, norm, px_per_m, notes)


def stitch(xy):
    """One identity from several (n, 2, T) tracks: at each time take the track nearest the last pick.

    Detectors split one dog into several track IDs and also pick up stray
    people; this follows the most-seen track and hops to the closest
    alternative when it disappears.
    """
    n, _, T = xy.shape
    out = np.full((2, T), np.nan)
    if n == 0:
        return out
    seen = np.isfinite(xy[:, 0])
    cur, last = int(np.argmax(seen.sum(1))), None
    for k in range(T):
        if not seen[:, k].any():
            continue
        if not seen[cur, k]:
            cand = np.flatnonzero(seen[:, k])
            ref = last if last is not None else xy[cand, :, k].mean(0)
            cur = int(cand[np.argmin(np.linalg.norm(xy[cand, :, k] - ref, axis=1))])
        out[:, k] = last = xy[cur, :, k]
    return out


def flock_series(tracks):
    """Flock centre, velocity and cohesion from the sheep, plus single dog and shepherd tracks."""
    kinds = np.array(tracks.kinds)
    dt = tracks.t[1] - tracks.t[0] if len(tracks.t) > 1 else data.DT
    sheep = tracks.xy[kinds == "sheep"] if len(kinds) else tracks.xy
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(sheep, axis=0) if len(sheep) else np.full((2, len(tracks.t)), np.nan)
        # Mean of each sheep's own velocity, so sheep entering or leaving the frame don't move the flock.
        vel = np.nanmean(np.gradient(sheep, dt, axis=-1), axis=0) if len(sheep) and len(tracks.t) > 1 else np.full_like(centre, np.nan)
        cohesion = np.nanmean(np.linalg.norm(sheep - centre, axis=1), axis=0) if len(sheep) else np.full(len(tracks.t), np.nan)
    n = np.isfinite(sheep[:, 0]).sum(0) if len(sheep) else np.zeros(len(tracks.t), int)
    return Flock(tracks.t, centre, vel, cohesion, n,
                 stitch(tracks.xy[kinds == "dog"]) if len(kinds) else np.full_like(centre, np.nan),
                 stitch(tracks.xy[kinds == "shepherd"]) if len(kinds) else np.full_like(centre, np.nan))


# --- phases ----------------------------------------------------------------


def _fill(a):
    """Carry the last seen value forward along the last axis (and the first one backward)."""
    a = np.asarray(a, float)
    ok = np.isfinite(a).all(0) if a.ndim > 1 else np.isfinite(a)
    if not ok.any():
        return a.copy()
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))
    idx[: np.argmax(ok)] = np.argmax(ok)
    return a[..., idx]


def _median(v):
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


@dataclass
class VideoSegment:
    start_s: float  # video seconds
    end_s: float
    label: str
    flock_speed: float
    dog_dist: float
    psi_bd: float
    heading_change: float
    cohesion: float
    shepherd_dist: float
    n_sheep: float  # median sheep in view
    dog_seen: float  # fraction of the segment with the dog in view


def label_phases(flock, rules=PH.Rules()):
    """Per-sample labels and segments with phases.Rules.

    A dog out of shot counts as far away ("away"); with no sheep in view the
    flock is treated as still ("idle"). dog_seen and n_sheep say how much of
    each segment rests on what was actually seen.
    """
    dog_seen = np.isfinite(flock.dog[0])
    flock_seen = np.isfinite(flock.centre[0])
    centre = _fill(flock.centre)
    vel = np.where(np.isfinite(flock.vel), flock.vel, 0.0)
    dog = np.where(dog_seen, flock.dog, np.inf)
    shepherd = _fill(flock.shepherd)
    cohesion = np.nan_to_num(_fill(flock.cohesion))
    with np.errstate(invalid="ignore"):
        f = PH.flock_features(vel, centre, dog, shepherd, cohesion, rules)
    f["dog_dist"][~dog_seen] = np.inf
    f["flock_speed"][~flock_seen] = 0.0
    lab = PH.clean(PH.classify(f, rules), max(1, round(rules.min_segment_s / data.DT)))
    segs = []
    starts, ends = PH._runs(lab)
    for a, b in zip(starts, ends):
        segs.append(VideoSegment(round(float(flock.t[a]), 2), round(float(flock.t[b - 1] + data.DT), 2), PH.LABELS[lab[a]],
                                 *(_median(f[k][a:b]) for k in PH.FEATURES),
                                 float(np.median(flock.n_sheep[a:b])), float(dog_seen[a:b].mean())))
    return lab, segs


# --- output ----------------------------------------------------------------


def _r(v, n=3):
    return "" if not np.isfinite(v) else round(float(v), n)


def write_tracks(tracks, path, offset=0.0):
    """Long-format tracks: one row per track per sample it was seen."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "video_t_s", "track", "kind", "x_m", "y_m", "x_norm", "y_norm"])
        for k, t in enumerate(tracks.t):
            for i, (tid, kind) in enumerate(zip(tracks.ids, tracks.kinds)):
                if np.isfinite(tracks.xy[i, 0, k]):
                    w.writerow([_r(t + offset, 2), _r(t, 2), tid, kind, *(_r(v) for v in tracks.xy[i, :, k]),
                                *(_r(v, 5) for v in tracks.norm[i, :, k])])


def write_flock(flock, path, offset=0.0):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "video_t_s", "flock_x", "flock_y", "flock_vx", "flock_vy", "cohesion", "n_sheep",
                    "dog_x", "dog_y", "shepherd_x", "shepherd_y"])
        for k, t in enumerate(flock.t):
            w.writerow([_r(t + offset, 2), _r(t, 2), *(_r(v) for v in flock.centre[:, k]), *(_r(v) for v in flock.vel[:, k]),
                        _r(flock.cohesion[k]), int(flock.n_sheep[k]), *(_r(v) for v in flock.dog[:, k]),
                        *(_r(v) for v in flock.shepherd[:, k])])


def write_phases(segs, path, session="", offset=0.0, video=""):
    """draft_phases.csv columns (tracking time) plus the video times; opens in viz/labeler.html."""
    cols = ["session", "start_s", "end_s", "label", *PH.FEATURES, "n_sheep", "dog_seen", "reviewed_label", "notes",
            "video", "video_start_s", "video_end_s"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for g in segs:
            w.writerow([session, _r(g.start_s + offset, 1), _r(g.end_s + offset, 1), g.label,
                        *(_r(getattr(g, k)) for k in PH.FEATURES), _r(g.n_sheep, 1), _r(g.dog_seen, 2), "", "",
                        video, _r(g.start_s), _r(g.end_s)])
