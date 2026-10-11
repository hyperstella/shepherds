import numpy as np
import pytest

from shepherd import data, phases as PH, video as V

PX_PER_M = 20.0
FPS = 25.0


def fake_detections(session, seconds=240, seed=0):
    """A top-down camera's view of real UWB tracks: box jitter, dropouts, ID switches, a stray person."""
    rng = np.random.default_rng(seed)
    T = round(seconds / data.DT)
    pos = session.pos[:, :, :T]
    t10 = np.arange(T) * data.DT
    frames = np.arange(round(seconds * FPS))
    t = frames / FPS
    lo = np.nanmin(pos, axis=(0, 2)) - 5
    width = int((np.nanmax(pos[..., 0, :]) - lo[0] + 5) * PX_PER_M)
    height = int((np.nanmax(pos[..., 1, :]) - lo[1] + 5) * PX_PER_M)
    kinds = ["sheep"] * data.N_SHEEP + ["dog", "shepherd"]
    rows, next_id = [], 1000
    for i, kind in enumerate(kinds):
        x = (np.interp(t, t10, pos[i, 0]) - lo[0]) * PX_PER_M
        y = height - (np.interp(t, t10, pos[i, 1]) - lo[1]) * PX_PER_M
        seen = rng.random(len(t)) > 0.15  # missed detections
        tid = i
        for f in frames[seen]:
            if rng.random() < 0.002:  # tracker hands out a new ID
                tid, next_id = next_id, next_id + 1
            size = (0.8 if kind == "dog" else 1.2 if kind == "sheep" else 0.6) * PX_PER_M
            rows.append((f, tid, kind, x[f] + rng.normal(0, 3), y[f] + rng.normal(0, 3), size, size * 0.6, 0.8))
    for f in frames[(frames > 100) & (frames < 115)]:  # a short-lived false positive
        rows.append((f, 9999, "dog", 10.0, 10.0, 15.0, 10.0, 0.3))
    for f in frames[: len(frames) // 3]:  # a spectator standing still at the edge
        rows.append((f, 5000, "shepherd", width - 5.0, 5.0, 12.0, 30.0, 0.6))
    return V._make_detections(FPS, width, height, sorted(rows)), T


@pytest.fixture(scope="module")
def session():
    return data.load_sessions()[0]


def test_scale_guess_from_sheep_boxes(session):
    det, _ = fake_detections(session, 30)
    assert V.estimate_px_per_m(det) == pytest.approx(PX_PER_M, rel=0.01)


def test_clean_tracks_recover_positions(session):
    det, T = fake_detections(session, 60)
    tr = V.clean_tracks(det, px_per_m=PX_PER_M)
    assert 9999 not in tr.ids  # under min_track_s
    assert tr.xy.shape[1:] == (2, len(tr.t))
    fl = V.flock_series(tr)
    true = session.pos[: data.N_SHEEP, :, :T].mean(0)
    k = min(T, len(fl.t))
    err = (fl.centre[:, :k] - fl.centre[:, :k].mean(1, keepdims=True)) - (true[:, :k] - true[:, :k].mean(1, keepdims=True))
    assert np.nanmax(np.abs(err)) < 0.5  # metres, after removing the camera offset
    dog_err = np.linalg.norm((fl.dog[:, :k] - fl.centre[:, :k]) - (session.pos[data.DOG, :, :k] - true[:, :k]), axis=0)
    assert np.nanmedian(dog_err) < 0.3


def test_phases_from_video_match_tracking(session):
    det, T = fake_detections(session, 240)
    fl = V.flock_series(V.clean_tracks(det, px_per_m=PX_PER_M))
    lab, segs = V.label_phases(fl)
    ref, _ = PH.label_session(session)
    k = min(T, len(lab))
    assert (lab[:k] == ref[:k]).mean() > 0.85
    assert segs[0].start_s == 0.0 and all(a.end_s == pytest.approx(b.start_s) for a, b in zip(segs, segs[1:]))


def test_missing_dog_reads_as_away():
    t = np.arange(200) * data.DT
    centre = np.stack([t * 1.0, np.zeros_like(t)])
    dog = np.full((2, len(t)), np.nan)
    fl = V.Flock(t, centre, np.stack([np.ones_like(t), np.zeros_like(t)]), np.ones_like(t), np.full(len(t), 10), dog, centre - [[3], [0]])
    lab, segs = V.label_phases(fl)
    assert set(lab) == {PH.LABELS.index("away")} and segs[0].dog_seen == 0.0


def test_csv_round_trip(session, tmp_path):
    det, _ = fake_detections(session, 20)
    V.write_detections(det, tmp_path / "d.csv")
    back = V.read_detections(tmp_path / "d.csv")
    assert back.fps == det.fps and len(back.frame) == len(det.frame)
    assert np.allclose(back.x, det.x, atol=0.01) and (back.kind == det.kind).all()
    tr = V.clean_tracks(back, px_per_m=PX_PER_M)
    fl = V.flock_series(tr)
    _, segs = V.label_phases(fl)
    V.write_tracks(tr, tmp_path / "t.csv")
    V.write_flock(fl, tmp_path / "f.csv")
    V.write_phases(segs, tmp_path / "p.csv", session=session.number, offset=0.0, video="x.mp4")
    lab = PH.read_reviewed(tmp_path / "p.csv", session)
    assert (lab[: round(segs[-1].end_s / data.DT)] >= 0).all()
