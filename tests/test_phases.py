import numpy as np
import pytest

from shepherd import data, phases as PH


@pytest.fixture(scope="module")
def sessions():
    return data.load_sessions()


def test_clean_absorbs_short_runs():
    lab = np.array([0] * 10 + [1] * 2 + [0] * 10 + [2] * 3 + [3] * 20)
    out = PH.clean(lab, 5)
    starts, ends = PH._runs(out)
    assert (ends - starts).min() >= 5
    assert out[10] == 0 and out[22] == 0  # ties and longer neighbours win


def test_segments_tile_each_session(sessions):
    for s in sessions:
        lab, segs = PH.label_session(s)
        assert len(lab) == s.pos.shape[-1]
        assert segs[0].start_s == 0.0
        assert all(a.end_s == b.start_s for a, b in zip(segs, segs[1:]))
        assert all(g.end_s - g.start_s >= PH.Rules().min_segment_s - 1e-9 for g in segs)


def test_hand_drives_are_drafted_as_drive_or_turn(sessions):
    names, hand = [], []
    for s in sessions:
        lab, _ = PH.label_session(s)
        names.append(np.array(PH.LABELS)[lab])
        hand.append(PH.hand_drive_mask(s))
    names, hand = np.concatenate(names), np.concatenate(hand)
    assert (names[hand] == "drive").mean() > 0.8
    assert np.isin(names[hand], ["drive", "turn"]).mean() > 0.95


def test_read_reviewed_matches_draft_and_prefers_reviewed_label(sessions, tmp_path):
    s = sessions[0]
    lab, segs = PH.label_session(s)
    path = tmp_path / "reviewed.csv"
    rows = ["track,start_s,end_s,label,reviewed_label,notes"]
    rows += [f"{g.track},{g.start_s},{g.end_s},{g.label},," for g in segs]
    rows.append(f"{s.number + 1},0.0,5.0,drive,,")  # other session, ignored
    path.write_text("\n".join(rows) + "\n")
    assert np.array_equal(PH.read_reviewed(path, s), lab)

    rows[1] = rows[1].rstrip(",") + ",flank,"
    rows.append(f"{s.number},1.0,2.0,unknown,,")  # not in LABELS, ignored
    path.write_text("\n".join(rows) + "\n")
    out = PH.read_reviewed(path, s)
    first = slice(0, round(segs[0].end_s / data.DT))
    assert (out[first] == PH.LABELS.index("flank")).all()
    assert np.array_equal(out[first.stop :], lab[first.stop :])


def test_track_matches_session_labels(sessions):
    s = sessions[0]
    lab_a, segs_a = PH.label_session(s)
    lab_b, _ = PH.label_track(PH.Track(1, s.pos, data.DT, data.N_SHEEP, data.DOG, data.SHEPHERD))
    np.testing.assert_array_equal(lab_a, lab_b)
    assert segs_a[0].track == 1


def test_generic_track_without_shepherd():
    # 5 sheep walking east at 1 m/s, dog 4 m behind, sampled at 1 Hz; then the flock stops.
    T = 60
    t = np.arange(T, dtype=float)
    x = np.minimum(t, 30.0)
    pos = np.zeros((6, 2, T))
    pos[:5, 0] = x + np.arange(5)[:, None] * 0.5
    pos[:5, 1] = np.arange(5)[:, None] * 0.5
    pos[5, 0], pos[5, 1] = x - 4.0, 1.0
    _, segs = PH.label_track(PH.Track("toy", pos, 1.0, 5, 5))
    names = [g.label for g in segs]
    assert names == ["drive", "idle"]
    assert segs[0].end_s == pytest.approx(31.0, abs=2)
    assert np.isnan(segs[0].shepherd_dist) and np.isnan(segs[0].hand_drive_overlap)


def test_away_from_edge_uses_nearest_sheep():
    f = {"psi_bd": np.array([170.0]), "heading_change": np.array([0.0]), "flock_speed": np.array([1.0]),
         "dog_dist": np.array([20.0]), "dog_nearest_sheep": np.array([5.0])}
    assert PH.LABELS[PH.classify(f)[0]] == "away"
    assert PH.LABELS[PH.classify(f, PH.Rules(away_from="edge"))[0]] == "drive"
