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
    rows = ["session,start_s,end_s,label,reviewed_label,notes"]
    rows += [f"{g.session},{g.start_s},{g.end_s},{g.label},," for g in segs]
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
