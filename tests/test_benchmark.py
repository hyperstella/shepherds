import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import benchmark_auth_sheep as B  # noqa: E402


def test_match_counts_flock_hits_people_and_misses():
    gts = [(0, 0, 10, 10, "sheep"), (20, 0, 30, 10, "goat"), (40, 0, 50, 10, "human"), (60, 0, 70, 10, "sheep")]
    preds = [(0, 0, 10, 10, 0.9), (21, 0, 31, 10, 0.8), (40, 0, 50, 10, 0.7), (0, 1, 10, 11, 0.6), (90, 0, 99, 9, 0.5)]
    res, missed = B.match(preds, gts)
    assert [(h, c) for _, h, c in res] == [(True, "sheep"), (True, "goat"), (False, "human"), (False, None), (False, None)]
    assert missed == ["sheep"]  # the duplicate box does not count twice


def test_ap_perfect_and_half():
    assert B.ap([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert B.ap([(0.9, True), (0.8, False)], 2) == pytest.approx(0.5)
    assert B.ap([], 3) == 0.0


def test_iou():
    a = np.array([[0, 0, 10, 10]]); b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert B.iou(a, b).tolist() == [[pytest.approx(1 / 3), 0.0]]
