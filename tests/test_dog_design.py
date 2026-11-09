import numpy as np

from shepherd import dog_design as DD

A, L, D = 3.57, 3.0, 0.25
f = lambda r: A * np.exp(-r / L)


def test_direction_match_pushes_the_dog_to_the_coverage_limit():
    out = DD.direction_match(14, 2.5, 1.0, D, None, r_dog=12.0)
    assert np.isclose(out["standoff"], out["d_cov"])  # the fan keeps narrowing, so the limit binds
    assert out["d_unconstrained"] > out["standoff"]
    assert out["J_lat"].min() >= out["J_best"] - 1e-9  # a lateral position never beats the axis


def test_single_dog_first_order_sits_behind_on_the_axis():
    one = DD.single_dog_first_order(f, None, D, 14, 2.5, 1.0, r_dog=12.0)
    assert one["standoff"] > 0 and abs(one["lateral"]) < 1e-9
    assert 0 < one["error_best"] < 1.0  # better than no dog, worse than exact


def test_translating_arc_is_non_negative_and_one_sided_for_a_wide_flock():
    arc = DD.translating_arc(f, None, D, 30, 2.5 * np.sqrt(30 / 14), 1.0, r_dog=12.0)
    assert (arc["counts"] >= 0).all() and arc["mass"] > 0
    assert arc["behind_mass_frac"] > 0.5 and arc["residual"] < 0.5


def test_candidates_fall_back_when_nothing_covers_the_flock():
    assert len(DD.candidate_dogs(3.66, 6.73)) > 0
    assert len(DD.candidate_dogs(2.5, 12.0)) < len(DD.candidate_dogs(2.5, None))
