import numpy as np

from shepherd import feasibility as FE

A, L, D = 3.57, 3.0, 0.25
f = lambda r: A * np.exp(-r / L)
g = lambda r: np.where(r <= 3.6, 0.02 * np.tanh(r - 1.5), 0.0)  # repel below 1.5 m, attract to 3.6 m


def test_confinement_drift_has_stiffness_D_over_s2():
    P, w, _, _, s = FE.goal_points(14, 2.5)
    assert np.isclose(FE.centroid_stiffness(P, w, -D * P / s ** 2, s), D / s ** 2, rtol=1e-2)  # grid quadrature


def test_pair_forces_cancel_in_the_centroid():
    # the goal-weighted mean of the cohesion drift is zero: cohesion can shape the flock, never move it
    P, w, n_star, step, s = FE.goal_points(14, 2.5)
    c = FE.cohesion_drift(g, P, n_star, step)
    assert np.abs((n_star[:, None] * c).sum(0)).max() < 1e-12 * np.abs(n_star[:, None] * c).sum()
    assert FE.centroid_stiffness(P, w, c, s) > 0  # but it does pull the shape in


def test_cohesion_takes_its_stiffness_off_the_dogs():
    P, w, n_star, step, s = FE.goal_points(14, 2.5)
    _, _, v = FE.required_drift_local(g, D, 14, 2.5)
    k_coh = FE.centroid_stiffness(P, w, FE.cohesion_drift(g, P, n_star, step), s)
    assert np.isclose(FE.centroid_stiffness(P, w, v, s), D / s ** 2 - k_coh, rtol=1e-2)


def test_single_ring_mass_scales_with_the_drift_asked():
    radii = (6.0, 8.0)
    one = FE.single_rings(f, None, D, 14, 2.5, radii)
    two = FE.single_rings(f, None, 2 * D, 14, 2.5, radii)
    for a, b in zip(one, two):
        assert np.isclose(b["mass"], 2 * a["mass"]) and np.isclose(a["drift_error"], b["drift_error"])
    assert one[1]["drift_error"] < 0.05  # a ring near the NNLS radius matches the bowl
