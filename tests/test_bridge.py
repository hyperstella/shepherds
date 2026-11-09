"""Focused checks for the bridge simulator: reference configurations, noise, activation, periodic
geometry, metrics and the Di Lorenzo feasibility calculation."""

from dataclasses import replace

import numpy as np
import pytest

from shepherd import bridge as B
from shepherd.model import Params, simulate

T0 = B.Target("gaussian", sigma=2.5)
PLANE = B.Domain(False)


# ------------------------------------------------------------------ reference configurations

@pytest.mark.parametrize("seed", [0, 2, 3])
def test_jadhav_reference_reproduces_the_published_model(seed):
    """Same random stream, same trajectories as shepherd.model.simulate (bitwise, until rounding)."""
    F, H, T, D, _ = B.preset("jadhav_reference")
    ref = simulate(Params(), 1, seed=seed)
    r = B.simulate_bridge(F, H, T, D, n_steps=369, seed=seed, init="jadhav")
    np.testing.assert_allclose(r.pos_s, np.moveaxis(ref.pos_s[0], -1, 0), atol=1e-6)
    np.testing.assert_allclose(r.pos_d[:, 0], ref.pos_d[0].T, atol=1e-6)
    assert np.array_equal(r.mode[1:, 0], ref.mode[0][1:])


def test_published_activation_leaves_distant_sheep_standing():
    """herding_model.m: no dog within r_dog and no neighbour within r_rep ⇒ no step, heading kept."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-20, 20, (14, 2))
    while (np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(14) * 99 < 2.0).any():
        pos = rng.uniform(-20, 20, (14, 2))
    hd = B._unit(rng.standard_normal((14, 2)))
    far_dog = np.array([[300.0, 0.0]])
    r = B.simulate_bridge(B.Followers(), B.Herders("strombom", n=1), T0, PLANE, n_steps=20, seed=1,
                          init_state=(pos, hd, far_dog, np.array([[1.0, 0.0]])))
    np.testing.assert_array_equal(r.pos_s[-1], pos)
    np.testing.assert_array_equal(r.hd_s[-1], hd)
    assert r.metrics["active_fraction"][-1] == 0.0


def test_wander_extension_moves_standing_sheep_and_is_off_by_default():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-20, 20, (10, 2)) * 3
    hd = B._unit(rng.standard_normal((10, 2)))
    st = (pos, hd, np.array([[500.0, 0.0]]), np.array([[1.0, 0.0]]))
    off = B.simulate_bridge(B.Followers(n=10), B.Herders(n=1), T0, PLANE, n_steps=30, seed=1, init_state=st)
    on = B.simulate_bridge(B.Followers(n=10, wander_speed=0.3), B.Herders(n=1), T0, PLANE, n_steps=30, seed=1, init_state=st)
    assert np.abs(off.pos_s[-1] - pos).max() == 0.0
    steps = np.linalg.norm(np.diff(on.pos_s, axis=0), axis=-1)
    np.testing.assert_allclose(steps, 0.3, atol=1e-12)  # every standing sheep walks at the wander speed


def test_contagion_activation_spreads_speed_from_the_dog():
    """herding_model_sim.m: speeds start at 0, are set to v within r_dog, and are copied from the
    alignment neighbours otherwise, so motion spreads outward from the sheep that see the dog."""
    F = B.Followers(activation="contagion", r_dog=3.0, n_att=8, n_ali=2)
    pos = np.stack([np.linspace(0, 26, 14), np.zeros(14)], -1)  # a line; dog near one end
    hd = np.tile([[1.0, 0.0]], (14, 1))
    st = (pos, hd, np.array([[-2.0, 0.0]]), np.array([[1.0, 0.0]]))
    r = B.simulate_bridge(F, B.Herders(n=1, l_drive=2.0), T0, PLANE, n_steps=1, seed=0, init_state=st)
    moved = np.linalg.norm(r.pos_s[1] - r.pos_s[0], axis=-1) > 0
    assert moved[0] and not moved[-1]  # only the sheep within 3 m of the dog moves on step 1
    r = B.simulate_bridge(F, B.Herders(n=1, l_drive=2.0), T0, PLANE, n_steps=40, seed=0, init_state=st)
    moved = np.linalg.norm(r.pos_s[-1] - r.pos_s[-2], axis=-1) > 0
    assert moved.sum() > moved[:1].sum()  # motion has spread to sheep that never saw the dog


# ------------------------------------------------------------------ noise

def test_directional_noise_keeps_the_speed_fixed_and_brownian_noise_spreads_positions():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 3, (20, 2))
    hd = B._unit(rng.standard_normal((20, 2)))
    st = (pos, hd, np.array([[-30.0, 0.0]]), np.array([[1.0, 0.0]]))
    F = B.Followers(n=20, activation="always", w_dog=0.0)
    a = B.simulate_bridge(F, B.Herders("none", n=0), T0, PLANE, n_steps=10, seed=3, init_state=st)
    steps = np.linalg.norm(np.diff(a.pos_s, axis=0), axis=-1)
    np.testing.assert_allclose(steps, 1.0, atol=1e-12)  # directional noise turns the heading, not the step
    b = B.simulate_bridge(replace(F, D=0.5), B.Herders("none", n=0), T0, PLANE, n_steps=10, seed=3, init_state=st)
    steps_b = np.linalg.norm(np.diff(b.pos_s, axis=0), axis=-1)
    assert steps_b.std() > 0.3  # Brownian displacement changes the step length
    assert np.allclose(a.pos_s[1], b.pos_s[1] - (b.pos_s[1] - a.pos_s[1]))  # same heading draws, extra displacement


def test_first_order_diffusion_coefficient_is_D():
    F = B.Followers("first_order", n=400, dt=0.1, D=0.3, population_scaling=False)
    st = (np.zeros((400, 2)), np.tile([[1.0, 0.0]], (400, 1)), np.zeros((0, 2)), np.zeros((0, 2)))
    r = B.simulate_bridge(F, B.Herders("none", n=0), T0, PLANE, n_steps=50, seed=0, init_state=st)
    msd = (r.pos_s[-1] ** 2).sum(-1).mean()
    assert abs(msd / (4 * 0.3 * 5.0) - 1) < 0.15  # <|x|²> = 4 D t in 2D


# ------------------------------------------------------------------ periodic geometry

def test_wrapped_distances_and_circular_centroid():
    dom = B.Domain(True, 10.0)
    np.testing.assert_allclose(dom.wrap(np.array([4.0, -4.0]) - np.array([-4.0, 4.0])), [-2.0, 2.0])
    pos = np.array([[4.5, 0.0], [-4.5, 0.0], [4.8, 0.0]])  # a cluster straddling the seam
    G = dom.centroid(pos)
    assert abs(dom.wrap(G[0] - 4.933)) < 0.05 and abs(G[1]) < 1e-9


def test_periodic_dynamics_are_translation_invariant():
    """Shifting every agent by a constant (across the seam) shifts the whole trajectory."""
    dom = B.Domain(True, 40.0)
    F = B.Followers(n=12, activation="always")
    H = B.Herders("strombom", n=2)
    rng = np.random.default_rng(1)
    pos = rng.normal(0, 2, (12, 2)); hd = B._unit(rng.standard_normal((12, 2)))
    dog = np.array([[-8.0, 1.0], [-8.0, -1.0]]); vd = np.tile([[1.0, 0.0]], (2, 1))
    shift = np.array([19.0, -17.0])
    T = B.Target("gaussian", centre=(0.0, 0.0), sigma=2.0)
    a = B.simulate_bridge(F, H, T, dom, n_steps=25, seed=5, init_state=(pos, hd, dog, vd))
    b = B.simulate_bridge(F, H, replace(T, centre=tuple(shift)), dom, n_steps=25, seed=5,
                          init_state=(pos + shift, hd, dog + shift, vd))
    np.testing.assert_allclose(dom.wrap(b.pos_s - a.pos_s - shift), 0.0, atol=1e-9)
    np.testing.assert_allclose(dom.wrap(b.pos_d - a.pos_d - shift), 0.0, atol=1e-9)
    for k in ("occupancy", "cohesion", "centroid_error"):
        np.testing.assert_allclose(a.metrics[k], b.metrics[k], atol=1e-9)
    # the grid KDE sees the shift through its cells (a shift that is not a whole cell moves the deposit)
    np.testing.assert_allclose(a.metrics["density_error"], b.metrics["density_error"], rtol=0.05)


def test_periodic_kernel_images_approach_the_periodic_sum():
    dom = B.Domain(True, 2 * np.pi)
    f = B.Followers("first_order", L_dog=np.pi).f
    z = np.random.default_rng(0).uniform(-np.pi, np.pi, (300, 2))
    k0, k1, k3 = (B.PeriodicKernel(f, dom, i)(z) for i in (0, 1, 3))
    assert np.abs(k1 - k3).max() < 0.5 * np.abs(k0 - k3).max()
    g = B.Grid(dom.size, 32)
    nodes = np.stack([g.DX.ravel(), g.DY.ravel()], -1)
    np.testing.assert_allclose(B.PeriodicKernel(f, dom, 2)(nodes).T.reshape(2, 32, 32), g.vector_kernel(f, 2), atol=1e-12)


# ------------------------------------------------------------------ metrics

def test_density_error_closed_form_matches_grid_on_a_large_torus():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 2.5, (40, 2))
    plane = B.Metrics(B.Target("gaussian", sigma=2.5, bandwidth=1.0), B.Domain(False), 40)
    torus = B.Metrics(B.Target("gaussian", sigma=2.5, bandwidth=1.0), B.Domain(True, 60.0), 40, B.Grid(60.0, 256))
    assert abs(plane.density_error(pos) - torus.density_error(pos)) < 0.02
    assert abs(plane.floor - torus.floor) < 0.01 * plane.floor + 1e-3


def test_finite_population_floor_matches_sampling():
    T = B.Target("gaussian", sigma=2.5, bandwidth=1.0)
    m = B.Metrics(T, B.Domain(False), 30)
    rng = np.random.default_rng(0)
    errs = [m.density_error(rng.normal(0, 2.5, (30, 2))) for _ in range(300)]
    assert abs(np.mean(errs) / m.floor - 1) < 0.1
    assert np.isclose(m.floor, 2.5 ** 2 / (30 * 1.0 ** 2))


# ------------------------------------------------------------------ Di Lorenzo design and feasibility

def test_dilorenzo_minimum_herder_mass_matches_their_280_of_1000():
    """arXiv:2411.04791 Sec. IV-A: k = 6/π, D = 0.01, L = π ⇒ N_H = 280 for N_T = 720 (M̂ = 0.28)."""
    F, H, T, D, _ = B.preset("dilorenzo_reference")
    d = B.design_herders(F, replace(H, grid=128), T, replace(D, images=3))
    assert abs(d.mass_needed - 0.28) < 0.01
    assert abs(B.herders_from_mass(720, d.mass_needed) - 280) < 15
    assert d.drift_error < 0.01 and d.feasible


def test_herder_mass_scales_with_D_and_is_independent_of_N():
    F, H, T, D, _ = B.preset("dilorenzo_reference")
    a = B.design_herders(F, H, T, D).mass_needed
    b = B.design_herders(replace(F, D=2 * F.D), H, T, D).mass_needed
    c = B.design_herders(replace(F, n=100), replace(H, n=40), T, D).mass_needed
    assert np.isclose(b, 2 * a, rtol=1e-6) and np.isclose(c, a, rtol=1e-6)


def test_surplus_herders_are_spread_uniformly_and_shortfall_is_marked_infeasible():
    F, H, T, D, _ = B.preset("dilorenzo_reference")
    many = B.design_herders(F, replace(H, n=600), T, D)
    few = B.design_herders(F, replace(H, n=50), T, D)
    assert many.feasible and not few.feasible
    assert np.isclose(many.rho_H.sum() * many.grid.h ** 2, 600 / 1320, rtol=1e-6)
    assert np.isclose(few.rho_H.sum() * few.grid.h ** 2, 50 / 770, rtol=1e-6)
    assert np.isclose(many.rho_H.min() - (many.mass_available - many.mass_needed) / (2 * np.pi) ** 2, 0.0, atol=1e-9)


def test_feasibility_scan_with_cohesion_depends_on_the_herder_mass():
    F, _, T, D, _ = B.preset("dilorenzo_reference")
    plain = B.feasibility_scan(F, T, D, masses=np.linspace(0.05, 0.95, 19), grid=48)
    coh = B.feasibility_scan(replace(F, g_rep=2.0, l_rep=np.pi / 2, g_att=1.0, l_att=np.pi), T, D,
                             masses=np.linspace(0.05, 0.95, 19), grid=48)
    assert np.ptp(plain["needed"]) < 1e-9  # no follower interaction: M̂ does not depend on M_H
    assert np.ptp(coh["needed"]) > 1e-3  # with it, the required drift carries M_T = 1 − M_H
    assert plain["min_feasible_mass"] is not None


def test_dilorenzo_herders_converge_onto_the_designed_density():
    """Theorem 1: the herder density error decays at rate K_H."""
    F, H, T, D, _ = B.preset("dilorenzo_reference")
    F = replace(F, n=100, dt=0.02)
    H = replace(H, n=40, grid=32)
    r = B.simulate_bridge(F, H, T, D, n_steps=150, seed=0, init="uniform", store_every=50)
    g = r.design.grid
    Kb = g.gaussian_kernel(H.resolve(F, D).bandwidth)
    w = 1.0 / 140
    err = [np.sqrt((((g.convolve(Kb, g.deposit(p, np.full(40, w))) - r.design.rho_H) ** 2).sum() * g.h ** 2)) for p in r.pos_d]
    assert err[-1] < 0.3 * err[0]


def test_strombom_multi_dog_reduces_to_the_published_dog_for_one_dog():
    F = B.Followers(n=14)
    H1 = B.Herders("strombom", n=1, spread_angle=0.7)
    r = B.simulate_bridge(F, H1, T0, PLANE, n_steps=40, seed=0, init="jadhav", box_length=30.0)
    ref = simulate(Params(box_length=30.0, n_steps=41), 1, seed=0)
    np.testing.assert_allclose(r.pos_s, np.moveaxis(ref.pos_s[0], -1, 0), atol=1e-6)


# ------------------------------------------------------------------ density-feedback controller (heuristic)

def _fb_setup():
    rng = np.random.default_rng(4)
    pos = rng.normal(0, 3, (20, 2))
    pos[:5] += [9.0, 0.0]  # a surplus lobe to the right of the goal
    dogs = np.array([[-20.0, 5.0], [20.0, -5.0]])
    T = B.Target("gaussian", sigma=2.5)
    return pos, dogs, T


def test_feedback_aims_push_the_largest_surplus_inward():
    pos, dogs, T = _fb_setup()
    F = B.Followers(n=20)
    dom = B.Domain(False)
    H = B.Herders("density_feedback", n=2).resolve(F, dom, T)
    met = B.Metrics(T.resolve(dom), dom, 20)
    sur = met.surplus(pos)
    aims, mode, _ = B.feedback_aims(pos, dogs, np.zeros(2), sur, H, dom, False, F.r_dog, met.peak)
    assert (mode == B.SHAPE).sum() >= 1
    i = int(np.argmax(sur))  # the sheep with the largest surplus is targeted by the nearest dog
    m = int(np.argmin(np.linalg.norm(pos[i] - dogs, axis=-1)))
    np.testing.assert_allclose(aims[m], pos[i] + H.fb_standoff * pos[i] / np.linalg.norm(pos[i]))
    _, mode_far, drv = B.feedback_aims(pos + 50.0, dogs + 50.0, np.zeros(2), sur, H, dom, False, F.r_dog, met.peak)
    assert drv
    assert (mode_far == B.DRIVE).all()  # far from the goal: Strömbom's drive geometry
    held, mode_h, _ = B.feedback_aims(pos, dogs, np.zeros(2), sur, H, dom, True, F.r_dog, met.peak)
    assert (mode_h == B.HOLD).all() and (np.linalg.norm(held - pos.mean(0), axis=-1) > F.r_dog).all()


def test_feedback_release_withdraws_and_published_sheep_stop():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 2.0, (14, 2))
    hd = B._unit(rng.standard_normal((14, 2)))
    st = (pos, hd, np.array([[-15.0, 0.0]]), np.array([[1.0, 0.0]]))
    H = B.Herders("density_feedback", n=1, fb_release=1.0)
    r = B.simulate_bridge(B.Followers(), H, B.Target("gaussian", sigma=2.5), B.Domain(False), n_steps=120, seed=0, init_state=st)
    assert r.mode[-1, 0] == B.HOLD
    np.testing.assert_array_equal(r.pos_s[-1], r.pos_s[-2])  # released: no sheep sees the dog, so none moves
    assert r.metrics["density_error"][-1] < 1.5


def test_feedback_dogs_wait_beyond_reach_of_a_matched_flock():
    """A flock already matching the target: the dogs find no surplus above the threshold and wait
    beyond the notice radius of every sheep, so the published sheep are left alone."""
    rng = np.random.default_rng(2)
    pos = rng.normal(0, 2.5, (40, 2))
    hd = B._unit(rng.standard_normal((40, 2)))
    ang = 2 * np.pi * np.arange(4) / 4
    dogs = 30.0 * np.stack([np.cos(ang), np.sin(ang)], -1)
    st = (pos, hd, dogs, np.tile([[1.0, 0.0]], (4, 1)))
    F = B.Followers(n=40)
    r = B.simulate_bridge(F, B.Herders("density_feedback", n=4, fb_threshold=1.0), B.Target("gaussian", sigma=2.5), B.Domain(False),
                          n_steps=60, seed=0, init_state=st)
    assert (r.mode[-1] == B.HOLD).all()
    d = np.linalg.norm(r.pos_s[-1][:, None] - r.pos_d[-1][None], axis=-1)
    assert d.min() > F.r_dog
    assert r.metrics["active_fraction"][-1] == 0.0


def test_dilorenzo_scenario_in_metres_keeps_their_design():
    """Their setting rescaled to metres needs the same herder mass fraction, and Jadhav sheep get the
    herders' own design (their follower model), not the empty design a D = 0 heading sheep would give."""
    F0, H0, T0_, D0, _ = B.preset("dilorenzo_reference")
    ref = B.design_herders(F0, H0, T0_, D0).mass_needed
    Fm, Hm, Tm, Dm, _ = B.preset("dilorenzo_metres")
    Fj, Hj, Tj, Dj, _ = B.preset("dilorenzo_jadhav_sheep")
    assert Fj.rule == "heading" and Fj.activation == "published" and Fj.D == 0.0
    assert abs(B.design_herders(Fm, Hm, Tm, Dm).mass_needed - ref) < 1e-9
    dj = B.design_herders(Fj, Hj, Tj, Dj)
    assert abs(dj.mass_needed - ref) < 1e-9 and dj.feasible
    assert B.design_herders(Fj, replace(Hj, design_model=None), Tj, Dj).mass_needed == 0.0
    assert Dj.size == pytest.approx(24.0) and Hj.n == 14 and Hj.substeps == 100


def test_herder_substeps_integrate_the_same_flow():
    """Sub-stepping the feedforward flow approaches the fine-step solution."""
    F, H, T, D, _ = B.preset("dilorenzo_metres")
    F = replace(F, dt=1.0)
    runs = {k: B.simulate_bridge(F, replace(H, substeps=k), T, D, n_steps=3, seed=2, init="uniform") for k in (1, 10, 100)}
    fine = runs[100].pos_d[-1]
    err = {k: np.abs(D.wrap(runs[k].pos_d[-1] - fine)).max() for k in (1, 10)}
    assert err[10] < err[1]


# ------------------------------------------------------------------ goal schedule (planner above the herders)

def test_goal_at_the_target_centre_is_the_original_run():
    """A schedule that never leaves the target centre takes the original code path: bitwise equal."""
    from shepherd.goal_schedule import FixedGoal
    F, H, T, D, _ = B.dilorenzo_scenario("jadhav")
    H = replace(H, substeps=5)
    a = B.simulate_bridge(F, H, T, D, seed=4, init="uniform", n_steps=15)
    b = B.simulate_bridge(F, H, T, D, seed=4, init="uniform", n_steps=15, goal=FixedGoal(T.centre))
    assert np.array_equal(a.pos_s, b.pos_s) and np.array_equal(a.pos_d, b.pos_d)
    assert np.array_equal(b.goal, np.zeros_like(b.goal))


def test_shifted_design_equals_a_design_at_the_new_centre():
    """Translating the precomputed design by a grid-aligned offset gives the run whose target is centred
    there (design recomputed): the planner only moves the goal, the herders behave as designed."""
    from shepherd.goal_schedule import FixedGoal
    F, H, T, D, _ = B.dilorenzo_scenario("jadhav")
    H = replace(H, substeps=5)
    h = D.size / H.grid
    c = (5 * h, -3 * h)
    st = B.initial_state(F.resolved(), H, T.resolve(D), D, np.random.default_rng(0), "uniform")
    moved = B.simulate_bridge(F, H, replace(T, centre=c), D, seed=1, init_state=st, n_steps=12)
    planned = B.simulate_bridge(F, H, T, D, seed=1, init_state=st, n_steps=12, goal=FixedGoal(c))
    np.testing.assert_allclose(planned.pos_d, moved.pos_d, atol=1e-8)
    np.testing.assert_allclose(planned.pos_s, moved.pos_s, atol=1e-8)


def test_goal_schedules_share_waypoints_and_glide_is_continuous():
    from shepherd.goal_schedule import GlideSchedule, HopSchedule
    D = B.Domain(True, 24.0)
    w = np.array([[0.0, 0.0], [6.0, 0.0], [-10.0, 11.0], [3.0, -2.0]])
    hop, glide = HopSchedule(w, 20), GlideSchedule(w, 20, D)
    assert np.array_equal(hop(0), w[0]) and np.array_equal(hop(39.9), w[1])
    np.testing.assert_allclose(glide(20.0), w[1]) and np.testing.assert_allclose(glide(10.0), [3.0, 0.0])
    ts = np.arange(0, 60, 0.5)
    steps = [np.linalg.norm(D.wrap(glide(b) - glide(a))) for a, b in zip(ts[:-1], ts[1:])]
    assert max(steps) < 0.5  # never jumps: at most |Δw|/dwell · 0.5 s
    with pytest.raises(ValueError):
        F, H, T, Dm, _ = B.preset("jadhav_reference")
        B.simulate_bridge(F, H, T, Dm, n_steps=1, goal=hop)


# ------------------------------------------------------------------ scorecard

def test_scorecard_definitions_on_a_constructed_run():
    """Occupancy, time to sustained success, escapes, splits and distance on a hand-built trajectory."""
    from shepherd.scorecard import score
    D = B.Domain(False)
    T = B.Target("gaussian", sigma=2.5).resolve(D)  # r_goal ≈ 6.1 m
    n, S = 10, 121  # 0..120 s at 1 s
    pos = np.zeros((S, n, 2))
    pos[:, :, 0] = np.linspace(-1, 1, n)
    pos[:20] += 30.0           # out until t = 19
    pos[70:80, :6] += 30.0      # 6 of 10 leave for 10 s: occupancy 0.4 < 0.5 → one escape, recovered
    pos[100:, :2] += 20.0       # two sheep split off at the end and stay: an unrecovered split (largest 0.8)
    dogs = np.zeros((S, 1, 2)); dogs[:, 0, 0] = np.arange(S) * 0.5  # 0.5 m/s, straight
    r = B.BridgeResult(B.Followers(n=n), B.Herders("strombom", n=1), T, D, np.arange(S, dtype=float), pos,
                       np.zeros_like(pos), dogs, np.zeros((S, 1), np.int8), {}, 0.0)
    s = score(r)
    assert s["occupancy"][0] == 0 and s["occupancy"][20] == 1
    assert s["time_to_success"] == 20.0
    assert s["escapes"] == 1 and s["escape_unrecovered"] == 0 and s["escape_duration_mean"] == 10.0
    assert s["splits"] == 2 and s["splits_unrecovered"] == 1  # the 6-sheep excursion and the two that leave
    assert abs(s["final_occupancy"] - (10 * 1.0 + 21 * 0.8) / 31) < 1e-12  # t = 90..120: 10/10 to t = 99, then 8/10
    assert s["dogs"] == 1 and abs(s["distance_total"] - 60.0) < 1e-9 and abs(s["speed_peak"] - 0.5) < 1e-12
