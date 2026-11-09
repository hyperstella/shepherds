import numpy as np

from shepherd.model import REPLAY, Params, simulate

P = Params(n_steps=40, box_length=30.0)
FIELDS = ["pos_s", "vel_s", "pos_d", "vel_d", "mode"]


def same(a, b, atol=0.0):
    for f in FIELDS:
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), atol=atol, rtol=0, err_msg=f)


def test_per_run_equal_values_match_scalar_path():
    a = simulate(P, 3, seed=1)
    b = simulate(P, 3, seed=1, per_run={"w_att": [1.5] * 3, "r_rep": np.full(3, 2.0), "noise": 0.3})
    same(a, b)


def test_mixed_batch_leaves_other_runs_untouched():
    a = simulate(P, 2, seed=5, per_run={"w_dog": [1.0, 1.0], "r_rep": [2.0, 2.0]})
    b = simulate(P, 2, seed=5, per_run={"w_dog": [1.0, 0.2], "r_rep": [2.0, 0.8]})
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(a, f)[0], getattr(b, f)[0], err_msg=f)
    assert not np.allclose(a.pos_s[1], b.pos_s[1], equal_nan=True)


def test_per_run_r_rep_moves_derived_lengths_like_scalar():
    a = simulate(Params(n_steps=40, box_length=30.0, r_rep=1.0), 2, seed=2)
    b = simulate(P, 2, seed=2, per_run={"r_rep": [1.0, 1.0]})
    same(a, b)


def test_sheep_weights_only_matter_up_to_a_common_scale():
    # new heading = unit(sum of weighted terms), so scaling all sheep weights
    # (noise included) changes nothing; the dog keeps its own noise.
    k = 2.5
    base = dict(w_inertia=0.5, w_rep=2.0, w_dog=1.0, w_att=1.5, w_ali=1.3, noise=0.3)
    a = simulate(P, 2, seed=9, per_run={**base, "noise_dog": 0.3})
    b = simulate(P, 2, seed=9, per_run={**{kk: k * v for kk, v in base.items()}, "noise_dog": 0.3})
    same(a, b, atol=1e-6)  # ulp-level differences, amplified a little by 40 chaotic steps


def test_dt_equal_tau_is_the_published_model():
    same(simulate(P, 2, seed=4), simulate(Params(n_steps=40, box_length=30.0, dt=1.0, tau_s=1.0, tau_d=1.0), 2, seed=4))


def test_long_relaxation_time_keeps_headings():
    r = simulate(Params(n_steps=20, box_length=30.0, tau_s=1e9), 1, seed=0)
    np.testing.assert_allclose(r.vel_s[..., -1], r.vel_s[..., 0], atol=1e-6)


def test_dog_replay_and_init():
    T = 25
    path = np.stack([np.linspace(-10, 0, T), np.zeros(T)])  # dog walks toward the flock
    init = {"pos_s": np.stack([np.linspace(3, 6, 14), np.zeros(14)], -1)[None]}
    r = simulate(Params(dt=0.5), 3, seed=0, init=init, dog_path=path)
    assert r.pos_s.shape[-1] == T and (r.mode[:, 1:] == REPLAY).all()
    np.testing.assert_array_equal(r.pos_d[0], path)
    np.testing.assert_array_equal(r.pos_s[:, :, :, 0], np.broadcast_to(init["pos_s"], (3, 14, 2)))
    np.testing.assert_allclose(r.vel_d[0, :, 1:], np.diff(path, axis=-1) / 0.5)
    # sheep are pushed away from the approaching dog (+x)
    assert (r.pos_s[:, :, 0, -1].mean() - r.pos_s[:, :, 0, 0].mean()) > 1.0
    # and move at most v * dt per step
    step = np.linalg.norm(np.diff(r.pos_s, axis=-1), axis=2)
    assert step.max() <= 0.5 + 1e-9


def test_pair_kernel_pulls_straggler_back():
    from dataclasses import replace

    from shepherd.kernels import PAIR_KNOTS

    # attraction that grows with distance, repulsion below 1 m
    coef = tuple(np.interp(PAIR_KNOTS[:-1], [0.0, 1.0, 6.0], [-0.05, 0.0, 0.06]))
    pos_s = np.array([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [6.0, 0.0]]])
    init = {"pos_s": pos_s, "vel_s": np.tile([0.0, 1.0], (1, 5, 1)), "pos_d": np.array([[0.0, -3.0]]),
            "vel_d": np.array([[0.0, 1.0]])}
    base = Params(n_sheep=5, k_perceive=4, n_att=2, n_steps=20, dt=0.2, noise=0.0, w_att=0.0, w_ali=0.0, w_dog=0.0, w_rep=0.0,
                  r_dog=50.0, v_dog=0.0, box_length=0.0)
    off = simulate(base, 1, seed=0, init=init)
    on = simulate(replace(base, pair_kernel=(PAIR_KNOTS, coef), g_pairk=1.0), 1, seed=0, init=init)
    gap = lambda r: np.linalg.norm(r.pos_s[0, 4, :, -1] - r.pos_s[0, :4, :, -1].mean(0))
    assert gap(on) < gap(off) - 0.5


def test_hold_withdraws_and_freezes_the_flock():
    from shepherd.model import HOLD

    rng = np.random.default_rng(0)
    init = {"pos_s": rng.normal(0, 1.5, (1, 14, 2)), "pos_d": np.array([[-9.0, 0.0]])}
    p = Params(n_steps=80, hold_radius=5.0)
    r = simulate(p, 2, seed=1, init=init)
    assert (r.mode[:, 1] == HOLD).all()  # the centroid starts inside hold_radius
    d = np.linalg.norm(r.pos_s[..., -1] - r.pos_d[:, None, :, -1], axis=-1)
    assert (d.min(-1) > p.r_dog).all()  # no sheep sees the dog at the end
    np.testing.assert_array_equal(r.pos_s[..., -1], r.pos_s[..., -2])  # and the flock is frozen
    g = r.pos_s[..., -1].mean(1)
    np.testing.assert_allclose(np.linalg.norm(r.pos_d[..., -1] - g, axis=-1), p.r_dog + 3.0, atol=0.3)


def test_hold_matches_published_model_until_the_flock_arrives():
    from dataclasses import replace

    from shepherd.model import HOLD

    q = Params(n_steps=70, box_length=25.0)
    a = simulate(q, 3, seed=7)
    b = simulate(replace(q, hold_radius=5.0), 3, seed=7)
    held = 0
    for r in range(3):
        t_hold = np.flatnonzero(b.mode[r] == HOLD)
        t0 = int(t_hold[0]) if len(t_hold) else q.n_steps
        held += len(t_hold) > 0
        for f in FIELDS:
            np.testing.assert_array_equal(getattr(a, f)[r][..., :t0], getattr(b, f)[r][..., :t0], err_msg=f)
        if len(t_hold):  # the dog only holds once the centroid is within hold_radius of the target
            assert np.linalg.norm(b.pos_s[r, :, :, t0 - 1].mean(0)) <= 5.0
    assert held > 0


def test_hold_reengages_when_the_flock_wanders():
    from shepherd.model import HOLD

    rng = np.random.default_rng(3)
    init = {"pos_s": rng.normal(0, 1.0, (1, 14, 2)), "pos_d": np.array([[-8.0, 0.0]])}
    r = simulate(Params(n_steps=300, social_always=True, hold_radius=3.0, hold_release=4.0), 4, seed=2, init=init)
    reengaged = 0
    for m in r.mode:
        t_hold = np.flatnonzero(m == HOLD)
        reengaged += len(t_hold) > 0 and bool((m[t_hold[0]:] != HOLD).any())
    assert reengaged > 0
