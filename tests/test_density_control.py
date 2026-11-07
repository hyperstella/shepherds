import numpy as np

from shepherd import density_control as DC
from shepherd.expected_heading import expected_headings
from shepherd.model import Params, simulate


def _driven(seed, n_runs=4, social_always=False):
    res = simulate(Params(n_steps=60, social_always=social_always), n_runs=n_runs, seed=seed)
    return res.pos_s[..., -1], res.vel_s[..., -1], res.pos_d[..., -1]


def _monte_carlo(pos, hd, dog, p, K=20000, seed=5):
    init = dict(pos_s=np.repeat(pos[None], K, 0), vel_s=np.repeat(hd[None], K, 0))
    dp = np.repeat(dog[None], K, 0)
    res = simulate(Params(n_steps=2, social_always=p.social_always), n_runs=K, seed=seed, init=init, dog_path=np.stack([dp, dp], -1))
    return res.pos_s[..., -1]


def test_batched_response_matches_expected_headings():
    pos, hd, dog = _driven(3)
    mu0, mv0 = expected_headings(pos, hd, dog)
    mu1, mv1 = DC.ExpectedResponse(pos, hd)(dog)
    assert np.allclose(mu0, mu1, atol=1e-12) and np.array_equal(mv0, mv1)


def test_response_under_the_modified_rule_matches_monte_carlo():
    p = Params(social_always=True)
    pos, hd, dog = _driven(4, n_runs=1, social_always=True)
    dog = pos[0].mean(0) + np.array([9.0, 0.0])  # beyond the notice radius for part of the flock
    mu, moves = DC.ExpectedResponse(pos, hd, p)(dog[None])
    x1 = _monte_carlo(pos[0], hd[0], dog, p)
    step = x1 - pos[0][None]
    se = step.std(0) / np.sqrt(len(step))
    assert moves.all()
    assert np.all(np.abs(step.mean(0) - mu[0]) <= 5 * se + 1e-12)


def test_l2_gradient_and_laplacian_match_finite_differences():
    fn = DC.Functional("l2", s=DC.goal_sigma(5.0), b=1.0)
    pos, _, _ = _driven(5, n_runs=2)
    x = pos - pos.mean(1, keepdims=True) + np.array([3.0, -1.0])
    v, g, lap = fn.terms(x)
    eps = 1e-5
    num, numlap = np.zeros_like(g), np.zeros(g.shape[:2])
    for i in range(x.shape[1]):
        for c in range(2):
            xp, xm = x.copy(), x.copy()
            xp[:, i, c] += eps
            xm[:, i, c] -= eps
            num[:, i, c] = (fn.value(xp) - fn.value(xm)) / (2 * eps)
            numlap[:, i] += (fn.value(xp) - 2 * v + fn.value(xm)) / eps ** 2
    assert np.allclose(num, g, atol=1e-9) and np.allclose(numlap, lap, atol=1e-5)


def test_moment_expected_change_is_exact():
    p = Params()
    fm = DC.Functional("moment")
    pos, hd, dog = _driven(3, n_runs=1)
    mu, mv = DC.ExpectedResponse(pos, hd, p)(dog)
    x1 = _monte_carlo(pos[0], hd[0], dog[0], p)
    dV = fm.value(x1) - fm.value(pos)
    assert abs(fm.expected_change(pos, mu, mv)[0] - dV.mean()) <= 5 * dV.std() / np.sqrt(len(dV))


def test_lookahead_moment_expected_change_is_exact():
    p = Params()
    f8 = DC.Functional("moment", horizon=8)
    pos, hd, dog = _driven(3, n_runs=1)
    mu, mv = DC.ExpectedResponse(pos, hd, p)(dog)
    x1 = _monte_carlo(pos[0], hd[0], dog[0], p, K=40000, seed=7)
    res = simulate(Params(n_steps=2), n_runs=40000, seed=7, init=dict(pos_s=np.repeat(pos, 40000, 0), vel_s=np.repeat(hd, 40000, 0)),
                   dog_path=np.stack([np.repeat(dog, 40000, 0)] * 2, -1))
    dV = f8.value(x1, res.vel_s[..., -1]) - f8.value(pos, hd)
    assert abs(f8.expected_change(pos, mu, mv, 1.0, None, hd)[0] - dV.mean()) <= 5 * dV.std() / np.sqrt(len(dV))
    assert np.allclose(f8.value(pos, hd), ((pos + 8 * hd) ** 2).sum(-1).mean(1))


def test_placement_recovers_a_drive_point():
    pos, hd, dog = _driven(4, n_runs=6)
    ctrl = DC.Controller(DC.Functional("moment"))
    h, s, _ = ctrl.place(DC.ExpectedResponse(pos, hd), dog)
    c = pos.mean(1)
    rel = h - c
    cos = (DC._unit(rel) * DC._unit(-c)).sum(-1)  # the goal is the origin
    assert (s < 0).all()
    assert (cos < -0.9).all()  # behind the flock on the goal line
    assert (np.linalg.norm(rel, axis=-1) >= 4).all() and (np.linalg.norm(rel, axis=-1) <= 10).all()


def test_hold_at_the_goal_keeps_a_packed_flock_in_sight():
    # Sheep sampled from the goal density sit within each other's repulsion radius. Without a
    # dog in range they graze apart (repulsion only), which raises the second moment; with the
    # dog in sight attraction is on and the placement finds a position that costs less.
    rng = np.random.default_rng(0)
    R, N = 6, 14
    pos = DC.goal_sigma(5.0) * rng.standard_normal((R, N, 2))
    th = 2 * np.pi * rng.random((R, N))
    hd = np.stack([np.cos(th), np.sin(th)], -1)
    dog = pos.mean(1) + np.array([-7.5, 0.0])
    ctrl = DC.Controller(DC.Functional("moment"))
    resp = DC.ExpectedResponse(pos, hd)
    h, s, _ = ctrl.place(resp, dog)
    graze = ctrl.functional.expected_change(pos, *resp(pos.mean(1) + np.array([1e3, 0.0])))
    in_range = (np.linalg.norm(pos - h[:, None], axis=-1) <= 12.0).mean(1)
    assert (graze > 0).all() and (s < graze).all()
    assert (in_range >= 0.9).all()


def test_release_is_free_for_a_spread_flock():
    # With every pair further apart than r_rep, published sheep without a dog in range stand
    # still: the out-of-range change is exactly zero, so the best placement never costs.
    rng = np.random.default_rng(1)
    R, N = 4, 14
    g = np.stack(np.meshgrid(np.arange(4) * 2.6, np.arange(4) * 2.6, indexing="ij"), -1).reshape(-1, 2)[:N]
    pos = g[None] - g[:N].mean(0) + 0.1 * rng.standard_normal((R, N, 2))
    th = 2 * np.pi * rng.random((R, N))
    hd = np.stack([np.cos(th), np.sin(th)], -1)
    ctrl = DC.Controller(DC.Functional("moment"))
    resp = DC.ExpectedResponse(pos, hd)
    graze = ctrl.functional.expected_change(pos, *resp(pos.mean(1) + np.array([1e3, 0.0])))
    _, s, _ = ctrl.place(resp, pos.mean(1) + np.array([-18.0, 0.0]))
    assert np.array_equal(graze, np.zeros(R)) and (s <= 1e-12).all()
