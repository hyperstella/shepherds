import numpy as np

from shepherd.expected_heading import expected_headings, expected_spread_change, noise_shrink
from shepherd.model import Params, simulate


def _state(seed, p):
    res = simulate(Params(n_sheep=p.n_sheep, box_length=2000.0, n_steps=160), n_runs=1, seed=seed)
    t = 150
    return res.pos_s[0, :, :, t], res.vel_s[0, :, :, t], res.pos_d[0, :, t]


def test_noise_shrink_small_q():
    # m(q) = 1 - q^2/4 + O(q^4)
    q = np.array([0.0, 0.01, 0.05])
    assert np.allclose(noise_shrink(q), 1 - q ** 2 / 4, atol=1e-6)


def test_matches_monte_carlo():
    p = Params()
    pos, hd, dog = _state(3, p)
    mu, moves = expected_headings(pos[None], hd[None], dog[None], p)
    K = 40000
    init = dict(pos_s=np.repeat(pos[None], K, 0), vel_s=np.repeat(hd[None], K, 0))
    dp = np.repeat(dog[None], K, 0)
    res = simulate(Params(n_steps=2), n_runs=K, seed=11, init=init, dog_path=np.stack([dp, dp], -1))
    step = res.pos_s[..., -1] - pos[None]
    mc = step.mean(0) / p.v_sheep
    se = step.std(0) / np.sqrt(K)
    assert np.all(np.abs(mc - mu[0]) <= 5 * se + 1e-12)
    assert np.array_equal(moves[0], np.linalg.norm(step, axis=-1).max(0) > 0)


def test_spread_change_matches_monte_carlo():
    # a dilated state with the dog 10 m to one side, so some sheep stand still and some only repel
    p = Params()
    pos, hd, dog = _state(5, p)
    c = pos.mean(0)
    pos = c + 1.6 * (pos - c)
    dog = c + np.array([10.0, 0.0])
    far = np.linalg.norm(pos - dog, axis=-1).argsort()[::-1]
    pos[far[1]] = pos[far[0]] + np.array([0.0, 1.0])
    _, moves = expected_headings(pos[None], hd[None], dog[None], p)
    sees = np.linalg.norm(pos - dog, axis=-1) <= p.r_dog
    assert 0 < sees.sum() < moves.sum() < p.n_sheep
    exact = expected_spread_change(pos[None], hd[None], dog[None], p)[0]
    K = 200000
    init = dict(pos_s=np.repeat(pos[None], K, 0), vel_s=np.repeat(hd[None], K, 0))
    dp = np.repeat(dog[None], K, 0)
    res = simulate(Params(n_steps=2), n_runs=K, seed=12, init=init, dog_path=np.stack([dp, dp], -1))
    m2 = lambda x: ((x - x.mean(1, keepdims=True)) ** 2).sum(-1).mean(1)
    d = m2(res.pos_s[..., -1]) - m2(pos[None])[0]
    assert abs(d.mean() - exact) <= 5 * d.std() / np.sqrt(K)
