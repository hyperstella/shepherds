import numpy as np

from shepherd.collecting import collecting_drift_bound, dog_decision, psi
from shepherd.model import COLLECT, DRIVE, SLOW, Params, simulate


def _snapshot(seed):
    res = simulate(Params(box_length=2000.0, n_steps=160), n_runs=1, seed=seed)
    t = 150
    return res.pos_s[0, :, :, t], res.vel_s[0, :, :, t], res.pos_d[0, :, t], res.vel_d[0, :, t]


def test_dog_decision_matches_model():
    res = simulate(Params(box_length=2000.0, n_steps=120), n_runs=40, seed=4)
    pos = np.moveaxis(res.pos_s, -1, 1)[:, :-1]
    dog = np.moveaxis(res.pos_d, -1, 1)[:, :-1]
    code = {0: DRIVE, 1: COLLECT, 2: SLOW}
    pred = np.vectorize(code.get)(dog_decision(pos, dog))
    assert np.array_equal(pred, res.mode[:, 1:])


def test_drift_bound_holds():
    # a sheep moved 14 m out to the side, the dog at four places: far (target frozen), inside, behind it, and within
    # r_rep of it (slow mode)
    p = Params()
    pos, hd, _, dog_v = _snapshot(9)
    c = pos.mean(0)
    side = np.array([-hd.mean(0)[1], hd.mean(0)[0]]) / np.linalg.norm(hd.mean(0))
    pos[0] = c + 14.0 * side
    for dog in (c - 20.0 * side, c + 6.0 * side, c + 17.0 * side, pos[0] + np.array([1.5, 0.5])):
        b = collecting_drift_bound(pos[None], hd[None], dog[None], dog_v[None], p)
        assert b["target"][0] == 0
        K = 40000
        rep = lambda a: np.repeat(a[None], K, 0)
        res = simulate(Params(n_steps=2), n_runs=K, seed=13, init=dict(pos_s=rep(pos), vel_s=rep(hd), pos_d=rep(dog), vel_d=rep(dog_v)))
        d = psi(res.pos_s[..., 1], res.pos_d[..., 1], np.zeros(K, int)) - psi(pos[None], dog[None], np.zeros(1, int))[0]
        assert d.mean() <= b["U"][0] + 5 * d.std() / np.sqrt(K)
