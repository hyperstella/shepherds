import numpy as np
import pytest

from shepherd import data, kernels as K


@pytest.fixture(scope="module")
def drives():
    return data.load_drives(data.load_sessions())


def test_hat_basis_interpolates_knots_and_vanishes_beyond_range():
    b = K.HatBasis((0.0, 0.5, 1.0, 2.0, 4.0))
    np.testing.assert_allclose(b(np.array(b.knots[:-1])), np.eye(b.size))
    assert np.all(b(np.array([4.0, 5.0, 9.0])) == 0)
    r = np.linspace(0, 2, 41)  # partition of unity before the last interval
    np.testing.assert_allclose(b(r).sum(-1), 1.0)


def test_apply_gains_scales_only_selected_kernels():
    m = K.KernelModel(terms=("E", "A", "D"))
    c = np.arange(m.n_coef, dtype=float) + 1
    out = K.apply_gains(m, c, {"E": np.array([2.0, 0.0])})
    sl = m.slices()
    np.testing.assert_allclose(out[0, sl["E"]], 2 * c[sl["E"]])
    np.testing.assert_allclose(out[1, sl["E"]], 0)
    np.testing.assert_allclose(out[:, sl["D"]], np.tile(c[sl["D"]], (2, 1)))


def test_weak_form_recovers_noise_free_dynamics(drives):
    """Fit exact synthetic tracks of known kernels: the learned forces should
    reproduce the true ones where data lie (small discretization error only)."""
    m = K.KernelModel(terms=("E", "D"), pair=K.HatBasis((0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)),
                      dog=K.HatBasis((0.0, 2.0, 4.0, 6.0, 10.0, 20.0)))
    sl = m.slices()
    c_true = np.zeros(m.n_coef)
    c_true[sl["E"]] = [-0.8, -0.3, 0.2, 0.3, 0.25, 0.1]
    c_true[sl["D"]] = [1.5, 1.0, 0.5, 0.2, 0.05]
    tracks, pieces = [], []
    for i, d in enumerate(drives[:6]):
        pos, vel = K.smooth_drive(d.pos[: K.N + 1])
        L = min(pos.shape[-1], 150)
        P, V = K.simulate(m, c_true, 0.0, pos[None, :K.N, :, 0], vel[None, :K.N, :, 0], pos[K.N, :, :L], seed=0)
        tracks.append((P[0], V[0]))
        pieces.append(K.Piece(d.session, i, P[0]))
    nrm = K.normal_equations(m, pieces, H=0.5, stride=0.2, tracks=tracks)
    c_hat = K.solve(m, nrm, lam=1e-6)
    X, y = zip(*(K.weak_form(m, *t, H=0.5) for t in tracks))
    X, y = np.concatenate(X), np.concatenate(y)
    r2 = 1 - np.sum((y - X @ c_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.98
    err = K.curve_error(m, c_true, c_hat, K.distance_weights(drives[:6]))
    assert err["E"] < 0.15 and err["D"] < 0.15, err


def test_speed_replay_sets_flock_mean_speed(drives):
    m = K.KernelModel(terms=("E",))
    c = np.zeros(m.n_coef)
    pos, vel = K.smooth_drive(drives[0].pos[: K.N + 1])
    speed = np.full(40, 1.7)
    _, V = K.simulate(m, c, 0.1, np.repeat(pos[None, :K.N, :, 0], 3, 0), np.repeat(vel[None, :K.N, :, 0], 3, 0),
                      pos[K.N, :, :40], seed=0, speed_path=speed)
    np.testing.assert_allclose(np.linalg.norm(V[:, :K.N, :, 1:], axis=2).mean(1), 1.7)
