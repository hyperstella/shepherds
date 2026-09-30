import numpy as np
import pytest
import scipy.io as sio

from shepherd import data, metrics, network
from shepherd.model import Params, simulate, simulate_reference

UP = data.DATA_DIR


@pytest.fixture(scope="module")
def sessions():
    return data.load_sessions()


def test_sessions_match_upstream_mat(sessions):
    m = sio.loadmat(UP / "sheep_all_dat.mat")
    assert len(sessions) == 9
    for s in sessions:
        np.testing.assert_array_equal(s.pos, m[f"pos_ev_{s.number}"])
        np.testing.assert_array_equal(s.vel, m[f"vel_ev_{s.number}"])
        np.testing.assert_allclose(s.phi, m[f"phi_ev_{s.number}"], atol=1e-12)


def test_drives_match_upstream_mat(sessions):
    m = sio.loadmat(UP / "drives_data.mat")
    concat = data.concat_drives_by_session(sessions)
    assert len(concat) == 8
    for k, c in enumerate(concat, start=1):
        np.testing.assert_array_equal(c.pos, m[f"pos_ev_{k}"])
        np.testing.assert_array_equal(c.vel, m[f"vel_ev_{k}"])
    assert len(data.load_drives(sessions)) == m["len_drvs"].size == 29


def test_xcorr_matches_definition():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=50)
    lags, r = network.xcorr_unbiased(x, y, 5)
    full = np.correlate(x, y, mode="full")  # index n-1 is lag 0
    n = len(x)
    expect = full[n - 1 - 5 : n + 5] / (n - np.abs(lags))
    np.testing.assert_allclose(r, expect)
    ux, uy = np.stack([x, y]), np.stack([y, x])
    _, C = network.directional_ccf(ux, uy, 5)
    np.testing.assert_allclose(C[0, 1], network.xcorr_unbiased(x, y, 5)[1] + network.xcorr_unbiased(y, x, 5)[1])


def test_leader_detected_for_lagged_copy():
    rng = np.random.default_rng(1)
    phi = np.cumsum(rng.normal(scale=0.2, size=400))
    lead, follow = phi[5:], phi[:-5]  # follower copies leader 5 samples later
    ux = np.cos(np.stack([lead, follow]))
    uy = np.sin(np.stack([lead, follow]))
    indeg, edges = network.leader_follower(ux, uy, maxlag=20)
    assert edges == [(0, 1, 5)]
    assert list(indeg) == [1, 0]


def test_signed_angle_convention():
    a = np.array([1.0, 0.0])
    assert np.isclose(metrics.signed_angle(a, np.array([0.0, 1.0])), np.pi / 2)
    assert np.isclose(metrics.signed_angle(a, np.array([0.0, -1.0])), -np.pi / 2)
    assert metrics.signed_angle(a, np.zeros(2)) == 0.0


@pytest.mark.parametrize("n_sheep", [5, 14])
def test_vectorized_model_equals_reference(n_sheep):
    # The two implementations sum vectors in different orders, so they differ
    # by ~1 ulp from the first step; the dynamics are chaotic and amplify this
    # after ~50 steps. Compare over a shorter horizon across several seeds.
    p = Params(n_sheep=n_sheep, n_steps=30, box_length=20.0, l_sep=3.0, k_perceive=min(10, n_sheep - 1), n_att=min(4, n_sheep - 1))
    modes = set()
    for seed in range(6):
        a = simulate(p, n_runs=4, seed=seed)
        b = simulate_reference(p, n_runs=4, seed=seed)
        for f in ["pos_s", "vel_s", "pos_d", "vel_d", "mode"]:
            np.testing.assert_allclose(getattr(a, f), getattr(b, f), atol=1e-8, err_msg=f)
        modes |= set(np.unique(a.mode).tolist())
    assert {1, 2}.issubset(modes)  # both drive and collect exercised
