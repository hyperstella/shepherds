import numpy as np
import pytest

from shepherd import data, metrics as M, protocol as PR
from shepherd.model import Params, simulate


@pytest.fixture(scope="module")
def drives():
    return data.load_drives(data.load_sessions())


@pytest.fixture(scope="module")
def runs():
    return simulate(Params(n_steps=120, box_length=40.0), n_runs=4, seed=3)


def test_metrics_accept_batch_axes(runs):
    N = runs.params.n_sheep
    pos = np.stack([runs.run(r)[0] for r in range(4)])
    vel = np.stack([runs.run(r)[1] for r in range(4)])
    for r in range(4):
        np.testing.assert_allclose(M.cohesion(pos, N)[r], M.cohesion(pos[r], N))
        np.testing.assert_allclose(M.polarization_from_vel(vel, N)[r], M.polarization_from_vel(vel[r], N))
        for a, b in zip(M.frame_observables(pos, vel, N), M.frame_observables(pos[r], vel[r], N)):
            np.testing.assert_allclose(a[r], b)
        for a, b in zip(M.viewing_angles(pos, vel, N), M.viewing_angles(pos[r], vel[r], N)):
            np.testing.assert_allclose(a[r], b)


def test_legacy_protocol_reproduces_paper_pipeline(drives):
    st = PR.hierarchy_stats(PR.hierarchy(PR.segments_legacy(drives), maxlag_s=2.0, cmin=0.5, signed=True))
    assert st["binmean_pearson"] == pytest.approx(0.874, abs=1e-3)  # results/summary.json
    assert st["dog_rank"] == 15
    # individual-level association is much weaker than the bin-mean figure
    assert st["spearman"] == pytest.approx(0.365, abs=1e-3)
    assert st["median_lag_s"] == pytest.approx(0.5)


def test_slow_samples_get_zero_heading():
    pos = np.zeros((3, 2, 5))
    pos[0, 0] = [0, 1, 2, 2.001, 3]  # 1 m per step, then nearly still
    seg = PR.from_positions(pos, dt=1.0, n_sheep=2)
    assert np.allclose(seg.head[0, :, 1], [1, 0])
    assert np.all(seg.head[0, :, 3] == 0)
    assert np.all(seg.head[1] == 0)


def test_batched_summaries_match_single(runs):
    N = runs.params.n_sheep
    pos = np.stack([runs.run(r)[0] for r in range(4)])
    batch = PR.summaries(PR.from_positions(pos, 1.0, N), chunk=3)  # also exercises chunking
    for r in range(4):
        single = PR.summaries(PR.from_positions(pos[r], 1.0, N))
        for k in PR.SUMMARY_KEYS:
            np.testing.assert_allclose(batch[k][r], single[k], equal_nan=True, err_msg=k)


def test_surrogate_shapes(drives):
    segs = PR.segments_from_drives(drives[:4], dt=0.2)
    out = PR.timeshift_surrogate(segs, n_surr=3, seed=0)
    assert out["spearman"].shape == (3,) and out["edges_per_segment"].shape == (3,)


def test_replay_windows_align_with_data_windows(drives):
    from shepherd import scenario as SC
    from shepherd.model import Params

    real = SC.window_segments(drives[:5], dt=0.5, horizon=10.0, burn=2.0)
    sim = SC.replay_windows(drives[:5], Params(), n_runs=2, seed=0, dt=0.5, horizon=10.0, burn=2.0)
    assert [s.label for s in real] == [s.label for s in sim]
    for r, s in zip(real, sim):
        assert s.pos.shape == (2, *r.pos.shape)
        np.testing.assert_allclose(s.pos[:, -1], np.broadcast_to(r.pos[-1], s.pos[:, -1].shape))  # dog replayed


def test_long_summaries_detect_a_dispersing_flock():
    T, N = 30, 14
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (N, 2))
    pos = np.zeros((N + 1, 2, T))
    for t in range(T):
        grow = 1 + 0.2 * t  # flock spreads out over time
        pos[:N, :, t] = base * grow + np.array([t * 1.0, 0])
    pos[N, :, :] = pos[:N].mean(0) - np.array([[5.0], [0.0]])
    seg = PR.from_positions(pos, 1.0, N)
    out = PR.long_summaries(seg)
    assert out["C_drift"] > 3 and out["frac_far"] > 0.2
    still = PR.from_positions(np.broadcast_to(pos[..., :1], pos.shape) + np.linspace(0, 1, T), 1.0, N)
    assert abs(PR.long_summaries(still)["C_drift"]) < 1e-9


def test_closed_loop_drives_follow_real_drives(drives):
    from shepherd import scenario as SC
    from shepherd.model import Params

    sim = SC.closed_loop_drives(drives, Params(), n_runs=2, seed=0, dt=1.0,
                                dog={"l_drive": 4.0, "flank_sigma": 2.0, "flank_tau": 2.7, "tau_d": 2.0})
    real = SC.real_long_segments(drives, dt=1.0)
    assert [s.label for s in sim] == [s.label for s in real]
    for s, r in zip(sim, real):
        assert s.pos.shape == (2, *r.pos.shape)
