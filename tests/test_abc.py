import numpy as np

from shepherd import abc as ABC


def toy(n, seed):
    """Summaries = (a + b, a - b, log c) + noise; truth a=1, b=0.5, c=2."""
    priors = {"a": ABC.Prior(0.1, 5.0), "b": ABC.Prior(-2.0, 2.0, log=False), "c": ABC.Prior(0.5, 8.0)}
    th = ABC.sobol_prior(priors, n, seed)
    rng = np.random.default_rng(seed)
    sims = np.stack([th["a"] + th["b"], th["a"] - th["b"], np.log(th["c"])], 1) + rng.normal(0, 0.05, (n, 3))
    return priors, th, sims


def test_prior_transforms_round_trip():
    for pr in (ABC.Prior(0.2, 3.0), ABC.Prior(3.0, 25.0, log=False)):
        x = pr.from_unit(np.linspace(0.01, 0.99, 50))
        np.testing.assert_allclose(pr.from_z(pr.to_z(x)), x, rtol=1e-9)


def test_regression_abc_recovers_toy_truth():
    priors, th, sims = toy(20000, 0)
    obs = np.array([1.5, 0.5, np.log(2.0)])
    post = ABC.fit(th, sims, obs, ABC.mad_scale(sims), priors, accept=0.02)
    for k, truth in {"a": 1.0, "b": 0.5, "c": 2.0}.items():
        lo, hi = np.percentile(post.samples[k], [5, 95])
        assert lo <= truth <= hi, (k, lo, hi)
        assert hi - lo < 0.5 * (priors[k].hi - priors[k].lo)
    s = post.summary(priors)
    assert all(v["contraction"] > 0.8 for v in s.values())


def test_aggregate_weights_and_masks():
    table = np.array([[[1.0, np.nan], [3.0, 4.0]]])  # (1 set, 2 windows, 2 stats)
    np.testing.assert_allclose(ABC.aggregate(table, [1.0, 3.0]), [[2.5, 4.0]])
    np.testing.assert_allclose(ABC.aggregate(table, [1.0, 3.0], mask=[True, False]), [[1.0, np.nan]])


def test_coverage_is_calibrated_on_toy():
    priors, th, sims = toy(8000, 1)
    cov = ABC.coverage(th, sims, priors, n_test=40, accept=0.03, seed=0)
    assert all(0.7 <= v <= 1.0 for v in cov.values()), cov
