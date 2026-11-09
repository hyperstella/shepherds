"""Fast, dependency-only smoke tests (no pytest needed).
Run: python josef/test_repro.py
"""
import unittest
import numpy as np
from model import mcrit_2d, n_herders, run_2d
from simulate_agents import exact_target_goal_fraction


class ReproductionTests(unittest.TestCase):
    def test_critical_mass_and_count(self):
        mass = mcrit_2d(0.01, 6/np.pi, np.pi)
        self.assertAlmostEqual(mass, 0.27738, places=4)
        self.assertEqual(int(np.ceil(n_herders(720, mass))), 277)

    def test_von_mises_target(self):
        self.assertAlmostEqual(exact_target_goal_fraction(6/np.pi),
                               0.796904786, places=8)

    def test_short_stochastic_run(self):
        r = run_2d(NT=720, NH=277, seed=0, T=0.02, M=16, dt=0.01)
        self.assertEqual(r['snap0'][0].shape, (277, 2))
        self.assertEqual(r['T'].shape, (720, 2))
        self.assertTrue(np.isfinite(r['T']).all())
        self.assertTrue(np.isfinite(r['H']).all())
        self.assertEqual(len(r['times']), 3)


if __name__ == '__main__':
    unittest.main()
