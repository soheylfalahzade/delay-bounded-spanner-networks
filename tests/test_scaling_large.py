"""Scaling study: the work-counting replica equals the library, the fit recovers known exponents, the generator is reproducible."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import scaling_large as sl  # noqa: E402
from scaling_fit import fit_exponent  # noqa: E402


class TestScalingLarge(unittest.TestCase):
    def test_replica_keeps_exactly_the_library_edges(self):
        for n, t in ((300, 1.5), (300, 2.0), (600, 3.0)):
            G = sl.synthetic_network(n, seed=n)
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            kept, work = sl.greedy_with_work(G, t)
            self.assertEqual(kept, set(H.edges()))
            self.assertEqual(work["probes"], H.graph["dijkstra_probes"])
            self.assertTrue(0 < work["probes"] <= 24 * G.number_of_edges())
            self.assertGreaterEqual(work["settled"], work["probes"] - G.number_of_edges())

    def test_measure_checks_the_replica(self):
        G = sl.synthetic_network(200, seed=3)
        row = sl.measure(G, 2.0)
        self.assertEqual(row["kept_edges"], round(row["kept_pct"] * row["edges"] / 100))
        self.assertGreater(row["probes_per_edge"], 0.9)

    def test_generator_is_reproducible_and_sized_like_a_road_network(self):
        a, b = sl.synthetic_network(400, seed=5), sl.synthetic_network(400, seed=5)
        self.assertEqual(sorted(a.edges()), sorted(b.edges()))
        self.assertTrue(1.5 < a.number_of_edges() / a.number_of_nodes() < 4.0)

    def test_fit_recovers_known_exponents(self):
        x = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
        for a in (1.0, 1.4):
            slope, lo, hi = fit_exponent(x, 2e-5 * x ** a)
            self.assertAlmostEqual(slope, a, places=6)
            self.assertTrue(lo <= slope <= hi)


if __name__ == "__main__":
    unittest.main()
