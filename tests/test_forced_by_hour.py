"""Hourly decomposition of forced edges, true bridge share, and an analytic anchor on the unit grid."""
import os
import sys
import unittest

import networkx as nx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import forced_by_hour as fb  # noqa: E402


def unit_grid(n: int) -> nx.DiGraph:
    """n x n street grid, every block of length 1, two-way, identical in all 24 hours."""
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            G.add_node((i, j))
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b in G:
                    for x, y in ((a, b), (b, a)):
                        G.add_edge(x, y, tt=[1.0] * 24, free_flow=1.0, length=1.0)
    return G


class TestUnitGridAnalytic(unittest.TestCase):
    """In an undirected unit grid every street lies on a unit square, so the best detour around a removed
    street has length 3 and every edge has detour ratio exactly 3: all edges are forced for t < 3 and none
    for t >= 3.  This is the analytic reason a strict grid cannot be sparsified below t = 3."""

    def test_every_detour_ratio_is_three(self):
        G = unit_grid(5)
        R, rff, edges = fb.hourly_ratios(G, cap=4.0)
        self.assertTrue(np.allclose(R, 3.0))
        self.assertTrue(np.allclose(rff, 3.0))

    def test_forced_share_steps_at_three(self):
        G = unit_grid(5)
        R, rff, _ = fb.hourly_ratios(G, cap=4.0)
        self.assertEqual(fb.classify(R, rff, 2.99)["forced_any_pct"], 100.0)
        self.assertEqual(fb.classify(R, rff, 3.0)["forced_any_pct"], 0.0)

    def test_greedy_keeps_everything_below_three_and_certifies_at_three(self):
        G = unit_grid(5)
        H = core.delay_bounded_greedy_spanner(G, 2.99, show_progress=False)
        self.assertEqual(H.number_of_edges(), G.number_of_edges())
        H3 = core.delay_bounded_greedy_spanner(G, 3.0, show_progress=False)
        self.assertLess(H3.number_of_edges(), G.number_of_edges())
        self.assertTrue(core.temporal_stretch_certificate(G, H3, 3.0)["certified"])


class TestTrueBridge(unittest.TestCase):
    def test_directed_cycle_every_arc_is_a_bridge(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.assertEqual(fb.true_bridge_pct(G), 100.0)

    def test_two_way_cycle_has_no_bridge(self):
        G = nx.DiGraph()
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            G.add_edge(a, b)
            G.add_edge(b, a)
        self.assertEqual(fb.true_bridge_pct(G), 0.0)


class TestClassification(unittest.TestCase):
    def test_categories_are_consistent_on_a_synthetic_network(self):
        spec = [c for c in core.CITY_REGISTRY if c["name"] == "Yazd"][0]
        G = core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)
        R, rff, edges = fb.hourly_ratios(G, cap=3.0003)
        for t in (1.5, 2.0, 3.0):
            c = fb.classify(R, rff, t)
            self.assertGreaterEqual(c["forced_any_pct"] + 1e-9, c["forced_all_hours_pct"])
            self.assertAlmostEqual(c["forced_any_pct"], c["forced_all_hours_pct"] + c["forced_some_hours_pct"], places=9)
            self.assertLessEqual(c["time_only_forced_pct"], c["forced_any_pct"] + 1e-9)
            # forced_any must equal the Theorem-2 count used elsewhere in the repository
            import retention_diagnostics as rd
            edf = rd.per_edge_max_ratio(G, 3.0003)
            self.assertAlmostEqual(c["forced_any_pct"], 100.0 * float((edf.max_ratio > t).mean()), places=9)


if __name__ == "__main__":
    unittest.main()
