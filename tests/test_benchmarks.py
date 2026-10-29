"""Tests for benchmarks/literature_benchmarks.py (Baswana-Sen, reduction to greedy, Yao/Theta bounds)."""
import math, os, sys, unittest
import networkx as nx
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import literature_benchmarks as lb  # noqa: E402
import run_spanner_benchmark as core  # noqa: E402


class TestBaswanaSen(unittest.TestCase):
    def test_stretch_bound(self):
        for seed in range(6):
            Gu = lb.random_graph("ER", 60, seed)
            n = Gu.number_of_nodes()
            full = lb.sp_dijkstra(lb.csr_from_edges(n, [(u, v, d["weight"]) for u, v, d in Gu.edges(data=True)], False), directed=False)
            for k in (2, 3):
                E = lb.baswana_sen(Gu, k, seed)
                st = lb.exact_stretch_undirected(n, [(a, b, Gu[a][b]["weight"]) for a, b in E], full)
                self.assertLessEqual(st, 2 * k - 1 + 1e-9)


class TestReduction(unittest.TestCase):
    def test_constant_profile_equals_path_greedy(self):
        rng = np.random.default_rng(1)
        for kind in ("uniform", "clustered"):
            P = lb.make_points(kind, 30, rng)
            Gu, _ = lb.complete_euclid(P)
            for t in (1.2, 1.7, 2.5):
                H = core.delay_bounded_greedy_spanner(lb.digraph_from_undirected(Gu), t, show_progress=False)
                ours = {(min(u, v), max(u, v)) for u, v in H.edges()}
                self.assertEqual(ours, lb.reference_path_greedy(Gu, t))


class TestConeGraphs(unittest.TestCase):
    def test_yao_theta_theoretical_bounds(self):
        rng = np.random.default_rng(2)
        P = lb.make_points("uniform", 60, rng)
        _, D = lb.complete_euclid(P)
        yb = 1 / (1 - 2 * math.sin(math.pi / 12))
        tb = 1 / (math.cos(2 * math.pi / 12) - math.sin(2 * math.pi / 12))
        self.assertLessEqual(lb.spanner_metrics(P, D, lb.yao_or_theta(P, 12, False))["stretch"], yb + 1e-9)
        self.assertLessEqual(lb.spanner_metrics(P, D, lb.yao_or_theta(P, 12, True))["stretch"], tb + 1e-9)


class TestTheorem1BruteForce(unittest.TestCase):
    def test_certified_implies_all_pairs_all_hours(self):
        import random
        rng = random.Random(5)
        for _ in range(8):
            G = lb.random_temporal_digraph(rng.randint(8, 14), 0.4, rng)
            for t in (1.3, 2.0):
                H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
                self.assertTrue(core.temporal_stretch_certificate(G, H, t)["certified"])
                self.assertTrue(lb.brute_force_ok(G, H, t))

    def test_union_of_snapshots_is_certified(self):
        import random
        rng = random.Random(9)
        G = lb.random_temporal_digraph(12, 0.4, rng)
        Hu = lb.union_of_snapshots_spanner(G, 1.5)
        self.assertTrue(core.temporal_stretch_certificate(G, Hu, 1.5)["certified"])
        self.assertTrue(lb.brute_force_ok(G, Hu, 1.5))


if __name__ == "__main__":
    unittest.main()
