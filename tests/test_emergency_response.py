"""Emergency response comparison: Dijkstra agrees with networkx, intact ratios respect the certificate."""
import os
import sys
import unittest

import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import emergency_response as er  # noqa: E402
import region_fault as rf  # noqa: E402


def city():
    spec = [c for c in core.CITY_REGISTRY if c["name"] == "Yazd"][0]
    return core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)


class TestEmergency(unittest.TestCase):
    def test_sssp_matches_networkx(self):
        G = city()
        adj = er.adjacency_hour(list(G.edges(data=True)), 17)
        s = sorted(G.nodes(), key=str)[0]
        mine = er.sssp(adj, s)
        ref = nx.single_source_dijkstra_path_length(G, s, weight=lambda u, v, d: d["tt"][17])
        self.assertEqual(set(mine), set(ref))
        for k in ref:
            self.assertAlmostEqual(mine[k], ref[k], places=9)

    def test_intact_ratios_respect_the_certificate(self):
        G = city()
        t = 2.0
        H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        stations = er.pick_stations(G, 4)
        ratios, unreachable, pairs = er.compare(G, set(H.edges()), set(), set(), stations, 17, t)
        self.assertEqual(unreachable, 0)
        self.assertGreater(pairs, 0)
        self.assertLessEqual(max(ratios), t + 1e-9)
        self.assertGreaterEqual(min(ratios), 1.0 - 1e-9)

    def test_stations_are_distinct(self):
        G = city()
        st = er.pick_stations(G, 5)
        self.assertEqual(len(st), len(set(st)))

    def test_fault_tolerant_spanner_respects_the_bound_on_its_training_faults(self):
        G = city()
        t = 2.0
        train = rf.sample_scenarios(G, 4, (0.1, 0.2), seed=2)
        HF = rf.fault_tolerant_greedy(G, t, train)
        stations = er.pick_stations(G, 3)
        for sc in train:
            ratios, unreachable, _ = er.compare(G, set(HF.edges()), sc["removed_nodes"], sc["removed_edges"], stations, 17, t)
            self.assertEqual(unreachable, 0)
            if ratios:
                self.assertLessEqual(max(ratios), t + 1e-9)


if __name__ == "__main__":
    unittest.main()
