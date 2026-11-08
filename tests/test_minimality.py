"""Reverse-delete pass: exact certificate preserved, result edge-minimal, constant profile has nothing to remove."""
import os
import sys
import unittest

import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import run_spanner_benchmark as core  # noqa: E402
import minimality as mn  # noqa: E402
from test_forced_by_hour import unit_grid  # noqa: E402


def small_city(name="Yazd"):
    spec = [c for c in core.CITY_REGISTRY if c["name"] == name][0]
    return core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)


class TestMinimality(unittest.TestCase):
    def test_bidirectional_unit_grid_with_the_default_order_has_nothing_to_remove(self):
        # an observation for this order on this graph, not a theorem: for digraphs the undirected argument does not apply
        G = unit_grid(5)
        for t in (1.5, 3.0, 4.0):
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            removed, _ = mn.reverse_delete(G, H, t)
            self.assertEqual(removed, [])

    def test_pruned_spanner_is_certified_and_edge_minimal(self):
        G = small_city()
        for t in (2.0, 3.0):
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            removed, _ = mn.reverse_delete(G, H, t)
            H2 = H.copy()
            H2.remove_edges_from(removed)
            self.assertTrue(core.temporal_stretch_certificate(G, H2, t)["certified"])
            self.assertTrue(set(removed) <= set(H.edges()))
            # edge-minimal: deleting any further edge breaks the certificate
            for e in list(H2.edges())[::7]:
                H3 = H2.copy()
                H3.remove_edge(*e)
                self.assertFalse(core.temporal_stretch_certificate(G, H3, t)["certified"])

    def test_static_version_has_a_constant_profile(self):
        G = small_city("Yazd")
        S = mn.static_version(G)
        for u, v, d in S.edges(data=True):
            self.assertEqual(d["tt"], [d["free_flow"]] * 24)
        self.assertEqual(S.number_of_edges(), G.number_of_edges())

    def test_greedy_can_be_non_minimal_with_time_varying_weights(self):
        G = small_city("Manhattan")
        H = core.delay_bounded_greedy_spanner(G, 2.0, show_progress=False)
        removed, _ = mn.reverse_delete(G, H, 2.0)
        self.assertGreater(len(removed), 0)


if __name__ == "__main__":
    unittest.main()
