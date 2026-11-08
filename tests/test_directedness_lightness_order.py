"""Directedness, lightness and edge-order analyses."""
import os
import sys
import unittest

import networkx as nx
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import run_spanner_benchmark as core  # noqa: E402
import directedness as dr  # noqa: E402
import lightness as lt  # noqa: E402
import edge_order as eo  # noqa: E402
import minimality as mn  # noqa: E402
from test_forced_by_hour import unit_grid  # noqa: E402


def small_city(name="Manhattan"):
    spec = [c for c in core.CITY_REGISTRY if c["name"] == name][0]
    return core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)


class TestDirectedness(unittest.TestCase):
    def test_two_way_adds_exactly_the_missing_reverse_arcs(self):
        G = small_city()
        G2 = dr.two_way(G)
        missing = sum(1 for u, v in G.edges() if not G.has_edge(v, u))
        self.assertEqual(G2.number_of_edges(), G.number_of_edges() + missing)
        self.assertEqual(dr.one_way_pct(G2), 0.0)
        for u, v in G.edges():
            self.assertTrue(G2.has_edge(u, v) and G2.has_edge(v, u))

    def test_one_way_share_of_a_directed_cycle_is_100(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        self.assertEqual(dr.one_way_pct(G), 100.0)

    def test_already_two_way_network_is_unchanged(self):
        G = unit_grid(4)
        self.assertEqual(dr.two_way(G).number_of_edges(), G.number_of_edges())
        a = pd.DataFrame(dr.measure(G, [2.0, 3.0])).T
        b = pd.DataFrame(dr.measure(dr.two_way(G), [2.0, 3.0])).T
        pd.testing.assert_frame_equal(a, b)                                 # NaN-aware comparison

    def test_rows_and_correlations(self):
        G = small_city()
        rows = dr.city_rows(G, "X", "registry", [2.0])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertAlmostEqual(r["diff_retention"], r["retention_directed"] - r["retention_two_way"])
        df = pd.DataFrame([dict(rows[0], city=f"c{i}", one_way_pct=float(i), diff_forced=float(i), diff_retention=-float(i))
                           for i in range(10)])
        c = dr.correlations(df)
        self.assertAlmostEqual(c[c.target == "diff_forced"].spearman_rho.iloc[0], 1.0)
        self.assertAlmostEqual(c[c.target == "diff_retention"].spearman_rho.iloc[0], -1.0)


class TestLightness(unittest.TestCase):
    def test_unit_grid_values(self):
        n = 5
        G = unit_grid(n)
        m = lt.measure(G, 2.99)
        self.assertEqual(m["segments"], 2 * n * (n - 1))
        self.assertAlmostEqual(m["mst_km"] * 1000.0, n * n - 1)
        self.assertAlmostEqual(m["kept_length_pct"], 100.0)            # nothing is removable below t = 3
        self.assertAlmostEqual(m["forced_length_pct"], 100.0)
        self.assertAlmostEqual(m["lightness_spanner"], m["lightness_graph"])
        m3 = lt.measure(G, 3.0)
        self.assertLess(m3["kept_length_pct"], 100.0)
        self.assertGreaterEqual(m3["lightness_spanner"], 1.0 - 1e-9)   # a connected spanner weighs at least the MST
        self.assertAlmostEqual(m3["forced_length_pct"], 0.0)

    def test_invariants_on_a_synthetic_network(self):
        G = small_city("Yazd")
        for t in (1.5, 2.0, 3.0):
            m = lt.measure(G, t)
            self.assertLessEqual(m["forced_length_pct"], m["kept_length_pct"] + 1e-9)
            self.assertLessEqual(m["lightness_lower_bound"], m["lightness_spanner"] + 1e-9)
            self.assertLessEqual(m["lightness_spanner"], m["lightness_graph"] + 1e-9)
            self.assertTrue(0.0 <= m["capture_length"] <= 1.0)

    def test_segments_count_two_way_street_once(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", length=10.0)
        G.add_edge("b", "a", length=10.0)
        G.add_edge("b", "c", length=5.0)
        self.assertEqual(lt.segments(G), {frozenset(("a", "b")): 10.0, frozenset(("b", "c")): 5.0})
        self.assertEqual(lt.mst_weight(lt.segments(G)), 15.0)


class TestEdgeOrder(unittest.TestCase):
    def test_default_order_reproduces_the_library_spanner(self):
        G = small_city("Yazd")
        kept, _ = eo.greedy_in_order(G, 2.0, eo.make_order(G, "temporal_max"))
        H = core.delay_bounded_greedy_spanner(G, 2.0, show_progress=False)
        self.assertEqual(set(kept), set(H.edges()))

    def test_every_order_is_certified_and_pruning_never_grows(self):
        G = small_city("Manhattan")
        for t in (2.0, 3.0):
            for name in ("temporal_max", "temporal_mean", "free_flow", "descending_max", "random"):
                kept, adj = eo.greedy_in_order(G, t, eo.make_order(G, name, 1))
                self.assertTrue(mn._certified(G, adj, set(), t), name)
                H = nx.DiGraph()
                H.add_nodes_from(G.nodes(data=True))
                for u, v in kept:
                    H.add_edge(u, v, **G[u][v])
                removed, _ = mn.reverse_delete(G, H, t)
                self.assertLessEqual(len(kept) - len(removed), len(kept))

    def test_orders_are_permutations_and_random_is_seeded(self):
        G = small_city("Yazd")
        base = sorted((u, v) for u, v in G.edges())
        for name in ("temporal_max", "temporal_mean", "free_flow", "descending_max", "random"):
            self.assertEqual(sorted((u, v) for u, v, _ in eo.make_order(G, name, 3)), base)
        a = [(u, v) for u, v, _ in eo.make_order(G, "random", 3)]
        b = [(u, v) for u, v, _ in eo.make_order(G, "random", 3)]
        c = [(u, v) for u, v, _ in eo.make_order(G, "random", 4)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_every_order_on_the_grid_is_certified_and_spans(self):
        G = unit_grid(5)
        for name in ("temporal_max", "free_flow", "descending_max", "random"):
            kept, adj = eo.greedy_in_order(G, 3.0, eo.make_order(G, name, 2))
            self.assertTrue(mn._certified(G, adj, set(), 3.0), name)
            self.assertGreaterEqual(len(kept), 24)                          # strongly connected on 25 nodes

    def test_directed_greedy_need_not_be_minimal_even_for_a_static_profile(self):
        # The undirected argument (Filtser and Solomon) does not cover digraphs: with unit weights the greedy rule applied
        # to the bidirectional grid in some order leaves removable arcs.  This documents the fact; it is not a bug.
        G = unit_grid(5)
        removable = []
        for name in ("temporal_max", "free_flow", "descending_max", "random"):
            kept, _ = eo.greedy_in_order(G, 3.0, eo.make_order(G, name, 2))
            H = nx.DiGraph()
            H.add_nodes_from(G.nodes(data=True))
            for u, v in kept:
                H.add_edge(u, v, **G[u][v])
            removable.append(len(mn.reverse_delete(G, H, 3.0)[0]))
        self.assertGreaterEqual(max(removable), 1)


if __name__ == "__main__":
    unittest.main()
