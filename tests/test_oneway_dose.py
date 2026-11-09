"""One-way dose-response: nested doses, endpoints, denominators."""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import directedness as dr  # noqa: E402
import oneway_dose as od  # noqa: E402
import retention_diagnostics as rd  # noqa: E402


def city():
    spec = [c for c in core.CITY_REGISTRY if c["name"] == "Manhattan"][0]
    return core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)


class TestDose(unittest.TestCase):
    def test_doses_are_nested_and_reach_the_two_way_version(self):
        G = city()
        order = od.dose_order(G, 7)
        prev = set(G.edges())
        for f in od.DOSES:
            G2, k = od.with_dose(G, order, f)
            now = set(G2.edges())
            self.assertTrue(prev <= now)
            self.assertEqual(len(now) - G.number_of_edges(), k)
            prev = now
        self.assertEqual(set(od.with_dose(G, order, 0.0)[0].edges()), set(G.edges()))
        self.assertEqual(set(od.with_dose(G, order, 1.0)[0].edges()), set(dr.two_way(G).edges()))

    def test_order_is_seeded_and_covers_exactly_the_one_way_arcs(self):
        G = city()
        a, b, c = od.dose_order(G, 7), od.dose_order(G, 7), od.dose_order(G, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(set(a), {(u, v) for u, v in G.edges() if not G.has_edge(v, u)})

    def test_rows_endpoints_match_the_directedness_analysis(self):
        G = city()
        rows = pd.DataFrame(od.city_rows(G, "X", "registry", [2.0], 7))
        zero = rows[rows.dose == 0.0].iloc[0]
        one = rows[rows.dose == 1.0].iloc[0]
        a = dr.measure(G, [2.0])[2.0]
        b = dr.measure(dr.two_way(G), [2.0])[2.0]
        self.assertAlmostEqual(zero.forced_all_pct, a["forced"])
        self.assertAlmostEqual(zero.retention_all_pct, a["retention"])
        self.assertAlmostEqual(one.forced_all_pct, b["forced"])
        self.assertAlmostEqual(one.retention_all_pct, b["retention"])
        self.assertAlmostEqual(zero.forced_original_pct, zero.forced_all_pct)     # no arcs added at dose 0
        self.assertEqual(int(zero.arcs_added), 0)


if __name__ == "__main__":
    unittest.main()
