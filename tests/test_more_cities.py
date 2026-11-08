"""More cities: BH adjustment, correlations, real-data-only loader, row construction."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import more_cities as mc  # noqa: E402


class TestMoreCities(unittest.TestCase):
    def test_bh_adjust_known_values(self):
        p = np.array([0.01, 0.04, 0.03, 0.20])
        adj = mc.bh_adjust(p)
        # sorted p: .01 .03 .04 .20 -> raw adj .04 .04 .0533 .20 ; monotone from the top
        self.assertTrue(np.allclose(adj, [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6))
        self.assertTrue((adj >= p - 1e-12).all())

    def test_correlations_detect_a_planted_monotone_relation(self):
        rng = np.random.default_rng(0)
        n = 20
        x = rng.normal(size=n)
        df = pd.DataFrame(dict(t=2.0, retention_pct=x * 2 + 50, capture=rng.normal(size=n), bridge_pct=x,
                               true_bridge_pct=rng.normal(size=n), mean_degree=rng.normal(size=n),
                               circuity_mean=rng.normal(size=n), orientation_entropy=rng.normal(size=n),
                               edge_length_cv=rng.normal(size=n)))
        c = mc.correlations(df, [2.0])
        hit = c[(c.target == "retention_pct") & (c.descriptor == "bridge_pct")].iloc[0]
        self.assertAlmostEqual(hit.spearman_rho, 1.0)
        self.assertTrue(bool(hit["significant_BH_0.05"]))
        self.assertEqual(int(c["significant_BH_0.05"].sum()), 1)

    def test_too_few_cities_gives_no_correlations(self):
        df = pd.DataFrame(dict(t=[2.0] * 3, retention_pct=[1, 2, 3], capture=[1, 2, 3], bridge_pct=[1, 2, 3],
                               true_bridge_pct=[1, 2, 3], mean_degree=[1, 2, 3], circuity_mean=[1, 2, 3],
                               orientation_entropy=[1, 2, 3], edge_length_cv=[1, 2, 3]))
        self.assertEqual(len(mc.correlations(df, [2.0])), 0)

    def test_failed_download_returns_none_and_nothing_synthetic(self):
        original = core.fetch_osm_network
        core.fetch_osm_network = lambda center, dist: None
        try:
            self.assertIsNone(mc.load_real("Nowhere", 0.0, 0.0))
        finally:
            core.fetch_osm_network = original

    def test_city_rows_respect_the_bound_and_capture_range(self):
        spec = [c for c in core.CITY_REGISTRY if c["name"] == "Yazd"][0]
        G = core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)
        rows = mc.city_rows(G, "Yazd", "Iran", "registry", [2.0, 3.0])
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertGreaterEqual(r["retention_pct"] + 1e-9, r["forced_lower_bound_pct"])
            self.assertTrue(0.0 <= r["capture"] <= 1.0)

    def test_added_city_list_is_well_formed(self):
        names = [c[0] for c in mc.ADDED_CITIES]
        self.assertEqual(len(names), len(set(names)))
        reg = {c["name"] for c in core.CITY_REGISTRY}
        self.assertFalse(reg & set(names))
        for _, _, lat, lon, _ in mc.ADDED_CITIES:
            self.assertTrue(-90 <= lat <= 90 and -180 <= lon <= 180)


if __name__ == "__main__":
    unittest.main()
