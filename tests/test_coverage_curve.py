"""Budget-coverage curve: exactness of the covered area and consistency with the repair."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import run_spanner_benchmark as core  # noqa: E402
import all_discs as ad  # noqa: E402
import coverage_curve as cc  # noqa: E402
from test_all_discs import city  # noqa: E402


@unittest.skipUnless(ad.HAVE_SHAPELY, "shapely not installed")
class TestCoverage(unittest.TestCase):
    T = 2.0

    @classmethod
    def setUpClass(cls):
        cls.G = city(6)
        cls.net = ad.Net(cls.G)
        r = 0.12 * float(cls.G.graph.get("dist", core.DEFAULT_DIST))
        cls.scen, _, _, _ = ad.enumerate_scenarios(cls.G, cls.net, r)
        H0 = core.delay_bounded_greedy_spanner(cls.G, cls.T, show_progress=False)
        cls.H0 = {cls.net.id[e] for e in H0.edges()}
        cls.covered0, cls.by_v, cls.total = cc.repair_sets(cls.net, cls.H0, cls.scen, cls.T, cls.T)
        cls.pts, cls.chosen = cc.curve(cls.covered0, cls.by_v, cls.total)

    def test_start_matches_the_failing_area_of_the_intact_spanner(self):
        stats, _, _ = ad.run_family(self.net, self.H0, self.scen, self.T, repair=False)
        self.assertAlmostEqual(self.pts[0][1], 100.0 - stats["failing_area_pct"], places=6)

    def test_curve_is_monotone_and_reaches_full_coverage(self):
        ks = [k for k, _ in self.pts]
        cs = [c for _, c in self.pts]
        self.assertEqual(ks, sorted(ks))
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(cs, cs[1:])))
        self.assertAlmostEqual(cs[-1], 100.0, places=6)

    def test_every_prefix_covers_at_least_the_claimed_area(self):
        for k, claimed in self.pts[:: max(1, len(self.pts) // 12)]:
            H = self.H0 | set(self.chosen[:k])
            stats, _, _ = ad.run_family(self.net, H, self.scen, self.T, repair=False)
            actual = 100.0 - stats["failing_area_pct"]
            self.assertGreaterEqual(actual, claimed - 1e-9)
        k, claimed = self.pts[-1]
        stats, _, _ = ad.run_family(self.net, self.H0 | set(self.chosen[:k]), self.scen, self.T, repair=False)
        self.assertAlmostEqual(100.0 - stats["failing_area_pct"], 100.0, places=6)

    def test_union_of_repair_sets_is_not_smaller_than_the_sequential_repair(self):
        _, st, _ = ad.run_family(self.net, self.H0, self.scen, self.T, repair=True)
        self.assertGreaterEqual(len(self.chosen), len(st.H) - len(self.H0))

    def test_levels(self):
        self.assertEqual(cc.arcs_for_level(self.pts, 0.0), 0)
        self.assertEqual(cc.arcs_for_level(self.pts, 100.0), self.pts[-1][0])


if __name__ == "__main__":
    unittest.main()
