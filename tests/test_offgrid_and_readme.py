"""Off-grid robust certificate (Theorem 3) and README-number consistency."""
import csv, os, random, re, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import literature_benchmarks as lb  # noqa: E402
import offgrid_guarantee as og  # noqa: E402


class TestOffGridRobust(unittest.TestCase):
    def test_robust_construction_has_no_offgrid_violation(self):
        rng = random.Random(4)
        for _ in range(3):
            G = lb.random_temporal_digraph(rng.randint(10, 14), 0.4, rng)
            og.add_fine_costs(G, 4)
            rho = og.rho_of(G, 4)
            for T in (1.3, 2.0):
                H = core.delay_bounded_greedy_spanner(G, T / rho ** 2, hours=list(range(96)), attr="tt_fine", show_progress=False)
                worst, bad = og.offgrid_violation(G, H, T, 10, rng)
                self.assertEqual(bad, 0)
                self.assertLessEqual(worst, T + 1e-9)


class TestPerEdgeRobust(unittest.TestCase):
    def test_per_edge_construction_has_no_offgrid_violation(self):
        rng = random.Random(8)
        for _ in range(3):
            G = lb.random_temporal_digraph(rng.randint(10, 14), 0.4, rng)
            for T in (1.3, 2.0):
                H, _, _ = og.per_edge_robust_spanner(G, T, 6)
                worst, bad = og.offgrid_violation(G, H, T, 10, rng)
                self.assertEqual(bad, 0)


class TestReadmeNumbers(unittest.TestCase):
    """The README quotes numbers from results/benchmarks; keep them in sync with the CSVs."""
    def test_section_5d_matches_csv(self):
        readme = os.path.join(ROOT, "README.md")
        csvp = os.path.join(ROOT, "results", "benchmarks", "retention_large_t.csv")
        if not (os.path.exists(readme) and os.path.exists(csvp)):
            self.skipTest("README or retention_large_t.csv missing")
        text = open(readme, encoding="utf-8").read()
        if "## 5d." not in text:
            self.skipTest("README section 5d absent")
        sec = text.split("## 5d.")[1].split("\n---")[0]
        rows = re.findall(r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$", sec, flags=re.M)
        self.assertTrue(rows)
        data = {}
        for r in csv.DictReader(open(csvp)):
            data.setdefault(float(r["t"]), []).append((float(r["forced_pct"]), float(r["retention_pct"])))
        for t, f, g, gap in rows:
            vals = data.get(float(t))
            if not vals:
                continue
            mf = sum(x[0] for x in vals) / len(vals)
            mr = sum(x[1] for x in vals) / len(vals)
            self.assertAlmostEqual(float(f), mf, delta=0.06)
            self.assertAlmostEqual(float(g), mr, delta=0.06)
            self.assertAlmostEqual(float(gap), mr - mf, delta=0.06)


if __name__ == "__main__":
    unittest.main()
