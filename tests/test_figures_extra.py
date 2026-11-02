"""The additional figures are drawn from the result files and refuse inconsistent input."""
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import make_figures_extra as mfe  # noqa: E402

RES = os.path.join(ROOT, "results", "benchmarks")


class TestFiguresExtra(unittest.TestCase):
    def test_all_five_figures_are_written(self):
        out = tempfile.mkdtemp()
        try:
            mfe.fig0(out)
            for f in (mfe.fig7, mfe.fig8, mfe.fig9, mfe.fig10):
                f(RES, out)
            for name in ("fig0_overview", "fig7_scale_12cities", "fig8_optimality_gap", "fig9_offgrid_cost", "fig10_severity"):
                for ext in (".png", ".pdf"):
                    self.assertGreater(os.path.getsize(os.path.join(out, name + ext)), 5000)
        finally:
            shutil.rmtree(out)

    def test_the_illustrative_network_is_a_real_certified_spanner_with_removed_edges(self):
        import run_spanner_benchmark as core
        G, _ = mfe.toy_network()
        H = core.delay_bounded_greedy_spanner(G, 2.0, show_progress=False)
        self.assertTrue(core.temporal_stretch_certificate(G, H, 2.0)["certified"])
        self.assertTrue(0 < H.number_of_edges() < G.number_of_edges())

    def test_an_uncertified_spanner_stops_the_figure(self):
        for name, fn in (("large_scale_3500m.csv", mfe.fig7), ("severity_sensitivity.csv", mfe.fig10)):
            tmp = tempfile.mkdtemp()
            try:
                d = pd.read_csv(os.path.join(RES, name))
                d.loc[0, "delay_bounded_certified"] = False
                d.to_csv(os.path.join(tmp, name), index=False)
                with self.assertRaises(AssertionError):
                    fn(tmp, tmp)
            finally:
                shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
