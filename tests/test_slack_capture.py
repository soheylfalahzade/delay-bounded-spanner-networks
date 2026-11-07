"""Slack capture: share of the removable slack that the construction removes."""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import slack_capture as sc  # noqa: E402


class TestSlackCapture(unittest.TestCase):
    def test_formula(self):
        got = sc.capture(pd.Series([98.0, 90.0, 100.0]), pd.Series([96.0, 80.0, 100.0]))
        self.assertAlmostEqual(got.iloc[0], 0.5)      # removed 2 of 4 removable
        self.assertAlmostEqual(got.iloc[1], 0.5)      # removed 10 of 20 removable
        self.assertTrue(pd.isna(got.iloc[2]))         # no slack: undefined, not 0 or 1

    def test_keeping_everything_has_zero_capture_even_when_the_gap_is_tiny(self):
        # the point of the metric: retention 100 vs bound 98 is a 2-point gap but captures nothing
        self.assertEqual(sc.capture(pd.Series([100.0]), pd.Series([98.0])).iloc[0], 0.0)

    def test_committed_results_respect_the_bound(self):
        path = os.path.join(ROOT, "results", "benchmarks", "slack_capture.csv")
        if not os.path.exists(path):
            self.skipTest("run benchmarks/slack_capture.py first")
        df = pd.read_csv(path)
        self.assertTrue((df.removed_pct <= df.removable_slack_pct + 1e-6).all())
        self.assertTrue(df.capture.dropna().between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
