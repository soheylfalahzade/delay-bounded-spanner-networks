"""Tests for tools/validate_congestion_shape.py: analytic controls plus negative controls
that prove the checker rejects degenerate or mismatched inputs."""
import importlib.util
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools",
                     "validate_congestion_shape.py")
_spec = importlib.util.spec_from_file_location("validate_congestion_shape", _PATH)
vcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vcs)

HOURS = np.arange(24)
# two-peak weekday speed shape (km/h) used as a known ground truth
TRUTH = 40.0 - 8.0 * np.exp(-((HOURS - 8) ** 2) / 8.0) - 9.0 * np.exp(-((HOURS - 17) ** 2) / 8.0)


def _days(n=20, noise=0.5, seed=1):
    rng = np.random.default_rng(seed)
    return TRUTH + rng.normal(0, noise, size=(n, 24))


class TestCompareShape(unittest.TestCase):
    def test_affine_copy_of_truth_correlates_perfectly(self):
        res = vcs.compare_shape(_days(), 0.5 * TRUTH + 3.0, n_boot=200)
        self.assertGreater(res["pearson"], 0.99)
        self.assertGreater(res["pearson_ci95"][0], 0.95)

    def test_inverted_model_is_strongly_negative(self):
        res = vcs.compare_shape(_days(), -TRUTH, n_boot=200)
        self.assertLess(res["pearson"], -0.99)

    def test_unrelated_model_is_not_high(self):
        rng = np.random.default_rng(7)
        res = vcs.compare_shape(_days(), rng.normal(size=24), n_boot=200)
        self.assertLess(abs(res["pearson"]), 0.8)

    def test_hour_shifted_model_scores_lower_than_aligned(self):
        aligned = vcs.compare_shape(_days(), TRUTH, n_boot=100)["pearson"]
        shifted = vcs.compare_shape(_days(), np.roll(TRUTH, 6), n_boot=100)["pearson"]
        self.assertLess(shifted, aligned - 0.3)

    def test_scale_and_offset_invariance(self):
        a = vcs.compare_shape(_days(), TRUTH, n_boot=50)["pearson"]
        b = vcs.compare_shape(_days(), 3.7 * TRUTH + 11.0, n_boot=50)["pearson"]
        self.assertAlmostEqual(a, b, places=9)

    def test_bootstrap_is_deterministic_for_fixed_seed(self):
        a = vcs.compare_shape(_days(), TRUTH, n_boot=100, seed=3)["pearson_ci95"]
        b = vcs.compare_shape(_days(), TRUTH, n_boot=100, seed=3)["pearson_ci95"]
        self.assertEqual(a, b)

    def test_missing_hours_are_interpolated(self):
        d = _days()
        d[:, 3] = np.nan
        res = vcs.compare_shape(d, TRUTH, n_boot=50)
        self.assertTrue(np.isfinite(res["pearson"]))
        self.assertGreater(res["pearson"], 0.95)


class TestNegativeControls(unittest.TestCase):
    def test_constant_model_rejected(self):
        with self.assertRaises(ValueError):
            vcs.compare_shape(_days(), np.full(24, 30.0))

    def test_constant_measurement_rejected(self):
        with self.assertRaises(ValueError):
            vcs.compare_shape(np.full((10, 24), 25.0), TRUTH)

    def test_wrong_length_model_rejected(self):
        with self.assertRaises(ValueError):
            vcs.compare_shape(_days(), TRUTH[:23])

    def test_nonfinite_model_rejected(self):
        bad = TRUTH.copy()
        bad[5] = np.nan
        with self.assertRaises(ValueError):
            vcs.compare_shape(_days(), bad)

    def test_too_few_days_rejected(self):
        with self.assertRaises(ValueError):
            vcs.compare_shape(_days(n=3), TRUTH)


class TestDailyMatrixAndCsv(unittest.TestCase):
    def _frame(self):
        rows = []
        for d in pd.date_range("2026-09-01", "2026-09-14"):           # Tue .. Mon
            for h in range(24):
                rows.append({"d": d, "h": h, "v": TRUTH[h], "n": 10})
        return pd.DataFrame(rows)

    def test_weekends_are_dropped(self):
        piv = vcs.daily_matrix(self._frame())
        self.assertTrue((piv.index.dayofweek < 5).all())
        self.assertEqual(len(piv), 10)

    def test_sparse_days_are_dropped(self):
        df = self._frame()
        day = pd.Timestamp("2026-09-02")
        df = df[~((df["d"] == day) & (df["h"] < 6))]                   # only 18 hours left
        piv = vcs.daily_matrix(df)
        self.assertNotIn(day, piv.index)

    def test_csv_aggregation_matches_manual_mean(self):
        rows = []
        for d in pd.date_range("2026-09-01", "2026-09-02"):
            for h in range(24):
                for s in (TRUTH[h] - 1.0, TRUTH[h] + 1.0):
                    rows.append({"data_as_of": f"{d.date()}T{h:02d}:15:00.000",
                                 "speed": s, "borough": "Manhattan"})
                rows.append({"data_as_of": f"{d.date()}T{h:02d}:30:00.000",
                             "speed": 999.0, "borough": "Queens"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            out = vcs.fetch_csv(path, "2026-09-01", "2026-09-03", "Manhattan", chunksize=7)
        self.assertEqual(len(out), 48)
        self.assertTrue(np.allclose(out[out["d"] == pd.Timestamp("2026-09-01")].sort_values("h")["v"],
                                    TRUTH))
        self.assertLess(out["v"].max(), 100.0)                         # Queens rows excluded


if __name__ == "__main__":
    unittest.main()
