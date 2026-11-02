"""Results of an interrupted or repeated larger-radius run are merged, not overwritten."""
import os
import unittest
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from large_scale_run import merge_results  # noqa: E402


def _rows(city, v):
    return [dict(city=city, t=t, delay_bounded_retention_pct=v) for t in (1.5, 2.0)]




class TestLargeScaleMerge(unittest.TestCase):
    def test_new_cities_are_added_and_old_ones_kept(self):
        old = pd.DataFrame(_rows("Isfahan", 98.0) + _rows("Tokyo", 90.0))
        out = merge_results(old, _rows("Cairo", 95.0))
        assert sorted(out.city.unique()) == ["Cairo", "Isfahan", "Tokyo"] and len(out) == 6


    def test_a_recomputed_city_replaces_its_old_rows(self):
        old = pd.DataFrame(_rows("Isfahan", 98.0) + _rows("Tokyo", 90.0))
        out = merge_results(old, _rows("Tokyo", 91.0))
        assert len(out) == 4 and set(out[out.city == "Tokyo"].delay_bounded_retention_pct) == {91.0}


    def test_empty_inputs(self):
        assert len(merge_results(None, _rows("A", 1.0))) == 2
        assert len(merge_results(pd.DataFrame(_rows("A", 1.0)), [])) == 2


if __name__ == "__main__":
    unittest.main()
