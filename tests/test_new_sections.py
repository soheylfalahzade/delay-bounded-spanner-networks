"""README sections 5l / 5m are generated from result files, never hand-written, and assertions stop stale claims."""
import importlib.util
import os
import tempfile
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load():
    spec = importlib.util.spec_from_file_location("uns", os.path.join(ROOT, "scripts", "update_readme_new_sections.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rf_frame(**over):
    row = dict(city="A", edges=100, t=2.0, regions_train=10, regions_heldout=20, nominal_retention_pct=90.0,
               nominal_forced_lb_pct=88.0, fault_forced_lb_pct=89.0, fault_tolerant_retention_pct=91.0,
               nominal_fails_train_regions_pct=40.0, nominal_mean_violating_edge_pct=0.2,
               nominal_fails_heldout_regions_pct=35.0, ft_fails_train_regions_pct=0.0, ft_fails_heldout_regions_pct=20.0)
    row.update(over)
    return pd.DataFrame([row])


def bh_frame():
    base = dict(city="A", edges=100, true_bridge_pct=5.0, forced_any_pct=90.0, forced_all_hours_pct=80.0,
                forced_some_hours_pct=10.0, forced_free_flow_pct=85.0, time_only_forced_pct=5.0,
                static_only_forced_pct=0.0, peak_not_night_pct=4.0, share_of_forced_that_is_time_dependent=0.1)
    return pd.DataFrame([dict(base, t=2.0), dict(base, t=3.0, forced_any_pct=70.0)])


class TestNewSections(unittest.TestCase):
    def test_region_fault_section_uses_the_data(self):
        mod = load()
        s = mod.region_fault_section(rf_frame())
        self.assertIn("1.00 to 1.00 points", s)           # 91 - 90
        self.assertIn("40% to 40%", s)
        self.assertIn("<!-- BEGIN:region_fault -->", s)

    def test_failed_certification_stops_the_renderer(self):
        mod = load()
        with self.assertRaises(AssertionError):
            mod.region_fault_section(rf_frame(ft_fails_train_regions_pct=5.0))

    def test_bound_ordering_violation_stops_the_renderer(self):
        mod = load()
        with self.assertRaises(AssertionError):
            mod.region_fault_section(rf_frame(fault_tolerant_retention_pct=85.0))

    def test_by_hour_section_numbers(self):
        mod = load()
        s = mod.by_hour_section(bh_frame())
        self.assertIn("At t = 2, 90.0% of the edges are forced at some hour, 80.0% at every hour, and 5.0% are forced only", s)

    def test_render_is_idempotent_and_skips_missing_csv(self):
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            rf_frame().to_csv(os.path.join(d, "region_fault.csv"), index=False)
            mod.BEN = d
            text = "intro\n\n" + mod.ANCHOR + "\n\nrest\n"
            once = mod.render(text)
            self.assertIn("5l. Region faults", once)
            self.assertNotIn("5m. Forced edges by hour", once)     # no CSV, no section
            self.assertEqual(mod.render(once), once)
            changed = once.replace("40% to 40%", "1% to 1%")
            self.assertEqual(mod.render(changed), once)             # a hand edit inside the block is overwritten


if __name__ == "__main__":
    unittest.main()
