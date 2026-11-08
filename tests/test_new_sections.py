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


def min_frame(**over):
    row = dict(city="A", real_osm=True, profile="hourly", edges=100, t=2.0, greedy_retention_pct=90.0, removable_edges=2,
               removable_pct_of_spanner=2.2, pruned_retention_pct=88.0, greedy_is_minimal=False, full_certificate_checks=2)
    row.update(over)
    return pd.DataFrame([row, dict(row, t=3.0, removable_edges=0, removable_pct_of_spanner=0.0, greedy_is_minimal=True)])


def exact_frame(**over):
    base = dict(city="A", real_osm=True, t=2.0, nodes=10, edges=30, skipped=False, forced=20, free=6, optimum=24,
                greedy=25, greedy_pruned=24)
    base.update(over)
    return pd.DataFrame([dict(base, window=0), dict(base, window=1, greedy=24),
                         dict(base, window=2, skipped=True, forced=None, free=None, optimum=None, greedy=None,
                              greedy_pruned=None)])


def emer_frame(over_intact=0.0):
    rows = []
    for sc in ("intact", "training faults", "held-out faults"):
        for sp in ("intact-network spanner", "fault-tolerant spanner"):
            rows.append(dict(city="A", real_osm=True, t=2.0, hour=17, stations=5, spanner=sp, scenario=sc, regions=1,
                             pairs=100, unreachable_pct=1.0 if sp.startswith("intact") and sc != "intact" else 0.0,
                             mean_ratio=1.1, p95_ratio=1.5, max_ratio=1.9,
                             over_t_pct=over_intact if sc == "intact" else 3.0))
    return pd.DataFrame(rows)


class TestMoreSections(unittest.TestCase):
    def test_minimality_section(self):
        mod = load()
        s = mod.minimality_section(min_frame())
        self.assertIn("with the hourly profile the construction is edge-minimal in 1 of 2", s)
        old = min_frame().drop(columns=["profile"])                      # result files written before the profile column
        self.assertIn("hourly profile", mod.minimality_section(old))
        with self.assertRaises(AssertionError):
            mod.minimality_section(min_frame(real_osm=False))

    def test_road_exact_section_counts_and_skips(self):
        mod = load()
        s = mod.road_exact_section(exact_frame())
        self.assertIn("(1 here)", s)
        self.assertIn("optimal in 50% of the cases", s)
        with self.assertRaises(AssertionError):
            mod.road_exact_section(exact_frame(optimum=19))        # optimum below the forced count

    def test_emergency_section_requires_certificate_on_the_intact_network(self):
        mod = load()
        self.assertIn("never exceeds t", mod.emergency_section(emer_frame()))
        with self.assertRaises(AssertionError):
            mod.emergency_section(emer_frame(over_intact=0.5))

    def test_directedness_lightness_edge_order_sections(self):
        mod = load()
        d = pd.DataFrame([dict(city=f"c{i}", source="registry", real_osm=True, arcs_directed=100, arcs_two_way=110,
                               one_way_pct=10.0, t=2.0, forced_directed=90.0, retention_directed=92.0, capture_directed=0.8,
                               forced_two_way=88.0, retention_two_way=91.0, capture_two_way=0.7, diff_forced=2.0,
                               diff_retention=1.0) for i in range(3)])
        self.assertIn("+1.00", mod.directedness_section(d))
        with self.assertRaises(AssertionError):
            mod.directedness_section(d.assign(real_osm=False))
        l = pd.DataFrame([dict(city="A", source="registry", real_osm=True, t=2.0, segments=50, total_km=10.0, mst_km=4.0,
                               kept_length_pct=90.0, kept_count_pct=91.0, forced_length_pct=85.0, lightness_graph=2.5,
                               lightness_spanner=2.25, lightness_lower_bound=2.1, capture_length=0.66)])
        self.assertIn("66%", mod.lightness_section(l))
        o = pd.DataFrame([dict(city=c, real_osm=True, edges=100, t=2.0, order=n, seed=0, retention_pct=r,
                               pruned_retention_pct=r - 1) for c in ("A", "B")
                          for n, r in (("temporal_max", 90.0), ("random", 92.0))])
        s = mod.edge_order_section(o)
        self.assertIn("+2.00", s)
        self.assertIn("+0.00", s)

    def test_sections_are_skipped_without_files_and_inserted_in_order(self):
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            min_frame().to_csv(os.path.join(d, "minimality.csv"), index=False)
            emer_frame().to_csv(os.path.join(d, "emergency_response.csv"), index=False)
            mod.BEN = d
            out = mod.render("x\n\n" + mod.ANCHOR + "\n")
            self.assertLess(out.index("5n."), out.index("5p."))
            self.assertNotIn("5o.", out)
            self.assertEqual(mod.render(out), out)


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
            self.assertEqual(mod.render(changed), once)             # a hand edit inside the section is overwritten

    def test_numbers_in_the_introduction_are_refreshed(self):
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            mod.BEN = d
            text = "intro\n\n" + mod.ANCHOR + "\n\nrest\n"
            rf_frame(regions_train=10).to_csv(os.path.join(d, "region_fault.csv"), index=False)
            first = mod.render(text)
            self.assertIn("family of 10 disc faults", first)
            rf_frame(regions_train=20).to_csv(os.path.join(d, "region_fault.csv"), index=False)
            second = mod.render(first)
            self.assertIn("family of 20 disc faults", second)
            self.assertNotIn("family of 10 disc faults", second)
            self.assertEqual(second.count("### 5l."), 1)
            self.assertTrue(second.endswith("rest\n"))
            self.assertIn(mod.ANCHOR, second)


if __name__ == "__main__":
    unittest.main()
