"""Region-fault experiment: geometry, certificate on (G-F, H-F), forced-edge bound under faults, fault-tolerant greedy."""
import os
import sys
import unittest

import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
import region_fault as rf  # noqa: E402


def small_city(grid_n=9):
    spec = [c for c in core.CITY_REGISTRY if c["name"] == "Manhattan"][0]
    return core.build_city_network(spec, grid_n=grid_n, force_synthetic=True, use_cache=False)


def faulted_graph(G, sc):
    Gf = nx.DiGraph()
    Gf.add_nodes_from((n, d) for n, d in G.nodes(data=True) if n not in sc["removed_nodes"])
    for u, v, d in rf.faulted_edges(G, sc):
        Gf.add_edge(u, v, **d)
    return Gf


class TestGeometry(unittest.TestCase):
    def test_point_segment_distance(self):
        self.assertAlmostEqual(rf._point_segment_dist(0, 1, -1, 0, 1, 0), 1.0)
        self.assertAlmostEqual(rf._point_segment_dist(3, 0, -1, 0, 1, 0), 2.0)   # beyond the end point

    def test_polyline_touch(self):
        poly = [(0.0, 0.0), (10.0, 0.0)]
        self.assertTrue(rf.polyline_touches_disc(poly, 5.0, 1.0, 1.5))
        self.assertFalse(rf.polyline_touches_disc(poly, 5.0, 3.0, 1.5))

    def test_edges_incident_to_removed_nodes_are_removed(self):
        G = small_city()
        frame = rf.local_frame(G)
        n0 = sorted(G.nodes(), key=str)[len(G) // 2]
        x, y = frame[0][n0]
        sc = rf.make_scenario(G, frame, x, y, 120.0)
        self.assertIn(n0, sc["removed_nodes"])
        for u, v in G.edges():
            if u in sc["removed_nodes"] or v in sc["removed_nodes"]:
                self.assertIn((u, v), sc["removed_edges"])


class TestFaultCertificate(unittest.TestCase):
    def test_fast_count_matches_library_certificate(self):
        G = small_city()
        t = 2.0
        H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        for sc in rf.sample_scenarios(G, 6, (0.08, 0.15), seed=3):
            bad, m = rf.count_violations(G, set(H.edges()), sc, t)
            Gf = faulted_graph(G, sc)
            Hf = nx.DiGraph()
            Hf.add_nodes_from(Gf.nodes(data=True))
            for u, v, d in H.edges(data=True):
                if (u, v) not in sc["removed_edges"]:
                    Hf.add_edge(u, v, **d)
            ref = core.temporal_stretch_certificate(Gf, Hf, t, refine_cap=0)
            self.assertEqual(bad, ref["violating_edges"])
            self.assertEqual(m, Gf.number_of_edges())

    def test_intact_spanner_is_not_automatically_fault_tolerant(self):
        G = small_city()
        t = 2.0
        H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        ev = rf.evaluate_under_faults(G, H, rf.sample_scenarios(G, 12, (0.1, 0.2), seed=5), t)
        self.assertGreater(ev["regions_failing"], 0)   # on this instance the nominal spanner does break


class TestForcedUnderFaults(unittest.TestCase):
    def test_shortcut_matches_full_recomputation(self):
        G = small_city()
        t = 2.0
        nominal = rf.forced_edges_nominal(G, t)
        for sc in rf.sample_scenarios(G, 4, (0.1, 0.2), seed=7):
            fast = rf.forced_edges_under_fault(G, sc, t, nominal)
            Gf = faulted_graph(G, sc)
            edf = rd.per_edge_max_ratio(Gf, t * 1.0001)
            full = {(r.u, r.v) for r in edf.itertuples() if r.max_ratio > t}
            self.assertEqual(fast, full)

    def test_forcedness_is_monotone_under_faults(self):
        G = small_city()
        t = 2.0
        nominal = rf.forced_edges_nominal(G, t)
        for sc in rf.sample_scenarios(G, 4, (0.1, 0.2), seed=8):
            surviving_nominal = {e for e in nominal if e not in sc["removed_edges"]}
            self.assertTrue(surviving_nominal <= rf.forced_edges_under_fault(G, sc, t, nominal))


class TestFaultTolerantGreedy(unittest.TestCase):
    def test_certified_on_family_and_sandwiched_between_bounds(self):
        G = small_city()
        t = 2.0
        train = rf.sample_scenarios(G, 6, (0.08, 0.15, 0.25), seed=11)
        H0 = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        HF = rf.fault_tolerant_greedy(G, t, train)
        ev = rf.evaluate_under_faults(G, HF, train, t)
        self.assertEqual(ev["regions_failing"], 0)
        self.assertTrue(core.temporal_stretch_certificate(G, HF, t)["certified"])   # intact network too
        nominal, lb = rf.fault_forced_lower_bound(G, train, t)
        self.assertLessEqual(len(lb), HF.number_of_edges())
        self.assertLessEqual(len(nominal), len(lb))
        self.assertLessEqual(H0.number_of_edges(), HF.number_of_edges())

    def test_no_edge_subset_below_the_fault_lower_bound_survives(self):
        # remove one fault-forced edge from the fault-tolerant spanner: the family must then be violated
        G = small_city()
        t = 2.0
        train = rf.sample_scenarios(G, 6, (0.08, 0.15, 0.25), seed=11)
        HF = rf.fault_tolerant_greedy(G, t, train)
        nominal, lb = rf.fault_forced_lower_bound(G, train, t)
        extra = sorted(lb - nominal, key=str)
        if not extra:
            self.skipTest("no fault-only forced edge on this instance")
        e = extra[0]
        H2 = HF.copy()
        H2.remove_edge(*e)
        ev = rf.evaluate_under_faults(G, H2, train, t)
        self.assertGreater(ev["regions_failing"], 0)


if __name__ == "__main__":
    unittest.main()
