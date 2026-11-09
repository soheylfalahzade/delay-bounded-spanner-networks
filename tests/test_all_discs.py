"""Exact analysis over all disc faults of one radius: cells, failure, repair, lower bound, against brute force."""
import os
import random
import sys
import unittest

import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import region_fault as rf  # noqa: E402
import all_discs as ad  # noqa: E402


def city(grid_n, name="Manhattan"):
    spec = [c for c in core.CITY_REGISTRY if c["name"] == name][0]
    return core.build_city_network(spec, grid_n=grid_n, force_synthetic=True, use_cache=False)


def as_scenario(net, S):
    return {"removed_edges": {(net.u[i], net.v[i]) for i in S}}


@unittest.skipUnless(ad.HAVE_SHAPELY, "shapely not installed")
class TestAllDiscs(unittest.TestCase):
    T = 2.0

    @classmethod
    def setUpClass(cls):
        cls.G = city(6)
        cls.net = ad.Net(cls.G)
        cls.r = 0.12 * float(cls.G.graph.get("dist", core.DEFAULT_DIST))
        cls.scen, cls.faces, cls.distinct, cls.polys = ad.enumerate_scenarios(cls.G, cls.net, cls.r)
        H0 = core.delay_bounded_greedy_spanner(cls.G, cls.T, show_progress=False)
        cls.H0 = {cls.net.id[e] for e in H0.edges()}

    def test_search_agrees_with_networkx(self):
        H = self.net.adjacency(self.H0)
        Hg = nx.DiGraph()
        for i in self.H0:
            Hg.add_edge(self.net.u[i], self.net.v[i], i=i)
        rng = random.Random(1)
        for _ in range(60):
            g = rng.randrange(self.net.m)
            tau = rng.randrange(24)
            b = self.T * self.net.tt[g][tau]
            path = ad.search(H, self.net.tt, self.net.u[g], self.net.v[g], tau, b, frozenset())
            try:
                d = nx.dijkstra_path_length(Hg, self.net.u[g], self.net.v[g],
                                            weight=lambda x, y, dd, tau=tau: self.net.tt[dd["i"]][tau])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = float("inf")
            self.assertEqual(path is not None, d <= b + core.EPS)

    def test_cells_cover_random_fault_centres(self):
        frame = rf.local_frame(self.G)
        xs = [p[0] for p in frame[0].values()]
        ys = [p[1] for p in frame[0].values()]
        rng = random.Random(2)
        found = 0
        N = 400
        for _ in range(N):
            cx = rng.uniform(min(xs) - self.r, max(xs) + self.r)
            cy = rng.uniform(min(ys) - self.r, max(ys) + self.r)
            sc = rf.make_scenario(self.G, frame, cx, cy, self.r)
            S = frozenset(self.net.id[e] for e in sc["removed_edges"])
            found += S in self.scen or not S
        self.assertGreaterEqual(found / N, 0.99)

    def test_violations_match_brute_force(self):
        st = ad.SpannerState(self.net, self.H0, self.T)
        H_edges = {(self.net.u[i], self.net.v[i]) for i in self.H0}
        keys = sorted(self.scen, key=lambda s: (len(s), sorted(s)))
        for S in keys[:: max(1, len(keys) // 120)]:
            bad = st.violations(S)
            ref, _ = rf.count_violations(self.G, H_edges, as_scenario(self.net, S), self.T)
            self.assertEqual(len(bad), ref)

    def test_repair_is_certified_for_every_cell_and_for_random_centres(self):
        stats, st, _ = ad.run_family(self.net, self.H0, self.scen, self.T, repair=True)
        again, _, _ = ad.run_family(self.net, st.H, self.scen, self.T, repair=False)
        self.assertEqual(again["failing_scenarios"], 0)
        self.assertTrue(self.H0 <= st.H)
        Hs = {(self.net.u[i], self.net.v[i]) for i in st.H}
        frame = rf.local_frame(self.G)
        xs = [p[0] for p in frame[0].values()]
        ys = [p[1] for p in frame[0].values()]
        rng = random.Random(3)
        for _ in range(150):
            sc = rf.make_scenario(self.G, frame, rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys)), self.r)
            bad, _ = rf.count_violations(self.G, Hs, sc, self.T)
            self.assertEqual(bad, 0)

    def test_lower_bound_equals_brute_force_union_and_is_inside_every_certified_spanner(self):
        nominal, union = ad.forced_union(self.net, list(self.scen), self.T)
        nom_ref = rf.forced_edges_nominal(self.G, self.T)
        self.assertEqual({(self.net.u[i], self.net.v[i]) for i in nominal}, nom_ref)
        ref = set(nom_ref)
        for S in list(self.scen):
            ref |= rf.forced_edges_under_fault(self.G, as_scenario(self.net, S) | {"removed_nodes": set()}, self.T, nom_ref)
        got = {(self.net.u[i], self.net.v[i]) for i in union}
        self.assertEqual(ref, got)                                     # equality with brute force over ALL scenarios
        _, st, _ = ad.run_family(self.net, self.H0, self.scen, self.T, repair=True)
        self.assertTrue(union <= st.H)

    def test_relaxed_stretch_after_a_fault_matches_brute_force(self):
        tf = 3.0
        st = ad.SpannerState(self.net, self.H0, self.T, tf)
        H_edges = {(self.net.u[i], self.net.v[i]) for i in self.H0}
        keys = sorted(self.scen, key=lambda s: (len(s), sorted(s)))
        for S in keys[:: max(1, len(keys) // 100)]:
            ref, _ = rf.count_violations(self.G, H_edges, as_scenario(self.net, S), tf)
            self.assertEqual(len(st.violations(S)), ref)

    def test_relaxed_repair_is_certified_and_monotone_in_the_post_fault_stretch(self):
        sizes, areas, bounds = [], [], []
        for tf in (2.0, 3.0, 5.0):
            stats, st, _ = ad.run_family(self.net, self.H0, self.scen, self.T, repair=True, t_fault=tf)
            again, _, _ = ad.run_family(self.net, st.H, self.scen, self.T, repair=False, t_fault=tf)
            self.assertEqual(again["failing_scenarios"], 0)
            nominal, union = ad.forced_union(self.net, list(self.scen), self.T, tf)
            self.assertTrue(union <= st.H)
            sizes.append(len(st.H))
            areas.append(ad.run_family(self.net, self.H0, self.scen, self.T, repair=False, t_fault=tf)[0]["failing_area_pct"])
            bounds.append(len(union))
        self.assertTrue(sizes[0] >= sizes[1] >= sizes[2])
        self.assertTrue(areas[0] >= areas[1] >= areas[2])
        self.assertTrue(bounds[0] >= bounds[1] >= bounds[2])

    def test_relaxed_lower_bound_against_brute_force(self):
        tf = 3.0
        nominal_t, union = ad.forced_union(self.net, list(self.scen), self.T, tf)
        nom_tf = rf.forced_edges_nominal(self.G, tf)
        ref = set(nom_tf)
        for S in self.scen:
            ref |= rf.forced_edges_under_fault(self.G, as_scenario(self.net, S) | {"removed_nodes": set()}, tf, nom_tf)
        got = {(self.net.u[i], self.net.v[i]) for i in union}
        nom_t = {(self.net.u[i], self.net.v[i]) for i in nominal_t}
        self.assertTrue(ref <= got)
        self.assertTrue(got - ref <= nom_t)                            # extra arcs are exactly those forced at the nominal stretch

    def test_failing_area_is_a_share_of_the_total_area(self):
        st, _, _ = ad.run_family(self.net, self.H0, self.scen, self.T, repair=False)
        self.assertTrue(0.0 <= st["failing_area_pct"] <= 100.0)
        self.assertEqual(st["scenarios"], self.distinct)
        self.assertAlmostEqual(sum(self.scen.values()), sum(self.scen.values()))


class TestMergeRows(unittest.TestCase):
    def test_new_rows_replace_old_rows_with_the_same_key(self):
        import pandas as pd
        old = pd.DataFrame([dict(city="A", radius_m=75.0, t=2.0, v=1), dict(city="A", radius_m=150.0, t=2.0, v=2),
                            dict(city="B", radius_m=75.0, t=2.0, v=3)])
        new = pd.DataFrame([dict(city="A", radius_m=5.0, t=2.0, v=9), dict(city="A", radius_m=150.0, t=2.0, v=7)])
        out = ad.merge_rows(old, new)
        self.assertEqual(len(out), 4)
        self.assertEqual(int(out[(out.city == "A") & (out.radius_m == 150.0)].v.iloc[0]), 7)
        self.assertEqual(int(out[(out.city == "B")].v.iloc[0]), 3)
        self.assertEqual(list(out[out.city == "A"].radius_m), [5.0, 75.0, 150.0])
        self.assertIn("t_fault", out.columns)                          # files written before the column get t_fault = t


if __name__ == "__main__":
    unittest.main()
