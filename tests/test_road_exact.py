"""Exact optimum on windows: agrees with brute force, sandwiched between the forced bound and the greedy size."""
import itertools
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import road_exact as rx  # noqa: E402
import minimality as mn  # noqa: E402
import retention_diagnostics as rd  # noqa: E402


def windows(n=6, k=9):
    spec = [c for c in core.CITY_REGISTRY if c["name"] == "Manhattan"][0]
    G = core.build_city_network(spec, grid_n=9, force_synthetic=True, use_cache=False)
    nodes = sorted(G.nodes(), key=str)
    out = []
    for i in range(0, len(nodes), max(1, len(nodes) // n)):
        W = rx.make_window(G, nodes[i], k)
        if W is not None and 15 <= W.number_of_edges() <= 45:
            out.append(W)
    return out[:n]


class TestRoadExact(unittest.TestCase):
    def test_matches_brute_force_over_all_edge_subsets(self):
        done = 0
        for W in windows():
            t = 3.0
            res = rx.exact_optimum(W, t, max_free=14)
            if res is None:
                continue
            opt, nf, nfree = res
            edges = list(W.edges())
            best = min(len(S) for r in range(len(edges) + 1) for S in itertools.combinations(edges, r)
                       if rx._feasible(W, set(S), t)) if len(edges) <= 20 else None
            if best is not None:
                self.assertEqual(opt, best)
            done += 1
        self.assertGreater(done, 0)

    def test_ordering_forced_optimum_pruned_greedy(self):
        solved = 0
        for W in windows():
            for t in (2.0, 3.0):
                res = rx.exact_optimum(W, t, max_free=14)
                if res is None:
                    continue
                opt, nf, _ = res
                H = core.delay_bounded_greedy_spanner(W, t, show_progress=False)
                removed, _ = mn.reverse_delete(W, H, t)
                self.assertLessEqual(nf, opt)
                self.assertLessEqual(opt, H.number_of_edges() - len(removed))
                self.assertLessEqual(H.number_of_edges() - len(removed), H.number_of_edges())
                solved += 1
        self.assertGreater(solved, 0)

    def test_evaluation_budget_skips_instead_of_approximating(self):
        for W in windows():
            res = rx.exact_optimum(W, 3.0, max_free=14)
            if res is not None and res[2] >= 3:
                self.assertIsNone(rx.exact_optimum(W, 3.0, max_free=14, max_evals=1))
                return
        self.skipTest("no window with enough free edges")

    def test_too_many_free_edges_is_skipped_not_approximated(self):
        W = windows()[0]
        self.assertIsNone(rx.exact_optimum(W, 5.0, max_free=0))


if __name__ == "__main__":
    unittest.main()
