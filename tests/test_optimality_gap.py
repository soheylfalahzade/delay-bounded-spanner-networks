"""Exact search for the minimum delay-bounded spanner (benchmarks/optimality_gap.py)."""
import itertools
import unittest
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import literature_benchmarks as lb  # noqa: E402
import optimality_gap as og  # noqa: E402


def _instance(seed, n=6, p=0.35):
    rng = random.Random(seed)
    while True:
        G = lb.random_temporal_digraph(n, p, rng)
        if 6 <= G.number_of_nodes() and 8 <= G.number_of_edges() <= 12:
            return G




class TestOptimalityGap(unittest.TestCase):
    def test_floyd_warshall_validity_matches_the_reference_check(self):
        for seed in range(4):
            G = _instance(seed)
            edges, cost = og.edge_arrays(G)
            n = G.number_of_nodes()
            DG = og.all_pairs(n, edges, cost)
            rng = random.Random(seed)
            for t in (1.5, 2.0):
                for _ in range(15):
                    keep = np.array([rng.random() < 0.8 for _ in edges])
                    H = G.__class__()
                    H.add_nodes_from(G.nodes())
                    for (u, v), k in zip(edges, keep):
                        if k:
                            H.add_edge(u, v, **G[u][v])
                    assert og.is_valid(n, edges, cost, keep, t, DG) == lb.brute_force_ok(G, H, t)


    def test_exact_minimum_equals_exhaustive_enumeration(self):
        for seed in range(3):
            G = _instance(seed + 10)
            edges, cost = og.edge_arrays(G)
            n, m = G.number_of_nodes(), len(edges)
            DG = og.all_pairs(n, edges, cost)
            for t in (1.5, 3.0):
                best = m
                for k in range(m + 1):
                    if any(og.is_valid(n, edges, cost, np.isin(np.arange(m), c), t, DG) for c in itertools.combinations(range(m), k)):
                        best = k
                        break
                assert og.minimum_spanner_size(n, edges, cost, t, DG)[0] == best


    def test_an_edge_is_forced_exactly_when_removing_it_alone_breaks_the_bound(self):
        for seed, t in ((21, 2.0), (22, 1.5), (23, 3.0)):
            G = _instance(seed)
            edges, cost = og.edge_arrays(G)
            n, m = G.number_of_nodes(), len(edges)
            DG = og.all_pairs(n, edges, cost)
            forced = set(og.forced_edges(n, edges, cost, t, DG))
            for e in range(m):
                mask = np.ones(m, dtype=bool)
                mask[e] = False
                assert (e in forced) == (not og.is_valid(n, edges, cost, mask, t, DG))


if __name__ == "__main__":
    unittest.main()
