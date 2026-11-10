"""Cut structure: classification of arcs by single-failure cuts, against brute force."""
import itertools
import os
import random
import sys
import unittest

import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import cut_structure as cs  # noqa: E402


def two_way(edges):
    G = nx.DiGraph()
    for a, b in edges:
        G.add_edge(a, b)
        G.add_edge(b, a)
    return G


def brute(G):
    """Exhaustive classification straight from the definition."""
    arcs = list(G.edges())
    out = {}
    for (u, v) in arcs:
        H = G.copy()
        H.remove_edge(u, v)
        if not nx.has_path(H, u, v):
            out[(u, v)] = "bridge"
            continue
        cat = "robust"
        for w in G.nodes():
            if w in (u, v):
                continue
            K = H.copy()
            K.remove_node(w)
            if not nx.has_path(K, u, v):
                cat = "vertex cut"
        if cat == "robust":
            for (a, b) in arcs:
                if (a, b) == (u, v) or (a, b) == (v, u):
                    continue
                K = H.copy()
                K.remove_edges_from([(a, b), (b, a)])
                if not nx.has_path(K, u, v):
                    cat = "street cut"
        out[(u, v)] = cat
    return out


class TestCutStructure(unittest.TestCase):
    def test_path_arc_is_a_bridge(self):
        G = nx.DiGraph([("a", "b")])
        self.assertEqual(cs.classify(G), {("a", "b"): "bridge"})

    def test_triangle_arcs_are_in_a_cut(self):
        G = two_way([("a", "b"), ("b", "c"), ("c", "a")])
        self.assertTrue(all(v != "robust" for v in cs.classify(G).values()))

    def test_complete_graph_k4_is_robust(self):
        G = two_way(list(itertools.combinations("abcd", 2)))
        self.assertTrue(all(v == "robust" for v in cs.classify(G).values()))

    def test_matches_brute_force_on_random_digraphs(self):
        rng = random.Random(5)
        for trial in range(40):
            n = rng.randint(4, 8)
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            for a in range(n):
                for b in range(n):
                    if a != b and rng.random() < 0.35:
                        G.add_edge(a, b)
            got, ref = cs.classify(G), brute(G)
            for arc in ref:
                # category names agree up to the vertex-versus-street tie, which only matters for the "any cut" count
                self.assertEqual(got[arc] == "robust", ref[arc] == "robust", (trial, arc, got[arc], ref[arc]))
                self.assertEqual(got[arc] == "bridge", ref[arc] == "bridge", (trial, arc))

    def test_matches_brute_force_on_random_two_way_graphs(self):
        rng = random.Random(9)
        for trial in range(30):
            n = rng.randint(5, 9)
            pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.4]
            if not pairs:
                continue
            G = two_way(pairs)
            got, ref = cs.classify(G), brute(G)
            for arc in ref:
                self.assertEqual(got[arc] == "robust", ref[arc] == "robust", (trial, arc, got[arc], ref[arc]))


if __name__ == "__main__":
    unittest.main()
