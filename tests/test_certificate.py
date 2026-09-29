#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Theorem-1 certificate machinery in run_spanner_benchmark.py.

These are NOT end-to-end pipeline tests -- they are small, hand-checkable
controls with an analytically known right answer, run against the actual
production functions (not a reimplementation), so a bug in the pruned
Dijkstra or the certificate logic cannot hide behind "it looked right on a
3000-node OSM graph". Run with:

    python -m unittest discover -s tests -v

or, if pytest is installed:

    pytest tests/ -v
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

import run_spanner_benchmark as spanner


def _toy_graph(edges):
    """Build a tiny single-hour DiGraph. `edges` is a list of (u, v, w)."""
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, tt=[float(w)], tt_ff=[float(w)], free_flow=float(w),
                   length=float(w), highway="residential", tier="local")
    for n in G.nodes():
        G.nodes[n]  # ensure node exists; no extra attrs needed for these tests
    return G


class TestFullNetworkAlwaysCertifies(unittest.TestCase):
    """H = G is always a valid t-spanner for every t >= 1: every edge is its
    own shortest path in itself, so dist_H(e) = w(e) exactly, and
    w(e) <= t * w(e) holds trivially. This must hold on ANY graph."""

    def test_full_network_certified_on_toy_graph(self):
        G = _toy_graph([("A", "B", 3.0), ("B", "C", 3.0), ("A", "C", 1.0)])
        for t in (1.0, 1.2, 5.0):
            cert = spanner.temporal_stretch_certificate(G, G, t, hours=[0])
            self.assertTrue(cert["certified"], f"full network must certify at t={t}")
            self.assertEqual(cert["violating_edges"], 0)

    def test_full_network_certified_on_random_synthetic_grid(self):
        G = spanner.synthetic_arterial_grid((31.9, 54.35), 600, n=7, seed_key="unittest-full")
        G = spanner.enrich_temporal_costs(G)
        for t in (1.0, 1.3, 2.5):
            cert = spanner.temporal_stretch_certificate(G, G, t)
            self.assertTrue(cert["certified"], f"full network must certify at t={t}")


class TestPlantedShortcutViolation(unittest.TestCase):
    """A->C is a genuine shortcut (cost 1) versus the only surviving detour
    A->B->C (cost 3+3=6). Removing A->C from H must break the certificate
    for any t < 6, with an EXACTLY computable attained stretch of 6.0 --
    this is the case that used to be silently misreported as a flat 8.0
    (a search-cutoff artifact) before the fix; this test would have caught
    that bug directly."""

    def setUp(self):
        self.G = _toy_graph([("A", "B", 3.0), ("B", "C", 3.0), ("A", "C", 1.0)])
        self.H = _toy_graph([("A", "B", 3.0), ("B", "C", 3.0)])
        # H must share the same node set as G for the certificate's Dijkstra probes.
        self.H.add_node("C")

    def test_violation_detected_below_threshold(self):
        cert = spanner.temporal_stretch_certificate(self.G, self.H, 1.5, hours=[0])
        self.assertFalse(cert["certified"])
        self.assertEqual(cert["violating_edges"], 1)
        self.assertEqual(cert["unreachable_violations"], 0)
        self.assertFalse(cert["attained_stretch_is_lower_bound"])
        self.assertAlmostEqual(cert["attained_worst_edge_stretch"], 6.0, places=9)

    def test_no_violation_above_threshold(self):
        # At t=6 the single remaining constraint (dist_H(A,C)=6 <= t*1) holds
        # with equality, and the other two edges are exact matches (t>=1
        # always covers those) -- so certification must flip to True.
        cert = spanner.temporal_stretch_certificate(self.G, self.H, 6.0, hours=[0])
        self.assertTrue(cert["certified"])
        self.assertEqual(cert["violating_edges"], 0)


class TestUnreachableViolationIsReportedAsInfinity(unittest.TestCase):
    """If H cannot reach v from u AT ALL on the relevant snapshot, the honest
    answer is 'unbounded', not a finite guess. This is the exact defect the
    code review caught: the old code reported a flat, fabricated 8.0 for
    every such case regardless of the graph."""

    def test_unreachable_edge_reports_infinite_stretch(self):
        G = _toy_graph([("A", "B", 3.0), ("B", "C", 3.0), ("A", "C", 1.0)])
        H = nx.DiGraph()
        H.add_edge("A", "B", tt=[3.0])
        H.add_node("C")  # B->C and A->C both absent: C is unreachable in H

        cert = spanner.temporal_stretch_certificate(G, H, 1.5, hours=[0])
        self.assertFalse(cert["certified"])
        self.assertGreaterEqual(cert["unreachable_violations"], 1)
        self.assertTrue(cert["attained_stretch_is_lower_bound"])
        self.assertTrue(math.isinf(cert["attained_worst_edge_stretch"]))


class TestGreedyConstructionAlwaysSelfCertifies(unittest.TestCase):
    """Property test: for ANY graph and ANY t >= 1, the output of
    delay_bounded_greedy_spanner must certify against its own parent graph.
    This is Theorem 1's guarantee by construction; if it ever fails, the
    construction algorithm itself has a bug, independent of any specific
    city's data."""

    def test_property_holds_across_seeds_and_t(self):
        for seed_key in ("prop-a", "prop-b", "prop-c"):
            G = spanner.synthetic_arterial_grid((31.9, 54.35), 600, n=8, seed_key=seed_key)
            G = spanner.enrich_temporal_costs(G)
            for t in (1.05, 1.4, 2.0, 3.0):
                H = spanner.delay_bounded_greedy_spanner(G, t, show_progress=False)
                cert = spanner.temporal_stretch_certificate(G, H, t, refine_cap=0)
                self.assertTrue(
                    cert["certified"],
                    f"greedy construction failed to self-certify: seed={seed_key}, t={t}, "
                    f"violating_edges={cert['violating_edges']}"
                )


class TestBoundedDijkstraAgainstNetworkX(unittest.TestCase):
    """Independent verification of the CORE primitive: bounded_dijkstra with
    an infinite budget must agree with networkx's own (independently
    implemented, widely trusted) shortest-path routine on random graphs.
    Every certificate in this project ultimately rests on this one function
    being correct."""

    def test_matches_networkx_on_random_graphs(self):
        rng = random.Random(1234)
        mismatches = []
        for trial in range(20):
            n = rng.randint(6, 14)
            p = rng.uniform(0.25, 0.55)
            G = nx.gnp_random_graph(n, p, seed=trial, directed=True)
            if G.number_of_edges() == 0:
                continue
            for u, v in G.edges():
                G[u][v]["weight"] = round(rng.uniform(0.5, 10.0), 3)

            adj = {node: {} for node in G.nodes()}
            for u, v, data in G.edges(data=True):
                adj[u][v] = [data["weight"]]

            nodes = list(G.nodes())
            for _ in range(15):
                s, t = rng.sample(nodes, 2)
                ours = spanner.bounded_dijkstra(adj, s, t, 0, spanner.INF)
                try:
                    truth = nx.shortest_path_length(G, s, t, weight="weight")
                except nx.NetworkXNoPath:
                    truth = float("inf")
                if math.isinf(truth) and math.isinf(ours):
                    continue
                if abs(ours - truth) > 1e-6:
                    mismatches.append((trial, s, t, ours, truth))

        self.assertEqual(mismatches, [],
                         f"bounded_dijkstra disagreed with networkx on: {mismatches[:5]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
