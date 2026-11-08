#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the delay-bounded greedy spanner edge-minimal?

For a single static weighting the greedy t-spanner has no proper t-spanner inside itself (every edge is forced
in H; Filtser and Solomon).  With hourly weights the construction orders edges by their maximum travel time, so an
edge kept early may become redundant after later edges are added in an hour in which it was cheaper to bypass.
This script tests that directly: after the greedy construction, edges of H are examined in reverse order and an edge
is deleted whenever H minus that edge still satisfies the exact Theorem-1 certificate for all of G at all 24 hours.
It reports how many edges are removable and verifies the pruned spanner exactly.

    python benchmarks/minimality.py --cities Yazd Isfahan Manhattan Tokyo --t 1.5 2 3
    python benchmarks/minimality.py --synthetic --cities Yazd --t 2          # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Set, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))
Edge = Tuple[object, object]


def _certified(G, adj, skip: Set[Edge], t: float) -> bool:
    """Edge-wise Theorem-1 certificate of the subgraph encoded by `adj` against all edges of G.
    Edges that are in the subgraph are satisfied by themselves."""
    for u, v, d in G.edges(data=True):
        if v in adj[u]:
            continue
        costs = d["tt"]
        for tau in H24:
            b = t * costs[tau]
            if core.bounded_dijkstra(adj, u, v, tau, b) > b + core.EPS:
                return False
    return True


def reverse_delete(G, H, t: float) -> Tuple[List[Edge], int]:
    """Delete edges of H, most expensive first, while the exact certificate against G still holds.
    Returns (removed edges, number of full certificate evaluations)."""
    kept = {(u, v) for u, v in H.edges()}
    order = [(u, v) for u, v, _ in core._edge_order(G, "temporal_max", "tt", H24) if (u, v) in kept]
    adj = {n: {} for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        if (u, v) in kept:
            adj[u][v] = d["tt"]
    removed: List[Edge] = []
    full_checks = 0
    for u, v in reversed(order):
        costs = adj[u].pop(v)
        # quick test: the edge itself must have a detour in H - e at every hour
        ok = True
        for tau in H24:
            b = t * costs[tau]
            if core.bounded_dijkstra(adj, u, v, tau, b) > b + core.EPS:
                ok = False
                break
        if ok:
            full_checks += 1
            ok = _certified(G, adj, set(), t)
        if ok:
            removed.append((u, v))
        else:
            adj[u][v] = costs
    return removed, full_checks


def run_city(spec: Dict, ts, synthetic: bool, grid_n: int):
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return []
    m = G.number_of_edges()
    rows = []
    for t in ts:
        H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        removed, full = reverse_delete(G, H, t)
        H2 = H.copy()
        H2.remove_edges_from(removed)
        assert core.temporal_stretch_certificate(G, H2, t)["certified"], "pruned spanner is not certified: bug"
        rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), edges=m, t=t,
                         greedy_retention_pct=100.0 * H.number_of_edges() / m,
                         removable_edges=len(removed),
                         removable_pct_of_spanner=100.0 * len(removed) / max(1, H.number_of_edges()),
                         pruned_retention_pct=100.0 * H2.number_of_edges() / m,
                         greedy_is_minimal=len(removed) == 0, full_certificate_checks=full))
        print(f"[ok] {spec['name']} t={t}: {len(removed)} of {H.number_of_edges()} spanner edges removable")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    specs = [c for c in core.CITY_REGISTRY if c["name"] in a.cities]
    rows = []
    for s in specs:
        rows += run_city(s, a.t, a.synthetic, a.grid_n)
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(3).to_csv(os.path.join(OUT, "minimality.csv"), index=False)
    with open(os.path.join(OUT, "minimality.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df))
    print(md_table(df))


if __name__ == "__main__":
    main()
