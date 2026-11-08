#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the processing order of the edges matter?

The greedy construction depends on the order in which edges are examined.  This script runs it on real networks with
several orders, certifies every result exactly, and also applies the reverse-delete pass of benchmarks/minimality.py so that
the comparison is made between edge-minimal spanners as well:

  temporal_max     ascending maximum travel time (the default of this repository)
  temporal_mean    ascending mean travel time
  centrality       maximum travel time damped by sampled edge betweenness (as in README 5i)
  free_flow        ascending free-flow travel time (a static order)
  random           uniformly random order, several seeds
  descending_max   DESCENDING maximum travel time (negative control)

    python benchmarks/edge_order.py --cities Yazd Isfahan Manhattan Tokyo --t 2 3
    python benchmarks/edge_order.py --synthetic --cities Yazd --t 2 --seeds 2        # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import minimality as mn  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))


def greedy_in_order(G, t: float, ordered):
    """Greedy rule of the repository applied to an arbitrary order of (u, v, data) triples."""
    adj = {n: {} for n in G.nodes()}
    kept = []
    for u, v, d in ordered:
        costs = d["tt"]
        add = False
        for tau in sorted(H24, key=lambda x: costs[x]):
            b = t * costs[tau]
            if core.bounded_dijkstra(adj, u, v, tau, b) > b + core.EPS:
                add = True
                break
        if add:
            adj[u][v] = costs
            kept.append((u, v))
    return kept, adj


def make_order(G, name: str, seed: int = 0):
    if name in ("temporal_max", "temporal_mean", "centrality"):
        return core._edge_order(G, name, "tt", H24)
    edges = list(G.edges(data=True))
    if name == "free_flow":
        return sorted(edges, key=lambda e: (e[2]["free_flow"], str(e[0]), str(e[1])))
    if name == "descending_max":
        return sorted(edges, key=lambda e: (-max(e[2]["tt"]), str(e[0]), str(e[1])))
    if name == "random":
        rng = random.Random(seed)
        edges.sort(key=lambda e: (str(e[0]), str(e[1])))
        rng.shuffle(edges)
        return edges
    raise ValueError(name)


def run_city(spec, ts, seeds: int, synthetic: bool, grid_n: int) -> List[Dict]:
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return []
    m = G.number_of_edges()
    rows = []
    for t in ts:
        for name in ("temporal_max", "temporal_mean", "centrality", "free_flow", "descending_max", "random"):
            for seed in (range(seeds) if name == "random" else [0]):
                kept, adj = greedy_in_order(G, t, make_order(G, name, seed))
                assert mn._certified(G, adj, set(), t), f"order {name} gave an uncertified spanner: bug"
                H = core.nx.DiGraph()
                H.add_nodes_from(G.nodes(data=True))
                for u, v in kept:
                    H.add_edge(u, v, **G[u][v])
                removed, _ = mn.reverse_delete(G, H, t)
                rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), edges=m, t=t,
                                 order=name, seed=seed, retention_pct=100.0 * len(kept) / m,
                                 pruned_retention_pct=100.0 * (len(kept) - len(removed)) / m))
        print(f"[ok] {spec['name']} t={t}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rows: List[Dict] = []
    for spec in [c for c in core.CITY_REGISTRY if c["name"] in a.cities]:
        rows += run_city(spec, a.t, a.seeds, a.synthetic, a.grid_n)
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "edge_order.csv"), index=False)
    summ = df.groupby(["t", "order"])[["retention_pct", "pruned_retention_pct"]].mean().reset_index()
    with open(os.path.join(OUT, "edge_order.md"), "w", encoding="utf-8") as f:
        f.write(md_table(summ, 3))
    print(md_table(summ, 3))


if __name__ == "__main__":
    main()
