#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why does the delay-bounded spanner keep most edges of a road network?

For every directed edge e=(u,v) and hour tau define the detour ratio
    r_tau(e) = d_{G - e, tau}(u, v) / w_tau(e)        (inf if e is a bridge)
Any edge with max_tau r_tau(e) > t is FORCED: whatever subset H of G a spanner
construction has built, d_{H,tau}(u,v) >= d_{G-e,tau}(u,v) > t * w_tau(e), so e
cannot be dropped by ANY edge-subset t-spanner. Hence
        retention(t)  >=  forced_fraction(t)
is a lower bound that holds for every algorithm, not just the greedy one.
The script reports, per city and t, the forced fraction, the retention achieved by
the delay-bounded greedy, and the gap between them (how much of the retention is
explained by the network structure itself).

    python benchmarks/retention_diagnostics.py                 # all registered cities (cached OSM)
    python benchmarks/retention_diagnostics.py --cities Yazd Isfahan --t 1.5 2 3 5
    python benchmarks/retention_diagnostics.py --synthetic     # offline smoke test
"""
from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_spanner_benchmark as core  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))


def detour_ratio(adj: Dict, u, v, w: float, tau: int, cap: float) -> float:
    """Dijkstra from u to v ignoring arc (u,v), stopping once the distance exceeds cap*w."""
    budget = cap * w
    dist = {u: 0.0}
    pq = [(0.0, 0, u)]
    cnt = 1
    while pq:
        d, _, x = heapq.heappop(pq)
        if d > dist.get(x, math.inf):
            continue
        if x == v:
            return d / w
        if d > budget:
            break
        for y, c in adj[x]:
            if x == u and y == v:
                continue
            nd = d + c
            if nd <= budget and nd < dist.get(y, math.inf):
                dist[y] = nd
                heapq.heappush(pq, (nd, cnt, y))
                cnt += 1
    return math.inf


def per_edge_max_ratio(G, cap: float) -> pd.DataFrame:
    adj_by_hour = []
    for tau in H24:
        adj_by_hour.append({x: [(y, d["tt"][tau]) for y, d in G[x].items()] for x in G.nodes()})
    rows = []
    for u, v, d in G.edges(data=True):
        rs = [detour_ratio(adj_by_hour[tau], u, v, d["tt"][tau], tau, cap) for tau in H24]
        rows.append(dict(u=u, v=v, max_ratio=max(rs), min_ratio=min(rs), median_ratio=float(np.median(rs)),
                         out_deg=G.out_degree(u), in_deg=G.in_degree(v)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=None, help="names from CITY_REGISTRY (default: all)")
    ap.add_argument("--t", type=float, nargs="+", default=[1.1, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=0, help="skip cities larger than this (0 = no limit)")
    args = ap.parse_args()

    cap = max(args.t) * 1.0001
    specs = [c for c in core.CITY_REGISTRY if (args.cities is None or c["name"] in args.cities)]
    if not specs:
        sys.exit("no matching cities; available: " + ", ".join(c["name"] for c in core.CITY_REGISTRY))
    rows: List[Dict] = []
    detail = []
    for spec in specs:
        G = core.build_city_network(spec, grid_n=args.grid_n, force_synthetic=args.synthetic)
        if args.max_nodes and G.number_of_nodes() > args.max_nodes:
            print(f"[skip] {spec['name']}: {G.number_of_nodes()} nodes")
            continue
        m = G.number_of_edges()
        edf = per_edge_max_ratio(G, cap)
        edf["city"] = spec["name"]
        detail.append(edf)
        for t in args.t:
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            forced = float((edf.max_ratio > t).mean())
            kept = H.number_of_edges() / m
            rows.append(dict(city=spec["name"], country=spec["country"], archetype=spec["archetype"],
                             real_osm=bool(G.graph.get("used_real_osm")), nodes=G.number_of_nodes(), edges=m, t=t,
                             retention_pct=100 * kept, forced_lower_bound_pct=100 * forced,
                             unexplained_pct=100 * (kept - forced),
                             share_of_retention_forced=(forced / kept if kept else float("nan")),
                             bridge_pct=100 * float(np.isinf(edf.max_ratio).mean()),
                             median_max_detour_ratio=float(np.median(edf.max_ratio.replace(np.inf, np.nan).dropna())) if np.isfinite(edf.max_ratio).any() else float("nan")))
            assert kept + 1e-12 >= forced, "lower bound violated: bug"
        print(f"[ok] {spec['name']}: {m} edges")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(3).to_csv(os.path.join(OUT, "retention_diagnostics.csv"), index=False)
    pd.concat(detail).to_csv(os.path.join(OUT, "retention_edge_ratios.csv"), index=False)
    piv = df.pivot_table(index=["city", "country", "real_osm"], columns="t",
                         values=["retention_pct", "forced_lower_bound_pct"]).round(1)
    with open(os.path.join(OUT, "retention_diagnostics.md"), "w", encoding="utf-8") as f:
        f.write(df.round(2).to_markdown(index=False))
    print("\n", piv.to_string())
    s = df.groupby("t")[["retention_pct", "forced_lower_bound_pct", "share_of_retention_forced"]].mean().round(3)
    print("\nMean over cities:\n", s.to_string())
    print("\nLower bound respected in every (city, t):", bool((df.retention_pct + 1e-9 >= df.forced_lower_bound_pct).all()))


if __name__ == "__main__":
    main()
