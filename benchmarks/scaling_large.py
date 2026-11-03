#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling study of the delay-bounded greedy construction.

Besides wall-clock time, a machine-independent measure of work is recorded: the number of bounded-Dijkstra probes and the number
of nodes they settle. If the nodes settled per probe stay bounded while the network grows, the construction is (near) linear in
the number of edges. The work counters come from a replica of the library loop that is checked, edge by edge, against the
library result (tests/test_scaling_large.py).

  python benchmarks/scaling_large.py --synthetic --sizes 1000 3000 10000 30000 100000 300000 --t 2
  python benchmarks/scaling_large.py --real --cities Isfahan Tokyo Manhattan --dists 1500 2500 3500 5000 7000 --t 2

Synthetic networks are thinned Delaunay triangulations (road-like degree); they are labelled as such in the output.
Output: results/benchmarks/scaling_large.csv (resumable; one row per network and t)
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import run_spanner_benchmark as core  # noqa: E402
from scaling_fit import fit_exponent  # noqa: E402,F401

OUT = os.path.join(ROOT, "results", "benchmarks")
CSV = os.path.join(OUT, "scaling_large.csv")
EPS = core.EPS


def synthetic_network(n_points, seed=1, keep=0.45):
    """Thinned Delaunay triangulation with hourly costs: congestion is strongest in the middle of the area and peaks at 08:00 and 17:00."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    side = np.sqrt(n_points)
    pts = rng.random((n_points, 2)) * side
    tri = Delaunay(pts)
    s = np.sort(np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]]), axis=1)
    s = np.unique(s, axis=0)
    length = np.hypot(*(pts[s[:, 0]] - pts[s[:, 1]]).T)
    sel = (length < 2.2) & (rng.random(len(s)) < keep)
    s, length = s[sel], length[sel]
    mid = (pts[s[:, 0]] + pts[s[:, 1]]) / 2
    load = np.exp(-np.hypot(*(mid - side / 2).T) / (0.35 * side)) * (0.8 + 0.4 * rng.random(len(s)))
    h = np.arange(24)
    peak = 0.55 * np.exp(-0.5 * ((h - 17.0) / 2.2) ** 2) + 0.35 * np.exp(-0.5 * ((h - 8.0) / 1.6) ** 2)
    tt = (length[:, None] * (1.0 + 3.2 * load[:, None] * peak[None, :])).tolist()
    G = nx.DiGraph()
    G.add_nodes_from(range(n_points))
    for (u, v), l, c in zip(s.tolist(), length.tolist(), tt):
        G.add_edge(u, v, tt=c, free_flow=l, length=l)
        G.add_edge(v, u, tt=c, free_flow=l, length=l)
    return G


def _bounded_with_work(adj, source, target, tau, budget):
    if source == target:
        return 0.0, 0
    if not adj.get(source):
        return core.INF, 0
    dist = {source: 0.0}
    pq = [(0.0, 0, source)]
    counter, settled = 1, 0
    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist.get(u, core.INF) + EPS:
            continue
        settled += 1
        if u == target:
            return d, settled
        for v, wvec in adj.get(u, {}).items():
            nd = d + wvec[tau]
            if nd > budget + EPS:
                continue
            if nd + EPS < dist.get(v, core.INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, counter, v))
                counter += 1
    return core.INF, settled


def greedy_with_work(G, t, attr="tt", order="temporal_max"):
    """Replica of core.delay_bounded_greedy_spanner that also counts probes and settled nodes. Returns (kept edge set, counters)."""
    hours = list(range(24))
    adj = {n: {} for n in G.nodes()}
    kept = set()
    probes = settled_total = 0
    for u, v, data in core._edge_order(G, order, attr, hours):
        costs = data[attr]
        add = False
        for tau in sorted(hours, key=lambda x: costs[x]):
            budget = t * costs[tau]
            probes += 1
            d, st = _bounded_with_work(adj, u, v, tau, budget)
            settled_total += st
            if d > budget + EPS:
                add = True
                break
        if add:
            adj[u][v] = costs
            kept.add((u, v))
    return kept, {"probes": probes, "settled": settled_total}


def measure(G, t):
    t0 = time.perf_counter()
    H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    secs = time.perf_counter() - t0
    t1 = time.perf_counter()
    kept, work = greedy_with_work(G, t)
    secs_replica = time.perf_counter() - t1
    assert kept == set(H.edges()), "the work-counting replica disagrees with the library"
    assert work["probes"] == H.graph["dijkstra_probes"], "the replica probes a different number of times than the library"
    m = G.number_of_edges()
    return dict(nodes=G.number_of_nodes(), edges=m, t=t, seconds=round(secs, 4), seconds_replica=round(secs_replica, 4), kept_edges=len(kept),
                kept_pct=round(100 * len(kept) / m, 3), probes=work["probes"], settled=work["settled"],
                probes_per_edge=round(work["probes"] / m, 4), settled_per_probe=round(work["settled"] / max(1, work["probes"]), 4),
                settled_per_edge=round(work["settled"] / m, 4))


def load():
    return pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()


def append(row):
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([row]).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="*", default=[1000, 3000, 10000, 30000, 100000])
    ap.add_argument("--cities", nargs="*", default=["Isfahan", "Tokyo", "Manhattan"])
    ap.add_argument("--dists", type=int, nargs="*", default=[1500, 2500, 3500, 5000])
    ap.add_argument("--t", type=float, nargs="+", default=[2.0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-seconds", type=int, default=10 ** 9, help="stop starting new rows after this many seconds (rerun to resume)")
    ap.add_argument("--attempts", type=int, default=6)
    a = ap.parse_args()
    done = load()
    key = lambda kind, label, size: set(zip(done[done.kind == kind].label, done[done.kind == kind]["size"], done[done.kind == kind].t)) if not done.empty else set()  # noqa: E731
    t_end = time.time() + a.max_seconds
    if a.synthetic:
        have = key("synthetic", None, None)
        for n in a.sizes:
            todo = [t for t in a.t if ("thinned Delaunay", n, t) not in have]
            if not todo or time.time() > t_end:
                continue
            G = synthetic_network(n, a.seed)
            for t in todo:
                row = dict(kind="synthetic", label="thinned Delaunay", size=n, **measure(G, t))
                append(row)
                print(f"[synthetic n={n}] edges={row['edges']} t={t:g}: {row['seconds']}s, settled/probe {row['settled_per_probe']}", flush=True)
            del G
    if a.real:
        from large_scale_run import build_real
        have = key("real", None, None)
        for name in a.cities:
            spec0 = next(c for c in core.CITY_REGISTRY if c["name"] == name)
            for dist in a.dists:
                todo = [t for t in a.t if (name, dist, t) not in have]
                if not todo or time.time() > t_end:
                    continue
                G = build_real(dict(spec0, dist=dist), 15, a.attempts, 30)
                if G is None:
                    print(f"[FAIL] {name} {dist} m: no real OSM network; skipped", flush=True)
                    continue
                for t in todo:
                    row = dict(kind="real", label=name, size=dist, **measure(G, t))
                    append(row)
                    print(f"[{name} {dist} m] edges={row['edges']} t={t:g}: {row['seconds']}s, settled/probe {row['settled_per_probe']}", flush=True)
                del G


if __name__ == "__main__":
    main()
