#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact optimum on windows cut from real road networks.

Section 5j compares the greedy construction with the exact optimum on small random digraphs only.  Here the same exact
search is run on small windows of the real OpenStreetMap city cores: a window is the subgraph induced by the first k
nodes of a breadth-first search from a random intersection, restricted to its largest strongly connected component.

Exactness.  The edge-wise certificate of Theorem 1 is necessary and sufficient for the all-pairs, all-hours bound: if H
is a t-spanner then every edge (u,v) of G has d_H(u,v) <= t d_G(u,v) <= t w(u,v).  Every forced edge (Theorem 2) is in every
spanner, so the optimum is |forced| plus the size of a smallest set of non-forced ("free") edges that completes the
certificate.  Subsets of the free edges are enumerated by increasing size; the first feasible size is optimal.  Windows with
more than --max-free free edges, or needing more than --max-evals feasibility checks, are skipped (reported), never approximated.

Compared on each window: exact optimum, greedy, and greedy followed by the reverse-delete pass of benchmarks/minimality.py.

    python benchmarks/road_exact.py --cities Yazd Isfahan Manhattan Tokyo --t 2 3 5
    python benchmarks/road_exact.py --synthetic --cities Yazd --windows 6        # offline smoke test
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
import minimality as mn  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))


def make_window(G: nx.DiGraph, seed_node, k: int) -> Optional[nx.DiGraph]:
    seen, q = {seed_node}, deque([seed_node])
    und = G.to_undirected(as_view=True)
    while q and len(seen) < k:
        x = q.popleft()
        for y in sorted(und[x], key=str):
            if y not in seen and len(seen) < k:
                seen.add(y)
                q.append(y)
    W = G.subgraph(seen).copy()
    if W.number_of_nodes() < 4:
        return None
    comp = max(nx.strongly_connected_components(W), key=len)
    W = W.subgraph(comp).copy()
    return W if W.number_of_nodes() >= 4 else None


def _feasible(W: nx.DiGraph, include: set, t: float) -> bool:
    adj = {n: {} for n in W.nodes()}
    for u, v, d in W.edges(data=True):
        if (u, v) in include:
            adj[u][v] = d["tt"]
    for u, v, d in W.edges(data=True):
        if (u, v) in include:
            continue
        for tau in H24:
            b = t * d["tt"][tau]
            if core.bounded_dijkstra(adj, u, v, tau, b) > b + core.EPS:
                return False
    return True


def exact_optimum(W: nx.DiGraph, t: float, max_free: int, max_evals: int = 0):
    """(optimum size, forced count, free count), or None if there are more than max_free free edges or the number of
    feasibility checks would exceed max_evals (0 = no limit).  A skipped window is never approximated."""
    edf = rd.per_edge_max_ratio(W, t * 1.0001)
    forced = {(r.u, r.v) for r in edf.itertuples() if r.max_ratio > t}
    free = sorted(({(u, v) for u, v in W.edges()} - forced), key=str)
    if len(free) > max_free:
        return None
    evals = 0
    for k in range(len(free) + 1):
        for S in itertools.combinations(free, k):
            evals += 1
            if max_evals and evals > max_evals:
                return None
            if _feasible(W, forced | set(S), t):
                return len(forced) + k, len(forced), len(free)
    raise AssertionError("the full edge set is always feasible: bug")


def run_city(spec: Dict, ts, n_windows: int, k_nodes: int, min_edges: int, max_edges: int, max_free: int,
             synthetic: bool, grid_n: int, seed: int, max_evals: int = 50000) -> List[Dict]:
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return []
    rng = random.Random(seed)
    nodes = sorted(G.nodes(), key=str)
    wins = []
    tries = 0
    while len(wins) < n_windows and tries < 40 * n_windows:
        tries += 1
        W = make_window(G, nodes[rng.randrange(len(nodes))], k_nodes)
        if W is not None and min_edges <= W.number_of_edges() <= max_edges:
            wins.append(W)
    rows = []
    for t in ts:
        for i, W in enumerate(wins):
            res = exact_optimum(W, t, max_free, max_evals)
            if res is None:
                rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), t=t, window=i,
                                 nodes=W.number_of_nodes(), edges=W.number_of_edges(), skipped=True))
                continue
            opt, nf, nfree = res
            H = core.delay_bounded_greedy_spanner(W, t, show_progress=False)
            removed, _ = mn.reverse_delete(W, H, t)
            g, p = H.number_of_edges(), H.number_of_edges() - len(removed)
            assert nf <= opt <= p <= g, "ordering forced <= optimum <= pruned <= greedy violated: bug"
            rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), t=t, window=i,
                             nodes=W.number_of_nodes(), edges=W.number_of_edges(), skipped=False, forced=nf,
                             free=nfree, optimum=opt, greedy=g, greedy_pruned=p))
        print(f"[ok] {spec['name']} t={t}: {len(wins)} windows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--min-edges", type=int, default=20)
    ap.add_argument("--max-edges", type=int, default=70)
    ap.add_argument("--max-free", type=int, default=16)
    ap.add_argument("--max-evals", type=int, default=50000, help="feasibility checks per window before it is skipped")
    ap.add_argument("--seed", type=int, default=core.SEED)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    specs = [c for c in core.CITY_REGISTRY if c["name"] in a.cities]
    rows = []
    for s in specs:
        rows += run_city(s, a.t, a.windows, a.nodes, a.min_edges, a.max_edges, a.max_free, a.synthetic, a.grid_n, a.seed, a.max_evals)
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "road_exact_raw.csv"), index=False)
    d = df[~df.skipped.astype(bool)]
    summ = (d.assign(excess=d.greedy - d.optimum, excess_pruned=d.greedy_pruned - d.optimum,
                     nontrivial=d.free > 0)
             .groupby("t").agg(windows=("window", "count"), mean_edges=("edges", "mean"), mean_free=("free", "mean"),
                               greedy_optimal=("excess", lambda x: 100 * (x == 0).mean()),
                               pruned_optimal=("excess_pruned", lambda x: 100 * (x == 0).mean()),
                               mean_excess=("excess", "mean"), max_excess=("excess", "max")).reset_index())
    with open(os.path.join(OUT, "road_exact.md"), "w", encoding="utf-8") as f:
        f.write(md_table(summ))
    print(md_table(summ))


if __name__ == "__main__":
    main()
