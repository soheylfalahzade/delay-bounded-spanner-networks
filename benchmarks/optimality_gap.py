#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact minimum-size delay-bounded spanners on tiny random time-varying digraphs, and the gap to the greedy construction.

For every instance the true optimum (fewest edges such that dist_H(u,v,tau) <= t * dist_G(u,v,tau) for all pairs and all 24
hours) is found by an exhaustive search over removal sets. Supersets of a valid subgraph are valid, so the search grows the
removal set one edge at a time and only extends sets whose every smaller subset is valid (Apriori style). The forced edges of
Theorem 2 (ratio of the best detour above t) are reported separately as the lower bound.

    python benchmarks/optimality_gap.py --instances 150 --max-n 8 --max-m 22 --seed 1     # resumable

Output: results/benchmarks/optimality_gap_raw.csv  (one row per instance and t)
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import literature_benchmarks as lb  # noqa: E402
import run_spanner_benchmark as core  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
T_LIST = (1.1, 1.5, 2.0, 3.0)
TOL = 1e-7


def edge_arrays(G):
    """Edges in a fixed order and the hourly cost tensor c[h, e]."""
    edges = list(G.edges())
    cost = np.array([G[u][v]["tt"] for u, v in edges], dtype=float).T  # (24, m)
    return edges, cost


def all_pairs(n, edges, cost, mask=None):
    """Floyd-Warshall for all 24 hours at once; mask selects the edges that are present."""
    D = np.full((cost.shape[0], n, n), np.inf)
    idx = range(len(edges)) if mask is None else np.flatnonzero(mask)
    for e in idx:
        u, v = edges[e]
        D[:, u, v] = np.minimum(D[:, u, v], cost[:, e])
    for i in range(n):
        D[:, i, i] = 0.0
    for k in range(n):
        D = np.minimum(D, D[:, :, k:k + 1] + D[:, k:k + 1, :])
    return D


def is_valid(n, edges, cost, mask, t, DG):
    DH = all_pairs(n, edges, cost, mask)
    fin = np.isfinite(DG)
    return not np.any(DH[fin] > t * DG[fin] + TOL)


def forced_edges(n, edges, cost, t, DG):
    """Theorem 2: edge e is in every t-spanner iff some hour has no detour of ratio <= t without e."""
    forced = []
    m = len(edges)
    for e in range(m):
        mask = np.ones(m, dtype=bool)
        mask[e] = False
        D = all_pairs(n, edges, cost, mask)
        u, v = edges[e]
        if np.any(D[:, u, v] > t * cost[:, e] + TOL):
            forced.append(e)
    return forced


def minimum_spanner_size(n, edges, cost, t, DG):
    """Exact minimum number of edges of a valid subgraph; returns (size, forced edge indices)."""
    m = len(edges)
    forced = forced_edges(n, edges, cost, t, DG)
    free = [e for e in range(m) if e not in set(forced)]
    valid = []
    for e in free:
        mask = np.ones(m, dtype=bool)
        mask[e] = False
        if is_valid(n, edges, cost, mask, t, DG):
            valid.append(frozenset([e]))
    best = 0 if not valid else 1
    level = set(valid)
    while level:
        nxt = set()
        keys = sorted(level, key=sorted)
        for a, b in itertools.combinations(keys, 2):
            c = a | b
            if len(c) != len(a) + 1 or c in nxt:
                continue
            if all((c - {x}) in level for x in c):
                mask = np.ones(m, dtype=bool)
                mask[list(c)] = False
                if is_valid(n, edges, cost, mask, t, DG):
                    nxt.add(c)
        if nxt:
            best = len(next(iter(nxt)))
        level = nxt
    return m - best, forced


def run(args):
    path = os.path.join(OUT, "optimality_gap_raw.csv")
    os.makedirs(OUT, exist_ok=True)
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["inst", "t"])
    finished = set(zip(done.inst.astype(int), done.t.astype(float)))
    rng = random.Random(args.seed)
    deadline = time.time() + args.max_seconds
    produced = 0
    for i in range(args.instances):
        n = rng.randint(5, args.max_n)
        G = lb.random_temporal_digraph(n, rng.uniform(0.2, 0.5), rng)  # always draw, so the stream is reproducible
        if G.number_of_nodes() < 5 or G.number_of_edges() > args.max_m or G.number_of_edges() < 10:
            continue
        if all((i, t) in finished for t in T_LIST):
            continue
        if time.time() > deadline:
            print("time budget reached; rerun to resume")
            break
        edges, cost = edge_arrays(G)
        nn = G.number_of_nodes()
        DG = all_pairs(nn, edges, cost)
        rows = []
        for t in T_LIST:
            if (i, t) in finished:
                continue
            opt, forced = minimum_spanner_size(nn, edges, cost, t, DG)
            Hg = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            Hu = lb.union_of_snapshots_spanner(G, t)
            rows.append(dict(inst=i, n=nn, m=len(edges), t=t, optimum=opt, greedy=Hg.number_of_edges(),
                             union=Hu.number_of_edges(), forced=len(forced)))
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        produced += 1
    print(f"instances written this run: {produced}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=150)
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--max-m", type=int, default=22)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-seconds", type=int, default=250)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
