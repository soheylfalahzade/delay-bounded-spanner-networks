#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-grid (continuous-time snapshot) guarantee.

Theorem 1 certifies the bound only at the 24 grid hours. With the piecewise-linear interpolation used
by the repository (td_cost), a spanner certified at the grid can slightly exceed t at intermediate times.

Theorem 3 (interpolation-robust certificate).  Let c_e(theta) be the piecewise-linear hourly profile and let
theta_0..theta_{M-1} be a uniform sub-grid (M = 24*m points).  Let
    rho = max over edges e and k of max( c_e(theta_{k+1}) / c_e(theta_k),  c_e(theta_k) / c_e(theta_{k+1}) ).
If H satisfies the edge-wise certificate with factor t' at every sub-grid point, then for EVERY theta in [0,24):
    dist_{H,theta}(x, y) <= rho^2 * t' * dist_{G,theta}(x, y)      for all x, y.
Proof sketch: for theta in [theta_k, theta_{k+1}] the cost c_e(theta) lies between its end-point values, so
c_e(theta) <= rho c_e(theta_k) and c_e(theta_k) <= rho c_e(theta). Take the H-path P for edge e at theta_k:
cost_theta(P) <= rho cost_{theta_k}(P) <= rho t' c_e(theta_k) <= rho^2 t' c_e(theta); Theorem 1 at the fixed
snapshot theta then gives the all-pairs bound.  Choosing t' = T / rho^2 yields a continuous-time bound T.

    python benchmarks/offgrid_guarantee.py --synthetic                     # random time-varying digraphs
    python benchmarks/offgrid_guarantee.py --dataset metrla  --speeds ~/data/METR-LA.csv
    python benchmarks/offgrid_guarantee.py --dataset pemsbay --speeds ~/data/PEMS-BAY.csv
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import literature_benchmarks as lb  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def add_fine_costs(G, m):
    """tt_fine[k] = interpolated cost at hour k/m (k = 0 .. 24m-1)."""
    for _, _, d in G.edges(data=True):
        d["tt_fine"] = [core.td_cost(d["tt"], (k / m) * 3600.0) for k in range(24 * m)]


def rho_of(G, m):
    r = 1.0
    M = 24 * m
    for _, _, d in G.edges(data=True):
        c = d["tt_fine"]
        for k in range(M):
            a, b = c[k], c[(k + 1) % M]
            r = max(r, a / b, b / a)
    return r


def offgrid_violation(G, H, T, samples, rng):
    """Largest observed ratio dist_H / dist_G over edges at random off-grid times (empirical check)."""
    worst = 1.0
    bad = 0
    for _ in range(samples):
        th = rng.uniform(0, 24)
        adjH = {x: [] for x in H.nodes()}
        for u, v, d in H.edges(data=True):
            adjH[u].append((v, core.td_cost(G[u][v]["tt"], th * 3600.0)))
        for u, v, d in G.edges(data=True):
            w = core.td_cost(d["tt"], th * 3600.0)
            dh = _dij(adjH, u, v, 50 * w)
            ratio = dh / w
            worst = max(worst, ratio)
            if ratio > T + 1e-9:
                bad += 1
    return worst, bad


def _dij(adj, s, t, budget):
    import heapq
    dist = {s: 0.0}
    pq = [(0.0, s)]
    while pq:
        d, x = heapq.heappop(pq)
        if d > dist.get(x, math.inf):
            continue
        if x == t:
            return d
        if d > budget:
            break
        for y, c in adj[x]:
            nd = d + c
            if nd < dist.get(y, math.inf) and nd <= budget:
                dist[y] = nd
                heapq.heappush(pq, (nd, y))
    return math.inf


def run_one(G, T_list, m, samples, rng, label):
    add_fine_costs(G, m)
    rho = rho_of(G, m)
    rows = []
    for T in T_list:
        Hg = core.delay_bounded_greedy_spanner(G, T, show_progress=False)                       # grid only
        wg, badg = offgrid_violation(G, Hg, T, samples, rng)
        tp = T / rho ** 2
        Hr = core.delay_bounded_greedy_spanner(G, tp, hours=list(range(24 * m)), attr="tt_fine", show_progress=False)
        cert = core.temporal_stretch_certificate(G, Hr, tp, hours=list(range(24 * m)), attr="tt_fine")["certified"]
        wr, badr = offgrid_violation(G, Hr, T, samples, rng)
        E = G.number_of_edges()
        rows.append(dict(graph=label, T=T, m=m, rho=round(rho, 4), t_prime=round(tp, 4),
                         grid_only_edges_pct=100 * Hg.number_of_edges() / E, grid_only_worst_offgrid=wg, grid_only_violating_samples=badg,
                         robust_edges_pct=100 * Hr.number_of_edges() / E, robust_certified_on_subgrid=bool(cert),
                         robust_worst_offgrid=wr, robust_violating_samples=badr))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dataset", choices=["metrla", "pemsbay"], default=None)
    ap.add_argument("--speeds", default=None)
    ap.add_argument("--T", type=float, nargs="+", default=[1.2, 1.5, 2.0])
    ap.add_argument("--m", type=int, default=4, help="sub-grid points per hour (4 = 15 min)")
    ap.add_argument("--samples", type=int, default=24, help="random off-grid times for the empirical check")
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    if a.synthetic:
        for i in range(a.instances):
            G = lb.random_temporal_digraph(rng.randint(12, 20), rng.uniform(0.25, 0.5), rng)
            rows += run_one(G, a.T, a.m, a.samples, rng, f"random-{i}")
    if a.dataset:
        G = lb.load_metrla(a.speeds, os.path.join(OUT, "metrla_cache"), dataset=a.dataset)
        rows += run_one(G, a.T, a.m, a.samples, rng, a.dataset)
    if not rows:
        ap.error("use --synthetic and/or --dataset")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "offgrid_guarantee.csv"), index=False)
    g = df.groupby(["T"]).agg(n=("graph", "size"), rho_mean=("rho", "mean"), grid_edges=("grid_only_edges_pct", "mean"),
                              grid_viol_samples=("grid_only_violating_samples", "sum"), grid_worst=("grid_only_worst_offgrid", "max"),
                              robust_edges=("robust_edges_pct", "mean"), robust_viol_samples=("robust_violating_samples", "sum"),
                              robust_worst=("robust_worst_offgrid", "max")).round(3)
    print(g.to_string())
    print("robust certificate always satisfied on sub-grid:", bool(df.robust_certified_on_subgrid.all()))
    print("robust construction: off-grid violations:", int(df.robust_violating_samples.sum()))
    if df.robust_violating_samples.sum() > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
