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

Theorem 3b (per-edge version, immune to outlier edges).  Let rho_e be the largest ratio of consecutive sub-grid
values of edge e alone and c^_e(theta_k) = rho_e * c_e(theta_k).  If every edge e outside H has an H-path P with
    sum_{e' in P} c^_{e'}(theta_k)  <=  (T / rho_e^2) * c^_e(theta_k)         for every sub-grid point k,
then dist_{H,theta} <= T * dist_{G,theta} for all theta (same argument; an edge with rho_e^2 >= T is simply kept).
A few extreme edges then only affect themselves instead of inflating one global rho.

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


def add_fine_costs(G, m, attr="tt_fine"):
    """d[attr][k] = interpolated cost at hour k/m (k = 0 .. 24m-1)."""
    for _, _, d in G.edges(data=True):
        d[attr] = [core.td_cost(d["tt"], (k / m) * 3600.0) for k in range(24 * m)]


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


def edge_rhos(G, m, attr="tt_fine"):
    M = 24 * m
    out = {}
    for u, v, d in G.edges(data=True):
        c = d[attr]
        out[(u, v)] = max(max(c[k] / c[(k + 1) % M], c[(k + 1) % M] / c[k]) for k in range(M))
    return out


def per_edge_robust_spanner(G, T, m):
    """Greedy construction certified for ALL theta by Theorem 3b. Returns (H, number of edges kept only because rho_e^2 >= T)."""
    import heapq
    M = 24 * m
    add_fine_costs(G, m, "tt_fine_pe")
    rho = edge_rhos(G, m, "tt_fine_pe")
    chat = {e: [rho[e] * x for x in G[e[0]][e[1]]["tt_fine_pe"]] for e in G.edges()}
    order = sorted(G.edges(), key=lambda e: (min(G[e[0]][e[1]]["tt_fine_pe"]), e))
    adj = {x: [] for x in G.nodes()}
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    forced_by_rho = 0
    last_fail = 0

    def reach(u, v, k, budget):
        dist = {u: 0.0}
        pq = [(0.0, u)]
        while pq:
            d, x = heapq.heappop(pq)
            if d > dist.get(x, math.inf):
                continue
            if x == v:
                return True
            if d > budget:
                return False
            for y, cl in adj[x]:
                nd = d + cl[k]
                if nd <= budget and nd < dist.get(y, math.inf):
                    dist[y] = nd
                    heapq.heappush(pq, (nd, y))
        return False

    for e in order:
        u, v = e
        te = T / rho[e] ** 2
        keep = te < 1.0
        if keep:
            forced_by_rho += 1
        else:
            ks = [last_fail] + [k for k in range(M) if k != last_fail]
            for k in ks:
                if not reach(u, v, k, te * chat[e][k]):
                    keep = True
                    last_fail = k
                    break
        if keep:
            adj[u].append((v, chat[e]))
            H.add_edge(u, v, **G[u][v])
    return H, forced_by_rho, rho


def run_one(G, T_list, candidates, samples, rng, label, pe_m=8):
    """For every target T pick the coarsest sub-grid m in `candidates` whose rho gives t' = T / rho^2 >= 1 + (T-1)/2
    (a finer grid shrinks rho towards 1); report infeasible if none qualifies."""
    rows = []
    E = G.number_of_edges()
    for T in T_list:
        Hg = core.delay_bounded_greedy_spanner(G, T, show_progress=False)                       # grid only
        wg, badg = offgrid_violation(G, Hg, T, samples, rng)
        chosen = None
        last_rho = float("nan")
        for m in candidates:
            add_fine_costs(G, m)
            rho = rho_of(G, m)
            last_rho = rho
            tp = T / rho ** 2
            if tp >= 1.0 + 0.5 * (T - 1.0) - 1e-12:
                chosen = (m, rho, tp)
                break
        Hp, nforced, rho_e = per_edge_robust_spanner(G, T, pe_m)
        wp, badp = offgrid_violation(G, Hp, T, samples, rng)
        rv = np.array(list(rho_e.values()))
        base = dict(graph=label, T=T, grid_only_edges_pct=100 * Hg.number_of_edges() / E,
                    grid_only_worst_offgrid=wg, grid_only_violating_samples=badg,
                    peredge_m=pe_m, peredge_edges_pct=100 * Hp.number_of_edges() / E, peredge_worst_offgrid=wp,
                    peredge_violating_samples=badp, peredge_rho_forced_edges_pct=100 * nforced / E,
                    rho_e_median=float(np.median(rv)), rho_e_p99=float(np.percentile(rv, 99)), rho_e_max=float(rv.max()))
        if chosen is None:
            rows.append(dict(base, m=candidates[-1], rho=round(last_rho, 4), t_prime=float("nan"), robust_edges_pct=float("nan"),
                             robust_certified_on_subgrid=False, robust_worst_offgrid=float("nan"), robust_violating_samples=0,
                             note=f"infeasible: rho^2 = {last_rho ** 2:.3f} too large even at m = {candidates[-1]}"))
            continue
        m, rho, tp = chosen
        Hr = core.delay_bounded_greedy_spanner(G, tp, hours=list(range(24 * m)), attr="tt_fine", show_progress=False)
        cert = core.temporal_stretch_certificate(G, Hr, tp, hours=list(range(24 * m)), attr="tt_fine")["certified"]
        wr, badr = offgrid_violation(G, Hr, T, samples, rng)
        rows.append(dict(base, m=m, rho=round(rho, 4), t_prime=round(tp, 4), robust_edges_pct=100 * Hr.number_of_edges() / E,
                         robust_certified_on_subgrid=bool(cert), robust_worst_offgrid=wr, robust_violating_samples=badr, note=""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dataset", choices=["metrla", "pemsbay"], default=None)
    ap.add_argument("--speeds", default=None)
    ap.add_argument("--T", type=float, nargs="+", default=[1.2, 1.5, 2.0])
    ap.add_argument("--m", type=int, nargs="+", default=[4, 8, 16], help="candidate sub-grid points per hour, coarse to fine (4 = 15 min)")
    ap.add_argument("--pe-m", type=int, default=8, help="sub-grid points per hour for the per-edge construction")
    ap.add_argument("--samples", type=int, default=24, help="random off-grid times for the empirical check")
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    if a.synthetic:
        for i in range(a.instances):
            G = lb.random_temporal_digraph(rng.randint(12, 20), rng.uniform(0.25, 0.5), rng)
            rows += run_one(G, a.T, a.m, a.samples, rng, f"random-{i}", a.pe_m)
    if a.dataset:
        G = lb.load_metrla(a.speeds, os.path.join(OUT, "metrla_cache"), dataset=a.dataset)
        rows += run_one(G, a.T, a.m, a.samples, rng, a.dataset, a.pe_m)
    if not rows:
        ap.error("use --synthetic and/or --dataset")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, f"offgrid_guarantee_{a.dataset or 'synthetic'}.csv"), index=False)
    print(df[["graph", "T", "m", "rho", "t_prime", "grid_only_edges_pct", "robust_edges_pct", "note"]].round(3).to_string(index=False))
    print("\nPer-edge robust construction (Theorem 3b), sub-grid m =", a.pe_m)
    print(df[["graph", "T", "grid_only_edges_pct", "peredge_edges_pct", "peredge_rho_forced_edges_pct", "peredge_violating_samples",
              "peredge_worst_offgrid", "rho_e_median", "rho_e_p99", "rho_e_max"]].round(3).to_string(index=False))
    print("per-edge robust: off-grid violations:", int(df.peredge_violating_samples.sum()))
    feas = df[df.note == ""]
    print("global-rho rows infeasible:", int((df.note != "").sum()), "of", len(df))
    if not len(feas):
        if df.peredge_violating_samples.sum() > 0:
            sys.exit(1)
        sys.exit(0)
    df = feas
    g = df.groupby(["T"]).agg(n=("graph", "size"), rho_mean=("rho", "mean"), m_max=("m", "max"), grid_edges=("grid_only_edges_pct", "mean"),
                              grid_viol_samples=("grid_only_violating_samples", "sum"), grid_worst=("grid_only_worst_offgrid", "max"),
                              robust_edges=("robust_edges_pct", "mean"), robust_viol_samples=("robust_violating_samples", "sum"),
                              robust_worst=("robust_worst_offgrid", "max")).round(3)
    print(g.to_string())
    print("robust certificate always satisfied on sub-grid:", bool(df.robust_certified_on_subgrid.all()))
    print("robust construction: off-grid violations:", int(df.robust_violating_samples.sum()))
    if df.robust_violating_samples.sum() > 0 or df.peredge_violating_samples.sum() > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
