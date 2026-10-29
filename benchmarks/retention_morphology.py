#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-t behaviour of edge retention and its relation to street-network morphology.

For every city: forced-edge lower bound (Theorem 2, docs/THEOREM_FORCED_EDGES.md) and
delay-bounded-greedy retention for t in a wide grid, the smallest grid t at which
retention drops to <= 90 / 80 %, and Spearman correlations (Benjamini-Hochberg
corrected across all tests) between retention / forced fraction and morphology
descriptors (orientation entropy, circuity, mean degree, edge-length CV, bridge share).

    python benchmarks/retention_morphology.py                 # all cached cities
    python benchmarks/retention_morphology.py --synthetic --cities Yazd Tokyo Manhattan Cairo
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
from retention_diagnostics import per_edge_max_ratio  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        run = min(run, p[idx] * n / rank)
        adj[idx] = run
    return adj


def first_t_below(df_city, col, level):
    ok = df_city[df_city[col] <= level]
    return float(ok.t.min()) if len(ok) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--t", type=float, nargs="+", default=[1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=0)
    args = ap.parse_args()

    specs = [c for c in core.CITY_REGISTRY if args.cities is None or c["name"] in args.cities]
    cap = max(args.t) * 1.0001
    rows, meta = [], []
    for spec in specs:
        G = core.build_city_network(spec, grid_n=args.grid_n, force_synthetic=args.synthetic)
        if args.max_nodes and G.number_of_nodes() > args.max_nodes:
            continue
        m = G.number_of_edges()
        edf = per_edge_max_ratio(G, cap)
        morph = core.network_morphology(G)
        meta.append(dict(city=spec["name"], country=spec["country"], archetype=spec["archetype"],
                         bridge_pct=100 * float(np.isinf(edf.max_ratio).mean()), **morph))
        for t in args.t:
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            rows.append(dict(city=spec["name"], t=t, retention_pct=100 * H.number_of_edges() / m,
                             forced_pct=100 * float((edf.max_ratio > t).mean())))
        print(f"[ok] {spec['name']}")
    R = pd.DataFrame(rows)
    R["gap_pct"] = R.retention_pct - R.forced_pct
    M = pd.DataFrame(meta)
    os.makedirs(OUT, exist_ok=True)
    R.round(3).to_csv(os.path.join(OUT, "retention_large_t.csv"), index=False)

    thr = []
    for c, g in R.groupby("city"):
        thr.append(dict(city=c, t_retention_le_90=first_t_below(g, "retention_pct", 90.0),
                        t_retention_le_80=first_t_below(g, "retention_pct", 80.0),
                        t_forced_le_90=first_t_below(g, "forced_pct", 90.0),
                        t_forced_le_80=first_t_below(g, "forced_pct", 80.0)))
    T = pd.DataFrame(thr).merge(M, on="city")
    T.round(3).to_csv(os.path.join(OUT, "retention_thresholds_by_city.csv"), index=False)

    desc = ["orientation_entropy", "circuity_mean", "mean_degree", "edge_length_cv", "bridge_pct"]
    tests = []
    for t in args.t:
        sub = R[R.t == t].merge(M, on="city")
        if len(sub) < 5:
            continue
        for target in ("retention_pct", "forced_pct", "gap_pct"):
            for d in desc:
                if d == "bridge_pct" and target == "forced_pct":
                    continue   # tautological: forced share tends to the bridge share as t grows
                x, y = sub[d].values, sub[target].values
                if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                    continue
                rho, p = spearmanr(x, y, nan_policy="omit")
                tests.append(dict(t=t, target=target, descriptor=d, n=len(sub), spearman_rho=rho, p_raw=p))
    C = pd.DataFrame(tests)
    if len(C):
        C["p_BH"] = bh_adjust(C.p_raw.values)
        C["significant_BH_0.05"] = C.p_BH < 0.05
        C.round(4).to_csv(os.path.join(OUT, "retention_morphology_correlations.csv"), index=False)

    piv = R.pivot_table(index="city", columns="t", values=["retention_pct", "forced_pct"]).round(1)
    print("\n", piv.to_string())
    print("\nMean over cities:\n", R.groupby("t")[["retention_pct", "forced_pct", "gap_pct"]].mean().round(2).to_string())
    print("\nSmallest grid t with retention <= 90% / 80% (nan = never within grid):\n",
          T[["city", "t_retention_le_90", "t_retention_le_80", "t_forced_le_90", "t_forced_le_80"]].to_string(index=False))
    if len(C):
        print(f"\nSpearman tests: {len(C)}; significant after BH: {int(C['significant_BH_0.05'].sum())}")
        print(C.sort_values("p_BH").head(12).round(4).to_string(index=False))
    with open(os.path.join(OUT, "retention_morphology_summary.md"), "w", encoding="utf-8") as f:
        f.write(R.groupby("t")[["retention_pct", "forced_pct", "gap_pct"]].mean().round(2).to_markdown())
        f.write("\n\n")
        f.write(T[["city", "t_retention_le_90", "t_retention_le_80"]].round(2).to_markdown(index=False))


if __name__ == "__main__":
    main()
