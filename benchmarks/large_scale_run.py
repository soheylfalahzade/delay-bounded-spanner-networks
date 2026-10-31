#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Larger-radius evaluation (default 5 km instead of the 1.5 km city cores).

For each city: build the OSM network with an overridden radius, then report for several t
  * retention of the delay-bounded greedy spanner and certification (Theorem 1),
  * forced-edge lower bound (Theorem 2),
  * static free-flow baseline (certified or not),
  * union of per-hour static greedy spanners (only at --union-t; 24 greedy runs).

    python benchmarks/large_scale_run.py                                   # Isfahan Tokyo Manhattan, 5 km
    python benchmarks/large_scale_run.py --cities Yazd Cairo --dist 4000 --t 1.5 2 3
    python benchmarks/large_scale_run.py --synthetic --cities Yazd         # offline smoke test
Output: results/benchmarks/large_scale_<dist>m.csv / .md
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
from retention_diagnostics import per_edge_max_ratio  # noqa: E402
from literature_benchmarks import union_of_snapshots_spanner  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Isfahan", "Tokyo", "Manhattan"])
    ap.add_argument("--dist", type=int, default=5000)
    ap.add_argument("--t", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    ap.add_argument("--union-t", type=float, default=2.0)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-edges", type=int, default=60000)
    a = ap.parse_args()

    rows = []
    for name in a.cities:
        spec = next((c for c in core.CITY_REGISTRY if c["name"] == name), None)
        if spec is None:
            print("[skip] unknown city", name)
            continue
        spec = dict(spec, dist=a.dist)
        t0 = time.time()
        G = core.build_city_network(spec, grid_n=a.grid_n, force_synthetic=a.synthetic)
        m = G.number_of_edges()
        print(f"[{name}] {G.number_of_nodes()} nodes / {m} edges (real OSM: {bool(G.graph.get('used_real_osm'))}) "
              f"built in {time.time() - t0:.0f}s", flush=True)
        if m > a.max_edges:
            print(f"[skip] {name}: more than --max-edges")
            continue
        edf = per_edge_max_ratio(G, max(a.t) * 1.0001)
        for t in a.t:
            t1 = time.time()
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            cert = core.temporal_stretch_certificate(G, H, t)
            S = core.static_freeflow_spanner(G, t, show_progress=False)
            cs = core.temporal_stretch_certificate(G, S, t)
            row = dict(city=name, dist_m=a.dist, real_osm=bool(G.graph.get("used_real_osm")), nodes=G.number_of_nodes(), edges=m, t=t,
                       delay_bounded_retention_pct=100 * H.number_of_edges() / m, delay_bounded_certified=cert["certified"],
                       forced_lower_bound_pct=100 * float((edf.max_ratio > t).mean()),
                       static_ff_retention_pct=100 * S.number_of_edges() / m, static_ff_certified=cs["certified"],
                       static_ff_violating_edges=cs["violating_edges"], union_retention_pct=np.nan, union_certified=np.nan)
            if abs(t - a.union_t) < 1e-9:
                U = union_of_snapshots_spanner(G, t)
                row["union_retention_pct"] = 100 * U.number_of_edges() / m
                row["union_certified"] = core.temporal_stretch_certificate(G, U, t)["certified"]
            rows.append(row)
            print(f"   t={t}: ours {row['delay_bounded_retention_pct']:.1f}% (certified={cert['certified']}), "
                  f"forced bound {row['forced_lower_bound_pct']:.1f}%, static-ff {row['static_ff_retention_pct']:.1f}% "
                  f"(certified={cs['certified']}), {time.time() - t1:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"large_scale_{a.dist}m" + ("_synthetic" if a.synthetic else ""))
    df.round(3).to_csv(stem + ".csv", index=False)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(df.round(2).to_markdown(index=False))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
