#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity of the conclusions to congestion severity.

Measured slowdowns differ from the model in both directions (freeways: model overstates, 1.50x vs 1.34-1.38x;
Manhattan streets: model understates, 1.50x vs 2.07x). The excess delay of every edge is rescaled,
    tt_k(h) = tt_ff * (1 + k * (tt(h)/tt_ff - 1)),
and retention / certification / forced bound / baselines are recomputed. k = 1 is the original model.

    python benchmarks/severity_sensitivity.py                                  # 4 cached city cores
    python benchmarks/severity_sensitivity.py --cities Yazd Cairo --k 0.5 1 2.14
    python benchmarks/severity_sensitivity.py --synthetic --cities Yazd         # offline smoke test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
from retention_diagnostics import per_edge_max_ratio  # noqa: E402
from literature_benchmarks import union_of_snapshots_spanner  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def rescale(G, k):
    H = copy.deepcopy(G)
    for _, _, d in H.edges(data=True):
        ff = min(d["tt"])
        d["tt"] = [ff * (1.0 + k * (x / ff - 1.0)) for x in d["tt"]]
    return H


def mean_slowdown(G):
    ratios = [max(d["tt"]) / min(d["tt"]) for _, _, d in G.edges(data=True)]
    return float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Isfahan", "Tokyo", "Cairo", "Manhattan"])
    ap.add_argument("--k", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.14])
    ap.add_argument("--t", type=float, nargs="+", default=[1.5, 2.0])
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    rows = []
    for name in a.cities:
        spec = next((c for c in core.CITY_REGISTRY if c["name"] == name), None)
        if spec is None:
            continue
        G0 = core.build_city_network(spec, grid_n=a.grid_n, force_synthetic=a.synthetic)
        if not a.synthetic and not G0.graph.get("used_real_osm"):
            print(f"[FAIL] {name}: not a real OSM network; skipped")
            continue
        m = G0.number_of_edges()
        for k in a.k:
            G = rescale(G0, k)
            edf = per_edge_max_ratio(G, max(a.t) * 1.0001)
            sl = mean_slowdown(G)
            for t in a.t:
                H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
                S = core.static_freeflow_spanner(G, t, show_progress=False)
                cs = core.temporal_stretch_certificate(G, S, t)
                U = union_of_snapshots_spanner(G, t) if abs(t - a.t[-1]) < 1e-9 else None
                rows.append(dict(city=name, k=k, mean_edge_slowdown=sl, t=t, edges=m,
                                 delay_bounded_retention_pct=100 * H.number_of_edges() / m,
                                 delay_bounded_certified=core.temporal_stretch_certificate(G, H, t)["certified"],
                                 forced_lower_bound_pct=100 * float((edf.max_ratio > t).mean()),
                                 static_ff_retention_pct=100 * S.number_of_edges() / m, static_ff_certified=cs["certified"],
                                 static_ff_violating_edges=cs["violating_edges"],
                                 static_ff_worst_stretch=cs["attained_worst_edge_stretch"],
                                 union_retention_pct=(np.nan if U is None else 100 * U.number_of_edges() / m)))
        print("[ok]", name, flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    suffix = "_synthetic" if a.synthetic else ""
    df.round(3).to_csv(os.path.join(OUT, f"severity_sensitivity{suffix}.csv"), index=False)
    g = df.groupby(["k", "t"]).agg(mean_slowdown=("mean_edge_slowdown", "mean"), ours=("delay_bounded_retention_pct", "mean"),
                                   forced=("forced_lower_bound_pct", "mean"), static_ff=("static_ff_retention_pct", "mean"),
                                   static_ff_uncertified_cities=("static_ff_certified", lambda x: int((~x).sum())),
                                   ours_all_certified=("delay_bounded_certified", "all")).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()
