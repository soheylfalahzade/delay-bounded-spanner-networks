#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How much does direction matter?

Real road networks contain one-way streets.  This script compares each network with its two-way version, in which every
arc that has no reverse arc gets one (same hourly travel times, reversed geometry); arcs that already have a reverse arc
are unchanged.  For both versions and each t it computes the forced lower bound (Theorem 2), the retention of the
delay-bounded greedy spanner and the share of the removable slack that it removes, and then relates the difference to the
share of one-way arcs across cities (Spearman, Benjamini-Hochberg adjusted).

The two versions are different graphs with different edge counts; percentages are of each graph's own arcs.

    python benchmarks/directedness.py --t 2 3
    python benchmarks/directedness.py --synthetic --registry-only --t 2        # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
import more_cities as mc  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def two_way(G):
    """Copy of G in which every arc without a reverse arc gets a reverse arc with the same hourly costs."""
    G2 = G.copy()
    for u, v, d in list(G.edges(data=True)):
        if not G.has_edge(v, u):
            nd = dict(d)
            if nd.get("coords"):
                nd["coords"] = list(reversed(nd["coords"]))
            G2.add_edge(v, u, **nd)
    return G2


def one_way_pct(G) -> float:
    m = G.number_of_edges()
    return 100.0 * sum(1 for u, v in G.edges() if not G.has_edge(v, u)) / max(1, m)


def measure(G, ts) -> Dict[float, Dict[str, float]]:
    cap = max(ts) * 1.0001
    edf = rd.per_edge_max_ratio(G, cap)
    m = G.number_of_edges()
    out = {}
    for t in ts:
        H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        forced = 100.0 * float((edf.max_ratio > t).mean())
        ret = 100.0 * H.number_of_edges() / m
        slack = 100.0 - forced
        out[t] = dict(forced=forced, retention=ret, capture=((100.0 - ret) / slack if slack > 1e-9 else float("nan")))
    return out


def city_rows(G, name: str, source: str, ts) -> List[Dict]:
    G2 = two_way(G)
    a, b = measure(G, ts), measure(G2, ts)
    rows = []
    for t in ts:
        rows.append(dict(city=name, source=source, real_osm=bool(G.graph.get("used_real_osm")),
                         arcs_directed=G.number_of_edges(), arcs_two_way=G2.number_of_edges(),
                         one_way_pct=one_way_pct(G), t=t,
                         forced_directed=a[t]["forced"], retention_directed=a[t]["retention"],
                         capture_directed=a[t]["capture"],
                         forced_two_way=b[t]["forced"], retention_two_way=b[t]["retention"],
                         capture_two_way=b[t]["capture"],
                         diff_forced=a[t]["forced"] - b[t]["forced"],
                         diff_retention=a[t]["retention"] - b[t]["retention"]))
    return rows


def correlations(df: pd.DataFrame, min_cities: int = 8) -> pd.DataFrame:
    rows = []
    for t, d in df.groupby("t"):
        if len(d) < min_cities or d.one_way_pct.nunique() < 2:
            continue
        for target in ("diff_forced", "diff_retention"):
            rho, p = stats.spearmanr(d.one_way_pct, d[target])
            rows.append(dict(t=t, target=target, n=len(d), spearman_rho=float(rho), p_raw=float(p)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_BH"] = mc.bh_adjust(out.p_raw.to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--registry-only", action="store_true")
    ap.add_argument("--max-cities", type=int, default=0, help="limit the number of ADDED cities (0 = all)")
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rows: List[Dict] = []
    for spec in core.CITY_REGISTRY:
        G = load_city(spec, a.grid_n, a.synthetic)
        if G is not None:
            rows += city_rows(G, spec["name"], "registry", a.t)
            print(f"[ok] {spec['name']}")
    if not a.registry_only:
        added = mc.ADDED_CITIES[: a.max_cities] if a.max_cities else mc.ADDED_CITIES
        for name, country, lat, lon, _ in added:
            G = mc.load_real(name, lat, lon)
            if G is not None:
                rows += city_rows(G, name, "added", a.t)
                print(f"[ok] {name}")
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "directedness.csv"), index=False)
    cor = correlations(df)
    cor.round(4).to_csv(os.path.join(OUT, "directedness_correlations.csv"), index=False)
    with open(os.path.join(OUT, "directedness.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df.groupby("t").mean(numeric_only=True).reset_index(), 2))
    print(md_table(df.groupby("t").mean(numeric_only=True).reset_index(), 2))


if __name__ == "__main__":
    main()
