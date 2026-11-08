#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Total street length instead of the number of edges.

The number of arcs is not what a road authority maintains; length is.  For each network and t this script reports, on
physical street segments (two arcs of a two-way street count once, a segment is kept if any of its arcs is kept):

  * the share of the total street length kept by the delay-bounded greedy spanner;
  * lightness = kept length / length of a minimum spanning tree of the street segments (the standard spanner measure);
  * the length of the forced segments (a segment with a forced arc is in every spanner) as a lower bound on the length of
    ANY edge-subset spanner, and the share of the removable length that the construction removes.

    python benchmarks/lightness.py --cities Yazd Isfahan Manhattan Tokyo --t 2 3
    python benchmarks/lightness.py --synthetic --registry-only --t 2         # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import networkx as nx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
import more_cities as mc  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def segments(G) -> Dict[frozenset, float]:
    """Physical street segments: one entry per unordered node pair, with the segment length in metres."""
    seg: Dict[frozenset, float] = {}
    for u, v, d in G.edges(data=True):
        key = frozenset((u, v))
        seg[key] = max(seg.get(key, 0.0), float(d.get("length", 0.0)))
    return seg


def mst_weight(seg: Dict[frozenset, float]) -> float:
    U = nx.Graph()
    for k, w in seg.items():
        a, b = tuple(k) if len(k) == 2 else (next(iter(k)),) * 2
        if a != b:
            U.add_edge(a, b, weight=w)
    return float(sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(U, data=True)))


def measure(G, t: float, ratios=None) -> Dict[str, float]:
    seg = segments(G)
    total = sum(seg.values())
    mst = mst_weight(seg)
    H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    kept_keys = {frozenset(e) for e in H.edges()}
    kept = sum(w for k, w in seg.items() if k in kept_keys)
    if ratios is None:
        ratios = rd.per_edge_max_ratio(G, t * 1.0001)
    forced_keys = {frozenset((r.u, r.v)) for r in ratios.itertuples() if r.max_ratio > t}
    forced = sum(w for k, w in seg.items() if k in forced_keys)
    assert forced <= kept + 1e-6 <= total + 2e-6, "forced length <= kept length <= total length violated: bug"
    slack = total - forced
    return dict(segments=len(seg), total_km=total / 1000.0, mst_km=mst / 1000.0,
                kept_length_pct=100.0 * kept / total, kept_count_pct=100.0 * H.number_of_edges() / G.number_of_edges(),
                forced_length_pct=100.0 * forced / total, lightness_graph=total / mst, lightness_spanner=kept / mst,
                lightness_lower_bound=forced / mst,
                capture_length=((total - kept) / slack if slack > 1e-9 else float("nan")))


def city_rows(G, name: str, source: str, ts) -> List[Dict]:
    cap = max(ts) * 1.0001
    edf = rd.per_edge_max_ratio(G, cap)
    return [dict(city=name, source=source, real_osm=bool(G.graph.get("used_real_osm")), t=t, **measure(G, t, edf))
            for t in ts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=None, help="registry cities (default: all twelve)")
    ap.add_argument("--t", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--registry-only", action="store_true")
    ap.add_argument("--max-cities", type=int, default=0)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rows: List[Dict] = []
    for spec in core.CITY_REGISTRY:
        if a.cities and spec["name"] not in a.cities:
            continue
        G = load_city(spec, a.grid_n, a.synthetic)
        if G is not None:
            rows += city_rows(G, spec["name"], "registry", a.t)
            print(f"[ok] {spec['name']}")
    if not a.registry_only and not a.cities:
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
    df.round(4).to_csv(os.path.join(OUT, "lightness.csv"), index=False)
    with open(os.path.join(OUT, "lightness.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df.groupby("t").mean(numeric_only=True).reset_index(), 2))
    print(md_table(df.groupby("t").mean(numeric_only=True).reset_index(), 2))


if __name__ == "__main__":
    main()
