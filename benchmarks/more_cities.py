#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
More cities: does the forced-edge picture hold beyond the twelve registry cities, and does a larger sample resolve the
morphology question?

The twelve registry cities give a low-powered morphology test (README 5d).  This script adds further city cores (about 1.5 km
radius, real OpenStreetMap data only; a city whose download fails is skipped, never replaced by a synthetic network),
computes for each city and t the forced lower bound (Theorem 2), the retention of the delay-bounded greedy spanner and the
share of the removable slack it removes, together with the morphology descriptors of the core.  Over the pooled sample it reports Spearman
correlations of retention and of slack capture with the descriptors, Benjamini-Hochberg-adjusted over all tests.

    python benchmarks/more_cities.py                         # registry cities + all added cities
    python benchmarks/more_cities.py --only-added --max-cities 6
    python benchmarks/more_cities.py --synthetic --max-cities 2 --registry-only        # offline smoke test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
import forced_by_hour as fb  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")

# name, country, latitude, longitude, archetype (the archetype is used only as a cache label; no synthetic fallback exists here)
ADDED_CITIES = [
    ("London", "UK", 51.5074, -0.1278, "historic-organic"), ("Paris", "France", 48.8566, 2.3522, "radial-grid"),
    ("Berlin", "Germany", 52.5200, 13.4050, "radial-grid"), ("Rome", "Italy", 41.9028, 12.4964, "historic-organic"),
    ("Istanbul", "Turkey", 41.0082, 28.9784, "hill-organic"), ("Mumbai", "India", 19.0760, 72.8777, "historic-organic"),
    ("Delhi", "India", 28.6139, 77.2090, "radial-grid"), ("Beijing", "China", 39.9042, 116.4074, "modern-grid"),
    ("Seoul", "South Korea", 37.5665, 126.9780, "hill-organic"), ("Singapore", "Singapore", 1.3521, 103.8198, "modern-grid"),
    ("Sydney", "Australia", -33.8688, 151.2093, "hill-organic"), ("Sao Paulo", "Brazil", -23.5505, -46.6333, "historic-organic"),
    ("Mexico City", "Mexico", 19.4326, -99.1332, "radial-grid"), ("Buenos Aires", "Argentina", -34.6037, -58.3816, "strict-grid"),
    ("Chicago", "USA", 41.8781, -87.6298, "strict-grid"), ("San Francisco", "USA", 37.7749, -122.4194, "strict-grid"),
    ("Toronto", "Canada", 43.6532, -79.3832, "modern-grid"), ("Nairobi", "Kenya", -1.2921, 36.8219, "historic-organic"),
    ("Lagos", "Nigeria", 6.5244, 3.3792, "historic-organic"), ("Moscow", "Russia", 55.7558, 37.6173, "radial-grid"),
    ("Bangkok", "Thailand", 13.7563, 100.5018, "historic-organic"), ("Jakarta", "Indonesia", -6.2088, 106.8456, "historic-organic"),
    ("Tabriz", "Iran", 38.0962, 46.2738, "historic-organic"), ("Ahvaz", "Iran", 31.3183, 48.6706, "modern-grid"),
    ("Kerman", "Iran", 30.2839, 57.0834, "historic-organic"), ("Rasht", "Iran", 37.2808, 49.5832, "historic-organic"),
]
DESCRIPTORS = ["bridge_pct", "true_bridge_pct", "mean_degree", "circuity_mean", "orientation_entropy", "edge_length_cv"]


def load_real(name: str, lat: float, lon: float, dist: int = 1500) -> Optional["core.nx.DiGraph"]:
    """Real OSM core of one added city, cached like the registry cities; None if the download fails."""
    center = (lat, lon)
    key = core._cache_key("added_city", center, dist, {"contract": True})
    pkl, meta = core._cache_paths(key, name)
    G = core.load_graph_cache(pkl, meta)
    if G is not None:
        return G
    G = core.fetch_osm_network(center, dist)
    if G is None:
        print(f"[skip] {name}: real OpenStreetMap data unavailable")
        return None
    raw_n, raw_e = G.number_of_nodes(), G.number_of_edges()
    G = core.contract_degree_two(G)
    G.graph.update({"raw_nodes": raw_n, "raw_edges": raw_e, "city": name, "used_real_osm": True, "center": list(center),
                    "dist": dist})
    G = core.enrich_temporal_costs(G)
    G.graph["cache_hit"] = False
    core.save_graph_cache(G, pkl, meta)
    return G


def city_rows(G, name: str, country: str, source: str, ts) -> List[Dict]:
    cap = max(ts) * 1.0001
    edf = rd.per_edge_max_ratio(G, cap)
    morph = core.network_morphology(G)
    m = G.number_of_edges()
    extra = dict(bridge_pct=100.0 * float(np.isinf(edf.max_ratio).mean()), true_bridge_pct=fb.true_bridge_pct(G),
                 mean_degree=morph["mean_degree"], circuity_mean=morph["circuity_mean"],
                 orientation_entropy=morph["orientation_entropy"], edge_length_cv=morph["edge_length_cv"])
    rows = []
    for t in ts:
        H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        forced = 100.0 * float((edf.max_ratio > t).mean())
        ret = 100.0 * H.number_of_edges() / m
        slack = 100.0 - forced
        assert ret + 1e-9 >= forced, "lower bound violated: bug"
        rows.append(dict(city=name, country=country, source=source, real_osm=bool(G.graph.get("used_real_osm")),
                         nodes=G.number_of_nodes(), edges=m, t=t, retention_pct=ret, forced_lower_bound_pct=forced,
                         capture=((100.0 - ret) / slack if slack > 1e-9 else float("nan")), **extra))
    return rows


def bh_adjust(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[idx] * m / rank)
        adj[idx] = prev
    return adj


def correlations(df: pd.DataFrame, ts, min_cities: int = 8) -> pd.DataFrame:
    rows = []
    for t in ts:
        d = df[df.t == t]
        if len(d) < min_cities:
            continue
        for target in ("retention_pct", "capture"):
            for desc in DESCRIPTORS:
                x, y = d[desc].astype(float), d[target].astype(float)
                ok = x.notna() & y.notna()
                if ok.sum() < min_cities or x[ok].nunique() < 2 or y[ok].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(x[ok], y[ok])
                rows.append(dict(t=t, target=target, descriptor=desc, n=int(ok.sum()), spearman_rho=float(rho),
                                 p_raw=float(p)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_BH"] = bh_adjust(out.p_raw.to_numpy())
        out["significant_BH_0.05"] = out.p_BH < 0.05
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--registry-only", action="store_true")
    ap.add_argument("--only-added", action="store_true")
    ap.add_argument("--max-cities", type=int, default=0, help="limit the number of ADDED cities (0 = all)")
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    rows: List[Dict] = []
    if not a.only_added:
        for spec in core.CITY_REGISTRY:
            G = load_city(spec, a.grid_n, a.synthetic)
            if G is not None:
                rows += city_rows(G, spec["name"], spec["country"], "registry", a.t)
                print(f"[ok] {spec['name']}")
    if not a.registry_only:
        added = ADDED_CITIES[: a.max_cities] if a.max_cities else ADDED_CITIES
        for name, country, lat, lon, _arch in added:
            G = load_real(name, lat, lon)
            if G is not None:
                rows += city_rows(G, name, country, "added", a.t)
                print(f"[ok] {name}")
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "more_cities.csv"), index=False)
    cor = correlations(df, a.t)
    cor.round(4).to_csv(os.path.join(OUT, "more_cities_correlations.csv"), index=False)
    summ = (df.groupby("t").agg(cities=("city", "nunique"), forced_mean=("forced_lower_bound_pct", "mean"),
                                retention_mean=("retention_pct", "mean"), capture_mean=("capture", "mean"),
                                capture_min=("capture", "min"), capture_max=("capture", "max")).reset_index())
    with open(os.path.join(OUT, "more_cities.md"), "w", encoding="utf-8") as f:
        f.write(md_table(summ, 3))
        if len(cor):
            f.write("\n\n" + md_table(cor, 3))
    print(md_table(summ, 3))
    if len(cor):
        print(md_table(cor, 3))


if __name__ == "__main__":
    main()
