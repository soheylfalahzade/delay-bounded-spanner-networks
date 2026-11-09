#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dose-response for one-way streets.

Section 5s compares each network with its fully two-way version.  Here the one-way arcs are made two-way gradually, in a
fixed random order (nested: the arcs added at 25% are among those added at 50%, and so on), at 0, 25, 50, 75 and 100 percent.
At each dose it reports the forced share and the retention of the delay-bounded greedy spanner, both over all arcs of the
modified network and over the ORIGINAL arcs only (so that the denominator does not change with the dose).

    python benchmarks/oneway_dose.py --t 2 3
    python benchmarks/oneway_dose.py --synthetic --registry-only --t 2         # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
import more_cities as mc  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)


def dose_order(G, seed: int):
    one_way = sorted(((u, v) for u, v in G.edges() if not G.has_edge(v, u)), key=str)
    random.Random(seed).shuffle(one_way)
    return one_way


def with_dose(G, order, frac: float):
    """G plus the reverse arc (same hourly costs, reversed geometry) of the first round(frac * len(order)) one-way arcs."""
    k = int(round(frac * len(order)))
    G2 = G.copy()
    for u, v in order[:k]:
        d = dict(G[u][v])
        if d.get("coords"):
            d["coords"] = list(reversed(d["coords"]))
        G2.add_edge(v, u, **d)
    return G2, k


def measure(G, G_dose, t: float, edf) -> Dict[str, float]:
    orig = set(G.edges())
    H = core.delay_bounded_greedy_spanner(G_dose, t, show_progress=False)
    forced = {(r.u, r.v) for r in edf.itertuples() if r.max_ratio > t}
    kept = set(H.edges())
    n_all, n_orig = G_dose.number_of_edges(), len(orig)
    return dict(forced_all_pct=100.0 * len(forced) / n_all, retention_all_pct=100.0 * len(kept) / n_all,
                forced_original_pct=100.0 * len(forced & orig) / n_orig,
                retention_original_pct=100.0 * len(kept & orig) / n_orig)


def city_rows(G, name: str, source: str, ts, seed: int) -> List[Dict]:
    order = dose_order(G, seed)
    one_way_pct = 100.0 * len(order) / max(1, G.number_of_edges())
    cap = max(ts) * 1.0001
    rows = []
    for frac in DOSES:
        G2, k = with_dose(G, order, frac)
        edf = rd.per_edge_max_ratio(G2, cap)
        for t in ts:
            rows.append(dict(city=name, source=source, real_osm=bool(G.graph.get("used_real_osm")),
                             one_way_pct=one_way_pct, dose=frac, arcs_added=k, t=t, **measure(G, G2, t, edf)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--registry-only", action="store_true")
    ap.add_argument("--max-cities", type=int, default=0)
    ap.add_argument("--seed", type=int, default=core.SEED)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rows: List[Dict] = []
    for spec in core.CITY_REGISTRY:
        G = load_city(spec, a.grid_n, a.synthetic)
        if G is not None:
            rows += city_rows(G, spec["name"], "registry", a.t, a.seed)
            print(f"[ok] {spec['name']}")
    if not a.registry_only:
        added = mc.ADDED_CITIES[: a.max_cities] if a.max_cities else mc.ADDED_CITIES
        for name, country, lat, lon, _ in added:
            G = mc.load_real(name, lat, lon)
            if G is not None:
                rows += city_rows(G, name, "added", a.t, a.seed)
                print(f"[ok] {name}")
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "oneway_dose.csv"), index=False)
    summ = df.groupby(["t", "dose"]).mean(numeric_only=True).reset_index()
    with open(os.path.join(OUT, "oneway_dose.md"), "w", encoding="utf-8") as f:
        f.write(md_table(summ, 2))
    print(md_table(summ, 2))


if __name__ == "__main__":
    main()
