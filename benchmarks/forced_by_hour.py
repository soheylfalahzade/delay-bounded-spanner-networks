#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does time variation change which edges are forced?

Theorem 2 calls an edge e forced if its detour ratio r_tau(e) = d_{G-e,tau}(u,v) / w_tau(e)
exceeds t at SOME hour tau.  This script splits the forced set by hour, to answer a question
a reader will ask: is the forced-edge bound just the static (free-flow) bound in disguise,
or does the hourly profile add forced edges?

For every edge and stretch t it classifies the edge as

  forced_all_hours     r_tau(e) > t at every hour          (forced independently of time of day)
  forced_some_hours    r_tau(e) > t at some but not all    (time-dependent forcedness)
  forced_free_flow     r_ff(e)  > t with free-flow weights (what a static analysis would see)
  time_only_forced     forced at some hour, NOT forced at free flow  (forced only because of time variation)
  static_only_forced   forced at free flow, NOT forced at any hour   (the static analysis over-forces)
  peak_not_night       forced at some peak hour (07-09, 16-18) and at no night hour (00-05)

It also reports the TRUE bridge share (arcs with no alternative directed path at all), which is a
stricter notion than the `bridge_pct` column of the retention CSVs (see true_bridge_pct()).

and records the share of edges forced at each hour.  Everything is computed exactly (detour
Dijkstra per edge per hour), with the same primitive as benchmarks/retention_diagnostics.py.

    python benchmarks/forced_by_hour.py --cities Yazd Isfahan Manhattan Tokyo --t 1.5 2 3
    python benchmarks/forced_by_hour.py --synthetic --cities Yazd --t 2          # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))
PEAK = (7, 8, 9, 16, 17, 18)
NIGHT = (0, 1, 2, 3, 4, 5)


def hourly_ratios(G, cap: float):
    """(array [E x 24] of detour ratios, array [E] of free-flow detour ratios, edge list)."""
    adj_h = [{x: [(y, d["tt"][tau]) for y, d in G[x].items()] for x in G.nodes()} for tau in H24]
    adj_ff = {x: [(y, d["free_flow"]) for y, d in G[x].items()] for x in G.nodes()}
    edges = list(G.edges(data=True))
    R = np.empty((len(edges), 24))
    rff = np.empty(len(edges))
    for i, (u, v, d) in enumerate(edges):
        for tau in H24:
            R[i, tau] = rd.detour_ratio(adj_h[tau], u, v, d["tt"][tau], tau, cap)
        rff[i] = rd.detour_ratio(adj_ff, u, v, d["free_flow"], 0, cap)
    return R, rff, edges


def true_bridge_pct(G) -> float:
    """Share of arcs (u,v) for which v is NOT reachable from u in G-(u,v) at all (unbounded, hour-independent).
    This is the true bridge share.  The `bridge_pct` column of retention_diagnostics.csv / retention_large_t.csv is
    different: it is the share of arcs with NO detour within `cap` times the arc's own travel time (cap =
    1.0001 * the largest t of that run), which is an upper bound on the true bridge share."""
    adj = {x: list(G[x]) for x in G.nodes()}
    n_bridge = 0
    for u, v in G.edges():
        seen, stack, found = {u}, [u], False
        while stack and not found:
            x = stack.pop()
            for y in adj[x]:
                if x == u and y == v:
                    continue
                if y == v:
                    found = True
                    break
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        n_bridge += 0 if found else 1
    return 100.0 * n_bridge / max(1, G.number_of_edges())


def classify(R: np.ndarray, rff: np.ndarray, t: float) -> Dict[str, float]:
    f = R > t
    any_h, all_h = f.any(axis=1), f.all(axis=1)
    ff = rff > t
    peak = f[:, list(PEAK)].any(axis=1)
    night = f[:, list(NIGHT)].any(axis=1)
    m = float(len(R))
    pct = lambda mask: 100.0 * float(mask.sum()) / m  # noqa: E731
    return dict(forced_any_pct=pct(any_h), forced_all_hours_pct=pct(all_h),
                forced_some_hours_pct=pct(any_h & ~all_h), forced_free_flow_pct=pct(ff),
                time_only_forced_pct=pct(any_h & ~ff), static_only_forced_pct=pct(ff & ~any_h),
                peak_not_night_pct=pct(peak & ~night),
                share_of_forced_that_is_time_dependent=(float((any_h & ~all_h).sum() / any_h.sum())
                                                         if any_h.sum() else float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--t", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=0)
    args = ap.parse_args()

    specs = [c for c in core.CITY_REGISTRY if args.cities is None or c["name"] in args.cities]
    if not specs:
        sys.exit("no matching cities; available: " + ", ".join(c["name"] for c in core.CITY_REGISTRY))
    cap = max(args.t) * 1.0001
    rows: List[Dict] = []
    profile: List[Dict] = []
    for spec in specs:
        G = core.build_city_network(spec, grid_n=args.grid_n, force_synthetic=args.synthetic)
        if args.max_nodes and G.number_of_nodes() > args.max_nodes:
            print(f"[skip] {spec['name']}: {G.number_of_nodes()} nodes")
            continue
        R, rff, edges = hourly_ratios(G, cap)
        tb = true_bridge_pct(G)
        for t in args.t:
            c = classify(R, rff, t)
            rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")),
                             edges=len(edges), t=t, true_bridge_pct=tb, **c))
            for tau in H24:
                profile.append(dict(city=spec["name"], t=t, hour=tau,
                                    forced_pct=100.0 * float((R[:, tau] > t).mean())))
        print(f"[ok] {spec['name']}: {len(edges)} edges")
    df = pd.DataFrame(rows)
    pf = pd.DataFrame(profile)
    os.makedirs(OUT, exist_ok=True)
    df.round(3).to_csv(os.path.join(OUT, "forced_by_hour.csv"), index=False)
    pf.round(3).to_csv(os.path.join(OUT, "forced_by_hour_profile.csv"), index=False)
    mean = df.groupby("t").mean(numeric_only=True).drop(columns=["edges"]).round(2)
    with open(os.path.join(OUT, "forced_by_hour.md"), "w", encoding="utf-8") as f:
        f.write(df.round(2).to_markdown(index=False))
        f.write("\n\nMean over cities:\n\n")
        f.write(mean.to_markdown())
    print("\nMean over cities:\n", mean.to_string())


if __name__ == "__main__":
    main()
