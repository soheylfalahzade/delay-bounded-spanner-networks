#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emergency response times on the full network and on the spanner, with and without a regional fault.

A response base (station) must reach every intersection.  For each station and each target the travel time in the full
network is compared with the travel time in the spanner, at one hour of the day (default 17:00, the evening peak):

    ratio = d_H(station, target) / d_G(station, target)

* intact network: the certificate guarantees ratio <= t, the interesting number is how close to t the ratios get;
* after a disc fault F: stations and targets outside F, travel times in G-F and H-F.  Two spanners are compared: the
  one built for the intact network, and the fault-tolerant greedy spanner built for a training family of faults
  (benchmarks/region_fault.py).  Targets reachable in G-F but not in H-F are counted as unreachable.

Stations are the nodes chosen by farthest-point sampling from the intersection nearest the centre of the network; they stand
for response bases, not for real hospitals or fire stations.  Faults are discs; the traffic profile is the parametric model.

    python benchmarks/emergency_response.py --cities Yazd Isfahan Manhattan Tokyo --t 2
    python benchmarks/emergency_response.py --synthetic --cities Yazd --regions 6 --heldout 8     # offline smoke test
"""
from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import region_fault as rf  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def sssp(adj: Dict, source) -> Dict:
    dist = {source: 0.0}
    pq = [(0.0, 0, source)]
    c = 1
    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, c, v))
                c += 1
    return dist


def adjacency_hour(edges, hour: int) -> Dict:
    adj: Dict = {}
    for u, v, d in edges:
        adj.setdefault(u, []).append((v, d["tt"][hour]))
    return adj


def pick_stations(G, k: int):
    nodes, _ = rf.local_frame(G)
    ids = sorted(nodes, key=str)
    c = min(ids, key=lambda n: math.hypot(*nodes[n]))
    chosen = [c]
    while len(chosen) < min(k, len(ids)):
        far = max(ids, key=lambda n: min(math.hypot(nodes[n][0] - nodes[s][0], nodes[n][1] - nodes[s][1])
                                         for s in chosen))
        chosen.append(far)
    return chosen


def compare(G, H_edges: Set, removed_nodes: Set, removed_edges: Set, stations, hour: int, t: float):
    """Ratios (and unreachable count) of H-F against G-F for the given stations."""
    Gf = [(u, v, d) for u, v, d in G.edges(data=True) if (u, v) not in removed_edges]
    Hf = [(u, v, d) for u, v, d in Gf if (u, v) in H_edges]
    aG, aH = adjacency_hour(Gf, hour), adjacency_hour(Hf, hour)
    ratios: List[float] = []
    unreachable = pairs = 0
    for s in stations:
        if s in removed_nodes:
            continue
        dG, dH = sssp(aG, s), sssp(aH, s)
        for x, g in dG.items():
            if x == s or g <= 0:
                continue
            pairs += 1
            h = dH.get(x)
            if h is None:
                unreachable += 1
            else:
                ratios.append(h / g)
    return ratios, unreachable, pairs


def summarise(ratios, unreachable, pairs, t) -> Dict:
    r = np.array(ratios) if ratios else np.array([float("nan")])
    return dict(pairs=pairs, unreachable_pct=100.0 * unreachable / max(1, pairs),
                mean_ratio=float(np.nanmean(r)), p95_ratio=float(np.nanpercentile(r, 95)),
                max_ratio=float(np.nanmax(r)), over_t_pct=100.0 * float((r > t + 1e-9).sum()) / max(1, pairs))


def run_city(spec, t, hour, n_stations, n_regions, n_heldout, radius_fracs, seed, synthetic, grid_n) -> List[Dict]:
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return []
    train = rf.sample_scenarios(G, n_regions, radius_fracs, seed)
    held = rf.sample_scenarios(G, n_heldout, radius_fracs, seed + 1000)
    H0 = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    HF = rf.fault_tolerant_greedy(G, t, train)
    stations = pick_stations(G, n_stations)
    sets = {"intact": [dict(removed_nodes=set(), removed_edges=set())], "training faults": train,
            "held-out faults": held}
    rows = []
    for sname, scs in sets.items():
        for hname, H in (("intact-network spanner", H0), ("fault-tolerant spanner", HF)):
            he = set(H.edges())
            R: List[float] = []
            U = P = 0
            for sc in scs:
                r, u, p = compare(G, he, sc["removed_nodes"], sc["removed_edges"], stations, hour, t)
                R += r
                U += u
                P += p
            rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), t=t, hour=hour,
                             stations=len(stations), spanner=hname, scenario=sname, regions=len(scs),
                             **summarise(R, U, P, t)))
        print(f"[ok] {spec['name']} {sname}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--hour", type=int, default=17)
    ap.add_argument("--stations", type=int, default=5)
    ap.add_argument("--regions", type=int, default=20)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--radius-fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=core.SEED)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    specs = [c for c in core.CITY_REGISTRY if c["name"] in a.cities]
    rows = []
    for s in specs:
        rows += run_city(s, a.t, a.hour, a.stations, a.regions, a.heldout, a.radius_fracs, a.seed, a.synthetic, a.grid_n)
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "emergency_response.csv"), index=False)
    with open(os.path.join(OUT, "emergency_response.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df))
    print(md_table(df))


if __name__ == "__main__":
    main()
