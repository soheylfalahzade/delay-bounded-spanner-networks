#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Region faults on a delay-bounded spanner.

Definition (Abam, de Berg, Farshi, Gudmundsson 2009, adapted to road networks).
For a convex region F, let G-F be the network after every intersection inside F and
every street segment that touches F has been removed, and let H-F be the same
operation applied to the spanner H.  H is *region-fault tolerant for a family
{F}* if, for every F in the family and every hour tau, H-F is a t-spanner of G-F.
The reference graph is G-F, not G: a fault is applied to both graphs, exactly as in
the geometric definition (where the reference is the complete graph minus F).

What this script measures (all on the SAME definition):

  A. Nominal H under faults.  The delay-bounded spanner H built for the intact
     network is not designed for faults.  For sampled regions F we count how often
     H-F fails to be a t-spanner of G-F, and how many street segments of G-F
     violate the bound (exact edge-wise certificate of Theorem 1 applied to the
     pair (G-F, H-F)).

  B. Forced-edge lower bound under faults.  If an edge e of G-F has detour ratio
     r_{G-F}(e) > t, then e belongs to every edge-subset spanner that is
     t-spanning for G-F.  The union of these edges over the sampled family, together
     with the nominal forced edges, is therefore a lower bound on the size of ANY
     edge-subset spanner that is region-fault tolerant for that family.
     Since G-F is a subgraph of G, r_{G-F}(e) >= r_G(e): nominally forced edges stay
     forced, so only nominally non-forced edges are re-examined.

  C. A fault-tolerant greedy construction.  The same greedy rule, but an edge is
     dropped only if it has a detour within budget in H-F for EVERY sampled F that
     leaves the edge intact and every hour (plus the intact network).  By the argument
     of Theorem 1 applied to each G-F, the result is certified for the sampled family;
     this is verified exactly.  Generalisation to regions NOT in the family is not
     guaranteed and is measured on held-out regions drawn with a different seed.

Scope (stated, not hidden): regions are discs; a street segment is treated as
touching F if its stored polyline comes within the disc; the family is a finite random
sample, so "tolerant" always means "for this family"; the traffic model is the
parametric one used elsewhere in this repository.

    python benchmarks/region_fault.py --cities Yazd Isfahan Manhattan Tokyo --t 2
    python benchmarks/region_fault.py --synthetic --cities Yazd --regions 8 --heldout 10   # offline smoke
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))
Edge = Tuple[object, object]


# ------------------------------------------------------------------ geometry
def local_frame(G: nx.DiGraph):
    """Return (node -> (x, y) in metres, edge -> polyline in metres) in a local frame."""
    lat0, lon0 = G.graph["center"]
    mlat, mlon = core.local_metric_factors(lat0)

    def conv(lon: float, lat: float) -> Tuple[float, float]:
        return ((lon - lon0) * mlon, (lat - lat0) * mlat)

    nodes = {n: conv(d["x"], d["y"]) for n, d in G.nodes(data=True)}
    polys = {}
    for u, v, d in G.edges(data=True):
        pts = d.get("coords") or [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        polys[(u, v)] = [conv(x, y) for x, y in pts]
    return nodes, polys


def _point_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    s = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + s * dx), py - (ay + s * dy))


def polyline_touches_disc(poly: Sequence[Tuple[float, float]], cx: float, cy: float, r: float) -> bool:
    if len(poly) == 1:
        return math.hypot(poly[0][0] - cx, poly[0][1] - cy) <= r
    for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
        if _point_segment_dist(cx, cy, ax, ay, bx, by) <= r:
            return True
    return False


def make_scenario(G: nx.DiGraph, frame, cx: float, cy: float, r: float) -> Dict:
    """Disc fault: nodes inside the disc are removed, and every edge that touches the
    disc or is incident to a removed node is removed."""
    nodes, polys = frame
    removed_nodes = {n for n, (x, y) in nodes.items() if math.hypot(x - cx, y - cy) <= r}
    removed_edges: Set[Edge] = set()
    for (u, v), poly in polys.items():
        if u in removed_nodes or v in removed_nodes or polyline_touches_disc(poly, cx, cy, r):
            removed_edges.add((u, v))
    return {"cx": cx, "cy": cy, "r": r, "removed_nodes": removed_nodes, "removed_edges": removed_edges}


def sample_scenarios(G: nx.DiGraph, n: int, radius_fracs: Sequence[float], seed: int) -> List[Dict]:
    """n disc faults centred on random intersections, radii cycling through radius_fracs * network radius.
    Discs that would leave fewer than half of the intersections are discarded."""
    frame = local_frame(G)
    nodes = frame[0]
    ids = sorted(nodes, key=str)
    rng = random.Random(seed)
    dist = float(G.graph.get("dist", core.DEFAULT_DIST))
    out: List[Dict] = []
    tries = 0
    while len(out) < n and tries < 50 * max(1, n):
        c = ids[rng.randrange(len(ids))]
        r = radius_fracs[len(out) % len(radius_fracs)] * dist
        sc = make_scenario(G, frame, nodes[c][0], nodes[c][1], r)
        tries += 1
        if len(sc["removed_nodes"]) <= 0.5 * len(ids) and len(sc["removed_edges"]) > 0:
            out.append(sc)
    return out


def faulted_edges(G: nx.DiGraph, sc: Dict) -> List[Tuple[object, object, dict]]:
    rem = sc["removed_edges"]
    return [(u, v, d) for u, v, d in G.edges(data=True) if (u, v) not in rem]


# ------------------------------------------------------------- A: certificate
def adjacency(edges: Sequence[Tuple[object, object, dict]], nodes) -> Dict:
    adj = {n: {} for n in nodes}
    for u, v, d in edges:
        adj[u][v] = d["tt"]
    return adj


def count_violations(G: nx.DiGraph, H_edges: Set[Edge], sc: Dict, t: float) -> Tuple[int, int]:
    """Exact Theorem-1 certificate of (H-F) against (G-F): number of edges of G-F whose detour in H-F
    exceeds t * w at some hour, and the number of edges of G-F.  Edges of H-F satisfy the bound trivially
    (the edge itself is the path), so only edges outside H are probed."""
    rem = sc["removed_edges"]
    alive = [(u, v, d) for u, v, d in G.edges(data=True) if (u, v) not in rem]
    adjH = {n: {} for n in G.nodes()}
    for u, v, d in alive:
        if (u, v) in H_edges:
            adjH[u][v] = d["tt"]
    bad = 0
    for u, v, d in alive:
        if (u, v) in H_edges:
            continue
        costs = d["tt"]
        for tau in H24:
            budget = t * costs[tau]
            if core.bounded_dijkstra(adjH, u, v, tau, budget) > budget + core.EPS:
                bad += 1
                break
    return bad, len(alive)


def evaluate_under_faults(G: nx.DiGraph, H: nx.DiGraph, scenarios: Sequence[Dict], t: float) -> Dict:
    H_edges = set(H.edges())
    n_fail = 0
    rates = []
    for sc in scenarios:
        bad, m = count_violations(G, H_edges, sc, t)
        n_fail += int(bad > 0)
        rates.append(bad / max(1, m))
    k = max(1, len(scenarios))
    return {"regions": len(scenarios), "regions_failing": n_fail, "regions_failing_pct": 100.0 * n_fail / k,
            "mean_violating_edge_pct": 100.0 * float(np.mean(rates)) if rates else float("nan"),
            "max_violating_edge_pct": 100.0 * float(np.max(rates)) if rates else float("nan")}


# --------------------------------------------------- B: forced-edge lower bound
def forced_edges_nominal(G: nx.DiGraph, t: float) -> Set[Edge]:
    cap = t * 1.0001
    df = rd.per_edge_max_ratio(G, cap)
    return {(r.u, r.v) for r in df.itertuples() if r.max_ratio > t}


def forced_edges_under_fault(G: nx.DiGraph, sc: Dict, t: float, nominal_forced: Set[Edge],
                             adj_by_hour_cache: Dict | None = None) -> Set[Edge]:
    """Edges of G-F with detour ratio > t in G-F.  Nominally forced edges that survive stay forced
    (G-F is a subgraph of G), so only the remaining surviving edges are examined."""
    rem = sc["removed_edges"]
    alive = [(u, v, d) for u, v, d in G.edges(data=True) if (u, v) not in rem]
    adj_by_hour = []
    for tau in H24:
        a = {n: [] for n in G.nodes()}
        for u, v, d in alive:
            a[u].append((v, d["tt"][tau]))
        adj_by_hour.append(a)
    cap = t * 1.0001
    out: Set[Edge] = set()
    for u, v, d in alive:
        if (u, v) in nominal_forced:
            out.add((u, v))
            continue
        for tau in H24:
            if rd.detour_ratio(adj_by_hour[tau], u, v, d["tt"][tau], tau, cap) > t:
                out.add((u, v))
                break
    return out


def fault_forced_lower_bound(G: nx.DiGraph, scenarios: Sequence[Dict], t: float,
                             nominal_forced: Set[Edge] | None = None) -> Tuple[Set[Edge], Set[Edge]]:
    nominal = forced_edges_nominal(G, t) if nominal_forced is None else nominal_forced
    union = set(nominal)
    for sc in scenarios:
        union |= forced_edges_under_fault(G, sc, t, nominal)
    return nominal, union


# --------------------------------------------- C: fault-tolerant greedy spanner
def fault_tolerant_greedy(G: nx.DiGraph, t: float, scenarios: Sequence[Dict]) -> nx.DiGraph:
    """Greedy rule applied to the intact network and to every scenario G-F: an edge is dropped only if,
    for every scenario that leaves it intact and every hour, the spanner built so far offers a detour
    within t * w in H-F.  Certified for the given family by Theorem 1 applied to each G-F."""
    ordered = core._edge_order(G, "temporal_max", "tt", H24)
    removed = [set()] + [sc["removed_edges"] for sc in scenarios]
    adjs = [{n: {} for n in G.nodes()} for _ in removed]
    kept = []
    for u, v, d in ordered:
        costs = d["tt"]
        slice_order = sorted(H24, key=lambda tau: costs[tau])
        add = False
        for s, rem in enumerate(removed):
            if (u, v) in rem:
                continue
            adj = adjs[s]
            for tau in slice_order:
                budget = t * costs[tau]
                if core.bounded_dijkstra(adj, u, v, tau, budget) > budget + core.EPS:
                    add = True
                    break
            if add:
                break
        if add:
            for s, rem in enumerate(removed):
                if (u, v) not in rem:
                    adjs[s][u][v] = costs
            kept.append((u, v, d))
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in kept:
        H.add_edge(u, v, **d)
    H.graph.update(G.graph)
    H.graph.update({"method": "fault_tolerant_greedy", "t": t, "regions_in_family": len(scenarios)})
    return H


# ------------------------------------------------------------------ driver
def run_city(spec: Dict, t: float, n_regions: int, n_heldout: int, radius_fracs: Sequence[float],
             seed: int, synthetic: bool, grid_n: int) -> Dict:
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return None
    m = G.number_of_edges()
    train = sample_scenarios(G, n_regions, radius_fracs, seed)
    held = sample_scenarios(G, n_heldout, radius_fracs, seed + 1000)
    H0 = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    HF = fault_tolerant_greedy(G, t, train)

    nominal_forced, fault_lb = fault_forced_lower_bound(G, train, t)
    cert_train_ft = evaluate_under_faults(G, HF, train, t)
    assert cert_train_ft["regions_failing"] == 0, "fault-tolerant greedy failed on its own family: bug"
    ev0_train = evaluate_under_faults(G, H0, train, t)
    ev0_held = evaluate_under_faults(G, H0, held, t)
    evF_held = evaluate_under_faults(G, HF, held, t)

    ret0 = 100.0 * H0.number_of_edges() / m
    retF = 100.0 * HF.number_of_edges() / m
    lb0 = 100.0 * len(nominal_forced) / m
    lbF = 100.0 * len(fault_lb) / m
    assert retF + 1e-9 >= lbF and ret0 + 1e-9 >= lb0 and retF + 1e-9 >= ret0, "ordering violated: bug"
    return dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), nodes=G.number_of_nodes(),
                edges=m, t=t, regions_train=len(train), regions_heldout=len(held),
                radius_fracs="/".join(f"{x:g}" for x in radius_fracs),
                nominal_retention_pct=ret0, nominal_forced_lb_pct=lb0,
                fault_forced_lb_pct=lbF, fault_tolerant_retention_pct=retF,
                extra_edges_for_fault_tolerance_pct=retF - ret0,
                nominal_fails_train_regions_pct=ev0_train["regions_failing_pct"],
                nominal_mean_violating_edge_pct=ev0_train["mean_violating_edge_pct"],
                nominal_fails_heldout_regions_pct=ev0_held["regions_failing_pct"],
                ft_fails_train_regions_pct=cert_train_ft["regions_failing_pct"],
                ft_fails_heldout_regions_pct=evF_held["regions_failing_pct"],
                ft_mean_violating_edge_heldout_pct=evF_held["mean_violating_edge_pct"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--regions", type=int, default=20, help="regions in the training family")
    ap.add_argument("--heldout", type=int, default=40, help="fresh regions (different seed) for generalisation")
    ap.add_argument("--radius-fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                    help="disc radii as fractions of the network radius")
    ap.add_argument("--seed", type=int, default=core.SEED)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    specs = [c for c in core.CITY_REGISTRY if c["name"] in args.cities]
    if not specs:
        sys.exit("no matching cities; available: " + ", ".join(c["name"] for c in core.CITY_REGISTRY))
    rows = []
    for spec in specs:
        row = run_city(spec, args.t, args.regions, args.heldout, args.radius_fracs, args.seed,
                       args.synthetic, args.grid_n)
        if row is None:
            continue
        rows.append(row)
        print(f"[ok] {spec['name']}: nominal fails {row['nominal_fails_train_regions_pct']:.0f}% of regions; "
              f"retention {row['nominal_retention_pct']:.1f} -> {row['fault_tolerant_retention_pct']:.1f} "
              f"(fault lower bound {row['fault_forced_lb_pct']:.1f}); held-out failure of FT "
              f"{row['ft_fails_heldout_regions_pct']:.0f}%")
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(3).to_csv(os.path.join(OUT, "region_fault.csv"), index=False)
    with open(os.path.join(OUT, "region_fault.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
