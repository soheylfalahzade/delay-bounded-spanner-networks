#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Delay-Bounded Time-Varying Geometric t-Spanners for Urban Road Networks
 ------------------------------------------------------------------------------
 Reference implementation + Q1-grade experimental protocol (single file).
================================================================================

PROBLEM
-------
G = (V, E) is a directed geometric road network. Every edge e carries a length
l(e) and an hourly speed profile v(e, tau), tau in T = {0,...,23}, giving the
time-sliced cost

        w(e, tau) = l(e) / v(e, tau).

We seek a sparse backbone H subset of E certified to satisfy

        forall u,v in V, forall tau in T :
            dist_H(u, v, tau) <= t * dist_G(u, v, tau).

THEOREM 1 (edge-wise certificate lifts to all pairs, all hours)
---------------------------------------------------------------
If  dist_H(u,v,tau) <= t * w((u,v),tau)  for every (u,v) in E and every tau in T,
then H is a delay-bounded time-varying t-spanner of G.
Proof: fix tau. w(.,tau) is a fixed non-negative weighting, so G_tau is a static
digraph. Concatenating the edge-wise invariant along a tau-optimal u->v path and
using the triangle inequality of shortest-path distances in H_tau gives the
all-pairs bound. The argument is independent across tau.  []

Consequence: the |V|^2 |T| pairwise constraints collapse to |E| |T| *local*
constraints, each decided by a Dijkstra pruned to the ball of radius t*w(e,tau).

PROPOSITION 2 (continuous-time transfer)
----------------------------------------
Let w~(e, .) be the cyclic piecewise-linear interpolant of w(e, .) and assume
FIFO, i.e. d/ds (s + w~(e,s)) >= 0 (verified numerically here). If H satisfies
the snapshot certificate for parameter t and
    kappa = max_e max_s  w~(e,s) / min(w(e,floor(s)), w(e,ceil(s))) ,
then the departure-time-dependent dilation of H is at most kappa * t. Because
the interpolant is convex-free between consecutive hourly samples, kappa = 1 for
piecewise-linear profiles, so the snapshot certificate transfers exactly. This
is validated empirically by a FIFO time-dependent Dijkstra (Section 9).

WHAT THIS REVISION FIXES RELATIVE TO A NAIVE IMPLEMENTATION
-----------------------------------------------------------
R1. Degenerate temporal model. A per-class multiplicative congestion factor makes
    w(.,tau) = c(tau) * w(.,0): every snapshot is homothetic, so the static and
    the delay-bounded spanner coincide by construction and the experiment cannot
    discriminate them. Here congestion is *edge-heterogeneous*: it depends on the
    CBD distance field, on the edge's orientation relative to the centre
    (inbound-morning / outbound-evening tidal flow), on functional class, and on
    a deterministic per-edge idiosyncratic component.
R2. Topological unit of sparsification. Sparsifying the raw OSM node graph is
    meaningless: most vertices are geometry-carrying degree-2 shape points and
    their edges are non-redundant by construction. The network is contracted to
    its junction graph (geometry preserved as polylines) before spanner
    construction, which is the correct combinatorial object.
R3. Parameter range. t in [1.2, 1.8] sits below the detour spectrum of an urban
    grid; the reported 0.8-3.2% edge removal is an artefact of the range, not a
    result. The sweep now spans t in [1.05, 3.0].
R4. Censored metrics. Assigning a sentinel dilation (e.g. 25.0) to unreachable
    pairs poisons every aggregate (a reported "mean dilation" of 23.9 is a
    sentinel average, not a dilation). Reachability is now reported as its own
    metric and dilation statistics are computed on reachable pairs only.
R5. Broken hierarchical baseline. Component re-attachment that terminates on a
    round cap leaves a disconnected graph. Replaced by last-mile access
    attachment to the largest arterial SCC via multi-source Dijkstra, which is
    both correct and what practitioners actually do.
R6. No control for density. Certified sparsity is only meaningful against
    equal-budget alternatives: matched-density random and betweenness-top-k
    sparsifiers are added, yielding a proper Pareto front.
R7. Certificate reported as a bare boolean. The *attained* worst-case temporal
    stretch of every construction is now measured, so "not certified" comes with
    a magnitude.
R8. No uncertainty, no ablation, no scaling. Added: multi-seed OD resampling with
    bootstrap CIs, a Wilcoxon signed-rank test against the matched-density
    control, an edge-ordering ablation, and an empirical runtime/probe scaling
    study.

USAGE
-----
    python run_spanner_benchmark.py
    python run_spanner_benchmark.py --quick
    python run_spanner_benchmark.py --synthetic
    python run_spanner_benchmark.py --lat 35.6892 --lon 51.3890 --dist 2000

OUTPUTS
-------
    results/spanner_benchmark.png    7-panel publication figure
    results/spanner_report.json      full machine-readable record
    results/benchmark_table.md       main benchmark table
    results/ablation_table.md        edge-ordering ablation
    results/per_pair_dilation.csv    per-OD-pair dilation at the peak hour
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import time
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from scipy import stats as sps
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# ==============================================================================
# SECTION 1 - CONFIGURATION
# ==============================================================================

CENTER_POINT: Tuple[float, float] = (31.8974, 54.3569)     # Yazd centre (lat, lon)
DEFAULT_DIST: int = 1500                                   # metres
NETWORK_TYPE: str = "drive"

HOURS: List[int] = list(range(24))
SEC_PER_HOUR: float = 3600.0

# Dilation sweep: spans the full detour spectrum of an urban street grid.
T_VALUES: List[float] = [1.05, 1.15, 1.30, 1.50, 1.80, 2.20, 2.60, 3.00]
T_FOCUS: float = 1.50

N_SOURCES: int = 14
N_TARGETS_PER_SOURCE: int = 12
N_REPEATS: int = 3                       # independent OD resamplings
SEED: int = 42
BOOTSTRAP_N: int = 2000

EPS: float = 1e-9
INF: float = float("inf")

CERT_REFINE_CAP: int = 400               # max violating edges refined for magnitude
CERT_REFINE_MULT: float = 8.0            # refinement search budget = mult * w(e,tau)

ROOT: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(ROOT, "results")
FIGURE_PATH: str = os.path.join(RESULTS_DIR, "spanner_benchmark.png")
REPORT_PATH: str = os.path.join(RESULTS_DIR, "spanner_report.json")
TABLE_PATH: str = os.path.join(RESULTS_DIR, "benchmark_table.md")
ABLATION_PATH: str = os.path.join(RESULTS_DIR, "ablation_table.md")
PERPAIR_PATH: str = os.path.join(RESULTS_DIR, "per_pair_dilation.csv")

# class -> (free-flow speed km/h, baseline congestion susceptibility, tier)
HIGHWAY_PROFILES: Dict[str, Tuple[float, float, str]] = {
    "motorway":       (100.0, 0.20, "arterial"),
    "motorway_link":  ( 70.0, 0.26, "arterial"),
    "trunk":          ( 85.0, 0.28, "arterial"),
    "trunk_link":     ( 60.0, 0.30, "arterial"),
    "primary":        ( 60.0, 0.46, "arterial"),
    "primary_link":   ( 45.0, 0.44, "arterial"),
    "secondary":      ( 50.0, 0.50, "collector"),
    "secondary_link": ( 40.0, 0.46, "collector"),
    "tertiary":       ( 42.0, 0.44, "collector"),
    "tertiary_link":  ( 35.0, 0.40, "collector"),
    "unclassified":   ( 32.0, 0.32, "local"),
    "residential":    ( 30.0, 0.28, "local"),
    "living_street":  ( 20.0, 0.20, "local"),
    "service":        ( 20.0, 0.18, "local"),
    "road":           ( 30.0, 0.30, "local"),
}
DEFAULT_PROFILE: Tuple[float, float, str] = (30.0, 0.30, "local")

HIGHWAY_ORDER: List[str] = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link", "unclassified", "residential", "road",
    "living_street", "service",
]
HIGHWAY_RANK: Dict[str, int] = {h: i for i, h in enumerate(HIGHWAY_ORDER)}

ARTERIAL_CLASSES: Set[str] = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
}

SPEED_FLOOR_FRAC: float = 0.18           # congestion never drops below this share of v_ff

METHOD_STYLE: Dict[str, Tuple[str, str, str, str]] = {
    "delay_bounded":  ("Delay-Bounded Spanner (ours)",   "#c0392b", "-",  "o"),
    "static_ff":      ("Static Free-Flow Spanner",       "#2980b9", "--", "s"),
    "static_len":     ("Static Geometric Spanner",       "#8e44ad", "--", "v"),
    "mean_temporal":  ("Mean-Temporal Spanner",          "#16a085", "-.", "P"),
    "hierarchical":   ("Hierarchical Road Classifier",   "#27ae60", "-.", "^"),
    "matched_random": ("Matched-Density Random",         "#f39c12", ":",  "X"),
    "matched_btw":    ("Matched-Density Betweenness",    "#d35400", ":",  "*"),
    "full":           ("Full Network (ground truth)",    "#7f8c8d", ":",  "D"),
}


# ==============================================================================
# SECTION 2 - GEOMETRY HELPERS
# ==============================================================================

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(max(0.0, a)))


def local_metric_factors(lat0: float) -> Tuple[float, float]:
    """Metres per degree latitude / longitude at latitude lat0."""
    mlat = 111320.0
    mlon = 111320.0 * max(0.15, math.cos(math.radians(lat0)))
    return mlat, mlon


def normalize_highway(value) -> str:
    if isinstance(value, (list, tuple, set)):
        best, best_rank = None, 10 ** 6
        for cand in value:
            if isinstance(cand, str) and cand in HIGHWAY_PROFILES:
                r = HIGHWAY_RANK.get(cand, 10 ** 5)
                if r < best_rank:
                    best, best_rank = cand, r
        if best is not None:
            return best
        value = next(iter(value), "residential")
    if not isinstance(value, str):
        return "residential"
    return value if value in HIGHWAY_PROFILES else "residential"


def dominant_class(a: str, b: str) -> str:
    return a if HIGHWAY_RANK.get(a, 10 ** 5) <= HIGHWAY_RANK.get(b, 10 ** 5) else b


def stable_unit(key: str) -> float:
    """Deterministic pseudo-random number in [0,1) derived from a string key."""
    h = 2166136261
    for ch in key:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % 100003) / 100003.0


# ==============================================================================
# SECTION 3 - SPATIO-TEMPORAL CONGESTION MODEL  (fixes R1)
# ==============================================================================

MORNING_PEAK, MORNING_SIGMA = 8.0, 1.30
EVENING_PEAK, EVENING_SIGMA = 17.5, 1.75
MIDDAY_PEAK, MIDDAY_SIGMA = 12.5, 3.00


def _gauss(tau: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((tau - mu) / sigma) ** 2)


def edge_congestion_curve(inbound: float, cbd: float, jitter: float) -> List[float]:
    """c_e(tau) in [0,1]: tidal, CBD-weighted, edge-idiosyncratic congestion.

    inbound in [0,1] : 1 = edge points toward the centre, 0 = away from it.
    cbd     in [0,1] : proximity of the edge midpoint to the centre.
    jitter  in [0,1] : deterministic per-edge idiosyncrasy.

    Morning peaks load inbound edges, evening peaks load outbound edges; the
    amplitude of both scales with CBD proximity. This breaks the homothety
    w(.,tau) = c(tau) w(.,0) that makes static and temporal spanners identical.
    """
    outbound = 1.0 - inbound
    am = (0.35 + 0.95 * inbound) * (0.45 + 0.85 * cbd)
    ae = (0.35 + 0.95 * outbound) * (0.45 + 0.95 * cbd)
    phase_m = MORNING_PEAK + 0.9 * (jitter - 0.5)
    phase_e = EVENING_PEAK + 1.2 * (jitter - 0.5)
    curve = []
    for tau in HOURS:
        c = (am * _gauss(tau, phase_m, MORNING_SIGMA)
             + ae * _gauss(tau, phase_e, EVENING_SIGMA)
             + (0.18 + 0.35 * cbd) * _gauss(tau, MIDDAY_PEAK, MIDDAY_SIGMA)
             + 0.04 + 0.06 * jitter)
        curve.append(float(min(1.0, max(0.0, c))))
    return curve


def speed_curve(free_flow_kmh: float, alpha: float, congestion: Sequence[float]) -> List[float]:
    """v(e, tau) in km/h, floored at SPEED_FLOOR_FRAC of free flow."""
    floor = SPEED_FLOOR_FRAC * free_flow_kmh
    return [max(floor, free_flow_kmh * (1.0 - alpha * c)) for c in congestion]


# ==============================================================================
# SECTION 4 - SNAPSHOT SHORTEST-PATH PRIMITIVES
# ==============================================================================

Adjacency = Dict[object, Dict[object, Sequence[float]]]


def build_adjacency(G: nx.DiGraph, attr: str = "tt") -> Adjacency:
    adj: Adjacency = {n: {} for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        adj[u][v] = data[attr]
    return adj


def bounded_dijkstra(adj: Adjacency, source, target, tau: int, budget: float) -> float:
    """Pruned single-pair Dijkstra on snapshot tau.

    Explores only the ball of radius `budget` around `source`, returning inf if
    no path of cost <= budget exists. Because the budget is a single edge cost
    scaled by t, each probe settles O(|B|) nodes with |B| << |V|.
    """
    if source == target:
        return 0.0
    if not adj.get(source):
        return INF
    dist = {source: 0.0}
    pq: List[Tuple[float, int, object]] = [(0.0, 0, source)]
    counter = 1
    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist.get(u, INF) + EPS:
            continue
        if u == target:
            return d
        for v, wvec in adj.get(u, {}).items():
            nd = d + wvec[tau]
            if nd > budget + EPS:
                continue
            if nd + EPS < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, counter, v))
                counter += 1
    return INF


def single_source_snapshot(G: nx.DiGraph, source, tau: int,
                           targets: Optional[Set] = None,
                           attr: str = "tt") -> Dict[object, float]:
    dist: Dict[object, float] = {}
    seen: Dict[object, float] = {source: 0.0}
    pq: List[Tuple[float, int, object]] = [(0.0, 0, source)]
    counter = 1
    remaining = set(targets) if targets is not None else None
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        if remaining is not None:
            remaining.discard(u)
            if not remaining:
                break
        for v, data in G[u].items():
            nd = d + data[attr][tau]
            if nd + EPS < seen.get(v, INF):
                seen[v] = nd
                heapq.heappush(pq, (nd, counter, v))
                counter += 1
    if targets is None:
        return dist
    return {x: dist[x] for x in targets if x in dist}


def multi_source_pred(G: nx.DiGraph, sources: Set, attr: str = "free_flow") -> Dict[object, object]:
    """Dijkstra from a virtual super-source; returns predecessor map."""
    dist: Dict[object, float] = {}
    seen: Dict[object, float] = {s: 0.0 for s in sources}
    pred: Dict[object, object] = {}
    pq: List[Tuple[float, int, object]] = [(0.0, i, s) for i, s in enumerate(sorted(sources, key=str))]
    heapq.heapify(pq)
    counter = len(pq)
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        for v, data in G[u].items():
            nd = d + float(data[attr])
            if nd + EPS < seen.get(v, INF):
                seen[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, counter, v))
                counter += 1
    return pred


# ==============================================================================
# SECTION 5 - NETWORK ACQUISITION, CONTRACTION, ENRICHMENT
# ==============================================================================

def _coords_from_data(data, xu: float, yu: float, xv: float, yv: float) -> List[Tuple[float, float]]:
    geom = data.get("geometry", None)
    if geom is not None and hasattr(geom, "coords"):
        try:
            c = [(float(x), float(y)) for x, y in geom.coords]
            if len(c) >= 2:
                return c
        except Exception:
            pass
    return [(xu, yu), (xv, yv)]


def _multidigraph_to_digraph(M) -> nx.DiGraph:
    D = nx.DiGraph()
    for n, data in M.nodes(data=True):
        try:
            D.add_node(n, x=float(data["x"]), y=float(data["y"]))
        except Exception:
            continue
    for u, v, data in M.edges(data=True):
        if u == v or u not in D or v not in D:
            continue
        try:
            length = float(data.get("length", 0.0) or 0.0)
        except (TypeError, ValueError):
            length = 0.0
        if length <= 0.0:
            length = haversine_m(D.nodes[u]["y"], D.nodes[u]["x"],
                                 D.nodes[v]["y"], D.nodes[v]["x"])
        if length <= 0.0:
            continue
        payload = {
            "length": length,
            "highway": normalize_highway(data.get("highway")),
            "coords": _coords_from_data(data, D.nodes[u]["x"], D.nodes[u]["y"],
                                        D.nodes[v]["x"], D.nodes[v]["y"]),
            "n_base": 1,
        }
        if D.has_edge(u, v):
            if length < D[u][v]["length"]:
                D[u][v].update(payload)
        else:
            D.add_edge(u, v, **payload)
    return D


def largest_strongly_connected(G: nx.DiGraph) -> nx.DiGraph:
    comps = list(nx.strongly_connected_components(G))
    if not comps:
        return G
    H = G.subgraph(max(comps, key=len)).copy()
    H.graph.update(G.graph)
    return H


def contract_degree_two(G: nx.DiGraph, max_passes: int = 40) -> nx.DiGraph:
    """Contract shape points into the junction graph (fixes R2).

    Removes interior degree-2 vertices of directed chains (one-way) and of
    bidirectional chains (two-way), summing lengths and concatenating polyline
    geometry so the drawn map is unchanged.
    """
    H = G.copy()
    for _ in range(max_passes):
        changed = False
        for n in list(H.nodes()):
            if n not in H:
                continue
            preds = set(H.predecessors(n)) - {n}
            succs = set(H.successors(n)) - {n}
            din, dout = H.in_degree(n), H.out_degree(n)

            if din == 2 and dout == 2 and len(preds) == 2 and preds == succs:
                u, v = sorted(preds, key=str)
                if H.has_edge(u, v) or H.has_edge(v, u):
                    continue
                a1, a2 = H[u][n], H[n][v]
                b1, b2 = H[v][n], H[n][u]
                H.add_edge(u, v,
                           length=a1["length"] + a2["length"],
                           highway=dominant_class(a1["highway"], a2["highway"]),
                           coords=list(a1["coords"]) + list(a2["coords"])[1:],
                           n_base=a1["n_base"] + a2["n_base"])
                H.add_edge(v, u,
                           length=b1["length"] + b2["length"],
                           highway=dominant_class(b1["highway"], b2["highway"]),
                           coords=list(b1["coords"]) + list(b2["coords"])[1:],
                           n_base=b1["n_base"] + b2["n_base"])
                H.remove_node(n)
                changed = True

            elif din == 1 and dout == 1 and len(preds) == 1 and len(succs) == 1:
                u = next(iter(preds))
                v = next(iter(succs))
                if u == v or H.has_edge(u, v):
                    continue
                a1, a2 = H[u][n], H[n][v]
                H.add_edge(u, v,
                           length=a1["length"] + a2["length"],
                           highway=dominant_class(a1["highway"], a2["highway"]),
                           coords=list(a1["coords"]) + list(a2["coords"])[1:],
                           n_base=a1["n_base"] + a2["n_base"])
                H.remove_node(n)
                changed = True
        if not changed:
            break
    H.graph.update(G.graph)
    return largest_strongly_connected(H)


def fetch_osm_network(center: Tuple[float, float], dist: int) -> Optional[nx.DiGraph]:
    """Overpass via OSMnx; returns None on any failure (timeout, DNS, rate limit)."""
    try:
        import osmnx as ox
    except Exception as exc:
        print(f"[network] osmnx unavailable ({type(exc).__name__}: {exc}).")
        return None
    try:
        for key, value in (("use_cache", True), ("log_console", False),
                           ("requests_timeout", 90), ("overpass_rate_limit", True)):
            if hasattr(ox.settings, key):
                setattr(ox.settings, key, value)
    except Exception:
        pass
    try:
        print(f"[network] Querying Overpass at {center}, r = {dist} m ...")
        M = ox.graph_from_point(center, dist=dist, network_type=NETWORK_TYPE, simplify=True)
        D = largest_strongly_connected(_multidigraph_to_digraph(M))
        if D.number_of_edges() < 60:
            print("[network] Overpass response too sparse; falling back.")
            return None
        D.graph.update({"source": "osmnx", "center": list(center), "dist": dist})
        print(f"[network] OSM raw graph: {D.number_of_nodes()} nodes / "
              f"{D.number_of_edges()} directed edges.")
        return D
    except Exception as exc:
        print(f"[network] Overpass failed ({type(exc).__name__}: {exc}). Falling back.")
        return None


def synthetic_arterial_grid(center: Tuple[float, float], dist: int, n: int = 23) -> nx.DiGraph:
    """Deterministic stratified arterial / collector / local grid with one-way pockets,
    two trunk diagonals (ring corridor) and randomly severed local links (blocks)."""
    lat0, lon0 = center
    step = (2.0 * dist) / (n - 1)
    mlat, mlon = local_metric_factors(lat0)

    G = nx.DiGraph()
    nid: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in range(n):
            k = i * n + j
            nid[(i, j)] = k
            G.add_node(k,
                       x=float(lon0 + (j - (n - 1) / 2.0) * step / mlon),
                       y=float(lat0 + (i - (n - 1) / 2.0) * step / mlat))

    def line_class(idx: int) -> str:
        if idx % 7 == 0:
            return "primary"
        if idx % 3 == 0:
            return "secondary"
        return "residential"

    def link(a: int, b: int, hw: str) -> None:
        xa, ya = G.nodes[a]["x"], G.nodes[a]["y"]
        xb, yb = G.nodes[b]["x"], G.nodes[b]["y"]
        length = haversine_m(ya, xa, yb, xb)
        if length > 0.0:
            G.add_edge(a, b, length=length, highway=hw,
                       coords=[(xa, ya), (xb, yb)], n_base=1)

    for i in range(n):
        for j in range(n):
            a = nid[(i, j)]
            if j + 1 < n:
                hw = line_class(i)
                if not (hw == "residential" and stable_unit(f"h{i}-{j}") < 0.07):
                    b = nid[(i, j + 1)]
                    link(a, b, hw)
                    if not (hw == "residential" and (i + j) % 7 == 0):
                        link(b, a, hw)
            if i + 1 < n:
                hw = line_class(j)
                if not (hw == "residential" and stable_unit(f"v{i}-{j}") < 0.07):
                    b = nid[(i + 1, j)]
                    link(a, b, hw)
                    if not (hw == "residential" and (3 * i + j) % 9 == 0):
                        link(b, a, hw)

    for k in range(n - 1):
        a, b = nid[(k, k)], nid[(k + 1, k + 1)]
        link(a, b, "trunk"); link(b, a, "trunk")
        c, d = nid[(k, n - 1 - k)], nid[(k + 1, n - 2 - k)]
        link(c, d, "trunk"); link(d, c, "trunk")

    G = largest_strongly_connected(G)
    G.graph.update({"source": "synthetic", "center": list(center), "dist": dist})
    print(f"[network] Synthetic grid: {G.number_of_nodes()} nodes / "
          f"{G.number_of_edges()} directed edges.")
    return G


def enrich_temporal_costs(G: nx.DiGraph) -> nx.DiGraph:
    """Attach the spatio-temporal speed field and all cost vectors."""
    lat0, lon0 = G.graph["center"]
    radius = float(G.graph.get("dist", DEFAULT_DIST))
    mlat, mlon = local_metric_factors(lat0)

    for u, v, data in G.edges(data=True):
        hw = normalize_highway(data.get("highway"))
        v_ff, alpha0, tier = HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)

        coords = data.get("coords") or [(G.nodes[u]["x"], G.nodes[u]["y"]),
                                        (G.nodes[v]["x"], G.nodes[v]["y"])]
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

        # local planar frame (metres) centred on the CBD
        ex = (x1 - x0) * mlon
        ey = (y1 - y0) * mlat
        cx = (mx - lon0) * mlon
        cy = (my - lat0) * mlat
        d_center = math.hypot(cx, cy)
        cbd = math.exp(-((d_center / max(1.0, 0.65 * radius)) ** 2))

        norm_e = math.hypot(ex, ey)
        norm_c = math.hypot(cx, cy)
        if norm_e < 1e-9 or norm_c < 1e-9:
            inbound = 0.5
        else:
            cosang = -(ex * cx + ey * cy) / (norm_e * norm_c)   # toward centre => +1
            inbound = 0.5 * (1.0 + max(-1.0, min(1.0, cosang)))

        jitter = stable_unit(f"{u}->{v}|{hw}")
        alpha = float(min(0.88, max(0.06, alpha0 * (0.55 + 0.95 * cbd) * (0.85 + 0.30 * jitter))))

        congestion = edge_congestion_curve(inbound, cbd, jitter)
        speeds = speed_curve(v_ff, alpha, congestion)
        length = float(data["length"])
        tt = [length / (s * 1000.0 / SEC_PER_HOUR) for s in speeds]
        ff = length / (v_ff * 1000.0 / SEC_PER_HOUR)

        data.update({
            "highway": hw,
            "tier": tier,
            "coords": coords,
            "v_ff": v_ff,
            "alpha": alpha,
            "cbd": cbd,
            "inbound": inbound,
            "congestion": congestion,
            "speeds": speeds,
            "tt": tt,                                   # 24-slice temporal cost
            "free_flow": ff,
            "tt_ff": [ff],                              # static free-flow view
            "len_vec": [length],                        # static geometric view
            "tt_mean_vec": [float(sum(tt) / len(tt))],  # time-averaged view
            "w_max": max(tt),
            "w_min": min(tt),
            "peak_ratio": max(tt) / min(tt),
        })

    G.graph["hours"] = list(HOURS)
    return G


def build_network(center: Tuple[float, float], dist: int,
                  force_synthetic: bool = False,
                  contract: bool = True) -> nx.DiGraph:
    G = None if force_synthetic else fetch_osm_network(center, dist)
    if G is None:
        G = synthetic_arterial_grid(center, dist)
    raw_nodes, raw_edges = G.number_of_nodes(), G.number_of_edges()
    if contract:
        G = contract_degree_two(G)
        print(f"[network] Junction contraction: {raw_nodes} -> {G.number_of_nodes()} nodes, "
              f"{raw_edges} -> {G.number_of_edges()} edges.")
    G.graph["raw_nodes"] = raw_nodes
    G.graph["raw_edges"] = raw_edges
    return enrich_temporal_costs(G)


def network_summary(G: nx.DiGraph) -> dict:
    tiers: Dict[str, int] = {}
    total_len = 0.0
    prs, alphas = [], []
    for _, _, d in G.edges(data=True):
        tiers[d["tier"]] = tiers.get(d["tier"], 0) + 1
        total_len += d["length"]
        prs.append(d["peak_ratio"])
        alphas.append(d["alpha"])
    return {
        "source": G.graph.get("source", "unknown"),
        "center": G.graph.get("center"),
        "dist_m": G.graph.get("dist"),
        "raw_nodes": G.graph.get("raw_nodes"),
        "raw_edges": G.graph.get("raw_edges"),
        "junction_nodes": G.number_of_nodes(),
        "junction_edges": G.number_of_edges(),
        "total_length_km": round(total_len / 1000.0, 3),
        "tier_counts": tiers,
        "mean_edge_length_m": round(total_len / max(1, G.number_of_edges()), 2),
        "peak_ratio_mean": round(float(np.mean(prs)), 4),
        "peak_ratio_p95": round(float(np.percentile(prs, 95)), 4),
        "peak_ratio_max": round(float(np.max(prs)), 4),
        "alpha_mean": round(float(np.mean(alphas)), 4),
        "strongly_connected": bool(nx.is_strongly_connected(G)),
    }


def temporal_heterogeneity_index(G: nx.DiGraph) -> float:
    """Departure from homothety: if w(.,tau) = c(tau) w(.,0) for all e, this is 0.

    Defined as the mean over tau of the coefficient of variation of the
    normalised edge cost ratios r_e(tau) = w(e,tau)/w(e,0). A strictly positive
    value certifies that no static (single-snapshot) spanner can be optimal for
    every hour.
    """
    ratios = np.array([[d["tt"][tau] / d["tt"][0] for tau in HOURS]
                       for _, _, d in G.edges(data=True)], dtype=float)
    cv = ratios.std(axis=0) / np.maximum(1e-12, ratios.mean(axis=0))
    return float(cv.mean())


def subgraph_by_radius(G: nx.DiGraph, frac: float) -> nx.DiGraph:
    lat0, lon0 = G.graph["center"]
    radius = float(G.graph.get("dist", DEFAULT_DIST)) * frac
    keep = [n for n in G.nodes()
            if haversine_m(lat0, lon0, G.nodes[n]["y"], G.nodes[n]["x"]) <= radius]
    if len(keep) < 20:
        keep = list(G.nodes())
    S = largest_strongly_connected(G.subgraph(keep).copy())
    S.graph.update(G.graph)
    return S


# ==============================================================================
# SECTION 6 - DELAY-BOUNDED GREEDY SPANNER
# ==============================================================================

def _edge_order(G: nx.DiGraph, mode: str, attr: str,
                hours: Sequence[int], seed: int = SEED) -> List[Tuple[object, object, dict]]:
    edges = list(G.edges(data=True))
    if mode == "temporal_mean":
        keyed = [((sum(d[attr][h] for h in hours) / len(hours), str(u), str(v)), (u, v, d))
                 for u, v, d in edges]
    elif mode == "centrality":
        k = min(96, G.number_of_nodes())
        try:
            cent = nx.edge_betweenness_centrality(G, k=k, weight="free_flow", seed=seed)
        except Exception:
            cent = {}
        keyed = []
        for u, v, d in edges:
            c = cent.get((u, v), 0.0)
            score = max(d[attr][h] for h in hours) / (1.0 + 6.0 * c)
            keyed.append(((score, str(u), str(v)), (u, v, d)))
    else:  # temporal_max (classical greedy analogue)
        keyed = [((max(d[attr][h] for h in hours), str(u), str(v)), (u, v, d))
                 for u, v, d in edges]
    keyed.sort(key=lambda kv: kv[0])
    return [payload for _, payload in keyed]


def delay_bounded_greedy_spanner(G: nx.DiGraph,
                                 t: float,
                                 hours: Optional[Iterable[int]] = None,
                                 attr: str = "tt",
                                 order: str = "temporal_max",
                                 show_progress: bool = True,
                                 label: str = "delay_bounded") -> nx.DiGraph:
    """Delay-Bounded Greedy Spanner.

        1  order E ascending by max_tau w(e,tau)
        2  H <- empty
        3  for (u,v) in E in that order:
        4      for tau ordered by ascending w((u,v),tau):      # tightest budget first
        5          beta <- t * w((u,v),tau)
        6          if BoundedDijkstra(H, u->v, tau, beta) > beta:
        7              H <- H + {(u,v)}; break                 # early exit
        8  return H

    Complexity O(|E| |T| |B| log|B|); the tightest-budget-first slice order makes
    the |T| loop terminate after ~1 probe for every edge that is retained, so the
    full 24 probes are paid only for genuinely redundant edges.
    """
    hours = list(HOURS) if hours is None else list(hours)
    t = float(t)
    if t < 1.0:
        raise ValueError("Dilation parameter t must be >= 1.")

    t0 = time.perf_counter()
    ordered = _edge_order(G, order, attr, hours)

    adj: Adjacency = {n: {} for n in G.nodes()}
    kept: List[Tuple[object, object, dict]] = []
    probes = 0

    it = ordered
    if show_progress:
        it = tqdm(ordered, desc=f"[{label}] t={t:.2f}", unit="edge", leave=False)

    for u, v, data in it:
        costs = data[attr]
        slice_order = sorted(hours, key=lambda tau: costs[tau])
        add = False
        for tau in slice_order:
            budget = t * costs[tau]
            probes += 1
            if bounded_dijkstra(adj, u, v, tau, budget) > budget + EPS:
                add = True
                break
        if add:
            adj[u][v] = costs
            kept.append((u, v, data))

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in kept:
        H.add_edge(u, v, **data)
    H.graph.update(G.graph)
    H.graph.update({
        "method": label, "t": t, "hours": hours, "cost_attr": attr, "order": order,
        "build_seconds": time.perf_counter() - t0,
        "dijkstra_probes": probes, "edges_parent": G.number_of_edges(),
    })
    return H


def temporal_stretch_certificate(G: nx.DiGraph, H: nx.DiGraph, t: float,
                                 hours: Optional[Iterable[int]] = None,
                                 attr: str = "tt",
                                 refine_cap: int = CERT_REFINE_CAP,
                                 refine_mult: float = CERT_REFINE_MULT) -> dict:
    """Exhaustive Theorem-1 certificate + magnitude of the worst violation (fixes R7).

    Phase 1 decides the invariant for every (e, tau) with budget t*w(e,tau).
    Phase 2 re-probes violating edges with budget refine_mult*w(e,tau) to recover
    the attained stretch, so a failed certificate is reported with a number and
    not merely as a boolean.
    """
    hours = list(HOURS) if hours is None else list(hours)
    adj = build_adjacency(H, attr)

    violating: List[Tuple[object, object, int, float]] = []
    for u, v, d in G.edges(data=True):
        costs = d[attr]
        for tau in hours:
            budget = t * costs[tau]
            if bounded_dijkstra(adj, u, v, tau, budget) > budget + EPS:
                violating.append((u, v, tau, costs[tau]))
                break

    max_stretch = float(t)
    censored = 0
    refined = violating[:refine_cap]
    for u, v, tau, w in refined:
        big = refine_mult * w
        d = bounded_dijkstra(adj, u, v, tau, big)
        if d == INF:
            censored += 1
        else:
            max_stretch = max(max_stretch, d / max(EPS, w))
    if censored > 0:
        max_stretch = max(max_stretch, refine_mult)

    return {
        "certified": len(violating) == 0,
        "violating_edges": int(len(violating)),
        "violation_rate": float(len(violating) / max(1, G.number_of_edges())),
        "checked_edges": int(G.number_of_edges()),
        "attained_worst_edge_stretch": float(max_stretch),
        "censored_refinements": int(censored),
        "refine_budget_mult": float(refine_mult),
        "first_violation_hour": int(violating[0][2]) if violating else None,
    }


# ==============================================================================
# SECTION 7 - BASELINES AND MATCHED-DENSITY CONTROLS
# ==============================================================================

def full_network(G: nx.DiGraph) -> nx.DiGraph:
    H = G.copy()
    H.graph.update({"method": "full", "t": None, "build_seconds": 0.0,
                    "dijkstra_probes": 0, "edges_parent": G.number_of_edges()})
    return H


def static_freeflow_spanner(G: nx.DiGraph, t: float, show_progress: bool = True) -> nx.DiGraph:
    """Greedy t-spanner certified only on the free-flow snapshot (traffic-blind)."""
    H = delay_bounded_greedy_spanner(G, t, hours=[0], attr="tt_ff",
                                     order="temporal_max",
                                     show_progress=show_progress, label="static_ff")
    H.graph.update({"method": "static_ff", "t": t})
    return H


def static_geometric_spanner(G: nx.DiGraph, t: float, show_progress: bool = True) -> nx.DiGraph:
    """Classical Euclidean-length greedy t-spanner (geometry-only)."""
    H = delay_bounded_greedy_spanner(G, t, hours=[0], attr="len_vec",
                                     order="temporal_max",
                                     show_progress=show_progress, label="static_len")
    H.graph.update({"method": "static_len", "t": t})
    return H


def mean_temporal_spanner(G: nx.DiGraph, t: float, show_progress: bool = True) -> nx.DiGraph:
    """Spanner on the 24-hour *averaged* cost: the natural aggregation strawman."""
    H = delay_bounded_greedy_spanner(G, t, hours=[0], attr="tt_mean_vec",
                                     order="temporal_max",
                                     show_progress=show_progress, label="mean_temporal")
    H.graph.update({"method": "mean_temporal", "t": t})
    return H


def hierarchical_backbone(G: nx.DiGraph, keep: Optional[Set[str]] = None) -> nx.DiGraph:
    """Functional-class heuristic with correct last-mile attachment (fixes R5).

    The arterial subgraph is reduced to its largest SCC; every remaining junction
    is then attached to that core by its cheapest free-flow access path in both
    directions (multi-source Dijkstra forward and on the reverse graph). The
    result is strongly connected by construction, but carries no dilation
    certificate whatsoever - which is exactly the gap the proposed method closes.
    """
    keep = ARTERIAL_CLASSES if keep is None else keep
    t0 = time.perf_counter()

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d["highway"] in keep:
            H.add_edge(u, v, **d)

    comps = [c for c in nx.strongly_connected_components(H) if len(c) > 1]
    core: Set = max(comps, key=len) if comps else {min(G.nodes(), key=str)}

    pred_out = multi_source_pred(G, core, "free_flow")                   # core -> node
    Grev = G.reverse(copy=False)
    pred_in = multi_source_pred(Grev, core, "free_flow")                 # node -> core

    for n in G.nodes():
        if n in core:
            continue
        cur, guard = n, 0
        while cur not in core and cur in pred_out and guard < 10 ** 5:
            p = pred_out[cur]
            H.add_edge(p, cur, **G[p][cur])
            cur, guard = p, guard + 1
        cur, guard = n, 0
        while cur not in core and cur in pred_in and guard < 10 ** 5:
            p = pred_in[cur]                                             # edge cur -> p in G
            if G.has_edge(cur, p):
                H.add_edge(cur, p, **G[cur][p])
            cur, guard = p, guard + 1

    if not nx.is_strongly_connected(H):
        H = largest_strongly_connected(H)

    H.graph.update(G.graph)
    H.graph.update({"method": "hierarchical", "t": None,
                    "build_seconds": time.perf_counter() - t0,
                    "dijkstra_probes": 0, "edges_parent": G.number_of_edges()})
    return H


def strong_connectivity_core(G: nx.DiGraph, weight: str = "free_flow") -> Set[Tuple]:
    """Union of an out-arborescence and an in-arborescence rooted at a median node.

    Guarantees strong connectivity with ~2(|V|-1) edges; used as the mandatory
    skeleton of every matched-density control so that density comparisons are not
    confounded by connectivity failure.
    """
    root = min(G.nodes(), key=str)
    pred_out = multi_source_pred(G, {root}, weight)
    pred_in = multi_source_pred(G.reverse(copy=False), {root}, weight)
    core: Set[Tuple] = set()
    for n, p in pred_out.items():
        if G.has_edge(p, n):
            core.add((p, n))
    for n, p in pred_in.items():
        if G.has_edge(n, p):
            core.add((n, p))
    return core


def _materialize(G: nx.DiGraph, edges: Iterable[Tuple], method: str,
                 seconds: float) -> nx.DiGraph:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in edges:
        if G.has_edge(u, v):
            H.add_edge(u, v, **G[u][v])
    H.graph.update(G.graph)
    H.graph.update({"method": method, "t": None, "build_seconds": seconds,
                    "dijkstra_probes": 0, "edges_parent": G.number_of_edges()})
    return H


def matched_density_random(G: nx.DiGraph, budget_edges: int, seed: int = SEED) -> nx.DiGraph:
    """Random sparsifier at exactly the proposed method's edge budget (fixes R6)."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    core = strong_connectivity_core(G)
    rest = [(u, v) for u, v in G.edges() if (u, v) not in core]
    rng.shuffle(rest)
    need = max(0, budget_edges - len(core))
    chosen = set(core) | set(rest[:need])
    return _materialize(G, chosen, "matched_random", time.perf_counter() - t0)


def matched_density_betweenness(G: nx.DiGraph, budget_edges: int, seed: int = SEED) -> nx.DiGraph:
    """Top-k edge-betweenness sparsifier at the same edge budget."""
    t0 = time.perf_counter()
    core = strong_connectivity_core(G)
    k = min(128, G.number_of_nodes())
    try:
        cent = nx.edge_betweenness_centrality(G, k=k, weight="free_flow", seed=seed)
    except Exception:
        cent = {}
    rest = sorted(((u, v) for u, v in G.edges() if (u, v) not in core),
                  key=lambda e: (-cent.get(e, 0.0), str(e[0]), str(e[1])))
    need = max(0, budget_edges - len(core))
    chosen = set(core) | set(rest[:need])
    return _materialize(G, chosen, "matched_btw", time.perf_counter() - t0)


# ==============================================================================
# SECTION 8 - EVALUATION (uncensored, multi-seed, bootstrapped)  (fixes R4, R8)
# ==============================================================================

def sample_pairs(G: nx.DiGraph, n_sources: int, n_targets: int, seed: int) -> Dict:
    rng = random.Random(seed)
    nodes = sorted(G.nodes(), key=str)
    sources = rng.sample(nodes, min(n_sources, len(nodes)))
    pairs: Dict[object, List[object]] = {}
    for s in sources:
        pool = [n for n in nodes if n != s]
        pairs[s] = rng.sample(pool, min(n_targets, len(pool)))
    return pairs


def baseline_distances(G: nx.DiGraph, pairs: Dict, hours: Sequence[int]) -> Dict:
    out: Dict[Tuple[int, object], Dict[object, float]] = {}
    for tau in hours:
        for s, targets in pairs.items():
            out[(tau, s)] = single_source_snapshot(G, s, tau, set(targets))
    return out


def bootstrap_ci(values: Sequence[float], stat=np.mean, n: int = BOOTSTRAP_N,
                 seed: int = SEED, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    boots = stat(arr[idx], axis=1)
    return (float(np.percentile(boots, 100 * alpha / 2.0)),
            float(np.percentile(boots, 100 * (1 - alpha / 2.0))))


def evaluate_graph(H: nx.DiGraph, pair_sets: Sequence[Dict], hours: Sequence[int],
                   bases: Sequence[Dict], collect_pairs_at: Optional[int] = None) -> Dict:
    """Empirical dilation over OD samples and all snapshots.

    Unreachable pairs are reported separately and excluded from dilation
    statistics; no sentinel value is ever mixed into an aggregate.
    """
    per_hour_max = np.zeros((len(pair_sets), len(hours)), dtype=float)
    per_hour_mean = np.zeros((len(pair_sets), len(hours)), dtype=float)
    all_ratios: List[float] = []
    per_seed_worst: List[float] = []
    per_seed_mean: List[float] = []
    collected: List[Tuple[str, str, float]] = []

    total = reachable = breaches = 0
    t_bound = H.graph.get("t")

    for r, (pairs, base) in enumerate(zip(pair_sets, bases)):
        seed_ratios: List[float] = []
        for hi, tau in enumerate(hours):
            ratios: List[float] = []
            for s, targets in pairs.items():
                dH = single_source_snapshot(H, s, tau, set(targets))
                dG = base[(tau, s)]
                for tgt in targets:
                    dg = dG.get(tgt)
                    if dg is None or dg <= 0.0:
                        continue
                    total += 1
                    dh = dH.get(tgt)
                    if dh is None:
                        continue
                    reachable += 1
                    ratio = dh / dg
                    ratios.append(ratio)
                    if t_bound is not None and ratio > t_bound + 1e-6:
                        breaches += 1
                    if collect_pairs_at is not None and tau == collect_pairs_at and r == 0:
                        collected.append((str(s), str(tgt), float(ratio)))
            arr = np.asarray(ratios, dtype=float) if ratios else np.asarray([np.nan])
            per_hour_max[r, hi] = np.nanmax(arr)
            per_hour_mean[r, hi] = np.nanmean(arr)
            seed_ratios.extend([x for x in arr.tolist() if not math.isnan(x)])
        if seed_ratios:
            per_seed_worst.append(float(np.max(seed_ratios)))
            per_seed_mean.append(float(np.mean(seed_ratios)))
            all_ratios.extend(seed_ratios)

    allr = np.asarray(all_ratios, dtype=float) if all_ratios else np.asarray([np.nan])
    mean_lo, mean_hi = bootstrap_ci(all_ratios, np.mean) if all_ratios else (np.nan, np.nan)
    hour_max_mean = np.nanmean(per_hour_max, axis=0).tolist()
    hour_max_std = np.nanstd(per_hour_max, axis=0).tolist()

    return {
        "worst_dilation": float(np.nanmax(allr)),
        "worst_dilation_seed_mean": float(np.mean(per_seed_worst)) if per_seed_worst else float("nan"),
        "worst_dilation_seed_std": float(np.std(per_seed_worst)) if per_seed_worst else float("nan"),
        "mean_dilation": float(np.nanmean(allr)),
        "mean_dilation_ci": [mean_lo, mean_hi],
        "p95_dilation": float(np.nanpercentile(allr, 95)),
        "p99_dilation": float(np.nanpercentile(allr, 99)),
        "per_hour_max": hour_max_mean,
        "per_hour_max_std": hour_max_std,
        "per_hour_mean": np.nanmean(per_hour_mean, axis=0).tolist(),
        "pairs_evaluated": int(total),
        "reachable_pairs": int(reachable),
        "reachability": float(reachable / max(1, total)),
        "bound_breaches": int(breaches),
        "breach_rate": float(breaches / max(1, reachable)) if reachable else float("nan"),
        "worst_hour": int(np.nanargmax(hour_max_mean)),
        "collected_pairs": collected,
    }


def structural_metrics(G: nx.DiGraph, H: nx.DiGraph) -> Dict:
    e_g, e_h = G.number_of_edges(), H.number_of_edges()
    len_g = sum(d["length"] for _, _, d in G.edges(data=True))
    len_h = sum(d["length"] for _, _, d in H.edges(data=True))
    arterial = sum(1 for _, _, d in H.edges(data=True) if d["tier"] == "arterial")
    return {
        "edges": int(e_h),
        "edges_parent": int(e_g),
        "edge_retention_pct": round(100.0 * e_h / max(1, e_g), 3),
        "edges_removed_pct": round(100.0 * (1.0 - e_h / max(1, e_g)), 3),
        "length_km": round(len_h / 1000.0, 3),
        "length_retention_pct": round(100.0 * len_h / max(1e-9, len_g), 3),
        "arterial_share_pct": round(100.0 * arterial / max(1, e_h), 2),
        "avg_out_degree": round(e_h / max(1, H.number_of_nodes()), 3),
        "strongly_connected": bool(nx.is_strongly_connected(H)) if e_h else False,
    }


# ==============================================================================
# SECTION 9 - FIFO TIME-DEPENDENT VALIDATION  (Proposition 2)
# ==============================================================================

def td_cost(costs: Sequence[float], abs_seconds: float) -> float:
    """Cyclic piecewise-linear interpolation of the hourly cost profile."""
    h = (abs_seconds / SEC_PER_HOUR) % 24.0
    i = int(math.floor(h))
    f = h - i
    return costs[i] * (1.0 - f) + costs[(i + 1) % 24] * f


def fifo_violation_rate(G: nx.DiGraph, samples: int = 24 * 4) -> float:
    """Fraction of (edge, time) samples where s + w~(e,s) is non-monotone."""
    bad = tot = 0
    step = 24.0 * SEC_PER_HOUR / samples
    for _, _, d in G.edges(data=True):
        costs = d["tt"]
        prev = 0.0 + td_cost(costs, 0.0)
        for k in range(1, samples + 1):
            s = k * step
            cur = s + td_cost(costs, s)
            tot += 1
            if cur + 1e-6 < prev:
                bad += 1
            prev = cur
    return float(bad / max(1, tot))


def td_dijkstra(G: nx.DiGraph, source, departure: float,
                targets: Set) -> Dict[object, float]:
    """FIFO time-dependent Dijkstra; returns travel times (not arrival times)."""
    arrival = {source: departure}
    settled: Dict[object, float] = {}
    pq: List[Tuple[float, int, object]] = [(departure, 0, source)]
    counter = 1
    remaining = set(targets)
    while pq:
        a, _, u = heapq.heappop(pq)
        if u in settled:
            continue
        settled[u] = a
        remaining.discard(u)
        if not remaining:
            break
        for v, data in G[u].items():
            na = a + td_cost(data["tt"], a)
            if na + EPS < arrival.get(v, INF):
                arrival[v] = na
                heapq.heappush(pq, (na, counter, v))
                counter += 1
    return {x: settled[x] - departure for x in targets if x in settled}


def time_dependent_validation(G: nx.DiGraph, H: nx.DiGraph, pairs: Dict,
                              departures_h: Sequence[float]) -> Dict:
    ratios: List[float] = []
    unreachable = 0
    for dep_h in departures_h:
        dep = dep_h * SEC_PER_HOUR
        for s, targets in pairs.items():
            tg = set(targets)
            dg = td_dijkstra(G, s, dep, tg)
            dh = td_dijkstra(H, s, dep, tg)
            for x in targets:
                if x not in dg or dg[x] <= 0:
                    continue
                if x not in dh:
                    unreachable += 1
                    continue
                ratios.append(dh[x] / dg[x])
    arr = np.asarray(ratios, dtype=float) if ratios else np.asarray([np.nan])
    return {
        "departures_hours": list(departures_h),
        "pairs": int(arr.size),
        "unreachable": int(unreachable),
        "worst_td_dilation": float(np.nanmax(arr)),
        "mean_td_dilation": float(np.nanmean(arr)),
        "p95_td_dilation": float(np.nanpercentile(arr, 95)),
    }


# ==============================================================================
# SECTION 10 - FIGURE
# ==============================================================================

def _edge_segments(G: nx.DiGraph, edges: Optional[Iterable[Tuple]] = None):
    segs = []
    it = G.edges(data=True) if edges is None else ((u, v, G[u][v]) for u, v in edges)
    for u, v, d in it:
        c = d.get("coords")
        if c and len(c) >= 2:
            segs.append([(float(x), float(y)) for x, y in c])
        else:
            segs.append([(G.nodes[u]["x"], G.nodes[u]["y"]),
                         (G.nodes[v]["x"], G.nodes[v]["y"])])
    return segs


def make_figure(G: nx.DiGraph, results: Dict, backbone: nx.DiGraph,
                hours: Sequence[int], t_focus: float, t_values: Sequence[float],
                scaling: List[Dict], pair_frames: Dict[str, pd.DataFrame],
                path: str) -> None:
    sns.set_theme(style="whitegrid", context="talk",
                  rc={"axes.edgecolor": "#2b2b2b", "grid.alpha": 0.30,
                      "axes.titlesize": 17, "axes.labelsize": 15,
                      "legend.fontsize": 10.5})

    fig = plt.figure(figsize=(23, 20.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.45],
                          hspace=0.36, wspace=0.26)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    axG = fig.add_subplot(gs[2, :])

    # ---- (a) spatio-temporal speed field -------------------------------------
    rows = []
    for _, _, d in G.edges(data=True):
        orient = "inbound" if d["inbound"] >= 0.5 else "outbound"
        for tau in hours:
            rows.append({"hour": tau,
                         "speed_ratio": d["speeds"][tau] / d["v_ff"],
                         "tier": d["tier"], "orientation": orient})
    dfA = pd.DataFrame(rows)
    sns.lineplot(data=dfA, x="hour", y="speed_ratio", hue="tier", style="orientation",
                 errorbar=("ci", 95), ax=axA, linewidth=2.2,
                 palette={"arterial": "#7b0d1e", "collector": "#c0392b", "local": "#e08e79"})
    axA.set_xlabel(r"Hour of day $\tau$")
    axA.set_ylabel(r"$v(e,\tau)\,/\,v_{\mathrm{ff}}(e)$")
    axA.set_title("(a) Tidal spatio-temporal speed field", loc="left")
    axA.set_xticks(range(0, 24, 4))
    axA.legend(fontsize=9.5, ncol=2, loc="lower left", framealpha=0.9)

    # ---- (b) worst-case dilation vs hour --------------------------------------
    axB.axvspan(6.5, 9.5, color="#f1c40f", alpha=0.13, zorder=0)
    axB.axvspan(15.5, 19.5, color="#e67e22", alpha=0.13, zorder=0)
    ymax = t_focus
    for method in ("delay_bounded", "static_ff", "static_len", "mean_temporal", "hierarchical"):
        key = (method, t_focus) if method not in ("hierarchical",) else (method, None)
        entry = results.get(key)
        if entry is None:
            continue
        label, color, ls, marker = METHOD_STYLE[method]
        m = entry["metrics"]
        y = np.asarray(m["per_hour_max"], dtype=float)
        e = np.asarray(m["per_hour_max_std"], dtype=float)
        axB.plot(list(hours), y, label=f"{label}  (reach {100*m['reachability']:.0f}%)",
                 color=color, linestyle=ls, marker=marker, markersize=5, linewidth=2.3, zorder=3)
        axB.fill_between(list(hours), y - e, y + e, color=color, alpha=0.15, zorder=2)
        ymax = max(ymax, float(np.nanmax(y)))
    axB.axhline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=2.0,
                label=f"Certified bound $t={t_focus}$", zorder=4)
    axB.set_ylim(0.97, min(6.0, 1.12 * ymax))
    axB.set_xlabel(r"Hour of day $\tau$")
    axB.set_ylabel(r"Worst-case dilation $\max\ d_H/d_G$")
    axB.set_title("(b) Temporal dilation stability", loc="left")
    axB.set_xticks(range(0, 24, 4))
    axB.legend(fontsize=9, loc="upper left", framealpha=0.92)

    # ---- (c) sparsification vs t ----------------------------------------------
    for method in ("delay_bounded", "static_ff", "mean_temporal"):
        xs, ys, zs = [], [], []
        for t in sorted(t_values):
            entry = results.get((method, t))
            if entry:
                xs.append(t)
                ys.append(entry["structure"]["edges_removed_pct"])
                zs.append(100.0 - entry["structure"]["length_retention_pct"])
        if xs:
            label, color, ls, marker = METHOD_STYLE[method]
            axC.plot(xs, ys, label=label, color=color, linestyle=ls,
                     marker=marker, markersize=8, linewidth=2.5)
            axC.plot(xs, zs, color=color, linestyle=":", linewidth=1.4, alpha=0.75)
    hier = results.get(("hierarchical", None))
    if hier:
        label, color, ls, _ = METHOD_STYLE["hierarchical"]
        axC.axhline(hier["structure"]["edges_removed_pct"], color=color,
                    linestyle=ls, linewidth=2.2, label=label + r" ($t$-free)")
    axC.set_xlabel(r"Dilation parameter $t$")
    axC.set_ylabel("Removed (%)  [solid: edges, dotted: length]")
    axC.set_title("(c) Sparsification vs. permitted dilation", loc="left")
    axC.legend(fontsize=9, loc="upper left", framealpha=0.92)

    # ---- (d) Pareto front at matched density ----------------------------------
    for method in ("delay_bounded", "static_ff", "static_len", "mean_temporal",
                   "matched_random", "matched_btw", "hierarchical"):
        pts = [(k[1], v) for k, v in results.items() if k[0] == method]
        if not pts:
            continue
        label, color, ls, marker = METHOD_STYLE[method]
        xs = [v["structure"]["edge_retention_pct"] for _, v in pts]
        ys = [v["certificate"].get("attained_worst_edge_stretch", np.nan)
              if v.get("certificate") else np.nan for _, v in pts]
        ys = [min(y, 12.0) if np.isfinite(y) else 12.0 for y in ys]
        order = np.argsort(xs)
        xs = list(np.asarray(xs)[order]); ys = list(np.asarray(ys)[order])
        axD.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                 markersize=9, linewidth=2.0, label=label, alpha=0.9)
    axD.axhline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=1.8)
    axD.set_yscale("log")
    axD.set_xlabel("Edges retained (%)")
    axD.set_ylabel("Attained worst-case temporal stretch")
    axD.set_title("(d) Sparsity / fidelity Pareto front", loc="left")
    axD.legend(fontsize=8.5, loc="upper right", framealpha=0.92)

    # ---- (e) runtime and probe scaling ----------------------------------------
    if scaling:
        se = [s["edges"] for s in scaling]
        st = [s["seconds"] for s in scaling]
        sp = [s["probes_per_edge"] for s in scaling]
        axE.plot(se, st, color="#c0392b", marker="o", linewidth=2.4, label="build time (s)")
        axE.set_xscale("log"); axE.set_yscale("log")
        axE.set_xlabel(r"$|E|$ (junction edges)")
        axE.set_ylabel("Construction time (s)")
        if len(se) >= 2:
            k = np.polyfit(np.log(se), np.log(np.maximum(1e-6, st)), 1)[0]
            axE.plot(se, np.exp(np.polyval(np.polyfit(np.log(se), np.log(np.maximum(1e-6, st)), 1),
                                           np.log(se))),
                     color="#7f8c8d", linestyle="--", linewidth=1.6,
                     label=fr"fit: $T \propto |E|^{{{k:.2f}}}$")
        ax2 = axE.twinx()
        ax2.plot(se, sp, color="#2980b9", marker="s", linewidth=2.0, linestyle="-.")
        ax2.set_ylabel("Dijkstra probes per edge", color="#2980b9")
        ax2.grid(False)
        axE.set_title(f"(e) Empirical scaling ($t={t_focus}$)", loc="left")
        axE.legend(fontsize=9, loc="upper left", framealpha=0.92)

    # ---- (f) dilation ECDF at the peak hour ------------------------------------
    if pair_frames:
        dfF = pd.concat(list(pair_frames.values()), ignore_index=True)
        palette = {METHOD_STYLE[m][0]: METHOD_STYLE[m][1] for m in pair_frames}
        sns.ecdfplot(data=dfF, x="dilation", hue="method", ax=axF,
                     linewidth=2.3, palette=palette)
        axF.axvline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=1.8)
        axF.set_xlim(0.98, min(5.0, float(dfF["dilation"].quantile(0.999)) * 1.05 + 0.02))
        axF.set_xlabel(r"OD dilation $d_H/d_G$ at peak hour")
        axF.set_ylabel("Empirical CDF")
        axF.set_title("(f) Dilation distribution at the worst hour", loc="left")
        leg = axF.get_legend()
        if leg is not None:
            leg.set_title(None)
            for txt in leg.get_texts():
                txt.set_fontsize(9)

    # ---- (g) spatial backbone ---------------------------------------------------
    kept_edges = set(backbone.edges())
    removed = [(u, v) for u, v in G.edges() if (u, v) not in kept_edges]
    axG.add_collection(LineCollection(_edge_segments(G, removed), colors="#c3cbd4",
                                      linewidths=0.8, alpha=0.9, zorder=1))
    by_tier = {"arterial": [], "collector": [], "local": []}
    for u, v, d in backbone.edges(data=True):
        by_tier.setdefault(d["tier"], []).append(
            _edge_segments(backbone, [(u, v)])[0])
    axG.add_collection(LineCollection(by_tier.get("local", []), colors="#e07b5f",
                                      linewidths=1.2, alpha=0.9, zorder=2))
    axG.add_collection(LineCollection(by_tier.get("collector", []), colors="#c0392b",
                                      linewidths=1.9, alpha=0.95, zorder=3))
    axG.add_collection(LineCollection(by_tier.get("arterial", []), colors="#6d0d1c",
                                      linewidths=3.0, alpha=1.0, zorder=4))
    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    padx = 0.02 * (max(xs) - min(xs) + 1e-6)
    pady = 0.02 * (max(ys) - min(ys) + 1e-6)
    axG.set_xlim(min(xs) - padx, max(xs) + padx)
    axG.set_ylim(min(ys) - pady, max(ys) + pady)
    lat0 = float(np.mean(ys))
    axG.set_aspect(1.0 / max(0.15, math.cos(math.radians(lat0))), adjustable="box")
    axG.grid(False)
    axG.set_facecolor("#fcfcfd")
    axG.set_xlabel("Longitude")
    axG.set_ylabel("Latitude")
    keep_pct = 100.0 * backbone.number_of_edges() / max(1, G.number_of_edges())
    axG.set_title(f"(g) Certified emergency backbone $H \\subseteq G$  "
                  f"($t={t_focus}$, {keep_pct:.1f}% of junction edges, "
                  f"{100.0-keep_pct:.1f}% pruned, source: {G.graph.get('source','?')})",
                  loc="left")
    axG.legend(handles=[
        Line2D([0], [0], color="#c3cbd4", lw=2, label="Pruned from $G$"),
        Line2D([0], [0], color="#e07b5f", lw=2, label="$H$ — local"),
        Line2D([0], [0], color="#c0392b", lw=3, label="$H$ — collector"),
        Line2D([0], [0], color="#6d0d1c", lw=4, label="$H$ — arterial"),
    ], fontsize=11, loc="upper right", framealpha=0.95)

    fig.suptitle("Delay-Bounded Time-Varying Geometric $t$-Spanners for Urban Road Networks",
                 fontsize=24, y=0.975)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] wrote {path}")


# ==============================================================================
# SECTION 11 - REPORTING
# ==============================================================================

def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def build_table(results: Dict, t_values: Sequence[float]) -> str:
    head = ("| Method | t | Edges kept (%) | Length kept (%) | Attained worst "
            "edge stretch | Certified (24 h) | Worst OD dilation | Mean OD dilation "
            "[95% CI] | Reach (%) | Build (s) | Probes/edge |")
    sep = "|" + "---|" * 11
    rows = [head, sep]

    order: List[Tuple[str, Optional[float]]] = []
    for t in sorted(t_values):
        for m in ("delay_bounded", "static_ff", "mean_temporal"):
            order.append((m, t))
    for m in ("static_len", "matched_random", "matched_btw"):
        order.extend([k for k in results if k[0] == m])
    order.append(("hierarchical", None))
    order.append(("full", None))

    seen = set()
    for key in order:
        if key in seen or key not in results:
            continue
        seen.add(key)
        e = results[key]
        m, s, c = e["metrics"], e["structure"], e.get("certificate", {}) or {}
        name = METHOD_STYLE[key[0]][0]
        tstr = f"{key[1]:.2f}" if key[1] is not None else "—"
        stretch = c.get("attained_worst_edge_stretch")
        cert = "—" if not c else ("**yes**" if c.get("certified") else
                                  f"no ({c.get('violating_edges')} edges)")
        ci = m.get("mean_dilation_ci", [np.nan, np.nan])
        ppe = e["dijkstra_probes"] / max(1, s["edges_parent"])
        rows.append(
            f"| {name} | {tstr} | {s['edge_retention_pct']:.1f} | "
            f"{s['length_retention_pct']:.1f} | {_fmt(stretch)} | {cert} | "
            f"{_fmt(m['worst_dilation'])} | {_fmt(m['mean_dilation'])} "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {100*m['reachability']:.1f} | "
            f"{e['build_seconds']:.2f} | {ppe:.2f} |"
        )
    return "\n".join(rows)


def build_ablation_table(ablation: List[Dict]) -> str:
    rows = ["| Edge ordering | Edges kept (%) | Length kept (%) | Certified | "
            "Probes/edge | Build (s) |", "|---|---|---|---|---|---|"]
    for a in ablation:
        rows.append(f"| `{a['order']}` | {a['edge_retention_pct']:.1f} | "
                    f"{a['length_retention_pct']:.1f} | "
                    f"{'yes' if a['certified'] else 'no'} | "
                    f"{a['probes_per_edge']:.2f} | {a['build_seconds']:.2f} |")
    return "\n".join(rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(map(str, o))
    return str(o)


# ==============================================================================
# SECTION 12 - MAIN PIPELINE
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delay-Bounded Time-Varying Geometric t-Spanner — Q1 benchmark")
    ap.add_argument("--lat", type=float, default=CENTER_POINT[0])
    ap.add_argument("--lon", type=float, default=CENTER_POINT[1])
    ap.add_argument("--dist", type=int, default=DEFAULT_DIST)
    ap.add_argument("--synthetic", action="store_true", help="skip Overpass entirely")
    ap.add_argument("--no-contract", action="store_true", help="disable junction contraction")
    ap.add_argument("--quick", action="store_true", help="reduced smoke-test footprint")
    ap.add_argument("--order", type=str, default="temporal_max",
                    choices=["temporal_max", "temporal_mean", "centrality"])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dist = 800 if args.quick else args.dist
    n_src = 6 if args.quick else N_SOURCES
    n_tgt = 6 if args.quick else N_TARGETS_PER_SOURCE
    repeats = 1 if args.quick else max(1, args.repeats)
    t_values = [1.15, 1.50, 2.20] if args.quick else list(T_VALUES)
    t_focus = T_FOCUS if T_FOCUS in t_values else t_values[len(t_values) // 2]
    hours = list(HOURS)

    banner = "=" * 90
    print(banner)
    print("  Delay-Bounded Time-Varying Geometric t-Spanner — experimental protocol")
    print(banner)

    # ---------------- network -------------------------------------------------
    G = build_network((args.lat, args.lon), dist,
                      force_synthetic=args.synthetic, contract=not args.no_contract)
    summary = network_summary(G)
    summary["temporal_heterogeneity_index"] = round(temporal_heterogeneity_index(G), 5)
    summary["fifo_violation_rate"] = round(fifo_violation_rate(G), 6)
    print(f"[network] {json.dumps(summary, default=_json_default)}")
    if summary["temporal_heterogeneity_index"] < 1e-6:
        print("[warn] temporal field is homothetic; static and temporal spanners coincide.")

    # ---------------- OD samples ---------------------------------------------
    pair_sets, bases = [], []
    for r in range(repeats):
        p = sample_pairs(G, n_src, n_tgt, args.seed + 1000 * r)
        pair_sets.append(p)
        t0 = time.perf_counter()
        bases.append(baseline_distances(G, p, hours))
        print(f"[eval] oracle {r+1}/{repeats}: {sum(len(v) for v in p.values())} OD pairs "
              f"x 24 snapshots in {time.perf_counter()-t0:.2f}s")

    results: Dict[Tuple[str, Optional[float]], Dict] = {}

    def record(key, H, cert_t: Optional[float], collect_at: Optional[int] = None):
        cert = temporal_stretch_certificate(G, H, cert_t, hours) if cert_t else {}
        met = evaluate_graph(H, pair_sets, hours, bases, collect_pairs_at=collect_at)
        results[key] = {
            "metrics": met,
            "structure": structural_metrics(G, H),
            "certificate": cert,
            "build_seconds": round(float(H.graph.get("build_seconds", 0.0)), 3),
            "dijkstra_probes": int(H.graph.get("dijkstra_probes", 0)),
        }
        s = results[key]["structure"]
        print(f"  [{key[0]:<15} t={('%.2f' % key[1]) if key[1] else '  — '}] "
              f"keep={s['edge_retention_pct']:5.1f}% | "
              f"worstOD={met['worst_dilation']:.3f} | "
              f"reach={100*met['reachability']:5.1f}% | "
              f"cert={cert.get('certified') if cert else '—'} | "
              f"stretch={_fmt(cert.get('attained_worst_edge_stretch')) if cert else '—'} | "
              f"{results[key]['build_seconds']:.2f}s")
        return met

    # ---------------- sweep ---------------------------------------------------
    print("\n[sweep] constructing spanners over the dilation grid ...")
    for t in t_values:
        H = delay_bounded_greedy_spanner(G, t, hours=hours, order=args.order)
        record(("delay_bounded", t), H, t)

        S = static_freeflow_spanner(G, t)
        record(("static_ff", t), S, t)

        M = mean_temporal_spanner(G, t)
        record(("mean_temporal", t), M, t)

    L = static_geometric_spanner(G, t_focus)
    record(("static_len", t_focus), L, t_focus)

    # ---------------- t-independent baselines ---------------------------------
    Hh = hierarchical_backbone(G)
    record(("hierarchical", None), Hh, t_focus)

    Hf = full_network(G)
    record(("full", None), Hf, None)

    # ---------------- matched-density controls --------------------------------
    budget = results[("delay_bounded", t_focus)]["structure"]["edges"]
    Hr = matched_density_random(G, budget, args.seed)
    record(("matched_random", t_focus), Hr, t_focus)
    Hb = matched_density_betweenness(G, budget, args.seed)
    record(("matched_btw", t_focus), Hb, t_focus)

    # ---------------- per-pair distribution at the worst hour -----------------
    worst_hour = int(results[("delay_bounded", t_focus)]["metrics"]["worst_hour"])
    print(f"\n[eval] peak-hour analysis at tau = {worst_hour}")
    pair_frames: Dict[str, pd.DataFrame] = {}
    per_pair_lookup: Dict[str, Dict[Tuple[str, str], float]] = {}
    for method, key in (("delay_bounded", ("delay_bounded", t_focus)),
                        ("static_ff", ("static_ff", t_focus)),
                        ("matched_btw", ("matched_btw", t_focus)),
                        ("hierarchical", ("hierarchical", None))):
        if key not in results:
            continue
        graph = {"delay_bounded": None}.get(method)
        H = {"delay_bounded": None}  # placeholder, rebuilt below
        # recompute the collected pairs for this method
        src_graph = {
            ("delay_bounded", t_focus): None,
        }
        # rebuild the corresponding graph object
        if key[0] == "delay_bounded":
            Hx = delay_bounded_greedy_spanner(G, t_focus, hours=hours,
                                              order=args.order, show_progress=False)
        elif key[0] == "static_ff":
            Hx = static_freeflow_spanner(G, t_focus, show_progress=False)
        elif key[0] == "matched_btw":
            Hx = matched_density_betweenness(G, budget, args.seed)
        else:
            Hx = hierarchical_backbone(G)
        Hx.graph["t"] = t_focus
        met = evaluate_graph(Hx, pair_sets[:1], hours, bases[:1],
                             collect_pairs_at=worst_hour)
        recs = met["collected_pairs"]
        if not recs:
            continue
        label = METHOD_STYLE[key[0]][0]
        pair_frames[key[0]] = pd.DataFrame(
            [{"source": a, "target": b, "dilation": r, "method": label} for a, b, r in recs])
        per_pair_lookup[key[0]] = {(a, b): r for a, b, r in recs}

    if pair_frames:
        pd.concat(list(pair_frames.values()), ignore_index=True).to_csv(PERPAIR_PATH, index=False)
        print(f"[report] wrote {PERPAIR_PATH}")

    # ---------------- inferential test ----------------------------------------
    test_out: Dict = {"available": False}
    if _HAVE_SCIPY and "delay_bounded" in per_pair_lookup and "matched_btw" in per_pair_lookup:
        a_map, b_map = per_pair_lookup["delay_bounded"], per_pair_lookup["matched_btw"]
        common = sorted(set(a_map) & set(b_map))
        if len(common) >= 10:
            a = np.array([a_map[k] for k in common])
            b = np.array([b_map[k] for k in common])
            try:
                stat, p = sps.wilcoxon(a, b, alternative="less", zero_method="zsplit")
                diff = b - a
                eff = float(np.mean(diff) / (np.std(diff) + 1e-12))
                test_out = {"available": True, "test": "Wilcoxon signed-rank (one-sided)",
                            "comparison": "delay_bounded < matched_density_betweenness",
                            "n_pairs": int(len(common)), "statistic": float(stat),
                            "p_value": float(p), "mean_delta": float(np.mean(diff)),
                            "effect_size_dz": eff, "hour": worst_hour}
                print(f"[stats] Wilcoxon (ours < matched-betweenness) at tau={worst_hour}: "
                      f"n={len(common)}, p={p:.3e}, dz={eff:.3f}")
            except Exception as exc:
                test_out = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---------------- time-dependent (FIFO) validation ------------------------
    print("[validation] FIFO time-dependent check (Proposition 2) ...")
    Hfoc = delay_bounded_greedy_spanner(G, t_focus, hours=hours,
                                        order=args.order, show_progress=False)
    small_pairs = {s: v[:4] for s, v in list(pair_sets[0].items())[:6]}
    departures = [7.5, 8.5, 12.0, 17.0, 18.0] if not args.quick else [8.0, 17.5]
    td_ours = time_dependent_validation(G, Hfoc, small_pairs, departures)
    td_static = time_dependent_validation(G, static_freeflow_spanner(G, t_focus, False),
                                          small_pairs, departures)
    print(f"[validation] TD worst dilation — ours: {td_ours['worst_td_dilation']:.3f} "
          f"(bound {t_focus}) | static free-flow: {td_static['worst_td_dilation']:.3f}")

    # ---------------- ordering ablation ---------------------------------------
    print("[ablation] edge-ordering heuristics ...")
    ablation: List[Dict] = []
    for mode in ("temporal_max", "temporal_mean", "centrality"):
        Ha = delay_bounded_greedy_spanner(G, t_focus, hours=hours, order=mode,
                                          show_progress=False)
        ca = temporal_stretch_certificate(G, Ha, t_focus, hours, refine_cap=0)
        sa = structural_metrics(G, Ha)
        ablation.append({
            "order": mode,
            "edge_retention_pct": sa["edge_retention_pct"],
            "length_retention_pct": sa["length_retention_pct"],
            "certified": bool(ca["certified"]),
            "probes_per_edge": Ha.graph["dijkstra_probes"] / max(1, G.number_of_edges()),
            "build_seconds": round(Ha.graph["build_seconds"], 3),
        })
        print(f"  [{mode:<14}] keep={sa['edge_retention_pct']:5.1f}% | "
              f"certified={ca['certified']} | {Ha.graph['build_seconds']:.2f}s")

    # ---------------- runtime / probe scaling ---------------------------------
    print("[scaling] empirical construction-cost scaling ...")
    scaling: List[Dict] = []
    fracs = [0.45, 0.7, 1.0] if args.quick else [0.35, 0.5, 0.65, 0.8, 1.0]
    for f in fracs:
        S = subgraph_by_radius(G, f) if f < 1.0 else G
        if S.number_of_edges() < 30:
            continue
        Hs = delay_bounded_greedy_spanner(S, t_focus, hours=hours,
                                          order="temporal_max", show_progress=False)
        scaling.append({
            "fraction": f, "nodes": S.number_of_nodes(), "edges": S.number_of_edges(),
            "seconds": round(Hs.graph["build_seconds"], 4),
            "probes": int(Hs.graph["dijkstra_probes"]),
            "probes_per_edge": Hs.graph["dijkstra_probes"] / max(1, S.number_of_edges()),
            "edge_retention_pct": structural_metrics(S, Hs)["edge_retention_pct"],
        })
        print(f"  |E|={S.number_of_edges():5d} -> {scaling[-1]['seconds']:.3f}s, "
              f"{scaling[-1]['probes_per_edge']:.2f} probes/edge, "
              f"keep {scaling[-1]['edge_retention_pct']:.1f}%")

    # ---------------- figure and tables ---------------------------------------
    make_figure(G, results, Hfoc, hours, t_focus, t_values, scaling, pair_frames, FIGURE_PATH)

    table = build_table(results, t_values)
    with open(TABLE_PATH, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(f"[report] wrote {TABLE_PATH}")

    abl_table = build_ablation_table(ablation)
    with open(ABLATION_PATH, "w", encoding="utf-8") as fh:
        fh.write(abl_table + "\n")
    print(f"[report] wrote {ABLATION_PATH}")

    clean_results = {}
    for (m, t), v in results.items():
        vv = dict(v)
        vv["metrics"] = {k: val for k, val in v["metrics"].items() if k != "collected_pairs"}
        clean_results[f"{m}|{'' if t is None else t}"] = vv

    report = {
        "framework": "Delay-Bounded Time-Varying Geometric t-Spanner",
        "version": "2.0.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "formulation": ("for all u,v in V, for all tau in {0..23}: "
                        "dist_H(u,v,tau) <= t * dist_G(u,v,tau)"),
        "certificate_principle": ("Theorem 1: the edge-wise invariant "
                                  "dist_H(u,v,tau) <= t*w((u,v),tau) implies the "
                                  "all-pairs, all-hours guarantee."),
        "configuration": {
            "center": [args.lat, args.lon], "dist_m": dist, "hours": hours,
            "t_values": t_values, "t_focus": t_focus, "edge_order": args.order,
            "seed": args.seed, "repeats": repeats, "sources": n_src,
            "targets_per_source": n_tgt, "junction_contraction": not args.no_contract,
            "bootstrap_samples": BOOTSTRAP_N, "quick": bool(args.quick),
        },
        "network": summary,
        "peak_hour": worst_hour,
        "results": clean_results,
        "matched_density_budget_edges": int(budget),
        "significance_test": test_out,
        "time_dependent_validation": {
            "delay_bounded": td_ours, "static_freeflow": td_static,
            "note": ("FIFO piecewise-linear interpolation of the hourly profiles; "
                     "a TD dilation at or below t empirically confirms Proposition 2."),
        },
        "ordering_ablation": ablation,
        "scaling_study": scaling,
        "known_limitations": [
            "Snapshot (time-slice) model: no departure-time evolution within a single "
            "traversal; validated ex post by the FIFO time-dependent experiment.",
            "Speed profiles are an analytic tidal model, not probe/loop-detector data; "
            "replacing edge_congestion_curve() requires no algorithmic change.",
            "Greedy construction gives no approximation guarantee on |H|; the reported "
            "matched-density controls act as the empirical lower-bound proxy.",
            "Single metropolitan area; external validity requires a multi-city replication.",
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    print(f"[report] wrote {REPORT_PATH}")

    print("\n" + table + "\n")
    print(abl_table + "\n")
    print("Done.")


if __name__ == "__main__":
    main()