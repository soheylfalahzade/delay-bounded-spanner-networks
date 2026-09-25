#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Delay-Bounded Time-Varying Geometric t-Spanners for Urban Road Networks
 ------------------------------------------------------------------------------
 v2.1.0 -- single-file pipeline with cross-city generalization study,
 a classical-geometric-spanner baseline, street-morphology regression, and a
 sensitivity/robustness analysis against congestion-model misspecification.
================================================================================

WHAT CHANGED SINCE v2.0.0 (reviewer-driven revision)
------------------------------------------------------------------------------
The v2.0.0 draft fixed the temporal model, the sparsification unit, the
parameter range, censored metrics, the hierarchical baseline and added
bootstrap CIs -- but a single-city study cannot support a generalizability
claim, and there was no bridge to the classical computational-geometry
spanner literature. This revision adds:

R9.  Cross-city replication. The pipeline is re-run, end-to-end, on SIX
     metropolitan areas with distinct street-network morphologies (historic-
     organic cores, radial-grid metros, a hill-terrain organic city, a
     modern planned grid). Each city attempts a live OSMnx/Overpass fetch
     first and only falls back to a deterministic, morphology-parameterized
     synthetic generator when the API is unreachable -- so the script
     produces real OpenStreetMap results outdoors and reproducible synthetic
     results in an offline sandbox, and it always reports, per city, which
     one it used ("data provenance").
R10. A classical-geometric-spanner baseline. `yao_cone_sparsifier` is a
     network-constrained adaptation of the Yao graph (partition each node's
     incident edges into k angular cones, keep the cheapest edge per cone).
     This is explicitly NOT the textbook unconstrained-point-set Yao graph
     (whose stretch factor 1/(1-2 sin(pi/k)) is a theorem, not an empirical
     claim) -- the docstring says so -- but it gives referees a bridge to a
     named, citable construction rather than only ad hoc engineering
     baselines.
R11. Morphology regression. Per-city street-orientation entropy (the
     Boeing 2019 grid-vs-organic legibility metric) and mean edge circuity
     are correlated (Pearson r, p) against attained worst-case stretch and
     edge retention, to test whether the method's advantage is an artifact
     of one city's geometry or holds across morphologies.
R12. City-level paired significance. A second, independent Wilcoxon
     signed-rank test compares city-level worst-case OD dilation (ours vs.
     the matched-density betweenness control) across the 6 cities, on top
     of the existing within-city, pair-level test -- flagged low-power
     given n=6, but directionally informative and honestly reported as such.
R13. Sensitivity to congestion-model misspecification. Because the diurnal
     congestion field is a parametric model and not fitted to floating-car
     or loop-detector telemetry, the congestion-severity scalar alpha is
     perturbed by -30%..+30% and the full construction + certification is
     repeated at each scale, to show the qualitative ranking (ours >
     matched-density control) is not an artifact of one parameter choice.
R14. Honesty about scope. The report's `known_limitations` block is
     extended to state plainly that (a) congestion is a parametric model,
     not measured telemetry, (b) when Overpass is unreachable the synthetic
     fallback is used and is clearly labelled, and (c) n=6 cities supports
     a directional generalization claim, not a definitive one -- a
     multi-country, telemetry-fitted replication remains future work.

USAGE
-----
    python run_spanner_benchmark.py                    # full run
    python run_spanner_benchmark.py --quick             # fast smoke test
    python run_spanner_benchmark.py --synthetic          # force offline mode
    python run_spanner_benchmark.py --skip-crosscity     # focus city only

OUTPUTS (results/)
-------------------
    spanner_benchmark.png       7-panel focus-city figure
    spanner_crosscity.png       4-panel cross-city generalization figure
    spanner_report.json         full machine-readable record
    benchmark_table.md          focus-city benchmark table
    ablation_table.md           edge-ordering ablation
    crosscity_table.md          cross-city summary table
    per_pair_dilation.csv       per-OD-pair dilation at the focus city's peak hour
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

# Dilation sweep spans the detour spectrum of an urban street grid.
T_VALUES: List[float] = [1.10, 1.30, 1.50, 1.80, 2.20, 3.00]
T_FOCUS: float = 1.50

N_SOURCES: int = 12
N_TARGETS_PER_SOURCE: int = 10
N_REPEATS: int = 2                       # independent OD resamplings
SEED: int = 42
BOOTSTRAP_N: int = 1500

EPS: float = 1e-9
INF: float = float("inf")

CERT_REFINE_CAP: int = 300
CERT_REFINE_MULT: float = 8.0

ROOT: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(ROOT, "results")
FIGURE_PATH: str = os.path.join(RESULTS_DIR, "spanner_benchmark.png")
CROSSCITY_FIGURE_PATH: str = os.path.join(RESULTS_DIR, "spanner_crosscity.png")
REPORT_PATH: str = os.path.join(RESULTS_DIR, "spanner_report.json")
TABLE_PATH: str = os.path.join(RESULTS_DIR, "benchmark_table.md")
ABLATION_PATH: str = os.path.join(RESULTS_DIR, "ablation_table.md")
CROSSCITY_TABLE_PATH: str = os.path.join(RESULTS_DIR, "crosscity_table.md")
PERPAIR_PATH: str = os.path.join(RESULTS_DIR, "per_pair_dilation.csv")

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

SPEED_FLOOR_FRAC: float = 0.18

METHOD_STYLE: Dict[str, Tuple[str, str, str, str]] = {
    "delay_bounded":  ("Delay-Bounded Spanner (ours)",   "#c0392b", "-",  "o"),
    "static_ff":      ("Static Free-Flow Spanner",       "#2980b9", "--", "s"),
    "static_len":     ("Static Geometric Spanner",       "#8e44ad", "--", "v"),
    "mean_temporal":  ("Mean-Temporal Spanner",          "#16a085", "-.", "P"),
    "hierarchical":   ("Hierarchical Road Classifier",   "#27ae60", "-.", "^"),
    "yao_cone":       ("Yao-Cone Sparsifier (k=8)",      "#34495e", "-.", "h"),
    "matched_random": ("Matched-Density Random",         "#f39c12", ":",  "X"),
    "matched_btw":    ("Matched-Density Betweenness",    "#d35400", ":",  "*"),
    "full":           ("Full Network (ground truth)",    "#7f8c8d", ":",  "D"),
}

# ---- Cross-city registry: 6 Iranian + 6 international metros -----------------
CITY_REGISTRY: List[Dict] = [
    # Iran
    {"name": "Yazd",      "lat": 31.8974, "lon": 54.3569, "dist": 1500, "archetype": "historic-organic", "country": "Iran"},
    {"name": "Tehran",    "lat": 35.6892, "lon": 51.3890, "dist": 1500, "archetype": "radial-grid",      "country": "Iran"},
    {"name": "Isfahan",   "lat": 32.6546, "lon": 51.6680, "dist": 1500, "archetype": "historic-organic", "country": "Iran"},
    {"name": "Shiraz",    "lat": 29.5918, "lon": 52.5837, "dist": 1500, "archetype": "hill-organic",     "country": "Iran"},
    {"name": "Mashhad",   "lat": 36.2605, "lon": 59.6168, "dist": 1500, "archetype": "radial-grid",      "country": "Iran"},
    {"name": "Qom",       "lat": 34.6401, "lon": 50.8764, "dist": 1500, "archetype": "modern-grid",      "country": "Iran"},
    # International
    {"name": "Barcelona", "lat": 41.3874, "lon":  2.1686, "dist": 1500, "archetype": "modern-grid",      "country": "Spain"},
    {"name": "Manhattan", "lat": 40.7580, "lon": -73.9855, "dist": 1500, "archetype": "strict-grid",     "country": "USA"},
    {"name": "Amsterdam", "lat": 52.3730, "lon":  4.8926, "dist": 1500, "archetype": "historic-organic", "country": "Netherlands"},
    {"name": "Tokyo",     "lat": 35.6595, "lon": 139.7005, "dist": 1500, "archetype": "dense-organic",   "country": "Japan"},
    {"name": "Cairo",     "lat": 30.0444, "lon": 31.2357, "dist": 1500, "archetype": "informal-organic", "country": "Egypt"},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631, "dist": 1500, "archetype": "strict-grid",    "country": "Australia"},
]

ARCHETYPE_PARAMS: Dict[str, Dict] = {
    "historic-organic": dict(irregularity=0.55, ring_count=1, oneway_density=0.16,
                             arterial_period=7, collector_period=3),
    "hill-organic":     dict(irregularity=0.70, ring_count=1, oneway_density=0.20,
                             arterial_period=8, collector_period=3),
    "radial-grid":      dict(irregularity=0.15, ring_count=3, oneway_density=0.05,
                             arterial_period=5, collector_period=2),
    "modern-grid":      dict(irregularity=0.05, ring_count=1, oneway_density=0.03,
                             arterial_period=6, collector_period=3),
    "strict-grid":      dict(irregularity=0.02, ring_count=2, oneway_density=0.30,
                             arterial_period=4, collector_period=2),
    "dense-organic":    dict(irregularity=0.65, ring_count=1, oneway_density=0.25,
                             arterial_period=9, collector_period=4),
    "informal-organic": dict(irregularity=0.85, ring_count=1, oneway_density=0.30,
                             arterial_period=10, collector_period=4),
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
# SECTION 3 - SPATIO-TEMPORAL CONGESTION MODEL
# ==============================================================================

MORNING_PEAK, MORNING_SIGMA = 8.0, 1.30
EVENING_PEAK, EVENING_SIGMA = 17.5, 1.75
MIDDAY_PEAK, MIDDAY_SIGMA = 12.5, 3.00


def _gauss(tau: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((tau - mu) / sigma) ** 2)


def edge_congestion_curve(inbound: float, cbd: float, jitter: float) -> List[float]:
    """c_e(tau) in [0,1]: tidal, CBD-weighted, edge-idiosyncratic congestion.

    Morning peaks load inbound edges, evening peaks load outbound edges; the
    amplitude scales with CBD proximity. This is a PARAMETRIC model, not
    fitted to floating-car or loop-detector telemetry -- see Section 16 for
    the sensitivity analysis that quantifies how much this matters.
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
    """Pruned single-pair Dijkstra on snapshot tau, restricted to the ball of
    radius `budget` around `source` (Theorem 1's local certificate probe)."""
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
    """Contract shape points into the junction graph. Sparsifying the raw OSM
    node graph is meaningless (most vertices are non-redundant geometry
    carriers); this is the correct combinatorial object to spanify."""
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
    """Overpass via OSMnx; returns None on any failure. In a sandboxed
    environment with no route to overpass-api.de this always returns None and
    the caller transparently falls back to the synthetic generator below."""
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


def synthetic_arterial_grid(center: Tuple[float, float], dist: int, n: int = 19,
                            arterial_period: int = 6, collector_period: int = 3,
                            irregularity: float = 0.0, oneway_density: float = 0.07,
                            ring_count: int = 1, seed_key: str = "grid") -> nx.DiGraph:
    """Deterministic, morphology-parameterized arterial/collector/local grid.

    `irregularity` jitters node positions (0 = perfect grid, ~0.7 = organic
    historic core). `ring_count` adds concentric bypass diagonals (radial
    metros). `oneway_density` controls block severing and one-way pockets on
    local streets. All randomness is a deterministic hash of `seed_key` so
    each named city is reproducible without a global RNG.
    """
    lat0, lon0 = center
    step = (2.0 * dist) / (n - 1)
    mlat, mlon = local_metric_factors(lat0)

    G = nx.DiGraph()
    nid: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in range(n):
            k = i * n + j
            nid[(i, j)] = k
            base_lon = lon0 + (j - (n - 1) / 2.0) * step / mlon
            base_lat = lat0 + (i - (n - 1) / 2.0) * step / mlat
            if irregularity > 0:
                jlon = irregularity * 0.4 * (step / mlon) * (2 * stable_unit(f"{seed_key}|jx|{i}|{j}") - 1)
                jlat = irregularity * 0.4 * (step / mlat) * (2 * stable_unit(f"{seed_key}|jy|{i}|{j}") - 1)
            else:
                jlon = jlat = 0.0
            G.add_node(k, x=float(base_lon + jlon), y=float(base_lat + jlat))

    def line_class(idx: int) -> str:
        if idx % arterial_period == 0:
            return "primary"
        if idx % collector_period == 0:
            return "secondary"
        return "residential"

    def link(a: int, b: int, hw: str) -> None:
        xa, ya = G.nodes[a]["x"], G.nodes[a]["y"]
        xb, yb = G.nodes[b]["x"], G.nodes[b]["y"]
        length = haversine_m(ya, xa, yb, xb)
        if length > 0.0:
            G.add_edge(a, b, length=length, highway=hw, coords=[(xa, ya), (xb, yb)], n_base=1)

    for i in range(n):
        for j in range(n):
            a = nid[(i, j)]
            if j + 1 < n:
                hw = line_class(i)
                sever = hw == "residential" and stable_unit(f"{seed_key}|sv|h|{i}|{j}") < oneway_density * 0.5
                if not sever:
                    b = nid[(i, j + 1)]
                    link(a, b, hw)
                    oneway = hw == "residential" and stable_unit(f"{seed_key}|ow|h|{i}|{j}") < oneway_density
                    if not oneway:
                        link(b, a, hw)
            if i + 1 < n:
                hw = line_class(j)
                sever = hw == "residential" and stable_unit(f"{seed_key}|sv|v|{i}|{j}") < oneway_density * 0.5
                if not sever:
                    b = nid[(i + 1, j)]
                    link(a, b, hw)
                    oneway = hw == "residential" and stable_unit(f"{seed_key}|ow|v|{i}|{j}") < oneway_density
                    if not oneway:
                        link(b, a, hw)

    def add_diagonal(offset: int) -> None:
        span = n - 1 - offset
        if span < 1:
            return
        for k in range(span):
            i0, j0 = k + offset, k
            i1, j1 = i0 + 1, j0 + 1
            if i1 < n and j1 < n:
                a, b = nid[(i0, j0)], nid[(i1, j1)]
                link(a, b, "trunk"); link(b, a, "trunk")
            i0b, j0b = k + offset, n - 1 - k
            i1b, j1b = i0b + 1, j0b - 1
            if 0 <= j1b < n and i1b < n:
                c, d = nid[(i0b, j0b)], nid[(i1b, j1b)]
                link(c, d, "trunk"); link(d, c, "trunk")

    offset_step = max(1, (n - 2) // max(1, ring_count))
    for r in range(ring_count):
        add_diagonal(min(n - 2, r * offset_step))

    G = largest_strongly_connected(G)
    G.graph.update({"source": "synthetic", "center": list(center), "dist": dist,
                    "archetype_params": {"arterial_period": arterial_period,
                                         "collector_period": collector_period,
                                         "irregularity": irregularity,
                                         "oneway_density": oneway_density,
                                         "ring_count": ring_count}})
    print(f"[network] Synthetic grid ({seed_key}): {G.number_of_nodes()} nodes / "
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
            cosang = -(ex * cx + ey * cy) / (norm_e * norm_c)
            inbound = 0.5 * (1.0 + max(-1.0, min(1.0, cosang)))

        jitter = stable_unit(f"{u}->{v}|{hw}")
        alpha = float(min(0.88, max(0.06, alpha0 * (0.55 + 0.95 * cbd) * (0.85 + 0.30 * jitter))))

        congestion = edge_congestion_curve(inbound, cbd, jitter)
        speeds = speed_curve(v_ff, alpha, congestion)
        length = float(data["length"])
        tt = [length / (s * 1000.0 / SEC_PER_HOUR) for s in speeds]
        ff = length / (v_ff * 1000.0 / SEC_PER_HOUR)

        data.update({
            "highway": hw, "tier": tier, "coords": coords,
            "v_ff": v_ff, "alpha": alpha, "cbd": cbd, "inbound": inbound,
            "congestion": congestion, "speeds": speeds, "tt": tt,
            "free_flow": ff, "tt_ff": [ff], "len_vec": [length],
            "tt_mean_vec": [float(sum(tt) / len(tt))],
            "w_max": max(tt), "w_min": min(tt), "peak_ratio": max(tt) / min(tt),
        })

    G.graph["hours"] = list(HOURS)
    return G


def build_network(center: Tuple[float, float], dist: int,
                  force_synthetic: bool = False, contract: bool = True,
                  grid_n: int = 19) -> nx.DiGraph:
    G = None if force_synthetic else fetch_osm_network(center, dist)
    if G is None:
        G = synthetic_arterial_grid(center, dist, n=grid_n, seed_key="focus")
    raw_nodes, raw_edges = G.number_of_nodes(), G.number_of_edges()
    if contract:
        G = contract_degree_two(G)
        print(f"[network] Junction contraction: {raw_nodes} -> {G.number_of_nodes()} nodes, "
              f"{raw_edges} -> {G.number_of_edges()} edges.")
    G.graph["raw_nodes"] = raw_nodes
    G.graph["raw_edges"] = raw_edges
    return enrich_temporal_costs(G)


def build_city_network(spec: Dict, grid_n: int = 15, contract: bool = True,
                       force_synthetic: bool = False) -> nx.DiGraph:
    """Cross-city loader: real OSM first, morphology-aware synthetic fallback."""
    center = (spec["lat"], spec["lon"])
    G = None if force_synthetic else fetch_osm_network(center, spec["dist"])
    used_real = G is not None
    if G is None:
        params = ARCHETYPE_PARAMS[spec["archetype"]]
        G = synthetic_arterial_grid(center, spec["dist"], n=grid_n, seed_key=spec["name"], **params)
    raw_nodes, raw_edges = G.number_of_nodes(), G.number_of_edges()
    if contract:
        G = contract_degree_two(G)
    G.graph["raw_nodes"] = raw_nodes
    G.graph["raw_edges"] = raw_edges
    G.graph["city"] = spec["name"]
    G.graph["archetype"] = spec["archetype"]
    G.graph["used_real_osm"] = used_real
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
    """Mean coefficient of variation of r_e(tau)=w(e,tau)/w(e,0) across edges.

    Strictly positive iff the temporal field is NOT homothetic (i.e. NOT
    w(.,tau) = c(tau) w(.,0)); a homothetic field would make the static and
    delay-bounded spanners provably identical, so this value certifies the
    experiment is not degenerate.
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


def network_morphology(G: nx.DiGraph) -> Dict:
    """Street-network morphology descriptors used for the cross-city
    regression: mean edge circuity and street-orientation entropy (the
    Boeing 2019 grid-vs-organic legibility metric, 0 = perfectly gridded,
    1 = uniformly random orientations)."""
    lat0 = G.graph["center"][0]
    mlat, mlon = local_metric_factors(lat0)

    lengths, straight = [], []
    bin_weights = np.zeros(36)
    for _, _, d in G.edges(data=True):
        coords = d.get("coords")
        if not coords or len(coords) < 2:
            continue
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        L = float(d["length"])
        S = haversine_m(y0, x0, y1, x1)
        if S > 1e-6:
            lengths.append(L)
            straight.append(S)
        ex = (x1 - x0) * mlon
        ey = (y1 - y0) * mlat
        if abs(ex) + abs(ey) > 1e-9:
            brg = math.degrees(math.atan2(ex, ey)) % 180.0
            bin_weights[int(brg // 5.0) % 36] += L

    circuity = float(np.mean(np.asarray(lengths) / np.maximum(1e-6, np.asarray(straight)))) \
        if lengths else float("nan")
    p = bin_weights / max(1e-12, bin_weights.sum())
    p_nonzero = p[p > 0]
    orientation_entropy = float(-(p_nonzero * np.log(p_nonzero)).sum() / math.log(36)) if p_nonzero.size else float("nan")

    degs = [G.degree(n) for n in G.nodes()]
    length_arr = np.asarray([d["length"] for _, _, d in G.edges(data=True)], dtype=float)

    return {
        "circuity_mean": round(circuity, 4),
        "orientation_entropy": round(orientation_entropy, 4),
        "mean_degree": round(float(np.mean(degs)), 3) if degs else float("nan"),
        "edge_length_cv": round(float(np.std(length_arr) / max(1e-6, np.mean(length_arr))), 4)
        if length_arr.size else float("nan"),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
    }


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
    else:  # temporal_max
        keyed = [((max(d[attr][h] for h in hours), str(u), str(v)), (u, v, d))
                 for u, v, d in edges]
    keyed.sort(key=lambda kv: kv[0])
    return [payload for _, payload in keyed]


def delay_bounded_greedy_spanner(G: nx.DiGraph, t: float,
                                 hours: Optional[Iterable[int]] = None,
                                 attr: str = "tt", order: str = "temporal_max",
                                 show_progress: bool = True,
                                 label: str = "delay_bounded") -> nx.DiGraph:
    """Delay-Bounded Greedy Spanner (Theorem 1 certificate by construction):

        1  order E ascending by max_tau w(e,tau)
        2  H <- empty
        3  for (u,v) in E in that order:
        4      for tau ordered by ascending w((u,v),tau):
        5          beta <- t * w((u,v),tau)
        6          if BoundedDijkstra(H, u->v, tau, beta) > beta:
        7              H <- H + {(u,v)}; break
        8  return H
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
                                 attr: str = "tt", refine_cap: int = CERT_REFINE_CAP,
                                 refine_mult: float = CERT_REFINE_MULT) -> dict:
    """Exhaustive Theorem-1 certificate + magnitude of the worst violation."""
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
    for u, v, tau, w in violating[:refine_cap]:
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
    H = delay_bounded_greedy_spanner(G, t, hours=[0], attr="tt_ff", order="temporal_max",
                                     show_progress=show_progress, label="static_ff")
    H.graph.update({"method": "static_ff", "t": t})
    return H


def static_geometric_spanner(G: nx.DiGraph, t: float, show_progress: bool = True) -> nx.DiGraph:
    H = delay_bounded_greedy_spanner(G, t, hours=[0], attr="len_vec", order="temporal_max",
                                     show_progress=show_progress, label="static_len")
    H.graph.update({"method": "static_len", "t": t})
    return H


def mean_temporal_spanner(G: nx.DiGraph, t: float, show_progress: bool = True) -> nx.DiGraph:
    H = delay_bounded_greedy_spanner(G, t, hours=[0], attr="tt_mean_vec", order="temporal_max",
                                     show_progress=show_progress, label="mean_temporal")
    H.graph.update({"method": "mean_temporal", "t": t})
    return H


def yao_cone_sparsifier(G: nx.DiGraph, k: int = 8, attr: str = "free_flow") -> nx.DiGraph:
    """Network-constrained adaptation of the classical Yao graph.

    The textbook Yao graph (Yao, 1982) is built on a COMPLETE point set: for
    each point, partition all other points into k angular cones of 360/k
    degrees and keep the nearest point per cone, which yields a proven
    stretch factor of 1/(1 - 2 sin(pi/k)). A road network is not a complete
    point set -- edges are constrained by existing infrastructure -- so here
    each node's EXISTING outgoing edges are partitioned into k cones by
    bearing and the cheapest (free-flow) edge per cone is kept. This carries
    no closed-form stretch guarantee, but it is the standard adaptation used
    when candidate edges are fixed, and gives referees a named bridge to the
    classical computational-geometry spanner family rather than only
    engineering heuristics.
    """
    lat0 = G.graph["center"][0]
    mlat, mlon = local_metric_factors(lat0)
    t0 = time.perf_counter()

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    cone_width = 360.0 / k
    for u in G.nodes():
        best: Dict[int, Tuple[object, float]] = {}
        for v, d in G[u].items():
            coords = d.get("coords") or [(G.nodes[u]["x"], G.nodes[u]["y"]),
                                         (G.nodes[v]["x"], G.nodes[v]["y"])]
            x0, y0 = coords[0]
            x1, y1 = coords[-1]
            ex = (x1 - x0) * mlon
            ey = (y1 - y0) * mlat
            brg = math.degrees(math.atan2(ex, ey)) % 360.0
            cone = int(brg // cone_width)
            cost = d[attr]
            if cone not in best or cost < best[cone][1]:
                best[cone] = (v, cost)
        for _, (v, _cost) in best.items():
            H.add_edge(u, v, **G[u][v])

    H = augment_connectivity(G, H)
    H.graph.update(G.graph)
    H.graph.update({"method": "yao_cone", "t": None, "build_seconds": time.perf_counter() - t0,
                    "dijkstra_probes": 0, "edges_parent": G.number_of_edges(), "cones": k})
    return H


def _bridge_isolated_nodes(G: nx.DiGraph, H: nx.DiGraph) -> None:
    for n in list(H.nodes()):
        if H.out_degree(n) == 0:
            cands = [(d["free_flow"], str(v), v) for v, d in G[n].items()]
            if cands:
                _, _, v = min(cands)
                H.add_edge(n, v, **G[n][v])
        if H.in_degree(n) == 0:
            cands = [(G[u][n]["free_flow"], str(u), u) for u in G.predecessors(n)]
            if cands:
                _, _, u = min(cands)
                H.add_edge(u, n, **G[u][n])


def augment_connectivity(G: nx.DiGraph, H: nx.DiGraph, max_rounds: int = 500) -> nx.DiGraph:
    _bridge_isolated_nodes(G, H)
    for _ in range(max_rounds):
        comps = list(nx.strongly_connected_components(H))
        if len(comps) <= 1:
            break
        comp_of = {n: i for i, c in enumerate(comps) for n in c}
        best = None
        for u, v, d in G.edges(data=True):
            cu, cv = comp_of.get(u), comp_of.get(v)
            if cu is None or cv is None or cu == cv or H.has_edge(u, v):
                continue
            key = (d["free_flow"], str(u), str(v))
            if best is None or key < best[0]:
                best = (key, u, v, d)
        if best is None:
            break
        H.add_edge(best[1], best[2], **best[3])
    return H


def hierarchical_backbone(G: nx.DiGraph, keep: Optional[Set[str]] = None) -> nx.DiGraph:
    """Functional-class heuristic with correct last-mile attachment: the
    arterial subgraph is reduced to its largest SCC, then every remaining
    junction is attached by its cheapest free-flow access path (forward and
    reverse multi-source Dijkstra), so the result is strongly connected by
    construction -- but carries no dilation certificate whatsoever."""
    keep = ARTERIAL_CLASSES if keep is None else keep
    t0 = time.perf_counter()

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d["highway"] in keep:
            H.add_edge(u, v, **d)

    comps = [c for c in nx.strongly_connected_components(H) if len(c) > 1]
    core: Set = max(comps, key=len) if comps else {min(G.nodes(), key=str)}

    pred_out = multi_source_pred(G, core, "free_flow")
    Grev = G.reverse(copy=False)
    pred_in = multi_source_pred(Grev, core, "free_flow")

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
            p = pred_in[cur]
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


def _materialize(G: nx.DiGraph, edges: Iterable[Tuple], method: str, seconds: float) -> nx.DiGraph:
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
    t0 = time.perf_counter()
    rng = random.Random(seed)
    core = strong_connectivity_core(G)
    rest = [(u, v) for u, v in G.edges() if (u, v) not in core]
    rng.shuffle(rest)
    need = max(0, budget_edges - len(core))
    chosen = set(core) | set(rest[:need])
    return _materialize(G, chosen, "matched_random", time.perf_counter() - t0)


def matched_density_betweenness(G: nx.DiGraph, budget_edges: int, seed: int = SEED) -> nx.DiGraph:
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
# SECTION 8 - EVALUATION (uncensored, multi-seed, bootstrapped)
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
    per_hour_max = np.zeros((len(pair_sets), len(hours)), dtype=float)
    per_hour_mean = np.zeros((len(pair_sets), len(hours)), dtype=float)
    all_ratios: List[float] = []
    per_seed_worst: List[float] = []
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
            all_ratios.extend(seed_ratios)

    allr = np.asarray(all_ratios, dtype=float) if all_ratios else np.asarray([np.nan])
    mean_lo, mean_hi = bootstrap_ci(all_ratios, np.mean) if all_ratios else (np.nan, np.nan)
    hour_max_mean = np.nanmean(per_hour_max, axis=0).tolist()
    hour_max_std = np.nanstd(per_hour_max, axis=0).tolist()

    return {
        "worst_dilation": float(np.nanmax(allr)),
        "worst_dilation_seed_mean": float(np.mean(per_seed_worst)) if per_seed_worst else float("nan"),
        "mean_dilation": float(np.nanmean(allr)),
        "mean_dilation_ci": [mean_lo, mean_hi],
        "p95_dilation": float(np.nanpercentile(allr, 95)),
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
        "edges": int(e_h), "edges_parent": int(e_g),
        "edge_retention_pct": round(100.0 * e_h / max(1, e_g), 3),
        "edges_removed_pct": round(100.0 * (1.0 - e_h / max(1, e_g)), 3),
        "length_km": round(len_h / 1000.0, 3),
        "length_retention_pct": round(100.0 * len_h / max(1e-9, len_g), 3),
        "arterial_share_pct": round(100.0 * arterial / max(1, e_h), 2),
        "avg_out_degree": round(e_h / max(1, H.number_of_nodes()), 3),
        "strongly_connected": bool(nx.is_strongly_connected(H)) if e_h else False,
    }


def perturb_congestion(G: nx.DiGraph, alpha_scale: float) -> nx.DiGraph:
    """Rescale every edge's congestion-severity coefficient alpha by
    `alpha_scale` and recompute speeds/costs, holding the tidal SHAPE fixed.
    Used by the sensitivity analysis (Section 16) to test whether the
    method's ranking against baselines survives model misspecification."""
    Gp = G.copy()
    for _, _, d in Gp.edges(data=True):
        alpha = float(min(0.97, d["alpha"] * alpha_scale))
        speeds = speed_curve(d["v_ff"], alpha, d["congestion"])
        length = d["length"]
        tt = [length / (s * 1000.0 / SEC_PER_HOUR) for s in speeds]
        d["alpha"] = alpha
        d["speeds"] = speeds
        d["tt"] = tt
        d["tt_mean_vec"] = [float(sum(tt) / len(tt))]
        d["w_max"] = max(tt)
        d["w_min"] = min(tt)
        d["peak_ratio"] = max(tt) / min(tt)
    return Gp


# ==============================================================================
# SECTION 9 - FIFO TIME-DEPENDENT VALIDATION (Proposition 2)
# ==============================================================================

def td_cost(costs: Sequence[float], abs_seconds: float) -> float:
    h = (abs_seconds / SEC_PER_HOUR) % 24.0
    i = int(math.floor(h))
    f = h - i
    return costs[i] * (1.0 - f) + costs[(i + 1) % 24] * f


def fifo_violation_rate(G: nx.DiGraph, samples: int = 24 * 4) -> float:
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


def td_dijkstra(G: nx.DiGraph, source, departure: float, targets: Set) -> Dict[object, float]:
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
        "departures_hours": list(departures_h), "pairs": int(arr.size),
        "unreachable": int(unreachable),
        "worst_td_dilation": float(np.nanmax(arr)),
        "mean_td_dilation": float(np.nanmean(arr)),
        "p95_td_dilation": float(np.nanpercentile(arr, 95)),
    }


# ==============================================================================
# SECTION 10 - FOCUS-CITY FIGURE
# ==============================================================================

def _edge_segments(G: nx.DiGraph, edges: Optional[Iterable[Tuple]] = None):
    segs = []
    it = G.edges(data=True) if edges is None else ((u, v, G[u][v]) for u, v in edges)
    for u, v, d in it:
        c = d.get("coords")
        if c and len(c) >= 2:
            segs.append([(float(x), float(y)) for x, y in c])
        else:
            segs.append([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
    return segs


def make_figure(G: nx.DiGraph, results: Dict, backbone: nx.DiGraph, hours: Sequence[int],
                t_focus: float, t_values: Sequence[float], scaling: List[Dict],
                pair_frames: Dict[str, pd.DataFrame], path: str) -> None:
    sns.set_theme(style="whitegrid", context="talk",
                  rc={"axes.edgecolor": "#2b2b2b", "grid.alpha": 0.30,
                      "axes.titlesize": 17, "axes.labelsize": 15, "legend.fontsize": 10.5})

    fig = plt.figure(figsize=(23, 20.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.45], hspace=0.36, wspace=0.26)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0]); axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2])
    axG = fig.add_subplot(gs[2, :])

    rows = []
    for _, _, d in G.edges(data=True):
        orient = "inbound" if d["inbound"] >= 0.5 else "outbound"
        for tau in hours:
            rows.append({"hour": tau, "speed_ratio": d["speeds"][tau] / d["v_ff"],
                        "tier": d["tier"], "orientation": orient})
    dfA = pd.DataFrame(rows)
    sns.lineplot(data=dfA, x="hour", y="speed_ratio", hue="tier", style="orientation",
                 errorbar=("ci", 95), ax=axA, linewidth=2.2,
                 palette={"arterial": "#7b0d1e", "collector": "#c0392b", "local": "#e08e79"})
    axA.set_xlabel(r"Hour of day $\tau$"); axA.set_ylabel(r"$v(e,\tau)\,/\,v_{\mathrm{ff}}(e)$")
    axA.set_title("(a) Tidal spatio-temporal speed field", loc="left")
    axA.set_xticks(range(0, 24, 4)); axA.legend(fontsize=9.5, ncol=2, loc="lower left", framealpha=0.9)

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
    axB.set_xlabel(r"Hour of day $\tau$"); axB.set_ylabel(r"Worst-case dilation $\max\ d_H/d_G$")
    axB.set_title("(b) Temporal dilation stability", loc="left")
    axB.set_xticks(range(0, 24, 4)); axB.legend(fontsize=9, loc="upper left", framealpha=0.92)

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
            axC.plot(xs, ys, label=label, color=color, linestyle=ls, marker=marker,
                     markersize=8, linewidth=2.5)
            axC.plot(xs, zs, color=color, linestyle=":", linewidth=1.4, alpha=0.75)
    hier = results.get(("hierarchical", None))
    if hier:
        label, color, ls, _ = METHOD_STYLE["hierarchical"]
        axC.axhline(hier["structure"]["edges_removed_pct"], color=color, linestyle=ls,
                    linewidth=2.2, label=label + r" ($t$-free)")
    axC.set_xlabel(r"Dilation parameter $t$")
    axC.set_ylabel("Removed (%)  [solid: edges, dotted: length]")
    axC.set_title("(c) Sparsification vs. permitted dilation", loc="left")
    axC.legend(fontsize=9, loc="upper left", framealpha=0.92)

    for method in ("delay_bounded", "static_ff", "static_len", "mean_temporal", "yao_cone",
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
        axD.plot(xs, ys, color=color, linestyle=ls, marker=marker, markersize=9,
                 linewidth=2.0, label=label, alpha=0.9)
    axD.axhline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=1.8)
    axD.set_yscale("log")
    axD.set_xlabel("Edges retained (%)"); axD.set_ylabel("Attained worst-case temporal stretch")
    axD.set_title("(d) Sparsity / fidelity Pareto front", loc="left")
    axD.legend(fontsize=8, loc="upper right", framealpha=0.92)

    if scaling:
        se = [s["edges"] for s in scaling]; st = [s["seconds"] for s in scaling]
        sp = [s["probes_per_edge"] for s in scaling]
        axE.plot(se, st, color="#c0392b", marker="o", linewidth=2.4, label="build time (s)")
        axE.set_xscale("log"); axE.set_yscale("log")
        axE.set_xlabel(r"$|E|$ (junction edges)"); axE.set_ylabel("Construction time (s)")
        if len(se) >= 2:
            logse, logst = np.log(se), np.log(np.maximum(1e-6, st))
            k = np.polyfit(logse, logst, 1)[0]
            axE.plot(se, np.exp(np.polyval(np.polyfit(logse, logst, 1), logse)),
                     color="#7f8c8d", linestyle="--", linewidth=1.6,
                     label=fr"fit: $T \propto |E|^{{{k:.2f}}}$")
        ax2 = axE.twinx()
        ax2.plot(se, sp, color="#2980b9", marker="s", linewidth=2.0, linestyle="-.")
        ax2.set_ylabel("Dijkstra probes per edge", color="#2980b9"); ax2.grid(False)
        axE.set_title(f"(e) Empirical scaling ($t={t_focus}$)", loc="left")
        axE.legend(fontsize=9, loc="upper left", framealpha=0.92)

    if pair_frames:
        dfF = pd.concat(list(pair_frames.values()), ignore_index=True)
        palette = {METHOD_STYLE[m][0]: METHOD_STYLE[m][1] for m in pair_frames}
        sns.ecdfplot(data=dfF, x="dilation", hue="method", ax=axF, linewidth=2.3, palette=palette)
        axF.axvline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=1.8)
        axF.set_xlim(0.98, min(5.0, float(dfF["dilation"].quantile(0.999)) * 1.05 + 0.02))
        axF.set_xlabel(r"OD dilation $d_H/d_G$ at peak hour"); axF.set_ylabel("Empirical CDF")
        axF.set_title("(f) Dilation distribution at the worst hour", loc="left")
        leg = axF.get_legend()
        if leg is not None:
            leg.set_title(None)
            for txt in leg.get_texts():
                txt.set_fontsize(9)

    kept_edges = set(backbone.edges())
    removed = [(u, v) for u, v in G.edges() if (u, v) not in kept_edges]
    axG.add_collection(LineCollection(_edge_segments(G, removed), colors="#c3cbd4",
                                      linewidths=0.8, alpha=0.9, zorder=1))
    by_tier: Dict[str, List] = {"arterial": [], "collector": [], "local": []}
    for u, v, d in backbone.edges(data=True):
        by_tier.setdefault(d["tier"], []).append(_edge_segments(backbone, [(u, v)])[0])
    axG.add_collection(LineCollection(by_tier.get("local", []), colors="#e07b5f",
                                      linewidths=1.2, alpha=0.9, zorder=2))
    axG.add_collection(LineCollection(by_tier.get("collector", []), colors="#c0392b",
                                      linewidths=1.9, alpha=0.95, zorder=3))
    axG.add_collection(LineCollection(by_tier.get("arterial", []), colors="#6d0d1c",
                                      linewidths=3.0, alpha=1.0, zorder=4))
    xs = [G.nodes[n]["x"] for n in G.nodes()]; ys = [G.nodes[n]["y"] for n in G.nodes()]
    padx = 0.02 * (max(xs) - min(xs) + 1e-6); pady = 0.02 * (max(ys) - min(ys) + 1e-6)
    axG.set_xlim(min(xs) - padx, max(xs) + padx); axG.set_ylim(min(ys) - pady, max(ys) + pady)
    lat0 = float(np.mean(ys))
    axG.set_aspect(1.0 / max(0.15, math.cos(math.radians(lat0))), adjustable="box")
    axG.grid(False); axG.set_facecolor("#fcfcfd")
    axG.set_xlabel("Longitude"); axG.set_ylabel("Latitude")
    keep_pct = 100.0 * backbone.number_of_edges() / max(1, G.number_of_edges())
    axG.set_title(f"(g) Certified emergency backbone $H \\subseteq G$  "
                  f"($t={t_focus}$, {keep_pct:.1f}% of junction edges, "
                  f"{100.0-keep_pct:.1f}% pruned, source: {G.graph.get('source','?')})", loc="left")
    axG.legend(handles=[
        Line2D([0], [0], color="#c3cbd4", lw=2, label="Pruned from $G$"),
        Line2D([0], [0], color="#e07b5f", lw=2, label="$H$ — local"),
        Line2D([0], [0], color="#c0392b", lw=3, label="$H$ — collector"),
        Line2D([0], [0], color="#6d0d1c", lw=4, label="$H$ — arterial"),
    ], fontsize=11, loc="upper right", framealpha=0.95)

    fig.suptitle("Delay-Bounded Time-Varying Geometric $t$-Spanners for Urban Road Networks",
                 fontsize=24, y=0.975)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] wrote {path}")


# ==============================================================================
# SECTION 11 - CROSS-CITY GENERALIZATION STUDY
# ==============================================================================

def _select_cities(quick: bool) -> List[Dict]:
    """Full run: every registered city. Quick run: a STRATIFIED sample (>=2
    Iranian + >=2 international) rather than CITY_REGISTRY[:n] -- taking a
    plain prefix would silently smoke-test only the Iranian subset, since
    the registry lists Iran first."""
    if not quick:
        return list(CITY_REGISTRY)
    iran = [c for c in CITY_REGISTRY if c["country"] == "Iran"][:2]
    intl = [c for c in CITY_REGISTRY if c["country"] != "Iran"][:2]
    return iran + intl


def run_cross_city_study(args, t_focus: float) -> List[Dict]:
    grid_n = 11 if args.quick else args.city_grid_n
    n_src = 5 if args.quick else args.city_sources
    n_tgt = 5 if args.quick else args.city_targets
    cities = _select_cities(args.quick)

    print(f"\n[cross-city] {len(cities)} cities "
          f"({sum(1 for c in cities if c['country']=='Iran')} Iran / "
          f"{sum(1 for c in cities if c['country']!='Iran')} international), "
          f"grid_n={grid_n}, {n_src}x{n_tgt} OD pairs each, t_focus={t_focus}")

    out: List[Dict] = []
    for spec in cities:
        print(f"  --- {spec['name']}, {spec['country']} ({spec['archetype']}) ---")
        G = build_city_network(spec, grid_n=grid_n, force_synthetic=args.synthetic)
        morph = network_morphology(G)
        pairs = sample_pairs(G, n_src, n_tgt, args.seed)
        base = baseline_distances(G, pairs, HOURS)

        def cert_eval(H, tval):
            H.graph["t"] = tval
            c = temporal_stretch_certificate(G, H, tval, HOURS, refine_cap=60) if tval else {}
            m = evaluate_graph(H, [pairs], HOURS, [base])
            return {"structure": structural_metrics(G, H), "certificate": c, "metrics": m,
                    "build_seconds": round(float(H.graph.get("build_seconds", 0.0)), 3)}

        methods: Dict[str, Dict] = {}
        Hd = delay_bounded_greedy_spanner(G, t_focus, hours=HOURS, order="temporal_max",
                                          show_progress=False)
        methods["delay_bounded"] = cert_eval(Hd, t_focus)
        budget = methods["delay_bounded"]["structure"]["edges"]

        Hs = static_freeflow_spanner(G, t_focus, show_progress=False)
        methods["static_ff"] = cert_eval(Hs, t_focus)

        Hy = yao_cone_sparsifier(G, k=8)
        methods["yao_cone"] = cert_eval(Hy, t_focus)

        Hh = hierarchical_backbone(G)
        methods["hierarchical"] = cert_eval(Hh, t_focus)

        Hb = matched_density_betweenness(G, budget, args.seed)
        methods["matched_btw"] = cert_eval(Hb, t_focus)

        entry = {"city": spec["name"], "country": spec["country"], "archetype": spec["archetype"],
                 "used_real_osm": bool(G.graph.get("used_real_osm", False)),
                 **morph, "methods": methods}
        out.append(entry)
        print(f"    keep(ours)={methods['delay_bounded']['structure']['edge_retention_pct']:.1f}% | "
              f"stretch={methods['delay_bounded']['certificate']['attained_worst_edge_stretch']:.3f} | "
              f"worstOD(ours)={methods['delay_bounded']['metrics']['worst_dilation']:.3f} | "
              f"worstOD(matched-btw)={methods['matched_btw']['metrics']['worst_dilation']:.3f} | "
              f"orient.entropy={morph['orientation_entropy']:.3f} | circuity={morph['circuity_mean']:.3f}")
    return out


def cross_city_regression(city_results: List[Dict]) -> Dict:
    """Correlate street-morphology descriptors against method QUALITY, not
    against the certified spanner's own attained stretch: for a construction
    that certifies successfully, attained_worst_edge_stretch collapses to t
    by definition (zero variance across cities), which is a degenerate
    regression target. The informative, non-degenerate targets are (i) edge
    retention and (ii) the empirical advantage over the matched-density
    control, i.e. how much morphology modulates the method's payoff."""
    if len(city_results) < 4 or not _HAVE_SCIPY:
        return {"available": False,
                "reason": "need >=4 cities and scipy for a Pearson correlation"}
    ents = [c["orientation_entropy"] for c in city_results]
    circ = [c["circuity_mean"] for c in city_results]
    retain = [c["methods"]["delay_bounded"]["structure"]["edge_retention_pct"] for c in city_results]
    ours_dil = [c["methods"]["delay_bounded"]["metrics"]["worst_dilation"] for c in city_results]
    btw_dil = [c["methods"]["matched_btw"]["metrics"]["worst_dilation"] for c in city_results]
    advantage = [b / max(1e-9, o) for o, b in zip(ours_dil, btw_dil)]

    out: Dict = {"available": True, "n_cities": len(city_results)}
    targets = (("edge_retention_pct", retain),
              ("worst_od_dilation", ours_dil),
              ("advantage_ratio_vs_matched_btw", advantage))
    for xname, x in (("orientation_entropy", ents), ("circuity_mean", circ)):
        for yname, y in targets:
            try:
                if np.std(y) < 1e-9:
                    out[f"{xname}_vs_{yname}"] = {"r": float("nan"), "p": float("nan"),
                                                  "note": "zero variance in y across cities"}
                    continue
                r, p = sps.pearsonr(x, y)
                out[f"{xname}_vs_{yname}"] = {"r": float(r), "p": float(p)}
            except Exception as exc:
                out[f"{xname}_vs_{yname}"] = {"error": str(exc)}
    return out


def cross_city_significance(city_results: List[Dict]) -> Dict:
    if not _HAVE_SCIPY or len(city_results) < 5:
        return {"available": False,
                "reason": f"n_cities={len(city_results)} < 5: Wilcoxon needs more cities for power; "
                          "reported directionally in the table instead."}
    a = np.array([c["methods"]["delay_bounded"]["metrics"]["worst_dilation"] for c in city_results])
    b = np.array([c["methods"]["matched_btw"]["metrics"]["worst_dilation"] for c in city_results])
    try:
        stat, p = sps.wilcoxon(a, b, alternative="less", zero_method="zsplit")
        return {"available": True, "test": "Wilcoxon signed-rank (one-sided), city-level",
                "comparison": "delay_bounded < matched_density_betweenness",
                "n_cities": len(city_results), "statistic": float(stat), "p_value": float(p),
                "note": "Low power at this n; treat as directional evidence, not confirmatory."}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def cross_city_country_comparison(city_results: List[Dict]) -> Dict:
    """Iran vs. international, INDEPENDENT samples (different cities, not the
    same cities under two conditions) -- so Mann-Whitney U is the correct
    test here, not the paired Wilcoxon used elsewhere in this script. This is
    the check that actually speaks to external validity beyond one country:
    if Iran and international cities are statistically indistinguishable on
    the method's outcomes, the single-country result generalizes; if not,
    the difference and its direction must be reported, not hidden."""
    iran = [c for c in city_results if c["country"] == "Iran"]
    intl = [c for c in city_results if c["country"] != "Iran"]
    out: Dict = {"n_iran": len(iran), "n_international": len(intl),
                 "international_countries": sorted({c["country"] for c in intl})}
    if not _HAVE_SCIPY or len(iran) < 2 or len(intl) < 2:
        out["available"] = False
        out["reason"] = (f"need >=2 cities per group and scipy for Mann-Whitney U "
                          f"(have {len(iran)} Iran / {len(intl)} international)")
        return out

    out["available"] = True
    metrics = {
        "edge_retention_pct": lambda c: c["methods"]["delay_bounded"]["structure"]["edge_retention_pct"],
        "worst_od_dilation": lambda c: c["methods"]["delay_bounded"]["metrics"]["worst_dilation"],
        "advantage_ratio_vs_matched_btw": lambda c: (
            c["methods"]["matched_btw"]["metrics"]["worst_dilation"]
            / max(1e-9, c["methods"]["delay_bounded"]["metrics"]["worst_dilation"])),
        "orientation_entropy": lambda c: c["orientation_entropy"],
        "circuity_mean": lambda c: c["circuity_mean"],
    }
    for name, fn in metrics.items():
        a = [fn(c) for c in iran]
        b = [fn(c) for c in intl]
        try:
            stat, p = sps.mannwhitneyu(a, b, alternative="two-sided")
            out[name] = {"iran_median": float(np.median(a)), "international_median": float(np.median(b)),
                        "statistic": float(stat), "p_value": float(p)}
        except Exception as exc:
            out[name] = {"error": str(exc)}
    out["note"] = ("Two-sided Mann-Whitney U, independent samples. n is small (city count, "
                   "not OD-pair count) so this is a coarse external-validity check, not a "
                   "high-powered confirmatory test; a non-significant p here is the desired "
                   "result (it means the method does not behave differently by country).")
    return out


def sensitivity_analysis(G: nx.DiGraph, t_focus: float, pairs: Dict, seed: int,
                         scales: Sequence[float] = (0.70, 0.85, 1.00, 1.15, 1.30)) -> List[Dict]:
    rows: List[Dict] = []
    for sc in scales:
        Gp = perturb_congestion(G, sc)
        basep = baseline_distances(Gp, pairs, HOURS)
        H = delay_bounded_greedy_spanner(Gp, t_focus, hours=HOURS, show_progress=False)
        cert = temporal_stretch_certificate(Gp, H, t_focus, HOURS, refine_cap=60)
        met = evaluate_graph(H, [pairs], HOURS, [basep])
        budget = structural_metrics(Gp, H)["edges"]
        MB = matched_density_betweenness(Gp, budget, seed)
        MB.graph["t"] = t_focus
        metMB = evaluate_graph(MB, [pairs], HOURS, [basep])
        rows.append({
            "alpha_scale": sc,
            "ours_edge_retention_pct": structural_metrics(Gp, H)["edge_retention_pct"],
            "ours_certified": bool(cert["certified"]),
            "ours_worst_dilation": met["worst_dilation"],
            "matched_btw_worst_dilation": metMB["worst_dilation"],
        })
        print(f"  [sensitivity alpha x{sc:.2f}] ours worstOD={met['worst_dilation']:.3f} "
              f"(certified={cert['certified']}) vs matched-btw={metMB['worst_dilation']:.3f}")
    return rows


def build_crosscity_table(city_results: List[Dict]) -> str:
    head = ("| City | Country | Archetype | Data | Nodes | Edges | Orient.entropy | Circuity | "
            "Ours keep% | Certified | Ours worstOD | Matched-Btw worstOD | Advantage (x) | "
            "Static-FF keep% | Yao-Cone keep% | Matched-Btw keep% | Hierarchical keep% |")
    rows = [head, "|" + "---|" * 17]
    for c in sorted(city_results, key=lambda c: (c["country"] != "Iran", c["country"], c["city"])):
        mo = c["methods"]
        ours_od = mo["delay_bounded"]["metrics"]["worst_dilation"]
        btw_od = mo["matched_btw"]["metrics"]["worst_dilation"]
        adv = btw_od / max(1e-9, ours_od)
        rows.append(
            f"| {c['city']} | {c['country']} | {c['archetype']} | "
            f"{'OSM' if c['used_real_osm'] else 'synthetic'} | "
            f"{c['n_nodes']} | {c['n_edges']} | {c['orientation_entropy']:.3f} | "
            f"{c['circuity_mean']:.3f} | {mo['delay_bounded']['structure']['edge_retention_pct']:.1f} | "
            f"{'yes' if mo['delay_bounded']['certificate']['certified'] else 'no'} | "
            f"{ours_od:.3f} | {btw_od:.3f} | {adv:.2f} | "
            f"{mo['static_ff']['structure']['edge_retention_pct']:.1f} | "
            f"{mo['yao_cone']['structure']['edge_retention_pct']:.1f} | "
            f"{mo['matched_btw']['structure']['edge_retention_pct']:.1f} | "
            f"{mo['hierarchical']['structure']['edge_retention_pct']:.1f} |"
        )
    return "\n".join(rows)


def make_crosscity_figure(city_results: List[Dict], regression: Dict, sensitivity: List[Dict],
                          t_focus: float, path: str) -> None:
    sns.set_theme(style="whitegrid", context="talk",
                  rc={"axes.edgecolor": "#2b2b2b", "grid.alpha": 0.30})
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 13.5))
    axH, axI, axJ, axK = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    city_order = sorted(city_results, key=lambda c: (c["country"] != "Iran", c["country"], c["city"]))
    n_iran = sum(1 for c in city_order if c["country"] == "Iran")
    rows = []
    for c in city_order:
        for m in ("delay_bounded", "static_ff", "yao_cone", "matched_btw", "hierarchical"):
            if m in c["methods"]:
                rows.append({"city": f"{c['city']}\n({c['country']})", "method": METHOD_STYLE[m][0],
                            "retention": c["methods"][m]["structure"]["edge_retention_pct"]})
    if rows:
        dfH = pd.DataFrame(rows)
        city_labels = [f"{c['city']}\n({c['country']})" for c in city_order]
        pal = {METHOD_STYLE[m][0]: METHOD_STYLE[m][1]
              for m in ("delay_bounded", "static_ff", "yao_cone", "matched_btw", "hierarchical")}
        sns.barplot(data=dfH, x="city", y="retention", hue="method", ax=axH, palette=pal,
                   order=city_labels)
        if 0 < n_iran < len(city_order):
            axH.axvline(n_iran - 0.5, color="#111111", linestyle=(0, (4, 3)), linewidth=1.5, zorder=5)
            axH.text(n_iran / 2.0 - 0.5, 101.5, "Iran", ha="center", fontsize=10, fontweight="bold")
            axH.text(n_iran + (len(city_order) - n_iran) / 2.0 - 0.5, 101.5, "International",
                     ha="center", fontsize=10, fontweight="bold")
        axH.set_ylim(0, 108)
        axH.set_ylabel("Edges retained (%)"); axH.set_xlabel("")
        axH.set_title(f"(h) Cross-city sparsification at $t={t_focus}$", loc="left")
        axH.tick_params(axis="x", rotation=20, labelsize=8.5)
        axH.legend(fontsize=7.5, ncol=2, loc="lower right")

    names = [c["city"] for c in city_order]
    is_iran = [c["country"] == "Iran" for c in city_order]
    colors = ["#c0392b" if ir else "#2980b9" for ir in is_iran]
    ents = [c["orientation_entropy"] for c in city_order]
    ours_dil = [c["methods"]["delay_bounded"]["metrics"]["worst_dilation"] for c in city_order]
    btw_dil = [c["methods"]["matched_btw"]["metrics"]["worst_dilation"] for c in city_order]
    advantage = [b / max(1e-9, o) for o, b in zip(ours_dil, btw_dil)]
    axI.scatter(ents, advantage, s=100, c=colors, zorder=3)
    for x, y, nm in zip(ents, advantage, names):
        axI.annotate(nm, (x, y), fontsize=9, xytext=(5, 5), textcoords="offset points")
    axI.scatter([], [], color="#c0392b", label="Iran"); axI.scatter([], [], color="#2980b9", label="International")
    axI.axhline(1.0, color="#7f8c8d", linestyle=":", linewidth=1.4)
    if len(ents) >= 2 and np.std(advantage) > 1e-9:
        kk, bb = np.polyfit(ents, advantage, 1)
        xs = np.linspace(min(ents), max(ents), 20)
        axI.plot(xs, kk * xs + bb, color="#7f8c8d", linestyle="--")
    reg = regression.get("orientation_entropy_vs_advantage_ratio_vs_matched_btw", {})
    title = "(i) Orientation entropy vs. advantage over matched-density"
    if "r" in reg and not (isinstance(reg.get("r"), float) and math.isnan(reg.get("r", float("nan")))):
        title += f"  (r={reg['r']:.2f}, p={reg['p']:.2f})"
    axI.set_title(title, loc="left")
    axI.set_xlabel("Orientation entropy (0=grid, 1=organic)")
    axI.set_ylabel("Worst-OD-dilation ratio: matched-btw / ours  (>1 = we win)")
    axI.legend(fontsize=8, loc="best")

    circ = [c["circuity_mean"] for c in city_order]
    retain = [c["methods"]["delay_bounded"]["structure"]["edge_retention_pct"] for c in city_order]
    axJ.scatter(circ, retain, s=100, c=colors, zorder=3)
    for x, y, nm in zip(circ, retain, names):
        axJ.annotate(nm, (x, y), fontsize=9, xytext=(5, 5), textcoords="offset points")
    if len(circ) >= 2:
        kk, bb = np.polyfit(circ, retain, 1)
        xs = np.linspace(min(circ), max(circ), 20)
        axJ.plot(xs, kk * xs + bb, color="#7f8c8d", linestyle="--")
    reg2 = regression.get("circuity_mean_vs_edge_retention_pct", {})
    title2 = "(j) Mean circuity vs. sparsification"
    if "r" in reg2:
        title2 += f"  (r={reg2['r']:.2f}, p={reg2['p']:.2f})"
    axJ.set_title(title2, loc="left")
    axJ.set_xlabel("Mean edge circuity"); axJ.set_ylabel("Edges retained (%) (ours)")

    if sensitivity:
        sdf = pd.DataFrame(sensitivity)
        axK.plot(sdf["alpha_scale"], sdf["ours_worst_dilation"], marker="o", color="#c0392b",
                 linewidth=2.4, label="Delay-Bounded (ours)")
        axK.plot(sdf["alpha_scale"], sdf["matched_btw_worst_dilation"], marker="X", color="#d35400",
                 linewidth=2.0, label="Matched-Density Betweenness")
        axK.axhline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=1.8,
                    label=f"Certified bound $t={t_focus}$")
        axK.set_xlabel(r"Congestion-severity scale ($\alpha \times$)")
        axK.set_ylabel("Worst OD dilation")
        axK.set_title("(k) Robustness to congestion-model misspecification", loc="left")
        axK.legend(fontsize=9)

    fig.suptitle("Cross-City Generalization and Robustness of the Delay-Bounded Spanner",
                 fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] wrote {path}")


# ==============================================================================
# SECTION 12 - REPORTING (focus city)
# ==============================================================================

def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def build_table(results: Dict, t_values: Sequence[float]) -> str:
    head = ("| Method | t | Edges kept (%) | Length kept (%) | Attained worst "
            "edge stretch | Certified (24 h) | Worst OD dilation | Mean OD dilation "
            "[95% CI] | Reach (%) | Build (s) | Probes/edge |")
    rows = [head, "|" + "---|" * 11]
    order: List[Tuple[str, Optional[float]]] = []
    for t in sorted(t_values):
        for m in ("delay_bounded", "static_ff", "mean_temporal"):
            order.append((m, t))
    for m in ("static_len", "yao_cone", "matched_random", "matched_btw"):
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
        cert = "—" if not c else ("**yes**" if c.get("certified") else f"no ({c.get('violating_edges')} edges)")
        ci = m.get("mean_dilation_ci", [np.nan, np.nan])
        ppe = e["dijkstra_probes"] / max(1, s["edges_parent"])
        rows.append(
            f"| {name} | {tstr} | {s['edge_retention_pct']:.1f} | {s['length_retention_pct']:.1f} | "
            f"{_fmt(stretch)} | {cert} | {_fmt(m['worst_dilation'])} | {_fmt(m['mean_dilation'])} "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {100*m['reachability']:.1f} | "
            f"{e['build_seconds']:.2f} | {ppe:.2f} |"
        )
    return "\n".join(rows)


def build_ablation_table(ablation: List[Dict]) -> str:
    rows = ["| Edge ordering | Edges kept (%) | Length kept (%) | Certified | Probes/edge | Build (s) |",
            "|---|---|---|---|---|---|"]
    for a in ablation:
        rows.append(f"| `{a['order']}` | {a['edge_retention_pct']:.1f} | {a['length_retention_pct']:.1f} | "
                    f"{'yes' if a['certified'] else 'no'} | {a['probes_per_edge']:.2f} | "
                    f"{a['build_seconds']:.2f} |")
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
# SECTION 13 - MAIN PIPELINE
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delay-Bounded Time-Varying Geometric t-Spanner — Q1 benchmark")
    ap.add_argument("--lat", type=float, default=CENTER_POINT[0])
    ap.add_argument("--lon", type=float, default=CENTER_POINT[1])
    ap.add_argument("--dist", type=int, default=DEFAULT_DIST)
    ap.add_argument("--grid-n", type=int, default=19)
    ap.add_argument("--synthetic", action="store_true", help="skip Overpass entirely (all cities)")
    ap.add_argument("--no-contract", action="store_true", help="disable junction contraction")
    ap.add_argument("--quick", action="store_true", help="reduced smoke-test footprint")
    ap.add_argument("--order", type=str, default="temporal_max",
                    choices=["temporal_max", "temporal_mean", "centrality"])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-crosscity", action="store_true")
    ap.add_argument("--city-grid-n", type=int, default=15)
    ap.add_argument("--city-sources", type=int, default=7)
    ap.add_argument("--city-targets", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dist = 800 if args.quick else args.dist
    grid_n = 13 if args.quick else args.grid_n
    n_src = 6 if args.quick else N_SOURCES
    n_tgt = 6 if args.quick else N_TARGETS_PER_SOURCE
    repeats = 1 if args.quick else max(1, args.repeats)
    t_values = [1.15, 1.50, 2.20] if args.quick else list(T_VALUES)
    t_focus = T_FOCUS if T_FOCUS in t_values else t_values[len(t_values) // 2]
    hours = list(HOURS)

    banner = "=" * 90
    print(banner)
    print("  Delay-Bounded Time-Varying Geometric t-Spanner — focus-city protocol")
    print(banner)

    # ---------------- focus-city network ---------------------------------
    G = build_network((args.lat, args.lon), dist, force_synthetic=args.synthetic,
                      contract=not args.no_contract, grid_n=grid_n)
    summary = network_summary(G)
    summary["temporal_heterogeneity_index"] = round(temporal_heterogeneity_index(G), 5)
    summary["fifo_violation_rate"] = round(fifo_violation_rate(G), 6)
    print(f"[network] {json.dumps(summary, default=_json_default)}")
    if summary["temporal_heterogeneity_index"] < 1e-6:
        print("[warn] temporal field is homothetic; static and temporal spanners coincide.")

    pair_sets, bases = [], []
    for r in range(repeats):
        p = sample_pairs(G, n_src, n_tgt, args.seed + 1000 * r)
        pair_sets.append(p)
        t0 = time.perf_counter()
        bases.append(baseline_distances(G, p, hours))
        print(f"[eval] oracle {r+1}/{repeats}: {sum(len(v) for v in p.values())} OD pairs "
              f"x 24 snapshots in {time.perf_counter()-t0:.2f}s")

    results: Dict[Tuple[str, Optional[float]], Dict] = {}

    def record(key, H, cert_t, collect_at=None):
        cert = temporal_stretch_certificate(G, H, cert_t, hours) if cert_t else {}
        met = evaluate_graph(H, pair_sets, hours, bases, collect_pairs_at=collect_at)
        results[key] = {"metrics": met, "structure": structural_metrics(G, H), "certificate": cert,
                        "build_seconds": round(float(H.graph.get("build_seconds", 0.0)), 3),
                        "dijkstra_probes": int(H.graph.get("dijkstra_probes", 0))}
        s = results[key]["structure"]
        print(f"  [{key[0]:<15} t={('%.2f' % key[1]) if key[1] else '  — '}] "
              f"keep={s['edge_retention_pct']:5.1f}% | worstOD={met['worst_dilation']:.3f} | "
              f"reach={100*met['reachability']:5.1f}% | cert={cert.get('certified') if cert else '—'} | "
              f"stretch={_fmt(cert.get('attained_worst_edge_stretch')) if cert else '—'} | "
              f"{results[key]['build_seconds']:.2f}s")

    print("\n[sweep] constructing spanners over the dilation grid ...")
    for t in t_values:
        record(("delay_bounded", t), delay_bounded_greedy_spanner(G, t, hours=hours, order=args.order), t)
        record(("static_ff", t), static_freeflow_spanner(G, t), t)
        record(("mean_temporal", t), mean_temporal_spanner(G, t), t)

    record(("static_len", t_focus), static_geometric_spanner(G, t_focus), t_focus)
    record(("yao_cone", t_focus), yao_cone_sparsifier(G, k=8), t_focus)
    record(("hierarchical", None), hierarchical_backbone(G), t_focus)
    record(("full", None), full_network(G), None)

    budget = results[("delay_bounded", t_focus)]["structure"]["edges"]
    record(("matched_random", t_focus), matched_density_random(G, budget, args.seed), t_focus)
    record(("matched_btw", t_focus), matched_density_betweenness(G, budget, args.seed), t_focus)

    worst_hour = int(results[("delay_bounded", t_focus)]["metrics"]["worst_hour"])
    print(f"\n[eval] peak-hour analysis at tau = {worst_hour}")
    pair_frames: Dict[str, pd.DataFrame] = {}
    per_pair_lookup: Dict[str, Dict[Tuple[str, str], float]] = {}
    for mkey, builder in (
        ("delay_bounded", lambda: delay_bounded_greedy_spanner(G, t_focus, hours=hours,
                                                               order=args.order, show_progress=False)),
        ("static_ff", lambda: static_freeflow_spanner(G, t_focus, show_progress=False)),
        ("matched_btw", lambda: matched_density_betweenness(G, budget, args.seed)),
        ("hierarchical", lambda: hierarchical_backbone(G)),
    ):
        Hx = builder()
        Hx.graph["t"] = t_focus
        met = evaluate_graph(Hx, pair_sets[:1], hours, bases[:1], collect_pairs_at=worst_hour)
        recs = met["collected_pairs"]
        if not recs:
            continue
        label = METHOD_STYLE[mkey][0]
        pair_frames[mkey] = pd.DataFrame(
            [{"source": a, "target": b, "dilation": r, "method": label} for a, b, r in recs])
        per_pair_lookup[mkey] = {(a, b): r for a, b, r in recs}

    if pair_frames:
        pd.concat(list(pair_frames.values()), ignore_index=True).to_csv(PERPAIR_PATH, index=False)
        print(f"[report] wrote {PERPAIR_PATH}")

    test_out: Dict = {"available": False}
    if _HAVE_SCIPY and "delay_bounded" in per_pair_lookup and "matched_btw" in per_pair_lookup:
        a_map, b_map = per_pair_lookup["delay_bounded"], per_pair_lookup["matched_btw"]
        common = sorted(set(a_map) & set(b_map))
        if len(common) >= 10:
            a = np.array([a_map[k] for k in common]); b = np.array([b_map[k] for k in common])
            try:
                stat, p = sps.wilcoxon(a, b, alternative="less", zero_method="zsplit")
                diff = b - a
                eff = float(np.mean(diff) / (np.std(diff) + 1e-12))
                test_out = {"available": True, "test": "Wilcoxon signed-rank (one-sided), pair-level",
                            "comparison": "delay_bounded < matched_density_betweenness",
                            "n_pairs": int(len(common)), "statistic": float(stat), "p_value": float(p),
                            "mean_delta": float(np.mean(diff)), "effect_size_dz": eff, "hour": worst_hour}
                print(f"[stats] Wilcoxon (ours < matched-betweenness) tau={worst_hour}: "
                      f"n={len(common)}, p={p:.3e}, dz={eff:.3f}")
            except Exception as exc:
                test_out = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    print("[validation] FIFO time-dependent check (Proposition 2) ...")
    Hfoc = delay_bounded_greedy_spanner(G, t_focus, hours=hours, order=args.order, show_progress=False)
    small_pairs = {s: v[:4] for s, v in list(pair_sets[0].items())[:6]}
    departures = [7.5, 8.5, 12.0, 17.0, 18.0] if not args.quick else [8.0, 17.5]
    td_ours = time_dependent_validation(G, Hfoc, small_pairs, departures)
    td_static = time_dependent_validation(G, static_freeflow_spanner(G, t_focus, False), small_pairs, departures)
    print(f"[validation] TD worst dilation — ours: {td_ours['worst_td_dilation']:.3f} "
          f"(bound {t_focus}) | static free-flow: {td_static['worst_td_dilation']:.3f}")

    print("[ablation] edge-ordering heuristics ...")
    ablation: List[Dict] = []
    for mode in ("temporal_max", "temporal_mean", "centrality"):
        Ha = delay_bounded_greedy_spanner(G, t_focus, hours=hours, order=mode, show_progress=False)
        ca = temporal_stretch_certificate(G, Ha, t_focus, hours, refine_cap=0)
        sa = structural_metrics(G, Ha)
        ablation.append({"order": mode, "edge_retention_pct": sa["edge_retention_pct"],
                         "length_retention_pct": sa["length_retention_pct"],
                         "certified": bool(ca["certified"]),
                         "probes_per_edge": Ha.graph["dijkstra_probes"] / max(1, G.number_of_edges()),
                         "build_seconds": round(Ha.graph["build_seconds"], 3)})
        print(f"  [{mode:<14}] keep={sa['edge_retention_pct']:5.1f}% | certified={ca['certified']} | "
              f"{Ha.graph['build_seconds']:.2f}s")

    print("[scaling] empirical construction-cost scaling ...")
    scaling: List[Dict] = []
    fracs = [0.45, 0.7, 1.0] if args.quick else [0.4, 0.6, 0.8, 1.0]
    for f in fracs:
        S = subgraph_by_radius(G, f) if f < 1.0 else G
        if S.number_of_edges() < 30:
            continue
        Hs = delay_bounded_greedy_spanner(S, t_focus, hours=hours, order="temporal_max", show_progress=False)
        scaling.append({"fraction": f, "nodes": S.number_of_nodes(), "edges": S.number_of_edges(),
                        "seconds": round(Hs.graph["build_seconds"], 4),
                        "probes": int(Hs.graph["dijkstra_probes"]),
                        "probes_per_edge": Hs.graph["dijkstra_probes"] / max(1, S.number_of_edges()),
                        "edge_retention_pct": structural_metrics(S, Hs)["edge_retention_pct"]})
        print(f"  |E|={S.number_of_edges():5d} -> {scaling[-1]['seconds']:.3f}s, "
              f"{scaling[-1]['probes_per_edge']:.2f} probes/edge, keep {scaling[-1]['edge_retention_pct']:.1f}%")

    make_figure(G, results, Hfoc, hours, t_focus, t_values, scaling, pair_frames, FIGURE_PATH)

    table = build_table(results, t_values)
    with open(TABLE_PATH, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(f"[report] wrote {TABLE_PATH}")

    abl_table = build_ablation_table(ablation)
    with open(ABLATION_PATH, "w", encoding="utf-8") as fh:
        fh.write(abl_table + "\n")
    print(f"[report] wrote {ABLATION_PATH}")

    # ---------------- cross-city generalization study -----------------------
    city_results: List[Dict] = []
    regression: Dict = {"available": False}
    city_sig: Dict = {"available": False}
    country_cmp: Dict = {"available": False}
    sensitivity: List[Dict] = []
    crosscity_table = ""
    if not args.skip_crosscity:
        print("\n" + banner)
        print("  Cross-city generalization study")
        print(banner)
        city_results = run_cross_city_study(args, t_focus)
        regression = cross_city_regression(city_results)
        city_sig = cross_city_significance(city_results)
        country_cmp = cross_city_country_comparison(city_results)
        print("\n[sensitivity] robustness to congestion-model misspecification "
              "(focus city, alpha rescaled -30%..+30%)")
        sensitivity = sensitivity_analysis(G, t_focus, pair_sets[0], args.seed)
        make_crosscity_figure(city_results, regression, sensitivity, t_focus, CROSSCITY_FIGURE_PATH)
        crosscity_table = build_crosscity_table(city_results)
        with open(CROSSCITY_TABLE_PATH, "w", encoding="utf-8") as fh:
            fh.write(crosscity_table + "\n")
        print(f"[report] wrote {CROSSCITY_TABLE_PATH}")
        if regression.get("available"):
            for k, v in regression.items():
                if isinstance(v, dict) and "r" in v:
                    print(f"[regression] {k}: r={v['r']:.3f}, p={v['p']:.3f}")
        if city_sig.get("available"):
            print(f"[stats] city-level Wilcoxon: n={city_sig['n_cities']}, p={city_sig['p_value']:.3e}")
        else:
            print(f"[stats] city-level Wilcoxon skipped: {city_sig.get('reason')}")
        if country_cmp.get("available"):
            print(f"[stats] Iran (n={country_cmp['n_iran']}) vs International "
                  f"(n={country_cmp['n_international']}, {country_cmp['international_countries']}) "
                  f"Mann-Whitney U:")
            for k in ("edge_retention_pct", "worst_od_dilation", "advantage_ratio_vs_matched_btw"):
                v = country_cmp.get(k, {})
                if "p_value" in v:
                    print(f"    {k}: Iran median={v['iran_median']:.3f}, "
                          f"Intl median={v['international_median']:.3f}, p={v['p_value']:.3f}")
        else:
            print(f"[stats] Iran-vs-international comparison skipped: {country_cmp.get('reason')}")

    clean_results = {}
    for (m, t), v in results.items():
        vv = dict(v)
        vv["metrics"] = {k: val for k, val in v["metrics"].items() if k != "collected_pairs"}
        clean_results[f"{m}|{'' if t is None else t}"] = vv

    clean_city_results = []
    for c in city_results:
        cc = dict(c)
        cc["methods"] = {m: {kk: (vv if kk != "metrics" else
                                  {k2: v2 for k2, v2 in vv.items() if k2 != "collected_pairs"})
                             for kk, vv in md.items()} for m, md in c["methods"].items()}
        clean_city_results.append(cc)

    report = {
        "framework": "Delay-Bounded Time-Varying Geometric t-Spanner",
        "version": "2.1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "formulation": ("for all u,v in V, for all tau in {0..23}: "
                        "dist_H(u,v,tau) <= t * dist_G(u,v,tau)"),
        "certificate_principle": ("Theorem 1: the edge-wise invariant "
                                  "dist_H(u,v,tau) <= t*w((u,v),tau) implies the "
                                  "all-pairs, all-hours guarantee."),
        "configuration": {
            "center": [args.lat, args.lon], "dist_m": dist, "grid_n": grid_n, "hours": hours,
            "t_values": t_values, "t_focus": t_focus, "edge_order": args.order,
            "seed": args.seed, "repeats": repeats, "sources": n_src, "targets_per_source": n_tgt,
            "junction_contraction": not args.no_contract, "bootstrap_samples": BOOTSTRAP_N,
            "quick": bool(args.quick), "crosscity_skipped": bool(args.skip_crosscity),
        },
        "network": summary,
        "peak_hour": worst_hour,
        "results": clean_results,
        "matched_density_budget_edges": int(budget),
        "significance_test_pair_level": test_out,
        "time_dependent_validation": {
            "delay_bounded": td_ours, "static_freeflow": td_static,
            "note": ("FIFO piecewise-linear interpolation of the hourly profiles; a TD "
                     "dilation at or below t empirically confirms Proposition 2."),
        },
        "ordering_ablation": ablation,
        "scaling_study": scaling,
        "cross_city": {
            "cities": clean_city_results,
            "morphology_regression": regression,
            "city_level_significance": city_sig,
            "iran_vs_international_comparison": country_cmp,
            "yao_cone_definition": ("Network-constrained adaptation of the classical Yao graph "
                                    "(k=8 angular cones per node, cheapest free-flow edge per cone); "
                                    "NOT the unconstrained-point-set construction with the closed-form "
                                    "stretch bound 1/(1-2 sin(pi/k))."),
        },
        "congestion_sensitivity": sensitivity,
        "known_limitations": [
            "Snapshot (time-slice) model: no departure-time evolution within a single "
            "traversal; validated ex post by the FIFO time-dependent experiment.",
            "The diurnal congestion field is a PARAMETRIC model (tidal Gaussian mixture, "
            "CBD-weighted), not fitted to floating-car or loop-detector telemetry; "
            "Section congestion_sensitivity quantifies robustness to this choice but does "
            "not substitute for a telemetry-fitted replication.",
            "Cities whose live OSMnx/Overpass fetch is unavailable (e.g. an offline sandbox) "
            "fall back to a deterministic, morphology-parameterized synthetic generator; each "
            "city's `used_real_osm` flag in `cross_city.cities` states which was used for that run.",
            f"n={len(CITY_REGISTRY)} cities across "
            f"{len({c['country'] for c in CITY_REGISTRY})} countries supports a directional "
            "cross-morphology, cross-country generalization claim (see morphology_regression, "
            "city_level_significance and iran_vs_international_comparison), not a definitive "
            "one; a larger, telemetry-fitted replication across more countries and continents "
            "remains future work.",
            "Greedy construction gives no approximation guarantee on |H|; the matched-density "
            "and Yao-cone controls are the empirical (not theoretical) lower-bound proxies used here.",
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    print(f"[report] wrote {REPORT_PATH}")

    print("\n" + table + "\n")
    print(abl_table + "\n")
    if crosscity_table:
        print(crosscity_table + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
