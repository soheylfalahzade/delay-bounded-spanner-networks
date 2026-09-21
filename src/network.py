"""Urban road network acquisition + diurnal speed enrichment.

Primary source: OSMnx / Overpass.
Fallback     : high-fidelity synthetic arterial-collector-local grid, so the
               pipeline is fully reproducible offline or behind a firewall.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from src.config import (CENTER_POINT, DEFAULT_DIST, EVENING_PEAK, EVENING_SIGMA,
                        HOURS, MORNING_PEAK, MORNING_SIGMA, NETWORK_TYPE)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------------
# Road hierarchy: free-flow speed (km/h), congestion amplitude, functional tier
# --------------------------------------------------------------------------------
HIGHWAY_PROFILES: Dict[str, Tuple[float, float, str]] = {
    "motorway":       (100.0, 0.22, "arterial"),
    "motorway_link":  ( 70.0, 0.28, "arterial"),
    "trunk":          ( 85.0, 0.30, "arterial"),
    "trunk_link":     ( 60.0, 0.32, "arterial"),
    "primary":        ( 60.0, 0.48, "arterial"),
    "primary_link":   ( 45.0, 0.45, "arterial"),
    "secondary":      ( 50.0, 0.52, "collector"),
    "secondary_link": ( 40.0, 0.48, "collector"),
    "tertiary":       ( 42.0, 0.46, "collector"),
    "tertiary_link":  ( 35.0, 0.42, "collector"),
    "unclassified":   ( 32.0, 0.34, "local"),
    "residential":    ( 30.0, 0.30, "local"),
    "living_street":  ( 20.0, 0.22, "local"),
    "service":        ( 20.0, 0.20, "local"),
    "road":           ( 30.0, 0.32, "local"),
}
DEFAULT_PROFILE = (30.0, 0.32, "local")

ARTERIAL_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link",
                    "primary", "primary_link", "secondary", "secondary_link"}


# --------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def normalize_highway(value) -> str:
    """OSM `highway` tags may be a list; collapse to the dominant class."""
    if isinstance(value, (list, tuple, set)):
        for candidate in value:
            if candidate in HIGHWAY_PROFILES:
                return candidate
        value = next(iter(value), "residential")
    if not isinstance(value, str):
        return "residential"
    return value if value in HIGHWAY_PROFILES else "residential"


# --------------------------------------------------------------------------------
# Diurnal congestion model
# --------------------------------------------------------------------------------
def congestion_index(tau: float) -> float:
    """Normalised congestion in [0, 1] for hour ``tau``.

    Bimodal: a sharp morning commute peak and a broader, heavier evening peak,
    with a mild midday plateau and a free-flowing night trough.
    """
    morning = 0.95 * math.exp(-0.5 * ((tau - MORNING_PEAK) / MORNING_SIGMA) ** 2)
    evening = 1.00 * math.exp(-0.5 * ((tau - EVENING_PEAK) / EVENING_SIGMA) ** 2)
    midday = 0.30 * math.exp(-0.5 * ((tau - 12.5) / 3.0) ** 2)
    night = 0.04
    return float(min(1.0, morning + evening + midday + night))


CONGESTION_CURVE: List[float] = [congestion_index(h) for h in HOURS]


def speed_curve(free_flow_kmh: float, amplitude: float) -> List[float]:
    """v(e, tau) in km/h; speeds never collapse below 25% of free flow."""
    floor = 0.25 * free_flow_kmh
    return [max(floor, free_flow_kmh * (1.0 - amplitude * c)) for c in CONGESTION_CURVE]


# --------------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------------
def _multidigraph_to_digraph(M: "nx.MultiDiGraph") -> nx.DiGraph:
    """Collapse parallel edges, keeping the shortest; preserves one-way semantics."""
    D = nx.DiGraph()
    for n, data in M.nodes(data=True):
        D.add_node(n, x=float(data["x"]), y=float(data["y"]))
    for u, v, data in M.edges(data=True):
        if u == v:
            continue
        length = float(data.get("length", 0.0))
        if length <= 0.0:
            length = haversine_m(D.nodes[u]["y"], D.nodes[u]["x"],
                                 D.nodes[v]["y"], D.nodes[v]["x"])
        if length <= 0.0:
            continue
        payload = {
            "length": length,
            "highway": normalize_highway(data.get("highway")),
            "geometry": data.get("geometry", None),
        }
        if D.has_edge(u, v):
            if length < D[u][v]["length"]:
                D[u][v].update(payload)
        else:
            D.add_edge(u, v, **payload)
    return D


def _largest_strongly_connected(G: nx.DiGraph) -> nx.DiGraph:
    comps = list(nx.strongly_connected_components(G))
    if not comps:
        return G
    giant = max(comps, key=len)
    H = G.subgraph(giant).copy()
    H.graph.update(G.graph)
    return H


def fetch_osm_network(center=CENTER_POINT, dist: int = DEFAULT_DIST) -> Optional[nx.DiGraph]:
    """Try Overpass via OSMnx. Returns None on *any* failure (timeout, DNS, API)."""
    try:
        import osmnx as ox
    except Exception as exc:                                   # pragma: no cover
        print(f"[network] osmnx unavailable ({exc}).")
        return None

    try:
        # OSMnx >= 1.2 uses the `settings` module (ox.config() was removed in 2.x)
        for key, value in (("use_cache", True), ("log_console", False),
                           ("requests_timeout", 90), ("overpass_rate_limit", True)):
            if hasattr(ox.settings, key):
                setattr(ox.settings, key, value)
    except Exception:
        pass

    try:
        print(f"[network] Querying Overpass for {center} r={dist} m ...")
        M = ox.graph_from_point(center, dist=dist,
                                network_type=NETWORK_TYPE, simplify=True)
        D = _multidigraph_to_digraph(M)
        D = _largest_strongly_connected(D)
        if D.number_of_edges() < 50:
            print("[network] Overpass response too sparse; using fallback.")
            return None
        D.graph["source"] = "osmnx"
        D.graph["center"] = list(center)
        D.graph["dist"] = dist
        print(f"[network] OSM network: {D.number_of_nodes()} nodes / "
              f"{D.number_of_edges()} directed edges.")
        return D
    except Exception as exc:
        print(f"[network] Overpass failed ({type(exc).__name__}: {exc}).")
        return None


def synthetic_arterial_grid(center=CENTER_POINT,
                            dist: int = DEFAULT_DIST,
                            n: int = 19) -> nx.DiGraph:
    """Deterministic arterial / collector / local grid with one-way pockets.

    Rows and columns are stratified so the hierarchy is realistic:
      * every 6th line  -> primary arterial
      * every 3rd line  -> secondary collector
      * everything else -> residential local street
    Two trunk diagonals emulate a ring/bypass corridor.
    """
    lat0, lon0 = center
    span = 2.0 * dist
    step = span / (n - 1)
    mlat = 111320.0
    mlon = 111320.0 * max(0.2, math.cos(math.radians(lat0)))

    G = nx.DiGraph()
    node_id = {}
    for i in range(n):
        for j in range(n):
            nid = i * n + j
            node_id[(i, j)] = nid
            lat = lat0 + (i - (n - 1) / 2.0) * step / mlat
            lon = lon0 + (j - (n - 1) / 2.0) * step / mlon
            G.add_node(nid, x=float(lon), y=float(lat))

    def line_class(idx: int) -> str:
        if idx % 6 == 0:
            return "primary"
        if idx % 3 == 0:
            return "secondary"
        return "residential"

    def link(a, b, hw):
        la, lo = G.nodes[a]["y"], G.nodes[a]["x"]
        lb, lob = G.nodes[b]["y"], G.nodes[b]["x"]
        length = haversine_m(la, lo, lb, lob)
        G.add_edge(a, b, length=length, highway=hw, geometry=None)

    for i in range(n):
        for j in range(n):
            a = node_id[(i, j)]
            if j + 1 < n:                       # horizontal segment, class of row i
                b = node_id[(i, j + 1)]
                hw = line_class(i)
                link(a, b, hw)
                if not (hw == "residential" and (i + j) % 7 == 0):
                    link(b, a, hw)              # one-way pockets on local streets
            if i + 1 < n:                       # vertical segment, class of column j
                b = node_id[(i + 1, j)]
                hw = line_class(j)
                link(a, b, hw)
                if not (hw == "residential" and (i * 3 + j) % 9 == 0):
                    link(b, a, hw)

    for k in range(n - 1):                      # trunk diagonals (bypass corridor)
        a, b = node_id[(k, k)], node_id[(k + 1, k + 1)]
        link(a, b, "trunk"); link(b, a, "trunk")
        c, d = node_id[(k, n - 1 - k)], node_id[(k + 1, n - 2 - k)]
        link(c, d, "trunk"); link(d, c, "trunk")

    G = _largest_strongly_connected(G)
    G.graph["source"] = "synthetic"
    G.graph["center"] = list(center)
    G.graph["dist"] = dist
    print(f"[network] Synthetic arterial-collector grid: {G.number_of_nodes()} nodes / "
          f"{G.number_of_edges()} directed edges.")
    return G


# --------------------------------------------------------------------------------
# Enrichment
# --------------------------------------------------------------------------------
def enrich_temporal_costs(G: nx.DiGraph) -> nx.DiGraph:
    """Attach v(e, tau) and w(e, tau) = length(e) / v(e, tau) to every edge."""
    for _, _, data in G.edges(data=True):
        hw = normalize_highway(data.get("highway"))
        free_flow_kmh, amplitude, tier = HIGHWAY_PROFILES.get(hw, DEFAULT_PROFILE)
        speeds = speed_curve(free_flow_kmh, amplitude)
        length = float(data["length"])
        tt = [length / (v * 1000.0 / 3600.0) for v in speeds]     # seconds
        data["highway"] = hw
        data["tier"] = tier
        data["speeds"] = speeds
        data["tt"] = tt
        data["free_flow"] = tt[0] if False else length / (free_flow_kmh * 1000.0 / 3600.0)
        data["tt_ff"] = [data["free_flow"]]        # single-slice view for the static baseline
        data["w_max"] = max(tt)
        data["w_min"] = min(tt)
        data["peak_ratio"] = max(tt) / min(tt)
    G.graph["hours"] = list(HOURS)
    G.graph["congestion_curve"] = list(CONGESTION_CURVE)
    return G


def build_network(center=CENTER_POINT, dist: int = DEFAULT_DIST,
                  force_synthetic: bool = False) -> nx.DiGraph:
    """Public entry point: resilient load + temporal enrichment."""
    G = None
    if not force_synthetic:
        G = fetch_osm_network(center, dist)
    if G is None:
        G = synthetic_arterial_grid(center, dist)
    return enrich_temporal_costs(G)


def network_summary(G: nx.DiGraph) -> dict:
    tiers: Dict[str, int] = {}
    total_len = 0.0
    for _, _, d in G.edges(data=True):
        tiers[d["tier"]] = tiers.get(d["tier"], 0) + 1
        total_len += d["length"]
    return {
        "source": G.graph.get("source", "unknown"),
        "center": G.graph.get("center"),
        "dist_m": G.graph.get("dist"),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "total_length_km": round(total_len / 1000.0, 3),
        "tier_counts": tiers,
        "mean_peak_ratio": round(
            sum(d["peak_ratio"] for _, _, d in G.edges(data=True)) / max(1, G.number_of_edges()), 4),
    }


if __name__ == "__main__":
    g = build_network()
    print(network_summary(g))
