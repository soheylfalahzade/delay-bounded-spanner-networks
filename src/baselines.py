"""Baselines: static metric spanner, hierarchical classifier, full network."""
from __future__ import annotations

import time
from typing import Optional, Set

import networkx as nx

from src.network import ARTERIAL_CLASSES
from src.spanner import delay_bounded_greedy_spanner


def full_network(G: nx.DiGraph) -> nx.DiGraph:
    """Ground truth: the unpruned network (dilation identically 1)."""
    H = G.copy()
    H.graph.update({"method": "full", "t": None, "build_seconds": 0.0,
                    "dijkstra_probes": 0, "edges_parent": G.number_of_edges()})
    return H


def static_metric_spanner(G: nx.DiGraph, t: float,
                          show_progress: bool = True) -> nx.DiGraph:
    """Classical greedy t-spanner on the *free-flow* metric only.

    Traffic-blind: it certifies dilation for the single free-flow snapshot and is
    therefore expected to breach the bound during the morning/evening peaks.
    """
    H = delay_bounded_greedy_spanner(
        G, t, hours=[0], attr="tt_ff", order="temporal_max",
        show_progress=show_progress, label="static")
    H.graph["method"] = "static"
    return H


def _bridge_isolated_nodes(G: nx.DiGraph, H: nx.DiGraph) -> None:
    """Give every node at least one cheap in- and out-edge (in place)."""
    for n in list(H.nodes()):
        if H.out_degree(n) == 0:
            cands = [(d["free_flow"], v) for v, d in G[n].items()]
            if cands:
                _, v = min(cands, key=lambda z: (z[0], str(z[1])))
                H.add_edge(n, v, **G[n][v])
        if H.in_degree(n) == 0:
            cands = [(G[u][n]["free_flow"], u) for u in G.predecessors(n)]
            if cands:
                _, u = min(cands, key=lambda z: (z[0], str(z[1])))
                H.add_edge(u, n, **G[u][n])


def augment_connectivity(G: nx.DiGraph, H: nx.DiGraph, max_rounds: int = 4000) -> nx.DiGraph:
    """Greedily re-attach strongly connected components with cheapest edges."""
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
            cost = d["free_flow"]
            if best is None or cost < best[0]:
                best = (cost, u, v, d)
        if best is None:
            break
        H.add_edge(best[1], best[2], **best[3])
    return H


def hierarchical_backbone(G: nx.DiGraph,
                          keep: Optional[Set[str]] = None,
                          show_progress: bool = False) -> nx.DiGraph:
    """Engineering heuristic: retain the functional road hierarchy only.

    Traffic-agnostic *and* geometry-agnostic: it has no dilation certificate at
    all, which is exactly the gap the proposed algorithm closes.
    """
    keep = ARTERIAL_CLASSES if keep is None else keep
    t0 = time.perf_counter()
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if d["highway"] in keep:
            H.add_edge(u, v, **d)
    H = augment_connectivity(G, H)
    H.graph.update(G.graph)
    H.graph.update({"method": "hierarchical", "t": None,
                    "build_seconds": time.perf_counter() - t0,
                    "dijkstra_probes": 0,
                    "edges_parent": G.number_of_edges()})
    return H
