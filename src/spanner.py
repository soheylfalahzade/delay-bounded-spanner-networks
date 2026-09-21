"""Delay-Bounded Greedy Spanner.

Given G = (V, E) with per-hour edge costs w(e, tau), construct H subset of E with

    forall u,v in V, forall tau in {0..23}:  dist_H(u,v,tau) <= t * dist_G(u,v,tau)

Correctness (Theorem 1, see README): it suffices to enforce the *edge-wise*
invariant  dist_H(u,v,tau) <= t * w((u,v), tau)  for every (u,v) in E and every
tau. Concatenating the invariant along a snapshot-optimal path lifts it to all
pairs, because each snapshot is a static metric.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from src.config import EPS, HOURS
from src.temporal import INF, bounded_dijkstra


def _edge_order(G: nx.DiGraph,
                mode: str,
                attr: str,
                hours: Sequence[int],
                seed: int = 42) -> List[Tuple[object, object, dict]]:
    """Candidate ordering. Classical greedy = ascending cost; the centrality
    variant biases structurally important corridors to be tested early."""
    edges = list(G.edges(data=True))

    if mode == "temporal_mean":
        keyed = [((sum(d[attr][h] for h in hours) / len(hours), str(u), str(v)), (u, v, d))
                 for u, v, d in edges]
    elif mode == "centrality":
        k = min(64, G.number_of_nodes())
        cent = nx.edge_betweenness_centrality(G, k=k, weight="free_flow", seed=seed)
        keyed = []
        for u, v, d in edges:
            c = cent.get((u, v), 0.0)
            score = max(d[attr][h] for h in hours) / (1.0 + 6.0 * c)
            keyed.append(((score, str(u), str(v)), (u, v, d)))
    else:  # "temporal_max" (default)
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
                                 label: str = "delay-bounded") -> nx.DiGraph:
    """Construct the certified delay-bounded t-spanner H of G."""
    hours = list(HOURS) if hours is None else list(hours)
    t = float(t)
    if t < 1.0:
        raise ValueError("Dilation parameter t must be >= 1.")

    t0 = time.perf_counter()
    ordered = _edge_order(G, order, attr, hours)

    adj = {n: {} for n in G.nodes()}
    kept: List[Tuple[object, object, dict]] = []
    probes = 0

    iterator = ordered
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(ordered, desc=f"[{label}] t={t:.2f}", unit="edge", leave=False)
        except Exception:
            pass

    for u, v, data in iterator:
        costs = data[attr]
        # Tightest budget first: cheapest snapshot is the likeliest to violate,
        # so the loop short-circuits after ~1 probe for edges that get kept.
        slice_order = sorted(hours, key=lambda tau: costs[tau])
        must_add = False
        for tau in slice_order:
            budget = t * costs[tau]
            probes += 1
            if bounded_dijkstra(adj, u, v, tau, budget) > budget + EPS:
                must_add = True
                break
        if must_add:
            adj[u][v] = costs
            kept.append((u, v, data))

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in kept:
        H.add_edge(u, v, **data)

    elapsed = time.perf_counter() - t0
    H.graph.update(G.graph)
    H.graph.update({
        "method": label,
        "t": t,
        "hours": hours,
        "cost_attr": attr,
        "order": order,
        "build_seconds": elapsed,
        "dijkstra_probes": probes,
        "edges_parent": G.number_of_edges(),
    })
    return H


def certify_edge_condition(G: nx.DiGraph,
                           H: nx.DiGraph,
                           t: float,
                           hours: Optional[Iterable[int]] = None,
                           attr: str = "tt") -> dict:
    """Exhaustive certificate check of the edge-wise invariant.

    Passing this check *proves* the all-pairs, all-hours guarantee via Theorem 1.
    """
    hours = list(HOURS) if hours is None else list(hours)
    adj = {n: {} for n in H.nodes()}
    for u, v, d in H.edges(data=True):
        adj[u][v] = d[attr]

    violations = 0
    worst = 0.0
    worst_hour = None
    for u, v, d in G.edges(data=True):
        costs = d[attr]
        for tau in hours:
            budget = t * costs[tau]
            dist = bounded_dijkstra(adj, u, v, tau, budget)
            if dist > budget + EPS:
                violations += 1
                ratio = INF
                if ratio > worst:
                    worst = float("inf")
                    worst_hour = tau
                break
    return {
        "certified": violations == 0,
        "violating_edges": violations,
        "checked_edges": G.number_of_edges(),
        "worst_violation_hour": worst_hour,
    }
