"""Time-sliced shortest-path primitives.

The network is modelled as a stack of 24 static snapshots G_tau = (V, E, w(., tau)).
All primitives here operate on a *single* snapshot, which is what makes the
delay-bounded greedy certificate provable (see README, Theorem 1).
"""
from __future__ import annotations

import heapq
from typing import Dict, Optional, Sequence

INF = float("inf")
EPS = 1e-9

Adjacency = Dict[object, Dict[object, Sequence[float]]]


def build_adjacency(G, attr: str = "tt") -> Adjacency:
    """Flatten a DiGraph into a plain dict-of-dicts adjacency of cost vectors."""
    adj: Adjacency = {n: {} for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        adj[u][v] = data[attr]
    return adj


def time_weight(tau: int, attr: str = "tt"):
    """Return a NetworkX-compatible weight callable for snapshot ``tau``."""
    def _w(u, v, data):
        return data[attr][tau]
    return _w


def bounded_dijkstra(adj: Adjacency,
                     source,
                     target,
                     tau: int,
                     budget: float) -> float:
    """Pruned single-pair Dijkstra on snapshot ``tau``.

    Explores only the ball of radius ``budget`` around ``source`` and stops as
    soon as ``target`` is settled. Returns ``inf`` when no path of cost
    <= ``budget`` exists. This is the ingredient that keeps the greedy spanner
    construction out of combinatorial blow-up: the budget is a *single edge*
    cost scaled by t, so each probe touches a tiny local neighbourhood.
    """
    if source == target:
        return 0.0
    out = adj.get(source)
    if not out:
        return INF

    dist = {source: 0.0}
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
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
                heapq.heappush(pq, (nd, v))
    return INF


def single_source_snapshot(G, source, tau: int,
                           targets: Optional[set] = None,
                           attr: str = "tt") -> Dict[object, float]:
    """Full single-source Dijkstra on snapshot ``tau`` (used for evaluation)."""
    dist = {source: 0.0}
    seen = {source: 0.0}
    pq = [(0.0, source)]
    remaining = set(targets) if targets is not None else None
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF) + EPS:
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
                heapq.heappush(pq, (nd, v))
    if targets is None:
        return dist
    return {t: dist[t] for t in targets if t in dist}
