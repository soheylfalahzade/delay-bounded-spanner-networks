#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why does every arc become forced under faults?  A graph-theoretic explanation.

Section 5v finds, in all four cities and at every disc radius from 5 m to 300 m, that the arcs forced in some G-F cover the
whole network, and the post-fault stretch sweep shows that this persists for large allowed stretch.  A fault of tiny radius
removes one street or one intersection, so the question is purely combinatorial: for an arc e = (u, v), is there ONE street
or ONE intersection whose removal leaves no u-to-v path at all in G-e?  If so, e must be in every fault-tolerant spanner
whatever the stretch (a stretch bound cannot help when no path exists).

For every arc e = (u, v) of the network (arcs with identical endpoints in the opposite direction are the two sides of one
street and are removed together):

  bridge        no directed path from u to v avoids e;
  vertex cut    some intersection w (not u or v) lies on every u-to-v path that avoids e;
  street cut    some street (an arc with its opposite arc) lies on every such path (and does not contain e);
  robust        none of the above: e keeps a path after any single street or intersection failure.

A shortest (hop) path P from u to v avoiding e is found first; a failure that leaves a path must miss P or leave another
path, so it suffices to try the interior vertices and the streets of P.

    python benchmarks/cut_structure.py --cities Yazd Isfahan Manhattan Tokyo
    python benchmarks/cut_structure.py --synthetic --cities Yazd              # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import retention_diagnostics as rd  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")


def bfs_path(adj, src, dst, blocked_arcs: Set[int], blocked_node=None) -> Optional[List[int]]:
    """Arc ids of a shortest-hop path src -> dst avoiding the blocked arcs and node; None if there is none."""
    if blocked_node is not None and blocked_node in (src, dst):
        return None
    par: Dict = {src: None}
    q = deque([src])
    while q:
        x = q.popleft()
        if x == dst:
            path = []
            while par[x] is not None:
                px, e = par[x]
                path.append(e)
                x = px
            return path[::-1]
        for y, e in adj[x]:
            if e in blocked_arcs or y in par or y == blocked_node:
                continue
            par[y] = (x, e)
            q.append(y)
    return None


def classify(G) -> Dict[Tuple, str]:
    arcs = list(G.edges())
    idx = {a: i for i, a in enumerate(arcs)}
    adj = {n: [] for n in G.nodes()}
    for i, (u, v) in enumerate(arcs):
        adj[u].append((v, i))
    twin = [idx.get((v, u)) for (u, v) in arcs]
    out: Dict[Tuple, str] = {}
    for e, (u, v) in enumerate(arcs):
        base = {e}
        P = bfs_path(adj, u, v, base)
        if P is None:
            out[(u, v)] = "bridge"
            continue
        cat = "robust"
        interior = [arcs[a][1] for a in P[:-1]]
        for w in interior:
            if bfs_path(adj, u, v, base, blocked_node=w) is None:
                cat = "vertex cut"
                break
        if cat == "robust":
            for a in P:
                if twin[a] == e:                       # the opposite arc of e itself: the same street as e
                    continue
                blocked = base | {a} | ({twin[a]} if twin[a] is not None else set())
                if bfs_path(adj, u, v, blocked) is None:
                    cat = "street cut"
                    break
        out[(u, v)] = cat
    return out


def city_row(G, name: str, t: float = 2.0) -> Dict:
    cats = classify(G)
    m = G.number_of_edges()
    cnt = pd.Series(list(cats.values())).value_counts()
    edf = rd.per_edge_max_ratio(G, t * 1.0001)
    forced = float((edf.max_ratio > t).mean()) * 100.0
    pct = lambda k: 100.0 * float(cnt.get(k, 0)) / m  # noqa: E731
    return dict(city=name, real_osm=bool(G.graph.get("used_real_osm")), arcs=m, bridge_pct=pct("bridge"),
                vertex_cut_pct=pct("vertex cut"), street_cut_pct=pct("street cut"), robust_pct=pct("robust"),
                in_some_single_failure_cut_pct=100.0 - pct("robust"), nominal_forced_pct=forced, t=t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rows = []
    for spec in [c for c in core.CITY_REGISTRY if c["name"] in a.cities]:
        G = load_city(spec, a.grid_n, a.synthetic)
        if G is not None:
            rows.append(city_row(G, spec["name"], a.t))
            print(f"[ok] {spec['name']}: robust {rows[-1]['robust_pct']:.1f}%", flush=True)
    if not rows:
        sys.exit("no network available; nothing written")
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.round(4).to_csv(os.path.join(OUT, "cut_structure.csv"), index=False)
    with open(os.path.join(OUT, "cut_structure.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df))
    print(md_table(df))


if __name__ == "__main__":
    main()
