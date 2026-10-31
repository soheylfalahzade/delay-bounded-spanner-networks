#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Literature-anchored validation harness for the delay-bounded spanner.

Sub-commands (python benchmarks/literature_benchmarks.py <cmd> --help):

  static     Farshi-Gudmundsson (ESA 2005 / WEA 2007) style protocol on planar
             point sets. With a constant 24-hour profile the delay-bounded
             construction must reduce to the classical path-greedy spanner
             (Althofer et al. 1993); checked edge-for-edge against an independent
             reference implementation. Also reports edges / lightness / max degree /
             hop diameter / exact stretch / crossings for Greedy, Yao and Theta graphs.
  general    General-graph protocol (Chimani-Stutzenstein style): Greedy vs
             Baswana-Sen (2k-1)-spanners on ER / BA / random-geometric graphs.
  theorem1   Exhaustive all-pairs, all-hours brute-force check of Theorem 1 on
             thousands of random small time-varying digraphs + negative controls,
             plus comparison with the certified "union of per-hour static greedy
             spanners" baseline.
  metrla     Real-telemetry benchmark: METR-LA sensor graph (Li et al., ICLR 2018)
             with measured hourly speeds. Needs METR-LA.csv (see --help).
  dimacs     Static road-network sanity check on a DIMACS-9 .gr file (cropped).
  all        static + general + theorem1 (no downloads needed).

All results are written as JSON/CSV/Markdown under results/benchmarks/.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_spanner_benchmark as core  # noqa: E402  (the repo's own pipeline)

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))
TOL = 1e-9


# ============================================================================
# generic helpers
# ============================================================================
def _save(name: str, obj) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))


def _md(df: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name.replace(".md", ".csv")), index=False)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))


def digraph_from_undirected(Gu: nx.Graph, weight: str = "weight") -> nx.DiGraph:
    """Symmetric digraph with a CONSTANT 24-hour profile (static reduction)."""
    G = nx.DiGraph()
    G.add_nodes_from(Gu.nodes())
    for u, v, d in Gu.edges(data=True):
        w = float(d[weight])
        for a, b in ((u, v), (v, u)):
            G.add_edge(a, b, tt=[w] * 24, length=w)
    return G


def csr_from_edges(n: int, edges: Sequence[Tuple[int, int, float]], directed: bool) -> csr_matrix:
    if not edges:
        return csr_matrix((n, n))
    r = [e[0] for e in edges]
    c = [e[1] for e in edges]
    w = [e[2] for e in edges]
    if not directed:
        r, c, w = r + c, c + r, w + w
    return csr_matrix((w, (r, c)), shape=(n, n))


def exact_stretch_undirected(n: int, spanner_edges, full_dist: np.ndarray) -> float:
    """max_{u != v} d_H(u,v) / d_G(u,v) (inf if H disconnects a connected pair)."""
    M = csr_from_edges(n, spanner_edges, directed=False)
    dH = sp_dijkstra(M, directed=False)
    mask = ~np.eye(n, dtype=bool) & np.isfinite(full_dist) & (full_dist > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dH[mask] / full_dist[mask]
    return float(np.max(ratio)) if ratio.size else 1.0


# ============================================================================
# planar point-set machinery (Farshi-Gudmundsson protocol)
# ============================================================================
def make_points(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.random((n, 2))
    if kind == "normal":
        return rng.normal(0.5, 0.15, (n, 2))
    if kind == "clustered":
        k = max(3, n // 25)
        centers = rng.random((k, 2))
        idx = rng.integers(0, k, n)
        return centers[idx] + rng.normal(0, 0.03, (n, 2))
    if kind == "perturbed_grid":
        s = int(math.ceil(math.sqrt(n)))
        g = np.array([(i / s, j / s) for i in range(s) for j in range(s)])[:n]
        return g + rng.normal(0, 0.15 / s, g.shape)
    raise ValueError(kind)


def complete_euclid(P: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    Gu = nx.Graph()
    Gu.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            Gu.add_edge(i, j, weight=float(D[i, j]))
    return Gu, D


def reference_path_greedy(Gu: nx.Graph, t: float) -> Set[Tuple[int, int]]:
    """Independent textbook path-greedy (Althofer et al. 1993): scan edges by
    increasing weight, add (u,v) iff d_H(u,v) > t*w. Uses networkx Dijkstra with a
    cutoff -- shares no code with run_spanner_benchmark.py."""
    H = nx.Graph()
    H.add_nodes_from(Gu.nodes())
    out: Set[Tuple[int, int]] = set()
    for u, v, d in sorted(Gu.edges(data=True), key=lambda e: (e[2]["weight"], e[0], e[1])):
        w = d["weight"]
        try:
            dist = nx.dijkstra_path_length(H, u, v, weight="weight") if nx.has_path(H, u, v) else math.inf
        except nx.NetworkXNoPath:
            dist = math.inf
        if dist > t * w + TOL:
            H.add_edge(u, v, weight=w)
            out.add((min(u, v), max(u, v)))
    return out


def yao_or_theta(P: np.ndarray, k: int, theta: bool) -> Set[Tuple[int, int]]:
    n = len(P)
    ang = 2 * math.pi / k
    E: Set[Tuple[int, int]] = set()
    for i in range(n):
        diff = P - P[i]
        dist = np.hypot(diff[:, 0], diff[:, 1])
        phi = np.mod(np.arctan2(diff[:, 1], diff[:, 0]), 2 * math.pi)
        cone = np.minimum((phi // ang).astype(int), k - 1)
        for c in range(k):
            m = (cone == c) & (np.arange(n) != i)
            if not m.any():
                continue
            idx = np.where(m)[0]
            if theta:
                bis = (c + 0.5) * ang
                proj = diff[idx, 0] * math.cos(bis) + diff[idx, 1] * math.sin(bis)
                j = int(idx[np.argmin(proj)])
            else:
                j = int(idx[np.argmin(dist[idx])])
            E.add((min(i, j), max(i, j)))
    return E


def count_crossings(P: np.ndarray, edges: Sequence[Tuple[int, int]]) -> int:
    if len(edges) < 2:
        return 0
    E = np.array(edges)
    A, B = P[E[:, 0]], P[E[:, 1]]

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    total = 0
    m = len(E)
    for s in range(0, m, 400):
        a1, b1 = A[s:s + 400, None, :], B[s:s + 400, None, :]
        a2, b2 = A[None, :, :], B[None, :, :]
        o1, o2 = orient(a1, b1, a2), orient(a1, b1, b2)
        o3, o4 = orient(a2, b2, a1), orient(a2, b2, b1)
        proper = (o1 * o2 < 0) & (o3 * o4 < 0)
        e1, e2 = E[s:s + 400, None, :], E[None, :, :]
        share = ((e1[..., 0] == e2[..., 0]) | (e1[..., 0] == e2[..., 1]) |
                 (e1[..., 1] == e2[..., 0]) | (e1[..., 1] == e2[..., 1]))
        idx_i = np.arange(s, min(s + 400, m))[:, None]
        upper = np.arange(m)[None, :] > idx_i
        total += int((proper & ~share & upper).sum())
    return total


def spanner_metrics(P: np.ndarray, D: np.ndarray, edges: Set[Tuple[int, int]]) -> Dict:
    n = len(P)
    el = sorted(edges)
    w = [(a, b, float(D[a, b])) for a, b in el]
    mst = minimum_spanning_tree(csr_from_edges(n, [(a, b, float(D[a, b])) for a in range(n) for b in range(a + 1, n)], False))
    deg = np.zeros(n, int)
    for a, b in el:
        deg[a] += 1
        deg[b] += 1
    hop = sp_dijkstra(csr_from_edges(n, [(a, b, 1.0) for a, b, _ in w], False), directed=False, unweighted=True)
    return {
        "edges": len(el),
        "lightness": sum(x[2] for x in w) / float(mst.sum()),
        "max_degree": int(deg.max()),
        "hop_diameter": float(np.max(hop[np.isfinite(hop)])),
        "stretch": exact_stretch_undirected(n, w, D),
        "crossings": count_crossings(P, el),
    }


def cmd_static(args) -> None:
    rng = np.random.default_rng(args.seed)
    rows, mismatches = [], []
    for kind in ("uniform", "normal", "clustered", "perturbed_grid"):
        for n in args.sizes:
            for rep in range(args.reps):
                P = make_points(kind, n, rng)
                Gu, D = complete_euclid(P)
                Gd = digraph_from_undirected(Gu)
                for t in args.t:
                    t0 = time.perf_counter()
                    Hd = core.delay_bounded_greedy_spanner(Gd, t, show_progress=False)
                    t_ours = time.perf_counter() - t0
                    ours = {(min(u, v), max(u, v)) for u, v in Hd.edges()}
                    sym_ok = all(Hd.has_edge(v, u) for u, v in Hd.edges())
                    ref = reference_path_greedy(Gu, t)
                    same = ours == ref
                    if not (same and sym_ok):
                        mismatches.append(dict(kind=kind, n=n, rep=rep, t=t, only_ours=len(ours - ref), only_ref=len(ref - ours)))
                    m = spanner_metrics(P, D, ours)
                    rows.append(dict(kind=kind, n=n, rep=rep, method="DelayBounded(const profile)", t=t, **m,
                                     matches_reference=bool(same and sym_ok), seconds=t_ours))
                    rows.append(dict(kind=kind, n=n, rep=rep, method="PathGreedy(reference)", t=t,
                                     **spanner_metrics(P, D, ref), matches_reference=True, seconds=float("nan")))
                for k in args.cones:
                    for theta in (False, True):
                        E = yao_or_theta(P, k, theta)
                        m = spanner_metrics(P, D, E)
                        bound = None
                        if not theta and k > 6:
                            bound = 1.0 / (1.0 - 2.0 * math.sin(math.pi / k))
                        if theta and k >= 9:
                            th = 2 * math.pi / k
                            bound = 1.0 / (math.cos(th) - math.sin(th))
                        rows.append(dict(kind=kind, n=n, rep=rep, method=f"{'Theta' if theta else 'Yao'}-{k}", t=float("nan"),
                                         **m, theoretical_stretch_bound=bound,
                                         bound_respected=(None if bound is None else bool(m["stretch"] <= bound + 1e-9))))
    df = pd.DataFrame(rows)
    agg = (df.groupby(["kind", "n", "method", "t"], dropna=False)
             [["edges", "lightness", "max_degree", "hop_diameter", "stretch", "crossings"]].mean().reset_index())
    _md(agg.round(3), "static_reduction_table.md")
    df.to_csv(os.path.join(OUT, "static_reduction_raw.csv"), index=False)
    summary = {
        "instances_checked_against_reference": int((df.method == "DelayBounded(const profile)").sum()),
        "mismatches": mismatches,
        "stretch_within_t": bool((df[df.method == "DelayBounded(const profile)"].eval("stretch <= t + 1e-9")).all()),
        "yao_theta_bound_violations": int((df.get("bound_respected") == False).sum()),  # noqa: E712
    }
    _save("static_reduction_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if mismatches or not summary["stretch_within_t"] or summary["yao_theta_bound_violations"]:
        sys.exit(1)


# ============================================================================
# Baswana-Sen randomized (2k-1)-spanner (weighted, undirected)
# ============================================================================
def baswana_sen(Gu: nx.Graph, k: int, seed: int = 0) -> Set[Tuple[int, int]]:
    rng = random.Random(seed)
    nodes = list(Gu.nodes())
    n = len(nodes)
    p = n ** (-1.0 / k)
    adj = {v: {u: float(d["weight"]) for u, d in Gu[v].items()} for v in nodes}
    cluster: Dict[int, Optional[int]] = {v: v for v in nodes}
    S: Set[Tuple[int, int]] = set()

    def add(a, b):
        S.add((min(a, b), max(a, b)))

    def drop(a, b):
        adj[a].pop(b, None)
        adj[b].pop(a, None)

    for _ in range(k - 1):
        snap = {v: dict(adj[v]) for v in nodes}          # decisions use phase-start state
        centers = {c for c in cluster.values() if c is not None}
        sampled = {c for c in centers if rng.random() < p}
        new_cluster: Dict[int, Optional[int]] = {}
        removals: List[Tuple[int, int]] = []
        for v in nodes:
            c = cluster[v]
            if c is None:
                new_cluster[v] = None
                continue
            if c in sampled:
                new_cluster[v] = c
                continue
            best: Dict[int, Tuple[float, int]] = {}
            for u, w in snap[v].items():
                cu = cluster[u]
                if cu is None or cu == c:
                    continue
                if cu not in best or (w, u) < best[cu]:
                    best[cu] = (w, u)
            sam = {cu: b for cu, b in best.items() if cu in sampled}
            if not sam:
                for cu, (w, u) in best.items():
                    add(v, u)
                for u in snap[v]:
                    removals.append((v, u))
                new_cluster[v] = None
            else:
                cstar, (wstar, ustar) = min(sam.items(), key=lambda kv: kv[1])
                add(v, ustar)
                new_cluster[v] = cstar
                drop_cl = {cstar}
                for cu, (w, u) in best.items():
                    if cu != cstar and (w, u) < (wstar, ustar):
                        add(v, u)
                        drop_cl.add(cu)
                for u in snap[v]:
                    if cluster[u] in drop_cl:
                        removals.append((v, u))
        for a, b in removals:
            drop(a, b)
        cluster = new_cluster
        for v in nodes:                                   # remove intra-cluster edges
            for u in list(adj[v]):
                if cluster[v] is not None and cluster[v] == cluster.get(u):
                    drop(v, u)
    for v in nodes:                                       # vertex-cluster joining
        if cluster[v] is None:
            continue
        best = {}
        for u, w in adj[v].items():
            cu = cluster[u]
            if cu is None or cu == cluster[v]:
                continue
            if cu not in best or (w, u) < best[cu]:
                best[cu] = (w, u)
        for cu, (w, u) in best.items():
            add(v, u)
    return S


def random_graph(kind: str, n: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    if kind == "ER":
        G = nx.gnp_random_graph(n, 6.0 / n, seed=seed)
    elif kind == "BA":
        G = nx.barabasi_albert_graph(n, 4, seed=seed)
    elif kind == "RGG":
        G = nx.random_geometric_graph(n, math.sqrt(8.0 / (math.pi * n)), seed=seed)
    else:
        raise ValueError(kind)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(1.0, 10.0)
    return G


def cmd_general(args) -> None:
    rows = []
    for kind in ("ER", "BA", "RGG"):
        for n in args.sizes:
            for rep in range(args.reps):
                Gu = random_graph(kind, n, args.seed + rep)
                nn = Gu.number_of_nodes()
                full = sp_dijkstra(csr_from_edges(nn, [(u, v, d["weight"]) for u, v, d in Gu.edges(data=True)], False), directed=False)
                mst_w = sum(d["weight"] for _, _, d in nx.minimum_spanning_tree(Gu).edges(data=True))
                for kk in (2, 3):
                    t = 2 * kk - 1
                    t0 = time.perf_counter()
                    Hd = core.delay_bounded_greedy_spanner(digraph_from_undirected(Gu), t, show_progress=False)
                    sec = time.perf_counter() - t0
                    ours = {(min(u, v), max(u, v)) for u, v in Hd.edges()}
                    ref = reference_path_greedy(Gu, t)
                    bs = baswana_sen(Gu, kk, seed=args.seed + rep)
                    for name, E, s in (("DelayBounded(const)", ours, sec), ("PathGreedy(ref)", ref, float("nan")), (f"BaswanaSen-k{kk}", bs, float("nan"))):
                        wl = [(a, b, Gu[a][b]["weight"]) for a, b in E]
                        rows.append(dict(graph=kind, n=nn, m=Gu.number_of_edges(), t=t, method=name, rep=rep,
                                         edges=len(E), edge_frac=len(E) / Gu.number_of_edges(),
                                         lightness=sum(x[2] for x in wl) / mst_w,
                                         stretch=exact_stretch_undirected(nn, wl, full),
                                         same_as_reference=(E == ref) if name.startswith("Delay") else None))
    df = pd.DataFrame(rows)
    agg = df.groupby(["graph", "n", "t", "method"])[["edges", "edge_frac", "lightness", "stretch"]].mean().reset_index()
    _md(agg.round(3), "general_graph_table.md")
    df.to_csv(os.path.join(OUT, "general_graph_raw.csv"), index=False)
    bs_ok = bool((df[df.method.str.startswith("Baswana")].eval("stretch <= t + 1e-9")).all())
    greedy_ok = bool((df[~df.method.str.startswith("Baswana")].eval("stretch <= t + 1e-9")).all())
    same = bool(df[df.method.str.startswith("Delay")]["same_as_reference"].all())
    summary = dict(baswana_sen_stretch_ok=bs_ok, greedy_stretch_ok=greedy_ok, delay_bounded_equals_reference=same)
    _save("general_graph_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not (bs_ok and greedy_ok and same):
        sys.exit(1)


# ============================================================================
# time-varying graphs: exhaustive Theorem 1 check + certified baselines
# ============================================================================
def random_temporal_digraph(n: int, p: float, rng: random.Random) -> nx.DiGraph:
    G = nx.gnp_random_graph(n, p, seed=rng.randrange(10 ** 9), directed=True)
    comp = max(nx.strongly_connected_components(G), key=len)
    G = nx.convert_node_labels_to_integers(G.subgraph(comp).copy())
    for u, v in G.edges():
        base = rng.uniform(1.0, 10.0)
        a, b = rng.uniform(0.0, 1.5), rng.uniform(0, 24)
        G[u][v]["tt"] = [base * (1.0 + a * max(0.0, math.sin(2 * math.pi * (h - b) / 24.0))) for h in H24]
        G[u][v]["free_flow"] = base
        G[u][v]["length"] = base
    return G


def hour_matrix(G: nx.DiGraph, h: int, n: int) -> csr_matrix:
    return csr_from_edges(n, [(u, v, d["tt"][h]) for u, v, d in G.edges(data=True)], directed=True)


def brute_force_ok(G: nx.DiGraph, H: nx.DiGraph, t: float) -> bool:
    n = G.number_of_nodes()
    for h in H24:
        dG = sp_dijkstra(hour_matrix(G, h, n), directed=True)
        dH = sp_dijkstra(hour_matrix(H, h, n), directed=True)
        fin = np.isfinite(dG)
        if np.any(dH[fin] > t * dG[fin] + 1e-7):
            return False
    return True


def union_of_snapshots_spanner(G: nx.DiGraph, t: float) -> nx.DiGraph:
    """Certified baseline: union over tau of the classical greedy spanner of the
    snapshot G_tau. Adding edges only shortens distances, so the union is a t-spanner
    of every snapshot (hence also certified by Theorem 1)."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for h in H24:
        Hh = core.delay_bounded_greedy_spanner(G, t, hours=[h], show_progress=False, label=f"snap{h}")
        for u, v in Hh.edges():
            H.add_edge(u, v, **G[u][v])
    return H


def cmd_theorem1(args) -> None:
    rng = random.Random(args.seed)
    rows = []
    unsound = 0
    conserv_den = conserv_num = 0
    for i in range(args.instances):
        n = rng.randint(8, args.max_n)
        G = random_temporal_digraph(n, rng.uniform(0.15, 0.5), rng)
        if G.number_of_nodes() < 5:
            continue
        for t in (1.1, 1.5, 2.0, 3.0):
            H = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
            cert = core.temporal_stretch_certificate(G, H, t)["certified"]
            ok = brute_force_ok(G, H, t)
            if cert and not ok:
                unsound += 1
            Hu = union_of_snapshots_spanner(G, t)
            cert_u = core.temporal_stretch_certificate(G, Hu, t)["certified"]
            ok_u = brute_force_ok(G, Hu, t)
            if cert_u and not ok_u:
                unsound += 1
            # negative control: delete 1-3 random edges of H
            Hn = H.copy()
            el = list(Hn.edges())
            for e in rng.sample(el, min(len(el), rng.randint(1, 3))):
                Hn.remove_edge(*e)
            cert_n = core.temporal_stretch_certificate(G, Hn, t)["certified"]
            ok_n = brute_force_ok(G, Hn, t)
            if cert_n and not ok_n:
                unsound += 1
            if not cert_n:
                conserv_den += 1
                conserv_num += int(ok_n)
            rows.append(dict(inst=i, n=G.number_of_nodes(), m=G.number_of_edges(), t=t,
                             ours_edges=H.number_of_edges(), union_edges=Hu.number_of_edges(),
                             ours_cert=cert, ours_brute=ok, union_cert=cert_u, union_brute=ok_u,
                             neg_cert=cert_n, neg_brute=ok_n))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "theorem1_raw.csv"), index=False)
    agg = df.groupby("t").agg(instances=("inst", "count"), ours_edges=("ours_edges", "mean"),
                              union_of_snapshots_edges=("union_edges", "mean"), graph_edges=("m", "mean")).reset_index()
    agg["ours_vs_union_edge_ratio"] = agg.ours_edges / agg.union_of_snapshots_edges
    _md(agg.round(3), "theorem1_edges_vs_union_baseline.md")
    summary = dict(
        instances=int(len(df)),
        soundness_violations=int(unsound),
        ours_all_certified_and_bruteforce_ok=bool(df.ours_cert.all() and df.ours_brute.all()),
        union_all_certified_and_bruteforce_ok=bool(df.union_cert.all() and df.union_brute.all()),
        negative_controls_rejected_by_certificate=int((~df.neg_cert).sum()),
        negative_controls_total=int(len(df)),
        certificate_conservativeness=(conserv_num / conserv_den if conserv_den else None),
        note="conservativeness = P(all-pairs bound still holds | edge-wise certificate fails)",
    )
    _save("theorem1_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if unsound or not summary["ours_all_certified_and_bruteforce_ok"]:
        sys.exit(1)


# ============================================================================
# METR-LA (Li et al., ICLR 2018): real measured speeds on a real sensor graph
# ============================================================================
DCRNN_RAW = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"


def _fetch(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_metrla(speeds_csv: str, cache: str, kernel_thresh: float = 0.1, dataset: str = "metrla") -> nx.DiGraph:
    """DCRNN sensor graph + measured speeds. dataset in {"metrla", "pemsbay"}."""
    if dataset == "metrla":
        dist_name, ids_name = "distances_la_2012.csv", "graph_sensor_ids.txt"
    else:
        dist_name, ids_name = "distances_bay_2017.csv", None
    dist_path = _fetch(DCRNN_RAW + dist_name, os.path.join(cache, dist_name))
    if dataset == "metrla":
        ids_path = _fetch(DCRNN_RAW + ids_name, os.path.join(cache, ids_name))
        ids = open(ids_path).read().strip().split(",")
        dd = pd.read_csv(dist_path, dtype={"from": str, "to": str})
    else:
        dd = pd.read_csv(dist_path, header=None, names=["from", "to", "cost"], dtype={"from": str, "to": str})
        ids = list(dict.fromkeys(dd["from"].tolist()))
    sp = pd.read_csv(speeds_csv, index_col=0)
    sp.index = pd.to_datetime(sp.index)
    sp.columns = [str(c) for c in sp.columns]
    ids = [s_ for s_ in ids if s_ in sp.columns]
    idx = {s_: i for i, s_ in enumerate(ids)}
    n = len(ids)
    Dm = np.full((n, n), np.inf)
    for r in dd.itertuples(index=False):
        if r[0] in idx and r[1] in idx:
            Dm[idx[r[0]], idx[r[1]]] = r[2]
    fin = Dm[np.isfinite(Dm)]
    sigma = float(fin.std())
    A = np.exp(-np.square(Dm / sigma))
    A[A < kernel_thresh] = 0.0                       # DCRNN adjacency rule
    sp = sp[ids].replace(0.0, np.nan)
    wk = sp[sp.index.dayofweek < 5]
    prof = wk.groupby(wk.index.hour).median()          # 24 x n  (mph)
    prof = prof.apply(lambda c: c.fillna(c.median()))
    prof = prof.fillna(float(np.nanmedian(prof.values)))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    mph = 0.44704
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0 and np.isfinite(Dm[i, j]) and Dm[i, j] > 0:
                v = [0.5 * (prof.iloc[h, i] + prof.iloc[h, j]) * mph for h in H24]
                tt = [Dm[i, j] / max(s_, 0.5) for s_ in v]
                G.add_edge(i, j, tt=tt, free_flow=min(tt), length=float(Dm[i, j]),
                           tt_ff=[min(tt)], len_vec=[float(Dm[i, j])], tt_mean_vec=[float(np.mean(tt))])
    G.graph["hourly_mean_speed_mph"] = [float(prof.iloc[h].mean()) for h in H24]
    return G


def model_shape_vs_telemetry(mean_speed: Sequence[float]) -> Dict:
    """Shape-only comparison of the parametric congestion curve with measured speeds."""
    ms = np.array(mean_speed, float)
    ms = ms / ms.max()
    cong = core.edge_congestion_curve(0.5, 0.6, 0.5)
    mod = np.array(core.speed_curve(100.0, 0.35, cong), float)
    mod = mod / mod.max()
    r = float(np.corrcoef(ms, mod)[0, 1])
    return dict(pearson_r_normalised_hourly_speed=r,
                measured_peak_to_offpeak_slowdown=float(1.0 / ms.min()),
                model_peak_to_offpeak_slowdown=float(1.0 / mod.min()))


def cmd_metrla(args) -> None:
    G = load_metrla(args.speeds, os.path.join(OUT, "metrla_cache"), dataset=args.dataset)
    n, m = G.number_of_nodes(), G.number_of_edges()
    shape = model_shape_vs_telemetry(G.graph["hourly_mean_speed_mph"])
    rows = []
    for t in args.t:
        ours = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
        uni = union_of_snapshots_spanner(G, t)
        stat = core.static_freeflow_spanner(G, t, show_progress=False)
        mean = core.mean_temporal_spanner(G, t, show_progress=False)
        for name, H in (("DelayBounded", ours), ("UnionOfSnapshots", uni), ("StaticFreeFlow", stat), ("MeanTemporal", mean)):
            c = core.temporal_stretch_certificate(G, H, t)
            rows.append(dict(t=t, method=name, edges_kept_pct=100.0 * H.number_of_edges() / m,
                             certified=c["certified"], violating_edges=c["violating_edges"],
                             attained_worst_edge_stretch=c["attained_worst_edge_stretch"]))
    df = pd.DataFrame(rows)
    _md(df.round(3), f"{args.dataset}_table.md")
    _save(f"{args.dataset}_summary.json", dict(dataset=args.dataset, nodes=n, edges=m, congestion_shape=shape,
                                      source="Li et al., ICLR 2018 (DCRNN); sensor graph from liyaguang/DCRNN"))
    print(df.round(3).to_string(index=False))
    print(json.dumps(shape, indent=2))


# ============================================================================
# DIMACS-9 road networks (static)
# ============================================================================
def load_dimacs(path: str, max_nodes: int, seed: int) -> nx.Graph:
    op = gzip.open if path.endswith(".gz") else open
    Gf = nx.Graph()
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("a "):
                _, u, v, w = line.split()
                u, v, w = int(u), int(v), float(w)
                if u != v and (not Gf.has_edge(u, v) or Gf[u][v]["weight"] > w):
                    Gf.add_edge(u, v, weight=w)
    rng = random.Random(seed)
    s = rng.choice(list(Gf.nodes()))
    keep = list(nx.bfs_tree(Gf, s).nodes())[:max_nodes]
    H = Gf.subgraph(keep).copy()
    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    return nx.convert_node_labels_to_integers(H)


def cmd_dimacs(args) -> None:
    Gu = load_dimacs(args.gr, args.max_nodes, args.seed)
    nn = Gu.number_of_nodes()
    rng = random.Random(args.seed)
    src = rng.sample(range(nn), min(args.sources, nn))
    M = csr_from_edges(nn, [(u, v, d["weight"]) for u, v, d in Gu.edges(data=True)], False)
    full = sp_dijkstra(M, directed=False, indices=src)
    rows = []
    for t in args.t:
        Hd = core.delay_bounded_greedy_spanner(digraph_from_undirected(Gu), t, show_progress=False)
        E = {(min(u, v), max(u, v)) for u, v in Hd.edges()}
        Mh = csr_from_edges(nn, [(a, b, Gu[a][b]["weight"]) for a, b in E], False)
        dh = sp_dijkstra(Mh, directed=False, indices=src)
        mask = np.isfinite(full) & (full > 0)
        rows.append(dict(method="DelayBounded(const)", t=t, nodes=nn, edges=Gu.number_of_edges(), kept=len(E),
                         kept_pct=100.0 * len(E) / Gu.number_of_edges(),
                         sampled_worst_stretch=float(np.max(dh[mask] / full[mask]))))
    for kk in (2, 3):
        E = baswana_sen(Gu, kk, args.seed)
        Mh = csr_from_edges(nn, [(a, b, Gu[a][b]["weight"]) for a, b in E], False)
        dh = sp_dijkstra(Mh, directed=False, indices=src)
        mask = np.isfinite(full) & (full > 0)
        rows.append(dict(method=f"BaswanaSen-k{kk}", t=2 * kk - 1, nodes=nn, edges=Gu.number_of_edges(), kept=len(E),
                         kept_pct=100.0 * len(E) / Gu.number_of_edges(),
                         sampled_worst_stretch=float(np.max(dh[mask] / full[mask]))))
    df = pd.DataFrame(rows)
    _md(df.round(3), "dimacs_table.md")
    print(df.round(3).to_string(index=False))


# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static")
    s.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--t", type=float, nargs="+", default=[1.1, 1.25, 1.5, 2.0])
    s.add_argument("--cones", type=int, nargs="+", default=[9, 12])
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_static)

    g = sub.add_parser("general")
    g.add_argument("--sizes", type=int, nargs="+", default=[150, 300])
    g.add_argument("--reps", type=int, default=3)
    g.add_argument("--seed", type=int, default=11)
    g.set_defaults(fn=cmd_general)

    th = sub.add_parser("theorem1")
    th.add_argument("--instances", type=int, default=150)
    th.add_argument("--max-n", type=int, default=22)
    th.add_argument("--seed", type=int, default=3)
    th.set_defaults(fn=cmd_theorem1)

    ml = sub.add_parser("metrla", help="download METR-LA.csv from https://zenodo.org/records/5724362 first")
    ml.add_argument("--speeds", required=True, help="path to METR-LA.csv or PEMS-BAY.csv")
    ml.add_argument("--dataset", choices=["metrla", "pemsbay"], default="metrla")
    ml.add_argument("--t", type=float, nargs="+", default=[1.1, 1.3, 1.5, 2.0, 3.0])
    ml.set_defaults(fn=cmd_metrla)

    dm = sub.add_parser("dimacs", help="DIMACS-9 .gr / .gr.gz (e.g. USA-road-d.NY.gr.gz)")
    dm.add_argument("--gr", required=True)
    dm.add_argument("--max-nodes", type=int, default=3000)
    dm.add_argument("--sources", type=int, default=40)
    dm.add_argument("--t", type=float, nargs="+", default=[1.1, 1.5, 2.0, 3.0])
    dm.add_argument("--seed", type=int, default=5)
    dm.set_defaults(fn=cmd_dimacs)

    al = sub.add_parser("all")
    al.set_defaults(fn=None)

    args = ap.parse_args()
    if args.cmd == "all":
        for name, fn, extra in (("static", cmd_static, argparse.Namespace(sizes=[50, 100], reps=3, t=[1.1, 1.25, 1.5, 2.0], cones=[9, 12], seed=7)),
                                ("general", cmd_general, argparse.Namespace(sizes=[150, 300], reps=3, seed=11)),
                                ("theorem1", cmd_theorem1, argparse.Namespace(instances=150, max_n=22, seed=3))):
            print(f"\n=== {name} ===")
            fn(extra)
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
