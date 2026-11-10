#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact analysis over ALL disc faults of one radius.

Section 5l samples faults.  Here the family is the whole continuous family D(c, r) of discs of radius r, centre c anywhere.
The set of street segments removed by D(c, r) is {e : dist(c, e) <= r}; it is constant on the cells of the arrangement of
the offset regions N_r(e) = {c : dist(c, e) <= r}, one region per segment.  So the continuous family contains only finitely
many distinct fault scenarios (the distinct edge sets of the cells), and everything can be decided exactly:

  1. Failure.  Does the delay-bounded spanner H built for the intact network satisfy the delay bound in G-F for every
     fault of the family?  Where it fails, how large is the area of failing centres, and what is the worst fault?
  2. Repair.  Starting from H, every violating arc of every scenario is added; because adding arcs never breaks a satisfied
     pair, one pass over the scenarios yields a spanner H* that is a t-spanner of G-F at every hour for EVERY disc of radius r
     (up to a set of centres of measure zero, namely discs tangent to a street polyline).  H* is re-verified from scratch.
  3. Lower bound.  The arcs forced in some G-F (detour ratio above t in G-F) must be in every edge-subset spanner that is
     fault tolerant for the family; their union is therefore a lower bound on the size of any such spanner.

Graceful degradation.  With --t-fault the stretch allowed AFTER a fault may exceed the nominal stretch t (H-F must be a
t_fault-spanner of G-F, with t_fault >= t); t_fault = t is the strict definition above.  The curves of retention and of
the lower bound against t_fault show how much slack makes tolerance non-trivial.

Exactness is up to the polygonal approximation of the offset regions (relative error of the radius below 1e-3 with
quad_segs=32) and to the model of section 5l (a segment touches the disc if its stored polyline does).  The weights of the
area statistics assume the fault centre is uniform over the filled footprint of the offset regions.

    python benchmarks/all_discs.py --cities Yazd Isfahan Manhattan Tokyo --t 2
    python benchmarks/all_discs.py --synthetic --cities Yazd --radius-fracs 0.1        # offline smoke test
"""
from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import polygonize, unary_union
    from shapely.strtree import STRtree
    HAVE_SHAPELY = True
except ImportError:  # pragma: no cover
    HAVE_SHAPELY = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import region_fault as rf  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
H24 = list(range(24))
EPS = core.EPS
INF = float("inf")


# ------------------------------------------------------------ indexed network
class Net:
    """Arcs indexed 0..m-1 in the order of G.edges(); adjacency stores (head, arc id)."""

    def __init__(self, G):
        self.edges = list(G.edges(data=True))
        self.m = len(self.edges)
        self.u = [e[0] for e in self.edges]
        self.v = [e[1] for e in self.edges]
        self.tt = [e[2]["tt"] for e in self.edges]
        self.id = {(a, b): i for i, (a, b, _) in enumerate(self.edges)}
        self.nodes = list(G.nodes())

    def adjacency(self, arc_ids) -> Dict:
        adj = {n: [] for n in self.nodes}
        for i in arc_ids:
            adj[self.u[i]].append((self.v[i], i))
        return adj


def search(adj, TT, src, dst, tau: int, budget: float, blocked, banned: int = -1):
    """A path src -> dst of length <= budget (+EPS) that avoids the arcs in `blocked` and `banned`; list of arc ids or None."""
    if src == dst:
        return []
    lim = budget + EPS
    dist = {src: 0.0}
    par = {}
    pq = [(0.0, 0, src)]
    c = 1
    while pq:
        d, _, x = heapq.heappop(pq)
        if d > dist.get(x, INF):
            continue
        if x == dst:
            path = []
            while x != src:
                x, e = par[x]
                path.append(e)
            return path
        for y, e in adj.get(x, ()):
            if e == banned or e in blocked:
                continue
            nd = d + TT[e][tau]
            if nd > lim:
                continue
            if nd < dist.get(y, INF):
                dist[y] = nd
                par[y] = (x, e)
                heapq.heappush(pq, (nd, c, y))
                c += 1
    return None


# -------------------------------------------------------- scenario enumeration
def polyline_with_endpoints(G, frame, u, v):
    nodes, polys = frame
    p = list(polys[(u, v)])
    if math.hypot(p[0][0] - nodes[u][0], p[0][1] - nodes[u][1]) > 1e-6:
        p.insert(0, nodes[u])
    if math.hypot(p[-1][0] - nodes[v][0], p[-1][1] - nodes[v][1]) > 1e-6:
        p.append(nodes[v])
    return p


def exact_set(net: Net, polys: List, cx: float, cy: float, r: float, candidates) -> FrozenSet[int]:
    return frozenset(i for i in candidates if rf.polyline_touches_disc(polys[i], cx, cy, r))


def enumerate_scenarios(G, net: Net, r: float, quad_segs: int = 32, max_faces: int = 2000000):
    """Distinct removed-arc sets of all discs of radius r, with the area of centres that realise each.
    Returns (dict frozenset -> area, number of faces, number of distinct sets, polylines)."""
    if not HAVE_SHAPELY:
        raise RuntimeError("shapely is required (it is installed together with osmnx): pip install shapely")
    frame = rf.local_frame(G)
    polys = [polyline_with_endpoints(G, frame, net.u[i], net.v[i]) for i in range(net.m)]
    geoms = [LineString(p) if len(p) > 1 else Point(p[0]) for p in polys]
    bufs = [g.buffer(r, quad_segs=quad_segs) for g in geoms]
    lines = unary_union([b.boundary for b in bufs])
    faces = list(polygonize(lines.geoms if hasattr(lines, "geoms") else [lines]))
    if len(faces) > max_faces:
        raise RuntimeError(f"{len(faces)} faces exceed --max-faces {max_faces}")
    tree = STRtree(bufs)
    area: Dict[FrozenSet[int], float] = {}
    for f in faces:
        p = f.representative_point()
        cand = tree.query(p, predicate="intersects")
        S = exact_set(net, polys, p.x, p.y, r, cand)
        area[S] = area.get(S, 0.0) + f.area
    return area, len(faces), len(area), polys


# ---------------------------------------------------- failure, repair, verification
class SpannerState:
    """Spanner as a set of arc ids with adjacency and the witnesses (paths in H) of every arc outside H."""

    def __init__(self, net: Net, H_ids: Set[int], t: float, t_fault: float = None):
        self.net, self.t = net, t
        self.t_fault = t if t_fault is None else t_fault
        assert self.t_fault >= t - 1e-12, "the stretch allowed after a fault cannot be below the nominal stretch"
        self.H = set(H_ids)
        self.adj = net.adjacency(self.H)
        self.index: Dict[int, List[Tuple[int, int]]] = {}
        for g in range(net.m):
            if g in self.H:
                continue
            for tau in H24:
                b = t * net.tt[g][tau]
                path = search(self.adj, net.tt, net.u[g], net.v[g], tau, b, frozenset())
                assert path is not None, "the intact-network spanner is not certified: bug"
                for h in path:
                    self.index.setdefault(h, []).append((g, tau))

    def add(self, g: int) -> None:
        if g not in self.H:
            self.H.add(g)
            self.adj[self.net.u[g]].append((self.net.v[g], g))

    def violations(self, S: FrozenSet[int]) -> Set[int]:
        """Arcs of G-S that lack a detour within t * w in H-S at some hour."""
        net, t = self.net, self.t_fault
        todo = set()
        for h in S:
            lst = self.index.get(h)
            if lst:
                todo.update(lst)
        bad: Set[int] = set()
        for g, tau in todo:
            if g in S or g in self.H or g in bad:
                continue
            b = t * net.tt[g][tau]
            if search(self.adj, net.tt, net.u[g], net.v[g], tau, b, S) is None:
                bad.add(g)
        return bad


def run_family(net: Net, H_ids: Set[int], scenarios: Dict[FrozenSet[int], float], t: float, repair: bool,
               t_fault: float = None):
    st = SpannerState(net, H_ids, t, t_fault)
    total = sum(scenarios.values())
    failing = 0
    area_fail = 0.0
    counts: List[int] = []
    worst = (0, None)
    for S, a in scenarios.items():
        bad = st.violations(S)
        counts.append(len(bad))
        if bad:
            failing += 1
            area_fail += a
            if len(bad) > worst[0]:
                worst = (len(bad), S)
            if repair:
                for g in bad:
                    st.add(g)
    stats = dict(scenarios=len(scenarios), failing_scenarios=failing,
                 failing_scenarios_pct=100.0 * failing / max(1, len(scenarios)),
                 failing_area_pct=100.0 * area_fail / total if total else float("nan"),
                 worst_violating_arcs=worst[0],
                 median_violating_arcs_among_failing=float(np.median([c for c in counts if c])) if failing else 0.0)
    return stats, st, worst[1]


# ------------------------------------------------------------------ lower bound
def forced_union(net: Net, scenarios, t: float, t_fault: float = None) -> Tuple[Set[int], Set[int]]:
    """(arcs forced at the nominal stretch t, union of those and the arcs forced at the post-fault stretch t_fault in some
    G-S).  Detour witnesses in G avoid re-searching: an arc whose witness path survives S keeps its detour, so only pairs
    whose witness is hit are searched again."""
    tf = t if t_fault is None else t_fault
    adj = net.adjacency(range(net.m))
    nominal: Set[int] = set()
    for e in range(net.m):
        for tau in H24:
            b = t * net.tt[e][tau]
            if search(adj, net.tt, net.u[e], net.v[e], tau, b, frozenset(), banned=e) is None:
                nominal.add(e)
                break
    union = set(nominal)
    index: Dict[int, List[Tuple[int, int]]] = {}
    for e in range(net.m):
        wit = []
        for tau in H24:
            b = tf * net.tt[e][tau]
            path = search(adj, net.tt, net.u[e], net.v[e], tau, b, frozenset(), banned=e)
            if path is None:
                union.add(e)                      # forced at t_fault even without a fault
                wit = []
                break
            wit.append((tau, path))
        for tau, path in wit:
            for h in path:
                index.setdefault(h, []).append((e, tau))
    for S in scenarios:
        todo = set()
        for h in S:
            lst = index.get(h)
            if lst:
                todo.update(lst)
        for e, tau in todo:
            if e in S or e in union:
                continue
            b = tf * net.tt[e][tau]
            if search(adj, net.tt, net.u[e], net.v[e], tau, b, S, banned=e) is None:
                union.add(e)
    return nominal, union


# ------------------------------------------------------------------------ driver
def run_city(spec, t: float, radius_fracs: Sequence[float], synthetic: bool, grid_n: int, quad_segs: int,
             max_faces: int, t_faults: Sequence[float] = (None,), sink=None, done=frozenset()) -> List[Dict]:
    """`sink(rows)` is called with the new rows of each (radius, post-fault stretch) as soon as they exist, so that a long run
    that is interrupted keeps what it has computed."""
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return []
    net = Net(G)
    H0 = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    H0_ids = {net.id[e] for e in H0.edges()}
    dist = float(G.graph.get("dist", core.DEFAULT_DIST))
    rows = []
    for frac in radius_fracs:
        r = frac * dist
        todo = [tf for tf in t_faults
                if (spec["name"], round(r, 1), float(t), float(t if tf is None else tf)) not in done]
        if not todo:
            print(f"[done-already] {spec['name']} r={r:.1f} m", flush=True)
            continue
        t_faults_here = tuple(todo)
        t0 = time.time()
        print(f"[start] {spec['name']} r={r:.1f} m", flush=True)
        try:
            scen, n_faces, n_dist, _ = enumerate_scenarios(G, net, r, quad_segs, max_faces)
        except RuntimeError as exc:
            print(f"[skip] {spec['name']} r={r:.0f} m: {exc}", flush=True)
            continue
        t1 = time.time()
        for tf in t_faults_here:
            tfv = t if tf is None else float(tf)
            st0, _, _ = run_family(net, H0_ids, scen, t, repair=False, t_fault=tfv)
            st1, repaired, _ = run_family(net, H0_ids, scen, t, repair=True, t_fault=tfv)
            st2, _, _ = run_family(net, repaired.H, scen, t, repair=False, t_fault=tfv)
            assert st2["failing_scenarios"] == 0, "repaired spanner still violates a scenario: bug"
            nominal, union = forced_union(net, scen.keys(), t, tfv)
            Hs = repaired.H
            assert union <= Hs, "a fault-forced arc is missing from a certified spanner: bug"
            assert len(union) <= len(Hs) and len(nominal) <= len(union)
            m = net.m
            rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), arcs=m, t=t, t_fault=tfv,
                             radius_m=round(r, 1), radius_frac=frac, faces=n_faces, distinct_scenarios=n_dist,
                             nominal_fails_scenarios_pct=st0["failing_scenarios_pct"],
                             nominal_fails_area_pct=st0["failing_area_pct"],
                             worst_violating_arcs=st0["worst_violating_arcs"],
                             nominal_retention_pct=100.0 * len(H0_ids) / m,
                             repaired_retention_pct=100.0 * len(Hs) / m,
                             added_arcs=len(Hs) - len(H0_ids),
                             nominal_forced_lb_pct=100.0 * len(nominal) / m,
                             family_forced_lb_pct=100.0 * len(union) / m,
                             verified_failing_scenarios=st2["failing_scenarios"],
                             seconds_enumeration=round(t1 - t0, 1), seconds_total=round(time.time() - t0, 1)))
            if sink is not None:
                sink(rows[-1:])
            print(f"[ok] {spec['name']} r={r:.0f} m t_fault={tfv:g}: {n_dist} scenarios, nominal fails {st0['failing_area_pct']:.1f}% "
                  f"of centres, +{len(Hs) - len(H0_ids)} arcs for a certified spanner", flush=True)
    return rows


def merge_rows(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Rows of `new` replace rows of `old` with the same (city, radius_m, t); other rows of `old` are kept."""
    old, new = old.copy(), new.copy()
    for d in (old, new):
        if "t_fault" not in d.columns:
            d["t_fault"] = d["t"]
    key = ["city", "radius_m", "t", "t_fault"]
    new_keys = set(map(tuple, new[key].round(3).to_numpy().tolist()))
    keep = old[[tuple(r) not in new_keys for r in old[key].round(3).to_numpy().tolist()]]
    return pd.concat([keep, new], ignore_index=True).sort_values(["city", "t", "t_fault", "radius_m"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--t-fault", type=float, nargs="+", default=None,
                    help="stretch allowed after a fault (>= t); several values give a degradation curve; default: same as t")
    ap.add_argument("--radius-fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--quad-segs", type=int, default=32)
    ap.add_argument("--max-faces", type=int, default=2000000)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="with --append: do not recompute (city, radius, t, t_fault) rows that are already in all_discs.csv")
    ap.add_argument("--append", action="store_true",
                    help="merge the new rows into an existing all_discs.csv instead of overwriting it")
    a = ap.parse_args()
    rows: List[Dict] = []
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "all_discs.csv")
    base = pd.read_csv(path) if (a.append and os.path.exists(path)) else None
    done = frozenset()
    if a.skip_done and base is not None:
        b = base.copy()
        if "t_fault" not in b.columns:
            b["t_fault"] = b["t"]
        done = frozenset((r.city, round(float(r.radius_m), 1), float(r.t), float(r.t_fault)) for r in b.itertuples())

    def sink(new_rows):
        nonlocal base
        new = pd.DataFrame(new_rows).round(4)
        base = new if base is None else merge_rows(base, new)
        base.round(4).to_csv(path, index=False)          # written after every result: an interruption loses nothing

    for spec in [c for c in core.CITY_REGISTRY if c["name"] in a.cities]:
        rows += run_city(spec, a.t, a.radius_fracs, a.synthetic, a.grid_n, a.quad_segs, a.max_faces,
                         tuple(a.t_fault) if a.t_fault else (None,), sink, done)
    if not rows:
        if done:
            print("every requested row is already in all_discs.csv; nothing to do")
            return
        sys.exit("no network available; nothing written")
    df = base
    with open(os.path.join(OUT, "all_discs.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df))
    print(md_table(df))


if __name__ == "__main__":
    main()
