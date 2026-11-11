#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget versus coverage: a non-vacuous notion of regional fault tolerance.

Sections 5v and 5x show that guaranteeing the delay bound for EVERY disc fault of a radius forces the whole network, so the
worst-case definition is vacuous on road networks.  The exact scenario enumeration of benchmarks/all_discs.py supports an
average-case alternative: the share of fault positions (area of disc centres) for which H-F is a t-spanner of G-F, as a
function of how many arcs are added to the intact-network spanner.

For each failing scenario S the arcs that would repair it are V(S) (arcs of G-S without a detour in H-S); a scenario is covered
once all of V(S) is in H.  Distinct repair sets are taken in decreasing order of (area / size) and added; a scenario is counted
as soon as its whole repair set is present.  The covered area of the curve is therefore a LOWER bound on the true coverage of
the spanner at that size (an added arc can also restore a detour for another violating arc, so some further scenarios may be
covered without their whole repair set); at the end of the curve it is exact (100%).  The order is a static greedy heuristic,
so the curve is a feasible trade-off, not an optimal one: "at least this much coverage with this many added arcs".

    python benchmarks/coverage_curve.py --cities Yazd Isfahan Manhattan Tokyo --radius-fracs 0.05 0.10
    python benchmarks/coverage_curve.py --synthetic --cities Yazd --radius-fracs 0.05         # offline smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, FrozenSet, List, Sequence, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
import all_discs as ad  # noqa: E402
from common import md_table, load_city  # noqa: E402

OUT = os.path.join(ROOT, "results", "benchmarks")
LEVELS = (50.0, 75.0, 90.0, 95.0, 99.0, 100.0)


def repair_sets(net, H_ids, scen, t: float, t_fault: float):
    """(area covered by the intact-network spanner, {repair set: area}, total area)."""
    st = ad.SpannerState(net, H_ids, t, t_fault)
    total = sum(scen.values())
    covered0 = 0.0
    by_v: Dict[FrozenSet[int], float] = {}
    for S, a in scen.items():
        V = frozenset(st.violations(S))
        if V:
            by_v[V] = by_v.get(V, 0.0) + a
        else:
            covered0 += a
    return covered0, by_v, total


def curve(covered0: float, by_v: Dict[FrozenSet[int], float], total: float):
    """List of (arcs added, covered area in percent), non-decreasing in both coordinates; also the chosen arcs in order."""
    items = sorted(by_v.items(), key=lambda kv: (-kv[1] / len(kv[0]), sorted(kv[0])))
    need = {V: len(V) for V in by_v}
    inv: Dict[int, List[FrozenSet[int]]] = {}
    for V in by_v:
        for g in V:
            inv.setdefault(g, []).append(V)
    chosen: List[int] = []
    seen = set()
    covered = covered0
    pts = [(0, 100.0 * covered / total)]
    for V, _ in items:
        new = [g for g in sorted(V) if g not in seen]
        if not new:
            continue
        for g in new:
            seen.add(g)
            chosen.append(g)
            for W in inv[g]:
                need[W] -= 1
                if need[W] == 0:
                    covered += by_v[W]
        pts.append((len(chosen), 100.0 * covered / total))
    return pts, chosen


def arcs_for_level(pts: Sequence[Tuple[int, float]], level: float) -> int:
    for k, c in pts:
        if c >= level - 1e-9:
            return k
    return pts[-1][0]


def run_city(spec, t: float, radius_fracs: Sequence[float], t_fault: float, synthetic: bool, grid_n: int,
             quad_segs: int, max_faces: int, sink=None, done=frozenset()) -> List[Dict]:
    """`sink(rows)` receives the rows of each radius as soon as they exist; radii whose (city, radius, t, t_fault) key is in
    `done` are skipped, so an interrupted run can be resumed."""
    G = load_city(spec, grid_n, synthetic)
    if G is None:
        return []
    net = ad.Net(G)
    H0 = core.delay_bounded_greedy_spanner(G, t, show_progress=False)
    H0_ids = {net.id[e] for e in H0.edges()}
    dist = float(G.graph.get("dist", core.DEFAULT_DIST))
    rows = []
    for frac in radius_fracs:
        r = frac * dist
        if (spec["name"], round(r, 1), float(t), float(t_fault)) in done:
            print(f"[done-already] {spec['name']} r={r:.1f} m", flush=True)
            continue
        print(f"[start] {spec['name']} r={r:.1f} m", flush=True)
        try:
            scen, _, n_dist, _ = ad.enumerate_scenarios(G, net, r, quad_segs, max_faces)
        except RuntimeError as exc:
            print(f"[skip] {spec['name']} r={r:.0f} m: {exc}", flush=True)
            continue
        covered0, by_v, total = repair_sets(net, H0_ids, scen, t, t_fault)
        pts, chosen = curve(covered0, by_v, total)
        # exact repair (arcs added scenario by scenario with the growing spanner) can need fewer arcs than the union
        _, repaired, _ = ad.run_family(net, H0_ids, scen, t, repair=True, t_fault=t_fault)
        full = len(repaired.H) - len(H0_ids)
        m = net.m
        new_rows = []
        for level in LEVELS:
            k = full if level >= 100.0 else min(arcs_for_level(pts, level), full)
            new_rows.append(dict(city=spec["name"], real_osm=bool(G.graph.get("used_real_osm")), arcs=m, t=t, t_fault=t_fault,
                             radius_m=round(r, 1), distinct_scenarios=n_dist, covered_by_intact_spanner_pct=pts[0][1],
                             coverage_level_pct=level, arcs_to_add=int(k), added_pct_of_arcs=100.0 * k / m,
                             retention_pct=100.0 * (len(H0_ids) + k) / m,
                             nominal_retention_pct=100.0 * len(H0_ids) / m))
        rows += new_rows
        if sink is not None:
            sink(new_rows)
        print(f"[ok] {spec['name']} r={r:.0f} m: intact spanner covers {pts[0][1]:.1f}%; 99% needs "
              f"{min(arcs_for_level(pts, 99.0), full)} arcs, 100% needs {full}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Yazd", "Isfahan", "Manhattan", "Tokyo"])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--t-fault", type=float, default=None)
    ap.add_argument("--radius-fracs", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--quad-segs", type=int, default=32)
    ap.add_argument("--max-faces", type=int, default=2000000)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--append", action="store_true", help="merge new rows into an existing coverage_curve.csv")
    ap.add_argument("--skip-done", action="store_true", help="with --append: skip radii already in the file")
    a = ap.parse_args()
    tf = a.t if a.t_fault is None else a.t_fault
    rows: List[Dict] = []
    path = os.path.join(OUT, "coverage_curve.csv")
    os.makedirs(OUT, exist_ok=True)
    base = pd.read_csv(path) if (a.append and os.path.exists(path)) else None
    done = frozenset()
    if a.skip_done and base is not None:
        done = frozenset((r.city, round(float(r.radius_m), 1), float(r.t), float(r.t_fault)) for r in base.itertuples())

    def sink(new_rows):
        nonlocal base
        new = pd.DataFrame(new_rows).round(4)
        if base is None:
            base = new
        else:
            key = ["city", "radius_m", "t", "t_fault", "coverage_level_pct"]
            kn = set(map(tuple, new[key].to_numpy().tolist()))
            keep = base[[tuple(r) not in kn for r in base[key].to_numpy().tolist()]]
            base = pd.concat([keep, new], ignore_index=True)
        base.to_csv(path, index=False)                       # written after every radius: an interruption loses nothing

    for spec in [c for c in core.CITY_REGISTRY if c["name"] in a.cities]:
        rows += run_city(spec, a.t, a.radius_fracs, tf, a.synthetic, a.grid_n, a.quad_segs, a.max_faces, sink, done)
    if not rows:
        if done:
            print("every requested row is already in coverage_curve.csv; nothing to do")
            return
        sys.exit("no network available; nothing written")
    df = base
    with open(os.path.join(OUT, "coverage_curve.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df))
    print(md_table(df))


if __name__ == "__main__":
    main()
