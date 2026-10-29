#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network maps that show Theorem 2 on the cities themselves.

Every street segment is classified at stretch bound t:
  forced                r(e) > t   (no detour within ratio t exists; every t-spanner must keep it)
  kept, optional        in the delay-bounded spanner although r(e) <= t
  removed               dropped by the delay-bounded spanner

    python benchmarks/make_network_maps.py                       # Isfahan Tokyo Cairo Manhattan, t = 2
    python benchmarks/make_network_maps.py --cities Yazd Tehran --t 3
    python benchmarks/make_network_maps.py --synthetic --cities Yazd Tokyo   # offline smoke test
Output: results/figures/fig5_network_maps_t<t>.png / .pdf
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import run_spanner_benchmark as core  # noqa: E402
from retention_diagnostics import per_edge_max_ratio  # noqa: E402

COL = {"forced": "#1f3a93", "optional": "#17a398", "removed": "#d6402b"}
LW = {"forced": 1.1, "optional": 1.3, "removed": 1.3}


def classify(G, H, t):
    edf = per_edge_max_ratio(G, t * 1.0001)
    ratio = {(r.u, r.v): r.max_ratio for r in edf.itertuples()}
    cls = {}
    for (u, v), r in ratio.items():
        if not H.has_edge(u, v):
            cls[(u, v)] = "removed"
        elif r > t:
            cls[(u, v)] = "forced"
        else:
            cls[(u, v)] = "optional"
    return cls


def draw(ax, G, cls, title):
    for k in ("removed", "optional", "forced"):                    # forced drawn last
        edges = [e for e, c in cls.items() if c == k]
        if edges:
            ax.add_collection(LineCollection(core._edge_segments(G, edges), colors=COL[k], linewidths=LW[k],
                                             alpha=0.95 if k != "removed" else 0.9, capstyle="round", zorder={"removed": 3, "optional": 4, "forced": 2}[k]))
    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    lat0 = float(np.mean(ys))
    ax.set_aspect(1.0 / max(0.2, np.cos(np.radians(lat0))))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    n = len(cls)
    pct = {k: 100.0 * sum(1 for c in cls.values() if c == k) / n for k in COL}
    ax.set_title(f"{title}\nforced {pct['forced']:.1f}%  |  removed {pct['removed']:.1f}%", fontsize=10, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=["Isfahan", "Tokyo", "Cairo", "Manhattan"])
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--grid-n", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "figures"))
    a = ap.parse_args()

    specs = [c for c in core.CITY_REGISTRY if c["name"] in a.cities]
    specs.sort(key=lambda c: a.cities.index(c["name"]))
    if not specs:
        sys.exit("no matching cities")
    ncol = 2 if len(specs) > 1 else 1
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 6.3 * nrow + 0.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, spec in zip(axes.ravel(), specs):
        G = core.build_city_network(spec, grid_n=a.grid_n, force_synthetic=a.synthetic)
        H = core.delay_bounded_greedy_spanner(G, a.t, show_progress=False)
        cls = classify(G, H, a.t)
        ax.axis("on")
        draw(ax, G, cls, f"{spec['name']} ({spec['country']}, {spec['archetype']})")
        print("[ok]", spec["name"])
    h = [Line2D([], [], color=COL["forced"], lw=2.4, label="forced: detour ratio > t (kept by any t-spanner)"),
         Line2D([], [], color=COL["optional"], lw=2.4, label="kept by the construction, not forced"),
         Line2D([], [], color=COL["removed"], lw=2.4, label="removed by the construction")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9.5)
    fig.suptitle(f"Forced edges (Theorem 2) on city street networks at t = {a.t:g}", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"fig5_network_maps_t{a.t:g}")
    fig.savefig(stem + ".png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    print("saved", stem + ".png")


if __name__ == "__main__":
    main()
