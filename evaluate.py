#!/usr/bin/env python3
"""Benchmark harness for the Delay-Bounded Time-Varying Geometric t-Spanner.

Usage:
    python evaluate.py                 # full run (Yazd centre, r = 1500 m)
    python evaluate.py --quick         # reduced footprint smoke test
    python evaluate.py --synthetic     # force the offline arterial grid
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baselines import full_network, hierarchical_backbone, static_metric_spanner
from src.config import (CENTER_POINT, DEFAULT_DIST, DILATION_CAP, FIGURE_PATH, HOURS,
                        N_SOURCES, N_TARGETS_PER_SOURCE, README_PATH, REPORT_PATH,
                        RESULTS_DIR, SEED, TABLE_PATH, T_FOCUS, T_VALUES)
from src.network import CONGESTION_CURVE, build_network, network_summary
from src.spanner import certify_edge_condition, delay_bounded_greedy_spanner
from src.temporal import single_source_snapshot

METHOD_STYLE = {
    "delay_bounded": ("Delay-Bounded Spanner (ours)", "#c0392b", "-", "o"),
    "static":        ("Static Metric Spanner",        "#2980b9", "--", "s"),
    "hierarchical":  ("Hierarchical Road Classifier", "#27ae60", "-.", "^"),
    "full":          ("Full Network (ground truth)",  "#7f8c8d", ":", "D"),
}


# ---------------------------------------------------------------------------------
# Sampling and metrics
# ---------------------------------------------------------------------------------
def sample_pairs(G: nx.DiGraph, n_sources: int, n_targets: int, seed: int) -> Dict:
    rng = random.Random(seed)
    nodes = sorted(G.nodes())
    sources = rng.sample(nodes, min(n_sources, len(nodes)))
    pairs = {}
    for s in sources:
        pool = [n for n in nodes if n != s]
        pairs[s] = rng.sample(pool, min(n_targets, len(pool)))
    return pairs


def baseline_distances(G: nx.DiGraph, pairs: Dict, hours: Sequence[int]) -> Dict:
    out = {}
    for tau in hours:
        for s, targets in pairs.items():
            out[(tau, s)] = single_source_snapshot(G, s, tau, set(targets))
    return out


def evaluate_graph(H: nx.DiGraph, pairs: Dict, hours: Sequence[int],
                   base: Dict) -> Dict:
    per_hour_max, per_hour_mean = [], []
    all_ratios, unreachable, breaches, total = [], 0, 0, 0
    t_bound = H.graph.get("t")

    for tau in hours:
        ratios = []
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
                    unreachable += 1
                    breaches += 1
                    ratios.append(DILATION_CAP)
                    continue
                r = dh / dg
                ratios.append(r)
                if t_bound is not None and r > t_bound + 1e-6:
                    breaches += 1
        arr = np.asarray(ratios, dtype=float) if ratios else np.asarray([1.0])
        per_hour_max.append(float(arr.max()))
        per_hour_mean.append(float(arr.mean()))
        all_ratios.extend(arr.tolist())

    allr = np.asarray(all_ratios, dtype=float)
    return {
        "worst_dilation": float(allr.max()),
        "mean_dilation": float(allr.mean()),
        "p95_dilation": float(np.percentile(allr, 95)),
        "per_hour_max": per_hour_max,
        "per_hour_mean": per_hour_mean,
        "pairs_evaluated": int(total),
        "unreachable_pairs": int(unreachable),
        "bound_breaches": int(breaches),
        "breach_rate": float(breaches / max(1, total)),
        "peak_hour_worst": int(int(np.argmax(per_hour_max))),
    }


def structural_metrics(G: nx.DiGraph, H: nx.DiGraph) -> Dict:
    e_g, e_h = G.number_of_edges(), H.number_of_edges()
    len_g = sum(d["length"] for _, _, d in G.edges(data=True))
    len_h = sum(d["length"] for _, _, d in H.edges(data=True))
    return {
        "edges": e_h,
        "edges_parent": e_g,
        "edge_retention_pct": round(100.0 * e_h / max(1, e_g), 3),
        "edges_removed_pct": round(100.0 * (1.0 - e_h / max(1, e_g)), 3),
        "length_km": round(len_h / 1000.0, 3),
        "length_retention_pct": round(100.0 * len_h / max(1e-9, len_g), 3),
        "avg_out_degree": round(e_h / max(1, H.number_of_nodes()), 3),
        "strongly_connected": bool(nx.is_strongly_connected(H)),
    }


# ---------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------
def edge_segments(G: nx.DiGraph):
    segs = []
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry", None)
        if geom is not None and hasattr(geom, "coords"):
            try:
                segs.append([(float(x), float(y)) for x, y in geom.coords])
                continue
            except Exception:
                pass
        segs.append([(G.nodes[u]["x"], G.nodes[u]["y"]),
                     (G.nodes[v]["x"], G.nodes[v]["y"])])
    return segs


def make_figure(results: Dict, G: nx.DiGraph, backbone: nx.DiGraph,
                hours: Sequence[int], t_focus: float, path: str) -> None:
    sns.set_theme(style="whitegrid", context="talk",
                  rc={"axes.edgecolor": "#333333", "grid.alpha": 0.35})

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.28, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ---- Subplot 1: dilation vs time of day -------------------------------------
    ax1.axvspan(6.5, 9.5, color="#f1c40f", alpha=0.14, zorder=0)
    ax1.axvspan(15.5, 19.5, color="#e67e22", alpha=0.14, zorder=0)
    for method in ("delay_bounded", "static", "hierarchical"):
        key = (method, t_focus) if method in ("delay_bounded", "static") else (method, None)
        entry = results.get(key)
        if entry is None:
            continue
        label, color, ls, marker = METHOD_STYLE[method]
        ax1.plot(hours, entry["metrics"]["per_hour_max"], label=label, color=color,
                 linestyle=ls, marker=marker, markersize=5.5, linewidth=2.4, zorder=3)
    ax1.axhline(t_focus, color="#111111", linestyle=(0, (4, 3)), linewidth=2.0,
                label=f"Certified bound $t={t_focus}$", zorder=4)
    ax1.set_xlabel("Hour of day $\\tau$")
    ax1.set_ylabel("Worst-case dilation  $\\max\\, d_H/d_G$")
    ax1.set_title("(a) Temporal dilation stability under diurnal congestion", loc="left")
    ax1.set_xticks(range(0, 24, 3))
    ax1.legend(fontsize=11, loc="upper left", framealpha=0.92)

    # ---- Subplot 2: sparsity vs t -----------------------------------------------
    for method in ("delay_bounded", "static"):
        xs, ys = [], []
        for t in sorted(T_VALUES):
            entry = results.get((method, t))
            if entry:
                xs.append(t)
                ys.append(entry["structure"]["edges_removed_pct"])
        if xs:
            label, color, ls, marker = METHOD_STYLE[method]
            ax2.plot(xs, ys, label=label, color=color, linestyle=ls,
                     marker=marker, markersize=9, linewidth=2.6)
    hier = results.get(("hierarchical", None))
    if hier:
        label, color, ls, _ = METHOD_STYLE["hierarchical"]
        ax2.axhline(hier["structure"]["edges_removed_pct"], color=color,
                    linestyle=ls, linewidth=2.4, label=label + " (t-independent)")
    ax2.set_xlabel("Dilation parameter $t$")
    ax2.set_ylabel("Edges removed (%)")
    ax2.set_title("(b) Sparsification vs. permitted dilation", loc="left")
    ax2.legend(fontsize=11, loc="lower right", framealpha=0.92)

    # ---- Subplot 3: spatial backbone --------------------------------------------
    base_segs = edge_segments(G)
    ax3.add_collection(LineCollection(base_segs, colors="#b9c2cc", linewidths=0.7,
                                      alpha=0.85, zorder=1))
    arterial, local = [], []
    for u, v, d in backbone.edges(data=True):
        seg = edge_segments(backbone.subgraph([u, v]).edge_subgraph([(u, v)]))
        seg = seg[0] if seg else [(G.nodes[u]["x"], G.nodes[u]["y"]),
                                  (G.nodes[v]["x"], G.nodes[v]["y"])]
        (arterial if d["tier"] == "arterial" else local).append(seg)
    ax3.add_collection(LineCollection(local, colors="#c0392b", linewidths=1.3,
                                      alpha=0.85, zorder=2))
    ax3.add_collection(LineCollection(arterial, colors="#7b0d1e", linewidths=2.6,
                                      alpha=0.95, zorder=3))
    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    padx = 0.02 * (max(xs) - min(xs) + 1e-6)
    pady = 0.02 * (max(ys) - min(ys) + 1e-6)
    ax3.set_xlim(min(xs) - padx, max(xs) + padx)
    ax3.set_ylim(min(ys) - pady, max(ys) + pady)
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(False)
    ax3.set_facecolor("#fdfdfd")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    src = G.graph.get("source", "unknown")
    keep = 100.0 * backbone.number_of_edges() / max(1, G.number_of_edges())
    ax3.set_title(f"(c) Certified emergency road backbone  $H \\subseteq G$  "
                  f"($t={t_focus}$, {keep:.1f}% of edges retained, source: {src})",
                  loc="left")
    from matplotlib.lines import Line2D
    ax3.legend(handles=[
        Line2D([0], [0], color="#b9c2cc", lw=2, label="Full network $G$"),
        Line2D([0], [0], color="#c0392b", lw=2, label="Backbone $H$ — collector/local"),
        Line2D([0], [0], color="#7b0d1e", lw=3, label="Backbone $H$ — arterial"),
    ], fontsize=12, loc="upper right", framealpha=0.95)

    fig.suptitle("Delay-Bounded Time-Varying Geometric $t$-Spanners for Urban Road Networks",
                 fontsize=21, y=0.965)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] wrote {path}")


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------
def build_table(results: Dict) -> str:
    rows = ["| Method | $t$ | Edges kept (%) | Removed (%) | Worst dilation (24 h) | Mean dilation | Breach rate | Certified | Build (s) |",
            "|---|---|---|---|---|---|---|---|---|"]
    order = []
    for t in sorted(T_VALUES):
        order.append(("delay_bounded", t))
        order.append(("static", t))
    order.append(("hierarchical", None))
    order.append(("full", None))
    for key in order:
        entry = results.get(key)
        if not entry:
            continue
        m, s, c = entry["metrics"], entry["structure"], entry.get("certificate", {})
        name = METHOD_STYLE[key[0]][0]
        tstr = f"{key[1]:.2f}" if key[1] is not None else "—"
        cert = "—" if not c else ("yes" if c.get("certified") else "**no**")
        rows.append(f"| {name} | {tstr} | {s['edge_retention_pct']:.1f} | "
                    f"{s['edges_removed_pct']:.1f} | {m['worst_dilation']:.3f} | "
                    f"{m['mean_dilation']:.3f} | {100*m['breach_rate']:.2f}% | {cert} | "
                    f"{entry['build_seconds']:.2f} |")
    return "\n".join(rows)


def inject_table(readme_path: str, table: str) -> None:
    start, end = "<!-- BENCHMARK_TABLE_START -->", "<!-- BENCHMARK_TABLE_END -->"
    if not os.path.exists(readme_path):
        return
    with open(readme_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if start not in text or end not in text:
        return
    head = text.split(start)[0]
    tail = text.split(end)[1]
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(head + start + "\n" + table + "\n" + end + tail)
    print(f"[report] benchmark table injected into {readme_path}")


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Delay-bounded spanner benchmark")
    ap.add_argument("--dist", type=int, default=DEFAULT_DIST)
    ap.add_argument("--lat", type=float, default=CENTER_POINT[0])
    ap.add_argument("--lon", type=float, default=CENTER_POINT[1])
    ap.add_argument("--synthetic", action="store_true", help="skip Overpass entirely")
    ap.add_argument("--quick", action="store_true", help="reduced smoke-test footprint")
    ap.add_argument("--order", type=str, default="temporal_max",
                    choices=["temporal_max", "temporal_mean", "centrality"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    dist = 700 if args.quick else args.dist
    n_src = 6 if args.quick else N_SOURCES
    n_tgt = 6 if args.quick else N_TARGETS_PER_SOURCE
    t_values = [1.2, 1.6] if args.quick else list(T_VALUES)
    t_focus = T_FOCUS if T_FOCUS in t_values else t_values[-1]
    hours = list(HOURS)

    print("=" * 84)
    print("  Delay-Bounded Time-Varying Geometric t-Spanner — benchmark")
    print("=" * 84)

    G = build_network((args.lat, args.lon), dist, force_synthetic=args.synthetic)
    summary = network_summary(G)
    print(f"[network] {summary}")

    pairs = sample_pairs(G, n_src, n_tgt, args.seed)
    print(f"[eval] sampling {sum(len(v) for v in pairs.values())} OD pairs "
          f"x {len(hours)} snapshots")
    t0 = time.perf_counter()
    base = baseline_distances(G, pairs, hours)
    print(f"[eval] ground-truth distance oracle built in {time.perf_counter()-t0:.2f}s")

    results: Dict = {}

    # --- proposed + static, across the t-grid ------------------------------------
    for t in t_values:
        H = delay_bounded_greedy_spanner(G, t, hours=hours, order=args.order)
        cert = certify_edge_condition(G, H, t, hours)
        metrics = evaluate_graph(H, pairs, hours, base)
        results[("delay_bounded", t)] = {
            "metrics": metrics,
            "structure": structural_metrics(G, H),
            "certificate": cert,
            "build_seconds": round(H.graph["build_seconds"], 3),
            "dijkstra_probes": H.graph["dijkstra_probes"],
        }
        print(f"  [delay-bounded t={t:.2f}] kept "
              f"{results[('delay_bounded', t)]['structure']['edge_retention_pct']:.1f}% | "
              f"worst={metrics['worst_dilation']:.3f} | certified={cert['certified']} | "
              f"{H.graph['build_seconds']:.2f}s")

        S = static_metric_spanner(G, t)
        S.graph["t"] = t
        cert_s = certify_edge_condition(G, S, t, hours)
        metrics_s = evaluate_graph(S, pairs, hours, base)
        results[("static", t)] = {
            "metrics": metrics_s,
            "structure": structural_metrics(G, S),
            "certificate": cert_s,
            "build_seconds": round(S.graph["build_seconds"], 3),
            "dijkstra_probes": S.graph["dijkstra_probes"],
        }
        print(f"  [static       t={t:.2f}] kept "
              f"{results[('static', t)]['structure']['edge_retention_pct']:.1f}% | "
              f"worst={metrics_s['worst_dilation']:.3f} | certified={cert_s['certified']} | "
              f"{S.graph['build_seconds']:.2f}s")

    # --- t-independent baselines --------------------------------------------------
    Hh = hierarchical_backbone(G)
    results[("hierarchical", None)] = {
        "metrics": evaluate_graph(Hh, pairs, hours, base),
        "structure": structural_metrics(G, Hh),
        "certificate": {},
        "build_seconds": round(Hh.graph["build_seconds"], 3),
        "dijkstra_probes": 0,
    }
    print(f"  [hierarchical] kept "
          f"{results[('hierarchical', None)]['structure']['edge_retention_pct']:.1f}% | "
          f"worst={results[('hierarchical', None)]['metrics']['worst_dilation']:.3f}")

    Hf = full_network(G)
    results[("full", None)] = {
        "metrics": evaluate_graph(Hf, pairs, hours, base),
        "structure": structural_metrics(G, Hf),
        "certificate": {},
        "build_seconds": 0.0,
        "dijkstra_probes": 0,
    }

    # --- outputs -------------------------------------------------------------------
    backbone = delay_bounded_greedy_spanner(G, t_focus, hours=hours, order=args.order,
                                            show_progress=False)
    make_figure(results, G, backbone, hours, t_focus, FIGURE_PATH)

    table = build_table(results)
    with open(TABLE_PATH, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    inject_table(README_PATH, table)

    report = {
        "framework": "Delay-Bounded Time-Varying Geometric t-Spanner",
        "version": "1.0.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "center": [args.lat, args.lon], "dist_m": dist, "hours": hours,
            "t_values": t_values, "t_focus": t_focus, "edge_order": args.order,
            "seed": args.seed, "sources": n_src, "targets_per_source": n_tgt,
            "quick": bool(args.quick),
        },
        "network": summary,
        "congestion_curve": [round(c, 4) for c in CONGESTION_CURVE],
        "results": {f"{m}|{'' if t is None else t}": v for (m, t), v in results.items()},
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[report] wrote {REPORT_PATH}")
    print("\n" + table + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
