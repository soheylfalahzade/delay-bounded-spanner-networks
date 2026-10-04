#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates assets/algorithm_demo.gif: a small, honest animation of the
Delay-Bounded Greedy Spanner actually running.

This is NOT a mockup -- it imports run_spanner_benchmark.py and replays the
exact same edge ordering (_edge_order) and the exact same pruned-Dijkstra
keep/skip decision (bounded_dijkstra) that the real algorithm uses, just on
a small toy grid chosen for legibility at GIF size rather than on a full
city. It is schematic/illustrative (not a benchmark result), lives outside
results/ for that reason, and is not referenced by verify_results.py or any
numeric claim in the README.

Usage:
    python tools/make_algorithm_animation.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

import run_spanner_benchmark as spanner

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
OUT_PATH = os.path.join(OUT_DIR, "algorithm_demo.gif")

T = 1.5
GRID_N = 6
HOURS = [0]  # single-hour toy run: this is about the KEEP/SKIP logic, not the 24h sweep


def build_toy_graph():
    G = spanner.synthetic_arterial_grid((31.9, 54.35), 550, n=GRID_N,
                                        seed_key="anim-demo", irregularity=0.08)
    G = spanner.enrich_temporal_costs(G)
    return G


def replay_greedy(G):
    """Re-runs the exact algorithm from delay_bounded_greedy_spanner(), one
    edge at a time, yielding (u, v, kept: bool, adj_snapshot) after each
    decision, so we can render the REAL decision sequence frame by frame."""
    ordered = spanner._edge_order(G, "temporal_max", "tt", HOURS)
    adj = {n: {} for n in G.nodes()}
    for u, v, data in ordered:
        costs = data["tt"]
        slice_order = sorted(HOURS, key=lambda tau: costs[tau])
        keep = False
        for tau in slice_order:
            budget = T * costs[tau]
            if spanner.bounded_dijkstra(adj, u, v, tau, budget) > budget + spanner.EPS:
                keep = True
                break
        if keep:
            adj[u][v] = costs
        yield u, v, keep


def node_positions(G):
    return {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}


def draw_frame(G, pos, kept_edges, tested_edge, skipped_edges, step, total, ax_lim):
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=120)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # full network, faint
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#30363d", linewidth=1.1, zorder=1)

    # already-skipped edges this run: very faint dashed
    for u, v in skipped_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#484f58", linewidth=1.0,
                linestyle=(0, (1, 2)), zorder=2)

    # kept edges so far: the certified backbone growing
    for u, v in kept_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#ff6b57", linewidth=3.0, zorder=3,
                solid_capstyle="round")

    # edge currently being tested: bright yellow highlight
    if tested_edge is not None:
        u, v = tested_edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#f2cc60", linewidth=4.0, zorder=4,
                solid_capstyle="round")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=16, color="#8b949e", zorder=2)

    pad_x = ax_lim[0]
    pad_y = ax_lim[1]
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    kept_n = len(kept_edges)
    total_n = G.number_of_edges()
    ax.set_title(
        f"Delay-Bounded Greedy Spanner  —  t = {T}   ({kept_n}/{total_n} edges kept so far)",
        color="#e6edf3", fontsize=12, pad=12, loc="center", fontweight="bold")

    legend = [
        Line2D([0], [0], color="#30363d", lw=2, label="Full network $G$"),
        Line2D([0], [0], color="#f2cc60", lw=3, label="Edge being tested"),
        Line2D([0], [0], color="#484f58", lw=2, linestyle=(0, (1, 2)), label="Skipped (redundant)"),
        Line2D([0], [0], color="#ff6b57", lw=3, label="Kept — certified backbone $H$"),
    ]
    leg = ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.02),
                    ncol=2, fontsize=8.5, frameon=False, labelcolor="#c9d1d9")

    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[anim] building toy graph ...")
    G = build_toy_graph()
    pos = node_positions(G)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = (0.06 * (max(xs) - min(xs) + 1e-9), 0.06 * (max(ys) - min(ys) + 1e-9))
    print(f"[anim] toy graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    decisions = list(replay_greedy(G))
    total = len(decisions)
    print(f"[anim] replayed {total} real keep/skip decisions from the actual algorithm")

    # Subsample decisions to a manageable frame count for a smooth, short GIF.
    target_decision_frames = 28
    step = max(1, total // target_decision_frames)

    frames = []
    kept, skipped = [], []
    for i, (u, v, keep) in enumerate(decisions):
        if keep:
            kept.append((u, v))
        else:
            skipped.append((u, v))
        if i % step == 0 or i == total - 1:
            frames.append(draw_frame(G, pos, list(kept), (u, v), list(skipped), i, total, pad))

    # Hold on the final certified backbone for a few extra frames, then loop.
    final = draw_frame(G, pos, kept, None, skipped, total, total, pad)
    frames.extend([final] * 8)

    print(f"[anim] writing {len(frames)} frames to {OUT_PATH}")
    frames[0].save(OUT_PATH, save_all=True, append_images=frames[1:],
                   duration=160, loop=0, optimize=True)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"[anim] done: {OUT_PATH} ({size_kb:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
